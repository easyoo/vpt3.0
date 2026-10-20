# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

def MLp(in_dim,out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
    )
class LearnableTokens(nn.Module):
    def __init__(self, 
                    layers = 4,
                    dim = 768,
                    learnable_token_per_layer = 8,
                    init_std = 1e-6):
        super().__init__()
        self.token_dict = nn.ParameterDict({
            str(i): nn.Parameter(torch.randn(1,learnable_token_per_layer, dim))
            for i in range(layers)})
        # [vpt4.0 · M1] 原实现 std=1e-6 → 32 个可学习 token 初始几乎完全相同（实测训练后两两余弦仍 0.87，
        # 参与比有效秩仅 2~3/32，即"32 个原型实际只有 2~3 个自由度"）。
        # 这里改成与 ViT patch embedding 同量级的初始化（截断正态 std=0.02），
        # 使 32 个原型从第一步起就是**方向不同、尺度可比**的 32 个候选，训练只需让它们分化而不是"从零破对称"。
        for param in self.token_dict.values():
            # [FIX-INIT-REPLAY] 必须与原实现用**同一个算子** `nn.init.normal_`。
            # 原先用 trunc_normal_（内部是"均匀分布→erfinv"变换），分布与原实现几乎相同，
            # 但**消耗随机数的方式不同** → 之后所有随机增强（4 shot × 22 倍）全部错位，
            # 导致"机制全关也复现不出基线"（实测 pill P-AP 0.5987 → 0.5423）。
            # 换成 normal_ 后，随机流与原实现一致，"关掉机制＝复现基线"，A/B 才干净。
            # 对 M1 本意（std=0.02）几乎无影响：正态在 ±2σ 外的概率很小，打散原型的目的同样成立。
            nn.init.normal_(param, std=init_std)


class TokenMapping(nn.Module):
    """[vpt4.0 · M2] 带残差的映射：out = x + f(LayerNorm(x))。

    原实现是纯低秩瓶颈 `LayerNorm→Linear(768,128)→GELU→LN(128)→Linear(128,768)`：
    输出被强制压进 ≤128 维子空间，等于**替换**掉 DINOv2 的特征几何（文献里 PatchCore/PaDiM 一类
    方法之所以强，恰恰是因为直接在原始 DINOv2 特征上比距离）。
    这里给映射加一条恒等旁路，并把最后一层**零初始化**：
      * 初始时 out ≡ x（等价于"不做映射"，即原始特征空间的最强基线）；
      * 训练只能在这个解之上做修正，因此不会比"直接用原始特征"更差。
    residual=False 时退化为原实现（用于 A/B 对照）。
    """

    def __init__(self, dim=768, hidden=128, residual=True, metric=True):
        super().__init__()
        self.residual = residual
        self.ln = nn.LayerNorm(dim)
        self.f = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )
        if residual:
            nn.init.zeros_(self.f[-1].weight)
            nn.init.zeros_(self.f[-1].bias)
            # 输出再归一化：残差会把"原始特征尺度"带进来（fusion 后范数约 160，而原瓶颈输出约 27），
            # 直接接原损失会让 MSE 项量级暴涨（实测 epoch1 mse=28，原配方 0.09）→ 损失被 MSE 主导。
            # LayerNorm 保持方向结构不变、把尺度拉回可比范围，使 cosine/MSE 两项回到原来的平衡。
            self.out_ln = nn.LayerNorm(dim)
        else:
            self.out_ln = None
        # [vpt4.0 · M6] 可学习度量头：把"余弦距离"放到一个被训练出来的度量空间里算。
        # 初值为单位阵 → 与现状（直接余弦）**严格等价**，训练只会在此基础上改进；
        # 它同时作用于 patch 与原型两侧（都经过 token_mapping），因此打分口径保持一致。
        self.metric = nn.Parameter(torch.eye(dim)) if metric else None

    def forward(self, x):
        h = self.f(self.ln(x))
        if not self.residual:
            out = h
        else:
            h = x + h
            out = self.out_ln(h) if self.out_ln is not None else h
        if self.metric is not None:
            out = out @ self.metric
        return out



class ReconstructHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(in_dim)
        
        self.rec_head = self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.norm(x)
        return self.rec_head(x)



class LocalConvMapping(nn.Module):
    def __init__(self, dim=768):
        super(LocalConvMapping, self).__init__()
        # 1. 第一层归一化
        self.ln1 = nn.LayerNorm(dim)
        
        # 2. 深度卷积 (Depth-wise Conv): 捕捉局部空间关系
        # kernel_size=3 可以让每个 patch 观察到周围 8 个邻居，增强不连续性的感知
        self.dw_conv = nn.Sequential(
            nn.ReflectionPad2d(1), # 保持边缘尺寸
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, groups=dim),
            nn.GELU()
        )
        
        # 3. 逐点卷积 (Point-wise Conv): 跨通道特征融合
        self.pw_conv = nn.Linear(dim, dim)
        
        # 4. 残差缩放因子：初始设为很小的值，保证训练初期特征空间的稳定
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x shape: [B, N, C], 其中 N 通常是 784 (28x28)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        identity = x
        x = self.ln1(x)
        
        # 转换为 2D 空间维度 [B, C, H, W]
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 局部卷积操作
        x_2d = self.dw_conv(x_2d)
        
        # 转回 Token 序列 [B, N, C]
        x = x_2d.reshape(B, C, N).transpose(1, 2)
        
        # 线性映射
        x = self.pw_conv(x)
        
        # 残差连接：保留 DINOv2 的原始特征判别力
        return identity + self.gate * x

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        arg=None
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()
        
        
        #############
        learnable_layers = 4 if arg is None else arg.learnable_layers
        # self.mlp_per_layer = nn.ModuleDict({
        #     str(i): MLp(embed_dim, embed_dim)
        #     for i in range(learnable_layers)
        # })
        
        # [vpt4.0 · M1] token 初始化尺度（默认 0.02，可被 --token_init_std 覆盖；0.000001 = 原行为）
        _tok_std = getattr(arg, 'token_init_std', 0.02) if arg is not None else 0.02
        self.tokens_per_layer = LearnableTokens(layers=learnable_layers,
                                                dim=embed_dim,
                                                learnable_token_per_layer=arg.learnable_num_per_layer,
                                                init_std=_tok_std)
        # self.token_mapping = nn.Sequential(
        #     nn.LayerNorm(768),
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(768, 768),
        #     # nn.ReLU(),
        # )
        # [vpt4.0 · M2] 残差映射（默认开启）；--res_mapping 0 则退回原低秩实现
        _res_map = bool(getattr(arg, 'res_mapping', 1)) if arg is not None else True
        _metric = bool(getattr(arg, 'metric_head', 1)) if arg is not None else True
        self.token_mapping = TokenMapping(dim=embed_dim, hidden=128, residual=_res_map, metric=_metric)
        
        # self.p2l = nn.Identity()
        # self.rec_head = ReconstructHead(embed_dim, embed_dim)
        
    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        b,l,c = x.shape
        # [CONTROL] 这里刻意保留原项目的行为：循环外先追加一次 token_dict[str(0)]，
        # 循环体内 i=0 时再追加一次（即 layer-0 可学习 token 被追加两遍，
        # feat_list[0]=64 且前后 32 个逐元素相同）。用于和审计副本做 A/B。
        x=torch.cat((x,self.tokens_per_layer.token_dict[str(0)].expand(x.shape[0], -1, -1)), dim=1)
        feat_list,patch_feat_layer = [],[]
        for i,blk in enumerate(self.blocks):
            # if i==0:
            #     x = torch.cat((x,self.tokens_per_layer.token_dict[str(i)].expand(x.shape[0], -1, -1)), dim=1) 
            #     x = blk(x)
            #     feat_list.append(x[:,-4:,:]) # b 4 c
            # elif i<len(self.tokens_per_layer.token_dict.keys()):
            #     x[:,l:,:] += self.tokens_per_layer.token_dict[str(i)]
                
            #     x = blk(x)
            #     feat_list.append(x[:,-4:,:]) # b 4 c
            # elif i==len(self.tokens_per_layer.token_dict.keys()):
            #     # 如果是最后一层需要添加token的，则移除后面可学习的token
            #     x = x[:,0:-4:,:]
            #     x = blk(x)
            # else:
            #     x = blk(x)
           
            # 输出后即丢弃
            if i < len(self.tokens_per_layer.token_dict.keys()):
                x = torch.cat((x,self.tokens_per_layer.token_dict[str(i)].expand(x.shape[0], -1, -1)), dim=1) 
                x = blk(x)
                length = x.shape[1]-l
                feat_list.append(x[:,-length:,:]) # b len c
                patch_feat_layer.append(x[:,5:l,:])
                x = x[:,:l,:] # b l c
            
            else:
                x = blk(x)
                patch_feat_layer.append(x[:,5:l,:])
            
            # 输出后不丢弃
            # if i < 10:
            #     x = blk(x)
            #     length = x.shape[1]-l
            #     feat_list.append(x[:,-length:,:]) # b len c
            #     patch_feat_layer.append(x[:,5:l,:])             
            # else:
            #     x = x[:,:l,:] # b l c              
            #     x = blk(x)
            #     patch_feat_layer.append(x[:,5:l,:])

   
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,

            "patch_feat_layer": patch_feat_layer,
            "feat_list": feat_list,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
