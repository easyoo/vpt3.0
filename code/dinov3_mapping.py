# -*- coding: utf-8 -*-
"""[主干实验] DINOv3 特征主干接入（只替换主干，核心创新完全不变）。

设计原则（与 vpt3.7 逐条对齐，保证"只动主干"）：
1. LearnableTokens / TokenMapping 直接 import 自 `dinov2.models.vision_transformer`，
   即**同一份代码对象**；可学习 token 的注入位置（layer 0~3）、丢弃时机、
   patch_feat_layer / feat_list 的构造顺序与 v3.7 完全一致（含 layer-0 双追加的历史行为）。
2. 主干用 timm 的 DINOv3 实现（RoPE + 4 个 register token，patch16），
   未预训练构建 + 从本地 safetensors 读权重，不依赖联网。
3. v3.7 的 DINOv2 主干用**绝对位置编码**，可学习 token 追加到序列末尾时天然没有位置信息；
   DINOv3 用 **RoPE**，而 RoPE 按"序列末尾 len(rope) 个 token"对齐。因此这里在追加 K 个
   可学习 token 时，把 rope 末尾补 K 行 (sin=0, cos=1)（等价于"不旋转"），
   使 prefix = N - len(rope) 仍等于 5（1 cls + 4 register），真实 patch 的旋转角不受影响。
   可学习 token 与 v3.7 一样是**位置无关**的。
"""
import torch
import torch.nn as nn
import timm

from dinov2.models.vision_transformer import LearnableTokens, TokenMapping

TIMM_NAMES = {
    'dinov3b': 'vit_base_patch16_dinov3.lvd1689m',
    'dinov3l': 'vit_large_patch16_dinov3.lvd1689m',
}
LOCAL_CKPT = {
    'dinov3b': '/home/jjquan/Downloads/dinov3/model.safetensors',
    'dinov3l': '/home/jjquan/Downloads/dinov3l/model.safetensors',
}


def _pad_rope_no_rotate(rope, k):
    """给 rope 末尾补 k 行"不旋转"位置（sin=0, cos=1）。

    timm 的 rope 张量形状为 [HW, 2*head_dim]，前一半是 sin、后一半是 cos
    （见 timm.layers.apply_rot_embed_cat 的 chunk(2) 拆法）。
    """
    if rope is None or k <= 0:
        return rope
    if isinstance(rope, (tuple, list)):
        sin, cos = rope
        d = sin.shape[-1]
        return (torch.cat([sin, sin.new_zeros(k, d)], dim=-2),
                torch.cat([cos, cos.new_ones(k, d)], dim=-2))
    d2 = rope.shape[-1]
    half = d2 // 2
    pad = torch.cat([rope.new_zeros(k, half), rope.new_ones(k, half)], dim=-1)
    return torch.cat([rope, pad], dim=-2)


class DinoV3Mapping(nn.Module):
    """对外暴露与 code_vpt4 里 DinoVisionTransformer 一致的接口子集。"""

    def __init__(self, arch, ckpt, arg, img_size=518):
        super().__init__()
        assert arch in TIMM_NAMES, arch
        self.arch = arch
        self.backbone = timm.create_model(
            TIMM_NAMES[arch], pretrained=False, num_classes=0,
            img_size=img_size, dynamic_img_size=True,
        )
        from safetensors.torch import load_file
        state = load_file(ckpt)
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        info = self.backbone.load_state_dict(state, strict=False)
        miss = [k for k in info.missing_keys if not k.startswith('head.')]
        print('[dinov3] 主干 %s 载入 %d 张量；缺失 %d / 多余 %d'
              % (arch, len(state), len(miss), len(info.unexpected_keys)))
        if miss or info.unexpected_keys:
            raise RuntimeError('DINOv3 权重键不匹配 missing=%s unexpected=%s'
                               % (miss[:6], info.unexpected_keys[:6]))
        self.embed_dim = int(self.backbone.embed_dim)
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)
        self.num_register_tokens = self.num_prefix_tokens - 1
        if self.num_register_tokens != 4:
            raise RuntimeError('本实现假定 1 cls + 4 register（v3.7 硬编码 5），实际 %d 个 register'
                               % self.num_register_tokens)
        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm
        learnable_layers = 4 if arg is None else arg.learnable_layers
        _tok_std = getattr(arg, 'token_init_std', 0.02) if arg is not None else 0.02
        self.tokens_per_layer = LearnableTokens(
            layers=learnable_layers, dim=self.embed_dim,
            learnable_token_per_layer=arg.learnable_num_per_layer, init_std=_tok_std)
        _res_map = bool(getattr(arg, 'res_mapping', 1)) if arg is not None else True
        _metric = bool(getattr(arg, 'metric_head', 1)) if arg is not None else True
        self.token_mapping = TokenMapping(dim=self.embed_dim, hidden=128,
                                          residual=_res_map, metric=_metric)

    def _embed(self, x, masks=None):
        if masks is not None:
            raise NotImplementedError('DINOv3 主干暂不支持 mask token')
        x = self._fit_patch_grid(x)
        t = self.backbone.patch_embed(x)
        t, rope = self.backbone._pos_embed(t)
        t = self.backbone.norm_pre(t)
        hp = x.shape[-2] // self.patch_size
        wp = x.shape[-1] // self.patch_size
        return t, rope, (hp, wp)

    def _fit_patch_grid(self, x):
        """[主干适配] 输入边长必须是 patch(16) 的整数倍。

        v3.7 的训练几何是 Resize(448) → CenterCrop(392)，392 只能是 14 的倍数（DINOv2）。
        为了**不改动数据几何**（同一裁剪窗、同一随机增强、同一 RNG 消耗），这里在主干内部把
        392 双线性重采样到 400（=25×16），使 patch16 能整除；保持"只换主干、只变主干看到的输入"
        这一最小改动。448 / 672 等生产推理尺寸本身可整除，不受影响。
        """
        p = self.patch_size
        h, w = x.shape[-2:]
        nh = ((h + p - 1) // p) * p
        nw = ((w + p - 1) // p) * p
        if (nh, nw) != (h, w):
            x = torch.nn.functional.interpolate(x, size=(nh, nw), mode='bilinear',
                                                align_corners=False)
        return x

    def prepare_tokens_with_masks(self, x, masks=None):
        t, _rope, _hw = self._embed(x, masks)
        return t

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            raise NotImplementedError('list 输入未使用')
        x, rope, _hw = self._embed(x, masks)
        b, l, c = x.shape
        x = torch.cat((x, self.tokens_per_layer.token_dict['0'].expand(x.shape[0], -1, -1)), dim=1)
        feat_list, patch_feat_layer = [], []
        n_learn = len(self.tokens_per_layer.token_dict.keys())
        for i, blk in enumerate(self.blocks):
            if i < n_learn:
                tok = self.tokens_per_layer.token_dict[str(i)]
                # 进入本层前序列里已有的"非 patch"多余 token（layer-0 的循环外追加会带 32 个）
                n_extra = x.shape[1] - l
                x = torch.cat((x, tok.expand(x.shape[0], -1, -1)), dim=1)
                x = blk(x, rope=_pad_rope_no_rotate(rope, n_extra + tok.shape[1]))
                length = x.shape[1] - l
                feat_list.append(x[:, -length:, :])
                patch_feat_layer.append(x[:, 5:l, :])
                x = x[:, :l, :]
            else:
                x = blk(x, rope=rope)
                patch_feat_layer.append(x[:, 5:l, :])
        x_norm = self.norm(x)
        return {
            'x_norm_clstoken': x_norm[:, 0],
            'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1],
            'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:],
            'x_prenorm': x,
            'masks': masks,
            'patch_feat_layer': patch_feat_layer,
            'feat_list': feat_list,
        }

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return ret['x_norm_clstoken']


def build_dinov3(arch, arg, img_size=518, ckpt=None):
    ck = ckpt or LOCAL_CKPT[arch]
    return DinoV3Mapping(arch, ck, arg, img_size=img_size)
