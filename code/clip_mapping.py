# -*- coding: utf-8 -*-
"""[主干实验 · 换家族] CLIP ViT-B/16 作为 mapping 分支的特征主干。

为什么试：现有全部视图（DINOv2-B/L、DINOv3-B/L）都属同一自监督家族、特征高度相关，
这正是集成收益饱和（+0.85~0.94）的原因；而且卡住的指标是**图像级**（语义/全局性质），
CLIP 的语义型特征与 DINO 的自监督局部特征互补性最强。

设计与 DINOv3Mapping 完全同构（核心创新一字未改，直接复用同一份类对象）：
  * LearnableTokens / TokenMapping 复用 `dinov2.models.vision_transformer` 的同一份实现；
  * 注入位置（layer 0~3）、丢弃时机、patch_feat_layer / feat_list 顺序与 v3.7 一致；
  * 唯一区别：CLIP 的前缀只有 1 个 cls token（无 register），故取 patch 用 `x[:, 1:l, :]`；
    位置编码是绝对式（timm 的 _pos_embed 在 dynamic_img_size 下自动插值），追加的
    可学习 token 天然没有位置信息 —— 与 v3.7 的 DINOv2 行为一致，无需 RoPE 补位。
边界：输入边长向上取整到 16 的倍数（训练几何 392 → 400），与 DINOv3 接入同一处理。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from dinov2.models.vision_transformer import LearnableTokens, TokenMapping

TIMM_NAMES = {
    'clipb': 'vit_base_patch16_clip_224.laion2b',
}
LOCAL_CKPT = {
    'clipb': '/home/jjquan/Downloads/clipb/model.safetensors',
}


class ClipMapping(nn.Module):
    def __init__(self, arch, ckpt, arg, img_size=518):
        super().__init__()
        assert arch in TIMM_NAMES, arch
        self.arch = arch
        self.backbone = timm.create_model(
            TIMM_NAMES[arch], pretrained=False, num_classes=0,
            # CLIP 是绝对位置编码：必须按**预训练尺寸 224** 建模型（pos_embed 197×768），
            # 再由 dynamic_img_size 在前向时插值到任意输入尺寸。
            img_size=224, dynamic_img_size=True,
        )
        from safetensors.torch import load_file
        state = load_file(ckpt)
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        info = self.backbone.load_state_dict(state, strict=False)
        miss = [k for k in info.missing_keys if not k.startswith('head.')]
        print('[clip] 主干 %s 载入 %d 张量；缺失 %d / 多余 %d'
              % (arch, len(state), len(miss), len(info.unexpected_keys)))
        if miss or info.unexpected_keys:
            raise RuntimeError('CLIP 权重键不匹配 missing=%s unexpected=%s'
                               % (miss[:6], info.unexpected_keys[:6]))
        self.embed_dim = int(self.backbone.embed_dim)
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)
        if self.num_prefix_tokens != 1:
            raise RuntimeError('本实现假定 CLIP 只有 1 个 cls 前缀，实际 %d' % self.num_prefix_tokens)
        self.num_register_tokens = 0
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

    def _fit(self, x):
        p = self.patch_size
        h, w = x.shape[-2:]
        nh = ((h + p - 1) // p) * p
        nw = ((w + p - 1) // p) * p
        if (nh, nw) != (h, w):
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
        return x

    def _embed(self, x, masks=None):
        if masks is not None:
            raise NotImplementedError('CLIP 主干不支持 mask token')
        x = self._fit(x)
        t = self.backbone.patch_embed(x)                 # [B, C, H', W']
        t = self.backbone._pos_embed(t)                  # 绝对位置编码（dynamic 自动插值）→ [B, N+1, C]
        t = self.backbone.norm_pre(t)
        return t

    def prepare_tokens_with_masks(self, x, masks=None):
        return self._embed(x, masks)

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            raise NotImplementedError('list 输入未使用')
        x = self._embed(x, masks)
        b, l, c = x.shape
        x = torch.cat((x, self.tokens_per_layer.token_dict['0'].expand(x.shape[0], -1, -1)), dim=1)
        feat_list, patch_feat_layer = [], []
        n_learn = len(self.tokens_per_layer.token_dict.keys())
        for i, blk in enumerate(self.blocks):
            if i < n_learn:
                tok = self.tokens_per_layer.token_dict[str(i)]
                x = torch.cat((x, tok.expand(x.shape[0], -1, -1)), dim=1)
                x = blk(x)
                length = x.shape[1] - l
                feat_list.append(x[:, -length:, :])
                patch_feat_layer.append(x[:, 1:l, :])       # CLIP：前缀仅 1 个 cls
                x = x[:, :l, :]
            else:
                x = blk(x)
                patch_feat_layer.append(x[:, 1:l, :])
        x_norm = self.norm(x)
        return {
            'x_norm_clstoken': x_norm[:, 0],
            'x_norm_regtokens': x_norm[:, 1:1],
            'x_norm_patchtokens': x_norm[:, 1:],
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


def build_clip(arch, arg, img_size=518, ckpt=None):
    return ClipMapping(arch, ckpt or LOCAL_CKPT[arch], arg, img_size=img_size)
