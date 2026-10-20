# -*- coding: utf-8 -*-
"""[主干实验] 把 DINOv3 主干包装成 HuggingFace DINOv2 的调用接口，供 bank 分支（MemoryBank）使用。

bank 分支原实现（其余部分一字不改）：
    outputs = model(images); hidden_states = outputs.hidden_states
    fused = [sum(hidden_states[l] for l in [2,3,4,5])[:, 1:, :],
             sum(hidden_states[l] for l in [6,7,8,9])[:, 1:, :]]
即"每一层输出 [B, 1+patch, C]，丢掉第 0 个 cls token 后取 patch"。

本适配器只做两件事：
1. 用 timm 的 DINOv3（RoPE + 4 register）逐层跑，产出与 HF 同构的 hidden_states 元组：
   索引 0 = 词嵌入输出，索引 i = 第 i 个 block 之后（与 HF DINOv2 的约定一致）。
2. **丢掉 4 个 register token**，使 `[:, 1:, :]` 恰好等于 patch 序列
   （register 是全局 token，混进 patch 特征库会污染 kNN）。

层分组沿用 bank 原配置 [2,3,4,5] / [6,7,8,9]（DINOv2-base 与 DINOv3-B 都是 12 层，相对深度一致，
非测试集调参）。输入边长不是 16 的倍数时（bank 的 392 裁剪）内部重采样到 400，与 mapping 分支同处理。
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

TIMM_NAMES = {
    'dinov3b': 'vit_base_patch16_dinov3.lvd1689m',
    'dinov3l': 'vit_large_patch16_dinov3.lvd1689m',
}
LOCAL_CKPT = {
    'dinov3b': '/home/jjquan/Downloads/dinov3/model.safetensors',
    'dinov3l': '/home/jjquan/Downloads/dinov3l/model.safetensors',
}


class DinoV3HFAdapter(nn.Module):
    def __init__(self, arch='dinov3b', ckpt=None, img_size=518):
        super().__init__()
        assert arch in TIMM_NAMES, arch
        self.arch = arch
        self.backbone = timm.create_model(
            TIMM_NAMES[arch], pretrained=False, num_classes=0,
            img_size=img_size, dynamic_img_size=True,
        )
        from safetensors.torch import load_file
        state = load_file(ckpt or LOCAL_CKPT[arch])
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        info = self.backbone.load_state_dict(state, strict=False)
        miss = [k for k in info.missing_keys if not k.startswith('head.')]
        if miss or info.unexpected_keys:
            raise RuntimeError(f'DINOv3 权重键不匹配 missing={miss[:5]} unexpected={info.unexpected_keys[:5]}')
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])
        self.num_register_tokens = int(self.backbone.num_prefix_tokens) - 1
        print(f'[dinov3-adapter] {arch} 载入完成：patch={self.patch_size} '
              f'register={self.num_register_tokens} blocks={len(self.backbone.blocks)}')

    def _fit(self, x):
        p = self.patch_size
        h, w = x.shape[-2:]
        nh = ((h + p - 1) // p) * p
        nw = ((w + p - 1) // p) * p
        if (nh, nw) != (h, w):
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
        return x

    def forward(self, images):
        x = self._fit(images)
        t = self.backbone.patch_embed(x)
        t, rope = self.backbone._pos_embed(t)
        t = self.backbone.norm_pre(t)
        npt = self.backbone.num_prefix_tokens          # 1 cls + 4 register
        states = [torch.cat([t[:, :1], t[:, npt:]], dim=1)]   # 索引 0：词嵌入（丢 register）
        for blk in self.backbone.blocks:
            t = blk(t, rope=rope)
            states.append(torch.cat([t[:, :1], t[:, npt:]], dim=1))
        return types.SimpleNamespace(hidden_states=tuple(states))


def build_adapter(arch='dinov3b'):
    return DinoV3HFAdapter(arch)
