# -*- coding: utf-8 -*-
"""DINOv3 主干封装冒烟测试：形状 / RoPE 对齐 / 与 v3.7 语义一致性检查。"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code_vpt4'))
sys.path.insert(0, HERE)

from dinov3_mapping import build_dinov3           # noqa: E402


class A:
    pass


def mkargs():
    a = A()
    a.learnable_layers = 4
    a.learnable_num_per_layer = 32
    a.res_mapping = 1
    a.metric_head = 1
    a.token_init_std = 0.02
    return a


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = mkargs()
    m = build_dinov3('dinov3b', a, img_size=518).to(dev).eval()
    print(f'[smoke] embed_dim={m.embed_dim} patch={m.patch_size} prefix={m.num_prefix_tokens} '
          f'blocks={len(m.blocks)}')
    for size, scale in [(448, 1.0), (448, 1.5)]:
        px = int(round(size * scale))
        x = torch.randn(2, 3, px, px, device=dev)
        with torch.no_grad():
            ret = m(x, is_training=True)
        g = int(round(ret['patch_feat_layer'][0].shape[1] ** 0.5))
        print(f'[smoke] 输入 {px}px → patch 网格 {g}x{g}={g*g} '
              f'(期望 {px//16}x{px//16}={px//16*(px//16)})  '
              f'层数={len(ret["patch_feat_layer"])}  feat_list 长度={[f.shape[1] for f in ret["feat_list"]]}')
        assert ret['patch_feat_layer'][0].shape[1] == (px // 16) ** 2, 'patch 数与分辨率不符'
        assert len(ret['patch_feat_layer']) == 12
        assert [f.shape[1] for f in ret['feat_list']] == [64, 32, 32, 32]
    # RoPE 对齐检查：把可学习 token 数设成 0（不往序列里加任何 token）时，
    # 逐层 patch 特征必须与 timm 原生 forward_intermediates 逐位一致。
    # 说明：不能用"token 置零"做等价性判断——额外的 token 即使数值为 0，也会以 key/value
    # 的形式参与注意力（softmax 分母变化），v3.7 的 DINOv2 路径同理。
    a0 = mkargs()
    a0.learnable_num_per_layer = 0
    m0 = build_dinov3('dinov3b', a0, img_size=518).to(dev).eval()
    m0.backbone.load_state_dict(m.backbone.state_dict())
    with torch.no_grad():
        x = torch.randn(1, 3, 448, 448, device=dev)
        ret = m0(x, is_training=True)
        _final, pure = m0.backbone.forward_intermediates(x, indices=[1, 2, 3, 4, 6, 8, 11], norm=False)
        d = 0.0
        for j, idx in enumerate([1, 2, 3, 4, 6, 8, 11]):
            pj = pure[j]
            if pj.dim() == 4:
                pj = pj.flatten(2).transpose(1, 2)
            else:
                pj = pj[:, 5:, :] if pj.dim() == 3 else pj[5:, :].unsqueeze(0)
            d = max(d, (ret['patch_feat_layer'][idx] - pj).abs().max().item())
        print(f'[smoke] 可学习 token 数为 0 时，逐层 patch 特征与 timm 原生 forward_intermediates 最大差 '
              f'{d:.3e}（应为 0，证明 RoPE 补位与前缀对齐正确）')
        assert d < 1e-4, d
    print('[smoke] OK')


if __name__ == '__main__':
    main()
