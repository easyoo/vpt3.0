# -*- coding: utf-8 -*-
"""CLIP 主干封装冒烟测试：形状 + 与 timm 原生逐层特征的一致性。"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code_vpt4'))
sys.path.insert(0, HERE)

from clip_mapping import build_clip           # noqa: E402


class A:
    pass


def mkargs(n=32):
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = n
    a.res_mapping = 1
    a.metric_head = 1
    a.token_init_std = 0.02
    return a


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = build_clip('clipb', mkargs()).to(dev).eval()
    print(f'[smoke] embed_dim={m.embed_dim} patch={m.patch_size} prefix={m.num_prefix_tokens} '
          f'blocks={len(m.blocks)}')
    for px in (400, 448, 672):
        x = torch.randn(2, 3, px, px, device=dev)
        with torch.no_grad():
            ret = m(x, is_training=True)
        g = int(round(ret['patch_feat_layer'][0].shape[1] ** 0.5))
        print(f'[smoke] {px}px → {g}x{g}={g*g} (期望 {(px//16)**2})  层数={len(ret["patch_feat_layer"])} '
              f'feat_list={[f.shape[1] for f in ret["feat_list"]]}')
        assert ret['patch_feat_layer'][0].shape[1] == (px // 16) ** 2
        assert len(ret['patch_feat_layer']) == 12
    # 等价性：不注入任何 token 时，逐层 patch 特征应与 timm 原生路径逐位一致
    m0 = build_clip('clipb', mkargs(0)).to(dev).eval()
    m0.backbone.load_state_dict(m.backbone.state_dict())
    with torch.no_grad():
        x = torch.randn(1, 3, 448, 448, device=dev)
        ret = m0(x, is_training=True)
        t = m0.backbone._pos_embed(m0.backbone.patch_embed(x))
        t = m0.backbone.norm_pre(t)
        idxs = [1, 2, 3, 4, 6, 8, 11]
        d = 0.0
        for i, blk in enumerate(m0.blocks):
            t = blk(t)
            if i in idxs:
                d = max(d, (ret['patch_feat_layer'][i] - t[:, 1:, :]).abs().max().item())
    print(f'[smoke] token 数为 0 时与 timm 原生逐层特征最大差 {d:.3e}')
    assert d < 1e-4, d
    print('[smoke] OK')


if __name__ == '__main__':
    main()
