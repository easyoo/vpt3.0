# -*- coding: utf-8 -*-
"""[审计探针] 量化 mapping 分支"原型"的有效多样性。

关注对象：token_mapping(fused_learnable_token) —— 即测试时用来算
"patch 到最近原型的余弦距离"的那组原型。

输出：
  - 原型两两余弦相似度（映射后 / 映射前）
  - 原型矩阵的奇异值谱与有效秩（participation ratio）
  - softmax 权重熵（temperature=0.07）→ 有效原型数
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/code')
import torchvision.transforms as T
from PIL import Image

from dinov2.models import vision_transformer


def build(ckpt_path, learnable_layers=1, learnable_num_per_layer=32, temperature=0.07):
    class A:
        pass
    args = A()
    args.learnable_layers = learnable_layers
    args.learnable_num_per_layer = learnable_num_per_layer
    model = vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=args)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['dino'], strict=False)
    model.eval()
    return model, args


def participation_ratio(sv):
    sv = sv / sv.sum()
    return float(1.0 / (sv ** 2).sum())


def report(name, protos):
    """protos: [K, D]"""
    pn = F.normalize(protos, dim=-1)
    C = pn @ pn.T
    K = C.shape[0]
    off = C[~torch.eye(K, dtype=bool)]
    sv = torch.linalg.svdvals(protos - protos.mean(0, keepdim=True))
    eff = participation_ratio(sv)
    # softmax 权重熵（等价于"一个 patch 实际用到几个原型"）
    p = F.softmax(C / 0.07, dim=-1)
    ent = -(p * torch.log(p + 1e-12)).sum(-1)
    print(f"  [{name}] K={K} cos(off-diag) mean={off.mean():.4f} max={off.max():.4f} "
          f"min={off.min():.4f}")
    print(f"  [{name}] 映射后原型有效秩(participation ratio)={eff:.2f}/{K} "
          f"| softmax 熵={ent.mean():.4f} → 有效原型数≈{float(torch.exp(ent.mean())):.2f}")
    print(f"  [{name}] 原型矩阵行范数 mean={protos.norm(dim=-1).mean():.4f} "
          f"std={protos.norm(dim=-1).std():.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image', default=None, help='正常图路径；不给就用固定随机图')
    ap.add_argument('--size', type=int, default=448, help='输入尺寸(与训练一致的 448)')
    args = ap.parse_args()

    model, _ = build(args.ckpt)
    tm = model.token_mapping
    with torch.no_grad():
        raw = model.tokens_per_layer.token_dict['0'][0]          # [K, D]
        mapped = tm(raw.unsqueeze(0))[0]                          # [K, D]
    print(f"checkpoint: {args.ckpt}")
    report('映射前 token', raw)
    report('映射后原型', mapped)

    if args.image and os.path.exists(args.image):
        tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = tf(Image.open(args.image).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            out = model(img, is_training=True)
        for i, f in enumerate(out['feat_list']):
            fn = F.normalize(f[0], dim=-1)
            C = fn @ fn.T
            K = C.shape[0]
            off = C[~torch.eye(K, dtype=bool)]
            print(f"  [真实前向] feat_list[{i}] shape={tuple(f.shape)} "
                  f"token 两两 cos: mean={off.mean():.6f} max={off.max():.6f}")
            print(f"  [真实前向] 该层 token 是否逐元素相同: "
                  f"max|f0-fk|={(f[0]-f[0,0:1]).abs().max():.3e}")


if __name__ == '__main__':
    main()
