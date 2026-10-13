# -*- coding: utf-8 -*-
"""[路径2 候选 · 零训练] 打分图归一化/聚合方式的对照实验。

依据失败样本分析（`probe_failures.py`，screw@10ep）：
  * 正常图 sp_score ∈ [0.06, 0.106]，异常图 sp_score ∈ [0.059, ~0.2]；
    两者**严重重叠**，最难的缺陷类型（scratch_head 0.103、manipulated_front 0.112）几乎落在正常区间内；
  * 这些类型的 in/out-GT 对比度只有 0.07 左右（好类型 thread_neck 是 0.22）→ 打分图**弥散**，
    被"整图统一的基线距离"淹没，而不是被真正定位出来。

因此本脚本比较"相对化"打分：把 patch 的绝对距离换成"相对本图/本邻域"的残差。
  base      : d（现状）
  nb        : 3x3 邻域平均（PatchCore 邻域聚合）
  rel_med   : d − median(d)            ← 去整图基线
  rel_nb    : d − 3x3 邻域均值          ← 局部对比度（本文候选）
  rel_nb+med: rel_nb 后再减中位数
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def build(mod, ckpt, fusion, device):
    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.fusion = fusion
    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def map28(model, imgs, fused_layer):
    ret = model(imgs, is_training=True)
    pfl, fl = ret['patch_feat_layer'], ret['feat_list']
    fp = model.layer_fusion(pfl, fused_layer) if hasattr(model, 'layer_fusion') \
        else torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
    flr = torch.sum(torch.stack(fl, dim=0), dim=0)
    p = F.normalize(model.token_mapping(fp), dim=-1)
    q = F.normalize(model.token_mapping(flr), dim=-1)
    s = (1.0 - torch.bmm(p, q.transpose(1, 2))).clamp_min(0).min(dim=-1).values
    g = int(round(s.shape[1] ** 0.5))
    return s.reshape(-1, g, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--fusion', default='sum', choices=['sum', 'norm'])
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--protocol', type=int, default=392)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_norm', 'main-fewshot-norm.py'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))  # 用 --script 所在目录的 dinov2 副本
    mod = load_mod(args.script)
    device = args.device
    fused_layer = [1, 2, 3, 4, 6, 8, 11]
    model = build(mod, args.ckpt, args.fusion, device)

    tf, gtf = mod.get_data_transforms(448, 392)
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rm = None if args.protocol == 392 else args.protocol
    variants = ['base', 'nb', 'rel_med', 'rel_nb', 'rel_nb+med']
    acc = {v: dict(gt_px=[], pr_px=[], gt_sp=[], pr_sp=[]) for v in variants}

    for img, gt, label, _ in tqdm(dl, ncols=60, desc=args.cls):
        img = img.to(device)
        m28 = map28(model, img, fused_layer)                     # [B,28,28]
        nb = F.avg_pool2d(m28.unsqueeze(1), 3, stride=1, padding=1)[:, 0]
        med = m28.flatten(1).median(dim=1).values.view(-1, 1, 1)
        d = {
            'base': m28,
            'nb': nb,
            'rel_med': m28 - med,
            'rel_nb': m28 - nb,
            'rel_nb+med': (m28 - nb) - (m28 - nb).flatten(1).median(dim=1).values.view(-1, 1, 1),
        }
        for v in variants:
            mm = d[v]
            am = F.interpolate(mm.unsqueeze(1), size=img.shape[-1] if rm is None else rm,
                               mode='bilinear', align_corners=True)
            gt_b = F.interpolate(gt, size=am.shape[-1], mode='nearest') if rm is not None else gt
            am = gk(am)
            gt_b = (gt_b > 0.5).float()
            if gt_b.shape[1] > 1:
                gt_b = torch.max(gt_b, dim=1, keepdim=True)[0]
            flat = am.flatten(1)
            k_top = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean(dim=1)
            acc[v]['gt_px'].append(gt_b[:, 0].cpu().numpy())
            acc[v]['pr_px'].append(am[:, 0].cpu().numpy())
            acc[v]['gt_sp'].append(label.numpy())
            acc[v]['pr_sp'].append(sp.cpu().numpy())

    names = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    print(f"\n[{args.cls}] {os.path.basename(args.ckpt)} fusion={args.fusion} protocol={args.protocol}")
    print("  " + " " * 14 + " ".join(f"{n:>8}" for n in names))
    res = {}
    for v in variants:
        a = acc[v]
        r = mod.ader_evaluator(np.concatenate(a['pr_px']), np.concatenate(a['pr_sp']),
                               np.concatenate(a['gt_px']), np.concatenate(a['gt_sp']))
        res[v] = r
        print(f"  {v:<14}" + " ".join(f"{x:8.4f}" for x in r))
    if args.out:
        with open(args.out, 'a') as fh:
            for v in variants:
                fh.write(f"{args.cls},{os.path.basename(args.ckpt)},{args.fusion},{args.protocol},{v}," +
                         ",".join(f"{x:.4f}" for x in res[v]) + "\n")


if __name__ == '__main__':
    main()
