# -*- coding: utf-8 -*-
"""[路径1 关键消融] token_mapping（768→128→768 低秩瓶颈）到底是帮忙还是损害？

代码事实：`token_mapping = LayerNorm(768) → Linear(768→128) → GELU → LayerNorm(128) → Linear(128→768)`，
**没有残差/跳连**，因此映射后的特征被强制压进 ≤128 维子空间（768 维里 ~83% 的方向被丢弃）。
而异常打分是"映射后 patch ↔ 映射后原型"的余弦距离（两侧过同一个映射）。

本脚本在**同一份已训练权重、同一次前向**上比较两种打分空间（无需训练、可在 CPU 上跑）：
  raw      : 直接用融合后的 patch 特征 vs 融合后的可学习原型（等价"不加映射"的经典做法）
  mapped   : 现状（两者都过 token_mapping）
若 raw ≈ 或优于 mapped，说明这个低秩瓶颈在损害性能，应改成**残差式映射**（identity + f），
这样起点等于"原始 DINOv2 特征"，训练只能在此基础上改进。

用法（CPU 也可）：
  python probe_identity.py --ckpt results/control35/maponly/model_screw_25.pth --class screw --device cpu --limit 20
"""
import argparse
import importlib.util
import os
import sys
import time

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


def _auroc(y, s):
    """Mann–Whitney AUROC（全向量化；并列值取平均秩，避免 Python 逐值循环导致的超慢）。"""
    y = y.astype(np.uint8)
    n_pos = int(y.sum()); n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1, [ss.size]))
    avg = (bounds[:-1] + bounds[1:] + 1) / 2.0          # 每个并列组的平均秩(1-based)
    ranks = np.empty(ss.size, dtype=np.float64)
    ranks[order] = np.repeat(avg, np.diff(bounds))
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ap(y, s):
    y = y.astype(np.uint8)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float('nan')
    order = np.argsort(-s, kind='mergesort')
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, ys.size + 1)
    return float((prec * ys).sum() / n_pos)


def _f1max(y, s):
    y = y.astype(np.uint8)
    order = np.argsort(-s, kind='mergesort')
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
    return float(np.nanmax(f1))


def cpu_metrics(pr_px, pr_sp, gt_px, gt_sp):
    """CPU 版指标：I-AUROC/I-AP/I-F1/P-AUROC/P-AP/P-F1（AUPRO 需 CUDA 累加器，置 nan）。"""
    pr_sp = np.asarray(pr_sp, dtype=np.float64)
    gt_sp = np.asarray(gt_sp, dtype=np.uint8)
    px = np.asarray(pr_px, dtype=np.float32).ravel()
    gy = np.asarray(gt_px, dtype=np.uint8).ravel()
    return [_auroc(gt_sp, pr_sp), _ap(gt_sp, pr_sp), _f1max(gt_sp, pr_sp),
            _auroc(gy, px), _ap(gy, px), _f1max(gy, px), float('nan')]


def score_from(patch, proto, out_size):
    """patch [N,D], proto [K,D] → [1,1,out,out] 最近原型的余弦距离图"""
    p = F.normalize(patch, dim=-1)
    q = F.normalize(proto, dim=-1)
    s = (1.0 - p @ q.T).clamp_min(0).min(dim=-1).values
    g = int(round(s.shape[0] ** 0.5))
    return F.interpolate(s.reshape(1, 1, g, g), size=out_size, mode='bilinear', align_corners=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--protocol', type=int, default=392)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 张（计时/冒烟用；0=全部）')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_control', 'main-fewshot-control.py'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    mod.args = a
    mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(args.device).eval()

    tf, gtf = mod.get_data_transforms(448, 392)
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    if args.limit:
        ds.img_paths = ds.img_paths[:args.limit]
        ds.gt_paths = ds.gt_paths[:args.limit]
        ds.labels = ds.labels[:args.limit]
        ds.types = ds.types[:args.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    rm = None if args.protocol == 392 else args.protocol
    fused_layer = [1, 2, 3, 4, 6, 8, 11]

    variants = ['raw', 'mapped']
    acc = {v: dict(gt_px=[], pr_px=[], gt_sp=[], pr_sp=[], contrast=[]) for v in variants}
    t0 = time.time()
    with torch.no_grad():
        for img, gt, label, _ in tqdm(dl, ncols=60, desc=args.cls):
            img = img.to(args.device)
            ret = model(img, is_training=True)
            pfl, fl = ret['patch_feat_layer'], ret['feat_list']
            fp = torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
            flr = torch.sum(torch.stack(fl, dim=0), dim=0)     # [B,K,D] 可学习原型（原始空间）
            spaces = {
                'raw': (F.normalize(fp, dim=-1), F.normalize(flr, dim=-1)),
                'mapped': (model.token_mapping(fp), model.token_mapping(flr)),
            }
            for v, (pf, pr) in spaces.items():
                for b in range(img.shape[0]):
                    am = score_from(pf[b], pr[b], img.shape[-1])
                    gt_b = F.interpolate(gt, size=rm, mode='nearest') if rm is not None else gt
                    am = gk(am)
                    gtb = (gt_b[b:b + 1] > 0.5).float()
                    if gtb.shape[1] > 1:
                        gtb = torch.max(gtb, dim=1, keepdim=True)[0]
                    flat = am.flatten(1)
                    k_top = max(1, int(flat.shape[1] * 0.01))
                    sp = torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean()
                    acc[v]['gt_px'].append(gtb[0, 0].cpu().numpy())
                    acc[v]['pr_px'].append(am[0, 0].cpu().numpy())
                    acc[v]['gt_sp'].append(int(label[b]))
                    acc[v]['pr_sp'].append(float(sp))
                    if float((gtb > 0.5).sum()) > 0:
                        acc[v]['contrast'].append(float(am[gtb > 0.5].mean() - am[gtb <= 0.5].mean()))
    dt = time.time() - t0
    print(f'\n[{args.cls}] {os.path.basename(args.ckpt)}  设备={args.device}  {len(ds)} 张，用时 {dt:.0f}s')
    hdr = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    print("  " + " " * 8 + " ".join(f"{n:>8}" for n in hdr) + "   in/out对比度")
    print("  （CPU 口径：AUROC/AP/F1 由 numpy 实现；P-AUPRO 需要 CUDA 累加器，这里为 nan）")
    res = {}
    for v in variants:
        a_ = acc[v]
        r = cpu_metrics(np.stack(a_['pr_px']), np.array(a_['pr_sp']),
                        np.stack(a_['gt_px']), np.array(a_['gt_sp']))
        res[v] = r
        print(f"  {v:<8}" + " ".join(f"{x:8.4f}" for x in r) +
              f"   {np.mean(a_['contrast']):.4f}")
    if len(res) == 2:
        d = [b - a for a, b in zip(res['raw'], res['mapped'])]
        print("  Δ(raw-)... ", " ".join(f"{x:+8.4f}" for x in d))
    if args.out:
        with open(args.out, 'a') as fh:
            for v in variants:
                fh.write(f"{args.cls},{os.path.basename(args.ckpt)},{args.device},{v}," +
                         ",".join(f"{x:.4f}" for x in res[v]) + "\n")


if __name__ == '__main__':
    main()
