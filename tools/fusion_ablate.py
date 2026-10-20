# -*- coding: utf-8 -*-
"""[无超参的打分/融和变换消融] 直接用**缓存的两支打分图**评估若干**参数自由**的变换。

候选（全部不需要拟合、也没有可调超参）：
  v0 基线   : X = 0.5·z(mapping) + 0.5·z(bank)（类内 z，随后高斯平滑 + top1% 图像分）
  v1 去偏置 : X' = X − per-image median(X)   ← 去掉"每张图自身的整体偏置"
  v2 图内标准化: X' = (X − mean) / std        ← 每张图各自的均值方差
  v3 秩融合 : X = 0.5·rank(z_mapping) + 0.5·rank(z_bank)（类内像素秩，再归一化到 [0,1]）

用途：找"不需要在测试集上选参数"的增益来源。每个数据集输出一份 CSV（逐类），并打印汇总。

用法：
  CUDA_VISIBLE_DEVICES=<uuid> python fusion_ablate.py --tag fab_mv2 \
      --map_dir .maps_simple_nocrop448_mv2 --bank_dir .errmap_nocrop448_2shot \
      --cls "bottle cable ..." --data_path /home/jjquan/MVTec-AD
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    spec = importlib.util.spec_from_file_location('audit_main_fa', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main_fa'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def per_image_norm(x, mode):
    """x: [N,1,H,W]"""
    flat = x.flatten(1)                              # [N, HW]
    if mode == 'median':
        med = flat.median(dim=1, keepdim=True).values   # [N,1]
        return x - med[:, :, None, None]
    if mode == 'std':
        mu = flat.mean(dim=1, keepdim=True)
        sd = flat.std(dim=1, keepdim=True)
        return (x - mu[:, :, None, None]) / (sd[:, :, None, None] + 1e-8)
    return x


def rank_cls(x):
    """类内像素秩，归一化到 [0,1]；x: [N,1,H,W]"""
    flat = x.reshape(-1)
    order = flat.argsort().argsort().float()      # 0..N-1
    return (order / (flat.numel() - 1)).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--map_dir', required=True)
    ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--w', type=float, default=0.5)
    ap.add_argument('--variants', default='v0_baseline,v1_img_median,v2_img_std,v3_rank',
                    help='要评测的候选（逗号分隔）：v0_baseline / v1_img_median / v2_img_std / v3_rank / '
                         'v4_min / v5_max / v6_prod / v7_harm')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    variants = [s for s in args.variants.split(',') if s]
    out = []
    for cls in args.cls.split():
        M = torch.load(os.path.join(HERE, args.map_dir, f'{cls}.map'), weights_only=False).float()
        Bk = torch.load(os.path.join(HERE, args.bank_dir, f'{cls}.map'), weights_only=False).float()
        assert M.shape == Bk.shape, (cls, M.shape, Bk.shape)
        ds = mod.MVTecDataset(root=os.path.join(args.data_path, cls),
                              transform=tf, gt_transform=gtf, phase='test', sort_items=False)
        gt = torch.stack([ds[i][1] for i in range(len(ds))], 0).float()
        lab = np.array([int(ds[i][2]) for i in range(len(ds))])
        assert gt.shape[0] == M.shape[0], (cls, gt.shape, M.shape)
        gt_px = (gt[:, 0].numpy() > 0.5)

        zM, zB = z_cls(M), z_cls(Bk)
        base = args.w * zM + (1 - args.w) * zB
        sp = lambda t: F.softplus(t)                      # >0 且单调，便于做乘积/调和平均
        cands = {
            'v0_baseline': base,
            'v1_img_median': per_image_norm(base, 'median'),
            'v2_img_std': per_image_norm(base, 'std'),
            'v3_rank': args.w * rank_cls(zM) + (1 - args.w) * rank_cls(zB),
            'v4_min': torch.minimum(zM, zB),              # 两分支都高才算异常（更保守）
            'v5_max': torch.maximum(zM, zB),              # 任一分支高即算异常
            'v6_prod': sp(zM) * sp(zB),                   # 概率式"与"
            'v7_harm': 2 * sp(zM) * sp(zB) / (sp(zM) + sp(zB) + 1e-8),
        }
        for name in variants:
            X = cands[name]
            with torch.no_grad():
                am = gk(X.to(args.device)).cpu()
            flat = am.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(1).numpy()
            r = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab))
            out.append([cls, name] + [v * 100 for v in r])
            print(f'  [{cls}] {name:<14}' + '  '.join(f'{n}={v*100:.2f}' for n, v in zip(NAMES, r)), flush=True)
        del ds

    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    path = os.path.join(HERE, 'logs', f'{args.tag}.csv')
    with open(path, 'w') as fh:
        for row in out:
            fh.write(','.join([row[0], row[1]] + [f'{v:.4f}' for v in row[2:]]) + '\n')
    print('saved', path)
    print(f"\n{'variant':<16}{'类':>3}" + ''.join(f'{n:>9}' for n in NAMES) + f"{'7avg':>9}")
    for name in variants:
        rows = [r[2:] for r in out if r[1] == name]
        a = [float(np.mean([r[i] for r in rows])) for i in range(7)]
        print(f"{name:<16}{len(rows):>3}" + ''.join(f'{x:>9.2f}' for x in a) + f"{sum(a)/7:>9.2f}")


if __name__ == '__main__':
    main()
