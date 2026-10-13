# -*- coding: utf-8 -*-
"""[融合评估] 只读磁盘上缓存的两个分支异常图，按任意规则融合后算官方指标。

缓存由 `cache_maps.py` 生成：
  .maps_audit/<cls>_mapping.map   （mapping 分支）
  .errmap_audit/<cls>.map         （原版 MemoryBank）
  .maps_audit/<cls>_gt.map        （GT，与上两者同顺序）
因为三者来自同一次 DataLoader 遍历，**逐图严格对齐**（同一测试样本）。

支持的融合规则：
  sum        直接相加（等同原版 type=5）
  prod       逐像素相乘
  max        逐像素取最大
  zsum       各自按"整数据集"标准化后相加
  zprod      各自按"整数据集"标准化后相乘
  wsum:w     加权相加 w*mapping + (1-w)*bank（如 wsum:0.3）
  zsum_img   各自按"每张图"标准化后相加
  rank_sum   各自按"每张图"秩归一化后相加

用法：
  python fuse_eval.py --classes carpet grid --rule sum
  python fuse_eval.py --classes all --rule prod
"""
import argparse
import glob
import importlib.util
import os
import statistics as st
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
ALL = "carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper".split()


def load_mod(script_path):
    spec = importlib.util.spec_from_file_location('audit_main', script_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def fuse(rule, m, b):
    m = m.float(); b = b.float()
    def z_cls(x):      # 按"类别内"（整个测试集全部像素）标准化 —— 用户要求的口径
        return (x - x.mean()) / (x.std() + 1e-8)
    def mm_cls(x):     # 按类别内 min-max 归一到 [0,1]
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    def rk_cls(x):     # 按类别内秩归一化到 [0,1]
        f = x.reshape(-1)
        r = torch.argsort(torch.argsort(f)).float()
        return (r / (f.numel() - 1)).reshape_as(x)
    if rule == 'sum':
        return m + b
    if rule in ('mapping_only', 'improved_only'):
        return m
    if rule == 'bank_only':
        return b
    if rule == 'prod':
        return m * b
    if rule == 'max':
        return torch.maximum(m, b)
    if rule == 'zsum':
        return z_cls(m) + z_cls(b)
    if rule == 'zprod':
        return z_cls(m) * z_cls(b)
    if rule == 'zmax':
        return torch.maximum(z_cls(m), z_cls(b))
    if rule.startswith('zwsum:'):
        w = float(rule.split(':')[1]); return w * z_cls(m) + (1 - w) * z_cls(b)
    if rule == 'mm_sum':
        return mm_cls(m) + mm_cls(b)
    if rule == 'mm_prod':
        return mm_cls(m) * mm_cls(b)
    if rule == 'mm_max':
        return torch.maximum(mm_cls(m), mm_cls(b))
    if rule == 'rank_sum':
        return rk_cls(m) + rk_cls(b)
    if rule == 'rank_prod':
        return rk_cls(m) * rk_cls(b)
    if rule.startswith('wsum:'):
        w = float(rule.split(':')[1]); return w * m + (1 - w) * b
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='all')
    ap.add_argument('--rule', default='sum')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_control', 'main-fewshot-control.py'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--mapping_source', default='mapping', choices=['mapping', 'improved'],
                    help="用哪张 mapping 图: 'mapping'=现状(type1), 'improved'=改善后(融合+1.5x+翻转)")
    ap.add_argument('--map_tag', default=None,
                    help='直接指定 mapping 缓存的标签：.maps_audit/<cls>_<map_tag>.map（如 improved_m4m6r3）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    classes = ALL if args.classes == 'all' else args.classes.split(',')

    rows = {}
    for cls in classes:
        src = 'improved' if getattr(args, 'mapping_source', 'mapping') == 'improved' else 'mapping'
        tag = args.map_tag if args.map_tag else src
        mp = os.path.join(HERE, '.maps_audit', f'{cls}_{tag}.map')
        bp = os.path.join(HERE, '.errmap_audit', f'{cls}.map')
        gp = os.path.join(HERE, '.maps_audit', f'{cls}_gt.map')
        if not (os.path.exists(mp) and os.path.exists(bp) and os.path.exists(gp)):
            print(f'[skip] {cls}: 缺缓存'); continue
        m = torch.load(mp, weights_only=False)
        b = torch.load(bp, weights_only=False)
        gt = torch.load(gp, weights_only=False).float()
        assert m.shape == b.shape == gt.shape, (cls, m.shape, b.shape, gt.shape)
        am = fuse(args.rule, m, b).to(args.device)
        gt = gt.to(args.device)
        am = gk(am)
        pr_px = am[:, 0].cpu().numpy(); gt_px = (gt[:, 0] > 0.5).cpu().numpy()
        flat = am.flatten(1); k_top = max(1, int(flat.shape[1] * 0.01))
        pr_sp = torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean(dim=1).cpu().numpy()
        # 图像级标签从 meta 读，保证与缓存顺序一致
        import json
        meta = json.load(open(os.path.join(HERE, '.maps_audit', f'{cls}_meta.json')))
        gt_sp = np.array(meta['labels'])
        r = mod.ader_evaluator(pr_px, pr_sp, gt_px, gt_sp)
        rows[cls] = r
        print(f'  {cls:<11}' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r)))
    if rows:
        mean = [st.mean([rows[c][i] for c in rows]) for i in range(7)]
        print(f'\n[{args.rule}] {len(rows)} 类平均: ' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, mean)))
        if args.out:
            with open(args.out, 'a') as fh:
                fh.write(f'{args.rule},{len(rows)},' + ','.join(f'{v:.4f}' for v in mean) + '\n')


if __name__ == '__main__':
    main()
