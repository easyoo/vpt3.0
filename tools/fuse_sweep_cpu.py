# -*- coding: utf-8 -*-
"""[CPU 版融合扫描] 只读磁盘缓存的两个分支异常图，一次遍历算完所有融合规则。
  * 用 numpy 指标（AUROC/AP/F1；P-AUPRO 需要 CUDA 累加器，此处 nan）
  * 归一化一律按"类别内"（整个测试集全部像素），符合要求
用法: python fuse_sweep_cpu.py [--rules a,b,...] [--mapping_source improved] [--out x.csv]
"""
import argparse, glob, json, os, re, statistics as st, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
ALL = "carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper".split()
sys.path.insert(0, HERE)
from probe_identity import cpu_metrics

RULES = ['mapping_only','bank_only','sum','prod','max','zsum','zprod','zmax',
         'mm_sum','mm_prod','mm_max','rank_sum','rank_prod',
         'wsum:0.3','wsum:0.5','wsum:0.7','zwsum:0.3','zwsum:0.5','zwsum:0.7']

def apply_rule(rule, m, b):
    z = lambda x: (x - x.mean()) / (x.std() + 1e-8)                 # 类别内
    mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)       # 类别内 min-max
    def rk(x):
        f = x.reshape(-1); order = np.argsort(f, kind='stable')
        r = np.empty(f.size, dtype=np.float64); r[order] = np.arange(1, f.size + 1)
        return (r / (f.size - 1)).reshape(x.shape)
    if rule == 'mapping_only': return m
    if rule == 'bank_only': return b
    if rule == 'sum': return m + b
    if rule == 'prod': return m * b
    if rule == 'max': return np.maximum(m, b)
    if rule == 'zsum': return z(m) + z(b)
    if rule == 'zprod': return z(m) * z(b)
    if rule == 'zmax': return np.maximum(z(m), z(b))
    if rule == 'mm_sum': return mm(m) + mm(b)
    if rule == 'mm_prod': return mm(m) * mm(b)
    if rule == 'mm_max': return np.maximum(mm(m), mm(b))
    if rule == 'rank_sum': return rk(m) + rk(b)
    if rule == 'rank_prod': return rk(m) * rk(b)
    if rule.startswith('wsum:'):
        w = float(rule.split(':')[1]); return w * m + (1 - w) * b
    if rule.startswith('zwsum:'):
        w = float(rule.split(':')[1]); return w * z(m) + (1 - w) * z(b)
    raise ValueError(rule)

def gauss_kernel(k=5, sigma=4.0):
    """与项目 get_gaussian_kernel 完全一致的核（对称、归一化）"""
    x = np.arange(k)
    xg, yg = np.meshgrid(x, x, indexing='ij')
    m = (k - 1) / 2.0
    kern = (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-(((xg - m) ** 2 + (yg - m) ** 2)) / (2 * sigma ** 2))
    return kern / kern.sum()


def gauss_np(am, kern=None):
    """与 PyTorch Conv2d(padding=k//2) 等价的零填充相关（核对称→相关=卷积）"""
    from scipy.ndimage import convolve
    if kern is None:
        kern = gauss_kernel()
    if am.ndim == 2:
        return convolve(am, kern, mode='constant', cval=0.0)
    return np.stack([convolve(x, kern, mode='constant', cval=0.0) for x in am], 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='all')
    ap.add_argument('--rules', default=','.join(RULES))
    ap.add_argument('--mapping_source', default='improved', choices=['mapping', 'improved'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit_classes', type=int, default=0)
    args = ap.parse_args()
    classes = ALL if args.classes == 'all' else args.classes.split(',')
    if args.limit_classes: classes = classes[:args.limit_classes]
    rules = args.rules.split(',')
    acc = {r: {} for r in rules}
    for cls in classes:
        mp = os.path.join(HERE, '.maps_audit', f'{cls}_{args.mapping_source}.map')
        bp = os.path.join(HERE, '.errmap_audit', f'{cls}.map')
        gp = os.path.join(HERE, '.maps_audit', f'{cls}_gt.map')
        meta = os.path.join(HERE, '.maps_audit', f'{cls}_meta.json')
        if not all(os.path.exists(p) for p in (mp, bp, gp, meta)):
            print(f'[skip] {cls} 缺缓存'); continue
        m = torch.load(mp, weights_only=False).numpy()[:, 0]
        b = torch.load(bp, weights_only=False).numpy()[:, 0]
        gt = (torch.load(gp, weights_only=False).numpy()[:, 0] > 0.5).astype(np.uint8)
        labels = np.array(json.load(open(meta))['labels'], dtype=np.uint8)
        for r in rules:
            am = apply_rule(r, m, b).astype(np.float32)
            am = gauss_np(am)
            flat = am.reshape(am.shape[0], -1)
            k_top = max(1, int(flat.shape[1] * 0.01))
            sp = -np.sort(-flat, axis=1)[:, :k_top].mean(axis=1)
            acc[r][cls] = cpu_metrics(am, sp, gt, labels)[:6]   # 不含 AUPRO
        print(f'  {cls} ok')
    print(f"\n{'规则':<16}{'类数':>4}" + ''.join(f'{n:>10}' for n in NAMES[:6]))
    rows = []
    for r in rules:
        have = sorted(acc[r])
        if not have: continue
        mean = [st.mean([acc[r][c][i] for c in have]) for i in range(6)]
        rows.append((r, len(have), mean))
    for r, n, mean in sorted(rows, key=lambda x: -x[2][4]):
        print(f"{r:<16}{n:>4}" + ''.join(f'{v:10.4f}' for v in mean))
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write('rule,n,' + ','.join(NAMES[:6]) + '\n')
            for r, n, mean in rows:
                fh.write(f'{r},{n},' + ','.join(f'{v:.4f}' for v in mean) + '\n')

if __name__ == '__main__':
    main()
