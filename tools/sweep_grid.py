# -*- coding: utf-8 -*-
"""[打分搜索框架 · 第3步] 跨类枚举打分组合，按多类平均排序。

先按 CPU 六指标（AUROC/AP/F1 × 图像/像素，AUPRO 置 nan）快速筛，
再对 Top-K 用官方 adeval（含 AUPRO）复算，避免被代理指标误导。

用法:
  python sweep_grid.py --grids .grids_visa4 --topk 15
  python sweep_grid.py --grids .grids_visa4 --official_topk 6     # 需 CUDA
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_grid import GridEval, NAMES          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', default='.grids_visa4')
    ap.add_argument('--learn_ops', default='min,geom64,bgeom,b03,b07,bmin,bbar,bn5')
    ap.add_argument('--raw_ops', default='min,kn5,kn8,geom,g03,g07,gn5,gmed,dbar')
    ap.add_argument('--ws', default='0.2,0.3,0.4,0.5,0.6,0.7,0.8')
    ap.add_argument('--topk', type=int, default=15)
    ap.add_argument('--official_topk', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.grids, '*.npz')))
    if not files:
        raise SystemExit(f'{args.grids} 下没有 npz')
    combos = list(itertools.product(args.learn_ops.split(','), args.raw_ops.split(','),
                                    [float(x) for x in args.ws.split(',')]))
    acc = {c: [] for c in combos}
    per_class = {c: {} for c in combos}
    for f in files:
        ge = GridEval(f)
        cls = os.path.basename(f)[:-4]
        for c in combos:
            r = ge.metrics(*c, official=False)
            vals = [x * 100 for x in r]
            fin = [v for v in vals if not np.isnan(v)]
            acc[c].append(sum(fin) / len(fin))
            per_class[c][cls] = vals
        del ge
    print(f"# {len(files)} 类 · {len(combos)} 个组合 · CPU 六指标均值排序")
    print(f"{'rank':<5}{'learn':<8}{'raw':<7}{'w':<5}{'6指标均值':>10}  "
          f"（AUPRO 不在内，需用 --official 复核）")
    ranked = sorted(combos, key=lambda c: -float(np.mean(acc[c])))
    for i, c in enumerate(ranked[:args.topk], 1):
        print(f"{i:<5}{c[0]:<8}{c[1]:<7}{c[2]:<5.1f}{np.mean(acc[c]):>10.2f}")

    if args.official_topk:
        print(f"\n=== 官方 adeval 复核 Top-{args.official_topk}（含 AUPRO）===")
        print(f"{'learn':<8}{'raw':<7}{'w':<5}" + "".join(f"{m:>9}" for m in NAMES)
              + f"{'7avg':>9}   '对比基线'")
        ges = {f: GridEval(f, device='cuda:0') for f in files}
        base = None
        for i, c in enumerate(ranked[:args.official_topk], 1):
            rows = [ges[f].metrics(*c, official=True) for f in files]
            a = [float(np.mean([r[k] for r in rows])) * 100 for k in range(7)]
            if c == ('min', 'min', 0.5):
                base = a
            print(f"{c[0]:<8}{c[1]:<7}{c[2]:<5.1f}" + "".join(f"{v:>9.2f}" for v in a)
                  + f"{sum(a)/7:>9.2f}")
        # 基线单独补一行（即使没进 Top-K）
        if base is None:
            b = ('min', 'min', 0.5)
            rows = [ges[f].metrics(*b, official=True) for f in files]
            base = [float(np.mean([r[k] for r in rows])) * 100 for k in range(7)]
        print(f"{'min':<8}{'min':<7}{0.5:<5.1f}" + "".join(f"{v:>9.2f}" for v in base)
              + f"{sum(base)/7:>9.2f}   ← 基线")

    if args.out:
        with open(args.out, 'w') as fh:
            fh.write('learn\traw\tw\t6avg\n')
            for c in ranked:
                fh.write(f"{c[0]}\t{c[1]}\t{c[2]}\t{np.mean(acc[c]):.4f}\n")
        print(f"\n写出 {args.out}")


if __name__ == '__main__':
    main()
