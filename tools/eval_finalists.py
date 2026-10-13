# -*- coding: utf-8 -*-
"""[打分搜索框架 · 第4步] 对指定组合用**官方 adeval（含 AUPRO、全分辨率）**算出 7 项指标，
跨类平均，并与基线 (min,min,w=0.5) 对比。

用法:
  CUDA_VISIBLE_DEVICES=<uuid> python eval_finalists.py --grids .grids2_visa4 \
      --combos "min:min:0.5,min:g512_0.3:0.5,b64_0.7:g512_0.3:0.5"
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_grid import GridEval, NAMES          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', default='.grids2_visa4')
    ap.add_argument('--combos', required=True, help='learn:raw:w, 逗号分隔')
    ap.add_argument('--save', default='')
    ap.add_argument('--per_class', type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.grids, '*.npz')))
    combos = []
    for c in args.combos.split(','):
        lo, ro, w = c.split(':')
        combos.append((lo, ro, float(w)))
    print(f"# {os.path.basename(args.grids)}  {len(files)} 类 · 官方 adeval · 全分辨率")
    rows = {}
    for f in files:
        ge = GridEval(f, device='cuda:0')
        cls = os.path.basename(f)[:-4]
        for c in combos:
            r = ge.metrics(c[0], c[1], c[2], official=True)
            rows.setdefault(c, {})[cls] = [x * 100 for x in r]
        del ge
    base = rows.get(('min', 'min', 0.5))
    print(f"{'learn':<9}{'raw':<9}{'w':<5}" + "".join(f"{m:>9}" for m in NAMES)
          + f"{'7avg':>9}{'Δ':>8}")
    lines = []
    for c in combos:
        a = [float(np.mean([rows[c][k][i] for k in rows[c]])) for i in range(7)]
        d = (sum(a) / 7 - sum(base['__x__'] if False else
                              [float(np.mean([base[k][i] for k in base])) for i in range(7)]) / 7) \
            if base else 0.0
        line = (f"{c[0]:<9}{c[1]:<9}{c[2]:<5.1f}" + "".join(f"{v:>9.2f}" for v in a)
                + f"{sum(a)/7:>9.2f}{d:>+8.2f}")
        print(line)
        lines.append(line)
        if args.per_class:
            per = ", ".join(f"{k}={sum(rows[c][k])/7:.2f}" for k in sorted(rows[c]))
            print(f"   逐类 7avg: {per}")
    if args.save:
        with open(args.save, 'w') as fh:
            fh.write(f"# {args.grids}  official adeval\n")
            fh.write(f"{'learn':<9}{'raw':<9}{'w':<5}" + "".join(f"{m:>9}" for m in NAMES)
                     + f"{'7avg':>9}\n")
            fh.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
