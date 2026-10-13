# -*- coding: utf-8 -*-
"""推理侧消融的逐类结果 + 按类配对 bootstrap（官方 7 指标）。

用法: CUDA_VISIBLE_DEVICES=<uuid> python ablate_infer_boot.py --grids .grids3_mv2 \
        --specs "A,B" --out logs/xxx.txt
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_comp import CompEval, parse_spec      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', required=True)
    ap.add_argument('--specs', required=True, help='逗号分隔，第 1 个作为基线')
    ap.add_argument('--out', default='')
    ap.add_argument('--n_boot', type=int, default=20000)
    args = ap.parse_args()
    specs = [parse_spec(t) for t in args.specs.split(',')]
    files = sorted(glob.glob(os.path.join(args.grids, '*.npz')))
    rows = {i: {} for i in range(len(specs))}
    for f in files:
        ge = CompEval(f)
        cls = os.path.basename(f)[:-4]
        for i, s in enumerate(specs):
            r = ge.metrics(s, stride=1, official=True)
            rows[i][cls] = [x * 100 for x in r]
        del ge
    cls_all = sorted(rows[0])
    lines = [f"# {args.grids} 官方 adeval 逐类 7 项平均（{len(cls_all)} 类）"]
    hdr = f"{'class':<12}" + "".join(f"{'spec'+str(i):>10}" for i in range(len(specs)))
    lines.append(hdr)
    for c in cls_all:
        lines.append(f"{c:<12}" + "".join(
            f"{sum(rows[i][c])/7:>10.2f}" if c in rows[i] else f"{'-':>10}" for i in range(len(specs))))
    for i in range(1, len(specs)):
        d = np.array([sum(rows[i][c]) / 7 - sum(rows[0][c]) / 7
                      for c in cls_all if c in rows[i]])
        rng = np.random.default_rng(0)
        b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
        lo, hi = np.percentile(b, [2.5, 97.5])
        lines.append(f"\nspec0 → spec{i}:  Δ均值={d.mean():+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  "
                     f"P(Δ>0)={float((b>0).mean()):.3f}  逐类为正 {int((d>0).sum())}/{len(d)}")
    out = "\n".join(lines)
    print(out)
    if args.out:
        open(args.out, 'w').write(out + "\n")


if __name__ == '__main__':
    main()
