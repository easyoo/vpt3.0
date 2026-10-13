# -*- coding: utf-8 -*-
"""[打分搜索 · 分阶段] 先用"单支裸图"廉价定位每侧最优算子，再只对少数配对扫融合权重。

阶段 A：`w=0`（纯原始库侧）扫所有 raw 算子；`w=1`（纯可学习侧）扫所有 learn 算子。
阶段 B：把两阶段各自最优的 top-m 配对，扫 w。
筛选用像素 stride 下采样（默认 2）加速，最终按需用官方 adeval 复核。

用法：
  python sweep_stage.py --grids .grids_visa4 --stage A
  python sweep_stage.py --grids .grids_visa4 --stage B --pairs "bgeom:geom,bgeom:gn5,bmin:geom" \
      --ws 0.2,0.3,0.4,0.5,0.6,0.7,0.8
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_grid import GridEval, NAMES          # noqa: E402


def sweep(files, combos, stride):
    """按"每次只加载一个类"遍历：峰值内存 = 单类的插值缓存。返回 {combo: [逐类均值]}"""
    acc = {c: [] for c in combos}
    for f in files:
        ge = GridEval(f)
        for c in combos:
            r = ge.metrics(c[0], c[1], c[2], official=False, stride=stride)
            vals = [x * 100 for x in r]
            fin = [v for v in vals if not np.isnan(v)]
            acc[c].append(sum(fin) / len(fin))
        del ge
    return acc


def summarize(acc, combos):
    out = {}
    for c in combos:
        out[c] = float(np.mean(acc[c]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', default='.grids_visa4')
    ap.add_argument('--stage', default='A')
    ap.add_argument('--raw_ops', default='min,kn5,kn8,geom,g03,g07,gn5,gmed,dbar')
    ap.add_argument('--learn_ops', default='min,geom64,bgeom,b03,b07,bmin,bbar,bn5')
    ap.add_argument('--pairs', default='', help='A 阶段产出后填，如 bgeom:geom,b03:gn5')
    ap.add_argument('--ws', default='0.2,0.3,0.4,0.5,0.6,0.7,0.8')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--topk', type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.grids, '*.npz')))
    print(f"# {len(files)} 类，stride={args.stride} 像素下采样筛选用", flush=True)

    if args.stage.upper() == 'A':
        combos = ([('min', ro, 0.0) for ro in args.raw_ops.split(',')]
                  + [(lo, 'min', 1.0) for lo in args.learn_ops.split(',')])
        acc = sweep(files, combos, args.stride)
        print(f"\n=== 阶段 A：单支裸图（w=0 纯 raw / w=1 纯 learn）===")
        print(f"{'侧':<7}{'算子':<9}{'6指标均值':>10}")
        for c in sorted(combos, key=lambda c: -np.mean(acc[c])):
            side = 'raw' if c[2] == 0.0 else 'learn'
            op = c[1] if side == 'raw' else c[0]
            print(f"{side:<7}{op:<9}{np.mean(acc[c]):>10.2f}")
    else:
        pairs = [p.split(':') for p in args.pairs.split(',') if p]
        combos = [(lo, ro, float(ws)) for lo, ro in pairs for ws in args.ws.split(',')]
        acc = sweep(files, combos, args.stride)
        print(f"\n=== 阶段 B：{len(pairs)} 个配对 × {len(args.ws.split(','))} 个 w ===")
        print(f"{'learn':<8}{'raw':<7}{'w':<5}{'6指标均值':>10}")
        for c in sorted(combos, key=lambda c: -np.mean(acc[c]))[:args.topk]:
            print(f"{c[0]:<8}{c[1]:<7}{c[2]:<5.1f}{np.mean(acc[c]):>10.2f}")


if __name__ == '__main__':
    main()
