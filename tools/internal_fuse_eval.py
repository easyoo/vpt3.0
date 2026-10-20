# -*- coding: utf-8 -*-
"""[参数自由] mapping 分支**内部**两种证据的合成方式消融（离线、用分量网格）。

现状：m = w·s_learn + (1-w)·s_raw（加权和，w=0.6 是测试集上选过的）。
候选（无超参）：
  sum  : 现状加权和
  geo  : sqrt(s_learn · s_raw)（几何平均；乘性"与"）
  max  : max(s_learn, s_raw)（取大；任一证据强即异常）
  min  : min(...)（取小；两者都强才算）
  prod : softplus 乘积（概率式"与"）

用法：
  CUDA_VISIBLE_DEVICES=<uuid> python internal_fuse_eval.py --grids .grids3_mv2 \
      --tag intfuse_mv2 --raw d1:q512:0.5 --learn dlmin:bq64:0.7 --w 0.6
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_comp import CompEval          # noqa: E402

NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--raw', default='d1:q512:0.5')
    ap.add_argument('--learn', default='dlmin:bq64:0.7')
    ap.add_argument('--w', type=float, default=0.6)
    ap.add_argument('--views', default='0')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    import importlib.util
    p = os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')
    sys.path.insert(0, os.path.dirname(p))
    sp = importlib.util.spec_from_file_location('audit_main_if', p)
    mod = importlib.util.module_from_spec(sp)
    sys.modules['audit_main_if'] = mod
    sp.loader.exec_module(mod)
    views = [int(c) for c in args.views]

    rl, rg, a = args.raw.split(':'); a = float(a)
    ll, lg, b = args.learn.split(':'); b = float(b)
    variants = ['sum', 'geo', 'max', 'min', 'prod']
    out = []
    for f in sorted(glob.glob(os.path.join(args.grids, '*.npz'))):
        cls = os.path.basename(f)[:-4]
        ge = CompEval(f)
        gt = ge.gt.to(torch.uint8)
        tot = {v: None for v in variants}
        for vi in views:
            rs = CompEval._pow(ge.comp('raw', rl, vi), ge.comp('raw', rg, vi), a)
            ls = CompEval._pow(ge.comp('learn', ll, vi), ge.comp('learn', lg, vi), b)
            sp_ = lambda t: F.softplus(t)
            maps = {
                'sum': args.w * ls + (1 - args.w) * rs,
                'geo': (ls.clamp_min(1e-6).sqrt() * rs.clamp_min(1e-6).sqrt()),
                'max': torch.maximum(ls, rs),
                'min': torch.minimum(ls, rs),
                'prod': sp_(ls) * sp_(rs),
            }
            for v in variants:
                tot[v] = maps[v] if tot[v] is None else tot[v] + maps[v]
        for v in variants:
            am = tot[v] / len(views)
            flat = am.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            spx = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(1).numpy()
            r = list(mod.ader_evaluator(am[:, 0].numpy(), spx, gt.numpy().astype(np.uint8), ge.label))
            out.append([cls, v] + [x * 100 for x in r])
        del ge
        print(f'  [{cls}] done', flush=True)

    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    path = os.path.join(HERE, 'logs', f'{args.tag}.csv')
    with open(path, 'w') as fh:
        for row in out:
            fh.write(','.join([row[0], row[1]] + [f'{v:.4f}' for v in row[2:]]) + '\n')
    print('saved', path)
    print(f"\n{'内部合成':<8}{'类':>3}" + ''.join(f'{n:>9}' for n in NAMES) + f"{'7avg':>9}")
    for v in variants:
        rows = [r[2:] for r in out if r[1] == v]
        avg = [float(np.mean([r[i] for r in rows])) for i in range(7)]
        print(f"{v:<8}{len(rows):>3}" + ''.join(f'{x:>9.2f}' for x in avg) + f"{sum(avg)/7:>9.2f}")


if __name__ == '__main__':
    main()
