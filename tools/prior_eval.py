# -*- coding: utf-8 -*-
"""[公平口径量化] 用**分量网格**离线重建 mapping 图（448 全分辨率），与**缓存的 bank 图**融合，
给出官方 adeval 的 7 项指标 —— 用于回答："把测试集上选出来的打分超参换成先验取值，掉多少？"

候选 spec 形如  raw_local:raw_global:alpha | learn_local:learn_global:beta | view_w
  e.g.  d1:q512:0.5|dlmin:bq64:0.7|0.6     ← 当前生产配方（α/β/视图权重都是测试集上选的）
        d1:q512:0.5|dlmin:bq64:0.5|0.5     ← 先验配方（两侧都用几何平均、视图等权）

用法：
  CUDA_VISIBLE_DEVICES=<uuid> python prior_eval.py --grids .grids3_mv2 \
      --bank_dir .errmap_nocrop448_2shot --tag prior_mv2 --fuse_w 0.5 \
      --specs "d1:q512:0.5|dlmin:bq64:0.7|0.6,d1:q512:0.5|dlmin:bq64:0.5|0.5"
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_comp import CompEval, parse_spec          # noqa: E402

NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def adeval(mod, am, gt, label):
    """am: [N,1,H,W] torch；gt: [N,H,W] uint8（与 am 同分辨率）"""
    flat = am.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    sp = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(1).numpy()
    return list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt.numpy().astype(np.uint8), label))


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', required=True)
    ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--specs', required=True)
    ap.add_argument('--fuse_w', type=float, default=0.5)
    ap.add_argument('--views', default='0', help="参与平均的视图，如 '0'（无翻转，v3.7 口径）或 '01'")
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    views = [int(c) for c in args.views]

    import importlib.util
    p = os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')
    sys.path.insert(0, os.path.dirname(p))
    spec = importlib.util.spec_from_file_location('audit_main_p', p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_main_p'] = mod
    spec.loader.exec_module(mod)

    specs = [parse_spec(t) for t in args.specs.split(',')]
    files = sorted(glob.glob(os.path.join(args.grids, '*.npz')))
    out = []
    for f in files:
        cls = os.path.basename(f)[:-4]
        ge = CompEval(f)
        gt = ge.gt.to(torch.uint8)
        bank_path = os.path.join(HERE, args.bank_dir, f'{cls}.map')
        bank = None
        if os.path.exists(bank_path):
            bank = torch.load(bank_path, weights_only=False).float()
            assert bank.shape[0] == gt.shape[0], (cls, bank.shape, gt.shape)
        for s in specs:
            (rl, rg, a), (ll, lg, b), w = s
            tot = None
            for vi in views:
                rs = CompEval._pow(ge.comp('raw', rl, vi), ge.comp('raw', rg, vi), a)
                ls = CompEval._pow(ge.comp('learn', ll, vi), ge.comp('learn', lg, vi), b)
                amv = w * ls + (1.0 - w) * rs
                tot = amv if tot is None else tot + amv
            m = tot / len(views)
            label = f'd1^{s[0][2]}{s[0][1]}|dlmin^{s[1][2]}{s[1][1]}|vw{s[2]}'
            with torch.no_grad():
                r_map = adeval(mod, m, gt, ge.label)
            out.append([cls, label, 'mapping'] + r_map)
            if bank is not None:
                fused = args.fuse_w * z_cls(m) + (1 - args.fuse_w) * z_cls(bank[:m.shape[0]])
                with torch.no_grad():
                    r_fus = adeval(mod, fused, gt, ge.label)
                out.append([cls, label, 'fused'] + r_fus)
        del ge
        print(f'  [{cls}] done', flush=True)

    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    path = os.path.join(HERE, 'logs', f'{args.tag}.csv')
    with open(path, 'w') as fh:
        for row in out:
            fh.write(','.join([row[0], row[1], row[2]] + [f'{v:.4f}' for v in row[3:]]) + '\n')
    print('saved', path, flush=True)

    # 汇总
    import collections
    acc = collections.defaultdict(list)
    for row in out:
        acc[(row[1], row[2])].append(row[3:])
    print(f"\n{'spec':<34}{'分支':<9}{'类':>3}" + ''.join(f'{n:>9}' for n in NAMES) + f"{'7avg':>9}")
    for (lb, br), v in acc.items():
        a = [float(np.mean([r[i] for r in v])) * 100 for i in range(7)]
        print(f"{lb:<34}{br:<9}{len(v):>3}" + ''.join(f'{x:>9.2f}' for x in a) + f"{sum(a)/7:>9.2f}")


if __name__ == '__main__':
    main()
