# -*- coding: utf-8 -*-
"""审计：提升的统计显著性（按类别配对 bootstrap）。
配对设计：同一类、同一权重下，旧打分图 vs 新打分图 -> 逐类 Δ -> bootstrap 均值与 95% CI。
不确定性只来自"类别抽样"；训练噪声被配对消掉。
"""
import collections
import glob
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
VISA = "candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum".split()
MV = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()


def load(pat):
    d = collections.defaultdict(dict)
    for f in glob.glob(pat):
        for line in open(f, errors='ignore'):
            if line.strip().startswith('CSV,'):
                p = line.strip().split(',')
                if len(p) == 10:
                    d[p[1]][p[2]] = [float(x) * 100 for x in p[3:]]
    return d


def paired(tagA, patA, tagB, patB, order, n_boot=20000, seed=0):
    A, B = load(patA), load(patB)
    cls = [c for c in order if c in A and 'fused' in A[c] and c in B and 'fused' in B[c]]
    if not cls:
        print(f"{tagA} vs {tagB}: 无配对数据")
        return
    outcomes = []
    for tag, idxs in (('AUROC项(P-AUPRO)', [6]), ('6项均值', list(range(6))),
                      ('I-AUROC', [0]), ('P-AP', [4])):
        dA = np.array([sum(A[c]['fused'][i] for i in idxs) / len(idxs) for c in cls])
        dB = np.array([sum(B[c]['fused'][i] for i in idxs) / len(idxs) for c in cls])
        d = dB - dA
        rng = np.random.default_rng(seed)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {tag:<16} Δ={d.mean():+7.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  "
              f"P(Δ>0)={float((boot > 0).mean()):.3f}  逐类为正 {int((d > 0).sum())}/{len(cls)}")
    print(f"  [{tagA}  ->  {tagB}]")


if __name__ == '__main__':
    print("=== VisA 4-shot ===")
    paired('旧min+w0.3(交付)', 'logs/v3fuse_v3visa4_*.log',
           '新打分+w0.5', 'logs/v3fw0.5_visa4_*.log', VISA)
    paired('旧min+w0.5', 'logs/oldw5vw0.5_*.log',
           '新打分+w0.5', 'logs/v3fw0.5_visa4_*.log', VISA)
    print("=== MVTec 2-shot ===")
    paired('旧min+w0.3', 'logs/v3fuse_v3mv2_*.log',
           '新打分+w0.5', 'logs/v3fw5mw0.5_*.log', MV)
    paired('旧min+w0.5', 'logs/oldw5mw0.5_*.log',
           '新打分+w0.5', 'logs/v3fw5mw0.5_*.log', MV)
    print("=== MVTec 4-shot ===")
    paired('旧min+w0.3', 'logs/v3fuse_v3mv4_*.log',
           '新打分+w0.5', 'logs/v3fw5fw0.5_*.log', MV)
    paired('旧min+w0.5', 'logs/oldw5fw0.5_*.log',
           '新打分+w0.5', 'logs/v3fw5fw0.5_*.log', MV)
