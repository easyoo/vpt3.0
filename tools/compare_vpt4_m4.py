# -*- coding: utf-8 -*-
"""[对比] vpt4 冒烟类 vs vpt3.0 mapping 基线（同一打分协议、同一轮次 35）"""
import os, re, sys, statistics as st
BASE = {   # vpt3.0 mapping 分支逐类基线（CPU 指标；P-AUPRO 需 GPU，见 fuse_rules_improved_GPU.csv 15类均值）
 'transistor':[0.7462,0.7001,0.6909,0.8841,0.3962,0.4614],
 'screw'     :[0.8768,0.9592,0.8824,0.9772,0.4126,0.4690],
 'capsule'   :[0.8899,0.9750,0.9237,0.9713,0.4528,0.4918],
 'metal_nut' :[1.0000,1.0000,1.0000,0.9498,0.7314,0.7710],
 'wood'      :[1.0000,1.0000,1.0000,0.9579,0.6478,0.6392],
}
NAMES=["I-AUROC","I-AP","I-F1","P-AUROC","P-AP","P-F1"]
def read(path):
    tag=None; out={}
    if not os.path.exists(path): return out
    for ln in open(path):
        s=ln.strip()
        if s.startswith('['): tag=s[1:].split(']')[0]
        elif s.startswith('CSV:') and tag: out[tag]=[float(x) for x in s[4:].split(',')[-7:]]
    return out
rows={}
for cls in BASE:
    d=read(f'results/vpt4/eval_m4_{cls}.log')
    if 'vpt4m4' in d: rows[cls]=d['vpt4m4']
print(f"{'class':<11}"+''.join(f'{n:>10}' for n in NAMES)+"   (Δ 为该指标相对基线)")
for cls in BASE:
    if cls not in rows: print(f'{cls:<11} 尚无结果'); continue
    a=rows[cls]; b=BASE[cls]
    print(f"{cls:<11}"+''.join(f'{a[i]:10.4f}' for i in range(6))+"   "+''.join(f'{(a[i]-b[i]):+8.4f}' for i in range(6)))
if rows:
    mean=[st.mean([rows[c][i] for c in rows]) for i in range(6)]
    bmean=[st.mean([BASE[c][i] for c in rows]) for i in range(6)]
    print(f"\n{len(rows)} 类均值     "+''.join(f'{mean[i]:10.4f}' for i in range(6))+"   "+''.join(f'{(mean[i]-bmean[i]):+8.4f}' for i in range(6)))
    print(f"相对提升(%)    "+''.join(f'{100*(mean[i]-bmean[i])/max(bmean[i],1e-9):+10.2f}' for i in range(6)))
