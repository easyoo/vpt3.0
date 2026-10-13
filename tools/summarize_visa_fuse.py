#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 VisA bank+融合逐类日志(logs/visa_fuse_<SAVE>_<cls>.log)的 CSV 行。

用法: python summarize_visa_fuse.py [SAVE] [cls...]
输出: 逐类 mapping / bank / fused 七项指标 + 12 类平均 与 Δ。
"""
import sys, os, glob, re

SAVE = sys.argv[1] if len(sys.argv) > 1 else "vpt35_visa4shot"
CLS = sys.argv[2:] or [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
METRICS = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]

def read_log(cls):
    p = f"logs/visa_fuse_{SAVE}_{cls}.log"
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("CSV,"):
                continue
            parts = line.strip().split(",")
            if len(parts) != 10:
                continue
            _, c, tag = parts[0], parts[1], parts[2]
            if c != cls:
                continue
            out[tag] = [float(x) * 100 for x in parts[3:]]
    return out

data, missing = {}, []
for c in CLS:
    d = read_log(c)
    if d is None or len(d) < 3:
        missing.append(c)
        continue
    data[c] = d

hdr = f"{'class':<12}" + "".join(f"{m:>9}" for m in METRICS)
print(f"SAVE={SAVE}  已有 {len(data)}/{len(CLS)} 类  (缺: {missing or '无'})\n")
for tag in ("mapping", "bank", "fused"):
    print(f"===== {tag} =====")
    print(hdr)
    rows = []
    for c in CLS:
        if c not in data:
            continue
        v = data[c][tag]
        rows.append(v)
        print(f"{c:<12}" + "".join(f"{x:>9.2f}" for x in v) + f"  |avg {sum(v)/7:>6.2f}")
    if rows:
        avg = [sum(r[i] for r in rows) / len(rows) for i in range(7)]
        print(f"{'MEAN(' + str(len(rows)) + ')':<12}" + "".join(f"{x:>9.2f}" for x in avg) + f"  |avg {sum(avg)/7:>6.2f}")
    print()

if len(data) == len(CLS):
    print("===== 平均 Δ (fused - mapping / fused - bank) =====")
    print(hdr)
    for tag, ref in (("fused", "mapping"), ("fused", "bank")):
        avg_a = [sum(data[c][tag][i] for c in CLS) / len(CLS) for i in range(7)]
        avg_b = [sum(data[c][ref][i] for c in CLS) / len(CLS) for i in range(7)]
        d = [a - b for a, b in zip(avg_a, avg_b)]
        print(f"{tag}-{ref:<7}" + "".join(f"{x:>+9.2f}" for x in d) + f"  |avg {sum(d)/7:>+6.2f}")
