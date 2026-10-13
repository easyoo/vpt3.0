# -*- coding: utf-8 -*-
"""汇总 v3 生产版三支结果：logs/v3fuse_<tag>_<cls>.log 的 CSV 行。"""
import collections
import glob
import os
import sys

MET = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
_VISA = "candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum".split()
_MV = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()
ORD = {'visa4': _VISA, 'mv2': _MV, 'mv4': _MV,
       'v3visa4': _VISA, 'v3mv2': _MV, 'v3mv4': _MV}


def main():
    tags = sys.argv[1:] or ['v3visa4', 'v3mv2', 'v3mv4']
    for tag in tags:
        d = collections.defaultdict(dict)
        for f in glob.glob(f'logs/v3fuse_{tag}_*.log'):
            for line in open(f, errors='ignore'):
                if line.strip().startswith('CSV,'):
                    p = line.strip().split(',')
                    if len(p) == 10:
                        d[p[1]][p[2]] = [float(x) * 100 for x in p[3:]]
        cls = [c for c in ORD.get(tag, sorted(d)) if c in d and len(d[c]) == 3]
        if not cls:
            print(f"=== {tag}: 暂无结果 ===")
            continue
        print(f"\n=== {tag}  ({len(cls)} 类) ===")
        print(f"{'':<10}" + "".join(f"{m:>9}" for m in MET) + f"{'7avg':>9}")
        avg = {}
        for t in ('mapping', 'bank', 'fused'):
            rows = [d[c][t] for c in cls]
            a = [sum(r[i] for r in rows) / len(rows) for i in range(7)]
            avg[t] = a
            print(f"{t:<10}" + "".join(f"{x:>9.2f}" for x in a) + f"{sum(a)/7:>9.2f}")
        for ref in ('mapping', 'bank'):
            dd = [avg['fused'][i] - avg[ref][i] for i in range(7)]
            print(f"Δfu-{ref[:4]:<5}" + "".join(f"{x:>+9.2f}" for x in dd)
                  + f"{sum(dd)/7:>+9.2f}")


if __name__ == '__main__':
    main()
