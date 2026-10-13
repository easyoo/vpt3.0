# -*- coding: utf-8 -*-
"""汇总消融实验：5 类筛选集上，各 ablation 配置的 7 项平均（新/旧两种打分口径），
并给出相对基线（ab_base）的 Δ。用法: python summ_ablate.py <save1> <save2> ..."""
import sys
import os

MET = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
CLS = ["transistor", "screw", "capsule", "metal_nut", "wood"]


def read_csv(path):
    """读取 eval 脚本打印的 CSV: 行（7 个数）"""
    if not os.path.exists(path):
        return None
    for line in open(path, errors='ignore'):
        if 'CSV:' in line:
            v = [float(x) for x in line.split('CSV:')[1].strip().split(',')]
            if len(v) == 7:
                return [x * 100 for x in v]
    return None


def collect(saves, kind):
    out = {}
    for s in saves:
        rows = []
        for c in CLS:
            r = read_csv(f'logs/ablate_{kind}_{s}_{c}.log')
            if r:
                rows.append(r)
        if rows:
            out[s] = ([sum(r[i] for r in rows) / len(rows) for i in range(7)], len(rows))
    return out


def main():
    saves = sys.argv[1:]
    if not saves:
        raise SystemExit('用法: python summ_ablate.py <save1> <save2> ...')
    for kind, name in (('new', 'vpt3.6 新打分（d1^0.5·q50^0.5 ⊕ dlmin^0.7·bq64^0.3, w=0.6）'),
                       ('old', 'vpt3.5 旧打分（min/min, w=0.5）')):
        res = collect(saves, kind)
        if not res:
            print(f"\n=== {name}：暂无结果 ===")
            continue
        base = res.get(saves[0])
        print(f"\n=== {name} · 5 类筛选（{CLS}）===")
        print(f"{'配置':<22}{'类数':>4}" + "".join(f"{m:>9}" for m in MET) + f"{'7avg':>9}{'Δ':>8}")
        for s in saves:
            if s not in res:
                print(f"{s:<22}  无结果")
                continue
            a, n = res[s]
            d = f"{sum(a)/7 - sum(base[0])/7:>+8.2f}" if base and s != saves[0] else ""
            print(f"{s:<22}{n:>4}" + "".join(f"{x:>9.2f}" for x in a) + f"{sum(a)/7:>9.2f}{d}")


if __name__ == '__main__':
    main()
