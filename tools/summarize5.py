# -*- coding: utf-8 -*-
# [vpt5.0] 5 类筛选协议：解析 eval_nocrop 的日志 → 7 指标平均 → 与 2-shot 基线对比。
# 用法: python summarize5.py logs/eval5_v5_t05_2s5_*.log
import glob
import re
import sys

NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']
CLS5 = ['transistor', 'screw', 'capsule', 'metal_nut', 'wood']


def baseline():
    """2-shot 交付版（m4m6r3_2shot）在 5 类上的 mapping-only 指标（来自逐类 CSV）。"""
    per = {}
    for c in CLS5:
        for ln in open(f'logs/fuse_2shot_{c}.log', errors='ignore'):
            if ln.startswith('CSV,') and ln.split(',')[2] == 'mapping':
                per[c] = [float(x) for x in ln.strip().split(',')[3:10]]
    return per


def main():
    pats = sys.argv[1:]
    if not pats:
        print('用法: python summarize5.py <eval 日志 glob>'); return
    files = []
    for p in pats:
        files += glob.glob(p)
    per = {}
    for fp in files:
        m = re.search(r'_(transistor|screw|capsule|metal_nut|wood)\.log$', fp)
        if not m:
            continue
        cls = m.group(1)
        for ln in open(fp, errors='ignore'):
            ln = ln.strip()
            if ln.startswith('CSV:'):
                per[cls] = [float(x) for x in ln[4:].split(',')]
    if not per:
        print('没解析到任何 CSV 行'); return
    base = baseline()
    n = len(per)
    avg = [sum(per[c][i] for c in per) / n for i in range(7)]
    # 基线必须限定在**同一批解析到的类别**上算，否则少类时会与 5 类基线错比（会失真）
    bavg = [sum(base[c][i] for c in per) / n for i in range(7)]
    print(f'解析到 {n}/5 类: {" ".join(sorted(per))}')
    if n < 5:
        print(f'⚠ 只解析到 {n}/5 类：基线已按这 {n} 类重算，结果不可与 5 类结论混用')
    print(f'{"指标":10s} {"新方案":>8s} {"基线":>8s} {"Δ":>8s} {"Δ%":>8s}')
    for i, nm in enumerate(NAMES):
        d = avg[i] - bavg[i]
        print(f'{nm:10s} {avg[i]*100:8.2f} {bavg[i]*100:8.2f} {d*100:+8.2f} {d/bavg[i]*100:+7.2f}%')
    new_m, base_m = sum(avg) / 7 * 100, sum(bavg) / 7 * 100
    print(f'{"7 项平均":10s} {new_m:8.2f} {base_m:8.2f} {new_m-base_m:+8.2f} {(new_m-base_m)/base_m*100:+7.2f}%'
          f'   {"✅ 达标" if (new_m-base_m)/base_m*100 >= 2 else "✘ 未达标"}（门槛 +2%）')
    print('\n逐类 7 项平均：')
    for c in CLS5:
        if c in per:
            print(f'  {c:11s} 新 {sum(per[c])/7*100:6.2f}   基线 {sum(base[c])/7*100:6.2f}'
                  f'   Δ {(sum(per[c])-sum(base[c]))/7*100:+6.2f}')


if __name__ == '__main__':
    main()
