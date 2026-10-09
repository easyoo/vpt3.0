# -*- coding: utf-8 -*-
"""[汇总] MVTec 全部 15 类的平均指标：原始口径 / base / 6视图推荐 / 2视图高效档。"""
import glob, os, statistics as st, sys

NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
ALL15 = "carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper".split()

def read_tag_log(path):
    """从 eval_*.log 里读出 {tag: [7 指标]}"""
    out, tag = {}, None
    if not os.path.exists(path):
        return out
    for ln in open(path):
        s = ln.strip()
        if s.startswith('['):
            tag = s[1:].split(']')[0]
        elif s.startswith('CSV:') and tag:
            out[tag] = [float(x) for x in s[4:].split(',')[-7:]]
    return out

# 原始口径：control35 第 35 轮（逐类）
raw35 = {}
csvp = 'results/control35/maponly/metrics_maponly.csv'
if os.path.exists(csvp):
    for line in open(csvp):
        p = line.strip().split(',')
        if len(p) >= 10 and int(p[1]) == 35:
            raw35[p[0]] = [float(x) for x in p[3:10]]

rows = {'原始口径(统一35轮)': raw35}
for label, pat in [('base(选最优轮)', 'results/ms35/eval_15cls_{}.log'),
                   ('推荐6视图', 'results/ms35/eval_15cls_{}.log'),
                   ('高效2视图', 'results/ms35/eval_fast_{}.log')]:
    rows.setdefault(label, {})
for cls in ALL15:
    d = read_tag_log(f'results/ms35/eval_15cls_{cls}.log')
    if '15cls-base' in d: rows['base(选最优轮)'][cls] = d['15cls-base']
    if '15cls-rec' in d: rows['推荐6视图'][cls] = d['15cls-rec']
    f = read_tag_log(f'results/ms35/eval_fast_{cls}.log')
    if 'fast2view' in f: rows['高效2视图'][cls] = f['fast2view']

print(f"{'协议':<20}{'类数':>5}" + ''.join(f'{n:>10}' for n in NAMES))
for label, d in rows.items():
    have = [c for c in ALL15 if c in d]
    if not have:
        print(f'{label:<20}{0:>5}   （暂无数据）'); continue
    m = [st.mean([d[c][i] for c in have]) for i in range(7)]
    print(f'{label:<20}{len(have):>5}' + ''.join(f'{x:10.4f}' for x in m))

# 逐类明细（高效2视图）
print('\n=== 逐类（高效2视图：融合 + 1.5×+翻转）===')
print(f"{'class':<12}" + ''.join(f'{n:>10}' for n in NAMES))
d = rows['高效2视图']
for c in ALL15:
    if c in d: print(f'{c:<12}' + ''.join(f'{x:10.4f}' for x in d[c]))
