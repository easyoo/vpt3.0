# -*- coding: utf-8 -*-
# [2-shot 实验] 汇总 fuse_nocrop 日志里的逐类指标 → 15 类平均（mapping / bank / fused）。
# 用法: python summarize_2shot.py logs/fuse_2shot_*.log
import collections
import glob
import sys

NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def main():
    pats = sys.argv[1:] or ['logs/fuse_2shot_*.log']
    files = []
    for p in pats:
        files += glob.glob(p)
    agg = collections.defaultdict(lambda: [0.0] * 7)
    cnt = collections.Counter()
    per_class = {}
    for fp in files:
        for ln in open(fp, errors='ignore'):
            ln = ln.strip()
            if not ln.startswith('CSV,'):
                continue
            _, cls, branch, *vals = ln.split(',')
            v = [float(x) for x in vals[:7]]
            if len(v) != 7:
                continue
            key = (cls, branch)
            per_class[key] = v
            for i, x in enumerate(v):
                agg[branch][i] += x
            cnt[branch] += 1
    print('%-9s %-8s' % ('branch', 'n') + ''.join('%10s' % n for n in NAMES) + '%10s' % 'mean7')
    for br in ('mapping', 'bank', 'fused'):
        n = cnt[br]
        if not n:
            continue
        avg = [x / n for x in agg[br]]
        print('%-9s %-8d' % (br, n) + ''.join('%10.4f' % x for x in avg) + '%10.4f' % (sum(avg) / 7))
    # 逐类（融合）明细
    fused = sorted(c for c, b in per_class if b == 'fused')
    if fused:
        print('\n逐类（fused）:')
        print('%-12s' % 'cls' + ''.join('%10s' % n for n in NAMES))
        for c in fused:
            print('%-12s' % c + ''.join('%10.4f' % x for x in per_class[(c, 'fused')]))


if __name__ == '__main__':
    main()
