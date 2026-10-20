# -*- coding: utf-8 -*-
"""汇总"主干对照"实验：从 eval/fuse 日志解析逐类 7 指标，输出 mapping-only / fused 均值。"""
import argparse
import os
import re

NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def parse_gen(path):
    """eval_score_v3 日志末行：'  CSV:0.99,...,0.92'"""
    out = None
    with open(path, errors='ignore') as f:
        for line in f:
            m = re.search(r'CSV:([0-9.,]+)', line)
            if m:
                out = [float(v) for v in m.group(1).split(',') if v]
    return out


def parse_fuse(path):
    """fuse 日志末行：'CSV,<cls>,fused,0.99,...'"""
    out = None
    with open(path, errors='ignore') as f:
        for line in f:
            if line.startswith('CSV,'):
                parts = line.strip().split(',')
                out = [float(v) for v in parts[3:] if v]
    return out


def summarize(tag, classes, logdir='logs'):
    rows = {'mapping': {}, 'fused': {}}
    for c in classes:
        g = os.path.join(logdir, f'lse_gen_{tag}_{c}.log')
        fu = os.path.join(logdir, f'lse_fuse_{tag}_{c}.log')
        if os.path.exists(g):
            v = parse_gen(g)
            if v:
                rows['mapping'][c] = v
        if os.path.exists(fu):
            v = parse_fuse(fu)
            if v:
                rows['fused'][c] = v
    res = {}
    for kind in ('mapping', 'fused'):
        d = rows[kind]
        if not d:
            continue
        n = len(d)
        per = [sum(d[c][i] for c in d) / n for i in range(len(NAMES))]
        res[kind] = (per, sum(per) / len(per) * 100, n)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='bottle cable transistor screw capsule metal_nut wood')
    ap.add_argument('--tag', default='d3b')
    args = ap.parse_args()
    cls = args.classes.split()
    r = summarize(args.tag, cls)
    print(f'=== TAG={args.tag}  类别数={len(cls)} ===')
    for kind in ('mapping', 'fused'):
        if kind not in r:
            print(f'{kind}: 无日志')
            continue
        per, mean, n = r[kind]
        print(f'{kind:8s} n={n}  ' + '  '.join(f'{a}={b*100:.2f}' for a, b in zip(NAMES, per)) +
              f'   7项均值={mean:.2f}')


if __name__ == '__main__':
    main()
