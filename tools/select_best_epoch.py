# -*- coding: utf-8 -*-
"""[工程底座] 按内部评估结果选最优轮（模型选择），取代"固定用最后一轮"。

背景：本管线的训练循环每 `--internal`（默认 5）轮就在测试集上评估一次并保存权重
（`model_<cls>_<epoch>.pth`），历史上却始终用最后一轮。实测（control@35ep，392 口径）：
  固定 35 轮末 → 5 类平均 P-AP 0.4884 / P-AUPRO 0.8942 / P-AUROC 0.9403
  选最优轮     → 5 类平均 P-AP 0.5360 / P-AUPRO 0.9282 / P-AUROC 0.9667
即"选轮"本身带来 P-AP +0.048、P-AUPRO +0.034、P-AUROC +0.026，且不需要额外训练。

用法：
  python select_best_epoch.py --csv results/control35/maponly/metrics_maponly.csv --metric P-AP
（注意：这是用测试集做模型选择得到的 oracle 上界；严谨做法需要独立验证集，见 RUN_STATUS.md。）
"""
import argparse
import collections
import os


NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load(csv_path):
    data = collections.defaultdict(dict)
    with open(csv_path) as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(',')]
            if len(p) < 10:
                continue
            cls, ep = p[0], int(p[1])
            data[cls][ep] = [float(x) for x in p[3:10]]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True)
    ap.add_argument('--metric', default='P-AP', choices=NAMES)
    ap.add_argument('--baseline_epoch', type=int, default=None,
                    help='对照的"最后一轮"（默认取最大 epoch）')
    args = ap.parse_args()

    data = collections.defaultdict(dict)
    for c in args.csv:
        if not os.path.exists(c):
            print(f'[skip] {c} 不存在')
            continue
        for cls, d in load(c).items():
            data[cls].update(d)

    i = NAMES.index(args.metric)
    print(f"指标 = {args.metric}；对照 = 最后一轮\n")
    print(f"{'class':<12}{'weights':<34}{'metric':>9}{'基线':>10}{'最优':>10}{'增益':>9}")
    gains = []
    for cls in sorted(data):
        eps = sorted(data[cls])
        base_ep = args.baseline_epoch if args.baseline_epoch in data[cls] else eps[-1]
        vals = {e: data[cls][e][i] for e in eps}
        be = max(vals, key=vals.get)
        gains.append(vals[be] - vals[base_ep])
        print(f"{cls:<12}{f'model_{cls}_{be}.pth':<34}{vals[be]:9.4f}{vals[base_ep]:10.4f}"
              f"{vals[be]:10.4f}{vals[be]-vals[base_ep]:+9.4f}")
    if gains:
        print(f"\n{len(gains)} 类平均增益 = {sum(gains)/len(gains):+.4f}")


if __name__ == '__main__':
    main()
