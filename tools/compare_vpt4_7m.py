# -*- coding: utf-8 -*-
"""[对比] 任意 vpt4 变体 vs vpt3.0「改善 mapping」基线 —— **7 项官方指标（含 P-AUPRO）**。

基线来源：results/ms35/all15_g35.csv 的第二行组（= vpt3.0 改善 mapping，392 口径、统一 35 轮、
与 vpt4 完全相同的打分协议 `--fuse_bank -1 --ms_scoring --ms_scales 1.5 --ms_flip 1`）。

用法：
  python compare_vpt4_7m.py m4px05:vpt4px05 [m4m6:vpt4m6 ...]
  python compare_vpt4_7m.py --all          # 自动扫描 results/vpt4/eval_*.log
"""
import argparse
import glob
import math
import os
import re
import statistics as st

NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
BASE_CSV = "results/ms35/all15_g35.csv"
CLASSES = ["transistor", "screw", "capsule", "metal_nut", "wood"]
ALL15 = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


def load_base():
    rows = {}
    for ln in open(BASE_CSV):
        s = ln.strip()
        if not s.startswith("CSV:"):
            continue
        p = s[4:].split(",")
        rows.setdefault(p[0], []).append([float(x) for x in p[4:]])
    return {c: v[1] for c, v in rows.items()}          # 第二行组 = vpt3.0 改善 mapping


def read_eval(save, tag, cls, pattern="results/vpt4/eval_{save}_{cls}.log"):
    path = pattern.format(save=save, cls=cls)
    if not os.path.exists(path):
        return None
    cur = None
    last = None
    for ln in open(path):
        s = ln.strip()
        if s.startswith("["):
            cur = s[1:].split("]")[0]
        elif s.startswith("CSV:") and (tag is None or cur == tag):
            last = [float(x) for x in s[4:].split(",")[-7:]]     # 同一日志可能被追加多次（重评）→ 取最后一条
    return last


def _mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return st.mean(vals) if vals else float('nan')


def report(name, save, tag, base, classes, pattern="results/vpt4/eval_{save}_{cls}.log"):
    got = {c: read_eval(save, tag, c, pattern) for c in classes}
    got = {c: v for c, v in got.items() if v}
    if not got:
        print(f"[{name}] 暂无结果（{save}）")
        return
    print(f"\n=== {name}  save={save} tag={tag}  {len(got)}/{len(classes)} 类 ===")
    print(f"{'class':<11}" + "".join(f"{n:>10}" for n in NAMES) + "   相对基线(Δ)")
    for c in classes:
        if c not in got:
            print(f"{c:<11} 尚无结果")
            continue
        a, b = got[c], base[c]
        print(f"{c:<11}" + "".join(f"{a[i]:10.4f}" if not math.isnan(a[i]) else f"{'n/a':>10}"
                                   for i in range(7))
              + "   " + "".join((f"{a[i]-b[i]:+8.4f}" if not math.isnan(a[i]) else f"{'n/a':>8}")
                                for i in range(7)))
    m = [_mean([got[c][i] for c in got]) for i in range(7)]
    b = [_mean([base[c][i] for c in got]) for i in range(7)]
    print(f"{'均值':<11}" + "".join(f"{m[i]:10.4f}" if not math.isnan(m[i]) else f"{'n/a':>10}"
                                   for i in range(7))
          + "   " + "".join((f"{m[i]-b[i]:+8.4f}" if not math.isnan(m[i]) else f"{'n/a':>8}")
                            for i in range(7)))
    print(f"{'Δ%':<11}" + "".join((f"{100*(m[i]-b[i])/b[i]:+9.2f}%" if not math.isnan(m[i]) else f"{'n/a':>10}")
                                  for i in range(7)))
    rel_all = [100 * (m[i] - b[i]) / b[i] for i in range(7) if not math.isnan(m[i])]
    miss = [NAMES[i] for i in range(7) if not math.isnan(m[i]) and 100 * (m[i] - b[i]) / b[i] < 1.5]
    print(f"未达 +1.5% 的指标：{miss if miss else '无 ✔（7/7 达标）'}")
    print(f"【验收口径】{len(rel_all)} 项相对提升的平均值 = {st.mean(rel_all):+.2f}%   "
          f"（门槛 +1.50% → {'达标 ✔' if st.mean(rel_all) >= 1.5 else '未达标 ✘'}）")
    print("           逐指标(%) = " + "  ".join(
        f"{NAMES[i]}:{(100*(m[i]-b[i])/b[i]):+.2f}" if not math.isnan(m[i]) else f"{NAMES[i]}:n/a"
        for i in range(7)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", help="save:tag，如 m4m6:vpt4m6")
    ap.add_argument("--all", action="store_true", help="扫描 results/vpt4/eval_*.log 自动列出")
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--all15", action="store_true", help="用 MVTec 全部 15 类（验收口径）")
    ap.add_argument("--pattern", default="results/vpt4/eval_{save}_{cls}.log",
                    help="评估日志路径模板，例如 CPU 口径用 results/vpt4/eval_{save}_cpu_{cls}.log")
    args = ap.parse_args()
    if args.all15:
        args.classes = ALL15

    base = load_base()
    pairs = list(args.pairs)
    if args.all:
        found = set()
        for p in glob.glob("results/vpt4/eval_*_*.log"):
            m = re.match(r"results/vpt4/eval_(.+)_([a-z_]+)\.log$", p)
            if not m:
                continue
            save, cls = m.group(1), m.group(2)
            if cls not in args.classes:
                continue
            tag = None
            for ln in open(p):
                s = ln.strip()
                if s.startswith("["):
                    tag = s[1:].split("]")[0]
            if tag:
                found.add(f"{save}:{tag}")
        pairs += sorted(found)
    if not pairs:
        pairs = ["m4px05:vpt4px05", "m4px02:vpt4px02", "m4:vpt4m4"]
    for pr in pairs:
        save, _, tag = pr.partition(":")
        report(pr, save, tag or None, base, args.classes, args.pattern)


if __name__ == "__main__":
    main()
