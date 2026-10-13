# -*- coding: utf-8 -*-
# [VisA 准备 2/3] 把官方 VisA（<cls>/Data/{Images,Masks}/...，划分见 split_csv/1cls.csv）
# 整理成 **MVTec 式目录**，这样训练/评估脚本（shot_utils、MVTecDataset）可以零改动复用：
#
#   <root>/<cls>/train/good/<name>.JPG                 ← 官方 train 划分的正常图（抽样底图来源）
#   <root>/<cls>/test/good/<name>.JPG                  ← 官方 test 划分的正常图
#   <root>/<cls>/test/anomaly/<name>.JPG               ← 官方 test 划分的异常图
#   <root>/<cls>/ground_truth/anomaly/<name>.png       ← 对应掩码（MVTecDataset 只 glob *.png）
#
# **重要（VisA 特有的一处坑）**：VisA 官方掩码是 **0/1 二值**（最大值 1），而 MVTec 是 **0/255**。
# 管线里 GT 的判定是 `(gt > 0.5)`（Resize+ToTensor 后 1 → 1/255 ≈ 0.004），
# 若直接沿用原图，**所有异常掩码都会变成空掩码** → adeval 的 AUPRO 除零 → NaN → `assert area >= 0` 崩溃。
# 因此这里把掩码**二值化重写成 0/255**（真实文件，非软链接），其余图片仍用软链接。
#
# 用法：python visa_build_root.py [输出根]
import numpy as np
from PIL import Image
import csv
import os
import sys

VISAROOT = '/mnt/data/datasets/VisA'
CSV = os.path.join(VISAROOT, 'split_csv', '1cls.csv')
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/jjquan/experiment/visual2/visa_prepared'
TYPE = 'anomaly'


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return False
    os.symlink(os.path.abspath(src), dst)
    return True


def write_mask_bin(src, dst):
    """把 VisA 的 0/1 掩码二值化写成 0/255（与 MVTec 掩码口径一致）。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and not os.path.islink(dst):
        return False
    a = np.asarray(Image.open(src).convert('L'))
    b = (a > 0).astype(np.uint8) * 255
    Image.fromarray(b, mode='L').save(dst)
    return True


def main():
    rows = list(csv.DictReader(open(CSV)))
    n = {'train': 0, 'test_good': 0, 'test_anom': 0, 'mask': 0}
    for r in rows:
        cls, split, label = r['object'], r['split'], r['label']
        img = os.path.join(VISAROOT, r['image'])
        base = os.path.basename(img)
        if split == 'train':
            link(img, os.path.join(OUT, cls, 'train', 'good', base)); n['train'] += 1
        elif label == 'normal':
            link(img, os.path.join(OUT, cls, 'test', 'good', base)); n['test_good'] += 1
        else:
            link(img, os.path.join(OUT, cls, 'test', TYPE, base)); n['test_anom'] += 1
            if r['mask']:
                m = os.path.join(VISAROOT, r['mask'])
                # MVTecDataset 只 glob *.png → 掩码统一落成 .png 名（本体就是 png）
                mname = os.path.splitext(base)[0] + '.png'
                write_mask_bin(m, os.path.join(OUT, cls, 'ground_truth', TYPE, mname)); n['mask'] += 1
    print(f'VisA → MVTec 式根目录：{OUT}')
    print(f"  train/good={n['train']}  test/good={n['test_good']}  test/anomaly={n['test_anom']}  masks={n['mask']}")
    for cls in sorted({r['object'] for r in rows}):
        t = len(os.listdir(os.path.join(OUT, cls, 'train', 'good')))
        g = len(os.listdir(os.path.join(OUT, cls, 'test', 'good')))
        a = len(os.listdir(os.path.join(OUT, cls, 'test', TYPE)))
        print(f'  {cls:12s} train={t:4d} test_good={g:3d} test_anom={a:3d}')


if __name__ == '__main__':
    main()
