# -*- coding: utf-8 -*-
# [VisA 准备 1/3] 按固定种子从 VisA 官方 train 划分里抽 1/2/4-shot 正常样本，
# 并生成 AnoStyler 配置（每张底图 num_gen=35，与 MVTec 口径一致）。
#
# VisA 协议：官方 split_csv/1cls.csv —— train=纯正常图，test=正常+异常（含掩码）。
# 抽样只在 train 里做，绝不触碰 test，避免泄漏。
#
# 输出：anostyler_visa/<K>shot/configs/<idx>_<cls>_<stem>.yaml
#       anostyler_visa/<K>shot/shots.json（记录了每类抽到哪些图，供复现与后续对齐）
import csv
import json
import os
import random

import yaml

VISAROOT = '/mnt/data/datasets/VisA'
CSV = os.path.join(VISAROOT, 'split_csv', '1cls.csv')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'anostyler_visa')
SHOTS = [1, 2, 4]
SEED = 1
NUM_GEN = 35                                  # 与 MVTec 2-shot/4-shot 实验一致
DEFECT = 'defect'                             # VisA 无官方缺陷类型名，统一用一个合成类型

BASE = {
    'seed': 77,
    'num_gen': NUM_GEN,
    'img_width': 512,
    'img_height': 512,
    'm_max': 5,
    'alpha': 0.7,
    'fg_mask': True,
    'sam_ckpt': './checkpoints/sam_vit_b_01ec64.pth',
    'crop_size': 128,
    'num_crops': 64,
    'max_step': 75,
    'lr': 0.0005,
    'thresh': 0.7,
    'lambda_tv': 0.002,
    'lambda_pdir': 9000,
    'lambda_gdir': 500,
    'lambda_c': 150,
    'lambda_mclip': 1000,
}


def load_train_normals():
    """返回 {cls: [相对路径, ...]}（仅 train 划分的 normal 图，已排序）。"""
    per = {}
    for r in csv.DictReader(open(CSV)):
        if r['split'] != 'train' or r['label'] != 'normal':
            continue
        per.setdefault(r['object'], []).append(r['image'])
    return {k: sorted(v) for k, v in per.items()}


def main():
    per = load_train_normals()
    classes = sorted(per)
    print(f'VisA train-normal：{len(classes)} 类  ' +
          ' '.join(f'{c}={len(per[c])}' for c in classes))
    for K in SHOTS:
        cfg_dir = os.path.join(OUT, f'{K}shot', 'configs')
        os.makedirs(cfg_dir, exist_ok=True)
        shots_rec, idx = {}, 0
        for cls in classes:
            files = per[cls]
            assert len(files) >= K, (cls, len(files), K)
            random.seed(SEED)                     # 每类同一种子 ⇒ K 之间是嵌套子集
            picks = sorted(random.sample(files, K))
            shots_rec[cls] = picks
            for rel in picks:
                stem = os.path.splitext(os.path.basename(rel))[0]
                cfg = dict(BASE)
                cfg['category'] = cls
                cfg['defect'] = DEFECT
                cfg['normal_image'] = os.path.join(VISAROOT, rel)
                cfg['save_path'] = os.path.join(OUT, f'{K}shot', 'genbyAnoStyler')
                name = f'{idx:04d}_{cls}_{stem}.yaml'
                with open(os.path.join(cfg_dir, name), 'w', encoding='utf-8') as fh:
                    yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
                idx += 1
        with open(os.path.join(OUT, f'{K}shot', 'shots.json'), 'w', encoding='utf-8') as fh:
            json.dump({'dataset': 'VisA', 'shots': K, 'seed': SEED, 'num_gen': NUM_GEN,
                       'defect_type': DEFECT, 'picks': shots_rec}, fh, indent=1, ensure_ascii=False)
        print(f'{K}-shot: 生成 {idx} 个配置（{len(classes)} 类 × {K} 张底图）→ {cfg_dir}')


if __name__ == '__main__':
    main()
