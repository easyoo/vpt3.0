# -*- coding: utf-8 -*-
"""对单个已落盘的异常图目录做评测（同一套口径：类内无需再归一化、高斯 k=5 σ=4、官方 adeval、
图像分 = 每图最高 1% 像素的均值）。用于拆开看 mapping 分支内部两侧的单独性能。
用法： python eval_single_map.py --map_dir .maps_d2_learn --classes "bottle ..." --data_path /home/jjquan/MVTec-AD
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_single', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_single'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map_dir', required=True)
    ap.add_argument('--classes', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    rows = []
    for cls in args.classes.split():
        # [FIX] 必须按侧车（生成异常图时记录的样本键序）对齐，否则 map 与 GT/标签错位，
        # 指标会变成垃圾值（上次就是因为漏了这一步，得到 learn-only 48.16 / raw-only 51.88 的错值）。
        sp = os.path.join(HERE, args.map_dir, f'{cls}.json')
        sc = None
        if os.path.exists(sp):
            try:
                sc = json.load(open(sp))
            except Exception:
                sc = None
        if sc and sc.get('keys'):
            d = mod.MVTecDataset(root=os.path.join(args.data_path, cls), transform=tf,
                                 gt_transform=gtf, phase='test', sort_items=True)
            keys = list(sc['keys'])
            if list(d.keys) != keys:
                raise RuntimeError(f'[align] {cls}: 数据集键序与侧车不一致，拒绝评测')
        else:
            d = mod.MVTecDataset(root=os.path.join(args.data_path, cls), transform=tf,
                                 gt_transform=gtf, phase='test', sort_items=False)
        gt = torch.stack([d[i][1] for i in range(len(d))], 0).float()
        lab = torch.tensor([int(d[i][2]) for i in range(len(d))])
        X = torch.load(os.path.join(HERE, args.map_dir, f'{cls}.map'), weights_only=False).float()
        with torch.no_grad():
            am = gk(X.to(args.device)).cpu()
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
        r = list(mod.ader_evaluator(am[:, 0].numpy(), sp, (gt[:, 0].numpy() > 0.5), lab.numpy()))
        rows.append(r)
        print(f'  {cls:12s} ' + '  '.join(f'{n}={v*100:.2f}' for n, v in zip(NAMES, r)))
    a = np.array(rows).mean(0) * 100
    print(f'  {"平均":12s} ' + '  '.join(f'{n}={v:.2f}' for n, v in zip(NAMES, a)) + f'   7项均值={a.mean():.3f}')


if __name__ == '__main__':
    main()
