# -*- coding: utf-8 -*-
"""混合打分：像素级用"多视图等权 z 平均图"，图像级用"各视图图像分再平均"。

动机（来自四视图实验的观察）：四视图的增益几乎全部落在像素级（P-AP/P-F1），
而图像级（I-AUROC/I-AP/I-F1）有时**微降** —— 原因是图像分目前从"融合后的图"取 top-1%，
融合在压低个别视图高响应区域的同时也压低了图像分。

本脚本把两者分开：
  像素图  = mean_i z(M_i)      （与 fuse_ens_maps --equal_all 完全一致）
  图像分  = mean_i ( top1%(smooth(M_i)) )   ← 各视图自己的图像分先算好再平均
其余（类内 z、侧车对齐、高斯 k=5 σ=4、官方 adeval）完全不变。
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
    spec = importlib.util.spec_from_file_location('audit_hybrid', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_hybrid'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def img_score(am):
    flat = am.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    return torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--view_dirs', required=True, help='逗号分隔，全部等权（含 bank 图目录）')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    dirs = args.view_dirs.split(',')

    def sc(d):
        p = os.path.join(HERE, d, f'{args.cls}.json')
        if not os.path.exists(p):
            return None
        try:
            return json.load(open(p))
        except Exception:
            return None

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    def mk(sort_items):
        d = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls), transform=tf,
                             gt_transform=gtf, phase='test', sort_items=sort_items)
        g = torch.stack([d[i][1] for i in range(len(d))], 0).float()
        l = torch.tensor([int(d[i][2]) for i in range(len(d))])
        return d, g, l

    scs = [sc(d) for d in dirs]
    maps = [torch.load(os.path.join(HERE, d, f'{args.cls}.map'), weights_only=False).float() for d in dirs]
    if all(s and s.get('keys') for s in scs):
        ds, gt, lab = mk(True)
    else:
        ds, gt, lab = mk(False)
        frame = list(ds.keys)
        def ro(t, s):
            if not (s and s.get('keys')) or list(s['keys']) == frame:
                return t
            idx = torch.tensor([list(s['keys']).index(x) for x in frame])
            return t[idx]
        maps = [ro(m, s) for m, s in zip(maps, scs)]

    gt_px = (gt[:, 0].numpy() > 0.5)
    F = sum(z_cls(m) for m in maps) / len(maps)
    with torch.no_grad():
        am_f = gk(F.to(args.device)).cpu()
        sp_each = [img_score(gk(m.to(args.device)).cpu()) for m in maps]
    sp_mean = np.mean(np.stack(sp_each, 0), 0)
    r_hybrid = list(mod.ader_evaluator(am_f[:, 0].numpy(), sp_mean, gt_px, lab.numpy()))
    r_fused = list(mod.ader_evaluator(am_f[:, 0].numpy(), img_score(am_f), gt_px, lab.numpy()))
    print(f'  [{args.cls}] hybrid ' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r_hybrid)))
    print(f'  [{args.cls}] fused  ' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r_fused)))
    print(f'CSV,{args.cls},hybrid,' + ','.join(f'{v:.4f}' for v in r_hybrid))
    print(f'CSV,{args.cls},fused,' + ','.join(f'{v:.4f}' for v in r_fused))


if __name__ == '__main__':
    main()
