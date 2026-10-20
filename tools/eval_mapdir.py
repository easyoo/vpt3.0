# -*- coding: utf-8 -*-
"""单独评估一个异常图目录（与 fuse_nocrop 的"单支"路径完全同口径）。

用途：比较 mapping 分支内部两侧各自的性能 —— `--w 1.0` 导出的可学习侧图、
`--w 0.0` 导出的"mapping 侧的 bank（原始库）"图，以及二者 0.6 混合后的 mapping 图。
"""
import argparse
import importlib.util
import json
import os
import sys

import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_mapdir', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_mapdir'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--dirs', required=True, help='逗号分隔；每个目录一项，输出各自指标')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    dirs = args.dirs.split(',')

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
    for d, X in zip(dirs, maps):
        with torch.no_grad():
            am = gk(X.to(args.device)).cpu()
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
        r = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab.numpy()))
        print(f'CSV,{args.cls},{os.path.basename(d)},' + ','.join(f'{v:.4f}' for v in r))


if __name__ == '__main__':
    main()
