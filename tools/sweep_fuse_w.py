# -*- coding: utf-8 -*-
"""一次加载、多个融合权重：F = w·z(mean_i z(M_i)) + (1-w)·z(B)。

用途：检查"两视图 mapping ⊕ bank"下 w=0.5（单视图时代定的先验）是否仍合理，
以及"每个视图等权"(w=2/3) 是否更好。**同一脚本在三个数据集上跑，避免只在某一个上有结论。**
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
    spec = importlib.util.spec_from_file_location('audit_sweepw', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_sweepw'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True)
    ap.add_argument('--map_dirs', required=True, help='逗号分隔')
    ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--ws', default='0.5,0.6,0.6667,0.7,0.75,0.8')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    ws = [float(x) for x in args.ws.split(',')]
    dirs = args.map_dirs.split(',')
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    acc = {w: [] for w in ws}
    for cls in args.classes.split():
        def sc(d):
            p = os.path.join(HERE, d, f'{cls}.json')
            if not os.path.exists(p):
                return None
            try:
                return json.load(open(p))
            except Exception:
                return None

        def mk(sort_items):
            d = mod.MVTecDataset(root=os.path.join(args.data_path, cls), transform=tf,
                                 gt_transform=gtf, phase='test', sort_items=sort_items)
            g = torch.stack([d[i][1] for i in range(len(d))], 0).float()
            l = torch.tensor([int(d[i][2]) for i in range(len(d))])
            return d, g, l

        names = dirs + [args.bank_dir]
        scs = [sc(n) for n in names]
        maps = [torch.load(os.path.join(HERE, d, f'{cls}.map'), weights_only=False).float() for d in dirs]
        B = torch.load(os.path.join(HERE, args.bank_dir, f'{cls}.map'), weights_only=False).float()
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
            maps = [ro(m, s) for m, s in zip(maps, scs[:-1])]
            B = ro(B, scs[-1])
        M = sum(z_cls(m) for m in maps) / len(maps)
        zM, zB = z_cls(M), z_cls(B)
        gt_px = (gt[:, 0].numpy() > 0.5)
        for w in ws:
            with torch.no_grad():
                am = gk((w * zM + (1 - w) * zB).to(args.device)).cpu()
            flat = am.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
            acc[w].append(list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab.numpy())))
    n = len(acc[ws[0]])
    print(f'--- {args.map_dirs} ⊕ {args.bank_dir} (n={n}) ---')
    for w in ws:
        a = np.array(acc[w]).mean(0)
        print(f'  w={w:<7} ' + '  '.join(f'{x}={v*100:.2f}' for x, v in zip(NAMES, a)) +
              f'   7项均值={a.mean()*100:.3f}')


if __name__ == '__main__':
    main()
