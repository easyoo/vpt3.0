# -*- coding: utf-8 -*-
"""逐图自适应门控融合：按 mapping 图与 bank 图在本图上的**一致性**决定该图的权重。

动机：bank 在图像级更可靠，mapping 在局部更敏感；两路一致时等权最佳，不一致时应偏向 bank。
权重只用本图的两路证据（无标签、无测试集调参）：
    c_i   = Pearson(zM_i, zB_i)  ∈ [-1,1]     （逐图像素级相关）
    w_i   = w_lo + (w_hi - w_lo)·clamp(c_i,0,1)
    F_i   = w_i·zM_i + (1-w_i)·zB_i            （z 为逐图标准化）
最终仍是一张异常图；平滑与 adeval 与 v3.7 相同。
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
    spec = importlib.util.spec_from_file_location('audit_af', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_af'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map_dir', required=True)
    ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--classes', required=True)
    ap.add_argument('--bounds', default='0.3:0.7,0.4:0.6,0.25:0.75')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    bounds = [tuple(float(v) for v in b.split(':')) for b in args.bounds.split(',')]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    acc = {b: [] for b in bounds}
    acc['zavg'] = []
    print(f'--- 自适应门控：M={args.map_dir} ⊕ B={args.bank_dir} ---')
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
        scM, scB = sc(args.map_dir), sc(args.bank_dir)
        M = torch.load(os.path.join(HERE, args.map_dir, f'{cls}.map'), weights_only=False).float()
        B = torch.load(os.path.join(HERE, args.bank_dir, f'{cls}.map'), weights_only=False).float()
        if scM and scM.get('keys') and scB and scB.get('keys'):
            ds, gt, lab = mk(True)
        else:
            ds, gt, lab = mk(False)
            frame = list(ds.keys)
            def ro(t, s):
                if not (s and s.get('keys')) or list(s['keys']) == frame:
                    return t
                idx = torch.tensor([list(s['keys']).index(x) for x in frame])
                return t[idx]
            M, B = ro(M, scM), ro(B, scB)
        # [口径修正] 融合值必须用**类别内 z**（与 v3.7 一致）；
        # 逐图标准化只用来算"两路在本图是否一致"的权重，不参与融合后的尺度。
        zM_cls = (M - M.mean()) / (M.std() + 1e-8)
        zB_cls = (B - B.mean()) / (B.std() + 1e-8)
        muM = M.mean(dim=(1, 2, 3), keepdim=True); sdM = M.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        muB = B.mean(dim=(1, 2, 3), keepdim=True); sdB = B.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        zM = (M - muM) / sdM
        zB = (B - muB) / sdB
        a = (zM * zB).mean(dim=(1, 2, 3), keepdim=True)                 # 逐图相关
        a = a.clamp(0.0, 1.0)
        gt_px = (gt[:, 0].numpy() > 0.5)
        cands = {'zavg': 0.5 * zM_cls + 0.5 * zB_cls}
        for lo, hi in bounds:
            w = lo + (hi - lo) * a
            cands[(lo, hi)] = w * zM_cls + (1 - w) * zB_cls
        for k, F in cands.items():
            with torch.no_grad():
                am = gk(F.to(args.device)).cpu()
            flat = am.flatten(1)
            kk = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, 1, descending=True)[0][:, :kk].mean(1).numpy()
            acc[k].append(list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab.numpy())))
    for k in ['zavg'] + bounds:
        a = np.array(acc[k]).mean(0)
        name = 'zavg(基线)' if k == 'zavg' else f'w={k[0]:.2f}~{k[1]:.2f}'
        print(f'  {name:16s} ' + '  '.join(f'{n}={v*100:.2f}' for n, v in zip(NAMES, a)) +
              f'   7项均值={a.mean()*100:.3f}')


if __name__ == '__main__':
    main()
