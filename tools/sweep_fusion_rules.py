# -*- coding: utf-8 -*-
"""扫描 mapping 图 M 与 bank 图 B 的**单图融合规则**（最终仍是一张图）。

规则（z 均为类内标准化）：
  zavg      0.5·zM + 0.5·zB                     （v3.7 基线）
  minboost  zM + zB + γ·min(zM,zB)             （两路都高的地方额外加权）
  product   zM·zB（标准化后）                    （要求两路一致）
  max       max(zM, zB)
  rank      0.5·rM + 0.5·rB                     （类内秩，抗离群）
  nscore    Φ⁻¹(rM)/2 + Φ⁻¹(rB)/2               （秩→正态分数，抗离群）
  logz      0.5·z(log1p(M/sM)) + 0.5·z(log1p(B/sB))（压缩重尾）

所有规则都只输出一张异常图；高斯平滑 k=5 σ=4、类内 top-1% 图像分、官方 adeval 与 v3.7 相同。
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
    spec = importlib.util.spec_from_file_location('audit_fr', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_fr'] = m
    spec.loader.exec_module(m)
    return m


def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def rank01(x):
    f = x.flatten()
    o = torch.argsort(f)
    r = torch.empty_like(f)
    r[o] = torch.arange(f.numel(), dtype=f.dtype, device=f.device)
    return (r / max(1, f.numel() - 1)).reshape(x.shape)


def nscore(x):
    r = rank01(x).clamp(1e-4, 1 - 1e-4)
    return torch.erfinv(2 * r - 1) * (2 ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map_dir', required=True)
    ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--classes', required=True)
    ap.add_argument('--rules', default='zavg,minboost0.25,minboost0.5,product,max,rank,nscore,logz')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    rules = args.rules.split(',')
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    acc = {r: [] for r in rules}
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
        zM, zB = z(M), z(B)
        gt_px = (gt[:, 0].numpy() > 0.5)
        cands = {}
        for r in rules:
            if r == 'zavg':
                F = 0.5 * zM + 0.5 * zB
            elif r.startswith('minboost'):
                g = float(r.replace('minboost', ''))
                F = zM + zB + g * torch.minimum(zM, zB)
            elif r == 'product':
                F = zM * zB
            elif r == 'max':
                F = torch.maximum(zM, zB)
            elif r == 'rank':
                F = 0.5 * rank01(M) + 0.5 * rank01(B)
            elif r == 'nscore':
                F = 0.5 * nscore(M) + 0.5 * nscore(B)
            elif r == 'logz':
                sm, sb = M.mean().clamp_min(1e-8), B.mean().clamp_min(1e-8)
                F = 0.5 * z(torch.log1p(M / sm)) + 0.5 * z(torch.log1p(B / sb))
            else:
                raise ValueError(r)
            cands[r] = F
        for r, F in cands.items():
            with torch.no_grad():
                am = gk(F.to(args.device)).cpu()
            flat = am.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
            acc[r].append(list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab.numpy())))
    print(f'--- M={args.map_dir}  ⊕  B={args.bank_dir} ---')
    for r in rules:
        a = np.array(acc[r]).mean(0)
        print(f'  {r:14s} ' + '  '.join(f'{n}={v*100:.2f}' for n, v in zip(NAMES, a)) +
              f'   7项均值={a.mean()*100:.3f}')


if __name__ == '__main__':
    main()
