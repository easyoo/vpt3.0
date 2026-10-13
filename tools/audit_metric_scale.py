# -*- coding: utf-8 -*-
"""审计：adeval 的指标是否对"单调变换"不变？

若不是，则新打分（d1^0.5·q50^0.5，分布形状与 min 不同）带来的 AUPRO 变化里
可能混入"阈值分箱"伪影，而不是排序质量变化。

对同一张已落盘的异常图做若干**严格单调**变换（排序完全不变），逐一评测。
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    s = importlib.util.spec_from_file_location('audit_m', p)
    m = importlib.util.module_from_spec(s)
    sys.modules['audit_m'] = m
    s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--transforms', default='id,sq,sqrt,log,linear10')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    mod = load_mod(os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    gt_px = []
    for _, gt, _, _ in dl:
        g = (gt > 0.5).float()
        if g.shape[1] > 1:
            g = torch.max(g, dim=1, keepdim=True)[0]
        gt_px.append(g[:, 0].numpy().astype(np.uint8))
    gt_px = np.concatenate(gt_px, 0)

    M = torch.load(args.map, weights_only=False).float()
    print(f"map {args.map} shape={tuple(M.shape)}  min={M.min():.5f} max={M.max():.5f}")
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    print(f"{'transform':<12}" + "".join(f"{n:>9}" for n in NAMES))
    for t in args.transforms.split(','):
        if t == 'id':
            x = M
        elif t == 'sq':
            x = M ** 2
        elif t == 'sqrt':
            x = M.clamp_min(0) ** 0.5
        elif t == 'log':
            x = torch.log1p(M)
        elif t == 'linear10':
            x = M * 10
        else:
            raise SystemExit(t)
        am = gk(x.to(args.device)).cpu()
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        sp = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        r = list(mod.ader_evaluator(am[:, 0].cpu().numpy(), sp.flatten().cpu().numpy(),
                                    gt_px, np.array([1 if g.sum() > 0 else 0 for g in gt_px])))
        print(f"{t:<12}" + "".join(f"{v:>9.4f}" for v in r))


if __name__ == '__main__':
    main()
