# -*- coding: utf-8 -*-
"""三路融合：mapping / segmap（分割头）/ bank，各自类内 z 标准化后按给定权重相加。

权重默认等分（1/3），**不在测试集上搜索**；仅用于检验"把训练过但推理未用的分割头图
作为第三个来源"是否能提升交付口径（mapping ⊕ bank）。

用法: python fuse3_nocrop.py --class transistor --map_dir A --seg_dir B --bank_dir C \
        --data_path /home/jjquan/MVTec-AD [--w "1,1,1"]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    s = importlib.util.spec_from_file_location('f3', p)
    m = importlib.util.module_from_spec(s)
    sys.modules['f3'] = m
    s.loader.exec_module(m)
    return m


def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--map_dir', required=True)
    ap.add_argument('--seg_dir', required=True)
    ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--w', default='1,1,1', help='mapping,seg,bank 三路权重（默认等分）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    wt = [float(x) for x in args.w.split(',')]
    wt = [x / sum(wt) for x in wt]

    M = torch.load(os.path.join(HERE, args.map_dir, f'{args.cls}.map'), weights_only=False).float()
    S = torch.load(os.path.join(HERE, args.seg_dir, f'{args.cls}.map'), weights_only=False).float()
    B = torch.load(os.path.join(HERE, args.bank_dir, f'{args.cls}.map'), weights_only=False).float()
    assert M.shape == B.shape == S.shape, (M.shape, S.shape, B.shape)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test', sort_items=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    gt_px, gt_sp = [], []
    for _, gt, label, _ in dl:
        g = (gt > 0.5).float()
        if g.shape[1] > 1:
            g = torch.max(g, dim=1, keepdim=True)[0]
        gt_px.append(g[:, 0].numpy().astype(np.uint8))
        gt_sp.append(label.numpy())
    gt_px = np.concatenate(gt_px, 0)
    gt_sp = np.concatenate(gt_sp).flatten()

    fused = wt[0] * z(M) + wt[1] * z(S) + wt[2] * z(B)
    am = gk(fused.to(args.device))
    flat = am.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    pr_sp = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(1)
    r = list(mod.ader_evaluator(am[:, 0].cpu().numpy(), pr_sp.flatten().cpu().numpy(),
                                gt_px, gt_sp))
    print(f"[fuse3 {args.cls} w={args.w}] " + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, r)))
    print("  CSV," + f"{args.cls},fuse3," + ",".join(f"{v:.4f}" for v in r))
    # 同时给出"仅加 seg 图"的两路对照（map⊕seg），便于看 seg 自身价值
    f2 = 0.5 * z(M) + 0.5 * z(S)
    am2 = gk(f2.to(args.device))
    flat2 = am2.flatten(1)
    pr_sp2 = torch.sort(flat2, dim=1, descending=True)[0][:, :k].mean(1)
    r2 = list(mod.ader_evaluator(am2[:, 0].cpu().numpy(), pr_sp2.flatten().cpu().numpy(),
                                 gt_px, gt_sp))
    print(f"[fuse2 {args.cls} map⊕seg] " + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, r2)))


if __name__ == '__main__':
    main()
