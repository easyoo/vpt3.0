# -*- coding: utf-8 -*-
"""秩融合（rank fusion）：每路异常图先做**类内秩归一化**，再多路等权平均。

为什么试它：
  * 现有融合是"类内 z 标准化 + 等权平均"。z 标准化被极端值主导（平方项），
    而图像级指标用的是**top-1% 均值**这个尾部统计量 —— 尾部正是 z 最不稳的地方。
  * 秩归一化对离群值稳健、跨路天然尺度无关、且是**无参数先验**（等权，不搜权重）。

流程：rank_i = (该类全部像素中该像素的排名) / (N-1) → [0,1]
      F = mean_i rank_i  → 高斯平滑(k=5,σ=4) → 官方 adeval（像素+图像分同口径）
其余（侧车对齐、GT、标签、图像分=top-1%）与 fuse_nocrop/fuse_ens_maps 完全一致。
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
    spec = importlib.util.spec_from_file_location('audit_rank', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_rank'] = m
    spec.loader.exec_module(m)
    return m


def rank_norm(x):
    """把整张 [N,1,H,W] 张量做全局秩归一化（与 fuse 的 z_cls 一样是"类内"操作）。"""
    flat = x.flatten()
    n = flat.numel()
    order = torch.argsort(flat)
    ranks = torch.empty(n, dtype=torch.float32, device=x.device)
    ranks[order] = torch.arange(n, dtype=torch.float32, device=x.device)
    return (ranks / max(1, n - 1)).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--view_dirs', required=True)
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
    F = sum(rank_norm(m) for m in maps) / len(maps)
    with torch.no_grad():
        am = gk(F.to(args.device)).cpu()
    flat = am.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
    r = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab.numpy()))
    print(f'  [{args.cls}] rankfusion ' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r)))
    print(f'CSV,{args.cls},rank,' + ','.join(f'{v:.4f}' for v in r))


if __name__ == '__main__':
    main()
