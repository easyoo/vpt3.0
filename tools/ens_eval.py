# -*- coding: utf-8 -*-
"""[公平实验] 离线图级融合评估（不做任何测试集调参）。

只读三类**已缓存**的打分图（同一 data loader 顺序、同一 448 不裁剪口径）：
  A  = v3.6 mapping 图  (.maps_nocrop448_m4m6r3_2shot)
  B  = v3.7 mapping 图  (.maps_simple_nocrop448_mv2)
  BK = 原始 MemoryBank 图 (.errmap_nocrop448_2shot)

候选（全部为**先验固定**的权重，不搜索）：
  A / B / ENS = 0.5·z(A) + 0.5·z(B)
  fused(X, w) = w·z(X) + (1-w)·z(BK)，X ∈ {A, B, ENS}，w ∈ {0.5, 0.6}
  —— 其中 0.5 是等权先验；0.6 是 v3.6/v3.7 历史沿用的取值，一并列出以量化"测试集调权重"的影响。

每个类输出 logs/ens_<cls>.log，内含若干 CSV 行（供汇总）。
"""
import argparse
import importlib.util
import os
import sys

import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--dirA', default='.maps_nocrop448_m4m6r3_2shot')
    ap.add_argument('--dirB', default='.maps_simple_nocrop448_mv2')
    ap.add_argument('--dirBK', default='.errmap_nocrop448_2shot')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--tag', default='ens', help='输出日志前缀：logs/<tag>_<cls>.log')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)

    A = torch.load(os.path.join(HERE, args.dirA, f'{args.cls}.map'), weights_only=False).float()
    B = torch.load(os.path.join(HERE, args.dirB, f'{args.cls}.map'), weights_only=False).float()
    BK = torch.load(os.path.join(HERE, args.dirBK, f'{args.cls}.map'), weights_only=False).float()
    assert A.shape == B.shape == BK.shape, (args.cls, A.shape, B.shape, BK.shape)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test', sort_items=False)
    gt = torch.stack([ds[i][1] for i in range(len(ds))], 0).float()
    lab = torch.tensor([int(ds[i][2]) for i in range(len(ds))])
    assert gt.shape[0] == A.shape[0], (args.cls, gt.shape, A.shape)

    zA, zB, zBK = z_cls(A), z_cls(B), z_cls(BK)
    zENS = 0.5 * zA + 0.5 * zB
    cands = [('A(v3.6_map)', A), ('B(v3.7_map)', B), ('ENS(0.5A+0.5B)', zENS)]
    for tag, Z in (('A', zA), ('B', zB), ('ENS', zENS)):
        for w in (0.5, 0.6):
            cands.append((f'fuse[{tag},w={w}]', w * Z + (1 - w) * zBK))

    lines = []
    for name, X in cands:
        with torch.no_grad():
            am = gk(X.to(args.device)).cpu()
        pr_px = am[:, 0].numpy()
        gt_px = (gt[:, 0].numpy() > 0.5)
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        pr_sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
        r = list(mod.ader_evaluator(pr_px, pr_sp, gt_px, lab.numpy()))
        lines.append(f'CSV,{args.cls},{name},' + ','.join(f'{v:.4f}' for v in r))
        print(f'  [{args.cls}] {name:<18}' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r)), flush=True)

    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    with open(os.path.join(HERE, 'logs', f'{args.tag}_{args.cls}.log'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
