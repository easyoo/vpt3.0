# -*- coding: utf-8 -*-
"""[新口径 · 融合评估] 训练裁剪(448→392) / 测试不裁剪(Resize 448)：

  mapping 分支：用 m4m6r3_15 权重在**不裁剪 448** 下重推 → .maps_nocrop448_m4m6r3/<cls>.map
  bank  分支：原版 MemoryBank 在**不裁剪 448** 下重推 → .errmap_nocrop448/<cls>.map
  融合      ：类内 z 标准化后 0.3·z(mapping) + 0.7·z(bank)，高斯平滑(k=5,σ=4) → 官方 adeval

同时给出两支单独 + 融合三组指标，便于看互补性。
用法：python fuse_nocrop.py --class transistor --size 448
"""
import argparse
import importlib.util
import os
import sys

import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


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
    ap.add_argument('--w', type=float, default=0.3, help='mapping 权重（0.3 = 0.3·z(M)+0.7·z(B)）')
    ap.add_argument('--map_dir', default='.maps_nocrop448_m4m6r3')
    ap.add_argument('--bank_dir', default='.errmap_nocrop448')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)

    mp = os.path.join(HERE, args.map_dir, f'{args.cls}.map')
    bp = os.path.join(HERE, args.bank_dir, f'{args.cls}.map')
    M = torch.load(mp, weights_only=False).float()
    B = torch.load(bp, weights_only=False).float()
    assert M.shape == B.shape, (args.cls, M.shape, B.shape)

    # GT / 标签（与上面两张图同序：同一 DataLoader，不裁剪 448）
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    gt = torch.stack([ds[i][1] for i in range(len(ds))], 0).float()
    lab = torch.tensor([int(ds[i][2]) for i in range(len(ds))])
    assert gt.shape[0] == M.shape[0], (args.cls, gt.shape, M.shape)

    F = args.w * z_cls(M) + (1 - args.w) * z_cls(B)
    out = {}
    for name, X in (('mapping', M), ('bank', B), ('fused', F)):
        with torch.no_grad():
            am = gk(X.to(args.device)).cpu()
        pr_px = am[:, 0].numpy()
        gt_px = (gt[:, 0].numpy() > 0.5)
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        pr_sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
        r = list(mod.ader_evaluator(pr_px, pr_sp, gt_px, lab.numpy()))
        out[name] = r
        print(f'  [{args.cls}] {name:<8}' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r)))
    # 便于脚本聚合
    for name in ('mapping', 'bank', 'fused'):
        print(f'CSV,{args.cls},{name},' + ','.join(f'{v:.4f}' for v in out[name]))


if __name__ == '__main__':
    main()
