# -*- coding: utf-8 -*-
"""多随机种子集成的离线评测（等权、先验，不涉及测试集调参）。

参照协议与 v3.7 完全一致：
  1. 每个种子的 mapping 打分图各自做**类内 z 标准化**；
  2. 集成 = 多种子 z 图**等权平均**（先验权重 1/S，无搜索）；
  3. 与 bank 图按 0.5/0.5 融合（先验）；高斯平滑 k=5,σ=4；官方 adeval。

候选：每个单种子 / 集成(ENS) / 每个单种子⊕bank / ENS⊕bank。
用法：
  CUDA_VISIBLE_DEVICES=<uuid> python seed_ens_eval.py --tag seedens_mv2 \
    --map_dirs .maps_simple_nocrop448_mv2,.maps_seed2_mv2,.maps_seed3_mv2 \
    --bank_dir .errmap_nocrop448_2shot --cls "bottle cable ..." --data_path /home/jjquan/MVTec-AD
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    spec = importlib.util.spec_from_file_location('audit_main_se', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main_se'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--map_dirs', required=True, help='逗号分隔，顺序即种子顺序（第一个视为基线种子）')
    ap.add_argument('--bank_dir', default='')
    ap.add_argument('--cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--fuse_w_list', default='0.5')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    dirs = [d for d in args.map_dirs.split(',') if d]
    fws = [float(w) for w in args.fuse_w_list.split(',')]

    out = []
    for cls in args.cls.split():
        maps = []
        for d in dirs:
            p = os.path.join(HERE, d, f'{cls}.map')
            if not os.path.exists(p):
                print(f'  [skip] {cls}: 缺少 {p}')
                maps = None
                break
            maps.append(torch.load(p, weights_only=False).float())
        if maps is None:
            continue
        shapes = {tuple(m.shape) for m in maps}
        assert len(shapes) == 1, (cls, shapes)
        ds = mod.MVTecDataset(root=os.path.join(args.data_path, cls),
                              transform=tf, gt_transform=gtf, phase='test', sort_items=False)
        gt = torch.stack([ds[i][1] for i in range(len(ds))], 0).float()
        lab = np.array([int(ds[i][2]) for i in range(len(ds))])
        assert gt.shape[0] == maps[0].shape[0], (cls, gt.shape, maps[0].shape)
        gt_px = (gt[:, 0].numpy() > 0.5)
        zs = [z_cls(m) for m in maps]
        ens = torch.stack(zs, 0).mean(0)                     # 等权（先验）
        cands = []
        for i, (d, z) in enumerate(zip(dirs, zs)):
            cands.append((f's{i+1}_solo', z))
        if len(zs) > 1:
            cands.append((f'ENS{len(zs)}', ens))
        if args.bank_dir:
            bank = torch.load(os.path.join(HERE, args.bank_dir, f'{cls}.map'),
                              weights_only=False).float()
            assert bank.shape == maps[0].shape, (cls, bank.shape, maps[0].shape)
            zB = z_cls(bank)
            for name, z in list(cands):        # 快照：避免边遍历边追加
                for w in fws:
                    cands.append((f'fuse[{name},w={w}]', w * z + (1 - w) * zB))
        for name, X in cands:
            with torch.no_grad():
                am = gk(X.to(args.device)).cpu()
            flat = am.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(1).numpy()
            r = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab))
            out.append([cls, name] + [v * 100 for v in r])
        del ds
        print(f'  [{cls}] done', flush=True)

    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    path = os.path.join(HERE, 'logs', f'{args.tag}.csv')
    with open(path, 'w') as fh:
        for row in out:
            fh.write(','.join([row[0], row[1]] + [f'{v:.4f}' for v in row[2:]]) + '\n')
    print('saved', path)
    import collections
    acc = collections.defaultdict(list)
    for row in out:
        acc[row[1]].append(row[2:])
    print(f"\n{'candidate':<24}{'类':>3}" + ''.join(f'{n:>9}' for n in NAMES) + f"{'7avg':>9}")
    for k in acc:
        v = acc[k]
        a = [float(np.mean([r[i] for r in v])) for i in range(7)]
        print(f"{k:<24}{len(v):>3}" + ''.join(f'{x:>9.2f}' for x in a) + f"{sum(a)/7:>9.2f}")


if __name__ == '__main__':
    main()
