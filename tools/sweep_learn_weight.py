# -*- coding: utf-8 -*-
"""扫描"可学习侧权重 w"：mapping = w·z(L) + (1-w)·z(R)，再 0.5/0.5 与增强 bank 融合。

为什么可以离线：R（mapping 分支的原始库侧）完全由**冻结主干**决定，两版模型里逐位相同，
所以任意 w 的 mapping 图都可由已落盘的 L、R 组合得到，无需重新前向。
所有 z 标准化都是**类内**（跨该类全部测试样本、全部像素），与 fuse_nocrop 一致。

用法：
  python sweep_learn_weight.py --L_dir /tmp/gmix_maps --R_dir .maps_v37_mv4_raw \
      --bank_dir .errmap_nocrop448 --classes "bottle ..." --ws 0.5,0.6,0.7,0.8,0.9,1.0
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
    spec = importlib.util.spec_from_file_location('audit_w', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_w'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L_dir', required=True, help='可学习侧异常图（w=1.0 的等价物）')
    ap.add_argument('--R_dir', required=True, help='mapping 原始库侧异常图（w=0.0）')
    ap.add_argument('--bank_dir', required=True, help='增强 bank 异常图')
    ap.add_argument('--classes', required=True)
    ap.add_argument('--ws', default='0.5,0.6,0.7,0.8,0.9,1.0')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    ws = [float(x) for x in args.ws.split(',')]
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
        names = [args.L_dir, args.R_dir, args.bank_dir]
        scs = [sc(n) for n in names]
        L = torch.load(os.path.join(HERE, args.L_dir, f'{cls}.map'), weights_only=False).float()
        R = torch.load(os.path.join(HERE, args.R_dir, f'{cls}.map'), weights_only=False).float()
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
            L, R, B = ro(L, scs[0]), ro(R, scs[1]), ro(B, scs[2])
        zL, zR, zB = z_cls(L), z_cls(R), z_cls(B)
        gt_px = (gt[:, 0].numpy() > 0.5)
        for w in ws:
            # [口径修正] v3.7 是**先按原始尺度混合，再 z 标准化**（两路尺度差很大，先 z 后混合不等价）
            M = w * L + (1 - w) * R
            F = 0.5 * z_cls(M) + 0.5 * zB
            with torch.no_grad():
                am = gk(F.to(args.device)).cpu()
            flat = am.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
            acc[w].append(list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab.numpy())))
    print(f'--- L={args.L_dir} ---')
    best = None
    for w in ws:
        a = np.array(acc[w]).mean(0)
        print(f'  w={w:.2f}  ' + '  '.join(f'{n}={v*100:.2f}' for n, v in zip(NAMES, a)) +
              f'   7项均值={a.mean()*100:.3f}')
        if best is None or a.mean() > best[1]:
            best = (w, a.mean() * 100)
    print(f'  → 最优 w={best[0]:.2f}  均值={best[1]:.3f}')


if __name__ == '__main__':
    main()
