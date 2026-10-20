# -*- coding: utf-8 -*-
"""多路异常图融合（与 fuse_nocrop.py 完全同一套归一化/对齐/评测口径，只是允许 >2 路）。

规则保持 v3.7 精神：
  1) 每张图在**类内**做 z 标准化（跨该类所有测试样本、所有像素，与 fuse_nocrop 的 z_cls 相同）；
  2) mapping 侧的多路先等权平均 → z 后再与 bank 侧按 --w 加权；
  3) 高斯平滑 (k=5, σ=4) → 官方 adeval。

用法：
  python fuse_ens_maps.py --class bottle --map_dirs .maps_base15_all,.maps_d3b15_all \
      --bank_dir .errmap_nocrop448_2shot --w 0.5
"""
import argparse
import importlib.util
import json
import os
import sys

import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main_ens', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main_ens'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--w', type=float, default=0.5, help='mapping 侧总权重')
    ap.add_argument('--map_dirs', required=True, help='逗号分隔的多路 mapping 图目录')
    ap.add_argument('--bank_dir', default='.errmap_nocrop448_2shot')
    ap.add_argument('--equal_all', type=int, default=0,
                    help='1 = 把 --map_dirs 与 --bank_dir 全部当作等权视图（F = mean_i z(X_i)），忽略 --w')
    ap.add_argument('--resize_to', type=int, default=0,
                    help='>0 时把所有异常图双线性重采样到该尺寸（用于 672 全尺寸 bank 与 448 图融合）')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    dirs = [d for d in args.map_dirs.split(',') if d]

    def sidecar(d):
        p = os.path.join(HERE, d, f'{args.cls}.json')
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return None
        return None

    maps = [torch.load(os.path.join(HERE, d, f'{args.cls}.map'), weights_only=False).float()
            for d in dirs]
    B = torch.load(os.path.join(HERE, args.bank_dir, f'{args.cls}.map'), weights_only=False).float()
    if int(args.resize_to) > 0:
        S = int(args.resize_to)
        def _rs(x):
            return x if x.shape[-1] == S else torch.nn.functional.interpolate(
                x, size=(S, S), mode='bilinear', align_corners=True)
        _before = {tuple(m.shape[-2:]) for m in maps} | {tuple(B.shape[-2:])}
        maps = [_rs(m) for m in maps]
        B = _rs(B)
        print(f'  [resize] {sorted(_before)} → {S}x{S}')
    for m in maps:
        if m.shape != B.shape:
            raise RuntimeError(f'形状不一致 {m.shape} vs {B.shape}')

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    # ---- 对齐（与 fuse_nocrop.py 完全同构）----
    def _mk_ds(sort_items):
        d = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls), transform=tf,
                             gt_transform=gtf, phase='test', sort_items=sort_items)
        g = torch.stack([d[i][1] for i in range(len(d))], 0).float()
        l = torch.tensor([int(d[i][2]) for i in range(len(d))])
        return d, g, l

    names = dirs + [args.bank_dir]
    scs = [sidecar(n) for n in names]
    both_sc = all(bool(s and s.get('keys')) for s in scs)
    if both_sc:
        ds, gt, lab = _mk_ds(True)
        keys = list(scs[0]['keys'])
        for s in scs:
            if list(s['keys']) != keys:
                raise RuntimeError('[align] 各路侧车键序不一致，拒绝融合')
        if getattr(ds, 'keys', None) != keys:
            raise RuntimeError('[align] 数据集键序与侧车不一致')
        print(f'  [align] 全路侧车 → 按键序对齐 {len(keys)} 个样本')
    else:
        ds, gt, lab = _mk_ds(False)          # 历史未排序序（与无侧车图一致）
        frame = list(ds.keys)

        def _reorder(t, sc, tag):
            if not (sc and sc.get('keys')):
                return t
            k = list(sc['keys'])
            if k == frame:
                return t
            if set(k) != set(frame):
                raise RuntimeError(f'[align] {tag} 侧车键集合与数据集不一致')
            idx = torch.tensor([k.index(x) for x in frame])
            print(f'  [align] {tag} 图带侧车 → 重排到历史坐标系（{len(idx)} 个样本）')
            return t[idx]
        for i, n in enumerate(names):
            if i < len(maps):
                maps[i] = _reorder(maps[i], scs[i], n)
            else:
                B = _reorder(B, scs[i], n)
        print(f'  [align] 公共坐标系 = 历史未排序序（{len(frame)} 个样本）；'
              f'侧车={[bool(s and s.get("keys")) for s in scs]}')

    if int(args.equal_all):
        F = sum(z_cls(x) for x in (maps + [B])) / (len(maps) + 1)
    else:
        M = sum(z_cls(m) for m in maps) / len(maps)
        F = args.w * z_cls(M) + (1 - args.w) * z_cls(B)
    out = {}
    for name, X in list(zip([os.path.basename(d) for d in dirs], maps)) + \
            [('BANK', B), ('fused', F)]:
        with torch.no_grad():
            am = gk(X.to(args.device)).cpu()
        pr_px = am[:, 0].numpy()
        gt_px = (gt[:, 0].numpy() > 0.5)
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        pr_sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
        out[name] = list(mod.ader_evaluator(pr_px, pr_sp, gt_px, lab.numpy()))
        print(f'  [{args.cls}] {name:<24}' +
              '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, out[name])))
    print(f'CSV,{args.cls},fused_ens,' + ','.join(f'{v:.4f}' for v in out['fused']))


if __name__ == '__main__':
    main()
