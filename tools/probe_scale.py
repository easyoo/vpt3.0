# -*- coding: utf-8 -*-
"""[路径2 候选 · 零训练] 多尺度打分：针对"小/局部缺陷"失败模式的测试期方案。

依据 `probe_failures.py`（screw@10ep）：最难的缺陷类型（scratch_head / manipulated_front）
在 in/out-GT 对比度上只有 ~0.07（好类型 thread_neck 是 0.22），说明打分图弥散。
根因之一：一个 14px 的 patch 在 392 输入下只覆盖很少的原始像素，
小缺陷的异常能量在 patch 内被平均掉 → 距离被稀释。

做法（不改权重、不改训练）：把已预处理好的 392x392 输入再放大 s 倍送进 backbone，
得到 28s x 28s 的更细打分图，再插值回 392 与其它尺度融合。
  base / s1.25 / s1.5 / ms_mean / ms_max
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def build(mod, ckpt, fusion, device):
    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.fusion = fusion
    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def map_at_scale(model, imgs, fused_layer, out_size):
    ret = model(imgs, is_training=True)
    pfl, fl = ret['patch_feat_layer'], ret['feat_list']
    fp = model.layer_fusion(pfl, fused_layer) if hasattr(model, 'layer_fusion') \
        else torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
    flr = torch.sum(torch.stack(fl, dim=0), dim=0)
    p = F.normalize(model.token_mapping(fp), dim=-1)
    q = F.normalize(model.token_mapping(flr), dim=-1)
    s = (1.0 - torch.bmm(p, q.transpose(1, 2))).clamp_min(0).min(dim=-1).values
    g = int(round(s.shape[1] ** 0.5))
    am = s.reshape(-1, 1, g, g)
    return F.interpolate(am, size=out_size, mode='bilinear', align_corners=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--fusion', default='sum', choices=['sum', 'norm'])
    ap.add_argument('--scales', default='1.0,1.25,1.5')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--protocol', type=int, default=392)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_norm', 'main-fewshot-norm.py'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tta_flip', action='store_true',
                    help='额外把每个尺度做一次水平翻转并回投平均（多尺度 x 翻转 TTA）')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))  # 用 --script 所在目录的 dinov2 副本
    mod = load_mod(args.script)
    device = args.device
    fused_layer = [1, 2, 3, 4, 6, 8, 11]
    model = build(mod, args.ckpt, args.fusion, device)
    scales = [float(s) for s in args.scales.split(',')]

    tf, gtf = mod.get_data_transforms(448, 392)
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rm = None if args.protocol == 392 else args.protocol
    names_v = [f's{s}' for s in scales] + ['ms_mean', 'ms_max']
    if args.tta_flip:
        names_v += ['ms_flip_mean', 'ms_flip_max', 'ms_flip_min', 'ms_flip_med',
                    'ms_flip_mean_med']
    acc = {v: dict(gt_px=[], pr_px=[], gt_sp=[], pr_sp=[]) for v in names_v}

    for img, gt, label, _ in tqdm(dl, ncols=60, desc=args.cls):
        img = img.to(device)
        maps = {}
        for s in scales:
            im = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
            maps[f's{s}'] = map_at_scale(model, im, fused_layer, img.shape[-2:])
            if args.tta_flip:
                maps[f'f{s}'] = map_at_scale(model, im.flip(3), fused_layer, img.shape[-2:]).flip(3)
        stack = torch.stack([maps[f's{s}'] for s in scales], dim=0)
        maps['ms_mean'] = stack.mean(dim=0)
        maps['ms_max'] = stack.max(dim=0).values
        if args.tta_flip:
            fstack = torch.stack([maps[f'f{s}'] for s in scales], dim=0)
            both = torch.cat([stack, fstack], dim=0)
            maps['ms_flip_mean'] = both.mean(dim=0)
            maps['ms_flip_max'] = both.max(dim=0).values
            # 「多尺度一致性」变体：只有所有视图都判异常才保留（压误报），以及逐像素中位数（抗离群）
            maps['ms_flip_min'] = both.min(dim=0).values
            maps['ms_flip_med'] = both.median(dim=0).values
            _m = maps['ms_flip_mean']
            # [BUGFIX] _m 是 4 维 [B,1,H,W]，必须用 view(-1,1,1,1) 才能正确广播；
            # 之前写成 view(-1,1,1) 会与 [B,1,H,W] 广播成 [B,B,H,W]，导致高斯核报 channels=16。
            maps['ms_flip_mean_med'] = _m - _m.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
        for v in names_v:
            am = maps[v]
            gt_b = F.interpolate(gt, size=am.shape[-1], mode='nearest') if rm is not None else gt
            am = gk(am)
            gt_b = (gt_b > 0.5).float()
            if gt_b.shape[1] > 1:
                gt_b = torch.max(gt_b, dim=1, keepdim=True)[0]
            flat = am.flatten(1)
            k_top = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean(dim=1)
            acc[v]['gt_px'].append(gt_b[:, 0].cpu().numpy())
            acc[v]['pr_px'].append(am[:, 0].cpu().numpy())
            acc[v]['gt_sp'].append(label.numpy())
            acc[v]['pr_sp'].append(sp.cpu().numpy())

    hdr = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    print(f"\n[{args.cls}] {os.path.basename(args.ckpt)} fusion={args.fusion} protocol={args.protocol}")
    print("  " + " " * 10 + " ".join(f"{n:>8}" for n in hdr))
    res = {}
    for v in names_v:
        a = acc[v]
        r = mod.ader_evaluator(np.concatenate(a['pr_px']), np.concatenate(a['pr_sp']),
                               np.concatenate(a['gt_px']), np.concatenate(a['gt_sp']))
        res[v] = r
        print(f"  {v:<10}" + " ".join(f"{x:8.4f}" for x in r))
    if args.out:
        with open(args.out, 'a') as fh:
            for v in names_v:
                fh.write(f"{args.cls},{os.path.basename(args.ckpt)},{args.fusion},{args.protocol},{v}," +
                         ",".join(f"{x:.4f}" for x in res[v]) + "\n")


if __name__ == '__main__':
    main()
