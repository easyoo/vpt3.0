# -*- coding: utf-8 -*-
"""[失败样本分析] 逐图诊断 mapping 分支（type=1）的失败模式。

输出（CSV + 终端汇总）：
  per-image: sp_score, label, defect_type, mean_in_gt, mean_out_gt, contrast(=in-out), px_ap
汇总：
  1) 按缺陷类型统计"异常图像级分数 / 正常图像级分数"的分离度 → 哪类缺陷最难
  2) 最严重的漏检（异常图分数最低）与最严重的误报（正常图分数最高）
  3) in/out GT 对比度 → 定位是否弥散（P-AP 低的主因）
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--protocol', type=int, default=392)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code', 'main-fewshot-audit.py'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--scale', default='1.0',
                    help="'1.0' = 现状；'ms' = 多尺度(1.0/1.25/1.5)平均（§9 方案）")
    args = ap.parse_args()

    def score_map(model, img, fused_layer, mode):
        def _one(im):
            ret = model(im, is_training=True)
            pfl, fl = ret['patch_feat_layer'], ret['feat_list']
            fp = torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
            flr = torch.sum(torch.stack(fl, dim=0), dim=0)
            pm = model.token_mapping(fp)
            lm = model.token_mapping(flr)
            am = mod.compute_anomaly_scores(pm.cpu().numpy(), lm.cpu().numpy(), im.shape[-1])
            return am
        if mode == 'ms':
            acc = None
            base = img.shape[-2:]
            for s in (1.0, 1.25, 1.5):
                im = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear',
                                                        align_corners=False)
                am = _one(im)
                if am.shape[-2:] != base:
                    am = F.interpolate(am, size=base, mode='bilinear', align_corners=True)
                acc = am if acc is None else acc + am
            return acc / 3.0
        return _one(img)

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))  # 用 --script 所在目录的 dinov2 副本
    mod = load_mod(args.script)
    device = args.device
    fused_layer = [1, 2, 3, 4, 6, 8, 11]

    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.temperature = 0.07
    a.scaled_factor = 3
    mod.args = a            # compute_anomaly_scores / get_data_transforms 依赖模块级全局
    mod.device = device
    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(device).eval()

    data_transform, gt_transform = mod.get_data_transforms(448, 392)
    test_data = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                                 transform=data_transform, gt_transform=gt_transform, phase='test')
    dl = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rm = None if args.protocol == 392 else args.protocol

    rows = []
    with torch.no_grad():
        for img, gt, label, paths in tqdm(dl, ncols=60, desc=args.cls):
            img = img.to(device)
            am = score_map(model, img, fused_layer, args.scale)
            if rm is not None:
                am = F.interpolate(am, size=rm, mode='bilinear', align_corners=False)
                gt_b = F.interpolate(gt, size=rm, mode='nearest')
            else:
                gt_b = gt
            am = gk(am)
            gt_b = (gt_b > 0.5).float()
            if gt_b.shape[1] > 1:
                gt_b = torch.max(gt_b, dim=1, keepdim=True)[0]
            for b in range(img.shape[0]):
                s = am[b, 0]
                g = gt_b[b, 0]
                flat = s.flatten().unsqueeze(0)
                k_top = max(1, int(flat.shape[1] * 0.01))
                sp = torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean().item()
                gsum = float((g > 0.5).sum().item())
                mean_in = float(s[g > 0.5].mean().item()) if gsum > 0 else float('nan')
                mean_out = float(s[g <= 0.5].mean().item())
                # 逐图像素 AP（用整图排序近似：anomaly 图内异常像素占比很低时足够近似）
                p = paths[b]
                dtype = os.path.basename(os.path.dirname(p))
                rows.append(dict(path=p, cls=args.cls, dtype=dtype, label=int(label[b]),
                                 sp_score=sp, mean_in=mean_in, mean_out=mean_out,
                                 contrast=(mean_in - mean_out) if gsum > 0 else float('nan'),
                                 gt_ratio=gsum / g.numel()))

    anom = [r for r in rows if r['label'] == 1]
    norm = [r for r in rows if r['label'] == 0]
    print(f"\n[{args.cls}] {os.path.basename(args.ckpt)}  "
          f"正常图 {len(norm)} 张，异常图 {len(anom)} 张")
    print(f"  正常图 sp_score: mean={np.mean([r['sp_score'] for r in norm]):.4f} "
          f"max={np.max([r['sp_score'] for r in norm]):.4f}")
    print(f"  异常图 sp_score: mean={np.mean([r['sp_score'] for r in anom]):.4f} "
          f"min={np.min([r['sp_score'] for r in anom]):.4f}")
    print(f"  异常图 in/out GT 对比度: mean={np.nanmean([r['contrast'] for r in anom]):.4f} "
          f"（越小说明异常图越弥散→像素级指标差）")
    print("  按缺陷类型（异常图）：")
    for t in sorted(set(r['dtype'] for r in anom)):
        sub = [r for r in anom if r['dtype'] == t]
        print(f"    {t:<14} n={len(sub):3d}  sp_score mean={np.mean([r['sp_score'] for r in sub]):.4f}  "
              f"contrast mean={np.nanmean([r['contrast'] for r in sub]):.4f}")
    print("  最严重漏检（异常图分数最低 5 张）：")
    for r in sorted(anom, key=lambda r: r['sp_score'])[:5]:
        print(f"    sp={r['sp_score']:.4f} contrast={r['contrast']:.4f} {r['dtype']}/{os.path.basename(r['path'])}")
    print("  最严重误报（正常图分数最高 5 张）：")
    for r in sorted(norm, key=lambda r: -r['sp_score'])[:5]:
        print(f"    sp={r['sp_score']:.4f} {os.path.basename(r['path'])}")

    if args.out:
        with open(args.out, 'w') as fh:
            fh.write("cls,dtype,label,sp_score,mean_in,mean_out,contrast,gt_ratio,path\n")
            for r in rows:
                fh.write(f"{r['cls']},{r['dtype']},{r['label']},{r['sp_score']:.6f},"
                         f"{r['mean_in']:.6f},{r['mean_out']:.6f},{r['contrast']:.6f},"
                         f"{r['gt_ratio']:.6f},{r['path']}\n")


if __name__ == '__main__':
    main()
