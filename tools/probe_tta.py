# -*- coding: utf-8 -*-
"""[路径2 候选 · 零训练] 测试期增强(TTA) 与 邻域聚合 对 mapping 分支的影响。

动机（来自失败样本分析的先验）：
  * 图像级指标在纹理类已经饱和，唯一战场是像素级 P-AUROC / P-AP / AUPRO；
  * 打分图是 28x28 的 patch 级最近原型余弦距离，边界处易弥散；
  * 训练时用了 8 个旋转增强，测试时却完全没有利用这种对称性。

本脚本在冻结权重下比较四种打分：
  base         : 现状（28x28 → 双线性上采样 → 高斯平滑）
  base+nb      : 先对 28x28 做 3x3 平均池化（邻域聚合，PatchCore 常用）
  base+tta     : 4 旋转 × {原图, 水平翻转} = 8 视图，回投到原坐标系后取平均
  base+tta+nb  : 两者结合
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
def map28(model, imgs, fused_layer):
    """返回 [B, 28, 28] 的原型最近余弦距离图（未上采样）"""
    ret = model(imgs, is_training=True)
    pfl, fl = ret['patch_feat_layer'], ret['feat_list']
    fp = model.layer_fusion(pfl, fused_layer) if hasattr(model, 'layer_fusion') \
        else torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
    flr = torch.sum(torch.stack(fl, dim=0), dim=0)
    p = F.normalize(model.token_mapping(fp), dim=-1)
    q = F.normalize(model.token_mapping(flr), dim=-1)
    sim = torch.bmm(p, q.transpose(1, 2))
    s = (1.0 - sim).clamp_min(0).min(dim=-1).values
    n = s.shape[1]
    g = int(round(n ** 0.5))
    return s.reshape(-1, g, g)


def views(img):
    """产出 (名称, 变换后的图, 把该视图的 map 还原回原坐标系的函数)"""
    out = []
    for k in range(4):
        im = torch.rot90(img, k, dims=(2, 3))
        out.append((f'rot{k}', im, (lambda m, k=k: torch.rot90(m, -k, dims=(1, 2)))))
    for k in range(4):
        im = torch.rot90(img.flip(3), k, dims=(2, 3))
        out.append((f'flip_rot{k}', im, (lambda m, k=k: torch.rot90(m, -k, dims=(1, 2)).flip(2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--fusion', default='sum', choices=['sum', 'norm'])
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--protocol', type=int, default=392)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_norm', 'main-fewshot-norm.py'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))  # 用 --script 所在目录的 dinov2 副本
    mod = load_mod(args.script)
    device = args.device
    fused_layer = [1, 2, 3, 4, 6, 8, 11]
    model = build(mod, args.ckpt, args.fusion, device)

    tf, gtf = mod.get_data_transforms(448, 392)
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test', sort_items=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rm = None if args.protocol == 392 else args.protocol
    variants = ['base', 'base+nb', 'base+tta', 'base+tta+nb']
    acc = {v: dict(gt_px=[], pr_px=[], gt_sp=[], pr_sp=[]) for v in variants}

    for img, gt, label, _ in tqdm(dl, ncols=60, desc=args.cls):
        img = img.to(device)
        m28 = map28(model, img, fused_layer)                       # [B,28,28]
        m28_tta = None
        for name, im, inv in views(img):
            m = map28(model, im, fused_layer)
            m = inv(m)
            m28_tta = m if m28_tta is None else m28_tta + m
        m28_tta = m28_tta / 8.0
        for v in variants:
            mm = m28
            if 'tta' in v:
                mm = m28_tta
            if 'nb' in v:
                mm = F.avg_pool2d(mm.unsqueeze(1), 3, stride=1, padding=1)[:, 0]
            am = F.interpolate(mm.unsqueeze(1), size=img.shape[-1] if rm is None else rm,
                               mode='bilinear', align_corners=True)
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

    names = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
    print(f"\n[{args.cls}] {os.path.basename(args.ckpt)} fusion={args.fusion} protocol={args.protocol}")
    print("  " + " " * 14 + " ".join(f"{n:>8}" for n in names))
    res = {}
    for v in variants:
        a = acc[v]
        r = mod.ader_evaluator(np.concatenate(a['pr_px']), np.concatenate(a['pr_sp']),
                               np.concatenate(a['gt_px']), np.concatenate(a['gt_sp']))
        res[v] = r
        print(f"  {v:<14}" + " ".join(f"{x:8.4f}" for x in r))
    if args.out:
        with open(args.out, 'a') as fh:
            for v in variants:
                fh.write(f"{args.cls},{os.path.basename(args.ckpt)},{args.fusion},{args.protocol},{v}," +
                         ",".join(f"{x:.4f}" for x in res[v]) + "\n")


if __name__ == '__main__':
    main()
