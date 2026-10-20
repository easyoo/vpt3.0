# -*- coding: utf-8 -*-
"""[对照实验] 用**不做中心裁剪**的预处理重算指标，量化"448→中心裁 392"对指标的影响。

本项目评估口径：`Resize(448) → ToTensor → CenterCrop(392) → Normalize`（保留 76.6% 面积）。
本脚本改为：`Resize(448) → ToTensor → Normalize`（**不裁剪**，保留 100% 面积），
其余完全一致：同一 checkpoint、同一打分图定义（compute_fused_map）、同一高斯平滑（k=5,σ=4）、
同一官方 adeval 指标。GT 也按同样几何 resize 到 448。

用法：python eval_nocrop.py --save m4m6r3_15 --class transistor [--no_tta]
"""
import argparse
import importlib.util
import os
import random
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash         # noqa: E402  [统一数据源]
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load_mod(path):
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='m4m6r3_15')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448, help='resize 边长（不裁剪）')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--res_mapping', type=int, default=0)
    ap.add_argument('--global_proto', type=int, default=1)
    ap.add_argument('--metric_head', type=int, default=1)
    ap.add_argument('--subspace', type=int, default=0, help='M27: 1=子空间式打分（须与训练端一致）')
    ap.add_argument('--layer_scale_learn', type=int, default=0, help='M30: 1=可学习逐层融合权重（须与训练端一致）')
    ap.add_argument('--learnable_layers', type=int, default=1,
                    help='[M34] 可学习原型在多少个深度读出（须与训练端一致；默认 1）')
    ap.add_argument('--softmin', type=int, default=0,
                    help='[A1] 1=距离用可训练温度的 softmin 聚合（须与训练端一致）')
    ap.add_argument('--scales', default='1.5')
    ap.add_argument('--flip', type=int, default=1)
    ap.add_argument('--w_learn', type=float, default=0.5)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt_dir', default=None,
                    help='权重目录；默认 results/vpt4/<save>（评基线时可传 results/control35/maponly）')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    ap.add_argument('--save_maps_dir', default='',
                    help='非空则把"测试不裁剪"口径的异常图落盘到该目录（<cls>.map），便于后续融合/可视化')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = args.learnable_layers
    a.softmin = args.softmin
    a.learnable_num_per_layer = args.num_tokens
    a.res_mapping = args.res_mapping
    a.global_proto = args.global_proto
    a.metric_head = args.metric_head
    a.subspace = args.subspace
    a.layer_scale_learn = args.layer_scale_learn
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    a.ms_bank_proto = 0
    mod.args = a
    mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ckpt = os.path.join(args.ckpt_dir or os.path.join('results', 'vpt4', args.save),
                        f'model_{args.cls}_35.pth')
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(args.device).eval()

    # 关键区别：**不做 CenterCrop**
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    # 原始特征库（4 shot，同一预处理）
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    # [统一数据源] 与训练 / bank 分支共用同一份 shot 清单
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test', sort_items=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    scales = tuple(float(x) for x in args.scales.split(','))

    gt_px, pr_px, gt_sp, pr_sp = [], [], [], []
    raw_px = []            # 未平滑的映射图（供后续"先融合后平滑"使用）
    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(args.device)
            am = mod.compute_fused_map(model, img, bank, a.fused_layer, scales=scales,
                                       tta_flip=bool(args.flip), w_learn=args.w_learn)
            raw_px.append(am.float().cpu())     # ← 落盘用：未平滑
            am = gk(am)
            gt = (gt.to(args.device) > 0.5).float()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_px.append(gt); pr_px.append(am); gt_sp.append(label)
            flat = am.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            pr_sp.append(torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1))
    if args.save_maps_dir:
        os.makedirs(args.save_maps_dir, exist_ok=True)
        _m = torch.cat(raw_px, 0).cpu()
        torch.save(_m, os.path.join(args.save_maps_dir, f'{args.cls}.map'))
        print(f'  [maps] 已保存(未平滑) {args.save_maps_dir}/{args.cls}.map  shape={tuple(_m.shape)}')
    gt_px = torch.cat(gt_px, 0)[:, 0].cpu().numpy()
    pr_px = torch.cat(pr_px, 0)[:, 0].cpu().numpy()
    gt_sp = torch.cat(gt_sp).flatten().cpu().numpy()
    pr_sp = torch.cat(pr_sp).flatten().cpu().numpy()
    r = list(mod.ader_evaluator(pr_px, pr_sp, gt_px, gt_sp))
    print(f"[nocrop{args.size}] {args.cls}  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, r)))
    print("  CSV:" + ",".join(f"{v:.4f}" for v in r))


if __name__ == '__main__':
    main()
