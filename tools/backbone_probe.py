# -*- coding: utf-8 -*-
"""原始库侧"纯 backbone 特征"探针：比较 DINOv2 ViT-B/14 与 ViT-L/14 的特征质量。

为什么可以不用训练：原始库侧打分（正常 patch 库 kNN + 全局分位数）**不含任何可学习参数**，
完全由 backbone 特征决定，因此这是一个**确定性**对比（没有种子噪声）。

实现要点：**绕过** vpt 的可学习 token 注入（那会污染 patch 特征），直接逐层跑 blocks 取
post-block 的 patch token，逐层求和 → L2 归一化 → d1^0.5·q^0.5（与生产原始侧同式）。

用法：
  CUDA_VISIBLE_DEVICES=<uuid> python backbone_probe.py --backbone base --class transistor
  CUDA_VISIBLE_DEVICES=<uuid> python backbone_probe.py --backbone large --class transistor
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash          # noqa: E402

NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']
SUBN = 1024
CKPT = {
    'base': '/home/jjquan/Downloads/dinov2_vitb14_reg4_pretrain.pth',
    'large': '/home/jjquan/Downloads/dinov2_vitl14_reg4_pretrain_converted.pth',
}


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    spec = importlib.util.spec_from_file_location('audit_main_bp', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main_bp'] = m
    spec.loader.exec_module(m)
    return m


def plain_patch_feats(model, x, layers):
    """不使用任何可学习 token：逐层跑 blocks，取 post-block patch token（[B,N,C]）之和。"""
    with torch.no_grad():
        t = model.prepare_tokens_with_masks(x)
        n_prefix = 1 + model.num_register_tokens
        acc = None
        for i, blk in enumerate(model.blocks):
            t = blk(t)
            if i in layers:
                part = t[:, n_prefix:, :]
                acc = part if acc is None else acc + part
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', choices=['base', 'large'], required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--layers', default='1,2,3,4,6,8,11')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save_map', default='')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    mod = load_mod(args.script)
    layers = [int(v) for v in args.layers.split(',')]

    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.res_mapping = 0
    a.metric_head = 1
    a.token_init_std = 0.02
    mod.args = a
    mod.device = args.device
    model = mod.vision_transformer.__dict__['vit_' + args.backbone](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    sd = torch.load(CKPT[args.backbone], map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[probe] backbone={args.backbone} 载入 {len(sd)} 个张量；'
          f'缺失 {len(missing)} / 多余 {len(unexpected)}（vpt 附加参数本就缺失，不影响本探针）')
    model.to(args.device).eval()
    dim = model.embed_dim

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    print(f'[shots] {args.cls} hash={shots_hash(picks)} picks={picks}')
    normals = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    normals = F.interpolate(normals, scale_factor=args.scale, mode='bilinear', align_corners=False)

    bank = []
    for i in range(normals.shape[0]):
        f = plain_patch_feats(model, normals[i:i + 1].to(args.device), layers)
        bank.append(F.normalize(f, dim=-1).reshape(-1, dim))
    bank = torch.cat(bank, 0)
    M = bank.shape[0]
    q_raw = F.normalize(bank, dim=-1)
    sub = torch.randperm(M, generator=torch.Generator().manual_seed(1234))[:min(SUBN, M)].to(args.device)
    print(f'[probe] 正常 patch 库 {tuple(bank.shape)}（{args.shots} shot × {args.scale}×）')

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls), transform=tf,
                          gt_transform=gtf, phase='test', sort_items=False)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    maps, gts, labs = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, gt, lab, _ = ds[i]
            x = F.interpolate(img[None], scale_factor=args.scale, mode='bilinear', align_corners=False)
            fp = plain_patch_feats(model, x.to(args.device), layers)
            g = int(round(fp.shape[1] ** 0.5))
            p = F.normalize(fp, dim=-1)
            d1, cols = None, []
            ch = 1024
            for a0 in range(0, M, ch):
                b0 = min(a0 + ch, M)
                sim = torch.bmm(p, q_raw[a0:b0].t().unsqueeze(0).expand(p.shape[0], -1, -1))
                dch = (1.0 - sim).clamp_min(0)
                mch = dch.min(-1).values
                d1 = mch if d1 is None else torch.minimum(d1, mch)
                sel = sub[(sub >= a0) & (sub < b0)] - a0
                if sel.numel():
                    cols.append(dch.index_select(-1, sel))
            ds_ = torch.sort(torch.cat(cols, -1), dim=-1).values
            q = ds_[..., min(512, ds_.shape[-1]) - 1]
            am = (d1.clamp_min(1e-6).pow(0.5) * q.clamp_min(1e-6).pow(0.5)).reshape(-1, 1, g, g)
            am = F.interpolate(am, size=(args.size, args.size), mode='bilinear', align_corners=True)
            maps.append(am[0].cpu())
            gt_px = (gt > 0.5).float()
            if gt_px.shape[0] > 1:
                gt_px = gt_px.max(0, keepdim=True)[0]
            gts.append((gt_px > 0.5).to(torch.uint8))
            labs.append(int(lab))
    X = torch.stack(maps, 0)
    gt = torch.stack(gts, 0)
    lab = np.array(labs)
    if args.save_map:
        os.makedirs(args.save_map, exist_ok=True)
        torch.save(X, os.path.join(args.save_map, f'{args.cls}.map'))
    am = gk(X.to(args.device)).cpu()
    flat = am.flatten(1)
    k = max(1, int(flat.shape[1] * 0.01))
    sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
    r = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt[:, 0].numpy(), lab))
    print(f'[probe] {args.backbone} raw-only(baseline formula) ' +
          '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r)) +
          f'   7avg={sum(r)/len(r)*100:.2f}')


if __name__ == '__main__':
    main()
