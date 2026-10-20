# -*- coding: utf-8 -*-
"""原始库侧"纯 backbone 特征"探针（DINOv3 版）。

与 backbone_probe.py 完全同一套协议（1.5× + 448、逐层 post-block 求和、L2 归一化、
d1^0.5·q^0.5、adeval 七指标），唯一区别是特征主干换成 DINOv3（timm 实现，RoPE）。

原始库侧打分（正常 patch 库 kNN + 全局分位数）不含任何可学习参数，完全由 backbone
特征决定，因此这是一个**确定性**对比（没有种子噪声）。

用法：
  python backbone_probe3.py --class transistor --layers 1,2,3,4,6,8,11
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
import timm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash          # noqa: E402

NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']
SUBN = 1024
D3_CKPT = {
    'dinov3b': '/home/jjquan/Downloads/dinov3/model.safetensors',
    'dinov3l': '/home/jjquan/Downloads/dinov3l/model.safetensors',
}
D3_TIMM = {
    'dinov3b': 'vit_base_patch16_dinov3.lvd1689m',
    'dinov3l': 'vit_large_patch16_dinov3.lvd1689m',
}


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    spec = importlib.util.spec_from_file_location('audit_main_bp3', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main_bp3'] = m
    spec.loader.exec_module(m)
    return m


def build_d3(backbone):
    from safetensors.torch import load_file
    model = timm.create_model(D3_TIMM[backbone], pretrained=False, num_classes=0)
    state = load_file(D3_CKPT[backbone])
    state = {k: v for k, v in state.items() if not k.startswith('head.')}
    info = model.load_state_dict(state, strict=False)
    print(f'[probe] {backbone} 载入 {len(state)} 张量；缺失 {len(info.missing_keys)} / 多余 {len(info.unexpected_keys)}')
    if len(info.missing_keys) > 20 or len(info.unexpected_keys) > 20:
        raise RuntimeError(f'权重键对不上：missing={info.missing_keys[:8]} unexpected={info.unexpected_keys[:8]}')
    return model


def plain_patch_feats_d3(model, x, layers):
    """DINOv3 纯特征：timm forward_intermediates 取 post-block patch token 之和。"""
    with torch.no_grad():
        # timm 返回 (最终特征, 逐层中间特征列表)；中间特征默认 NCHW
        _final, outs = model.forward_intermediates(x, indices=list(layers), norm=False)
    npt = model.num_prefix_tokens
    acc = None
    for o in outs:
        # NCHW 输出只含 patch token（不含 cls/register），NLC 输出含前缀
        part = o.flatten(2).transpose(1, 2) if o.dim() == 4 else o[:, npt:, :]
        acc = part if acc is None else acc + part
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', choices=['dinov3b', 'dinov3l'], default='dinov3b')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--layers', default='1,2,3,4,6,8,11')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    mod = load_mod(args.script)
    layers = [int(v) for v in args.layers.split(',')]
    model = build_d3(args.backbone).to(args.device).eval()
    dim = model.embed_dim
    print(f'[probe] 主干 {args.backbone} dim={dim} patch={model.patch_embed.patch_size} 层={layers}')

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
        f = plain_patch_feats_d3(model, normals[i:i + 1].to(args.device), layers)
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
            fp = plain_patch_feats_d3(model, x.to(args.device), layers)
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
