# -*- coding: utf-8 -*-
"""[路径1/2 关键消融 · CPU 可跑] 学习到的 token_mapping 到底值不值得？

现状：打分 = 余弦距离( token_mapping(patch特征), token_mapping(可学习原型) )。
候选（经典做法）：不做任何映射，直接用原始融合特征，原型来自"正常训练图"的 patch 库（PatchCore 思路）。

本脚本在同一次前向上同时算出 4 种配置（特征空间/原型来源不同，其余完全一致）：
  1) mapped+learn : 现状（映射空间 + 可学习原型）
  2) raw+learn    : 原始空间 + 可学习原型（映射被训练过，此列仅作参考）
  3) raw+bankK    : 原始空间 + 4 张正常图 k-center 选 K 个原型（经典 no-mapping 基线）
  4) raw+bankAll  : 原始空间 + 全部 4×784 个正常 patch 作为原型
  5) mapped+bankK : 映射空间 + 同一批 k-center 原型（分离"空间"与"原型来源"两个因素）

指标用纯 numpy 实现（CPU 无 adeval 的 CUDA 累加器），P-AUPRO 不可得。
"""
import argparse
import importlib.util
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_identity import cpu_metrics, score_from  # 复用 CPU 指标与打分函数

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def kcenter(feats, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    first = int(torch.randint(0, feats.shape[0], (1,), generator=g).item())
    d = 1.0 - feats @ feats[first]
    idx = [first]
    for _ in range(k - 1):
        i = int(torch.argmax(d).item())
        idx.append(i)
        d = torch.minimum(d, 1.0 - feats @ feats[i])
    return torch.stack([feats[i] for i in idx], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_control', 'main-fewshot-control.py'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    mod.args = a
    mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=False)['dino'], strict=False)
    model.to(args.device).eval()
    fused_layer = [1, 2, 3, 4, 6, 8, 11]

    @torch.no_grad()
    def feats_of(imgs):
        ret = model(imgs.to(args.device), is_training=True)
        pfl, fl = ret['patch_feat_layer'], ret['feat_list']
        fp = torch.sum(torch.stack([pfl[i] for i in fused_layer], dim=0), dim=0)
        flr = torch.sum(torch.stack(fl, dim=0), dim=0)
        raw = (F.normalize(fp, dim=-1), F.normalize(flr, dim=-1))
        mp = model.token_mapping(fp)
        mpr = model.token_mapping(flr)
        mapped = (F.normalize(mp, dim=-1), F.normalize(mpr, dim=-1))
        return raw, mapped

    tf, gtf = mod.get_data_transforms(448, 392)
    # 正常训练图（与训练脚本同口径：seed 固定后随机抽 shot 张原图）
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    files = sorted(os.listdir(root))
    random.seed(args.seed)
    picks = random.sample(files, args.shots)
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    raw_n, mapped_n = feats_of(normal)
    bank_raw = raw_n[0].reshape(-1, raw_n[0].shape[-1])
    bank_map = mapped_n[0].reshape(-1, mapped_n[0].shape[-1])
    cs_raw, cs_map = kcenter(bank_raw, args.k), kcenter(bank_map, args.k)
    cs_raw128 = kcenter(bank_raw, 128) if bank_raw.shape[0] > 128 else bank_raw
    print(f'[{args.cls}] 正常图 {args.shots} 张 → bank {tuple(bank_raw.shape)}；'
          f'k-center{args.k} 完成（raw / mapped 两个空间）')

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    if args.limit:
        ds.img_paths, ds.gt_paths, ds.labels, ds.types = (
            ds.img_paths[:args.limit], ds.gt_paths[:args.limit],
            ds.labels[:args.limit], ds.types[:args.limit])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    variants = ['mapped+learn', 'raw+learn', f'raw+bank{args.k}', 'raw+bankAll',
                f'mapped+bank{args.k}', f'mapped+learn+bank{args.k}',
                'mapped+learn+bankAll',
                'fuse_raw_avg', 'fuse_z_avg', 'fuse_rank_avg',
                'fuse_k32', 'fuse_k128', 'fuse_w03', 'fuse_w07',
                'hyb_zpx_rawsp', 'hyb_zpx_banksp', 'hyb_zpx_maxsp']
    acc = {v: dict(gt_px=[], pr_px=[], gt_sp=[], pr_sp=[], contrast=[]) for v in variants}

    t0 = time.time()
    with torch.no_grad():
        for img, gt, label, _ in tqdm(dl, ncols=60, desc=args.cls):
            raw, mapped = feats_of(img)
            B = img.shape[0]
            for b in range(B):
                src = {
                    'mapped+learn': (mapped[0][b], mapped[1][b]),
                    'raw+learn': (raw[0][b], raw[1][b]),
                    f'raw+bank{args.k}': (raw[0][b], cs_raw),
                    'raw+bankAll': (raw[0][b], bank_raw),
                    f'mapped+bank{args.k}': (mapped[0][b], cs_map),
                    f'mapped+learn+bank{args.k}': (mapped[0][b],
                                                   torch.cat([mapped[1][b], cs_map], 0)),
                    'mapped+learn+bankAll': (mapped[0][b],
                                             torch.cat([mapped[1][b], bank_map], 0)),
                }
                gtb = (gt[b:b + 1] > 0.5).float()
                if gtb.shape[1] > 1:
                    gtb = torch.max(gtb, dim=1, keepdim=True)[0]
                for v, (pf, pr) in src.items():
                    am = gk(score_from(pf, pr, img.shape[-1]))
                    if v.startswith('fuse'):
                        continue
                    flat = am.flatten(1)
                    k_top = max(1, int(flat.shape[1] * 0.01))
                    sp = float(torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean())
                    acc[v]['gt_px'].append(gtb[0, 0].cpu().numpy())
                    acc[v]['pr_px'].append(am[0, 0].cpu().numpy())
                    acc[v]['gt_sp'].append(int(label[b]))
                    acc[v]['pr_sp'].append(sp)
                    if float((gtb > 0.5).sum()) > 0:
                        acc[v]['contrast'].append(float(am[gtb > 0.5].mean() - am[gtb <= 0.5].mean()))
                # 打分级融合：现状(映射+可学习原型) × 原始空间正常 patch 库
                m_learn = gk(score_from(*src['mapped+learn'], img.shape[-1]))
                m_bank = gk(score_from(*src['raw+bankAll'], img.shape[-1]))
                def _z(x):
                    return (x - x.mean()) / (x.std() + 1e-8)
                def _rank(x):
                    f = x.flatten()
                    r = torch.argsort(torch.argsort(f)).float()
                    return (r / (f.numel() - 1)).reshape_as(x)
                fus = {
                    'fuse_raw_avg': 0.5 * (m_learn + m_bank),
                    'fuse_z_avg': 0.5 * (_z(m_learn) + _z(m_bank)),
                    'fuse_rank_avg': 0.5 * (_rank(m_learn) + _rank(m_bank)),
                }
                m_bank32 = gk(score_from(raw[0][b], cs_raw, img.shape[-1]))
                m_bank128 = gk(score_from(raw[0][b], cs_raw128, img.shape[-1]))
                fus['fuse_k32'] = 0.5 * (m_learn + m_bank32)
                fus['fuse_k128'] = 0.5 * (m_learn + m_bank128)
                fus['fuse_w03'] = 0.3 * m_learn + 0.7 * m_bank
                fus['fuse_w07'] = 0.7 * m_learn + 0.3 * m_bank
                for v, am in fus.items():
                    flat = am.flatten(1)
                    k_top = max(1, int(flat.shape[1] * 0.01))
                    sp = float(torch.sort(flat, dim=1, descending=True)[0][:, :k_top].mean())
                    acc[v]['gt_px'].append(gtb[0, 0].cpu().numpy())
                    acc[v]['pr_px'].append(am[0, 0].cpu().numpy())
                    acc[v]['gt_sp'].append(int(label[b]))
                    acc[v]['pr_sp'].append(sp)
                    if float((gtb > 0.5).sum()) > 0:
                        acc[v]['contrast'].append(float(am[gtb > 0.5].mean() - am[gtb <= 0.5].mean()))
                # 混合：像素图用 z 融合，图像级分数用"固定尺度"的原始分数（避免逐图归一化破坏跨图可比性）
                fus['hyb_zpx_rawsp'] = fus['fuse_z_avg']
                sp_src = {
                    'hyb_zpx_rawsp': fus['fuse_raw_avg'],
                    'hyb_zpx_banksp': m_bank,
                    'hyb_zpx_maxsp': torch.maximum(m_learn, m_bank),
                }
                for v, sp_map in sp_src.items():
                    acc[v]['gt_px'].append(gtb[0, 0].cpu().numpy())
                    acc[v]['pr_px'].append(fus['fuse_z_avg'][0, 0].cpu().numpy())
                    acc[v]['gt_sp'].append(int(label[b]))
                    flat_sp = sp_map.flatten(1)
                    k_top = max(1, int(flat_sp.shape[1] * 0.01))
                    acc[v]['pr_sp'].append(float(torch.sort(flat_sp, dim=1, descending=True)[0][:, :k_top].mean()))
                    _zmap = fus['fuse_z_avg']
                    if float((gtb > 0.5).sum()) > 0:
                        acc[v]['contrast'].append(float(_zmap[gtb > 0.5].mean() - _zmap[gtb <= 0.5].mean()))

    hdr = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1"]
    print(f'\n[{args.cls}] {os.path.basename(args.ckpt)}  {len(ds)} 张，{time.time()-t0:.0f}s（CPU）')
    print("  " + " " * 16 + " ".join(f"{n:>8}" for n in hdr) + "   in/out对比度")
    res = {}
    for v in variants:
        d = acc[v]
        r = cpu_metrics(np.stack(d['pr_px']), np.array(d['pr_sp']),
                        np.stack(d['gt_px']), np.array(d['gt_sp']))[:6]
        res[v] = r
        print(f"  {v:<16}" + " ".join(f"{x:8.4f}" for x in r) + f"   {np.mean(d['contrast']):.4f}")
    if args.out:
        with open(args.out, 'a') as fh:
            for v in variants:
                fh.write(f"{args.cls},{os.path.basename(args.ckpt)},cpu,{v}," +
                         ",".join(f"{x:.4f}" for x in res[v]) + "\n")


if __name__ == '__main__':
    main()
