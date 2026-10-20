# -*- coding: utf-8 -*-
"""扫描“可学习图 M + 原始 448 bank + 全尺寸 672 bank”的单图融合规则。

每个候选最终只产生一张异常图：先对各分支做类内归一化，再线性/一致性组合，
最后执行与 v3.7 相同的 Gaussian(5,4) 平滑和 top-1% 图像分。
归一化不读取标签。权重网格在脚本中预注册，开发集选出的前几名应放到 held-out 类别复核。
"""
import argparse
import csv
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def load_mod(path):
    spec = importlib.util.spec_from_file_location('fusion3_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['fusion3_main'] = m
    spec.loader.exec_module(m)
    return m


def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def rank01(x):
    flat = x.flatten()
    order = torch.argsort(flat)
    r = torch.empty_like(flat)
    r[order] = torch.arange(flat.numel(), dtype=flat.dtype, device=flat.device)
    return (r / max(1, flat.numel() - 1)).reshape(x.shape)


def nscore(x):
    r = rank01(x).clamp(1e-4, 1 - 1e-4)
    return torch.erfinv(2 * r - 1) * (2 ** 0.5)


def robust_z(x):
    """中位数/IQR 标准化，降低极端异常像素对整类尺度估计的影响。"""
    f = x.flatten()
    step = max(1, f.numel() // 1_000_000)
    fs = f[::step]                      # 固定步长抽样，规避 torch.quantile 2^24 上限
    med = torch.quantile(fs, 0.50)
    q25 = torch.quantile(fs, 0.25)
    q75 = torch.quantile(fs, 0.75)
    return (x - med) / (q75 - q25).clamp_min(1e-8)


def clipped_z(x):
    """先截断 0.1%–99.9% 尾部，再做普通 z；与 top-1% 图像分口径更匹配。"""
    f = x.flatten()
    step = max(1, f.numel() // 1_000_000)
    fs = f[::step]
    lo = torch.quantile(fs, 0.001)
    hi = torch.quantile(fs, 0.999)
    return z(x.clamp(lo, hi))


def tail_scale(x):
    """用类内中位数和 99% 分位把各分支映射到可比较的尾部分数空间。"""
    f = x.flatten()
    step = max(1, f.numel() // 1_000_000)
    fs = f[::step]
    med = torch.quantile(fs, 0.50)
    q99 = torch.quantile(fs, 0.99)
    return (x - med) / (q99 - med).clamp_min(1e-8)


WEIGHTS = {
    'eq': (1 / 3, 1 / 3, 1 / 3),
    'm25b35_40': (0.25, 0.35, 0.40),
    'm30b30_40': (0.30, 0.30, 0.40),
    'm30b40_30': (0.30, 0.40, 0.30),
    'm40b30_30': (0.40, 0.30, 0.30),
    'm20b40_40': (0.20, 0.40, 0.40),
    'm40b20_40': (0.40, 0.20, 0.40),
    'm40b40_20': (0.40, 0.40, 0.20),
    'm50b25_25': (0.50, 0.25, 0.25),
    'm25b50_25': (0.25, 0.50, 0.25),
    'm25b25_50': (0.25, 0.25, 0.50),
}


def candidates(M, B1, B2):
    """逐项生成候选图，避免同时保存数十张 [N,1,448,448] 造成主机内存峰值。"""
    zs = {'z': (z(M), z(B1), z(B2)),
          'rz': (robust_z(M), robust_z(B1), robust_z(B2)),
          'cz': (clipped_z(M), clipped_z(B1), clipped_z(B2)),
          'tail': (tail_scale(M), tail_scale(B1), tail_scale(B2)),
          'logz': (z(torch.log1p(M / M.mean().clamp_min(1e-8))),
                   z(torch.log1p(B1 / B1.mean().clamp_min(1e-8))),
                   z(torch.log1p(B2 / B2.mean().clamp_min(1e-8)))),
          'rank': (rank01(M), rank01(B1), rank01(B2)),
          'ns': (nscore(M), nscore(B1), nscore(B2))}
    for tn, xs in zs.items():
        for wn, ws in WEIGHTS.items():
            yield f'{tn}_{wn}', sum(w * x for w, x in zip(ws, xs))
        # 每种尺度下都额外测试 bank-only 和低 mapping 权重。
        xs0, xb1, xb2 = xs
        yield f'{tn}_bank_eq', 0.5 * xb1 + 0.5 * xb2
        for mw in (0.05, 0.10, 0.15):
            yield f'{tn}_lowM{mw:.2f}', mw * xs0 + (1 - mw) * (0.5 * xb1 + 0.5 * xb2)
        xavg = (xs0 + xb1 + xb2) / 3
        xspan = torch.maximum(torch.maximum(xs0, xb1), xb2) - \
                torch.minimum(torch.minimum(xs0, xb1), xb2)
        for lam in (0.05, 0.10, 0.20):
            yield f'{tn}_avg_span{lam:.2f}', xavg - lam * xspan

    # 普通 z 与稳健 z 的混合，降低"极端尾部被压缩/放大"之间的方差。
    xz = (zs['z'][0] + zs['z'][1] + zs['z'][2]) / 3
    xr = (zs['rz'][0] + zs['rz'][1] + zs['rz'][2]) / 3
    for wr in (0.25, 0.50, 0.75):
        yield f'mix_zrz{wr:.2f}', (1 - wr) * xz + wr * xr

    zm, zb1, zb2 = zs['z']
    # M 与两张 bank 性能差距较大时，测试 5%–15% 的小权重，仍保持“两 bank + M”三路融合。
    for mw in (0.05, 0.10, 0.15):
        yield f'z_lowM{mw:.2f}_beq', mw * zm + (1 - mw) * (0.5 * zb1 + 0.5 * zb2)

    # 一致性/残差门控：以两 bank 为主，只允许 mapping 在强响应或与 bank 同向时补充；
    # 对三路分歧较大的像素施加软惩罚。仍输出单张异常图。
    zb = 0.5 * (zb1 + zb2)
    zavg = (zm + zb1 + zb2) / 3
    resid = zm - zb
    for a in (0.10, 0.25, 0.50):
        yield f'z_bank_posM{a:.2f}', zb + a * torch.relu(zm)
        yield f'z_bank_resid{a:.2f}', zb + a * torch.relu(resid)
        yield f'z_bank_tanhM{a:.2f}', zb + a * torch.tanh(zm)
    for th in (0.25, 0.50, 1.00):
        for a in (0.10, 0.25, 0.50):
            yield f'z_bank_hiM{a:.2f}_t{th:.2f}', zb + a * torch.relu(zm - th)
    for lam in (0.05, 0.10, 0.20):
        yield f'zavg_dis2_{lam:.2f}', zavg - lam * resid.pow(2)
        yield f'zavg_disabs_{lam:.2f}', zavg - lam * resid.abs()
        span = torch.maximum(torch.maximum(zm, zb1), zb2) - \
               torch.minimum(torch.minimum(zm, zb1), zb2)
        yield f'zavg_span_{lam:.2f}', zavg - lam * span
    for scale in (0.5, 1.0, 2.0):
        yield f'z_lse{scale:.1f}', torch.logsumexp(
            torch.stack([scale * zm, scale * zb1, scale * zb2], 0), dim=0) / scale
    # 两个 bank 先融合，再作为单一 bank 分支与 M 组合。
    for bw in (0.25, 0.50, 0.75):
        bank = (1 - bw) * zb1 + bw * zb2
        for mw in (0.25, 1 / 3, 0.40, 0.50):
            yield f'z_m{mw:.2f}_b{bw:.2f}', mw * zm + (1 - mw) * bank
            bank_mix = (1 - bw) * torch.maximum(zb1, zb2) + bw * torch.minimum(zb1, zb2)
            yield f'z_bankmax_m{mw:.2f}_b{bw:.2f}', mw * zm + (1 - mw) * bank_mix

    # 三路一致高响应时额外增强，最终仍是一张图。
    zavg = (zm + zb1 + zb2) / 3
    zmin = torch.minimum(torch.minimum(zm, zb1), zb2)
    for gamma in (0.10, 0.25, 0.50):
        yield f'zavg_minboost{gamma:.2f}', zavg + gamma * zmin
    yield 'z_product', zm * (0.5 * zb1 + 0.5 * zb2)
    yield 'z_max', torch.maximum(torch.maximum(zm, zb1), zb2)
    yield 'bank_z_eq', 0.5 * zb1 + 0.5 * zb2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True)
    ap.add_argument('--map_dir', required=True)
    ap.add_argument('--bank448_dir', default='.errmap_nocrop448')
    ap.add_argument('--bank672_dir', default='.errmap672_mv4')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--topn', type=int, default=25)
    ap.add_argument('--full_metrics', action='store_true',
                    help='默认先用 6 项指标筛选；最终短名单再加 P-F1')
    ap.add_argument('--only_rules', default='',
                    help='逗号分隔的融合规则白名单；用于全类只复核短名单')
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    rows = []
    per_class = []
    use_metrics = (['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                    'P-F1_max', 'AUPRO'] if args.full_metrics else
                   ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'AUPRO'])
    metric_names = NAMES if args.full_metrics else NAMES[:5] + [NAMES[-1]]

    for cls in args.classes.split():
        # 两个 bank 目录是历史生成物，没有 sidecar，v3.7/fuse_ens_maps 的统一口径是
        # 以 MVTecDataset(sort_items=False) 的历史顺序为公共坐标系，再按 sidecar 重排 M。
        ds = mod.MVTecDataset(root=os.path.join(args.data_path, cls), transform=tf,
                              gt_transform=gtf, phase='test', sort_items=False)
        gt = torch.stack([ds[i][1] for i in range(len(ds))], 0).float()
        lab = torch.tensor([int(ds[i][2]) for i in range(len(ds))]).numpy()
        M = torch.load(os.path.join(HERE, args.map_dir, f'{cls}.map'),
                       map_location='cpu', weights_only=False).float()
        B1 = torch.load(os.path.join(HERE, args.bank448_dir, f'{cls}.map'),
                        map_location='cpu', weights_only=False).float()
        B2 = torch.load(os.path.join(HERE, args.bank672_dir, f'{cls}.map'),
                        map_location='cpu', weights_only=False).float()
        if M.shape[0] != gt.shape[0] or B1.shape[0] != gt.shape[0] or B2.shape[0] != gt.shape[0]:
            raise RuntimeError(f'{cls}: 样本数不一致 {M.shape[0]}/{B1.shape[0]}/{B2.shape[0]}/{gt.shape[0]}')
        if B2.shape[-1] != args.size:
            B2 = F.interpolate(B2, size=(args.size, args.size), mode='bilinear',
                               align_corners=True)
        # bank 图没有侧车，沿用 v3.7 的既有历史顺序；M 图若有 sidecar，则重排到该坐标系。
        sc = os.path.join(HERE, args.map_dir, f'{cls}.json')
        if os.path.exists(sc):
            import json
            keys = json.load(open(sc)).get('keys')
            if keys and list(keys) != list(ds.keys):
                if set(keys) != set(ds.keys):
                    raise RuntimeError(f'{cls}: M 图键集合与数据集不一致')
                idx = torch.tensor([list(keys).index(x) for x in ds.keys])
                M = M[idx]
                print(f'  [align] {cls}: M sidecar → 历史顺序 ({len(idx)} 个样本)')
        cands = candidates(M, B1, B2)
        allowed = {x.strip() for x in args.only_rules.split(',') if x.strip()}
        gt_px = gt[:, 0].numpy() > 0.5
        rr = []
        for name, Fmap in cands:
            if allowed and name not in allowed:
                continue
            with torch.no_grad():
                am = gk(Fmap.to(args.device)).cpu()
            flat = am.flatten(1)
            kk = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, 1, descending=True)[0][:, :kk].mean(1).numpy()
            metric = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab,
                                             use_metrics=use_metrics))
            rr.append((float(np.mean(metric) * 100), name, metric))
            rows.append([cls, name] + [float(x) for x in metric] + [float(np.mean(metric))])
        rr.sort(reverse=True)
        print(f'--- {cls} ---')
        for val, name, metric in rr[:args.topn]:
            print(f'  {name:25s} ' + '  '.join(
                f'{n}={v * 100:.2f}' for n, v in zip(metric_names, metric)) + f'   均值={val:.3f}')
        per_class.append((cls, rr[0][1], rr[0][0]))

    names = sorted({r[1] for r in rows})
    agg = []
    for name in names:
        rr = [r for r in rows if r[1] == name]
        agg.append((float(np.mean([r[-1] for r in rr])),
                    name, np.mean([r[2:-1] for r in rr], axis=0)))
    agg.sort(reverse=True)
    print('=== 融合候选宏平均（类别等权）===')
    for val, name, metric in agg[:args.topn]:
        print(f'  {name:25s} ' + '  '.join(
            f'{n}={v * 100:.2f}' for n, v in zip(metric_names, metric)) +
            f'   {len(metric)}项均值={val * 100:.3f}')
    print('=== 每类独立最优（诊断用）===')
    for cls, name, val in per_class:
        print(f'  {cls:12s} {name:25s} {val:.3f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'fusion'] + metric_names + ['mean'])
        w.writerows(rows)


if __name__ == '__main__':
    main()
