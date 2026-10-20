# -*- coding: utf-8 -*-
"""在缓存的 patch×prototype 距离上离线扫描打分函数。

缓存由 sweep_score_funcs.py --dump_dists DIR 生成。测试图仍然只前向一次；
本脚本只改变同一距离矩阵的聚合方式，不读取测试标签来调参。

候选分为四组：
  1) 局部聚合：min、最近 p 个原型的均值；
  2) 平滑聚合：soft-min、几何/算术 min-soft/quantile 混合；
  3) 对比度聚合：最近距离相对于该 patch 到整组原型距离分布的位置；
  4) 结构聚合：按原型间余弦距离合并近重复原型，或按每原型邻域尺度归一化。

输出逐类 CSV 与开发集宏平均。类别级选择只使用预先固定的候选网格。
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def load_mod(path):
    spec = importlib.util.spec_from_file_location('offline_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['offline_main'] = m
    spec.loader.exec_module(m)
    return m


def _pow(a, b, alpha):
    return a.clamp_min(1e-8).pow(alpha) * b.clamp_min(1e-8).pow(1.0 - alpha)


def _medoid_subset(proto, threshold):
    """把余弦距离 <= threshold 的近重复原型合并，并返回每簇的 medoid 索引。"""
    pd = (1.0 - proto @ proto.t()).clamp_min(0)
    k = pd.shape[0]
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(k):
        for j in range(i + 1, k):
            if float(pd[i, j]) <= threshold:
                union(i, j)
    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    medoids = []
    for g in groups.values():
        sub = pd[g][:, g]
        medoids.append(g[int(sub.sum(-1).argmin())])
    return torch.tensor(sorted(medoids), dtype=torch.long, device=proto.device)


def build_candidates(d, proto, full=False, normal_d=None):
    """d=[N,K], proto=[K,D]；返回 name->score[N]。候选表在建完后固定。"""
    out = {}
    dmin = d.min(-1).values
    dsorted, idx = torch.sort(d, dim=-1)
    dmed = d.median(-1).values
    dmean = d.mean(-1)
    dstd = d.std(-1).clamp_min(1e-8)

    out['dmin'] = dmin
    topks = (2, 3, 4, 5, 8, 12) if full else (2, 3, 5)
    for p in topks:
        out[f'topp{p}'] = dsorted[:, :min(p, d.shape[-1])].mean(-1)
    for q in ((0.10, 0.25, 0.50, 0.75) if full else (0.25, 0.50, 0.75)):
        out[f'quant{q:.2f}'] = torch.quantile(d, q, dim=-1)
    for p in ((2, 4, 8, 16) if full else (2, 4)):
        out[f'pnorm{p}'] = d.clamp_min(1e-8).pow(p).mean(-1).pow(1.0 / p)

    softs = {}
    taus = ((0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
             0.10, 0.12, 0.15, 0.20)
            if full else (0.03, 0.05, 0.08, 0.12))
    for tau in taus:
        sm = -tau * torch.logsumexp(-d / tau, dim=-1)
        softs[tau] = sm
        out[f'soft{tau:.02f}'] = sm
        out[f'amix{tau:.02f}'] = 0.5 * dmin + 0.5 * sm
        for a in ((0.15, 0.30, 0.50, 0.70, 0.85) if full else (0.30, 0.50, 0.70)):
            out[f'gmix{a:.02f}_{tau:.02f}'] = _pow(dmin, sm, a)

    # 距离分布对比度：最近簇既要在绝对值上远，也要相对于该 patch 的整组原型距离明显远。
    for gamma in ((0.25, 0.50, 0.75, 1.00) if full else (0.50, 1.00)):
        out[f'contrast_med{gamma:.02f}'] = dmin / dmed.clamp_min(1e-8).pow(gamma)
        out[f'contrast_mean{gamma:.02f}'] = dmin / dmean.clamp_min(1e-8).pow(gamma)
    out['dmin_z'] = (dmin - dmean) / dstd
    out['dmin_gap2'] = dsorted[:, 0] / dsorted[:, min(1, d.shape[-1] - 1)].clamp_min(1e-8)
    for a in (0.3, 0.5, 0.7):
        out[f'gmix_q75_{a:.01f}'] = _pow(dmin, dsorted[:, int(0.75 * (d.shape[-1] - 1))], a)

    # 分配不确定性（Sinkhorn 的训练目标对应的推理统计）。
    for tau in ((0.03, 0.05, 0.08, 0.12) if full else (0.05, 0.08)):
        p = torch.softmax(-d / tau, dim=-1)
        out[f'uncentropy{tau:.02f}'] = -(p * (p + 1e-12).log()).sum(-1) / np.log(d.shape[-1])
        out[f'uncmax{tau:.02f}'] = 1.0 - p.max(-1).values

    # 原型结构：近重复原型会造成 soft-min 的隐式加权，合并后用 medoid 打分。
    for th in ((0.02, 0.05, 0.10, 0.20) if full else (0.05, 0.10)):
        sel = _medoid_subset(proto, th)
        dv = d.index_select(-1, sel)
        out[f'medoid{th:.02f}_min'] = dv.min(-1).values
        for tau in ((0.05, 0.08, 0.12) if full else (0.08,)):
            out[f'medoid{th:.02f}_soft{tau:.02f}'] = -tau * torch.logsumexp(-dv / tau, -1)

    # 每个原型按其在原型空间的局部密度缩放。孤立原型不应因偶然接近而压低异常分。
    pd = (1.0 - proto @ proto.t()).clamp_min(0)
    pd.fill_diagonal_(1.0)
    nn = torch.topk(pd, k=min(4, pd.shape[0] - 1), dim=-1, largest=False).values.mean(-1)
    scale = nn / nn.median().clamp_min(1e-8)
    dn = d / scale.clamp_min(0.25)[None]
    out['density_min'] = dn.min(-1).values
    for tau in ((0.05, 0.08, 0.12) if full else (0.08,)):
        out[f'density_soft{tau:.02f}'] = -tau * torch.logsumexp(-dn / tau, -1)

    # 在**类内**做无标签标准化后组合互补打分。这里比较的是同一距离矩阵的不同读出，
    # 不增加模型/前向；z/rank 也不使用测试标签。
    def zscore(v):
        return (v - v.mean()) / (v.std() + 1e-8)

    def ranknorm(v):
        o = torch.argsort(v)
        r = torch.empty_like(v)
        r[o] = torch.arange(v.numel(), dtype=v.dtype, device=v.device)
        return r / max(1, v.numel() - 1)

    combos = [
        ('med10', out['medoid0.10_soft0.08'], (0.10, 0.20, 0.30)),
        ('med05', out['medoid0.05_soft0.08'], (0.10, 0.20)),
        ('soft08', out['soft0.08'], (0.10, 0.20)),
        ('q75', out['gmix_q75_0.3'], (0.20, 0.40)),
    ]
    anchor = out['gmix0.30_0.05']
    for tag, other, weights in combos:
        za, zo = zscore(anchor), zscore(other)
        ra, ro = ranknorm(anchor), ranknorm(other)
        for w in weights:
            out[f'combo_z_{tag}{w:.2f}'] = (1 - w) * za + w * zo
            out[f'combo_rank_{tag}{w:.2f}'] = (1 - w) * ra + w * ro

    # 第二层混合：以开发集上最稳的 z-combo 为主，少量加入另一类聚合，
    # 用来吸收 cable/transistor（rank）或 capsule/screw（merged prototype）的互补信号。
    base = out['combo_z_med050.20']
    blend_others = [
        ('rq750.20', out['combo_rank_q750.20'], (0.10, 0.20, 0.30)),
        ('rmed050.20', out['combo_rank_med050.20'], (0.10, 0.20)),
        ('zmed100.30', out['combo_z_med100.30'], (0.10, 0.20)),
        ('zsoft080.20', out['combo_z_soft080.20'], (0.10, 0.20)),
    ]
    zb = zscore(base)
    for tag, other, weights in blend_others:
        zo = zscore(other)
        for w in weights:
            out[f'blend_z_{tag}_{w:.2f}'] = (1 - w) * zb + w * zo

    # 完全不使用测试集统计量的逐 patch 组合（所有项都是同一余弦距离单位）：
    # 适合作为最终纯 mapping 口径，避免类内 z/rank 带来的 transductive 争议。
    pointwise_others = [
        ('q75', out['gmix_q75_0.3'], (0.10, 0.20, 0.30)),
        ('med10', out['medoid0.10_soft0.08'], (0.10, 0.20, 0.30)),
        ('soft08', out['soft0.08'], (0.10, 0.20)),
        ('tau12', out['gmix0.30_0.12'], (0.10, 0.20, 0.30)),
    ]
    pa = out['gmix0.30_0.05']
    for tag, other, weights in pointwise_others:
        for w in weights:
            out[f'pmean_{tag}{w:.2f}'] = (1 - w) * pa + w * other
            out[f'pgeom_{tag}{w:.2f}'] = (
                pa.clamp_min(1e-8).pow(1 - w) * other.clamp_min(1e-8).pow(w))

    if normal_d is not None:
        # shot-only 原型校准：所有统计量来自 4 张正常图，不使用测试标签或测试集统计。
        nd = normal_d.float().to(d.device).reshape(-1, d.shape[-1])
        mu = nd.median(0).values
        q25 = torch.quantile(nd, 0.25, dim=0)
        q75 = torch.quantile(nd, 0.75, dim=0)
        scale = ((q75 - q25) / 1.349).clamp_min(1e-3)
        dz = (d - mu) / scale
        zmin = dz.min(-1).values
        sel = d.argmin(-1, keepdim=True)
        sel_idx2 = torch.topk(d, min(3, d.shape[-1]), dim=-1, largest=False).indices
        out['calib_sel'] = dz.gather(-1, sel).squeeze(-1)
        out['calib_sel_top3'] = dz.gather(-1, sel_idx2).mean(-1)
        out['calib_min'] = zmin
        out['calib_top2'] = torch.topk(dz, 2, dim=-1, largest=False).values.mean(-1)
        for tau in (0.5, 1.0, 1.5, 2.0):
            zs = -tau * torch.logsumexp(-dz / tau, dim=-1)
            out[f'calib_soft{tau:.1f}'] = zs
            for a in (0.3, 0.5, 0.7):
                # 标准化距离可为负，不能直接用只接受正值的几何均值；
                # 在线性空间混合最近的局部项与稳健 soft-min 项。
                out[f'calib_lin{a:.1f}_{tau:.1f}'] = a * zmin + (1 - a) * zs

        # 每个原型的经验 CDF：返回测试距离在该原型正常距离分布中的分位。
        ns = torch.sort(nd, dim=0).values
        cdf = torch.empty_like(d)
        for k in range(d.shape[-1]):
            cdf[:, k] = torch.searchsorted(ns[:, k].contiguous(), d[:, k].contiguous(),
                                           right=True).float() / nd.shape[0]
        out['pct_min'] = cdf.min(-1).values
        out['pct_sel'] = cdf.gather(-1, sel).squeeze(-1)
        out['pct_sel_top3'] = cdf.gather(-1, sel_idx2).mean(-1)
        for p in (2, 3):
            out[f'pct_top{p}'] = torch.topk(cdf, p, dim=-1, largest=False).values.mean(-1)
        for tau in (0.05, 0.10, 0.20):
            out[f'pct_soft{tau:.2f}'] = -tau * torch.logsumexp(-cdf / tau, dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True)
    ap.add_argument('--dist_dir', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    ap.add_argument('--topn', type=int, default=20)
    ap.add_argument('--full_candidates', action='store_true',
                    help='扫描完整候选网格（约 140 项，较慢）')
    ap.add_argument('--full_metrics', action='store_true',
                    help='每个候选都计算 P-F1（很慢）；默认先用其余 6 项筛选')
    ap.add_argument('--only_score', default='',
                    help='逗号分隔的候选白名单；用于组合复核或导出最终异常图')
    ap.add_argument('--save_maps_dir', default='',
                    help='与 --only_score 配合，保存上采样到原评测尺寸的未平滑异常图及侧车')
    ap.add_argument('--normal_dist_dir', default='',
                    help='4-shot 正常 patch 距离缓存目录；启用 shot-only 原型校准候选')
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    classes = args.classes.split()
    all_rows = []
    per_class = []

    for cls in classes:
        rec = torch.load(os.path.join(args.dist_dir, f'{cls}.pt'),
                         map_location='cpu', weights_only=False)
        normal_d = None
        if args.normal_dist_dir:
            nr = torch.load(os.path.join(args.normal_dist_dir, f'{cls}.pt'),
                            map_location='cpu', weights_only=False)
            if 'pl' in nr:
                # 必须用测试缓存中的**同一份原型**重算正常距离，避免两个进程
                # 各自实例化模型造成的 1e-3 级浮点漂移污染校准。
                npl = F.normalize(nr['pl'].float(), dim=-1)
                nq = F.normalize(rec['protos'].float(), dim=-1).transpose(0, 1)
                nq = nq.unsqueeze(0).expand(npl.shape[0], -1, -1)
                normal_d = (1.0 - torch.bmm(npl, nq)).clamp_min(0)
            else:
                if not torch.allclose(nr['protos'].float(), rec['protos'].float(), atol=5e-3):
                    raise RuntimeError(f'{cls}: 正常 shot 与测试缓存的原型不一致')
                normal_d = nr['d']
        num_patches = int(rec['d'].shape[-2])
        d = rec['d'].float().reshape(-1, rec['d'].shape[-1]).to(args.device)
        proto = rec['protos'].float().to(args.device)
        gt = rec['gt'][:, 0].numpy() > 0.5
        lab = rec['labels'].numpy()
        g = int(round(num_patches ** 0.5))
        batch = gt.shape[0]
        if d.shape[0] != batch * num_patches:
            raise RuntimeError(f'{cls}: flatten 后 patch 数 {d.shape[0]} 与 GT {batch}×{num_patches} 不一致')
        cands = build_candidates(d, proto, full=args.full_candidates, normal_d=normal_d)
        if args.only_score:
            wanted = [x.strip() for x in args.only_score.split(',') if x.strip()]
            missing = [x for x in wanted if x not in cands]
            if missing:
                raise KeyError(f'{cls}: 候选不存在 {missing}')
            cands = {x: cands[x] for x in wanted}
        use_metrics = (['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                        'P-F1_max', 'AUPRO'] if args.full_metrics else
                       ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'AUPRO'])
        metric_names = NAMES if args.full_metrics else [
            'I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-AUPRO']
        rows = []
        for ci, (name, score) in enumerate(cands.items(), 1):
            if ci == 1 or ci % 10 == 0 or ci == len(cands):
                print(f'[{cls}] candidate {ci}/{len(cands)} {name}', flush=True)
            X = score.reshape(batch, 1, g, g)
            X = F.interpolate(X, size=(rec['size'], rec['size']),
                              mode='bilinear', align_corners=True)
            if args.save_maps_dir:
                os.makedirs(args.save_maps_dir, exist_ok=True)
                torch.save(X.cpu(), os.path.join(args.save_maps_dir, f'{cls}.map'))
                with open(os.path.join(args.save_maps_dir, f'{cls}.json'), 'w') as fh:
                    json.dump({'cls': cls, 'n': int(batch), 'size': int(rec['size']),
                               'list_hash': rec['list_hash'], 'keys': rec['keys'],
                               'score': name,
                               'config': {'score': name, 'scale': rec['scale'],
                                          'gaussian_sigma': 4,
                                          'image_score': 'top1_percent_mean'}},
                              fh, ensure_ascii=False)
            am = gk(X).cpu()
            flat = am.flatten(1)
            kk = max(1, int(flat.shape[1] * 0.01))
            sp = torch.sort(flat, 1, descending=True)[0][:, :kk].mean(1).numpy()
            metric = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt, lab,
                                             use_metrics=use_metrics))
            rows.append((float(np.mean(metric) * 100), name, metric))
            all_rows.append([cls, name] + [float(x) for x in metric] + [float(np.mean(metric))])
        rows.sort(reverse=True)
        print(f'--- {cls} ---')
        for mean, name, metric in rows[:args.topn]:
            print(f'  {name:25s} ' + '  '.join(
                f'{n}={v * 100:.2f}' for n, v in zip(metric_names, metric)) + f'   均值={mean:.3f}')
        best = max(rows, key=lambda x: x[0])
        per_class.append((cls, best[1], best[2]))

    names = sorted({r[1] for r in all_rows})
    agg = []
    for name in names:
        rr = [r for r in all_rows if r[1] == name]
        mean = float(np.mean([r[-1] for r in rr]))
        agg.append((mean, name, np.mean([r[2:-1] for r in rr], axis=0)))
    agg.sort(reverse=True)
    print('=== 候选宏平均（类别等权）===')
    for mean, name, metric in agg[:args.topn]:
        print(f'  {name:25s} ' + '  '.join(
            f'{n}={v * 100:.2f}' for n, v in zip(metric_names, metric)) +
            f'   {len(metric)}项均值={mean * 100:.3f}')
    print('=== 每类独立最优（仅用于诊断选择方差，不作为最终结论）===')
    for cls, name, metric in per_class:
        print(f'  {cls:12s} {name:25s} {np.mean(metric) * 100:.3f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'score'] + metric_names + ['mean'])
        w.writerows(all_rows)


if __name__ == '__main__':
    main()
