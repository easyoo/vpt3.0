# -*- coding: utf-8 -*-
"""bank-conditioned 与局部空间融合搜索。

默认固定两张输入图：mapping M、bank448 B1，禁止读取 bank672。
仅为复现历史作废实验保留显式 --allow_bank672_legacy 开关。
候选只改变融合规则，不重训、不增加模型或视图。
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
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def _default_script():
    """Support both audit-workspace and packaged vpt3.8 layouts."""
    for path in (
        os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'),
        os.path.join(HERE, '..', 'code', 'main-fewshot-vpt4.py'),
        os.path.join(HERE, 'main-fewshot-vpt4.py'),
    ):
        if os.path.exists(path):
            return os.path.normpath(path)
    return os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')


def load_mod(path):
    spec = importlib.util.spec_from_file_location('adv_fusion_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['adv_fusion_main'] = m
    spec.loader.exec_module(m)
    return m


def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def z_per_image(x):
    """Image-local z-score; uses no statistics from other test images."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - mean) / (std + 1e-8)


def identity(x):
    """No normalization; retain the original cached branch scores."""
    return x


def box(x, k):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2)


def local_z(x, k):
    m = box(x, k)
    v = box(x * x, k) - m * m
    return (x - m) / v.clamp_min(1e-8).sqrt()


def candidates(M, B1, B2):
    zm, zb1, zb2 = z(M), z(B1), z(B2)
    zb = 0.5 * (zb1 + zb2)
    zeq = (zm + zb1 + zb2) / 3
    yield 'z_eq', zeq
    yield 'bank_z_eq', zb

    # mapping 只能通过 bank 的支持门控进入；bank 分歧越大，门控越弱。
    for tau in (0.25, 0.50, 1.00, 2.00):
        gate = torch.exp(-torch.abs(zb1 - zb2) / tau)
        for a in (0.10, 0.25, 0.50, 0.75):
            yield f'agree_t{tau:.2f}_a{a:.2f}', zb + a * zm * gate
            yield f'agree_pos_t{tau:.2f}_a{a:.2f}', zb + a * torch.relu(zm) * gate

    # bank 支持强度作为 soft gate。
    for scale in (0.5, 1.0, 2.0):
        gate = torch.sigmoid(scale * zb)
        for a in (0.10, 0.25, 0.50):
            yield f'bgate_s{scale:.1f}_a{a:.2f}', zb + a * zm * gate
            yield f'bgate_pos_s{scale:.1f}_a{a:.2f}', zb + a * torch.relu(zm) * gate

    # 三路一致时门控；只有 bank 与 mapping 同向才让 mapping 加成。
    for scale in (0.5, 1.0, 2.0):
        gate = torch.sigmoid(scale * torch.minimum(zb, zm))
        for a in (0.10, 0.25, 0.50):
            yield f'consensus_s{scale:.1f}_a{a:.2f}', zeq + a * gate * torch.relu(zm)

    # 局部均值与局部标准化：比较区域一致异常与孤立异常。
    for k in (5, 9, 15, 21, 25, 31, 41):
        sm, sb1, sb2 = box(zm, k), box(zb1, k), box(zb2, k)
        se = z((sm + sb1 + sb2) / 3)
        yield f'box_eq{k}', se
        yield f'mix_box_eq{k}', 0.5 * zeq + 0.5 * se
        for a in (0.10, 0.25, 0.50):
            yield f'z_detail{k}_a{a:.2f}', zeq + a * (zm - sm)
            yield f'bank_detail{k}_a{a:.2f}', zeq + a * (zb - 0.5 * (sb1 + sb2))

    for k in (9, 15, 25):
        lm, lb1, lb2 = local_z(zm, k), local_z(zb1, k), local_z(zb2, k)
        le = z((lm + lb1 + lb2) / 3)
        yield f'localz_eq{k}', le
        yield f'mix_localz_eq{k}', 0.5 * zeq + 0.5 * le
        for a in (0.10, 0.25, 0.50):
            yield f'mix_localz_map{k}_a{a:.2f}', zeq + a * lm

    # 多尺度区域池化平均，降低单一窗口对目标尺寸的敏感性。
    multi = []
    for k in (15, 25, 41):
        sm, sb1, sb2 = box(zm, k), box(zb1, k), box(zb2, k)
        multi.append(z((sm + sb1 + sb2) / 3))
    yield 'box_ms_eq', torch.stack(multi, 0).mean(0)


def candidates_two_branch(M, B1, normalize=z):
    """只用 mapping + 原始 448 bank，禁止任何 672 bank 信息。"""
    zm, zb = normalize(M), normalize(B1)
    zeq = normalize(0.5 * (zm + zb))
    yield 'z_eq2', zeq
    for k in (5, 9, 15, 21, 25, 31, 41):
        sm, sb = box(zm, k), box(zb, k)
        se = normalize(0.5 * (sm + sb))
        yield f'box_eq2_{k}', se
        yield f'mix_box_eq2_{k}', 0.5 * zeq + 0.5 * se
    multi = []
    for k in (15, 25, 41):
        sm, sb = box(zm, k), box(zb, k)
        multi.append(normalize(0.5 * (sm + sb)))
    yield 'box_ms_eq2', torch.stack(multi, 0).mean(0)


def candidates_no_raw_bank(L, B):
    """只保留可学习侧 L 与 88 张增强样本构建的 MemoryBank B。

    原 mapping raw 侧的 4-shot 原始 bank 被删除。由于 B 已包含原图及其增强，
    这里只用低权重 L 校正 B，并保留原 25×25 区域池化与 Gaussian 后处理。
    """
    zl, zb = z(L), z(B)
    for w in (0.10, 0.20, 0.30, 0.40):
        fl = box(zl, 25)
        fb = box(zb, 25)
        yield f'box_Lw{w:.2f}_B', z(w * fl + (1.0 - w) * fb)
    for w in (0.15, 0.20, 0.30, 0.40):
        base = w * zl + (1.0 - w) * zb
        fl = box(zl, 25)
        fb = box(zb, 25)
        regional = z(w * fl + (1.0 - w) * fb)
        yield f'mix_Lw{w:.2f}_B', 0.5 * z(base) + 0.5 * regional


def candidates_no_raw_bank_v2(L, B):
    """只使用增强 bank B 和可学习侧 L，测试全局/区域/局部残差叠加。"""
    zl, zb = z(L), z(B)
    rb25, rl25 = box(zb, 25), box(zl, 25)
    yield 'B_reg25', z(rb25)
    for w in (0.05, 0.10, 0.20):
        yield f'B_reg25_L{w:.2f}', z((1.0 - w) * rb25 + w * rl25)
    for w in (0.05, 0.10, 0.20):
        yield f'B_global_L{w:.2f}', z((1.0 - w) * zb + w * zl)

    # 局部残差：增强 bank 相对其邻域的高频证据，用于补回 raw 侧的定位锐度。
    for k in (9, 15):
        db = zb - box(zb, k)
        dl = zl - box(zl, k)
        for a in (0.10, 0.25, 0.50):
            yield f'B25_resB{k}_a{a:.2f}', z(rb25 + a * torch.relu(db))
            yield (f'B25_L0.10_resB{k}_a{a:.2f}',
                   z(0.90 * rb25 + 0.10 * rl25 + a * torch.relu(db)))
            yield (f'B25_L0.10_resL{k}_a{a:.2f}',
                   z(0.90 * rb25 + 0.10 * rl25 + a * torch.relu(dl)))


def candidates_no_raw_bank_v3(L, B):
    """L 与增强 B 的非线性互补：正响应、乘积、交集和 bank 门控。"""
    zl, zb = z(L), z(B)
    rb, rl = box(zb, 25), box(zl, 25)
    yield 'B_reg25', z(rb)
    for a in (0.05, 0.10, 0.25, 0.50):
        yield f'B_posL_a{a:.2f}', z(rb + a * torch.relu(rl))
        yield f'B_minL_a{a:.2f}', z(rb + a * torch.minimum(
            torch.relu(rb), torch.relu(rl)))
        yield f'B_prodL_a{a:.2f}', z(rb + a * torch.relu(rb) * torch.relu(rl))
    for s in (0.5, 1.0, 2.0):
        gate = torch.sigmoid(s * rb)
        for a in (0.10, 0.25, 0.50):
            yield f'B_gateL_s{s:.1f}_a{a:.2f}', z(rb + a * gate * torch.relu(rl))
    yield 'B_geomL', z(torch.sqrt(
        torch.relu(rb + 1.0).clamp_min(1e-6) *
        torch.relu(rl + 1.0).clamp_min(1e-6)) - 1.0)
    yield 'B_geomL2', z(torch.sqrt(
        torch.relu(rb + 2.0).clamp_min(1e-6) *
        torch.relu(rl + 2.0).clamp_min(1e-6)) - 2.0)


def candidates_bank_only(L, B):
    """纯增强 bank 的后处理搜索；L 只用于少量可选的低权重校正。"""
    zl, zb = z(L), z(B)
    yield 'B_raw', zb
    for k in (5, 9, 15, 21, 25, 31, 41):
        rb = box(zb, k)
        yield f'B_box{k}', z(rb)
        yield f'B_mix{k}', 0.5 * zb + 0.5 * z(rb)
        yield f'B_box{k}_L0.03', z(0.97 * rb + 0.03 * box(zl, k))
        yield f'B_box{k}_L0.05', z(0.95 * rb + 0.05 * box(zl, k))
    yield 'B_box_ms', torch.stack(
        [z(box(zb, k)) for k in (15, 25, 41)], 0).mean(0)
    multi = [z(box(zb, k)) for k in (15, 21, 25, 31, 41)]
    yield 'B_box_ms2', torch.stack(multi, 0).mean(0)
    yield 'B_box_ms3', (
        0.10 * multi[0] + 0.15 * multi[1] + 0.20 * multi[2] +
        0.25 * multi[3] + 0.30 * multi[4])
    multi_l = [z(box(zl, k)) for k in (15, 21, 25, 31, 41)]
    bms3 = (0.10 * multi[0] + 0.15 * multi[1] + 0.20 * multi[2] +
            0.25 * multi[3] + 0.30 * multi[4])
    lms3 = (0.10 * multi_l[0] + 0.15 * multi_l[1] + 0.20 * multi_l[2] +
            0.25 * multi_l[3] + 0.30 * multi_l[4])
    for w in (0.02, 0.03, 0.05, 0.08):
        yield f'B_box_ms3_L{w:.2f}', z((1.0 - w) * bms3 + w * lms3)


def candidates_stage_no_raw(L, S1, S2):
    """仅使用增强 bank 的两个 stage 和可学习侧 L；不读取原始 4-shot bank。"""
    zl, z1, z2 = z(L), z(S1), z(S2)
    rb1, rb2, rl = box(z1, 25), box(z2, 25), box(zl, 25)
    for w in (0.0, 0.05, 0.10, 0.20):
        for a in (0.25, 0.50, 0.75):
            rb = a * rb1 + (1.0 - a) * rb2
            yield f'stage_box_a{a:.2f}_L{w:.2f}', z((1.0 - w) * rb + w * rl)
    # 两 stage 差异作为互补信号。
    for a in (0.10, 0.25, 0.50):
        rb = 0.5 * (rb1 + rb2)
        yield f'stage_absdiff_a{a:.2f}', z(rb + a * torch.abs(rb1 - rb2))
    yield 'stage_s1_only', z(rb1)
    yield 'stage_s2_only', z(rb2)


def candidates_enhanced_raw(L, B, R):
    """B 与 L 来自增强 bank，R 是同一 88 视图在 mapping raw 空间的弱辅助图。"""
    zl, zb, zr = z(L), z(B), z(R)
    rb, rl, rr = box(zb, 25), box(zl, 25), box(zr, 25)
    for wl in (0.0, 0.05, 0.10):
        for wr in (0.05, 0.10, 0.20, 0.30):
            den = 1.0 + wl + wr
            yield f'B_L{wl:.2f}_R{wr:.2f}', z((rb + wl * rl + wr * rr) / den)
    for wl in (0.05, 0.10):
        for wr in (0.05, 0.10, 0.20):
            base = (zb + wl * zl + wr * zr) / (1.0 + wl + wr)
            reg = z((rb + wl * rl + wr * rr) / (1.0 + wl + wr))
            yield f'mix_B_L{wl:.2f}_R{wr:.2f}', 0.5 * z(base) + 0.5 * reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True)
    ap.add_argument('--map_dir', required=True)
    ap.add_argument('--bank448_dir', default='.errmap_nocrop448')
    ap.add_argument('--bank672_dir', default='.errmap672_mv4',
                    help='仅历史作废实验使用；默认不会读取')
    ap.add_argument('--two_branch', action='store_true',
                    help='兼容旧命令；两路 448 bank 现为默认模式')
    ap.add_argument('--allow_bank672_legacy', action='store_true',
                    help='仅复现历史作废结果时允许读取 672 bank')
    ap.add_argument('--no_raw_bank', action='store_true',
                    help='输入图为 learn-only L；删除 mapping 的 4-shot raw bank，'
                         '仅用低权重 L 校正增强 448 bank')
    ap.add_argument('--no_raw_bank_v2', action='store_true',
                    help='输入图为 learn-only L；使用增强 bank 的区域、全局和局部残差重融合')
    ap.add_argument('--no_raw_bank_v3', action='store_true',
                    help='输入图为 learn-only L；使用增强 bank 的非线性门控/交集融合')
    ap.add_argument('--bank_only_search', action='store_true',
                    help='搜索纯增强 bank 的平滑尺度及其极小权重 L 校正')
    ap.add_argument('--stage_no_raw', action='store_true',
                    help='输入为 L、增强 bank stage1、stage2；不读取原始 4-shot bank')
    ap.add_argument('--enhanced_raw', action='store_true',
                    help='输入为 L、增强 bank B、以及仅由 88 视图构建的 mapping raw 弱图 R')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=_default_script())
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--only_rules', default='')
    ap.add_argument('--full_metrics', action='store_true')
    ap.add_argument('--topn', type=int, default=20)
    ap.add_argument('--gauss_k', type=int, default=5)
    ap.add_argument('--gauss_sigma', type=float, default=4.0)
    ap.add_argument('--image_top_ratio', type=float, default=0.01,
                    help='图像异常分数使用的最热像素比例；默认 0.01')
    ap.add_argument('--z_mode', choices=('class', 'image', 'none'), default='class',
                    help='class=当前口径（测试集整类统计量，转导）；'
                         'image=逐图标准化（不使用其他测试图）；'
                         'none=两个分支和融合结果都不做 z-score')
    ap.add_argument('--require_mapping_sidecar', action='store_true',
                    help='缺少 mapping 样本顺序侧车时立即失败，禁止静默按索引错配')
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=args.gauss_k, sigma=args.gauss_sigma).to(args.device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    use_metrics = (['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                    'P-F1_max', 'AUPRO'] if args.full_metrics else
                   ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'AUPRO'])
    metric_names = NAMES if args.full_metrics else NAMES[:5] + [NAMES[-1]]
    rows = []

    for cls in args.classes.split():
        ds = mod.MVTecDataset(root=os.path.join(args.data_path, cls), transform=tf,
                              gt_transform=gtf, phase='test', sort_items=False)
        gt = torch.stack([ds[i][1] for i in range(len(ds))], 0).float()
        lab = torch.tensor([int(ds[i][2]) for i in range(len(ds))]).numpy()
        M = torch.load(os.path.join(HERE, args.map_dir, f'{cls}.map'),
                       map_location='cpu', weights_only=False).float()
        if M.shape[-1] != args.size:
            M = F.interpolate(M, size=(args.size, args.size), mode='bilinear',
                              align_corners=True)
        B1 = torch.load(os.path.join(HERE, args.bank448_dir, f'{cls}.map'),
                        map_location='cpu', weights_only=False).float()
        if B1.shape[-1] != args.size:
            B1 = F.interpolate(B1, size=(args.size, args.size), mode='bilinear',
                               align_corners=True)
        B2 = None
        if args.allow_bank672_legacy:
            B2 = torch.load(os.path.join(HERE, args.bank672_dir, f'{cls}.map'),
                            map_location='cpu', weights_only=False).float()
            B2 = F.interpolate(B2, size=(args.size, args.size), mode='bilinear',
                               align_corners=True)
        sc = os.path.join(HERE, args.map_dir, f'{cls}.json')
        if args.require_mapping_sidecar and not os.path.exists(sc):
            raise FileNotFoundError(
                f'缺少 mapping 样本顺序侧车，禁止融合: {sc}')
        if os.path.exists(sc):
            keys = json.load(open(sc)).get('keys')
            if keys and list(keys) != list(ds.keys):
                idx = torch.tensor([list(keys).index(x) for x in ds.keys])
                M = M[idx]
        bsc = os.path.join(HERE, args.bank448_dir, f'{cls}.json')
        if os.path.exists(bsc):
            bank_keys = json.load(open(bsc)).get('keys')
            if bank_keys and list(bank_keys) != list(ds.keys):
                idx = torch.tensor([list(bank_keys).index(x) for x in ds.keys])
                B1 = B1[idx]
        allowed = {x.strip() for x in args.only_rules.split(',') if x.strip()}
        gt_px = gt[:, 0].numpy() > 0.5
        rr = []
        if args.bank_only_search:
            gen = candidates_bank_only(M, B1)
        elif args.no_raw_bank_v3:
            gen = candidates_no_raw_bank_v3(M, B1)
        elif args.enhanced_raw:
            gen = candidates_enhanced_raw(M, B1, B2)
        elif args.stage_no_raw:
            gen = candidates_stage_no_raw(M, B1, B2)
        elif args.no_raw_bank_v2:
            gen = candidates_no_raw_bank_v2(M, B1)
        elif args.no_raw_bank:
            gen = candidates_no_raw_bank(M, B1)
        elif B2 is not None:
            gen = candidates(M, B1, B2)
        else:
            normalize = {
                'class': z,
                'image': z_per_image,
                'none': identity,
            }[args.z_mode]
            gen = candidates_two_branch(M, B1, normalize=normalize)
        for name, Fmap in gen:
            if allowed and name not in allowed:
                continue
            with torch.no_grad():
                am = gk(Fmap.to(args.device)).cpu()
            flat = am.flatten(1)
            kk = max(1, int(flat.shape[1] * args.image_top_ratio))
            sp = torch.sort(flat, 1, descending=True)[0][:, :kk].mean(1).numpy()
            metric = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab,
                                             use_metrics=use_metrics))
            rows.append([cls, name] + [float(x) for x in metric] + [float(np.mean(metric))])
            rr.append((float(np.mean(metric) * 100), name, metric))
        rr.sort(reverse=True)
        print(f'--- {cls} ---', flush=True)
        for val, name, metric in rr[:args.topn]:
            print(f'  {name:28s} ' + '  '.join(
                f'{n}={v * 100:.2f}' for n, v in zip(metric_names, metric)) +
                f'   均值={val:.3f}', flush=True)

    names = sorted({r[1] for r in rows})
    agg = []
    for name in names:
        rr = [r for r in rows if r[1] == name]
        agg.append((float(np.mean([r[-1] for r in rr])), name,
                    np.mean([r[2:-1] for r in rr], axis=0)))
    agg.sort(reverse=True)
    print('=== 融合候选宏平均 ===')
    for val, name, metric in agg[:args.topn]:
        print(f'  {name:28s} ' + '  '.join(
            f'{n}={v * 100:.2f}' for n, v in zip(metric_names, metric)) +
            f'   均值={val * 100:.3f}', flush=True)
    with open(args.out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'fusion'] + metric_names + ['mean'])
        w.writerows(rows)


if __name__ == '__main__':
    main()
