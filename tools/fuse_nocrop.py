# -*- coding: utf-8 -*-
"""[vpt3.7 · 融合评估] 训练裁剪(448→392) / 测试不裁剪(Resize N)：

  mapping 分支：eval_score_v3.py 在**不裁剪 N** 下重推 → <map_dir>/<cls>.map（v3.7: .maps_simple_nocrop448_*）
  bank  分支：cache_bank_nocrop.py 在**同一几何**下重推 → <bank_dir>/<cls>.map（v3.7: .errmap_nocrop448*）
  融合      ：类内 z 标准化后 **0.5·z(mapping) + 0.5·z(bank)**，高斯平滑(k=5,σ=4) → 官方 adeval

  [FIX-1] 默认 --w 由 0.3 改为 0.5，默认目录更新为 v3.7 口径（旧默认对应 v3.0/v3.5 的 0.3/0.7）。
  [FIX-4] 若 <cls>.json 侧车存在（记录生成异常图时的样本键顺序），按其键顺序对齐 GT/标签；
          没有侧车（历史图）时退回"未排序"的历史顺序，保证旧结果可复现。

同时给出两支单独 + 融合三组指标，便于看互补性。
用法：python fuse_nocrop.py --class transistor --size 448 --w 0.5
"""
import argparse
import importlib.util
import json
import os
import sys

import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def z_cls(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--w', type=float, default=0.5,
                    help='mapping 权重（v3.7 默认 0.5 → 0.5·z(M)+0.5·z(B)；旧版曾用 0.3）')
    ap.add_argument('--map_dir', default='.maps_simple_nocrop448_mv2')
    ap.add_argument('--bank_dir', default='.errmap_nocrop448_2shot')
    ap.add_argument('--align_by_sidecar', type=int, default=1,
                    help='[FIX-4] 1=有 <cls>.json 侧车时按其样本键对齐 GT/标签（推荐）；0=强制按当前数据集顺序')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)

    mp = os.path.join(HERE, args.map_dir, f'{args.cls}.map')
    bp = os.path.join(HERE, args.bank_dir, f'{args.cls}.map')
    M = torch.load(mp, weights_only=False).float()
    B = torch.load(bp, weights_only=False).float()
    assert M.shape == B.shape, (args.cls, M.shape, B.shape)

    # [FIX-4] 优先按侧车（生成异常图时记录的样本顺序）对齐；历史图无侧车则退回未排序顺序
    def _sidecar(d):
        p = os.path.join(HERE, d, f'{args.cls}.json')
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception as e:
                print(f'  [warn] 侧车解析失败 {p}: {e}')
        return None

    sc_m, sc_b = _sidecar(args.map_dir), _sidecar(args.bank_dir)
    if not int(args.align_by_sidecar):        # 显式关闭对齐：沿用历史行为
        sc_m = sc_b = None
    both_sc = bool(sc_m and sc_b and sc_m.get('keys') and sc_b.get('keys'))
    if both_sc and sc_m['keys'] != sc_b['keys']:
        raise RuntimeError('[align] mapping 与 bank 的侧车键顺序不一致，拒绝融合（避免静默错位）')

    # 公共坐标系：两张图都有侧车 → 用"排序序"（两侧一致）；否则退回"历史未排序序"，
    # 并把带侧车的那张图**重排**到该坐标系（混合场景：新 mapping + 旧 bank 等）
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    def _mk_ds(sort_items):
        d = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                             transform=tf, gt_transform=gtf, phase='test',
                             sort_items=sort_items)
        g = torch.stack([d[i][1] for i in range(len(d))], 0).float()
        l = torch.tensor([int(d[i][2]) for i in range(len(d))])
        return d, g, l

    if both_sc:
        ds, gt, lab = _mk_ds(True)                     # 排序序 == 侧车键序
        keys = list(sc_m['keys'])
        if ds.keys != keys:
            raise RuntimeError('[align] 数据集键序与侧车不一致（数据集可能已变动）')
        print(f'  [align] 新旧图均带侧车 → 按侧车键序对齐 {len(keys)} 个样本 hash={sc_m.get("list_hash")}')
    else:
        ds, gt, lab = _mk_ds(False)                    # 历史未排序序（与无侧车图一致）
        frame = list(ds.keys)
        def _reorder(tensor, sc, tag):
            if not (sc and sc.get('keys')):
                return tensor
            k = list(sc['keys'])
            if k == frame:
                return tensor
            if set(k) != set(frame):
                raise RuntimeError(f'[align] {tag} 侧车键集合与数据集不一致')
            idx = torch.tensor([k.index(x) for x in frame])
            print(f'  [align] {tag} 图带侧车 → 重排到历史坐标系（{len(idx)} 个样本）')
            return tensor[idx]
        M = _reorder(M, sc_m, 'mapping')
        B = _reorder(B, sc_b, 'bank')
        print(f'  [align] 公共坐标系 = 历史未排序序（{len(frame)} 个样本）；'
              f'mapping侧车={bool(sc_m)} bank侧车={bool(sc_b)}')

    assert gt.shape[0] == M.shape[0], (args.cls, gt.shape, M.shape)

    F = args.w * z_cls(M) + (1 - args.w) * z_cls(B)
    out = {}
    for name, X in (('mapping', M), ('bank', B), ('fused', F)):
        with torch.no_grad():
            am = gk(X.to(args.device)).cpu()
        pr_px = am[:, 0].numpy()
        gt_px = (gt[:, 0].numpy() > 0.5)
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        pr_sp = torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
        r = list(mod.ader_evaluator(pr_px, pr_sp, gt_px, lab.numpy()))
        out[name] = r
        print(f'  [{args.cls}] {name:<8}' + '  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES, r)))
    # 便于脚本聚合
    for name in ('mapping', 'bank', 'fused'):
        print(f'CSV,{args.cls},{name},' + ','.join(f'{v:.4f}' for v in out[name]))


if __name__ == '__main__':
    main()
