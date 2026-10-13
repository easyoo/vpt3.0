# -*- coding: utf-8 -*-
"""[打分搜索框架 · 第2步] 离线枚举打分组合（读 probe_score_grid.py 落的网格图）。

组合语义：`fused = w·learn_op + (1−w)·raw_op`，逐视图插值到 448 →（翻转）→ 平均 →
高斯平滑(k=5,σ=4) → top-1% 图像分数 → 指标（与官方 adeval 口径一致）。

用法：
  # 快速筛选（CPU 指标，AUPRO 置 nan，秒级）
  python eval_grid.py --npz .grids_visa4/candle.npz --sweep
  # 指定组合
  python eval_grid.py --npz .grids_visa4/candle.npz --combo learn=bgeom,raw=geom,w=0.5
"""
import argparse
import importlib.util
import itertools
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_identity import cpu_metrics         # noqa: E402
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]


def load_mod(path):
    # 代码模块内部 `from dinov2.models import ...` 需要自身目录在 sys.path 上
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def gaussian_kernel_5x5_sigma4():
    """复刻 code 里的 get_gaussian_kernel(5, 4)：5×5 深度可分离卷积，padding=2。"""
    ks, sigma, ch = 5, 4, 1
    x = torch.arange(ks)
    xg = x.repeat(ks).view(ks, ks)
    xy = torch.stack([xg, xg.t()], dim=-1).float()
    mean = (ks - 1) / 2.
    var = sigma ** 2.
    k = (1. / (2. * math.pi * var)) * torch.exp(-((xy - mean) ** 2).sum(-1) / (2 * var))
    k = (k / k.sum()).view(1, 1, ks, ks).repeat(ch, 1, 1, 1)
    conv = torch.nn.Conv2d(ch, ch, ks, groups=ch, bias=False, padding=ks // 2)
    conv.weight.data = k
    conv.weight.requires_grad_(False)
    return conv


class GridEval:
    def __init__(self, npz, script=None, device='cpu'):
        z = np.load(npz, allow_pickle=True)
        self.z = z
        self.keys = [k for k in z.files if '__v' in k]
        self.gt = torch.tensor(z['gt'].astype(np.float32))
        self.label = z['label'].astype(np.uint8)
        self.size = int(z['gt'].shape[-1])
        self.n_views = 1 + max(int(k.split('__v')[1]) for k in self.keys)
        self.script = script or os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')
        self._gk = None
        self._cache = {}
        self._mod = None
        self.device = device

    def _kernel(self):
        if self._gk is None:
            self._gk = gaussian_kernel_5x5_sigma4()
        return self._gk

    def view_map(self, branch, op, vi):
        """插值到 448 的单视图图 [N,1,S,S]，带缓存。"""
        key = (branch, op, vi)
        if key in self._cache:
            return self._cache[key]
        g = self.z[f'{branch}_{op}__v{vi}']                   # [N,g,g] float16
        t = torch.tensor(g.astype(np.float32)).unsqueeze(1)
        t = F.interpolate(t, size=(self.size, self.size), mode='bilinear', align_corners=True)
        if vi == 1:
            t = t.flip(3)
        self._cache[key] = t
        return t

    def fused(self, learn_op, raw_op, w):
        tot = None
        for vi in range(self.n_views):
            ml = self.view_map('learn', learn_op, vi)
            mr = self.view_map('raw', raw_op, vi)
            am = w * ml + (1.0 - w) * mr
            tot = am if tot is None else tot + am
        return tot / self.n_views

    def score(self, am):
        gk = self._kernel()
        am = gk(am)
        flat = am.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        return am, torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

    def metrics(self, learn_op, raw_op, w, official=False, stride=1):
        am, sp = self.score(self.fused(learn_op, raw_op, w))
        gt_px = self.gt.numpy().astype(np.uint8)          # [N, S, S]
        if stride > 1:                                    # 仅用于快速筛选：像素下采样
            am = am[:, :, ::stride, ::stride]
            gt_px = gt_px[:, ::stride, ::stride]
        if official:
            if self._mod is None:
                self._mod = load_mod(self.script)
            r = list(self._mod.ader_evaluator(am[:, 0].numpy(), sp.flatten().numpy(),
                                              gt_px, self.label))
        else:
            r = cpu_metrics(am[:, 0].numpy(), sp.flatten().numpy(), gt_px, self.label)
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--combo', default='', help='learn=<op>,raw=<op>,w=<float>')
    ap.add_argument('--sweep', action='store_true', help='枚举内置组合网格')
    ap.add_argument('--official', type=int, default=0, help='1=用官方 adeval（需 CUDA）')
    ap.add_argument('--learn_ops', default='min,bgeom')
    ap.add_argument('--raw_ops', default='min,geom,g07,gn5')
    ap.add_argument('--ws', default='0.3,0.5,0.7')
    args = ap.parse_args()

    ge = GridEval(args.npz, device='cuda:0' if args.official else 'cpu')
    print(f"# {os.path.basename(args.npz)}  keys={len(ge.keys)}  N={ge.gt.shape[0]}  "
          f"views={ge.n_views}")
    print(f"{'learn':<8}{'raw':<8}{'w':<6}" + "".join(f"{m:>9}" for m in NAMES) + f"{'7avg':>9}")

    combos = []
    if args.combo:
        d = dict(kv.split('=') for kv in args.combo.split(','))
        combos.append((d['learn'], d['raw'], float(d.get('w', 0.5))))
    if args.sweep:
        for lo, ro, w in itertools.product(args.learn_ops.split(','), args.raw_ops.split(','),
                                           [float(x) for x in args.ws.split(',')]):
            combos.append((lo, ro, w))

    for lo, ro, w in combos:
        r = ge.metrics(lo, ro, w, official=bool(args.official))
        vals = [x * 100 for x in r]
        fin = [v for v in vals if not np.isnan(v)]
        print(f"{lo:<8}{ro:<8}{w:<6.1f}" + "".join(
            ("      nan" if np.isnan(v) else f"{v:>9.2f}") for v in vals)
            + f"{sum(fin)/len(fin):>9.2f}")


if __name__ == '__main__':
    main()
