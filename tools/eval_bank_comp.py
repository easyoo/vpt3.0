# -*- coding: utf-8 -*-
"""[bank 分支] 离线枚举打分 d1^a * q_p^(1-a)，两 stage 的组合方式可选。

  stage 分： s_s = d1_s^a * q_p_s^(1-a)
  组合：     score 级：s = (s_1 + s_2)/2      （各 stage 先打分再平均）
             comp  级：d1=(d1_1+d1_2)/2, q=(q_1+q_2)/2 后再打分
评测：32x32 网格 -> 插值 448（与官方 bank 口径一致）-> 高斯平滑 -> top-1% -> 指标
"""
import argparse
import glob
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


def gk5():
    ks, sigma = 5, 4
    x = torch.arange(ks)
    xg = x.repeat(ks).view(ks, ks)
    xy = torch.stack([xg, xg.t()], dim=-1).float()
    mean = (ks - 1) / 2.
    var = sigma ** 2
    k = (1. / (2. * math.pi * var)) * torch.exp(-((xy - mean) ** 2).sum(-1) / (2 * var))
    k = (k / k.sum()).view(1, 1, ks, ks)
    conv = torch.nn.Conv2d(1, 1, ks, bias=False, padding=ks // 2)
    conv.weight.data = k
    for p in conv.parameters():
        p.requires_grad_(False)
    return conv


class BankEval:
    def __init__(self, npz):
        z = np.load(npz, allow_pickle=True)
        self.z = z
        self.gt = torch.tensor(z['gt'].astype(np.float32))
        self.label = z['label'].astype(np.uint8)
        self.size = int(z['gt'].shape[-1])
        self._gk = gk5()
        self._c = {}

    def comp(self, stage, name):
        key = (stage, name)
        if key not in self._c:
            a = torch.tensor(self.z[f's{stage}_{name}'].astype(np.float32)).unsqueeze(1)
            self._c[key] = F.interpolate(a, size=(self.size, self.size), mode='bilinear',
                                         align_corners=True)
        return self._c[key]

    def build(self, d1name, qname, a_par, combine):
        e = 1e-6
        if combine == 'score':
            outs = []
            for s in (1, 2):
                d1 = self.comp(s, d1name).clamp_min(e)
                q = self.comp(s, qname).clamp_min(e)
                outs.append(d1.pow(a_par) * q.pow(1 - a_par))
            return (outs[0] + outs[1]) / 2
        d1 = ((self.comp(1, d1name) + self.comp(2, d1name)) / 2).clamp_min(e)
        q = ((self.comp(1, qname) + self.comp(2, qname)) / 2).clamp_min(e)
        return d1.pow(a_par) * q.pow(1 - a_par)

    def metrics(self, spec, stride=1, official=False):
        d1name, qname, a_par, combine = spec
        am = self._gk(self.build(d1name, qname, a_par, combine))
        flat = am.flatten(1)
        kk = max(1, int(flat.shape[1] * 0.01))
        sp = torch.sort(flat, dim=1, descending=True)[0][:, :kk].mean(dim=1)
        gt = self.gt.numpy().astype(np.uint8)
        aa = am[:, 0].numpy()
        if stride > 1:
            aa = aa[:, ::stride, ::stride]
            gt = gt[:, ::stride, ::stride]
        if official:
            import importlib.util
            p = os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')
            sys.path.insert(0, os.path.dirname(p))
            s = importlib.util.spec_from_file_location('audit_bk', p)
            m = importlib.util.module_from_spec(s)
            sys.modules['audit_bk'] = m
            s.loader.exec_module(m)
            return list(m.ader_evaluator(am[:, 0].numpy(), sp.flatten().numpy(), gt, self.label))
        return cpu_metrics(aa, sp.flatten().numpy(), gt, self.label)

    def save_map(self, spec, path):
        d1name, qname, a_par, combine = spec
        am = self.build(d1name, qname, a_par, combine)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(am.cpu(), path)
        return path


def parse(txt):
    p = txt.split(':')
    return (p[0], p[1], float(p[2]), p[3] if len(p) > 3 else 'score')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='')
    ap.add_argument('--grids', default='')
    ap.add_argument('--combo', default='')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--official', type=int, default=0)
    ap.add_argument('--stride', type=int, default=3)
    args = ap.parse_args()

    if args.npz and args.combo:
        ge = BankEval(args.npz)
        r = ge.metrics(parse(args.combo), stride=1, official=bool(args.official))
        print(f"{args.npz}  {args.combo}")
        print("  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, r)))
        return

    files = sorted(glob.glob(os.path.join(args.grids, '*.npz')))
    specs = []
    for q in ('q64', 'q256', 'q512', 'q1024'):
        for a_par in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            specs.append(('d1', q, a_par, 'score'))
    specs.append(('d1', 'mean', 0.5, 'score'))
    specs.append(('d1', 'q512', 0.5, 'comp'))
    specs.append(('mean', 'mean', 1.0, 'score'))
    print(f"# {os.path.basename(args.grids)} {len(files)} 类 bank 打分散列", flush=True)
    acc = {s: [] for s in specs}
    for f in files:
        ge = BankEval(f)
        for s in specs:
            r = ge.metrics(s, stride=args.stride)
            v = [x * 100 for x in r]
            fin = [u for u in v if not np.isnan(u)]
            acc[s].append(sum(fin) / len(fin))
        del ge
    for s, v in sorted(acc.items(), key=lambda kv: -np.mean(kv[1]))[:20]:
        print(f"{s[0]:<6}{s[1]:<7}a={s[2]:<4}{s[3]:<6}{np.mean(v):>8.2f}")


if __name__ == '__main__':
    main()
