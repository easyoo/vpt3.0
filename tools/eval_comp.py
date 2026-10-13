# -*- coding: utf-8 -*-
"""[打分搜索框架 v3 评测器] 从**分量网格**离线构造任意打分并评测。

   raw_score   = raw_local^a  * raw_global^(1-a)
   learn_score = learn_local^b * learn_global^(1-b)
   final       = w*learn_score + (1-w)*raw_score

用法：
  python eval_comp.py --grids .grids3_visa4 --stage 1            # 扫 raw 全局项与指数
  python eval_comp.py --grids .grids3_visa4 --stage 2 --best ...  # 扫 w 与 learn 侧
  CUDA_VISIBLE_DEVICES=x python eval_comp.py --grids .grids3_visa4 --official \
      --specs "d1:q512:0.5|dlmin:bq64:0.7|0.5"
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
    conv = torch.nn.Conv2d(1, 1, ks, groups=1, bias=False, padding=ks // 2)
    conv.weight.data = k
    for p in conv.parameters():
        p.requires_grad_(False)
    return conv


class CompEval:
    def __init__(self, npz):
        z = np.load(npz, allow_pickle=True)
        self.z = z
        self.gt = torch.tensor(z['gt'].astype(np.float32))
        self.label = z['label'].astype(np.uint8)
        self.size = int(z['gt'].shape[-1])
        self.n_views = 1 + max(int(k.split('__v')[1]) for k in z.files if '__v' in k)
        self._gk = gk5()
        self._c = {}

    def comp(self, br, op, vi):
        if '/' in op:                      # 跨分支引用：'raw/q64' 表示取 raw 分支的 q64 分量
            br, op = op.split('/', 1)
        key = (br, op, vi)
        if key not in self._c:
            g = self.z[f'{br}_{op}__v{vi}']
            t = torch.tensor(g.astype(np.float32)).unsqueeze(1)
            t = F.interpolate(t, size=(self.size, self.size), mode='bilinear', align_corners=True)
            if vi == 1:
                t = t.flip(3)
            self._c[key] = t
        return self._c[key]

    @staticmethod
    def _pow(local, glob, a):
        e = 1e-6
        return local.clamp_min(e).pow(a) * glob.clamp_min(e).pow(1.0 - a)

    def build(self, spec):
        (rl, rg, a), (ll, lg, b), w = spec
        tot = None
        for vi in range(self.n_views):
            rs = self._pow(self.comp('raw', rl, vi), self.comp('raw', rg, vi), a)
            ls = self._pow(self.comp('learn', ll, vi), self.comp('learn', lg, vi), b)
            am = w * ls + (1.0 - w) * rs
            tot = am if tot is None else tot + am
        return tot / self.n_views

    def metrics(self, spec, stride=1, official=False):
        am = self._gk(self.build(spec))
        flat = am.flatten(1)
        kk = max(1, int(flat.shape[1] * 0.01))
        sp = torch.sort(flat, dim=1, descending=True)[0][:, :kk].mean(dim=1)
        gt = self.gt.numpy().astype(np.uint8)
        a = am[:, 0].numpy()
        if stride > 1:
            a = a[:, ::stride, ::stride]
            gt = gt[:, ::stride, ::stride]
        if official:
            import importlib.util
            p = os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py')
            sys.path.insert(0, os.path.dirname(p))
            s = importlib.util.spec_from_file_location('audit_main2', p)
            m = importlib.util.module_from_spec(s)
            sys.modules['audit_main2'] = m
            s.loader.exec_module(m)
            return list(m.ader_evaluator(am[:, 0].numpy(), sp.flatten().numpy(), gt,
                                         self.label))
        return cpu_metrics(a, sp.flatten().numpy(), gt, self.label)


def parse_spec(txt):
    r, l, w = txt.split('|')
    rl, rg, a = r.split(':')
    ll, lg, b = l.split(':')
    return (rl, rg, float(a)), (ll, lg, float(b)), float(w)


def sweep(files, specs, stride):
    acc = {s: [] for s in specs}
    for f in files:
        ge = CompEval(f)
        for s in specs:
            r = ge.metrics(s, stride=stride)
            v = [x * 100 for x in r]
            fin = [u for u in v if not np.isnan(u)]
            acc[s].append(sum(fin) / len(fin))
        del ge
    return {s: float(np.mean(acc[s])) for s in specs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', default='.grids3_visa4')
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--specs', default='')
    ap.add_argument('--official', type=int, default=0)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--learn_spec', default='dlmin:bq64:0.7',
                    help='阶段1 固定的 learn 侧写法')
    ap.add_argument('--save', default='')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.grids, '*.npz')))
    print(f"# {os.path.basename(args.grids)}  {len(files)} 类", flush=True)

    if args.specs:
        specs = [parse_spec(t) for t in args.specs.split(',')]
        if args.official:
            rows = {s: [] for s in specs}
            for f in files:
                ge = CompEval(f)
                for s in specs:
                    rows[s].append([x * 100 for x in ge.metrics(s, stride=1, official=True)])
                del ge
            avg = {s: [float(np.mean([r[i] for r in rows[s]])) for i in range(7)] for s in specs}
            ref = avg[specs[0]]
            print(f"{'raw':<20}{'learn':<22}{'w':<5}" + "".join(f"{m:>9}" for m in NAMES)
                  + f"{'7avg':>9}{'Δ':>8}")
            lines = []
            for s in specs:
                a = avg[s]
                line = (f"{s[0][0]+'^'+str(s[0][2])+'*'+s[0][1]:<20}"
                        f"{s[1][0]+'^'+str(s[1][2])+'*'+s[1][1]:<22}{s[2]:<5.1f}"
                        + "".join(f"{v:>9.2f}" for v in a)
                        + f"{sum(a)/7:>9.2f}{sum(a)/7-sum(ref)/7:>+8.2f}")
                print(line)
                lines.append(line)
            if args.save:
                with open(args.save, 'w') as fh:
                    fh.write(f"# {args.grids} official adeval\n")
                    fh.write("\n".join(lines) + "\n")
            return
        res = sweep(files, specs, args.stride)
        for s, v in sorted(res.items(), key=lambda kv: -kv[1]):
            print(f"{s[0][0]+'^'+str(s[0][2])+'*'+s[0][1]:<28}"
                  f"{s[1][0]+'^'+str(s[1][2])+'*'+s[1][1]:<30}w={s[2]:<4} {v:.2f}")
        return

    if args.stage == 1:
        ls = args.learn_spec.split(':')
        lspec = (ls[0], ls[1], float(ls[2]))
        specs = []
        for rg in ['q64', 'q256', 'q512', 'q1024']:
            for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                specs.append((('d1', rg, a), lspec, 0.5))
        res = sweep(files, specs, args.stride)
        print(f"{'raw':<22}{'学习侧':<22}{'w':<5}{'6指标均值':>10}")
        for s, v in sorted(res.items(), key=lambda kv: -kv[1])[:args.topk]:
            print(f"{'d1^'+str(s[0][2])+'*'+s[0][1]:<22}"
                  f"{s[1][0]+'^'+str(s[1][2])+'*'+s[1][1]:<22}{s[2]:<5.1f}{v:>10.2f}")
        if args.save:
            with open(args.save, 'w') as fh:
                for s, v in sorted(res.items(), key=lambda kv: -kv[1]):
                    fh.write(f"{s}\t{v:.4f}\n")
    else:
        # 阶段2：给定 top raw 设定，扫 w 与 learn 侧
        raw_set = ['d1:q512:0.3', 'd1:q512:0.4', 'd1:q512:0.5', 'd1:q256:0.4']
        learn_set = ['dlmin:bq64:0.7', 'dlmin:bq256:0.7', 'dlmin:bq64:0.5', 'min:dlmed:0.5']
        specs = []
        for r in raw_set:
            rp = r.split(':')
            for l in learn_set:
                lp = l.split(':')
                for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    specs.append(((rp[0], rp[1], float(rp[2])),
                                  (lp[0], lp[1], float(lp[2])), w))
        res = sweep(files, specs, args.stride)
        print(f"{'raw':<18}{'learn':<22}{'w':<5}{'6指标均值':>10}")
        for s, v in sorted(res.items(), key=lambda kv: -kv[1])[:args.topk]:
            print(f"{s[0][0]+'^'+str(s[0][2])+'*'+s[0][1]:<18}"
                  f"{s[1][0]+'^'+str(s[1][2])+'*'+s[1][1]:<22}{s[2]:<5.1f}{v:>10.2f}")
        if args.save:
            with open(args.save, 'w') as fh:
                for s, v in sorted(res.items(), key=lambda kv: -kv[1]):
                    fh.write(f"{s}\t{v:.4f}\n")


if __name__ == '__main__':
    main()
