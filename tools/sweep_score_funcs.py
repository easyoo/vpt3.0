# -*- coding: utf-8 -*-
"""打分函数扫描：模型固定，每张测试图只前向一次，拿到 patch×原型 距离矩阵 d 后
同时评估多种**单图**聚合函数。

候选（d = 1 − cos，均为 [B,N,K]）：
  dmin                     到最近原型
  soft(τ)                  −τ·logΣexp(−d/τ)
  gmix(a,τ)                dmin^a · soft(τ)^(1−a)
  gmix3(a,b,τ)             dmin^a · soft(τ)^b · dmed^(1−a−b)   （dmed=到全部原型的中位数）
  topp(p)                  最近 p 个原型的平均距离
"""
import argparse
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


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_sf', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_sf'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', required=True)
    ap.add_argument('--save', default='m4m6r3_4shot_clpt3')
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--proto_set', default='unique')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    ap.add_argument('--dump_dists', default='',
                    help='可选：把 patch×prototype 余弦距离矩阵与对齐信息缓存到该目录，'
                         '后续可在不重复前向的情况下扫描更多打分函数')
    ap.add_argument('--dump_only', action='store_true',
                    help='只缓存距离矩阵，不评估内置候选（推荐给离线扫描使用）')
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.res_mapping = 0
    a.global_proto = 1
    a.metric_head = 1
    a.token_init_std = 0.02
    mod.args = a
    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8, num_register_tokens=4,
        interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = os.path.join(args.ckpt_dir or os.path.join('results', 'vpt4', args.save),
                      f'model_{args.cls}_35.pth')
    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['dino'], strict=False)
    model.to(args.device).eval()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls), transform=tf,
                          gt_transform=gtf, phase='test')
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    cands = {}
    def add(name, fn):
        cands.setdefault(name, []).append(fn)
    gts, labs, dist_store, proto_store = [], [], [], []
    per = {}
    dl = torch.utils.data.DataLoader(ds, batch_size=int(args.batch), shuffle=False,
                                     num_workers=4)
    with torch.no_grad():
        for img, gt, lab, _ in dl:
            x = img.to(args.device)
            if abs(args.scale - 1.0) > 1e-8:
                x = F.interpolate(x, scale_factor=args.scale, mode='bilinear', align_corners=False)
            ret = model(x, is_training=True)
            fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
            flr = torch.cat(ret['feat_list'], dim=1)
            _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
            flr = torch.cat([flr[:, :_g.shape[1]], _g], dim=1) if args.proto_set == 'unique' else torch.cat([flr, _g], 1)
            pl = F.normalize(model.token_mapping(fp), dim=-1)
            ql = F.normalize(model.token_mapping(flr), dim=-1)
            proto_store.append(ql[0].cpu().half())
            d = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)        # [B,N,K]
            dist_store.append(d.cpu().half())
            bs = d.shape[0]
            g = int(round(d.shape[1] ** 0.5))
            if not args.dump_only:
                df = d.reshape(-1, d.shape[-1])
                outs = {}
                outs['dmin'] = df.min(-1).values
                outs['dmed'] = df.median(-1).values
                for tau in (0.02, 0.03, 0.05, 0.08, 0.12):
                    outs[f'soft{tau}'] = -tau * torch.logsumexp(-df / tau, dim=-1)
                for aa in (0.3, 0.5, 0.7):
                    for tau in (0.03, 0.05, 0.08):
                        outs[f'gmix{aa}_{tau}'] = (outs['dmin'].clamp_min(1e-8) ** aa) * \
                                                  (outs[f'soft{tau}'].clamp_min(1e-8) ** (1 - aa))
                for p in (2, 3, 5, 10):
                    outs[f'topp{p}'] = torch.topk(
                        df, min(p, df.shape[-1]), dim=-1, largest=False).values.mean(-1)
                for k, v in outs.items():
                    per.setdefault(k, []).append(v.reshape(bs, 1, g, g))
            gts.append(gt)
            labs.extend(int(v) for v in lab)
    if args.dump_dists:
        os.makedirs(args.dump_dists, exist_ok=True)
        torch.save({
            'd': torch.cat(dist_store, 0),
            # 每类原型相同；保存一份，后续可计算原型间拓扑/密度自适应打分。
            'protos': torch.stack(proto_store, 0)[0],
            'gt': torch.cat(gts, 0).float(),
            'labels': torch.tensor(labs, dtype=torch.int64),
            'keys': list(ds.keys),
            'list_hash': ds.list_hash,
            'ckpt': ck,
            'size': int(args.size),
            'scale': float(args.scale),
            'proto_set': str(args.proto_set),
        }, os.path.join(args.dump_dists, f'{args.cls}.pt'))
        print(f'[dists] 保存 {args.dump_dists}/{args.cls}.pt '
              f'shape={tuple(torch.cat(dist_store, 0).shape)}')
    gt_t = torch.cat(gts, 0).float()
    gt_px = gt_t[:, 0].numpy() > 0.5
    lab = np.array(labs)
    print(f'--- {args.cls} ---')
    if args.dump_only:
        return
    rows = []
    for k, L in per.items():
        X = torch.cat(L, 0).float()
        with torch.no_grad():
            # 与 eval_score_v3 的正式路径严格一致：先在 patch 网格上形成候选图，
            # 双线性上采样到数据集分辨率，再执行同一个 Gaussian filter。
            # 漏掉上采样会让 [B,g,g] 与 [B,448,448] GT 的样本数不一致。
            X = F.interpolate(X.to(args.device), size=(args.size, args.size),
                              mode='bilinear', align_corners=True)
            am = gk(X).cpu()
        flat = am.flatten(1)
        kk = max(1, int(flat.shape[1] * 0.01))
        sp = torch.sort(flat, 1, descending=True)[0][:, :kk].mean(1).numpy()
        r = list(mod.ader_evaluator(am[:, 0].numpy(), sp, gt_px, lab))
        rows.append((np.mean(r) * 100, k, r))
    rows.sort(reverse=True)
    for m, k, r in rows[:8]:
        print(f'  {k:12s} ' + '  '.join(f'{n}={v*100:.2f}' for n, v in zip(NAMES, r)) + f'   均值={m:.3f}')


if __name__ == '__main__':
    main()
