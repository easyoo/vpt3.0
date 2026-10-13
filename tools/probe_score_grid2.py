# -*- coding: utf-8 -*-
"""[打分搜索框架 v2] 打分统一成 d1^a x q_p^(1-a) 并系统扫描（一次推理全落盘）。

依据阶段 A 的发现（logs/sweep_visa4.txt）：
  * 全局项用中位数（gmed 75.33）明显优于用均值（geom 74.00），原始 min 为 71.53；
  * 指数上让全局项权重更高更好（a=0.3 优于 a=0.5 优于 a=0.7）。
因此把打分族统一写成"局部项 x 全局项"，全局项 = 到正常 patch 库距离分布的**分位数**：
    score = d1^a * q_p^(1-a)    q_p = 距离分布的 p/1024 分位（在 1024 个子采样 patch 上估计）
p 越小越接近 min（局部），p 越大越接近中位数（全局）；p=1024 即中位数。

落盘：<branch>_<op>__v<view> float16 [N,g,g]；gt/label/meta 同 v1。
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash         # noqa: E402

SUBN = 1024
PS = (8, 64, 256, 512, 1024)
ALPHAS = (0.3, 0.5, 0.7)
RAW_OPS = (['min', 'kn5', 'kn8'] + [f'g{p}_{a}' for p in PS for a in ALPHAS])
LEARN_OPS = (['min', 'geom64', 'bmin'] + [f'b{p}_{a}' for p in (64, 1024) for a in ALPHAS])


def load_mod(path):
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def _pow(d1, gq, a):
    e = 1e-6
    return d1.clamp_min(e).pow(a) * gq.clamp_min(e).pow(1.0 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='vpt35_visa4shot')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/experiment/visual2/visa_prepared')
    ap.add_argument('--out_dir', default='.grids2_visa4')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--scales', default='1.5')
    ap.add_argument('--flip', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.softmin = 0
    a.learnable_num_per_layer = args.num_tokens
    a.res_mapping = 0
    a.global_proto = 1
    a.metric_head = 1
    a.subspace = 0
    a.layer_scale_learn = 0
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    a.ms_bank_proto = 0
    mod.args = a
    mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ckpt = os.path.join(args.ckpt_dir or os.path.join('results', 'vpt4', args.save),
                        f'model_{args.cls}_35.pth')
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['dino'],
                          strict=False)
    model.to(args.device).eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)
    M = bank.shape[0]
    q_raw = F.normalize(bank, dim=-1)
    with torch.no_grad():
        q_bank_l = F.normalize(model.token_mapping(bank), dim=-1)
    gen = torch.Generator().manual_seed(1234)
    sub_idx = torch.randperm(M, generator=gen)[:min(SUBN, M)].to(args.device)
    print(f'[sub] 全局分位数用 {sub_idx.numel()}/{M} 个 patch 子采样估计')

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    if args.limit:
        ds.img_paths = ds.img_paths[:args.limit]
        ds.gt_paths = ds.gt_paths[:args.limit]
        ds.labels = ds.labels[:args.limit]
        ds.types = ds.types[:args.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    scales = tuple(float(x) for x in args.scales.split(','))

    keys = [f'{b}_{o}__v{vi}' for vi in range(2 if args.flip else 1)
            for b, ops in (('raw', RAW_OPS), ('learn', LEARN_OPS)) for o in ops]
    store = {k: [] for k in keys}
    gt_all, label_all = [], []

    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(args.device)
            views = []
            for s in scales:
                v_img = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear',
                                                           align_corners=False)
                views.extend([v_img, v_img.flip(3)] if args.flip else [v_img])
            for vi, view in enumerate(views):
                ret = model(view, is_training=True)
                fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
                _gg = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                flr = torch.cat([flr, _gg], dim=1)
                n_el = fp.shape[1]
                gg = int(round(n_el ** 0.5))

                p = F.normalize(fp, dim=-1)
                d = (1.0 - torch.bmm(p, q_raw.t().unsqueeze(0)
                                     .expand(p.shape[0], -1, -1))).clamp_min(0)
                top8 = torch.topk(d, 8, dim=-1, largest=False).values
                dsub = torch.sort(d.index_select(-1, sub_idx), dim=-1).values
                raw = {'min': top8[..., 0], 'kn5': top8[..., :5].mean(-1), 'kn8': top8.mean(-1)}
                for pp in PS:
                    qv = dsub[..., min(pp, dsub.shape[-1]) - 1]
                    for al in ALPHAS:
                        raw[f'g{pp}_{al}'] = _pow(top8[..., 0], qv, al)

                pl = F.normalize(model.token_mapping(fp), dim=-1)
                ql = F.normalize(model.token_mapping(flr), dim=-1)
                dl = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                db = (1.0 - torch.bmm(pl, q_bank_l.t().unsqueeze(0)
                                      .expand(pl.shape[0], -1, -1))).clamp_min(0)
                db8 = torch.topk(db, 8, dim=-1, largest=False).values
                dbsub = torch.sort(db.index_select(-1, sub_idx), dim=-1).values
                dl1 = dl.min(-1).values
                learn = {'min': dl1, 'geom64': _pow(dl1, dl.mean(-1), 0.5), 'bmin': db8[..., 0]}
                for pp in (64, 1024):
                    qv = dbsub[..., min(pp, dbsub.shape[-1]) - 1]
                    for al in ALPHAS:
                        learn[f'b{pp}_{al}'] = _pow(dl1, qv, al)

                for br, mp in (('raw', raw), ('learn', learn)):
                    for o, s_t in mp.items():
                        store[f'{br}_{o}__v{vi}'].append(s_t.reshape(-1, gg, gg).half().cpu())
            gt_px = (gt > 0.5).float()
            if gt_px.shape[1] > 1:
                gt_px = torch.max(gt_px, dim=1, keepdim=True)[0]
            gt_all.append((gt_px[:, 0] > 0.5).to(torch.uint8).cpu())
            label_all.append(label)

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {k: torch.cat(v, 0).numpy() for k, v in store.items()}
    payload['gt'] = torch.cat(gt_all, 0).numpy()
    payload['label'] = torch.cat(label_all).flatten().numpy().astype(np.uint8)
    payload['meta'] = np.array(json.dumps({'save': args.save, 'cls': args.cls,
                                           'shots': args.shots, 'size': args.size,
                                           'scales': args.scales, 'flip': args.flip,
                                           'raw_ops': RAW_OPS, 'learn_ops': LEARN_OPS}))
    p = os.path.join(args.out_dir, f'{args.cls}.npz')
    np.savez_compressed(p, **payload)
    print(f'[grids2] 保存 {p}  keys={len(payload)}')


if __name__ == '__main__':
    main()
