# -*- coding: utf-8 -*-
"""[打分搜索框架 v3] 只落盘**分量**（不再落组合），从而支持离线任意 (局部算子, 全局算子, 指数, 融合权重)。

分量定义（都在 patch 网格上）：
  raw 侧：d1（最近邻）、q8/q64/q256/q512/q1024（距离分布的 1024-子采样分位数，q512=中位数）
  learn侧：dlmin（到 64 个可学习原型的最小距离）、dlmed（其分布中位数）、
          bq8/bq64/bq256/bq512/bq1024（到"映射后原始库"距离分布的分位数）、bmin

任意打分 = (local)^a * (global)^(1-a)，例如 VisA 上最优的 d1^0.3·q512^0.7。
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
RAW_COMPS = ['d1', 'q8', 'q64', 'q256', 'q512', 'q1024']
LEARN_COMPS = ['dlmin', 'dlmed', 'bmin', 'bq8', 'bq64', 'bq256', 'bq512', 'bq1024']


def load_mod(path):
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='vpt35_visa4shot')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/experiment/visual2/visa_prepared')
    ap.add_argument('--out_dir', default='.grids3_visa4')
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
    print(f'[sub] 全局分位数用 {sub_idx.numel()}/{M} 个子采样估计')

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
            for b, ops in (('raw', RAW_COMPS), ('learn', LEARN_COMPS)) for o in ops]
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
                dsub = torch.sort(d.index_select(-1, sub_idx), dim=-1).values
                raw = {'d1': d.min(-1).values}
                for pp in (8, 64, 256, 512, 1024):
                    raw[f'q{pp}'] = dsub[..., min(pp, dsub.shape[-1]) - 1]

                pl = F.normalize(model.token_mapping(fp), dim=-1)
                ql = F.normalize(model.token_mapping(flr), dim=-1)
                dl_ = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                db = (1.0 - torch.bmm(pl, q_bank_l.t().unsqueeze(0)
                                      .expand(pl.shape[0], -1, -1))).clamp_min(0)
                dbsub = torch.sort(db.index_select(-1, sub_idx), dim=-1).values
                learn = {'dlmin': dl_.min(-1).values, 'dlmed': dl_.median(-1).values,
                         'bmin': db.min(-1).values}
                for pp in (8, 64, 256, 512, 1024):
                    learn[f'bq{pp}'] = dbsub[..., min(pp, dbsub.shape[-1]) - 1]

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
                                           'raw_comps': RAW_COMPS, 'learn_comps': LEARN_COMPS}))
    p = os.path.join(args.out_dir, f'{args.cls}.npz')
    np.savez_compressed(p, **payload)
    print(f'[grids3] 保存 {p}  keys={len(payload)}')


if __name__ == '__main__':
    main()
