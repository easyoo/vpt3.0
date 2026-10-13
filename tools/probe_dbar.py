# -*- coding: utf-8 -*-
"""[诊断] 三处打分点的 `d̄ / d_min` 分布 —— 判断 geom 在该处是否有信息量。

原理：`s = √(d_min · d̄)`。若 `d̄ ≈ d_min`（比例 →1），则该处的参考集"高度冗余"，
geom 退化回 min；若比例远大于 1，则全局项带来真实增量。

用法: python probe_dbar.py --save vpt35_visa4shot --class candle --shots 4 --size 448 \
        --data_path /home/jjquan/experiment/visual2/visa_prepared \
        --ckpt_dir results/visa/vpt35_visa4shot
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots         # noqa: E402


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
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--limit', type=int, default=8, help='只跑前 N 个 batch（默认 8，够统计）')
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
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)
    q_raw = F.normalize(bank, dim=-1)

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    stats = {'learn': [], 'raw': []}
    with torch.no_grad():
        for bi, (img, gt, label, _) in enumerate(dl):
            if bi >= args.limit:
                break
            img = img.to(args.device)
            ret = model(img, is_training=True)
            fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
            flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
            _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
            flr = torch.cat([flr, _g], dim=1)

            p_l = F.normalize(model.token_mapping(fp), dim=-1)
            q_l = F.normalize(model.token_mapping(flr), dim=-1)
            d_l = (1.0 - torch.bmm(p_l, q_l.transpose(1, 2))).clamp_min(0)
            p_r = F.normalize(fp, dim=-1)
            d_r = (1.0 - torch.bmm(p_r, q_raw.t().unsqueeze(0).expand(p_r.shape[0], -1, -1))).clamp_min(0)
            for key, d in (('learn', d_l), ('raw', d_r)):
                dmin = d.min(dim=-1).values.flatten()
                dbar = d.mean(dim=-1).flatten()
                stats[key].append(torch.stack([dmin, dbar], -1).cpu().numpy())

    print(f"=== {args.cls}（shot={args.shots}, save={args.save}, 前 {args.limit} 个 batch）===")
    for key in ('learn', 'raw'):
        v = np.concatenate(stats[key], 0)
        dmin, dbar = v[:, 0], v[:, 1]
        ratio = dbar / np.clip(dmin, 1e-8, None)
        M = 64 if key == 'learn' else bank.shape[0]
        print(f"\n[{key}] 参考集 M={M}  (可学习原型 64 / 原始库 K×784={bank.shape[0]})")
        print(f"  d_min : p10={np.percentile(dmin,10):.4f} p50={np.percentile(dmin,50):.4f} "
              f"p90={np.percentile(dmin,90):.4f}  mean={dmin.mean():.4f}")
        print(f"  d_bar : p10={np.percentile(dbar,10):.4f} p50={np.percentile(dbar,50):.4f} "
              f"p90={np.percentile(dbar,90):.4f}  mean={dbar.mean():.4f}")
        print(f"  d_bar/d_min : p10={np.percentile(ratio,10):.2f} p50={np.percentile(ratio,50):.2f} "
              f"p90={np.percentile(ratio,90):.2f}  mean={ratio.mean():.2f}")
        print(f"  → geom 相对 min 的平均抬升倍数 √(ratio) = {np.sqrt(ratio).mean():.2f}")
        lg = np.log(np.clip(dbar, 1e-8, None)); lm = np.log(np.clip(dmin, 1e-8, None))
        print(f"  log 变异比 std(log d̄)/std(log d_min) = {lg.std()/lm.std():.3f}"
              f"   ← geom 里全局项的权重（0=退化成 min，0.5=两边各半）")


if __name__ == '__main__':
    main()
