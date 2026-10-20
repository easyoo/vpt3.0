# -*- coding: utf-8 -*-
"""诊断：可学习原型侧的"全局项"到底有没有信息量？

对每张测试图，逐 patch 计算：
  dlmin = 到最近可学习原型的 cos 距离
  dlmed = 到**全部 64 个可学习原型**距离的中位数（"和所有原型的一般距离"）
  q512  = 到正常 patch 库距离的中位数（原方案的全局项）
输出三者的均值/标准差/变异系数(CV=std/mean) 以及 corr(dlmin, dlmed)。
CV 越小 → 该量在 patch 间越接近常数 → 作为"全局项"越无信息量。
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
from shot_utils import get_shots          # noqa: E402


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    spec = importlib.util.spec_from_file_location('dm_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['dm_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=2)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--save', default='m4m6r3_2shot')
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    mod = load_mod(args.script)
    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.softmin = 0
    a.learnable_num_per_layer = 32
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
    ck = os.path.join(args.ckpt_dir or f'results/vpt4/{args.save}', f'model_{args.cls}_35.pth')
    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['dino'],
                          strict=False)
    model.to(args.device).eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])   # 掩码：只 resize（本探针不用 GT）
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=1)
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)
    q_raw = F.normalize(bank, dim=-1)
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls), transform=tf,
                          gt_transform=gtf, phase='test', sort_items=True)

    D = {'dlmin': [], 'dlmed': [], 'q512': []}
    with torch.no_grad():
        for i in range(min(args.limit, len(ds))):
            img, _, _, _ = ds[i]
            v = F.interpolate(img[None], scale_factor=args.scale, mode='bilinear',
                              align_corners=False).to(args.device)
            ret = model(v, is_training=True)
            fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
            flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
            _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
            flr = torch.cat([flr, _g], dim=1)
            pl = F.normalize(model.token_mapping(fp), dim=-1)
            ql = F.normalize(model.token_mapping(flr), dim=-1)
            d_proto = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)   # [1,N,64]
            D['dlmin'].append(d_proto.min(-1).values.flatten().cpu())
            D['dlmed'].append(d_proto.median(-1).values.flatten().cpu())
            p_raw = F.normalize(fp, dim=-1)
            d_raw = (1.0 - torch.bmm(p_raw, q_raw.t().unsqueeze(0))).clamp_min(0)
            D['q512'].append(d_raw.median(-1).values.flatten().cpu())

    out = {k: torch.cat(v).numpy() for k, v in D.items()}
    print(f'[{args.cls}] {min(args.limit, len(ds))} 张测试图，patch 数 {out["dlmin"].size}')
    for k in ('dlmin', 'dlmed', 'q512'):
        x = out[k]
        print(f'  {k:<6} mean={x.mean():.4f}  std={x.std():.4f}  CV(std/mean)={x.std()/x.mean():.4f}')
    c = np.corrcoef(out['dlmin'], out['dlmed'])[0, 1]
    print(f'  corr(dlmin, dlmed) = {c:.4f}   '
          f'（dlmed 跨 patch 近似常数 → 当"全局项"几乎不提供信息）')


if __name__ == '__main__':
    main()
