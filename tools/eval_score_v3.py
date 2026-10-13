# -*- coding: utf-8 -*-
"""[生产版] 按选定配方重推 mapping 异常图并落盘，用于与 bank 融合。

配方（分量、指数均可配）：
  raw_score   = d1^raw_alpha  * q_global^(1-raw_alpha)
  learn_score = dlmin^l_alpha * bq_global^(1-l_alpha)
  final       = w*learn_score + (1-w)*raw_score   （逐视图 → 插值 448 → 翻转 → 平均）

用法:
  python eval_score_v3.py --save vpt35_visa4shot --class candle --shots 4 \
    --data_path /home/jjquan/experiment/visual2/visa_prepared \
    --ckpt_dir results/visa/vpt35_visa4shot \
    --raw_alpha 0.5 --raw_global 512 --l_alpha 0.7 --l_global 64 --w 0.5 \
    --save_maps_dir .maps_v3_nocrop448_visa4
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
from shot_utils import get_shots, shots_hash         # noqa: E402
NAMES = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
SUBN = 1024


def load_mod(path):
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def _pow(local, glob, a):
    e = 1e-6
    return local.clamp_min(e).pow(a) * glob.clamp_min(e).pow(1.0 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='vpt35_visa4shot')
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/experiment/visual2/visa_prepared')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--scales', default='1.5')
    ap.add_argument('--flip', type=int, default=1)
    ap.add_argument('--raw_alpha', type=float, default=0.5)
    ap.add_argument('--raw_global', type=int, default=512, help='raw 全局项分位（1024 子采样中第 k 小）')
    ap.add_argument('--l_alpha', type=float, default=0.7)
    ap.add_argument('--l_global', type=int, default=64, help='learn 全局项分位')
    ap.add_argument('--w', type=float, default=0.5, help='learn 侧权重')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    ap.add_argument('--sub_seed', type=int, default=1234)
    ap.add_argument('--save_maps_dir', default='')
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
    gen = torch.Generator().manual_seed(args.sub_seed)
    sub_idx = torch.randperm(M, generator=gen)[:min(SUBN, M)].to(args.device)

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    scales = tuple(float(x) for x in args.scales.split(','))

    def score_of(d, sub):
        ds_ = torch.sort(d.index_select(-1, sub), dim=-1).values
        d1 = d.min(-1).values
        q = ds_[..., min(args.raw_global, ds_.shape[-1]) - 1]
        return _pow(d1, q, args.raw_alpha)

    raw_store, gt_px_l, gt_sp_l = [], [], []
    base_hw = (args.size, args.size)
    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(args.device)
            acc, n_view = None, 0
            for s in scales:
                v_img = img if s == 1.0 else F.interpolate(img, scale_factor=s, mode='bilinear',
                                                           align_corners=False)
                for vi, view in enumerate([v_img, v_img.flip(3)] if args.flip else [v_img]):
                    ret = model(view, is_training=True)
                    fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                    flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
                    _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                    flr = torch.cat([flr, _g], dim=1)
                    n_el = fp.shape[1]
                    gg = int(round(n_el ** 0.5))

                    p = F.normalize(fp, dim=-1)
                    d = (1.0 - torch.bmm(p, q_raw.t().unsqueeze(0)
                                         .expand(p.shape[0], -1, -1))).clamp_min(0)
                    m_raw = score_of(d, sub_idx).reshape(-1, 1, gg, gg)

                    pl = F.normalize(model.token_mapping(fp), dim=-1)
                    ql = F.normalize(model.token_mapping(flr), dim=-1)
                    dl_ = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                    db = (1.0 - torch.bmm(pl, q_bank_l.t().unsqueeze(0)
                                          .expand(pl.shape[0], -1, -1))).clamp_min(0)
                    dls = torch.sort(db.index_select(-1, sub_idx), dim=-1).values
                    dlmin = dl_.min(-1).values
                    ql_ = dls[..., min(args.l_global, dls.shape[-1]) - 1]
                    m_learn = _pow(dlmin, ql_, args.l_alpha).reshape(-1, 1, gg, gg)

                    am = F.interpolate(args.w * m_learn + (1.0 - args.w) * m_raw,
                                       size=base_hw, mode='bilinear', align_corners=True)
                    if vi == 1:
                        am = am.flip(3)
                    acc = am if acc is None else acc + am
                    n_view += 1
            gt_px = (gt.to(args.device) > 0.5).float()
            if gt_px.shape[1] > 1:
                gt_px = torch.max(gt_px, dim=1, keepdim=True)[0]
            raw_store.append(acc / max(1, n_view))
            gt_px_l.append(gt_px)
            gt_sp_l.append(label)

    raw_map = torch.cat(raw_store, 0)
    if args.save_maps_dir:
        os.makedirs(args.save_maps_dir, exist_ok=True)
        torch.save(raw_map.cpu(), os.path.join(args.save_maps_dir, f'{args.cls}.map'))
        print(f'  [maps] 保存 {args.save_maps_dir}/{args.cls}.map shape={tuple(raw_map.shape)}')

    am = gk(raw_map)
    flat = am.flatten(1)
    kk = max(1, int(flat.shape[1] * 0.01))
    pr_sp = torch.sort(flat, dim=1, descending=True)[0][:, :kk].mean(dim=1)
    r = list(mod.ader_evaluator(am[:, 0].cpu().numpy(), pr_sp.flatten().cpu().numpy(),
                                torch.cat(gt_px_l, 0)[:, 0].cpu().numpy(),
                                torch.cat(gt_sp_l).flatten().cpu().numpy()))
    print(f"[v3] {args.cls}  raw(d1^{args.raw_alpha}*q{args.raw_global}) "
          f"learn(dlmin^{args.l_alpha}*bq{args.l_global}) w={args.w}")
    print("  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, r)))
    print("  CSV:" + ",".join(f"{v:.4f}" for v in r))


if __name__ == '__main__':
    main()
