# -*- coding: utf-8 -*-
"""打印推理路径上每一步的真实张量形状（用于写文档/核对实现）。"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots          # noqa: E402


def load_mod(p, name='m'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='vpt35_visa4shot')
    ap.add_argument('--class', dest='cls', default='candle')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--data_path', default='/home/jjquan/experiment/visual2/visa_prepared')
    ap.add_argument('--ckpt_dir', default='results/visa/vpt35_visa4shot')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()

    mod = load_mod(os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'), 'mi')

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
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'model_{args.cls}_35.pth'),
                                     map_location='cpu', weights_only=False)['dino'], strict=False)
    model.to(args.device).eval()
    print(f"patch_size=14  img_size(构建)=518  num_register_tokens=4  fused_layer={a.fused_layer}")

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=1)
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    print(f"\n[库] shot 图: {tuple(normal.shape)}")
    bank = mod.build_raw_bank(model, normal, a.fused_layer, k=0)
    print(f"[库] bank(raw patch 库, 未归一化): {tuple(bank.shape)}  "
          f"= {len(picks)} shot x {bank.shape[0]//len(picks)} patch/shot")
    q_raw = F.normalize(bank, dim=-1)
    q_bank_l = F.normalize(model.token_mapping(bank), dim=-1)
    print(f"[库] q_raw(归一化): {tuple(q_raw.shape)}；映射后可学习空间库: {tuple(q_bank_l.shape)}")
    sub = torch.randperm(bank.shape[0], generator=torch.Generator().manual_seed(1234))[:1024]
    print(f"[库] 全局分位数子采样索引: {tuple(sub.shape)}（种子 1234）")

    img = normal[:args.batch].to(args.device)
    print(f"\n[输入] 448 图: {tuple(img.shape)}")
    for sname, scale in (('1.0x', 1.0), ('1.5x', 1.5)):
        v = img if scale == 1.0 else F.interpolate(img, scale_factor=scale, mode='bilinear',
                                                   align_corners=False)
        with torch.no_grad():
            ret = model(v, is_training=True)
        pfl, fl = ret['patch_feat_layer'], ret['feat_list']
        print(f"[视图 {sname}] 输入 {tuple(v.shape)}；patch 网格 {int(v.shape[-1]//14)}x{int(v.shape[-1]//14)}"
              f"；patch_feat_layer: {len(pfl)} 层 x {tuple(pfl[0].shape)}；feat_list: "
              f"{len(fl)} 项 x {tuple(fl[0].shape)}")
        fp = torch.sum(torch.stack([pfl[j] for j in a.fused_layer], 0), 0)
        flr = torch.sum(torch.stack(fl, 0), 0)
        print(f"    fp=Σ{fused_str(a.fused_layer)}: {tuple(fp.shape)}；"
              f"flr=Σfeat_list: {tuple(flr.shape)}（=64 = 32 个 layer-0 token 被追加两次）")
        gl = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
        flr2 = torch.cat([flr, gl], dim=1)
        print(f"    ⊕ 全局锚点 token_dict['0'] {tuple(gl.shape)} → 原型集 {tuple(flr2.shape)}")
        d = (1.0 - torch.bmm(F.normalize(fp, dim=-1), q_raw.t().unsqueeze(0)
                             .expand(fp.shape[0], -1, -1))).clamp_min(0)
        ds = torch.sort(d.index_select(-1, sub.to(d.device)), dim=-1).values
        print(f"    ① 原始库侧: d={tuple(d.shape)} → top-1024 子采样排序 {tuple(ds.shape)}；"
              f"d1={tuple(d.min(-1).values.shape)} q50={tuple(ds[..., 511].shape)}")
        pl = F.normalize(model.token_mapping(fp), dim=-1)
        ql = F.normalize(model.token_mapping(flr2), dim=-1)
        dl = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
        db = (1.0 - torch.bmm(pl, q_bank_l.t().unsqueeze(0).expand(pl.shape[0], -1, -1))).clamp_min(0)
        dbs = torch.sort(db.index_select(-1, sub.to(db.device)), dim=-1).values
        print(f"    ② 可学习侧: pl={tuple(pl.shape)} ql={tuple(ql.shape)} dl={tuple(dl.shape)}"
              f" → dlmin={tuple(dl.min(-1).values.shape)}；"
              f"db={tuple(db.shape)} → bq64={tuple(dbs[..., 63].shape)}")
    print("\n[文档用常量] " + fused_str(a.fused_layer))


def fused_str(fl):
    return "patch_feat_layer[" + ",".join(str(i) for i in fl) + "]"


if __name__ == '__main__':
    main()
