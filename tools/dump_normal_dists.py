# -*- coding: utf-8 -*-
"""为每个类别缓存 4-shot 正常 patch 到可学习原型的余弦距离分布。

用途：打分阶段用 shot-only 分位数/尺度把每个原型校准到可比较的典型性空间。
这不是额外测试前向：测试图仍只前向一次；这里只处理训练侧的正常 shot。
"""
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
from shot_utils import get_shots, shots_hash  # noqa: E402


def load_mod(path):
    spec = importlib.util.spec_from_file_location('normal_dist_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['normal_dist_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True)
    ap.add_argument('--save', default='m4m6r3_4shot_clpt3')
    ap.add_argument('--ckpt_dir', default=None)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--shot_seed', type=int, default=1)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--proto_set', default='unique')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
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
    a.learnable_num_per_layer = 32
    a.res_mapping = 0
    a.global_proto = 1
    a.metric_head = 1
    a.token_init_std = 0.02
    mod.args = a

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                    T.Normalize(mean, std)])
    os.makedirs(args.out_dir, exist_ok=True)
    for cls in args.classes.split():
        model = mod.vision_transformer.__dict__['vit_base'](
            patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
            num_register_tokens=4, interpolate_antialias=False,
            interpolate_offset=0.1, arg=a)
        ck = os.path.join(args.ckpt_dir or os.path.join('results', 'vpt4', args.save),
                          f'model_{cls}_35.pth')
        model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['dino'],
                              strict=False)
        model.to(args.device).eval()
        picks = get_shots(args.data_path, cls, shots=args.shots, seed=args.shot_seed)
        root = os.path.join(args.data_path, cls, 'train', 'good')
        imgs = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB'))
                            for f in picks], 0).to(args.device)
        if abs(args.scale - 1.0) > 1e-8:
            imgs = F.interpolate(imgs, scale_factor=args.scale, mode='bilinear',
                                 align_corners=False)
        ds, pls, protos = [], [], None
        with torch.no_grad():
            for i in range(0, imgs.shape[0], 2):
                ret = model(imgs[i:i + 2], is_training=True)
                fp = torch.sum(torch.stack(
                    [ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
                flr = torch.cat(ret['feat_list'], dim=1)
                g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                flr = (torch.cat([flr[:, :g.shape[1]], g], dim=1)
                       if args.proto_set == 'unique' else torch.cat([flr, g], 1))
                pl = F.normalize(model.token_mapping(fp), dim=-1)
                ql = F.normalize(model.token_mapping(flr), dim=-1)
                protos = ql[0].cpu().half()
                pls.append(pl.cpu().half())
                dist = (1.0 - torch.bmm(pl, ql.transpose(1, 2))).clamp_min(0)
                ds.append(dist.cpu().half())
        d = torch.cat(ds, 0)
        torch.save({'d': d, 'pl': torch.cat(pls, 0), 'protos': protos, 'picks': picks,
                    'shot_hash': shots_hash(picks), 'num_patches': int(d.shape[1]),
                    'scale': float(args.scale), 'proto_set': args.proto_set},
                   os.path.join(args.out_dir, f'{cls}.pt'))
        print(f'[normal] {cls} shot{args.shots} {picks} d={tuple(d.shape)}', flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
