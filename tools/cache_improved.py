# -*- coding: utf-8 -*-
"""[落盘缓存] 保存**改善后 mapping 分支**的逐图异常分数图（推荐档口径）。

改善后 mapping = 打分图融合(可学习原型图 ⊕ 原始特征空间正常 patch 库图) + 1.5× 缩放 + 水平翻转
（即 production 的 `compute_fused_map`，2 次前向）。产出：
  .maps_audit/<cls>_improved.map   [N,1,392,392] float32（未做高斯平滑，便于后续任意融合）
"""
import argparse
import importlib.util
import os
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(p):
    spec = importlib.util.spec_from_file_location('audit_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_norm', 'main-fewshot-norm.py'))
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--scales', default='1.5')
    ap.add_argument('--flip', type=int, default=1)
    ap.add_argument('--w_learn', type=float, default=0.5)
    # [vpt4] 用我们自己的 checkpoint（code_vpt4）时必须显式给架构/机制开关，
    # 否则 strict=False 会静默丢掉映射权重、或把 M4/M6 关掉（这正是本轮踩过的坑）。
    ap.add_argument('--out_suffix', default='improved',
                    help='输出文件名后缀：.maps_audit/<cls>_<suffix>.map（默认覆盖旧缓存 improved）')
    ap.add_argument('--res_mapping', type=int, default=0)
    ap.add_argument('--global_proto', type=int, default=0)
    ap.add_argument('--metric_head', type=int, default=0)
    ap.add_argument('--num_tokens', type=int, default=32)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)

    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = args.num_tokens
    a.res_mapping = args.res_mapping
    a.global_proto = args.global_proto
    a.metric_head = args.metric_head
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    mod.args = a
    mod.device = args.device

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['dino'], strict=False)
    model.to(args.device).eval()

    tf, gtf = mod.get_data_transforms(448, 392)
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    files = sorted(os.listdir(root))
    random.seed(args.seed)
    picks = random.sample(files, args.shots)
    normal = torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
    bank = mod.build_raw_bank(model, normal, a.fused_layer)      # 原始特征空间正常 patch 库
    print(f'[improved] 原始特征库 {tuple(bank.shape)}')

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    scales = tuple(float(x) for x in args.scales.split(','))
    out = []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(dl, ncols=60, desc='improved'):
            img = img.to(args.device)
            am = mod.compute_fused_map(model, img, bank, a.fused_layer, scales=scales,
                                       tta_flip=bool(args.flip), w_learn=args.w_learn)
            out.append(am.float().cpu())
    map_dir = os.path.join(HERE, '.maps_audit'); os.makedirs(map_dir, exist_ok=True)
    p = os.path.join(map_dir, f'{args.cls}_{args.out_suffix}.map')
    torch.save(torch.cat(out, 0), p)
    print(f'[improved] 已保存 {p}  shape={tuple(torch.cat(out,0).shape)}')


if __name__ == '__main__':
    main()
