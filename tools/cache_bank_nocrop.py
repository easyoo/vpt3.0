# -*- coding: utf-8 -*-
"""[落盘] 在"测试不裁剪"口径下重建**原版 MemoryBank** 的逐图异常图。

与评估完全同一几何：`Resize(448) → ToTensor → Normalize`（**不做 CenterCrop**），
bank 的 4 张 shot 也用同一几何 + 原版 augmentation（22 张/图）。
输出：<out_dir>/<cls>.map  [N,1,448,448] float32（未平滑，便于后续融合）。

用法：python cache_bank_nocrop.py --class transistor --size 448 --out_dir .errmap_nocrop448
"""
import argparse
import importlib.util
import os
import random
import sys

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash        # noqa: E402  [统一数据源]


def load_mod(p):
    spec = importlib.util.spec_from_file_location('m', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['m'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_dir', default='.errmap_nocrop448')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt2', 'main-fewshot-vpt2.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    from util.memorybank import MemoryBank

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    mod.args = a
    mod.device = args.device

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    # [统一数据源] 与训练/其它脚本共用同一份 shot 清单（改 shots 数时两边仍然一致）
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
    imgs = [tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks]
    normal_imgs = torch.cat([mod.augmentation(im) for im in imgs], 0).to(args.device)
    print(f'[bank-nocrop{args.size}] {args.cls}: {len(picks)} shot → 增强后 {tuple(normal_imgs.shape)}')
    bank = MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_imgs, info=print, device=args.device)

    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    out = []
    for img, gt, label, _ in tqdm(dl, ncols=60, desc=f'bank{args.size}'):
        am = bank.query_batch(img.to(args.device))
        out.append(torch.tensor(am).unsqueeze(1).to(args.device))
    m = torch.cat(out, 0).cpu()
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f'{args.cls}.map')
    torch.save(m, p)
    print(f'[bank-nocrop{args.size}] 已保存 {p} shape={tuple(m.shape)}')


if __name__ == '__main__':
    main()
