# -*- coding: utf-8 -*-
"""[bank 分支 · 打分搜索] 一次推理把 bank 的"距离分位数分量"按 patch 网格落盘。

背景：原版 MemoryBank 的查询是 `faiss.search(q, k=1)` 硬编码的 1-NN，
两个 stage 的 1-NN 距离平均后作为异常图。这里改成 `k=1024`，
同时拿到每个 patch 的 d1 与距离分布的多个分位数（q8/q64/q256/q512/q1024），
之后可离线枚举 `d1^a · q_p^(1-a)` 与两 stage 的组合方式。

用法: python bank_grid.py --class candle --shots 4 --size 448 --knn 1024 \
        --data_path /home/jjquan/experiment/visual2/visa_prepared --out_dir .bgrid_visa4
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F1
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash         # noqa: E402
QS = (8, 64, 256, 512, 1024)


def load_mod(path, name='m'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--knn', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--data_path', default='/home/jjquan/experiment/visual2/visa_prepared')
    ap.add_argument('--out_dir', default='.bgrid_visa4')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--mb', default=os.path.join(HERE, 'code_vpt2', 'util', 'memorybank.py'))
    ap.add_argument('--vpt', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    mb = load_mod(args.mb, 'mb_mod')
    vpt = load_mod(args.vpt, 'vpt_mod')

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    vpt.args = a
    vpt.device = args.device

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
    print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
    imgs = [tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks]
    normal_imgs = torch.cat([vpt.augmentation(im) for im in imgs], 0).to(args.device)
    print(f'[bank] {args.cls}: {len(picks)} shot → 增强 {tuple(normal_imgs.shape)}')
    bank = mb.MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_imgs, info=print,
                         device=args.device)
    K = args.knn

    ds = vpt.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    if args.limit:
        ds.img_paths = ds.img_paths[:args.limit]
        ds.gt_paths = ds.gt_paths[:args.limit]
        ds.labels = ds.labels[:args.limit]
        ds.types = ds.types[:args.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    keys = [f's{s}_{c}' for s in (1, 2) for c in (['d1'] + [f'q{q}' for q in QS] + ['mean'])]
    store = {k: [] for k in keys}
    gt_all, label_all = [], []

    with torch.no_grad():
        for bi, (img, gt, label, _) in enumerate(dl):
            img = img.to(args.device)
            qf = mb.extract_fused_features(bank.model_t, img)
            qf = [F1.normalize(x, dim=-1) for x in qf]
            for si, (feat, idx) in enumerate(zip(qf, [bank.faiss_index_1, bank.faiss_index_2]), 1):
                B, N, C = feat.shape
                g = int(round(N ** 0.5))
                flat = feat.reshape(-1, C).cpu().contiguous().numpy()
                d, _ = idx.search(flat, k=K)
                if bank.use_cos:
                    d = 1.0 - d
                d = np.sort(np.clip(d, 0.0, None), axis=1)
                store[f's{si}_d1'].append(d[:, 0].reshape(B, g, g).astype(np.float16))
                for q in QS:
                    store[f's{si}_q{q}'].append(
                        d[:, min(q, d.shape[1]) - 1].reshape(B, g, g).astype(np.float16))
                store[f's{si}_mean'].append(d.mean(1).reshape(B, g, g).astype(np.float16))
            gt_px = (gt > 0.5).float()
            if gt_px.shape[1] > 1:
                gt_px = torch.max(gt_px, dim=1, keepdim=True)[0]
            gt_all.append((gt_px[:, 0] > 0.5).to(torch.uint8).cpu())
            label_all.append(label)
            print(f'  batch {bi} done', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {k: np.concatenate(v, 0) for k, v in store.items()}
    payload['gt'] = torch.cat(gt_all, 0).numpy()
    payload['label'] = torch.cat(label_all).flatten().numpy().astype(np.uint8)
    payload['meta'] = np.array(json.dumps({'cls': args.cls, 'shots': args.shots,
                                           'size': args.size, 'knn': K,
                                           'qs': list(QS), 'use_cos': bool(bank.use_cos)}))
    p = os.path.join(args.out_dir, f'{args.cls}.npz')
    np.savez_compressed(p, **payload)
    print(f'[bgrid] 保存 {p} k={K} keys={len(payload)}')


if __name__ == '__main__':
    main()
