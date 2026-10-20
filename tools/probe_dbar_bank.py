# -*- coding: utf-8 -*-
"""[诊断 · bank 分支] MemoryBank 的 `d̄ / d_min` 分布。

原版 `query_batch` 用 `faiss_index.search(q, k=1)` 只取最近邻；这里改成 k=KNN 同时拿到
d_min 与 d̄（前 K 个近邻的均值近似全库均值），用于判断 geom 在 bank 分支是否有信息量。

用法: python probe_dbar_bank.py --class candle --shots 4 --size 448 \
        --data_path /home/jjquan/experiment/visual2/visa_prepared [--knn 512] [--limit 6]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F1
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots          # noqa: E402


def load_mod(path):
    spec = importlib.util.spec_from_file_location('mb_mod', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['mb_mod'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--knn', type=int, default=512)
    ap.add_argument('--limit', type=int, default=6)
    ap.add_argument('--data_path', default='/home/jjquan/experiment/visual2/visa_prepared')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--mb', default=os.path.join(HERE, 'code_vpt2', 'util', 'memorybank.py'))
    ap.add_argument('--vpt', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    mb = load_mod(args.mb)
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.vpt)))
    vpt = load_mod(args.vpt)

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
    imgs = [tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks]
    normal_imgs = torch.cat([vpt.augmentation(im) for im in imgs], 0).to(args.device)
    print(f'[bank] {args.cls}: {len(picks)} shot → 增强 {tuple(normal_imgs.shape)}')
    bank = mb.MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_imgs, info=print,
                         device=args.device)
    K = args.knn

    ds = vpt.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test', sort_items=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    store = {1: [], 2: []}
    with torch.no_grad():
        for bi, (img, gt, label, _) in enumerate(dl):
            if bi >= args.limit:
                break
            img = img.to(args.device)
            qf = mb.extract_fused_features(bank.model_t, img)
            qf = [F1.normalize(x, dim=-1) for x in qf]
            for si, (feat, idx) in enumerate(zip(qf, [bank.faiss_index_1, bank.faiss_index_2]), 1):
                flat = feat.reshape(-1, feat.shape[-1]).cpu().contiguous().numpy()
                d, _ = idx.search(flat, k=K)
                if bank.use_cos:
                    d = 1.0 - d
                d = np.clip(d, 0.0, None)
                store[si].append(np.stack([d[:, 0], d.mean(1)], -1))

    print(f"=== {args.cls} bank 分支（shot={args.shots}, k={K}, 前 {args.limit} batch）===")
    for si in (1, 2):
        v = np.concatenate(store[si], 0)
        dmin, dbar = v[:, 0], v[:, 1]
        ratio = dbar / np.clip(dmin, 1e-8, None)
        print(f"\n[stage-{si}]  (库 patch 数 = {bank.faiss_index_1.ntotal if si == 1 else bank.faiss_index_2.ntotal})")
        print(f"  d_min : p10={np.percentile(dmin,10):.4f} p50={np.percentile(dmin,50):.4f} "
              f"p90={np.percentile(dmin,90):.4f}")
        print(f"  d_bar : p10={np.percentile(dbar,10):.4f} p50={np.percentile(dbar,50):.4f} "
              f"p90={np.percentile(dbar,90):.4f}")
        print(f"  d_bar/d_min : p10={np.percentile(ratio,10):.2f} p50={np.percentile(ratio,50):.2f} "
              f"p90={np.percentile(ratio,90):.2f}")
        # 关键判据：d̄ 的对数变异 / d_min 的对数变异
        lg = np.log(np.clip(dbar, 1e-8, None)); lm = np.log(np.clip(dmin, 1e-8, None))
        print(f"  log 变异比  std(log d̄)/std(log d_min) = {lg.std()/lm.std():.3f}")


if __name__ == '__main__':
    main()
