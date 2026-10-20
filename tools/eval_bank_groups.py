# -*- coding: utf-8 -*-
"""只使用增强 bank，按原始 shot 的 22 个增强视图分组查询。

不保留/不读取原始 4-shot bank。对每个测试 patch 得到 4 个“增强组距离”，
之后离线比较 min/mean/median 等组聚合方式。
"""
import argparse
import importlib.util
import json
import os
import sys

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shot_utils import get_shots, shots_hash  # noqa: E402


def load_mod(path, name):
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
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--bank_chunk', type=int, default=2)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--vpt', default=os.path.join(HERE, 'code_vpt2', 'main-fewshot-vpt2.py'))
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.vpt)))
    vpt = load_mod(args.vpt, 'vpt_group_mod')
    mb = load_mod(os.path.join(HERE, 'code_vpt2', 'util', 'memorybank.py'),
                  'mb_group_mod')

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
    groups = [vpt.augmentation(im) for im in imgs]
    normal_imgs = torch.cat(groups, 0)
    print(f'[bank-groups] {len(groups)} 组 × {groups[0].shape[0]} 视图 = '
          f'{normal_imgs.shape[0]} 张增强图')

    ch = max(1, int(args.bank_chunk))
    bank = mb.MemoryBank(target_cls=args.cls, use_json=False,
                         imgs=normal_imgs[:ch].to(args.device), info=print,
                         device=args.device)
    for i in range(ch, normal_imgs.shape[0], ch):
        bank.extract_features(normal_imgs[i:i + ch].to(args.device))

    views_per_group = groups[0].shape[0]
    patches_per_view = None
    feats = []
    for stage_idx, idx in enumerate([bank.faiss_index_1, bank.faiss_index_2], 1):
        flat = idx.reconstruct_n(0, idx.ntotal)
        patches_per_view = flat.shape[0] // normal_imgs.shape[0]
        grouped = []
        for gi in range(len(groups)):
            lo = gi * views_per_group * patches_per_view
            hi = (gi + 1) * views_per_group * patches_per_view
            sub = np.ascontiguousarray(flat[lo:hi])
            gidx = faiss.IndexFlatIP(sub.shape[1])
            gidx.add(sub)
            grouped.append(gidx)
        feats.append(grouped)
        print(f'[bank-groups] stage{stage_idx}: 每组 '
              f'{views_per_group * patches_per_view} patches')

    ds = vpt.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=tf, gt_transform=gtf, phase='test')
    dl = torch.utils.data.DataLoader(ds, batch_size=max(1, int(args.batch)),
                                     shuffle=False, num_workers=4)
    out = [[], []]
    for img, _, _, _ in tqdm(dl, ncols=60, desc='group-knn'):
        with torch.no_grad():
            qfs = mb.extract_fused_features(bank.model_t, img.to(args.device))
            qfs = [F.normalize(rearrange(x, 'b l c->b c l'), dim=-1) for x in qfs]
        bsz = img.shape[0]
        for si, qf in enumerate(qfs):
            n_tok = qf.shape[-1]
            g = int(round(n_tok ** 0.5))
            assert g * g == n_tok
            qflat = rearrange(qf, 'b c l->(b l) c').cpu().contiguous().numpy()
            maps = []
            for idx in feats[si]:
                d, _ = idx.search(qflat, k=1)
                dist = torch.tensor(1.0 - d[:, 0], dtype=torch.float32)
                m = dist.reshape(bsz, 1, g, g)
                m = F.interpolate(m, (args.size, args.size), mode='bilinear',
                                  align_corners=True)
                maps.append(m)
            out[si].append(torch.stack(maps, 1))  # [B, G, 1, H, W]

    os.makedirs(args.out_dir, exist_ok=True)
    for si in (0, 1):
        arr = torch.cat(out[si], 0)
        p = os.path.join(args.out_dir, f'{args.cls}_s{si + 1}.npy')
        np.save(p, arr.numpy().astype(np.float16))
        print(f'[bank-groups] 保存 {p} shape={tuple(arr.shape)}')
    with open(os.path.join(args.out_dir, f'{args.cls}.json'), 'w') as f:
        json.dump({'cls': args.cls, 'groups': len(groups), 'views_per_group': views_per_group,
                   'size': args.size, 'shots': args.shots,
                   'picks': picks, 'shot_hash': shots_hash(picks)}, f, indent=1)


if __name__ == '__main__':
    main()
