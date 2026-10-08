# -*- coding: utf-8 -*-
"""[用户要求] 在原版 MemoryBank 与 mapping 分支之间做分数叠加，看是否 1+1>2。

复刻原版流程（`visual/main-fewshot.py` 的非 map_only 分支）：
  1) 用 4 张正常图（+ 原版 augmentation，共 88 张）构建 `MemoryBank`
     —— 它内部是**另一个模型** `facebook/dinov2-base`（AutoModel）+ faiss 双索引；
  2) 对测试集逐图 `query_batch` 得到 `visual_bank_map`（缓存到 .errmap_audit/<cls>.map）；
  3) 用原版 `evaluation_batch` 的组合口径评估：
       type=1 仅 mapping 原型距离（现状）
       type=3 仅 memory bank
       type=5 **mapping + memory bank**（本次要看的叠加）
       type=7 mapping + 分割头误差图 + memory bank
     全部 392 口径、官方 adeval 指标。
"""
import argparse
import importlib.util
import os
import random
import sys

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(script_path):
    spec = importlib.util.spec_from_file_location('audit_main', script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--class', dest='cls', required=True)
    ap.add_argument('--types', default='1,3,5,7')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--anomaly_data_root', default='/home/jjquan/experiment/visual2/anomaly_samples')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_control', 'main-fewshot-control.py'))
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default=None)
    ap.add_argument('--rebuild_errmap', action='store_true')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    # 只用副本自带的 util（原仓库的 util/vis.py 依赖未安装的 accimage）
    mod = load_mod(args.script)
    from util.memorybank import MemoryBank       # 原版实现

    class A:
        pass
    a = A()
    a.learnable_layers = 1
    a.learnable_num_per_layer = 32
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
    head = nn.Sequential(nn.Linear(768, 384), nn.ReLU(inplace=True),
                         nn.Linear(384, 1), nn.Sigmoid()).to(args.device)
    head.load_state_dict(ck['seg_head']); head.eval()
    mod.segmentation_head = head

    data_transform, gt_transform = mod.get_data_transforms(448, 392)
    # ---- 1) 原版 MemoryBank（4 张正常图 + 原版 augmentation，共 88 张）----
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    files = sorted(os.listdir(root))
    random.seed(args.seed)
    picks = random.sample(files, args.shots)
    imgs = []
    for f in picks:
        imgs.append(data_transform(Image.open(os.path.join(root, f)).convert('RGB')))
    # 与原版一致：逐张调用 augmentation()（它内部按单图 batch 旋转/平移），拼起来共 shots×22 张
    aug_parts = [mod.augmentation(im) for im in imgs]
    normal_imgs = torch.cat(aug_parts, dim=0).to(args.device)
    print(f'[bank] 正常图 {len(picks)} 张 → 增强后 {tuple(normal_imgs.shape)}')
    bank = MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_imgs,
                      info=print, device=args.device)

    # ---- 2) 测试集 errmap（缓存）----
    test_data = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                                 transform=data_transform, gt_transform=gt_transform, phase='test')
    dl = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    cache_dir = os.path.join(HERE, '.errmap_audit')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{args.cls}.map')
    if os.path.exists(cache) and not args.rebuild_errmap:
        errmap = torch.load(cache, weights_only=False)
        print(f'[bank] 载入缓存 errmap {cache}，shape={tuple(errmap.shape)}')
    else:
        out = []
        for img, gt, label, _ in tqdm(dl, ncols=60, desc='errmap'):
            am = bank.query_batch(img.to(args.device))
            am = torch.tensor(am).unsqueeze(1).to(args.device)
            out.append(am)
        errmap = torch.cat(out, dim=0)
        torch.save(errmap.cpu(), cache)
        print(f'[bank] 计算完成并缓存，shape={tuple(errmap.shape)}，size={errmap.numel()/1e6:.1f}M')
    errmap = errmap.to(args.device)

    # ---- 3) 逐类型评估 ----
    rows = {}
    for t in [int(x) for x in args.types.split(',')]:
        r = mod.evaluation_batch(model, dl, args.device, max_ratio=0.01,
                                 resize_mask=None, visual_bank_map=errmap, type=t)
        rows[t] = r
        print(f'  type={t}: ' + '  '.join(f'{n}={v:.4f}' for n, v in
              zip(["I-AUROC","I-AP","I-F1","P-AUROC","P-AP","P-F1","P-AUPRO"], r)))
    if args.out:
        with open(args.out, 'a') as fh:
            for t, r in rows.items():
                fh.write(f'{args.cls},{os.path.basename(args.ckpt)},type{t},' +
                         ','.join(f'{v:.4f}' for v in r) + '\n')


if __name__ == '__main__':
    main()
