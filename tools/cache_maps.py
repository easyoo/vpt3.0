# -*- coding: utf-8 -*-
"""[落盘缓存] 把 mapping 分支与 bank 分支的**逐图异常分数图**保存到磁盘，
后续可以只读缓存、任意换融合规则（相加 / 相乘 / 加权 / z 归一化 …）重新算指标，无需再跑网络。

产出（每类）：
  .maps_audit/<cls>_mapping.map   mapping 分支(现状打分 type=1)的 392×392 异常图, float32, [N,1,392,392]
  .errmap_audit/<cls>.map         原版 MemoryBank 的异常图（已存在则直接复用）, float32
  .maps_audit/<cls>_gt.map        GT 掩码 [N,1,392,392] uint8（与上面同顺序）
  .maps_audit/<cls>_meta.json     {labels, paths, ckpt} —— 便于核对"两个分支是否同一张图"
"""
import argparse
import importlib.util
import json
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
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--script', default=os.path.join(HERE, 'code_control', 'main-fewshot-control.py'))
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--skip_bank', action='store_true', help='bank 图已缓存时跳过重建')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    from util.memorybank import MemoryBank

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

    data_transform, gt_transform = mod.get_data_transforms(448, 392)
    test_data = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                                 transform=data_transform, gt_transform=gt_transform, phase='test')
    dl = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    map_dir = os.path.join(HERE, '.maps_audit'); os.makedirs(map_dir, exist_ok=True)
    bank_dir = os.path.join(HERE, '.errmap_audit'); os.makedirs(bank_dir, exist_ok=True)
    bank_path = os.path.join(bank_dir, f'{args.cls}.map')
    map_path = os.path.join(map_dir, f'{args.cls}_mapping.map')
    gt_path = os.path.join(map_dir, f'{args.cls}_gt.map')
    meta_path = os.path.join(map_dir, f'{args.cls}_meta.json')

    # ---- bank 分支异常图（沿用已有缓存；没有才重算）----
    if os.path.exists(bank_path) and args.skip_bank:
        print(f'[skip] bank 图已存在: {bank_path}')
    elif os.path.exists(bank_path):
        print(f'[skip] bank 图已存在（复用）: {bank_path}')
    else:
        root = os.path.join(args.data_path, args.cls, 'train', 'good')
        # [统一数据源] 与训练 / mapping 分支共用同一份 shot 清单
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from shot_utils import get_shots, shots_hash
        picks = get_shots(args.data_path, args.cls, shots=args.shots, seed=args.seed)
        print(f'[shot_utils] {args.cls} shot{args.shots} hash={shots_hash(picks)} picks={picks}')
        imgs = [data_transform(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks]
        normal_imgs = torch.cat([mod.augmentation(im) for im in imgs], dim=0).to(args.device)
        bank = MemoryBank(target_cls=args.cls, use_json=False, imgs=normal_imgs,
                          info=print, device=args.device)
        out = []
        for img, gt, label, _ in tqdm(dl, ncols=60, desc='bank-errmap'):
            am = torch.tensor(bank.query_batch(img.to(args.device))).unsqueeze(1).to(args.device)
            out.append(am.float())
        torch.save(torch.cat(out, 0).cpu(), bank_path)
        print(f'[bank] 已保存 {bank_path}')

    # ---- mapping 分支异常图 + GT + meta ----
    m_out, gt_out, labels, paths = [], [], [], []
    with torch.no_grad():
        for img, gt, label, p in tqdm(dl, ncols=60, desc='mapping-map'):
            img = img.to(args.device)
            ret = model(img, is_training=True)
            pfl, fl = ret['patch_feat_layer'], ret['feat_list']
            fp = torch.sum(torch.stack([pfl[i] for i in a.fused_layer], dim=0), dim=0)
            flr = torch.sum(torch.stack(fl, dim=0), dim=0)
            pm = model.token_mapping(fp); lm = model.token_mapping(flr)
            am = mod.compute_anomaly_scores(pm.cpu().numpy(), lm.cpu().numpy(), img.shape[-1])
            m_out.append(am.float().cpu())
            gt_b = (gt > 0.5).float()
            if gt_b.shape[1] > 1:
                gt_b = torch.max(gt_b, dim=1, keepdim=True)[0]
            gt_out.append(gt_b.cpu().to(torch.uint8))
            labels.extend([int(x) for x in label])
            paths.extend(list(p))
    torch.save(torch.cat(m_out, 0), map_path)
    torch.save(torch.cat(gt_out, 0), gt_path)
    json.dump({'labels': labels, 'paths': paths, 'ckpt': args.ckpt,
               'n': len(labels)}, open(meta_path, 'w'), indent=1)
    print(f'[mapping] 已保存 {map_path}\n[gt] {gt_path}\n[meta] {meta_path}')
    print(f'  样本数 = {len(labels)}（两个分支共用同一次 DataLoader 遍历 → 顺序一致）')


if __name__ == '__main__':
    main()
