# -*- coding: utf-8 -*-
"""[CPU 探针] 量一下 mapping 分支异常分数的真实分布，用来判断 M7 的 margin 是否合理。

背景：M7 = mean relu(s_normal - s_anomaly + margin)。
若 margin 相对"正常/异常的天然分差"太小 → 损失恒为 0，等于没加；
若太大 → 所有样本都进 hinge，梯度噪声大。所以必须先量分布，再定 margin。

打分口径与推理完全一致（每张测试图用**它自己**的原型集自参照）：
    s = 1 - max_k cos(patch, proto_k)

用法：
  python probe_m7_scores.py --save m4px05 --class transistor --n_images 6
  python probe_m7_scores.py --save m4 --class screw --n_images 6 --metric_head 0
"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_audit_module(script_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(script_path)))
    spec = importlib.util.spec_from_file_location('audit_main', script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='m4px05')
    ap.add_argument('--class', dest='cls', default='transistor')
    ap.add_argument('--n_images', type=int, default=6, help='取前 n 张测试图（含好图与坏图）')
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--num_tokens', type=int, default=32)
    ap.add_argument('--res_mapping', type=int, default=0)
    ap.add_argument('--global_proto', type=int, default=1)
    ap.add_argument('--metric_head', type=int, default=1)
    ap.add_argument('--script', default=os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    mod = load_audit_module(args.script)

    class A:
        pass
    a = A()
    a.fused_layer = [1, 2, 3, 4, 6, 8, 11]
    a.fused_layer_learnable = [0]
    a.learnable_layers = 1
    a.learnable_num_per_layer = args.num_tokens
    a.res_mapping = args.res_mapping
    a.global_proto = args.global_proto
    a.metric_head = args.metric_head
    a.temperature = 0.07
    a.scaled_factor = 3
    a.fusion = 'sum'
    mod.args = a
    mod.device = 'cpu'

    model = mod.vision_transformer.__dict__['vit_base'](
        patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
        num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1, arg=a)
    ckpt = os.path.join('results', 'vpt4', args.save, f'model_{args.cls}_35.pth')
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['dino'], strict=False)
    print(f'[probe] ckpt={ckpt}')
    print(f'[probe] 缺失键={list(missing)} 多余键={list(unexpected)}')
    model.eval()

    data_transform, gt_transform = mod.get_data_transforms(448, 392)
    ds = mod.MVTecDataset(root=os.path.join(args.data_path, args.cls),
                          transform=data_transform, gt_transform=gt_transform, phase='test', sort_items=False)
    print(f'[probe] {args.cls} 测试集 {len(ds)} 张，取前 {args.n_images} 张')

    grid = int(round(1024 ** 0.5))      # 32x32 patch
    rows = []
    for i in range(min(args.n_images, len(ds))):
        img, gt, label, _ = ds[i]
        with torch.no_grad():
            ret = model(img.unsqueeze(0), is_training=True)
            fp = torch.sum(torch.stack([ret['patch_feat_layer'][j] for j in a.fused_layer], 0), 0)
            flr = torch.sum(torch.stack(ret['feat_list'], 0), 0)
            if args.global_proto:
                _g = model.tokens_per_layer.token_dict['0'].expand(flr.shape[0], -1, -1)
                flr = torch.cat([flr, _g], dim=1)
            # 与推理端 compute_fused_map._map28 **完全同口径**：两侧都要过 token_mapping！
            # （踩过的坑：漏掉 token_mapping 时分数尺度会从 ~0.03 变成 ~0.9，margin 会选错一个数量级）
            s = mod.proto_distance_scores(model.token_mapping(fp),
                                          model.token_mapping(flr)).reshape(-1)    # [N]
            if i == 0:      # 额外做一次"与真实推理函数"的交叉验证
                # bank 传占位张量即可：w_learn=1.0 时原始特征库那一项权重为 0
                _dummy_bank = torch.randn(8, 768)
                ref_map = mod.compute_fused_map(model, img.unsqueeze(0), _dummy_bank, a.fused_layer,
                                                scales=(1.0,), tta_flip=False, w_learn=1.0)
                gg = int(round(s.numel() ** 0.5))
                ref_small = F.interpolate(ref_map, size=(gg, gg), mode='bilinear',
                                          align_corners=True).reshape(-1)
                d = (ref_small - s).abs().max()
                print(f'  [一致性] 与 compute_fused_map(w_learn=1.0,单视图) 的最大偏差 = {float(d):.3e}')
        g = int(round(s.numel() ** 0.5))
        m = F.adaptive_max_pool2d(gt.unsqueeze(0).float(), output_size=(g, g)).flatten() > 0.5
        rows.append((int(label), s[~m], s[m]))       # (label, 正常像素分数, 异常像素分数)
        if len(rows):
            n = s[~m]
            print(f'  图{i:>3} label={int(label)} 正常像素 {n.numel():>5} 个: '
                  f'p50={n.median():.3f} p95={torch.quantile(n, 0.95):.3f} '
                  + (f'| 异常像素 {int(m.sum()):>4} 个: p05={torch.quantile(s[m], 0.05):.3f} '
                     f'p50={s[m].median():.3f}' if int(m.sum()) > 0 else '| 无异常像素'))

    norm = torch.cat([r[1] for r in rows])
    anom = torch.cat([r[2] for r in rows if r[2].numel() > 0]) if any(r[2].numel() for r in rows) else torch.zeros(0)
    print(f'\n汇总：正常像素 {norm.numel()} 个（p50={norm.median():.3f}, p95={torch.quantile(norm,0.95):.3f}, '
          f'p99={torch.quantile(norm,0.99):.3f}）')
    if anom.numel():
        print(f'      异常像素 {anom.numel()} 个（p05={torch.quantile(anom,0.05):.3f}, '
              f'p50={anom.median():.3f}, p95={torch.quantile(anom,0.95):.3f}）')
        print(f'      天然分差(异常p50 - 正常p95) = {float(anom.median() - torch.quantile(norm,0.95)):+.3f}')
    print('\nmargin 扫描（困难挖掘 n=256，与训练时一致）：')
    for mg in (0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100):
        if anom.numel() == 0:
            break
        v = float(mod.cross_image_rank_loss(norm, anom, margin=mg, n_sample=256))
        ns = torch.topk(norm, min(256, norm.numel())).values
        as_ = torch.topk(anom, min(256, anom.numel()), largest=False).values
        d = (ns.unsqueeze(1) - as_.unsqueeze(0) + mg)
        frac = float((d > 0).float().mean())
        tag = '  <-- 该项恒 0，无梯度' if v == 0 else ''
        print(f'  margin={mg:.3f}  L_M7={v:.5f}  激活对比例={frac*100:5.1f}%  '
              f'(L-margin={v-mg:+.5f} 为真实结构项){tag}')

    if anom.numel():
        tau = float(torch.clamp(torch.quantile(norm, 0.75) - torch.quantile(norm, 0.25), min=1e-6))
        print(f'\n归一化模式（tau = IQR(正常分数) = {tau:.5f}）：')
        for mg in (0.1, 0.25, 0.5, 1.0):
            v = float(mod.cross_image_rank_loss(norm, anom, margin=mg, n_sample=256, normalize=True))
            ns = torch.topk(norm, min(256, norm.numel())).values
            as_ = torch.topk(anom, min(256, anom.numel()), largest=False).values
            d = ((ns.unsqueeze(1) - as_.unsqueeze(0)) / tau).clamp(-20, 20) + mg
            frac = float((d > 0).float().mean())
            print(f'  margin={mg:.2f} tau  L_M7={v:.4f}  激活对比例={frac*100:5.1f}%'
                  f'   → w_rank=0.2 时对总损失的贡献≈{0.2*v:.3f}')


if __name__ == '__main__':
    main()
