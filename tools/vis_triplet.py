# -*- coding: utf-8 -*-
"""[可视化] 逐测试样本输出三联异常图：mapping-only | bank-only | 叠加(0.3/0.7)。

要点
  1. **按类别归一化**：两支各在"整类全部像素"上做 z 标准化（z=(x-μ_cls)/σ_cls），
     叠加图 = 0.3·z(mapping) + 0.7·z(bank) —— 与评估所用融合口径完全一致，
     因此不同样本之间的颜色**可以直接横向比较**；
  2. 同一类别的所有图共用**同一色标区间**（取该类归一化值的 0.5%~99.5% 分位），
     否则每图自适应色标会掩盖跨图差异；
  3. 每个样本一张 PNG：三个子图 + 绿色 GT 轮廓 + 标题里的图级分数（top-1% 均值）。

用法
  python vis_triplet.py --classes all --out_dir results/vpt4/vis_triplet
  python vis_triplet.py --classes transistor,tile --max 20        # 每类只画 20 张
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def use_cjk_font():
    """让图内中文正常显示（服务器有 Noto Sans CJK；缺字体时自动退回英文标签）。"""
    import matplotlib.font_manager as fm
    cands = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
             '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']
    for p in cands:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                name = fm.FontProperties(fname=p).get_name()
                plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                return True
            except Exception:
                pass
    return False

HERE = os.path.dirname(os.path.abspath(__file__))
ALL15 = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()


def load_gaussian():
    import importlib.util
    sys.path.insert(0, os.path.join(HERE, 'code_vpt4'))
    spec = importlib.util.spec_from_file_location('mm', os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['mm'] = mod
    spec.loader.exec_module(mod)
    return mod.get_gaussian_kernel(kernel_size=5, sigma=4)


def z_cls(x):
    """类别内 z 标准化（整类全部像素的均值/标准差）——与评估融合口径一致。"""
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='all')
    ap.add_argument('--map_tag', default='improved_m4m6r3')
    ap.add_argument('--w', type=float, default=0.3, help='mapping 权重（叠加 = w·z(mapping)+(1-w)·z(bank)）')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'results', 'vpt4', 'vis_triplet'))
    ap.add_argument('--dpi', type=int, default=110)
    ap.add_argument('--max', type=int, default=0, help='每类最多画多少张（0=全部）')
    ap.add_argument('--smooth', type=int, default=1, help='1=按评估口径做 σ=4 高斯平滑（推荐，图面与指标一致）')
    ap.add_argument('--scale_pct', type=float, default=99.5, help='色标上界取该类归一化值的多少分位')
    args = ap.parse_args()

    classes = ALL15 if args.classes == 'all' else args.classes.split(',')
    cjk = use_cjk_font()
    gk = load_gaussian() if args.smooth else None
    # 原图按数据变换的同一几何取像素：Resize(448) → CenterCrop(392)，与异常图/GT 逐像素对齐
    disp_tf = T.Compose([T.Resize((448, 448)), T.CenterCrop(392)])
    os.makedirs(args.out_dir, exist_ok=True)
    cmap = plt.get_cmap('jet')
    total = 0

    for cls in classes:
        mp = os.path.join(HERE, '.maps_audit', f'{cls}_{args.map_tag}.map')
        bp = os.path.join(HERE, '.errmap_audit', f'{cls}.map')
        gp = os.path.join(HERE, '.maps_audit', f'{cls}_gt.map')
        if not (os.path.exists(mp) and os.path.exists(bp) and os.path.exists(gp)):
            print(f'[skip] {cls} 缺缓存'); continue
        M = torch.load(mp, weights_only=False).float()
        B = torch.load(bp, weights_only=False).float()
        G = torch.load(gp, weights_only=False).float()
        assert M.shape == B.shape == G.shape, (cls, M.shape, B.shape, G.shape)
        import json
        meta = json.load(open(os.path.join(HERE, '.maps_audit', f'{cls}_meta.json')))
        paths = meta['paths']
        assert len(paths) == M.shape[0], (cls, len(paths), M.shape[0])
        zM, zB = z_cls(M), z_cls(B)
        F = args.w * zM + (1 - args.w) * zB
        if gk is not None:
            with torch.no_grad():
                zM, zB, F = gk(zM), gk(zB), gk(F)
        # 图级分数（top-1% 均值，与评估的图像级分数同口径，且在类别归一化尺度上 → 可跨图比较）
        def img_score(X):
            flat = X[:, 0].reshape(X.shape[0], -1)
            k = max(1, int(flat.shape[1] * 0.01))
            return torch.sort(flat, 1, descending=True)[0][:, :k].mean(1).numpy()
        sM, sB, sF = img_score(zM), img_score(zB), img_score(F)
        # 类别固定色标（三张图共用）
        vals = np.concatenate([zM[:, 0].numpy().ravel(), zB[:, 0].numpy().ravel(), F[:, 0].numpy().ravel()])
        vmin, vmax = float(np.percentile(vals, 100 - args.scale_pct)), float(np.percentile(vals, args.scale_pct))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        out_cls = os.path.join(args.out_dir, cls)
        os.makedirs(out_cls, exist_ok=True)
        idxs = range(M.shape[0]) if args.max <= 0 else range(min(args.max, M.shape[0]))
        rows = []
        for i in idxs:
            g = G[i, 0].numpy() > 0.5
            lab = int(g.any())
            defect = 'good' if lab == 0 else 'defect'
            name = f'{i:04d}_{defect}_sF{sF[i]:.2f}.png'
            fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4), dpi=args.dpi)
            raw = np.asarray(disp_tf(Image.open(paths[i]).convert('RGB')))
            panels = [(axes[0], raw, '输入原图（392×392, resize+center-crop）', None, True),
                      (axes[1], zM[i, 0].numpy(), 'mapping-only (learnable protos)', sM[i], False),
                      (axes[2], zB[i, 0].numpy(), 'bank-only (MemoryBank)', sB[i], False),
                      (axes[3], F[i, 0].numpy(), f'fused  {args.w:.1f}·z(M)+{1-args.w:.1f}·z(B)', sF[i], False)]
            for ax, X, ttl, s, is_raw in panels:
                if is_raw:
                    ax.imshow(X)
                else:
                    ax.imshow(X, cmap=cmap, vmin=vmin, vmax=vmax)
                if lab:
                    # GT 轮廓：白色加粗；先垫一层深色描边，保证落在亮黄/亮红区域时也看得清
                    ax.contour(g, levels=[0.5], colors=['black'], linewidths=4.6)
                    ax.contour(g, levels=[0.5], colors=['white'], linewidths=2.8)
                ax.set_title(ttl + (f'\nimg-score={s:.2f}' if s is not None else ''), fontsize=8.5)
                ax.set_xticks([]); ax.set_yticks([])
            supt = (f'{cls}  #{i:04d}  gt={defect}   (z 归一化按类别：整类 μ/σ；色标类内固定)'
                    if cjk else
                    f'{cls}  #{i:04d}  gt={defect}   (class-level z-normalization; class-fixed color scale)')
            fig.suptitle(supt, fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(out_cls, name)); plt.close(fig)
            rows.append(dict(cls=cls, idx=i, file=name, gt=defect,
                             s_mapping=round(float(sM[i]), 3), s_bank=round(float(sB[i]), 3),
                             s_fused=round(float(sF[i]), 3)))
            total += 1
        with open(os.path.join(out_cls, '_index.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f'[{cls}] 已输出 {len(rows)} 张 → {out_cls}  色标区间=({vmin:.2f}, {vmax:.2f})')
    print(f'\n完成：共 {total} 张三联图，根目录 {args.out_dir}')


if __name__ == '__main__':
    main()
