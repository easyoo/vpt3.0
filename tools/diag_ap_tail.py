# -*- coding: utf-8 -*-
# [诊断] 量化"正常像素的高分尾部"对 P-AP / P-F1 的贡献上限。
# 做法：取已缓存的 mapping 异常图，把**全局分数最高的 q 比例的正常像素**强制压到最低分
# （oracle 完美尾部抑制，不是一个可用方法，只用来判断"尾部抑制"这条训练方向值不值得投入），
# 再按官方 adeval 口径重算指标。
# 用法: python diag_ap_tail.py [class...]
import importlib.util
import os
import sys

import numpy as np
import torch
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_mod(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location('audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['audit_main'] = m
    spec.loader.exec_module(m)
    return m


NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']


def main():
    cls_list = sys.argv[1:] or ['transistor', 'screw', 'capsule', 'metal_nut', 'wood']
    mod = load_mod(os.path.join(HERE, 'code_vpt5', 'main-fewshot-vpt4.py'))
    gk = mod.get_gaussian_kernel(kernel_size=5, sigma=4)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize(mean, std)])
    gtf = T.Compose([T.Resize((448, 448)), T.ToTensor()])
    qs = [0.0, 0.001, 0.005, 0.01, 0.02]

    res = {}
    print('%-11s %-7s' % ('class', 'q') + ''.join('%9s' % n for n in NAMES))
    for cls in cls_list:
        mp = os.path.join(HERE, '.maps_nocrop448_m4m6r3_2shot', f'{cls}.map')
        am = torch.load(mp, weights_only=False).float()          # [N,1,448,448]（未平滑）
        ds = mod.MVTecDataset(root=os.path.join('/home/jjquan/MVTec-AD', cls),
                              transform=tf, gt_transform=gtf, phase='test', sort_items=False)
        gts, labs = [], []
        for _, gt, label, _ in ds:
            gts.append(gt); labs.append(label)
        g = torch.cat(gts, 0)
        if g.dim() == 4:                       # [N,1,H,W]（DataLoader 之外可能是 3D）
            g = torch.max(g, dim=1)[0]
        gt_px = (g > 0.5).float().numpy()      # [N,H,W]
        gt_sp = np.array(labs)
        am_s = gk(am)
        for q in qs:
            a = am_s.clone()
            if q > 0:
                nrm = a[:, 0]                        # [N,H,W] 视图
                normal_mask = torch.from_numpy(gt_px < 0.5)
                v = nrm[normal_mask]
                if v.numel() > 0:
                    k = max(1, int(v.numel() * q))
                    thr = torch.topk(v, k).values.min()
                    sel = normal_mask & (nrm >= thr)
                    nrm[sel] = -1.0
            pr_px = a[:, 0].numpy()
            flat = a.flatten(1)
            kk = max(1, int(flat.shape[1] * 0.01))
            pr_sp = torch.sort(flat, dim=1, descending=True)[0][:, :kk].mean(dim=1).numpy()
            r = list(mod.ader_evaluator(pr_px, pr_sp, gt_px, gt_sp))
            print('%-11s %-7s' % (cls, q) + ''.join('%9.2f' % (x * 100) for x in r))
            res[(cls, q)] = np.array(r)
        print()
    print('=== %d 类平均（Δ 相对 q=0）===' % len(cls_list))
    print('%-7s' % 'q' + ''.join('%18s' % n for n in NAMES) + '%10s' % '7项平均')
    for q in qs:
        avg = np.mean([res[(c, q)] for c in cls_list], axis=0)
        base = np.mean([res[(c, 0.0)] for c in cls_list], axis=0)
        cells = ''
        for i in range(7):
            cells += '%10.2f(%+5.1f)' % (avg[i] * 100, (avg[i] - base[i]) * 100)
        print('%-7s' % q + cells + '%10.2f' % (avg.mean() * 100))


if __name__ == '__main__':
    main()
