# -*- coding: utf-8 -*-
"""Recompute only image-level metrics for top-0.5% vs top-1% aggregation."""
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'code_vpt4'))


def load_mod(path):
    spec = importlib.util.spec_from_file_location('top_ratio_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['top_ratio_main'] = m
    spec.loader.exec_module(m)
    return m


def f1_max(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(np.nanmax(2 * p * r / (p + r + 1e-7)))


def z_class(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def box(x, k=25):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2)


SPECS = [
    ('MVTec1', '/home/jjquan/MVTec-AD',
     ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
      'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
      'wood', 'zipper'],
     '.maps_v37_mvtec1shot', '.errmap672_mvtec1'),
    ('MVTec2', '/home/jjquan/MVTec-AD',
     ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
      'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
      'wood', 'zipper'],
     '.maps_simple_nocrop448_mv2', '.errmap672_mv2'),
    ('MVTec4', '/home/jjquan/MVTec-AD',
     ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
      'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
      'wood', 'zipper'],
     '.maps_simple_nocrop448_mv4', '.errmap672_mv4'),
    ('VisA1', '/home/jjquan/experiment/visual2/visa_prepared',
     ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
      'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
     '.maps_v37_visa1shot_promptv2', '.errmap672_visa1_promptv2'),
    ('VisA2', '/home/jjquan/experiment/visual2/visa_prepared',
     ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
      'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
     '.maps_v37_visa2shot_promptv2', '.errmap672_visa2_promptv2'),
    ('VisA4', '/home/jjquan/experiment/visual2/visa_prepared',
     ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
      'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
     '.maps_simple_nocrop448_visa4', '.errmap672_visa4'),
]


def run_spec(text):
    tag, data_path, classes, map_dir, bank_dir = text
    mod = load_mod(os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'))
    tf = T.Compose([T.Resize((448, 448)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    gtf = T.Compose([T.Resize((448, 448)), T.ToTensor()])
    gk = mod.get_gaussian_kernel(5, 4)
    rows = []
    for cls in classes:
        ds = mod.MVTecDataset(os.path.join(data_path, cls), tf, gtf,
                              phase='test', sort_items=False)
        label = np.array([int(ds[i][2]) for i in range(len(ds))])
        M = torch.load(os.path.join(HERE, map_dir, f'{cls}.map'),
                       map_location='cpu', weights_only=False).float()
        B = torch.load(os.path.join(HERE, bank_dir, f'{cls}.map'),
                       map_location='cpu', weights_only=False).float()
        if M.shape[-1] != 448:
            M = F.interpolate(M, size=(448, 448), mode='bilinear',
                              align_corners=True)
        if B.shape[-1] != 448:
            B = F.interpolate(B, size=(448, 448), mode='bilinear',
                              align_corners=True)
        for branch, val in [('mapping', M), ('bank', B)]:
            sc = os.path.join(HERE, map_dir if branch == 'mapping' else bank_dir,
                              f'{cls}.json')
            if os.path.exists(sc):
                keys = json.load(open(sc)).get('keys') or []
                if keys and list(keys) != list(ds.keys):
                    val[:] = val[torch.tensor(
                        [list(keys).index(x) for x in ds.keys])]
        fused = z_class(0.5 * box(z_class(M)) + 0.5 * box(z_class(B)))
        flat = gk(fused).numpy()[:, 0].reshape(len(ds), -1)
        for ratio in (0.005, 0.01):
            kk = max(1, int(flat.shape[1] * ratio))
            score = np.sort(flat, axis=1)[:, -kk:].mean(1)
            rows.append({
                'class': cls, 'ratio': ratio,
                'I-AUROC': roc_auc_score(label, score),
                'I-AP': average_precision_score(label, score),
                'I-F1': f1_max(label, score),
            })
    out = os.path.join(HERE, 'logs', f'{tag}_top_ratio_audit.json')
    json.dump(rows, open(out, 'w', encoding='utf-8'), indent=2)
    for ratio in (0.005, 0.01):
        rr = [r for r in rows if r['ratio'] == ratio]
        print(tag, ratio, {k: float(np.mean([r[k] for r in rr]))
                           for k in ('I-AUROC', 'I-AP', 'I-F1')}, flush=True)
    return tag


def main():
    import multiprocessing as mp
    with mp.Pool(3) as pool:
        print(pool.map(run_spec, SPECS), flush=True)
    print('TOP_RATIO_AUDIT_DONE', flush=True)


if __name__ == '__main__':
    main()
