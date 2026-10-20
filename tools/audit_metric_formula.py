# -*- coding: utf-8 -*-
"""Independent metric recomputation on one representative class per setting."""
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


def load_mod(path):
    spec = importlib.util.spec_from_file_location('metric_audit_main', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['metric_audit_main'] = m
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
    ('MVTec1', 'MVTec-AD', 1, '/home/jjquan/MVTec-AD', 'bottle',
     '.maps_v37_mvtec1shot', '.errmap672_mvtec1',
     'logs/mvtec1_mapping672_box_eq2_25.csv'),
    ('MVTec2', 'MVTec-AD', 2, '/home/jjquan/MVTec-AD', 'bottle',
     '.maps_simple_nocrop448_mv2', '.errmap672_mv2',
     'logs/mapping672_mv2_box_eq2_25.csv'),
    ('MVTec4', 'MVTec-AD', 4, '/home/jjquan/MVTec-AD', 'bottle',
     '.maps_simple_nocrop448_mv4', '.errmap672_mv4',
     'logs/mapping672_mv4_box_eq2_25.csv'),
    ('VisA1', 'VisA', 1, '/home/jjquan/experiment/visual2/visa_prepared', 'capsules',
     '.maps_v37_visa1shot_promptv2', '.errmap672_visa1_promptv2',
     'logs/visa1_mapping672_box_eq2_25.csv'),
    ('VisA2', 'VisA', 2, '/home/jjquan/experiment/visual2/visa_prepared', 'capsules',
     '.maps_v37_visa2shot_promptv2', '.errmap672_visa2_promptv2',
     'logs/visa2_promptv2_mapping672_box_eq2_25.csv'),
    ('VisA4', 'VisA', 4, '/home/jjquan/experiment/visual2/visa_prepared', 'capsules',
     '.maps_simple_nocrop448_visa4', '.errmap672_visa4',
     'logs/mapping672_visa4_box_eq2_25.csv'),
]


def main():
    code_dir = os.path.join(HERE, 'code_vpt4')
    sys.path.insert(0, code_dir)
    mod = load_mod(os.path.join(code_dir, 'main-fewshot-vpt4.py'))
    tf = T.Compose([T.Resize((448, 448)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    gtf = T.Compose([T.Resize((448, 448)), T.ToTensor()])
    gk = mod.get_gaussian_kernel(5, 4)
    report = {}
    for tag, dataset, shots, data_path, cls, map_dir, bank_dir, csv_path in SPECS:
        ds = mod.MVTecDataset(os.path.join(data_path, cls), tf, gtf,
                              phase='test', sort_items=False)
        gt = torch.stack([ds[i][1] for i in range(len(ds))], 0).float()
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
        fused = z_class(0.5 * box(z_class(M), 25) + 0.5 * box(z_class(B), 25))
        amap = gk(fused).numpy()[:, 0]
        flat = amap.reshape(len(ds), -1)
        kk = max(1, int(flat.shape[1] * 0.005))
        image_score = np.sort(flat, axis=1)[:, -kk:].mean(1)
        gt_px = gt[:, 0].numpy() > 0.5
        independent = {
            'I-AUROC': roc_auc_score(label, image_score),
            'I-AP': average_precision_score(label, image_score),
            'I-F1': f1_max(label, image_score),
            'P-AUROC': roc_auc_score(gt_px.ravel(), flat.ravel()),
            'P-AP': average_precision_score(gt_px.ravel(), flat.ravel()),
            'P-F1': f1_max(gt_px.ravel(), flat.ravel()),
        }
        csv_row = next(x for x in csv.DictReader(open(os.path.join(HERE, csv_path)))
                       if x['class'] == cls)
        diffs = {
            k: abs(independent[k] - float(csv_row[k]))
            for k in independent
        }
        report[tag] = {
            'dataset': dataset,
            'shots': shots,
            'class': cls,
            'independent': independent,
            'csv': {k: float(csv_row[k]) for k in independent},
            'abs_diff': diffs,
            'max_abs_diff': max(diffs.values()),
        }
        print(tag, report[tag], flush=True)
    out = os.path.join(HERE, 'logs', 'metric_formula_audit.json')
    json.dump(report, open(out, 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)
    print(f'METRIC_FORMULA_AUDIT_OK -> {out}')


if __name__ == '__main__':
    main()
