# -*- coding: utf-8 -*-
"""Aggregate and validate all six mapping + 672-bank evaluation CSVs."""
import csv
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1',
           'P-AUPRO']
MVTEC_CLASSES = {
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper',
}
VISA_CLASSES = {
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
    'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
}
SPECS = [
    ('MVTec-AD', 1, 'logs/mvtec1_mapping672_box_eq2_25.csv', 15),
    ('MVTec-AD', 2, 'logs/mapping672_mv2_box_eq2_25.csv', 15),
    ('MVTec-AD', 4, 'logs/mapping672_mv4_box_eq2_25.csv', 15),
    ('VisA', 1, 'logs/visa1_mapping672_box_eq2_25.csv', 12),
    ('VisA', 2, 'logs/visa2_promptv2_mapping672_box_eq2_25.csv', 12),
    ('VisA', 4, 'logs/mapping672_visa4_box_eq2_25.csv', 12),
]

SUBAD = {
    ('MVTec-AD', 1): {'I-AUROC': 97.1, 'I-AP': 98.6,
                      'P-AUROC': 97.5, 'P-AUPRO': 94.5},
    ('MVTec-AD', 2): {'I-AUROC': 97.5, 'I-AP': 98.8,
                      'P-AUROC': 97.8, 'P-AUPRO': 94.9},
    ('MVTec-AD', 4): {'I-AUROC': 98.0, 'I-AP': 99.0,
                      'P-AUROC': 97.9, 'P-AUPRO': 95.1},
    ('VisA', 1): {'I-AUROC': 93.2, 'I-AP': 93.0,
                  'P-AUROC': 98.2, 'P-AUPRO': 95.5},
    ('VisA', 2): {'I-AUROC': 93.8, 'I-AP': 93.4,
                  'P-AUROC': 98.3, 'P-AUPRO': 95.7},
    ('VisA', 4): {'I-AUROC': 94.7, 'I-AP': 94.3,
                  'P-AUROC': 98.4, 'P-AUPRO': 96.0},
}


def main():
    rows = []
    ok = True
    for dataset, shots, rel, expected_n in SPECS:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            rows.append({'dataset': dataset, 'shots': shots, 'status': 'missing',
                         'expected_classes': expected_n})
            ok = False
            continue
        data = list(csv.DictReader(open(path)))
        complete = len(data) == expected_n
        expected_set = MVTEC_CLASSES if dataset == 'MVTec-AD' else VISA_CLASSES
        classes = [r.get('class', '') for r in data]
        issues = []
        if len(classes) != len(set(classes)):
            issues.append('duplicate_class')
        if set(classes) != expected_set:
            issues.append('class_set')
        for row_index, row in enumerate(data):
            if row.get('fusion') != 'box_eq2_25':
                issues.append(f'fusion_rule@{row_index}')
            for key in METRICS:
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError):
                    issues.append(f'{key}@{row_index}')
                    continue
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    issues.append(f'{key}_range@{row_index}')
            vals = [float(row[k]) for k in METRICS]
            expected_mean = sum(vals) / len(vals)
            if not math.isclose(float(row['mean']), expected_mean,
                                rel_tol=0.0, abs_tol=1e-10):
                issues.append(f'mean@{row_index}')
        complete &= not issues
        item = {
            'dataset': dataset,
            'shots': shots,
            'status': 'complete' if complete else 'incomplete',
            'classes': len(data),
            'expected_classes': expected_n,
            'csv': rel,
            'issues': sorted(set(issues)),
        }
        if data:
            for key in METRICS + ['mean']:
                item[key] = statistics.mean(float(r[key]) for r in data) * 100
        rows.append(item)
        ok &= complete

    out_json = os.path.join(HERE, 'logs', 'mapping672_all_results.json')
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump({'ok': ok, 'results': rows}, fh, indent=2)

    lines = [
        '# Mapping + 672 MemoryBank 两路融合',
        '',
        '```text',
        'z(0.5*box25(z(M)) + 0.5*box25(z(B672->448)))',
        'Gaussian(k=5,sigma=4), image score = mean(top 0.5% pixels)',
        '```',
        '',
        '| 数据集 | Shot | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO | 7项均值 | 状态 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in rows:
        if r['status'] == 'missing':
            vals = ['-'] * 8
        else:
            vals = [f"{r[k]:.4f}" for k in METRICS + ['mean']]
        lines.append(
            f"| {r['dataset']} | {r['shots']} | " +
            ' | '.join(vals) +
            f" | {r['status']} ({r.get('classes', 0)}/{r['expected_classes']}) |"
        )
    lines += [
        '',
        '## 与 SUBAD 1/2/4-shot 对比',
        '',
        "SUBAD 数字来自 CVPR'26 论文 Table 1，取 5 个随机种子均值；"
        '其 `AUPR` 对应本表的 `I-AP`，`PRO` 对应本表的 `P-AUPRO`。'
        '论文未报告 `I-F1 / P-AP / P-F1`，因此用 `-` 占位。'
        '“4项均值”只平均双方都有报告的四项指标。',
        '',
        '| 数据集 | Shot | 方法 | I-AUROC | I-AP(AUPR) | I-F1 | '
        'P-AUROC | P-AP | P-F1 | P-AUPRO(PRO) | 4项均值 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    shared = ['I-AUROC', 'I-AP', 'P-AUROC', 'P-AUPRO']
    for r in rows:
        key = (r['dataset'], r['shots'])
        ours_mean = statistics.mean(r[k] for k in shared)
        sub = SUBAD[key]
        sub_mean = statistics.mean(sub[k] for k in shared)
        lines.append(
            f"| {r['dataset']} | {r['shots']} | 两路融合 | "
            f"{r['I-AUROC']:.4f} | {r['I-AP']:.4f} | {r['I-F1']:.4f} | "
            f"{r['P-AUROC']:.4f} | {r['P-AP']:.4f} | {r['P-F1']:.4f} | "
            f"{r['P-AUPRO']:.4f} | {ours_mean:.4f} |"
        )
        lines.append(
            f"| {r['dataset']} | {r['shots']} | SUBAD | "
            f"{sub['I-AUROC']:.1f} | {sub['I-AP']:.1f} | - | "
            f"{sub['P-AUROC']:.1f} | - | - | {sub['P-AUPRO']:.1f} | "
            f"{sub_mean:.4f} |"
        )
    lines += [
        '',
        '相对 SUBAD 的四项均值：',
        '',
        '| 数据集 | Shot | 两路融合 − SUBAD |',
        '|---|---:|---:|',
    ]
    for r in rows:
        key = (r['dataset'], r['shots'])
        ours_mean = statistics.mean(r[k] for k in shared)
        sub_mean = statistics.mean(SUBAD[key][k] for k in shared)
        lines.append(
            f"| {r['dataset']} | {r['shots']} | {ours_mean - sub_mean:+.4f} |")
    lines += ['', '缓存一致性结果见 `logs/mapping672_cache_audit.json`。', '']
    out_md = os.path.join(HERE, 'MAPPING672_ALL_RESULTS.md')
    with open(out_md, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('\n'.join(lines))
    if not ok:
        raise SystemExit('MAPPING672_RESULTS_INCOMPLETE')


if __name__ == '__main__':
    main()
