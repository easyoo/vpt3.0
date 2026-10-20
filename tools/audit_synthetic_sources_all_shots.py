# -*- coding: utf-8 -*-
"""Audit AnoStyler source images for MVTec/VisA 2-shot and 4-shot."""
import glob
import json
import os
from collections import defaultdict

import yaml

from shot_utils import get_shots

HERE = os.path.dirname(os.path.abspath(__file__))

SPECS = [
    {
        'name': 'MVTec2',
        'data': '/home/jjquan/MVTec-AD',
        'shots': 2,
        'configs': os.path.join(HERE, 'anostyler_2shot', 'configs', '*.yaml'),
        'generated': os.path.join(HERE, 'anostyler_2shot', 'genbyAnoStyler_2shot'),
        'classes': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                    'leather', 'metal_nut', 'pill', 'screw', 'tile',
                    'toothbrush', 'transistor', 'wood', 'zipper'],
    },
    {
        'name': 'MVTec4',
        'data': '/home/jjquan/MVTec-AD',
        'shots': 4,
        'configs': '/home/jjquan/AnoStyler/generated_configs/*.yaml',
        'generated': '/home/jjquan/AnoStyler/genbyAnoStyler',
        'classes': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                    'leather', 'metal_nut', 'pill', 'screw', 'tile',
                    'toothbrush', 'transistor', 'wood', 'zipper'],
    },
    {
        'name': 'VisA2',
        'data': '/home/jjquan/experiment/visual2/visa_prepared',
        'shots': 2,
        'configs': os.path.join(HERE, 'anostyler_visa', '2shot_promptv2',
                                'configs', '*.yaml'),
        'generated': os.path.join(HERE, 'anostyler_visa', '2shot_promptv2',
                                  'genbyAnoStyler'),
        'classes': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
                    'pipe_fryum'],
    },
    {
        'name': 'VisA4',
        'data': '/home/jjquan/experiment/visual2/visa_prepared',
        'shots': 4,
        'configs': os.path.join(HERE, 'anostyler_visa', '4shot',
                                'configs', '*.yaml'),
        'generated': os.path.join(HERE, 'anostyler_visa', '4shot',
                                  'genbyAnoStyler'),
        'classes': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
                    'pipe_fryum'],
    },
]


def main():
    report = {}
    for spec in SPECS:
        grouped = defaultdict(list)
        for p in glob.glob(spec['configs']):
            cfg = yaml.safe_load(open(p, encoding='utf-8'))
            grouped[cfg['category']].append({
                'config': p,
                'source': os.path.basename(cfg['normal_image']),
                'prompt': cfg['defect'],
                'stem': os.path.splitext(os.path.basename(cfg['normal_image']))[0],
            })
        items = {}
        for cls in spec['classes']:
            shots = set(get_shots(spec['data'], cls, shots=spec['shots'], seed=1))
            rows = grouped.get(cls, [])
            sources = {x['source'] for x in rows}
            outputs = 0
            missing = []
            for row in rows:
                prompt = row['prompt']
                base = os.path.join(spec['generated'], cls,
                                    prompt, row['stem'])
                ni = len(glob.glob(os.path.join(base, 'image', 'gen_ano_*.jpg')))
                nm = len(glob.glob(os.path.join(base, 'mask', 'gen_mask_*.jpg')))
                outputs += min(ni, nm)
                if ni != 35 or nm != 35:
                    missing.append({'stem': row['stem'], 'images': ni, 'masks': nm})
            items[cls] = {
                'shots': sorted(shots),
                'synthetic_sources': sorted(sources),
                'source_configs': len(rows),
                'overlap': len(shots & sources),
                'outside_shots': sorted(sources - shots),
                'missing_shots': sorted(shots - sources),
                'generated_pairs': outputs,
                'incomplete_sources': missing,
                'generic_prompts': sum(
                    str(x['prompt']).strip().lower() in
                    {'', 'defect', 'defect0', 'anomaly'} for x in rows),
            }
        report[spec['name']] = {
            'shots': spec['shots'],
            'classes': items,
        }
    out = os.path.join(HERE, 'logs', 'synthetic_sources_all_shots.json')
    json.dump(report, open(out, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    for name, ds in report.items():
        bad = {c: v for c, v in ds['classes'].items()
               if v['overlap'] != ds['shots'] or v['incomplete_sources']}
        print(name, 'bad_classes', bad)
    print(f'SYNTHETIC_SOURCE_AUDIT -> {out}')


if __name__ == '__main__':
    main()
