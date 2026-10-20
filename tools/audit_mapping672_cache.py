# -*- coding: utf-8 -*-
"""Audit mapping/672-bank caches for shot identity, order, shape and config.

The two branches historically used different Dataset classes:
  * mapping maps created on 2026-10-13 use vpt3.7's legacy (unsorted) order;
  * cache_bank_nocrop.py uses vpt2's legacy (unsorted) order;
  * newly generated mapping maps use vpt4 sorted order and carry a sidecar.

The fusion script uses legacy order as the common coordinate system and reorders
mapping maps by sidecar when needed. This audit verifies every cache against that
contract instead of comparing only class counts.
"""
import argparse
import glob
import hashlib
import json
import os
import re

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper',
]
VISA_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
    'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
]

SPECS = [
    {
        'name': 'MVTec1',
        'shots': 1,
        'expected_classes': MVTEC_CLASSES,
        'data_path': '/home/jjquan/MVTec-AD',
        'mapping_globs': ['logs/mvtec1_final_eval/map_*.log'],
        'bank_globs': ['logs/mvtec1_final_eval/bank672_*.log'],
        'expected_aug': 22,
        'map_dir': '.maps_v37_mvtec1shot',
        'bank_dir': '.errmap672_mvtec1',
        'mapping_order': 'sorted',
        'bank_order': 'legacy',
    },
    {
        'name': 'MVTec2',
        'shots': 2,
        'expected_classes': MVTEC_CLASSES,
        'data_path': '/home/jjquan/MVTec-AD',
        'mapping_globs': ['logs/simple_mv2_*.log'],
        'bank_globs': ['logs/mv672_2shot_*.log'],
        'expected_aug': 44,
        'map_dir': '.maps_simple_nocrop448_mv2',
        'bank_dir': '.errmap672_mv2',
        'mapping_order': 'legacy',
        'bank_order': 'legacy',
    },
    {
        'name': 'MVTec4',
        'shots': 4,
        'expected_classes': MVTEC_CLASSES,
        'data_path': '/home/jjquan/MVTec-AD',
        'mapping_globs': ['logs/simple_mv4_*.log'],
        'bank_globs': ['logs/m672_bank_*.log'],
        'expected_aug': 88,
        'map_dir': '.maps_simple_nocrop448_mv4',
        'bank_dir': '.errmap672_mv4',
        'mapping_order': 'legacy',
        'bank_order': 'legacy',
    },
    {
        'name': 'VisA1',
        'shots': 1,
        'expected_classes': VISA_CLASSES,
        'data_path': '/home/jjquan/experiment/visual2/visa_prepared',
        'mapping_globs': ['logs/visa1_final_eval/map_*.log'],
        'bank_globs': ['logs/visa1_final_eval/bank672_*.log'],
        'expected_aug': 22,
        'map_dir': '.maps_v37_visa1shot_promptv2',
        'bank_dir': '.errmap672_visa1_promptv2',
        'mapping_order': 'sorted',
        'bank_order': 'legacy',
    },
    {
        'name': 'VisA2',
        'shots': 2,
        'expected_classes': VISA_CLASSES,
        'data_path': '/home/jjquan/experiment/visual2/visa_prepared',
        'mapping_globs': ['logs/visa2_promptv2_final_eval/map_*.log'],
        'bank_globs': ['logs/visa2_promptv2_final_eval/bank672_*.log'],
        'expected_aug': 44,
        'map_dir': '.maps_v37_visa2shot_promptv2',
        'bank_dir': '.errmap672_visa2_promptv2',
        'mapping_order': 'sorted',
        'bank_order': 'legacy',
    },
    {
        'name': 'VisA4',
        'shots': 4,
        'expected_classes': VISA_CLASSES,
        'data_path': '/home/jjquan/experiment/visual2/visa_prepared',
        'mapping_globs': ['logs/simple_visa4_*.log'],
        'bank_globs': ['logs/v672_*.log', 'logs/visa672_*.log'],
        'expected_aug': 88,
        'map_dir': '.maps_simple_nocrop448_visa4',
        'bank_dir': '.errmap672_visa4',
        'mapping_order': 'legacy',
        'bank_order': 'legacy',
    },
]


def digest(keys):
    return hashlib.sha1('\n'.join(keys).encode()).hexdigest()[:12]


def dataset_keys(data_path, cls, sort_items):
    root = os.path.join(data_path, cls)
    defect_types = os.listdir(os.path.join(root, 'test'))
    if sort_items:
        defect_types = sorted(defect_types)
    keys = []
    for defect in defect_types:
        paths = []
        for ext in ('*.png', '*.JPG', '*.bmp'):
            paths += glob.glob(os.path.join(root, 'test', defect, ext))
        if sort_items or defect != 'good':
            paths.sort()
        keys.extend(f'{defect}/{os.path.basename(p)}' for p in paths)
    return keys


def shot_records(paths, shots):
    pat = re.compile(
        r'\[shot_utils\] (\S+) shot' + str(shots) +
        r' hash=([0-9a-f]+).*?picks=(\[[^\n]+\])')
    out = {}
    for path in paths:
        text = open(path, errors='ignore').read()
        for cls, h, picks in pat.findall(text):
            out.setdefault(cls, {'hash': h, 'picks': picks})
    return out


def mapping_config(paths):
    pat = re.compile(
        r'\[v3\] (\S+)\s+raw\(d1\^0\.5\*q512\) '
        r'learn\(dlmin\^0\.7\*bq0\) w=0\.6')
    out = set()
    global_off = 0
    for path in paths:
        text = open(path, errors='ignore').read()
        out.update(pat.findall(text))
        global_off += text.count(
            'learn 侧全局项已关闭：s_learn = dlmin')
    return sorted(out), global_off


def bank_aug_counts(paths, shots):
    pat = re.compile(
        r'\[bank-nocrop672\] (\S+): ' + str(shots) +
        r' shot → 增强后 \((\d+), 3, 672, 672\)')
    out = {}
    for path in paths:
        text = open(path, errors='ignore').read()
        for cls, count in pat.findall(text):
            out.setdefault(cls, int(count))
    return out


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def classify_order(keys, legacy, sorted_keys):
    if not keys:
        return 'unknown(no keys)'
    if list(keys) == legacy:
        return 'legacy'
    if list(keys) == sorted_keys:
        return 'sorted'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='',
                    help='逗号分隔的数据集名，例如 MVTec2,VisA2；空=全部')
    args = ap.parse_args()
    wanted = {x.strip() for x in args.only.split(',') if x.strip()}
    specs = [x for x in SPECS if not wanted or x['name'] in wanted]
    if wanted - {x['name'] for x in specs}:
        raise SystemExit(f'unknown datasets: {sorted(wanted - {x["name"] for x in specs})}')
    report = []
    ok = True
    for spec in specs:
        map_logs = []
        for g in spec['mapping_globs']:
            map_logs.extend(glob.glob(os.path.join(HERE, g)))
        bank_logs = []
        for g in spec['bank_globs']:
            bank_logs.extend(glob.glob(os.path.join(HERE, g)))
        maps = shot_records(map_logs, spec['shots'])
        banks = shot_records(bank_logs, spec['shots'])
        cfg_classes, global_off = mapping_config(map_logs)
        aug = bank_aug_counts(bank_logs, spec['shots'])

        classes = spec['expected_classes']
        missing_map = [c for c in classes if c not in maps]
        missing_bank = [c for c in classes if c not in banks]
        extra_map = sorted(set(maps) - set(classes))
        extra_bank = sorted(set(banks) - set(classes))
        shot_mismatch = [c for c in classes if maps.get(c) != banks.get(c)]
        missing_cfg = [c for c in classes if c not in set(cfg_classes)]
        bad_aug = [c for c in classes if aug.get(c) != spec['expected_aug']]

        shapes = {}
        orders = {}
        bad_shape = []
        bad_sidecar = []
        for cls in classes:
            mpath = os.path.join(HERE, spec['map_dir'], cls + '.map')
            bpath = os.path.join(HERE, spec['bank_dir'], cls + '.map')
            if not (os.path.exists(mpath) and os.path.exists(bpath)):
                bad_shape.append({'class': cls, 'reason': 'missing_map'})
                continue
            m = torch.load(mpath, map_location='cpu', weights_only=False)
            b = torch.load(bpath, map_location='cpu', weights_only=False)
            if (not torch.isfinite(m).all().item() or
                    not torch.isfinite(b).all().item()):
                bad_shape.append({'class': cls, 'reason': 'non_finite'})
            if m.shape[0] != b.shape[0] or m.shape[-1] != 448 or b.shape[-1] != 672:
                bad_shape.append({
                    'class': cls, 'reason': 'shape',
                    'mapping': list(m.shape), 'bank672': list(b.shape),
                })

            legacy = dataset_keys(spec['data_path'], cls, sort_items=False)
            sorted_keys = dataset_keys(spec['data_path'], cls, sort_items=True)
            mjson = load_json(os.path.join(HERE, spec['map_dir'], cls + '.json'))
            bjson = load_json(os.path.join(HERE, spec['bank_dir'], cls + '.json'))
            map_keys = (mjson or {}).get('keys') or []
            bank_keys = (bjson or {}).get('keys') or []
            mo = (classify_order(map_keys, legacy, sorted_keys)
                  if map_keys else spec['mapping_order'] + '(code-provenance)')
            bo = (classify_order(bank_keys, legacy, sorted_keys)
                  if bank_keys else spec['bank_order'] + '(code-provenance)')
            orders[cls] = {
                'n': len(legacy),
                'legacy_hash': digest(legacy),
                'sorted_hash': digest(sorted_keys),
                'mapping_order': mo,
                'bank_order': bo,
                'mapping_sidecar': bool(mjson),
                'bank_sidecar': bool(bjson),
                'fusion_reorders_mapping': mo.startswith('sorted'),
                'fusion_reorders_bank': bo.startswith('sorted'),
            }
            if spec['mapping_order'] == 'sorted' and mo != 'sorted':
                bad_sidecar.append({'class': cls, 'branch': 'mapping', 'order': mo})
            if spec['bank_order'] == 'legacy' and not bo.startswith('legacy'):
                bad_sidecar.append({'class': cls, 'branch': 'bank', 'order': bo})
            if mjson and int(mjson.get('n', -1)) != int(m.shape[0]):
                bad_sidecar.append({'class': cls, 'branch': 'mapping', 'reason': 'n'})
            if bjson and int(bjson.get('n', -1)) != int(b.shape[0]):
                bad_sidecar.append({'class': cls, 'branch': 'bank', 'reason': 'n'})
            shapes[cls] = {'mapping': list(m.shape), 'bank672': list(b.shape)}
            del m, b

        item_ok = (
            not missing_map and not missing_bank and
            not extra_map and not extra_bank and
            not shot_mismatch and not missing_cfg and
            not bad_aug and not bad_shape and not bad_sidecar and
            global_off == len(classes)
        )
        item = {
            'name': spec['name'],
            'shots': spec['shots'],
            'expected_classes': len(classes),
            'classes_mapping': len(maps),
            'classes_bank': len(banks),
            'missing_mapping': missing_map,
            'missing_bank': missing_bank,
            'extra_mapping': extra_map,
            'extra_bank': extra_bank,
            'shot_mismatches': shot_mismatch,
            'mapping_formula_missing': missing_cfg,
            'mapping_global_off_count': global_off,
            'augmentation_mismatches': bad_aug,
            'expected_augmented_views': spec['expected_aug'],
            'shape_mismatches': bad_shape,
            'sidecar_order_mismatches': bad_sidecar,
            'orders': orders,
            'shapes': shapes,
            'ok': item_ok,
        }
        ok &= item_ok
        report.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    out = os.path.join(HERE, 'logs', 'mapping672_cache_audit.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'ok': ok, 'datasets': report}, fh, indent=2, ensure_ascii=False)
    if not ok:
        raise SystemExit('CACHE_AUDIT_FAILED')
    print(f'CACHE_AUDIT_OK -> {out}')


if __name__ == '__main__':
    main()
