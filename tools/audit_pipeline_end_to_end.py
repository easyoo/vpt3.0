# -*- coding: utf-8 -*-
"""End-to-end provenance audit for synthetic data, shots, training and caches."""
import glob
import hashlib
import json
import os
import re
from collections import Counter

import numpy as np
import yaml
from PIL import Image

from shot_utils import get_shots, shots_hash

HERE = os.path.dirname(os.path.abspath(__file__))
MVTEC = '/home/jjquan/MVTec-AD'
VISA = '/home/jjquan/experiment/visual2/visa_prepared'
MVTEC1_ROOT = ('/home/jjquan/experiment/visual2/'
               'anomaly_samples_mvtec_1shot_promptv2/mvtec')
VISA1_ROOT = ('/home/jjquan/experiment/visual2/'
              'anomaly_samples_visa_1shot_promptv2/visa')
VISA2_RAW = ('/home/jjquan/experiment/visual2/'
             'anomaly_samples_visa_2shot_promptv2/visa')
MVTEC_MANIFEST = os.path.join(HERE, 'anostyler_mvtec', '1shot_promptv2',
                              'configs_manifest.json')
VISA_PROMPTS = os.path.join(HERE, 'anostyler_visa', '2shot_promptv2',
                            'prompts.json')
CLASSES = {
    MVTEC: ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper'],
    VISA: ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
           'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
           'pipe_fryum'],
}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_mask_stats(img_path, mask_path):
    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        same_size = im.size == mk.size
        m = np.asarray(mk.convert('L'))
        positive = float((m > 0).mean())
        return {'same_size': same_size, 'image_size': list(im.size),
                'mask_size': list(mk.size), 'mask_positive_fraction': positive}


def audit_synth_dir(root, expected_classes, source_bases):
    if isinstance(source_bases, str):
        source_bases = [source_bases]
    source_bases = [os.path.realpath(x) for x in source_bases]
    images = sorted(glob.glob(os.path.join(root, '*', '*', 'image', '*.jpg')))
    masks = sorted(glob.glob(os.path.join(root, '*', '*', 'mask', '*.jpg')))
    im_hashes = [md5(x) for x in images]
    mk_hashes = [md5(x) for x in masks]
    stats = [image_mask_stats(i, m) for i, m in zip(images, masks)]
    targets = []
    bad_targets = []
    for p in glob.glob(os.path.join(root, '*', '*', '*', '*')):
        if os.path.islink(p):
            real = os.path.realpath(p)
            targets.append(real)
            if not any(real.startswith(base) for base in source_bases):
                bad_targets.append({'link': p, 'target': real})
    return {
        'images': len(images),
        'masks': len(masks),
        'unique_image_hashes': len(set(im_hashes)),
        'unique_mask_hashes': len(set(mk_hashes)),
        'size_mismatches': sum(not x['same_size'] for x in stats),
        'empty_masks': sum(x['mask_positive_fraction'] <= 0 for x in stats),
        'full_masks': sum(x['mask_positive_fraction'] >= 0.999 for x in stats),
        'mask_positive_fraction_min': min(
            (x['mask_positive_fraction'] for x in stats), default=None),
        'mask_positive_fraction_mean': float(np.mean(
            [x['mask_positive_fraction'] for x in stats])) if stats else None,
        'mask_positive_fraction_max': max(
            (x['mask_positive_fraction'] for x in stats), default=None),
        'symlinks': len(targets),
        'unique_link_targets': len(set(targets)),
        'bad_symlink_targets': bad_targets,
        'expected_images': 35 * len(expected_classes),
    }


def audit_mvtec_synth():
    manifest = json.load(open(MVTEC_MANIFEST, encoding='utf-8'))
    items = {x['cls']: x for x in manifest['items']}
    checks = {}
    test_hashes = set()
    gt_hashes = set()
    for cls in CLASSES[MVTEC]:
        for p in glob.glob(os.path.join(MVTEC, cls, 'test', '*', '*')):
            if os.path.isfile(p):
                test_hashes.add(md5(p))
        for p in glob.glob(os.path.join(MVTEC, cls, 'ground_truth', '*', '*')):
            if os.path.isfile(p):
                gt_hashes.add(md5(p))
    raw_root = manifest['save_path']
    for cls in CLASSES[MVTEC]:
        item = items[cls]
        picks = get_shots(MVTEC, cls, shots=1, seed=1)
        stem = os.path.splitext(picks[0])[0]
        cfg = yaml.safe_load(open(item['config'], encoding='utf-8'))
        src = os.path.join(MVTEC, cls, 'train', 'good', picks[0])
        raw_dir = os.path.join(raw_root, cls, item['prompt'], stem)
        imgs = sorted(glob.glob(os.path.join(raw_dir, 'image', 'gen_ano_*.jpg')))
        msks = sorted(glob.glob(os.path.join(raw_dir, 'mask', 'gen_mask_*.jpg')))
        gen_hashes = {md5(p) for p in imgs}
        checks[cls] = {
            'shots': picks,
            'shot_hash_manifest': item['shot_hash'],
            'shot_hash_recomputed': shots_hash(picks),
            'stem_match': stem == item['stem'],
            'normal_image_match': os.path.abspath(cfg['normal_image']) == os.path.abspath(src),
            'prompt_match': cfg['defect'] == item['prompt'],
            'prompt_specific': str(cfg['defect']).strip().lower() not in
                               {'', 'defect', 'anomaly'},
            'images': len(imgs),
            'masks': len(msks),
            'pairs': sum(os.path.exists(os.path.join(
                raw_dir, 'mask', os.path.basename(x).replace(
                    'gen_ano_', 'gen_mask_'))) for x in imgs),
            'unique_image_hashes': len(gen_hashes),
            'source_equals_generated': md5(src) in gen_hashes,
            'test_hash_overlap': len(gen_hashes & test_hashes),
            'gt_hash_overlap': len({md5(p) for p in msks} & gt_hashes),
            'pair_stats': image_mask_stats(imgs[0], msks[0]) if imgs and msks else None,
        }
    root_check = audit_synth_dir(MVTEC1_ROOT, CLASSES[MVTEC], raw_root)
    return {'manifest_items': len(items), 'classes': checks,
            'root': root_check}


def audit_visa_synth():
    prompts = json.load(open(VISA_PROMPTS, encoding='utf-8'))
    checks = {}
    test_hashes = set()
    gt_hashes = set()
    for cls in CLASSES[VISA]:
        for p in glob.glob(os.path.join(VISA, cls, 'test', '*', '*')):
            if os.path.isfile(p):
                test_hashes.add(md5(p))
        for p in glob.glob(os.path.join(VISA, cls, 'ground_truth', '*', '*')):
            if os.path.isfile(p):
                gt_hashes.add(md5(p))
    for cls in CLASSES[VISA]:
        picks = get_shots(VISA, cls, shots=1, seed=1)
        stem = os.path.splitext(os.path.basename(picks[0]))[0]
        raw_dir = os.path.join(VISA2_RAW, cls, f'src{stem}')
        imgs = sorted(glob.glob(os.path.join(raw_dir, 'image', f'{stem}_gen_ano_*.jpg')))
        msks = sorted(glob.glob(os.path.join(raw_dir, 'mask', f'{stem}_gen_mask_*.jpg')))
        gen_hashes = {md5(p) for p in imgs}
        prompt = prompts[cls]['prompt']
        two_shot_basenames = {
            os.path.basename(x) for x in prompts[cls]['files']
        }
        checks[cls] = {
            'shots': picks,
            'stem': stem,
            'source_in_2shot': picks[0] in two_shot_basenames,
            'prompt': prompt,
            'prompt_specific': str(prompt).strip().lower() not in
                               {'', 'defect', 'anomaly'},
            'images': len(imgs),
            'masks': len(msks),
            'pairs': sum(os.path.exists(os.path.join(
                raw_dir, 'mask', os.path.basename(x).replace(
                    'gen_ano_', 'gen_mask_'))) for x in imgs),
            'unique_image_hashes': len(gen_hashes),
            'test_hash_overlap': len(gen_hashes & test_hashes),
            'gt_hash_overlap': len({md5(p) for p in msks} & gt_hashes),
            'pair_stats': image_mask_stats(imgs[0], msks[0]) if imgs and msks else None,
        }
    root_check = audit_synth_dir(
        VISA1_ROOT, CLASSES[VISA],
        [VISA2_RAW, os.path.join(HERE, 'anostyler_visa', '2shot_promptv2',
                                'genbyAnoStyler')])
    return {'classes': checks, 'root': root_check}


def audit_shots():
    out = {}
    for data, classes in CLASSES.items():
        item = {}
        for cls in classes:
            lists = {}
            for shots in (1, 2, 4):
                picks = get_shots(data, cls, shots=shots, seed=1)
                lists[str(shots)] = {
                    'picks': picks,
                    'hash': shots_hash(picks),
                    'unique': len(set(picks)) == shots,
                    'all_train_good': all(os.path.isfile(os.path.join(
                        data, cls, 'train', 'good', x)) for x in picks),
                }
            lists['nested_1_in_2'] = set(lists['1']['picks']).issubset(lists['2']['picks'])
            lists['nested_2_in_4'] = set(lists['2']['picks']).issubset(lists['4']['picks'])
            item[cls] = lists
        out[data] = item
    return out


def audit_training_logs():
    specs = [
        ('MVTec1', 'logs/train_mvtec1shot_vpt37_mvtec1shot_promptv2fix_seed1_*.log',
         'results/vpt4/vpt37_mvtec1shot_promptv2fix_seed1', CLASSES[MVTEC]),
        ('VisA1', 'logs/train_visa_vpt37_visa1shot_promptv2_*.log',
         'results/visa/vpt37_visa1shot_promptv2', CLASSES[VISA]),
    ]
    out = {}
    for name, pattern, ckpt_dir, classes in specs:
        items = {}
        for cls in classes:
            path = os.path.join(HERE, pattern.replace('*', cls))
            text = open(path, errors='ignore').read() if os.path.exists(path) else ''
            final = os.path.join(HERE, ckpt_dir, f'model_{cls}_35.pth')
            intermediate = glob.glob(os.path.join(HERE, ckpt_dir,
                                                  f'model_{cls}_[0-9]*.pth'))
            items[cls] = {
                'log_exists': bool(text),
                'total_35': '--total_epochs 35' in text,
                'samples_35': 'Total samples loaded: 35 from 1 categories' in text,
                'epoch35_seen': '[35/35]' in text,
                'shot_line': bool(re.search(r'\[shots\]', text)),
                'checkpoint35': os.path.isfile(final),
                'all_intermediate': len(intermediate),
            }
        out[name] = items
    return out


def main():
    report = {
        'synthetic_mvtec_1shot': audit_mvtec_synth(),
        'synthetic_visa_1shot': audit_visa_synth(),
        'shots': audit_shots(),
        'training': audit_training_logs(),
    }
    out = os.path.join(HERE, 'logs', 'pipeline_end_to_end_audit.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'PIPELINE_AUDIT -> {out}')


if __name__ == '__main__':
    main()
