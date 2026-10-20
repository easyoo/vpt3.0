# -*- coding: utf-8 -*-
"""Create MVTec 1-shot AnoStyler configs with class-specific prompts."""
import json
import os

import yaml

from shot_utils import get_shots, shots_hash

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/jjquan/MVTec-AD'
OUT = os.path.join(HERE, 'anostyler_mvtec', '1shot_promptv2')
NUM_GEN = 35
SEED = 1

PROMPTS = {
    'bottle': 'broken or contaminated bottle surface',
    'cable': 'cut insulation or missing wire',
    'capsule': 'cracked or dented capsule',
    'carpet': 'color stain or cut pile',
    'grid': 'bent or broken grid',
    'hazelnut': 'cracked or cut hazelnut shell',
    'leather': 'color stain or cut leather',
    'metal_nut': 'bent or scratched metal nut',
    'pill': 'discolored or cracked pill',
    'screw': 'scratched or deformed screw',
    'tile': 'cracked or rough tile',
    'toothbrush': 'defective toothbrush bristles',
    'transistor': 'bent or cut transistor lead',
    'wood': 'stained or scratched wood',
    'zipper': 'broken or damaged zipper teeth',
}


def main():
    cfg_dir = os.path.join(OUT, 'configs')
    save_dir = os.path.join(OUT, 'genbyAnoStyler')
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    manifest = []
    for idx, cls in enumerate(sorted(PROMPTS)):
        picks = get_shots(DATA, cls, shots=1, seed=SEED)
        prompt = PROMPTS[cls]
        for img in picks:
            stem = os.path.splitext(img)[0]
            cfg = {
                'seed': 77,
                'num_gen': NUM_GEN,
                'img_width': 512,
                'img_height': 512,
                'm_max': 5,
                'alpha': 0.7,
                'fg_mask': True,
                'sam_ckpt': './checkpoints/sam_vit_b_01ec64.pth',
                'crop_size': 128,
                'num_crops': 64,
                'max_step': 75,
                'lr': 0.0005,
                'thresh': 0.7,
                'lambda_tv': 0.002,
                'lambda_pdir': 9000,
                'lambda_gdir': 500,
                'lambda_c': 150,
                'lambda_mclip': 1000,
                'category': cls,
                'defect': prompt,
                'normal_image': os.path.join(DATA, cls, 'train', 'good', img),
                'save_path': save_dir,
            }
            name = f'1shot_{idx:04d}_{cls}_{stem}.yaml'
            path = os.path.join(cfg_dir, name)
            with open(path, 'w', encoding='utf-8') as fh:
                yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
            manifest.append({
                'idx': idx, 'cls': cls, 'stem': stem, 'prompt': prompt,
                'normal_image': cfg['normal_image'], 'config': path,
                'shot_hash': shots_hash(picks),
            })
    with open(os.path.join(OUT, 'configs_manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump({'dataset': 'MVTec-AD', 'shots': 1, 'seed': SEED,
                   'num_gen': NUM_GEN, 'save_path': save_dir,
                   'items': manifest}, fh, indent=2, ensure_ascii=False)
    print(f'created {len(manifest)} configs -> {cfg_dir}')


if __name__ == '__main__':
    main()
