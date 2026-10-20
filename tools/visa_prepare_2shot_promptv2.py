# -*- coding: utf-8 -*-
"""Create VisA 2-shot AnoStyler configs with class-specific defect prompts."""
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS_JSON = os.path.join(HERE, 'anostyler_visa', '2shot', 'shots.json')
OUT = os.path.join(HERE, 'anostyler_visa', '2shot_promptv2')
VISAROOT = '/mnt/data/datasets/VisA'

PROMPTS = {
    'candle': 'surface cracks or broken wax',
    'capsules': 'dented shell or surface cracks',
    'cashew': 'surface cracks or burn marks',
    'chewinggum': 'surface stains or deformation',
    'fryum': 'broken edges or burn marks',
    'macaroni1': 'surface cracks or discoloration',
    'macaroni2': 'surface cracks or discoloration',
    'pcb1': 'missing components or solder defects',
    'pcb2': 'missing components or solder defects',
    'pcb3': 'missing components or solder defects',
    'pcb4': 'missing components or solder defects',
    'pipe_fryum': 'surface cracks or deformation',
}


def main():
    cfg_dir = os.path.join(OUT, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)
    shot_info = json.load(open(SHOTS_JSON, encoding='utf-8'))
    records = {}
    for idx, (cls, files) in enumerate(sorted(shot_info['picks'].items())):
        prompt = PROMPTS[cls]
        records[cls] = {'prompt': prompt, 'files': files}
        for rel in files:
            stem = os.path.splitext(os.path.basename(rel))[0]
            cfg = {
                'seed': 77,
                'num_gen': 35,
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
                'normal_image': os.path.join(VISAROOT, rel),
                'save_path': os.path.join(OUT, 'genbyAnoStyler'),
            }
            name = f'{idx:04d}_{cls}_{stem}.yaml'
            with open(os.path.join(cfg_dir, name), 'w', encoding='utf-8') as fh:
                yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
    with open(os.path.join(OUT, 'prompts.json'), 'w', encoding='utf-8') as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    print(f'created {sum(len(v["files"]) for v in records.values())} configs -> {cfg_dir}')


if __name__ == '__main__':
    main()
