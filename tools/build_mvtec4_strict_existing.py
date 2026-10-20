# -*- coding: utf-8 -*-
"""Build a strict MVTec 4-shot view from the existing AnoStyler 4x35 output.

The old AnoStyler root already contains exactly four normal source images per
category and 35 image/mask pairs per source.  This script makes those four
source images the *only* train/good images visible in a dataset view, then
links the existing anomalies into a training root.
"""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/jjquan/MVTec-AD')
SRC = Path('/home/jjquan/AnoStyler/genbyAnoStyler')
VIEW = Path('/home/jjquan/experiment/visual2/MVTec-AD_strict4')
ANOROOT = Path('/home/jjquan/experiment/visual2/anomaly_samples_mvtec_4shot_strict/mvtec')
MANIFEST = ROOT / 'data_prep' / 'mvtec4_strict_existing_manifest.json'

CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper',
]
NUM_GEN = 35


def replace_link(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    all_items = []
    total_pairs = 0
    for cls in CLASSES:
        sources = []
        for defect_dir in sorted((SRC / cls).iterdir()):
            if not defect_dir.is_dir():
                continue
            for sample_dir in sorted(defect_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                stems = [
                    p.stem for p in (sample_dir / 'image').glob('gen_ano_*.jpg')
                ]
                mstems = [
                    p.stem for p in (sample_dir / 'mask').glob('gen_mask_*.jpg')
                ]
                if len(stems) < NUM_GEN:
                    raise RuntimeError(
                        f'{cls}/{defect_dir.name}/{sample_dir.name}: '
                        f'image count {len(stems)} < {NUM_GEN}')
                if len(mstems) < NUM_GEN:
                    raise RuntimeError(
                        f'{cls}/{defect_dir.name}/{sample_dir.name}: '
                        f'mask count {len(mstems)} < {NUM_GEN}')
                sources.append({
                    'cls': cls,
                    'defect': defect_dir.name,
                    'stem': sample_dir.name,
                    'sample_dir': sample_dir,
                })

        if len(sources) != 4:
            raise RuntimeError(f'{cls}: expected 4 source images, got {len(sources)}')
        stems = [x['stem'] for x in sources]
        if len(set(stems)) != 4:
            raise RuntimeError(f'{cls}: duplicate source stems: {stems}')

        normal_dir = DATA / cls / 'train' / 'good'
        view_good = VIEW / cls / 'train' / 'good'
        view_good.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            matches = list(normal_dir.glob(stem + '.*'))
            if len(matches) != 1:
                raise RuntimeError(f'{cls}/{stem}: normal-image matches={matches}')
            replace_link(matches[0], view_good / matches[0].name)

        replace_link(DATA / cls / 'test', VIEW / cls / 'test')
        replace_link(DATA / cls / 'ground_truth', VIEW / cls / 'ground_truth')

        for item in sources:
            dst_dir = ANOROOT / cls / item['defect']
            got = 0
            for i in range(NUM_GEN):
                simg = item['sample_dir'] / 'image' / f'gen_ano_{i}.jpg'
                smsk = item['sample_dir'] / 'mask' / f'gen_mask_{i}.jpg'
                if not simg.exists() or not smsk.exists():
                    raise RuntimeError(f'missing pair: {simg} / {smsk}')
                replace_link(
                    simg, dst_dir / 'image' / f'{item["stem"]}_gen_ano_{i}.jpg')
                replace_link(
                    smsk, dst_dir / 'mask' / f'{item["stem"]}_gen_mask_{i}.jpg')
                got += 1
            total_pairs += got
            all_items.append({
                'class': cls,
                'source_stem': item['stem'],
                'defect': item['defect'],
                'generated_pairs': got,
                'source_sample_dir': str(item['sample_dir']),
            })

        print(f'{cls:12s} sources={stems} pairs={sum(x["generated_pairs"] for x in all_items if x["class"] == cls)}')

    if total_pairs != len(CLASSES) * 4 * NUM_GEN:
        raise RuntimeError(f'expected {len(CLASSES)*4*NUM_GEN} pairs, got {total_pairs}')

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset': 'MVTec-AD',
        'shots': 4,
        'num_gen_per_source': NUM_GEN,
        'source_root': str(SRC),
        'dataset_view': str(VIEW),
        'anomaly_root': str(ANOROOT),
        'total_pairs': total_pairs,
        'items': all_items,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'complete: {total_pairs} pairs -> {ANOROOT}')
    print(f'dataset view -> {VIEW}')
    print(f'manifest -> {MANIFEST}')


if __name__ == '__main__':
    main()
