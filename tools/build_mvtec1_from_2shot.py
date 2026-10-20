# -*- coding: utf-8 -*-
"""Build MVTec 1-shot anomaly root by extracting the exact 1-shot source.

AnoStyler generates each source independently, so the source image in the
existing nested 2-shot output is exactly the artifact a dedicated 1-shot run
would produce. Reusing it avoids redundant computation without changing data.
"""
import glob
import os

from shot_utils import get_shots, shots_hash

DATA = '/home/jjquan/MVTec-AD'
SRC = '/home/jjquan/experiment/visual2/anomaly_samples_2shot/mvtec'
DST = '/home/jjquan/experiment/visual2/anomaly_samples_mvtec_1shot/mvtec'
CLASSES = ('bottle cable capsule carpet grid hazelnut leather metal_nut pill screw '
           'tile toothbrush transistor wood zipper').split()


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    total = 0
    for cls in CLASSES:
        picks = get_shots(DATA, cls, shots=1, seed=1)
        stem = os.path.splitext(picks[0])[0]
        pairs = 0
        for defect_dir in sorted(glob.glob(os.path.join(SRC, cls, '*'))):
            defect = os.path.basename(defect_dir)
            images = sorted(glob.glob(os.path.join(defect_dir, 'image',
                                                   f'{stem}_gen_ano_*.jpg')))
            masks = sorted(glob.glob(os.path.join(defect_dir, 'mask',
                                                  f'{stem}_gen_mask_*.jpg')))
            if not images and not masks:
                continue
            if len(images) != len(masks):
                raise SystemExit(f'{cls}/{defect}: image/mask mismatch')
            for img, msk in zip(images, masks):
                link(img, os.path.join(DST, cls, defect, 'image',
                                       os.path.basename(img)))
                link(msk, os.path.join(DST, cls, defect, 'mask',
                                       os.path.basename(msk)))
            pairs += len(images)
        if pairs < 35:
            raise SystemExit(f'{cls}: only {pairs} anomaly pairs')
        print(f'{cls:11s} shot={picks[0]} hash={shots_hash(picks)} pairs={pairs}')
        total += pairs
    print(f'complete: {total} anomalies -> {DST}')


if __name__ == '__main__':
    main()
