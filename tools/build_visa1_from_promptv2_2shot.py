# -*- coding: utf-8 -*-
"""Build VisA 1-shot prompt-v2 anomaly root from nested 2-shot artifacts."""
import os

from shot_utils import get_shots, shots_hash

DATA = '/home/jjquan/experiment/visual2/visa_prepared'
SRC = '/home/jjquan/experiment/visual2/anomaly_samples_visa_2shot_promptv2/visa'
DST = '/home/jjquan/experiment/visual2/anomaly_samples_visa_1shot_promptv2/visa'
CLASSES = ('candle capsules cashew chewinggum fryum macaroni1 '
           'macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum').split()


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    total = 0
    for cls in CLASSES:
        picks = get_shots(DATA, cls, shots=1, seed=1)
        stem = os.path.splitext(os.path.basename(picks[0]))[0]
        src_dir = os.path.join(SRC, cls, f'src{stem}')
        if not os.path.isdir(src_dir):
            raise SystemExit(f'{cls}: missing source dir {src_dir}')
        dst_type = os.path.join(DST, cls, f'src{stem}')
        pairs = 0
        for i in range(35):
            src_img = os.path.join(src_dir, 'image', f'{stem}_gen_ano_{i}.jpg')
            src_msk = os.path.join(src_dir, 'mask', f'{stem}_gen_mask_{i}.jpg')
            if not (os.path.exists(src_img) and os.path.exists(src_msk)):
                continue
            link(src_img, os.path.join(dst_type, 'image',
                                       f'{stem}_gen_ano_{i}.jpg'))
            link(src_msk, os.path.join(dst_type, 'mask',
                                       f'{stem}_gen_mask_{i}.jpg'))
            pairs += 1
        if pairs != 35:
            raise SystemExit(f'{cls}: {pairs}/35 anomaly pairs')
        print(f'{cls:12s} shot={picks[0]} hash={shots_hash(picks)} pairs={pairs}')
        total += pairs
    print(f'complete: {total} anomalies -> {DST}')


if __name__ == '__main__':
    main()
