# -*- coding: utf-8 -*-
"""Materialize MVTec 1-shot AnoStyler output as training-loader symlinks."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, 'anostyler_mvtec', '1shot_promptv2',
                        'configs_manifest.json')
DST = ('/home/jjquan/experiment/visual2/'
       'anomaly_samples_mvtec_1shot_promptv2/mvtec')


def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.abspath(src), dst)


def main():
    manifest = json.load(open(MANIFEST, encoding='utf-8'))
    total = 0
    for item in manifest['items']:
        cls, stem, defect = item['cls'], item['stem'], item['prompt']
        src_dir = os.path.join(manifest['save_path'], cls, defect, stem)
        dst_dir = os.path.join(DST, cls, defect)
        got = 0
        for i in range(manifest['num_gen']):
            src_img = os.path.join(src_dir, 'image', f'gen_ano_{i}.jpg')
            src_msk = os.path.join(src_dir, 'mask', f'gen_mask_{i}.jpg')
            if not (os.path.exists(src_img) and os.path.exists(src_msk)):
                continue
            link(src_img, os.path.join(dst_dir, 'image',
                                       f'{stem}_gen_ano_{i}.jpg'))
            link(src_msk, os.path.join(dst_dir, 'mask',
                                       f'{stem}_gen_mask_{i}.jpg'))
            got += 1
        print(f'{cls:12s} {stem:6s} {got}/{manifest["num_gen"]}')
        total += got
    if total != len(manifest['items']) * manifest['num_gen']:
        raise SystemExit(f'incomplete synthesis: {total} images')
    print(f'complete: {total} image/mask pairs -> {DST}')


if __name__ == '__main__':
    main()
