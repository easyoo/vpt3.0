# -*- coding: utf-8 -*-
"""Run all pending AnoStyler class-prompt configs from one config directory."""
import argparse
import concurrent.futures
import glob
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ANOSTYLER = os.path.join(HERE, 'AnoStyler_promptv2')
PY = '/home/jjquan/miniconda3/envs/anostyler/bin/python'


def complete(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))
    stem = os.path.splitext(os.path.basename(cfg['normal_image']))[0]
    image_dir = os.path.join(cfg['save_path'], cfg['category'],
                             cfg['defect'], stem, 'image')
    mask_dir = os.path.join(cfg['save_path'], cfg['category'],
                            cfg['defect'], stem, 'mask')
    return (
        len(glob.glob(os.path.join(image_dir, 'gen_ano_*.jpg'))) >= int(cfg['num_gen'])
        and len(glob.glob(os.path.join(mask_dir, 'gen_mask_*.jpg'))) >= int(cfg['num_gen'])
    )


def run_one(cfg_path, log_dir, gpu, prefix):
    cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))
    stem = os.path.splitext(os.path.basename(cfg['normal_image']))[0]
    name = f'{prefix}_{cfg["category"]}_{stem}'
    if complete(cfg_path):
        return name, 0, 'already complete'
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = gpu
    log_path = os.path.join(log_dir, name + '.log')
    with open(log_path, 'w') as log:
        proc = subprocess.run([PY, 'main.py', '--config', cfg_path],
                              cwd=ANOSTYLER, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    return name, proc.returncode, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_dir', required=True)
    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--gpu', required=True,
                    help='UUID or CUDA_VISIBLE_DEVICES value')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    config_dir = os.path.abspath(args.config_dir)
    log_dir = os.path.abspath(args.log_dir)
    configs = sorted(glob.glob(os.path.join(config_dir, '*.yaml')))
    pending = [p for p in configs if not complete(p)]
    print(f'total={len(configs)} pending={len(pending)} workers={args.workers} gpu={args.gpu}',
          flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, p, log_dir, args.gpu, args.prefix)
                   for p in pending]
        for fut in concurrent.futures.as_completed(futures):
            name, rc, path = fut.result()
            print(f'{name} rc={rc} {path}', flush=True)
            if rc:
                raise SystemExit(rc)
    print('ALL_PROMPTV2_GENERATION_DONE', flush=True)


if __name__ == '__main__':
    main()
