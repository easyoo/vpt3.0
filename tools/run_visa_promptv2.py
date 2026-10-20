# -*- coding: utf-8 -*-
"""Run class-specific-prompt AnoStyler generation for VisA 2-shot."""
import concurrent.futures
import glob
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'AnoStyler_promptv2')
CFG_DIR = os.path.join(HERE, 'anostyler_visa', '2shot_promptv2', 'configs')
LOG_DIR = os.path.join(HERE, 'logs', 'visa_promptv2')
PY = '/home/jjquan/miniconda3/envs/anostyler/bin/python'
GPU = 'GPU-3bf0407c-aa78-8aab-cd2f-1c7b3356c5be'
WORKERS = 3


def complete(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))
    stem = os.path.splitext(os.path.basename(cfg['normal_image']))[0]
    image_dir = os.path.join(cfg['save_path'], cfg['category'], cfg['defect'],
                             stem, 'image')
    return len(glob.glob(os.path.join(image_dir, 'gen_ano_*.jpg'))) >= int(
        cfg['num_gen'])


def run_one(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))
    name = '%s_%s' % (cfg['category'],
                      os.path.splitext(os.path.basename(cfg['normal_image']))[0])
    log_path = os.path.join(LOG_DIR, name + '.log')
    if complete(cfg_path):
        return name, 0, 'already complete'
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = GPU
    with open(log_path, 'w') as log:
        proc = subprocess.run([PY, 'main.py', '--config', cfg_path],
                              cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    return name, proc.returncode, log_path


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    configs = sorted(glob.glob(os.path.join(CFG_DIR, '*.yaml')))
    pending = [p for p in configs if not complete(p)]
    print('total=%d pending=%d workers=%d gpu=%s' %
          (len(configs), len(pending), WORKERS, GPU), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, p) for p in pending]
        for fut in concurrent.futures.as_completed(futures):
            name, rc, path = fut.result()
            print('%s rc=%d %s' % (name, rc, path), flush=True)
            if rc:
                raise SystemExit(rc)
    print('ALL_PROMPTV2_GENERATION_DONE', flush=True)


if __name__ == '__main__':
    main()
