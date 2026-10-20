# -*- coding: utf-8 -*-
"""[统一数据源] 训练 / mapping 分支 / bank 分支 **共用同一份 shot 清单**。

问题：此前三处各自 `random.seed(seed) + random.sample(sorted(train/good), 4)`，
虽然当前结果一致，但依赖"同种子+同排序+同随机实现"的约定，代码上没有任何保证。

做法：
  1. 首次调用时按 `seed + shots` 生成清单，**落盘**到 `.shots/<cls>_shot{K}_seed{S}.json`；
  2. 之后任何脚本（训练、评估、建 bank）都**读同一份清单**，即使 numpy/random 版本变化也不会漂移；
  3. 提供 `assert_same_shots()` 供脚本自检，不一致直接报错（而不是静默用不同数据）。

用法：
    from shot_utils import get_shots
    picks = get_shots(data_path, cls, shots=4, seed=1)     # -> ['034.png', ...]
"""
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SHOT_DIR = os.environ.get('VPT_SHOT_DIR', os.path.join(HERE, '.shots'))


def _paths(data_path, cls):
    root = os.path.join(data_path, cls, 'train', 'good')
    return sorted(f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)))


def _key(files):
    return hashlib.sha1('\n'.join(files).encode()).hexdigest()[:12]


def get_shots(data_path, cls, shots=4, seed=1, shot_dir=None, verbose=False):
    """返回该类别的 shot 文件名列表（**所有分支共用**，首次生成后落盘复用）。"""
    d = shot_dir or SHOT_DIR
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, f'{cls}_shot{shots}_seed{seed}.json')
    files = _paths(data_path, cls)
    if os.path.exists(fp):
        rec = json.load(open(fp))
        picks = rec['picks']
        assert rec.get('files_hash') == _key(files), (
            f'[shot_utils] {cls}: train/good 文件列表已变化，请删除 {fp} 后重建')
        assert len(picks) == shots, (fp, len(picks), shots)
    else:
        random.seed(seed)
        picks = random.sample(files, shots)
        json.dump({'cls': cls, 'shots': shots, 'seed': seed, 'picks': picks,
                   'n_files': len(files), 'files_hash': _key(files),
                   'data_path': data_path}, open(fp, 'w'), indent=1, ensure_ascii=False)
    if verbose:
        print(f'[shot_utils] {cls} shot{shots}/seed{seed} → {picks}')
    return picks


def shots_hash(picks):
    return hashlib.sha1('\n'.join(picks).encode()).hexdigest()[:12]


def assert_same_shots(a, b, msg=''):
    """两个分支/两次运行的 shot 清单必须逐项相同，否则直接报错。"""
    if list(a) != list(b):
        raise RuntimeError(f'[shot_utils] shot 清单不一致{("：" + msg) if msg else ""}\n A={list(a)}\n B={list(b)}')
    return True


def load_images_tensor(picks, data_path, cls, tf, torch_module=None):
    """按统一清单读取 shot 图片为张量（训练/建库两处共用同一实现）。"""
    from PIL import Image
    import torch
    root = os.path.join(data_path, cls, 'train', 'good')
    return torch.stack([tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks], 0)
