# -*- coding: utf-8 -*-
"""[FIX-3 回归测试] bank 分支 query_batch 在**尾批只有 1 张图**时不得崩溃/形状错乱。

背景：MemoryBank.query_batch 末尾用 .squeeze() 归约，当 batch=1 时 (1,1,H,W) → (H,W)，
下游 torch.tensor(am).unsqueeze(1) → (H,1,W)，与其它批次 (B,1,H,W) 拼接时报维度不一致错误
（got 4 and 3）。MVTec carpet/tile 测试集 N=117、batch=4 时必现。

用法：CUDA_VISIBLE_DEVICES=<uuid> python test_bank_tail_batch.py
"""
import argparse
import importlib.util
import os
import sys

import torch
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))


def _default_script():
    for c in (os.path.join(HERE, 'code_vpt4', 'main-fewshot-vpt4.py'),
              os.path.join(HERE, '..', 'code', 'main-fewshot-vpt4.py'),
              os.path.join(HERE, 'code_vpt2', 'main-fewshot-vpt2.py')):
        if os.path.exists(c):
            return os.path.normpath(c)
    raise FileNotFoundError('找不到主脚本，请用 --script 指定')


def load_mod(p):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    spec = importlib.util.spec_from_file_location('tbb_main', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['tbb_main'] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='/home/jjquan/MVTec-AD')
    ap.add_argument('--cls', default='bottle')
    ap.add_argument('--size', type=int, default=112, help='须为 14 的倍数（patch=14）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--script', default=_default_script())
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
    mod = load_mod(args.script)
    from util.memorybank import MemoryBank
    sys.path.insert(0, HERE)
    from shot_utils import get_shots

    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    picks = get_shots(args.data_path, args.cls, shots=1, seed=1)
    root = os.path.join(args.data_path, args.cls, 'train', 'good')
    imgs = [tf(Image.open(os.path.join(root, f)).convert('RGB')) for f in picks]
    normal = torch.cat([mod.augmentation(im) for im in imgs], 0).to(args.device)
    print(f'[test] {args.cls}: 1 shot → 增强 {tuple(normal.shape)}；size={args.size}')

    bank = MemoryBank(target_cls=args.cls, use_json=False, imgs=normal,
                      info=lambda *a, **k: None, device=args.device)
    q = tf(Image.open(os.path.join(root, picks[0])).convert('RGB')).unsqueeze(0).to(args.device)

    am1 = bank.query_batch(q)                                   # 尾批=1 的情形
    am4 = bank.query_batch(q.repeat(4, 1, 1, 1))
    assert am1.shape == (1, args.size, args.size), f'尾批=1 形状错误: {am1.shape}'
    assert am4.shape == (4, args.size, args.size), f'批=4 形状错误: {am4.shape}'
    print(f'[test] query_batch 形状 ✓ batch=1 → {tuple(am1.shape)}；batch=4 → {tuple(am4.shape)}')

    def _to4d(a):
        t = torch.tensor(a)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim == 3:
            t = t.unsqueeze(1)
        assert t.ndim == 4
        return t

    cat = torch.cat([torch.zeros(3, 1, args.size, args.size), _to4d(am1)], 0)
    assert cat.shape == (4, 1, args.size, args.size)
    d = (torch.tensor(am4[0]) - torch.tensor(am1[0])).abs().max().item()
    assert d < 1e-6, f'单张与批量结果不一致 max|Δ|={d}'
    print(f'[test] 拼接 ✓ shape={tuple(cat.shape)}；单张 vs 批量 max|Δ|={d:.2e}')
    print('[test] ALL PASSED (FIX-3)')


if __name__ == '__main__':
    main()
