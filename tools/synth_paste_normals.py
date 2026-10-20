# -*- coding: utf-8 -*-
# [vpt5.0 · M28 数据侧] 用**大量正常底图**重合成异常，直接把"正常监督"的覆盖面放大一个量级。
#
# 证据链：
#   * M24（把合成底图来源从 2 张扩到 4 张/类的 AnoStyler 集并集）是本轮**唯一**把 P-AP 拉正的机制（+1.89）。
#   * 诊断（diag_ap_tail.py）：差距来自"未被 2 张 shot 覆盖的正常变化"（尾部高分正常像素）。
#   * 现有所有损失里，辅助合成图里 **mask=0 的像素就是"正常"监督**（M7/M22/M11/M20 都用到）。
#   ⇒ 只要用更多正常图当底图，就等于把"正常监督"的覆盖面直接放大。
#
# 做法：每类随机取 20 张 train/good（**不含**测试集，也不排除那 2 张 shot），
# 用 `synthpaste` 的纹理贴合方式各生成 20 张异常图 → 新的缺陷类型目录 `synthnorm`。
import glob
import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vpt3.5', 'tools'))
from shot_utils import get_shots                     # noqa: E402

DATA = '/home/jjquan/MVTec-AD'
TEX_ROOT = '/home/jjquan/experiment/visual2/anomaly_samples_2shot_big/mvtec'
OUT_ROOT = '/home/jjquan/experiment/visual2/anomaly_samples_2shot_big/mvtec'
CLS5 = ['transistor', 'screw', 'capsule', 'metal_nut', 'wood']
N_NORMAL = 20          # 每类用多少张正常底图
PER_BASE = 20          # 每张底图生成多少张
SIZE = 512
SEED = 11


def random_blob_mask(h, w, area_ratio, feather=9):
    small = 32
    m = Image.new('L', (small, small), 0)
    px = m.load()
    a = area_ratio * small * small
    k = random.uniform(0.7, 1.4)
    rw = max(2, int((a * k) ** 0.5)); rh = max(2, int((a / k) ** 0.5))
    cx, cy = random.randint(rw, small - rw), random.randint(rh, small - rh)
    for y in range(small):
        for x in range(small):
            if ((x - cx) / rw) ** 2 + ((y - cy) / rh) ** 2 <= 1.0:
                px[x, y] = 255
    m = m.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m, dtype=np.float32) / 255.0


def main():
    random.seed(SEED); np.random.seed(SEED)
    total = 0
    for cls in CLS5:
        shots = get_shots(DATA, cls, shots=2, seed=1)
        good_dir = os.path.join(DATA, cls, 'train', 'good')
        pool = sorted(f for f in os.listdir(good_dir) if f not in shots)   # 除 2 张 shot 之外
        picks = random.sample(pool, min(N_NORMAL, len(pool)))
        tex_paths = glob.glob(os.path.join(TEX_ROOT, cls, '*', 'image', '*.jpg'))
        d_img = os.path.join(OUT_ROOT, cls, 'synthnorm', 'image')
        d_msk = os.path.join(OUT_ROOT, cls, 'synthnorm', 'mask')
        os.makedirs(d_img, exist_ok=True); os.makedirs(d_msk, exist_ok=True)
        i = 0
        for name in picks:
            base = Image.open(os.path.join(good_dir, name)).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
            for _ in range(PER_BASE):
                tex = Image.open(random.choice(tex_paths)).convert('RGB')
                tw = random.randint(48, 160)
                x0 = random.randint(0, max(0, tex.width - tw)); y0 = random.randint(0, max(0, tex.height - tw))
                patch = tex.crop((x0, y0, x0 + tw, y0 + tw)).resize((SIZE, SIZE), Image.BILINEAR)
                arr = np.asarray(patch, dtype=np.float32) * random.uniform(0.8, 1.2) + random.uniform(-18, 18)
                patch = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                mask = random_blob_mask(SIZE, SIZE, random.uniform(0.004, 0.03),
                                        random.choice([5, 9, 15]))
                m3 = np.repeat(mask[:, :, None], 3, axis=2)
                out = np.asarray(base, dtype=np.float32) * (1 - m3) + np.asarray(patch, dtype=np.float32) * m3
                Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
                    os.path.join(d_img, 'n%04d.jpg' % i), quality=92)
                Image.fromarray((mask * 255).astype(np.uint8)).save(
                    os.path.join(d_msk, 'n%04d.jpg' % i), quality=92)
                i += 1; total += 1
        print(f'{cls:12s} 底图 {len(picks)} 张 × {PER_BASE} → {i} 张（正常监督覆盖 {len(picks)} 个正常背景）')
    print(f'\n合计 {total} 张 → {OUT_ROOT}/<cls>/synthnorm/')


if __name__ == '__main__':
    main()
