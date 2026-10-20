# -*- coding: utf-8 -*-
# [vpt5.0 · M26 数据侧] 低成本重合成：把"已有合成图里的缺陷纹理"贴到 2-shot 正常图上。
#
# 证据链：M24（把 2-shot 与 4-shot 两套 AnoStyler 合成集并集：每类 70→210）是本轮**唯一**
# 把 P-AP 拉正的机制（+1.89，+3.94%），说明"合成监督的多样性"才是 P-AP 的瓶颈。
# 但 AnoStyler 每 35 张要 ~30 分钟，扩不动。M26 用"纹理库 + 随机平滑掩码"在**分钟级**
# 生成上千张新合成图：底图 = 该类的 2-shot 正常图（与 AnoStyler 一致），
# 纹理 = 已有 AnoStyler 合成图里随机裁的缺陷块，掩码 = 随机形状+羽化。
#
# 输出：直接往 <ANOROOT>/mvtec/<cls>/synthpaste/{image,mask}/ 写 jpg（新"缺陷类型"目录），
# 训练时 AnoStylerGeneratedDataset 会当作一个额外的缺陷类型，每轮采样 anom_per_type 张。
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
PER_CLS = 400
SIZE = 512
SEED = 7


def random_blob_mask(h, w, area_ratio, feather=9):
    """随机形状的平滑掩码：在低分辨率上画一个椭圆/矩形再放大+羽化。"""
    small = 32
    m = Image.new('L', (small, small), 0)
    px = m.load()
    a = area_ratio * small * small
    k = random.uniform(0.7, 1.4)
    rw = max(2, int((a * k) ** 0.5))
    rh = max(2, int((a / k) ** 0.5))
    cx, cy = random.randint(rw, small - rw), random.randint(rh, small - rh)
    for y in range(small):
        for x in range(small):
            if ((x - cx) / rw) ** 2 + ((y - cy) / rh) ** 2 <= 1.0:
                px[x, y] = 255
    m = m.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m, dtype=np.float32) / 255.0


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    total = 0
    for cls in CLS5:
        shots = get_shots(DATA, cls, shots=2, seed=1)
        normals = [Image.open(os.path.join(DATA, cls, 'train', 'good', s)).convert('RGB') for s in shots]
        tex_paths = glob.glob(os.path.join(TEX_ROOT, cls, '*', 'image', '*.jpg'))
        if not tex_paths:
            print(f'{cls}: 无纹理来源，跳过'); continue
        d_img = os.path.join(OUT_ROOT, cls, 'synthpaste', 'image')
        d_msk = os.path.join(OUT_ROOT, cls, 'synthpaste', 'mask')
        os.makedirs(d_img, exist_ok=True); os.makedirs(d_msk, exist_ok=True)
        for i in range(PER_CLS):
            base = normals[i % len(normals)].resize((SIZE, SIZE), Image.BILINEAR)
            tex = Image.open(random.choice(tex_paths)).convert('RGB')
            tw = random.randint(48, 160)
            x0 = random.randint(0, max(0, tex.width - tw))
            y0 = random.randint(0, max(0, tex.height - tw))
            patch = tex.crop((x0, y0, x0 + tw, y0 + tw)).resize((SIZE, SIZE), Image.BILINEAR)
            # 纹理做一点光度抖动，避免模型记住固定的绝对色
            arr = np.asarray(patch, dtype=np.float32)
            arr = arr * random.uniform(0.8, 1.2) + random.uniform(-18, 18)
            patch = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            mask = random_blob_mask(SIZE, SIZE, area_ratio=random.uniform(0.004, 0.03),
                                    feather=random.choice([5, 9, 15]))
            m3 = np.repeat(mask[:, :, None], 3, axis=2)
            out = np.asarray(base, dtype=np.float32) * (1 - m3) + np.asarray(patch, dtype=np.float32) * m3
            Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
                os.path.join(d_img, 'p%04d.jpg' % i), quality=92)
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                os.path.join(d_msk, 'p%04d.jpg' % i), quality=92)
            total += 1
        print(f'{cls:12s} 生成 {PER_CLS} 张 → {d_img}')
    print(f'\n合计生成 {total} 张（贴纹理合成，底图=该类 2-shot 正常图）')


if __name__ == '__main__':
    main()
