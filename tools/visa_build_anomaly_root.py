# -*- coding: utf-8 -*-
# [VisA 准备 3/3] 把 AnoStyler 的 VisA 合成产物整理成训练 DataLoader 需要的结构：
#   <out>/visa/<cls>/src<stem>/{image/<stem>_gen_ano_i.jpg, mask/<stem>_gen_mask_i.jpg}
#
# **为什么要"每个底图一个类型目录"**：训练侧 `AnoStylerGeneratedDataset(..., samples_per_anomaly_type=35)`
# 每个"缺陷类型"最多采样 35 张。若把 K 张底图的产物全塞进同一个类型目录，每个 epoch 只会用到 35 张，
# 4-shot 就退化成 1-shot 的监督量（实测踩过：pcb1 4-shot 只加载 35 张）。
# 按底图分目录后 → 每类每轮加载 **K×35** 张，与 MVTec 实验的组织口径完全一致。
# 训练脚本里用的是 os.path.join(args.anomaly_data_root, 'visa')，
# 所以 --anomaly_data_root 指向 /home/jjquan/experiment/visual2/anomaly_samples_visa_<K>shot 即可。
#
# 用法：python visa_build_anomaly_root.py <K> [src] [dst_visa_dir]
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
K = sys.argv[1] if len(sys.argv) > 1 else '1'
SRC = os.path.join(HERE, 'anostyler_visa', K + 'shot', 'genbyAnoStyler')
DST = '/home/jjquan/experiment/visual2/anomaly_samples_visa_' + K + 'shot/visa'
if len(sys.argv) > 2:
    SRC = sys.argv[2]
if len(sys.argv) > 3:
    DST = sys.argv[3]
NUM_GEN = 35


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    n = 0
    for cls_dir in sorted(glob.glob(os.path.join(SRC, '*'))):
        cls = os.path.basename(cls_dir)
        for type_dir in sorted(glob.glob(os.path.join(cls_dir, '*'))):
            t = os.path.basename(type_dir)
            for stem_dir in sorted(glob.glob(os.path.join(type_dir, '*'))):
                stem = os.path.basename(stem_dir)
                got = 0
                t_src = 'src' + stem                    # 每个底图一个"缺陷类型" → 每类型恰好 35 张
                for i in range(NUM_GEN):
                    si = os.path.join(stem_dir, 'image', 'gen_ano_%d.jpg' % i)
                    sm = os.path.join(stem_dir, 'mask', 'gen_mask_%d.jpg' % i)
                    if not (os.path.exists(si) and os.path.exists(sm)):
                        continue
                    link(si, os.path.join(DST, cls, t_src, 'image', '%s_gen_ano_%d.jpg' % (stem, i)))
                    link(sm, os.path.join(DST, cls, t_src, 'mask', '%s_gen_mask_%d.jpg' % (stem, i)))
                    got += 1
                n += got
                if got < NUM_GEN:
                    print('  WARN %s/%s/%s: %d/%d 张' % (cls, t, stem, got, NUM_GEN))
        c = len(glob.glob(os.path.join(DST, cls, '*', 'image', '*.jpg')))
        print('  %-12s %d 张' % (cls, c))
    print('\n合计 %d 张 → %s' % (n, DST))


if __name__ == '__main__':
    main()
