# -*- coding: utf-8 -*-
# [vpt5.0 · M24 数据侧] 把 2-shot 与 4-shot 两套 AnoStyler 合成异常**合并**成一个更大的监督池。
#
# 动机：所有"在现有合成集上加权/加损失"的尝试都为负 → 瓶颈很可能是**代理监督的多样性**：
# 2-shot 每类只有 2 张底图 × 35 = 70 张合成图，模型很容易贴合这 2 张底图的合成伪影。
# 合并后每类 210 张（2-shot 70 + 4-shot 140），而 DataLoader 仍是每类每轮采样 35/缺陷类型，
# 于是每个 epoch 看到的是**更大池子里的随机子集** → 监督多样性显著增加，训练预算不变。
#
# 注意：这不改变 2-shot 的"正常参考只有 2 张"这一设定（推理时仍然只用 2 张 shot）；
# 只是把**辅助合成异常**的池子扩大（合成图全部来自 train/good，且与测试集零重叠）。
import glob
import os

SRC2 = '/home/jjquan/experiment/visual2/anomaly_samples_2shot/mvtec'
SRC4 = '/home/jjquan/experiment/visual2/anomaly_samples/mvtec'
DST = '/home/jjquan/experiment/visual2/anomaly_samples_2shot_big/mvtec'


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    n2 = n4 = 0
    for cls in sorted(os.listdir(SRC2)):
        types = set(os.listdir(os.path.join(SRC2, cls)))
        if os.path.isdir(os.path.join(SRC4, cls)):
            types |= set(os.listdir(os.path.join(SRC4, cls)))     # 缺陷类型取并集
        for t2 in sorted(types):
            d_img = os.path.join(DST, cls, t2, 'image')
            d_msk = os.path.join(DST, cls, t2, 'mask')
            # 2-shot 原样
            for p in glob.glob(os.path.join(SRC2, cls, t2, 'image', '*.jpg')):
                b = os.path.basename(p)
                link(p, os.path.join(d_img, b))
                link(os.path.join(SRC2, cls, t2, 'mask', b.replace('gen_ano', 'gen_mask')),
                     os.path.join(d_msk, b.replace('gen_ano', 'gen_mask')))
                n2 += 1
            # 4-shot 加前缀 s4_（含 2-shot 里没有的缺陷类型）
            s4 = os.path.join(SRC4, cls, t2, 'image')
            if os.path.isdir(s4):
                for p in glob.glob(os.path.join(s4, '*.jpg')):
                    b = os.path.basename(p)
                    link(p, os.path.join(d_img, 's4_' + b))
                    link(os.path.join(SRC4, cls, t2, 'mask', b.replace('gen_ano', 'gen_mask')),
                         os.path.join(d_msk, 's4_' + b.replace('gen_ano', 'gen_mask')))
                    n4 += 1
        c2 = len(glob.glob(os.path.join(DST, cls, '*', 'image', '*.jpg')))
        print('%-12s 合并后 %d 张（2-shot 池 + 4-shot 池）' % (cls, c2))
    print('\n合计：2-shot 链接 %d 张 + 4-shot 链接 %d 张 → %s' % (n2, n4, DST))


if __name__ == '__main__':
    main()
