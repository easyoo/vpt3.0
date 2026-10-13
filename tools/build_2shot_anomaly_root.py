# -*- coding: utf-8 -*-
# [2-shot 实验] 把 AnoStyler 的输出整理成训练 DataLoader 需要的目录结构。
#
# AnoStyler 输出: <save_path>/<cls>/<defect>/<stem>/{image/gen_ano_i.jpg, mask/gen_mask_i.jpg}
# 训练需要:      <root>/mvtec/<cls>/<defect>/{image/<stem>_gen_ano_i.jpg, mask/<stem>_gen_mask_i.jpg}
#
# 与 4-shot 的训练数据根目录(/home/jjquan/experiment/visual2/anomaly_samples)保持同一套命名，
# 用**符号链接**而不是拷贝（省磁盘，且保证与 AnoStyler 产物逐位一致）。
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'anostyler_2shot', 'genbyAnoStyler_2shot')
DST = '/home/jjquan/experiment/visual2/anomaly_samples_2shot/mvtec'


def main():
    man = json.load(open(os.path.join(HERE, 'anostyler_2shot', 'configs_manifest.json')))
    num_gen = man['num_gen']
    n_img = n_mask = 0
    for it in man['items']:
        s_dir = os.path.join(SRC, it['cls'], it['defect'], it['stem'])
        d_dir = os.path.join(DST, it['cls'], it['defect'])
        os.makedirs(os.path.join(d_dir, 'image'), exist_ok=True)
        os.makedirs(os.path.join(d_dir, 'mask'), exist_ok=True)
        got = 0
        for i in range(num_gen):
            simg = os.path.join(s_dir, 'image', 'gen_ano_%d.jpg' % i)
            smsk = os.path.join(s_dir, 'mask', 'gen_mask_%d.jpg' % i)
            if not (os.path.exists(simg) and os.path.exists(smsk)):
                continue
            limg = os.path.join(d_dir, 'image', '%s_gen_ano_%d.jpg' % (it['stem'], i))
            lmsk = os.path.join(d_dir, 'mask', '%s_gen_mask_%d.jpg' % (it['stem'], i))
            for src, dst in ((simg, limg), (smsk, lmsk)):
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
            n_img += 1
            n_mask += 1
            got += 1
        print('[link] %-11s %-22s %-6s -> %d/%d' % (it['cls'], it['defect'], it['stem'], got, num_gen))
    print('\n完成: 图片链接 %d 个 / mask 链接 %d 个 -> %s' % (n_img, n_mask, DST))


if __name__ == '__main__':
    main()
