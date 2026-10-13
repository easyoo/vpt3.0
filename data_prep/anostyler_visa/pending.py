# -*- coding: utf-8 -*-
# 列出"还没合成完"的 VisA 配置（三个 shot 设定合并扫描）。
# 完成判据：<save_path>/<cls>/<defect>/<stem>/image/gen_ano_*.jpg 数量 >= num_gen。
import glob
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
done = total = 0
for K in (1, 2, 4):
    for cfg in sorted(glob.glob(os.path.join(HERE, f'{K}shot', 'configs', '*.yaml'))):
        c = yaml.safe_load(open(cfg))
        stem = os.path.splitext(os.path.basename(os.path.basename(c['normal_image'])))[0]
        d = os.path.join(c['save_path'], c['category'], c['defect'], stem, 'image')
        n = len(glob.glob(os.path.join(d, 'gen_ano_*.jpg')))
        total += 1
        if n >= int(c['num_gen']):
            done += 1
        else:
            print('%s|%s|%s|%s|%d' % (cfg, c['category'], c['defect'], stem, n))
print('#done=%d total=%d' % (done, total), file=sys.stderr)
