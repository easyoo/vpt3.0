#!/bin/bash
# 纯 mapping（不加任何正常样本库）+ 多尺度/翻转，统一 epoch=35，15 类，官方指标
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
ALL="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
for cls in $ALL; do
  timeout 2400 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --protocol 392 --fusion sum --ms_scoring --ms_scales 1.5 --ms_flip 1 \
    --tag g35-msonly >> results/ms35/eval_g35_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] $cls@35 ms-only 完成"
done
echo "[$(date +%F\ %H:%M:%S)] MSONLY_DONE"
