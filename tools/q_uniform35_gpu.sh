#!/bin/bash
# 统一 epoch=35 × 15 类 × {现状打分, 推荐2视图档}，GPU 官方指标（含 P-AUPRO）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
ALL="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
for cls in $ALL; do
  [ -f "results/control35/maponly/model_${cls}_35.pth" ] || { echo "[skip] $cls"; continue; }
  timeout 2400 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --protocol 392 --fusion sum --tag g35-base >> results/ms35/eval_g35_${cls}.log 2>&1
  timeout 2400 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --protocol 392 --fusion sum --fuse_bank -1 --ms_scoring \
    --ms_scales 1.5 --ms_flip 1 --tag g35-rec2v >> results/ms35/eval_g35_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] $cls@35 完成"
done
grep -h "^  CSV" results/ms35/eval_g35_*.log > results/ms35/all15_g35.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] G35_DONE"
