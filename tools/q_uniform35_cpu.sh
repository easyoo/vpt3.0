#!/bin/bash
# 统一 epoch=35（所有类别同一轮次）× 15 类 × {现状打分, 推荐2视图档}，CPU 指标
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
ALL="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
for cls in $ALL; do
  [ -f "results/control35/maponly/model_${cls}_35.pth" ] || { echo "[skip] $cls"; continue; }
  echo "[$(date +%H:%M:%S)] $cls@35 评估中…"
  # ① 现状打分
  timeout 3600 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --protocol 392 --fusion sum --device cpu --cpu_metrics --tag u35-base \
    >> results/ms35/eval_u35_${cls}.log 2>&1
  # ② 推荐档：融合 + 1.5× 缩放 + 翻转（2 次前向）
  timeout 3600 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --protocol 392 --fusion sum --device cpu --cpu_metrics \
    --fuse_bank -1 --ms_scoring --ms_scales 1.5 --ms_flip 1 --tag u35-rec2v \
    >> results/ms35/eval_u35_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] $cls@35 完成"
done
echo "[$(date +%F\ %H:%M:%S)] U35_CPU_DONE"
