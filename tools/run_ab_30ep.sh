#!/bin/bash
# 路径1（工程修复）A/B 实验：只跑 mapping 分支（type=1），30 轮，bank 全程禁用
#   control = 原行为（仅加 harness：map_only / 设备 / 路径 / 可选依赖 / 向量化打分）
#   fixed   = control + 3 处行为性修复（learnable token 重复追加、fused_learnable 覆盖、type1 评估分辨率）
# 用法: bash run_ab_30ep.sh [class ...]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
# 只用 GPU1（按 UUID 指定，等价于 CUDA_VISIBLE_DEVICES=1，且不会被 GPU0 的硬件错误影响）
export CUDA_VISIBLE_DEVICES=GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1,::1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUDIT_DEVICE=cuda:0
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
MVTEC=/home/jjquan/MVTec-AD
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples
CLASSES="${*:-screw hazelnut wood capsule metal_nut}"
mkdir -p logs results/control results/fixed

for variant in control fixed; do
  if [ "$variant" = "control" ]; then SCRIPT=code_control/main-fewshot-control.py; else SCRIPT=code/main-fewshot-audit.py; fi
  for cls in $CLASSES; do
    echo "[$(date +%F\ %H:%M:%S)] === $variant / $cls / 30ep / mapping-only ==="
    timeout 3600 $PY "$SCRIPT" --train_class "$cls" --total_epochs 30 --internal 5 \
      --map_only --batch_size 16 --seed 1 \
      --save_dir /home/jjquan/experiment/visual/audit_mapping/results/$variant \
      --save_name maponly --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
      > "logs/${variant}_${cls}.log" 2>&1
    echo "[$(date +%F\ %H:%M:%S)] $variant/$cls exit=$?"
  done
done
echo "[$(date +%F\ %H:%M:%S)] AB_DONE"
