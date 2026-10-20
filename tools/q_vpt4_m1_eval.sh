#!/bin/bash
# vpt4 冒烟评估：沿用 vpt3.0 的 mapping 打分协议（改善后 mapping，bank 分支禁用）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
for cls in "$@"; do
  ckpt=$(ls -1 results/vpt4/m1/model_${cls}_35.pth 2>/dev/null | head -1)
  [ -z "$ckpt" ] && { echo "[skip] $cls 无 35 轮权重"; continue; }
  timeout 2400 $PY eval_offline.py --script code_vpt4/main-fewshot-vpt4.py \
    --ckpt "$ckpt" --class $cls --protocol 392 --fusion sum \
    --fuse_bank -1 --ms_scoring --ms_scales 1.5 --ms_flip 1 --tag vpt4m1 \
    >> results/vpt4/eval_m1_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] eval $cls 完成"
done
echo "[$(date +%F\ %H:%M:%S)] VPT4_EVAL_DONE"
