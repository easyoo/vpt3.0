#!/bin/bash
# 续跑：control 剩余 4 类 + fixed 全部 5 类（control/screw 已完成）
# 只跑 mapping 分支（--map_only），30 轮，bank 全程禁用
set -u
cd /home/jjquan/experiment/visual/audit_mapping
export CUDA_VISIBLE_DEVICES=GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1,::1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUDIT_DEVICE=cuda:0
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
MVTEC=/home/jjquan/MVTec-AD
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples
mkdir -p logs results/control results/fixed

run_one () {
  local variant="$1" cls="$2"
  if [ "$variant" = "control" ]; then SCRIPT=code_control/main-fewshot-control.py; else SCRIPT=code/main-fewshot-audit.py; fi
  echo "[$(date +%F\ %H:%M:%S)] === $variant / $cls / 30ep / mapping-only ==="
  timeout 3600 $PY "$SCRIPT" --train_class "$cls" --total_epochs 30 --internal 5 \
    --map_only --batch_size 16 --seed 1 \
    --save_dir /home/jjquan/experiment/visual/audit_mapping/results/$variant \
    --save_name maponly --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/${variant}_${cls}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] $variant/$cls exit=$?"
}

for cls in hazelnut wood capsule metal_nut; do run_one control "$cls"; done
for cls in screw hazelnut wood capsule metal_nut; do run_one fixed "$cls"; done
echo "[$(date +%F\ %H:%M:%S)] REST_DONE"
