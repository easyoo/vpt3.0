#!/bin/bash
# 队列 B：norm（FIX-12）5 类 × 35 轮。串行执行，单进程约 8.4GB。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
mkdir -p logs results/norm35
for cls in $CLASSES; do
  echo "[$(date +%F\ %H:%M:%S)] norm / $cls / 35ep"
  timeout 3600 $PY code_norm/main-fewshot-norm.py --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 1 --fusion norm \
    --save_dir results/norm35 --save_name maponly \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/norm35_${cls}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] norm/$cls exit=$?"
done
echo "[$(date +%F\ %H:%M:%S)] QUEUE_NORM_DONE"
