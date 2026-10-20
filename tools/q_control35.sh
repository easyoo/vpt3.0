#!/bin/bash
# 队列 A：control 5 类 × 35 轮 + 噪声地板（seed=2 的 3 类）。
# 通过 WAIT_FOR_NORM 控制启动时机，避免与其它训练叠加导致显存爆掉。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
mkdir -p logs results/control35 results/control35_seed2
if [ "${WAIT_SEC:-0}" -gt 0 ]; then
  echo "[$(date +%F\ %H:%M:%S)] 等待 ${WAIT_SEC}s 后开始 control 队列"
  sleep "${WAIT_SEC}"
fi
for cls in $CLASSES; do
  echo "[$(date +%F\ %H:%M:%S)] control / $cls / 35ep"
  timeout 3600 $PY code_control/main-fewshot-control.py --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 1 \
    --save_dir results/control35 --save_name maponly \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/control35_${cls}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] control/$cls exit=$?"
done
for cls in screw hazelnut metal_nut; do
  echo "[$(date +%F\ %H:%M:%S)] control-seed2 / $cls / 35ep"
  timeout 3600 $PY code_control/main-fewshot-control.py --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 2 \
    --save_dir results/control35_seed2 --save_name maponly \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/control35_seed2_${cls}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] control-seed2/$cls exit=$?"
done
echo "[$(date +%F\ %H:%M:%S)] QUEUE_CONTROL_DONE"
