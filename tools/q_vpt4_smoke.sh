#!/bin/bash
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
for cls in "$@"; do
  echo "[$(date +%H:%M:%S)] vpt4 train $cls (35ep, map_only)"
  timeout 3600 $PY code_vpt4/main-fewshot-vpt4.py --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 1 \
    --token_init_std 0.02 --res_mapping 0 --w_div 0.1 \
    --save_dir results/vpt4 --save_name m13 \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/vpt4_${cls}.log" 2>&1
  echo "[$(date +%H:%M:%S)] vpt4/$cls exit=$?"
done
echo "[$(date +%F\ %H:%M:%S)] VPT4_SMOKE_DONE"
