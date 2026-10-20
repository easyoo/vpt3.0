#!/bin/bash
# 15 类：原版 MemoryBank 分数 × mapping 分支分数（同一遍历、保证同一样本），统一 epoch=35
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
GROUP="$1"
case "$GROUP" in
 1) CLS="carpet grid leather tile wood" ;;
 2) CLS="bottle cable capsule hazelnut metal_nut" ;;
 3) CLS="pill screw toothbrush transistor zipper" ;;
 *) echo "usage: $0 1|2|3"; exit 1 ;;
esac
for cls in $CLS; do
  timeout 3600 $PY eval_bank_combo.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --types 1,3,5 --out results/ms35/bank15.csv >> logs/bank15_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] group$GROUP $cls 完成"
done
echo "[$(date +%F\ %H:%M:%S)] BANK15_GROUP${GROUP}_DONE"
