#!/bin/bash
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
case "$1" in
 1) CLS="carpet grid leather tile wood" ;;
 2) CLS="bottle cable capsule hazelnut metal_nut" ;;
 3) CLS="pill screw toothbrush transistor zipper" ;;
 *) echo usage; exit 1 ;;
esac
for cls in $CLS; do
  timeout 2400 $PY cache_improved.py --ckpt results/control35/maponly/model_${cls}_35.pth --class $cls \
    >> logs/cacheimproved_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] improved $cls 完成"
done
echo "[$(date +%F\ %H:%M:%S)] IMPROVED_GROUP$1_DONE"
