#!/bin/bash
# 缓存 mapping 分支异常图（bank 图已缓存，会自动复用），按组并行
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
  timeout 2400 $PY cache_maps.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --skip_bank >> logs/cachemaps_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] cache $cls 完成"
done
echo "[$(date +%F\ %H:%M:%S)] CACHE_GROUP$1_DONE"
