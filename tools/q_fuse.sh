#!/bin/bash
# [路径2 · FEAT-FUSE] GPU 复验：把"映射空间可学习原型"与"原始空间正常特征库"两张打分图融合
# CPU 配对实验（probe_rawbank.py，4 类）已达到 P-AP 平均 +0.060；这里用官方 adeval 指标（含 AUPRO）复验。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
mkdir -p logs results/ms35
require_gpu || exit 3
declare -A BEST=( [screw]=25 [hazelnut]=20 [wood]=10 [capsule]=10 [metal_nut]=35 )
for cls in $CLASSES; do
  ep=${BEST[$cls]}
  for cfg in "base:" "bankAll:--fuse_bank -1" "bankAll_w03:--fuse_bank -1 --fuse_w 0.3" \
             "bankAll_z:--fuse_bank -1 --fuse_z 1" "bankAll_ms:--fuse_bank -1 --ms_scoring" \
             "bankAll_w03_ms:--fuse_bank -1 --fuse_w 0.3 --ms_scoring"; do
    tag=${cfg%%:*}; extra=${cfg#*:}
    echo "[$(date +%F\ %H:%M:%S)] $cls (epoch $ep) $tag"
    timeout 1800 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_${ep}.pth \
      --class $cls --protocol 392 --fusion sum --tag $tag $extra \
      >> results/ms35/eval_fuse_${cls}.log 2>&1
  done
done
grep -h "^  CSV" results/ms35/eval_fuse_*.log > results/ms35/fuse35.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] FUSE_QUEUE_DONE"
