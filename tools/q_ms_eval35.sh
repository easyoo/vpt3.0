#!/bin/bash
# 用生产实现(evaluation_batch_ms)在 control@35ep 的 5 类上复验多尺度+翻转打分
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
mkdir -p results/ms35
for cls in $CLASSES; do
  for extra in "" "--ms_scoring"; do
    tag=$([ -z "$extra" ] && echo base || echo ms)
    echo "[$(date +%F\ %H:%M:%S)] $cls $tag"
    timeout 1800 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_35.pth \
      --class $cls --protocol 392 --fusion sum $extra --tag $tag \
      >> results/ms35/eval_${cls}.log 2>&1
  done
done
grep -h "^  CSV" results/ms35/eval_*.log > results/ms35/production_ms35.csv
echo "[$(date +%F\ %H:%M:%S)] MS_EVAL_DONE"
