#!/bin/bash
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
for rule in mapping_only bank_only zwsum:0.3 rank_sum wsum:0.3; do
  echo "=== $rule ==="
  timeout 3600 $PY fuse_eval.py --classes all --rule $rule --mapping_source improved \
    --out results/ms35/fuse_rules_improved_GPU.csv 2>&1 | tail -1
done
echo "[$(date +%F\ %H:%M:%S)] GPU_FUSE_DONE"
