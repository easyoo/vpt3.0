#!/bin/bash
# 规则扫描：同一 harness 下比较各种融合方式（15 类，官方指标）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
OUT=results/ms35/fuse_rules.csv
for rule in mapping_only bank_only sum prod max zsum zprod zmax mm_sum mm_prod mm_max rank_sum rank_prod wsum:0.3 wsum:0.5 zwsum:0.3 zwsum:0.5 zwsum:0.7; do
  echo "=== $rule ==="
  timeout 3600 $PY fuse_eval.py --classes all --rule "$rule" --out $OUT 2>&1 | tail -2
done
echo "[$(date +%F\ %H:%M:%S)] SWEEP_DONE"
