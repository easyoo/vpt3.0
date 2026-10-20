#!/bin/bash
# 改善后 mapping 分支 × 原版 bank：融合规则扫描（15 类，官方指标）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
OUT=results/ms35/fuse_rules_improved.csv
for rule in improved_only bank_only sum max mm_sum mm_max zmax rank_sum rank_prod wsum:0.3 wsum:0.5 wsum:0.7 zsum zwsum:0.3 zwsum:0.5; do
  echo "=== $rule ==="
  timeout 3600 $PY fuse_eval.py --classes all --rule "$rule" --mapping_source improved --out $OUT 2>&1 | tail -1
done
echo "[$(date +%F\ %H:%M:%S)] SWEEP_IMPROVED_DONE"
