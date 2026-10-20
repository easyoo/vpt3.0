#!/bin/bash
# 类内 z 归一化加权和的权重细扫（w = mapping 占比）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
case "$1" in
 1) CLS="carpet,grid,leather,tile,wood" ;;
 2) CLS="bottle,cable,capsule,hazelnut,metal_nut" ;;
 3) CLS="pill,screw,toothbrush,transistor,zipper" ;;
 *) echo usage; exit 1 ;;
esac
$PY -u fuse_sweep_cpu.py --classes $CLS --mapping_source improved \
  --rules mapping_only,bank_only,zwsum:0.10,zwsum:0.15,zwsum:0.20,zwsum:0.25,zwsum:0.30,zwsum:0.35,zwsum:0.40,wsum:0.30,wsum:0.40 \
  --out results/ms35/weight_sweep_g${1}.csv > logs/weightsweep_g${1}.log 2>&1
echo "[$(date +%H:%M:%S)] WEIGHT_SWEEP_GROUP$1_DONE"
