#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
cd "$ROOT"

run_group() {
  local tag="$1" classes="$2"
  "$PY" sweep_fusion_advanced.py \
    --script "$ROOT/code/main-fewshot-vpt4.py" \
    --classes "$classes" \
    --map_dir .maps_mvtec4_strict \
    --bank448_dir .errmap672_mvtec4_strict \
    --size 672 \
    --data_path /home/jjquan/experiment/visual2/MVTec-AD_strict4 \
    --image_top_ratio 0.005 \
    --z_mode none \
    --only_rules box_eq2_25 \
    --full_metrics \
    --out_csv "logs/mvtec4_strict_raw672_${tag}.csv" \
    >"logs/mvtec4_strict_raw672_${tag}.log" 2>&1
}

run_group g1 'bottle cable capsule carpet grid' &
run_group g2 'hazelnut leather metal_nut pill screw' &
run_group g3 'tile toothbrush transistor wood zipper' &
wait

head -n 1 logs/mvtec4_strict_raw672_g1.csv > logs/mvtec4_strict_raw672.csv
for tag in g1 g2 g3; do
  tail -n +2 "logs/mvtec4_strict_raw672_${tag}.csv" >> logs/mvtec4_strict_raw672.csv
done
printf 'MVTEC4_STRICT_FUSION_DONE\n'
