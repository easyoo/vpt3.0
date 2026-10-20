#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export CUDA_VISIBLE_DEVICES=1
cd "$ROOT"

run_group() {
  local tag="$1"
  local classes="$2"
  "$PY" sweep_fusion_advanced.py \
    --classes "$classes" \
    --map_dir .maps_v37_visa2shot_promptv2 \
    --bank448_dir .errmap672_visa2_promptv2 \
    --size 672 \
    --data_path /home/jjquan/experiment/visual2/visa_prepared \
    --image_top_ratio 0.005 \
    --z_mode none \
    --only_rules box_eq2_25 \
    --full_metrics \
    --out_csv "logs/mapping672_visa2_rawfusion_size672_${tag}.csv" \
    >"logs/mapping672_visa2_rawfusion_size672_${tag}.log" 2>&1
}

run_group g1 'candle capsules cashew chewinggum' &
run_group g2 'fryum macaroni1 macaroni2 pcb1' &
run_group g3 'pcb2 pcb3 pcb4 pipe_fryum' &
wait

head -n 1 logs/mapping672_visa2_rawfusion_size672_g1.csv \
  > logs/mapping672_visa2_rawfusion_size672_top005.csv
for tag in g1 g2 g3; do
  tail -n +2 "logs/mapping672_visa2_rawfusion_size672_${tag}.csv" \
    >> logs/mapping672_visa2_rawfusion_size672_top005.csv
done

printf 'VISA2_RAW_SIZE672_DONE\n'
