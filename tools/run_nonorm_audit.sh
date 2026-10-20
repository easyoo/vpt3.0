#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=1
cd "$ROOT"

run_one() {
  local tag="$1"
  local data="$2"
  local classes="$3"
  local map_dir="$4"
  local bank_dir="$5"
  "$PY" sweep_fusion_advanced.py \
    --classes "$classes" \
    --map_dir "$map_dir" \
    --bank448_dir "$bank_dir" \
    --size 448 \
    --data_path "$data" \
    --image_top_ratio 0.005 \
    --z_mode none \
    --only_rules box_eq2_25 \
    --full_metrics \
    --out_csv "logs/mapping672_${tag}_rawfusion_top005.csv" \
    >"logs/mapping672_${tag}_rawfusion_top005.log" 2>&1
}

MVTEC_CLASSES='bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'
VISA_CLASSES='candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'

run_one mv1 /home/jjquan/MVTec-AD "$MVTEC_CLASSES" \
  .maps_v37_mvtec1shot .errmap672_mvtec1 &
run_one mv2 /home/jjquan/MVTec-AD "$MVTEC_CLASSES" \
  .maps_simple_nocrop448_mv2 .errmap672_mv2 &
run_one mv4 /home/jjquan/MVTec-AD "$MVTEC_CLASSES" \
  .maps_simple_nocrop448_mv4 .errmap672_mv4 &
wait

run_one visa1 /home/jjquan/experiment/visual2/visa_prepared "$VISA_CLASSES" \
  .maps_v37_visa1shot_promptv2 .errmap672_visa1_promptv2 &
run_one visa2 /home/jjquan/experiment/visual2/visa_prepared "$VISA_CLASSES" \
  .maps_v37_visa2shot_promptv2 .errmap672_visa2_promptv2 &
run_one visa4 /home/jjquan/experiment/visual2/visa_prepared "$VISA_CLASSES" \
  .maps_simple_nocrop448_visa4 .errmap672_visa4 &
wait

printf 'NONORM_AUDIT_DONE\n'
