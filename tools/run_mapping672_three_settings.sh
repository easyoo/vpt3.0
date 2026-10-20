#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
cd "$ROOT"

run_one() {
  local tag="$1"
  local data="$2"
  local classes="$3"
  local map_dir="$4"
  local bank672_dir="$5"
  "$PY" sweep_fusion_advanced.py \
    --classes "$classes" \
    --map_dir "$map_dir" \
    --bank448_dir "$bank672_dir" \
    --size 448 \
    --data_path "$data" \
    --device cpu \
    --image_top_ratio 0.005 \
    --only_rules box_eq2_25 \
    --full_metrics \
    --out_csv "logs/mapping672_${tag}_box_eq2_25.csv"
}

run_one mv2 /home/jjquan/MVTec-AD \
  'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper' \
  .maps_simple_nocrop448_mv2 .errmap672_mv2

run_one mv4 /home/jjquan/MVTec-AD \
  'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper' \
  .maps_simple_nocrop448_mv4 .errmap672_mv4

run_one visa4 /home/jjquan/experiment/visual2/visa_prepared \
  'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum' \
  .maps_simple_nocrop448_visa4 .errmap672_visa4

printf 'MAPPING672_THREE_SETTINGS_DONE\n'
