#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
DATA=/home/jjquan/experiment/visual2/visa_prepared
MAP_DIR="$ROOT/.maps_v37_visa2shot_promptv2"
BANK672_DIR="$ROOT/.errmap672_visa2_promptv2"
OUT_CSV="$ROOT/logs/visa2_promptv2_mapping672_box_eq2_25.csv"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
CLASSES=(candle capsules cashew chewinggum fryum macaroni1
         macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)

cd "$ROOT"

for _ in $(seq 1 360); do
  map_n=$(find "$MAP_DIR" -maxdepth 1 -type f -name '*.map' | wc -l)
  b672_n=$(find "$BANK672_DIR" -maxdepth 1 -type f -name '*.map' | wc -l)
  printf 'WAIT map=%s/12 bank672=%s/12\n' "$map_n" "$b672_n"
  if [[ "$map_n" -ge 12 && "$b672_n" -ge 12 ]]; then
    break
  fi
  sleep 60
done

if [[ ! -s "$MAP_DIR/pipe_fryum.map" || ! -s "$BANK672_DIR/pipe_fryum.map" ]]; then
  printf 'ERROR: mapping/672 异常图未在等待时限内齐全\n' >&2
  exit 1
fi

printf 'START mapping+672 fusion box_eq2_25 top0.5%%\n'
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "$PY" sweep_fusion_advanced.py \
  --classes "${CLASSES[*]}" \
  --map_dir "$MAP_DIR" \
  --bank448_dir "$BANK672_DIR" \
  --size 448 \
  --data_path "$DATA" \
  --image_top_ratio 0.005 \
  --only_rules box_eq2_25 \
  --full_metrics \
  --require_mapping_sidecar \
  --out_csv "$OUT_CSV"
printf 'DONE fusion csv=%s\n' "$OUT_CSV"
