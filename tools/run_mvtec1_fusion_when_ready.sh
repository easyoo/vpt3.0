#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
DATA=/home/jjquan/MVTec-AD
MAP_DIR="$ROOT/.maps_v37_mvtec1shot"
BANK672_DIR="$ROOT/.errmap672_mvtec1"
OUT_CSV="$ROOT/logs/mvtec1_mapping672_box_eq2_25.csv"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw
         tile toothbrush transistor wood zipper)

cd "$ROOT"
for _ in $(seq 1 600); do
  map_n=$(find "$MAP_DIR" -maxdepth 1 -type f -name '*.map' | wc -l)
  b672_n=$(find "$BANK672_DIR" -maxdepth 1 -type f -name '*.map' | wc -l)
  printf 'WAIT map=%s/15 bank672=%s/15\n' "$map_n" "$b672_n"
  if [[ "$map_n" -ge 15 && "$b672_n" -ge 15 ]]; then
    break
  fi
  sleep 60
done

if [[ ! -s "$MAP_DIR/zipper.map" || ! -s "$BANK672_DIR/zipper.map" ]]; then
  printf 'ERROR: MVTec 1-shot mapping/672 异常图未在等待时限内齐全\n' >&2
  exit 1
fi

OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32 \
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
