#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
DATA=/home/jjquan/experiment/visual2/visa_prepared
CKPT="$ROOT/results/visa/vpt37_visa2shot_promptv2"
MAP_DIR="$ROOT/.maps_v37_visa2shot_promptv2"
BANK672_DIR="$ROOT/.errmap672_visa2_promptv2"
LOG_DIR="$ROOT/logs/visa2_promptv2_final_eval"
GPU="${FINAL_GPU:-1}"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
CLASSES=(
  candle capsules cashew chewinggum fryum macaroni1
  macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum
)

mkdir -p "$MAP_DIR" "$BANK672_DIR" "$LOG_DIR"
cd "$ROOT"

run_mapping() {
  for cls in "${CLASSES[@]}"; do
    out="$MAP_DIR/$cls.map"
    log="$LOG_DIR/map_${cls}.log"
    if [[ -s "$out" ]]; then
      printf 'SKIP mapping %s\n' "$cls"
      continue
    fi
    printf 'START mapping %s\n' "$cls"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval_score_v3.py \
      --save vpt37_visa2shot_promptv2 \
      --class "$cls" \
      --shots 2 \
      --size 448 \
      --data_path "$DATA" \
      --ckpt_dir "$CKPT" \
      --scales 1.5 \
      --flip 0 \
      --l_global 0 \
      --w 0.6 \
      --device cuda:0 \
      --save_maps_dir "$MAP_DIR" >"$log" 2>&1
    printf 'DONE mapping %s\n' "$cls"
  done
}

run_bank672() {
  for cls in "${CLASSES[@]}"; do
    out="$BANK672_DIR/$cls.map"
    log="$LOG_DIR/bank672_${cls}.log"
    if [[ -s "$out" ]]; then
      printf 'SKIP bank672 %s\n' "$cls"
      continue
    fi
    printf 'START bank672 %s\n' "$cls"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" cache_bank_nocrop.py \
      --class "$cls" \
      --shots 2 \
      --seed 1 \
      --size 672 \
      --data_path "$DATA" \
      --out_dir "$BANK672_DIR" \
      --device cuda:0 \
      --batch 4 \
      --bank_chunk 4 \
      --gpu_query \
      --query_bank_chunk 8192 >"$log" 2>&1
    printf 'DONE bank672 %s\n' "$cls"
  done
}

mode="${1:-all}"
case "$mode" in
  map) run_mapping ;;
  bank672) run_bank672 ;;
  all)
    run_mapping
    run_bank672
    ;;
  *)
    printf 'usage: %s {map|bank672|all}\n' "$0" >&2
    exit 2
    ;;
esac
