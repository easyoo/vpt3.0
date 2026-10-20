#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
DATA=/home/jjquan/MVTec-AD
CKPT="$ROOT/results/vpt4/vpt37_mvtec1shot_promptv2fix_seed1"
MAP_DIR="$ROOT/.maps_v37_mvtec1shot"
BANK672_DIR="$ROOT/.errmap672_mvtec1"
LOG_DIR="$ROOT/logs/mvtec1_final_eval"
GPU="${FINAL_GPU:-1}"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw
         tile toothbrush transistor wood zipper)

mkdir -p "$MAP_DIR" "$BANK672_DIR" "$LOG_DIR"
cd "$ROOT"

wait_checkpoint() {
  local cls="$1"
  for _ in $(seq 1 480); do
    [[ -s "$CKPT/model_${cls}_35.pth" ]] && return 0
    sleep 30
  done
  printf 'ERROR: checkpoint for %s not ready\n' "$cls" >&2
  return 1
}

run_mapping() {
  for cls in "${CLASSES[@]}"; do
    out="$MAP_DIR/$cls.map"
    log="$LOG_DIR/map_${cls}.log"
    if [[ -s "$out" ]]; then
      printf 'SKIP map %s\n' "$cls"
      continue
    fi
    wait_checkpoint "$cls"
    printf 'START map %s\n' "$cls"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval_score_v3.py \
      --save vpt37_mvtec1shot_promptv2fix_seed1 \
      --class "$cls" \
      --shots 1 \
      --size 448 \
      --data_path "$DATA" \
      --ckpt_dir "$CKPT" \
      --scales 1.5 \
      --flip 0 \
      --l_global 0 \
      --w 0.6 \
      --device cuda:0 \
      --save_maps_dir "$MAP_DIR" >"$log" 2>&1
    printf 'DONE map %s\n' "$cls"
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
      --shots 1 \
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
