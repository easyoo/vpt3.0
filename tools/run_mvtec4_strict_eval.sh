#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA=/home/jjquan/experiment/visual2/MVTec-AD_strict4
CKPT="$ROOT/results/vpt4/vpt37_mvtec4shot_strict_promptv2_seed1"
MAP_DIR="$ROOT/.maps_mvtec4_strict"
BANK_DIR="$ROOT/.errmap672_mvtec4_strict"
LOG_DIR="$ROOT/logs/mvtec4_strict_eval"
GPU="${FINAL_GPU:-1}"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
export VPT_SHOT_DIR="$ROOT/data_prep/shots_mvtec4_strict"
CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw
         tile toothbrush transistor wood zipper)
cd "$ROOT"
mkdir -p "$MAP_DIR" "$BANK_DIR" "$LOG_DIR" "$VPT_SHOT_DIR"

map_one() {
  local cls="$1"
  local out="$MAP_DIR/$cls.map"
  [[ -s "$out" ]] && { printf 'SKIP map %s\n' "$cls"; return; }
  printf 'START map %s\n' "$cls"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval_score_v3.py \
    --save vpt37_mvtec4shot_strict_promptv2_seed1 \
    --class "$cls" --shots 4 --size 448 --data_path "$DATA" \
    --ckpt_dir "$CKPT" --scales 1.5 --flip 0 --l_global 0 --w 0.6 \
    --device cuda:0 --save_maps_dir "$MAP_DIR" \
    >"$LOG_DIR/map_${cls}.log" 2>&1
  printf 'DONE map %s\n' "$cls"
}

bank_one() {
  local cls="$1"
  local out="$BANK_DIR/$cls.map"
  [[ -s "$out" ]] && { printf 'SKIP bank %s\n' "$cls"; return; }
  printf 'START bank %s\n' "$cls"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" cache_bank_nocrop.py \
    --class "$cls" --shots 4 --seed 1 --size 672 --data_path "$DATA" \
    --out_dir "$BANK_DIR" --device cuda:0 --batch 4 --bank_chunk 4 \
    --gpu_query --query_bank_chunk 8192 \
    >"$LOG_DIR/bank672_${cls}.log" 2>&1
  printf 'DONE bank %s\n' "$cls"
}

run_phase() {
  local fn="$1"
  local a=("${CLASSES[@]:0:8}")
  local b=("${CLASSES[@]:8}")
  for cls in "${a[@]}"; do "$fn" "$cls"; done &
  local pa=$!
  for cls in "${b[@]}"; do "$fn" "$cls"; done &
  local pb=$!
  wait "$pa" "$pb"
}

run_phase map_one
run_phase bank_one
printf 'ALL_MVTEC4_STRICT_EVAL_DONE\n'
