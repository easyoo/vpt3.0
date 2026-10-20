#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
SRC="$ROOT/anostyler_mvtec/1shot_promptv2/genbyAnoStyler"
cd "$ROOT"

for _ in $(seq 1 480); do
  n=$(find "$SRC" -type f -name 'gen_ano_*.jpg' 2>/dev/null | wc -l)
  printf 'WAIT mvtec1_synth=%s/525\n' "$n"
  [[ "$n" -ge 525 ]] && break
  sleep 60
done

if [[ "$(find "$SRC" -type f -name 'gen_ano_*.jpg' | wc -l)" -lt 525 ]]; then
  printf 'ERROR: MVTec 1-shot synthesis did not finish\n' >&2
  exit 1
fi

/home/jjquan/miniconda3/envs/vpt/bin/python run_promptv2_generation.py \
  --config_dir "$ROOT/anostyler_visa/1shot_promptv2/configs" \
  --log_dir "$ROOT/logs/visa1_promptv2" \
  --prefix visa1 \
  --gpu GPU-3bf0407c-aa78-8aab-cd2f-1c7b3356c5be \
  --workers 3
