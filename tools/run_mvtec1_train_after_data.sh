#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
SRC="$ROOT/anostyler_mvtec/1shot_promptv2/genbyAnoStyler"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
cd "$ROOT"

for _ in $(seq 1 480); do
  n=$(find "$SRC" -type f -name 'gen_ano_*.jpg' 2>/dev/null | wc -l)
  printf 'WAIT mvtec1_synth=%s/525\n' "$n"
  if [[ "$n" -ge 525 ]]; then
    break
  fi
  sleep 60
done

[[ "$(find "$SRC" -type f -name 'gen_ano_*.jpg' | wc -l)" -ge 525 ]] || {
  printf 'ERROR: MVTec 1-shot synthesis incomplete\n' >&2
  exit 1
}

"$PY" build_mvtec1_anomaly_root.py
flock "$ROOT/logs/gpu1_train.lock" \
  env FINAL_GPU="${FINAL_GPU:-1}" bash run_mvtec1shot_35e.sh
