#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
SRC="$ROOT/anostyler_visa/1shot_promptv2/genbyAnoStyler"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
cd "$ROOT"

for _ in $(seq 1 480); do
  n=$(find "$SRC" -type f -name 'gen_ano_*.jpg' 2>/dev/null | wc -l)
  printf 'WAIT visa1_synth=%s/420\n' "$n"
  [[ "$n" -ge 420 ]] && break
  sleep 60
done

[[ "$(find "$SRC" -type f -name 'gen_ano_*.jpg' | wc -l)" -ge 420 ]] || {
  printf 'ERROR: VisA 1-shot synthesis incomplete\n' >&2
  exit 1
}

"$PY" visa_build_anomaly_root.py 1 \
  "$SRC" \
  /home/jjquan/experiment/visual2/anomaly_samples_visa_1shot_promptv2/visa
FINAL_GPU="${FINAL_GPU:-1}" bash run_visa1shot_35e.sh
