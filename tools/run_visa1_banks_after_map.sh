#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
MAP_DIR="$ROOT/.maps_v37_visa1shot_promptv2"
cd "$ROOT"

for _ in $(seq 1 240); do
  n=$(find "$MAP_DIR" -maxdepth 1 -type f -name '*.map' | wc -l)
  printf 'WAIT visa1 mapping=%s/12\n' "$n"
  [[ "$n" -ge 12 ]] && break
  sleep 30
done

[[ -s "$MAP_DIR/pipe_fryum.map" ]] || {
  printf 'ERROR: VisA 1-shot mapping incomplete\n' >&2
  exit 1
}

FINAL_GPU="${FINAL_GPU:-1}" bash run_visa1_final_eval.sh bank672 > logs/visa1_bank672_final_driver.log 2>&1 &
p1=$!

wait "$p1"
printf 'VIS_A1_BANKS_DONE\n'
