#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
cd "$ROOT"

FILES=(
  logs/mvtec1_mapping672_box_eq2_25.csv
  logs/mapping672_mv2_box_eq2_25.csv
  logs/mapping672_mv4_box_eq2_25.csv
  logs/visa1_mapping672_box_eq2_25.csv
  logs/visa2_promptv2_mapping672_box_eq2_25.csv
  logs/mapping672_visa4_box_eq2_25.csv
)

for _ in $(seq 1 720); do
  ready=0
  for f in "${FILES[@]}"; do
    if [[ -s "$f" && "$(wc -l < "$f")" -gt 1 ]]; then
      ready=$((ready + 1))
    fi
  done
  printf 'WAIT result_csv=%s/6\n' "$ready"
  [[ "$ready" -eq 6 ]] && break
  sleep 60
done

for f in "${FILES[@]}"; do
  if [[ ! -s "$f" || "$(wc -l < "$f")" -le 1 ]]; then
    printf 'ERROR: missing result %s\n' "$f" >&2
    exit 1
  fi
done

"$PY" audit_mapping672_cache.py
"$PY" summarize_mapping672_all.py
printf 'MAPPING672_FINAL_RESULTS_READY\n'
