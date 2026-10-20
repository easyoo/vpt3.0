#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
MVTEC_CKPT="$ROOT/results/vpt4/vpt37_mvtec1shot_promptv2fix_seed1"
VISA_CKPT="$ROOT/results/visa/vpt37_visa1shot_promptv2"
VISA2_BANK672="$ROOT/.errmap672_visa2_promptv2"
cd "$ROOT"
mkdir -p "$MVTEC_CKPT" "$VISA_CKPT" "$VISA2_BANK672"

for _ in $(seq 1 600); do
  m=$(find "$MVTEC_CKPT" -maxdepth 1 -type f -name 'model_*_35.pth' 2>/dev/null | wc -l)
  v=$(find "$VISA_CKPT" -maxdepth 1 -type f -name 'model_*_35.pth' 2>/dev/null | wc -l)
  b=$(find "$VISA2_BANK672" -maxdepth 1 -type f -name '*.map' 2>/dev/null | wc -l)
  printf 'WAIT mvtec1=%s/15 visa1=%s/12 visa2_bank672=%s/12\n' "$m" "$v" "$b"
  if [[ "$m" -ge 15 && "$v" -ge 12 && "$b" -ge 12 ]]; then
    break
  fi
  sleep 60
done

[[ "$(find "$MVTEC_CKPT" -maxdepth 1 -type f -name 'model_*_35.pth' | wc -l)" -ge 15 ]] || exit 1
[[ "$(find "$VISA_CKPT" -maxdepth 1 -type f -name 'model_*_35.pth' | wc -l)" -ge 12 ]] || exit 1
[[ "$(find "$VISA2_BANK672" -maxdepth 1 -type f -name '*.map' | wc -l)" -ge 12 ]] || exit 1

FINAL_GPU="${FINAL_GPU:-1}" bash run_mvtec1_final_eval.sh all
FINAL_GPU="${FINAL_GPU:-1}" bash run_visa1_final_eval.sh all
printf 'ALL_1SHOT_EVALS_DONE\n'
