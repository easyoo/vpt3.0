#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA=/home/jjquan/experiment/visual2/MVTec-AD_strict4
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples_mvtec_4shot_strict
SAVE=vpt37_mvtec4shot_strict_promptv2_seed1
GPU="${FINAL_GPU:-1}"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
CODE="$ROOT/code/main-fewshot-vpt4.py"
export VPT_SHOT_DIR="$ROOT/data_prep/shots_mvtec4_strict"
cd "$ROOT"
mkdir -p "results/vpt4/$SAVE" logs "$VPT_SHOT_DIR"

train_one() {
  local cls="$1"
  local ckpt="results/vpt4/$SAVE/model_${cls}_35.pth"
  local log="logs/train_mvtec4strict_${SAVE}_${cls}.log"
  if [[ -s "$ckpt" ]]; then
    printf 'SKIP mvtec4strict %s\n' "$cls"
    return
  fi
  printf 'START mvtec4strict %s\n' "$cls"
  local code_hash
  code_hash=$(md5sum "$CODE" | cut -c1-12)
  {
    printf '===== [%s] train(MVTec strict 4-shot) save=%s class=%s gpu=%s code_md5=%s =====\n' \
      "$(date +'%F %T')" "$SAVE" "$cls" "$GPU" "$code_hash"
    printf 'data_path=%s\nanomaly_root=%s\nshot_dir=%s\n' "$DATA" "$ANOROOT" "$VPT_SHOT_DIR"
  } >"$log"
  CUDA_VISIBLE_DEVICES="$GPU" timeout 3600 "$PY" "$CODE" \
    --train_class "$cls" \
    --data_path "$DATA" \
    --anomaly_data_root "$ANOROOT" \
    --shot 4 \
    --total_epochs 35 \
    --internal 5 \
    --map_only \
    --batch_size 16 \
    --seed 1 \
    --token_init_std 0.02 \
    --res_mapping 0 \
    --save_dir results/vpt4 \
    --save_name "$SAVE" \
    --global_proto 1 \
    --w_div 0.1 \
    --w_pixel 0.0 \
    --metric_head 1 \
    --w_rank 0.02 \
    --rank_margin 0.25 \
    --rank_norm 1 >>"$log" 2>&1
  for epoch in 5 10 15 20 25 30; do
    rm -f "results/vpt4/$SAVE/model_${cls}_${epoch}.pth"
  done
  printf 'DONE mvtec4strict %s\n' "$cls"
}

queue_a=(bottle cable capsule carpet grid hazelnut leather metal_nut)
queue_b=(pill screw tile toothbrush transistor wood zipper)

( for cls in "${queue_a[@]}"; do train_one "$cls"; done ) &
pa=$!
( for cls in "${queue_b[@]}"; do train_one "$cls"; done ) &
pb=$!
wait "$pa" "$pb"
printf 'ALL_MVTEC4_STRICT_TRAIN_DONE save=%s\n' "$SAVE"
