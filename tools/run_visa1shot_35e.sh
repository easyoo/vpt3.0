#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
DATA=/home/jjquan/experiment/visual2/visa_prepared
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples_visa_1shot_promptv2
SAVE=vpt37_visa1shot_promptv2
GPU="${FINAL_GPU:-1}"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
CLASSES=(candle capsules cashew chewinggum fryum macaroni1
         macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)

cd "$ROOT"
mkdir -p "results/visa/$SAVE" logs

for cls in "${CLASSES[@]}"; do
  ckpt="results/visa/$SAVE/model_${cls}_35.pth"
  log="logs/train_visa_vpt37_visa1shot_promptv2_${cls}.log"
  if [[ -s "$ckpt" ]]; then
    printf 'SKIP visa1 %s\n' "$cls"
    continue
  fi
  printf 'START visa1 %s\n' "$cls"
  CODE_HASH=$(md5sum code_vpt4/main-fewshot-vpt4.py | cut -c1-12)
  {
    printf '===== [%s] train(VisA 1-shot promptv2) save=%s class=%s gpu=%s code_md5=%s =====\n' \
      "$(date +'%F %T')" "$SAVE" "$cls" "$GPU" "$CODE_HASH"
    printf 'anomaly_root=%s\n' "$ANOROOT"
  } >"$log"
  CUDA_VISIBLE_DEVICES="$GPU" timeout 3600 "$PY" code_vpt4/main-fewshot-vpt4.py \
    --dataset VisA \
    --data_path "$DATA" \
    --anomaly_data_root "$ANOROOT" \
    --train_class "$cls" \
    --shot 1 \
    --total_epochs 35 \
    --internal 5 \
    --map_only \
    --batch_size 16 \
    --seed 1 \
    --token_init_std 0.02 \
    --res_mapping 0 \
    --save_dir results/visa \
    --save_name "$SAVE" \
    --global_proto 1 \
    --w_div 0.1 \
    --w_pixel 0.0 \
    --metric_head 1 \
    --w_rank 0.02 \
    --rank_margin 0.25 \
    --rank_norm 1 >>"$log" 2>&1
  for epoch in 5 10 15 20 25 30; do
    old="results/visa/$SAVE/model_${cls}_${epoch}.pth"
    if [[ -f "$old" ]]; then
      unlink "$old"
    fi
  done
  printf 'DONE visa1 %s\n' "$cls"
done

printf 'ALL_VISA_1SHOT_DONE save=%s\n' "$SAVE"
