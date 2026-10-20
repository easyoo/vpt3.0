#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjquan/experiment/visual/audit_mapping
DATA=/home/jjquan/MVTec-AD
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples_mvtec_1shot_promptv2
SAVE=vpt37_mvtec1shot_promptv2fix_seed1
GPU="${FINAL_GPU:-1}"
PY="${PY:-/home/jjquan/miniconda3/envs/vpt/bin/python}"
CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw
         tile toothbrush transistor wood zipper)

cd "$ROOT"
mkdir -p "results/vpt4/$SAVE" logs

for cls in "${CLASSES[@]}"; do
  ckpt="results/vpt4/$SAVE/model_${cls}_35.pth"
  log="logs/train_mvtec1shot_${SAVE}_${cls}.log"
  if [[ -s "$ckpt" ]]; then
    printf 'SKIP mvtec1 %s\n' "$cls"
    continue
  fi
  printf 'START mvtec1 %s\n' "$cls"
  CODE_HASH=$(md5sum code_vpt4/main-fewshot-vpt4.py | cut -c1-12)
  {
    printf '===== [%s] train(MVTec 1-shot) save=%s class=%s gpu=%s code_md5=%s =====\n' \
      "$(date +'%F %T')" "$SAVE" "$cls" "$GPU" "$CODE_HASH"
    printf 'anomaly_root=%s\n' "$ANOROOT"
  } >"$log"
  CUDA_VISIBLE_DEVICES="$GPU" timeout 3600 "$PY" code_vpt4/main-fewshot-vpt4.py \
    --train_class "$cls" \
    --data_path "$DATA" \
    --anomaly_data_root "$ANOROOT" \
    --shot 1 \
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
    old="results/vpt4/$SAVE/model_${cls}_${epoch}.pth"
    if [[ -f "$old" ]]; then
      unlink "$old"
    fi
  done
  printf 'DONE mvtec1 %s\n' "$cls"
done

printf 'ALL_MVTEC_1SHOT_DONE save=%s\n' "$SAVE"
