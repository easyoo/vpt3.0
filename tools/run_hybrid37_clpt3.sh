#!/bin/bash
# 受控实验：只替换训练权重为 clpt3，其余严格保持 v3.7 mapping 推理配置。
set -euo pipefail
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=${PYPATH:-python}
SAVE=${SAVE:-m4m6r3_4shot_clpt3}
TAG=${TAG:-clpt3}
MAPDIR=${MAPDIR:-.maps_clpt3_hybrid37_nocrop448}
for cls in "$@"; do
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} \
  MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} \
  "$PYPATH" -u eval_score_v3.py \
    --class "$cls" \
    --save "$SAVE" \
    --ckpt_dir "results/vpt4/$SAVE" \
    --save_maps_dir "$MAPDIR" \
    --shots 4 --size 448 --data_path /home/jjquan/MVTec-AD \
    --scales 1.5 --flip 0 --l_global 0 \
    --raw_alpha 0.5 --raw_global 512 --l_alpha 0.7 --w 0.6 \
    --batch 8 --device cuda:0 \
    > "logs/hybrid37_${TAG}_${cls}.log" 2>&1
  printf '[hybrid37-%s] %s exit=0 %s\n' "$TAG" "$cls" "$(date +%H:%M:%S)"
done
