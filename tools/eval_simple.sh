#!/bin/bash
# [v3.6-simplified] 按"精简版"配置跑 mapping 打分：**关闭 learn 侧全局项 + 关闭翻转视图**
#   s_raw   = d1^0.5 · q50^0.5          （不变）
#   s_learn = dlmin                     （不再对 bank 做 token_mapping、不再算分位数）
#   m       = w · s_learn + (1−w) · s_raw，单视图（scale 1.5，无翻转）
# 用法: SAVE=<save> SHOTS=<K> DATA=<root> CKPT=<dir> TAG=<name> FORCE_GPU=<uuid> \
#       bash eval_simple.sh [cls...]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
SAVE="${SAVE:?}"; TAG="${TAG:?}"; SHOTS="${SHOTS:-2}"
MAPDIR="${MAPDIR:-.maps_simple_nocrop448_${TAG}}"
G="${FORCE_GPU:-$AUDIT_GPU_UUID}"
export CUDA_VISIBLE_DEVICES="$G" EXPECTED_GPU_UUID="$G" AUDIT_GPU_UUID="$G"
CLS_DEFAULT="transistor screw capsule metal_nut wood"
POS=("$@"); [ ${#POS[@]} -eq 0 ] && POS=($CLS_DEFAULT)
for cls in "${POS[@]}"; do
  $PY eval_score_v3.py --save "$SAVE" --class "$cls" --shots "$SHOTS" --size 448 \
      --data_path "${DATA:-$MVTEC}" --ckpt_dir "${CKPT:-results/vpt4/$SAVE}" \
      --scales "${SCALES:-1.5}" --flip "${FLIP:-0}" \
      --l_global 0 --raw_alpha 0.5 --raw_global 512 --l_alpha 0.7 --w 0.6 \
      --save_maps_dir "$MAPDIR" \
      > "logs/simple_${TAG}_${cls}.log" 2>&1
  echo "[simple] $TAG/$cls exit=$? $(grep -m1 'CSV:' logs/simple_${TAG}_${cls}.log | sed 's/.*CSV://')"
done
