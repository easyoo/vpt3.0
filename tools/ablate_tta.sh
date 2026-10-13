#!/bin/bash
# [消融] 测试期 TTA 的视图数：0 TTA / 单视图(1.5×) / 双视图(1.5×+翻转，现状)
# 用法: SAVE=ab_base TAG=none SCALES=1.0 FLIP=0 FORCE_GPU=<uuid> bash ablate_tta.sh [cls...]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
SAVE="${SAVE:?}"; TAG="${TAG:?}"; SCALES="${SCALES:-1.5}"; FLIP="${FLIP:-0}"
shift || true
CLS_ALL="transistor screw capsule metal_nut wood"
POS=("$@"); [ ${#POS[@]} -eq 0 ] && POS=($CLS_ALL)
G="${FORCE_GPU:-$AUDIT_GPU_UUID}"
export CUDA_VISIBLE_DEVICES="$G" EXPECTED_GPU_UUID="$G" AUDIT_GPU_UUID="$G"
for cls in "${POS[@]}"; do
  $PY eval_score_v3.py --save "$SAVE" --class "$cls" --shots "${SHOTS:-2}" --size 448 \
      --data_path "${DATA:-$MVTEC}" --ckpt_dir "${CKPT:-results/vpt4/$SAVE}" \
      --scales "$SCALES" --flip "$FLIP" \
      --raw_alpha 0.5 --raw_global 512 --l_alpha 0.7 --l_global 64 --w 0.6 \
      > "logs/ablate_tta_${TAG}_${cls}.log" 2>&1
  echo "[$TAG] $cls $(grep -m1 'CSV:' logs/ablate_tta_${TAG}_${cls}.log | sed 's/.*CSV://')"
done
