#!/bin/bash
# [消融实验] 对某个 ablation save 的 5 类权重做两种打分口径的评估：
#   (1) 新打分（vpt3.6, eval_score_v3.py）
#   (2) 旧打分（vpt3.5, eval_nocrop.py）
# 两次评估都用**交付代码 code_vpt4**（消融只改训练损失权重，模型结构完全一致）。
#
# 用法: SAVE=ab_base FORCE_GPU=<uuid> bash eval_ablate.sh [cls...]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
SAVE="${SAVE:?用法: SAVE=<save> bash eval_ablate.sh [cls...]}"; shift || true
CLS_DEFAULT="transistor screw capsule metal_nut wood"
POS=("$@"); [ ${#POS[@]} -eq 0 ] && POS=($CLS_DEFAULT)
SHOTS="${SHOTS:-2}"
GPU_UUID="${FORCE_GPU:-$AUDIT_GPU_UUID}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID" EXPECTED_GPU_UUID="$GPU_UUID" AUDIT_GPU_UUID="$GPU_UUID"
MAPDIR=".maps_v3_ablate_${SAVE}"
MH="${METRIC_HEAD:-1}"; GP="${GLOBAL_PROTO:-1}"      # 结构消融时须与训练一致
for cls in "${POS[@]}"; do
  $PY eval_score_v3.py --save "$SAVE" --class "$cls" --shots "$SHOTS" --size 448 \
      --data_path "$MVTEC" --ckpt_dir "results/vpt4/$SAVE" --save_maps_dir "$MAPDIR" \
      --metric_head "$MH" --global_proto "$GP" \
      --raw_alpha 0.5 --raw_global 512 --l_alpha 0.7 --l_global 64 --w 0.6 \
      > "logs/ablate_new_${SAVE}_${cls}.log" 2>&1
  echo "[new] $SAVE/$cls exit=$? $(grep -m1 'CSV:' logs/ablate_new_${SAVE}_${cls}.log | sed 's/.*CSV://')"
  $PY eval_nocrop.py --save "$SAVE" --class "$cls" --shots "$SHOTS" --size 448 \
      --data_path "$MVTEC" --ckpt_dir "results/vpt4/$SAVE" \
      --metric_head "$MH" --global_proto "$GP" \
      > "logs/ablate_old_${SAVE}_${cls}.log" 2>&1
  echo "[old] $SAVE/$cls exit=$? $(grep -m1 'CSV:' logs/ablate_old_${SAVE}_${cls}.log | sed 's/.*CSV://')"
done
