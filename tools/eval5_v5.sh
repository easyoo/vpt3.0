#!/bin/bash
# [vpt5.0] 5 类筛选评估：mapping-only / 测试不裁剪 448 / 官方 adeval。
# 用法: bash eval5_v5.sh <save_name>
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
SAVE="${1:?用法: bash eval5_v5.sh <save_name>}"
NT="${NUM_TOKENS:-32}"          # 与训练端 --learnable_num_per_layer 保持一致
MH="${EVAL_METRIC_HEAD:-1}"     # 与训练端 --metric_head 保持一致
GP="${EVAL_GLOBAL_PROTO:-1}"    # 与训练端 --global_proto 保持一致
SUB="${EVAL_SUBSPACE:-0}"       # 与训练端 --subspace 保持一致
LS="${EVAL_LS:-0}"              # 与训练端 --layer_scale_learn 保持一致
LL="${EVAL_LL:-1}"              # 与训练端 --learnable_layers 保持一致
SM="${EVAL_SOFTMIN:-0}"         # 与训练端 --softmin 保持一致
CLS5="transistor screw capsule metal_nut wood"
DRIVER="logs/2shot_eval_driver_5_${SAVE}.log"
log () { echo "[$(date +'%F %T')] $*" | tee -a "$DRIVER"; }

for cls in $CLS5; do
  while [ ! -s "results/vpt4/${SAVE}/model_${cls}_35.pth" ]; do log "等待 $SAVE/$cls 权重 …"; sleep 60; done
done
log "5 类权重齐全，开始评估（mapping-only）"

for cls in $CLS5; do
  if [ -n "${FORCE_GPU:-}" ]; then          # 显式绑卡（与训练共存，评估只占 3~4GB）
    U="$FORCE_GPU"
  else
    U=""
    for _t in 1 2 3; do
      U=$($PY pick_gpu.py --owner $$ --tag "eval5:${SAVE}:${cls}") || U=""
      [ -n "$U" ] && break
      sleep 60
    done
    [ -z "$U" ] && { log "严重错误：$cls 拿不到 GPU，中止"; exit 4; }
  fi
  export CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U" AUDIT_GPU_UUID="$U"
  $PY eval_nocrop.py --save "$SAVE" --class "$cls" --shots 2 --size 448 \
      --num_tokens "$NT" \
      --metric_head "$MH" --global_proto "$GP" \
      --subspace "$SUB" \
      --layer_scale_learn "$LS" \
      --learnable_layers "$LL" \
      --softmin "$SM" \
      --script code_vpt5/main-fewshot-vpt4.py \
      > "logs/eval5_${SAVE}_${cls}.log" 2>&1
  rc=$?
  [ -n "${FORCE_GPU:-}" ] || $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  metrics=$(grep -m1 "CSV:" "logs/eval5_${SAVE}_${cls}.log" | sed 's/.*CSV://')
  log "$cls exit=$rc  $metrics"
done
log "评估完成，汇总："
$PY summarize5.py "logs/eval5_${SAVE}_*.log" | tee "logs/eval5_${SAVE}_summary.txt" | tee -a "$DRIVER"
