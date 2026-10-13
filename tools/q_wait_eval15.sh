#!/bin/bash
# 等到 15 类权重齐全后，用带优先级的评估脚本（v2）跑 15 类评估；期间每分钟检查一次。
# 用法：EXTRA_ARGS="..." bash q_wait_eval15.sh <save_name> <tag> [最大等待分钟=180]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
SAVE="${1:?用法: q_wait_eval15.sh <save_name> <tag> [max_min]}"
TAG="${2:?缺少 tag}"
MAXMIN="${3:-180}"
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

for i in $(seq 1 "$MAXMIN"); do
  n=0
  for c in $CLS; do [ -f "results/vpt4/${SAVE}/model_${c}_35.pth" ] && n=$((n+1)); done
  if [ "$n" -eq 15 ]; then
    echo "[$(date +'%F %T')] 15 类权重齐全，开始评估 save=$SAVE tag=$TAG"
    break
  fi
  [ $((i % 5)) -eq 0 ] && echo "[$(date +%H:%M:%S)] 等待权重 $n/15（已等 ${i} 分钟）"
  sleep 60
done
EXTRA_ARGS="${EXTRA_ARGS:-}" bash q_rotate_eval_v2.sh "$SAVE" "$TAG" $CLS
echo "[$(date +'%F %T')] 15 类评估结束 save=$SAVE"
