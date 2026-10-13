#!/bin/bash
# vpt4 · 15 类验收跑（训练 + 评估），逐类轮换空闲卡、无空闲卡休眠 120s 重试。
#
# 用法：
#   EXTRA_ARGS="--global_proto 1 --w_div 0.1 --w_pixel 0.2 --metric_head 0" \
#     bash q_all15_vpt4.sh m4px02b
#   （EVAL_EXTRA 默认 = 训练侧同样的 --num_tokens/--res_mapping/--global_proto/--metric_head）
set -u
cd /home/jjquan/experiment/visual/audit_mapping

SAVE="${1:?用法: EXTRA_ARGS=... bash q_all15_vpt4.sh <save_name> [tag]}"
TAG="${2:-vpt4${SAVE}}"
EXTRA="${EXTRA_ARGS:?必须提供 EXTRA_ARGS（训练侧额外参数）}"
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

# 评估侧参数：从训练侧参数里挑出必须一致的四项
EVAL_EXTRA="--num_tokens 32 --res_mapping 0"
grep -q -- '--global_proto 1' <<< "$EXTRA" && EVAL_EXTRA="$EVAL_EXTRA --global_proto 1" || EVAL_EXTRA="$EVAL_EXTRA --global_proto 0"
grep -q -- '--metric_head 0' <<< "$EXTRA" && EVAL_EXTRA="$EVAL_EXTRA --metric_head 0" || EVAL_EXTRA="$EVAL_EXTRA --metric_head 1"

echo "[$(date +'%F %T')] 15 类训练开始 save=$SAVE tag=$TAG"
echo "  train-extra: $EXTRA"
echo "  eval-extra : $EVAL_EXTRA"
EXTRA_ARGS="$EXTRA" bash q_rotate_train.sh "$SAVE" $CLS
echo "[$(date +'%F %T')] 15 类训练结束，开始评估"
EXTRA_ARGS="$EVAL_EXTRA" bash q_rotate_eval.sh "$SAVE" "$TAG" $CLS
echo "[$(date +'%F %T')] 15 类验收跑完成；统计：python compare_vpt4_7m.py --all15 $SAVE:$TAG"
