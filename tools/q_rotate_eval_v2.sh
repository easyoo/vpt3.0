#!/bin/bash
# v2：与 q_rotate_eval.sh 相同，但挑卡时带 --priority（短任务优先）+ 15s 轮询。
# 为什么需要：训练任务每跑完一类就立刻重抢卡，评估（几分钟的短任务）会被"饿死"几十分钟。
# 机制：评估在等待期间插 prio 标记；训练在下一次挑卡时看到标记就让路，于是释放出来的卡落给评估。
# 用法同 q_rotate_eval.sh：
#   [EXTRA_ARGS="..."] bash q_rotate_eval_v2.sh <save_name> <tag> <cls...> [-- <额外参数...>]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh

SAVE="${1:?用法: q_rotate_eval_v2.sh <save_name> <tag> <cls...>}"; shift
TAG="${1:?缺少 tag}"; shift
EXTRA="${EXTRA_ARGS:-}"
POS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift; EXTRA="$EXTRA $*"; break
  fi
  POS+=("$1"); shift
done
if [ ${#POS[@]} -eq 0 ]; then echo "错误：没有指定类别"; exit 2; fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN save=$SAVE tag=$TAG"; echo "DRY_RUN extra=[$EXTRA]"; echo "DRY_RUN classes=[${POS[*]}]"; exit 0
fi

mkdir -p results/vpt4 logs
for cls in "${POS[@]}"; do
  ckpt="results/vpt4/${SAVE}/model_${cls}_35.pth"
  [ -f "$ckpt" ] || { echo "[skip] $cls 无权重 ($ckpt)"; continue; }
  U=""
  for _try in 1 2 3; do
    U=$($PY pick_gpu.py --owner $$ --priority --sleep 15 --tag "eval:${SAVE}:${cls}" --max_used_mib 3000) || U=""
    [ -n "$U" ] && break
    echo "[$(date +%H:%M:%S)] 第${_try}次挑卡失败（pick_gpu 异常退出），30s 后重试"
    sleep 30
  done
  if [ -z "$U" ]; then echo "[$(date +%H:%M:%S)] 严重错误：连续 3 次拿不到 GPU，中止本轮"; exit 4; fi
  export AUDIT_GPU_UUID="$U" CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U"
  echo "[$(date +%H:%M:%S)] eval $cls on ${U:4:8} (save=$SAVE tag=$TAG)"
  timeout 2400 $PY eval_offline.py --script code_vpt4/main-fewshot-vpt4.py \
    --ckpt "$ckpt" --class $cls --protocol 392 --fusion sum \
    --fuse_bank -1 --ms_scoring --ms_scales 1.5 --ms_flip 1 --tag "$TAG" $EXTRA \
    >> "results/vpt4/eval_${SAVE}_${cls}.log" 2>&1
  rc=$?
  $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] eval $cls 结束 exit=$rc"
done
echo "[$(date +%F\ %H:%M:%S)] ROTATE_EVAL_DONE save=$SAVE"
