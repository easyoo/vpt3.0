#!/bin/bash
# 轮换卡训练：每个类别开始前重新挑一张**空闲**卡；无空闲卡时由 pick_gpu.py 自动休眠 120s 重试；
# 存在多张空闲卡时优先换一张（避免长时间占用同一张物理卡导致故障）。
#
# 用法（两种写法等价）：
#   [EXTRA_ARGS="--global_proto 1 --metric_head 1"] bash q_rotate_train.sh <save_name> <cls...>
#   bash q_rotate_train.sh <save_name> <cls...> -- --global_proto 1 --metric_head 1
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh

SAVE="${1:?用法: q_rotate_train.sh <save_name> <cls...> [-- <额外参数...>]}"; shift
EXTRA="${EXTRA_ARGS:-}"
POS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then                    # 修复：'--' 出现在类别列表之后也要能识别
    shift; EXTRA="$EXTRA $*"; break
  fi
  POS+=("$1"); shift
done
if [ ${#POS[@]} -eq 0 ]; then echo "错误：没有指定类别"; exit 2; fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN save=$SAVE"; echo "DRY_RUN extra=[$EXTRA]"; echo "DRY_RUN classes=[${POS[*]}]"; exit 0
fi

mkdir -p logs results/vpt4
MANIFEST="logs/manifest_train_${SAVE}.txt"
for cls in "${POS[@]}"; do
  U=""
  for _try in 1 2 3; do                       # 挑卡失败就重试，绝不静默跳过类别
    U=$($PY pick_gpu.py --owner $$ --tag "train:${SAVE}:${cls}") || U=""
    [ -n "$U" ] && break
    echo "[$(date +%H:%M:%S)] 第${_try}次挑卡失败（pick_gpu 异常退出），30s 后重试"
    sleep 30
  done
  if [ -z "$U" ]; then echo "[$(date +%H:%M:%S)] 严重错误：连续 3 次拿不到 GPU，中止本轮（避免漏跑类别）"; exit 4; fi
  export AUDIT_GPU_UUID="$U" CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U"
  LOG="logs/train_${SAVE}_${cls}.log"
  CODE_HASH=$(md5sum code_vpt4/main-fewshot-vpt4.py | cut -c1-12)
  {
    echo "===== [$(date +'%F %T')] train save=$SAVE class=$cls gpu=${U:4:8} owner=$$ code_md5=$CODE_HASH ====="
    echo "cmd: $PY code_vpt4/main-fewshot-vpt4.py --train_class $cls --total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --token_init_std 0.02 --res_mapping 0 --save_dir results/vpt4 --save_name $SAVE $EXTRA"
  } >> "$LOG"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +'%F %T')" "$cls" "${U:4:8}" "$CODE_HASH" "$EXTRA" >> "$MANIFEST"
  echo "[$(date +%H:%M:%S)] train $cls on ${U:4:8} (save=$SAVE)"
  timeout 3600 $PY code_vpt4/main-fewshot-vpt4.py --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 1 --token_init_std 0.02 --res_mapping 0 \
    --save_dir results/vpt4 --save_name "$SAVE" $EXTRA \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" >> "$LOG" 2>&1
  rc=$?
  $PY pick_gpu.py --release "$U" >/dev/null 2>&1       # 本类跑完立刻释放，下一个类重新挑卡
  echo "[$(date +%H:%M:%S)] train $cls 结束 exit=$rc"
  if [ "$rc" -ne 0 ]; then echo "[$(date +%H:%M:%S)] 严重错误：$cls 训练失败（exit=$rc），中止本轮"; exit 5; fi
done
echo "[$(date +'%F %H:%M:%S')] ROTATE_TRAIN_DONE save=$SAVE"
