#!/bin/bash
# [vpt5.0] 轮换卡训练：code_vpt5（含 M11 尾部排序损失）+ 2-shot 数据管线。
# 与 q_rotate_train_2shot.sh 的唯一区别是 **代码目录 = code_vpt5**（其余口径完全一致）。
#
# 用法: EXTRA_ARGS="--w_tail 0.05 --tail_topk 64" bash q_rotate_train_v5.sh <save_name> [cls...]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh

SAVE="${1:?用法: q_rotate_train_v5.sh <save_name> [cls...]}"; shift
CLS_DEFAULT="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
POS=("$@")
[ ${#POS[@]} -eq 0 ] && POS=($CLS_DEFAULT)

CODE=code_vpt5
SHOTS=2
EP="${EPOCHS:-35}"                                 # [M33] 训练轮数可覆盖（默认 35）
ANOROOT_2SHOT="${ANOROOT_OVERRIDE:-/home/jjquan/experiment/visual2/anomaly_samples_2shot}"
EXTRA="${EXTRA_ARGS:-}"

mkdir -p logs results/vpt4
MANIFEST="logs/manifest_train_${SAVE}.txt"
for cls in "${POS[@]}"; do
  U=""
  for _try in 1 2 3; do
    U=$($PY pick_gpu.py --owner $$ --tag "trainv5:${SAVE}:${cls}") || U=""
    [ -n "$U" ] && break
    echo "[$(date +%H:%M:%S)] 第${_try}次挑卡失败，30s 后重试"; sleep 30
  done
  if [ -z "$U" ]; then echo "[$(date +%H:%M:%S)] 严重错误：连续 3 次拿不到 GPU，中止"; exit 4; fi
  export AUDIT_GPU_UUID="$U" CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U"
  LOG="logs/train_v5_${SAVE}_${cls}.log"
  CODE_HASH=$(md5sum ${CODE}/main-fewshot-vpt4.py | cut -c1-12)
  {
    echo "===== [$(date +'%F %T')] train(v5) save=$SAVE class=$cls gpu=${U:4:8} owner=$$ code_md5=$CODE_HASH ====="
    echo "cmd: $PY ${CODE}/main-fewshot-vpt4.py --train_class $cls --total_epochs $EP --internal 5 --map_only --batch_size 16 --seed 1 --shot $SHOTS --token_init_std 0.02 --res_mapping 0 --anomaly_data_root $ANOROOT_2SHOT --save_dir results/vpt4 --save_name $SAVE $EXTRA"
  } >> "$LOG"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +'%F %T')" "$cls" "${U:4:8}" "$CODE_HASH" "$EXTRA" >> "$MANIFEST"
  echo "[$(date +%H:%M:%S)] train v5 $cls (shot=$SHOTS) on ${U:4:8} (save=$SAVE)"
  timeout $((60 * EP * 3)) $PY ${CODE}/main-fewshot-vpt4.py --train_class "$cls" --total_epochs "$EP" --internal 5 \
    --map_only --batch_size 16 --seed 1 --shot "$SHOTS" --token_init_std 0.02 --res_mapping 0 \
    --anomaly_data_root "$ANOROOT_2SHOT" \
    --save_dir results/vpt4 --save_name "$SAVE" $EXTRA \
    --data_path "$MVTEC" >> "$LOG" 2>&1
  rc=$?
  $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] train v5 $cls 结束 exit=$rc"
  if [ "$rc" -ne 0 ]; then echo "[$(date +%H:%M:%S)] 严重错误：$cls 训练失败(exit=$rc)，中止"; exit 5; fi
done
echo "[$(date +'%F %T')] ROTATE_TRAIN_V5_DONE save=$SAVE"
