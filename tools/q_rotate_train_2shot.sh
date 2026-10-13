#!/bin/bash
# [2-shot 实验] 轮换卡训练：每个类别开始前重新挑一张空闲卡；无空闲卡时休眠 120s 重试。
#
# 与 4-shot 的 q_rotate_train.sh 的唯一区别：
#   --shot 2  +  --anomaly_data_root <2-shot AnoStyler 根目录>
# 其余（35 轮 / batch 16 / seed 1 / 同一套 vpt4.0 超参）完全一致。
#
# 用法: bash q_rotate_train_2shot.sh <save_name> [cls...]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh

SAVE="${1:?用法: q_rotate_train_2shot.sh <save_name> [cls...]}"; shift
CLS_DEFAULT="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
POS=("$@")
[ ${#POS[@]} -eq 0 ] && POS=($CLS_DEFAULT)

SHOTS=2
ANOROOT_2SHOT=/home/jjquan/experiment/visual2/anomaly_samples_2shot
EXTRA="${EXTRA_ARGS:-}"

mkdir -p logs results/vpt4
MANIFEST="logs/manifest_train_${SAVE}.txt"
for cls in "${POS[@]}"; do
  U=""
  for _try in 1 2 3; do
    U=$($PY pick_gpu.py --owner $$ --tag "train:${SAVE}:${cls}") || U=""
    [ -n "$U" ] && break
    echo "[$(date +%H:%M:%S)] 第${_try}次挑卡失败，30s 后重试"; sleep 30
  done
  if [ -z "$U" ]; then echo "[$(date +%H:%M:%S)] 严重错误：连续 3 次拿不到 GPU，中止"; exit 4; fi
  export AUDIT_GPU_UUID="$U" CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U"
  LOG="logs/train_2shot_${SAVE}_${cls}.log"
  CODE_HASH=$(md5sum code_vpt4/main-fewshot-vpt4.py | cut -c1-12)
  {
    echo "===== [$(date +'%F %T')] train(2shot) save=$SAVE class=$cls gpu=${U:4:8} owner=$$ code_md5=$CODE_HASH ====="
    echo "cmd: $PY code_vpt4/main-fewshot-vpt4.py --train_class $cls --total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot $SHOTS --token_init_std 0.02 --res_mapping 0 --anomaly_data_root $ANOROOT_2SHOT --save_dir results/vpt4 --save_name $SAVE $EXTRA"
  } >> "$LOG"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +'%F %T')" "$cls" "${U:4:8}" "$CODE_HASH" "$EXTRA" >> "$MANIFEST"
  echo "[$(date +%H:%M:%S)] train $cls (shot=$SHOTS) on ${U:4:8} (save=$SAVE)"
  timeout 3600 $PY code_vpt4/main-fewshot-vpt4.py --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 1 --shot "$SHOTS" --token_init_std 0.02 --res_mapping 0 \
    --anomaly_data_root "$ANOROOT_2SHOT" \
    --save_dir results/vpt4 --save_name "$SAVE" $EXTRA \
    --data_path "$MVTEC" >> "$LOG" 2>&1
  rc=$?
  $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] train $cls 结束 exit=$rc"
  if [ "$rc" -ne 0 ]; then echo "[$(date +%H:%M:%S)] 严重错误：$cls 训练失败(exit=$rc)，中止"; exit 5; fi
done
echo "[$(date +'%F %T')] ROTATE_TRAIN_2SHOT_DONE save=$SAVE"
