#!/bin/bash
# [消融实验] 与 q_rotate_train_2shot.sh 完全同一套口径，唯一区别是代码目录 = code_ablate
# （code_ablate = code_vpt4 的副本，仅新增 --w_pos/--w_mse/--w_push/--w_seg 四个权重开关，默认值与原行为一致）。
#
# 用法: EXTRA_ARGS="--w_push 0" bash q_ablate.sh <save_name> [cls...]
#       若指定 FORCE_GPU=<uuid> 则固定绑卡（并发用）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh

SAVE="${1:?用法: EXTRA_ARGS='...' bash q_ablate.sh <save_name> [cls...]}"; shift
CLS_DEFAULT="transistor screw capsule metal_nut wood"
POS=("$@")
[ ${#POS[@]} -eq 0 ] && POS=($CLS_DEFAULT)

CODE=code_ablate
SHOTS=2
INT="${INTERNAL:-5}"                     # [精简] INTERNAL=35 时只在最后一轮存权重/评估（省 6/7 存档）
ANOROOT_2SHOT=/home/jjquan/experiment/visual2/anomaly_samples_2shot
EXTRA="${EXTRA_ARGS:-}"

mkdir -p logs results/vpt4
MANIFEST="logs/manifest_ablate_${SAVE}.txt"
for cls in "${POS[@]}"; do
  if [ -n "${FORCE_GPU:-}" ]; then
    U="$FORCE_GPU"
  else
    U=""
    for _try in 1 2 3; do
      U=$($PY pick_gpu.py --owner $$ --tag "ablate:${SAVE}:${cls}") || U=""
      [ -n "$U" ] && break
      sleep 60
    done
    [ -z "$U" ] && { echo "拿不到 GPU，中止"; exit 4; }
  fi
  export AUDIT_GPU_UUID="$U" CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U"
  LOG="logs/ablate_train_${SAVE}_${cls}.log"
  CODE_HASH=$(md5sum ${CODE}/main-fewshot-vpt4.py | cut -c1-12)
  {
    echo "===== [$(date +'%F %T')] ablate save=$SAVE class=$cls gpu=${U:4:8} code_md5=$CODE_HASH ====="
    echo "cmd: $PY ${CODE}/main-fewshot-vpt4.py --train_class $cls --total_epochs 35 --internal $INT --map_only --batch_size 16 --seed 1 --shot 2 --token_init_std 0.02 --res_mapping 0 --anomaly_data_root $ANOROOT_2SHOT --save_dir results/vpt4 --save_name $SAVE $EXTRA"
  } >> "$LOG"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +'%F %T')" "$cls" "${U:4:8}" "$CODE_HASH" "$EXTRA" >> "$MANIFEST"
  echo "[$(date +%H:%M:%S)] $SAVE/$cls start on ${U:4:8}"
  timeout 3600 $PY ${CODE}/main-fewshot-vpt4.py --train_class "$cls" --total_epochs 35 --internal "$INT" \
    --map_only --batch_size 16 --seed 1 --shot "$SHOTS" --token_init_std 0.02 --res_mapping 0 \
    --anomaly_data_root "$ANOROOT_2SHOT" --save_dir results/vpt4 --save_name "$SAVE" $EXTRA \
    --data_path "$MVTEC" >> "$LOG" 2>&1
  rc=$?
  [ -n "${FORCE_GPU:-}" ] || $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] $SAVE/$cls 结束 exit=$rc"
done
echo "[$(date +'%F %T')] ABLATE_DONE save=$SAVE"
