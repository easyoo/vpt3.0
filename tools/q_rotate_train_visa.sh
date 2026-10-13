#!/bin/bash
# [VisA] 轮换卡训练：**用原本的 vpt3.5 代码**（code_vpt4，md5 08d8528b4a59），
# 数据 = VisA 官方 train 划分里抽的 K-shot 正常图 + 对应 AnoStyler 合成异常（每类 K×35 张）。
#
# 用法: EXTRA_ARGS="--global_proto 1 --w_div 0.1 --w_pixel 0.0 --metric_head 1 \
#                   --w_rank 0.02 --rank_margin 0.25 --rank_norm 1" \
#         SHOTS=4 bash q_rotate_train_visa.sh <save_name> [cls...]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh

SAVE="${1:?用法: SHOTS=4 bash q_rotate_train_visa.sh <save_name> [cls...]}"; shift
CLS_ALL="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
POS=("$@"); [ ${#POS[@]} -eq 0 ] && POS=($CLS_ALL)

CODE=code_vpt4                                      # ← 原本的 vpt3.5
K="${SHOTS:-4}"
DATA=/home/jjquan/experiment/visual2/visa_prepared
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples_visa_${K}shot
EXTRA="${EXTRA_ARGS:-}"

mkdir -p logs results/visa
MANIFEST="logs/manifest_train_${SAVE}.txt"
for cls in "${POS[@]}"; do
  U=""
  for _try in 1 2 3; do
    U=$($PY pick_gpu.py --owner $$ --tag "trainvisa:${SAVE}:${cls}") || U=""
    [ -n "$U" ] && break
    echo "[$(date +%H:%M:%S)] 第${_try}次挑卡失败，30s 后重试"; sleep 30
  done
  [ -z "$U" ] && { echo "[$(date +%H:%M:%S)] 连续 3 次拿不到 GPU，中止"; exit 4; }
  export AUDIT_GPU_UUID="$U" CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U"
  LOG="logs/train_visa_${SAVE}_${cls}.log"
  CODE_HASH=$(md5sum ${CODE}/main-fewshot-vpt4.py | cut -c1-12)
  {
    echo "===== [$(date +'%F %T')] train(VisA ${K}-shot) save=$SAVE class=$cls gpu=${U:4:8} owner=$$ code_md5=$CODE_HASH ====="
    echo "cmd: $PY ${CODE}/main-fewshot-vpt4.py --dataset VisA --data_path $DATA --anomaly_data_root $ANOROOT --train_class $cls --shot $K --total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --token_init_std 0.02 --res_mapping 0 --save_dir results/visa --save_name $SAVE $EXTRA"
  } >> "$LOG"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +'%F %T')" "$cls" "${U:4:8}" "$CODE_HASH" "$EXTRA" >> "$MANIFEST"
  echo "[$(date +%H:%M:%S)] train VisA $cls (shot=$K) on ${U:4:8} (save=$SAVE)"
  timeout 3600 $PY ${CODE}/main-fewshot-vpt4.py --dataset VisA --data_path "$DATA" \
    --anomaly_data_root "$ANOROOT" --train_class "$cls" --shot "$K" \
    --total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 \
    --token_init_std 0.02 --res_mapping 0 --save_dir results/visa --save_name "$SAVE" $EXTRA >> "$LOG" 2>&1
  rc=$?
  $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[$(date +%H:%M:%S)] train VisA $cls 结束 exit=$rc"
  [ "$rc" -ne 0 ] && { echo "[$(date +%H:%M:%S)] $cls 失败(exit=$rc)，中止"; exit 5; }
done
echo "[$(date +'%F %T')] ROTATE_TRAIN_VISA_DONE save=$SAVE"
