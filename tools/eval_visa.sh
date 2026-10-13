#!/bin/bash
# [VisA] 逐类评估：mapping-only / 测试不裁剪 448 / 官方 adeval（与 MVTec 实验同口径）。
# 用法: SAVE=vpt35_visa4shot SHOTS=4 bash eval_visa.sh
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
SAVE="${SAVE:?用法: SAVE=<save_name> SHOTS=4 bash eval_visa.sh}"
K="${SHOTS:-4}"
DATA=/home/jjquan/experiment/visual2/visa_prepared
CLS="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
DRIVER="logs/visa_eval_driver_${SAVE}.log"
log () { echo "[$(date +'%F %T')] $*" | tee -a "$DRIVER"; }

for cls in $CLS; do
  while [ ! -s "results/visa/${SAVE}/model_${cls}_35.pth" ]; do log "等待 $SAVE/$cls 权重 …"; sleep 60; done
done
log "12 类权重齐全，开始评估（VisA ${K}-shot, mapping-only）"
for cls in $CLS; do
  U=""
  for _t in 1 2 3; do
    U=$($PY pick_gpu.py --owner $$ --tag "evalvisa:${SAVE}:${cls}") || U=""
    [ -n "$U" ] && break
    sleep 60
  done
  [ -z "$U" ] && { log "严重错误：$cls 拿不到 GPU，中止"; exit 4; }
  export CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U" AUDIT_GPU_UUID="$U"
  $PY eval_nocrop.py --save "$SAVE" --class "$cls" --shots "$K" --size 448 \
      --data_path "$DATA" --ckpt_dir "results/visa/$SAVE" \
      --script code_vpt4/main-fewshot-vpt4.py \
      > "logs/eval_visa_${SAVE}_${cls}.log" 2>&1
  rc=$?
  $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  log "$cls exit=$rc  $(grep -m1 'CSV:' logs/eval_visa_${SAVE}_${cls}.log | sed 's/.*CSV://')"
done
log "评估完成"
