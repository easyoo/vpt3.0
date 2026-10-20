#!/bin/bash
# [2-shot 实验] 评估：等 15 类权重齐 → 逐类(轮换空闲卡)
#   1) mapping 分支（测试不裁剪 448，shot=2 的原始特征库）→ 落盘未平滑异常图
#   2) bank 分支（原版 MemoryBank，shot=2）
#   3) 类内 z 融合 0.3/0.7 + 官方 adeval
#   4) 汇总 15 类平均
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh

SAVE="${1:-m4m6r3_2shot}"; shift || true
MAPDIR=".maps_nocrop448_${SAVE}"
BANK_DIR=".errmap_nocrop448_2shot"
CLS_DEFAULT="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
CLS="${*:-$CLS_DEFAULT}"
TAG="${EVAL_TAG:-$(echo $CLS | tr ' ' '_' | cut -c1-24)}"
LOG=logs/2shot_eval_driver_${TAG}.log
log () { echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }

# ---- 1) 等权重齐 ----
for cls in $CLS; do
  while [ ! -s "results/vpt4/${SAVE}/model_${cls}_35.pth" ]; do
    log "等待权重 $SAVE/model_${cls}_35.pth …"; sleep 120
  done
done
log "15 类权重齐全，开始评估"

# ---- 2) 逐类评估 ----
for cls in $CLS; do
  U=""
  for _t in 1 2 3; do
    U=$($PY pick_gpu.py --owner $$ --tag "eval2shot:${cls}") || U=""
    [ -n "$U" ] && break
    log "$cls 未拿到卡，60s 后重试"; sleep 60
  done
  [ -z "$U" ] && { log "严重错误：$cls 连续 3 次拿不到 GPU，中止"; exit 4; }
  export CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U" AUDIT_GPU_UUID="$U"
  log "$cls 评估开始 (gpu=${U:4:8})"
  $PY eval_nocrop.py --save "$SAVE" --class "$cls" --shots 2 --size 448 \
      --save_maps_dir "$MAPDIR" > "logs/eval_2shot_${SAVE}_${cls}.log" 2>&1
  rc1=$?
  $PY cache_bank_nocrop.py --class "$cls" --shots 2 --size 448 \
      --out_dir "$BANK_DIR" > "logs/cachebank_2shot_${cls}.log" 2>&1
  rc2=$?
  $PY fuse_nocrop.py --class "$cls" --size 448 --map_dir "$MAPDIR" --bank_dir "$BANK_DIR" \
      > "logs/fuse_2shot_${cls}.log" 2>&1
  rc3=$?
  $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  log "$cls 完成 exit=($rc1,$rc2,$rc3)"
  if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ] || [ "$rc3" -ne 0 ]; then
    log "!! $cls 有步骤失败，请查看 logs/*_2shot_${cls}.log"
  fi
done

# ---- 3) 汇总 ----
$PY summarize_2shot.py 'logs/fuse_2shot_*.log' | tee logs/2shot_15_summary.txt | tail -25
log "评估完成，汇总见 logs/2shot_15_summary.txt"
