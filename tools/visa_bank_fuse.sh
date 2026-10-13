#!/bin/bash
# [VisA] 跑 bank 分支并与 mapping 分支融合（与 MVTec 完全同一套口径）：
#   1) mapping：重跑并**落盘未平滑异常图**（.maps_nocrop448_visa<K>）
#   2) bank   ：原版 MemoryBank（dinov2-base 中层双 stage + faiss，K shot × 22 增强）→ .errmap_nocrop448_visa<K>
#   3) 融合   ：类内 z 标准化后 0.3·z(mapping) + 0.7·z(bank)，再单次高斯平滑 → 官方 adeval
# 用法: SAVE=vpt35_visa4shot SHOTS=4 bash visa_bank_fuse.sh
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
SAVE="${SAVE:?用法: SAVE=<save_name> SHOTS=4 bash visa_bank_fuse.sh}"
K="${SHOTS:-4}"
DATA=/home/jjquan/experiment/visual2/visa_prepared
MAPDIR=".maps_nocrop448_visa${K}"
BANKDIR=".errmap_nocrop448_visa${K}"
CLS_DEFAULT="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
shift || true                                   # 允许只传类别子集（切半并行用）
CLS="${*:-$CLS_DEFAULT}"
DRIVER="logs/visa_bank_fuse_${SAVE}.log"
log () { echo "[$(date +'%F %T')] $*" | tee -a "$DRIVER"; }

for cls in $CLS; do
  if grep -q "CSV,${cls},fused" "logs/visa_fuse_${SAVE}_${cls}.log" 2>/dev/null; then
    log "$cls 已完成，跳过"; continue
  fi
  if [ -n "${FORCE_GPU:-}" ]; then          # 固定绑卡（多开并发用，不再走 pick_gpu 独占声明）
    U="$FORCE_GPU"
  else
    U=""
    for _t in 1 2 3; do
      U=$($PY pick_gpu.py --owner $$ --tag "visabank:${SAVE}:${cls}") || U=""
      [ -n "$U" ] && break
      sleep 60
    done
    [ -z "$U" ] && { log "严重错误：$cls 拿不到 GPU，中止"; exit 4; }
  fi
  export CUDA_VISIBLE_DEVICES="$U" EXPECTED_GPU_UUID="$U" AUDIT_GPU_UUID="$U"

  # 1) mapping 分支（落盘未平滑异常图 + 记录 mapping-only 指标）
  $PY eval_nocrop.py --save "$SAVE" --class "$cls" --shots "$K" --size 448 \
      --data_path "$DATA" --ckpt_dir "results/visa/$SAVE" --save_maps_dir "$MAPDIR" \
      --script code_vpt4/main-fewshot-vpt4.py \
      > "logs/visa_map_${SAVE}_${cls}.log" 2>&1
  rc1=$?

  # 2) bank 分支（原版 MemoryBank，同 K shot × 22 增强）
  $PY cache_bank_nocrop.py --class "$cls" --shots "$K" --size 448 \
      --data_path "$DATA" --out_dir "$BANKDIR" \
      > "logs/visa_bank_${SAVE}_${cls}.log" 2>&1
  rc2=$?

  # 3) 融合 0.3/0.7 + 官方指标
  $PY fuse_nocrop.py --class "$cls" --size 448 --w 0.3 \
      --map_dir "$MAPDIR" --bank_dir "$BANKDIR" --data_path "$DATA" \
      > "logs/visa_fuse_${SAVE}_${cls}.log" 2>&1
  rc3=$?
  [ -n "${FORCE_GPU:-}" ] || $PY pick_gpu.py --release "$U" >/dev/null 2>&1
  log "$cls exit=($rc1,$rc2,$rc3)"
done
log "完成：融合逐类日志 logs/visa_fuse_${SAVE}_*.log"
