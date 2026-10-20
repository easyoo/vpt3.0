#!/bin/bash
# [主干实验 · bank 分支] 为另外两个设定建 DINOv3 bank：
#   A) MVTec 4-shot  → .errmap_d3b_4shot（15 类）
#   B) VisA  4-shot  → .errmap_d3b_visa4（12 类）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
M15="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
V12="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
DATA_VISA=/home/jjquan/experiment/visual2/visa_prepared

one(){ # $1=cls $2=shots $3=out_dir $4=data_path
  [ -s "$3/$1.map" ] && return 0
  local U; U=$("$PYPATH" pick_gpu.py --owner "bk_$(basename $3)_$1" --tag "bk$(basename $3):$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" cache_bank_nocrop.py --class "$1" --shots "$2" --size 448 \
    --data_path "$4" --out_dir "$3" --backbone dinov3b --batch 8 --bank_chunk 4 \
    > "logs/bank_d3b_$(basename $3)_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[bank$(basename $3)] $1 exit=$rc $(date +%H:%M:%S)" >> logs/bank_d3b_driver.log; return $rc; }
export -f one; export PYPATH DATA_VISA

echo "== MVTec 4-shot bank(DINOv3) =="
echo "$M15" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'one {} 4 .errmap_d3b_4shot /home/jjquan/MVTec-AD'
echo "== VisA 4-shot bank(DINOv3) =="
echo "$V12" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'one {} 4 .errmap_d3b_visa4 $DATA_VISA'
echo "[bankD3-4shot/visa] DONE $(date +%H:%M:%S)" >> logs/bank_d3b_driver.log
