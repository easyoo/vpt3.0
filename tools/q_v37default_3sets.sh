#!/bin/bash
# [口径补齐] 用 **vpt3.7 默认口径**（--l_global 0 --flip 0 --w 0.6，融合仍 w=0.5）测三个数据集。
#   背景：仓库 docs/COMPARE_three_datasets.md 里的表用的是 v3.6 口径（--l_global 64 --flip 1），
#   与 v3.7 代码的默认值不一致 → 需要把"v3.7 默认口径"的三数据集数字单独测出来。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
M15="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
V12="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
VISA=/home/jjquan/experiment/visual2/visa_prepared
COMMON="--size 448 --scales 1.5 --flip 0 --l_global 0 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"     # ← vpt3.7 默认值

gen(){ # $1=cls $2=shots $3=save $4=outdir $5=data
  [ -s "$4/$1.map" ] && return 0
  local U; U=$("$PYPATH" pick_gpu.py --owner "v37_$(basename $4)_$1" --tag "v37:$(basename $4):$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --shots "$2" \
    --save "$3" --ckpt_dir "results/${6}/$3" --save_maps_dir "$4" --data_path "$5" $COMMON \
    > "logs/v37_$(basename $4)_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON VISA

echo "== MVTec 2-shot (v3.7 默认口径)：复用已有的 .maps_base15_all =="
echo "== MVTec 4-shot =="
echo "$M15" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} 4 m4m6r3_15 .maps_v37_mv4 /home/jjquan/MVTec-AD vpt4'
echo "== VisA 4-shot =="
echo "$V12" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} 4 vpt35_visa4shot .maps_v37_visa4 $VISA visa'
echo "[v37-3sets] gen DONE $(date +%H:%M:%S)" >> logs/v37_driver.log

fuse(){ local U; U=$("$PYPATH" pick_gpu.py --owner "f37_$1_$2" --tag "f37:$2:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir "$2" --bank_dir "$3" --data_path "$4" --device cuda:0 \
    > "logs/v37_fuse_$(basename $2)_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse
echo "$M15" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_v37_mv4 .errmap_nocrop448 /home/jjquan/MVTec-AD'
echo "$V12" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_v37_visa4 .errmap_nocrop448_visa4 $VISA'
echo "[v37-3sets] DONE $(date +%H:%M:%S)" >> logs/v37_driver.log
