#!/bin/bash
# [四视图等权 · 另外两个设定] 视图 = mapping(ViT-B), mapping(DINOv3-B), bank(DINOv2), bank(DINOv3)
#   A) MVTec 4-shot（15 类）  B) VisA 4-shot（12 类）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
M15="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
V12="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
DATA_VISA=/home/jjquan/experiment/visual2/visa_prepared

fe(){ # $1=cls $2=map_dirs $3=bank_dir $4=tag $5=data_path $6=out_log_prefix
  local U; U=$("$PYPATH" pick_gpu.py --owner "4v_$4_$1" --tag "4v:$4:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" --map_dirs "$2" \
    --bank_dir "$3" --w 0.5 --equal_all 1 --device cuda:0 --data_path "$5" \
    --script code_vpt4/main-fewshot-vpt4.py > "$6_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fe; export PYPATH DATA_VISA

echo "== MVTec 4-shot 四视图 =="
echo "$M15" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_base4L_all,.maps_d3b4L_all,.errmap_nocrop448 .errmap_d3b_4shot mv4  /home/jjquan/MVTec-AD logs/combov4_mv4shot'
echo "== VisA 4-shot 四视图 =="
echo "$V12" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_visa_baseL,.maps_visa_d3bL,.errmap_nocrop448_visa4 .errmap_d3b_visa4 visa4 $DATA_VISA logs/combov4_visa4'
echo "[4view-others] DONE $(date +%H:%M:%S)" >> logs/bank_d3b_driver.log
