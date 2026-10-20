#!/bin/bash
# [完整性复核] 独立重算三个设定的"四视图等权"头条结果，写全新日志（verify4v_*），
#   与先前记录逐类对比，确认没有被并发流水线的竞态污染。
#   已先验证：fuse_ens_maps.py 同命令连跑 3 次数值逐位一致（确定性 OK）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
M15="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
V12="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
VISA=/home/jjquan/experiment/visual2/visa_prepared

fe(){ local U; U=$("$PYPATH" pick_gpu.py --owner "vf_$4_$1" --tag "vf:$4:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" --map_dirs "$2" \
    --bank_dir "$3" --equal_all 1 --device cuda:0 --data_path "$5" \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/verify4v_${4}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fe; export PYPATH VISA

echo "$M15" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'fe {} .maps_baseL_all,.maps_d3bL_all,.errmap_nocrop448_2shot .errmap_d3b_2shot mv2 /home/jjquan/MVTec-AD'
echo "$M15" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'fe {} .maps_base4L_all,.maps_d3b4L_all,.errmap_nocrop448 .errmap_d3b_4shot mv4 /home/jjquan/MVTec-AD'
echo "$V12" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'fe {} .maps_visa_baseL,.maps_visa_d3bL,.errmap_nocrop448_visa4 .errmap_d3b_visa4 visa4 $VISA'
echo "[verify4v] DONE $(date +%H:%M:%S)" >> logs/verify_driver.log
