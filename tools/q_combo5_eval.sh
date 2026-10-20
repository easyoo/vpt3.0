#!/bin/bash
# [视图数扫描] 在四视图基础上再加一路，全部等权（先验，不搜权重）
#   V5 : mapping(ViT-B) + mapping(DINOv3-B) + mapping(DINOv3-L) + bank(DINOv2) + bank(DINOv3)
#   V5b: mapping(ViT-B) + mapping(DINOv3-B 7层) + mapping(DINOv3-B 官方层) + bank(DINOv2) + bank(DINOv3)
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
fe(){ local U; U=$("$PYPATH" pick_gpu.py --owner "c5_$1_$3" --tag "c5:$3:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" --map_dirs "$2" \
    --bank_dir .errmap_d3b_2shot --w 0.5 --equal_all 1 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/combo_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fe; export PYPATH
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_baseL_all,.maps_d3bL_all,.maps_d3lL_all,.errmap_nocrop448_2shot V5'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_baseL_all,.maps_d3bL_all,.maps_d3b15aL_all,.errmap_nocrop448_2shot V5b'
echo "[combo5] DONE $(date +%H:%M:%S)" >> logs/bank_d3b_driver.log
