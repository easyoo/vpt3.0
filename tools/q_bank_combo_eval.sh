#!/bin/bash
# [bank 主干对照] 用已落盘的异常图做组合（全部等权/先验，不搜权重）：
#   A: mapping(ViT-B)  ⊕ bank(DINOv3)
#   B: mapping(DINOv3) ⊕ bank(DINOv3)
#   C: mapping(集成)   ⊕ bank(DINOv3)
#   D: 四视图等权 mapping(ViT-B), mapping(DINOv3), bank(DINOv2), bank(DINOv3)
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

fn(){ local U; U=$("$PYPATH" pick_gpu.py --owner "cb_$1_$3" --tag "cb:$3:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir "$2" --bank_dir .errmap_d3b_2shot --device cuda:0 \
    > "logs/combo_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fn; export PYPATH
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fn {} .maps_baseL_all A'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fn {} .maps_d3bL_all B'

fe(){ local U; U=$("$PYPATH" pick_gpu.py --owner "ce_$1_$3" --tag "ce:$3:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" --map_dirs "$2" \
    --bank_dir "$4" --w 0.5 --equal_all "$5" --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/combo_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fe
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_baseL_all,.maps_d3bL_all C .errmap_d3b_2shot 0'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_baseL_all,.maps_d3bL_all,.errmap_nocrop448_2shot D .errmap_d3b_2shot 1'
echo "[combo] DONE $(date +%H:%M:%S)" >> logs/bank_d3b_driver.log
