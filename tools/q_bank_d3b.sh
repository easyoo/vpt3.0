#!/bin/bash
# [主干实验 · bank 分支] 用 DINOv3-B 特征重建 2-shot bank 异常图（除主干外，MemoryBank 逻辑不变）
#   out: .errmap_d3b_2shot/<cls>.map（与 .errmap_nocrop448_2shot 一一对应）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
one(){ local c=$1
  [ -s ".errmap_d3b_2shot/$c.map" ] && return 0
  local U; U=$("$PYPATH" pick_gpu.py --owner "bk_$c" --tag "bankD3:$c") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" cache_bank_nocrop.py --class "$c" --shots 2 --size 448 \
    --out_dir .errmap_d3b_2shot --backbone dinov3b --batch 8 --bank_chunk 4 \
    > "logs/bank_d3b_$c.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[bankD3] $c exit=$rc $(date +%H:%M:%S)" >> logs/bank_d3b_driver.log
  return $rc; }
export -f one; export PYPATH
echo "$CLS" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'one {}'
echo "[bankD3] DONE $(date +%H:%M:%S)" >> logs/bank_d3b_driver.log
