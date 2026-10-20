#!/bin/bash
# [增强 bank] DINOv2-base、448 分辨率、MemoryBank 逻辑不变，仅把每张 shot 的增强从 22 张提到 44 张
#   （库容量翻倍），与可学习分支保持同一主干（DINOv2）。
#   产物：.errmap_enh_4shot/<cls>.map
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
one(){ local c=$1
  [ -s ".errmap_enh_4shot/$c.map" ] && return 0
  local U; U=$("$PYPATH" pick_gpu.py --owner "enh_$c" --tag "enh:$c") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" cache_bank_nocrop.py --class "$c" --shots 4 --size 448 \
    --out_dir .errmap_enh_4shot --backbone base --aug_copies 2 --batch 8 --bank_chunk 4 \
    > "logs/bank_enh_$c.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
  echo "[bank_enh] $c exit=$rc $(date +%H:%M:%S)" >> logs/bank_enh_driver.log
  return $rc; }
export -f one; export PYPATH
echo "$CLS" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'one {}'
echo "[bank_enh] DONE $(date +%H:%M:%S)" >> logs/bank_enh_driver.log
