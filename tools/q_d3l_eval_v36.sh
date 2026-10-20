#!/bin/bash
# [第三主干] DINOv3-L/16（15 类 2-shot）用 v3.6 口径评估 + 与 ViT-B/DINOv3-B 做多路集成
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

for c in $CLS; do
  [ -s "results/vpt4/m4m6r3_2shot_d3l15/model_${c}_35.pth" ] || { echo "缺权重 $c"; exit 3; }
done

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gl_$1" --tag "gl:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" \
    --save m4m6r3_2shot_d3l15 --ckpt_dir results/vpt4/m4m6r3_2shot_d3l15 \
    --save_maps_dir .maps_d3lL_all --backbone dinov3l --fused_layers 2,4,6,8,12,16,22 \
    $COMMON > "logs/lse_gen_v36_m4m6r3_2shot_d3l15_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'

fuse(){ local U; U=$("$PYPATH" pick_gpu.py --owner "fl_$1" --tag "fl:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir .maps_d3lL_all --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/lse_fuse_v36_.maps_d3lL_all_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {}'

ens(){ local U; U=$("$PYPATH" pick_gpu.py --owner "el_$1_$3" --tag "el:$3:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" --map_dirs "$2" \
    --bank_dir .errmap_nocrop448_2shot --w 0.5 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/ens_v36_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f ens
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {} .maps_baseL_all,.maps_d3lL_all viewD2D3L'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {} .maps_baseL_all,.maps_d3bL_all,.maps_d3lL_all viewD2D3B3L'
echo "[d3l-eval] DONE $(date +%H:%M:%S)" >> logs/d3l15_driver.log
