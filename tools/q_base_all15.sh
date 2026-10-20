#!/bin/bash
# [对照] v3.7 基线（DINOv2 ViT-B/14）· MVTec 15 类 2-shot · **只做评估 + 融合**
#   与 q_d3b_all15.sh 使用完全相同的 eval/fuse 命令（唯一差别是没有 --backbone dinov3b）。
#   注意：本脚本**不训练**，直接使用已存在的 v3.7 权重 results/vpt4/m4m6r3_2shot。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
TAG=base15
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

for c in $CLS; do
  [ -s "results/vpt4/m4m6r3_2shot/model_${c}_35.pth" ] || { echo "缺少权重 $c"; exit 3; }
done

gen(){ U=$("$PYPATH" pick_gpu.py --owner "gen_$1" --tag "$TAG:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --shots 2 --seed 1 \
    --save "m4m6r3_2shot" --ckpt_dir "results/vpt4/m4m6r3_2shot" \
    --data_path /home/jjquan/MVTec-AD \
    --size 448 --scales 1.5 --flip 0 --l_global 0 --w 0.6 --raw_global 512 --raw_alpha 0.5 \
    --batch 8 --save_maps_dir ".maps_${TAG}_all" > "logs/lse_gen_${TAG}_$1.log" 2>&1
  rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH TAG
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'

W=0.5; export W
fuse(){ U=$("$PYPATH" pick_gpu.py --owner "fuse_$1" --tag "$TAG:fuse:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w "$W" \
    --map_dir ".maps_${TAG}_all" --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/lse_fuse_${TAG}_$1.log" 2>&1
  rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {}'
echo "[$TAG] EVAL+FUSE DONE $(date +%H:%M:%S)" >> logs/base15_driver.log
