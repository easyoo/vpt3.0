#!/bin/bash
# [补跑] q_d3b15_official_layers.sh 的评估+融合阶段（首次运行时 LAYERS 未导出，
#        导致 --fused_layers 为空 → 评估直接报错）。训练权重已存在，这里只做 eval+fuse。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
TAG=d3b15a
LAYERS="2,5,8,11"
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

for c in $CLS; do
  [ -s "results/vpt4/m4m6r3_2shot_${TAG}/model_${c}_35.pth" ] || { echo "缺少权重 $c"; exit 3; }
done

gen(){ U=$("$PYPATH" pick_gpu.py --owner "gen_$1" --tag "$TAG:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --shots 2 --seed 1 \
    --save "m4m6r3_2shot_${TAG}" --ckpt_dir "results/vpt4/m4m6r3_2shot_${TAG}" \
    --data_path /home/jjquan/MVTec-AD --backbone dinov3b --fused_layers "2,5,8,11" \
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
echo "[$TAG] EVAL+FUSE DONE $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
