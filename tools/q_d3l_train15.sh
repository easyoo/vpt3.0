#!/bin/bash
# [主干实验 · 第三个主干] DINOv3-L/16（ViT-L，24 层，dim 1024）· MVTec 15 类 2-shot
#   层集合：按"相对深度平移"规则 [2,4,6,8,12,16,22]（与把 DINOv2 的 7 层规则平移到 24 层一致）
#   显存：ViT-L 在 400px 下 batch16 会 OOM → 用 batch 8（工程约束，非测试集调参）
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
TAG=d3l15
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 8 --seed 1 --shot 2 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3l \
--fused_layer_csv 2,4,6,8,12,16,22 \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples_2shot \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_2shot_$TAG/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$TAG:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name "m4m6r3_2shot_$TAG" > "logs/${TAG}_train_$c.log" 2>&1
    rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$TAG] $c exit=$rc $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE TAG
for grp in "bottle cable capsule carpet" "grid hazelnut leather metal_nut" \
           "pill screw tile toothbrush" "transistor wood zipper"; do
  ( for c in $grp; do one $c $((71100 + RANDOM % 100)); done ) &
done
wait
echo "[$TAG] TRAIN DONE $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
