#!/bin/bash
# [主干实验] DINOv3-B · MVTec 15 类 · 2-shot · **官方稠密层规则**
#   层集合 = DINOv3 官方 dense eval 的 BackboneLayersSet.FOUR_EVEN_INTERVALS：
#     ViT/S(12) = ViT/B(12) = [2, 5, 8, 11]（0-based，见 dinov3/eval/segmentation/models/__init__.py L36-38）
#   这是**已发表、非测试集调参**的规则；对照实验 q_d3b_all15.sh 用的是"DINOv2 的 [1,2,3,4,6,8,11] 按相对深度平移"。
#   其余全部与 q_d3b_all15.sh 相同（同数据/同种子/同 bank 缓存/同打分口径）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
TAG=d3b15a
LAYERS="2,5,8,11"
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 2 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3b \
--fused_layer_csv $LAYERS \
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
  ( for c in $grp; do one $c $((91100 + RANDOM % 100)); done ) &
done
wait

gen(){ U=$("$PYPATH" pick_gpu.py --owner "gen_$1" --tag "$TAG:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --shots 2 --seed 1 \
    --save "m4m6r3_2shot_$TAG" --ckpt_dir "results/vpt4/m4m6r3_2shot_$TAG" \
    --data_path /home/jjquan/MVTec-AD --backbone dinov3b --fused_layers "$LAYERS" \
    --size 448 --scales 1.5 --flip 0 --l_global 0 --w 0.6 --raw_global 512 --raw_alpha 0.5 \
    --batch 8 --save_maps_dir ".maps_${TAG}_all" > "logs/lse_gen_${TAG}_$1.log" 2>&1
  rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'

W=0.5; export W
fuse(){ U=$("$PYPATH" pick_gpu.py --owner "fuse_$1" --tag "$TAG:fuse:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w "$W" \
    --map_dir ".maps_${TAG}_all" --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/lse_fuse_${TAG}_$1.log" 2>&1
  rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {}'
echo "[$TAG] PIPELINE DONE $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
