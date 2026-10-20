#!/bin/bash
# [损失机制 M10 · 全量] 在 DINOv3 主干上打开图像级跨图排序损失（--w_img_rank 0.02），
#   MVTec 15 类 2-shot。子集上把该模型当第 5 路视图后 7 类达到 +1.09（四视图 +0.94）。
#   训练完用 v3.6 口径评估，再与现有四视图组合对比。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
TAG=d3bimg
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 2 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3b \
--w_img_rank 0.02 \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples_2shot \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_2shot_$TAG/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$TAG:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name "m4m6r3_2shot_$TAG" > "logs/${TAG}_train_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$TAG] $c exit=$rc $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE TAG
for grp in "bottle cable capsule carpet" "grid hazelnut leather metal_nut" \
           "pill screw tile toothbrush" "transistor wood zipper"; do
  ( for c in $grp; do one $c $((83100 + RANDOM % 100)); done ) &
done
wait

COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"
gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gi_$1" --tag "$TAG:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" \
    --save "m4m6r3_2shot_$TAG" --ckpt_dir "results/vpt4/m4m6r3_2shot_$TAG" \
    --save_maps_dir ".maps_${TAG}_all" --backbone dinov3b $COMMON > "logs/lse_gen_${TAG}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON TAG
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'

v5(){ local U; U=$("$PYPATH" pick_gpu.py --owner "v5_$1" --tag "v5:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" \
    --map_dirs .maps_baseL_all,.maps_d3bL_all,.maps_d3bimg_all,.errmap_nocrop448_2shot \
    --bank_dir .errmap_d3b_2shot --equal_all 1 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/combo_V5img_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f v5
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'v5 {}'
echo "[$TAG] PIPELINE DONE $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
