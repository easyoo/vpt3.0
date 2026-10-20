#!/bin/bash
# [CLUSTER v2 · 15 类] 更多簇 + 更强平衡：learnable_num_per_layer 32→64，w_cluster 0.05→0.1。
#   动机：v1（32 token / w=0.05）只把"被用到的原型数"从 2/96 提到 6/96，簇仍太少。
#   评测：可学习分支单独（w=1.0）+ 单图打分（几何平均 dlmin·softmin）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
TAG=clL4b
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 4 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 \
--learnable_layers 4 --learnable_num_per_layer 32 --w_cluster 0.05 --cluster_eps 0.05 \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"
COMMON="--shots 4 --size 448 --w 1.0 --flip 0 --l_global 0 --data_path /home/jjquan/MVTec-AD --batch 8 \
--l_score_mode 8 --l_softmin_tau 0.05 --proto_set all --num_tokens 32"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_4shot_$TAG/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$TAG:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name "m4m6r3_4shot_$TAG" > "logs/${TAG}_train_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$TAG] $c exit=$rc $(date +%H:%M:%S)" >> logs/cluster_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE TAG
for grp in "bottle cable capsule carpet" "grid hazelnut leather metal_nut" \
           "pill screw tile toothbrush" "transistor wood zipper"; do
  ( for c in $grp; do one $c $((91100 + RANDOM % 100)); done ) &
done
wait
echo "[$TAG] TRAIN DONE $(date +%H:%M:%S)" >> logs/cluster_driver.log

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gv2_$1" --tag "gv2:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "m4m6r3_4shot_$TAG" \
    --ckpt_dir "results/vpt4/m4m6r3_4shot_$TAG" --save_maps_dir ".maps_${TAG}_learn" $COMMON \
    > "logs/${TAG}_gen_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON TAG
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'
echo "[$TAG] DONE $(date +%H:%M:%S)" >> logs/cluster_driver.log
