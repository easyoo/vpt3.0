#!/bin/bash
# [CLUSTER · 15 类正式验证] MVTec 4-shot，在 v3.7 配方上加 Sinkhorn 平衡分配聚类损失。
#   子集结果：可学习侧单独 68.18 → 71.99（+3.81）；机制检查：可用原型 2/96 → 6/96。
#   评价：**可学习分支单独**（--w 1.0），15 类，对比 baseline 74.932，目标 ≥ 76.43（+1.5）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
TAG=clnear
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 4 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 \
--w_cluster 0.05 --cluster_eps 0.05 --w_near 0.2 \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"
COMMON="--shots 4 --size 448 --w 1.0 --flip 0 --l_global 0 --data_path /home/jjquan/MVTec-AD --batch 8"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_4shot_$TAG/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$TAG:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name "m4m6r3_4shot_$TAG" > "logs/${TAG}_train_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$TAG] $c exit=$rc nan=$(grep -c nan logs/${TAG}_train_$c.log) $(date +%H:%M:%S)" >> logs/cluster_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE TAG
for grp in "bottle cable capsule carpet" "grid hazelnut leather metal_nut" \
           "pill screw tile toothbrush" "transistor wood zipper"; do
  ( for c in $grp; do one $c $((93100 + RANDOM % 100)); done ) &
done
wait
echo "[$TAG] TRAIN DONE $(date +%H:%M:%S)" >> logs/cluster_driver.log

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "g15_$1" --tag "g15:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "m4m6r3_4shot_$TAG" \
    --ckpt_dir "results/vpt4/m4m6r3_4shot_$TAG" --save_maps_dir ".maps_${TAG}_learn" $COMMON \
    > "logs/${TAG}_gen_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON TAG
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'
echo "[$TAG] DONE $(date +%H:%M:%S)" >> logs/cluster_driver.log
