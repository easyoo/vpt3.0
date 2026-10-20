#!/bin/bash
# [PUSH-TOPK] 难负例聚合方式对照：原实现 top-10 均值（10）vs max（1）vs top-3 均值（3）。
#   在 7 类子集、MVTec 4-shot、聚类损失配方上训练；评测可学习分支单独（w=1.0，几何混合）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
SUB="bottle cable transistor screw capsule metal_nut wood"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 4 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 \
--w_cluster 0.05 --cluster_eps 0.05 \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"
COMMON="--shots 4 --size 448 --w 1.0 --flip 0 --l_global 0 --data_path /home/jjquan/MVTec-AD --batch 8 \
--l_score_mode 8 --l_softmin_tau 0.05 --proto_set unique"

one(){ local c=$1 tag=$2 k=$3 i=$4
  [ -s "results/vpt4/m4m6r3_shot4_$tag/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$tag:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --push_topk "$k" --save_name "m4m6r3_shot4_$tag" > "logs/pt_${tag}_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$tag] $c exit=$rc $(date +%H:%M:%S)" >> logs/pt_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE
( for c in bottle cable transistor; do one $c pt1 1 94101; done ) &
( for c in screw capsule metal_nut wood; do one $c pt1 1 94102; done ) &
wait
( for c in bottle cable transistor; do one $c pt3 3 94103; done ) &
( for c in screw capsule metal_nut wood; do one $c pt3 3 94104; done ) &
wait
echo "[pt] TRAIN DONE $(date +%H:%M:%S)" >> logs/pt_driver.log

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gpt_$2_$1" --tag "gpt:$2:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "m4m6r3_shot4_$2" \
    --ckpt_dir "results/vpt4/m4m6r3_shot4_$2" --save_maps_dir ".maps_pt_$2" $COMMON \
    > "logs/pt_gen_${2}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} pt1'
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} pt3'
echo "[pt] DONE $(date +%H:%M:%S)" >> logs/pt_driver.log
