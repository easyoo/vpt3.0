#!/bin/bash
# [CLUSTER · 子集验证] MVTec 4-shot，7 类子集。
#   诊断依据：训练好的 v3.7 模型里 96 个可学习原型只有 **2 个** 被正常 patch 当最近邻用到，
#   最大簇占 79.4% → patch 特征实际塌成"一个簇"，与"正常 patch 应形成若干簇"的直觉相反。
#   损失 = SwAV 式 Sinkhorn 平衡分配（行和=1 的 patch 归属、列均衡 → 各簇被等量使用）
#          + CE(log_softmax(sim), Q)。
#   评价口径：可学习分支单独（--w 1.0，即只用 learn 侧异常分数）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
SUB="bottle cable transistor screw capsule metal_nut wood"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 4 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"
COMMON="--shots 4 --size 448 --w 1.0 --flip 0 --l_global 0 --data_path /home/jjquan/MVTec-AD --batch 8"

one(){ local c=$1 tag=$2 i=$3
  [ -s "results/vpt4/m4m6r3_4shot_$tag/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$tag:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --w_cluster "$4" --cluster_eps 0.05 --save_name "m4m6r3_4shot_$tag" > "logs/cl_${tag}_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$tag] $c exit=$rc $(date +%H:%M:%S)" >> logs/cluster_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE
( for c in bottle cable transistor screw; do one $c cl005 86101 0.05; done ) &
( for c in capsule metal_nut wood; do one $c cl015 86102 0.15; done ) &
wait
echo "[CLUSTER] TRAIN DONE $(date +%H:%M:%S)" >> logs/cluster_driver.log

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gcl_$2_$1" --tag "gcl:$2:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "m4m6r3_4shot_$2" \
    --ckpt_dir "results/vpt4/m4m6r3_4shot_$2" --save_maps_dir ".maps_cl_${2}_sub" $COMMON \
    > "logs/cl_gen_${2}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} cl005'
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} cl015'
echo "[CLUSTER] DONE $(date +%H:%M:%S)" >> logs/cluster_driver.log
