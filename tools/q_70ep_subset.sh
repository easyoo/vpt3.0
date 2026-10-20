#!/bin/bash
# [训练侧变量] 轮数：35 → 70。依据：DINOv3 训练到第 35 轮时 loss 仍在下降（最后一个 epoch 降 ~7%），
#   即预算内并未收敛；§14 也记录过"71~100% 的类最佳轮≠35"。轮数是**训练超参**，不涉及测试集调参。
#   为控制磁盘，--save_every 35（只存第 35、70 轮；默认 1 = 历史行为不变）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
SUB="bottle cable transistor screw capsule metal_nut wood"
BASE="--internal 5 --map_only --batch_size 16 --seed 1 --shot 2 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3b \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples_2shot \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_2shot_d3b70/model_${c}_70.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "d3b70:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --total_epochs 70 --save_every 35 --save_name m4m6r3_2shot_d3b70 > "logs/d3b70_train_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[d3b70] $c exit=$rc $(date +%H:%M:%S)" >> logs/d3b70_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE
( for c in bottle cable transistor screw; do one $c 84101; done ) &
( for c in capsule metal_nut wood; do one $c 84102; done ) &
wait

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "g70_$1" --tag "g70:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" \
    --save m4m6r3_2shot_d3b70 --ckpt_dir results/vpt4/m4m6r3_2shot_d3b70 \
    --save_maps_dir .maps_d3b70_sub --backbone dinov3b --pth_epoch 70 $COMMON \
    > "logs/d3b70_gen_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'

fn(){ local U; U=$("$PYPATH" pick_gpu.py --owner "f70_$1" --tag "f70:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir .maps_d3b70_sub --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/d3b70_fuse_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fn
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fn {}'

fe(){ local U; U=$("$PYPATH" pick_gpu.py --owner "e70_$1" --tag "e70:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" \
    --map_dirs .maps_baseL_all,.maps_d3b70_sub,.errmap_nocrop448_2shot --bank_dir .errmap_d3b_2shot \
    --equal_all 1 --device cuda:0 --script code_vpt4/main-fewshot-vpt4.py > "logs/d3b70_ens_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fe
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {}'
echo "[d3b70] DONE $(date +%H:%M:%S)" >> logs/d3b70_driver.log
