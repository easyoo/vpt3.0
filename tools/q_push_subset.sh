#!/bin/bash
# [损失配比 · 子集验证] DINOv3 主干下 neg_push 项比 ViT-B 大 30×（0.25 vs 0.0085），
#   怀疑训练被"推开异常"主导 → 把负推权重下调（--w_push 0.1 / 0.3）在 7 类子集上验证。
#   其余与 v3.7 完全一致；--w_push 1.0 = 原行为（已验证逐位一致）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
SUB="bottle cable transistor screw capsule metal_nut wood"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 2 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3b \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples_2shot \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

one(){ local c=$1 tag=$2 wp=$3 i=$4
  [ -s "results/vpt4/m4m6r3_2shot_$tag/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$tag:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --w_push "$wp" --save_name "m4m6r3_2shot_$tag" > "logs/push_${tag}_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$tag] $c exit=$rc $(date +%H:%M:%S)" >> logs/push_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE
( for c in bottle cable transistor; do one $c push01 0.1 41101; done ) &
( for c in screw capsule metal_nut wood; do one $c push01 0.1 41102; done ) &
wait
( for c in bottle cable transistor; do one $c push03 0.3 41103; done ) &
( for c in screw capsule metal_nut wood; do one $c push03 0.3 41104; done ) &
wait

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gp_$1_$2" --tag "gp:$2:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" \
    --save "m4m6r3_2shot_$2" --ckpt_dir "results/vpt4/m4m6r3_2shot_$2" \
    --save_maps_dir ".maps_${2}_sub" --backbone dinov3b $COMMON > "logs/push_gen_${2}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} push01'
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} push03'
echo "[push] DONE $(date +%H:%M:%S)" >> logs/push_driver.log
