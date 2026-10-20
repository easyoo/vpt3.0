#!/bin/bash
# [换家族主干 · 子集验证] CLIP ViT-B/16（LAION-2B）作为 mapping 分支主干。
#   动机：现有全部视图同属 DINO 家族、高度相关 → 集成收益饱和；且卡住的是图像级（语义）指标。
#   除主干外一切照旧（同数据/同种子/同损失/同打分口径）。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
SUB="bottle cable transistor screw capsule metal_nut wood"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 2 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone clipb \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples_2shot \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_2shot_clip/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    local U; U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "clip:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name m4m6r3_2shot_clip > "logs/clip_train_$c.log" 2>&1
    local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[clip] $c exit=$rc $(date +%H:%M:%S)" >> logs/clip_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE
( for c in bottle cable transistor screw; do one $c 85101; done ) &
( for c in capsule metal_nut wood; do one $c 85102; done ) &
wait

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gc_$1" --tag "gc:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" \
    --save m4m6r3_2shot_clip --ckpt_dir results/vpt4/m4m6r3_2shot_clip \
    --save_maps_dir .maps_clip_sub --backbone clipb $COMMON > "logs/clip_gen_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'

fn(){ local U; U=$("$PYPATH" pick_gpu.py --owner "fc_$1" --tag "fc:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir .maps_clip_sub --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/clip_fuse_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fn
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fn {}'

fe(){ local U; U=$("$PYPATH" pick_gpu.py --owner "ec_$1_$3" --tag "ec:$3:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" --map_dirs "$2" \
    --bank_dir .errmap_d3b_2shot --equal_all 1 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/clip_ens_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fe
# 四视图基线（同子集）+ 五视图（把 DINOv3 换成 CLIP）+ 六视图（都上）
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_baseL_all,.maps_d3bL_all,.errmap_nocrop448_2shot,.errmap_d3b_2shot base4'
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_baseL_all,.maps_clip_sub,.errmap_nocrop448_2shot,.errmap_d3b_2shot cd2clip'
echo "$SUB" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fe {} .maps_baseL_all,.maps_d3bL_all,.maps_clip_sub,.errmap_nocrop448_2shot,.errmap_d3b_2shot all5'
echo "[clip] DONE $(date +%H:%M:%S)" >> logs/clip_driver.log
