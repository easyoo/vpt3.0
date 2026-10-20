#!/bin/bash
# [MVTec 4-shot · 主干对照 + 集成] 与 2-shot 完全同一套口径：
#   训练：seed 1 / shot 4 / 35 epochs / map_only / 异常合成根 = anomaly_samples（4-shot 用）
#   打分：v3.6 口径 --l_global 64 --flip 1 --l_alpha 0.7 --w 0.6；融合 w=0.5，bank 用 4-shot 缓存
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
TAG=d3b4
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 4 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3b \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_4shot_$TAG/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$TAG:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name "m4m6r3_4shot_$TAG" > "logs/${TAG}_train_$c.log" 2>&1
    rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$TAG] $c exit=$rc $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE TAG
for grp in "bottle cable capsule carpet" "grid hazelnut leather metal_nut" \
           "pill screw tile toothbrush" "transistor wood zipper"; do
  ( for c in $grp; do one $c $((61100 + RANDOM % 100)); done ) &
done
wait

COMMON="--shots 4 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"
gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "g4_$3_$1" --tag "g4:$3:$1") || return 1
  local BB=""; [ "$3" = "d3b4" ] && BB="--backbone dinov3b"
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "$2" \
    --ckpt_dir "results/vpt4/$2" --save_maps_dir "$4" $BB $COMMON > "logs/lse_gen_v36_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "== 4-shot 基线(ViT-B) v3.6 口径 =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} m4m6r3_15 base4 .maps_base4L_all'
echo "== 4-shot DINOv3-B v3.6 口径 =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} m4m6r3_4shot_d3b4 d3b4 .maps_d3b4L_all'

BANK=.errmap_nocrop448
fuse(){ local U; U=$("$PYPATH" pick_gpu.py --owner "f4_$1_$2" --tag "f4:$2:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir "$2" --bank_dir $BANK --device cuda:0 > "logs/lse_fuse_v36_4shot_$(basename $2)_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse; export BANK
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_base4L_all'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_d3b4L_all'

ens(){ local U; U=$("$PYPATH" pick_gpu.py --owner "e4_$1" --tag "e4:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" \
    --map_dirs .maps_base4L_all,.maps_d3b4L_all --bank_dir $BANK --w 0.5 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/ens_v36_4shot_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f ens
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {}'
echo "[$TAG] 4SHOT PIPELINE DONE $(date +%H:%M:%S)" >> logs/${TAG}_driver.log
