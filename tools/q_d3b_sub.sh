#!/bin/bash
# [主干实验] DINOv3-B 主干 · 7 类子集流水线（与 v3.7 / §15 的 ViT-B、ViT-L 对照同协议）
#   - 训练/推理：同一份 shot 清单（seed 1, 2shot），同一 anomaly 合成样本，同一 batch
#   - bank 分支保持不变（DINOv2 ViT-B 缓存 .errmap_nocrop448_2shot），以隔离"主干"这一个变量
#   - 打分不扫最优，固定 v3.7 口径：1.5 尺度、不翻转、l_global 0、内部 w=0.6、融合 w=0.5
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
TAG=d3b
BASE="--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --shot 2 \
--token_init_std 0.02 --res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 \
--metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3b \
--anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples_2shot \
--data_path /home/jjquan/MVTec-AD --save_dir results/vpt4"

one(){ local c=$1 i=$2
  [ -s "results/vpt4/m4m6r3_2shot_$TAG/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "$TAG:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name "m4m6r3_2shot_$TAG" > "logs/d3b_train_$c.log" 2>&1
    rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[$TAG] $c exit=$rc $(date +%H:%M:%S)" >> logs/d3b_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE TAG
( for c in bottle transistor capsule wood; do one $c 73101; done ) &
( for c in cable screw metal_nut; do one $c 73102; done ) &
wait

SUB="bottle cable transistor screw capsule metal_nut wood"
gen(){ U=$("$PYPATH" pick_gpu.py --owner "gen_$1" --tag "$TAG:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --shots 2 --seed 1 \
    --save "m4m6r3_2shot_$TAG" --ckpt_dir "results/vpt4/m4m6r3_2shot_$TAG" \
    --data_path /home/jjquan/MVTec-AD --backbone dinov3b \
    --size 448 --scales 1.5 --flip 0 --l_global 0 --w 0.6 --raw_global 512 --raw_alpha 0.5 \
    --batch 8 --save_maps_dir ".maps_${TAG}_sub" > "logs/lse_gen_${TAG}_$1.log" 2>&1
  rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH SUB TAG
echo "$SUB" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'gen {}'

W=0.5; export W
fuse(){ U=$("$PYPATH" pick_gpu.py --owner "fuse_$1" --tag "$TAG:fuse:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w "$W" \
    --map_dir ".maps_${TAG}_sub" --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/lse_fuse_${TAG}_$1.log" 2>&1
  rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse
echo "$SUB" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'fuse {}'
echo "[$TAG] PIPELINE DONE $(date +%H:%M:%S)" >> logs/d3b_driver.log
