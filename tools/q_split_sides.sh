#!/bin/bash
# [视图分解] 每个 mapping 分支内部本来是 `w·可学习侧 + (1-w)·原始侧`（w=0.6，ViT-B 时代定的）。
#   这里把两侧分别导出成独立异常图，以便在"视图集成"框架下等权处理：
#     --w 1.0 → 纯可学习侧；--w 0.0 → 纯原始侧
#   ViT-B 与 DINOv3-B 各导一份（4 组），供离线组合。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

gen(){ # $1=cls $2=save $3=ckpt_dir $4=outdir $5=w $6=backbone
  [ -s "$4/$1.map" ] && return 0
  local U; U=$("$PYPATH" pick_gpu.py --owner "sp_$4_$1" --tag "sp:$(basename $4):$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "$2" \
    --ckpt_dir "$3" --save_maps_dir "$4" --w "$5" --backbone "$6" $COMMON \
    > "logs/split_$(basename $4)_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON

do_set(){ echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c "gen {} $1 $2 $3 $4 $5"; }
do_set m4m6r3_2shot         results/vpt4/m4m6r3_2shot         .maps_d2_learn 1.0 base
do_set m4m6r3_2shot         results/vpt4/m4m6r3_2shot         .maps_d2_raw   0.0 base
do_set m4m6r3_2shot_d3b15   results/vpt4/m4m6r3_2shot_d3b15   .maps_d3b_learn 1.0 dinov3b
do_set m4m6r3_2shot_d3b15   results/vpt4/m4m6r3_2shot_d3b15   .maps_d3b_raw   0.0 dinov3b
echo "[split] DONE $(date +%H:%M:%S)" >> logs/split_driver.log
