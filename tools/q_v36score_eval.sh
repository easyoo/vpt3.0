#!/bin/bash
# [口径统一] 用 **v3.6 打分口径**（--l_global 64 --flip 1 --l_alpha 0.7 --w 0.6）重跑异常图，
#   目的：把"基线 / DINOv3-B / 两主干集成"三者放在**与靶子 v3.6(87.65) 完全相同的口径**下比较。
#   说明：v3.7 精简版（--l_global 0 --flip 0）在 MVTec 2-shot 融合为 87.51，比 v3.6 低 0.14，
#         直接拿它当基线会低估靶子、高估我们的增益。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

gen(){ # $1=class $2=tag(save) $3=ckpt_dir $4=outdir $5=backbone(可空)
  local U; U=$("$PYPATH" pick_gpu.py --owner "gen_$2_$1" --tag "$2:gen:$1") || return 1
  local BB=""; [ -n "$5" ] && BB="--backbone $5"
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "$2" \
    --ckpt_dir "$3" --save_maps_dir "$4" $BB $COMMON > "logs/lse_gen_$2_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON

echo "== 基线(ViT-B) v3.6 口径 =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} m4m6r3_2shot results/vpt4/m4m6r3_2shot .maps_baseL_all ""'
echo "== DINOv3-B v3.6 口径 =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} m4m6r3_2shot_d3b results/vpt4/m4m6r3_2shot_d3b .maps_d3bL_all dinov3b'

fuse(){ local U; U=$("$PYPATH" pick_gpu.py --owner "fuse_$1_$2" --tag "fuse:$2:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir "$2" --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/lse_fuse_v36_$(basename $2)_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse
echo "== 融合(各自 w=0.5) =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_baseL_all'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_d3bL_all'

ens(){ local U; U=$("$PYPATH" pick_gpu.py --owner "ens_$1" --tag "ens:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" \
    --map_dirs "$2" --bank_dir .errmap_nocrop448_2shot --w 0.5 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/ens_v36_$(echo $2 | tr ',' '+')_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f ens
echo "== 集成 =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {} .maps_baseL_all,.maps_d3bL_all'
echo "[v36score] DONE $(date +%H:%M:%S)" >> logs/v36score_driver.log
