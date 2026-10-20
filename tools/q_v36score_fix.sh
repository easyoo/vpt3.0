#!/bin/bash
# [补跑] 上面 q_v36score_eval.sh 里 DINOv3 的 ckpt 目录写错了（15 类权重在 m4m6r3_2shot_d3b15，
#        不是只含 7 类的 m4m6r3_2shot_d3b）→ 8 个类没有生成异常图。这里用正确目录补跑，
#        然后重做 DINOv3 侧的融合与"两主干集成"。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
MISS="carpet grid hazelnut leather pill tile toothbrush zipper"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gen3_$1" --tag "fix:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" \
    --save m4m6r3_2shot_d3b15 --ckpt_dir results/vpt4/m4m6r3_2shot_d3b15 \
    --save_maps_dir .maps_d3bL_all --backbone dinov3b $COMMON > "logs/lse_gen_m4m6r3_2shot_d3b15_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON
echo "$MISS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {}'

fuse(){ local U; U=$("$PYPATH" pick_gpu.py --owner "fuse3_$1" --tag "fix:fuse:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir .maps_d3bL_all --bank_dir .errmap_nocrop448_2shot --device cuda:0 \
    > "logs/lse_fuse_v36_.maps_d3bL_all_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {}'

ens(){ local U; U=$("$PYPATH" pick_gpu.py --owner "ens3_$1" --tag "fix:ens:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" \
    --map_dirs .maps_baseL_all,.maps_d3bL_all --bank_dir .errmap_nocrop448_2shot --w 0.5 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py \
    > "logs/ens_v36_.maps_baseL_all+.maps_d3bL_all_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f ens
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {}'
echo "[fix] DONE $(date +%H:%M:%S)" >> logs/v36score_driver.log
