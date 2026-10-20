#!/bin/bash
# [多种子 × 多主干 集成] 用 v3.6 打分口径重推 seed2 / seed3 的 mapping 异常图，
#   然后与"ViT-B(seed1) + DINOv3-B"组合成多路集成（全部等权、先验，不搜权重）。
#   背景：seed2/seed3 训练用的是同一份 shot 清单（hash 79bf97a84286），
#         所以它们与 seed1 的差别只在模型初始化/训练随机性 → 干净的多种子集成。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
CLS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
COMMON="--shots 2 --seed 1 --data_path /home/jjquan/MVTec-AD \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"

gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gen_$2_$1" --tag "$2:gen:$1") || return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "$2" \
    --ckpt_dir "results/vpt4/$2" --save_maps_dir "$3" $COMMON > "logs/lse_gen_v36_${2}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON

echo "== seed2 (v3.6 口径) =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} m4m6r3_2shot_seed2 .maps_seed2L_all'
echo "== seed3 (v3.6 口径) =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} m4m6r3_2shot_seed3 .maps_seed3L_all'

ens(){ local U; U=$("$PYPATH" pick_gpu.py --owner "ens_$1_$3" --tag "ens:$3:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" --map_dirs "$2" \
    --bank_dir .errmap_nocrop448_2shot --w 0.5 --device cuda:0 \
    --script code_vpt4/main-fewshot-vpt4.py > "logs/ens_v36_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f ens

echo "== 集成组合 =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {} .maps_baseL_all,.maps_seed2L_all d2seed12'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {} .maps_baseL_all,.maps_seed2L_all,.maps_d3bL_all d2seed12d3'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {} .maps_baseL_all,.maps_seed2L_all,.maps_seed3L_all,.maps_d3bL_all all4'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {} .maps_baseL_all,.maps_seed2L_all,.maps_seed3L_all d2seed123'
echo "[seedv36] DONE $(date +%H:%M:%S)" >> logs/seed_v36_driver.log
