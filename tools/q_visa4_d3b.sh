#!/bin/bash
# [VisA 4-shot · 主干对照 + 集成] 与 MVTec 同一套口径（v3.6 打分 / 融合 w=0.5）
#   训练：12 类 · shot 4 · seed 1 · 35 epochs · map_only · 异常合成根 anomaly_samples_visa_4shot
#   评估：eval_score_v3（v3.6 口径）；bank 用既有 VisA 4-shot 缓存 .errmap_nocrop448_visa4
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PYPATH=/home/jjquan/miniconda3/envs/vpt/bin/python
DATA=/home/jjquan/experiment/visual2/visa_prepared
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples_visa_4shot
BANK=.errmap_nocrop448_visa4
CLS="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
BASE="--dataset VisA --data_path $DATA --anomaly_data_root $ANOROOT --shot 4 \
--total_epochs 35 --internal 5 --map_only --batch_size 16 --seed 1 --token_init_std 0.02 \
--res_mapping 0 --global_proto 1 --w_div 0.1 --w_pixel 0.0 --metric_head 1 \
--w_rank 0.02 --rank_margin 0.25 --rank_norm 1 --backbone dinov3b --save_dir results/visa"

one(){ local c=$1 i=$2
  [ -s "results/visa/vpt35_visa4shot_d3b/model_${c}_35.pth" ] && return 0
  for t in 1 2 3; do
    U=$("$PYPATH" pick_gpu.py --owner "$i" --tag "visaD3:$c") || { sleep 60; continue; }
    CUDA_VISIBLE_DEVICES="$U" "$PYPATH" code_vpt4/main-fewshot-vpt4.py --train_class "$c" $BASE \
      --save_name "vpt35_visa4shot_d3b" > "logs/visa4d3_train_$c.log" 2>&1
    rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1
    echo "[visaD3] $c exit=$rc $(date +%H:%M:%S)" >> logs/visa4d3_driver.log
    [ "$rc" -eq 0 ] && return 0; sleep 30
  done; }
export -f one; export PYPATH BASE
for grp in "candle capsules cashew" "chewinggum fryum macaroni1" "macaroni2 pcb1 pcb2" "pcb3 pcb4 pipe_fryum"; do
  ( for c in $grp; do one $c $((51100 + RANDOM % 100)); done ) &
done
wait

COMMON="--shots 4 --seed 1 --data_path $DATA \
--size 448 --scales 1.5 --flip 1 --l_global 64 --l_alpha 0.7 --w 0.6 \
--raw_global 512 --raw_alpha 0.5 --batch 8"
gen(){ local U; U=$("$PYPATH" pick_gpu.py --owner "gv_$3_$1" --tag "gv:$3:$1") || return 1
  local BB=""; [ "$3" = "d3b" ] && BB="--backbone dinov3b"
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" eval_score_v3.py --class "$1" --save "$2" \
    --ckpt_dir "results/visa/$2" --save_maps_dir "$4" $BB $COMMON > "logs/visa4_gen_v36_${3}_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f gen; export PYPATH COMMON DATA
echo "== VisA 4-shot 基线(ViT-B) =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} vpt35_visa4shot base .maps_visa_baseL'
echo "== VisA 4-shot DINOv3-B =="
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'gen {} vpt35_visa4shot_d3b d3b .maps_visa_d3bL'

fuse(){ local U; U=$("$PYPATH" pick_gpu.py --owner "fv_$1_$2" --tag "fv:$2:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_nocrop.py --class "$1" --size 448 --w 0.5 \
    --map_dir "$2" --bank_dir $BANK --device cuda:0 --data_path $DATA \
    > "logs/visa4_fuse_v36_$(basename $2)_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f fuse; export BANK
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_visa_baseL'
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'fuse {} .maps_visa_d3bL'

ens(){ local U; U=$("$PYPATH" pick_gpu.py --owner "ev_$1" --tag "ev:$1"); [ -z "$U" ] && return 1
  CUDA_VISIBLE_DEVICES="$U" "$PYPATH" fuse_ens_maps.py --class "$1" \
    --map_dirs .maps_visa_baseL,.maps_visa_d3bL --bank_dir $BANK --w 0.5 --device cuda:0 \
    --data_path $DATA --script code_vpt4/main-fewshot-vpt4.py > "logs/visa4_ens_v36_$1.log" 2>&1
  local rc=$?; "$PYPATH" pick_gpu.py --release "$U" >/dev/null 2>&1; return $rc; }
export -f ens
echo "$CLS" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'ens {}'
echo "[visaD3] PIPELINE DONE $(date +%H:%M:%S)" >> logs/visa4d3_driver.log
