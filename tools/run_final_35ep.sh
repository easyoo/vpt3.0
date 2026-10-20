#!/bin/bash
# 最终 A/B（35 轮，只跑 mapping 分支）：control vs norm(FIX-12 多层特征对齐融合)
# 两条队列并行，每条队列内部串行，避免显存爆掉（单进程约 8.4GB / 24.5GB）。
# A 队列 = control 5 类 + control 复现性对照(screw 重跑)
# B 队列 = norm 5 类
set -u
cd /home/jjquan/experiment/visual/audit_mapping
export CUDA_VISIBLE_DEVICES=GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1,::1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUDIT_DEVICE=cuda:0
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
MVTEC=/home/jjquan/MVTec-AD
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples
CLASSES="screw hazelnut wood capsule metal_nut"
mkdir -p logs results/control35 results/norm35 results/control35_seed2

run () {  # run <variant> <cls> <save_dir> <logtag> <extra...>
  local variant="$1" cls="$2" outdir="$3" logtag="$4"; shift 4
  local script="code_control/main-fewshot-control.py"
  [ "$variant" = "norm" ] && script="code_norm/main-fewshot-norm.py"
  echo "[$(date +%F\ %H:%M:%S)] $variant / $cls / 35ep / $logtag"
  timeout 3600 $PY "$script" --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 1 \
    --save_dir "$outdir" --save_name maponly \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" "$@" \
    > "logs/${logtag}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] $variant/$cls exit=$?"
}

queue_control () {
  for cls in $CLASSES; do run control "$cls" results/control35 "control35_${cls}"; done
  # 噪声地板：同一配置换 seed（只影响 tokens_per_layer/token_mapping/seg_head 的随机初始化）
  for cls in screw hazelnut metal_nut; do
    run control "$cls" results/control35_seed2 "control35_seed2_${cls}" --seed 2
  done
  echo "[$(date +%F\ %H:%M:%S)] QUEUE_CONTROL_DONE"
}

queue_norm () {
  for cls in $CLASSES; do run norm "$cls" results/norm35 "norm35_${cls}" --fusion norm; done
  echo "[$(date +%F\ %H:%M:%S)] QUEUE_NORM_DONE"
}

queue_control &
queue_norm &
wait
echo "[$(date +%F\ %H:%M:%S)] FINAL_35EP_DONE"
