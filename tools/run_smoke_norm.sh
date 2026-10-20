#!/bin/bash
# 冒烟测试：验证 (a) fusion=sum 能逐位复现 control 的行为；(b) fusion=norm 能正常训练
set -u
cd /home/jjquan/experiment/visual/audit_mapping
export CUDA_VISIBLE_DEVICES=GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1,::1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUDIT_DEVICE=cuda:0
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
for mode in sum norm; do
  echo "[$(date +%F\ %H:%M:%S)] smoke fusion=$mode"
  timeout 900 $PY code_norm/main-fewshot-norm.py --train_class screw --total_epochs 5 --internal 5 \
    --map_only --fusion $mode --batch_size 16 --seed 1 \
    --save_dir /home/jjquan/experiment/visual/audit_mapping/results/smoke_${mode} --save_name maponly \
    --data_path /home/jjquan/MVTec-AD --anomaly_data_root /home/jjquan/experiment/visual2/anomaly_samples \
    > logs/smoke_fusion_${mode}.log 2>&1
  echo "[$(date +%F %H:%M:%S)] smoke fusion=$mode exit=$?"
done
