#!/bin/bash
# 所有实验共用的环境设置（GPU1 by UUID / 离线 / 路径）
# [FIX-13] 默认绑定用户指定的 GPU1（物理卡 UUID）。若获授权改用别的卡，只需：
#   AUDIT_GPU_UUID=GPU-xxxx bash q_next.sh
# 数字索引在本机不可用（GPU0 枚举异常会导致 CUDA_VISIBLE_DEVICES=1 静默落到别的物理卡）。
export AUDIT_GPU_UUID="${AUDIT_GPU_UUID:-GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822}"
export CUDA_VISIBLE_DEVICES="$AUDIT_GPU_UUID"
export EXPECTED_GPU_UUID="$AUDIT_GPU_UUID"                          # 进程内 UUID 断言用
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1,::1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUDIT_DEVICE=cuda:0
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
MVTEC=/home/jjquan/MVTec-AD
ANOROOT=/home/jjquan/experiment/visual2/anomaly_samples
CLASSES="screw hazelnut wood capsule metal_nut"

# [FIX-13] 训练/评估前先断言"确实绑在指定的物理卡上"。
# 背景：本机 GPU0 处于 Unknown Error，数字索引 CUDA_VISIBLE_DEVICES=1 会静默落到物理 GPU2
# （实测 torch 报 UUID=3bf0407c = GPU2，而同事的任务正在那张卡上跑）。
require_gpu () {
  "$PY" "$(dirname "${BASH_SOURCE[0]}")/gpu_guard.py" || {
    echo "[common_env] GPU 守卫失败，拒绝在该卡上启动任务"; return 1; }
}
