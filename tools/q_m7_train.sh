#!/bin/bash
# vpt4 · M7（跨图排序一致性损失）5 类筛选实验
#   m4m5r   = M1+M3 + M4(全局原型) + M5(像素BCE, w=0.2) + M7(跨图排序, w=0.2)，**关闭 M6** → 隔离 M7 的贡献
#   m4m5m6r = 同上 + M6(可学习度量头)                                          → M7 与 M6 的叠加
# 全部 mapping-only（禁用 bank），35 轮，5 类，逐类轮换空闲卡。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
CLS="transistor screw capsule metal_nut wood"

# 权重由实测分数分布标定（probe_m7_scores.py）：
#   归一化后 L_M7 ≈ 1.6（tau=IQR≈0.0025，margin=0.25 时 74% 样本对激活）
#   → 取 w_rank=0.02 使 M7 在训练首轮约占总损失的 10%，避免压过 cos/MSE/分割项
setsid nohup env EXTRA_ARGS="--global_proto 1 --w_div 0.1 --w_pixel 0.2 --metric_head 0 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1" \
  bash q_rotate_train.sh m4m5r $CLS > logs/q_m4m5r.log 2>&1 &

# 组合版（M6+M7）：需要时取消注释再跑
# setsid nohup env EXTRA_ARGS="--global_proto 1 --w_div 0.1 --w_pixel 0.2 --metric_head 1 --w_rank 0.02 --rank_margin 0.25 --rank_norm 1" \
#   bash q_rotate_train.sh m4m5m6r $CLS > logs/q_m4m5m6r.log 2>&1 &

echo "[$(date +'%F %T')] M7 实验已入队（m4m5r）"
