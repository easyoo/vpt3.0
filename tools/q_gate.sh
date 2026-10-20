#!/bin/bash
# 路径2 候选：浅层残差门控（--fusion gate）
# 与 FIX-12（硬性逐层归一化）不同：门控初值 0 → 起点与 control 逐位等价，不会比现状更差；
# 只有梯度真的认为浅层有帮助时才会打开。5 类 × 35 轮，训练后用 base/ms 两套打分评估。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
mkdir -p logs results/gate35 results/ms35
require_gpu || exit 3
for cls in $CLASSES; do
  echo "[$(date +%F\ %H:%M:%S)] gate / $cls / 35ep"
  timeout 3600 $PY code_gate/main-fewshot-gate.py --train_class "$cls" --total_epochs 35 \
    --internal 5 --map_only --batch_size 16 --seed 1 --fusion gate \
    --save_dir results/gate35 --save_name maponly \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/gate35_${cls}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] gate/$cls exit=$?"
done
for cls in $CLASSES; do
  for extra in "" "--ms_scoring"; do
    tag=$([ -z "$extra" ] && echo base || echo ms)
    timeout 1800 $PY eval_offline.py --ckpt results/gate35/maponly/model_${cls}_35.pth \
      --class $cls --protocol 392 --fusion gate $extra --tag gate-${tag} \
      >> results/ms35/eval_gate_${cls}.log 2>&1
  done
done
grep -h "^  CSV" results/ms35/eval_gate_*.log > results/ms35/gate35.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] GATE_QUEUE_DONE"
