#!/bin/bash
# 路径1 追加：FIX-14 分割头损失（逐样本 Dice + 跳过全背景样本）× 35 轮
# 目标指标：type=2（分割头误差图）与 type=4（两者相加）；type=1 作为不受影响的对照。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
mkdir -p logs results/dice35
require_gpu || exit 3
for cls in $CLASSES; do
  echo "[$(date +%F\ %H:%M:%S)] dice / $cls / 35ep"
  timeout 3600 $PY code_dice/main-fewshot-dice.py --train_class "$cls" --total_epochs 35 \
    --internal 5 --map_only --batch_size 16 --seed 1 \
    --save_dir results/dice35 --save_name maponly \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/dice35_${cls}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] dice/$cls exit=$?"
done
# 评估三种打分（type=1 对照；type=2 分割头；type=4 相加），与 control35 做配对比较
for cls in $CLASSES; do
  for t in 1 2 4; do
    timeout 1800 $PY eval_offline.py --ckpt results/dice35/maponly/model_${cls}_35.pth \
      --class $cls --protocol 392 --fusion sum --type $t --tag dice-type$t \
      >> results/ms35/eval_dice_${cls}.log 2>&1
  done
done
grep -h "^  CSV" results/ms35/eval_dice_*.log > results/ms35/dice35.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] DICE_QUEUE_DONE"
