#!/bin/bash
# 下一轮实验队列（GPU 恢复后运行）
#   1) 补齐打分变体：ms_flip_min / ms_flip_med / ms_flip_mean_med（多尺度一致性，目标是压掉 metal_nut 的误报）
#   2) 训练期尺度抖动变体 code_sj（35 轮，5 类）
#   3) 用生产实现评估 jitter35：base 与 ms_flip_mean
# 用法：bash q_next.sh [--skip-train]
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
mkdir -p logs results/jitter35 results/ms35 results/probe
require_gpu || exit 3

echo "[$(date +%F\ %H:%M:%S)] [1/3] 打分变体（control@35）"
for cls in $CLASSES; do
  timeout 1800 $PY probe_scale.py --ckpt results/control35/maponly/model_${cls}_35.pth \
    --class $cls --tta_flip --out results/probe/scale35_variants.csv \
    > logs/variants35_${cls}.log 2>&1
  echo "[$(date +%F\ %H:%M:%S)] variants/$cls exit=$? → $(grep -c . results/probe/scale35_variants.csv 2>/dev/null) 行"
done

if [ "${1:-}" != "--skip-train" ]; then
  echo "[$(date +%F\ %H:%M:%S)] [2/3] 训练期尺度抖动 35 轮（5 类）"
  for cls in $CLASSES; do
    timeout 3600 $PY code_sj/main-fewshot-jitter.py --train_class "$cls" --total_epochs 35 \
      --internal 5 --map_only --batch_size 16 --seed 1 --scale_jitter 1.0,1.25,1.5 \
      --save_dir results/jitter35 --save_name maponly \
      --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
      > "logs/jitter35_${cls}.log" 2>&1
    echo "[$(date +%F\ %H:%M:%S)] jitter/$cls exit=$?"
  done
  echo "[$(date +%F\ %H:%M:%S)] [3/3] 评估 jitter35（base / ms）"
  for cls in $CLASSES; do
    for extra in "" "--ms_scoring"; do
      tag=$([ -z "$extra" ] && echo base || echo ms)
      timeout 1800 $PY eval_offline.py --ckpt results/jitter35/maponly/model_${cls}_35.pth \
        --class $cls --protocol 392 --fusion sum $extra --tag jitter-$tag \
        >> results/ms35/eval_jitter_${cls}.log 2>&1
    done
  done
  grep -h "^  CSV" results/ms35/eval_jitter_*.log > results/ms35/jitter35.csv 2>/dev/null
fi

echo "[$(date +%F\ %H:%M:%S)] [4/4] 组合方案：按 P-AP 选出的最优轮 × {base, ms}"
# 逐类最优轮由 select_best_epoch.py 给出（此处写死，避免依赖实时解析）
for spec in capsule:10 hazelnut:20 metal_nut:35 screw:25 wood:10; do
  cls=${spec%%:*}; ep=${spec##*:}
  for extra in "" "--ms_scoring"; do
    tag=$([ -z "$extra" ] && echo base || echo ms)
    timeout 1800 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_${ep}.pth \
      --class $cls --protocol 392 --fusion sum $extra --tag best-${tag} \
      >> results/ms35/eval_best_${cls}.log 2>&1
  done
done
grep -h "^  CSV" results/ms35/eval_best_*.log > results/ms35/best_epoch_ms.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] NEXT_QUEUE_DONE"
