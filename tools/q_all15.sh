#!/bin/bash
# 补齐 MVTec 全部 15 类：训练 10 个缺失类（control 配方 / 35 轮 / map_only），
# 然后用推荐配置（选最优轮 + 打分融合 + 多尺度）评估全部 15 类，输出 15 类平均。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
MISSING="carpet grid leather tile bottle cable pill toothbrush transistor zipper"
ALL="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"

for cls in $MISSING; do
  if [ -f "results/control35/maponly/model_${cls}_35.pth" ]; then echo "[skip-train] $cls"; continue; fi
  echo "[$(date +%F\ %H:%M:%S)] train $cls (35ep)"
  timeout 3600 $PY code_control/main-fewshot-control.py --train_class "$cls" --total_epochs 35 --internal 5 \
    --map_only --batch_size 16 --seed 1 \
    --save_dir results/control35 --save_name maponly \
    --data_path "$MVTEC" --anomaly_data_root "$ANOROOT" \
    > "logs/control35_${cls}.log" 2>&1
  echo "[$(date +%F\ %H:%M:%S)] train/$cls exit=$?"
done

echo "[$(date +%F\ %H:%M:%S)] 训练全部完成，开始 15 类评估"
for cls in $ALL; do
  # 按 P-AP 选最优轮
  ep=$($PY - "$cls" <<'PY'
import sys, csv
cls=sys.argv[1]; rows=[]
for line in open('results/control35/maponly/metrics_maponly.csv'):
    p=line.strip().split(',')
    if len(p)>=10 and p[0]==cls: rows.append((float(p[7]),int(p[1])))
print(max(rows)[1] if rows else 35)
PY
)
  for extra in "" "--fuse_bank -1 --ms_scoring"; do
    tag=$([ -z "$extra" ] && echo base || echo rec)
    timeout 2400 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_${ep}.pth \
      --class $cls --protocol 392 --fusion sum $extra --tag 15cls-$tag \
      >> results/ms35/eval_15cls_${cls}.log 2>&1
  done
  echo "[$(date +%F\ %H:%M:%S)] eval $cls@$ep done"
done
grep -h "^  CSV" results/ms35/eval_15cls_*.log > results/ms35/all15.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] ALL15_DONE"
