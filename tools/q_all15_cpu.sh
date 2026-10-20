#!/bin/bash
# GPU 故障期间的 CPU 评估：15 类 × 2 视图档（--ms_scales 1.5 --ms_flip 1 + 融合）
# 指标用 numpy 实现（与 GPU 口径配对一致）；P-AUPRO 需要 CUDA 累加器，此处为 nan。
set -u
cd /home/jjquan/experiment/visual/audit_mapping
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
ALL="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
for cls in $ALL; do
  ep=$($PY - "$cls" <<'PY'
import sys
cls=sys.argv[1]; rows=[]
for line in open('results/control35/maponly/metrics_maponly.csv'):
    p=line.strip().split(',')
    if len(p)>=10 and p[0]==cls: rows.append((float(p[7]),int(p[1])))
print(max(rows)[1] if rows else 35)
PY
)
  /usr/bin/time -f "%e s" timeout 3600 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_${ep}.pth \
    --class $cls --protocol 392 --fusion sum --device cpu --cpu_metrics \
    --fuse_bank -1 --ms_scoring --ms_scales 1.5 --ms_flip 1 --tag cpu-fast2view \
    >> results/ms35/eval_cpu15_${cls}.log 2>&1
  echo "[$(date +%H:%M:%S)] $cls@$ep 完成"
done
grep -h "^  CSV" results/ms35/eval_cpu15_*.log > results/ms35/all15_cpu_fast2view.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] CPU15_DONE"
