#!/bin/bash
# 在全部 15 类上评估"高效档"：--ms_scales 1.5 --ms_flip 1（2 次前向）+ 融合
set -u
cd /home/jjquan/experiment/visual/audit_mapping
source ./common_env.sh
require_gpu || exit 3
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
ALL="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
for cls in $ALL; do
  if [ ! -f "results/control35/maponly/model_${cls}_35.pth" ]; then echo "[skip] $cls 未训练"; continue; fi
  ep=$($PY - "$cls" <<'PY'
import sys
cls=sys.argv[1]; rows=[]
for line in open('results/control35/maponly/metrics_maponly.csv'):
    p=line.strip().split(',')
    if len(p)>=10 and p[0]==cls: rows.append((float(p[7]),int(p[1])))
print(max(rows)[1] if rows else 35)
PY
)
  timeout 2400 $PY eval_offline.py --ckpt results/control35/maponly/model_${cls}_${ep}.pth \
    --class $cls --protocol 392 --fusion sum --fuse_bank -1 --ms_scoring \
    --ms_scales 1.5 --ms_flip 1 --tag fast2view \
    >> results/ms35/eval_fast_${cls}.log 2>&1
  echo "[$(date +%F\ %H:%M:%S)] fast2view $cls@$ep done"
done
grep -h "^  CSV" results/ms35/eval_fast_*.log > results/ms35/all15_fast.csv 2>/dev/null
echo "[$(date +%F\ %H:%M:%S)] FAST_DONE"
