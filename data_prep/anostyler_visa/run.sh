#!/bin/bash
# [VisA] AnoStyler 批量合成：1/2/4-shot 共 84 个配置（12 类 × (1+2+4) 张底图 × 35 张/底图）。
#   * 只用**当前空闲**的卡（按显存余量判定），不挤占同事任务；
#   * 每卡并发 ANOSTYLER_PARALLEL 个进程（单进程约 4.6GB 显存）；
#   * 断点续跑：pending.py 只列没跑完的配置，重复执行本脚本即可续跑。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ANOSTYLER_DIR=/home/jjquan/AnoStyler
PY_ANOSTYLER=/home/jjquan/miniconda3/envs/anostyler/bin/python
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
GPUS=(${ANOSTYLER_GPUS:-GPU-0e3a7e1c-105c-997d-921f-b4247b4dc822 GPU-3bf0407c-aa78-8aab-cd2f-1c7b3356c5be})
PARALLEL="${ANOSTYLER_PARALLEL:-3}"
NEED_FREE_MB="${ANOSTYLER_NEED_FREE_MB:-16000}"
cd "$ANOSTYLER_DIR" || exit 3
mkdir -p "$HERE/logs"
DRIVER="$HERE/logs/driver.log"

log () { echo "[$(date +'%F %T')] $*" | tee -a "$DRIVER"; }

free_mb () {
  nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v g="$1" '$1==g {print $2; exit}' | tr -dc '0-9'
}

pick_gpus () {
  local ok=""
  for g in "${GPUS[@]}"; do
    f="$(free_mb "$g")"; f="${f:-0}"; case "$f" in ''|*[!0-9]*) f=0;; esac
    [ "$f" -ge "$NEED_FREE_MB" ] && ok="$ok $g"
  done
  echo "$ok"
}

CONF_LIST=()
while IFS='|' read -r cfg cls defect stem n; do
  CONF_LIST+=("$cfg|$cls|$defect|$stem|$n")
done < <("$PY" "$HERE/pending.py" 2>>"$DRIVER")

TOTAL=${#CONF_LIST[@]}
log "VisA 合成开始：待跑 $TOTAL 个配置 / 每卡 $PARALLEL 并发 / 卡: ${GPUS[*]}"

i=0
while [ "$i" -lt "$TOTAL" ]; do
  avail=""
  for _ in $(seq 1 240); do
    avail="$(pick_gpus)"; [ -n "$avail" ] && break
    log "无空闲卡（需 ${NEED_FREE_MB}MB 余量），休眠 120s 后重试"; sleep 120
  done
  [ -z "$avail" ] && { log "长时间无空闲卡，中止"; exit 4; }
  pids=""; idx=$i
  for g in $avail; do
    for j in $(seq 1 "$PARALLEL"); do
      [ "$idx" -ge "$TOTAL" ] && break
      item="${CONF_LIST[$idx]}"
      cfg="${item%%|*}"; rest="${item#*|}"; cls="${rest%%|*}"
      rest="${rest#*|}"; defect="${rest%%|*}"; stem="${rest#*|}"; stem="${stem%%|*}"
      base="${cls}_${defect}_${stem}"
      log "启动 $base（已有 ${item##*|} 张）on ${g:4:8}"
      CUDA_VISIBLE_DEVICES="$g" "$PY_ANOSTYLER" main.py --config "$cfg" \
          > "$HERE/logs/${base}.log" 2>&1 &
      pids="$pids $!"; idx=$((idx + 1))
    done
  done
  for p in $pids; do wait "$p"; done
  i=$idx
  d=$("$PY" "$HERE/pending.py" 2>/dev/null | wc -l)
  log "本轮结束：已完成 $((TOTAL - d))/$TOTAL，剩余 $d"
done
log "全部完成"
touch "$HERE/DONE"
