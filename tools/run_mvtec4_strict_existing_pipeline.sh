#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/jjquan/miniconda3/envs/vpt/bin/python
cd "$ROOT"

"$PY" "$ROOT/tools/build_mvtec4_strict_existing.py"
bash "$ROOT/tools/run_mvtec4_strict_35e.sh"
bash "$ROOT/tools/run_mvtec4_strict_eval.sh"
bash "$ROOT/tools/run_mvtec4_strict_fusion.sh"
printf 'MVTEC4_STRICT_EXISTING_PIPELINE_DONE\n'
