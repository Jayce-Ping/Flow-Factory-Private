#!/bin/bash
set -euo pipefail

REPO=/root/Flow-Factory-Private
PYTHON=/opt/conda/envs/ff/bin/python
LOG=/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/diagnostics/teacher_gap_v1/analysis/full_x0_run.log

restore_local_keepalive() {
  pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null || true
  setsid "$PYTHON" "$REPO/.scratch/gpu_keepalive.py" \
    > /root/gpu_keepalive.log 2>&1 < /dev/null &
}

trap restore_local_keepalive EXIT INT TERM
pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null || true
sleep 2

export HF_HOME=/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/.cache/huggingface
export TORCH_HOME=/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/.cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPYCACHEPREFIX=/tmp/ffpyc

cd "$REPO"
"$PYTHON" scripts/xopd_analysis/analyze_full_capture_x0.py "$@" > "$LOG" 2>&1
