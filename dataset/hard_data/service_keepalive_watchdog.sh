#!/bin/bash
# Request-aware GPU keepalive for nodes serving one distributed vLLM engine.
#
# Polls the head API's Prometheus gauges. Any running/waiting request stops the
# local matmul burner immediately; the burner starts only after a continuous
# idle grace period. API-unreachable/model-startup is fail-safe: burner off.
set -uo pipefail

METRICS_URL=${QWEN_METRICS_URL:-http://28.7.185.156:8000/metrics}
POLL_SECONDS=${QWEN_KEEPALIVE_POLL_SECONDS:-2}
IDLE_GRACE_SECONDS=${QWEN_KEEPALIVE_IDLE_GRACE_SECONDS:-20}
REPO=${FLOW_FACTORY_REPO:-/root/Flow-Factory-Private}
PYTHON=${KEEPALIVE_PYTHON:-/opt/conda/envs/ff/bin/python}
KEEPALIVE_SCRIPT=${KEEPALIVE_SCRIPT:-$REPO/.scratch/gpu_keepalive.py}
LOG=${QWEN_KEEPALIVE_LOG:-/root/qwen_keepalive_watchdog.log}

if [ "$POLL_SECONDS" -lt 1 ] || [ "$IDLE_GRACE_SECONDS" -lt "$POLL_SECONDS" ]; then
  echo "invalid watchdog timing: poll=$POLL_SECONDS idle_grace=$IDLE_GRACE_SECONDS" >&2
  exit 2
fi
if [ ! -x "$PYTHON" ] || [ ! -f "$KEEPALIVE_SCRIPT" ]; then
  echo "missing keepalive runtime: python=$PYTHON script=$KEEPALIVE_SCRIPT" >&2
  exit 2
fi

burner_running() {
  pgrep -f "^$PYTHON .*gpu_keepalive\\.py" >/dev/null ||
    pgrep -f "^python .*gpu_keepalive\\.py" >/dev/null
}

start_burner() {
  burner_running && return 0
  cd "$REPO" || exit 1
  setsid nohup "$PYTHON" "$KEEPALIVE_SCRIPT" \
    > /root/gpu_keepalive.log 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] idle -> keepalive started" >> "$LOG"
}

stop_burner() {
  if burner_running; then
    pkill -9 -f "^$PYTHON .*gpu_keepalive\\.py" 2>/dev/null || true
    pkill -9 -f "^python .*gpu_keepalive\\.py" 2>/dev/null || true
    echo "[$(date '+%F %T')] request/startup -> keepalive stopped" >> "$LOG"
  fi
}

request_count() {
  curl --noproxy '*' -fsS --max-time 2 "$METRICS_URL" |
    awk '
      /^vllm:num_requests_running\{/ {sum += $NF; found = 1}
      /^vllm:num_requests_waiting\{/ {sum += $NF; found = 1}
      END {if (!found) exit 1; printf "%d\n", sum}
    '
}

trap 'stop_burner' EXIT INT TERM
idle_seconds=0
state=""
while true; do
  if requests=$(request_count 2>/dev/null); then
    if [ "$requests" -gt 0 ]; then
      stop_burner
      idle_seconds=0
      if [ "$state" != active ]; then
        echo "[$(date '+%F %T')] active requests=$requests" >> "$LOG"
        state=active
      fi
    else
      idle_seconds=$((idle_seconds + POLL_SECONDS))
      if [ "$idle_seconds" -ge "$IDLE_GRACE_SECONDS" ]; then
        start_burner
        if [ "$state" != idle ]; then
          echo "[$(date '+%F %T')] idle requests=0 grace=${idle_seconds}s" >> "$LOG"
          state=idle
        fi
      fi
    fi
  else
    # API unavailable usually means model load/restart. Never contend with it.
    stop_burner
    idle_seconds=0
    if [ "$state" != unavailable ]; then
      echo "[$(date '+%F %T')] metrics unavailable; keepalive disabled" >> "$LOG"
      state=unavailable
    fi
  fi
  sleep "$POLL_SECONDS"
done
