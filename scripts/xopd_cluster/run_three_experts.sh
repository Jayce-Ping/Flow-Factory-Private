#!/bin/bash
# Sequentially run the three 4B-student XOPD experiments for the capacity study:
#     Run #1  OCR              (eval_teacher_at_start: true  -> the ONE shared teacher baseline)
#     Run #2  geneval_enhanced (eval_teacher_at_start: false -> reuses Run #1's ceiling)
#     Run #3  geneval_enh+ocr mixed 1:1 (eval_teacher_at_start: false)
#
# Contract:
#   * Before each run: stop keepalive, rsync repo (src/config/xopd_configs/scripts) to the 3
#     workers, and verify a per-file SHA of the actual working tree matches on all 4 nodes
#     (abort on drift -- Git HEAD alone is NOT enough; workers have diverged before).
#   * Each run blocks until rank0 exits. A stall watchdog kills a run whose rank0 log stops
#     advancing for STALL_SECS (default 90 min) so a NCCL/HF hang cannot wedge the chain.
#   * A run counts as SUCCESS only on exit code 0 AND "Training completed successfully" in the
#     rank0 log AND no error tail. On ANY failure the chain STOPS (no blind retry), restarts
#     keepalive, and exits non-zero.
#   * Whenever no run is training (startup, inter-run prep, after finish/failure) keepalive runs
#     on all 4 nodes so GPU utilization stays high (this cluster reaps idle allocations).
#
# Usage:
#   setsid bash scripts/xopd_cluster/run_three_experts.sh > /root/three_experts.log 2>&1 < /dev/null &
set -uo pipefail
REPO=/root/Flow-Factory-Private
cd "$REPO"
WORKERS=(28.7.185.215 28.7.185.156 28.7.195.15)
LAUNCHER=scripts/xopd_cluster/run_4node_xopd.sh
STALL_SECS=${STALL_SECS:-5400}          # 90 min with no rank0 log write -> hung
SYNC_PATHS="src config xopd_configs scripts"
HASH_CMD='cd /root/Flow-Factory-Private && find src config xopd_configs scripts -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64'

# name:port  (distinct ports so a stale socket from a previous run never collides)
CONFIGS=(
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_1kep.yaml:29570"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_geneval_enh_1kep.yaml:29571"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_geneval_enh_ocr_mixed_1kep.yaml:29572"
)

log() { echo "[orchestrator $(date '+%F %T')] $*"; }

start_keepalive() {
  log "starting GPU keepalive on all 4 nodes"
  local KA='source /opt/conda/etc/profile.d/conda.sh && conda activate ff && cd /root/Flow-Factory-Private && pkill -f "[g]pu_keepalive.py" 2>/dev/null; setsid python .scratch/gpu_keepalive.py > /root/gpu_keepalive.log 2>&1 < /dev/null &'
  for ip in "${WORKERS[@]}"; do ssh -o StrictHostKeyChecking=no -f "$ip" "bash -lc '$KA'" || true; done
  bash -lc "$KA" || true
}
stop_keepalive() {
  log "stopping GPU keepalive on all 4 nodes"
  for ip in "${WORKERS[@]}"; do ssh -o StrictHostKeyChecking=no "$ip" "pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null" || true; done
  pkill -9 -f "[g]pu_keepalive.py" 2>/dev/null || true
}
kill_all_runs() {
  log "killing any stale ff-train on all 4 nodes"
  for ip in "${WORKERS[@]}"; do ssh -o StrictHostKeyChecking=no "$ip" "pkill -9 -f '[f]f-train' 2>/dev/null; pkill -9 -f 'flow_factory.train' 2>/dev/null; pkill -9 -f '[_]ocr_worker.py' 2>/dev/null" || true; done
  pkill -9 -f "[f]f-train" 2>/dev/null || true
  pkill -9 -f "flow_factory.train" 2>/dev/null || true
  pkill -9 -f "[_]ocr_worker.py" 2>/dev/null || true
}

sync_and_verify() {
  log "rsync repo -> workers + SHA-verify working tree on all nodes"
  for ip in "${WORKERS[@]}"; do
    rsync -a --delete -e "ssh -o StrictHostKeyChecking=no" $SYNC_PATHS "$ip:$REPO/" || { log "rsync to $ip failed"; return 1; }
  done
  local local_sha; local_sha=$(bash -c "$HASH_CMD")
  log "node0 tree sha=$local_sha"
  local ip remote_sha
  for ip in "${WORKERS[@]}"; do
    remote_sha=$(ssh -o StrictHostKeyChecking=no "$ip" "$HASH_CMD")
    if [ "$remote_sha" != "$local_sha" ]; then
      log "SHA MISMATCH on $ip ($remote_sha != $local_sha)"; return 1
    fi
    log "$ip tree sha=$remote_sha OK"
  done
  return 0
}

watchdog() {  # $1=config ; kills a hung run so the foreground launcher returns
  local cfg="$1" age
  while true; do
    sleep 300
    [ -f /root/ff_xopd_rank0.log ] || continue
    age=$(( $(date +%s) - $(stat -c %Y /root/ff_xopd_rank0.log) ))
    if [ "$age" -gt "$STALL_SECS" ]; then
      log "WATCHDOG: rank0 log stale ${age}s > ${STALL_SECS}s -> killing hung run '$cfg'"
      for ip in "${WORKERS[@]}"; do ssh -o StrictHostKeyChecking=no "$ip" "pkill -9 -f '[f]f-train $cfg' 2>/dev/null; pkill -9 -f 'flow_factory.train $cfg' 2>/dev/null" || true; done
      pkill -9 -f "[f]f-train $cfg" 2>/dev/null || true
      pkill -9 -f "flow_factory.train $cfg" 2>/dev/null || true
      return 0
    fi
  done
}

# Always leave GPUs busy when the orchestrator exits (success or failure).
trap 'log "orchestrator exiting -> ensuring keepalive"; kill "${WD:-}" 2>/dev/null || true; start_keepalive' EXIT

log "=== three-expert XOPD chain start (STALL_SECS=$STALL_SECS) ==="
kill_all_runs                # drop any prior (debug) run
start_keepalive              # hold GPUs during the first sync

for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"; port="${entry##*:}"; name=$(basename "$cfg" .yaml)
  log "===================== RUN: $name (port $port) ====================="

  if ! sync_and_verify; then log "ABORT: sync/verify failed before $name"; exit 1; fi

  stop_keepalive              # free GPUs for the real job
  watchdog "$cfg" & WD=$!
  log "launching $name ..."
  MASTER_PORT="$port" FLOW_FACTORY_EVAL_GLOO_BARRIER=1 FLOW_FACTORY_EVAL_DEBUG=1 \
    bash "$LAUNCHER" "$cfg"
  rc=$?
  kill "$WD" 2>/dev/null || true; WD=""

  if [ "$rc" -ne 0 ]; then log "ABORT: $name exited rc=$rc (no retry)"; exit 1; fi
  if ! grep -q "Training completed successfully" /root/ff_xopd_rank0.log; then
    log "ABORT: $name exited 0 but no completion marker -> treating as failure"; exit 1
  fi
  if tail -80 /root/ff_xopd_rank0.log | grep -qiE 'Traceback|Detected mismatch|out of memory'; then
    log "ABORT: $name log shows an error tail"; exit 1
  fi

  rid=$(grep -oE 'runs/[A-Za-z0-9]+' /root/ff_xopd_rank0.log | tail -1)
  log "SUCCESS: $name completed (rc=0, wandb=$rid)"
  cp -f /root/ff_xopd_rank0.log "/root/ff_xopd_${name}_$(date +%Y%m%d_%H%M%S).log" || true
  start_keepalive             # hold GPUs during the next inter-run prep
done

log "=== ALL THREE RUNS COMPLETED SUCCESSFULLY ==="
