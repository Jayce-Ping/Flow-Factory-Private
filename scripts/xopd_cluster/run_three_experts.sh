#!/bin/bash
# Sequentially run the XOPD capacity + ablation experiments on this 4-node cluster:
#     Run #1  OCR                     (eval_teacher_at_start: true  -> the ONE shared teacher baseline)
#     Run #2  geneval_enhanced        (eval_teacher_at_start: false -> reuses Run #1's ceiling)
#     Run #3  OCR x0-space  (full-timestep clean-latent d_k; xopd_dk_space=x0)  -- prefer OTHER cluster
#     Run #4  OCR selective (late-timestep; DEFERRED until v vs x0 picks winner)
#     Run #5  OCR v-space   (full-timestep raw-velocity d_k; xopd_dk_space=v)
#
# Plan (2026-07-16): MSE(v) >> MSE(xt) (xt last-step dominance). Next ablate MSE(x0) vs MSE(v);
# selective ONLY after that. Prefer launching Run #3 (xspace) on an idle second cluster in parallel
# with this cluster's Run #5 (vmse). See docs/xopd/progress/2026-07-16-loss-space-ablation-plan.md.
# NOTE: MIXED trains on a SEPARATE cluster and is NOT in this chain. Loss spaces:
# MSE(v):MSE(xt):MSE(x0) = 1:dt^2:sigma^2 (xt is default for Runs #1/#2/#4 until #4 is re-forked).
#
# Resuming / handing off (env):
#   START_AT=<i>     start the chain at CONFIGS index i (skip already-done runs). Default 0.
#   STOP_AT=<i>      last CONFIGS index to run (inclusive); empty = run to the end.
#   WAIT_FOR=<sub>   before starting, WAIT (no kill, no keepalive) until an in-flight ff-train whose
#                    cmdline contains <sub> finishes -- used to hand off from a run another
#                    orchestrator launched, without disturbing it. Requires completion marker.
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
#   # this cluster: only vmse (idx 4), already in flight via START_AT=4
#   START_AT=4 STOP_AT=4 setsid bash ... &
#   # other / idle cluster: only x0-MSE (idx 2) — DO NOT chain selective yet
#   START_AT=2 STOP_AT=2 setsid bash ... &
#   # RELAY the MoF2-FSDP run (idx 5) AFTER the in-flight strat4 run finishes (waits for its
#   # "Training completed successfully" marker; does NOT kill it):
#   WAIT_FOR=strat4 START_AT=5 STOP_AT=5 setsid bash scripts/xopd_cluster/run_three_experts.sh \
#     > /root/mof2_fsdp_relay.log 2>&1 < /dev/null &
#   # (Run #5 = MoF 2xbase, geneval+ocr MIX, FSDP HYBRID_SHARD student + replicated 32B teacher, full-timestep MSE(v).)
set -uo pipefail
REPO=/root/Flow-Factory-Private
cd "$REPO"
WORKERS=(28.7.185.215 28.7.185.156 28.7.195.15)
LAUNCHER=scripts/xopd_cluster/run_4node_xopd.sh
STALL_SECS=${STALL_SECS:-5400}          # 90 min with no rank0 log write -> hung
START_AT=${START_AT:-0}                 # CONFIGS index to start at (skip completed runs)
STOP_AT=${STOP_AT:-}                    # LAST CONFIGS index to run (inclusive); empty = run to the end.
                                        # Use with START_AT to run a contiguous sub-range, e.g.
                                        # START_AT=2 STOP_AT=3 runs only indices 2,3 (skips a trailing run).
WAIT_FOR=${WAIT_FOR:-}                  # await an in-flight ff-train (cmdline substring) before starting
SYNC_PATHS="src config xopd_configs scripts"
HASH_CMD='cd /root/Flow-Factory-Private && find src config xopd_configs scripts -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64'

# name:port  (distinct ports so a stale socket from a previous run never collides)
CONFIGS=(
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_1kep.yaml:29570"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_geneval_enh_1kep.yaml:29571"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml:29573"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_selective_teacher_1kep.yaml:29574"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_vmse_1kep.yaml:29575"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_mof2_mix_fsdp_vmse_1kep.yaml:29585"
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

wait_for_running() {  # $1 = ff-train cmdline substring; block (no kill/keepalive) until it is gone
  local pat="$1" beat=0 age
  log "WAIT_FOR: awaiting in-flight run matching '$pat' before starting the chain (not killing it)"
  while pgrep -f "flow_factory.train.*$pat" >/dev/null 2>&1 || pgrep -f "[f]f-train.*$pat" >/dev/null 2>&1; do
    sleep 120
    beat=$((beat + 1))
    if [ $((beat % 15)) -eq 0 ]; then     # ~every 30 min: heartbeat with rank0 log age (hang visibility)
      age=$([ -f /root/ff_xopd_rank0.log ] && echo $(( $(date +%s) - $(stat -c %Y /root/ff_xopd_rank0.log) )) || echo NA)
      log "WAIT_FOR: still awaiting '$pat' (rank0 log age=${age}s; not killing -- monitor GPU util if this grows unbounded)"
    fi
  done
  log "WAIT_FOR: '$pat' no longer running"
  if ! grep -q "Training completed successfully" /root/ff_xopd_rank0.log 2>/dev/null; then
    log "WAIT_FOR: '$pat' ended WITHOUT a completion marker -> ABORT (investigate before follow-ups)"
    exit 1
  fi
  log "WAIT_FOR: '$pat' completed successfully; proceeding"
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

log "=== XOPD chain start (STALL_SECS=$STALL_SECS, START_AT=$START_AT, WAIT_FOR='${WAIT_FOR:-none}') ==="
if [ -n "$WAIT_FOR" ]; then
  wait_for_running "$WAIT_FOR"   # hand off from an in-flight run WITHOUT killing it / touching keepalive
else
  kill_all_runs                  # fresh start: drop any prior (debug) run
fi
start_keepalive                  # hold GPUs during the first sync (awaited run, if any, is done)

for i in "${!CONFIGS[@]}"; do
  if [ "$i" -lt "$START_AT" ]; then
    log "SKIP index $i ($(basename "${CONFIGS[$i]%%:*}" .yaml)) < START_AT=$START_AT"
    continue
  fi
  if [ -n "$STOP_AT" ] && [ "$i" -gt "$STOP_AT" ]; then
    log "STOP: index $i > STOP_AT=$STOP_AT -> chain sub-range done"
    break
  fi
  entry="${CONFIGS[$i]}"
  cfg="${entry%%:*}"; port="${entry##*:}"; name=$(basename "$cfg" .yaml)
  log "===================== RUN [$i]: $name (port $port) ====================="

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

log "=== CHAIN COMPLETED SUCCESSFULLY (from index $START_AT) ==="
