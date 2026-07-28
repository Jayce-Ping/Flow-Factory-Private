#!/bin/bash
# Sequentially run the XOPD loss-space ablation on this 4-node cluster.
#
# PLAN (updated 2026-07-23): finish the FULL OCR loss-space ablation FIRST, pick the single best d_k
# loss, and ONLY THEN move on to other experiments. Phase 2 is DEFERRED until the winner is chosen.
#
# Phase 1 -- OCR loss-space sweep (OCR-only data, otherwise identical; goal: pick the winning d_k):
#     xt-MSE   (default, last-step dominated)       done  (h5j2xknk)
#     v-MSE    (raw velocity)                       done  (5nyhyylw)
#     x0-MSE   (clean latent, sigma^2 reweight)     other cluster (maydy3cw)
#     x0_norm  (DiffusionNFT/DMD self-normalized)   <== NEXT (CONFIGS index 8)
#     selective late/early + strat4 (all v-MSE)     per-timestep coverage variants
#   Identity (ODE): MSE(v):MSE(xt):MSE(x0) = 1:dt^2:sigma^2; x0_norm = x0-MSE / sg(mean|x0_s-x0_t|).
#   Finding so far: MSE(v) >> MSE(xt) (xt last-step dominance). v vs x0 vs x0_norm decides the winner.
#
# Phase 2 -- DEFERRED until Phase 1 picks the best loss: DMD (xopd_configs/.../dmd_ocr_1kep, needs a
#   GPU smoke first), MoF-2 mix, mixed-data. NOT auto-chained here yet.
#   See docs/xopd/dmd_cross_model_design.md and docs/xopd/progress/2026-07-16-loss-space-ablation-plan.md.
#
# NOTE: indices 5-7 (MoF-2 mix, mixed-data) predate this revision and are Phase-2-adjacent; they are
# kept for START_AT/STOP_AT stability. The Phase-1 winner search is xt/v/x0/x0_norm (+ selective).
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
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_mof2_mix_fsdp_vmse_soft_nolb_1kep.yaml:29586"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_geneval_enh_ocr_mixed_vmse_1kep.yaml:29587"
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_x0norm_1kep.yaml:29588"   # Phase 1: OCR x0_norm loss-space ablation (stopped at ep255, treated as finished)
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_mof2_mix_fsdp_vmse_sigtopk_1kep.yaml:29589"  # MoF-2 router arm 3: sigmoid gate + top-1 sparse. STOPPED at ~ep160: the router was single-expert
                                                                                                # from init (E0 never selected; logit gap +0.44 vs +-0.02 across prompts), so arm 4 supersedes it.
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_mof2_mix_fsdp_vmse_sigtopk_cs_soft_t02_1kep.yaml:29590"  # MoF-2 router arm 4: arm 3 + k-means cold-start (soft targets, T=0.2). Cold-start runs but converges
                                                                                                           # degenerately (acc 0.48, maxprob 1.0 within 10 of 300 steps); not the lr -- a CPU sweep learns the same clusters at every lr.
  "xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_mof2_mix_fsdp_vmse_sigsoft_nolb_1kep.yaml:29591"  # MoF-2 router arm 5: NON-CONVEX dense soft mixing -- independent sigmoid gates over ALL experts, no load balance
)

log() { echo "[orchestrator $(date '+%F %T')] $*"; }

start_keepalive() {
  log "starting GPU keepalive on all 4 nodes"
  # The pre-kill must be anchored to the python process: an unanchored pattern also matches THIS
  # launching shell (its command line names the script and its log), so pkill killed the shell
  # before it could start anything -- keepalive then silently never ran and the nodes sat idle.
  local KA='source /opt/conda/etc/profile.d/conda.sh && conda activate ff && cd /root/Flow-Factory-Private && pkill -9 -f "^python .*gpu_keepalive\.py" 2>/dev/null; setsid nohup python .scratch/gpu_keepalive.py > /root/gpu_keepalive.log 2>&1 < /dev/null &'
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
