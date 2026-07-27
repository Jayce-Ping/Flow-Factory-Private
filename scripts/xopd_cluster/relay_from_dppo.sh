#!/bin/bash
# Hand the 4-node cluster back from the DPPO rebuttal run to the XOPD chain.
#
#   phase 1  wait until the in-flight DPPO run reaches STOP_EPOCH (or stops advancing)
#   phase 2  kill it on all 4 nodes, wait for the 32 GPUs to come back
#   phase 3  DMD GPU smoke on node0 only (keepalive covers the other 3 nodes meanwhile)
#   phase 4  on smoke success, hand off to run_three_experts.sh for the MoF-2 long run
#
# STOP_EPOCH is the epoch whose *start* triggers the stop, so the previous epoch's checkpoint and
# eval have both landed: an epoch runs sampling -> training -> save -> eval, so seeing
# "Epoch 1001 Sampling" means epoch 1000 (a multiple of both save_freq and eval_freq) is complete.
#
# The DMD trainer has never run on a GPU, so the smoke gates the long run: on smoke failure the
# chain stops with keepalive up rather than burning days on an unverified trainer. Any exit leaves
# keepalive running on all 4 nodes -- this cluster reaps allocations whose GPU utilization is low.
#
# Usage:
#   setsid bash scripts/xopd_cluster/relay_from_dppo.sh > /root/relay_from_dppo.log 2>&1 < /dev/null &
set -uo pipefail

REPO=/root/Flow-Factory-Private
WORKERS=(28.7.185.215 28.7.185.156 28.7.195.15)
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# --- phase 1: the run we are waiting on ------------------------------------------------------
WATCH_LOG=${WATCH_LOG:-/root/e3_mcK64cps_rank0.log}
STOP_EPOCH=${STOP_EPOCH:-1001}
DPPO_STALL_SECS=${DPPO_STALL_SECS:-1800}   # log silent this long -> the run is dead, relay now
DPPO_CKPT_DIR=${DPPO_CKPT_DIR:-/root/DPPO-rebuttal/code/saves/grpo_sd3-5_01234_cps3_inner1_kl-adv-1e-6_mc-K64_geneval2/checkpoints}

# --- phases 3-4 ------------------------------------------------------------------------------
SMOKE_CFG=${SMOKE_CFG:-xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_dmd_mix_smoke.yaml}
SMOKE_PORT=${SMOKE_PORT:-29580}
SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-9000}       # 2.5h: 32B teacher load + one 8-GPU DMD epoch
MOF_INDEX=${MOF_INDEX:-9}                  # run_three_experts.sh CONFIGS index (MoF-2 sigtopk)

log() { echo "[relay $(date '+%F %T')] $*"; }

start_keepalive() {  # $* = node list ("local" for node0); no args = all four
  local nodes=("$@"); [ ${#nodes[@]} -eq 0 ] && nodes=(local "${WORKERS[@]}")
  local KA='source /opt/conda/etc/profile.d/conda.sh && conda activate ff && cd /root/Flow-Factory-Private && pkill -f "[g]pu_keepalive.py" 2>/dev/null; setsid python .scratch/gpu_keepalive.py > /root/gpu_keepalive.log 2>&1 < /dev/null &'
  log "starting GPU keepalive on: ${nodes[*]}"
  local n
  for n in "${nodes[@]}"; do
    if [ "$n" = local ]; then bash -lc "$KA" || true; else $SSH -f "$n" "bash -lc '$KA'" || true; fi
  done
}

stop_keepalive() {
  log "stopping GPU keepalive on all 4 nodes"
  local ip
  for ip in "${WORKERS[@]}"; do $SSH "$ip" "pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null" || true; done
  pkill -9 -f "[g]pu_keepalive.py" 2>/dev/null || true
}

kill_trainers() {
  log "killing ff-train / flow_factory.train on all 4 nodes"
  local pat="pkill -9 -f '[f]f-train' 2>/dev/null; pkill -9 -f '[f]low_factory.train' 2>/dev/null; pkill -9 -f '[a]ccelerate launch' 2>/dev/null; true"
  local ip
  for ip in "${WORKERS[@]}"; do $SSH "$ip" "$pat" || true; done
  bash -c "$pat" || true
}

# Total GPU memory in use on a node, in MiB (0 when the node is free).
node_gpu_mem() {
  local ip=$1 q="nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc"
  if [ "$ip" = local ]; then bash -c "$q" 2>/dev/null || echo 999999
  else $SSH "$ip" "$q" 2>/dev/null || echo 999999; fi
}

wait_for_free_gpus() {  # $1 = seconds to wait before giving up
  local deadline=$(( $(date +%s) + ${1:-300} )) n mem busy
  while [ "$(date +%s)" -lt "$deadline" ]; do
    busy=""
    for n in local "${WORKERS[@]}"; do
      mem=$(node_gpu_mem "$n"); [ "${mem:-999999}" -gt 2048 ] && busy="$busy $n(${mem}MiB)"
    done
    [ -z "$busy" ] && { log "all 32 GPUs are free"; return 0; }
    log "waiting for GPUs to drain:$busy"
    sleep 15
  done
  log "TIMEOUT waiting for GPUs to drain:$busy"
  return 1
}

abort() {  # keep the allocation alive and stop the chain
  log "!!! $* -- stopping the chain, leaving keepalive up for all 4 nodes"
  start_keepalive
  exit 1
}

cd "$REPO"
log "=== relay armed: watching $WATCH_LOG for 'Epoch $STOP_EPOCH Sampling' ==="
[ -f "$WATCH_LOG" ] || abort "watch log $WATCH_LOG does not exist"

# ---------------------------------------------------------------------------- phase 1: wait ---
trigger=
beat=0
while :; do
  if grep -q "Epoch ${STOP_EPOCH} Sampling" "$WATCH_LOG" 2>/dev/null; then trigger=epoch; break; fi
  age=$(( $(date +%s) - $(stat -c %Y "$WATCH_LOG") ))
  if [ "$age" -gt "$DPPO_STALL_SECS" ]; then trigger=dead; break; fi
  if [ $((beat % 20)) -eq 0 ]; then   # every ~10 min
    log "still waiting: $(grep -oE 'Epoch [0-9]+ Sampling' "$WATCH_LOG" | tail -1), log age ${age}s"
  fi
  beat=$((beat + 1))
  sleep 30
done

if [ "$trigger" = epoch ]; then
  done_epoch=$((STOP_EPOCH - 1))
  log "TRIGGER: epoch $STOP_EPOCH started -> epoch $done_epoch is complete"
  grep -q "Epoch ${done_epoch}\]" "$WATCH_LOG" \
    && log "  eval for epoch $done_epoch found in the log" \
    || log "  WARNING: no eval line for epoch $done_epoch (continuing anyway)"
  [ -d "$DPPO_CKPT_DIR/checkpoint-$done_epoch" ] \
    && log "  checkpoint-$done_epoch present" \
    || log "  WARNING: $DPPO_CKPT_DIR/checkpoint-$done_epoch missing (continuing anyway)"
else
  log "TRIGGER: $WATCH_LOG has been silent for >${DPPO_STALL_SECS}s -> DPPO run is gone, relaying now"
fi

# ---------------------------------------------------------------------------- phase 2: stop ---
kill_trainers
sleep 10
wait_for_free_gpus 300 || abort "GPUs did not drain after killing the DPPO run"

# --------------------------------------------------------------------- phase 3: DMD smoke -----
# Single-node smoke: keepalive holds the other three nodes so they are not left idle.
start_keepalive "${WORKERS[@]}"
log "=== phase 3: DMD smoke on node0 ($SMOKE_CFG, port $SMOKE_PORT, timeout ${SMOKE_TIMEOUT}s) ==="
timeout "$SMOKE_TIMEOUT" env MASTER_PORT="$SMOKE_PORT" XOPD_WORKERS="" \
  bash scripts/xopd_cluster/run_4node_xopd.sh "$SMOKE_CFG"
smoke_rc=$?
cp -f /root/ff_xopd_rank0.log /root/relay_dmd_smoke_rank0.log 2>/dev/null || true
log "smoke exited rc=$smoke_rc (log kept at /root/relay_dmd_smoke_rank0.log)"

[ "$smoke_rc" -eq 124 ] && abort "DMD smoke timed out after ${SMOKE_TIMEOUT}s"
[ "$smoke_rc" -ne 0 ] && abort "DMD smoke failed (rc=$smoke_rc)"
grep -q "Training completed successfully" /root/relay_dmd_smoke_rank0.log \
  || abort "DMD smoke exited 0 but never logged 'Training completed successfully'"
log "DMD smoke PASSED"

# ------------------------------------------------------------------- phase 4: MoF-2 long run --
kill_trainers          # nothing should survive the smoke; make sure of it before a 32-GPU launch
stop_keepalive         # the orchestrator manages keepalive from here on
sleep 5
log "=== phase 4: handing off to run_three_experts.sh (CONFIGS index $MOF_INDEX) ==="
START_AT="$MOF_INDEX" STOP_AT="$MOF_INDEX" bash scripts/xopd_cluster/run_three_experts.sh
chain_rc=$?
log "=== orchestrator exited rc=$chain_rc ==="
[ "$chain_rc" -ne 0 ] && abort "MoF-2 chain failed (rc=$chain_rc)"
start_keepalive
log "=== relay complete ==="
