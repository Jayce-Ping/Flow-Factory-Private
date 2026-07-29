#!/bin/bash
# Sequentially eval one XOPD checkpoint three ways on 32 GPUs: normal routing, expert 0 only,
# expert 1 only (model.mof_force_expert). Each variant needs all four nodes, so they run one after
# another; a variant is stopped as soon as both eval sets have logged, because the epoch-0 eval runs
# BEFORE training and the training epoch that would follow is of no interest.
#
# The 'blend' variant is the control: its numbers must land on the training curve at that epoch,
# which is what makes the two single-expert numbers trustworthy.
#
# Two things this has to get right, both learned the hard way:
#   * WAIT for the GPUs to actually drain between variants. A leftover rank from a previous attempt
#     holding ~40 GiB makes the next launch die with a confusing OOM at model load.
#   * Decide success/failure from THIS variant's launcher process, not by grepping a log file that
#     all variants share -- a stale traceback from the previous variant otherwise reads as a fresh
#     failure and kills a run that was doing fine.
#
# Usage: setsid bash scripts/xopd_analysis/run_forced_expert_eval.sh > /root/forced_expert_eval.log 2>&1 &
set -uo pipefail
REPO=/root/Flow-Factory-Private
cd "$REPO"
WORKERS=(28.7.185.215 28.7.185.156 28.7.195.15)
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
LAUNCHER=scripts/xopd_cluster/run_4node_xopd.sh
RANK0_LOG=/root/ff_xopd_rank0.log
KEYS=("eval/geneval_gs1/reward_geneval_mean" "eval/ocr_gs1/reward_ocr_mean")
TIMEOUT=${TIMEOUT:-4200}
PORT_BASE=${PORT_BASE:-29610}
RESULTS=/root/forced_expert_eval_results.txt

log() { echo "[forced-eval $(date '+%F %T')] $*"; }

start_keepalive() {
  local KA='source /opt/conda/etc/profile.d/conda.sh && conda activate ff && cd /root/Flow-Factory-Private && pkill -9 -f "^python .*gpu_keepalive\.py" 2>/dev/null; setsid nohup python .scratch/gpu_keepalive.py > /root/gpu_keepalive.log 2>&1 < /dev/null &'
  log "keepalive on all 4 nodes"
  for ip in "${WORKERS[@]}"; do $SSH -f "$ip" "bash -lc '$KA'" || true; done
  bash -lc "$KA" || true
}
stop_keepalive() {
  for ip in "${WORKERS[@]}"; do $SSH "$ip" "pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null" || true; done
  pkill -9 -f "[g]pu_keepalive.py" 2>/dev/null || true
}
kill_runs() {
  local pat="pkill -9 -f '[f]f-train' 2>/dev/null; pkill -9 -f '[f]low_factory.train' 2>/dev/null; pkill -9 -f '[a]ccelerate launch' 2>/dev/null; true"
  for ip in "${WORKERS[@]}"; do $SSH "$ip" "$pat" || true; done
  bash -c "$pat" || true
}
node_gpu_mem() {
  local q="nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc"
  if [ "$1" = local ]; then bash -c "$q" 2>/dev/null || echo 999999
  else $SSH "$1" "$q" 2>/dev/null || echo 999999; fi
}
wait_gpus_free() {  # $1 = seconds
  local deadline=$(( $(date +%s) + ${1:-300} )) n mem busy
  while [ "$(date +%s)" -lt "$deadline" ]; do
    busy=""
    for n in local "${WORKERS[@]}"; do
      mem=$(node_gpu_mem "$n"); [ "${mem:-999999}" -gt 2048 ] && busy="$busy $n(${mem}MiB)"
    done
    [ -z "$busy" ] && return 0
    log "waiting for GPUs to drain:$busy"
    sleep 15
  done
  log "TIMEOUT: GPUs still busy:$busy"
  return 1
}

SYNC_PATHS="src config xopd_configs scripts"
HASH_CMD='cd /root/Flow-Factory-Private && find src config xopd_configs scripts -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64'
sync_and_verify() {
  # run_4node_xopd.sh syncs nothing, and the workers need BOTH the generated configs and the
  # mof_force_expert plumbing -- without the latter the knob lands in extra_kwargs with a warning
  # and e0/e1 would silently evaluate as the ordinary blend.
  local local_sha ip remote_sha
  for ip in "${WORKERS[@]}"; do
    rsync -a --delete -e "$SSH" $SYNC_PATHS "$ip:$REPO/" || { log "rsync to $ip FAILED"; return 1; }
  done
  local_sha=$(bash -c "$HASH_CMD")
  for ip in "${WORKERS[@]}"; do
    remote_sha=$($SSH "$ip" "$HASH_CMD")
    [ "$remote_sha" = "$local_sha" ] || { log "SHA MISMATCH on $ip"; return 1; }
  done
  log "workers synced + SHA-verified (tree ${local_sha:0:16})"
}

: > "$RESULTS"
i=0
for tag in blend e0 e1; do
  cfg="xopd_configs/ode_pathwise/_eval_mof2_sigsoft_ep100_${tag}.yaml"
  port=$((PORT_BASE + i)); i=$((i + 1))
  log "================= variant '$tag' ($cfg, port $port)"
  kill_runs
  sync_and_verify || { log "aborting: workers not in sync"; start_keepalive; exit 1; }
  stop_keepalive
  wait_gpus_free 300 || { log "aborting: GPUs never drained"; start_keepalive; exit 1; }
  : > "$RANK0_LOG"

  MASTER_PORT="$port" setsid bash "$LAUNCHER" "$cfg" > "/root/forced_eval_${tag}_launch.log" 2>&1 &
  lpid=$!
  log "launcher pid $lpid"
  deadline=$(( $(date +%s) + TIMEOUT ))
  status=running
  while :; do
    got=0
    for k in "${KEYS[@]}"; do grep -aq "$k" "$RANK0_LOG" 2>/dev/null && got=$((got + 1)); done
    if [ "$got" -eq "${#KEYS[@]}" ]; then status=ok; break; fi
    kill -0 "$lpid" 2>/dev/null || { status=exited; break; }
    [ "$(date +%s)" -ge "$deadline" ] && { status=timeout; break; }
    sleep 20
  done
  log "variant '$tag': $status"
  {
    echo "### $tag ($status)"
    grep -aoE 'eval/(geneval_gs1|ocr_gs1)/reward_[a-z_]+_mean=[0-9.]+' "$RANK0_LOG" | sort -u
  } >> "$RESULTS"
  cp -f "$RANK0_LOG" "/root/forced_eval_${tag}_rank0.log"
  kill_runs
  sleep 15
done

start_keepalive
log "================= done; results in $RESULTS"
cat "$RESULTS"
