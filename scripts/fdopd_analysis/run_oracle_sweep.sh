#!/bin/bash
# Run the zero-training oracle at each lambda in turn, one 32-GPU job at a time.
#
# Each config evaluates the composed field and exits on its own (fdopd_oracle_eval makes start()
# return after one evaluate()), so the driver just waits for the launcher to exit before starting
# the next lambda. About 35 minutes per lambda.
#
# Two things this has to get right:
#   * WAIT for the GPUs to drain between lambdas. A leftover rank holding memory makes the next
#     launch die at model load with an error that points nowhere near the cause.
#   * Never pkill on a pattern that also matches this script's own command line. The config paths
#     contain "oracle", so a pattern like '[o]racle' kills the driver itself; match on the module
#     instead.
#
# Usage: setsid bash scripts/fdopd_analysis/run_oracle_sweep.sh > /root/oracle_sweep.log 2>&1 &
set -uo pipefail
REPO=/root/Flow-Factory-Private
cd "$REPO"
WORKERS=(28.7.185.215 28.7.185.156 28.7.195.15)
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
LAUNCHER=scripts/xopd_cluster/run_4node_xopd.sh
CONFIG_DIR=examples/flow_direct_opd/lora/flux2_klein
LAMBDAS=("${@:-050 100}")
PORT_BASE=${PORT_BASE:-29820}

log() { echo "[oracle-sweep $(date '+%F %T')] $*"; }

kill_runs() {
  local pat="pkill -9 -f '[f]low_factory.train' 2>/dev/null; true"
  for ip in "${WORKERS[@]}"; do $SSH "$ip" "$pat" || true; done
  bash -c "$pat" || true
}

gpu_mem() {
  local q="nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc"
  if [ "$1" = local ]; then bash -c "$q" 2>/dev/null || echo 999999
  else $SSH "$1" "$q" 2>/dev/null || echo 999999; fi
}

wait_gpus_free() {
  local deadline=$(( $(date +%s) + ${1:-300} )) node mem busy
  while [ "$(date +%s)" -lt "$deadline" ]; do
    busy=""
    for node in local "${WORKERS[@]}"; do
      mem=$(gpu_mem "$node"); [ "${mem:-999999}" -gt 2048 ] && busy="$busy $node(${mem}MiB)"
    done
    [ -z "$busy" ] && return 0
    log "waiting for GPUs to drain:$busy"
    sleep 15
  done
  log "TIMEOUT: GPUs still busy:$busy"
  return 1
}

i=0
for tag in ${LAMBDAS[@]}; do
  config="$CONFIG_DIR/_oracle_klein9b_to_4b_lambda${tag}.yaml"
  [ -f "$config" ] || { log "missing config $config"; exit 1; }
  port=$((PORT_BASE + i)); i=$((i + 1))
  log "================= lambda 0.${tag} ($config, port $port)"
  kill_runs
  wait_gpus_free 300 || { log "aborting: GPUs never drained"; exit 1; }

  MASTER_PORT="$port" bash "$LAUNCHER" "$config" > "/root/oracle_${tag}_launch.log" 2>&1
  status=$?
  cp -f /root/ff_xopd_rank0.log "/root/oracle_${tag}_rank0.log" 2>/dev/null || true
  log "lambda 0.${tag} finished with exit $status"
  grep -a "reward_geneval_mean=\|reward_ocr_mean=\|oracle field over" "/root/oracle_${tag}_rank0.log" 2>/dev/null \
    | tail -3 | cut -c1-260
  sleep 20
done

kill_runs
log "================= sweep done"
