#!/bin/bash
# Canonical multi-node x 8-GPU XOPD launcher (Flow-Factory-Private, 28.7.193.116 + N workers).
# Defaults to the 4-node cluster; set XOPD_WORKERS to run on a subset (e.g. a 2-node run that
# frees a node for a vLLM judge + a node for keepalive). NUM_MACHINES is derived from the worker
# list, so the config's num_processes is overridden to NUM_MACHINES * 8 via the env below.
#
# Foreground/blocking: launches ranks 1..N-1 on the workers (ssh, one backgrounded ff-train per
# node) and rank 0 locally in the FOREGROUND, so a caller (scripts/xopd_cluster/run_three_experts.sh)
# can wait on it and read the exit code. On exit/interrupt it kills the ff-train procs for THIS
# config on every node so the next run starts from a clean slate.
#
# The cluster's IB fabric is broken, so NCCL runs over bond1 TCP (the env below is the proven
# 4-node ZeRO-2 path). HF Hub is forced offline: every shard is pre-cached, and a single rank
# hitting huggingface.co (429 rate-limit) would desync the collective group and hang the run.
#
# Usage:
#   MASTER_PORT=29570 bash scripts/xopd_cluster/run_4node_xopd.sh <config.yaml> [extra ff-train args...]
#   XOPD_WORKERS="28.7.185.215" MASTER_PORT=29600 bash scripts/xopd_cluster/run_4node_xopd.sh <cfg>  # 2-node
#
# Env knobs (defaults in parens):
#   MASTER_IP (28.7.193.116)  MASTER_PORT (29540)
#   NCCL_DEBUG (WARN)         -> raise to INFO only when debugging NCCL; INFO floods the log
#   FLOW_FACTORY_EVAL_GLOO_BARRIER (1) -> CPU monitored_barrier at eval stage boundaries; a single
#                                         rank that died (e.g. HF 429) fails fast with a named rank
#   FLOW_FACTORY_EVAL_DEBUG (1)        -> per-rank eval stage markers in the logs (localizes hangs)
set -uo pipefail
CONFIG=${1:?"usage: run_4node_xopd.sh <config.yaml> [extra args]"}; shift || true
EXTRA="$*"

MASTER_IP=${MASTER_IP:-28.7.193.116}
MASTER_PORT=${MASTER_PORT:-29540}
# Worker IPs (ranks 1..N-1). Override with XOPD_WORKERS to run on fewer/more nodes,
# e.g. XOPD_WORKERS="28.7.185.215" -> 2-node run (node0 rank0 + this worker rank1),
# leaving the other nodes free for a vLLM judge / keepalive. NUM_MACHINES is derived.
read -r -a WORKERS <<< "${XOPD_WORKERS:-28.7.185.215 28.7.185.156 28.7.195.15}"
NUM_MACHINES=$((1 + ${#WORKERS[@]}))

NCCL_DEBUG_LEVEL=${NCCL_DEBUG:-WARN}
GLOO_BARRIER=${FLOW_FACTORY_EVAL_GLOO_BARRIER:-1}
EVAL_DEBUG=${FLOW_FACTORY_EVAL_DEBUG:-1}

# Per-rank environment. Heredoc is EXPANDED (<<EOF) so the knob values above bake in; there are
# no other shell refs inside, so nothing else is accidentally interpolated.
read -r -d '' COMMON <<EOF || true
export http_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128
# Never route intra-cluster traffic through the external proxy: the GEditBench eval reward's
# OpenAI/httpx client must reach the vLLM judge (a bond1 IP, e.g. 28.7.195.15:8000) DIRECTLY,
# otherwise every request is proxied to star-proxy and fails (that reward -> nan / eval hang).
export no_proxy="localhost,127.0.0.1,28.7.195.15,28.7.193.116,28.7.185.215,28.7.185.156"
export NO_PROXY="\$no_proxy"
source /opt/conda/etc/profile.d/conda.sh; conda activate ff
unset PYTHONPATH
export PYTHONPYCACHEPREFIX=/tmp/ffpyc
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_GID_INDEX=3 NCCL_IB_SL=3 NCCL_IB_TC=160
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_IB_CUDA_SUPPORT=1 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_GDR_LEVEL=2 NCCL_PXN_DISABLE=1
export NCCL_COLLNET_ENABLE=0 SHARP_COLL_ENABLE_SAT=0
export NCCL_CHECKS_DISABLE=1 NCCL_LL_THRESHOLD=16384
export NCCL_SOCKET_IFNAME=bond1 UCX_NET_DEVICES=bond1
export NCCL_DEBUG=${NCCL_DEBUG_LEVEL}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLOW_FACTORY_EVAL_GLOO_BARRIER=${GLOO_BARRIER} FLOW_FACTORY_EVAL_DEBUG=${EVAL_DEBUG}
pkill -9 -f "[g]pu_burn.py" 2>/dev/null || true
pkill -9 -f "[g]pu_keepalive.py" 2>/dev/null || true
sleep 2
cd /root/Flow-Factory-Private
EOF

LAUNCH_ENV="export MASTER_IP=$MASTER_IP MASTER_PORT=$MASTER_PORT NUM_MACHINES=$NUM_MACHINES GPUS_PER_NODE=8"

cleanup() {
  echo ">>> [launcher] cleanup: killing ff-train for '$CONFIG' on all nodes"
  for ip in "${WORKERS[@]}"; do
    ssh -o StrictHostKeyChecking=no "$ip" \
      "pkill -9 -f '[f]f-train $CONFIG' 2>/dev/null; pkill -9 -f 'flow_factory.train $CONFIG' 2>/dev/null" || true
  done
  pkill -9 -f "[f]f-train $CONFIG" 2>/dev/null || true
  pkill -9 -f "flow_factory.train $CONFIG" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ">>> [launcher] config=$CONFIG num_machines=$NUM_MACHINES workers=(${WORKERS[*]}) master=$MASTER_IP:$MASTER_PORT nccl_debug=$NCCL_DEBUG_LEVEL gloo_barrier=$GLOO_BARRIER eval_debug=$EVAL_DEBUG"
rank=1
for ip in "${WORKERS[@]}"; do
  echo ">>> [launcher] rank $rank on $ip"
  ssh -o StrictHostKeyChecking=no "$ip" \
    "$COMMON; $LAUNCH_ENV MACHINE_RANK=$rank; nohup ff-train $CONFIG $EXTRA > /root/ff_xopd_rank${rank}.log 2>&1 & echo rank$rank PID \$!"
  rank=$((rank + 1))
done

echo ">>> [launcher] rank 0 on launcher node (FOREGROUND, blocking)"
eval "$COMMON"
eval "$LAUNCH_ENV MACHINE_RANK=0"
ff-train "$CONFIG" $EXTRA > /root/ff_xopd_rank0.log 2>&1
RC=$?
echo ">>> [launcher] rank0 exited with code $RC"
exit $RC
