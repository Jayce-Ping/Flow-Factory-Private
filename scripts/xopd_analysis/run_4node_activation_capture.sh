#!/bin/bash
# Four-node x 8-GPU launcher for activation-capture diagnostics.
set -euo pipefail

MASTER_IP=${MASTER_IP:-28.7.186.81}
MASTER_PORT=${MASTER_PORT:-29740}
read -r -a WORKERS <<< "${XOPD_WORKERS-28.7.193.116 28.7.193.87 28.7.187.4}"
NUM_NODES=$((1 + ${#WORKERS[@]}))
SCRIPT=scripts/xopd_analysis/capture_teacher_student_activations.py
EXTRA=("$@")
PATTERN='[c]apture_teacher_student_activations.py'

read -r -d '' COMMON <<'EOF' || true
source /opt/conda/etc/profile.d/conda.sh
conda activate ff
cd /root/Flow-Factory-Private
unset PYTHONPATH CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPYCACHEPREFIX=/tmp/ffpyc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=WARN
pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null || true
sleep 2
EOF

restart_keepalive() {
  for ip in "${WORKERS[@]}"; do
    ssh -o StrictHostKeyChecking=no -f "$ip" \
      "source /opt/conda/etc/profile.d/conda.sh && conda activate ff && \
       pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null || true; \
       setsid python /root/Flow-Factory-Private/.scratch/gpu_keepalive.py \
       > /root/gpu_keepalive.log 2>&1 < /dev/null &" || true
  done
  pkill -9 -f '[g]pu_keepalive.py' 2>/dev/null || true
  setsid /opt/conda/envs/ff/bin/python .scratch/gpu_keepalive.py \
    > /root/gpu_keepalive.log 2>&1 < /dev/null &
}

cleanup() {
  for ip in "${WORKERS[@]}"; do
    ssh -o StrictHostKeyChecking=no "$ip" \
      "pkill -9 -f '$PATTERN' 2>/dev/null || true" || true
  done
  pkill -9 -f "$PATTERN" 2>/dev/null || true
  restart_keepalive
}
trap cleanup EXIT INT TERM

rank=1
for ip in "${WORKERS[@]}"; do
  ssh -o StrictHostKeyChecking=no "$ip" \
    "$COMMON; nohup /opt/conda/envs/ff/bin/torchrun \
      --nnodes=$NUM_NODES --nproc-per-node=8 --node-rank=$rank \
      --master-addr=$MASTER_IP --master-port=$MASTER_PORT \
      $SCRIPT ${EXTRA[*]} > /root/activation_capture_rank${rank}.log 2>&1 &"
  rank=$((rank + 1))
done

eval "$COMMON"
/opt/conda/envs/ff/bin/torchrun \
  --nnodes="$NUM_NODES" --nproc-per-node=8 --node-rank=0 \
  --master-addr="$MASTER_IP" --master-port="$MASTER_PORT" \
  "$SCRIPT" "${EXTRA[@]}" > /root/activation_capture_rank0.log 2>&1
