#!/bin/bash
# Four-node x 8-GPU launcher for activation-capture diagnostics.
set -euo pipefail

MASTER_IP=${MASTER_IP:-28.7.193.116}
MASTER_PORT=${MASTER_PORT:-29740}
WORKERS=(28.7.185.215 28.7.185.156 28.7.195.15)
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
       setsid python /root/gpu_keepalive.py > /root/gpu_keepalive.log 2>&1 < /dev/null &" || true
  done
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
      --nnodes=4 --nproc-per-node=8 --node-rank=$rank \
      --master-addr=$MASTER_IP --master-port=$MASTER_PORT \
      $SCRIPT ${EXTRA[*]} > /root/activation_capture_rank${rank}.log 2>&1 &"
  rank=$((rank + 1))
done

eval "$COMMON"
/opt/conda/envs/ff/bin/torchrun \
  --nnodes=4 --nproc-per-node=8 --node-rank=0 \
  --master-addr="$MASTER_IP" --master-port="$MASTER_PORT" \
  "$SCRIPT" "${EXTRA[@]}" > /root/activation_capture_rank0.log 2>&1
