#!/bin/bash
# Two-node vLLM service for the official Qwen3-VL-235B-A22B-Instruct constructor.
set -euo pipefail

ACTION=${1:-status}
HEAD_SSH=${QWEN_HEAD_SSH:-28.7.185.156}
WORKER_SSH=${QWEN_WORKER_SSH:-28.7.195.15}
HEAD_IP=${QWEN_HEAD_IP:-28.7.185.156}
WORKER_IP=${QWEN_WORKER_IP:-28.7.195.15}
PORT=${QWEN_PORT:-8000}
RAY_PORT=${QWEN_RAY_PORT:-6379}
MODEL=${QWEN_MODEL:-/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/models/Qwen3-VL-235B-A22B-Instruct}
SERVED_NAME=${QWEN_SERVED_NAME:-qwen3-vl-235b-a22b-instruct}
ENV=/opt/conda/envs/vllm-judge
SELF=/root/Flow-Factory-Private/dataset/hard_data/serve_qwen3vl_235b.sh
WATCHDOG=/root/Flow-Factory-Private/dataset/hard_data/service_keepalive_watchdog.sh
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"

verify_prerequisites() {
  for node in "$HEAD_SSH" "$WORKER_SSH"; do
    $SSH "$node" "test -x $ENV/bin/vllm" || {
      echo "missing $ENV on $node; run install_vllm_judge.sh there" >&2
      exit 1
    }
  done
  test -f "$MODEL/config.json" || {
    echo "missing model config: $MODEL/config.json" >&2
    exit 1
  }
}

stop_service() {
  for node in "$HEAD_SSH" "$WORKER_SSH"; do
    $SSH "$node" \
      "pkill -9 -f '[v]llm serve.*Qwen3-VL-235B' 2>/dev/null || true; \
       pkill -9 -f '[s]ervice_keepalive_watchdog.sh' 2>/dev/null || true; \
       pkill -9 -f '^python .*gpu_keepalive\\.py' 2>/dev/null || true; \
       $ENV/bin/python -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true"
  done
}

start_service() {
  verify_prerequisites
  stop_service
  for node in "$HEAD_SSH" "$WORKER_SSH"; do
    $SSH "$node" "pkill -9 -f '^python .*gpu_keepalive\\.py' 2>/dev/null || true"
  done

  $SSH "$HEAD_SSH" \
    "NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1 \
     $ENV/bin/python -m ray.scripts.scripts start --head \
     --node-ip-address=$HEAD_IP --port=$RAY_PORT --num-gpus=8"
  $SSH "$WORKER_SSH" \
    "NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1 \
     $ENV/bin/python -m ray.scripts.scripts start --address=$HEAD_IP:$RAY_PORT \
     --node-ip-address=$WORKER_IP --num-gpus=8"

  $SSH -f "$HEAD_SSH" \
    "QWEN_MODEL='$MODEL' QWEN_SERVED_NAME='$SERVED_NAME' QWEN_PORT='$PORT' \
     QWEN_HEAD_IP='$HEAD_IP' setsid nohup bash '$SELF' local-serve \
     > /root/qwen3vl_235b_serve.log 2>&1 < /dev/null"

  echo "waiting for http://$HEAD_IP:$PORT/v1/models ..."
  for _ in $(seq 1 240); do
    if curl --noproxy '*' -fsS --max-time 3 "http://$HEAD_IP:$PORT/v1/models" >/dev/null 2>&1; then
      for node in "$HEAD_SSH" "$WORKER_SSH"; do
        $SSH -f "$node" \
          "QWEN_METRICS_URL='http://$HEAD_IP:$PORT/metrics' \
           setsid nohup bash '$WATCHDOG' > /root/qwen_keepalive_watchdog_runner.log 2>&1 < /dev/null"
      done
      echo "Qwen service ready: http://$HEAD_IP:$PORT/v1 ($SERVED_NAME)"
      return 0
    fi
    sleep 10
  done
  echo "Qwen service did not become ready within 40 minutes" >&2
  $SSH "$HEAD_SSH" "tail -80 /root/qwen3vl_235b_serve.log" >&2 || true
  return 1
}

serve_local() {
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
  export VLLM_HOST_IP=${QWEN_HEAD_IP:-28.7.185.156}
  export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
  exec "$ENV/bin/python" -m vllm.entrypoints.cli.main serve \
    "${QWEN_MODEL:?QWEN_MODEL is required}" \
    --served-model-name "${QWEN_SERVED_NAME:-qwen3-vl-235b-a22b-instruct}" \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --enforce-eager \
    --host 0.0.0.0 \
    --port "${QWEN_PORT:-8000}" \
    --gpu-memory-utilization "${QWEN_GPU_UTIL:-0.85}" \
    --trust-remote-code \
    --max-model-len "${QWEN_MAX_LEN:-32768}" \
    --max-num-seqs "${QWEN_MAX_NUM_SEQS:-16}" \
    --limit-mm-per-prompt '{"image":10}'
}

status_service() {
  if curl --noproxy '*' -fsS --max-time 5 "http://$HEAD_IP:$PORT/v1/models"; then
    echo
    $SSH "$HEAD_SSH" "$ENV/bin/python -m ray.scripts.scripts status" || true
    return 0
  fi
  echo "Qwen service is not reachable at http://$HEAD_IP:$PORT/v1" >&2
  return 1
}

case "$ACTION" in
  start) start_service ;;
  stop) stop_service ;;
  status) status_service ;;
  local-serve) serve_local ;;
  *)
    echo "usage: $0 {start|stop|status|local-serve}" >&2
    exit 2
    ;;
esac
