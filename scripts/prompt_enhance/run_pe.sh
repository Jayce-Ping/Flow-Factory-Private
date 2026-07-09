#!/bin/bash
# Prompt-enhance a geneval JSONL over N GPUs on this node (one vLLM process per GPU), then aggregate.
# Usage: run_pe.sh <input.jsonl> <outdir> <name>   e.g. run_pe.sh dataset/geneval/train.jsonl /root/pe_out train
set -uo pipefail
INPUT=${1:?}; OUTDIR=${2:?}; NAME=${3:?}
MODEL=${MODEL:-Qwen/Qwen3-VL-32B-Instruct}
N=${N:-8}
source /opt/conda/etc/profile.d/conda.sh; conda activate vllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$OUTDIR/shards"
n_in=$(grep -c . "$INPUT")
echo "[$(date +%T)] PE $NAME: $n_in lines over $N GPUs (model=$MODEL)"
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i nohup python /root/pe_geneval_hf.py \
    --input "$INPUT" --output "$OUTDIR/shards/${NAME}_shard${i}.jsonl" \
    --model "$MODEL" --shard "$i" --num_shards "$N" \
    > "$OUTDIR/shards/${NAME}_shard${i}.log" 2>&1 &
  pids+=($!)
  sleep 3   # stagger model loads to avoid a host-RAM spike
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date +%T)] shards done; aggregating"
python /root/pe_aggregate.py \
  --shards_glob "$OUTDIR/shards/${NAME}_shard*.jsonl" \
  --n_expected "$n_in" --output "$OUTDIR/${NAME}.jsonl"
echo "[$(date +%T)] PE $NAME complete"
