#!/usr/bin/env bash
# Wrapper for scripts/mof_evaluate.py — multi-GPU launch via accelerate.
#
# Usage:
#   bash scripts/run_mof_evaluate.sh \
#       <config.yaml> \
#       <checkpoint-dir> \
#       [output-dir] \
#       [extra args ...]
#
# Examples:
#   # Default mode=both, output saves/evaluate_figures/
#   bash scripts/run_mof_evaluate.sh \
#       mof_configs/nft/geneval_pickscore_ocr_lut.yaml \
#       saves/run-name/checkpoints/checkpoint-100
#
#   # Custom output dir + override eval params
#   bash scripts/run_mof_evaluate.sh \
#       mof_configs/nft/geneval_pickscore_ocr_lut.yaml \
#       saves/run-name/checkpoints/checkpoint-100 \
#       saves/evaluate_figures/run-name-ckpt100 \
#       --num-inference-steps 40 --guidance-scale 4.5
#
#   # No-EMA only (skip EMA pass)
#   bash scripts/run_mof_evaluate.sh \
#       <config> <checkpoint> "" --mode no_ema
#
#   # Override the accelerate config (default: config/accelerate_configs/multi_gpu.yaml)
#   ACCELERATE_CONFIG=config/accelerate_configs/single_gpu.yaml \
#       bash scripts/run_mof_evaluate.sh <config> <checkpoint>
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <config.yaml> <checkpoint-dir> [output-dir] [extra args ...]" >&2
    exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
OUTPUT_DIR="${3:-saves/evaluate_figures}"
shift 2
# Drop the (possibly empty) third positional so $@ holds only the extra args.
if [[ $# -gt 0 ]]; then shift || true; fi

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-config/accelerate_configs/multi_gpu.yaml}"

echo "[run_mof_evaluate] config:        ${CONFIG}"
echo "[run_mof_evaluate] checkpoint:    ${CHECKPOINT}"
echo "[run_mof_evaluate] output dir:    ${OUTPUT_DIR}"
echo "[run_mof_evaluate] accelerate:    ${ACCELERATE_CONFIG}"
echo "[run_mof_evaluate] extra args:    $*"

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    scripts/mof_evaluate.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
