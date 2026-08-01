#!/bin/bash
# Lightweight wrapper around reconstruct.py.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${HARD_DATA_PYTHON:-/opt/conda/envs/ff/bin/python}

[ -x "$PYTHON" ] || {
  echo "missing Python executable: $PYTHON" >&2
  exit 1
}

cd "$ROOT"
exec "$PYTHON" dataset/hard_data/reconstruct.py "$@"
