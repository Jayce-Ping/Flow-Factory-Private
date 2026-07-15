#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

# Thin wrapper around download_and_build.py (HF load + images/ + train/test.jsonl).
# Optional args are forwarded (e.g. --force, --max_samples 8).
"$PYTHON" "$SCRIPT_DIR/download_and_build.py" --out_dir "$SCRIPT_DIR" "$@"

echo "GEditBench-v2 build completed under $SCRIPT_DIR"
