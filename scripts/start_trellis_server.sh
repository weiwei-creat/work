#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8080}"
PIPELINE_PATH="${2:-/home/gaok/coding/TRELLIS}"
CONDA_ENV="${3:-trellis5080}"
SAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Starting TRELLIS Server ==="
echo "  Port:          ${PORT}"
echo "  Pipeline path: ${PIPELINE_PATH}"
echo "  Conda env:     ${CONDA_ENV}"

export PYTHONPATH="${PIPELINE_PATH}:${PYTHONPATH:-}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

exec conda run -n "${CONDA_ENV}" --no-capture-output \
    python "${SAGE_ROOT}/scripts/trellis_flask_server.py" \
    --port "${PORT}" \
    --pipeline-path "${PIPELINE_PATH}"
