#!/usr/bin/env bash
set -euo pipefail

SAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRELLIS_DIR="${SAGE_ROOT}/trellis"
BACKUP_DIR="/tmp/trellis_checkpoints_backup"

echo "=== Clone Microsoft TRELLIS with China mirror ==="

# Try mirrors in order
if git clone --depth 1 https://gitclone.com/github.com/microsoft/TRELLIS.git "${TRELLIS_DIR}" 2>/dev/null; then
    echo "✓ Cloned via gitclone.com"
elif git clone --depth 1 https://hub.原梓潼.com/microsoft/TRELLIS.git "${TRELLIS_DIR}" 2>/dev/null; then
    echo "✓ Cloned via hub.原梓潼.com"
elif git clone --depth 1 https://ghfast.top/github.com/microsoft/TRELLIS.git "${TRELLIS_DIR}" 2>/dev/null; then
    echo "✓ Cloned via ghfast.top"
elif git clone --depth 1 https://ghproxy.net/github.com/microsoft/TRELLIS.git "${TRELLIS_DIR}" 2>/dev/null; then
    echo "✓ Cloned via ghproxy.net"
elif git clone --depth 1 https://github.com/microsoft/TRELLIS.git "${TRELLIS_DIR}"; then
    echo "✓ Cloned via GitHub (slow)"
else
    echo "✗ All mirrors failed. Try manually: git clone https://github.com/microsoft/TRELLIS.git"
    exit 1
fi

# Restore checkpoint files
if [[ -d "${BACKUP_DIR}" ]]; then
    echo "=== Restoring checkpoint files ==="
    mkdir -p "${TRELLIS_DIR}/checkpoints"
    cp -r "${BACKUP_DIR}/"* "${TRELLIS_DIR}/checkpoints/" 2>/dev/null || true
    echo "✓ Checkpoints restored"
    echo "  $(find "${TRELLIS_DIR}/checkpoints" -type f | wc -l) files restored"
fi

# Install TRELLIS in trellis conda env
echo "=== Install TRELLIS into conda env 'trellis' ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trellis

cd "${TRELLIS_DIR}"
pip install -e . 2>&1 | tail -3

echo ""
echo "=== Done ==="
echo "TRELLIS installed at: ${TRELLIS_DIR}"
echo "To test: conda activate trellis && python -c 'from trellis.pipelines import TrellisTextTo3DPipeline; print(\"OK\")'"
