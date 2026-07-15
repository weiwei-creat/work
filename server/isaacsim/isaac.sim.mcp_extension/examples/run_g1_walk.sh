#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-command launcher for the Phase-B G1 walking visualization.
#
# Usage:
#   ./run_g1_walk.sh <layout_id>              # windowed, loops the walk forever
#   ./run_g1_walk.sh <layout_id> --headless   # offscreen render -> PNG frames + mp4
#   ./run_g1_walk.sh                           # uses the most recent layout in results/
#
# It resolves all paths itself and uses Isaac Sim's own python.sh, so you do NOT
# need to activate conda. Override defaults with env vars if needed:
#   ISAAC_SIM_PATH (default: $HOME/coding/isaacsim)
#   SAGE_RENDER_OUT, SAGE_G1_USD, SAGE_WALK_LOOPS, SAGE_CAM_BACK/UP/TARGET_Z, DISPLAY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# examples -> isaac.sim.mcp_extension -> isaacsim -> server -> SAGE_ROOT
SAGE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RESULTS_DIR="${SAGE_ROOT}/server/results"

LAYOUT_ID="${1:-}"
MODE="${2:-}"

# Default to the most recently modified layout if none given.
if [ -z "${LAYOUT_ID}" ]; then
    LAYOUT_ID="$(ls -dt "${RESULTS_DIR}"/*/ 2>/dev/null | head -1 | xargs -r basename || true)"
    if [ -z "${LAYOUT_ID}" ]; then
        echo "[ERROR] No layout_id given and no layouts found in ${RESULTS_DIR}" >&2
        echo "Usage: $0 <layout_id> [--headless]" >&2
        exit 1
    fi
    echo "[INFO] No layout_id given; using most recent: ${LAYOUT_ID}"
fi

LAYOUT_DIR="${RESULTS_DIR}/${LAYOUT_ID}"
if [ ! -d "${LAYOUT_DIR}" ]; then
    echo "[ERROR] Layout dir not found: ${LAYOUT_DIR}" >&2
    exit 1
fi
if [ ! -f "${LAYOUT_DIR}/${LAYOUT_ID}_g1_nav_path.json" ]; then
    echo "[ERROR] Nav path not found: ${LAYOUT_DIR}/${LAYOUT_ID}_g1_nav_path.json" >&2
    echo "        Run phase-A generation for a unitree_g1 task first." >&2
    exit 1
fi

ISAACSIM_PATH="${ISAAC_SIM_PATH:-${HOME}/coding/isaacsim}"
PYTHON_SH="${ISAACSIM_PATH}/_build/linux-x86_64/release/python.sh"
if [ ! -x "${PYTHON_SH}" ]; then
    echo "[ERROR] Isaac python.sh not found: ${PYTHON_SH}" >&2
    echo "        Set ISAAC_SIM_PATH to your Isaac Sim install root." >&2
    exit 1
fi

export SAGE_LAYOUT_DIR="${LAYOUT_DIR}"
export SAGE_LAYOUT_ID="${LAYOUT_ID}"
export SAGE_RENDER_OUT="${SAGE_RENDER_OUT:-/tmp/g1_render_${LAYOUT_ID}}"

if [ "${MODE}" = "--headless" ]; then
    export SAGE_RENDER_HEADLESS=1
    echo "[INFO] Headless render -> ${SAGE_RENDER_OUT}"
else
    export SAGE_RENDER_HEADLESS=0
    export DISPLAY="${DISPLAY:-:0}"
    export XAUTHORITY="${XAUTHORITY:-${HOME}/.Xauthority}"
    echo "[INFO] Windowed mode on DISPLAY=${DISPLAY} (looping walk; close window to exit)"
fi

echo "[INFO] layout_id = ${LAYOUT_ID}"
echo "[INFO] launching Isaac Sim (no conda needed) ..."
exec "${PYTHON_SH}" "${SCRIPT_DIR}/g1_walk_render.py"
