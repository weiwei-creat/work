#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash render_preview.sh /path/to/layout_id/layout_id.json" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYOUT_JSON_PATH="$1"
LAYOUT_ID="$(basename "${LAYOUT_JSON_PATH}" .json)"

python "${SCRIPT_DIR}/isaaclab/layout_preview.py" --layout_id "${LAYOUT_ID}"
