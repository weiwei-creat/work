#!/usr/bin/env bash
set -euo pipefail

echo "[setup-sage] This clone-based script is deprecated."
echo "[setup-sage] Forwarding to scripts/setup_sage_fresh_conda.sh ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/setup_sage_fresh_conda.sh" "$@"

SOURCE_ENV="${SOURCE_ENV:-env_isaaclab}"
TARGET_ENV="${TARGET_ENV:-sage}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/setup_sage_$(date +%Y%m%d_%H%M%S).log}"
SKIP_CLONE=0
RECREATE=0
SKIP_HEAVY=0
SKIP_LOCAL=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Clone an existing IsaacLab conda environment and install SAGE dependencies.

Options:
  --source-env NAME   Source conda env to clone. Default: env_isaaclab
  --target-env NAME   Target conda env to create/update. Default: sage
  --skip-clone        Do not clone; install into an existing target env.
  --recreate          Remove target env before cloning. Requires confirmation.
  --skip-heavy        Skip packages that may compile or need external indexes.
  --skip-local        Skip editable installs for local IsaacLab/M2T2/robomimic.
  -h, --help          Show this help.

Examples:
  ./scripts/setup_sage_from_env_isaaclab.sh
  ./scripts/setup_sage_from_env_isaaclab.sh --target-env sage --skip-heavy
  SOURCE_ENV=env_isaaclab TARGET_ENV=sage ./scripts/setup_sage_from_env_isaaclab.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-env)
            SOURCE_ENV="$2"
            shift 2
            ;;
        --target-env)
            TARGET_ENV="$2"
            shift 2
            ;;
        --skip-clone)
            SKIP_CLONE=1
            shift
            ;;
        --recreate)
            RECREATE=1
            shift
            ;;
        --skip-heavy)
            SKIP_HEAVY=1
            shift
            ;;
        --skip-local)
            SKIP_LOCAL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [setup-sage] $*" | tee -a "${LOG_FILE}"
}

have_env() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

run_pip() {
    conda run --no-capture-output -n "${TARGET_ENV}" python -m pip "$@"
}

run_step() {
    local title="$1"
    shift

    log "START: ${title}"
    set +e
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -ne 0 ]]; then
        log "FAILED: ${title} (exit ${status})"
        exit "${status}"
    fi
    log "DONE: ${title}"
}

run_optional_step() {
    local title="$1"
    shift

    log "START: ${title}"
    set +e
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -ne 0 ]]; then
        log "WARN: ${title} failed (exit ${status}); continuing."
        return "${status}"
    fi
    log "DONE: ${title}"
}

run_pip_install_batch() {
    local title="$1"
    shift

    log "Packages: $*"
    run_step "${title}" run_pip install --progress-bar raw "$@"
}

run_optional_pip_install_batch() {
    local title="$1"
    shift

    log "Packages: $*"
    run_optional_step "${title}" run_pip install --progress-bar raw "$@"
}

run_pip_install_each() {
    local title="$1"
    shift

    log "START GROUP: ${title}"
    local package
    for package in "$@"; do
        log "PACKAGE START: ${package}"
        run_step "${title}: ${package}" run_pip install --progress-bar raw "${package}"
        log "PACKAGE DONE: ${package}"
    done
    log "DONE GROUP: ${title}"
}

run_optional_pip_install_each() {
    local title="$1"
    shift

    log "START GROUP: ${title}"
    local package
    for package in "$@"; do
        log "PACKAGE START: ${package}"
        run_optional_step "${title}: ${package}" run_pip install --progress-bar raw "${package}" || true
        log "PACKAGE FINISHED: ${package}"
    done
    log "DONE GROUP: ${title}"
}

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"
log "Writing detailed install log to ${LOG_FILE}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not available in PATH." >&2
    exit 1
fi

if [[ "${SKIP_CLONE}" -eq 0 ]]; then
    if ! have_env "${SOURCE_ENV}"; then
        echo "[ERROR] Source conda env '${SOURCE_ENV}' does not exist." >&2
        echo "        Check with: conda env list" >&2
        exit 1
    fi
fi

if [[ "${SKIP_CLONE}" -eq 0 ]]; then
    if have_env "${TARGET_ENV}"; then
        if [[ "${RECREATE}" -eq 1 ]]; then
            read -r -p "Remove existing conda env '${TARGET_ENV}' and recreate it? [y/N] " answer
            if [[ "${answer}" != "y" && "${answer}" != "Y" ]]; then
                echo "[ERROR] Aborted." >&2
                exit 1
            fi
            log "Removing existing env '${TARGET_ENV}'..."
            run_step "Remove existing env ${TARGET_ENV}" conda env remove -y -n "${TARGET_ENV}"
        else
            log "Target env '${TARGET_ENV}' already exists; reusing it."
            log "Use --recreate to clone '${SOURCE_ENV}' again from scratch."
        fi
    fi

    if ! have_env "${TARGET_ENV}"; then
        log "Cloning '${SOURCE_ENV}' -> '${TARGET_ENV}'..."
        run_step "Clone conda env ${SOURCE_ENV} -> ${TARGET_ENV}" conda create -y -n "${TARGET_ENV}" --clone "${SOURCE_ENV}"
    fi
else
    if ! have_env "${TARGET_ENV}"; then
        echo "[ERROR] --skip-clone was set, but target env '${TARGET_ENV}' does not exist." >&2
        exit 1
    fi
fi

run_pip_install_batch "Upgrade pip tooling" -U pip setuptools wheel packaging ninja

log "Installing SAGE core Python dependencies in smaller observable batches..."
run_pip_install_each "Core APIs and utilities" \
    "anthropic>=0.65.0" \
    "mcp[cli]>=1.10.1" \
    "python-dotenv>=1.1.1" \
    httpx openai requests rich tqdm loguru \
    compress-pickle compress-json \
    hydra-core omegaconf s3fs

run_pip_install_each "Numeric, plotting, and ML utilities" \
    matplotlib seaborn pandas \
    "numpy>=1.24,<2" scipy scikit-learn scikit-image \
    h5py tensorboard tensorboardX "umap-learn"

run_pip_install_each "Geometry, mesh, and scene packages" \
    trimesh open3d==0.19.0 rtree shapely editdistance plyfile \
    pygltflib embreex manifold3d xatlas libigl mapbox-earcut yourdfpy

run_pip_install_each "Image, video, and rendering helpers" \
    imageio "imageio[ffmpeg]" opencv-python pillow==9.5.0 \
    bpy extcolors Pylette kornia==0.7.2 meshcat

run_pip_install_each "Robot and simulation Python packages" \
    gymnasium mujoco==3.3.4 robosuite sapien

run_pip_install_each "Model clients and foundation-model helpers" \
    open-clip-torch==2.32.0 sentence-transformers==4.1.0 \
    transformers==4.41.2 diffusers==0.11.1 accelerate==0.23.0 \
    "qwen-agent[gui,rag,code_interpreter,mcp]==0.0.26" \
    wandb

if [[ "${SKIP_HEAVY}" -eq 0 ]]; then
    log "Installing heavy/compiled dependencies..."
    run_optional_pip_install_batch "Heavy dependency: nvidia-curobo" --extra-index-url https://pypi.nvidia.com nvidia-curobo || {
        echo "[WARN] nvidia-curobo install failed. You can retry manually after checking CUDA/NVIDIA index access." >&2
    }
    run_optional_pip_install_batch "Heavy dependency: nvdiffrast" "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git" || {
        echo "[WARN] nvdiffrast install failed. You can retry manually if rendering/mesh ops need it." >&2
    }
    run_optional_pip_install_batch "Heavy dependency: pytorch3d" "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable" || {
        echo "[WARN] pytorch3d install failed. You can retry manually after confirming torch/CUDA compatibility." >&2
    }
else
    log "Skipping heavy/compiled dependencies."
fi

if [[ "${SKIP_LOCAL}" -eq 0 ]]; then
    log "Installing local editable packages..."
    if [[ -x "${REPO_ROOT}/IsaacLab/isaaclab.sh" ]]; then
        (
            set +u
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "${TARGET_ENV}"
            run_step "Editable install: IsaacLab extensions" "${REPO_ROOT}/IsaacLab/isaaclab.sh" -i all
        )
    else
        log "IsaacLab/isaaclab.sh is not executable; skipping IsaacLab editable install."
    fi

    if [[ "${SKIP_HEAVY}" -eq 0 ]]; then
        run_optional_pip_install_batch "Editable install: M2T2 pointnet2_ops" -e "${REPO_ROOT}/M2T2/pointnet2_ops" || {
            echo "[WARN] pointnet2_ops editable install failed. This usually needs a working CUDA compiler toolchain." >&2
        }
    else
        log "Skipping pointnet2_ops because --skip-heavy was set."
    fi
    run_optional_pip_install_batch "Editable install: M2T2" -e "${REPO_ROOT}/M2T2" || {
        echo "[WARN] M2T2 editable install failed." >&2
    }
    run_optional_pip_install_batch "Editable install: robomimic" -e "${REPO_ROOT}/robomimic" || {
        echo "[WARN] robomimic editable install failed." >&2
    }
else
    log "Skipping local editable packages."
fi

log "Running a lightweight import check..."
run_step "Lightweight import check" conda run --no-capture-output -n "${TARGET_ENV}" python - <<'PY'
import importlib

mods = [
    "torch",
    "omni.isaac.lab",
    "mcp",
    "openai",
    "trimesh",
    "open3d",
    "mujoco",
    "robosuite",
]

missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)))

if missing:
    print("[WARN] Some imports failed:")
    for name, exc in missing:
        print(f"  - {name}: {exc}")
else:
    print("[OK] Basic imports succeeded.")
PY

log "Done."
log "Detailed log: ${LOG_FILE}"
echo
echo "Activate it with:"
echo "  conda activate ${TARGET_ENV}"
