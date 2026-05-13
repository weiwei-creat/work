#!/usr/bin/env bash
set -euo pipefail

TARGET_ENV="${TARGET_ENV:-sage}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_PATH="${ISAACSIM_PATH:-/home/gaok/coding/isaacsim/_build/linux-x86_64/release}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/setup_sage_fresh_$(date +%Y%m%d_%H%M%S).log}"
RECREATE=0
SKIP_HEAVY=0
SKIP_LOCAL=0
INSTALL_ISAACSIM_PIP=0
BLENDER_VERSION="${BLENDER_VERSION:-3.6.18}"
BLENDER_INSTALL_DIR="${BLENDER_INSTALL_DIR:-${HOME}/.local/share/blender}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Create a fresh SAGE conda environment and install dependencies from scratch.

Options:
  --target-env NAME   Target conda env to create. Default: sage
  --python VERSION    Python version. Default: 3.10.12
  --isaacsim-path DIR Local Isaac Sim directory. Default: /home/gaok/coding/isaacsim/_build/linux-x86_64/release
  --recreate          Remove target env first if it already exists.
  --skip-heavy        Skip packages that may compile or need git/CUDA toolchains.
  --skip-local        Skip editable installs for local IsaacLab/M2T2/robomimic.
  --install-isaacsim-pip
                      Install Isaac Sim pip packages instead of only linking local Isaac Sim.
  -h, --help          Show this help.

Examples:
  ./scripts/setup_sage_fresh_conda.sh
  ./scripts/setup_sage_fresh_conda.sh --skip-heavy
  ./scripts/setup_sage_fresh_conda.sh --recreate --target-env sage
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-env)
            TARGET_ENV="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --isaacsim-path)
            ISAACSIM_PATH="$2"
            shift 2
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
        --install-isaacsim-pip)
            INSTALL_ISAACSIM_PIP=1
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

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [setup-sage] $*" | tee -a "${LOG_FILE}"
}

have_env() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

run_in_env() {
    conda run --no-capture-output -n "${TARGET_ENV}" "$@"
}

run_pip() {
    run_in_env python -m pip "$@"
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

setup_blender_bpy() {
    local blender_short
    blender_short="$(echo "${BLENDER_VERSION}" | cut -d. -f1-2)"
    local blender_python_dir="${BLENDER_INSTALL_DIR}/blender-${BLENDER_VERSION}-linux-x64/${blender_short}/python/lib/python3.10/site-packages"
    local site_packages
    site_packages="$(run_in_env python -c 'import site; print(site.getsitepackages()[0])')"

    if [[ -d "${blender_python_dir}" ]]; then
        log "Blender ${BLENDER_VERSION} already extracted to ${BLENDER_INSTALL_DIR}"
    else
        local url="https://download.blender.org/release/Blender$(echo ${BLENDER_VERSION} | cut -d. -f1-2)/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
        local tarball="/tmp/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
        log "Downloading Blender ${BLENDER_VERSION} from ${url} ..."
        mkdir -p "${BLENDER_INSTALL_DIR}"
        wget -q --show-progress -O "${tarball}" "${url}" || {
            log "WARN: Failed to download Blender. Install it manually and set BLENDER_INSTALL_DIR."
            return 1
        }
        log "Extracting Blender to ${BLENDER_INSTALL_DIR} ..."
        tar -xf "${tarball}" -C "${BLENDER_INSTALL_DIR}"
        rm -f "${tarball}"
    fi

    log "Linking Blender bpy into conda env site-packages..."
    echo "${blender_python_dir}" > "${site_packages}/blender.pth"
    log "Blender bpy linked successfully."
}

link_local_isaacsim() {
    if [[ ! -f "${ISAACSIM_PATH}/isaac-sim.sh" ]]; then
        echo "[ERROR] Isaac Sim was not found at '${ISAACSIM_PATH}'." >&2
        echo "        Pass --isaacsim-path /path/to/isaacsim/release, or use --install-isaacsim-pip." >&2
        exit 1
    fi

    log "Using local Isaac Sim: ${ISAACSIM_PATH}"
    run_step "Link local Isaac Sim into bundled IsaacLab" \
        ln -sfn "${ISAACSIM_PATH}" "${REPO_ROOT}/IsaacLab/_isaac_sim"
}

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not available in PATH." >&2
    exit 1
fi

log "Writing detailed install log to ${LOG_FILE}"

if have_env "${TARGET_ENV}"; then
    if [[ "${RECREATE}" -eq 1 ]]; then
        log "Removing existing env '${TARGET_ENV}'..."
        run_step "Remove existing env ${TARGET_ENV}" conda env remove -y -n "${TARGET_ENV}"
    else
        echo "[ERROR] Conda env '${TARGET_ENV}' already exists." >&2
        echo "        Use --recreate to remove and rebuild it." >&2
        exit 1
    fi
fi

log "Creating fresh conda env '${TARGET_ENV}' with Python ${PYTHON_VERSION}..."
run_step "Create conda env ${TARGET_ENV}" \
    conda create -y -n "${TARGET_ENV}" \
    -c nvidia -c pytorch -c conda-forge -c defaults \
    "python=${PYTHON_VERSION}" pip setuptools wheel \
    importlib_metadata \
    libgl libegl libglvnd libglu libxcb \
    sysroot_linux-64

run_pip_install_batch "Upgrade pip tooling" -U pip setuptools wheel packaging ninja

run_pip_install_batch "PyTorch CUDA 12.4 stack" \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

if [[ "${INSTALL_ISAACSIM_PIP}" -eq 1 ]]; then
    run_pip_install_batch "Isaac Sim 4.2 pip packages" \
        --extra-index-url https://pypi.nvidia.com \
        isaacsim==4.2.0.2 \
        isaacsim-extscache-physics==4.2.0.2 \
        isaacsim-extscache-kit==4.2.0.2 \
        isaacsim-extscache-kit-sdk==4.2.0.2
else
    link_local_isaacsim
fi

log "Installing SAGE Python dependencies one package at a time..."
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
    extcolors Pylette kornia==0.7.2 meshcat

run_step "Install Blender for bpy module" setup_blender_bpy

run_pip_install_each "Robot and simulation Python packages" \
    gymnasium mujoco==3.3.4 robosuite sapien

run_pip_install_each "Model clients and foundation-model helpers" \
    open-clip-torch==2.32.0 sentence-transformers==4.1.0 \
    transformers==4.41.2 diffusers==0.11.1 accelerate==0.23.0 \
    "qwen-agent[gui,rag,code_interpreter,mcp]==0.0.26" \
    wandb

if [[ "${SKIP_HEAVY}" -eq 0 ]]; then
    log "Installing heavy/compiled dependencies..."
    run_optional_pip_install_batch "Heavy dependency: nvidia-curobo" \
        --extra-index-url https://pypi.nvidia.com nvidia-curobo || true
    run_optional_pip_install_batch "Heavy dependency: nvdiffrast" \
        "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git" || true
    run_optional_pip_install_batch "Heavy dependency: pytorch3d" \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable" || true
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
        run_optional_pip_install_batch "Editable install: M2T2 pointnet2_ops" \
            -e "${REPO_ROOT}/M2T2/pointnet2_ops" || true
    else
        log "Skipping pointnet2_ops because --skip-heavy was set."
    fi
    run_optional_pip_install_batch "Editable install: M2T2" -e "${REPO_ROOT}/M2T2" || true
    run_optional_pip_install_batch "Editable install: robomimic" -e "${REPO_ROOT}/robomimic" || true
else
    log "Skipping local editable packages."
fi

run_step "Lightweight import check" conda run --no-capture-output -n "${TARGET_ENV}" python - <<'PY'
import importlib

mods = [
    "torch",
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
