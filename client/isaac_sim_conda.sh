#!/usr/bin/env bash

# Isaac Sim with Conda Environment Wrapper
# This script runs Isaac Sim while using the conda environment for Python

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Explicit launch-time values must win over machine-specific entries copied in
# from an older .env file.
_EXPLICIT_ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-}"
_EXPLICIT_CONDA_PYTHON="${CONDA_PYTHON:-}"
_EXPLICIT_CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

if [ -f "${SAGE_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${SAGE_ROOT}/.env"
    set +a
fi

[ -n "${_EXPLICIT_ISAAC_SIM_PATH}" ] && ISAAC_SIM_PATH="${_EXPLICIT_ISAAC_SIM_PATH}"
[ -n "${_EXPLICIT_CONDA_PYTHON}" ] && CONDA_PYTHON="${_EXPLICIT_CONDA_PYTHON}"
[ -n "${_EXPLICIT_CONDA_ENV_NAME}" ] && CONDA_ENV_NAME="${_EXPLICIT_CONDA_ENV_NAME}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-sage}"
ISAACSIM_PATH="${ISAAC_SIM_PATH:-/home/ubuntu/isaac}"
ISAACLAB_PATH="${ISAACLAB_PATH:-${SAGE_ROOT}/IsaacLab}"
RELEASE_ROOT="${ISAACSIM_PATH}/_build/linux-x86_64/release"
CONDA_ENV_PYTHON="${CONDA_PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"

echo "[INFO] Starting Isaac Sim with conda environment '${CONDA_ENV_NAME}'..."
if [ -x "${CONDA_ENV_PYTHON}" ]; then
    export PATH="$(dirname "${CONDA_ENV_PYTHON}"):${PATH:-}"
    PYTHON_EXE="${CONDA_ENV_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
else
    echo "[ERROR] Could not find a Python executable for Isaac Sim" >&2
    exit 1
fi
echo "${PYTHON_EXE}"

# Set up Isaac Sim environment variables
export ISAACSIM_PATH="${ISAACSIM_PATH}"
export ISAACLAB_PATH="${ISAACLAB_PATH}"
export RESOURCE_NAME="IsaacSim"
export OLD_PYTHONPATH="${PYTHONPATH:-}"

EXPERIENCE_KIT="${SAGE_ISAAC_EXPERIENCE:-}"
ISAAC_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experience)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --experience requires a kit filename or absolute path" >&2
                exit 1
            fi
            EXPERIENCE_KIT="$2"
            shift 2
            ;;
        --experience=*)
            EXPERIENCE_KIT="${1#--experience=}"
            shift
            ;;
        *)
            ISAAC_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ISAAC_ARGS[@]}"

resolve_experience_path() {
    local install_root="$1"
    local default_kit="$2"
    local selected_kit="${EXPERIENCE_KIT:-${default_kit}}"

    if [[ "${selected_kit}" = /* ]]; then
        printf '%s\n' "${selected_kit}"
    else
        printf '%s\n' "${install_root}/apps/${selected_kit}"
    fi
}

if [ -x "${ISAACSIM_PATH}/kit/kit" ]; then
    # Isaac Sim Kit environment variables for extracted binary installs.
    export CARB_APP_PATH="${ISAACSIM_PATH}/kit"
    export EXP_PATH="${ISAACSIM_PATH}/apps"
    export ISAAC_PATH="${ISAACSIM_PATH}"

    ISAAC_PYTHON_PATHS="${ISAACSIM_PATH}/kit/python/lib/python3.10/site-packages"
    ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/python_packages"
    ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/exts/omni.isaac.kit"
    ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/kit/kernel/py"
    ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/kit/plugins/bindings-python"
    ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/exts/omni.isaac.lula/pip_prebundle"
    ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/exts/omni.exporter.urdf/pip_prebundle"

    for pip_prebundle_dir in "${ISAACSIM_PATH}"/extscache/*/pip_prebundle "${ISAACSIM_PATH}"/exts/*/pip_prebundle; do
        if [ -d "${pip_prebundle_dir}" ]; then
            ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${pip_prebundle_dir}"
        fi
    done
    # Binary Isaac Sim 4.5 and the sage environment both use Python 3.10.
    # Keep Kit's bundled modules first, then expose portable dependencies used
    # by the SAGE extension (xatlas, trimesh, scipy, imageio, ...).
    CONDA_SITE_PACKAGES="$("${PYTHON_EXE}" -c 'import site; print(site.getsitepackages()[0])')"
    export SAGE_CONDA_SITE_PACKAGES="${CONDA_SITE_PACKAGES}"
    export PYTHONPATH="${ISAAC_PYTHON_PATHS}:${PYTHONPATH:-}:${CONDA_SITE_PACKAGES}"

    ISAAC_LIB_PATHS="${ISAACSIM_PATH}:${ISAACSIM_PATH}/kit:${ISAACSIM_PATH}/kit/kernel/plugins"
    ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/libs/iray:${ISAACSIM_PATH}/kit/plugins"
    ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/bindings-python"
    ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/carb_gfx"
    ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/rtx"
    ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/gpu.foundation"

    for schema_lib in "${ISAACSIM_PATH}"/exts/omni.usd.schema.isaac/plugins/*/lib; do
        if [ -d "${schema_lib}" ]; then
            ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${schema_lib}"
        fi
    done
    export LD_LIBRARY_PATH="${ISAAC_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
elif [ -x "${RELEASE_ROOT}/isaac-sim.sh" ]; then
    # Source-build Isaac Sim 5.x owns its Python 3.11 paths; avoid injecting
    # stale Python 3.10 paths from older binary-install launchers.
    export CARB_APP_PATH="${RELEASE_ROOT}/kit"
    export EXP_PATH="${RELEASE_ROOT}/apps"
    export ISAAC_PATH="${RELEASE_ROOT}"
fi

# Override Python executable to use conda python
export PYTHONEXE="${PYTHON_EXE}"

echo "[INFO] Using Python from conda environment: ${PYTHONEXE}"
echo "[INFO] Isaac Sim path: ${ISAACSIM_PATH}"
echo "[INFO] Starting Isaac Sim..."

if [ -x "${ISAACSIM_PATH}/kit/kit" ]; then
    EXPERIENCE_PATH="$(resolve_experience_path "${ISAACSIM_PATH}" "omni.isaac.sim.kit")"
    if [ ! -f "${EXPERIENCE_PATH}" ]; then
        echo "[ERROR] Isaac experience not found: ${EXPERIENCE_PATH}" >&2
        exit 1
    fi

    cd "${ISAACSIM_PATH}"
    echo "[INFO] Using Isaac experience: ${EXPERIENCE_PATH}"
    if [ -d "${ISAACLAB_PATH}/source/extensions" ]; then
        echo "[INFO] Including Isaac Lab extensions from: ${ISAACLAB_PATH}/source/extensions"
        exec "${ISAACSIM_PATH}/kit/kit" "${EXPERIENCE_PATH}" \
            --ext-folder "${ISAACSIM_PATH}/apps" \
            --ext-folder "${ISAACLAB_PATH}/source/extensions" \
            "$@"
    else
        echo "[INFO] Running Isaac Sim without Isaac Lab extensions"
        exec "${ISAACSIM_PATH}/kit/kit" "${EXPERIENCE_PATH}" \
            --ext-folder "${ISAACSIM_PATH}/apps" \
            "$@"
    fi
elif [ -x "${RELEASE_ROOT}/isaac-sim.sh" ]; then
    EXPERIENCE_PATH="$(resolve_experience_path "${RELEASE_ROOT}" "isaacsim.exp.full.kit")"
    if [ ! -f "${EXPERIENCE_PATH}" ]; then
        echo "[ERROR] Isaac experience not found: ${EXPERIENCE_PATH}" >&2
        exit 1
    fi

    cd "${RELEASE_ROOT}"
    NO_ROS_ENV=false
    for arg in "$@"; do
        if [ "${arg}" = "--no-ros-env" ]; then
            NO_ROS_ENV=true
            echo "[INFO] Skipping automatic ROS environment setup"
            break
        fi
    done
    if [ "${NO_ROS_ENV}" = "false" ] && [ -f "${RELEASE_ROOT}/setup_ros_env.sh" ]; then
        # Match isaac-sim.sh behavior while still honoring the selected experience.
        # shellcheck disable=SC1091
        source "${RELEASE_ROOT}/setup_ros_env.sh"
    fi

    echo "[INFO] Using Isaac experience: ${EXPERIENCE_PATH}"
    if [ -d "${ISAACLAB_PATH}/source/extensions" ]; then
        echo "[INFO] Including Isaac Lab extensions from: ${ISAACLAB_PATH}/source/extensions"
        exec "${RELEASE_ROOT}/kit/kit" "${EXPERIENCE_PATH}" \
            --ext-folder "${ISAACLAB_PATH}/source/extensions" \
            "$@"
    else
        echo "[INFO] Running Isaac Sim without Isaac Lab extensions"
        exec "${RELEASE_ROOT}/kit/kit" "${EXPERIENCE_PATH}" "$@"
    fi
else
    echo "[ERROR] Could not find an Isaac Sim launcher under ${ISAACSIM_PATH}" >&2
    exit 1
fi
