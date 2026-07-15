#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SAGE_CONDA_ENV:-sage}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/client/environment.yml"
PYPI_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
if [[ -z "${CONDA_BIN}" && -x "/home/ubuntu/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/home/ubuntu/miniconda3/bin/conda"
fi
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# conda env create forwards the entire pip section to one pip invocation. The
# exported environment mixes ordinary PyPI packages with +cu124 PyTorch wheels,
# which makes every package query both indexes and causes an all-or-nothing
# rollback on a transient PyTorch-index failure. Split it into deterministic
# transactions instead.
awk '/^  - pip:$/ {exit} {print}' "${ENV_FILE}" > "${TMP_DIR}/conda.yml"
awk '
    /^  - pip:$/ {pip_section=1; next}
    pip_section && /^      - / {
        sub(/^      - /, "")
        if ($0 !~ /^(torch|torchvision|torchaudio)==/ &&
            $0 !~ /^nvidia-/ &&
            $0 !~ /^omni-isaac-lab/ &&
            $0 !~ /^pytorch3d==/ &&
            $0 !~ /^nvdiffrast==/ &&
            $0 !~ /^nvidia-curobo==/) print
    }
' "${ENV_FILE}" > "${TMP_DIR}/requirements.txt"

if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
    echo "[sage-setup] conda is not available" >&2
    exit 1
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[sage-setup] updating existing environment '${ENV_NAME}'"
    "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${TMP_DIR}/conda.yml"
else
    echo "[sage-setup] creating environment '${ENV_NAME}'"
    "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${TMP_DIR}/conda.yml"
fi

echo "[sage-setup] installing the PyTorch CUDA 12.4 stack"
PIP_INDEX_URL="${PYTORCH_INDEX}" PIP_EXTRA_INDEX_URL="${PYPI_INDEX}" \
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    --retries 20 --timeout 60 \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

echo "[sage-setup] installing ordinary Python dependencies"
PIP_INDEX_URL="${PYPI_INDEX}" "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    --retries 20 --timeout 60 -r "${TMP_DIR}/requirements.txt"

# nvdiffrast is only used by optional local 3D-generation paths. It is not
# published on PyPI and compiling it requires a full CUDA Toolkit (nvcc), not
# merely the NVIDIA driver/runtime. Objathor-only reproduction does not import
# it, so keep the base environment valid on machines without a toolkit.
if command -v nvcc >/dev/null 2>&1; then
    PIP_INDEX_URL="${PYPI_INDEX}" "${CONDA_BIN}" run -n "${ENV_NAME}" \
        python -m pip install setuptools wheel ninja
    CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}" \
        "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
        --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
else
    echo "[sage-setup] nvcc not found; skipping optional nvdiffrast (Objathor mode is unaffected)"
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip check
"${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
    'import torch, openai, mcp, matplotlib, trimesh, open3d, cv2; print("[sage-setup] imports OK"); print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

echo "[sage-setup] environment '${ENV_NAME}' is ready"
echo "[sage-setup] cuRobo is intentionally optional and should be installed only for robot planning workflows"
