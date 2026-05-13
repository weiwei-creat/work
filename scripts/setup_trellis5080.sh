#!/usr/bin/env bash
set -euo pipefail

# ── Config ──────────────────────────────────────────
ENV_NAME="${1:-trellis5080}"
TRELLIS_DIR="${2:-/home/gaok/coding/TRELLIS}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
KAOLIN_INDEX="${KAOLIN_INDEX:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html}"

echo "=== Create env: ${ENV_NAME} ==="
conda create -y -n "${ENV_NAME}" python=3.10

echo "=== Install PyTorch 2.6.0 + CUDA 12.6 ==="
conda run -n "${ENV_NAME}" pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url "${PYTORCH_INDEX}"

echo "=== Verify PyTorch ==="
conda run -n "${ENV_NAME}" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Archs: {torch.cuda.get_arch_list()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Blackwell OK: {\"sm_120\" in str(torch.cuda.get_arch_list()) or \"sm_120a\" in str(torch.cuda.get_arch_list())}')
"

echo "=== Install kaolin ==="
conda run -n "${ENV_NAME}" pip install kaolin==0.18.0 -f "${KAOLIN_INDEX}"

echo "=== Install TRELLIS basic deps ==="
conda run -n "${ENV_NAME}" pip install \
    pillow imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh open3d xatlas pyvista pymeshfix igraph transformers \
    flask flask-cors werkzeug psutil

conda run -n "${ENV_NAME}" pip install \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

echo "=== Install CUDA extensions (from source) ==="
export CUDA_HOME
CUDA_HOME="$(conda run -n "${ENV_NAME}" python3 -c 'import sysconfig; print(sysconfig.get_config_var("CONDA_PREFIX"))')"
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="12.0"

# Reinstall nvidia-cuda-runtime to match
conda run -n "${ENV_NAME}" pip install nvidia-cuda-runtime-cu12 --force-reinstall 2>/dev/null || true

# spconv
echo "--- spconv ---"
conda run -n "${ENV_NAME}" pip uninstall -y spconv cumm spconv-cu120 cumm-cu120 2>/dev/null || true
conda run -n "${ENV_NAME}" pip install spconv-cu120 2>&1 | tail -3

# nvdiffrast
echo "--- nvdiffrast ---"
conda run -n "${ENV_NAME}" pip install git+https://github.com/NVlabs/nvdiffrast.git 2>&1 | tail -3 || \
    echo "[WARN] nvdiffrast failed; will skip mesh rendering"

# flash-attn
echo "--- flash-attn ---"
conda run -n "${ENV_NAME}" pip install flash-attn --no-build-isolation 2>&1 | tail -3 || \
    echo "[WARN] flash-attn failed; set ATTN_BACKEND=sdpa"

# diff-gaussian-rasterization (mip-splatting)
echo "--- diff-gaussian-rasterization ---"
conda run -n "${ENV_NAME}" pip install \
    "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization/" \
    --no-build-isolation 2>&1 | tail -3 || \
    echo "[WARN] diff-gaussian-rasterization failed"

# diffoctreerast
echo "--- diffoctreerast ---"
conda run -n "${ENV_NAME}" pip install \
    git+https://github.com/JeffreyXiang/diffoctreerast.git 2>&1 | tail -3 || \
    echo "[WARN] diffoctreerast failed"

echo ""
echo "=== Test TRELLIS import ==="
conda run -n "${ENV_NAME}" bash -c "
export PYTHONPATH=${TRELLIS_DIR}
export ATTN_BACKEND=sdpa
python3 -c '
from trellis.pipelines import TrellisTextTo3DPipeline
print(\"TRELLIS import: OK\")
pipeline = TrellisTextTo3DPipeline.from_pretrained(\"${TRELLIS_DIR}\")
print(\"Pipeline loaded: OK\")
print(\"Device:\", pipeline.device)
' 2>&1 | grep -E 'OK|Error|device'
" || echo "[WARN] Pipeline test failed"

echo ""
echo "=== Done ==="
echo "Activate: conda activate ${ENV_NAME}"
echo "Start server: ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa SPCONV_ALGO=native bash scripts/start_trellis_server.sh 8080 ${TRELLIS_DIR} ${ENV_NAME}"
