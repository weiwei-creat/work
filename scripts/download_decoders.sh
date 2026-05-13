#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="/home/gaok/coding/TRELLIS/JeffreyXiang/TRELLIS-image-large"
REPO="JeffreyXiang/TRELLIS-image-large"

mkdir -p "${LOCAL_DIR}"

FILES=(
  ckpts/ss_dec_conv3d_16l8_fp16.json
  ckpts/ss_dec_conv3d_16l8_fp16.safetensors
  ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json
  ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors
  ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json
  ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors
  ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json
  ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors
)

for f in "${FILES[@]}"; do
  echo "Downloading ${f}..."
  hf download "${REPO}" "${f}" --local-dir "${LOCAL_DIR}"
done

echo "Done. Files:"
find "${LOCAL_DIR}" -type f | sort
