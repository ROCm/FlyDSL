#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ASM_SRC="${SCRIPT_DIR}/flash_attn_opus.v1.s"
CO_OUT="${SCRIPT_DIR}/flash_attn_opus.v1.co"

echo "=== Compile flash_attn_opus.v1.s ==="
/opt/rocm/llvm/bin/clang++ \
  -x assembler \
  -target amdgcn-amd-amdhsa \
  --offload-arch=gfx950 \
  "${ASM_SRC}" \
  -o "${CO_OUT}"

echo "=== Build Python extension opus_asm_ext ==="
rm -rf build
python3 setup.py build_ext --inplace

echo "=== Build complete ==="
ls -lah "${CO_OUT}" opus_asm_ext*.so
