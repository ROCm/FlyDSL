#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

function compile_asm() {
  local src="$1"
  local out="$2"
  echo "=== Compile $(basename "${src}") ==="
  /opt/rocm/llvm/bin/clang++ \
    -x assembler \
    -target amdgcn-amd-amdhsa \
    --offload-arch=gfx950 \
    "${src}" \
    -o "${out}"
}

compile_asm \
  "${SCRIPT_DIR}/flash_attn_opus.v1.s" \
  "${SCRIPT_DIR}/flash_attn_opus.v1.co"

compile_asm \
  "${SCRIPT_DIR}/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s" \
  "${SCRIPT_DIR}/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.co"

compile_asm \
  "${SCRIPT_DIR}/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk0_gm0.s" \
  "${SCRIPT_DIR}/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk0_gm0.co"

echo "=== Build Python extensions ==="
rm -rf build
python3 setup.py build_ext --inplace

echo "=== Build complete ==="
ls -lah ./*.co ./*_asm_ext*.so
