#!/usr/bin/env bash
set -euo pipefail

TOP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${TOP}"

if [[ -z "${OPUS_INCLUDE_DIR:-}" ]]; then
  if [[ -d "${TOP}/../../aiter/csrc/include" ]]; then
    export OPUS_INCLUDE_DIR="${TOP}/../../aiter/csrc/include"
  else
    export OPUS_INCLUDE_DIR="/home/carhuang/repo/aiter/csrc/include"
  fi
fi

if [[ -z "${ARCH:-}" ]]; then
  ARCH="$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx/ {print $2; exit}')"
  export ARCH="${ARCH:-gfx950}"
fi

JOBS="${JOBS:-$(nproc)}"

echo "Building OPUS GQA Python shared library"
echo "  OPUS_INCLUDE_DIR=${OPUS_INCLUDE_DIR}"
echo "  ARCH=${ARCH}"
echo "  JOBS=${JOBS}"

make clean
make -j"${JOBS}" all lib
python3 -m pip install -e .

python3 - <<'PY'
import opus_attn

print(f"Installed opus_attn from {opus_attn.__file__}")
PY

rm -rf opus_attn_gqa.egg-info __pycache__
