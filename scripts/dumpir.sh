#!/bin/bash
# dumpir.sh <script> <arguments>
# Example: bash scripts/dumpir.sh python tests/kernels/benchmark/matrixTranspose.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Locate the build directory (default: .flir/build; fallback: build/).
BUILD_DIR="${FLIR_BUILD_DIR:-${FLIR_BUILD_DIR:-${REPO_ROOT}/.flir/build}}"
if [ ! -d "${BUILD_DIR}" ] && [ -d "${REPO_ROOT}/build" ]; then
  BUILD_DIR="${REPO_ROOT}/build"
fi
PYTHON_PACKAGE_ROOT="${BUILD_DIR}/python_packages/flydsl"

export FLIR_DUMP_IR=1
#export FLIR_DUMP_DIR=.kernel_dumps

export PYTHONPATH="${REPO_ROOT}/flydsl/src:${PYTHON_PACKAGE_ROOT}:${REPO_ROOT}:${PYTHONPATH}"

exec "$@"
