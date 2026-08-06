#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# FlyDSL Test Suite
# Fail-fast: exits immediately on first test failure.
#
# Local (default): skips large_shape tests for fast iteration.
# CI:              RUN_TESTS_FULL=1 bash scripts/run_tests.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Broad test runs must be deterministic; dedicated autotune tests opt in per test.
export FLYDSL_AUTOTUNE=0

# Auto-select GPU with the most free VRAM (skip if HIP_VISIBLE_DEVICES is already set).
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]] && command -v python3 &>/dev/null; then
    _best_gpu=$(python3 -c "
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    print(best)
" 2>/dev/null || true)
    if [[ -n "${_best_gpu}" ]]; then
        export HIP_VISIBLE_DEVICES="${_best_gpu}"
        echo "[run_tests] Auto-selected GPU ${_best_gpu} (most free VRAM)"
    fi
fi

BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
MLIR_LIBS_DIR="${BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs"

export PYTHONPATH="${BUILD_DIR}/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"
export FLYDSL_RUN_QUANT=1
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MLIR_LIBS_DIR}:"* ]]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Compile backend selects which example directory and pytest suites apply.
_compile_backend="${FLYDSL_COMPILE_BACKEND:-rocm}"
_compile_backend="${_compile_backend,,}"

pytest_args=(-v --no-header --tb=short)
_marker_expr=""
if [ "${RUN_TESTS_FULL:-0}" != "1" ]; then
    _marker_expr="not large_shape"
fi
if [ "${_compile_backend}" == "cuda" ]; then
    # tests/kernels, tests/system and the AOT suite are ROCm kernels and
    # rocm_lower tests; only the backend-agnostic tier is meaningful here.
    pytest_paths=(tests/language/ tests/unit/)
    _marker_expr="l0_backend_agnostic${_marker_expr:+ and ${_marker_expr}}"
else
    pytest_paths=(tests/kernels/ tests/language/ tests/unit/ tests/system/ tests/python/examples/)
fi
if [ -n "${_marker_expr}" ]; then
    pytest_args+=(-m "${_marker_expr}")
fi

# ---------------------------------------------------------------------------
# 1. All pytest-based tests (kernels + language + unit + system + examples)
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "Pytest: ${pytest_paths[*]}"
echo "========================================================================"

python3 -m pytest "${pytest_paths[@]}" "${pytest_args[@]}"

# ---------------------------------------------------------------------------
# 2. Standalone example scripts (not pytest)
#
# examples/*.py       -> target-neutral examples, run on every compile backend
# examples/rocm/*.py  -> ROCm/HIP examples, run on the rocm compile backend
# examples/cuda/*.py  -> NVVM/CUDA examples, run on the cuda compile backend
# examples/cuda/bench -> developer benchmark harnesses, never run here (they
#                        need an external CUTLASS checkout and take minutes)
# ---------------------------------------------------------------------------
if [ "${_compile_backend}" == "cuda" ]; then
    _example_subdir="cuda"
else
    _example_subdir="rocm"
fi
_example_dirs=("${REPO_ROOT}/examples" "${REPO_ROOT}/examples/${_example_subdir}")

echo ""
echo "========================================================================"
echo "Examples (examples/ + examples/${_example_subdir}/)"
echo "========================================================================"

# Whitelist from tests/arch_compat.py (single source of truth for arch compat).
_RDNA_EXAMPLE_WHITELIST=$(python3 -c "from tests.arch_compat import RDNA_COMPATIBLE_EXAMPLES; print(' '.join(RDNA_COMPATIBLE_EXAMPLES))" 2>/dev/null || echo "")
_gpu_arch=$(python3 -c "from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())" 2>/dev/null || echo "unknown")

for _dir in "${_example_dirs[@]}"; do
for example in "${_dir}"/*.py; do
    [ -f "${example}" ] || continue
    name="${example#${REPO_ROOT}/examples/}"
    if [[ "${_compile_backend}" != "cuda" && "${_gpu_arch}" != gfx9* ]] && ! echo "${_RDNA_EXAMPLE_WHITELIST}" | grep -qw "$(basename "${example}")"; then
        echo "  SKIP  ${name}  (not in RDNA whitelist, arch: ${_gpu_arch})"
        continue
    fi
    output=$(python3 "${example}" 2>&1) || {
        echo "  FAIL  ${name}"; echo "$output" | tail -10 | sed 's/^/        /'; exit 1
    }
    if echo "$output" | grep -qE "Result correct: False|All passed: False"; then
        echo "  FAIL  ${name}"; echo "$output" | tail -10 | sed 's/^/        /'; exit 1
    fi
    echo "  PASS  ${name}"
done
done

# ---------------------------------------------------------------------------
# 3. MLIR FileCheck tests
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "MLIR FileCheck Tests"
echo "========================================================================"

FLY_OPT="${BUILD_DIR}/bin/fly-opt"
FILECHECK=""
_enabled_backends="${FLYDSL_BACKENDS:-rocdl}"
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    _mlir_dir=$(grep '^MLIR_DIR:' "${BUILD_DIR}/CMakeCache.txt" | sed 's|^MLIR_DIR:[A-Z]*=||')
    [ -n "${_mlir_dir}" ] && FILECHECK="${_mlir_dir}/../../../bin/FileCheck"
    _cache_backends=$(grep '^FLYDSL_BACKENDS:' "${BUILD_DIR}/CMakeCache.txt" | sed 's|^FLYDSL_BACKENDS:[A-Z]*=||' || true)
    [ -n "${_cache_backends}" ] && _enabled_backends="${_cache_backends}"
fi
[ -z "${FILECHECK}" ] || [ ! -x "${FILECHECK}" ] && FILECHECK="$(which FileCheck 2>/dev/null || true)"

if [ -z "${FILECHECK}" ] || [ ! -x "${FILECHECK}" ]; then
    echo "  SKIP  FileCheck not found; skipping MLIR lit tests."
else

backend_enabled() {
    [[ ";${_enabled_backends};" == *";$1;"* ]]
}

# A test needs a backend if it runs that backend's conversion pass OR mentions
# its dialect anywhere: target atom types (!fly_rocdl.*, !fly_nvvm.*) fail to
# parse under target-neutral passes too when the dialect is not registered.
for f in $(find "${REPO_ROOT}/tests/mlir" -name "*.mlir" -type f 2>/dev/null | sort); do
    run_line=$(grep '^// RUN:' "$f" | head -1 | sed 's|^// RUN: *||')
    [ -z "$run_line" ] && continue
    if grep -q 'fly_nvvm\|convert-fly-to-nvvm' "$f" && ! backend_enabled nvvm; then
        echo "  SKIP  ${f#${REPO_ROOT}/tests/mlir/}  (nvvm backend not enabled)"
        continue
    fi
    if grep -q 'fly_rocdl\|convert-fly-to-rocdl' "$f" && ! backend_enabled rocdl; then
        echo "  SKIP  ${f#${REPO_ROOT}/tests/mlir/}  (rocdl backend not enabled)"
        continue
    fi
    cmd=$(echo "$run_line" | sed "s|%fly-opt|${FLY_OPT}|g; s|%FileCheck|${FILECHECK}|g; s|%s|${f}|g; s|FileCheck|${FILECHECK}|g")
    if eval "$cmd" > /tmp/filecheck_out.log 2>&1; then
        echo "  PASS  ${f#${REPO_ROOT}/tests/mlir/}"
    else
        echo "  FAIL  ${f#${REPO_ROOT}/tests/mlir/}"
        tail -5 /tmp/filecheck_out.log | sed 's/^/        /'
        exit 1
    fi
done

fi

echo ""
echo "========================================================================"
echo "All tests passed."
echo "========================================================================"
