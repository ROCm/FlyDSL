#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# FlyDSL Test Suite
# Fail-fast: exits immediately on first test failure.
#
# Local (default): skips large_shape tests for fast iteration.
# CI/full:         RUN_TESTS_FULL=1 bash scripts/run_tests.sh

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

pytest_markers="not multi_gpu and not benchmark"
if [ "${RUN_TESTS_FULL:-0}" != "1" ]; then
    pytest_markers+=" and not large_shape"
fi
# pytest.ini already enables verbose mode; -q brings the broad suite back to
# compact per-file progress instead of emitting thousands of node IDs to CI.
pytest_args=(-q --no-header --tb=short -m "${pytest_markers}")

# ---------------------------------------------------------------------------
# 1. Single-GPU correctness tests (kernels + language + unit + system + extension + examples)
# Multi-GPU and benchmark tests have dedicated CI jobs/steps and must not be
# repeated here, especially on single-GPU jobs backed by an 8-GPU host.
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "Pytest: kernels + language + unit + system + extension + examples"
echo "========================================================================"

python3 -m pytest \
    tests/kernels/ \
    tests/language/ \
    tests/unit/ \
    tests/system/ \
    tests/extension/ \
    tests/python/examples/ \
    "${pytest_args[@]}"

# ---------------------------------------------------------------------------
# 2. Standalone example scripts (not pytest)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Examples (examples/)"
echo "========================================================================"

# Architecture assignments from tests/arch_compat.py, checked before execution.
_gpu_arch=$(python3 -c "from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())" 2>/dev/null || echo "unknown")
for example in "${REPO_ROOT}"/examples/*.py "${REPO_ROOT}"/examples/extension/*/*.py; do
    [ -f "${example}" ] || continue
    # Named by their path under examples/, so nested ones stay unambiguous.
    name="${example#"${REPO_ROOT}"/examples/}"
    allowed_arches=$(python3 -c 'import sys; from tests.arch_compat import EXAMPLE_ARCHITECTURES; print(" ".join(EXAMPLE_ARCHITECTURES[sys.argv[1]]))' "${name}")
    read -r -a arch_patterns <<< "${allowed_arches}"
    supported=false
    for pattern in "${arch_patterns[@]}"; do
        if [[ "${_gpu_arch}" == ${pattern} ]]; then
            supported=true
            break
        fi
    done
    if [[ "${supported}" != true ]]; then
        echo "  SKIP  ${name}  (requires: ${allowed_arches}; arch: ${_gpu_arch})"
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

# ---------------------------------------------------------------------------
# 3. MLIR FileCheck tests
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "MLIR FileCheck Tests"
echo "========================================================================"

FLY_OPT="${BUILD_DIR}/bin/fly-opt"
FILECHECK=""
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    _mlir_dir=$(grep '^MLIR_DIR:' "${BUILD_DIR}/CMakeCache.txt" | sed 's|^MLIR_DIR:[A-Z]*=||')
    [ -n "${_mlir_dir}" ] && FILECHECK="${_mlir_dir}/../../../bin/FileCheck"
fi
[ -z "${FILECHECK}" ] || [ ! -x "${FILECHECK}" ] && FILECHECK="$(which FileCheck 2>/dev/null || true)"

if [ -z "${FILECHECK}" ] || [ ! -x "${FILECHECK}" ]; then
    # Fail-open by default for local runs without a built FileCheck, but say so
    # loudly: a green run_tests.sh does NOT mean the MLIR tests passed when this
    # fires. Set FLYDSL_REQUIRE_FILECHECK=1 (CI should) to make it an error.
    case "${FLYDSL_REQUIRE_FILECHECK:-0}" in
        1 | [Tt]rue | [Yy]es | [Oo]n)
            echo "  FAIL  FileCheck not found and FLYDSL_REQUIRE_FILECHECK is set; MLIR lit tests cannot run."
            exit 1
            ;;
    esac
    echo "  SKIP  FileCheck not found; MLIR lit tests DID NOT RUN (not a pass)."
    MLIR_TESTS_SKIPPED=1
else

for f in $(find "${REPO_ROOT}/tests/mlir" -name "*.mlir" -type f 2>/dev/null | sort); do
    # A file may carry several RUN lines, typically one per --check-prefix. Run
    # every one of them: only checking the first silently skips the rest.
    mapfile -t run_lines < <(grep '^// RUN:' "$f" | sed 's|^// RUN: *||')
    [ ${#run_lines[@]} -eq 0 ] && continue
    for run_line in "${run_lines[@]}"; do
        # Map every substitution to a placeholder first, then expand. Expanding
        # in place would let a later rule rewrite text an earlier one inserted:
        # with `%FileCheck`, the bare-FileCheck rule would match inside the path
        # just substituted and yield `/usr/bin//usr/bin/FileCheck`. `%FileCheck`
        # must also be consumed before the bare form, or it degrades to `%<path>`.
        cmd=$(echo "$run_line" | sed \
            -e "s|%fly-opt|@FLY_OPT@|g" \
            -e "s|%FileCheck|@FILECHECK@|g" \
            -e "s|FileCheck|@FILECHECK@|g" \
            -e "s|%s|@SRC@|g" \
            -e "s|@FLY_OPT@|${FLY_OPT}|g" \
            -e "s|@FILECHECK@|${FILECHECK}|g" \
            -e "s|@SRC@|${f}|g")
        if ! eval "$cmd" > /tmp/filecheck_out.log 2>&1; then
            echo "  FAIL  ${f#${REPO_ROOT}/tests/mlir/}"
            echo "        RUN: ${run_line}"
            tail -5 /tmp/filecheck_out.log | sed 's/^/        /'
            exit 1
        fi
    done
    echo "  PASS  ${f#${REPO_ROOT}/tests/mlir/}"
done

fi

echo ""
echo "========================================================================"
if [ "${MLIR_TESTS_SKIPPED:-0}" = "1" ]; then
    # Never claim a clean run when a whole stage was skipped -- that is the
    # exact confusion this guard exists to prevent.
    echo "All tests passed EXCEPT the MLIR lit stage, which DID NOT RUN."
else
    echo "All tests passed."
fi
echo "========================================================================"
