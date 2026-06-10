#!/usr/bin/env bash
# A8W4 GEMM precision check on real gfx1250 hardware (current pipeline).
# Same env workarounds as run_gemm_a8w4_realhw.sh, then runs the precision driver.
#
# Usage:
#   bash scripts/run_check_a8w4_precision.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRITON_PY="/root/upstream_triton/python"
if ! PYTHONPATH="$TRITON_PY" python3 -c "import triton" 2>/dev/null; then
    echo "ERR: triton at $TRITON_PY does not import. Edit TRITON_PY in this script." >&2
    exit 1
fi

export PYTHONPATH="$TRITON_PY:${REPO_ROOT}/build-fly/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"
export FLYDSL_ROOT="$REPO_ROOT"

# SDMA on this box faults on H2D; force blit-engine copies.
export HSA_ENABLE_SDMA=0
# lld resolves its toolchain relative to ROCM_PATH; a pre-set _rocm_sdk_devel
# path breaks the ld.lld lookup, so force /opt/rocm where the symlink lives.
export ROCM_PATH=/opt/rocm
export FLYDSL_RUNTIME_ENABLE_CACHE=1
unset FLYDSL_DUMP_IR FLYDSL_DUMP_DIR

echo "Using triton: $TRITON_PY"
exec python3 -u "$REPO_ROOT/scripts/check_a8w4_precision.py" "$@"
