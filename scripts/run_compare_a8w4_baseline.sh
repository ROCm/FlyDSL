#!/usr/bin/env bash
# Compare A8W4 GEMM perf (current vs baseline kernel) on real gfx1250 hardware.
# Sets the same env workarounds as run_gemm_a8w4_realhw.sh, then runs the
# Python comparison driver.
#
# Usage:
#   bash scripts/run_compare_a8w4_baseline.sh
#   bash scripts/run_compare_a8w4_baseline.sh --warmup 8 --iters 30
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRITON_PY="/root/upstream_triton/python"
if ! PYTHONPATH="$TRITON_PY" python3 -c "import triton" 2>/dev/null; then
    echo "ERR: triton at $TRITON_PY does not import. Edit TRITON_PY in this script." >&2
    exit 1
fi

#export PYTHONPATH="$TRITON_PY:${REPO_ROOT}/build-fly/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"
#export LD_LIBRARY_PATH="${REPO_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"


export ROCM_PATH=/home/jli10004/workspace/rocm-toolkit-samebank

export PYTHONPATH=/home/jli10004/workspace/FlyDSL/build-fly-coexec-samebank-gemm-branch/python_packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/jli10004/workspace/FlyDSL/build-fly-coexec-samebank-gemm-branch/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}

export FLYDSL_ROOT="$REPO_ROOT"

# SDMA on this box faults on H2D; force blit-engine copies.
export HSA_ENABLE_SDMA=0
# lld resolves its toolchain (ld.lld + amdgcn device libs) relative to ROCM_PATH.
# Keep the samebank toolkit set above (its llvm/bin/ld.lld -> matching LLD 23);
# /opt/rocm has no ld.lld and would force the incompatible /usr/bin/ld.lld (LLD 18).
export ROCM_PATH=/home/jli10004/workspace/rocm-toolkit-samebank
# Benchmark needs the runtime cache on so compile happens once per kernel.
export FLYDSL_RUNTIME_ENABLE_CACHE=1
# Never dump IR during timing — it floods stdout and slows compile.
unset FLYDSL_DUMP_IR FLYDSL_DUMP_DIR

echo "Using triton: $TRITON_PY"
python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free={f/1e9:.1f}GB / {t/1e9:.1f}GB')" 2>/dev/null || true

exec python3 -u "$REPO_ROOT/scripts/compare_a8w4_baseline.py" --include-tdmopt "$@"
