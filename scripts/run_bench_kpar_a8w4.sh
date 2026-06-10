#!/usr/bin/env bash
# Benchmark k_warp (k_warp=4) A8W4 GEMM vs the current main kernel on real
# gfx1250 hardware. Same env workarounds as the other run_*.sh scripts.
#
# Usage:
#   bash scripts/run_bench_kpar_a8w4.sh
#   bash scripts/run_bench_kpar_a8w4.sh -M 1 --repeat 5 --warmup 20 --iters 100
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

export HSA_ENABLE_SDMA=0
export ROCM_PATH=/opt/rocm
export FLYDSL_RUNTIME_ENABLE_CACHE=1
# unset FLYDSL_DUMP_IR FLYDSL_DUMP_DIR

echo "Using triton: $TRITON_PY"
python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free={f/1e9:.1f}GB / {t/1e9:.1f}GB')" 2>/dev/null || true

exec python3 -u "$REPO_ROOT/scripts/bench_kpar_a8w4.py" "$@"
