#!/usr/bin/env bash
# Temporary launcher: run the A8W4 GEMM benchmark on real gfx1250 hardware,
# bypassing the broken editable triton install.
#
# Why: pip's editable `triton` points at /root/triton-mi450 which has no
# compiled libtriton.so, so `import triton` fails. We prepend a triton build
# that DOES have the .so (upstream_triton) to PYTHONPATH. For mxscale+random
# this benchmark only imports triton; data prep is pure torch.
#
# Usage:
#   bash scripts/run_gemm_a8w4_realhw.sh                 # default args below
#   bash scripts/run_gemm_a8w4_realhw.sh -M 64 --iters 50   # override/add args
#   (any args you pass are appended and override the defaults)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── working triton (has compiled libtriton.so) ──
TRITON_PY="/root/upstream_triton/python"
if ! PYTHONPATH="$TRITON_PY" python3 -c "import triton" 2>/dev/null; then
    echo "ERR: triton at $TRITON_PY does not import. Edit TRITON_PY in this script." >&2
    echo "     Candidates: /root/upstream_triton/python , /workspace/qiwan/pyenv/lib/python3.12/site-packages" >&2
    exit 1
fi

export PYTHONPATH="$TRITON_PY:${REPO_ROOT}/build-fly/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"
export FLYDSL_ROOT="$REPO_ROOT"

# SDMA on this box faults on H2D (null-address page fault); force blit-engine
# copies instead. Without this the benchmark dies before the kernel runs.
export HSA_ENABLE_SDMA=0

# lld (ld.lld) resolves its toolchain relative to ROCM_PATH; without it the
# backend link step fails with "could not find path component ... ld.lld".
export ROCM_PATH=/opt/rocm

# ── default benchmark args (override by passing your own on the CLI) ──
DEFAULT_ARGS=(
    --data-format a8w4
    --scale-mode mxscale
    -M 1 -N 12288 -K 3072
    --tile-m 16 --tile-n 64 --tile-k 512
    --m-warp 1 --n-warp 4
    --num-buffers 4
    --split-k 1
    --cluster-m 1 --cluster-n 1
    --l2-prefetch-distance 0
    --out-dtype bf16
    --fill-mode random
    --benchmark
    --warmup 8
    --iters 30
    --use-graph
)

echo "Using triton: $TRITON_PY"
python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free={f/1e9:.1f}GB / {t/1e9:.1f}GB')" 2>/dev/null || true
echo "Args: ${DEFAULT_ARGS[*]} $*"
echo "========================================"

exec python3 -u "$REPO_ROOT/tests/kernels/test_gemm_fp8fp4_gfx1250.py" "${DEFAULT_ARGS[@]}" "$@"
