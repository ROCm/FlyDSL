#!/usr/bin/env bash
# Run PTPC A8W4 GEMM benchmark on FFM-Lite (gfx1250 cmodel) with ISA dump.
#
# Usage:
#   bash scripts/run_gemm_ptpc_a8w4_ffm.sh          # default M=1,64
#   bash scripts/run_gemm_ptpc_a8w4_ffm.sh 1         # M=1 only
#   bash scripts/run_gemm_ptpc_a8w4_ffm.sh 1 64 128  # M=1,64,128
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── FFM-Lite env ──
FFM_DIR=$(ls -d /data/docker/overlay2/*/diff/home/user/ffm-env/rocdtif-7.13-am+ffmlite-mi400.*-rel-* 2>/dev/null | head -1)
[ -z "$FFM_DIR" ] && { echo "ERR: no rocdtif-7.13+ ffm-lite found" >&2; exit 1; }
echo "Sourcing FFM-Lite env: $FFM_DIR"
set +u
source "$FFM_DIR/ffmlite_env.sh"
set -u

# ── FlyDSL paths ──
export FLYDSL_ROOT="$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/build-fly/python_packages:${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"

# ── ISA dump ──
export FLYDSL_DUMP_IR=1
export FLYDSL_RUNTIME_ENABLE_CACHE=0
DUMP_DIR="${FLYDSL_DUMP_DIR:-$HOME/.flydsl/debug}"
echo "ISA will be dumped to: $DUMP_DIR"

# ── M values ──
if [ $# -gt 0 ]; then
    M_VALUES=("$@")
else
    M_VALUES=(1 64)
fi

LOGDIR="/tmp/gemm_ptpc_a8w4_ffm"
mkdir -p "$LOGDIR"

for M in "${M_VALUES[@]}"; do
    LOGFILE="$LOGDIR/M${M}.log"
    echo ""
    echo "========================================"
    echo "  Running PTPC A8W4 GEMM: M=$M N=12288 K=3072"
    echo "  Log: $LOGFILE"
    echo "========================================"

    python3 -u "$REPO_ROOT/tests/kernels/test_gemm_fp8fp4_gfx1250.py" \
        --data-format a8w4 \
        --scale-mode mxscale \
        -M "$M" -N 12288 -K 3072 \
        --tile-m 16 --tile-n 128 --tile-k 256 \
        --m-warp 1 --n-warp 4 \
        --num-buffers 4 \
        --split-k 1 \
        --cluster-m 1 --cluster-n 1 \
        --l2-prefetch-distance 0 \
        --out-dtype bf16 \
        --benchmark \
        --warmup 0 \
        --iters 0 \
        --use-graph \
        2>&1 | tee "$LOGFILE"

    echo ""
    echo "── ISA files for M=$M ──"
    find "$DUMP_DIR" -name "*.s" -newer "$LOGFILE" -o -name "*final_isa*" -newer "$LOGFILE" 2>/dev/null | sort
done

echo ""
echo "All ISA dumps under: $DUMP_DIR"
ls -lhrt "$DUMP_DIR"/*/*.s 2>/dev/null | tail -10
