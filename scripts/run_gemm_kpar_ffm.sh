#!/usr/bin/env bash
# Run k_warp GEMM correctness test on FFM-Lite (gfx1250 cmodel).
#
# Usage:
#   bash scripts/run_gemm_kpar_ffm.sh                         # default fp4 M=1 N=512 K=512
#   bash scripts/run_gemm_kpar_ffm.sh --data-format fp8
#   bash scripts/run_gemm_kpar_ffm.sh -M 1 -N 512 -K 512 --k-warp 4 --tile-k 512
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

LOGFILE="/tmp/gemm_kpar_ffm.log"
echo ""
echo "========================================"
echo "  k_warp GEMM test (gfx1250 cmodel)"
echo "  Args: $*"
echo "  Log: $LOGFILE"
echo "========================================"

python3 -u "$REPO_ROOT/tests/kernels/test_gemm_kpar_ffm.py" "$@" 2>&1 | tee "$LOGFILE"

echo ""
echo "── ISA files ──"
find "$DUMP_DIR" -name "*.s" -newer "$LOGFILE" -o -name "*final_isa*" -newer "$LOGFILE" 2>/dev/null | sort

echo ""
echo "Done. ISA dumps under: $DUMP_DIR"
