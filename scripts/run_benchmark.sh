#!/bin/sh
# POSIX sh compatible (also works in bash).
set -eu
# Enable pipefail when supported (bash/ksh/zsh); ignore if unavailable (dash/posix sh).
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi
cd "$(dirname "$0")/.."

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# Locate the build directory (default: .flir/build; fallback: build/).
BUILD_DIR="${FLIR_BUILD_DIR:-${REPO_ROOT}/.flir/build}"
if [ ! -d "${BUILD_DIR}" ] && [ -d "${REPO_ROOT}/build" ]; then
  BUILD_DIR="${REPO_ROOT}/build"
fi
PYTHON_PACKAGE_ROOT="${BUILD_DIR}/python_packages/flydsl"
export PYTHONPATH="${REPO_ROOT}/flydsl/src:${PYTHON_PACKAGE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

BENCH_LOG_DIR="${BENCH_LOG_DIR:-/tmp/flir_bench}"
mkdir -p "${BENCH_LOG_DIR}"

SUCCESS_COUNT=0
FAIL_COUNT=0

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Softmax/LayerNorm shapes: "M,N,dtype"
SOFTMAX_SHAPES='
32768,8192,bf16
'
LAYERNORM_SHAPES='
32768,8192,bf16
'

# Preshuffle GEMM shapes: "dtype,M,N,K,tile_m,tile_n,tile_k"
GEMM_SHAPES='
fp8,16,40960,5120,16,128,256
fp8,16,77824,5120,16,128,256
fp8,5120,5120,8320,64,256,128
fp8,9728,8192,8320,64,256,128
int8,9728,8192,8320,64,256,128
'

# MoE shapes: "tokens,model_dim,inter_dim,experts,topk,tile_m,tile_n,tile_k,tile_n2,tile_k2"
MOE_SHAPES='
32768,8192,8192,16,4,64,128,128,256,128
64,6144,1024,128,8,16,64,256,64,256
'


# Memory bound threshold (M or tokens <= threshold => memory bound)
MEMORY_BOUND_THRESHOLD=512

# ============================================================================
# Helper functions
# ============================================================================

_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_benchmark.sh                  # run all benchmarks (default)
  bash scripts/run_benchmark.sh softmax          # run only softmax
  bash scripts/run_benchmark.sh layernorm moe      # run only selected benchmarks
  bash scripts/run_benchmark.sh --only softmax,moe
  bash scripts/run_benchmark.sh --list

Supported ops:
  softmax | layernorm | gemm | moe

Notes:
  - `layernorm` is accepted as an alias of `layernorm` (script runs layernorm).
USAGE
}

_die() {
  echo "error: $*" >&2
  echo "" >&2
  _usage >&2
  exit 2
}

_show_fail_log() {
  # Args: log_path op_name
  log_path="$1"
  op_name="${2:-unknown}"
  if [ -f "${log_path}" ]; then
    echo "" >&2
    echo "-------------------- ${op_name} log (tail) --------------------" >&2
    tail -n 200 "${log_path}" >&2 || true
    echo "-------------------- end of ${op_name} log --------------------" >&2
    echo "" >&2
  else
    echo "[warn] ${op_name} log missing: ${log_path}" >&2
  fi
}

print_bound_info() {
  size=$1
  name=$2
  if [ "$size" -le "$MEMORY_BOUND_THRESHOLD" ]; then
    echo "    [Memory Bound Shape: small $name=$size]"
  else
    echo "    [Compute Bound Shape: large $name=$size]"
  fi
}

# Print one-line perf row (like run_tests.sh style).
_fmt_table_header() {
  # Use fixed widths and truncate long strings to keep columns aligned.
  # (bash printf supports precision on %s: %-W.Ps)
  printf "\n%-14.14s %-34.34s %-10.10s %10s %10s\n" "op" "shape" "dtype" "TB/s" "TFLOPS"
  printf "%-14.14s %-34.34s %-10.10s %10s %10s\n" "--------------" "----------------------------------" "----------" "----------" "----------"
}

_emit_row() {
  op="$1"; shape="$2"; dtype="$3"; tbps="$4"; tflops="$5"
  printf "%-14.14s %-34.34s %-10.10s %10s %10s\n" "${op}" "${shape}" "${dtype}" "${tbps}" "${tflops}"
}

_normalize_op() {
  # Normalize aliases to canonical op names.
  op="${1:-}"
  case "${op}" in
    layernorm) echo "layernorm" ;;
    *) echo "${op}" ;;
  esac
}

# Default: run all benchmarks unless user selected a subset.
RUN_SOFTMAX=1
RUN_LAYERNORM=1
RUN_PRESHUFFLE_GEMM=1
RUN_MOE=1

_enable_only_ops() {
  RUN_SOFTMAX=0
  RUN_LAYERNORM=0
  RUN_PRESHUFFLE_GEMM=0
  RUN_MOE=0
  for op in "$@"; do
    op="$(_normalize_op "${op}")"
    case "${op}" in
      softmax) RUN_SOFTMAX=1 ;;
      layernorm) RUN_LAYERNORM=1 ;;
      gemm) RUN_PRESHUFFLE_GEMM=1 ;;
      moe) RUN_MOE=1 ;;
      "" ) ;;
      *) _die "unknown op '${op}'" ;;
    esac
  done
}

# Append one selected op to a space-separated list.
SELECTED_OPS=""
SELECTED_OPS_COUNT=0
_add_selected_op() {
  # Arg: op
  v="$1"
  # Skip empty items (e.g., trailing commas).
  [ -n "${v}" ] || return 0
  if [ -z "${SELECTED_OPS}" ]; then
    SELECTED_OPS="${v}"
  else
    SELECTED_OPS="${SELECTED_OPS} ${v}"
  fi
  SELECTED_OPS_COUNT=$((SELECTED_OPS_COUNT + 1))
}

# Parse args: if any ops are provided, run only those; otherwise run all.
if [ "$#" -gt 0 ]; then
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        _usage
        exit 0
        ;;
      --list)
        echo "softmax"
        echo "layernorm"
        echo "gemm"
        echo "moe"
        exit 0
        ;;
      --only)
        shift
        [ "$#" -gt 0 ] || _die "--only requires a comma-separated op list"
        oldIFS=$IFS
        IFS=,
        # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
        set -- $1
        IFS=$oldIFS
        for op in "$@"; do
          _add_selected_op "$op"
        done
        ;;
      --only=*)
        v="${1#--only=}"
        [ -n "${v}" ] || _die "--only= requires a comma-separated op list"
        oldIFS=$IFS
        IFS=,
        # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
        set -- $v
        IFS=$oldIFS
        for op in "$@"; do
          _add_selected_op "$op"
        done
        ;;
      --*)
        _die "unknown flag '$1'"
        ;;
      *)
        _add_selected_op "$1"
        ;;
    esac
    shift
  done
  if [ "${SELECTED_OPS_COUNT}" -gt 0 ]; then
    # shellcheck disable=SC2086 # want word-splitting for ops list
    _enable_only_ops ${SELECTED_OPS}
  fi
fi

_py_parse_and_emit() {
  # Args: op shape dtype log_path [M N]
  python3 - "$@" <<'PY'
import re, sys

op = sys.argv[1]
shape = sys.argv[2]
dtype = sys.argv[3]
path = sys.argv[4]
MN = sys.argv[5:]  # deprecated (kept for backward-compat)

tbps = None
tflops = None

txt = ""
try:
    with open(path, "r", errors="ignore") as f:
        txt = f.read()
except Exception:
    txt = ""

# GEMM-style: "Throughput: ..., XX.XX TFLOPS, BW: Y.YYY TB/s"
m = None
for m in re.finditer(r"Throughput:.*?([0-9.]+)\s*TFLOPS.*?BW:\s*([0-9.]+)\s*TB/s", txt):
    pass
if m:
    tflops = float(m.group(1))
    tbps = float(m.group(2))

# MoE-style: "FLIR MoE stageX[dt]: ... XX.XX TFLOPS ... Y.YYY TB/s"
if tbps is None or tflops is None:
    m = None
    for m in re.finditer(r"FLIR MoE .*?\:\s*[0-9.]+\s*us,\s*([0-9.]+)\s*TFLOPS.*?([0-9.]+)\s*TB/s", txt):
        pass
    if m:
        tflops = float(m.group(1))
        tbps = float(m.group(2))

# Softmax/Norm-style: "Kernel avg time: X ms" + "Bandwidth: Y GB/s"
if tbps is None:
    m_bw = None
    for m_bw in re.finditer(r"Bandwidth:\s*([0-9.]+)\s*GB/s", txt):
        pass
    if m_bw:
        tbps = float(m_bw.group(1)) / 1000.0


def fmt(x):
    return "-" if x is None else f"{x:.3f}"

print(f"{op}\t{shape}\t{dtype}\t{fmt(tbps)}\t{fmt(tflops)}")
PY
}

# ============================================================================
# Run Benchmarks
# ============================================================================

echo "========================================================================"
echo "Benchmarks (logs under ${BENCH_LOG_DIR})"
echo "========================================================================"
_fmt_table_header

# Softmax (log → parse → one-line row)
if [ "${RUN_SOFTMAX}" -eq 1 ]; then
  for shape in $SOFTMAX_SHAPES; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    M=$1; N=$2; dtype=$3
    export ROCDSL_SOFTMAX_SHAPES="$shape"
    log="${BENCH_LOG_DIR}/softmax_${M}x${N}_${dtype}.log"
    if python3 tests/kernels/test_softmax.py >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "softmax failed. Log: ${log}" >&2
      _show_fail_log "${log}" "softmax"
    fi
    row="$(_py_parse_and_emit softmax "${M}x${N}" "${dtype}" "${log}")"
    # row is tab-separated; default IFS includes tabs.
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
fi

# layernorm (script used to label this as LayerNorm; keep output truthful)
if [ "${RUN_LAYERNORM}" -eq 1 ]; then
  for shape in $LAYERNORM_SHAPES; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    M=$1; N=$2; dtype=$3
    export ROCDSL_LAYERNORM_SHAPES="$shape"
    log="${BENCH_LOG_DIR}/layernorm_${M}x${N}_${dtype}.log"
    if python3 tests/kernels/test_layernorm.py >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "layernorm failed. Log: ${log}" >&2
      _show_fail_log "${log}" "layernorm"
    fi
    row="$(_py_parse_and_emit layernorm "${M}x${N}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
fi

# Preshuffle GEMM
if [ "${RUN_PRESHUFFLE_GEMM}" -eq 1 ]; then
  for shape in $GEMM_SHAPES; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    dtype=$1; M=$2; N=$3; K=$4; tile_m=$5; tile_n=$6; tile_k=$7
    log="${BENCH_LOG_DIR}/preshuffle_gemm_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}.log"
    if python3 tests/kernels/test_preshuffle_gemm.py \
      --in_dtype "$dtype" \
      --num_warmup 10 \
      --num_iters 100 \
      -M "$M" \
      -N "$N" \
      -K "$K" \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" \
      --tile_k "$tile_k" >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "gemm failed. Log: ${log}" >&2
      _show_fail_log "${log}" "gemm"
    fi
    row="$(_py_parse_and_emit gemm "${M}x${N}x${K}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
fi

# MoE
if [ "${RUN_MOE}" -eq 1 ]; then
  for shape in $MOE_SHAPES; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    tokens=$1; model_dim=$2; inter_dim=$3; experts=$4; topk=$5; tile_m=$6; tile_n=$7; tile_k=$8; tile_n2=$9; tile_k2=${10}
    log="${BENCH_LOG_DIR}/moe_t${tokens}_md${model_dim}_id${inter_dim}_e${experts}_k${topk}.log"
    if python3 tests/kernels/test_moe_gemm.py \
      --in_dtype fp8 \
      -dim "$model_dim,$inter_dim" \
      -t "$tokens" \
      -e "$experts" \
      -k "$topk" \
      --num_warmup 10 \
      --num_iters 100 \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" \
      --tile_k "$tile_k" \
      --tile_n2 "$tile_n2" \
      --tile_k2 "$tile_k2" \
      --compare_aiter_ck false >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "moe failed. Log: ${log}" >&2
      _show_fail_log "${log}" "moe"
    fi
    # Emit stage1 + stage2 rows (parse from log; keep terminal output concise).
    # Keep shape string compact (no spaces/commas) so table alignment stays stable.
    shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"

    dt_s1="$(grep -Eo 'FLIR MoE stage1\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
    tf_s1="$(grep -Eo 'FLIR MoE stage1\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    tb_s1="$(grep -Eo 'FLIR MoE stage1\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    if [ -n "${dt_s1}" ] && [ -n "${tf_s1}" ] && [ -n "${tb_s1}" ]; then
      _emit_row "moe_gemm1" "${shape_moe}" "${dt_s1}" "${tb_s1}" "${tf_s1}"
    fi

    dt_s2="$(grep -Eo 'FLIR MoE stage2\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
    tf_s2="$(grep -Eo 'FLIR MoE stage2\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    tb_s2="$(grep -Eo 'FLIR MoE stage2\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    if [ -n "${dt_s2}" ] && [ -n "${tf_s2}" ] && [ -n "${tb_s2}" ]; then
      _emit_row "moe_gemm2" "${shape_moe}" "${dt_s2}" "${tb_s2}" "${tf_s2}"
    fi
  done
fi

# Summary
TOTAL=$((SUCCESS_COUNT + FAIL_COUNT))
echo ""
echo "========================================================================"
echo "Benchmark Summary"
echo "========================================================================"
echo "Total: ${TOTAL} tests"
echo "Success: ${SUCCESS_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Logs: ${BENCH_LOG_DIR}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
  echo "All benchmarks passed! "
  exit 0
else
  echo "Some benchmarks failed. Check the output above for details."
  exit 1
fi
