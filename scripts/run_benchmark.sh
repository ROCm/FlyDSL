#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."

BENCH_LOG_DIR="${BENCH_LOG_DIR:-/tmp/flir_bench}"
mkdir -p "${BENCH_LOG_DIR}"

SUCCESS_COUNT=0
FAIL_COUNT=0

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Softmax/LayerNorm shapes: "M,N,dtype"
SOFTMAX_SHAPES=("32768,8192,bf16")
LAYERNORM_SHAPES=("32768,8192,bf16")

# Preshuffle GEMM shapes: "dtype,M,N,K,tile_m,tile_n,tile_k"
GEMM_SHAPES=(
  "fp8,16,40960,5120,16,128,256"
  "fp8,5120,5120,8320,64,256,128"
  "fp8,9728,8192,8320,64,256,128"
  "int8,9728,8192,8320,64,256,128"
)

# MoE shapes: "tokens,model_dim,inter_dim,experts,topk,tile_m,tile_n,tile_k,tile_n2,tile_k2"
MOE_SHAPES=(
  "32768,9728,8192,16,4,64,128,128,256,128"
  "64,6144,1024,128,8,16,64,256,64,256"
)


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
  bash scripts/run_benchmark.sh rmsnorm moe      # run only selected benchmarks
  bash scripts/run_benchmark.sh --only softmax,moe
  bash scripts/run_benchmark.sh --list

Supported ops:
  softmax | rmsnorm | gemm | moe

Notes:
  - `layernorm` is accepted as an alias of `rmsnorm` (script runs RMSNorm).
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
  local log_path="$1"
  local op_name="${2:-unknown}"
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
  local size=$1
  local name=$2
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
  local op="$1" shape="$2" dtype="$3" tbps="$4" tflops="$5"
  printf "%-14.14s %-34.34s %-10.10s %10s %10s\n" "${op}" "${shape}" "${dtype}" "${tbps}" "${tflops}"
}

_normalize_op() {
  # Normalize aliases to canonical op names.
  local op="${1:-}"
  case "${op}" in
    layernorm) echo "rmsnorm" ;;
    *) echo "${op}" ;;
  esac
}

# Default: run all benchmarks unless user selected a subset.
RUN_SOFTMAX=1
RUN_RMSNORM=1
RUN_PRESHUFFLE_GEMM=1
RUN_MOE=1

_enable_only_ops() {
  RUN_SOFTMAX=0
  RUN_RMSNORM=0
  RUN_PRESHUFFLE_GEMM=0
  RUN_MOE=0
  local op
  for op in "$@"; do
    op="$(_normalize_op "${op}")"
    case "${op}" in
      softmax) RUN_SOFTMAX=1 ;;
      rmsnorm) RUN_RMSNORM=1 ;;
      gemm) RUN_PRESHUFFLE_GEMM=1 ;;
      moe) RUN_MOE=1 ;;
      "" ) ;;
      *) _die "unknown op '${op}'" ;;
    esac
  done
}

# Parse args: if any ops are provided, run only those; otherwise run all.
if [ "$#" -gt 0 ]; then
  selected_ops=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        _usage
        exit 0
        ;;
      --list)
        echo "softmax"
        echo "rmsnorm"
        echo "gemm"
        echo "moe"
        exit 0
        ;;
      --only)
        shift
        [ "$#" -gt 0 ] || _die "--only requires a comma-separated op list"
        IFS=',' read -r -a _ops <<< "$1"
        selected_ops+=("${_ops[@]}")
        ;;
      --only=*)
        v="${1#--only=}"
        [ -n "${v}" ] || _die "--only= requires a comma-separated op list"
        IFS=',' read -r -a _ops <<< "${v}"
        selected_ops+=("${_ops[@]}")
        ;;
      --*)
        _die "unknown flag '$1'"
        ;;
      *)
        selected_ops+=("$1")
        ;;
    esac
    shift
  done
  if [ "${#selected_ops[@]}" -gt 0 ]; then
    _enable_only_ops "${selected_ops[@]}"
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
  for shape in "${SOFTMAX_SHAPES[@]}"; do
    IFS=',' read -r M N dtype <<< "$shape"
    export ROCDSL_SOFTMAX_SHAPES="$shape"
    log="${BENCH_LOG_DIR}/softmax_${M}x${N}_${dtype}.log"
    if python3 tests/kernels/test_softmax.py >"${log}" 2>&1; then
      ((SUCCESS_COUNT++))
    else
      ((FAIL_COUNT++))
      echo "softmax failed. Log: ${log}" >&2
      _show_fail_log "${log}" "softmax"
    fi
    row="$(_py_parse_and_emit softmax "${M}x${N}" "${dtype}" "${log}")"
    IFS=$'\t' read -r op_s shape_s dtype_s tbps_s tflops_s <<<"${row}"
    _emit_row "${op_s}" "${shape_s}" "${dtype_s}" "${tbps_s}" "${tflops_s}"
  done
fi

# RMSNorm (script used to label this as LayerNorm; keep output truthful)
if [ "${RUN_RMSNORM}" -eq 1 ]; then
  for shape in "${LAYERNORM_SHAPES[@]}"; do
    IFS=',' read -r M N dtype <<< "$shape"
    export ROCDSL_RMSNORM_SHAPES="$shape"
    log="${BENCH_LOG_DIR}/rmsnorm_${M}x${N}_${dtype}.log"
    if python3 tests/kernels/test_rmsnorm.py >"${log}" 2>&1; then
      ((SUCCESS_COUNT++))
    else
      ((FAIL_COUNT++))
      echo "rmsnorm failed. Log: ${log}" >&2
      _show_fail_log "${log}" "rmsnorm"
    fi
    row="$(_py_parse_and_emit rmsnorm "${M}x${N}" "${dtype}" "${log}")"
    IFS=$'\t' read -r op_s shape_s dtype_s tbps_s tflops_s <<<"${row}"
    _emit_row "${op_s}" "${shape_s}" "${dtype_s}" "${tbps_s}" "${tflops_s}"
  done
fi

# Preshuffle GEMM
if [ "${RUN_PRESHUFFLE_GEMM}" -eq 1 ]; then
  for shape in "${GEMM_SHAPES[@]}"; do
    IFS=',' read -r dtype M N K tile_m tile_n tile_k <<< "$shape"
    log="${BENCH_LOG_DIR}/preshuffle_gemm_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}.log"
    if python3 tests/kernels/test_preshuffle_gemm.py \
      --in_dtype "$dtype" \
      -M "$M" \
      -N "$N" \
      -K "$K" \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" \
      --tile_k "$tile_k" >"${log}" 2>&1; then
      ((SUCCESS_COUNT++))
    else
      ((FAIL_COUNT++))
      echo "gemm failed. Log: ${log}" >&2
      _show_fail_log "${log}" "gemm"
    fi
    row="$(_py_parse_and_emit gemm "${M}x${N}x${K}" "${dtype}" "${log}")"
    IFS=$'\t' read -r op_s shape_s dtype_s tbps_s tflops_s <<<"${row}"
    _emit_row "${op_s}" "${shape_s}" "${dtype_s}" "${tbps_s}" "${tflops_s}"
  done
fi

# MoE
if [ "${RUN_MOE}" -eq 1 ]; then
  for shape in "${MOE_SHAPES[@]}"; do
    IFS=',' read -r tokens model_dim inter_dim experts topk tile_m tile_n tile_k tile_n2 tile_k2 <<< "$shape"
    log="${BENCH_LOG_DIR}/moe_t${tokens}_md${model_dim}_id${inter_dim}_e${experts}_k${topk}.log"
    if python3 tests/kernels/test_moe_gemm.py \
      --in_dtype fp8 \
      -dim "$model_dim,$inter_dim" \
      -t "$tokens" \
      -e "$experts" \
      -k "$topk" \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" \
      --tile_k "$tile_k" \
      --tile_n2 "$tile_n2" \
      --tile_k2 "$tile_k2" \
      --compare_aiter_ck false >"${log}" 2>&1; then
      ((SUCCESS_COUNT++))
    else
      ((FAIL_COUNT++))
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
