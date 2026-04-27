#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# POSIX sh compatible (also works in bash).
set -eu
# Enable pipefail when supported (bash/ksh/zsh); ignore if unavailable (dash/posix sh).
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi
cd "$(dirname "$0")/.."

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# Locate the build directory (default: build-fly; fallback: build/).
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
if [ ! -d "${BUILD_DIR}" ] && [ -d "${REPO_ROOT}/build" ]; then
  BUILD_DIR="${REPO_ROOT}/build"
fi
PYTHON_PACKAGE_ROOT="${BUILD_DIR}/python_packages"
# Ensure build packages take priority over pip-installed flydsl
export PYTHONPATH="${PYTHON_PACKAGE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
MLIR_LIBS_DIR="${PYTHON_PACKAGE_ROOT}/flydsl/_mlir/_mlir_libs"
if [ -d "${MLIR_LIBS_DIR}" ]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

BENCH_LOG_DIR="${BENCH_LOG_DIR:-/tmp/flydsl_bench}"
mkdir -p "${BENCH_LOG_DIR}"

# Auto-select GPU with the most free VRAM (skip if HIP_VISIBLE_DEVICES is already set).
if [ -z "${HIP_VISIBLE_DEVICES:-}" ] && command -v python3 >/dev/null 2>&1; then
    _best_gpu=$(python3 -c "
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    print(best)
" 2>/dev/null || true)
    if [ -n "${_best_gpu}" ]; then
        export HIP_VISIBLE_DEVICES="${_best_gpu}"
        echo "[run_benchmark] Auto-selected GPU ${_best_gpu} (most free VRAM)"
    fi
fi

# Detect GPU architecture — CDNA-only benchmarks are guarded by IS_CDNA.
GPU_ARCH=$(python3 -c "from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())" 2>/dev/null || echo "unknown")
IS_CDNA=false
IS_RDNA4=false
case "${GPU_ARCH}" in gfx9*) IS_CDNA=true ;; esac
case "${GPU_ARCH}" in gfx120*) IS_RDNA4=true ;; esac
echo "[run_benchmark] GPU arch: ${GPU_ARCH} (CDNA=${IS_CDNA}, RDNA4=${IS_RDNA4})"

SUCCESS_COUNT=0
FAIL_COUNT=0

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Softmax/LayerNorm/RMSNorm shapes: "M,N,dtype"
SOFTMAX_SHAPES='
32768,8192,bf16
'
LAYERNORM_SHAPES='
32768,8192,bf16
'
RMSNORM_SHAPES='
32768,8192,bf16
'

# Preshuffle GEMM shapes: "dtype,M,N,K,tile_m,tile_n,tile_k"
GEMM_SHAPES='
fp8,16,40960,5120,16,128,256
fp8,16,77824,5120,16,128,256
fp8,256,2112,7168,64,64,256
fp8,512,2112,7168,64,64,256
fp8,5120,5120,8320,64,256,128
fp8,9728,8192,8320,64,256,128
fp8,8192,8192,8192,128,256,128
int8,9728,8192,8320,64,256,128
int4,9728,8192,8320,64,256,128
bf16,5120,5120,8320,64,256,128
'

# Async preshuffle GEMM shapes:
# "dtype,M,N,K,tile_m,tile_n,tile_k[,waves_per_eu]"
GEMM_SHAPES_ASYNC='
fp8,256,2112,7168,64,64,256
fp8,512,2112,7168,64,64,256
fp8,5120,5120,8320,128,256,128,2
fp8,9728,8192,8320,128,256,128,2
fp8,8192,8192,8192,128,256,128,2
int8,9728,8192,8320,128,256,128,2
'

# FP4 GEMM shapes (requires --wfp4, gfx950 only): "M,N,K,tile_m,tile_n,tile_k"
GEMM_FP4_SHAPES='
8192,8192,8192,64,128,256
8192,8192,8192,64,256,256
8192,8192,8192,128,256,256
8192,8192,8192,128,256,128
'

# Async FP4 GEMM shapes:
# "M,N,K,tile_m,tile_n,tile_k[,waves_per_eu]"
GEMM_FP4_SHAPES_ASYNC='

8192,8192,8192,128,256,128,2
8192,8192,8192,128,256,256,2
'

# MoE shapes: "tokens,model_dim,inter_dim,experts,topk,tile_m,tile_n,tile_k,tile_n2,tile_k2"
MOE_SHAPES='
32768,8192,8192,16,4,64,128,128,256,128
64,6144,1024,128,8,16,64,256,64,256
'

# MoE FP4 shapes (requires --in_dtype fp4, gfx950 only): same format as MOE_SHAPES
MOE_FP4_SHAPES='
16,7168,256,257,9,64,256,256,256,256
128,7168,256,257,9,64,256,256,256,256
2048,7168,256,257,9,64,256,256,256,256
16384,7168,256,257,9,64,256,256,256,256
32768,7168,256,257,9,64,256,256,256,256
16,7168,2048,32,8,64,256,256,256,256
128,7168,2048,32,8,64,256,256,256,256
2048,7168,2048,32,8,64,256,256,256,256
8192,7168,2048,32,8,64,256,256,256,256
32768,7168,2048,32,8,64,256,256,256,256
'

# MoE W4A16 groupwise shapes (int4_bf16, group_size=32): same format as MOE_SHAPES
# Kimi 2.5 TP8: model_dim=7168, inter_dim=256, E=384, topk=8
MOE_W4A16_SHAPES='
128,7168,256,384,8,16,128,128,128,256
256,7168,256,384,8,16,128,128,128,256
512,7168,256,384,8,16,128,128,128,256
'

# MoE A8W4 shapes (FP8 activation + MX-FP4 weight, gfx950 only): same format as MOE_SHAPES.
# GPT-OSS inspired: model_dim=3072, inter_dim=3072, E=128, topk=4; sweep tokens from 512 to
# bracket memory- and compute-bound regimes.  tile_m>=32 / tile_k>=256 are MX-FP4 layout requirements.
MOE_A8W4_SHAPES='
512,3072,3072,128,4,32,128,256,256,256
1024,3072,3072,128,4,32,128,256,256,256
2048,3072,3072,128,4,32,128,256,256,256
4096,3072,3072,128,4,32,128,256,256,256
8192,3072,3072,128,4,32,128,256,256,256
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
  softmax | layernorm | rmsnorm | gemm | moe
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
  # op column is wide enough to host "moe_<family>_s2_atomic" / "_reduce" suffixes.
  printf "\n%-22.22s %-34.34s %-10.10s %10s %10s\n" "op" "shape" "dtype" "TB/s" "TFLOPS"
  printf "%-22.22s %-34.34s %-10.10s %10s %10s\n" "----------------------" "----------------------------------" "----------" "----------" "----------"
}

_emit_row() {
  op="$1"; shape="$2"; dtype="$3"; tbps="$4"; tflops="$5"
  printf "%-22.22s %-34.34s %-10.10s %10s %10s\n" "${op}" "${shape}" "${dtype}" "${tbps}" "${tflops}"
}

_normalize_op() {
  # Normalize aliases to canonical op names.
  op="${1:-}"
  case "${op}" in
    layernorm) echo "layernorm" ;;
    *) echo "${op}" ;;
  esac
}

# Default: run softmax, norms, and GEMM unless user selected a subset.
# Use positional args or --only to enable others: softmax, layernorm, rmsnorm, gemm, moe
RUN_SOFTMAX=1
RUN_LAYERNORM=1
RUN_RMSNORM=1
RUN_PRESHUFFLE_GEMM=1
RUN_MOE=1

_enable_only_ops() {
  RUN_SOFTMAX=0
  RUN_LAYERNORM=0
  RUN_RMSNORM=0
  RUN_PRESHUFFLE_GEMM=0
  RUN_MOE=0
  for op in "$@"; do
    op="$(_normalize_op "${op}")"
    case "${op}" in
      softmax) RUN_SOFTMAX=1 ;;
      layernorm) RUN_LAYERNORM=1 ;;
      rmsnorm) RUN_RMSNORM=1 ;;
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
        echo "rmsnorm"
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

# MoE-style: "FlyDSL MoE stageX[dt]: ... XX.XX TFLOPS ... Y.YYY TB/s"
if tbps is None or tflops is None:
    m = None
    for m in re.finditer(r"FlyDSL MoE .*?\:\s*[0-9.]+\s*us,\s*([0-9.]+)\s*TFLOPS.*?([0-9.]+)\s*TB/s", txt):
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

_emit_moe_s2_rows() {
  # Args: op_prefix shape log_path
  # Extract separate atomic/reduce rows from MoE stage2 log lines. A line looks like:
  #   FlyDSL MoE stage2 [moe_gemm2] fp4 atomic | 7168x2048, ... | 1163.2 us, 1654.24 TFLOPS, 0.377 TB/s
  # Emit two table rows (op_prefix_atomic, op_prefix_reduce). Falls back to single row
  # tagged "mixed" if the log only has one mode (e.g., --gemm2_mode was overridden).
  op_prefix="$1"; shape="$2"; log_path="$3"
  python3 - "$op_prefix" "$shape" "$log_path" <<'PY'
import re, sys

op_prefix, shape, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path, "r", errors="ignore") as f:
        txt = f.read()
except Exception:
    txt = ""

pat = re.compile(
    r"FlyDSL MoE stage2 \[[^]]+\]\s+(\S+)\s+(atomic|reduce)\b.*?"
    r"([0-9.]+)\s*TFLOPS.*?([0-9.]+)\s*TB/s"
)
# keep last occurrence per mode
found = {}
for m in pat.finditer(txt):
    dtype, mode = m.group(1), m.group(2)
    found[mode] = (dtype, float(m.group(3)), float(m.group(4)))

def fmt(x):
    return "-" if x is None else f"{x:.3f}"

# Always emit atomic row first (if any), then reduce row.
emitted = False
for mode in ("atomic", "reduce"):
    if mode not in found:
        continue
    dtype, tflops, tbps = found[mode]
    print(f"{op_prefix}_{mode}\t{shape}\t{dtype}\t{fmt(tbps)}\t{fmt(tflops)}")
    emitted = True

if not emitted:
    # Nothing parsed — emit empty row so caller knows.
    print(f"{op_prefix}_atomic\t{shape}\t-\t-\t-")
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

# RMSNorm
if [ "${RUN_RMSNORM}" -eq 1 ]; then
  for shape in $RMSNORM_SHAPES; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    M=$1; N=$2; dtype=$3
    export ROCDSL_RMSNORM_SHAPES="$shape"
    log="${BENCH_LOG_DIR}/rmsnorm_${M}x${N}_${dtype}.log"
    if python3 tests/kernels/test_rmsnorm.py >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "rmsnorm failed. Log: ${log}" >&2
      _show_fail_log "${log}" "rmsnorm"
    fi
    row="$(_py_parse_and_emit rmsnorm "${M}x${N}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
fi

# Preshuffle GEMM (CDNA only — uses MFMA)
if [ "${RUN_PRESHUFFLE_GEMM}" -eq 1 ] && [ "${IS_CDNA}" = "true" ]; then
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
    gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}"
    row="$(_py_parse_and_emit gemm "${gemm_shape_tag}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done

  GEMM_USE_ASYNC_COPY="${GEMM_USE_ASYNC_COPY:-1}"
  GEMM_WAVES_PER_EU="${GEMM_WAVES_PER_EU:-2}"

  for shape in $GEMM_SHAPES_ASYNC; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    dtype=$1; M=$2; N=$3; K=$4; tile_m=$5; tile_n=$6; tile_k=$7
    shape_waves_per_eu="${8:-}"

    async_copy_flag=""
    async_copy_tag="async_copy"
    if [ "${GEMM_USE_ASYNC_COPY}" = "1" ] || [ "${GEMM_USE_ASYNC_COPY}" = "true" ]; then
      async_copy_flag="--use_async_copy"
    fi
    waves_per_eu="${shape_waves_per_eu:-${GEMM_WAVES_PER_EU}}"
    waves_per_eu_tag="${waves_per_eu}"

    log="${BENCH_LOG_DIR}/preshuffle_gemm_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}_${async_copy_tag}_${waves_per_eu_tag}.log"
    if python3 tests/kernels/test_preshuffle_gemm.py \
      --in_dtype "$dtype" \
      --num_warmup 10 \
      --num_iters 100 \
      -M "$M" \
      -N "$N" \
      -K "$K" \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" \
      --tile_k "$tile_k" \
      ${async_copy_flag} \
      --waves_per_eu "${waves_per_eu}" >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "gemm failed. Log: ${log}" >&2
      _show_fail_log "${log}" "gemm"
    fi
    shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_${waves_per_eu}tg"
    row="$(_py_parse_and_emit gemm_async "${shape_tag}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
  
  # FP4 GEMM (gfx950 only)
  for shape in $GEMM_FP4_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    M=$1; N=$2; K=$3; tile_m=$4; tile_n=$5; tile_k=$6
    dtype="fp4"
    log="${BENCH_LOG_DIR}/preshuffle_gemm_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}.log"
    if python3 tests/kernels/test_preshuffle_gemm.py \
      --wfp4 \
      --in_dtype fp4 \
      --num_warmup 10 \
      --num_iters 100 \
      -M "$M" \
      -N "$N" \
      -K "$K" \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" \
      --tile_k "$tile_k" >"${log}" 2>&1; then
      # Check if test was skipped due to architecture
      if grep -q "Skipping FP4 GEMM test\|Skipped" "${log}"; then
        gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}"
        _emit_row "gemm" "${gemm_shape_tag}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}"
        row="$(_py_parse_and_emit gemm "${gemm_shape_tag}" "${dtype}" "${log}")"
        set -- $row
        _emit_row "$1" "$2" "$3" "$4" "$5"
      fi
    else
      # Skip gracefully on unsupported architectures or missing features
      if grep -q "gfx950\|invalid choice\|Skipped\|not supported" "${log}" 2>/dev/null; then
        gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}"
        _emit_row "gemm" "${gemm_shape_tag}" "${dtype}" "skip" "skip"
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "gemm fp4 failed. Log: ${log}" >&2
        _show_fail_log "${log}" "gemm_fp4"
      fi

    fi
  done

  # FP4 GEMM async problem sizes (gfx950 only)
  GEMM_FP4_USE_ASYNC_COPY="${GEMM_FP4_USE_ASYNC_COPY:-1}"
  GEMM_FP4_WAVES_PER_EU="${GEMM_FP4_WAVES_PER_EU:-2}"

  for shape in $GEMM_FP4_SHAPES_ASYNC; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    M=$1; N=$2; K=$3; tile_m=$4; tile_n=$5; tile_k=$6
    shape_waves_per_eu="${7:-}"
    dtype="fp4"

    async_copy_flag=""
    async_copy_tag="async_copy"
    if [ "${GEMM_FP4_USE_ASYNC_COPY}" = "1" ] || [ "${GEMM_FP4_USE_ASYNC_COPY}" = "true" ]; then
      async_copy_flag="--use_async_copy"
    fi
    waves_per_eu="${shape_waves_per_eu:-${GEMM_FP4_WAVES_PER_EU}}"
    waves_per_eu_tag="${waves_per_eu}"

    log="${BENCH_LOG_DIR}/preshuffle_gemm_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}_${async_copy_tag}_${waves_per_eu_tag}.log"
    if python3 tests/kernels/test_preshuffle_gemm.py \
      --wfp4 \
      --in_dtype fp4 \
      --num_warmup 10 \
      --num_iters 100 \
      -M "$M" \
      -N "$N" \
      -K "$K" \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" \
      --tile_k "$tile_k" \
      ${async_copy_flag} \
      --waves_per_eu "${waves_per_eu}" >"${log}" 2>&1; then
      if grep -q "Skipping FP4 GEMM test\|Skipped" "${log}"; then
        gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_${waves_per_eu}tg"
        _emit_row "gemm_fp4_async" "${gemm_shape_tag}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_${waves_per_eu}tg"
        row="$(_py_parse_and_emit gemm_fp4_async "${gemm_shape_tag}" "${dtype}" "${log}")"
        set -- $row
        _emit_row "$1" "$2" "$3" "$4" "$5"
      fi
    else
      if grep -q "gfx950\|invalid choice\|Skipped\|not supported" "${log}" 2>/dev/null; then
        gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_${waves_per_eu}tg"
        _emit_row "gemm_fp4_async" "${gemm_shape_tag}" "${dtype}" "skip" "skip"
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "gemm fp4 async failed. Log: ${log}" >&2
        _show_fail_log "${log}" "gemm_fp4_async"
      fi
    fi
  done
fi

# MoE (CDNA only — uses MFMA)
if [ "${RUN_MOE}" -eq 1 ] && [ "${IS_CDNA}" = "true" ]; then
  for shape in $MOE_SHAPES; do
    oldIFS=$IFS
    IFS=,
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
      --skip_ref false \
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

    dt_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
    tf_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    tb_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    if [ -n "${dt_s1}" ] && [ -n "${tf_s1}" ] && [ -n "${tb_s1}" ]; then
      _emit_row "moe_gemm1" "${shape_moe}" "${dt_s1}" "${tb_s1}" "${tf_s1}"
    fi

    _emit_moe_s2_rows "moe_gemm2" "${shape_moe}" "${log}" | while IFS="$(printf '\t')" read -r _op _sh _dt _tb _tf; do
      _emit_row "${_op}" "${_sh}" "${_dt}" "${_tb}" "${_tf}"
    done
  done

  # MoE FP4 (gfx950 only)
  for shape in $MOE_FP4_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    tokens=$1; model_dim=$2; inter_dim=$3; experts=$4; topk=$5; tile_m=$6; tile_n=$7; tile_k=$8; tile_n2=$9; tile_k2=${10}
    dtype="fp4"
    log="${BENCH_LOG_DIR}/moe_fp4_t${tokens}_md${model_dim}_id${inter_dim}_e${experts}_k${topk}.log"
    if python3 tests/kernels/test_moe_gemm.py \
      --in_dtype fp4 \
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
      --skip_ref false \
      --compare_aiter_ck false >"${log}" 2>&1; then
      # Check if test was skipped due to architecture
      if grep -q "requires gfx950\|Skipping FP4" "${log}"; then
        shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"
        _emit_row "moe_fp4" "${shape_moe}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"

        dt_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
        tf_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
        tb_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
        if [ -n "${dt_s1}" ] && [ -n "${tf_s1}" ] && [ -n "${tb_s1}" ]; then
          _emit_row "moe_fp4_s1" "${shape_moe}" "${dt_s1}" "${tb_s1}" "${tf_s1}"
        fi

        _emit_moe_s2_rows "moe_fp4_s2" "${shape_moe}" "${log}" | while IFS="$(printf '\t')" read -r _op _sh _dt _tb _tf; do
          _emit_row "${_op}" "${_sh}" "${_dt}" "${_tb}" "${_tf}"
        done
      fi
    else
      # Skip gracefully on unsupported architectures
      if grep -q "requires gfx950\|Skipping FP4\|not supported" "${log}" 2>/dev/null; then
        shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"
        _emit_row "moe_fp4" "${shape_moe}" "${dtype}" "skip" "skip"
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "moe fp4 failed. Log: ${log}" >&2
        _show_fail_log "${log}" "moe_fp4"
      fi
    fi
  done

  # MoE W4A16 groupwise (int4_bf16, group_size=32)
  for shape in $MOE_W4A16_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    set -- $shape
    IFS=$oldIFS
    tokens=$1; model_dim=$2; inter_dim=$3; experts=$4; topk=$5; tile_m=$6; tile_n=$7; tile_k=$8; tile_n2=$9; tile_k2=${10}
    log="${BENCH_LOG_DIR}/moe_w4a16_t${tokens}_md${model_dim}_id${inter_dim}_e${experts}_k${topk}.log"
    if python3 tests/kernels/test_moe_gemm.py \
      --in_dtype int4_bf16 \
      --group_size 32 \
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
      --skip_ref false \
      --compare_aiter_ck false >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "moe w4a16 failed. Log: ${log}" >&2
      _show_fail_log "${log}" "moe_w4a16"
    fi
    shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"

    dt_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
    tf_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    tb_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
    if [ -n "${dt_s1}" ] && [ -n "${tf_s1}" ] && [ -n "${tb_s1}" ]; then
      _emit_row "moe_w4a16_s1" "${shape_moe}" "${dt_s1}" "${tb_s1}" "${tf_s1}"
    fi

    _emit_moe_s2_rows "moe_w4a16_s2" "${shape_moe}" "${log}" | while IFS="$(printf '\t')" read -r _op _sh _dt _tb _tf; do
      _emit_row "${_op}" "${_sh}" "${_dt}" "${_tb}" "${_tf}"
    done
  done

  # MoE A8W4 — FP8 activation + MX-FP4 weight (gfx950 only). End-to-end 2-stage:
  # stage1 a_dtype=fp8,b_dtype=fp4 -> silu(gate)*up fp16 -> MX-FP8 re-quant -> stage2.
  for shape in $MOE_A8W4_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    tokens=$1; model_dim=$2; inter_dim=$3; experts=$4; topk=$5; tile_m=$6; tile_n=$7; tile_k=$8; tile_n2=$9; tile_k2=${10}
    dtype="a8w4"
    shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"
    log="${BENCH_LOG_DIR}/moe_a8w4_t${tokens}_md${model_dim}_id${inter_dim}_e${experts}_k${topk}.log"
    if python3 tests/kernels/test_moe_gemm.py \
      --in_dtype a8w4 \
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
      --skip_ref false \
      --compare_aiter_ck false >"${log}" 2>&1; then
      # CLI prints "Skipping a8w4: requires gfx950+" on unsupported archs.
      if grep -q "requires gfx950\|Skipping a8w4" "${log}"; then
        _emit_row "moe_a8w4" "${shape_moe}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        dt_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
        tf_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
        tb_s1="$(grep -Eo 'FlyDSL MoE stage1\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
        if [ -n "${dt_s1}" ] && [ -n "${tf_s1}" ] && [ -n "${tb_s1}" ]; then
          _emit_row "moe_a8w4_s1" "${shape_moe}" "${dt_s1}" "${tb_s1}" "${tf_s1}"
        fi

        _emit_moe_s2_rows "moe_a8w4_s2" "${shape_moe}" "${log}" | while IFS="$(printf '\t')" read -r _op _sh _dt _tb _tf; do
          _emit_row "${_op}" "${_sh}" "${_dt}" "${_tb}" "${_tf}"
        done
      fi
    else
      if grep -q "requires gfx950\|Skipping a8w4\|not supported" "${log}" 2>/dev/null; then
        _emit_row "moe_a8w4" "${shape_moe}" "${dtype}" "skip" "skip"
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "moe a8w4 failed. Log: ${log}" >&2
        _show_fail_log "${log}" "moe_a8w4"
      fi
    fi
  done
fi

# RDNA4 WMMA GEMM benchmarks (via benchmark_common.py)
if [ "${IS_RDNA4}" = "true" ]; then
  echo ""
  echo "========================================================================"
  echo "RDNA4 WMMA Benchmarks"
  echo "========================================================================"
  log="${BENCH_LOG_DIR}/rdna_wmma_sweep.log"
  if python3 -c "from tests.kernels.benchmark_common import run_wmma_sweep, print_perf_table; rows = run_wmma_sweep(); print_perf_table(rows)" >"${log}" 2>&1; then
    cat "${log}"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "RDNA4 WMMA benchmark failed. Log: ${log}" >&2
    tail -20 "${log}" >&2
  fi
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
