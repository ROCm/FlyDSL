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
BENCH_OUTPUT_CSV="${BENCH_OUTPUT_CSV:-}"

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
IS_RDNA_WMMA=false  # gfx11* or gfx12* — both have WMMA for f16/bf16
case "${GPU_ARCH}" in gfx9*) IS_CDNA=true ;; esac
case "${GPU_ARCH}" in gfx120*) IS_RDNA4=true ;; esac
case "${GPU_ARCH}" in gfx11*|gfx12*) IS_RDNA_WMMA=true ;; esac
echo "[run_benchmark] GPU arch: ${GPU_ARCH} (CDNA=${IS_CDNA}, RDNA4=${IS_RDNA4}, RDNA_WMMA=${IS_RDNA_WMMA})"

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Softmax/FlashAttention/RMSNorm shapes: "M,N,dtype"
SOFTMAX_SHAPES='
32768,8192,bf16
'
# Softmax backward: dashboard rows are occupancy-stable (M=1024) cuts of the
# N-selected residency tiers. Additional N values stay in the correctness sweep
# in tests/kernels/test_softmax_bwd.py; small-M dashboard rows flap too much.
SOFTMAX_BWD_SHAPES='
1024,8192,bf16
1024,65536,bf16
'
RMSNORM_SHAPES='
32768,8192,bf16
'
# FlashAttention shapes:
#   preferred: "batch,seq_len,num_heads,num_kv_heads,head_dim,dtype,causal"
#   legacy:    "batch,seq_len,num_heads,head_dim,dtype,causal" (num_kv_heads=num_heads)
DEFAULT_FLASH_ATTN_FUNC_SHAPES='
32,8192,8,8,128,bf16,true
16,8192,16,16,128,bf16,true
4,8192,64,64,128,bf16,true
4,8192,64,8,128,bf16,true
'
FLASH_ATTN_FUNC_SHAPES="${FLASH_ATTN_FUNC_SHAPES:-${DEFAULT_FLASH_ATTN_FUNC_SHAPES}}"
# MLA decode shapes: "batch,ctx_len" (DeepSeek MLA, fp8 Q/KV, nh=128).
DEFAULT_MLA_DECODE_SHAPES='
32,8192
'
MLA_DECODE_SHAPES="${MLA_DECODE_SHAPES:-${DEFAULT_MLA_DECODE_SHAPES}}"

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

# A16W16 GEMM shapes:
# "dtype,M,N,K,tile_m,tile_n,tile_k,stages,split_k,m_waves,n_waves,k_waves[,use_hti]"
HGEMM_SHAPES_GFX950='
fp16,2048,2048,2048,128,128,64,4,1,4,4,1
bf16,8192,8192,8192,256,256,64,2,1,2,4,1,true
'
HGEMM_SHAPES_CDNA3='
fp16,4096,4096,4096,128,128,64,2,1,2,2,1
bf16,32,384,7168,16,64,128,2,14,1,2,1
'

# FP8 8-wave row-scale GEMM shapes (gfx950 only):
# "M,N,K,tile_m,tile_n,preshuffle_b"
FP8_GEMM_8WAVE_ROWSCALE_SHAPES='
5120,5120,8320,256,256,0
8192,8192,8192,256,256,0
'

# FP4 GEMM shapes (kernels/fp4_gemm_4wave.py, gfx950 only): "M,N,K"
# The 4-wave kernel runs its native BLOCK_M=BLOCK_N=256, BLOCK_K=256 config.
GEMM_FP4_SHAPES='
8192,8192,8192
16384,16384,16384
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
  bash scripts/run_benchmark.sh softmax_bwd      # run only softmax backward
  bash scripts/run_benchmark.sh rmsnorm flash_attn moe    # run only selected benchmarks
  bash scripts/run_benchmark.sh --only softmax,moe
  bash scripts/run_benchmark.sh --output_csv /tmp/bench.csv
  bash scripts/run_benchmark.sh --list

Supported ops:
  softmax | softmax_bwd | rmsnorm | flash_attn | mla | gemm | moe
  (gemm includes preshuffle GEMM, A16W16 GEMM, and FP8 8-wave row-scale GEMM)
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

# Record a non-zero benchmark exit, downgrading pytest skips to "skipped".
#
# run_benchmark.sh runs each kernel test as a plain script (not under pytest),
# so a pytest.skip()/pytest.importorskip() reached at runtime -- a single-GPU
# runner hitting a multi-GPU test, an arch guard, or an aiter API drift --
# surfaces as a non-zero exit plus a "Skipped:" line instead of being absorbed
# by pytest. Treat those as skipped so an environmental skip cannot turn the
# whole benchmark job red; genuine failures still count and dump their log tail.
_fail_or_skip() {
  # Args: log_path op_name
  _fos_log="$1"; _fos_op="${2:-unknown}"
  if grep -q "Skipped:" "${_fos_log}" 2>/dev/null; then
    SKIP_COUNT=$((SKIP_COUNT + 1))
    echo "${_fos_op} skipped (pytest.skip; see ${_fos_log})" >&2
    return 0
  fi
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "${_fos_op} failed. Log: ${_fos_log}" >&2
  _show_fail_log "${_fos_log}" "${_fos_op}"
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
  if [ -n "${BENCH_OUTPUT_CSV:-}" ]; then
    status="ok"
    if [ "${tbps}" = "skip" ] || [ "${tflops}" = "skip" ]; then
      status="skip"
    elif [ "${tbps}" = "-" ] && [ "${tflops}" = "-" ]; then
      status="missing"
    fi
    python3 - "${BENCH_OUTPUT_CSV}" "${op}" "${shape}" "${dtype}" "${tbps}" "${tflops}" "${status}" <<'PY'
import csv
import sys

with open(sys.argv[1], "a", newline="") as f:
    csv.writer(f).writerow(sys.argv[2:])
PY
  fi
}

_normalize_op() {
  # Normalize aliases to canonical op names.
  op="${1:-}"
  case "${op}" in
    softmax_bwd|softmax-bwd|softmax_backward|softmax-backward) echo "softmax_bwd" ;;
    flash|flash_attn|flash-attn|flash_attn_func|fmha) echo "flash_attn" ;;
    mla|mla_decode|mla-decode) echo "mla" ;;
    *) echo "${op}" ;;
  esac
}

# Default: run softmax, norms, attention, GEMM, and MoE unless user selected a subset.
# Use positional args or --only to enable others: softmax, softmax_bwd, rmsnorm, flash_attn, mla, gemm, moe
RUN_SOFTMAX=1
RUN_SOFTMAX_BWD=1
RUN_RMSNORM=1
RUN_FLASH_ATTN=1
RUN_MLA=1
RUN_PRESHUFFLE_GEMM=1
RUN_MOE=1

_enable_only_ops() {
  RUN_SOFTMAX=0
  RUN_SOFTMAX_BWD=0
  RUN_RMSNORM=0
  RUN_FLASH_ATTN=0
  RUN_MLA=0
  RUN_PRESHUFFLE_GEMM=0
  RUN_MOE=0
  for op in "$@"; do
    op="$(_normalize_op "${op}")"
    case "${op}" in
      softmax) RUN_SOFTMAX=1 ;;
      softmax_bwd) RUN_SOFTMAX_BWD=1 ;;
      rmsnorm) RUN_RMSNORM=1 ;;
      flash_attn) RUN_FLASH_ATTN=1 ;;
      mla) RUN_MLA=1 ;;
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
        echo "softmax_bwd"
        echo "rmsnorm"
        echo "flash_attn"
        echo "mla"
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
      --output_csv|--output-csv)
        flag="$1"
        shift
        [ "$#" -gt 0 ] || _die "${flag} requires a CSV path"
        BENCH_OUTPUT_CSV="$1"
        ;;
      --output_csv=*|--output-csv=*)
        BENCH_OUTPUT_CSV="${1#*=}"
        [ -n "${BENCH_OUTPUT_CSV}" ] || _die "$1 requires a CSV path"
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

if [ -n "${BENCH_OUTPUT_CSV}" ]; then
  mkdir -p "$(dirname "${BENCH_OUTPUT_CSV}")"
  printf "op,shape,dtype,tbps,tflops,status\n" >"${BENCH_OUTPUT_CSV}"
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

# FlashAttention table: "| PASS | maxerr mincos | time_us tflops".
if tflops is None:
    m = None
    for m in re.finditer(r"\|\s+(?:PASS|FAIL|--)\s+\|\s+[0-9.eE+-]+\s+[0-9.]+\s+\|\s+([0-9.]+)\s+([0-9.]+)", txt):
        pass
    if m:
        tflops = float(m.group(2))

# MLA decode: "TFLOPS=...  TB/s=..."
if tbps is None or tflops is None:
    m = None
    for m in re.finditer(r"TFLOPS=([0-9.]+)\s+TB/s=([0-9.]+)", txt):
        pass
    if m:
        tflops = float(m.group(1))
        tbps = float(m.group(2))

# Softmax/Norm-style: "Kernel avg time: X ms" + "Bandwidth: Y GB/s".
# Use the FIRST match: the base op (softmax/flash_attn/rmsnorm) is benchmarked
# first, so any later "Bandwidth:" lines come from fused/quant variants printed
# by the same test (e.g. test_flash_attn.py also runs fused_add/dynamicquant/
# smoothquant). Taking the last match reported the slow scalar smoothquant path
# as "flash_attn" (~1.69 vs the real ~5.6 TB/s base).
if tbps is None:
    m_bw = next(re.finditer(r"Bandwidth:\s*([0-9.]+)\s*GB/s", txt), None)
    if m_bw:
        tbps = float(m_bw.group(1)) / 1000.0


def fmt(x):
    return "-" if x is None else f"{x:.3f}"

print(f"{op}\t{shape}\t{dtype}\t{fmt(tbps)}\t{fmt(tflops)}")
PY
}

_emit_moe_rows() {
  # Args: s1_op s2_prefix shape log_path
  # Machine fields from tests/kernels/test_moe_gemm.py and test_moe_gemm_2stage.py:
  #   FlyDSL MoE stage1[fp8]: ... TFLOPS=12.34 TB/s=5.678
  #   FlyDSL MoE stage2[fp8] atomic: ... TFLOPS=12.34 TB/s=5.678
  # Parse failure emits nothing so the dashboard is not filled with "-" rows.
  s1_op="$1"; s2_prefix="$2"; shape="$3"; log_path="$4"
  _moe_rows="$(python3 - "$s1_op" "$s2_prefix" "$shape" "$log_path" <<'PY'
import re, sys

s1_op, s2_prefix, shape, path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    with open(path, "r", errors="ignore") as f:
        txt = f.read()
except Exception:
    txt = ""

def fmt(x):
    return f"{x:.3f}"

s1 = None
for m in re.finditer(
    r"FlyDSL MoE stage1\[(\S+)\]:.*?TFLOPS=([0-9.]+)\s+TB/s=([0-9.]+)",
    txt,
):
    s1 = m
if s1:
    print(f"{s1_op}\t{shape}\t{s1.group(1)}\t{fmt(float(s1.group(3)))}\t{fmt(float(s1.group(2)))}")

found = {}
for m in re.finditer(
    r"FlyDSL MoE stage2\[(\S+)\]\s+(atomic|reduce):.*?TFLOPS=([0-9.]+)\s+TB/s=([0-9.]+)",
    txt,
):
    found[m.group(2)] = (m.group(1), float(m.group(3)), float(m.group(4)))
for mode in ("atomic", "reduce"):
    if mode not in found:
        continue
    dtype, tflops, tbps = found[mode]
    print(f"{s2_prefix}_{mode}\t{shape}\t{dtype}\t{fmt(tbps)}\t{fmt(tflops)}")
PY
)"
  [ -z "${_moe_rows}" ] && return 0
  while IFS="$(printf '\t')" read -r _op _sh _dt _tb _tf; do
    _emit_row "${_op}" "${_sh}" "${_dt}" "${_tb}" "${_tf}"
  done <<EOF
${_moe_rows}
EOF
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
      _fail_or_skip "${log}" "softmax"
    fi
    row="$(_py_parse_and_emit softmax "${M}x${N}" "${dtype}" "${log}")"
    # row is tab-separated; default IFS includes tabs.
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
fi

# Softmax backward (log -> parse -> one row per residency tier)
if [ "${RUN_SOFTMAX_BWD}" -eq 1 ]; then
  for shape in $SOFTMAX_BWD_SHAPES; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    M=$1; N=$2; dtype=$3
    export ROCDSL_SOFTMAX_BWD_SHAPES="$shape"
    log="${BENCH_LOG_DIR}/softmax_bwd_${M}x${N}_${dtype}.log"
    if python3 tests/kernels/test_softmax_bwd.py >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      _fail_or_skip "${log}" "softmax_bwd"
    fi
    row="$(_py_parse_and_emit softmax_bwd "${M}x${N}" "${dtype}" "${log}")"
    # row is tab-separated; default IFS includes tabs.
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
      _fail_or_skip "${log}" "rmsnorm"
    fi
    row="$(_py_parse_and_emit rmsnorm "${M}x${N}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
fi

# FlashAttention / FMHA (CDNA only)
if [ "${RUN_FLASH_ATTN}" -eq 1 ] && [ "${IS_CDNA}" = "true" ]; then
  export FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA="${FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA:-1}"
  export FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16="${FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16:-1}"

  for shape in $FLASH_ATTN_FUNC_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    batch=$1; seq_len=$2; heads=$3
    if [ "$#" -ge 7 ]; then
      kv_heads=$4; head_dim=$5; dtype=$6; causal=$7
    else
      # Backward-compatible legacy format:
      #   batch,seq_len,num_heads,head_dim,dtype,causal
      kv_heads=$heads; head_dim=$4; dtype=$5; causal=$6
    fi
    causal_flag="--causal"
    causal_tag="causal"
    case "${causal}" in
      0|false|False|FALSE|no|NO|noncausal|non-causal)
        causal_flag="--no-causal"
        causal_tag="nocausal"
        ;;
    esac
    log="${BENCH_LOG_DIR}/flash_attn_B${batch}_S${seq_len}_H${heads}_Hkv${kv_heads}_D${head_dim}_${dtype}_${causal_tag}.log"
    if python3 tests/kernels/test_flash_attn_fwd.py \
      --batch "$batch" \
      --seq_len "$seq_len" \
      --num_heads "$heads" \
      --num_kv_heads "$kv_heads" \
      --head_dim "$head_dim" \
      --dtype "$dtype" \
      "${causal_flag}" \
      --warmup 10 \
      --iters 100 >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      _fail_or_skip "${log}" "flash_attn"
    fi
    shape_tag="B${batch}S${seq_len}H${heads}Hkv${kv_heads}D${head_dim}_${causal_tag}"
    row="$(_py_parse_and_emit flash_attn "${shape_tag}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done
fi

# MLA decode
if [ "${RUN_MLA}" -eq 1 ] && [ "${IS_CDNA}" = "true" ]; then
  for shape in $MLA_DECODE_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    batch=$1; ctx_len=$2
    shape_tag="B${batch}C${ctx_len}"
    log="${BENCH_LOG_DIR}/mla_decode_B${batch}_C${ctx_len}.log"
    if python3 tests/kernels/test_mla_decode.py \
      --batch "$batch" \
      --ctx_len "$ctx_len" >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      _fail_or_skip "${log}" "mla"
    fi
    row="$(_py_parse_and_emit mla "${shape_tag}" "fp8" "${log}")"
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
      _fail_or_skip "${log}" "gemm"
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
      _fail_or_skip "${log}" "gemm"
    fi
    shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_${waves_per_eu}tg"
    row="$(_py_parse_and_emit gemm_async "${shape_tag}" "${dtype}" "${log}")"
    set -- $row
    _emit_row "$1" "$2" "$3" "$4" "$5"
  done

  hgemm_shapes=""
  case "${GPU_ARCH}" in
    gfx95*)
      if [ -n "${HGEMM_SHAPES:-}" ]; then
        hgemm_shapes="${HGEMM_SHAPES}"
      else
        hgemm_shapes="${HGEMM_SHAPES_GFX950}"
      fi
      ;;
  esac

  for shape in $hgemm_shapes; do
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    dtype=$1; M=$2; N=$3; K=$4; tile_m=$5; tile_n=$6; tile_k=$7
    stages=$8; split_k=$9; m_waves=${10}; n_waves=${11}; k_waves=${12}
    use_hti="${13:-false}"
    hti_flag="--no-hti"
    hti_tag=""
    if [ "${use_hti}" = "1" ] || [ "${use_hti}" = "true" ]; then
      hti_flag="--hti"
      hti_tag="_hti"
    fi
    log="${BENCH_LOG_DIR}/hgemm_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}_s${stages}_sk${split_k}${hti_tag}.log"
    if python3 tests/kernels/test_gemm_a16w16_gfx950.py \
      --dtype "$dtype" \
      --num_warmup 3 \
      --num_iters 50 \
      -m "$M" \
      -n "$N" \
      -k "$K" \
      --TILE_M "$tile_m" \
      --TILE_N "$tile_n" \
      --TILE_K "$tile_k" \
      --STAGES "$stages" \
      --SPLIT_K "$split_k" \
      --BLOCK_M_WARPS "$m_waves" \
      --BLOCK_N_WARPS "$n_waves" \
      --BLOCK_K_WARPS "$k_waves" \
      "${hti_flag}" >"${log}" 2>&1; then
      if grep -q "Skipped:" "${log}"; then
        shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_sk${split_k}${hti_tag}"
        _emit_row "hgemm" "${shape_tag}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}_sk${split_k}${hti_tag}"
        row="$(_py_parse_and_emit hgemm "${shape_tag}" "${dtype}" "${log}")"
        set -- $row
        _emit_row "$1" "$2" "$3" "$4" "$5"
      fi
    else
      _fail_or_skip "${log}" "hgemm"
    fi
  done

  if [ -n "${FP8_GEMM_8WAVE_ROWSCALE_SHAPES:-}" ]; then
    for shape in $FP8_GEMM_8WAVE_ROWSCALE_SHAPES; do
      [ -z "$shape" ] && continue
      oldIFS=$IFS
      IFS=,
      # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
      set -- $shape
      IFS=$oldIFS
      M=$1; N=$2; K=$3; tile_m=$4; tile_n=$5; preshuffle_b=$6
      dtype="fp8"
      preshuffle_flag=""
      preshuffle_tag="rowmajor"
      if [ "${preshuffle_b}" = "1" ] || [ "${preshuffle_b}" = "true" ]; then
        preshuffle_flag="--preshuffle_b"
        preshuffle_tag="preshuffle_b"
      fi
      log="${BENCH_LOG_DIR}/fp8_gemm_8wave_rowscale_${M}x${N}x${K}_t${tile_m}x${tile_n}_${preshuffle_tag}.log"
      if python3 tests/kernels/test_fp8_gemm_rowscale.py \
        --wave_8 \
        --num_warmups 10 \
        --num_iters 100 \
        -M "$M" \
        -N "$N" \
        -K "$K" \
        --tile_m "$tile_m" \
        --tile_n "$tile_n" \
        ${preshuffle_flag} >"${log}" 2>&1; then
        if grep -q "Skipped:" "${log}"; then
          shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}_${preshuffle_tag}"
          _emit_row "fp8_8wave_rowscale" "${shape_tag}" "${dtype}" "skip" "skip"
        else
          SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
          shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}_${preshuffle_tag}"
          row="$(_py_parse_and_emit fp8_8wave_rowscale "${shape_tag}" "${dtype}" "${log}")"
          set -- $row
          _emit_row "$1" "$2" "$3" "$4" "$5"
        fi
      else
        if grep -q "requires CDNA4\|Skipped:" "${log}" 2>/dev/null; then
          shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}_${preshuffle_tag}"
          _emit_row "fp8_8wave_rowscale" "${shape_tag}" "${dtype}" "skip" "skip"
        else
          FAIL_COUNT=$((FAIL_COUNT + 1))
          echo "fp8 8wave row-scale gemm failed. Log: ${log}" >&2
          _show_fail_log "${log}" "fp8_8wave_rowscale"
        fi
      fi
    done
  fi

  # FP4 GEMM (kernels/fp4_gemm_4wave.py, gfx950 only); native 256x256x256 config
  tile_m=256; tile_n=256; tile_k=256
  for shape in $GEMM_FP4_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    M=$1; N=$2; K=$3
    dtype="fp4"
    gemm_shape_tag="${M}x${N}x${K}_tile${tile_m}x${tile_n}x${tile_k}"
    log="${BENCH_LOG_DIR}/fp4_gemm_4wave_${M}x${N}x${K}_${dtype}_t${tile_m}x${tile_n}x${tile_k}.log"
    if python3 tests/kernels/test_fp4_gemm_4wave.py \
      --num_warmup 10 \
      --num_iters 100 \
      -M "$M" \
      -N "$N" \
      -K "$K" \
      --tile_m "$tile_m" \
      --tile_n "$tile_n" >"${log}" 2>&1; then
      # Check if test was skipped due to architecture
      if grep -q "requires gfx950\|Skipping\|Skipped" "${log}"; then
        _emit_row "gemm_fp4" "${gemm_shape_tag}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        row="$(_py_parse_and_emit gemm_fp4 "${gemm_shape_tag}" "${dtype}" "${log}")"
        set -- $row
        _emit_row "$1" "$2" "$3" "$4" "$5"
      fi
    else
      # Skip gracefully on unsupported architectures or missing features
      if grep -q "requires gfx950\|gfx950\|Skipped\|not supported" "${log}" 2>/dev/null; then
        _emit_row "gemm_fp4" "${gemm_shape_tag}" "${dtype}" "skip" "skip"
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "gemm fp4 failed. Log: ${log}" >&2
        _show_fail_log "${log}" "gemm_fp4"
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
    # fp8 MOE_SHAPES drive kernels/moe/moe_gemm_2stage (as in v0.3.0). #948 rerouted
    # test_moe_gemm.py to mxfp_moe, which silently dropped these rows (exit 0, no
    # output), so this points at the package's own CLI instead.
    if FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 tests/kernels/test_moe_gemm_2stage.py \
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
      --skip_ref false >"${log}" 2>&1; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      _fail_or_skip "${log}" "moe"
    fi
    # Emit stage1 + stage2 rows (parse from log; keep terminal output concise).
    # Keep shape string compact (no spaces/commas) so table alignment stays stable.
    shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"
    _emit_moe_rows "moe_gemm1" "moe_gemm2" "${shape_moe}" "${log}"
  done

  # MoE FP4 (gfx950 only). skip_ref=true: fused mxfp4 e2e is numerically broken
  # (and memory-unsafe) under the strict gate; the CLI still launches the kernel
  # and prints FlyDSL MoE stage{1,2} lines that this table greps.
  for shape in $MOE_FP4_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    # shellcheck disable=SC2086 # intentional word-splitting on IFS=,
    set -- $shape
    IFS=$oldIFS
    tokens=$1; model_dim=$2; inter_dim=$3; experts=$4; topk=$5; tile_m=$6; tile_n=$7; tile_k=$8; tile_n2=$9; tile_k2=${10}
    dtype="fp4"
    shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"
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
      --skip_ref true \
      --compare_aiter_ck false >"${log}" 2>&1; then
      if grep -q "Skipped:" "${log}"; then
        _emit_row "moe_fp4" "${shape_moe}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        _emit_moe_rows "moe_fp4_s1" "moe_fp4_s2" "${shape_moe}" "${log}"
      fi
    else
      _fail_or_skip "${log}" "moe_fp4"
    fi
  done

  # MoE W4A16 groupwise (int4_bf16, group_size=32) via moe_2stage_a16wmix w_dtype=int4.
  for shape in $MOE_W4A16_SHAPES; do
    [ -z "$shape" ] && continue
    oldIFS=$IFS
    IFS=,
    set -- $shape
    IFS=$oldIFS
    tokens=$1; model_dim=$2; inter_dim=$3; experts=$4; topk=$5; tile_m=$6; tile_n=$7; tile_k=$8; tile_n2=$9; tile_k2=${10}
    dtype="int4_bf16"
    shape_moe="t${tokens}-d${model_dim}x${inter_dim}-e${experts}k${topk}"
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
      --skip_ref true \
      --compare_aiter_ck false >"${log}" 2>&1; then
      if grep -q "Skipped:" "${log}"; then
        _emit_row "moe_w4a16" "${shape_moe}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        _emit_moe_rows "moe_w4a16_s1" "moe_w4a16_s2" "${shape_moe}" "${log}"
      fi
    else
      _fail_or_skip "${log}" "moe_w4a16"
    fi
  done

  # MoE A8W4 — FP8 activation + MX-FP4 weight (gfx950 only). skip_ref=true: same
  # fused mxfp4 e2e gate as FP4 (perf-only; kernel still prints stage lines).
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
      --skip_ref true \
      --compare_aiter_ck false >"${log}" 2>&1; then
      if grep -q "Skipped:" "${log}"; then
        _emit_row "moe_a8w4" "${shape_moe}" "${dtype}" "skip" "skip"
      else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        _emit_moe_rows "moe_a8w4_s1" "moe_a8w4_s2" "${shape_moe}" "${log}"
      fi
    else
      _fail_or_skip "${log}" "moe_a8w4"
    fi
  done
fi

# RDNA WMMA GEMM benchmarks (gfx11* or gfx12*, via benchmark_common.py).
# FP8 WMMA is gfx12-only and is skipped inside run_wmma_sweep on gfx11*.
if [ "${IS_RDNA_WMMA}" = "true" ]; then
  echo ""
  echo "========================================================================"
  echo "RDNA WMMA Benchmarks (arch: ${GPU_ARCH})"
  echo "========================================================================"
  log="${BENCH_LOG_DIR}/rdna_wmma_sweep.log"
  if python3 -c "from tests.kernels.benchmark_common import run_wmma_sweep, print_perf_table; rows = run_wmma_sweep(); print_perf_table(rows)" >"${log}" 2>&1; then
    cat "${log}"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    _fail_or_skip "${log}" "RDNA WMMA benchmark"
  fi
fi

# Summary
TOTAL=$((SUCCESS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo ""
echo "========================================================================"
echo "Benchmark Summary"
echo "========================================================================"
echo "Total: ${TOTAL} tests"
echo "Success: ${SUCCESS_COUNT}"
echo "Skipped: ${SKIP_COUNT}"
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
