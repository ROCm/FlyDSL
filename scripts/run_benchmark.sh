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
PRESHUFFLE_GEMM_SHAPES=(
  "fp8,16,20480,8192,16,64,512"
  "fp8,5120,5120,8320,64,256,128"
  "fp8,9728,8192,8320,64,256,128"
  "int8,9728,8192,8320,64,256,128"
)

# MoE shapes: "tokens,model_dim,inter_dim,experts,topk,tile_m,tile_n,tile_k,tile_n2,tile_k2"
MOE_SHAPES=(
  "32768,8192,8192,16,4,64,128,128,256,128"
  # "32768,4864,4864,16,4,64,128,128,256,128"
  # "16,8192,8192,64,4,16,64,256,256,128"
)


# Memory bound threshold (M or tokens <= threshold => memory bound)
MEMORY_BOUND_THRESHOLD=512

# ============================================================================
# Helper functions
# ============================================================================

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
  printf "\n%-14s %-22s %-8s %10s %10s\n" "op" "shape" "dtype" "TB/s" "TFLOPS"
  printf "%-14s %-22s %-8s %10s %10s\n" "--------------" "----------------------" "--------" "----------" "----------"
}

_emit_row() {
  local op="$1" shape="$2" dtype="$3" tbps="$4" tflops="$5"
  printf "%-14s %-22s %-8s %10s %10s\n" "${op}" "${shape}" "${dtype}" "${tbps}" "${tflops}"
}

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
for shape in "${SOFTMAX_SHAPES[@]}"; do
  IFS=',' read -r M N dtype <<< "$shape"
  export ROCDSL_SOFTMAX_SHAPES="$shape"
  log="${BENCH_LOG_DIR}/softmax_${M}x${N}_${dtype}.log"
  if python3 tests/kernels/test_softmax.py >"${log}" 2>&1; then
    ((SUCCESS_COUNT++))
  else
    ((FAIL_COUNT++))
    echo "softmax failed. Log: ${log}" >&2
  fi
  row="$(_py_parse_and_emit softmax "${M}x${N}" "${dtype}" "${log}")"
  IFS=$'\t' read -r op_s shape_s dtype_s tbps_s tflops_s <<<"${row}"
  _emit_row "${op_s}" "${shape_s}" "${dtype_s}" "${tbps_s}" "${tflops_s}"
done

# RMSNorm (script used to label this as LayerNorm; keep output truthful)
for shape in "${LAYERNORM_SHAPES[@]}"; do
  IFS=',' read -r M N dtype <<< "$shape"
  export ROCDSL_RMSNORM_SHAPES="$shape"
  log="${BENCH_LOG_DIR}/rmsnorm_${M}x${N}_${dtype}.log"
  if python3 tests/kernels/test_rmsnorm.py >"${log}" 2>&1; then
    ((SUCCESS_COUNT++))
  else
    ((FAIL_COUNT++))
    echo "rmsnorm failed. Log: ${log}" >&2
  fi
  row="$(_py_parse_and_emit rmsnorm "${M}x${N}" "${dtype}" "${log}")"
  IFS=$'\t' read -r op_s shape_s dtype_s tbps_s tflops_s <<<"${row}"
  _emit_row "${op_s}" "${shape_s}" "${dtype_s}" "${tbps_s}" "${tflops_s}"
done

# Preshuffle GEMM
for shape in "${PRESHUFFLE_GEMM_SHAPES[@]}"; do
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
    echo "preshuffle_gemm failed. Log: ${log}" >&2
  fi
  row="$(_py_parse_and_emit preshuffle_gemm "${M}x${N}x${K}" "${dtype}" "${log}")"
  IFS=$'\t' read -r op_s shape_s dtype_s tbps_s tflops_s <<<"${row}"
  _emit_row "${op_s}" "${shape_s}" "${dtype_s}" "${tbps_s}" "${tflops_s}"
done

# MoE
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
    --num_iters 5 \
    --num_warmup 2 \
    --compare_aiter_ck false >"${log}" 2>&1; then
    ((SUCCESS_COUNT++))
  else
    ((FAIL_COUNT++))
    echo "moe failed. Log: ${log}" >&2
  fi
  # Emit stage1 + stage2 rows (parse from log; keep terminal output concise).
  shape_moe="t${tokens}-md${model_dim}-id${inter_dim}-e${experts}-k${topk}"

  dt_s1="$(grep -Eo 'FLIR MoE stage1\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
  tf_s1="$(grep -Eo 'FLIR MoE stage1\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
  tb_s1="$(grep -Eo 'FLIR MoE stage1\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
  if [ -n "${dt_s1}" ] && [ -n "${tf_s1}" ] && [ -n "${tb_s1}" ]; then
    _emit_row "moe_s1" "${shape_moe}" "${dt_s1}" "${tb_s1}" "${tf_s1}"
  fi

  dt_s2="$(grep -Eo 'FLIR MoE stage2\[[^]]+\]:' "${log}" | tail -1 | cut -d'[' -f2 | cut -d']' -f1 || true)"
  tf_s2="$(grep -Eo 'FLIR MoE stage2\[[^]]+\]:.* ([0-9.]+) TFLOPS' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
  tb_s2="$(grep -Eo 'FLIR MoE stage2\[[^]]+\]:.* ([0-9.]+) TB/s' "${log}" | tail -1 | awk '{print $(NF-1)}' || true)"
  if [ -n "${dt_s2}" ] && [ -n "${tf_s2}" ] && [ -n "${tb_s2}" ]; then
    _emit_row "moe_s2" "${shape_moe}" "${dt_s2}" "${tb_s2}" "${tf_s2}"
  fi
done

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
