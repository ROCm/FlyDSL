#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Preshuffle GEMM benchmark: v1 (old pipeline) vs v2 (layout API)
#
# Usage:
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh          # all shapes
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh fp16     # fp16 only
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh bf16     # bf16 only
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh fp8      # fp8 only
#   bash tests/kernels/bench_gemm_v1_vs_v2.sh sweep     # tile sweep (fp16, M=128)

set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONPATH=./
export FLYDSL_RUNTIME_ENABLE_CACHE=0

SEP="=============================================================================="
ITERS=20
WARMUP=5
FILTER="${1:-all}"

# ── Helper: run v1 (old pipeline) ────────────────────────────────────────────
run_v1() {
    local dtype=$1 M=$2 N=$3 K=$4 tm=$5 tn=$6 tk=$7
    python tests/kernels/test_preshuffle_gemm.py \
        --in_dtype "$dtype" -M "$M" -N "$N" -K "$K" \
        --tile_m "$tm" --tile_n "$tn" --tile_k "$tk" \
        --out_dtype bf16 --num_iters "$ITERS" --num_warmup "$WARMUP" \
        --no_aiter_bench 2>&1 | grep -E "Throughput|Tile|={10,}"
}

# ── Helper: run v2 (layout API) with a single config ─────────────────────────
# Prints the comparison table row: v2 us/TFLOPS | old us/TFLOPS | ratio
run_v2() {
    local dtype=$1 M=$2 N=$3 K=$4 tm=$5 tn=$6 tk=$7
    python tests/kernels/bench_preshuffle_gemm_v2.py \
        --dtype "$dtype" -M "$M" -N "$N" -K "$K" \
        --tile_m "$tm" --tile_n "$tn" --tile_k "$tk" \
        --iters "$ITERS" --warmup "$WARMUP" --no-check 2>&1 \
        | grep -E "^\s+[0-9]|tile\s+k\s" | tail -2
}

# ── Run one shape on both pipelines ──────────────────────────────────────────
bench_shape() {
    local dtype=$1 M=$2 N=$3 K=$4 tm=$5 tn=$6 tk=$7
    echo ""
    echo "--- ${dtype^^} M=${M} N=${N} K=${K} tile=${tm}x${tn}x${tk} (k_iters=$((tk/32))) ---"
    echo "[v1] old pipeline:"
    run_v1 "$dtype" "$M" "$N" "$K" "$tm" "$tn" "$tk"
    echo "[v2] layout API:"
    run_v2 "$dtype" "$M" "$N" "$K" "$tm" "$tn" "$tk"
}

# ── FP16 shapes ──────────────────────────────────────────────────────────────
run_fp16() {
    echo "$SEP"
    echo "  FP16 Benchmarks"
    echo "$SEP"

    # Best tile for layout API (k_iters=2)
    bench_shape fp16  128  5120 8192   64 128  64
    bench_shape fp16  128  5120 8192  128 128  64

    # k_iters=4
    bench_shape fp16  128  5120 8192   64 128 128

    # Large compute-bound (old pipe reaches ~200T here)
    bench_shape fp16 5120  5120 8192   64 128  64
    bench_shape fp16 5120  5120 8192   64 256  64
    bench_shape fp16 5120  5120 8192   64 128 128

    # Small M stress test (k_iters=16)
    bench_shape fp16   32  5120 8192   32  64 512
}

# ── BF16 shapes ──────────────────────────────────────────────────────────────
run_bf16() {
    echo "$SEP"
    echo "  BF16 Benchmarks"
    echo "$SEP"

    bench_shape bf16  128  5120 8192   64 128  64
    bench_shape bf16  128  5120 8192   64 128 128
    bench_shape bf16 5120  5120 8192   64 128  64
    bench_shape bf16 5120  5120 8192   64 256  64
}

# ── FP8 shapes ───────────────────────────────────────────────────────────────
run_fp8() {
    echo "$SEP"
    echo "  FP8 Benchmarks (v2 delegates to old path for fp8)"
    echo "$SEP"

    bench_shape fp8    16  5120 8192   16  64 256
    bench_shape fp8   128  5120 8192   64 128 128
    bench_shape fp8  5120  5120 8320   64 256 128
}

# ── Tile sweep ───────────────────────────────────────────────────────────────
run_sweep() {
    echo "$SEP"
    echo "  FP16 tile sweep (M=128, N=5120, K=8192)"
    echo "$SEP"
    python tests/kernels/bench_preshuffle_gemm_v2.py \
        --dtype fp16 --sweep -M 128 -N 5120 -K 8192 \
        --iters "$ITERS" --warmup "$WARMUP" --no-check
    echo ""
    echo "$SEP"
    echo "  FP16 tile sweep (M=5120, N=5120, K=8192)"
    echo "$SEP"
    python tests/kernels/bench_preshuffle_gemm_v2.py \
        --dtype fp16 --sweep -M 5120 -N 5120 -K 8192 \
        --iters "$ITERS" --warmup "$WARMUP" --no-check
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "$SEP"
echo "  Preshuffle GEMM: v1 (old pipeline) vs v2 (layout API)"
echo "  GPU: $(python -c 'from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())' 2>/dev/null || echo 'unknown')"
echo "  Iters: ${ITERS}, Warmup: ${WARMUP}"
echo "$SEP"

case "$FILTER" in
    fp16)  run_fp16 ;;
    bf16)  run_bf16 ;;
    fp8)   run_fp8 ;;
    sweep) run_sweep ;;
    all)
        run_fp16
        run_bf16
        run_fp8
        ;;
    *)
        echo "Usage: $0 [fp16|bf16|fp8|sweep|all]"
        exit 1
        ;;
esac

echo ""
echo "$SEP"
echo "  Done."
echo "$SEP"
