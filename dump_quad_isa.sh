#!/bin/bash
# Dump ISA and check accuracy for fp8_quadrant tile
# Usage: bash dump_quad_isa.sh [K]

K=${1:-3072}

FFM_DIR=$(ls -d /data/docker/overlay2/*/diff/home/user/ffm-env/rocdtif-7.13-am+ffmlite-mi400.*-rel-* 2>/dev/null | head -1)
[ -z "$FFM_DIR" ] && { echo "ERR: no rocdtif-7.13+ found" >&2; exit 1; }

source "$FFM_DIR/ffmlite_env.sh"
export FLYDSL_ROOT=/data/zanzhang/FlyDSL-main
export PYTHONPATH="/data/zanzhang/FlyDSL-main:${PYTHONPATH}"

DUMP_DIR=/tmp/isa_quad_dump
ARGS="--data-format a8w4 --scale-mode mxscale -M 1 -N 12288 -K ${K} \
      --tile-m 16 --tile-n 256 --tile-k 512 \
      --m-warp 1 --n-warp 4 --num-buffers 3 --split-k 1 \
      --cluster-m 1 --cluster-n 1 --l2-prefetch-distance 0 \
      --out-dtype bf16"

echo "=== Accuracy check (K=${K}) ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 tests/kernels/test_gemm_fp8fp4_gfx1250.py ${ARGS} --fill-mode random 2>&1 | tail -4

echo ""
echo "=== ISA dump (K=${K}) ==="
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=${DUMP_DIR} COMPILE_ONLY=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 tests/kernels/test_gemm_fp8fp4_gfx1250.py ${ARGS} 2>&1 | grep "COMPILE_ONLY"

ISA=${DUMP_DIR}/kernel_mxscale_gemm_0/21_final_isa.s
cp ${ISA} /data/zanzhang/FlyDSL-main/21_final_isa_quad.s
echo "Saved: 21_final_isa_quad.s ($(wc -l < /data/zanzhang/FlyDSL-main/21_final_isa_quad.s) lines, $(grep -c v_nop /data/zanzhang/FlyDSL-main/21_final_isa_quad.s) v_nops)"
echo ""
echo "=== barrier/wait distribution ==="
grep -n "s_barrier_wait\|s_wait_dscnt 0x0\|v_nop" /data/zanzhang/FlyDSL-main/21_final_isa_quad.s | grep -v "offset\|#" | head -30
