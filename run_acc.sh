DUMP_DIR=/tmp/isa_quad_dump
ARGS="--data-format a8w4 --scale-mode mxscale -M 1 -N 12288 -K 3072 \
      --tile-m 16 --tile-n 64 --tile-k 512 \
      --m-warp 1 --n-warp 4 --num-buffers 4 --split-k 1 \
      --cluster-m 1 --cluster-n 1 --l2-prefetch-distance 0 \
      --out-dtype bf16"

echo "=== Accuracy check (K=${K}) ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 tests/kernels/test_gemm_fp8fp4_gfx1250.py ${ARGS} --fill-mode random 2>&1 | tail -4


