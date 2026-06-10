#!/bin/bash
FLYDSL_DIR=/data/jli/flydsl-ws/mi450-cmodel-env/flydsl-prev
source "$FLYDSL_DIR/env-cmodel-fly.sh"

python3 "$FLYDSL_DIR/tests/kernels/test_gemm_fp8fp4_gfx1250.py" \
  --data-format a8w4 \
  --scale-mode mxscale \
  -M 1 -N 12288 -K 3072 \
  --tile-m 16 --tile-n 256 --tile-k 512 \
  --m-warp 1 --n-warp 4 \
  --num-buffers 4 \
  --split-k 1 \
  --cluster-m 1 --cluster-n 1 \
  --l2-prefetch-distance 0 \
  --out-dtype bf16 \
  --fill-mode random \
  --benchmark \
  --warmup 1 --iters 2 \
  --use-graph

# python3 "$SCRIPT_DIR/tests/kernels/test_gemm_fp8fp4_gfx1250.py" \
#   --data-format a8w4 \
#   --scale-mode mxscale \
#   -M 64 -N 12288 -K 3072 \
#   --tile-m 16 --tile-n 256 --tile-k 512 \
#   --m-warp 1 --n-warp 4 \
#   --num-buffers 4 \
#   --split-k 1 \
#   --cluster-m 1 --cluster-n 1 \
#   --l2-prefetch-distance 0 \
#   --out-dtype bf16 \
#   --fill-mode random \
#   --benchmark \
#   --warmup 1 --iters 2 \
#   --use-graph
