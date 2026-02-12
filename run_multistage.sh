#!/usr/bin/env bash
set -euo pipefail

# Multistage preshuffle GEMM quick test.
#
# Notes:
# - `tests/kernels/test_preshuffle_gemm.py` now imports `kernels/preshuffle_gemm_multistage.py`.
# - `--lds_stage 3` means A is prefetched 2 tiles ahead (lookahead = stages-1).
# - `--use_async_copy` is only supported on gfx950 (the test will skip otherwise).

rm -rf ~/.cache/flydsl/ dumps_multistage || true

# Correctness + small perf smoke (runs on any arch; no async copy).
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=dumps_multistage \
python tests/kernels/test_preshuffle_gemm.py \
  --in_dtype fp8 \
  -M 4096 -N 4096 -K 4096 \
  --tile_m 128 --tile_n 128 --tile_k 128 \
  --lds_stage 3 \
  --num_iters 10 --num_warmup 3 \
  --no_aiter_bench

