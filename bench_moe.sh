#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# MoE 2-stage GEMM benchmark sweep
#
# Problem sizes:
#   tokens:  256 / 512 / 1024 / 1536 / 2048
#   experts: 384, ep=8  => 48 experts per EP
#   model_dim/inter_dim: 3584/1024, 5120/1536, 7168/2048
#
# Supported dtypes: fp8, int8smooth, a8w4smooth (and others from test_moe_gemm.py CLI)
#
# Usage:
#   bash bench_moe.sh                              # default: fp8, all sizes
#   bash bench_moe.sh fp8                          # fp8 only
#   bash bench_moe.sh int8smooth                   # int8smooth only
#   bash bench_moe.sh a8w4smooth                   # a8w4smooth (stage1 + stage2)
#   bash bench_moe.sh fp8,int8smooth,a8w4smooth    # all three dtypes sequentially
#   bash bench_moe.sh fp8 20 5                     # fp8, 20 iters, 5 warmup

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=./
export PYTHONPATH=/workspace/FLIR/build-fly/python_packages:${PYTHONPATH}
export FLYDSL_RUNTIME_ENABLE_CACHE=0
export FLIR_REBUILD=1
export FLIR_A8W4SMOOTH_QPARAM_FORMAT=packed4
export FLIR_A8W4SMOOTH_INTERLEAVE_K64=1
export FLIR_A8W4SMOOTH_OVERFLOW_GUARD=1

IN_DTYPES="${1:-fp8}"
NUM_ITERS="${2:-20}"
NUM_WARMUP="${3:-5}"

# TOKENS=(256 512 1024 1536 2048)
TOKENS=(256 4096)
# DIMS=("3584,1024" "5120,1536" "7168,2048")
DIMS=("3584,1024")
EXPERTS=48    # 384 total / ep=8
TOPK=8

SEP="================================================================================"

echo "$SEP"
echo "  MoE 2-stage GEMM Benchmark"
echo "  dtype=$IN_DTYPES  experts=$EXPERTS  topk=$TOPK  iters=$NUM_ITERS  warmup=$NUM_WARMUP"
echo "  tokens: ${TOKENS[*]}"
echo "  dims:   ${DIMS[*]}"
echo "$SEP"
echo ""

# Split comma-separated dtypes
IFS=',' read -ra DTYPE_LIST <<< "$IN_DTYPES"

for IN_DTYPE in "${DTYPE_LIST[@]}"; do

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  dtype = $IN_DTYPE"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

for DIM in "${DIMS[@]}"; do
    MODEL_DIM="${DIM%,*}"
    INTER_DIM="${DIM#*,}"
    echo "──────────────────────────────────────────────────────────────────"
    echo "  model_dim=$MODEL_DIM  inter_dim=$INTER_DIM"
    echo "──────────────────────────────────────────────────────────────────"
    for T in "${TOKENS[@]}"; do
        echo ""
        echo ">>> tokens=$T  dim=${MODEL_DIM}x${INTER_DIM}  E=$EXPERTS  topk=$TOPK"

        if [ "$IN_DTYPE" = "a8w4smooth" ]; then
            # a8w4smooth: __main__ only runs stage1; use inline python for both stages
            python -u -c "
import os, sys, torch
os.environ.setdefault('FLIR_A8W4SMOOTH_QPARAM_FORMAT', 'packed4')
os.environ.setdefault('FLIR_A8W4SMOOTH_INTERLEAVE_K64', '1')
os.environ.setdefault('FLIR_A8W4SMOOTH_OVERFLOW_GUARD', '1')
torch.set_default_device('cuda')
sys.path.insert(0, '.')
from tests.kernels.test_moe_gemm import run_moe_stage1_a8w4smooth, run_moe_stage2_a8w4smooth

kwargs = dict(
    tokens=$T, model_dim=$MODEL_DIM, inter_dim=$INTER_DIM,
    experts=$EXPERTS, topk=$TOPK,
    tile_m=32, tile_n=128, tile_k=256,
    seed=0, num_iters=$NUM_ITERS, num_warmup=$NUM_WARMUP,
    moe_sort_mode='torch', skip_ref=True,
)
run_moe_stage1_a8w4smooth(**kwargs)
run_moe_stage2_a8w4smooth(**kwargs)
" 2>&1 | grep -E "TFLOPS|TB/s|stage[12]|Error|SKIP|skip" || true
        else
            # fp8 / int8 / bf16 etc: use the standard CLI
            python -u tests/kernels/test_moe_gemm.py \
                --in_dtype "$IN_DTYPE" \
                -dim "$DIM" \
                -t "$T" \
                -e "$EXPERTS" \
                -k "$TOPK" \
                --tile_m 32 \
                --tile_n 128 \
                --tile_k 256 \
                --tile_n2 256 \
                --tile_k2 256 \
                --gemm2_mode atomic \
                --moe_sort_mode torch \
                --compare_aiter_ck false \
                --skip_ref true \
                --num_iters "$NUM_ITERS" \
                --num_warmup "$NUM_WARMUP" \
                2>&1 | grep -E "TFLOPS|TB/s|stage[12]|Error|SKIP|skip" || true
        fi
        echo "---"
    done
    echo ""
done

done

echo "$SEP"
echo "  Benchmark complete."
echo "$SEP"
