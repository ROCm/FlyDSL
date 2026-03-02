export FLIR_DUMP_IR=1
export FLIR_DUMP_DIR=dumps
export HIP_VISIBLE_DEVICES=2
FLIR_REBUILD=1 AITER_LOG_MORE=1 FLIR_A8W4SMOOTH_QPARAM_FORMAT=packed4 FLIR_A8W4SMOOTH_INTERLEAVE_K64=1 FLIR_A8W4SMOOTH_OVERFLOW_GUARD=0 python tests/kernels/test_moe_gemm.py --in_dtype a8w4smooth --gemm2_mode atomic --moe_sort_mode torch --compare_aiter_ck false --skip_ref false -dim 5120,1536 -t 32 -e 16 -k 8 --tile_m 32 --tile_n 64 --tile_k 256 --tile_n2 128 --tile_k2 256 --num_warmup 10 --num_iters 100
# FLIR_REBUILD=1 AITER_LOG_MORE=1 python tests/kernels/test_moe_gemm.py --in_dtype int8smooth --gemm2_mode atomic --moe_sort_mode torch --compare_aiter_ck false --skip_ref false -dim 5120,1536 -t 110 -e 16 -k 8 --tile_m 32 --tile_n 64 --tile_k 128 --tile_n2 128 --tile_k2 128 --num_warmup 10 --num_iters 100
