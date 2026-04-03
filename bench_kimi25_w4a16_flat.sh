#!/bin/bash
# Kimi 2.5 TP8 W4A16 groupwise(g=32) MoE GEMM benchmark
# Compares preshuffle vs CK-tile flat layout for both Stage1 and Stage2
# Also compares bit-manipulation vs v_cvt_pk_bf16_f32 for f32->bf16 packing
# model_dim=7168, inter_dim=256, E=384, topk=8
# Assembly dumped to ./dumps/
export FLYDSL_RUNTIME_ENABLE_CACHE=0
export PYTHONPATH=./build/python_packages:./
export HSA_TOOLS_LIB=""
export HSA_TOOLS_REPORT_LOAD_FAILURE=0
export FLYDSL_DUMP_IR=1
export FLYDSL_DUMP_DIR=./dumps

python -c "
import torch
from tests.kernels.test_moe_gemm import run_moe_stage1, run_moe_stage2

# ---- Correctness (small shape, skip_ref=False) ----
print('====== Correctness Check ======')
ckw = dict(tokens=32, model_dim=1024, inter_dim=256, experts=8, topk=2,
           in_dtype='int4_bf16', group_size=32, doweight_stage1=False,
           num_iters=2, skip_ref=False)
a2c = torch.randn(32, 2, 256, device='cuda', dtype=torch.bfloat16) * 0.1
for flat in [False, True]:
    for cvt in [False, True]:
        tag = ('flat' if flat else 'preshuffle') + ('+cvt_pk' if cvt else '+bitmanip')
        print(f'  S1 {tag} tk=256:', end=' '); run_moe_stage1(**ckw, tile_m=16, tile_n=64, tile_k=256, use_flat_layout=flat, use_cvt_pk_bf16=cvt)
        print(f'  S2 {tag} tk=256:', end=' '); run_moe_stage2(**ckw, tile_m=16, tile_n=128, tile_k=256, a2_fp8_in=a2c, a2_scale_in=None, use_flat_layout=flat, use_cvt_pk_bf16=cvt)
print('Correctness OK\n')

# ---- Benchmark (Kimi 2.5 TP8 shape) ----
kw = dict(
    tokens=333, model_dim=7168, inter_dim=256, experts=384, topk=8,
    in_dtype='int4_bf16', group_size=32, doweight_stage1=False,
    num_iters=20, num_warmup=5, skip_ref=True,
)

print('====== Stage1 (tile 16x64x256) ======')
print('--- preshuffle + bitmanip ---')
run_moe_stage1(**kw, tile_m=16, tile_n=64, tile_k=256, use_flat_layout=False, use_cvt_pk_bf16=False)
print('--- preshuffle + cvt_pk_bf16 ---')
run_moe_stage1(**kw, tile_m=16, tile_n=64, tile_k=256, use_flat_layout=False, use_cvt_pk_bf16=True)
print('--- flat + bitmanip ---')
run_moe_stage1(**kw, tile_m=16, tile_n=64, tile_k=256, use_flat_layout=True, use_cvt_pk_bf16=False)
print('--- flat + cvt_pk_bf16 ---')
run_moe_stage1(**kw, tile_m=16, tile_n=64, tile_k=256, use_flat_layout=True, use_cvt_pk_bf16=True)

a2 = torch.randn(333, 8, 256, device='cuda', dtype=torch.bfloat16) * 0.1

print('====== Stage2 (tile 16x128x256) ======')
print('--- preshuffle + bitmanip ---')
run_moe_stage2(**kw, tile_m=16, tile_n=128, tile_k=256, a2_fp8_in=a2, a2_scale_in=None, use_flat_layout=False, use_cvt_pk_bf16=False)
print('--- preshuffle + cvt_pk_bf16 ---')
run_moe_stage2(**kw, tile_m=16, tile_n=128, tile_k=256, a2_fp8_in=a2, a2_scale_in=None, use_flat_layout=False, use_cvt_pk_bf16=True)
print('--- flat + bitmanip ---')
run_moe_stage2(**kw, tile_m=16, tile_n=128, tile_k=256, a2_fp8_in=a2, a2_scale_in=None, use_flat_layout=True, use_cvt_pk_bf16=False)
print('--- flat + cvt_pk_bf16 ---')
run_moe_stage2(**kw, tile_m=16, tile_n=128, tile_k=256, a2_fp8_in=a2, a2_scale_in=None, use_flat_layout=True, use_cvt_pk_bf16=True)
"
