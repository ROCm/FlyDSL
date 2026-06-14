#!/usr/bin/env python3
"""Dump ISA for the tail-only sub-depth NaN investigation.

Three cases (a8w4, tile 32/32, m_warp=2 n_warp=2, nb=2, l2pf=0):
  fail_loopiters0_depth1  : K=512  depth=1     -> NaN  (tail-only, sub-depth)
  pass_loopiters0_full    : K=512  depth=None  -> PASS (tail-only, full-depth)
  pass_mainloop_depth1    : K=1024 depth=1     -> PASS (main loop, sub-depth)
"""
import os
import shutil

os.environ["FLYDSL_DUMP_IR"] = "1"

import torch
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (
    random_fp8_data, preshuffle_e8m0_scale, _get_padded_problem_shape,
    _pad_mxscale_inputs,
)
from tests.kernels.utils import fp4_utils
import flydsl.compiler as flyc

SCALE_BLOCK = 32
DATA_FORMAT = "a8w4"
TILE_M, TILE_N, TILE_K = 32, 32, 256
M_WARP, N_WARP = 2, 2
NUM_BUFFERS = 2
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stagec_isa")


def do_compile(M, N, K, pf_depth_wmma, label):
    dump_dir = os.path.join(OUTDIR, "_raw_" + label)
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.environ["FLYDSL_DUMP_DIR"] = dump_dir

    ps = _get_padded_problem_shape(DATA_FORMAT, M, N, K, TILE_M, TILE_N, TILE_K, 1)
    pm, pn, pk = ps["M"], ps["N"], ps["K"]

    torch.manual_seed(0)
    a = random_fp8_data(M, K)
    b = fp4_utils.random_fp4_packed(N, K)
    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)
    a, b, a_scale, b_scale = _pad_mxscale_inputs(a, b, a_scale, b_scale, ps)
    skt = TILE_K // SCALE_BLOCK
    a_scale = preshuffle_e8m0_scale(a_scale, TILE_M // M_WARP, scale_k_per_tile=skt)
    b_scale = preshuffle_e8m0_scale(b_scale, TILE_N // N_WARP, scale_k_per_tile=skt)
    K_packed = pk // ps["pack_b"]
    b = fp4_utils.preshuffle_b_16x16_tiled(b, pn, K_packed, TILE_N, TILE_K // ps["pack_b"], ksmajor=True)

    a_gpu, b_gpu = a.cuda(), b.cuda()
    as_gpu, bs_gpu = a_scale.cuda(), b_scale.cuda()
    c_gpu = torch.zeros(pm, pn, dtype=torch.bfloat16, device="cuda")

    num_k_tiles = K // TILE_K
    loop_iters = (num_k_tiles - (NUM_BUFFERS - 1)) // NUM_BUFFERS
    print(f"\n=== {label}: M={M} N={N} K={K} depth={pf_depth_wmma} "
          f"num_k_tiles={num_k_tiles} loop_iters={loop_iters} ===")

    launch_fn = compile_mxscale_gemm(
        data_format=DATA_FORMAT, N=pn, K=pk,
        tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K,
        m_warp=M_WARP, n_warp=N_WARP, num_buffers=NUM_BUFFERS,
        out_dtype="bf16", wave_specialized_tdm=True,
        l2_prefetch_distance=0, pf_depth_wmma=pf_depth_wmma,
    )
    flyc.compile(launch_fn, c_gpu, a_gpu, b_gpu, as_gpu, bs_gpu,
                 pm, pn, pk, pn, torch.cuda.current_stream())

    # Copy the final ISA to a clean named file in OUTDIR.
    src = None
    for root, _, files in os.walk(dump_dir):
        for f in files:
            if f.endswith("_final_isa.s"):
                src = os.path.join(root, f)
    if src:
        dst = os.path.join(OUTDIR, f"{label}.s")
        shutil.copy(src, dst)
        print(f"[ok] {label}.s")


os.makedirs(OUTDIR, exist_ok=True)
do_compile(32, 32, 512, 1, "fail_loopiters0_depth1")
do_compile(32, 32, 512, None, "pass_loopiters0_full")
do_compile(32, 32, 1024, 1, "pass_mainloop_depth1")

print("\n\nISA files in", OUTDIR)
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith(".s"):
        print("  ", os.path.join(OUTDIR, f))
