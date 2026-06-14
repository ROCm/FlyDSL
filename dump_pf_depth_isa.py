#!/usr/bin/env python3
"""Dump ISA: real sub-depth (pf_depth_wmma=1) vs full prefetch (None).

Config chosen so depth=1 is LEGAL and takes the real sub-depth path (no fallback):
  a8w4, tile_m=tile_n=32 (warp 2x2 -> wmma_m_rep=wmma_n_rep=1 -> WMMAs-per-ks=1),
  tile_k=256 (k_wmma_steps=2 -> total WMMAs per tile A=2),
  K=512 -> num_k_tiles=2 == num_buffers=2 -> no buffer reuse, no fallback.
  l2_prefetch_distance=0 to skip the (undefined) _l2_prefetch_b_at branch.
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
M, N, K = 32, 32, 512
TILE_M, TILE_N, TILE_K = 32, 32, 256
M_WARP, N_WARP = 2, 2          # 4 waves (required by wave_specialized_tdm)
NUM_BUFFERS = 2


def do_compile(pf_depth_wmma, label):
    dump_dir = f"/tmp/gemm_isa_{label}"
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
    b = fp4_utils.preshuffle_b_16x16_tiled(
        b, pn, K_packed, TILE_N, TILE_K // ps["pack_b"], ksmajor=True
    )

    a_gpu, b_gpu = a.cuda(), b.cuda()
    as_gpu, bs_gpu = a_scale.cuda(), b_scale.cuda()
    c_gpu = torch.zeros(pm, pn, dtype=torch.bfloat16, device="cuda")

    print(f"\n{'='*64}")
    print(f"  pf_depth_wmma={pf_depth_wmma}  ({label})")
    print(f"  M={M} N={N} K={K}  tile=({TILE_M},{TILE_N},{TILE_K})  "
          f"warp=({M_WARP},{N_WARP})  bufs={NUM_BUFFERS}")
    print(f"  num_k_tiles={K // TILE_K}  _pf_wpks=1  _A_wmma=2")
    print(f"  dump -> {dump_dir}")
    print(f"{'='*64}")

    launch_fn = compile_mxscale_gemm(
        data_format=DATA_FORMAT, N=pn, K=pk,
        tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K,
        m_warp=M_WARP, n_warp=N_WARP, num_buffers=NUM_BUFFERS,
        out_dtype="bf16",
        wave_specialized_tdm=True,
        l2_prefetch_distance=0,        # skip undefined _l2_prefetch_b_at branch
        pf_depth_wmma=pf_depth_wmma,
    )
    flyc.compile(
        launch_fn, c_gpu, a_gpu, b_gpu, as_gpu, bs_gpu,
        pm, pn, pk, pn, torch.cuda.current_stream(),
    )
    print(f"[ok] compiled {label}")


do_compile(1, "depth1")      # real sub-depth
do_compile(None, "full")     # full prefetch

print("\n\n=== ISA (.s) files ===")
for label in ("depth1", "full"):
    for root, _, files in os.walk(f"/tmp/gemm_isa_{label}"):
        for f in sorted(files):
            if f.endswith(".s"):
                print(f"  {os.path.join(root, f)}")
