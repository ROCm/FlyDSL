#!/usr/bin/env python3
"""A8W4 GEMM runner for ATT tracing. Runs twice: warmup + traced dispatch."""
import sys, os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (
    SCALE_BLOCK, _get_padded_problem_shape, _pad_mxscale_inputs,
    fp4_utils, preshuffle_e8m0_scale, random_fp8_data,
)

M, N, K = 64, 12288, 3072
tm, tn, tk, mw, nw, nb = 16, 64, 512, 1, 4, 4

torch.manual_seed(0)
a = random_fp8_data(M, K)
b = fp4_utils.random_fp4_packed(N, K)
a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)

padded = _get_padded_problem_shape("a8w4", M, N, K, tm, tn, tk, 1)
a_p, b_p, as_p, bs_p = _pad_mxscale_inputs(a, b, a_scale, b_scale, padded)

skt = tk // SCALE_BLOCK
a_s = preshuffle_e8m0_scale(as_p.clone(), tm // mw, scale_k_per_tile=skt, coalesced=False)
b_s = preshuffle_e8m0_scale(bs_p.clone(), tn // nw, scale_k_per_tile=skt, coalesced=False)
K_packed = padded["K"] // padded["pack_b"]
b_shuf = fp4_utils.preshuffle_b_16x16(b_p.clone(), padded["N"], K_packed)

a_gpu = a_p.cuda().contiguous()
b_gpu = b_shuf.cuda().contiguous()
as_gpu = a_s.cuda().contiguous()
bs_gpu = b_s.cuda().contiguous()

launch_fn = compile_mxscale_gemm(
    data_format="a8w4", N=padded["N"], K=padded["K"],
    tile_m=tm, tile_n=tn, tile_k=tk, m_warp=mw, n_warp=nw, num_buffers=nb,
    use_tdm_store=True, out_dtype="bf16", wave_specialized_tdm=True, l2_prefetch_distance=2,
)

def run_once():
    c_gpu = torch.zeros(padded["M"], padded["N"], dtype=torch.bfloat16, device="cuda")
    flyc.compile(
        launch_fn, c_gpu.contiguous(), a_gpu, b_gpu, as_gpu, bs_gpu,
        padded["M"], padded["N"], padded["K"], padded["N"],
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    return c_gpu

# Round 1: warmup (JIT compile + first dispatch)
print("Warmup...", flush=True)
run_once()

# Round 2: traced dispatch
print("Traced run...", flush=True)
run_once()

print("Done.")
