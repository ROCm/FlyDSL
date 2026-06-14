#!/usr/bin/env python3
"""Isolate the M>tile_m (multi M-block) a8w4 bug for config
(M,N,K,tm,tn,tk,mw,nw,nb)=(32,12288,3072,16,64,512,1,4,4).

Modes via env: PF_QUADRANT/PF_PIPELINE/PF_FULL_PREFETCH (set by caller).
Prints a per-M-block correctness summary and a per-operand isolation sweep.
"""
import os
import torch
import torch.nn.functional as F

from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
import flydsl.compiler as flyc
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (
    preshuffle_e8m0_scale, _get_padded_problem_shape, _pad_mxscale_inputs,
    reference_a8w4_gemm, random_fp8_data,
)
from tests.kernels.utils import fp4_utils

M, N, K = 32, 12288, 3072
TM, TN, TK = 16, 64, 512
MW, NW, NB = 1, 4, 4
PFD = 4
SCALE_BLOCK = 32

ps = _get_padded_problem_shape("a8w4", M, N, K, TM, TN, TK, 1)
pm, pn, pk = ps["M"], ps["N"], ps["K"]

fn = compile_mxscale_gemm(
    data_format="a8w4", N=pn, K=pk, tile_m=TM, tile_n=TN, tile_k=TK,
    m_warp=MW, n_warp=NW, num_buffers=NB, wave_specialized_tdm=True,
    l2_prefetch_distance=0, out_dtype="bf16", use_tdm_store=True,
    scale_load_path="tdm", pf_depth_wmma=PFD,
)


def run(a, b, asc, bsc):
    ref = reference_a8w4_gemm(a, b, asc, bsc, M, N, K)
    ap, bp, ascp, bscp = _pad_mxscale_inputs(a.clone(), b.clone(), asc.clone(), bsc.clone(), ps)
    skt = TK // SCALE_BLOCK
    ascp = preshuffle_e8m0_scale(ascp, TM // MW, scale_k_per_tile=skt)
    bscp = preshuffle_e8m0_scale(bscp, TN // NW, scale_k_per_tile=skt)
    bp = fp4_utils.preshuffle_b_16x16_tiled(bp, pn, pk // ps["pack_b"], TN, TK // ps["pack_b"], ksmajor=True)
    cg = torch.zeros(pm, pn, dtype=torch.bfloat16, device="cuda")
    flyc.compile(fn, cg.contiguous(), ap.cuda().contiguous(), bp.cuda().contiguous(),
                 ascp.cuda().contiguous(), bscp.cuda().contiguous(),
                 pm, pn, pk, pn, torch.cuda.current_stream())
    torch.cuda.synchronize()
    out = cg[:M, :N].float().cpu()
    return out, ref.float()


def cos(o, r):
    return F.cosine_similarity(o.flatten().double().unsqueeze(0), r.flatten().double().unsqueeze(0)).item()


# ---- full random: per-M-block cosine map ----
torch.manual_seed(0)
A = random_fp8_data(M, K)
B = fp4_utils.random_fp4_packed(N, K)
AS = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
BS = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)
out, ref = run(A, B, AS, BS)
print(f"ALL random: overall cos={cos(out, ref):.6f}")
for blk in range(M // TM):
    r0, r1 = blk * TM, (blk + 1) * TM
    print(f"  M-block {blk} (rows {r0}:{r1}): cos={cos(out[r0:r1], ref[r0:r1]):.6f}")

# ---- per-operand isolation: 3 const, 1 random ----
Ac = torch.full((M, K), 0x38, dtype=torch.uint8)            # ~1.0 fp8
Bc = torch.full((N, K // 2), 0x22, dtype=torch.uint8)
ASc = torch.full((M, K // SCALE_BLOCK), 127, dtype=torch.uint8)
BSc = torch.full((N, K // SCALE_BLOCK), 127, dtype=torch.uint8)
print("\ncomplement isolation (1 const + 3 random):")
for tag, a, b, asc, bsc in [
    ("A  const", Ac, B, AS, BS),
    ("B  const", A, Bc, AS, BS),
    ("AS const", A, B, ASc, BS),
    ("BS const", A, B, AS, BSc),
]:
    o, r = run(a, b, asc, bsc)
    per = " ".join(f"blk{blk}={cos(o[blk*TM:(blk+1)*TM], r[blk*TM:(blk+1)*TM]):.4f}" for blk in range(M // TM))
    print(f"  {tag}: overall={cos(o, r):.6f}  {per}")
