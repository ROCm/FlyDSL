# SPDX-License-Identifier: Apache-2.0
"""Isolated fused-stage2 (gemm2 + combine) comparison: v1 vs v2, single-GPU, CUDAGraph-timed.

  * v1 = base MegaMoE stage2 = compile_mixed_moe_gemm2 (ONE kernel: gemm2 + weighted token-accumulate
         combine into out[token_id]).
  * v2 = exp MegaMoEV2 stage2 = compile_mega_moe_stage2 (ported aiter gemm2 + weighted P2P scatter into
         peer combine buffers; here the npes peer table points at ONE local buffer = self-scatter, so
         the compute/LDS/store path matches production, only remote xGMI latency is understated).

Both run the SAME down-proj (model_dim/inter_dim/experts/tokens/BM) with matched tile work
(cumsum/num_valid bound to `tokens`). Set HIP_VISIBLE_DEVICES to a healthy device."""
from __future__ import annotations

import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.mega_moe.mega_moe_exp.mega_moe_stage2 import compile_mega_moe_stage2
from kernels.moe.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

MODEL_DIM = 4096
INTER_DIM = 2048
EXPERTS = 64
TOPK = 8
NPES = 4
BM = 32
MAX_TOK = 512
RECV_CAP = 4096


def bench_cudagraph(cf, make_args, warmup=8, rep=100):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            cf(*make_args(side))
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cf(*make_args(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


def run(tokens):
    dev = torch.device("cuda", 0)
    max_blocks = (tokens + EXPERTS * BM + BM - 1) // BM
    a2rows = max_blocks * BM
    aq = torch.zeros(a2rows, INTER_DIM, dtype=torch.float8_e4m3fn, device=dev)
    _sc = a2rows * (INTER_DIM // 32) * 4 + (1 << 20)
    ascale1 = torch.zeros(((a2rows + 255) // 256) * 256 * (((INTER_DIM // 32) + 7) // 8) * 8 * 4 + _sc,
                          dtype=torch.uint8, device=dev)
    ascale2 = torch.zeros(_sc, dtype=torch.uint8, device=dev)
    bq = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 2) + (1 << 16), dtype=torch.uint8, device=dev)
    bscale = torch.zeros(EXPERTS * (MODEL_DIM // 32) * (INTER_DIM // 256) * 64 * 4 + (1 << 16),
                         dtype=torch.uint8, device=dev)
    eids = (torch.arange(max_blocks, device=dev, dtype=torch.int32) % EXPERTS)
    sweights = torch.ones(a2rows, dtype=torch.float32, device=dev)
    nv = torch.zeros(4, dtype=torch.int32, device=dev); nv[0] = tokens

    # ---- V1: mixed_moe_gemm2 (gemm2 + weighted token-accumulate combine) ----
    stids1 = (torch.arange(a2rows, device=dev, dtype=torch.int32) % tokens)
    v1 = compile_mixed_moe_gemm2(
        model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK,
        tile_m=BM, tile_n=128, tile_k=256, doweight_stage2=True, a_dtype="fp8",
        b_dtype="fp4", out_dtype="bf16", accumulate=True, persist_m=-1, sort_block_m=BM,
    )
    out1 = torch.zeros(tokens, MODEL_DIM, dtype=torch.bfloat16, device=dev)
    bias = torch.empty((0,), dtype=torch.float32, device=dev)

    def v1_make(st):
        return (out1, aq.view(-1), bq, ascale1, bscale, stids1, eids, sweights, nv, bias,
                tokens, MODEL_DIM, INTER_DIM, int(max_blocks), st)

    cf1 = flyc.compile(v1, *v1_make(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    t1 = bench_cudagraph(cf1, v1_make)

    # ---- V2: compile_mega_moe_stage2 (ported aiter gemm2 + P2P scatter combine) ----
    comb_inp_nbytes = MAX_TOK * TOPK * MODEL_DIM * 2
    v2 = compile_mega_moe_stage2(
        model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK, rank=0, npes=NPES,
        max_tok=MAX_TOK, recv_cap=RECV_CAP, comb_inp_nbytes=comb_inp_nbytes, BM=BM,
        HIDDEN_MAX=MODEL_DIM, INTER_MAX=INTER_DIM, a_dtype="fp8", persist=False, cu_num=304,
    )
    _idx = torch.arange(a2rows, device=dev, dtype=torch.int32)
    stids2 = ((_idx % TOPK) << 24) | (_idx % RECV_CAP)
    trb = (torch.arange(max_blocks, device=dev, dtype=torch.int32) * BM)
    cumsum = torch.zeros(8, dtype=torch.int32, device=dev)
    cumsum[0] = ((tokens + BM - 1) // BM) * BM
    _t = torch.arange(RECV_CAP, device=dev, dtype=torch.int32)
    tis = (_t % NPES) * MAX_TOK + ((_t // NPES) % MAX_TOK)
    comb_inp = torch.zeros(comb_inp_nbytes // 2, dtype=torch.bfloat16, device=dev)
    p2p_tbl = torch.full((NPES,), comb_inp.data_ptr(), dtype=torch.int64, device=dev)

    def v2_make(st):
        s = fx.Stream(st.cuda_stream)
        p = lambda t: fx.Int64(t.data_ptr())  # noqa: E731
        return (p(aq), p(ascale2), p(bq), p(bscale), p(eids), p(cumsum), p(stids2), p(sweights),
                p(trb), p(tis), p(p2p_tbl), fx.Int32(max_blocks), fx.Int32(max_blocks),
                fx.Int32(INTER_DIM), fx.Int32(MODEL_DIM), fx.Int32(0), fx.Int32(0), s)

    cf2 = flyc.compile(v2, *v2_make(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    t2 = bench_cudagraph(cf2, v2_make)

    print(f"[stage2 gemm2+combine] tokens={tokens:5d} a2rows={a2rows:6d}  "
          f"v1(mixed_moe_gemm2)={t1:.4f} ms  v2(aiter+p2p)={t2:.4f} ms  ratio(v2/v1)={t2 / t1:.3f}",
          flush=True)


def main():
    toks = [int(x) for x in sys.argv[1:]] or [128, 512, 2048]
    for t in toks:
        run(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
