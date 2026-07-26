# SPDX-License-Identifier: Apache-2.0
"""Isolated apples-to-apples gemm2 kernel comparison: V1 (mixed_moe_gemm2) vs V2 (ported aiter
compile_gemm2_a4w4_port). Both run the SAME down-proj gemm2 (identical model_dim/inter_dim/experts/
tokens/BM, identical zeroed inputs), each with its own LOCAL-write epilog (no P2P scatter, no combine).
This isolates the gemm2 KERNEL so we can tell whether the V2-vs-V1 fused-stage2 gap comes from gemm2
or from the fused P2P scatter. Single-GPU (set HIP_VISIBLE_DEVICES to a healthy device)."""
from __future__ import annotations

import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.mega_moe.mega_moe_exp.group_gemm2 import compile_gemm2_a4w4_port
from kernels.moe.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

# v4_flash a8w4, per-rank (world=4 -> epr=64)
MODEL_DIM = 4096   # N_OUT (down-proj output)
INTER_DIM = 2048   # D_INTER (contraction)
EXPERTS = 64       # per-rank experts
TOPK = 8
BM = 32
TN, TK = 128, 256  # V1 tile_n / tile_k


def bench_cudagraph(cf, make_args, warmup=5, rep=100):
    """make_args(cuda_stream) -> arg tuple; kernel MUST launch on the passed stream (else the CUDAGraph
    capture is empty and replay times ~0)."""
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
    # per-kernel scale buffers (V1 / V2 use different swizzled layouts; size generously to avoid OOB)
    _sc = a2rows * (INTER_DIM // 32) * 4 + (1 << 20)
    ascale1 = torch.zeros(((a2rows + 255) // 256) * 256 * (((INTER_DIM // 32) + 7) // 8) * 8 * 4 + _sc,
                          dtype=torch.uint8, device=dev)
    ascale2 = torch.zeros(_sc, dtype=torch.uint8, device=dev)
    bq = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 2) + (1 << 16), dtype=torch.uint8, device=dev)
    bscale = torch.zeros(EXPERTS * (MODEL_DIM // 32) * (INTER_DIM // 256) * 64 * 4 + (1 << 16),
                         dtype=torch.uint8, device=dev)
    # spread tiles across experts (realistic B-weight cache traffic; all-expert-0 is unrealistically fast)
    eids = (torch.arange(max_blocks, device=dev, dtype=torch.int32) % EXPERTS)
    stids = (torch.arange(a2rows, device=dev, dtype=torch.int32) % tokens)     # token ids < tokens
    sweights = torch.ones(a2rows, dtype=torch.float32, device=dev)
    cumsum = torch.zeros(8, dtype=torch.int32, device=dev)
    # match V1's work: V2 computes cumsum[0]//BM m-tiles -> set to ceildiv(tokens,BM)*BM.
    cumsum[0] = ((tokens + BM - 1) // BM) * BM
    nv = torch.zeros(4, dtype=torch.int32, device=dev)
    nv[0] = tokens
    bias = torch.empty((0,), dtype=torch.float32, device=dev)

    # ---- V1: mixed_moe_gemm2 (token-accumulate local bf16 write) ----
    v1 = compile_mixed_moe_gemm2(
        model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK,
        tile_m=BM, tile_n=TN, tile_k=TK, doweight_stage2=True, a_dtype="fp8",
        b_dtype="fp4", out_dtype="bf16", accumulate=True, persist_m=-1, sort_block_m=BM,
    )
    out1 = torch.zeros(tokens, MODEL_DIM, dtype=torch.bfloat16, device=dev)

    def v1_make(st):
        return (out1, aq.view(-1), bq, ascale1, bscale, stids, eids, sweights, nv, bias,
                tokens, MODEL_DIM, INTER_DIM, int(max_blocks), st)

    cf1 = flyc.compile(v1, *v1_make(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    t1 = bench_cudagraph(cf1, v1_make)

    # ---- V2: ported aiter gemm2 (atomic local bf16 write; eids are LOCAL 0..EXPERTS-1) ----
    v2 = compile_gemm2_a4w4_port(
        BM=BM, a_dtype="fp8", epilog="atomic", HIDDEN_MAX=MODEL_DIM, INTER_MAX=INTER_DIM,
    )
    out2 = torch.zeros(tokens, MODEL_DIM, dtype=torch.bfloat16, device=dev)

    def v2_make(st):
        s = fx.Stream(st.cuda_stream)
        p = lambda t: fx.Int64(t.data_ptr())  # noqa: E731
        return (p(aq), p(ascale2), p(bq), p(bscale), p(eids), p(cumsum), p(stids), p(sweights),
                fx.Int32(tokens), fx.Int32(max_blocks), fx.Int32(max_blocks),
                fx.Int32(INTER_DIM), fx.Int32(MODEL_DIM), fx.Int32(0), fx.Int32(0),
                p(out2), p(out2), s)

    cf2 = flyc.compile(v2, *v2_make(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    t2 = bench_cudagraph(cf2, v2_make)

    print(f"[g2 V1-vs-V2] tokens={tokens:5d} a2rows={a2rows:6d}  "
          f"V1(mixed_moe_gemm2)={t1:.4f} ms  V2(aiter-port)={t2:.4f} ms  "
          f"ratio(V2/V1)={t2 / t1:.3f}", flush=True)


def main():
    toks = [int(x) for x in sys.argv[1:]] or [128, 256, 512, 1024, 2048]
    for t in toks:
        run(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
