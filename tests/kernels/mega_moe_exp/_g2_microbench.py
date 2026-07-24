# SPDX-License-Identifier: Apache-2.0
"""Isolated stage-2 gemm2 microbench (compile_group_gemm2 standalone atomic epilog).

Times ONLY the down-proj gemm2 kernel (no stage1, no combine), via flydsl do_bench. Used to compare
the aiter-port refactor vs the pre-refactor group_gemm2 at identical shapes. BM=32 for a fair
before/after (pre-refactor is fixed BM=32)."""

from __future__ import annotations

import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.mega_moe.mega_moe_exp.group_gemm2 import compile_group_gemm2


def bench_cudagraph(cf, make_args, warmup=5, rep=100):
    """CUDAGraph capture + replay timing -> pure stage2 GPU kernel time (no per-launch dispatch)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            cf(*make_args(fx.Stream(side.cuda_stream)))
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cf(*make_args(fx.Stream(torch.cuda.current_stream().cuda_stream)))
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep

MODEL_DIM = 7168  # N_OUT
INTER_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3072  # D_INTER (contraction)
EXPERTS = 48  # per-rank (v4_pro ep8)
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
BM = 32


def main():
    launch = compile_group_gemm2(model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, a_dtype="fp8", grid_mult=1)
    dev = torch.device("cuda", 0)
    max_blocks = (TOKENS + EXPERTS * BM + BM - 1) // BM
    a2rows = max_blocks * BM
    out = torch.zeros(TOKENS, MODEL_DIM, dtype=torch.bfloat16, device=dev)
    aq = torch.zeros(a2rows, INTER_DIM, dtype=torch.float8_e4m3fn, device=dev)
    ascale = torch.zeros(a2rows * (INTER_DIM // 32), dtype=torch.uint8, device=dev)
    bq = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 2), dtype=torch.uint8, device=dev)
    bscale = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 32), dtype=torch.uint8, device=dev)
    eids = torch.zeros(max_blocks, dtype=torch.int32, device=dev)  # all expert 0 (valid)
    stids = (torch.arange(a2rows, device=dev, dtype=torch.int32) % TOKENS)  # token ids < TOKENS
    sweights = torch.ones(a2rows, dtype=torch.float32, device=dev)
    nv = torch.zeros(4, dtype=torch.int32, device=dev)
    nv[0] = TOKENS  # num_valid -> total_work = ceil(TOKENS/BM) * num_n_blocks tiles

    def make_args(stream_fx):
        return (
            fx.Int64(out.data_ptr()),
            fx.Int64(aq.data_ptr()),
            fx.Int64(ascale.data_ptr()),
            fx.Int64(bq.data_ptr()),
            fx.Int64(bscale.data_ptr()),
            fx.Int64(eids.data_ptr()),
            fx.Int64(stids.data_ptr()),
            fx.Int64(sweights.data_ptr()),
            fx.Int64(nv.data_ptr()),
            fx.Int32(TOKENS),
            fx.Int32(max_blocks),
            stream_fx,
        )

    cf = flyc.compile(launch, *make_args(fx.Stream(torch.cuda.current_stream().cuda_stream)))
    torch.cuda.synchronize()
    t = bench_cudagraph(cf, make_args)
    print(f"[g2-microbench-cudagraph] model_dim={MODEL_DIM} inter_dim={INTER_DIM} experts={EXPERTS} "
          f"tokens={TOKENS} BM={BM} -> {t:.4f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
