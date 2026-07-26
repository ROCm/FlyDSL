# SPDX-License-Identifier: Apache-2.0
"""Compile smoke test for the ported aiter gemm2 + fused P2P scatter stage2 (no multi-GPU needed).
Forces trace + LLVM codegen of (a) standalone gemm2 atomic (compile_gemm2_a4w4_port) and (b) the
fused stage2 (compile_mega_moe_stage2). Run single-GPU: HIP_VISIBLE_DEVICES=4 python ... """
from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.mega_moe.mega_moe_exp.group_gemm2 import compile_gemm2_a4w4_port
from kernels.mega_moe.mega_moe_exp.mega_moe_stage2 import compile_mega_moe_stage2

MODEL_DIM = 4096
INTER_DIM = 2048
EXPERTS = 64
TOPK = 8
NPES = 4
MAX_TOK = 1024
BM = 32


def _p(t):
    return fx.Int64(t.data_ptr())


def main():
    dev = torch.device("cuda", 0)
    st = fx.Stream(torch.cuda.current_stream().cuda_stream)
    d = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)  # scratch device buffer for all ptrs
    cs = torch.zeros(8, dtype=torch.int32, device=dev)

    # (a) standalone gemm2 atomic
    g2 = compile_gemm2_a4w4_port(
        BM=BM, a_dtype="fp8", epilog="atomic", HIDDEN_MAX=MODEL_DIM, INTER_MAX=INTER_DIM,
    )
    a = (_p(d), _p(d), _p(d), _p(d), _p(d), _p(cs), _p(d), _p(d),
         fx.Int32(64), fx.Int32(16), fx.Int32(16), fx.Int32(INTER_DIM), fx.Int32(MODEL_DIM),
         fx.Int32(0), fx.Int32(0), _p(d), _p(d), st)
    flyc.compile(g2, *a)
    torch.cuda.synchronize()
    print("[smoke] compile_gemm2_a4w4_port(atomic) OK", flush=True)

    # (b) fused stage2 (P2P scatter)
    s2 = compile_mega_moe_stage2(
        model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK, rank=0, npes=NPES,
        max_tok=MAX_TOK, recv_cap=NPES * 64, BM=BM, HIDDEN_MAX=MODEL_DIM, INTER_MAX=INTER_DIM,
        persist=False, cu_num=304,
    )
    b = (_p(d), _p(d), _p(d), _p(d), _p(d), _p(cs), _p(d), _p(d), _p(d), _p(d), _p(d),
         fx.Int32(16), fx.Int32(16), fx.Int32(INTER_DIM), fx.Int32(MODEL_DIM), fx.Int32(0), fx.Int32(0), st)
    flyc.compile(s2, *b)
    torch.cuda.synchronize()
    print("[smoke] compile_mega_moe_stage2(scatter_p2p, persist=False) OK", flush=True)

    # (c) fused stage2 persist=True + spart on (default 402)
    s2p = compile_mega_moe_stage2(
        model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, topk=TOPK, rank=0, npes=NPES,
        max_tok=MAX_TOK, recv_cap=NPES * 64, BM=BM, HIDDEN_MAX=MODEL_DIM, INTER_MAX=INTER_DIM,
        persist=True, cu_num=304,
    )
    flyc.compile(s2p, *b)
    torch.cuda.synchronize()
    print("[smoke] compile_mega_moe_stage2(persist=True) OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
