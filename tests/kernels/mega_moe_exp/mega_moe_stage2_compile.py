# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""COMPILE_ONLY gfx950 check for mega_moe_stage2 (group_gemm2 + cross-rank P2P-scatter combine)."""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.mega_moe.mega_moe_exp.mega_moe_stage2 import compile_mega_moe_stage2

MODEL_DIM = 2048
INTER_DIM = 1024
EXPERTS = 8
TOPK = 1
NPES = 8
MAX_TOK = 512
_MAX_BLOCKS = 4096 // 32 + 8
_MAX_RECV = NPES * MAX_TOK


def main():
    launch = compile_mega_moe_stage2(
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        experts=EXPERTS,
        topk=TOPK,
        rank=0,
        npes=NPES,
        max_tok=MAX_TOK,
        a_dtype="fp8",
        grid_mult=1,
    )
    dev = torch.device("cuda", 0)
    a2rows = _MAX_BLOCKS * 32
    aq = torch.zeros(a2rows, INTER_DIM, dtype=torch.float8_e4m3fn, device=dev)
    ascale = torch.zeros(a2rows * (INTER_DIM // 32), dtype=torch.uint8, device=dev)
    bq = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 2), dtype=torch.uint8, device=dev)
    bscale = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 32), dtype=torch.uint8, device=dev)
    eids = torch.zeros(_MAX_BLOCKS, dtype=torch.int32, device=dev)
    stids = torch.zeros(a2rows, dtype=torch.int32, device=dev)
    sweights = torch.zeros(a2rows, dtype=torch.float32, device=dev)
    trb = torch.zeros(_MAX_BLOCKS, dtype=torch.int32, device=dev)
    nv = torch.zeros(4, dtype=torch.int32, device=dev)
    tis = torch.zeros(_MAX_RECV, dtype=torch.int32, device=dev)
    p2p = torch.zeros(NPES, dtype=torch.int64, device=dev)
    s = fx.Stream(torch.cuda.current_stream().cuda_stream)
    args = (
        fx.Int64(aq.data_ptr()),
        fx.Int64(ascale.data_ptr()),
        fx.Int64(bq.data_ptr()),
        fx.Int64(bscale.data_ptr()),
        fx.Int64(eids.data_ptr()),
        fx.Int64(stids.data_ptr()),
        fx.Int64(sweights.data_ptr()),
        fx.Int64(trb.data_ptr()),
        fx.Int64(nv.data_ptr()),
        fx.Int64(tis.data_ptr()),
        fx.Int64(p2p.data_ptr()),
        fx.Int32(_MAX_BLOCKS),
        s,
    )
    flyc.compile(launch, *args)
    print("[OK] mega_moe_stage2 compiled (arch=gfx950)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
