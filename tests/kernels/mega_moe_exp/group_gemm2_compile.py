# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""COMPILE_ONLY gfx950 check for group_gemm2 (aiter-v2 down-proj + local atomic combine)."""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.mega_moe.mega_moe_exp.group_gemm2 import compile_group_gemm2

MODEL_DIM = 2048  # N_OUT (down-proj output)
INTER_DIM = 1024  # D_INTER (contraction)
EXPERTS = 8
_MAX_BLOCKS = 4096 // 32 + 8
_TOKENS = 4096


def main():
    launch = compile_group_gemm2(model_dim=MODEL_DIM, inter_dim=INTER_DIM, experts=EXPERTS, grid_mult=1)
    dev = torch.device("cuda", 0)
    a2rows = _MAX_BLOCKS * 32
    out = torch.zeros(_TOKENS, MODEL_DIM, dtype=torch.bfloat16, device=dev)
    aq = torch.zeros(a2rows, INTER_DIM, dtype=torch.float8_e4m3fn, device=dev)
    ascale = torch.zeros(a2rows * (INTER_DIM // 32), dtype=torch.uint8, device=dev)
    bq = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 2), dtype=torch.uint8, device=dev)
    bscale = torch.zeros(EXPERTS * MODEL_DIM * (INTER_DIM // 32), dtype=torch.uint8, device=dev)
    eids = torch.zeros(_MAX_BLOCKS, dtype=torch.int32, device=dev)
    stids = torch.zeros(a2rows, dtype=torch.int32, device=dev)
    sweights = torch.zeros(a2rows, dtype=torch.float32, device=dev)
    nv = torch.zeros(4, dtype=torch.int32, device=dev)
    s = fx.Stream(torch.cuda.current_stream().cuda_stream)
    args = (
        fx.Int64(out.data_ptr()),
        fx.Int64(aq.data_ptr()),
        fx.Int64(ascale.data_ptr()),
        fx.Int64(bq.data_ptr()),
        fx.Int64(bscale.data_ptr()),
        fx.Int64(eids.data_ptr()),
        fx.Int64(stids.data_ptr()),
        fx.Int64(sweights.data_ptr()),
        fx.Int64(nv.data_ptr()),
        fx.Int32(_TOKENS),
        fx.Int32(_MAX_BLOCKS),
        s,
    )
    flyc.compile(launch, *args)
    print("[OK] group_gemm2 compiled (arch=gfx950)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
