# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""COMPILE_ONLY gfx950 check for mega_moe_stage1 (split dispatch producers + group_gemm1 consumers)."""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.mega_moe.mega_moe_exp.mega_moe_stage1 import compile_mega_moe_stage1

MODEL_DIM = 2048
INTER_DIM = 1024
NPES = 8
EPR = 8
EXPERTS = NPES * EPR
FUSE_TOPK = 1
TILE_M, TILE_N, TILE_K = 32, 256, 256
SBM = 32
FUSE_CAP = 512
FUSE_MTPR = 512
GRID_MULT = 8
_SCALE_DIM = MODEL_DIM // 32
_NVM = 4096
_MAX_BLOCKS = (_NVM + SBM - 1) // SBM + 8


def _alloc():
    dev = torch.device("cuda", 0)
    a2rows = _MAX_BLOCKS * SBM
    return dict(
        out=torch.zeros(a2rows, INTER_DIM, dtype=torch.float8_e4m3fn, device=dev),
        x=torch.zeros(_NVM, MODEL_DIM, dtype=torch.float8_e4m3fn, device=dev),
        w=torch.zeros(EPR * (2 * INTER_DIM) * (MODEL_DIM // 2), dtype=torch.uint8, device=dev),
        scale_x=torch.zeros(_NVM * (MODEL_DIM // 32), dtype=torch.uint8, device=dev),
        scale_w=torch.zeros(EPR * (2 * INTER_DIM) * (MODEL_DIM // 32), dtype=torch.uint8, device=dev),
        trb=torch.zeros(_MAX_BLOCKS, dtype=torch.int32, device=dev),
        se=torch.zeros(_MAX_BLOCKS, dtype=torch.int32, device=dev),
        nv=torch.zeros(4, dtype=torch.int32, device=dev),
        out_scale=torch.zeros(
            ((a2rows + 255) // 256) * 256 * ((((INTER_DIM // 32) + 7) // 8) * 8) + INTER_DIM,
            dtype=torch.uint8,
            device=dev,
        ),
    )


def main():
    launch = compile_mega_moe_stage1(
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        experts=EXPERTS,
        topk=FUSE_TOPK,
        rank=0,
        experts_per_rank=EPR,
        fuse_npes=NPES,
        fuse_topk=FUSE_TOPK,
        fuse_cap=FUSE_CAP,
        fuse_mtpr=FUSE_MTPR,
        fuse_scale_dim=_SCALE_DIM,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        sort_block_m=SBM,
        num_waves=4,
        grid_mult=GRID_MULT,
        wgm=1,
        mfma_amajor=True,
        swizzle_a=True,
        a_dtype="fp8",
        out_dtype="fp8",
        num_dispatch_cu=32,
    )
    t = _alloc()
    dev = torch.device("cuda", 0)
    s = fx.Stream(torch.cuda.current_stream().cuda_stream)
    _disp = torch.zeros(64, dtype=torch.int64, device=dev)
    _in_tok = torch.zeros(FUSE_MTPR * MODEL_DIM, dtype=torch.uint8, device=dev)
    _in_idx = torch.zeros(FUSE_MTPR * max(1, FUSE_TOPK), dtype=torch.int32, device=dev)
    _in_wts = torch.zeros(FUSE_MTPR * max(1, FUSE_TOPK), dtype=torch.float32, device=dev)
    _in_sc = torch.zeros(FUSE_MTPR * (MODEL_DIM // 32), dtype=torch.uint8, device=dev)
    _ready = torch.zeros(1, dtype=torch.int32, device=dev)
    args = (
        t["out"],
        t["x"],
        t["w"],
        t["scale_x"],
        t["scale_w"],
        t["trb"],
        t["se"],
        t["nv"],
        t["out_scale"],
        fx.Int32(_NVM),
        fx.Int32(INTER_DIM * 2),
        fx.Int32(MODEL_DIM),
        fx.Int32(_MAX_BLOCKS),
        fx.Int64(_disp.data_ptr()),
        fx.Int32(1),
        fx.Int64(_in_tok.data_ptr()),
        fx.Int64(_in_idx.data_ptr()),
        fx.Int64(_in_wts.data_ptr()),
        fx.Int64(_in_sc.data_ptr()),
        fx.Int64(_ready.data_ptr()),
        s,
    )
    flyc.compile(launch, *args)
    print("[OK] mega_moe_stage1 compiled (arch=gfx950)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
