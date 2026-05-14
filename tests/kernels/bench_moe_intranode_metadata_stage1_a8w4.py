#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Micro-benchmark for :func:`kernels.moe_peer_gather_mxfp8_a8w4.compile_peer_gather_mxfp8_rows`.

Example (single GPU)::

    python tests/kernels/bench_moe_intranode_metadata_stage1_a8w4.py --model-dim 7168 --max-recv 4096 --iters 50

On 8×MI355X use ``torchrun`` only when extending this script to multi-PE P2P bases.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLYDSL_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

import torch

import flydsl.expr as fx

from kernels.recv_meta_a8w4 import RECV_META_ROW_BYTES, pack_recv_meta_row_i32
from kernels.moe_peer_gather_mxfp8_a8w4 import compile_peer_gather_mxfp8_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dim", type=int, default=1024)
    p.add_argument("--max-recv", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    dev = torch.device("cuda:0")
    md = int(args.model_dim)
    max_recv = int(args.max_recv)
    gather = compile_peer_gather_mxfp8_rows(model_dim=md, max_recv_cap=max_recv)
    x = torch.randint(0, 255, (max_recv, md), dtype=torch.uint8, device=dev)
    recv = torch.zeros(max_recv * RECV_META_ROW_BYTES, dtype=torch.uint8, device=dev)
    for slot in range(max_recv):
        w = list(
            pack_recv_meta_row_i32(
                src_pe=0,
                kp=slot % 2,
                src_token=slot,
                expert_id=0,
                router_w=1.0,
                dest_encode=0,
                dst_slot=slot,
            )
        )
        row = torch.tensor(w, dtype=torch.int32, device=dev).view(torch.uint8)
        sl = slice(slot * RECV_META_ROW_BYTES, (slot + 1) * RECV_META_ROW_BYTES)
        recv[sl].copy_(row)
    p2p = torch.zeros(1, dtype=torch.int64, device=dev)
    p2p[0] = x.data_ptr()
    out = torch.empty_like(x)
    stream = fx.Stream(None)

    for _ in range(args.warmup):
        gather(
            fx.Int64(recv.data_ptr()),
            fx.Int64(p2p.data_ptr()),
            fx.Int64(out.data_ptr()),
            fx.Int32(max_recv),
            fx.Int32(md),
            fx.Int32(1),
            stream=stream,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        gather(
            fx.Int64(recv.data_ptr()),
            fx.Int64(p2p.data_ptr()),
            fx.Int64(out.data_ptr()),
            fx.Int32(max_recv),
            fx.Int32(md),
            fx.Int32(1),
            stream=stream,
        )
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / max(1, args.iters)
    bytes_moved = max_recv * md * 2  # read + write
    gbps = bytes_moved / dt / 1e9
    print(f"peer_gather: model_dim={md} max_recv={max_recv} iters={args.iters} "
          f"time_ms={dt*1000:.4f} eff_GB/s={gbps:.3f}")


if __name__ == "__main__":
    main()
