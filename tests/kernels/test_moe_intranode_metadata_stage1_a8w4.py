# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for intranode metadata dispatch + MX-FP8 peer gather (A8W4 staging path).

Smoke tests run on a single GPU.  Full shmem cross-PE checks require ``torchrun`` + mori::

    torchrun --standalone --nproc_per_node=2 \\
        tests/kernels/test_moe_intranode_metadata_stage1_a8w4.py

On MI355X 8-card machines, use ``tests/kernels/test_moe_intranode_metadata_multigpu.py``
(``pytest -m multi_gpu`` or ``torchrun --nproc_per_node=8`` on that file) for NCCL + mori,
real ``recv_meta`` / TIS / ``total_recv`` checks across all ranks.
"""

from __future__ import annotations

import os
import struct
import sys

import pytest
import torch

pytest.importorskip("flydsl._mlir", reason="FlyDSL MLIR stack required")

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLYDSL_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

from kernels.recv_meta_a8w4 import (
    RECV_META_ROW_BYTES,
    pack_recv_meta_row_i32,
    sort_recv_meta_rows_by_src_pe,
    unpack_recv_meta_row_i32,
)
from kernels.moe_peer_gather_mxfp8_a8w4 import compile_peer_gather_mxfp8_rows


def test_recv_meta_pack_roundtrip():
    row = pack_recv_meta_row_i32(
        src_pe=3,
        kp=1,
        src_token=7,
        expert_id=42,
        router_w=0.25,
        dest_encode=12345,
        dst_slot=99,
    )
    d = unpack_recv_meta_row_i32(row)
    assert d["src_pe"] == 3
    assert d["kp"] == 1
    assert d["src_token"] == 7
    assert d["expert_id"] == 42
    assert abs(d["router_w"] - 0.25) < 1e-6
    assert d["dest_encode"] == 12345
    assert d["dst_slot"] == 99


@pytest.mark.parametrize(
    "src_pe,kp,src_token,expert_id,router_w,dest_encode,dst_slot",
    [
        (0, 0, 0, 0, 0.0, 0, 0),
        (7, 3, 12345, 99, 1.25, -1, 1023),
        (255, 255, 0x00FEDC, 0x7FFFFFFF, -0.0, 0x80000001, 0xFFFFFFFF),
    ],
)
def test_recv_meta_pack_roundtrip_parametrized(
    src_pe, kp, src_token, expert_id, router_w, dest_encode, dst_slot
):
    row = pack_recv_meta_row_i32(
        src_pe=src_pe,
        kp=kp,
        src_token=src_token,
        expert_id=expert_id,
        router_w=router_w,
        dest_encode=dest_encode,
        dst_slot=dst_slot,
    )
    d = unpack_recv_meta_row_i32(row)
    assert d["src_pe"] == (src_pe & 0xFF)
    assert d["kp"] == (kp & 0xFF)
    assert d["src_token"] == (src_token & 0xFFFFFFFF)
    assert d["expert_id"] == expert_id
    assert abs(d["router_w"] - float(router_w)) < 1e-6
    assert (d["dest_encode"] & 0xFFFFFFFF) == (dest_encode & 0xFFFFFFFF)
    assert d["dst_slot"] == (dst_slot & 0xFFFFFFFF)


def test_sort_recv_meta_single_row_noop():
    buf = bytearray(RECV_META_ROW_BYTES)
    w = list(
        pack_recv_meta_row_i32(
            src_pe=5,
            kp=1,
            src_token=9,
            expert_id=1,
            router_w=2.0,
            dest_encode=3,
            dst_slot=0,
        )
    )
    buf[:] = struct.pack("<8I", *w)
    before = bytes(buf)
    sort_recv_meta_rows_by_src_pe(buf, total_recv=1)
    assert bytes(buf) == before


def test_sort_recv_meta_preserves_row_multiset():
    n = 5
    buf = bytearray(n * RECV_META_ROW_BYTES)
    specs = [(i % 3, i * 11, i) for i in range(n)]
    for slot, (pe, tok, dst) in enumerate(specs):
        w = list(
            pack_recv_meta_row_i32(
                src_pe=pe,
                kp=slot & 1,
                src_token=tok,
                expert_id=slot,
                router_w=float(slot) * 0.1,
                dest_encode=slot,
                dst_slot=dst,
            )
        )
        off = slot * RECV_META_ROW_BYTES
        buf[off : off + RECV_META_ROW_BYTES] = struct.pack("<8I", *w)
    multiset_before = sorted(
        buf[i * RECV_META_ROW_BYTES : (i + 1) * RECV_META_ROW_BYTES].hex()
        for i in range(n)
    )
    sort_recv_meta_rows_by_src_pe(buf, total_recv=n)
    multiset_after = sorted(
        buf[i * RECV_META_ROW_BYTES : (i + 1) * RECV_META_ROW_BYTES].hex()
        for i in range(n)
    )
    assert multiset_before == multiset_after
    keys = []
    for i in range(n):
        off = i * RECV_META_ROW_BYTES
        i32s = struct.unpack("<8I", buf[off : off + RECV_META_ROW_BYTES])
        u = unpack_recv_meta_row_i32(i32s)
        keys.append((u["src_pe"], u["src_token"]))
    assert keys == sorted(keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("md,max_recv", [(64, 4), (128, 2), (256, 1)])
def test_peer_gather_identity_parametrized(md, max_recv):
    import flydsl.expr as fx

    gather = compile_peer_gather_mxfp8_rows(model_dim=md, max_recv_cap=max_recv)
    dev = torch.device("cuda:0")
    x = torch.arange(max_recv * md, device=dev, dtype=torch.uint8).reshape(max_recv, md)
    recv = torch.zeros(max_recv * RECV_META_ROW_BYTES, dtype=torch.uint8, device=dev)
    for slot in range(max_recv):
        w = list(
            pack_recv_meta_row_i32(
                src_pe=0,
                kp=slot & 1,
                src_token=slot,
                expert_id=slot,
                router_w=1.0,
                dest_encode=slot,
                dst_slot=slot,
            )
        )
        row = torch.tensor(w, dtype=torch.int32, device=dev).view(torch.uint8)
        sl = slice(slot * RECV_META_ROW_BYTES, (slot + 1) * RECV_META_ROW_BYTES)
        recv[sl].copy_(row)
    p2p = torch.zeros(1, dtype=torch.int64, device=dev)
    p2p[0] = x.data_ptr()
    out = torch.zeros(max_recv, md, device=dev, dtype=torch.uint8)
    gather(
        fx.Int64(recv.data_ptr()),
        fx.Int64(p2p.data_ptr()),
        fx.Int64(out.data_ptr()),
        fx.Int32(max_recv),
        fx.Int32(md),
        fx.Int32(1),
        stream=fx.Stream(None),
    )
    torch.cuda.synchronize()
    assert torch.equal(out, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_peer_gather_scatter_dst_rows_precision():
    """out[dst_slot] == X[src_token] for P2P base on PE0; unused dst rows stay zero."""
    import flydsl.expr as fx

    md = 32
    max_recv = 4
    dev = torch.device("cuda:0")
    gather = compile_peer_gather_mxfp8_rows(model_dim=md, max_recv_cap=max_recv)
    n_src = 4
    x = torch.zeros(n_src, md, device=dev, dtype=torch.uint8)
    for r in range(n_src):
        x[r].fill_(200 + r)
    recv = torch.zeros(max_recv * RECV_META_ROW_BYTES, dtype=torch.uint8, device=dev)
    meta_rows = [(1, 3), (3, 0)]
    for slot, (src_tok, dst_slot) in enumerate(meta_rows):
        w = list(
            pack_recv_meta_row_i32(
                src_pe=0,
                kp=0,
                src_token=src_tok,
                expert_id=slot,
                router_w=1.0,
                dest_encode=slot,
                dst_slot=dst_slot,
            )
        )
        row = torch.tensor(w, dtype=torch.int32, device=dev).view(torch.uint8)
        off = slot * RECV_META_ROW_BYTES
        recv[off : off + RECV_META_ROW_BYTES] = row
    p2p = torch.zeros(1, dtype=torch.int64, device=dev)
    p2p[0] = x.data_ptr()
    out = torch.zeros(max_recv, md, device=dev, dtype=torch.uint8)
    total_recv = len(meta_rows)
    gather(
        fx.Int64(recv.data_ptr()),
        fx.Int64(p2p.data_ptr()),
        fx.Int64(out.data_ptr()),
        fx.Int32(total_recv),
        fx.Int32(md),
        fx.Int32(1),
        stream=fx.Stream(None),
    )
    torch.cuda.synchronize()
    assert torch.equal(out[3], x[1])
    assert torch.equal(out[0], x[3])
    for r in (1, 2):
        assert (out[r] == 0).all()


@pytest.mark.parametrize(
    "model_dim,inter_dim,experts,topk,tile_m,tile_n,tile_k",
    [
        (256, 128, 8, 2, 32, 64, 256),
        (512, 256, 16, 2, 32, 64, 256),
        (384, 128, 8, 2, 32, 64, 256),
    ],
)
def test_fused_intranode_peer_gather_compile_parametrized(
    model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k
):
    from kernels.moe_fused_dispatch_gather_gemm1_a8w4 import (
        compile_fused_dispatch_gather_gemm1_a8w4,
    )

    f = compile_fused_dispatch_gather_gemm1_a8w4(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        intranode_peer_gather=True,
    )
    assert callable(f)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_peer_gather_mxfp8_compile_and_run_identity():
    import flydsl.expr as fx

    md = 64
    dev = torch.device("cuda:0")
    max_recv = 4
    gather = compile_peer_gather_mxfp8_rows(model_dim=md, max_recv_cap=max_recv)
    x = torch.arange(max_recv * md, device=dev, dtype=torch.uint8).reshape(max_recv, md)
    recv = torch.zeros(max_recv * RECV_META_ROW_BYTES, dtype=torch.uint8, device=dev)
    for slot in range(max_recv):
        w = list(
            pack_recv_meta_row_i32(
                src_pe=0,
                kp=slot % 2,
                src_token=slot,
                expert_id=slot,
                router_w=1.0,
                dest_encode=slot,
                dst_slot=slot,
            )
        )
        row = torch.tensor(w, dtype=torch.int32, device=dev).view(torch.uint8)
        sl = slice(slot * RECV_META_ROW_BYTES, (slot + 1) * RECV_META_ROW_BYTES)
        recv[sl].copy_(row)

    p2p = torch.zeros(1, dtype=torch.int64, device=dev)
    p2p[0] = x.data_ptr()
    out = torch.zeros_like(x)
    gather(
        fx.Int64(recv.data_ptr()),
        fx.Int64(p2p.data_ptr()),
        fx.Int64(out.data_ptr()),
        fx.Int32(max_recv),
        fx.Int32(md),
        fx.Int32(1),
        stream=fx.Stream(None),
    )
    torch.cuda.synchronize()
    assert torch.equal(out, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sort_recv_meta_rows_by_src_pe():
    n = 4
    buf = bytearray(n * RECV_META_ROW_BYTES)
    rows_spec = [
        (2, 10, 0),
        (0, 5, 1),
        (1, 99, 2),
        (0, 3, 3),
    ]
    for slot, (pe, tok, dst) in enumerate(rows_spec):
        w = list(
            pack_recv_meta_row_i32(
                src_pe=pe,
                kp=slot % 2,
                src_token=tok,
                expert_id=slot,
                router_w=1.0,
                dest_encode=slot,
                dst_slot=dst,
            )
        )
        row = struct.pack("<8I", *w)
        off = slot * RECV_META_ROW_BYTES
        buf[off : off + RECV_META_ROW_BYTES] = row
    sort_recv_meta_rows_by_src_pe(buf, total_recv=n)
    keys = []
    for slot in range(n):
        off = slot * RECV_META_ROW_BYTES
        i32s = struct.unpack("<8I", buf[off : off + RECV_META_ROW_BYTES])
        d = unpack_recv_meta_row_i32(i32s)
        keys.append((d["src_pe"], d["src_token"]))
    assert keys == sorted(keys)


def test_intranode_peer_gather_requires_model_dim_multiple_of_four():
    from kernels.moe_fused_dispatch_gather_gemm1_a8w4 import (
        compile_fused_dispatch_gather_gemm1_a8w4,
    )

    with pytest.raises(ValueError, match="model_dim % 4"):
        compile_fused_dispatch_gather_gemm1_a8w4(
            model_dim=63,
            inter_dim=64,
            experts=4,
            topk=2,
            tile_m=32,
            tile_n=64,
            tile_k=64,
            doweight_stage1=False,
            intranode_peer_gather=True,
        )


def test_metadata_dispatch_jit_smoke_compile():
    from kernels.moe_metadata_dispatch_recv_meta import make_metadata_dispatch_recv_meta_jit

    j = make_metadata_dispatch_recv_meta_jit(
        rank=0,
        npes=1,
        experts_per_rank=8,
        experts_per_token=2,
        max_tok_per_rank=4,
        block_num=1,
        warp_num_per_block=1,
    )
    assert j is not None


if __name__ == "__main__":
    # Allow `torchrun ... test_file.py` for multi-GPU extensions (mori init left to callers).
    pytest.main([__file__, "-v", "-m", "not slow"] + sys.argv[1:])
