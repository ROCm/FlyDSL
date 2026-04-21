"""
FlyDSL-only tests for InterNode V1LL dispatch/combine helpers (no mori op / no HIP combine).

Run from repo root::

    cd FlyDSL && PYTHONPATH=python:.. pytest tests/kernels/test_dispatch_combine_internode_v1ll_flydsl.py -q

8-GPU ``copy_to_staging`` (``torchrun``) lives in
``test_dispatch_combine_internode_v1ll_flydsl_dist.py``.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flydsl.expr as fx

from kernels.dispatch_combine_internode_v1ll_copy_staging import (
    make_ep_dispatch_copy_to_staging_jit,
)
from kernels.dispatch_combine_internode_v1ll_combine import (
    make_ep_combine_all_jit,
    make_ep_combine_sync_jit,
)


def _xfer_bytes_bf16(*, hidden_dim: int, experts_per_token: int, scale_dim: int, scale_type_size: int) -> int:
    index_bytes = experts_per_token * 4
    weight_bytes = experts_per_token * 4
    scale_bytes = scale_dim * scale_type_size
    hidden_bytes = hidden_dim * 2
    return hidden_bytes + index_bytes + weight_bytes + scale_bytes + 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_jit_build_copy_staging():
    mp, wpb = 4, 2
    j = make_ep_dispatch_copy_to_staging_jit(
        rank=0,
        world_size=8,
        max_tok_per_rank=16,
        hidden_dim=128,
        experts_per_token=8,
        scale_dim=0,
        scale_type_size=1,
        multiprocessor_count=mp,
        warp_num_per_block=wpb,
        data_type=torch.bfloat16,
    )
    assert callable(j)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_jit_build_combine():
    mp, wpb = 4, 2
    hd, k, ner = 128, 8, 16
    for hw in (False, True):
        make_ep_combine_sync_jit(
            hidden_dim=hd,
            experts_per_token=k,
            multiprocessor_count=mp,
            warp_num_per_block=wpb,
            has_weights=hw,
            data_type=torch.bfloat16,
        )
        make_ep_combine_all_jit(
            rank=0,
            world_size=8,
            gpu_per_node=8,
            max_tok_per_rank=16,
            hidden_dim=hd,
            experts_per_token=k,
            num_experts_per_rank=ner,
            multiprocessor_count=mp,
            warp_num_per_block=wpb,
            has_weights=hw,
            data_type=torch.bfloat16,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.xfail(
    reason="FlyDSL scf.for with dynamic lower bound gw_id: multi-CTA schedule incomplete (xfail until compiler fix).",
    strict=False,
)
def test_launch_copy_to_staging_multi_warp_smoke():
    """``mp=4``, ``wpb=2`` → 8 global warps; ``cur_tok=2`` → ``lim=8``; full hidden per token."""
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    rank = 0
    world_size = 8
    max_tok = 16
    hidden_dim = 64
    k = 8
    scale_dim = 0
    scale_type_size = 1
    cur_tok = 2
    mp, wpb = 4, 2

    xfer = _xfer_bytes_bf16(
        hidden_dim=hidden_dim,
        experts_per_token=k,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
    )
    inp = torch.randn(cur_tok, hidden_dim, dtype=torch.bfloat16, device=device)
    idx = torch.randint(0, 32 * world_size, (cur_tok, k), dtype=torch.int32, device=device)
    wts = torch.randn(cur_tok, k, dtype=torch.float32, device=device)
    staging = torch.zeros(cur_tok * xfer, dtype=torch.uint8, device=device)
    scales = torch.zeros(1, dtype=torch.uint8, device=device)

    jit = make_ep_dispatch_copy_to_staging_jit(
        rank=rank,
        world_size=world_size,
        max_tok_per_rank=max_tok,
        hidden_dim=hidden_dim,
        experts_per_token=k,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        multiprocessor_count=mp,
        warp_num_per_block=wpb,
        data_type=torch.bfloat16,
    )
    jit(
        fx.Int64(inp.data_ptr()),
        fx.Int64(idx.data_ptr()),
        fx.Int64(wts.data_ptr()),
        fx.Int64(scales.data_ptr()),
        fx.Int64(staging.data_ptr()),
        fx.Int32(cur_tok),
    )
    torch.cuda.synchronize()

    hb = hidden_dim * 2
    ib = k * 4
    st = staging.view(torch.uint8).cpu()
    inp_cpu = inp.cpu().view(torch.uint8).reshape(cur_tok, hb)
    for t in range(cur_tok):
        assert (st[t * xfer : t * xfer + hb].numpy() == inp_cpu[t].numpy()).all()
    idx_cpu = idx.cpu()
    for t in range(cur_tok):
        off = t * xfer + hb
        got = st[off : off + ib].view(torch.int32)
        assert torch.equal(got, idx_cpu[t].view(torch.int32))
    wts_cpu = wts.cpu().view(torch.uint8).reshape(cur_tok, k * 4)
    for t in range(cur_tok):
        off = t * xfer + hb + ib
        assert (st[off : off + k * 4].numpy() == wts_cpu[t].numpy()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_launch_copy_to_staging_smoke():
    """Single-GPU smoke: pack a few bf16 tokens; verify hidden slice in staging."""
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    rank = 0
    world_size = 8
    max_tok = 16
    hidden_dim = 64
    k = 8
    scale_dim = 0
    scale_type_size = 1
    # Choose cur_tok so cur_tok * ceil(gw_num/cur_tok) == gw_num and every (token,part) is
    # assigned once (see copy_staging loop stride gw_num); e.g. gw_num=8, cur_tok=2 -> lim=8.
    # Single global warp (mp=1, wpb=1): one (token,part) covers full hidden row.
    cur_tok = 1
    mp, wpb = 1, 1

    xfer = _xfer_bytes_bf16(
        hidden_dim=hidden_dim,
        experts_per_token=k,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
    )
    inp = torch.randn(cur_tok, hidden_dim, dtype=torch.bfloat16, device=device)
    idx = torch.randint(0, 32 * world_size, (cur_tok, k), dtype=torch.int32, device=device)
    wts = torch.randn(cur_tok, k, dtype=torch.float32, device=device)
    staging = torch.zeros(cur_tok * xfer, dtype=torch.uint8, device=device)
    scales = torch.zeros(1, dtype=torch.uint8, device=device)  # unused when scale_dim=0

    jit = make_ep_dispatch_copy_to_staging_jit(
        rank=rank,
        world_size=world_size,
        max_tok_per_rank=max_tok,
        hidden_dim=hidden_dim,
        experts_per_token=k,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        multiprocessor_count=mp,
        warp_num_per_block=wpb,
        data_type=torch.bfloat16,
    )
    jit(
        fx.Int64(inp.data_ptr()),
        fx.Int64(idx.data_ptr()),
        fx.Int64(wts.data_ptr()),
        fx.Int64(scales.data_ptr()),
        fx.Int64(staging.data_ptr()),
        fx.Int32(cur_tok),
    )
    torch.cuda.synchronize()

    hb = hidden_dim * 2
    st = staging.view(torch.uint8).cpu()
    inp_cpu = inp.cpu().view(torch.uint8).reshape(cur_tok, hb)
    for t in range(cur_tok):
        got = st[t * xfer : t * xfer + hb].numpy()
        ref = inp_cpu[t].numpy()
        assert (got == ref).all(), f"token {t} hidden mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_launch_combine_sync_smoke():
    """combine_sync: inp -> comb_inp for total_recv tokens; weights -> shmem slot."""
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    hidden_dim = 48
    k = 8
    total_recv = 5
    mp, wpb = 4, 2

    inp = torch.randn(total_recv, hidden_dim, dtype=torch.bfloat16, device=device)
    comb_inp = torch.zeros_like(inp)
    wts = torch.randn(total_recv, k, dtype=torch.float32, device=device)
    wts_shmem = torch.zeros(total_recv * k, dtype=torch.int32, device=device)
    total_recv_buf = torch.tensor([total_recv], dtype=torch.int32, device=device)

    jit = make_ep_combine_sync_jit(
        hidden_dim=hidden_dim,
        experts_per_token=k,
        multiprocessor_count=mp,
        warp_num_per_block=wpb,
        has_weights=True,
        data_type=torch.bfloat16,
    )
    jit(
        fx.Int64(inp.data_ptr()),
        fx.Int64(comb_inp.data_ptr()),
        fx.Int64(wts.data_ptr()),
        fx.Int64(wts_shmem.data_ptr()),
        fx.Int64(total_recv_buf.data_ptr()),
    )
    torch.cuda.synchronize()

    assert torch.allclose(comb_inp, inp)
    w_back = wts_shmem.view(torch.float32)
    assert torch.allclose(w_back.view(total_recv, k), wts)
