"""
8-GPU FlyDSL ``copy_to_staging`` smoke (one rank per GPU, no mori op).

Run (single node 8 GPUs). Use ``torchrun -m pytest`` so pytest runs as a module (do not pass
``pytest`` as a script path). ``--standalone`` sets local rendezvous for one node::

    cd FlyDSL
    PYTHONPATH=python:.. torchrun --standalone --nproc_per_node=8 -m pytest \\
      tests/kernels/test_dispatch_combine_internode_v1ll_flydsl_dist.py::test_copy_to_staging_8ranks \\
      -v --tb=short

Do **not** mix with full ``tests/kernels/`` under ``torchrun`` (each rank would run every test).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flydsl.expr as fx

from kernels.dispatch_combine_internode_v1ll_copy_staging import (
    make_ep_dispatch_copy_to_staging_jit,
)


def _xfer_bytes_bf16(*, hidden_dim: int, experts_per_token: int, scale_dim: int, scale_type_size: int) -> int:
    index_bytes = experts_per_token * 4
    weight_bytes = experts_per_token * 4
    scale_bytes = scale_dim * scale_type_size
    hidden_bytes = hidden_dim * 2
    return hidden_bytes + index_bytes + weight_bytes + scale_bytes + 4


def _require_8_gpu_torchrun():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("Run with: torchrun --nproc_per_node=8 ... (WORLD_SIZE=8)")
    if "LOCAL_RANK" not in os.environ:
        pytest.skip("Expected torchrun (LOCAL_RANK set)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_copy_to_staging_8ranks():
    """Each rank packs its own tokens; verify hidden / indices / weights / flat id."""
    _require_8_gpu_torchrun()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        device_id=torch.device("cuda", local_rank),
    )
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 8

        device = torch.device("cuda", local_rank)
        max_tok = 16
        hidden_dim = 32
        k = 8
        scale_dim = 0
        scale_type_size = 1
        cur_tok = 2
        mp, wpb = 2, 2

        xfer = _xfer_bytes_bf16(
            hidden_dim=hidden_dim,
            experts_per_token=k,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
        )
        inp = torch.randn(cur_tok, hidden_dim, dtype=torch.bfloat16, device=device)
        idx = torch.randint(
            rank * 1000,
            rank * 1000 + 256,
            (cur_tok, k),
            dtype=torch.int32,
            device=device,
        )
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
        dist.barrier()

        hb = hidden_dim * 2
        ib = k * 4
        wb = k * 4
        sb = 0
        flat_stride = world_size * max_tok
        flat_base = rank * flat_stride

        st = staging.view(torch.uint8)
        inp_b = inp.view(torch.uint8).reshape(cur_tok, hb)
        for t in range(cur_tok):
            assert torch.equal(st[t * xfer : t * xfer + hb], inp_b[t])
            assert torch.equal(
                st[t * xfer + hb : t * xfer + hb + ib].view(torch.int32),
                idx[t].view(torch.int32),
            )
            assert torch.equal(
                st[t * xfer + hb + ib : t * xfer + hb + ib + wb].view(torch.float32),
                wts[t],
            )
            flat = int(
                st[t * xfer + hb + ib + wb + sb : t * xfer + hb + ib + wb + sb + 4]
                .view(torch.int32)[0]
                .item()
            )
            assert flat == flat_base + t

        dist.barrier()
    finally:
        dist.destroy_process_group()
