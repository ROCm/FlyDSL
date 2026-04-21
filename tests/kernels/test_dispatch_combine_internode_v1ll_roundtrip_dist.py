"""
8-GPU round-trip test for InterNodeV1LL:
dispatch -> combine should reconstruct input (k=1, weight=1).

Run:
    cd FlyDSL
    PYTHONPATH=python:.. torchrun --standalone --nproc_per_node=8 -m pytest \
      tests/kernels/test_dispatch_combine_internode_v1ll_roundtrip_dist.py::test_roundtrip_identity_8ranks \
      -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import mori.shmem as ms

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kernels.dispatch_combine_internode_v1ll_op import (
    FlyDSLDispatchCombineInterNodeV1LLConfig,
    FlyDSLDispatchCombineInterNodeV1LLOp,
)
from mori.ops.dispatch_combine import (
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
)


def _require_8_gpu_torchrun():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("Run with torchrun --nproc_per_node=8 (WORLD_SIZE=8)")
    if "LOCAL_RANK" not in os.environ:
        pytest.skip("Expected torchrun (LOCAL_RANK set)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_roundtrip_identity_8ranks():
    _require_8_gpu_torchrun()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", local_rank),
    )
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    try:
        assert world_size == 8
        dev = torch.device("cuda", local_rank)

        hidden_dim = 128
        max_tok = 16
        cur_tok = 8
        experts_per_rank = 4
        topk = 1

        common_kwargs = dict(
            data_type=torch.bfloat16,
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            scale_dim=0,
            scale_type_size=1,
            max_token_type_size=2,
            max_num_inp_token_per_rank=max_tok,
            num_experts_per_rank=experts_per_rank,
            num_experts_per_token=topk,
            warp_num_per_block=2,
            block_num=8,
            max_total_recv_tokens=0,
            use_external_inp_buf=True,
            gpu_per_node=8,
            rdma_block_num=4,
            num_qp_per_pe=1,
            quant_type="none",
        )

        fly_cfg = FlyDSLDispatchCombineInterNodeV1LLConfig(**common_kwargs)
        mori_cfg = EpDispatchCombineConfig(
            **common_kwargs,
            kernel_type=EpDispatchCombineKernelType.InterNodeV1LL,
        )

        fly_op = FlyDSLDispatchCombineInterNodeV1LLOp(
            fly_cfg, use_flydsl_copy_staging=True
        )
        mori_op = EpDispatchCombineOp(mori_cfg)

        g = torch.Generator(device=dev)
        g.manual_seed(2026 + rank)
        inp = torch.randn(cur_tok, hidden_dim, dtype=torch.bfloat16, device=dev, generator=g)
        # k=1 and weight=1 => round-trip should be identity after combine.
        wts = torch.ones(cur_tok, 1, dtype=torch.float32, device=dev)

        # Route to varying destination ranks to exercise inter-node path.
        dst_rank = torch.arange(cur_tok, device=dev, dtype=torch.int32)
        dst_rank = (dst_rank + rank + 1) % world_size
        idx = (dst_rank * experts_per_rank).view(cur_tok, 1).contiguous()

        fly_d = fly_op.dispatch(inp, wts, None, idx)
        fly_c = fly_op.combine(fly_d[0], None, fly_d[3])

        mori_d = mori_op.dispatch(inp, wts, None, idx)
        mori_c = mori_op.combine(mori_d[0], None, mori_d[3])

        torch.cuda.synchronize()
        dist.barrier()

        fly_out = fly_c[0][:cur_tok]
        mori_out = mori_c[0][:cur_tok]

        assert torch.allclose(
            fly_out, inp, atol=1e-2, rtol=1e-2
        ), f"rank{rank}: FlyDSL round-trip mismatch"
        assert torch.allclose(
            mori_out, inp, atol=1e-2, rtol=1e-2
        ), f"rank{rank}: mori round-trip mismatch"
        assert torch.allclose(
            fly_out, mori_out, atol=1e-2, rtol=1e-2
        ), f"rank{rank}: FlyDSL vs mori round-trip mismatch"

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
