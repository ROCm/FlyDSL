"""
8-GPU parity test: FlyDSL InterNodeV1LL op dispatch vs mori dispatch.

Run:
    cd FlyDSL
    PYTHONPATH=python:.. torchrun --standalone --nproc_per_node=8 -m pytest \
      tests/kernels/test_dispatch_combine_internode_v1ll_op_parity_dist.py::test_dispatch_parity_8ranks \
      -v --tb=short
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
def test_dispatch_parity_8ranks():
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

        hidden_dim = 64
        topk = 2
        experts_per_rank = 4
        max_tok = 16
        cur_tok = 8

        # Keep launch config deterministic and lightweight for parity testing.
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

        # Build inputs with deterministic routing to avoid duplicate expert-on-same-PE ambiguity.
        g = torch.Generator(device=dev)
        g.manual_seed(1337 + rank)
        inp = torch.randn(cur_tok, hidden_dim, dtype=torch.bfloat16, device=dev, generator=g)
        wts = torch.rand(cur_tok, topk, dtype=torch.float32, device=dev, generator=g)
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-6)

        pe0 = rank
        pe1 = (rank + 1) % world_size
        exp0 = pe0 * experts_per_rank
        exp1 = pe1 * experts_per_rank + 1
        idx = torch.empty(cur_tok, topk, dtype=torch.int32, device=dev)
        idx[:, 0] = exp0
        idx[:, 1] = exp1

        f_out, f_w, _f_s, f_i, f_total = fly_op.dispatch(inp, wts, None, idx)
        m_out, m_w, _m_s, m_i, m_total = mori_op.dispatch(inp, wts, None, idx)
        torch.cuda.synchronize()
        dist.barrier()

        f_n = int(f_total.item())
        m_n = int(m_total.item())
        assert f_n == m_n, f"rank{rank}: total_recv mismatch {f_n} vs {m_n}"

        if f_n > 0:
            f_src = fly_op.get_dispatch_src_token_pos()[:f_n]
            m_src = mori_op.get_dispatch_src_token_pos()[:m_n]
            f_ord = torch.argsort(f_src)
            m_ord = torch.argsort(m_src)

            assert torch.equal(
                f_src[f_ord], m_src[m_ord]
            ), f"rank{rank}: src-token map mismatch"
            assert torch.equal(
                f_i[:f_n][f_ord], m_i[:m_n][m_ord]
            ), f"rank{rank}: indices mismatch"
            assert torch.allclose(
                f_w[:f_n][f_ord], m_w[:m_n][m_ord], atol=1e-5, rtol=1e-5
            ), f"rank{rank}: weights mismatch"
            assert torch.allclose(
                f_out[:f_n][f_ord], m_out[:m_n][m_ord], atol=1e-2, rtol=1e-2
            ), f"rank{rank}: token output mismatch"

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
