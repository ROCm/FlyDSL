# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wiring notes for FlyDSL internode v1 LL dispatch.

Kernel implementations live in ``dispatch_combine_internode_v1ll_kernel.py`` (``make_copy_staging_jit``,
``make_dispatch_internode_v1ll_main_jit``, ``make_dispatch_internode_v1ll_copy_main_fused_jit``).

Symmetric buffer **sizes and field layout** must match mori ``EpDispatchCombineHandle`` for
``KernelType::InterNodeV1`` / ``InterNodeV1LL`` (see ``mori/src/ops/dispatch_combine/dispatch_combine.cpp``):
``dispatch_inp``, ``staging``, ``dispatch_out``, ``shmem_out_indices``, weights, optional scales,
``interNodeChunkFlagMemObj``, ``nodeRecvTokenNumMemObj``, ``recvTokenNumMemObj``, ``dispTokOffsetMemObj``,
``dispTokIdToSrcTokIdMemObj``, device buffers ``blockFlagCounter``, ``interNodeBlocksBarrier``,
``interNodeDispSendMap``, ``interNodeDispDestTokIdMap``, ``destPeTokenCounter``, ``dispatchGridBarrier``,
``combineGridBarrier``, ``crossDeviceBarrierFlag``, ``totalRecvTokenNum``, etc.

Precompute P2P device pointer tables with ``mori.shmem.shmem_ptr_p2p`` exactly like
``FlyDSLDispatchCombineIntraNodeOp`` (per-PE rows for remote symmetric windows), then pass ``fx.Int64``
``data_ptr()`` values into the two JIT launches (copy grid = SM count, main grid = ``block_num``).

This file intentionally does **not** duplicate the full allocation block yet; integrate next to avoid
diverging from mori without CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dispatch_combine_internode_v1ll_kernel import (
    make_copy_staging_jit,
    make_dispatch_internode_v1ll_main_jit,
)


@dataclass
class FlyDSLDispatchInternodeV1LLConfig:
    rank: int
    world_size: int
    gpu_per_node: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    data_type: torch.dtype = torch.bfloat16
    warp_num_per_block: int = 16
    block_num: int = 80
    rdma_block_num: int = 32
    num_qp_per_pe: int = 1
    copy_grid_blocks: int = 80
    scale_dim: int = 0
    scale_type_size: int = 0
    enable_std_moe: bool = False

    @property
    def xfer_bytes(self) -> int:
        esz = torch.tensor([], dtype=self.data_type).element_size()
        hb = self.hidden_dim * esz
        ib = self.num_experts_per_token * 4
        wb = self.num_experts_per_token * 4
        sb = self.scale_dim * self.scale_type_size
        return hb + ib + wb + sb + 4

    @property
    def max_recv(self) -> int:
        return self.world_size * self.max_num_inp_token_per_rank


def build_jit_handles(cfg: FlyDSLDispatchInternodeV1LLConfig):
    """Return ``(copy_jit, main_jit)`` for the given static config."""
    r = cfg.rank
    copy_jit = make_copy_staging_jit(
        rank=r,
        npes=cfg.world_size,
        experts_per_token=cfg.num_experts_per_token,
        hidden_dim=cfg.hidden_dim,
        max_tok_per_rank=cfg.max_num_inp_token_per_rank,
        copy_grid_blocks=cfg.copy_grid_blocks,
        warp_num_per_block=cfg.warp_num_per_block,
        scale_dim=cfg.scale_dim,
        scale_type_size=cfg.scale_type_size,
        data_type=cfg.data_type,
    )
    main_jit = make_dispatch_internode_v1ll_main_jit(
        rank=r,
        npes=cfg.world_size,
        gpu_per_node=cfg.gpu_per_node,
        experts_per_rank=cfg.num_experts_per_rank,
        experts_per_token=cfg.num_experts_per_token,
        hidden_dim=cfg.hidden_dim,
        max_tok_per_rank=cfg.max_num_inp_token_per_rank,
        block_num=cfg.block_num,
        rdma_block_num=cfg.rdma_block_num,
        warp_num_per_block=cfg.warp_num_per_block,
        num_qp_per_pe=cfg.num_qp_per_pe,
        scale_dim=cfg.scale_dim,
        scale_type_size=cfg.scale_type_size,
        enable_std_moe=cfg.enable_std_moe,
        data_type=cfg.data_type,
    )
    return copy_jit, main_jit
