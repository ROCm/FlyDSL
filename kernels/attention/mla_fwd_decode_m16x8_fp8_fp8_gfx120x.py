# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x WMMA form of the m16x8 latent decode.

Main's module is an MFMA kernel for a query width of 576 and a value width of
512. This module keeps those arguments and runs the latent attention on the
WMMA flash kernel. A width past one LDS row is split there. The same math
covers bf16, fp16, fp8 e4m3, fp8 e5m2, and int8.
"""

from __future__ import annotations

import torch

from flydsl.runtime.device import get_rocm_arch

QK_HEAD_DIM = 576
V_HEAD_DIM = 512


def launch_mla_fwd_decode_m16x8_fp8_fp8(
    query: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_page_indices: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    final_output: torch.Tensor,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    softmax_scale: float,
    num_cus: int = 1,
    lds_size: int = 0,
    stream=None,
) -> None:
    """Decode one query row per sequence into ``final_output``.

    A 2D query is ``[sequences * heads, qk]`` and a 2D cache is ``[pages, qk]``,
    the same flat layout main passes to its launcher. ``num_cus`` and
    ``lds_size`` are accepted from that launcher. The WMMA launch owns the grid.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"launch_mla_fwd_decode_m16x8_fp8_fp8 requires gfx120x, got {arch}"
    if query.dtype != kv_buffer.dtype:
        raise ValueError(f"gfx120x MLA keeps Q and KV in one dtype, got {query.dtype} and {kv_buffer.dtype}")
    batch = int(work_indptr.numel()) - 1
    if batch < 1:
        raise ValueError("work_indptr must describe at least one sequence")
    if query.dim() == 2:
        rows, qk_dim = (int(v) for v in query.shape)
        if rows % batch != 0:
            raise ValueError(f"flat query rows {rows} must be a multiple of batch {batch}")
        num_heads = rows // batch
        query_3d = query.view(batch, num_heads, qk_dim)
    elif query.dim() == 3:
        query_3d = query
        num_heads = int(query.shape[1])
        qk_dim = int(query.shape[2])
    else:
        raise ValueError(f"query must be [rows, qk] or [batch, heads, qk], got {tuple(query.shape)}")
    value_dim = int(final_output.shape[-1])
    if kv_buffer.dim() == 2:
        pages, cache_dim = (int(v) for v in kv_buffer.shape)
        if cache_dim < qk_dim or cache_dim < value_dim:
            raise ValueError(f"latent dim {cache_dim} is shorter than query {qk_dim} or value {value_dim}")
        kv_4d = kv_buffer.view(pages, 1, 1, cache_dim)
    elif kv_buffer.dim() == 4:
        kv_4d = kv_buffer
    else:
        raise ValueError(f"kv_buffer must be [pages, dim] or [pages, page, heads, dim], got {tuple(kv_buffer.shape)}")
    final_3d = final_output.view(batch, num_heads, value_dim)
    from kernels.attention.mla_fwd_decode_gfx120x import flydsl_mla_fwd_decode

    flydsl_mla_fwd_decode(
        query_3d,
        kv_4d,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_3d,
        split_output,
        split_lse,
        float(softmax_scale),
        stream=stream,
        num_cus=num_cus,
        lds_size=lds_size,
    )
