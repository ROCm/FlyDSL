# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x paged decode. Same arguments as ``pa_decode_tile``.

The KV cache is the vectorized layout that kernel uses: K is
``[blocks, Hkv, D / vec, page, vec]`` and V is ``[blocks, Hkv, page / vec, Dv, vec]``.
Query is packed ``[batch * query_length, Hq, D]``. This launch reshapes the query
and runs the gfx120x WMMA paged kernel. ``query_length == 1`` attends the valid
cache. A longer query uses bottom-right causal.
"""

from __future__ import annotations

import torch

from flydsl.runtime.device import get_rocm_arch


def pa_decode_tile(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: float | torch.Tensor | None,
    value_scale: float | torch.Tensor | None,
    softmax_scale: float | None = None,
    stream=None,
    *,
    num_partitions: int | None = None,
    pmax: torch.Tensor | None = None,
    psum: torch.Tensor | None = None,
    pout: torch.Tensor | None = None,
) -> None:
    """Write decode attention into ``output``. Q and the cache share a dtype."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"pa_decode_tile requires gfx120x, got {arch}"
    if pmax is not None or psum is not None or pout is not None:
        raise ValueError("gfx120x pa_decode_tile writes the combined output and does not take partial buffers")
    if num_partitions not in (None, 1):
        raise ValueError(f"gfx120x pa_decode_tile launches one split, got num_partitions={num_partitions}")
    if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
        raise ValueError(
            "gfx120x pa_decode_tile keeps Q and KV in one dtype, "
            f"got query {query.dtype}, key {key_cache.dtype}, value {value_cache.dtype}"
        )
    num_seqs = int(context_lengths.shape[0])
    total_q, num_q_heads, head_dim = (int(x) for x in query.shape)
    if total_q % num_seqs != 0:
        raise ValueError(f"query rows {total_q} must be a multiple of batch {num_seqs}")
    query_length = total_q // num_seqs
    if int(output.shape[0]) != total_q or int(output.shape[1]) != num_q_heads:
        raise ValueError(f"output {tuple(output.shape)} does not match query {tuple(query.shape)}")
    q = query.view(num_seqs, query_length, num_q_heads, head_dim)
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func

    got = flydsl_flash_attn_func(
        q,
        key_cache,
        value_cache,
        causal=query_length > 1,
        block_table=block_tables,
        seqlen_k=context_lengths,
        kv_cache_layout="vectorized",
        sm_scale=softmax_scale,
        q_descale=key_scale,
        k_descale=key_scale,
        v_descale=value_scale,
        stream=stream,
    )
    output.copy_(got.reshape(output.shape))
