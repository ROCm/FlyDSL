# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x paged decode. Same arguments as ``pa_decode_tile``.

The KV cache is the vectorized layout that kernel uses: K is
``[blocks, Hkv, D / vec, page, vec]`` and V is ``[blocks, Hkv, page / vec, Dv, vec]``.
Query is packed ``[batch * query_length, Hq, D]``. This launch reshapes the query
and runs the gfx120x WMMA paged kernel. ``query_length == 1`` attends the valid
cache. A longer query uses bottom-right causal. One partition copies that output
into ``pout`` when the partial buffers are provided. The partial max and sum stay
unused.
"""

from __future__ import annotations

import torch

from flydsl.runtime.device import get_rocm_arch


def _copy_single_partition_output(
    output: torch.Tensor,
    pout: torch.Tensor,
    *,
    batch: int,
    num_kv_heads: int,
) -> None:
    """Copy the final output into partition 0 of the partial workspace.

    Query heads are kv-major: head ``kv * group + g`` lands at eqgs
    ``token * group + g``.
    """
    if tuple(pout.shape) == tuple(output.shape):
        pout.copy_(output)
        return
    if output.dim() != 3 or pout.dim() != 5:
        raise ValueError(
            f"temporary_output shape {tuple(pout.shape)} does not match output {tuple(output.shape)} "
            "or [batch, kv_heads, partitions, query_group, head]"
        )
    rows, num_q_heads, dv = (int(x) for x in output.shape)
    if batch < 1 or rows % batch != 0:
        raise ValueError(f"output rows {rows} are not divisible by batch {batch}")
    if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
        raise ValueError(f"cannot map {num_q_heads} query heads onto {num_kv_heads} kv heads")
    query_length = rows // batch
    group = num_q_heads // num_kv_heads
    packed = (
        output.reshape(batch, query_length, num_kv_heads, group, dv)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, num_kv_heads, query_length * group, dv)
    )
    if int(pout.shape[2]) < 1:
        raise ValueError("temporary_output has no partition slot")
    slot = pout[:, :, 0, :, :]
    if tuple(slot.shape) != tuple(packed.shape):
        raise ValueError(f"temporary_output partition slot {tuple(slot.shape)} != {tuple(packed.shape)}")
    slot.copy_(packed)


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
    sliding_window: tuple[int, int] | int | None = None,
) -> None:
    """Write decode attention into ``output``. Q and the cache share a dtype.

    One WMMA launch writes the combined output. When ``num_partitions`` is 1 and
    ``pmax``, ``psum``, and ``pout`` are all set, ``pout`` receives that output
    at partition 0. ``pmax`` and ``psum`` stay unused: ``flydsl_flash_attn_func``
    returns the attention output, not the online softmax max and sum, and those
    two values are not invented.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"pa_decode_tile requires gfx120x, got {arch}"
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
    if key_cache.dim() == 5:
        layout = "vectorized"
    elif key_cache.dim() == 4:
        layout = "linear"
    else:
        raise ValueError(f"gfx120x pa_decode_tile cache rank {key_cache.dim()} is not 4 or 5")
    if isinstance(sliding_window, int):
        window = (int(sliding_window), 0) if sliding_window > 0 else None
    else:
        window = sliding_window
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func

    got = flydsl_flash_attn_func(
        q,
        key_cache,
        value_cache,
        causal=query_length > 1 or window is not None,
        block_table=block_tables,
        seqlen_k=context_lengths,
        kv_cache_layout=layout,
        sliding_window=window,
        sm_scale=softmax_scale,
        q_descale=key_scale,
        k_descale=key_scale,
        v_descale=value_scale,
        stream=stream,
    )
    output.copy_(got.reshape(output.shape))
    if num_partitions == 1 and pmax is not None and psum is not None and pout is not None:
        # No online max or sum comes back from the flash launch. Leave both buffers alone.
        kv_heads = int(key_cache.shape[1] if layout == "vectorized" else key_cache.shape[2])
        _copy_single_partition_output(output, pout, batch=num_seqs, num_kv_heads=kv_heads)
