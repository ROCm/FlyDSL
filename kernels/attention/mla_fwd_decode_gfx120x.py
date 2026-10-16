# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x MLA decode. Same arguments as ``flydsl_mla_fwd_decode``.

``work_indptr`` is the page CSR and ``kv_page_indices`` lists the pages.
``work_info_set`` is accepted for the call shape. The latent row's first
``value`` width is V. The query width is K. A width past one LDS row is split
by the WMMA kernel. One launch writes ``final_output``. ``split_output``
receives that result when it has the same number of elements. ``split_lse``
receives the fp32 LSE when it can hold those elements.
"""

from __future__ import annotations

import torch

from flydsl.runtime.device import get_rocm_arch


def _write_partition(dst: torch.Tensor, src: torch.Tensor, *, prefix: bool) -> None:
    """Copy one partition into ``dst`` without allocating a split workspace.

    Equal element counts copy in logical order. A larger ``dst`` receives the
    leading logical elements only when ``prefix`` is set and a leading slice
    fits. A smaller buffer is left untouched.
    """
    flat = src.reshape(-1)
    n = int(flat.numel())
    size = int(dst.numel())
    if size < n or (not prefix and size != n):
        return
    if size == n:
        dst.copy_(flat.reshape(dst.shape))
        return
    if dst.is_contiguous():
        dst.view(-1)[:n].copy_(flat)
        return
    tail = 1
    for dim in dst.shape[1:]:
        tail *= int(dim)
    slots = n // tail if tail else 0
    if tail and n % tail == 0 and dst.dim() > 0 and slots <= int(dst.shape[0]):
        dst[:slots].copy_(flat.reshape((slots, *tuple(dst.shape[1:]))))


def flydsl_mla_fwd_decode(
    query: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_page_indices: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    final_output: torch.Tensor,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    softmax_scale: float,
    stream=None,
    *,
    num_cus: int = 1,
    lds_size: int = 0,
) -> None:
    """Write one MLA partition into ``final_output`` and matching split buffers."""
    del work_info_set
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"flydsl_mla_fwd_decode requires gfx120x, got {arch}"
    if query.dtype != kv_buffer.dtype:
        raise ValueError(f"gfx120x MLA keeps Q and KV in one dtype, got {query.dtype} and {kv_buffer.dtype}")
    indptr = work_indptr.to(dtype=torch.int64, device="cpu")
    batch = int(indptr.numel()) - 1
    if batch < 1:
        raise ValueError("work_indptr must describe at least one sequence")
    total_q, num_heads, qk_dim = (int(x) for x in query.shape)
    if total_q % batch != 0:
        raise ValueError(f"query rows {total_q} must be a multiple of batch {batch}")
    query_length = total_q // batch
    value_dim = int(final_output.shape[-1])
    if int(final_output.shape[0]) != total_q or int(final_output.shape[1]) != num_heads:
        raise ValueError(f"final_output {tuple(final_output.shape)} does not match query {tuple(query.shape)}")
    pages, page_size, kv_heads, cache_dim = (int(x) for x in kv_buffer.shape)
    del pages
    if cache_dim < qk_dim or cache_dim < value_dim:
        raise ValueError(f"latent dim {cache_dim} is shorter than query {qk_dim} or value {value_dim}")
    if num_heads % kv_heads != 0:
        raise ValueError(f"query heads {num_heads} must be a multiple of KV heads {kv_heads}")
    counts = [int(indptr[i + 1] - indptr[i]) for i in range(batch)]
    if any(n < 0 for n in counts):
        raise ValueError("work_indptr must be nondecreasing")
    max_pages = max(counts) if counts else 0
    block_tables = torch.zeros(batch, max(max_pages, 1), dtype=torch.int32, device=query.device)
    seqlen = torch.zeros(batch, dtype=torch.int32, device=query.device)
    indices = kv_page_indices.to(dtype=torch.int64, device="cpu")
    for seq in range(batch):
        start, stop = int(indptr[seq]), int(indptr[seq + 1])
        if stop > int(indices.numel()):
            raise ValueError("kv_page_indices is shorter than work_indptr")
        n = stop - start
        if n:
            block_tables[seq, :n] = indices[start:stop].to(device=query.device, dtype=torch.int32)
        seqlen[seq] = n * page_size
    q = query.view(batch, query_length, num_heads, qk_dim)
    k_cache = kv_buffer[..., :qk_dim].contiguous()
    v_cache = kv_buffer[..., :value_dim].contiguous()
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func

    # The WMMA launch owns the grid.
    del num_cus, lds_size
    got, lse = flydsl_flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=query_length > 1,
        block_table=block_tables,
        seqlen_k=seqlen,
        kv_cache_layout="linear",
        sm_scale=float(softmax_scale),
        return_lse=True,
        stream=stream,
    )
    final_output.copy_(got.reshape(final_output.shape))
    _write_partition(split_output, final_output, prefix=False)
    _write_partition(split_lse, lse, prefix=True)
