# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness coverage for the tile-programming PA decode path."""

from __future__ import annotations

import math

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.pa_decode_tile import pa_decode_tile

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

ARCH = str(get_rocm_arch()).split(":", 1)[0]
IS_GFX95 = ARCH.startswith("gfx95")
SUPPORTED_ARCH = ARCH == "gfx942" or IS_GFX95
FP8_DTYPE = torch.float8_e4m3fn if IS_GFX95 else torch.float8_e4m3fnuz
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _quantize_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = tensor.float().abs().max() / FP8_MAX
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    return (tensor.float() / scale).to(FP8_DTYPE), scale.reshape(1)


def _reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_length: int,
    query_length: int,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
) -> torch.Tensor:
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    outputs = []
    token_positions = torch.arange(context_length, device=query.device)
    causal_limit = context_length - query_length + torch.arange(query_length, device=query.device)
    causal_mask = token_positions[None, :] <= causal_limit[:, None]

    for sequence in range(block_tables.shape[0]):
        pages = block_tables[sequence].long()
        key = (
            key_cache[pages].permute(0, 3, 1, 2, 4).reshape(-1, key_cache.shape[1], head_dim)[:context_length].float()
            * key_scale
        ).expand(-1, num_q_heads, -1)
        value = (
            value_cache[pages]
            .permute(0, 2, 4, 1, 3)
            .reshape(-1, value_cache.shape[1], head_dim)[:context_length]
            .float()
            * value_scale
        ).expand(-1, num_q_heads, -1)
        q = query[sequence * query_length : (sequence + 1) * query_length].float()
        scores = torch.einsum("qhd,khd->hqk", q, key) * (head_dim**-0.5)
        scores.masked_fill_(~causal_mask[None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, value).to(query.dtype))

    return torch.cat(outputs)


@pytest.mark.skipif(not SUPPORTED_ARCH, reason=f"requires gfx942 or gfx95*, got {ARCH}")
def test_fp8_head_dim_192_matches_torch() -> None:
    batch = 2
    query_length = 4
    num_q_heads = 16
    num_kv_heads = 1
    head_dim = 192
    page_size = 64
    context_length = 1027
    pages_per_sequence = math.ceil(context_length / page_size)
    num_pages = batch * pages_per_sequence
    generator = torch.Generator(device="cuda").manual_seed(20260821)

    query = torch.empty((batch * query_length, num_q_heads, head_dim), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.5, 0.5, generator=generator
    )
    key_source = torch.empty(
        (num_pages, num_kv_heads, head_dim // 16, page_size, 16), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.5, 0.5, generator=generator)
    value_source = torch.empty(
        (num_pages, num_kv_heads, page_size // 16, head_dim, 16), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.5, 0.5, generator=generator)
    key, key_scale = _quantize_per_tensor(key_source)
    value, value_scale = _quantize_per_tensor(value_source)
    block_tables = torch.arange(num_pages - 1, -1, -1, dtype=torch.int32, device="cuda").reshape(
        batch, pages_per_sequence
    )
    context_lengths = torch.full((batch,), context_length, dtype=torch.int32, device="cuda")

    expected = _reference(
        query,
        key,
        value,
        block_tables,
        context_length,
        query_length,
        key_scale,
        value_scale,
    )
    actual = torch.full_like(query, float("nan"))
    pa_decode_tile(
        output=actual,
        query=query,
        key_cache=key,
        value_cache=value,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=key_scale,
        value_scale=value_scale,
        softmax_scale=head_dim**-0.5,
        num_partitions=4,
    )
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)


@pytest.mark.large_shape
@pytest.mark.skipif(not SUPPORTED_ARCH, reason=f"requires gfx942 or gfx95*, got {ARCH}")
def test_fp8_cache_offset_above_2gib() -> None:
    num_q_heads = 16
    num_kv_heads = 1
    head_dim = 192
    page_size = 64
    page_bytes = num_kv_heads * head_dim * page_size
    high_page = math.ceil(2**31 / page_bytes)
    num_pages = high_page + 1
    context_length = 2
    query_length = 1
    query = torch.ones((1, num_q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    key = torch.empty((num_pages, num_kv_heads, head_dim // 16, page_size, 16), dtype=FP8_DTYPE, device="cuda")
    value = torch.empty((num_pages, num_kv_heads, page_size // 16, head_dim, 16), dtype=FP8_DTYPE, device="cuda")
    # Rounded-up tiles use page 0 for bounded fallback loads. Keep both that
    # page and the selected high page finite so masked PV lanes cannot see NaNs.
    key[0].zero_()
    value[0].zero_()
    key[high_page].zero_()
    value[high_page].zero_()

    # Two valid tokens make the result sensitive to both the K and V addresses.
    # If K wraps to page 0, the scores become equal and the opposing values
    # average to zero instead of selecting the positive token.
    key[high_page, 0, :, 0, :].fill_(-0.25)
    key[high_page, 0, :, 1, :].fill_(0.25)
    value[high_page, 0, 0, :, 0].fill_(-1.0)
    value[high_page, 0, 0, :, 1].fill_(1.0)
    block_tables = torch.tensor([[high_page]], dtype=torch.int32, device="cuda")
    context_lengths = torch.full((1,), context_length, dtype=torch.int32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    expected = _reference(
        query,
        key,
        value,
        block_tables,
        context_length,
        query_length,
        scale,
        scale,
    )
    assert expected.float().abs().min().item() > 0.5
    actual = torch.full_like(query, float("nan"))

    pa_decode_tile(
        output=actual,
        query=query,
        key_cache=key,
        value_cache=value,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=scale,
        value_scale=scale,
        softmax_scale=head_dim**-0.5,
        num_partitions=1,
    )
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
