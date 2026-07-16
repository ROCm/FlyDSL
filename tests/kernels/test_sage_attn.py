# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the CDNA SageAttention kernel."""

import math

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.sage_attn_cdna import build_sage_attn_cdna_module

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
pytestmark.append(
    pytest.mark.skipif(
        not isinstance(_ARCH, str) or not (_ARCH.startswith("gfx942") or _ARCH.startswith("gfx950")),
        reason=f"SageAttention requires gfx942 or gfx950, got {_ARCH}",
    )
)


def _fp8_dtype():
    return torch.float8_e4m3fn if _ARCH.startswith("gfx95") else torch.float8_e4m3fnuz


def _quantize(q, k, v, block_m, block_n, softmax_scale):
    batch, seq_len, heads, head_dim = q.shape
    q_blocks = math.ceil(seq_len / block_m)
    k_blocks = math.ceil(seq_len / block_n)

    q_scaled = q.float() * (softmax_scale * math.log2(math.e))
    q_view = q_scaled.view(batch, q_blocks, block_m, heads, head_dim)
    q_amax = q_view.abs().amax(dim=(2, 4)).permute(0, 2, 1).contiguous()
    q_scale = (q_amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    q_scale_bsh = q_scale.permute(0, 2, 1)[:, :, None, :, None]
    q_int8 = torch.round(q_view / q_scale_bsh).clamp(-127, 127).to(torch.int8).view_as(q)

    k_view = k.float().view(batch, k_blocks, block_n, heads, head_dim)
    k_amax = k_view.abs().amax(dim=(2, 4)).permute(0, 2, 1).contiguous()
    k_scale = (k_amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    k_scale_bsh = k_scale.permute(0, 2, 1)[:, :, None, :, None]
    k_int8 = torch.round(k_view / k_scale_bsh).clamp(-127, 127).to(torch.int8).view_as(k)

    fp8_dtype = _fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    v_scale = (v.float().abs().amax(dim=1) / fp8_max).clamp_min(torch.finfo(torch.float32).tiny)
    v_fp8 = (v.float() / v_scale[:, None]).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    v_blocked = (
        v_fp8.permute(0, 2, 3, 1).reshape(batch, heads, head_dim, k_blocks, block_n).permute(0, 1, 3, 2, 4).contiguous()
    )
    return q_int8, q_scale, k_int8, k_scale, v_blocked, v_scale


@pytest.mark.parametrize("causal", [False, True])
def test_sage_attn_bf16(causal):
    torch.manual_seed(42)
    batch, seq_len, heads, head_dim = 1, 256, 8, 128
    block_m, block_n = 256, 64
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    softmax_scale = head_dim**-0.5
    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = _quantize(q, k, v, block_m, block_n, softmax_scale)

    out = torch.empty_like(q)
    dummy = torch.empty(1, device="cuda", dtype=torch.float32)
    kernel = build_sage_attn_cdna_module(
        num_q_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        causal=causal,
        sm_scale=softmax_scale,
        block_m=block_m,
        block_n=block_n,
        gfx942_w256=True,
        v_transposed=True,
    )
    kernel(
        q_int8.reshape(-1),
        k_int8.reshape(-1),
        v_fp8.reshape(-1),
        out.reshape(-1),
        q_scale,
        k_scale,
        v_scale,
        dummy,
        dummy,
        dummy,
        dummy,
        batch,
        seq_len,
        seq_len,
        q_scale.shape[2],
        stream=torch.cuda.current_stream(),
    )

    ref = F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        is_causal=causal,
        scale=head_dim**-0.5,
    ).transpose(1, 2)
    cosine = F.cosine_similarity(out.float().reshape(-1, head_dim), ref.reshape(-1, head_dim), dim=1)
    assert torch.isfinite(out).all()
    assert cosine.min().item() > 0.97
    assert cosine.mean().item() > 0.99
