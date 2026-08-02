#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 A8W4/W4A4 preshuffle GEMM and SVDQuant epilogue coverage."""

import pytest
import torch

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.preshuffle_gemm_a8w4 import compile_preshuffle_gemm_a8w4, compile_preshuffle_gemm_w4
from tests.kernels.utils import gemm_common_utils

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _to_bytes(tensor):
    return tensor if tensor.dtype in (torch.uint8, torch.int8) else tensor.view(torch.uint8)


def _pack_a8w4(x, weight):
    x_q, x_scale = gemm_common_utils.per_1x32_f8_quant(x)
    weight_q, weight_scale, _ = gemm_common_utils.per_1x32_f4_quant(weight)
    return (
        _to_bytes(x_q),
        _to_bytes(gemm_common_utils.shuffle_scale_w4(x_scale, 1, False)),
        _to_bytes(gemm_common_utils.shuffle_weight_w4(weight_q, 16, False, False)),
        _to_bytes(gemm_common_utils.shuffle_scale_w4(weight_scale, 1, False)),
    )


def _args(out, x, weight, x_scale, weight_scale, d, l2, bias, m, n):
    return (
        out.view(-1),
        x.view(-1),
        weight.view(-1),
        x_scale.view(-1),
        weight_scale.view(-1),
        bias.view(-1),
        d.view(-1),
        l2.view(-1),
        m,
        n,
        torch.cuda.current_stream(),
    )


@pytest.mark.parametrize("rank,svd_use_mfma", [(16, True), (8, True), (16, False)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_a8w4_svd_epilogue_matches_separate(rank, svd_use_mfma, with_bias):
    """Fused d @ L2.T (and bias) must match the residual-only kernel plus torch."""
    if str(get_rocm_arch()) != "gfx950":
        pytest.skip(f"A8W4 SVDQuant requires gfx950, got {get_rocm_arch()}")

    m, n, k, tile_m, tile_n, tile_k = 33, 256, 256, 32, 128, 128
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    x = torch.randn(m, k, device=device, dtype=dtype) * 0.5
    weight = torch.randn(n, k, device=device, dtype=dtype) * 0.1
    d = torch.randn(m, rank, device=device, dtype=dtype) * 0.3
    l2 = torch.randn(n, rank, device=device, dtype=dtype) * 0.3
    bias = torch.randn(n, device=device, dtype=dtype) if with_bias else torch.empty(0, device=device, dtype=dtype)
    x_q, x_scale, weight_q, weight_scale = _pack_a8w4(x, weight)
    empty = torch.empty(0, device=device, dtype=dtype)

    residual = torch.empty(m, n, device=device, dtype=dtype)
    residual_launch = compile_preshuffle_gemm_a8w4(
        M=m, N=n, K=k, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16"
    )
    residual_compiled = flyc.compile(
        residual_launch,
        *_args(residual, x_q, weight_q, x_scale, weight_scale, empty, empty, empty, m, n),
    )
    residual_compiled(*_args(residual, x_q, weight_q, x_scale, weight_scale, empty, empty, empty, m, n))

    fused = torch.empty_like(residual)
    epilogue = "svd_bias" if with_bias else "svd"
    fused_launch = compile_preshuffle_gemm_a8w4(
        M=m,
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype="bf16",
        epilogue=epilogue,
        rank=rank,
        svd_use_mfma=svd_use_mfma,
    )
    fused_compiled = flyc.compile(fused_launch, *_args(fused, x_q, weight_q, x_scale, weight_scale, d, l2, bias, m, n))
    fused_compiled(*_args(fused, x_q, weight_q, x_scale, weight_scale, d, l2, bias, m, n))
    torch.cuda.synchronize()

    reference = residual.float() + d.float() @ l2.float().T
    if with_bias:
        reference += bias.float()
    cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(), reference.flatten(), dim=0)
    assert torch.isfinite(fused).all()
    assert cosine > 0.999, f"fused SVD epilogue cosine={cosine.item():.6f}"


def test_w4a4_factory_supports_svd_epilogue():
    """The W4 factory must expose the same standalone SVD epilogue contract."""
    if str(get_rocm_arch()) != "gfx950":
        pytest.skip(f"W4A4 SVDQuant requires gfx950, got {get_rocm_arch()}")

    launch = compile_preshuffle_gemm_w4(M=32, N=256, K=256, tile_m=32, tile_n=128, tile_k=128, epilogue="svd", rank=16)
    assert callable(launch)
