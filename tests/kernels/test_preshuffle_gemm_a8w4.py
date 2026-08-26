#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""SVDQuant low-rank up-projection epilogue for the A8W4 preshuffle GEMM - gfx950.

Covers the "svd" / "svd_bias" epilogues of kernels/gemm/mxfp4_preshuffle.launch_gemm:
y = a @ b.T + d @ L2.T (+ bias). Both the MFMA fast path (rank % 16 == 0, bf16 operands)
and the scalar fallback (rank = 8) are exercised. The fused result is compared against
the residual-only kernel plus a torch up-projection, which isolates the epilogue from
the quantized GEMM itself.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.mxfp4_preshuffle import launch_gemm
from tests.kernels.utils import gemm_common_utils as G

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

M, N, K = 33, 256, 256  # ragged M exercises the OOB-bounded d / C buffers
TILE_M, TILE_N, TILE_K = 32, 128, 128


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.contiguous().data_ptr())


def _pack_a8w4(a, b):
    """fp8 activation + MXFP4 weight in the layout launch_gemm expects.

    shuffle_scale_w4 requires a row count that is a multiple of 32, so A is quantized
    padded (see Mp below) and only the packed data is sliced back to M.
    """
    a_q, a_sc = G.per_1x32_f8_quant(a)
    b_q, b_sc, _ = G.per_1x32_f4_quant(b)
    return (
        a_q[:M],
        G.shuffle_scale_w4(a_sc, 1, False),
        G.shuffle_weight_w4(b_q, 16, False, False),
        G.shuffle_scale_w4(b_sc, 1, False),
    )


def _run(c, a_q, a_sc, b_shuf, b_sc, bias, d, l2, epilogue, rank, svd_use_mfma):
    launch_gemm(
        _ptr(c),
        _ptr(a_q),
        _ptr(b_shuf),
        _ptr(a_sc),
        _ptr(b_sc),
        _ptr(bias),
        _ptr(d),
        _ptr(l2),
        M,
        N,
        torch.cuda.current_stream(),
        N,
        K,
        TILE_M,
        TILE_N,
        TILE_K,
        "fp8",
        "bf16",
        "fp4",
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        epilogue,
        rank,
        svd_use_mfma,
    )
    torch.cuda.synchronize()
    return c


@pytest.mark.skipif(get_rocm_arch() != "gfx950", reason="A8W4 SVDQuant requires gfx950")
@pytest.mark.parametrize("rank,svd_use_mfma", [(16, True), (8, True), (16, False)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_svd_epilogue_matches_separate(rank, svd_use_mfma, with_bias):
    """Fused d @ L2.T (and bias) must match the residual-only kernel plus torch."""
    dev = torch.device("cuda")
    torch.manual_seed(0)
    Mp = (M + 31) // 32 * 32
    a = torch.randn(Mp, K, device=dev, dtype=torch.float32) * 0.5
    b = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.1
    d = torch.randn(M, rank, device=dev, dtype=torch.bfloat16) * 0.3
    l2 = torch.randn(N, rank, device=dev, dtype=torch.bfloat16) * 0.3
    empty = torch.empty(0, device=dev, dtype=torch.bfloat16)
    bias = torch.randn(N, device=dev, dtype=torch.bfloat16) if with_bias else empty
    a_q, a_sc, b_shuf, b_sc = _pack_a8w4(a, b)

    residual = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    _run(residual, a_q, a_sc, b_shuf, b_sc, empty, empty, empty, "none", 0, True)

    fused = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    _run(
        fused,
        a_q,
        a_sc,
        b_shuf,
        b_sc,
        bias,
        d,
        l2,
        "svd_bias" if with_bias else "svd",
        rank,
        svd_use_mfma,
    )

    reference = residual.float() + d.float() @ l2.float().T
    if with_bias:
        reference = reference + bias.float()
    cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(), reference.flatten(), dim=0)
    assert torch.isfinite(fused).all()
    assert cosine > 0.999, f"fused SVD epilogue cosine={cosine.item():.6f}"
