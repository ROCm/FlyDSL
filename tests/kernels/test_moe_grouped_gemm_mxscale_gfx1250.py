#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Smoke tests for gfx1250 grouped A8W4 MXScale MoE GEMM wrappers."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

import flydsl  # noqa: F401 -- preload comgr before torch/HIP loads LLVM
from flydsl.runtime.device import get_rocm_arch
from kernels.moe_grouped_gemm_mxscale_gfx1250 import (
    compile_moe_grouped_gemm1_a8w4_masked,
    compile_moe_grouped_gemm2_a8w4_masked,
)
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (
    SCALE_BLOCK,
    preshuffle_e8m0_scale,
    random_fp8_data,
)
from tests.kernels.utils import fp4_utils


pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    arch = str(get_rocm_arch())
    if not arch.startswith("gfx1250"):
        pytest.skip(f"grouped MXScale MoE kernels require gfx1250, got {arch}")


def _reference_a8w4(a_fp8, b_fp4, a_scale, b_scale, m, n, k):
    a_f32 = fp4_utils.fp8_e4m3_to_f32(a_fp8.view(torch.uint8))[:m, :k]
    b_f32 = fp4_utils.mxfp4_to_f32(b_fp4.view(torch.uint8))[:n, :k]
    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[:m, :k]
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[:n, :k]
    return torch.matmul(a_f32 * a_sc, (b_f32 * b_sc).T)


def _prep_scale(scale, *, warp_tile, scale_k_per_tile):
    return preshuffle_e8m0_scale(scale, warp_tile, scale_k_per_tile=scale_k_per_tile)


def _prep_b(weight, *, rows, packed_cols):
    return fp4_utils.preshuffle_b_16x16(weight.contiguous().view(rows, packed_cols), rows, packed_cols).view_as(weight)


def _common_shape():
    return dict(E=2, max_m=16, model_dim=256, inter_dim=256, tile_m=16, tile_n=64, tile_k=128, m_warp=1, n_warp=2)


def test_moe_grouped_mxscale_gemm1_a8w4_masked_smoke():
    _require_gfx1250()

    torch.manual_seed(2)
    device = "cuda"
    s = _common_shape()
    E, max_m, model_dim, inter_dim = s["E"], s["max_m"], s["model_dim"], s["inter_dim"]
    masked_m = torch.tensor([9, 5], dtype=torch.int32, device=device)

    x_raw = torch.stack([random_fp8_data(max_m, model_dim) for _ in range(E)]).contiguous()
    w_raw = torch.stack([fp4_utils.random_fp4_packed(2 * inter_dim, model_dim) for _ in range(E)]).contiguous()
    # Stage1 applies silu(gate) * up in f16; keep block scales small enough
    # that random FP8/FP4 smoke data does not overflow the fused epilogue.
    scale_x_raw = torch.full((E, max_m, model_dim // SCALE_BLOCK), 120, dtype=torch.uint8)
    scale_w_raw = torch.full((E, 2 * inter_dim, model_dim // SCALE_BLOCK), 120, dtype=torch.uint8)

    warp_tile_m = s["tile_m"] // s["m_warp"]
    warp_tile_n = s["tile_n"] // s["n_warp"]
    scale_k_per_tile = s["tile_k"] // SCALE_BLOCK
    x_scale = torch.stack([_prep_scale(scale_x_raw[e], warp_tile=warp_tile_m, scale_k_per_tile=scale_k_per_tile) for e in range(E)]).cuda()
    w_scale = torch.stack([_prep_scale(scale_w_raw[e], warp_tile=warp_tile_n, scale_k_per_tile=scale_k_per_tile) for e in range(E)]).cuda()
    x = x_raw.cuda()
    w = torch.stack([_prep_b(w_raw[e], rows=2 * inter_dim, packed_cols=model_dim // 2) for e in range(E)]).cuda()
    y = torch.empty(E, max_m, inter_dim, device=device, dtype=torch.float16)

    kernel = compile_moe_grouped_gemm1_a8w4_masked(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        max_m=max_m,
        tile_m=s["tile_m"],
        tile_n=s["tile_n"],
        tile_k=s["tile_k"],
        m_warp=s["m_warp"],
        n_warp=s["n_warp"],
        out_dtype="f16",
        num_buffers=2,
        expert_sched_mode=False,
    )
    kernel(y, x, w, x_scale, w_scale, masked_m, max_m, inter_dim, model_dim, E, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    for e in range(E):
        valid = int(masked_m[e].item())
        gate_up = _reference_a8w4(x_raw[e], w_raw[e], scale_x_raw[e], scale_w_raw[e], valid, 2 * inter_dim, model_dim)
        gate_up = gate_up.to(torch.float16).float()
        ref = torch.nn.functional.silu(gate_up[:, :inter_dim]) * gate_up[:, inter_dim:]
        ref = ref.to(torch.float16).float()
        torch.testing.assert_close(y[e, :valid].float(), ref.to(y.device).float(), rtol=1e-1, atol=1e-1)


def test_moe_grouped_mxscale_gemm2_a8w4_masked_smoke():
    _require_gfx1250()

    torch.manual_seed(3)
    device = "cuda"
    s = _common_shape()
    E, max_m, model_dim, inter_dim = s["E"], s["max_m"], s["model_dim"], s["inter_dim"]
    masked_m = torch.tensor([11, 7], dtype=torch.int32, device=device)

    x_raw = torch.stack([random_fp8_data(max_m, inter_dim) for _ in range(E)]).contiguous()
    w_raw = torch.stack([fp4_utils.random_fp4_packed(model_dim, inter_dim) for _ in range(E)]).contiguous()
    scale_x_raw = torch.full((E, max_m, inter_dim // SCALE_BLOCK), 127, dtype=torch.uint8)
    scale_w_raw = torch.full((E, model_dim, inter_dim // SCALE_BLOCK), 127, dtype=torch.uint8)

    warp_tile_m = s["tile_m"] // s["m_warp"]
    warp_tile_n = s["tile_n"] // s["n_warp"]
    scale_k_per_tile = s["tile_k"] // SCALE_BLOCK
    x_scale = torch.stack([_prep_scale(scale_x_raw[e], warp_tile=warp_tile_m, scale_k_per_tile=scale_k_per_tile) for e in range(E)]).cuda()
    w_scale = torch.stack([_prep_scale(scale_w_raw[e], warp_tile=warp_tile_n, scale_k_per_tile=scale_k_per_tile) for e in range(E)]).cuda()
    x = x_raw.cuda()
    w = torch.stack([_prep_b(w_raw[e], rows=model_dim, packed_cols=inter_dim // 2) for e in range(E)]).cuda()
    y = torch.empty(E, max_m, model_dim, device=device, dtype=torch.float16)

    kernel = compile_moe_grouped_gemm2_a8w4_masked(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        max_m=max_m,
        tile_m=s["tile_m"],
        tile_n=s["tile_n"],
        tile_k=s["tile_k"],
        m_warp=s["m_warp"],
        n_warp=s["n_warp"],
        out_dtype="f16",
        num_buffers=2,
        expert_sched_mode=False,
    )
    kernel(y, x, w, x_scale, w_scale, masked_m, max_m, model_dim, inter_dim, E, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    for e in range(E):
        valid = int(masked_m[e].item())
        ref = _reference_a8w4(x_raw[e], w_raw[e], scale_x_raw[e], scale_w_raw[e], valid, model_dim, inter_dim)
        torch.testing.assert_close(y[e, :valid].float(), ref.to(y.device).float(), rtol=1e-1, atol=1e-1)
