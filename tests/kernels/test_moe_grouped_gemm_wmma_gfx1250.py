#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Smoke tests for gfx1250 masked grouped MoE WMMA GEMM kernels."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

import flydsl  # noqa: F401 -- preload comgr before torch/HIP loads LLVM
from flydsl.runtime.device import get_rocm_arch
from kernels.moe_grouped_gemm_wmma_gfx1250 import (
    compile_moe_grouped_gemm1_masked,
    compile_moe_grouped_gemm2_masked,
)


pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    arch = str(get_rocm_arch())
    if not arch.startswith("gfx1250"):
        pytest.skip(f"grouped WMMA MoE kernels require gfx1250, got {arch}")


def _alloc_grouped_wmma_weight(
    shape: tuple[int, int, int],
    *,
    tile_n: int,
    tile_k: int,
    device: str,
    dtype: torch.dtype,
    scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (kernel_weight, reference_weight) for grouped WMMA kernels."""
    if tile_n != tile_k:
        raise ValueError("grouped WMMA weight allocation currently requires tile_n == tile_k")
    e, n, k = shape
    if n % tile_n != 0 or k % tile_k != 0:
        raise ValueError("weight shape must be divisible by tile_n and tile_k")

    ref = (torch.randn(shape, device=device, dtype=dtype) * scale).contiguous()
    kernel = torch.empty_like(ref)
    for n0 in range(0, n, tile_n):
        for k0 in range(0, k, tile_k):
            kernel[:, n0:n0 + tile_n, k0:k0 + tile_k] = (
                ref[:, n0:n0 + tile_n, k0:k0 + tile_k]
                .transpose(1, 2)
                .contiguous()
            )
    return kernel, ref


def test_moe_grouped_gemm1_masked_smoke():
    _require_gfx1250()

    torch.manual_seed(0)
    device = "cuda"
    E, max_m, model_dim, inter_dim = 1, 16, 64, 64
    tile_n, tile_k = 32, 32
    masked_m = torch.tensor([9], dtype=torch.int32, device=device)

    x = (torch.randn(E, max_m, model_dim, device=device, dtype=torch.float16) * 0.1).contiguous()
    w, w_ref = _alloc_grouped_wmma_weight(
        (E, 2 * inter_dim, model_dim),
        tile_n=tile_n,
        tile_k=tile_k,
        device=device,
        dtype=torch.float16,
    )
    y = torch.empty(E, max_m, inter_dim, device=device, dtype=torch.float16)

    kernel = compile_moe_grouped_gemm1_masked(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        max_m=max_m,
        tile_m=16,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=1,
        n_warp=2,
        in_dtype="fp16",
        out_dtype="f16",
        num_buffers=2,
        expert_sched_mode=False,
    )
    kernel(y, x, w, masked_m, max_m, inter_dim, model_dim, E, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    for e in range(E):
        valid = int(masked_m[e].item())
        gate = x[e, :valid].float() @ w_ref[e, :inter_dim].float().t()
        up = x[e, :valid].float() @ w_ref[e, inter_dim:].float().t()
        ref = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(
            y[e, :valid].float(),
            ref,
            rtol=5e-2,
            atol=5e-2,
        )


def test_moe_grouped_gemm2_masked_smoke():
    _require_gfx1250()

    torch.manual_seed(1)
    device = "cuda"
    E, max_m, model_dim, inter_dim = 1, 16, 64, 64
    tile_n, tile_k = 32, 32
    masked_m = torch.tensor([11], dtype=torch.int32, device=device)

    x = (torch.randn(E, max_m, inter_dim, device=device, dtype=torch.float16) * 0.1).contiguous()
    w, w_ref = _alloc_grouped_wmma_weight(
        (E, model_dim, inter_dim),
        tile_n=tile_n,
        tile_k=tile_k,
        device=device,
        dtype=torch.float16,
    )
    y = torch.empty(E, max_m, model_dim, device=device, dtype=torch.float16)

    kernel = compile_moe_grouped_gemm2_masked(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        max_m=max_m,
        tile_m=16,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=1,
        n_warp=2,
        in_dtype="fp16",
        out_dtype="f16",
        num_buffers=2,
        expert_sched_mode=False,
    )
    kernel(y, x, w, masked_m, max_m, model_dim, inter_dim, E, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    for e in range(E):
        valid = int(masked_m[e].item())
        out = y[e, :valid].float()
        assert torch.isfinite(out).all()
        assert out.abs().max() > 0
        ref = x[e, :valid].float() @ w_ref[e].float().t()
        torch.testing.assert_close(
            out,
            ref,
            rtol=5e-2,
            atol=5e-2,
        )
