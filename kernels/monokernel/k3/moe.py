# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Kimi-K3-specific launchers for the existing A16W4 MoE kernels."""

from __future__ import annotations

import functools

import torch

from kernels.common.tensor_shim import _run_compiled
from kernels.moe.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from kernels.moe.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid
from kernels.moe.moe_2stage_a16wmix.utils import a16wmix_resolve_arch

_HIDDEN = 3584
_INTER = 384
_EXPERTS = 896
_TOP_K = 16
_BLOCK_M = 16

# These are the production Kimi-K3 entries selected from AITER's tuned table.
# Keeping the fixed-shape choices here avoids changing the generic MoE package or
# depending on an external CSV at runtime.
_GEMM1_CONFIGS = {
    1: (32, 256, 2, 4),
    2: (32, 128, 4, 0),
    3: (64, 256, 2, 0),
    4: (64, 256, 2, 4),
    5: (64, 256, 2, 4),
    6: (64, 256, 2, 4),
    7: (64, 256, 2, 4),
    8: (128, 256, 1, 1),
}
_GEMM2_XCD = {1: 4, 2: 0, 3: 0, 4: 4, 5: 4, 6: 4, 7: 4, 8: 0}


def _check_samples(samples: int) -> None:
    if samples not in _GEMM1_CONFIGS:
        raise ValueError(f"Kimi-K3 MoE supports 1-8 samples, got {samples}")


@functools.cache
def _compile_gemm1(samples: int):
    _check_samples(samples)
    tile_n, tile_k, k_wave, xcd_swizzle = _GEMM1_CONFIGS[samples]
    return compile_gemm1_a16w4_port(
        BM=_BLOCK_M,
        D_HIDDEN=_HIDDEN,
        D_INTER=_INTER,
        NE=_EXPERTS,
        TOPK=_TOP_K,
        TILE_N=tile_n,
        TILE_K=tile_k,
        act="situv2",
        b_cache_mod=2,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=None,
        w_dtype="fp4",
        w_layout="standard",
        k_wave=k_wave,
        rocm_arch=a16wmix_resolve_arch(),
    )


@functools.cache
def _compile_gemm2(samples: int):
    _check_samples(samples)
    return compile_gemm2_a16w4_port(
        BM=_BLOCK_M,
        NE=_EXPERTS,
        N_OUT=_HIDDEN,
        D_INTER=_INTER,
        TILE_N=128,
        TILE_K=128,
        b_cache_mod=2,
        xcd_swizzle=_GEMM2_XCD[samples],
        waves_per_eu=None,
        w_dtype="fp4",
        persist=False,
        rocm_arch=a16wmix_resolve_arch(),
    )


def kimi_k3_mxfp4_gemm1(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    samples: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """Run Kimi-K3 routed expert up/gate plus SiTUv2."""

    tile_n, _, _, _ = _GEMM1_CONFIGS[samples]
    launch = _compile_gemm1(samples)
    grid = gemm1_a16w4_grid(
        _BLOCK_M,
        INTER=_INTER,
        TILE_N=tile_n,
        max_m_blocks=int(sorted_expert_ids.numel()),
    )
    _run_compiled(
        launch,
        activation.data_ptr(),
        weight.data_ptr(),
        weight_scale.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        sorted_token_ids.data_ptr(),
        samples,
        int(grid),
        float(situ_beta),
        1.0 / float(situ_beta),
        float(situ_linear_beta),
        1.0 / float(situ_linear_beta),
        float("inf"),
        output.data_ptr(),
        torch.cuda.current_stream(),
    )
    return output


def kimi_k3_mxfp4_gemm2(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    output: torch.Tensor,
    *,
    samples: int,
    max_sorted: int,
) -> torch.Tensor:
    """Run Kimi-K3 routed expert down projection and weighted scatter."""

    launch = _compile_gemm2(samples)
    grid = gemm2_a16w4_grid(
        _BLOCK_M,
        N_OUT=_HIDDEN,
        TILE_N=128,
        max_m_blocks=int(sorted_expert_ids.numel()),
        persist=False,
    )
    _run_compiled(
        launch,
        activation.data_ptr(),
        weight.data_ptr(),
        weight_scale.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        samples,
        max_sorted,
        int(grid),
        output.data_ptr(),
        torch.cuda.current_stream(),
    )
    return output
