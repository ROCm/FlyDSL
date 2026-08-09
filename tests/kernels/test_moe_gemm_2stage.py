#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Numeric correctness tests for the ``moe_gemm_2stage`` fp8 2-stage kernels.

Covers stage1 (gate-up + silu, ``compile_moe_gemm1``) and stage2
(down-projection, ``compile_moe_gemm2``) against torch fp8 references, checking
cosine similarity (not just that the builders trace/lower). The package is
fp8-only on CDNA3 (gfx94*) / CDNA4 (gfx95*).

Stage2 is exercised in both atomic (``accumulate=True``) and reduce
(``accumulate=False``) modes, across f16/bf16/f32 outputs (f32 is atomic-only),
and at two tile_m values (16 = decode-ish, 64 = prefill-ish).

Input construction (routing/sorting, per-token fp8 quant, weight preshuffle)
and the torch references are harvested from the proven standalone harness that
scored cos=1.0 across a wide sweep; helpers come from ``tests/utils`` and
``tests/kernels/test_ref`` rather than being reimplemented inline.
"""

import math
import os
import sys
from typing import Tuple

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (os.path.join(_REPO_ROOT, "build", "python_packages"), _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.moe.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2  # noqa: E402
from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2  # noqa: E402
from tests.utils import pertoken_quant, shuffle_weight  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = get_rocm_arch()
# gfx950 (MI350) uses OCP standard float8_e4m3fn; older MI300 uses fnuz.
_DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in _ARCH else torch.float8_e4m3fnuz


def _fp8_supported() -> bool:
    return ("gfx95" in _ARCH) or ("gfx94" in _ARCH)


_requires_fp8 = pytest.mark.skipif(not _fp8_supported(), reason="fp8 2-stage MoE requires gfx94*/gfx95*")


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).reshape(-1)
    b = b.to(torch.float32).reshape(-1)
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((a @ b) / denom)


def _out_torch_dtype(out_dtype: str) -> torch.dtype:
    return {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[out_dtype]


# ---------------------------------------------------------------------------
# Routing / sorting: pure-torch reference for aiter's moe_sorting (torch-native
# path). Harvested verbatim from the proven standalone stage2 harness.
# ---------------------------------------------------------------------------
def _moe_sorting_torch_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = topk_ids.device
    M, topk = topk_ids.shape

    max_num_tokens_padded = int(topk_ids.numel() + int(num_experts) * int(block_size) - int(topk))
    max_num_m_blocks = int((max_num_tokens_padded + int(block_size) - 1) // int(block_size))

    init_val = (int(topk) << 24) | int(M)
    sorted_ids = torch.full((max_num_tokens_padded,), init_val, dtype=torch.int32, device=device)
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((2,), dtype=torch.int32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    for expert_id in range(int(num_experts)):
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = int(token_id.numel())
        expert_blocks = int((tokens_num + int(block_size) - 1) // int(block_size))
        tokens_num_pad = int(expert_blocks * int(block_size))
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (topk_id.to(torch.int32) << 24) | token_id.to(
            torch.int32
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[token_id, topk_id].to(
            torch.float32
        )
        sorted_ids_begin = int(sorted_ids_begin + tokens_num_pad)
        sorted_expert_ids[sorted_expert_ids_begin : sorted_expert_ids_begin + expert_blocks] = int(expert_id)
        sorted_expert_ids_begin = int(sorted_expert_ids_begin + expert_blocks)

    num_tokens_post_pad[0] = int(sorted_ids_begin)
    num_tokens_post_pad[1] = int(topk_ids.shape[0])
    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


def _build_routing(topk_ids, topk_weights, *, experts, tile_m):
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad = _moe_sorting_torch_native(
        topk_ids.to(torch.int32),
        topk_weights.to(torch.float32),
        num_experts=int(experts),
        block_size=int(tile_m),
    )
    num_valid_ids = num_tokens_post_pad[:1].contiguous()
    blocks = int(sorted_expert_ids.numel())
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks


def _make_routing(tokens, experts, topk, tile_m, device, seed):
    torch.manual_seed(int(seed))
    score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    routing = _build_routing(topk_ids, topk_weights, experts=experts, tile_m=tile_m)
    return topk_ids, topk_weights, routing


# ---------------------------------------------------------------------------
# Stage1 (gate-up + silu).
# ---------------------------------------------------------------------------
def _run_gemm1(*, tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, out_dtype, seed=0):
    device = torch.device("cuda")
    doweight_stage1 = False

    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks = routing

    # randn inputs: silu(gate)*up compresses tiny activations and amplifies
    # relative fp8 error, so use unit-variance data (matches the upstream harness).
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )

    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_DTYPE_FP8)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_DTYPE_FP8)

    w1_shuffled = shuffle_weight(w1_q)
    w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim).contiguous()
    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)

    x_q = x_q.contiguous().view(tokens, model_dim)
    scale_x_1d = scale_x.view(-1).contiguous()
    scale_w1_1d = scale_w1_flat.view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)

    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=_out_torch_dtype(out_dtype))

    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        out_dtype=out_dtype,
    )

    def _args(o):
        return (
            o,
            x_q.view(-1),
            w1_shuffled_flat.view(-1),
            scale_x_1d,
            scale_w1_1d,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_valid_ids,
            tokens,
            inter_dim,
            model_dim,
            int(blocks),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(exe, *_args(out))
    compiled(*_args(out))
    torch.cuda.synchronize()

    ref = torch_moe_gemm1(
        x_q,
        w1_q_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=doweight_stage1,
    )
    return out, ref


# ---------------------------------------------------------------------------
# Stage2 (down-projection).
# ---------------------------------------------------------------------------
def _run_gemm2(*, tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, out_dtype, accumulate, seed=0):
    device = torch.device("cuda")
    doweight_stage1 = False
    doweight_stage2 = not doweight_stage1

    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks = routing

    s = 0.2
    x_fp32 = torch.rand((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.rand((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(model_dim)
    )
    w2_fp32 = torch.rand((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(inter_dim)
    )

    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_DTYPE_FP8)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_DTYPE_FP8)
    w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=_DTYPE_FP8)

    # Stage2 input A2 = quantized reference stage1 output.
    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)
    out1_ref = torch_moe_gemm1(
        x_q,
        w1_q_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=doweight_stage1,
    )
    a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=_DTYPE_FP8)

    w2_shuffled = shuffle_weight(w2_q)
    w2_kernel = w2_shuffled.view(experts * model_dim, inter_dim).contiguous().view(-1)
    a2_scale_1d = a2_scale.view(-1).contiguous()
    w2_scale_1d = scale_w2.view(experts * model_dim, 1).view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)

    out_dt = _out_torch_dtype(out_dtype)
    if accumulate:
        out = torch.zeros((tokens, model_dim), device=device, dtype=out_dt)
    else:
        out = torch.zeros((tokens * topk, model_dim), device=device, dtype=out_dt)

    exe = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        out_dtype=out_dtype,
        accumulate=accumulate,
    )

    def _args(o):
        return (
            o,
            a2_q.view(-1),
            w2_kernel.view(-1),
            a2_scale_1d,
            w2_scale_1d,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_valid_ids,
            tokens,
            model_dim,
            inter_dim,
            int(blocks),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(exe, *_args(out))
    out.zero_()
    compiled(*_args(out))
    torch.cuda.synchronize()

    ref = torch_moe_gemm2(
        a2_q,
        w2_q,
        a2_scale,
        scale_w2,
        topk_ids.to(torch.int64),
        topk_weights,
        model_dim=model_dim,
        doweight_stage2=doweight_stage2,
    )
    if accumulate:
        got = out
    else:
        # reduce mode: kernel scatters per (token, topk-slot); sum over the slot
        # dim to match the reference's [tokens, model_dim] reduction.
        got = out.view(tokens, topk, model_dim).sum(dim=1)
    return got, ref


# Small tile-valid fp8 shape: model_dim=256, inter_dim=128, experts=4, topk=2.
_SHAPE = dict(model_dim=256, inter_dim=128, experts=4, topk=2)


@_requires_fp8
@pytest.mark.parametrize("out_dtype", ["f16", "bf16"])
def test_moe_gemm1_numeric_prefill(out_dtype):
    """Stage1 gate-up + silu matches the torch fp8 reference (prefill tile_m=64)."""
    out, ref = _run_gemm1(
        tokens=128,
        **_SHAPE,
        tile_m=64,
        tile_n=64,
        tile_k=128,
        out_dtype=out_dtype,
    )
    cos = _cosine_sim(out, ref)
    assert cos > 0.99, f"stage1 cos={cos:.5f} (out_dtype={out_dtype}, tile_m=64)"


@_requires_fp8
@pytest.mark.parametrize("out_dtype", ["f16", "bf16"])
def test_moe_gemm1_numeric_decode(out_dtype):
    """Stage1 gate-up + silu matches the torch fp8 reference (decode tile_m=16).

    Regression guard for the decode-path gather/scatter bug: the A-gather and
    output-scatter TV layouts read 32 M-rows even when BM=16, so un-seeded
    sorted_lds slots (16..31) decoded to a valid token 0 / slot 0 and piled
    garbage onto the first-routed token. Fixed by seeding a sentinel token id
    (== M, hardware-OOB) into every readable LDS slot before the real ids.
    """
    out, ref = _run_gemm1(
        tokens=8,
        **_SHAPE,
        tile_m=16,
        tile_n=64,
        tile_k=128,
        out_dtype=out_dtype,
    )
    cos = _cosine_sim(out, ref)
    assert cos > 0.99, f"stage1 cos={cos:.5f} (out_dtype={out_dtype}, tile_m=16)"


@_requires_fp8
@pytest.mark.parametrize(
    "accumulate,out_dtype",
    [
        (True, "f16"),
        (True, "bf16"),
        (True, "f32"),  # f32 is atomic-only (kernel rejects f32 + reduce)
        (False, "f16"),
        (False, "bf16"),
    ],
)
@pytest.mark.parametrize("tile_m,tokens", [(16, 8), (64, 128)])
def test_moe_gemm2_numeric(accumulate, out_dtype, tile_m, tokens):
    """Stage2 down-projection matches the torch fp8 reference (cosine).

    Covers atomic (accumulate=True) and reduce (accumulate=False) modes, all
    supported output dtypes, and both decode-ish (tile_m=16) and prefill-ish
    (tile_m=64) paths.
    """
    got, ref = _run_gemm2(
        tokens=tokens,
        **_SHAPE,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        out_dtype=out_dtype,
        accumulate=accumulate,
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, f"stage2 cos={cos:.5f} ({mode}, out_dtype={out_dtype}, tile_m={tile_m})"


@_requires_fp8
def test_moe_gemm2_rejects_f32_reduce():
    """f32 output is atomic-only; reduce mode must fail-fast."""
    with pytest.raises(ValueError):
        compile_moe_gemm2(
            **_SHAPE,
            tile_m=16,
            tile_n=64,
            tile_k=128,
            doweight_stage2=True,
            out_dtype="f32",
            accumulate=False,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
