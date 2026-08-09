#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Numeric correctness tests for the ``moe_gemm_2stage`` fp8/int8 2-stage kernels.

Covers stage1 (gate-up + silu) and stage2 (down-projection) against torch
references by cosine similarity, for both fp8 and int8 inputs, on CDNA3 (gfx94*)
/ CDNA4 (gfx95*).

Stage2 runs both atomic (``accumulate=True``) and reduce (``accumulate=False``)
modes, f16/bf16/f32 outputs (f32 atomic-only), and two tile_m values (16 decode,
64 prefill). Routing/quant/preshuffle helpers come from ``tests/utils`` and
``tests/kernels/test_ref``.
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


def _quant_dtype(in_dtype: str) -> torch.dtype:
    """Kernel input torch dtype for the given ``in_dtype``."""
    return torch.int8 if in_dtype in ("int8", "int8smooth", "int4") else _DTYPE_FP8


def _pack_shuffled_int8_to_packed_int4(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a PRESHUFFLED int8 tensor (values in [-8, 7]) into packed int4 bytes.

    W4A8 weight packing: each contiguous 8-value block [v0..v7] -> 4 bytes with
    ``byte_j = v_j | (v_{j+4} << 4)`` (low nibbles hold v0..v3, high hold v4..v7),
    matching the in-kernel 7-op unpack (even={v0..v3}, odd={v4..v7}). Perturbing
    this de-interleave order (e.g. swapping the low/high nibble halves) breaks the
    result -- exercised by test_moe_gemm1_int4_perturb.
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    for j in range(4):
        out[:, j] = u[:, j] | (u[:, j + 4] << 4)
    return out.view(-1).to(torch.int8)


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


def _launch(exe, out, *, x, w, scale_x, scale_w, routing, dim0, dim1, tokens, zero_out=False):
    """Compile + run a stage kernel with the shared positional arg layout.

    ``dim0``/``dim1`` fill the two shape slots (gemm1: inter_dim, model_dim;
    gemm2: model_dim, inter_dim). All 1D operands are flattened here."""
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks = routing

    def _args(o):
        return (
            o,
            x.view(-1),
            w.view(-1),
            scale_x.view(-1).contiguous(),
            scale_w.view(-1).contiguous(),
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights.contiguous().view(-1),
            num_valid_ids,
            tokens,
            dim0,
            dim1,
            int(blocks),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(exe, *_args(out))
    if zero_out:
        out.zero_()
    compiled(*_args(out))
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Stage1 (gate-up + silu).
# ---------------------------------------------------------------------------
def _run_gemm1(
    *, tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, out_dtype, in_dtype="fp8", seed=0
):
    device = torch.device("cuda")
    doweight_stage1 = False
    _qd = _quant_dtype(in_dtype)

    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)

    # randn inputs: silu(gate)*up compresses tiny activations and amplifies
    # relative fp8 error, so use unit-variance data (matches the upstream harness).
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )

    _is_int4 = in_dtype == "int4"
    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_qd)
    # W4A8 weight is quantized to signed int4 range [-8, 7] (dtypeMax=7).
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_qd, **({"dtypeMax": 7} if _is_int4 else {}))

    w1_shuffled_flat = shuffle_weight(w1_q).view(experts * (2 * inter_dim), model_dim).contiguous()
    if _is_int4:
        # Preshuffled int8 -> packed 2 int4/byte; the reference still uses w1_q_flat.
        w1_shuffled_flat = _pack_shuffled_int8_to_packed_int4(w1_shuffled_flat)
    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)
    x_q = x_q.contiguous().view(tokens, model_dim)

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
        in_dtype=in_dtype,
    )
    _launch(
        exe,
        out,
        x=x_q,
        w=w1_shuffled_flat,
        scale_x=scale_x,
        scale_w=scale_w1_flat,
        routing=routing,
        dim0=inter_dim,
        dim1=model_dim,
        tokens=tokens,
    )

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
def _run_gemm2(
    *,
    tokens,
    model_dim,
    inter_dim,
    experts,
    topk,
    tile_m,
    tile_n,
    tile_k,
    out_dtype,
    accumulate,
    in_dtype="fp8",
    seed=0,
):
    device = torch.device("cuda")
    doweight_stage1 = False
    doweight_stage2 = not doweight_stage1
    _qd = _quant_dtype(in_dtype)

    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)

    s = 0.2
    x_fp32 = torch.rand((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.rand((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(model_dim)
    )
    w2_fp32 = torch.rand((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(inter_dim)
    )

    _is_int4 = in_dtype == "int4"
    _wmax = {"dtypeMax": 7} if _is_int4 else {}
    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_qd)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_qd, **_wmax)
    w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=_qd, **_wmax)

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
    a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=_qd)

    w2_kernel = shuffle_weight(w2_q).view(experts * model_dim, inter_dim).contiguous().view(-1)
    if _is_int4:
        w2_kernel = _pack_shuffled_int8_to_packed_int4(w2_kernel)
    w2_scale_flat = scale_w2.view(experts * model_dim, 1)

    out_dt = _out_torch_dtype(out_dtype)
    out = torch.zeros((tokens, model_dim) if accumulate else (tokens * topk, model_dim), device=device, dtype=out_dt)

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
        in_dtype=in_dtype,
    )
    _launch(
        exe,
        out,
        x=a2_q,
        w=w2_kernel,
        scale_x=a2_scale,
        scale_w=w2_scale_flat,
        routing=routing,
        dim0=model_dim,
        dim1=inter_dim,
        tokens=tokens,
        zero_out=True,
    )

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
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int4"])
@pytest.mark.parametrize("out_dtype", ["f16", "bf16"])
@pytest.mark.parametrize("tile_m,tokens", [(64, 128), (16, 8)])
def test_moe_gemm1_numeric(in_dtype, out_dtype, tile_m, tokens):
    """Stage1 gate-up + silu matches the torch reference (prefill tile_m=64,
    decode tile_m=16), for fp8, int8, and W4A8 (int4 weight) inputs.

    The tile_m=16 case is the decode-path gather/scatter regression guard: the
    A-gather and output-scatter TV layouts read 32 M-rows even when BM=16, so
    un-seeded sorted_lds slots (16..31) decoded to a valid token 0 / slot 0 and
    piled garbage onto the first-routed token. Fixed by seeding a sentinel token
    id (== M, hardware-OOB) into every readable LDS slot before the real ids.

    int8 shares the fp8 pipeline (i32 MFMA acc converted to f32 in dequant).
    """
    out, ref = _run_gemm1(
        tokens=tokens,
        **_SHAPE,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        out_dtype=out_dtype,
        in_dtype=in_dtype,
    )
    cos = _cosine_sim(out, ref)
    assert cos > 0.99, f"stage1 cos={cos:.5f} (in_dtype={in_dtype}, out_dtype={out_dtype}, tile_m={tile_m})"


@_requires_fp8
def test_moe_gemm1_int4_perturb():
    """W4A8 de-interleave order is load-bearing: swapping the low/high nibble halves
    of every packed byte (which re-orders the in-kernel unpack's even={v0..v3} /
    odd={v4..v7} split) must destroy correctness. Runs the SAME kernel with the
    correct packing and with the perturbed packing and asserts the perturbed cosine
    collapses -- guarding the nibble ordering that ``_pack_shuffled_int8_to_packed_int4``
    and ``load_weight_int4_frag`` jointly define. gemm2 shares that loader; its
    all-positive weight data masks the swap, so the discriminating check lives here.
    """
    device = torch.device("cuda")
    tokens, tile_m = 128, 64
    topk_ids, topk_weights, routing = _make_routing(tokens, _SHAPE["experts"], _SHAPE["topk"], tile_m, device, 0)
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks = routing
    x_fp32 = torch.randn((tokens, _SHAPE["model_dim"]), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn(
        (_SHAPE["experts"], 2 * _SHAPE["inter_dim"], _SHAPE["model_dim"]), device=device, dtype=torch.float32
    ) * (1.0 / math.sqrt(_SHAPE["model_dim"]))
    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
    w1_shuf = shuffle_weight(w1_q).view(_SHAPE["experts"] * (2 * _SHAPE["inter_dim"]), _SHAPE["model_dim"]).contiguous()
    w1_q_flat = w1_q.view(_SHAPE["experts"] * (2 * _SHAPE["inter_dim"]), _SHAPE["model_dim"])
    scale_w1_flat = scale_w1.view(_SHAPE["experts"] * (2 * _SHAPE["inter_dim"]), 1)
    packed_ok = _pack_shuffled_int8_to_packed_int4(w1_shuf)
    # Perturbation: swap low/high nibble of every byte (even/odd de-interleave swap).
    _b = packed_ok.to(torch.int16) & 0xFF
    packed_bad = (((_b & 0xF) << 4) | ((_b >> 4) & 0xF)).to(torch.uint8).to(torch.int8)

    exe = compile_moe_gemm1(
        **_SHAPE, tile_m=tile_m, tile_n=64, tile_k=128, doweight_stage1=False, out_dtype="bf16", in_dtype="int4"
    )

    def _run(packed_w):
        out = torch.empty((tokens, _SHAPE["topk"], _SHAPE["inter_dim"]), device=device, dtype=torch.bfloat16)
        args = (
            out,
            x_q.contiguous().view(-1),
            packed_w.view(-1),
            scale_x.view(-1).contiguous(),
            scale_w1_flat.view(-1).contiguous(),
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights.contiguous().view(-1),
            num_valid_ids,
            tokens,
            _SHAPE["inter_dim"],
            _SHAPE["model_dim"],
            int(blocks),
            torch.cuda.current_stream(),
        )
        compiled = flyc.compile(exe, *args)
        compiled(*args)
        torch.cuda.synchronize()
        return out

    ref = torch_moe_gemm1(
        x_q,
        w1_q_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=_SHAPE["inter_dim"],
        doweight_stage1=False,
    )
    cos_ok = _cosine_sim(_run(packed_ok), ref)
    cos_bad = _cosine_sim(_run(packed_bad), ref)
    assert cos_ok > 0.99, f"correct W4A8 packing should match: cos={cos_ok:.5f}"
    assert cos_bad < 0.5, f"perturbed nibble de-interleave should break: cos={cos_bad:.5f}"


# ---------------------------------------------------------------------------
# Stage1 int8smooth: X is pre-expanded per route and stored SLOT-MAJOR
# ([topk*tokens, K], row = slot*tokens + token), with a per-(token,slot) scale
# in the same order. The kernel decodes the fused sorted id (slot<<24 | token)
# and gathers X / scale_x at slot*tokens + token. Only stage1 differs from int8.
# ---------------------------------------------------------------------------
def _build_stage1_int8smooth(x_fp32, w1_fp32, topk_ids):
    """Slot-major int8smooth stage1 inputs (mirrors CK moe_smoothquant output)."""
    device = x_fp32.device
    tokens, model_dim = x_fp32.shape
    topk = topk_ids.shape[1]
    experts = w1_fp32.shape[0]

    smooth = 0.75 + 0.5 * torch.rand((experts, model_dim), device=device, dtype=torch.float32)
    x_route = x_fp32[:, None, :].expand(tokens, topk, model_dim) * smooth[topk_ids.to(torch.int64)]
    amax = torch.amax(torch.abs(x_route), dim=-1, keepdim=True)
    scale_x = amax / 127.0
    scale_x[scale_x == 0] = 1.0
    x_q = (x_route / scale_x).to(torch.int8)
    # slot-major [topk, tokens, K] / [topk, tokens, 1]
    x_q_sm = x_q.permute(1, 0, 2).contiguous()
    sx_sm = scale_x.permute(1, 0, 2).contiguous()

    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
    w1_q_flat = w1_q.view(experts * (2 * w1_fp32.shape[1] // 2), model_dim)
    scale_w1_flat = scale_w1.view(-1, 1)
    w1_kernel = shuffle_weight(w1_q).view(w1_q_flat.shape[0], model_dim).contiguous()

    return dict(
        x_q=x_q_sm.view(tokens * topk, model_dim).contiguous(),
        scale_x_1d=sx_sm.view(-1).contiguous(),
        x_ref=x_q.contiguous(),  # token-major [tokens, topk, K] for the reference
        sx_ref=scale_x.contiguous(),  # token-major [tokens, topk, 1]
        w1_q_flat=w1_q_flat,
        w1_kernel=w1_kernel,
        scale_w1_flat=scale_w1_flat,
        scale_w1_1d=scale_w1_flat.view(-1).contiguous(),
    )


def _run_gemm1_int8smooth(*, tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, out_dtype, seed=0):
    device = torch.device("cuda")
    doweight_stage1 = False
    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)

    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )

    d = _build_stage1_int8smooth(x_fp32, w1_fp32, topk_ids)
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
        in_dtype="int8smooth",
    )
    _launch(
        exe,
        out,
        x=d["x_q"],
        w=d["w1_kernel"],
        scale_x=d["scale_x_1d"],
        scale_w=d["scale_w1_1d"],
        routing=routing,
        dim0=inter_dim,
        dim1=model_dim,
        tokens=tokens,
    )

    ref = torch_moe_gemm1(
        d["x_ref"],
        d["w1_q_flat"],
        d["sx_ref"],
        d["scale_w1_flat"],
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=doweight_stage1,
    )
    return out, ref


@_requires_fp8
@pytest.mark.parametrize("out_dtype", ["f16", "bf16"])
@pytest.mark.parametrize("tile_m,tokens", [(64, 128), (16, 8)])
def test_moe_gemm1_int8smooth_numeric(out_dtype, tile_m, tokens):
    """Stage1 int8smooth (slot-major A/scale_x gather) matches the torch reference.

    int8smooth differs from int8 ONLY in stage1: X is pre-expanded to
    [topk*tokens, K] in slot-major order and both the A-gather and the
    activation-scale load index row = slot*tokens + token (fused id: slot<<24 |
    token). Swapping that decode back to plain ``token`` breaks this test (cosine
    drops below 0.99), which is the whole point of the feature.
    """
    out, ref = _run_gemm1_int8smooth(
        tokens=tokens,
        **_SHAPE,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        out_dtype=out_dtype,
    )
    cos = _cosine_sim(out, ref)
    assert cos > 0.99, f"stage1 int8smooth cos={cos:.5f} (out_dtype={out_dtype}, tile_m={tile_m})"


@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int4"])
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
def test_moe_gemm2_numeric(accumulate, out_dtype, tile_m, tokens, in_dtype):
    """Stage2 down-projection matches the torch reference (cosine), for fp8, int8,
    and W4A8 (int4 weight) inputs.

    Covers atomic (accumulate=True) and reduce (accumulate=False) modes, all
    supported output dtypes, and both decode-ish (tile_m=16) and prefill-ish
    (tile_m=64) paths. int8/int4 share the fp8 pipeline (i32 MFMA acc, f32 dequant);
    int4 swaps the weight load for an explicit ki-correct preshuffle loader.
    """
    got, ref = _run_gemm2(
        tokens=tokens,
        **_SHAPE,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        out_dtype=out_dtype,
        accumulate=accumulate,
        in_dtype=in_dtype,
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, f"stage2 cos={cos:.5f} ({mode}, in_dtype={in_dtype}, out_dtype={out_dtype}, tile_m={tile_m})"


def _run_gemm2_int8smooth(*, tokens, model_dim, inter_dim, experts, topk, tile_m, out_dtype, accumulate, seed=0):
    """Stage2 int8smooth: the smooth scale is applied host-side to A2 before quant,
    so the kernel sees plain int8 A2 + a per-route scale -- identical to the int8
    path in gemm2. This exercises that in_dtype='int8smooth' routes to that path."""
    device = torch.device("cuda")
    doweight_stage2 = True
    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)

    s = 0.2
    x_fp32 = torch.rand((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.rand((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(model_dim)
    )
    w2_fp32 = torch.rand((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(inter_dim)
    )

    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
    w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
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
        doweight_stage1=False,
    )  # [tokens, topk, inter_dim] fp32

    # int8smooth: per-expert smooth scale folded into A2 before per-(token,slot) quant.
    smooth2 = 0.75 + 0.5 * torch.rand((experts, inter_dim), device=device, dtype=torch.float32)
    a2_in = out1_ref * smooth2[topk_ids.to(torch.int64)]
    a2_q, a2_scale = pertoken_quant(a2_in, quant_dtype=torch.int8)

    w2_kernel = shuffle_weight(w2_q).view(experts * model_dim, inter_dim).contiguous().view(-1)
    w2_scale_flat = scale_w2.view(experts * model_dim, 1)

    out_dt = _out_torch_dtype(out_dtype)
    out = torch.zeros((tokens, model_dim) if accumulate else (tokens * topk, model_dim), device=device, dtype=out_dt)

    exe = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        doweight_stage2=doweight_stage2,
        out_dtype=out_dtype,
        accumulate=accumulate,
        in_dtype="int8smooth",
    )
    _launch(
        exe,
        out,
        x=a2_q,
        w=w2_kernel,
        scale_x=a2_scale,
        scale_w=w2_scale_flat,
        routing=routing,
        dim0=model_dim,
        dim1=inter_dim,
        tokens=tokens,
        zero_out=True,
    )

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
    got = out if accumulate else out.view(tokens, topk, model_dim).sum(dim=1)
    return got, ref


@_requires_fp8
@pytest.mark.parametrize("accumulate,out_dtype", [(True, "f16"), (True, "bf16")])
@pytest.mark.parametrize("tile_m,tokens", [(16, 8), (64, 128)])
def test_moe_gemm2_int8smooth_numeric(accumulate, out_dtype, tile_m, tokens):
    """Stage2 int8smooth matches the torch reference. int8smooth shares the int8
    stage2 path (smooth scale is baked into A2 host-side); this checks the dtype
    routes correctly."""
    got, ref = _run_gemm2_int8smooth(
        tokens=tokens,
        **_SHAPE,
        tile_m=tile_m,
        out_dtype=out_dtype,
        accumulate=accumulate,
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, f"stage2 int8smooth cos={cos:.5f} ({mode}, out_dtype={out_dtype}, tile_m={tile_m})"


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


# ===========================================================================
# TASK A: broadened shape coverage.
#
# Constraints are DERIVED from the kernel builders (gemm1.py / gemm2.py asserts):
#   gemm1: K=model_dim; tile_k in (128,256); model_dim % tile_k == 0 AND
#          (model_dim//tile_k) even; tile_n in [64,256] %64; tile_m in [16,256]
#          %16; contiguous_n = max(tile_n//2, 64); inter_dim % contiguous_n == 0.
#   gemm2: K=inter_dim; tile_k in (128,256); inter_dim % tile_k == 0; tile_n in
#          [64,256] %64; tile_m %16; contiguous_n = tile_n; model_dim % tile_n == 0;
#          epilogue: tile_m % 8 == 0 (row-thread grid) -- implied by %16.
# Invalid combinations are skipped with an explicit reason instead of dropped.
# ===========================================================================


def _skip_if_invalid_gemm1(*, model_dim, inter_dim, tile_m, tile_n, tile_k):
    if tile_k not in (128, 256):
        pytest.skip(f"gemm1 requires tile_k in (128,256), got {tile_k}")
    if model_dim % tile_k != 0 or (model_dim // tile_k) % 2 != 0:
        pytest.skip(f"gemm1 requires model_dim({model_dim}) an EVEN multiple of tile_k({tile_k})")
    if not (64 <= tile_n <= 256 and tile_n % 64 == 0):
        pytest.skip(f"gemm1 requires tile_n in [64,256] multiple of 64, got {tile_n}")
    if not (16 <= tile_m <= 256 and tile_m % 16 == 0):
        pytest.skip(f"gemm1 requires tile_m a 16-multiple in [16,256], got {tile_m}")
    contiguous_n = max(tile_n // 2, 64)
    if inter_dim % contiguous_n != 0:
        pytest.skip(f"gemm1 requires inter_dim({inter_dim}) % contiguous_n({contiguous_n}) == 0")


def _skip_if_invalid_gemm2(*, model_dim, inter_dim, tile_m, tile_n, tile_k):
    if tile_k not in (128, 256):
        pytest.skip(f"gemm2 requires tile_k in (128,256), got {tile_k}")
    if inter_dim % tile_k != 0:
        pytest.skip(f"gemm2 requires inter_dim({inter_dim}) % tile_k({tile_k}) == 0")
    if not (64 <= tile_n <= 256 and tile_n % 64 == 0):
        pytest.skip(f"gemm2 requires tile_n in [64,256] multiple of 64, got {tile_n}")
    if not (16 <= tile_m <= 256 and tile_m % 16 == 0):
        pytest.skip(f"gemm2 requires tile_m a 16-multiple in [16,256], got {tile_m}")
    if model_dim % tile_n != 0:
        pytest.skip(f"gemm2 requires model_dim({model_dim}) % tile_n({tile_n}) == 0")


# Shape coverage. model_dim/inter_dim kept small (128/256) to stay fast; the
# variety lives in tokens (ragged M), experts, topk, and the tile_* triples.
#
# Split into a small FAST core (default CI) and an exhaustive `large_shape` sweep.

# FAST core: (tokens, experts, topk, tile_m, tile_n, tile_k). Each row targets a
# distinct M-stress: ragged tails, padding-heavy routing, topk extremes, tile_k=256.
_FAST_GEMM1 = [
    (1, 8, 1, 16, 64, 128),  # single token, topk=1: almost all sorted slots padding
    (3, 4, 2, 16, 64, 128),  # tiny, ragged vs tile_m=16
    (7, 8, 2, 32, 64, 128),  # ragged tail vs tile_m=32
    (31, 32, 6, 64, 128, 128),  # not a multiple of tile_m=64; many experts, topk=6
    (33, 8, 2, 64, 128, 128),  # just over 32
    (129, 128, 8, 128, 256, 128),  # just over 128; 128 experts, topk=8
]
# gemm2 fast core reuses the same M/tile rows plus a tile_k=256 case (needs inter_dim=256).
_FAST_GEMM2 = _FAST_GEMM1 + [
    (17, 8, 2, 16, 64, 256),  # tile_k=256 path (requires inter_dim % 256 == 0)
]

# Exhaustive (large_shape): full cross of M cases x tile triples x dims.
_M_CASES = [
    (1, 8, 1),
    (3, 4, 2),
    (7, 8, 2),
    (31, 32, 6),
    (33, 8, 2),
    (129, 128, 8),
]
_TILE_CASES = [
    (16, 64, 128),
    (32, 64, 128),
    (64, 128, 128),
    (128, 256, 256),
    (16, 64, 256),
]
_DIM_CASES = [
    (256, 128),  # gemm1 K=256 (even x128); gemm2 K=128
    (256, 256),  # gemm2 K=256 exercises tile_k=256
]


@_requires_fp8
@pytest.mark.parametrize("tokens,experts,topk,tile_m,tile_n,tile_k", _FAST_GEMM1)
def test_moe_gemm1_shapes_fast(tokens, experts, topk, tile_m, tile_n, tile_k):
    """FAST stage1 shape coverage: ragged M tails, varied experts/topk, tile triples.
    One dtype (int8) to stay quick; the dtype x shape cross is in the large_shape sweep."""
    _skip_if_invalid_gemm1(model_dim=256, inter_dim=128, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    out, ref = _run_gemm1(
        tokens=tokens,
        model_dim=256,
        inter_dim=128,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype="bf16",
        in_dtype="int8",
    )
    cos = _cosine_sim(out, ref)
    assert cos > 0.99, f"stage1 fast cos={cos:.5f} (tile=({tile_m},{tile_n},{tile_k}), M=({tokens},{experts},{topk}))"


@_requires_fp8
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("tokens,experts,topk,tile_m,tile_n,tile_k", _FAST_GEMM2)
def test_moe_gemm2_shapes_fast(accumulate, tokens, experts, topk, tile_m, tile_n, tile_k):
    """FAST stage2 shape coverage (atomic + reduce): ragged M tails, varied
    experts/topk, tile triples. inter_dim=256 so tile_k in (128,256) both apply."""
    _skip_if_invalid_gemm2(model_dim=256, inter_dim=256, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    got, ref = _run_gemm2(
        tokens=tokens,
        model_dim=256,
        inter_dim=256,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype="bf16",
        accumulate=accumulate,
        in_dtype="int8",
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, f"stage2 fast cos={cos:.5f} ({mode}, tile=({tile_m},{tile_n},{tile_k}), M=({tokens},{experts},{topk}))"


@pytest.mark.large_shape
@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int4"])
@pytest.mark.parametrize("model_dim,inter_dim", _DIM_CASES)
@pytest.mark.parametrize("tile_m,tile_n,tile_k", _TILE_CASES)
@pytest.mark.parametrize("tokens,experts,topk", _M_CASES)
def test_moe_gemm1_shapes(in_dtype, model_dim, inter_dim, tile_m, tile_n, tile_k, tokens, experts, topk):
    """Exhaustive stage1 shape sweep across all four dtype-capable inputs, dims,
    tile triples, and ragged M cases (large_shape: excluded from fast CI)."""
    _skip_if_invalid_gemm1(model_dim=model_dim, inter_dim=inter_dim, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    out, ref = _run_gemm1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype="bf16",
        in_dtype=in_dtype,
    )
    cos = _cosine_sim(out, ref)
    assert cos > 0.99, (
        f"stage1 shapes cos={cos:.5f} (in={in_dtype}, dims=({model_dim},{inter_dim}), "
        f"tile=({tile_m},{tile_n},{tile_k}), M=({tokens},{experts},{topk}))"
    )


@pytest.mark.large_shape
@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int4"])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("model_dim,inter_dim", _DIM_CASES)
@pytest.mark.parametrize("tile_m,tile_n,tile_k", _TILE_CASES)
@pytest.mark.parametrize("tokens,experts,topk", _M_CASES)
def test_moe_gemm2_shapes(in_dtype, accumulate, model_dim, inter_dim, tile_m, tile_n, tile_k, tokens, experts, topk):
    """Exhaustive stage2 shape sweep (atomic + reduce) across all four dtype-capable
    inputs, dims, tile triples, and ragged M cases (large_shape: excluded from fast CI)."""
    _skip_if_invalid_gemm2(model_dim=model_dim, inter_dim=inter_dim, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    got, ref = _run_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype="bf16",
        accumulate=accumulate,
        in_dtype=in_dtype,
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, (
        f"stage2 shapes cos={cos:.5f} ({mode}, in={in_dtype}, dims=({model_dim},{inter_dim}), "
        f"tile=({tile_m},{tile_n},{tile_k}), M=({tokens},{experts},{topk}))"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
