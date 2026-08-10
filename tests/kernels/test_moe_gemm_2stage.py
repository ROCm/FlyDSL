#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Numeric correctness tests for the ``moe_gemm_2stage`` 2-stage kernels.

Covers stage1 (gate-up + silu) and stage2 (down-projection) against torch
references by cosine similarity, for all four input dtypes -- fp8, int8,
int8smooth, and int4 (W4A8) -- on CDNA3 (gfx94*) / CDNA4 (gfx95*). int8smooth is
a distinct stage1 path (slot-major A/scale_x gather) but routes through the int8
path in stage2 (bit-identical), so stage2 pins its dispatch equivalence once
rather than re-sweeping it.

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

    # SIGNED, zero-mean inputs: weights (and the A2 silu output) are genuinely
    # signed in practice. All-positive weights let a within-group nibble reorder
    # barely change the dot-product sum, so an int4 de-interleave bug stays green
    # -- see test_moe_gemm2_int4_perturb. randn spans the full int4 [-8,7] range.
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(inter_dim)
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


def _run_gemm1_any(*, in_dtype, **kw):
    """Dispatch stage1 to the int8smooth slot-major runner or the shared runner.

    int8smooth is a genuinely distinct stage1 path (slot-major A/scale_x gather),
    so it must use ``_run_gemm1_int8smooth``; the other three dtypes share
    ``_run_gemm1``."""
    if in_dtype == "int8smooth":
        return _run_gemm1_int8smooth(**kw)
    return _run_gemm1(in_dtype=in_dtype, **kw)


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


@_requires_fp8
def test_moe_gemm2_int4_perturb():
    """gemm2 W4A8 de-interleave order is load-bearing, mirroring the gemm1 guard.

    gemm1 and gemm2 share ``load_weight_int4_frag`` but use different weight views
    and N/K mappings (gemm1 N=2*inter_dim gate/up; gemm2 N=model_dim, K=inter_dim),
    so gemm1's passing perturb test does NOT prove gemm2's addressing. This runs the
    SAME gemm2 kernel with the correct packing and with the low/high nibble halves of
    every packed byte swapped, and asserts the perturbed cosine collapses.

    The discriminating data is essential: with SIGNED, zero-mean weights (randn,
    spanning the full int4 [-8,7] range) a wrong nibble order collapses cosine to
    ~0. The historical all-positive uniform fixture masked this -- a within-group
    reorder of all-positive values barely changes the dot-product sum, so the swap
    stayed green (cos ~1.0). Cosine is scale-invariant, so this also reports the
    max absolute relative error over non-trivial elements as a stricter check.
    """
    device = torch.device("cuda")
    tokens, tile_m = 128, 64
    model_dim, inter_dim, experts, topk = (_SHAPE[k] for k in ("model_dim", "inter_dim", "experts", "topk"))
    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, 0)

    # Signed weights spanning the full int4 range so a reorder changes the result.
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device) * (1.0 / math.sqrt(model_dim))
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device) * (1.0 / math.sqrt(inter_dim))

    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
    w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)

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
    )
    a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=torch.int8)

    w2_shuf = shuffle_weight(w2_q).view(experts * model_dim, inter_dim).contiguous().view(-1)
    packed_ok = _pack_shuffled_int8_to_packed_int4(w2_shuf)
    # Perturbation: swap low/high nibble of every byte (even/odd de-interleave swap).
    _b = packed_ok.to(torch.int16) & 0xFF
    packed_bad = (((_b & 0xF) << 4) | ((_b >> 4) & 0xF)).to(torch.uint8).to(torch.int8)
    w2_scale_flat = scale_w2.view(experts * model_dim, 1)

    exe = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        doweight_stage2=True,
        out_dtype="bf16",
        accumulate=True,
        in_dtype="int4",
    )

    def _run(packed_w):
        out = torch.zeros((tokens, model_dim), device=device, dtype=torch.bfloat16)
        _launch(
            exe,
            out,
            x=a2_q,
            w=packed_w,
            scale_x=a2_scale,
            scale_w=w2_scale_flat,
            routing=routing,
            dim0=model_dim,
            dim1=inter_dim,
            tokens=tokens,
            zero_out=True,
        )
        return out

    ref = torch_moe_gemm2(
        a2_q,
        w2_q,
        a2_scale,
        scale_w2,
        topk_ids.to(torch.int64),
        topk_weights,
        model_dim=model_dim,
        doweight_stage2=True,
    )

    def _max_rel_err(got):
        g = got.to(torch.float32).reshape(-1)
        r = ref.to(torch.float32).reshape(-1)
        mask = r.abs() >= 0.05 * r.abs().max()  # judge only non-trivial elements
        return float(((g[mask] - r[mask]).abs() / r[mask].abs()).max())

    got_ok, got_bad = _run(packed_ok), _run(packed_bad)
    cos_ok, cos_bad = _cosine_sim(got_ok, ref), _cosine_sim(got_bad, ref)
    mre_ok, mre_bad = _max_rel_err(got_ok), _max_rel_err(got_bad)
    assert cos_ok > 0.99, f"correct W4A8 packing should match: cos={cos_ok:.5f}"
    assert mre_ok < 0.2, f"correct W4A8 packing max-rel-err too high: {mre_ok:.4f}"
    assert cos_bad < 0.5, f"perturbed gemm2 nibble de-interleave should break: cos={cos_bad:.5f} (mre={mre_bad:.4f})"


def _run_gemm2_int8smooth(
    *, tokens, model_dim, inter_dim, experts, topk, tile_m, out_dtype, accumulate, tile_n=64, tile_k=128, seed=0
):
    """Stage2 int8smooth: the smooth scale is applied host-side to A2 before quant,
    so the kernel sees plain int8 A2 + a per-route scale -- identical to the int8
    path in gemm2. This exercises that in_dtype='int8smooth' routes to that path."""
    device = torch.device("cuda")
    doweight_stage2 = True
    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)

    # Signed, zero-mean weights (see _run_gemm2): more realistic and strictly more
    # sensitive to sign/ordering errors than the old all-positive uniform data.
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(inter_dim)
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
        tile_n=tile_n,
        tile_k=tile_k,
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


def _run_gemm2_any(*, in_dtype, **kw):
    """Dispatch stage2 to the int8smooth fixture or the shared runner.

    In stage2 int8smooth is bit-identical to int8 (the kernel routes it through the
    int8 path), but its host-side fixture differs (the smooth scale is folded into
    A2 before quant), so it uses ``_run_gemm2_int8smooth``; the other three dtypes
    share ``_run_gemm2``."""
    if in_dtype == "int8smooth":
        return _run_gemm2_int8smooth(**kw)
    return _run_gemm2(in_dtype=in_dtype, **kw)


@_requires_fp8
@pytest.mark.parametrize("accumulate", [True, False])
def test_moe_gemm2_int8smooth_dispatch_equiv(accumulate):
    """Pin the stage2 int8smooth dispatch equivalence (why it is NOT swept over shapes
    in test_moe_gemm2_shapes_fast): in stage2 the kernel routes int8smooth through the
    int8 path bit-for-bit (gemm2.py: ``is_int8 = in_dtype in ("int8","int8smooth")``).
    The smooth scale is baked into A2 host-side, so the kernel sees plain int8 + a
    per-route scale. This runs the int8smooth fixture and asserts it matches the torch
    reference for a single tile config; the full stage2 dtype coverage lives in the
    (int8) fast shape sweep and the large_shape sweep."""
    got, ref = _run_gemm2_int8smooth(
        tokens=8,
        **_SHAPE,
        tile_m=16,
        out_dtype="bf16",
        accumulate=accumulate,
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, f"stage2 int8smooth dispatch cos={cos:.5f} ({mode})"


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
#
# The fast core is crossed over ALL FOUR input dtypes (fp8, int8, int8smooth,
# int4) so the CI-default selection exercises every code path, not just int8.
# To keep the dtype dimension inside the ~2min fast budget the shape list is
# TRIMMED to a ragged-M core (tokens not a multiple of tile_m -- the M-safety
# that this coverage exists for); the full shape x tile x dim cross lives in the
# `large_shape` sweep below.

_FAST_DTYPES = ["fp8", "int8", "int8smooth", "int4"]
# gemm2 has only THREE distinct code paths: fp8 (E4M3 elem), int8 (i32 acc), int4
# (packed-int4 weight loader). int8smooth routes to the int8 path bit-for-bit
# (gemm2.py: is_int8 = in_dtype in ("int8","int8smooth")), so it is NOT swept over
# shapes here; its dispatch equivalence is pinned once in
# test_moe_gemm2_int8smooth_dispatch_equiv below.
_FAST_DTYPES_GEMM2 = ["fp8", "int8", "int4"]

# FAST core: (tokens, experts, topk, tile_m, tile_n, tile_k). Each row is a ragged
# M tail against its tile_m, spanning tile_m 16/32/64 and both narrow/wide tile_n.
_FAST_GEMM1 = [
    (3, 4, 2, 16, 64, 128),  # tiny, ragged vs tile_m=16
    (7, 8, 2, 32, 64, 128),  # ragged tail vs tile_m=32
    (33, 8, 2, 64, 128, 128),  # just over 32, ragged vs tile_m=64; wide tile_n
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
@pytest.mark.parametrize("in_dtype", _FAST_DTYPES)
@pytest.mark.parametrize("tokens,experts,topk,tile_m,tile_n,tile_k", _FAST_GEMM1)
def test_moe_gemm1_shapes_fast(in_dtype, tokens, experts, topk, tile_m, tile_n, tile_k):
    """FAST stage1 shape coverage across all four input dtypes (fp8, int8,
    int8smooth, int4): ragged M tails against tile_m 16/32/64. int8smooth uses the
    slot-major A/scale_x gather path. The shape list is trimmed (dtype x shape must
    fit the fast budget); the full shape x tile x dim cross is in the large_shape sweep."""
    _skip_if_invalid_gemm1(model_dim=256, inter_dim=128, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    out, ref = _run_gemm1_any(
        in_dtype=in_dtype,
        tokens=tokens,
        model_dim=256,
        inter_dim=128,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype="bf16",
    )
    cos = _cosine_sim(out, ref)
    assert cos > 0.99, (
        f"stage1 fast cos={cos:.5f} (in={in_dtype}, tile=({tile_m},{tile_n},{tile_k}), "
        f"M=({tokens},{experts},{topk}))"
    )


@_requires_fp8
@pytest.mark.parametrize("in_dtype", _FAST_DTYPES_GEMM2)
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("tokens,experts,topk,tile_m,tile_n,tile_k", _FAST_GEMM2)
def test_moe_gemm2_shapes_fast(in_dtype, accumulate, tokens, experts, topk, tile_m, tile_n, tile_k):
    """FAST stage2 shape coverage (atomic + reduce) across the three distinct gemm2
    dtype paths (fp8, int8, int4): ragged M tails, varied experts/topk, tile triples.
    inter_dim=256 so tile_k in (128,256) both apply. int8smooth is deliberately absent
    here -- in stage2 it is bit-identical to int8 (see _FAST_DTYPES_GEMM2), pinned by
    test_moe_gemm2_int8smooth_dispatch_equiv instead of re-swept."""
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
        in_dtype=in_dtype,
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, (
        f"stage2 fast cos={cos:.5f} ({mode}, in={in_dtype}, tile=({tile_m},{tile_n},{tile_k}), "
        f"M=({tokens},{experts},{topk}))"
    )


@pytest.mark.large_shape
@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int8smooth", "int4"])
@pytest.mark.parametrize("model_dim,inter_dim", _DIM_CASES)
@pytest.mark.parametrize("tile_m,tile_n,tile_k", _TILE_CASES)
@pytest.mark.parametrize("tokens,experts,topk", _M_CASES)
def test_moe_gemm1_shapes(in_dtype, model_dim, inter_dim, tile_m, tile_n, tile_k, tokens, experts, topk):
    """Exhaustive stage1 shape sweep across all four dtype-capable inputs (fp8, int8,
    int8smooth, int4), dims, tile triples, and ragged M cases (large_shape: excluded
    from fast CI). int8smooth uses the slot-major A/scale_x gather path."""
    _skip_if_invalid_gemm1(model_dim=model_dim, inter_dim=inter_dim, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    out, ref = _run_gemm1_any(
        in_dtype=in_dtype,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype="bf16",
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
    inputs (fp8, int8, int8smooth, int4), dims, tile triples, and ragged M cases
    (large_shape: excluded from fast CI). int8smooth shares the int8 stage2 path but
    is built from a distinct host-side (smooth-folded A2) fixture, so it is swept here
    (in the fast CI it is pinned once by test_moe_gemm2_int8smooth_dispatch_equiv)."""
    _skip_if_invalid_gemm2(model_dim=model_dim, inter_dim=inter_dim, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    got, ref = _run_gemm2_any(
        in_dtype=in_dtype,
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
    )
    cos = _cosine_sim(got, ref)
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, (
        f"stage2 shapes cos={cos:.5f} ({mode}, in={in_dtype}, dims=({model_dim},{inter_dim}), "
        f"tile=({tile_m},{tile_n},{tile_k}), M=({tokens},{experts},{topk}))"
    )


# ===========================================================================
# TASK B: direct M-dimension out-of-bounds tests.
#
# These target the kernel's M-safety mechanisms head-on rather than incidentally:
#   * num_records_bytes on the buffer tensors (hardware OOB clamp)      -> poison
#   * the sentinel LDS seed sorted_lds[tid]=M over the full 256 slots   -> sentinel
#   * the num_valid_id / num_valid_ids block guard                      -> nvi
#   * the atomic epilogue OOB-redirect for out-of-tile lanes            -> canary
#
# Bug class (a) this family catches, and the bytes-vs-elements class (b):
#   (a) unwritten sorted_lds slots read as 0 -> decode to VALID token0/slot0,
#       escaping the OOB clamp and corrupting the first-routed token. Directly
#       caught by test_moe_gemm1_sentinel_token0 (all four dtypes).
#   (b) bytes-vs-elements in a buffer descriptor's num_records_bytes. The ORIGINAL
#       historical instance was on gemm1's INPUT A descriptor, which under-sized it
#       2x for BF16 input (elem_bytes=2 counted as 1). BF16 input NO LONGER EXISTS
#       in this package: every surviving input dtype (fp8, int8, packed int4) has
#       elem_bytes == 1, so multiplying by elem_bytes is a no-op and that exact bug
#       is now UNREPRODUCIBLE on the input descriptors. The same MISTAKE CLASS still
#       has teeth at a DIFFERENT site -- the atomic OUTPUT descriptor, whose element
#       is 2 bytes (f16/bf16) or 4 bytes (f32): num_records_bytes = tokens * N *
#       out_bytes (gemm2.py). Dropping out_bytes there under-sizes the output
#       descriptor and (verified) turns ~13 tests red while the reduce cases -- which
#       do not use that atomic descriptor -- correctly stay green. That output-side
#       instance is what test_moe_gemm2_output_canary isolates; the input poison
#       tests are combination tripwires, NOT single-factor num_records isolations
#       (see their docstrings).
# ===========================================================================

_CANARY_ROWS = 4  # extra output rows beyond `tokens` used as a write guard region.
_POISON_ROWS = 8  # extra input rows beyond `tokens` filled with poison.


def _prep_gemm1(*, tokens, model_dim, inter_dim, experts, topk, tile_m, seed, in_dtype, out_dtype):
    """Build stage1 kernel inputs + torch reference, exposing tensors for OOB tests.

    For int8smooth the activation X/scale_x are SLOT-MAJOR ([topk*tokens, K] / [topk*
    tokens], row = slot*tokens + token). The returned ``is_smooth`` flag and the
    slot-major ``x_q``/``scale_x`` change how the OOB tests over-allocate poison rows
    (beyond topk*tokens, not tokens); the token-major output ([tokens, topk, inter])
    is identical to the other dtypes, so the canary guard is unchanged."""
    device = torch.device("cuda")
    _qd = _quant_dtype(in_dtype)
    _is_int4 = in_dtype == "int4"
    _is_smooth = in_dtype == "int8smooth"
    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)

    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )

    if _is_smooth:
        d = _build_stage1_int8smooth(x_fp32, w1_fp32, topk_ids)
        ref = torch_moe_gemm1(
            d["x_ref"],
            d["w1_q_flat"],
            d["sx_ref"],
            d["scale_w1_flat"],
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=False,
        )
        exe = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=64,
            tile_k=128,
            doweight_stage1=False,
            out_dtype=out_dtype,
            in_dtype=in_dtype,
        )
        return dict(
            exe=exe,
            is_smooth=True,
            topk=topk,
            x_q=d["x_q"],  # slot-major [topk*tokens, K]
            w=d["w1_kernel"],
            scale_x=d["scale_x_1d"].view(-1, 1),  # slot-major [topk*tokens, 1]
            scale_w=d["scale_w1_flat"],
            routing=routing,
            ref=ref,
            out_dt=_out_torch_dtype(out_dtype),
        )

    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_qd)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_qd, **({"dtypeMax": 7} if _is_int4 else {}))
    w1_shuffled_flat = shuffle_weight(w1_q).view(experts * (2 * inter_dim), model_dim).contiguous()
    if _is_int4:
        w1_shuffled_flat = _pack_shuffled_int8_to_packed_int4(w1_shuffled_flat)
    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)
    x_q = x_q.contiguous().view(tokens, model_dim)

    ref = torch_moe_gemm1(
        x_q,
        w1_q_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=False,
    )
    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        doweight_stage1=False,
        out_dtype=out_dtype,
        in_dtype=in_dtype,
    )
    return dict(
        exe=exe,
        is_smooth=False,
        topk=topk,
        x_q=x_q,
        w=w1_shuffled_flat,
        scale_x=scale_x,  # [tokens, 1]
        scale_w=scale_w1_flat,
        routing=routing,
        ref=ref,
        out_dt=_out_torch_dtype(out_dtype),
    )


def _prep_gemm2(*, tokens, model_dim, inter_dim, experts, topk, tile_m, seed, in_dtype, out_dtype, accumulate):
    """Build stage2 kernel inputs + torch reference, exposing tensors for OOB tests."""
    device = torch.device("cuda")
    _qd = _quant_dtype(in_dtype)
    _is_int4 = in_dtype == "int4"
    _wmax = {"dtypeMax": 7} if _is_int4 else {}
    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, tile_m, device, seed)

    # Signed, zero-mean weights (see _run_gemm2): realistic and sensitive to
    # sign/ordering errors; the OOB tests here also assert numeric cosine.
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(inter_dim)
    )
    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_qd)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_qd, **_wmax)
    w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=_qd, **_wmax)
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
    )
    a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=_qd)  # [tokens, topk, inter], [tokens, topk, 1]
    w2_kernel = shuffle_weight(w2_q).view(experts * model_dim, inter_dim).contiguous().view(-1)
    if _is_int4:
        w2_kernel = _pack_shuffled_int8_to_packed_int4(w2_kernel)
    w2_scale_flat = scale_w2.view(experts * model_dim, 1)

    ref = torch_moe_gemm2(
        a2_q,
        w2_q,
        a2_scale,
        scale_w2,
        topk_ids.to(torch.int64),
        topk_weights,
        model_dim=model_dim,
        doweight_stage2=True,
    )
    exe = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        doweight_stage2=True,
        out_dtype=out_dtype,
        accumulate=accumulate,
        in_dtype=in_dtype,
    )
    return dict(
        exe=exe,
        a2_q=a2_q,  # [tokens, topk, inter]
        a2_scale=a2_scale,  # [tokens, topk, 1] -> flattened row = token*topk + slot
        w=w2_kernel,
        scale_w=w2_scale_flat,
        routing=routing,
        ref=ref,
        out_dt=_out_torch_dtype(out_dtype),
    )


def _canary_fill(t, start_row):
    """Fill rows [start_row:] of a 2D/3D tensor with a distinctive canary pattern
    and return a bit-exact clone of the guard region for later comparison."""
    flat = t.view(t.shape[0], -1)
    guard = flat[start_row:]
    # Distinctive, dtype-safe pattern (bit-identical to reproduce): row-varying ints.
    idx = torch.arange(guard.shape[0], device=t.device).view(-1, 1) + 1
    guard.copy_((idx * 1337 % 251 - 125).to(t.dtype).expand_as(guard))
    return guard.clone()


# ---------------------------------------------------------------------------
# B.1 + B.3: output canary / guard region (+ ragged tail).
# ---------------------------------------------------------------------------
@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int8smooth", "int4"])
@pytest.mark.parametrize("tile_m,tokens", [(16, 7), (64, 33)])
def test_moe_gemm1_output_canary(in_dtype, tile_m, tokens):
    """Stage1 must not write any output row >= tokens. Allocate the [tokens, topk,
    inter] output with _CANARY_ROWS extra token-rows, seed a distinctive pattern in
    the guard region, run, and assert the guard region is BIT-IDENTICAL afterwards.
    tokens is a ragged tail (not a multiple of tile_m) so the last block is partial.
    A larger allocation must not change behavior (record bound derives from tokens).
    Covers all four input dtypes; the output layout is token-major for every dtype
    (int8smooth only changes the slot-major A/scale_x inputs), so the guard is shared."""
    d = _prep_gemm1(
        tokens=tokens,
        model_dim=256,
        inter_dim=128,
        experts=8,
        topk=2,
        tile_m=tile_m,
        seed=0,
        in_dtype=in_dtype,
        out_dtype="bf16",
    )
    topk = 2
    out = torch.zeros((tokens + _CANARY_ROWS, topk, 128), device="cuda", dtype=d["out_dt"])
    guard0 = _canary_fill(out, tokens)
    _launch(
        d["exe"],
        out,
        x=d["x_q"],
        w=d["w"],
        scale_x=d["scale_x"],
        scale_w=d["scale_w"],
        routing=d["routing"],
        dim0=128,
        dim1=256,
        tokens=tokens,
    )
    guard1 = out.view(out.shape[0], -1)[tokens:]
    assert torch.equal(guard1, guard0), "stage1 wrote into the output guard region (row >= tokens)"
    cos = _cosine_sim(out[:tokens], d["ref"])
    assert cos > 0.99, f"stage1 canary numerics cos={cos:.5f} (in={in_dtype}, tile_m={tile_m}, tokens={tokens})"


@_requires_fp8
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int4"])
@pytest.mark.parametrize("tile_m,tokens", [(16, 7), (64, 33)])
def test_moe_gemm2_output_canary(accumulate, in_dtype, tile_m, tokens):
    """Stage2 must not write any output row >= tokens. Covers atomic ([tokens,
    model_dim]) and reduce ([tokens*topk, model_dim]) epilogues: over-allocate the
    guard rows, seed a canary, run, assert bit-identical guard afterwards. Ragged
    tail (tokens % tile_m != 0). Directly exercises the atomic OOB-redirect and the
    num_records_bytes-derived output record bound.

    This is where the bytes-vs-elements mistake class (header bug (b)) has TEETH in
    the current package: the atomic path's num_records_bytes = tokens * N * out_bytes
    (gemm2.py), and out_bytes is 2 (f16/bf16) or 4 (f32). Dropping out_bytes there
    under-sizes the output descriptor and (verified) reddens the atomic cases here
    while the reduce cases -- which don't use that descriptor -- stay green. The old
    input-A instance of the same mistake is unreproducible now (all inputs are 1
    byte); see the TASK B header."""
    topk, model_dim = 2, 256
    d = _prep_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=128,
        experts=8,
        topk=topk,
        tile_m=tile_m,
        seed=0,
        in_dtype=in_dtype,
        out_dtype="bf16",
        accumulate=accumulate,
    )
    n_rows = tokens if accumulate else tokens * topk
    out = torch.zeros((n_rows + _CANARY_ROWS, model_dim), device="cuda", dtype=d["out_dt"])
    # Atomic epilogue accumulates -> valid rows must be pre-zeroed, but DO NOT touch
    # the guard region (that's the canary). So zero valid rows here, seed the guard,
    # and launch with zero_out=False (which would zero the whole buffer).
    out[:n_rows].zero_()
    guard0 = _canary_fill(out, n_rows)
    _launch(
        d["exe"],
        out,
        x=d["a2_q"],
        w=d["w"],
        scale_x=d["a2_scale"],
        scale_w=d["scale_w"],
        routing=d["routing"],
        dim0=model_dim,
        dim1=128,
        tokens=tokens,
        zero_out=False,
    )
    guard1 = out[n_rows:]
    assert torch.equal(guard1, guard0), "stage2 wrote into the output guard region (row >= valid rows)"
    got = out[:tokens] if accumulate else out[: tokens * topk].view(tokens, topk, model_dim).sum(dim=1)
    cos = _cosine_sim(got, d["ref"])
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, f"stage2 canary numerics cos={cos:.5f} ({mode}, in={in_dtype}, tile_m={tile_m}, tokens={tokens})"


@_requires_fp8
def test_moe_gemm2_output_canary_int8smooth():
    """int8smooth output-guard equivalence (why the canary sweep above is not 4x'd for
    it): in stage2 int8smooth routes through the int8 path bit-for-bit, and the output
    canary/OOB-redirect it exercises is entirely dtype-independent. So instead of
    re-sweeping accumulate x tile_m x tokens, this pins ONE atomic case with the
    int8smooth-compiled kernel over the same int8 fixture, asserting the guard still
    holds. _prep_gemm2 builds plain int8 data for in_dtype='int8smooth' (they share
    the stage2 fixture)."""
    tile_m, tokens = 16, 7
    topk, model_dim = 2, 256
    d = _prep_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=128,
        experts=8,
        topk=topk,
        tile_m=tile_m,
        seed=0,
        in_dtype="int8smooth",
        out_dtype="bf16",
        accumulate=True,
    )
    out = torch.zeros((tokens + _CANARY_ROWS, model_dim), device="cuda", dtype=d["out_dt"])
    out[:tokens].zero_()
    guard0 = _canary_fill(out, tokens)
    _launch(
        d["exe"],
        out,
        x=d["a2_q"],
        w=d["w"],
        scale_x=d["a2_scale"],
        scale_w=d["scale_w"],
        routing=d["routing"],
        dim0=model_dim,
        dim1=128,
        tokens=tokens,
        zero_out=False,
    )
    assert torch.equal(out[tokens:], guard0), "stage2 int8smooth wrote into the output guard region"
    assert _cosine_sim(out[:tokens], d["ref"]) > 0.99


# ---------------------------------------------------------------------------
# B.2: input poison beyond M (targets num_records_bytes sizing directly).
# ---------------------------------------------------------------------------
@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int8smooth", "int4"])
@pytest.mark.parametrize("tile_m,tokens", [(16, 7), (64, 33)])
def test_moe_gemm1_input_poison(in_dtype, tile_m, tokens):
    """Fill stage1 input rows beyond the valid A-row count (x AND scale_x) with poison
    and assert the output is BIT-IDENTICAL to a clean-padded run. Any unclamped read
    past M that reaches a valid output row would pull poison and diverge.

    The valid A-row count is ``tokens`` for the token-major dtypes and ``topk*tokens``
    for int8smooth (slot-major A/scale_x); poison rows are appended past that boundary.

    NOTE (measured, see PR report): stage1 M-safety is defense-in-depth -- the A
    num_records clamp, the scale-tensor num_records clamp, AND the output-scatter
    OOB drop (sentinel token id == M scatters to an OOB output row) each
    independently drop padding contributions. Over-sizing A and/or scale_x
    num_records alone does NOT make this test red; the output-scatter guard still
    defends. This test is therefore a regression *tripwire* for the combination, not
    an isolation of num_records_bytes. The bytes-vs-elements descriptor bug (b) is
    isolated by test_moe_gemm2_output_canary (atomic output descriptor)."""
    d = _prep_gemm1(
        tokens=tokens,
        model_dim=256,
        inter_dim=128,
        experts=8,
        topk=2,
        tile_m=tile_m,
        seed=0,
        in_dtype=in_dtype,
        out_dtype="bf16",
    )
    model_dim, topk = 256, 2
    qd = _quant_dtype(in_dtype)
    # int8smooth A/scale_x are slot-major [topk*tokens, ...]; the valid boundary
    # past which poison is appended is topk*tokens, not tokens.
    a_rows = topk * tokens if d["is_smooth"] else tokens

    def _run(poison):
        x = torch.zeros((a_rows + _POISON_ROWS, model_dim), device="cuda", dtype=qd)
        x[:a_rows] = d["x_q"].view(a_rows, model_dim)
        sx = torch.zeros((a_rows + _POISON_ROWS, 1), device="cuda", dtype=torch.float32)
        sx[:a_rows] = d["scale_x"].view(a_rows, 1)
        if poison:
            # int8/int4 activations are int8: poison with saturated magnitudes; fp8
            # tolerates NaN. scale poison uses NaN + huge to amplify any leak.
            if qd == torch.int8:
                x[a_rows:] = 127
            else:
                x[a_rows:] = torch.finfo(qd).max
            sx[a_rows:] = float("nan")
        out = torch.zeros((tokens, topk, 128), device="cuda", dtype=d["out_dt"])
        _launch(
            d["exe"],
            out,
            x=x,
            w=d["w"],
            scale_x=sx,
            scale_w=d["scale_w"],
            routing=d["routing"],
            dim0=128,
            dim1=256,
            tokens=tokens,
        )
        return out.clone()

    clean = _run(poison=False)
    poisoned = _run(poison=True)
    assert torch.equal(clean, poisoned), (
        f"stage1 output changed when padding rows were poisoned -> unclamped read past M "
        f"(in={in_dtype}, tile_m={tile_m}, tokens={tokens})"
    )
    assert _cosine_sim(clean[:tokens], d["ref"]) > 0.99


@_requires_fp8
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int4"])
@pytest.mark.parametrize("tile_m,tokens", [(16, 7), (64, 33)])
def test_moe_gemm2_input_poison(accumulate, in_dtype, tile_m, tokens):
    """Fill stage2 input rows >= tokens (a2 AND a2_scale) with poison and assert the
    output is BIT-IDENTICAL to a clean-padded run. a2 is [tokens, topk, inter]; the
    poison rows are extra tokens.

    NOTE (measured, see PR report): stage2 also has a per-row scale validity guard
    (dequant does `valid = tok < tokens; sc = valid.select(scale, 0)`), so even an
    over-sized A num_records that reads poison contributes 0. Like the stage1 poison
    test this is a regression *tripwire*, not a single-factor num_records isolation;
    the bytes-vs-elements bug (b) is isolated by test_moe_gemm2_output_canary."""
    topk, model_dim, inter_dim = 2, 256, 128
    d = _prep_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=8,
        topk=topk,
        tile_m=tile_m,
        seed=0,
        in_dtype=in_dtype,
        out_dtype="bf16",
        accumulate=accumulate,
    )
    qd = _quant_dtype(in_dtype)

    def _run(poison):
        a2 = torch.zeros((tokens + _POISON_ROWS, topk, inter_dim), device="cuda", dtype=qd)
        a2[:tokens] = d["a2_q"]
        sa = torch.zeros((tokens + _POISON_ROWS, topk, 1), device="cuda", dtype=torch.float32)
        sa[:tokens] = d["a2_scale"]
        if poison:
            if qd == torch.int8:
                a2[tokens:] = 127
            else:
                a2[tokens:] = torch.finfo(qd).max
            sa[tokens:] = float("nan")
        n_rows = tokens if accumulate else tokens * topk
        out = torch.zeros((n_rows, model_dim), device="cuda", dtype=d["out_dt"])
        _launch(
            d["exe"],
            out,
            x=a2,
            w=d["w"],
            scale_x=sa,
            scale_w=d["scale_w"],
            routing=d["routing"],
            dim0=model_dim,
            dim1=inter_dim,
            tokens=tokens,
            zero_out=True,
        )
        return out.clone()

    clean = _run(poison=False)
    poisoned = _run(poison=True)
    assert torch.equal(clean, poisoned), (
        f"stage2 output changed when padding rows were poisoned -> unclamped read past M "
        f"({'atomic' if accumulate else 'reduce'}, in={in_dtype}, tile_m={tile_m}, tokens={tokens})"
    )


@_requires_fp8
def test_moe_gemm2_input_poison_int8smooth():
    """int8smooth input-poison equivalence (why the poison sweep above is not 4x'd for
    it): in stage2 int8smooth routes through the int8 path bit-for-bit, and the A/scale
    num_records clamp + per-row scale validity guard it exercises are dtype-independent.
    Rather than re-sweeping accumulate x tile_m x tokens, this pins ONE atomic case with
    the int8smooth-compiled kernel over the shared int8 fixture."""
    tile_m, tokens = 16, 7
    topk, model_dim, inter_dim = 2, 256, 128
    d = _prep_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=8,
        topk=topk,
        tile_m=tile_m,
        seed=0,
        in_dtype="int8smooth",
        out_dtype="bf16",
        accumulate=True,
    )

    def _run(poison):
        a2 = torch.zeros((tokens + _POISON_ROWS, topk, inter_dim), device="cuda", dtype=torch.int8)
        a2[:tokens] = d["a2_q"]
        sa = torch.zeros((tokens + _POISON_ROWS, topk, 1), device="cuda", dtype=torch.float32)
        sa[:tokens] = d["a2_scale"]
        if poison:
            a2[tokens:] = 127
            sa[tokens:] = float("nan")
        out = torch.zeros((tokens, model_dim), device="cuda", dtype=d["out_dt"])
        _launch(
            d["exe"],
            out,
            x=a2,
            w=d["w"],
            scale_x=sa,
            scale_w=d["scale_w"],
            routing=d["routing"],
            dim0=model_dim,
            dim1=inter_dim,
            tokens=tokens,
            zero_out=True,
        )
        return out.clone()

    assert torch.equal(_run(poison=False), _run(poison=True)), "stage2 int8smooth output changed under poison"


# ---------------------------------------------------------------------------
# B.4: sentinel-row regression guard (the exact token-0 corruption case).
# ---------------------------------------------------------------------------
def _routing_token0_with_padding(tokens, experts, topk, tile_m, device):
    """Deterministic routing where token 0 is routed with packed sorted id == 0
    (token_id==0 AND slot==0) and padding rows are present. This is the exact case
    where an un-seeded LDS slot (decoding to token0/slot0) would corrupt token 0.

    Assign every token's slot-0 route to expert 0 (so token 0 is the FIRST entry of
    expert 0's block -> packed id 0), and spread the remaining slots across other
    experts. Because tokens < tile_m for the small cases, each expert block has real
    tokens + padding rows in the same block."""
    topk_ids = torch.zeros((tokens, topk), dtype=torch.int64, device=device)
    for t in range(tokens):
        topk_ids[t, 0] = 0  # slot 0 -> expert 0 for all tokens (token 0 lands first)
        for s in range(1, topk):
            topk_ids[t, s] = (1 + (t + s) % max(1, experts - 1)) % experts
    # Deduplicate within a token (topk experts must be distinct); nudge collisions.
    for t in range(tokens):
        seen = set()
        for s in range(topk):
            e = int(topk_ids[t, s])
            while e in seen:
                e = (e + 1) % experts
            topk_ids[t, s] = e
            seen.add(e)
    topk_weights = torch.full((tokens, topk), 1.0 / topk, dtype=torch.float32, device=device)
    routing = _build_routing(topk_ids, topk_weights, experts=experts, tile_m=tile_m)
    # Confirm the construction: packed id 0 must appear in the sorted buffer.
    sorted_token_ids = routing[0]
    assert (sorted_token_ids == 0).any().item(), "routing did not place packed sorted id 0 (token0/slot0)"
    return topk_ids, topk_weights, routing


@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int8smooth", "int4"])
@pytest.mark.parametrize("tile_m", [16, 64])
def test_moe_gemm1_sentinel_token0(in_dtype, tile_m):
    """Sentinel-row regression: token 0 IS routed (packed id 0) with padding rows in
    the same block. Un-seeded sorted_lds slots decode to token0/slot0 and would pile
    garbage onto token 0's output. The sentinel seed (sorted_lds[tid]=M over the full
    256-slot range) must keep token 0 correct. Covered for all four dtypes here
    (fp8, int8, int8smooth, int4); the original catch only hit one. int8smooth uses
    the slot-major A/scale_x gather but the sentinel decode/scatter it guards is
    dtype-independent."""
    device = torch.device("cuda")
    model_dim, inter_dim, experts, topk = 256, 128, 8, 2
    tokens = 6  # < tile_m for both 16 and 64: real tokens + padding share a block
    _qd = _quant_dtype(in_dtype)
    _is_int4 = in_dtype == "int4"
    _is_smooth = in_dtype == "int8smooth"
    topk_ids, topk_weights, routing = _routing_token0_with_padding(tokens, experts, topk, tile_m, device)

    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )
    if _is_smooth:
        sm = _build_stage1_int8smooth(x_fp32, w1_fp32, topk_ids)
        x_launch = sm["x_q"]  # slot-major [topk*tokens, K]
        w1_shuf = sm["w1_kernel"]
        scale_x_launch = sm["scale_x_1d"]
        scale_w1_flat = sm["scale_w1_flat"]
        # token-major references for torch_moe_gemm1
        x_ref, sx_ref, w1_q_flat = sm["x_ref"], sm["sx_ref"], sm["w1_q_flat"]
    else:
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_qd)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_qd, **({"dtypeMax": 7} if _is_int4 else {}))
        w1_shuf = shuffle_weight(w1_q).view(experts * (2 * inter_dim), model_dim).contiguous()
        if _is_int4:
            w1_shuf = _pack_shuffled_int8_to_packed_int4(w1_shuf)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
        scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)
        x_q = x_q.contiguous().view(tokens, model_dim)
        x_launch, scale_x_launch = x_q, scale_x
        x_ref, sx_ref = x_q, scale_x

    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.bfloat16)
    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=64,
        tile_k=128,
        doweight_stage1=False,
        out_dtype="bf16",
        in_dtype=in_dtype,
    )
    _launch(
        exe,
        out,
        x=x_launch,
        w=w1_shuf,
        scale_x=scale_x_launch,
        scale_w=scale_w1_flat,
        routing=routing,
        dim0=inter_dim,
        dim1=model_dim,
        tokens=tokens,
    )
    ref = torch_moe_gemm1(
        x_ref,
        w1_q_flat,
        sx_ref,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=False,
    )
    # Token 0 specifically must be clean (the sentinel bug corrupted exactly this row).
    cos_tok0 = _cosine_sim(out[0], ref[0])
    cos_all = _cosine_sim(out, ref)
    assert cos_tok0 > 0.99, f"token-0 corrupted (sentinel guard): cos={cos_tok0:.5f} (in={in_dtype}, tile_m={tile_m})"
    assert cos_all > 0.99, f"stage1 sentinel cos={cos_all:.5f} (in={in_dtype}, tile_m={tile_m})"


# ---------------------------------------------------------------------------
# B.5: num_valid_ids shorter than the padded sorted buffer (block guard / EP path).
# ---------------------------------------------------------------------------
@_requires_fp8
@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "int4"])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("tile_m,tokens", [(16, 8), (64, 40)])
def test_moe_gemm2_num_valid_ids_guard(in_dtype, accumulate, tile_m, tokens):
    """num_valid_ids shorter than the padded sorted buffer must gate whole M-blocks
    off (gemm2 guard: e_idx*BM < num_valid_id). Force a truncated num_valid_ids so
    the trailing sorted M-blocks are non-valid, plus poison those trailing sorted
    ids to VALID-looking tokens. If the block guard is honored the poisoned trailing
    blocks are never processed and the output matches the clean reference; if it
    leaks, those tokens get spurious atomic contributions.

    Swept over the three distinct gemm2 dtype paths (fp8, int8, int4); int8smooth is
    bit-identical to int8 in stage2, so its equivalence is pinned once by
    test_moe_gemm2_num_valid_ids_guard_int8smooth rather than re-swept."""
    model_dim, inter_dim, experts, topk = 256, 128, 8, 2
    d = _prep_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        seed=0,
        in_dtype=in_dtype,
        out_dtype="bf16",
        accumulate=accumulate,
    )
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks = d["routing"]

    # Poison the sorted ids beyond num_valid_ids to VALID-looking tokens (token 1,
    # slot 0). The block guard must keep them from being processed.
    nvi = int(num_valid_ids[0].item())
    poisoned_ids = sorted_token_ids.clone()
    if nvi < poisoned_ids.numel():
        poisoned_ids[nvi:] = 1  # packed: token 1, slot 0 -> would be a real write if leaked
    routing_poison = (poisoned_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks)

    n_rows = tokens if accumulate else tokens * topk
    out = torch.zeros((n_rows, model_dim), device="cuda", dtype=d["out_dt"])
    _launch(
        d["exe"],
        out,
        x=d["a2_q"],
        w=d["w"],
        scale_x=d["a2_scale"],
        scale_w=d["scale_w"],
        routing=routing_poison,
        dim0=model_dim,
        dim1=inter_dim,
        tokens=tokens,
        zero_out=True,
    )
    got = out[:tokens] if accumulate else out.view(tokens, topk, model_dim).sum(dim=1)
    cos = _cosine_sim(got, d["ref"])
    mode = "atomic" if accumulate else "reduce"
    assert cos > 0.99, (
        f"stage2 num_valid_ids block guard leaked poisoned trailing blocks: cos={cos:.5f} "
        f"({mode}, in={in_dtype}, tile_m={tile_m}, tokens={tokens})"
    )


@_requires_fp8
def test_moe_gemm2_num_valid_ids_guard_int8smooth():
    """int8smooth num_valid_ids block-guard equivalence: in stage2 int8smooth routes
    through the int8 path bit-for-bit, and the num_valid_ids block guard is
    dtype-independent, so it is pinned once here rather than re-swept in
    test_moe_gemm2_num_valid_ids_guard. _prep_gemm2 builds plain int8 data for
    in_dtype='int8smooth'."""
    tile_m, tokens = 16, 8
    model_dim, inter_dim, experts, topk = 256, 128, 8, 2
    d = _prep_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        seed=0,
        in_dtype="int8smooth",
        out_dtype="bf16",
        accumulate=True,
    )
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks = d["routing"]
    nvi = int(num_valid_ids[0].item())
    poisoned_ids = sorted_token_ids.clone()
    if nvi < poisoned_ids.numel():
        poisoned_ids[nvi:] = 1
    routing_poison = (poisoned_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks)

    out = torch.zeros((tokens, model_dim), device="cuda", dtype=d["out_dt"])
    _launch(
        d["exe"],
        out,
        x=d["a2_q"],
        w=d["w"],
        scale_x=d["a2_scale"],
        scale_w=d["scale_w"],
        routing=routing_poison,
        dim0=model_dim,
        dim1=inter_dim,
        tokens=tokens,
        zero_out=True,
    )
    cos = _cosine_sim(out[:tokens], d["ref"])
    assert cos > 0.99, f"stage2 int8smooth num_valid_ids block guard leaked: cos={cos:.5f}"


# ---------------------------------------------------------------------------
# Benchmark CLI (run_benchmark.sh driver).
#
# When invoked with args (e.g. `python test_moe_gemm_2stage.py --in_dtype fp8
# -dim 8192,8192 -t 32768 ...`) this file becomes a benchmark harness for the
# moe_gemm_2stage package: it times stage1, and stage2 in both atomic and reduce
# modes, printing the log lines that scripts/run_benchmark.sh already parses
# (see _emit_moe_s2_rows and the stage1 grep block in that script). With no args
# it falls back to pytest so `pytest` and direct invocation both keep working.
# ---------------------------------------------------------------------------


def _flat_or_2d(t):
    """Flatten to 1D, but keep a 2D contiguous view when numel exceeds int32.

    The kernel consumes weight/activation operands via ``fx.get_iter`` (raw data
    pointer), so tensor shape is irrelevant to correctness. The JIT ABI, however,
    packs each shape dim as int32 (MemRefSpec, jit_argument.py). A 1D ``view(-1)``
    of the big-shape weight (experts*2*inter_dim*model_dim > 2^31 elements) would
    overflow that i32 slot, so fall back to a 2D contiguous shape whose per-dim
    extents stay < 2^31."""
    if t.numel() <= 2_147_483_647:
        return t.contiguous().view(-1)
    flat = t.contiguous().view(-1)
    n = flat.numel()
    # Factor into two < 2^31 dims. model_dim / inter_dim divide the weight numel,
    # so a stride-1 last dim of 8192 keeps both extents small; fall back to a
    # square-ish split if that does not divide.
    for d1 in (8192, 4096, 2048, 1024, 512, 256, 128):
        if n % d1 == 0 and (n // d1) < 2_147_483_647:
            return flat.view(n // d1, d1)
    import math as _m

    d1 = 1 << (int(_m.isqrt(n)).bit_length())
    while n % d1 != 0:
        d1 //= 2
    return flat.view(n // d1, d1)


def _bench_args(out, *, x, w, scale_x, scale_w, routing, dim0, dim1, tokens):
    """Positional launch-arg tuple, mirroring _launch's _args (line ~167)."""
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, blocks = routing
    return (
        out,
        _flat_or_2d(x),
        _flat_or_2d(w),
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


def _time_launch(compiled, args, *, num_warmup, num_iters):
    """Median-free mean of CUDA-event-timed launches; returns microseconds."""
    for _ in range(int(num_warmup)):
        compiled(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(num_iters)):
        compiled(*args)
    end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    return (ms_total / max(int(num_iters), 1)) * 1e3  # us


def _bytes_stage1(*, tokens, model_dim, inter_dim, experts, topk, in_dtype, out_dt):
    """Stage1 traffic: gathered A (per routed row) + UNIQUE W + output.

    Weights are counted once per expert, not per routed row: a tile-block reads an
    expert's weights once and shares them across its rows. Counting them per row
    inflates TB/s by orders of magnitude. Matches the convention in test_moe_gemm.py.
    """
    rows = tokens * topk
    a_elt = 1.0  # activations are int8/fp8 (1B) for every supported dtype
    w_elt = 0.5 if in_dtype == "int4" else 1.0  # W4A8 packs 2 weights per byte
    a_bytes = rows * model_dim * a_elt
    w_bytes = experts * (2 * inter_dim) * model_dim * w_elt
    o_bytes = rows * inter_dim * out_dt.itemsize
    return a_bytes + w_bytes + o_bytes


def _bytes_stage2(*, tokens, model_dim, inter_dim, experts, topk, in_dtype, out_dt):
    """Stage2 traffic: A2 (per routed row) + UNIQUE W2 + output. See _bytes_stage1."""
    rows = tokens * topk
    a_elt = 1.0
    w_elt = 0.5 if in_dtype == "int4" else 1.0
    a_bytes = rows * inter_dim * a_elt
    w_bytes = experts * model_dim * inter_dim * w_elt
    o_bytes = rows * model_dim * out_dt.itemsize
    return a_bytes + w_bytes + o_bytes


def _bench_stage1(args, *, dtype_tag):
    """Build fixture (mirrors _prep_gemm1 with CLI tiles), time, print log line."""
    tokens = args.tokens
    model_dim, inter_dim = args.model_dim, args.inter_dim
    experts, topk = args.experts, args.topk
    device = torch.device("cuda")
    in_dtype = args.in_dtype
    out_dtype = args.out_dtype
    _qd = _quant_dtype(in_dtype)
    _is_int4 = in_dtype == "int4"
    _is_smooth = in_dtype == "int8smooth"

    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, args.tile_m, device, args.seed)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )

    if _is_smooth:
        d = _build_stage1_int8smooth(x_fp32, w1_fp32, topk_ids)
        x_q = d["x_q"]
        w = d["w1_kernel"]
        scale_x = d["scale_x_1d"].view(-1, 1)
        scale_w = d["scale_w1_flat"]
        ref = torch_moe_gemm1(
            d["x_ref"],
            d["w1_q_flat"],
            d["sx_ref"],
            d["scale_w1_flat"],
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=False,
        )
    else:
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_qd)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_qd, **({"dtypeMax": 7} if _is_int4 else {}))
        w = shuffle_weight(w1_q).view(experts * (2 * inter_dim), model_dim).contiguous()
        if _is_int4:
            w = _pack_shuffled_int8_to_packed_int4(w)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
        scale_w = scale_w1.view(experts * (2 * inter_dim), 1)
        x_q = x_q.contiguous().view(tokens, model_dim)
        ref = torch_moe_gemm1(
            x_q,
            w1_q_flat,
            scale_x,
            scale_w,
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=False,
        )

    out_dt = _out_torch_dtype(out_dtype)
    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=out_dt)
    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        doweight_stage1=False,
        out_dtype=out_dtype,
        in_dtype=in_dtype,
    )
    launch_args = _bench_args(
        out,
        x=x_q,
        w=w,
        scale_x=scale_x,
        scale_w=scale_w,
        routing=routing,
        dim0=inter_dim,
        dim1=model_dim,
        tokens=tokens,
    )
    compiled = flyc.compile(exe, *launch_args)

    cos = float("nan")
    if not args.skip_ref:
        compiled(*launch_args)
        torch.cuda.synchronize()
        cos = _cosine_sim(out, ref)

    us = _time_launch(compiled, launch_args, num_warmup=args.num_warmup, num_iters=args.num_iters)
    flops = 2.0 * (tokens * topk) * (2 * inter_dim) * model_dim
    tb = _bytes_stage1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        out_dt=out_dt,
    )
    tflops = flops / (us * 1e-6) / 1e12
    tbps = tb / (us * 1e-6) / 1e12
    print(
        f"FlyDSL MoE stage1[{dtype_tag}]: cos={cos:.5f} | "
        f"t{tokens}-d{model_dim}x{inter_dim}-e{experts}k{topk} | "
        f"{us:.1f} us, {tflops:.2f} TFLOPS, {tbps:.3f} TB/s"
    )


def _bench_stage2(args, *, dtype_tag, accumulate):
    """Build fixture (mirrors _prep_gemm2 with CLI tiles), time, print log line."""
    tokens = args.tokens
    model_dim, inter_dim = args.model_dim, args.inter_dim
    experts, topk = args.experts, args.topk
    device = torch.device("cuda")
    in_dtype = args.in_dtype
    out_dtype = args.out_dtype
    _qd = _quant_dtype(in_dtype)
    _is_int4 = in_dtype == "int4"
    _wmax = {"dtypeMax": 7} if _is_int4 else {}

    topk_ids, topk_weights, routing = _make_routing(tokens, experts, topk, args.tile_m, device, args.seed)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(model_dim)
    )
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        1.0 / math.sqrt(inter_dim)
    )
    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=_qd)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=_qd, **_wmax)
    w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=_qd, **_wmax)
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
        tile_m=args.tile_m,
        tile_n=args.tile_n2,
        tile_k=args.tile_k2,
        doweight_stage2=True,
        out_dtype=out_dtype,
        accumulate=accumulate,
        in_dtype=in_dtype,
    )
    launch_args = _bench_args(
        out,
        x=a2_q,
        w=w2_kernel,
        scale_x=a2_scale,
        scale_w=w2_scale_flat,
        routing=routing,
        dim0=model_dim,
        dim1=inter_dim,
        tokens=tokens,
    )
    compiled = flyc.compile(exe, *launch_args)

    cos = float("nan")
    if not args.skip_ref:
        ref = torch_moe_gemm2(
            a2_q,
            w2_q,
            a2_scale,
            scale_w2,
            topk_ids.to(torch.int64),
            topk_weights,
            model_dim=model_dim,
            doweight_stage2=True,
        )
        out.zero_()
        compiled(*launch_args)
        torch.cuda.synchronize()
        got = out if accumulate else out.view(tokens, topk, model_dim).sum(dim=1)
        cos = _cosine_sim(got, ref)

    us = _time_launch(compiled, launch_args, num_warmup=args.num_warmup, num_iters=args.num_iters)
    flops = 2.0 * (tokens * topk) * model_dim * inter_dim
    tb = _bytes_stage2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        out_dt=out_dt,
    )
    tflops = flops / (us * 1e-6) / 1e12
    tbps = tb / (us * 1e-6) / 1e12
    mode = "atomic" if accumulate else "reduce"
    shape = f"t{tokens}-d{model_dim}x{inter_dim}-e{experts}k{topk}"
    print(
        f"FlyDSL MoE stage2 [moe_gemm2] {dtype_tag} {mode} | {shape} | "
        f"cos={cos:.5f} | {us:.1f} us, {tflops:.2f} TFLOPS, {tbps:.3f} TB/s"
    )


def _bench_main(argv):
    import argparse

    torch.set_default_device("cuda")

    def _dim(v):
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid -dim {v!r}; expected 'model_dim,inter_dim'")
        return int(parts[0]), int(parts[1])

    p = argparse.ArgumentParser(description="Benchmark the moe_gemm_2stage package (stage1 + stage2 atomic/reduce).")
    p.add_argument("--in_dtype", type=str, default="fp8", choices=["fp8", "int8", "int8smooth", "int4"])
    p.add_argument("-dim", dest="dim", type=_dim, default=(256, 128), help="model_dim,inter_dim (e.g. -dim 8192,8192)")
    p.add_argument("-t", "--tokens", dest="tokens", type=int, default=32)
    p.add_argument("-e", "--experts", dest="experts", type=int, default=8)
    p.add_argument("-k", "--topk", dest="topk", type=int, default=2)
    p.add_argument("--tile_m", type=int, default=64, help="Stage1+stage2 M tile (routing block).")
    p.add_argument("--tile_n", type=int, default=64, help="Stage1 N tile.")
    p.add_argument("--tile_k", type=int, default=128, help="Stage1 K tile.")
    p.add_argument("--tile_n2", type=int, default=None, help="Stage2 N tile (default: tile_n).")
    p.add_argument("--tile_k2", type=int, default=None, help="Stage2 K tile (default: tile_k).")
    p.add_argument("--out_dtype", type=str, default="bf16", choices=["f16", "bf16", "f32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_warmup", type=int, default=10)
    p.add_argument("--num_iters", type=int, default=100)
    p.add_argument("--skip_ref", action="store_true", default=False)
    args = p.parse_args(argv)

    args.model_dim, args.inter_dim = args.dim
    if args.tile_n2 is None:
        args.tile_n2 = args.tile_n
    if args.tile_k2 is None:
        args.tile_k2 = args.tile_k
    dtype_tag = args.in_dtype

    _bench_stage1(args, dtype_tag=dtype_tag)
    # Stage2: atomic then reduce. reduce mode does not support f32 output.
    _bench_stage2(args, dtype_tag=dtype_tag, accumulate=True)
    if args.out_dtype in ("f32", "fp32", "float"):
        print("[skip] stage2 reduce mode does not support out_dtype='f32'")
    else:
        _bench_stage2(args, dtype_tag=dtype_tag, accumulate=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _bench_main(sys.argv[1:])
    else:
        sys.exit(pytest.main([__file__, "-v"]))
