# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The fp8 MoE layer with 32x32-block weight scales (``compile_moe_fp8_b32``) against an fp64 reference.

Weights are fp8 (E4M3) row-major ``[E, N, K]`` with one f32 scale per 32x32 block, the format vLLM's Triton fused_moe
serves a TP-sharded 128x128-block checkpoint in once it refines the scales; tokens bf16, routes sorted in aiter's
moe_sorting format. Each element of the output must lie within ``RTOL |ref| + ATOL S`` of the fp64 layer on the exact
bf16 tokens and dequantized weights, ``S`` the element's error scale (the root sum of squares of its products, through
SiLU's slopes and the routed weights): the stock upconverts each weight to bf16 once, accumulates in fp32 and adds the
routes into the bf16 output.
"""

import unittest.mock as mock

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.moe_2stage_a16wmix.fp8_b32 import compile_moe_fp8_b32

_ARCH = str(get_rocm_arch())
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _ARCH.startswith("gfx950"), reason=f"fp8 32x32-block MoE needs gfx950, got {_ARCH}"),
]

FP8 = torch.float8_e4m3fn
RTOL, ATOL = 2.0**-7, 2.0**-5


def _sort(ids, weights, experts, block):
    tokens, topk = ids.shape
    local = ids.reshape(-1).long()
    token = torch.arange(tokens).repeat_interleave(topk)
    slot = torch.arange(topk).repeat(tokens)
    counts = torch.bincount(local, minlength=experts)
    padded = (counts + block - 1) // block * block
    start = torch.cumsum(padded, 0) - padded
    order = torch.argsort(local * tokens + token)
    expert = local[order]
    dest = start[expert] + torch.arange(expert.numel()) - (torch.cumsum(counts, 0) - counts)[expert]
    size = -(-(tokens * topk + experts * block - topk) // block) * block
    out_ids = torch.full((size,), (topk << 24) | tokens, dtype=torch.int32)
    out_ids[dest] = ((slot[order] << 24) | token[order]).to(torch.int32)
    out_w = torch.zeros(size, dtype=torch.float32)
    out_w[dest] = weights.reshape(-1)[order].float()
    eids = torch.full((size // block,), -1, dtype=torch.int32)
    owners = torch.repeat_interleave(torch.arange(experts), padded // block)
    eids[: owners.numel()] = owners.to(torch.int32)
    valid = torch.tensor([int(padded.sum()), tokens], dtype=torch.int32)
    return out_ids, out_w, eids, valid


def _experts(experts, rows, cols, gen):
    values = torch.randn((experts, rows, cols), generator=gen, device="cuda")
    blocks = values.view(experts, rows // 32, 32, cols // 32, 32)
    amax = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-12)
    q = (blocks * (448.0 / amax)[:, :, None, :, None]).view(experts, rows, cols).to(FP8)
    spread = torch.exp2(torch.rand((experts, rows // 32, cols // 32), generator=gen, device="cuda") * 4 - 2)
    return q, (amax / 448.0 * spread / cols**0.5).float()


def _dequant(q, scale):
    rows, cols = q.shape
    return (q.double().view(rows // 32, 32, cols // 32, 32) * scale.double()[:, None, :, None]).view(rows, cols)


def _reference(x, w1, s1, w2, s2, ids, weights, inter):
    tokens, model = x.shape
    out = torch.zeros((tokens, model), dtype=torch.float64, device=x.device)
    square = torch.zeros_like(out)
    xd = x.double()
    for expert in torch.unique(ids).tolist():
        token, slot = torch.where(ids == expert)
        a = _dequant(w1[expert], s1[expert])
        y = xd[token] @ a.T
        g, u = y[:, :inter], y[:, inter:]
        h = F.silu(g) * u
        b = _dequant(w2[expert], s2[expert])
        route = weights[token, slot].double()[:, None]
        out.index_add_(0, token, route * (h @ b.T))
        rms = ((xd[token] ** 2) @ (a * a).T).sqrt()
        sig = torch.sigmoid(g)
        dg, du = (sig * (1 + g * (1 - sig)) * u).abs(), F.silu(g).abs()
        sigma = (dg * rms[:, :inter]) ** 2 + (du * rms[:, inter:]) ** 2
        square.index_add_(0, token, route**2 * ((h * h + sigma) @ (b * b).T))
    return out, square.sqrt()


@pytest.mark.parametrize(
    "tokens, model, inter, experts, topk",
    [
        (1, 512, 32, 4, 2),  # the smallest layer
        (7, 2560, 160, 64, 10),  # Qwen3.8-Flash-Next's TP4 rank (160 rows: K of stage 2 padded to its tile)
        (64, 2560, 160, 512, 10),  # its 512 experts, a decode step's 16 verify requests
        (300, 2560, 160, 512, 10),  # a prefill chunk's tail
        (129, 1024, 96, 33, 5),
        (2048, 2560, 160, 512, 10),  # a prefill chunk
        (40, 4096, 256, 16, 4),
        (17, 7168, 512, 9, 3),
    ],
)
def test_moe_fp8_b32(tokens, model, inter, experts, topk):
    block = 16 if tokens <= 64 else 64
    x, w1, s1, w2, s2, ids, weights, sids, sw, eids, valid, out = _build_case(
        tokens, model, inter, experts, topk, block
    )
    launch = compile_moe_fp8_b32(model_dim=model, inter_dim=inter, experts=experts, topk=topk, block_m=block)
    launch(out, x, w1, s1, w2, s2, sids, sw, eids, valid, tokens)
    torch.cuda.synchronize()
    _check(out, x, w1, s1, w2, s2, ids, weights, inter)


def _build_case(tokens, model, inter, experts, topk, block):
    gen = torch.Generator(device="cuda").manual_seed(tokens * 31 + inter)
    x = (
        torch.randn((tokens, model), generator=gen, device="cuda")
        * torch.exp2(torch.rand((tokens, 1), generator=gen, device="cuda") * 6 - 3)
    ).to(torch.bfloat16)
    w1, s1 = _experts(experts, 2 * inter, model, gen)
    w2, s2 = _experts(experts, model, inter, gen)
    score = torch.rand((tokens, experts), generator=gen, device="cuda")
    ids = torch.sort(score, dim=1, descending=True, stable=True).indices[:, :topk].to(torch.int32)
    weights = torch.softmax(torch.rand((tokens, topk), generator=gen, device="cuda"), dim=1)
    sids, sw, eids, valid = (t.cuda() for t in _sort(ids.cpu(), weights.cpu(), experts, block))
    out = torch.full((tokens, model), float("nan"), dtype=torch.bfloat16, device="cuda")
    return x, w1, s1, w2, s2, ids, weights, sids, sw, eids, valid, out


def _check(out, x, w1, s1, w2, s2, ids, weights, inter):
    ref, spread = _reference(x, w1, s1, w2, s2, ids.long(), weights, inter)
    err = (out.double() - ref).abs()
    allowed = RTOL * ref.abs() + ATOL * spread + 2.0**-133
    worst = float((err / allowed).max())
    assert torch.isfinite(out).all()
    assert worst <= 1.0, f"worst error {worst:.3f} of the bound"


def test_moe_fp8_b32_tail_mask():
    """A dirty intermediate's K padding must not leak through gemm2's A tail.

    block_m=32 leaves the padded K columns uninitialised (only block_m=16 with padded K zeroes them from the
    stage-1 grid), so filling every ``torch.empty`` bf16 intermediate with NaN exercises the gemm2 A-tail mask:
    without it, bf16 0 (the zeroed weight) x NaN reaches the output.
    """
    tokens, model, inter, experts, topk, block = 64, 2560, 160, 8, 4, 32
    x, w1, s1, w2, s2, ids, weights, sids, sw, eids, valid, out = _build_case(
        tokens, model, inter, experts, topk, block
    )
    launch = compile_moe_fp8_b32(model_dim=model, inter_dim=inter, experts=experts, topk=topk, block_m=block)
    real_empty = torch.empty

    def _poisoned_empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        if isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.bfloat16 and t.dim() == 2:
            t.fill_(float("nan"))
        return t

    with mock.patch.object(torch, "empty", _poisoned_empty):
        launch(out, x, w1, s1, w2, s2, sids, sw, eids, valid, tokens)
    torch.cuda.synchronize()
    _check(out, x, w1, s1, w2, s2, ids, weights, inter)


def test_moe_fp8_b32_noncurrent_stream():
    tokens, model, inter, experts, topk, block = 17, 512, 64, 8, 2, 16
    x, w1, s1, w2, s2, ids, weights, sids, sw, eids, valid, out = _build_case(
        tokens, model, inter, experts, topk, block
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch = compile_moe_fp8_b32(model_dim=model, inter_dim=inter, experts=experts, topk=topk, block_m=block)
    launch(out, x, w1, s1, w2, s2, sids, sw, eids, valid, tokens, stream=stream)
    torch.cuda.synchronize()
    _check(out, x, w1, s1, w2, s2, ids, weights, inter)


def test_moe_fp8_b32_zero_tokens():
    model, inter, experts, topk, block = 512, 32, 4, 2, 16
    gen = torch.Generator(device="cuda").manual_seed(7)
    w1, s1 = _experts(experts, 2 * inter, model, gen)
    w2, s2 = _experts(experts, model, inter, gen)
    x = torch.empty((0, model), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((0, model), dtype=torch.bfloat16, device="cuda")
    empty_i32 = torch.empty((0,), dtype=torch.int32, device="cuda")
    valid = torch.zeros((2,), dtype=torch.int32, device="cuda")
    launch = compile_moe_fp8_b32(model_dim=model, inter_dim=inter, experts=experts, topk=topk, block_m=block)
    launch(out, x, w1, s1, w2, s2, empty_i32, empty_i32.float(), empty_i32, valid, 0)
    assert out.shape == (0, model)
