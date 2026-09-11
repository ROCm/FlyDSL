# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Grouped MXFP8 correctness, including expert permutations and K padding."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.fp8_gemm_utils import preshuffle_b, xcd_remap_pid
from kernels.moe.mxfp8_moe_8wave import (
    compile_mxfp8_moe_gemm_8w,
    compile_mxfp8_moe_quant,
    compile_mxfp8_moe_reduce,
    compile_mxfp8_moe_unpack_routes,
)
from tests.kernels.utils.gemm_common_utils import e8m0_to_f32, per_1x32_f8_quant, shuffle_scale_w4

pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(str(get_rocm_arch()) != "gfx950", reason="requires gfx950 scaled MFMA"),
]


def quantize_mxfp8(x):
    """RoundUp scale policy used by AITER's MXFP8 MoE path."""
    blocks = x.float().reshape(-1, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-30) / 448.0))
    scale = torch.pow(2.0, exponent)
    q = (blocks / scale[:, None]).to(torch.float8_e4m3fn).reshape(x.shape)
    return q, (exponent + 127).to(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 32)


def prepare_weights(weight, stage):
    """Test-only packing: logical GGUU weights -> the kernel's storage contract."""
    experts, n, k = weight.shape
    k_pad = (k + 255) // 256 * 256
    weight = torch.nn.functional.pad(weight, (0, k_pad - k))
    q, s = quantize_mxfp8(weight)
    ref = q.float() * e8m0_to_f32(s).repeat_interleave(32, -1)
    if stage == 1:
        q = q.view(experts, 2, n // 32, 16, k_pad).permute(0, 2, 1, 3, 4).contiguous().view_as(q)
    packed = torch.stack([preshuffle_b(w.view(torch.int8)).reshape(n, k_pad) for w in q])
    scales = shuffle_scale_w4(s.reshape(experts * n, -1), experts, stage == 1)
    return packed, scales, ref


@pytest.mark.parametrize("stage,k,n", [(2, 384, 256), (2, 512, 768), (2, 768, 512), (1, 512, 768), (1, 6144, 768)])
@pytest.mark.parametrize("swizzle", [0, 4])
def test_mxfp8_moe_8wave(stage, k, n, swizzle):
    torch.manual_seed(42)
    experts = 3
    # More than one tile per expert, an expert with no routes, and a deliberately
    # nonmonotonic tile map distinguish expert offsets from M tile offsets.
    ids = torch.tensor([2, 2, 0], dtype=torch.int32, device="cuda")
    m = ids.numel() * 256
    k_pad = (k + 255) // 256 * 256
    a = torch.randn(m, k, device="cuda") * 0.1
    a[257:512] = 0
    a = torch.nn.functional.pad(a, (0, k_pad - k))
    aq, sa = per_1x32_f8_quant(a)
    b, sb, bref = prepare_weights(torch.randn(experts, n, k, device="cuda") * 0.1, stage)
    c = torch.full((m, n // 2 if stage == 1 else n), float("nan"), device="cuda", dtype=torch.bfloat16)
    args = (
        aq.view(torch.int8).flatten(),
        b.flatten(),
        c.flatten(),
        shuffle_scale_w4(sa, 1, False).flatten(),
        sb.flatten(),
        ids,
        ids,
        m,
        n,
        torch.cuda.current_stream(),
    )
    compiled = flyc.compile(compile_mxfp8_moe_gemm_8w(K=k_pad, logical_k=k, stage=stage, xcd_swizzle=swizzle), *args)
    compiled(*args)
    snapshot = c.clone()
    for _ in range(3):
        compiled(*args)
        torch.testing.assert_close(c, snapshot, rtol=0, atol=0)
    af = aq.float() * e8m0_to_f32(sa).repeat_interleave(32, -1)
    ref = torch.cat([af[i * 256 : (i + 1) * 256] @ bref[e].T for i, e in enumerate(ids.tolist())])
    if stage == 1:
        gate, up = ref.chunk(2, dim=-1)
        gate = gate.clamp(max=7)
        ref = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)
    torch.testing.assert_close(c, ref.bfloat16(), rtol=0.02, atol=0.015)


@pytest.mark.parametrize("k,gather", [(384, False), (768, False), (6144, True)])
def test_mxfp8_moe_quant(k, gather):
    torch.manual_seed(43)
    m, source_rows = 256, 129
    x = torch.randn(source_rows if gather else m, k, device="cuda", dtype=torch.bfloat16)
    row_map = torch.randperm(m, device="cuda").to(torch.int32)
    row_map[row_map >= source_rows] = -1
    kp = (k + 255) // 256 * 256
    q = torch.empty(m, kp, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty(m, kp // 32, device="cuda", dtype=torch.uint8)
    args = (x.flatten(), q.view(torch.int8).flatten(), scale.flatten(), row_map, m, torch.cuda.current_stream())
    f = flyc.compile(compile_mxfp8_moe_quant(K=k, gather=gather), *args)
    f(*args)
    ref = x[row_map.clamp_min(0).long()] if gather else x
    if gather:
        ref[row_map < 0] = 0
    rq, rs = quantize_mxfp8(torch.nn.functional.pad(ref, (0, kp - k)))
    torch.testing.assert_close(q.float(), rq.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale, shuffle_scale_w4(rs, 1, False), rtol=0, atol=0)


def test_mxfp8_moe_reduce():
    m, n, topk = 31, 256, 5
    inverse = torch.randperm(m * topk, device="cuda").to(torch.int32).reshape(m, topk)
    x = torch.randn(m * topk + 256, n, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(m, topk, device="cuda")
    y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    args = (x.flatten(), y.flatten(), inverse.flatten(), weights.flatten(), m, torch.cuda.current_stream())
    f = flyc.compile(compile_mxfp8_moe_reduce(N=n, topk=topk), *args)
    f(*args)
    ref = (x[inverse.long()].float() * weights.unsqueeze(-1)).sum(1).bfloat16()
    torch.testing.assert_close(y, ref, rtol=0.01, atol=0.015)


@pytest.mark.parametrize("m,n,group", [(341, 3, 4), (342, 3, 4), (345, 3, 4), (37, 33, 3)])
def test_mxfp8_moe_ragged_xcd(m, n, group):
    @flyc.kernel
    def kernel(out: fx.Tensor):
        bm, bn = xcd_remap_pid(m, n, group_m=group, allow_ragged=True)
        out[fx.block_idx.x] = bm * n + bn

    @flyc.jit
    def launch(out: fx.Tensor, stream: fx.Stream):
        kernel(out).launch(grid=(m * n, 1, 1), block=(1, 1, 1), stream=stream)

    out = torch.empty(m * n, device="cuda", dtype=torch.int32)
    args = (out, torch.cuda.current_stream())
    flyc.compile(launch, *args)(*args)
    torch.testing.assert_close(out.sort().values, torch.arange(m * n, device="cuda", dtype=torch.int32))


@pytest.mark.parametrize("k", [512, 6144])
def test_mxfp8_moe_gather_and_scatter_scales(k):
    torch.manual_seed(44)
    tokens, topk, n, rows = 64, 2, 768, 512
    inverse = torch.stack(
        [torch.randperm(tokens, device="cuda"), torch.randperm(tokens, device="cuda") + 256], -1
    ).int()
    row_map = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    row_map[inverse.flatten().long()] = torch.arange(tokens, device="cuda").repeat_interleave(topk).int()
    eids = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.1
    q = torch.empty(tokens, k, device="cuda", dtype=torch.int8)
    sa = torch.full((rows, k // 32), 127, device="cuda", dtype=torch.uint8)
    qargs = (x.flatten(), q.flatten(), sa.flatten(), inverse.flatten(), tokens, torch.cuda.current_stream())
    flyc.compile(compile_mxfp8_moe_quant(K=k, gather=False, scatter_scale_topk=topk), *qargs)(*qargs)
    stage = 1
    b, sb, bf = prepare_weights(torch.randn(2, n, k, device="cuda") * 0.1, stage)
    out = torch.zeros(rows, n // 2, device="cuda", dtype=torch.bfloat16)
    gargs = (
        q.flatten(),
        b.flatten(),
        out.flatten(),
        sa.flatten(),
        sb.flatten(),
        eids,
        row_map,
        rows,
        n,
        torch.cuda.current_stream(),
    )
    builder = compile_mxfp8_moe_gemm_8w(K=k, stage=stage, gather_a=True)
    compiled = flyc.compile(builder, *gargs)
    compiled(*gargs)
    snapshot = out.clone()
    for _ in range(5):
        compiled(*gargs)
        torch.testing.assert_close(out, snapshot, rtol=0, atol=0)
    qr, sr = quantize_mxfp8(x)
    xf = qr.float() * e8m0_to_f32(sr).repeat_interleave(32, -1)
    expected = torch.cat(
        [xf[row_map[:256].clamp_min(0).long()] @ bf[1].T, xf[row_map[256:].clamp_min(0).long()] @ bf[0].T]
    )
    gate, up = expected.chunk(2, -1)
    gate = gate.clamp(max=7)
    expected = (gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)).bfloat16()
    expected[row_map < 0] = 0
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.01)


def test_mxfp8_moe_unpack_routes():
    tokens, topk, rows = 63, 3, 256
    routes = torch.randperm(tokens * topk, device="cuda", dtype=torch.int32)
    packed = torch.full((rows,), (topk << 24) | tokens, device="cuda", dtype=torch.int32)
    packed[: routes.numel()] = ((routes % topk) << 24) | (routes // topk)
    row_map = torch.empty_like(packed)
    inverse = torch.full_like(routes, -1)
    args = (packed, row_map, inverse, rows, tokens, torch.cuda.current_stream())
    flyc.compile(compile_mxfp8_moe_unpack_routes(topk=topk), *args)(*args)
    torch.testing.assert_close(row_map[: routes.numel()], routes // topk)
    assert (row_map[routes.numel() :] == -1).all()
    torch.testing.assert_close(inverse[routes.long()], torch.arange(routes.numel(), device="cuda", dtype=torch.int32))


@pytest.mark.parametrize(
    "stage,k,n,tile", [(2, 384, 256, (256, 256)), (1, 512, 768, (256, 256)), (1, 512, 768, (128, 512))]
)
def test_dynamic_rows_and_weight_stride(stage, k, n, tile):
    torch.manual_seed(54)
    rows, active = 1024, 512
    kp = (k + 255) // 256 * 256
    a = torch.randn(rows, k, device="cuda") * 0.1
    aq, sa = quantize_mxfp8(torch.nn.functional.pad(a, (0, kp - k)))
    b, sb, bref = prepare_weights(torch.randn(2, n, k, device="cuda") * 0.1, stage)
    # AITER stores B with physical K384, while A/scales are padded to K512.
    b = b.view(2, n // 16, kp // 64, 4, 16, 16)[:, :, : k // 64].contiguous().flatten()
    ids = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    valid = torch.tensor([active], device="cuda", dtype=torch.int32)
    out = torch.full((rows, n // 2 if stage == 1 else n), 42.0, device="cuda", dtype=torch.bfloat16)
    args = (
        aq.view(torch.int8).flatten(),
        b,
        out.flatten(),
        shuffle_scale_w4(sa, 1, False).flatten(),
        sb.flatten(),
        ids,
        ids,
        valid,
        rows,
        n,
        torch.cuda.current_stream(),
    )
    fn = flyc.compile(
        compile_mxfp8_moe_gemm_8w(
            K=kp, logical_k=k, b_k=k, stage=stage, dynamic_rows=True, tile_m=tile[0], tile_n=tile[1], swiglu_limit=5.0
        ),
        *args
    )
    fn(*args)
    af = aq.float() * e8m0_to_f32(sa).repeat_interleave(32, -1)
    ref = torch.cat([af[:256] @ bref[1].T, af[256:512] @ bref[0].T])
    if stage == 1:
        gate, up = ref.chunk(2, -1)
        gate = gate.clamp(max=5)
        ref = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-5, 5) + 1)
    torch.testing.assert_close(out[:active], ref.bfloat16(), rtol=0.02, atol=0.015)
    assert (out[active:] == 42).all()
    snapshot = out.clone()
    for _ in range(3):
        fn(*args)
        assert torch.equal(out, snapshot)
    valid.zero_()
    out.fill_(42)
    fn(*args)
    assert (out == 42).all()


def test_sorted_reduce_missing_routes():
    tokens, n, topk = 31, 256, 5
    x = torch.randn(tokens * topk, n, device="cuda", dtype=torch.bfloat16)
    inverse = torch.randperm(tokens * topk, device="cuda").int().reshape(tokens, topk)
    inverse[:, 0] = -1
    weights = torch.rand(tokens * topk, device="cuda")
    out = torch.empty(tokens, n, device="cuda", dtype=torch.bfloat16)
    args = (x.flatten(), out.flatten(), inverse.flatten(), weights, tokens, torch.cuda.current_stream())
    flyc.compile(compile_mxfp8_moe_reduce(N=n, topk=topk, sorted_weights=True), *args)(*args)
    ix = inverse[:, 1:].long()
    ref = (x[ix].float() * weights[ix, None]).sum(1).bfloat16()
    torch.testing.assert_close(out, ref, rtol=0.01, atol=0.015)
