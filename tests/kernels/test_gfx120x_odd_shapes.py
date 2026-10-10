#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Odd-shape coverage across gfx120x product hosts.

Prefer shapes that are not tile multiples. The kernel zero-fills those tails.
Intentional hard raises stay documented here (paged float head dim, GQA).
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func  # noqa: E402
from kernels.gemm.rdna4_fused_mlp_nmajor import fused_swiglu_mlp_nmajor  # noqa: E402
from kernels.gemm.rdna4_iu4_gemm import iu4_gemm  # noqa: E402
from kernels.gemm.rdna4_scaled_mm_fp8 import scaled_mm_fp8  # noqa: E402
from kernels.gemm.rdna4_w8a16_linear import w8a16_linear  # noqa: E402
from kernels.quant.rdna4_awq_w4a16 import gemv_awq_w4a16  # noqa: E402
from kernels.quant.rdna4_convrot_w4a4 import convrot_w4a4_linear, quantize_convrot_w4a4_weight  # noqa: E402
from kernels.quant.rdna4_int4_codec import pack_int4_row_major  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"odd-shape suite requires gfx120x, got {ARCH}", allow_module_level=True)


def _pack_uint4_row_major(values: torch.Tensor) -> torch.Tensor:
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


@pytest.mark.parametrize("head_dim", [65, 80, 96, 127])
@pytest.mark.parametrize("seq", [33, 64, 77])
def test_fa_bf16_odd_head_dim_and_seq(head_dim: int, seq: int) -> None:
    """Dense FA soft-pads head_dim to next %32 tile in [64, 480]; odd seq is fine."""
    q = torch.randn(1, seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()


def test_fa_head_dim_above_lds_still_raises() -> None:
    q = torch.randn(1, 32, 1, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        flydsl_flash_attn_func(q, q, q, causal=False)


@pytest.mark.parametrize("k", [17, 31, 48, 63])
def test_iu4_gemm_odd_k_pack_aware(k: int) -> None:
    """K not multiple of 16 stays in the kernel. Pack still needs an even K."""
    m, n = 16, 32
    a = torch.randint(-8, 7, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 7, (n, k), device="cuda", dtype=torch.int8)
    # Pack needs even K. The extra nibble is zero and is not a host GEMM pad.
    if k % 2 != 0:
        a = torch.nn.functional.pad(a, (0, 1))
        b = torch.nn.functional.pad(b, (0, 1))
    scale_a = torch.ones(m, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(n, device="cuda", dtype=torch.float32)
    y = iu4_gemm(
        pack_int4_row_major(a),
        pack_int4_row_major(b),
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    assert y.shape == (m, n)
    a0 = a[:, :k]
    b0 = b[:, :k]
    ref = (a0.float() @ b0.float().T) * scale_a.float().unsqueeze(1) * scale_b.float().unsqueeze(0)
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("k", [33, 80, 100])
def test_int8_linear_odd_k(k: int) -> None:
    m, n = 16, 32
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(n, device="cuda", dtype=torch.float32).abs() + 0.01
    y = w8a16_linear(x, weight, weight_scale)
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k", [33, 96])
def test_scaled_mm_fp8_odd_k(k: int) -> None:
    m, n = 32, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    sa = float(a.abs().amax().clamp(min=1e-12) / 448.0)
    sb = float(b.abs().amax().clamp(min=1e-12) / 448.0)
    aq = (a.float() / sa).to(torch.float8_e4m3fn)
    bq = (b.float() / sb).to(torch.float8_e4m3fn)
    y = scaled_mm_fp8(
        aq,
        bq,
        torch.tensor([sa], device="cuda", dtype=torch.float32),
        torch.tensor([sb], device="cuda", dtype=torch.float32),
    )
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k", [48, 100])
def test_awq_odd_k_stays_in_kernel(k: int) -> None:
    """Scales cover ceil(K/G). X and packed W stay at the caller's K."""
    m, n, g = 2, 16, 64
    dtype = torch.bfloat16
    k_target = ((k + g - 1) // g) * g
    groups = k_target // g
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    # Pack logical K (pad one col if odd); host soft-pads to group when scales cover ceil(K/G).
    k_even = k if k % 2 == 0 else k + 1
    codes_e = torch.nn.functional.pad(codes, (0, k_even - k))
    qw = _pack_uint4_row_major(codes_e)
    wscales = (torch.randn((groups, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
    wzeros = torch.randn((groups, n), device="cuda", dtype=dtype) * 0.01
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    y = gemv_awq_w4a16(x, qw, wscales, wzeros, bias=None, group_size=g)
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k", [48, 80])
def test_convrot_odd_k_stays_in_kernel(k: int) -> None:
    m, n = 8, 32
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    qweight, wscales = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)
    y = convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=64)
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k,ffn", [(8, 8), (12, 16), (16, 12)])
def test_fused_swiglu_odd_ffn_or_k(k: int, ffn: int) -> None:
    """Short K or FFN stays in the LDS kernel and matches the eager SwiGLU."""
    from tests.kernels.oracles import reference_swiglu_mlp

    m = 16
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_gate = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
    w_up = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
    w_down = torch.randn(k, ffn, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = fused_swiglu_mlp_nmajor(x, w_gate, w_up, w_down, stream=stream)
    stream.synchronize()
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    assert y.shape == (m, k)
    assert x.shape == (m, k)
    torch.testing.assert_close(y.float(), ref.float(), atol=1.5e-1, rtol=1.5e-1)


def test_gqa_non_divisible_still_raises() -> None:
    q = torch.randn(1, 32, 3, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        flydsl_flash_attn_func(q, k, v, causal=False)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return 1.0 if torch.equal(x, y) else 0.0
    return float(torch.dot(x, y) / denom)


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, floor: float) -> None:
    score = _cos(out, ref)
    if score < floor:
        raise AssertionError(f"{name} cosine {score:.5f} < {floor}")


def test_autotune_boundary_shapes() -> None:
    """Shapes that are not the 64 or 128 tiles.

    Hosts use a fixed 256-thread block. The kernel masks lanes past the real
    size. Each section names itself if it fails.
    """
    import flydsl.compiler as flyc
    from flydsl.compiler.jit_argument import PointerJitArg
    from kernels.gemm.rdna4_int8_linear import int8_linear
    from kernels.gemm.rdna4_mxfp8_block_gemm import mxfp8_block_gemm
    from kernels.norm.adaln_gfx120x import build_adaln_module
    from kernels.norm.rms_rope_gfx120x import build_rms_rope_module
    from kernels.norm.rope_gfx120x import build_rope_module
    from kernels.quant.rdna4_fp8_quant import dequantize_fp8, fp8_quant_direct
    from kernels.quant.rdna4_mxfp8_e8m0 import dequantize_mxfp8_device, quantize_mxfp8_device
    from kernels.quant.rdna4_quantize_int8_rowwise import dequantize_int8_rowwise, quantize_int8_rowwise
    from tests.kernels.oracles import reference_swiglu_mlp

    def _ptr(t: torch.Tensor) -> PointerJitArg:
        import flydsl.expr as fx

        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    def _run(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            raise AssertionError(f"{name} failed: {exc}") from exc

    eps = 1e-6
    stream = torch.cuda.current_stream()

    def adaln(n: int) -> None:
        rows = 3
        x = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16) * 0.1
        shift = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16) * 0.1
        out = torch.empty_like(x)
        launch = build_adaln_module(n, "bfloat16", False)
        args = (x, scale, shift, out, rows, 1, 1, eps, stream)
        compiled = flyc.compile(launch, *args)
        compiled(*args)
        xf = x.float()
        ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        ref = ref * (1.0 + scale.float()) + shift.float()
        _check(f"adaln/n={n}", out, ref, 0.99)

    def rms(hd: int) -> None:
        rows = 3
        x = torch.randn(rows, hd, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(hd, device="cuda", dtype=torch.bfloat16)
        pairs = hd // 2
        freqs = torch.randn(rows, pairs, 2, 2, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        launch = build_rms_rope_module(hd, "bfloat16")
        args = (x, scale, freqs, out, rows, 1, eps, stream)
        compiled = flyc.compile(launch, *args)
        compiled(*args)
        norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * scale.float()
        xp = norm.reshape(rows, pairs, 2)
        f = freqs.float()
        y0 = f[..., 0, 0] * xp[..., 0] + f[..., 0, 1] * xp[..., 1]
        y1 = f[..., 1, 0] * xp[..., 0] + f[..., 1, 1] * xp[..., 1]
        ref = torch.stack((y0, y1), dim=-1).reshape(rows, hd)
        _check(f"rms-rope/hd={hd}", out, ref, 0.98)

    def rope(hd: int, rows: int) -> None:
        pairs = hd // 2
        n_pairs = rows * pairs
        x = torch.randn(rows, hd, device="cuda", dtype=torch.bfloat16)
        freqs = torch.randn(rows, pairs, 2, 2, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        launch = build_rope_module("bfloat16", n_pairs_total=n_pairs)
        args = (
            _ptr(x),
            _ptr(freqs),
            _ptr(out),
            n_pairs,
            pairs,
            1,
            rows,
            1,
            rows,
            1,
            n_pairs,
            stream,
        )
        compiled = flyc.compile(launch, *args)
        compiled(*args)
        xp = x.float().reshape(rows, pairs, 2)
        f = freqs.float()
        y0 = f[..., 0, 0] * xp[..., 0] + f[..., 0, 1] * xp[..., 1]
        y1 = f[..., 1, 0] * xp[..., 0] + f[..., 1, 1] * xp[..., 1]
        ref = torch.stack((y0, y1), dim=-1).reshape_as(x)
        _check(f"rope/hd={hd}/rows={rows}", out, ref, 0.98)

    def rowwise(k: int) -> None:
        x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
        q, scale = quantize_int8_rowwise(x)
        y = dequantize_int8_rowwise(q, scale)
        assert y.shape == x.shape
        _check(f"int8-rowwise/k={k}", y, x, 0.98)

    def fp8_roundtrip(k: int) -> None:
        x = torch.randn(5, k, device="cuda", dtype=torch.bfloat16)
        flat = x.reshape(-1).contiguous()
        q = torch.empty(flat.numel(), device="cuda", dtype=torch.float8_e4m3fn)
        amax = flat.float().abs().amax().clamp(min=1e-6)
        scale = torch.tensor([float(amax / 448.0)], device="cuda", dtype=torch.float32)
        fp8_quant_direct(_ptr(flat), _ptr(q), _ptr(scale), int(flat.numel()), "bfloat16", False, 448.0, 256, 1, stream)
        y = dequantize_fp8(q, scale, out_dtype=torch.bfloat16)
        _check(f"fp8-roundtrip/k={k}", y.reshape_as(x), x, 0.98)

    def scaled(m: int, n: int, k: int, kind: str) -> None:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        cap = 57344.0 if kind == "e5m2" else 448.0
        dt = torch.float8_e5m2 if kind == "e5m2" else torch.float8_e4m3fn
        sa = float(a.float().abs().amax().clamp(min=1e-6) / cap)
        sb = float(b.float().abs().amax().clamp(min=1e-6) / cap)
        aq = (a.float() / sa).to(dt)
        bq = (b.float() / sb).to(dt)
        y = scaled_mm_fp8(
            aq,
            bq,
            torch.tensor([sa], device="cuda"),
            torch.tensor([sb], device="cuda"),
        )
        ref = (aq.float() @ bq.float().T) * sa * sb
        _check(f"scaled-mm/{kind}/{m}x{n}x{k}", y, ref, 0.98)

    def i8(m: int, n: int, k: int) -> None:
        a = torch.randint(-20, 21, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-20, 21, (n, k), device="cuda", dtype=torch.int8)
        xs = torch.rand(m, device="cuda").abs() + 0.05
        ws = torch.rand(n, device="cuda").abs() + 0.05
        y = int8_linear(a, b, xs, ws)
        ref = (a.float() @ b.float().T) * xs[:, None] * ws[None, :]
        _check(f"int8/{m}x{n}x{k}", y, ref, 0.999)

    def mx(m: int, n: int, k: int) -> None:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        aq, asc = quantize_mxfp8_device(a)
        bq, bsc = quantize_mxfp8_device(b)
        y = mxfp8_block_gemm(aq, bq, asc, bsc)
        ref = dequantize_mxfp8_device(aq, asc) @ dequantize_mxfp8_device(bq, bsc).T
        _check(f"mxfp8/{m}x{n}x{k}", y, ref, 0.98)

    def attn(sq: int, sk: int, d: int, hq: int, hkv: int, dtype: torch.dtype) -> None:
        q = torch.randn(1, sq, hq, d, device="cuda", dtype=dtype)
        k = torch.randn(1, sk, hkv, d, device="cuda", dtype=dtype)
        v = torch.randn(1, sk, hkv, d, device="cuda", dtype=dtype)
        out = flydsl_flash_attn_func(q, k, v, causal=False)
        qq = q.float().transpose(1, 2)
        kk = k.float().transpose(1, 2)
        vv = v.float().transpose(1, 2)
        if hkv != hq:
            kk = kk.repeat_interleave(hq // hkv, dim=1)
            vv = vv.repeat_interleave(hq // hkv, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv).transpose(1, 2)
        _check(f"attn/{dtype}/sq={sq}/sk={sk}/d={d}/h={hq}/{hkv}", out, ref, 0.98)

    def swiglu(m: int, k: int, ffn: int) -> None:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
        down = torch.randn(k, ffn, device="cuda", dtype=torch.bfloat16)
        y = fused_swiglu_mlp_nmajor(x, gate, up, down, stream=stream)
        ref = reference_swiglu_mlp(x, gate, up, down)
        _check(f"swiglu/{m}x{k}x{ffn}", y, ref, 0.98)

    for n in (96, 160, 384):
        _run(f"adaln/{n}", lambda n=n: adaln(n))
    for hd in (80, 96):
        _run(f"rms/{hd}", lambda hd=hd: rms(hd))
        _run(f"rope/{hd}", lambda hd=hd: rope(hd, 5))
    for k in (100, 300, 700):
        _run(f"rowwise/{k}", lambda k=k: rowwise(k))
    _run("fp8/100", lambda: fp8_roundtrip(100))
    for shape in ((17, 40, 50), (17, 40, 200), (96, 80, 40), (130, 150, 160)):
        _run(f"scaled/{shape}", lambda shape=shape: scaled(*shape, "e4m3"))
    _run("scaled/e5m2", lambda: scaled(19, 33, 70, "e5m2"))
    _run("int8/odd", lambda: i8(13, 27, 41))
    _run("mxfp8/odd", lambda: mx(17, 23, 96))
    _run("attn/bf16-short", lambda: attn(17, 17, 72, 2, 2, torch.bfloat16))
    _run("attn/bf16-blockm", lambda: attn(129, 129, 96, 2, 2, torch.bfloat16))
    _run("attn/fp16-cross", lambda: attn(97, 200, 64, 2, 2, torch.float16))
    _run("attn/gqa", lambda: attn(20, 48, 80, 4, 2, torch.bfloat16))
    _run("swiglu/odd", lambda: swiglu(7, 24, 40))


def test_empty_mnk_still_raises_on_iu4_native_path() -> None:
    """Empty M raises on private native path and public int4 host (no iu8 demotion)."""
    from kernels.quant.rdna4_convrot_w4a4 import _convrot_w4a4_native_iu4, convrot_w4a4_linear

    x = torch.randn(0, 64, device="cuda", dtype=torch.bfloat16)
    qw = torch.zeros(32, 32, device="cuda", dtype=torch.int8)
    ws = torch.ones(32, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="positive MNK"):
        _convrot_w4a4_native_iu4(x, qw, ws, convrot_groupsize=64)
    with pytest.raises(ValueError, match="positive MNK"):
        convrot_w4a4_linear(x, qw, ws, convrot_groupsize=64, linear_dtype="int4")
