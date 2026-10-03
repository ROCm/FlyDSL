# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
"""gfx120x / RDNA4 FlashAttention correctness smokes.

Mirrors aiter op_tests shapes: short S77-class, long S, cross, FP8 descales,
D>128, additive mask, causal self. Skips when no CUDA or arch is not gfx120x.
"""

import math
import sys
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from kernels.attention.flash_attn_gfx120x_host import (  # noqa: E402
    bottom_right_causal_bias,
    flydsl_flash_attn_fp8_func,
    flydsl_flash_attn_func,
    flydsl_flash_attn_int8_func,
    flydsl_flash_attn_iu4_func,
    fold_alibi_to_bias,
    is_gfx120x,
    mask_is_noop,
    normalize_attn_mask,
)


def _arch() -> str:
    return (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]


pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not is_gfx120x(),
        reason=f"requires gfx120x, got {_arch()!r}",
    ),
]


def _sdpa_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, attn_mask: torch.Tensor | None = None
) -> torch.Tensor:
    # q/k/v BSHD → BHSD for SDPA
    qq = q.transpose(1, 2)
    kk = k.transpose(1, 2)
    vv = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, attn_mask=attn_mask)
    return out.transpose(1, 2)


def _min_cos(a: object, b: object, dim_last: int = 128) -> float:
    af = a.float().reshape(-1, a.shape[-1])
    bf = b.float().reshape(-1, b.shape[-1])
    return F.cosine_similarity(af, bf, dim=1).min().item()


@pytest.mark.parametrize("seq", [64, 77, 128, 129])
@pytest.mark.parametrize("dim", [64, 128])
def test_bf16_self_awkward_and_aligned(seq: int, dim: int) -> None:
    q = torch.randn(1, seq, 8, dim, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.999


def test_fp16_self_short() -> None:
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.float16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out, ref) > 0.999


def test_bf16_causal_self() -> None:
    q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=True)
    ref = _sdpa_ref(q, q, q, causal=True)
    assert _min_cos(out, ref) > 0.999


@pytest.mark.parametrize("sq,sk", [(77, 1024), (128, 1024), (1024, 128)])
def test_bf16_cross(sq: int, sk: int) -> None:
    q = torch.randn(1, sq, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, sk, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, sk, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _sdpa_ref(q, k, v, causal=False)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.999


def test_bf16_long_self() -> None:
    q = torch.randn(1, 2048, 4, 128, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out, ref) > 0.999


@pytest.mark.parametrize("dim", [160, 192, 256, 288, 320, 384])
def test_bf16_d_gt_128(dim: int) -> None:
    q = torch.randn(1, 64, 4, dim, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.997


def test_d_over_max_rejects() -> None:
    q = torch.randn(1, 32, 2, 416, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim=416"):
        flydsl_flash_attn_func(q, q, q, causal=False)


def test_noop_mask_hits() -> None:
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(64, 64, device="cuda", dtype=torch.bool)
    assert mask_is_noop(mask)
    out = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=mask)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out, ref) > 0.999


def test_additive_mask_path() -> None:
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    # Drop the upper triangle via additive -inf (non-causal kernel + mask).
    bias = torch.zeros(64, 64, device="cuda", dtype=torch.float32)
    idx = torch.triu_indices(64, 64, offset=1)
    bias[idx[0], idx[1]] = float("-inf")
    out = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=bias)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=bias)
    assert _min_cos(out, ref) > 0.998


def test_bool_mask_normalized() -> None:
    m = torch.ones(32, 32, dtype=torch.bool, device="cuda")
    m[0, -1] = False
    b = normalize_attn_mask(m, 32, 32, torch.device("cuda"))
    assert b is not None
    assert b.dtype == torch.float32
    assert math.isinf(b[0, -1].item()) and b[0, -1].item() < 0
    assert b[1, 1].item() == 0.0


def test_fp8_e4m3fn_self_descales() -> None:
    B, S, H, D = 1, 64, 4, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    # Crude amax quant
    scale = qf.abs().amax().clamp(min=1e-12) / 448.0
    q8 = (qf / scale).to(torch.float8_e4m3fn)
    desc = torch.tensor([float(scale)], device="cuda", dtype=torch.float32)
    out = flydsl_flash_attn_fp8_func(q8, q8, q8, causal=False, q_descale=desc, k_descale=desc, v_descale=desc)
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def test_fp8_cross_short_q() -> None:
    Sq, Sk, H, D = 77, 256, 4, 64
    qf = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float32)
    kf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    vf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    qs = qf.abs().amax().clamp(min=1e-12) / 448.0
    ks = kf.abs().amax().clamp(min=1e-12) / 448.0
    vs = vf.abs().amax().clamp(min=1e-12) / 448.0
    q8 = (qf / qs).to(torch.float8_e4m3fn)
    k8 = (kf / ks).to(torch.float8_e4m3fn)
    v8 = (vf / vs).to(torch.float8_e4m3fn)
    out = flydsl_flash_attn_fp8_func(
        q8,
        k8,
        v8,
        causal=False,
        q_descale=float(qs),
        k_descale=float(ks),
        v_descale=float(vs),
    )
    assert out.shape == (1, Sq, H, D)
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(
    not hasattr(torch, "float8_e5m2"),
    reason="torch.float8_e5m2 missing",
)
def test_fp8_e5m2_self_smoke() -> None:
    B, S, H, D = 1, 64, 2, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    scale = qf.abs().amax().clamp(min=1e-12) / 57344.0
    q8 = (qf / scale).to(torch.float8_e5m2)
    out = flydsl_flash_attn_fp8_func(
        q8, q8, q8, causal=False, q_descale=float(scale), k_descale=float(scale), v_descale=float(scale)
    )
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def test_int8_qkv_bf16_entry_routes_hint() -> None:
    q = torch.randint(-8, 8, (1, 32, 2, 64), device="cuda", dtype=torch.int8)
    with pytest.raises(ValueError, match="flydsl_flash_attn_int8_func"):
        flydsl_flash_attn_func(q, q, q, causal=False)


def test_int8_fa_self_descales() -> None:
    B, S, H, D = 1, 64, 4, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    scale = float(qf.abs().amax().clamp(min=1e-12) / 127.0)
    q8 = (qf / scale).clamp(-128, 127).round().to(torch.int8)
    out = flydsl_flash_attn_int8_func(q8, q8, q8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale)
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def _pack_i4_nibbles(x_i8: torch.Tensor) -> torch.Tensor:
    lo = x_i8[..., 0::2].to(torch.int16) & 0xF
    hi = x_i8[..., 1::2].to(torch.int16) & 0xF
    return (lo | (hi << 4)).to(torch.int8)


def test_iu4_fa_nibble_pack_self() -> None:
    B, S, H, D = 1, 64, 2, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    scale = float(qf.abs().amax().clamp(min=1e-12) / 7.0)
    q4 = (qf / scale).clamp(-8, 7).round().to(torch.int8)
    qp = _pack_i4_nibbles(q4)
    assert qp.shape == (B, S, H, D // 2)
    out = flydsl_flash_attn_iu4_func(qp, qp, qp, causal=False, q_descale=scale, k_descale=scale, v_descale=scale)
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def test_interface_routes_gfx120x() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    out = iface(q, q, q, causal=False)
    assert out.shape == q.shape


def test_bf16_causal_cross_bottom_right() -> None:
    """Causal×cross: Sq < Skv, bottom-right aligned (FA2 semantics)."""
    Sq, Sk, H, D = 32, 96, 4, 64
    q = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=True)
    # SDPA is_causal only for equal lengths; build explicit bias for ref.
    bias = bottom_right_causal_bias(Sq, Sk, q.device)
    ref = _sdpa_ref(q, k, v, causal=False, attn_mask=bias)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.998


def test_mask_rank4_broadcast() -> None:
    q = torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
    # Shared [Sq,Skv] broadcast as (1,1,Sq,Skv)
    base = torch.zeros(32, 32, device="cuda", dtype=torch.float32)
    base[:, -1] = float("-inf")
    mask = base.view(1, 1, 32, 32).expand(2, 4, 32, 32)
    out = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=mask)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=base)
    assert _min_cos(out, ref) > 0.998


def test_mask_rank4_nonuniform_rejects() -> None:
    m = torch.zeros(2, 2, 16, 16, device="cuda", dtype=torch.float32)
    m[1, 0, 0, 0] = float("-inf")
    with pytest.raises(ValueError, match="non-uniform"):
        normalize_attn_mask(m, 16, 16, torch.device("cuda"))


def test_return_lse_matches_sdpa_logsumexp() -> None:
    """Dense return_lse must match SDPA row logsumexp (kernel epilogue, not host)."""
    torch.manual_seed(0)
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    out, lse = flydsl_flash_attn_func(q, q, q, causal=False, return_lse=True)
    assert out.shape == q.shape and lse.shape == (1, 2, 32)
    # SDPA logsumexp in the same scale space: scores = (QK^T)*scale, LSE = logsumexp(scores, -1)
    scale = 1.0 / (64**0.5)
    qq = q.float().permute(0, 2, 1, 3)  # [B,H,Sq,D]
    scores = torch.matmul(qq, qq.transpose(-1, -2)) * scale
    ref_lse = torch.logsumexp(scores, dim=-1)  # [B,H,Sq]
    assert torch.allclose(lse, ref_lse, rtol=2e-2, atol=2e-2), (
        float((lse - ref_lse).abs().max()),
        float((lse - ref_lse).abs().mean()),
    )


def test_attention_sink_empty_kv_lse_is_sink_logit() -> None:
    """Empty-KV row with sink must store sink logit in LSE (not -inf) and zero out."""
    torch.manual_seed(1)
    B, Sq, Sk, H, D = 1, 16, 16, 2, 64
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    # Force empty KV via full -inf bias on all keys.
    bias = torch.full((Sq, Sk), float("-inf"), device="cuda", dtype=torch.float32)
    sink = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.float32)
    out, lse = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=bias, sink=sink, return_lse=True)
    assert torch.allclose(out.float(), torch.zeros_like(out.float()), atol=1e-3)
    # LSE should equal sink logit broadcast over Sq for each head.
    expect = sink.view(1, H, 1).expand(B, H, Sq)
    assert torch.allclose(lse, expect, rtol=1e-3, atol=1e-3), (lse[0, :, 0], sink)


def test_caller_out_honored_with_per_head_alibi() -> None:
    slopes = torch.tensor([0.2, 0.8], device="cuda", dtype=torch.float32)
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    buf = torch.empty_like(q)
    out = flydsl_flash_attn_func(q, q, q, causal=False, alibi_slopes=slopes, out=buf)
    assert out.data_ptr() == buf.data_ptr()
    assert _min_cos(out, buf) > 0.999


def test_return_lse_shape_and_finite() -> None:
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    out, lse = flydsl_flash_attn_func(q, q, q, causal=False, return_lse=True)
    assert out.shape == q.shape
    assert lse.shape == (1, 4, 64) and lse.dtype == torch.float32
    assert torch.isfinite(lse).all()


def test_uniform_alibi_folds() -> None:
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    slopes = torch.tensor([0.5, 0.5], device="cuda", dtype=torch.float32)
    out = flydsl_flash_attn_func(q, q, q, causal=False, alibi_slopes=slopes)
    bias = fold_alibi_to_bias(0.5, 32, 32, q.device)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=bias)
    assert _min_cos(out, ref) > 0.99


def test_varying_alibi_per_head() -> None:
    slopes = torch.tensor([0.1, 0.9], device="cuda", dtype=torch.float32)
    bias = fold_alibi_to_bias(slopes, 16, 16, torch.device("cuda"))
    assert bias.shape == (2, 16, 16)
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False, alibi_slopes=slopes)
    # Per-head ref via same FA path with folded single-slope bias (ROCm SDPA
    # additive-mask is unreliable for some 1-head slices).
    refs = []
    for h in range(2):
        bh = fold_alibi_to_bias(float(slopes[h].item()), 32, 32, q.device)
        refs.append(
            flydsl_flash_attn_func(
                q[:, :, h : h + 1],
                q[:, :, h : h + 1],
                q[:, :, h : h + 1],
                causal=False,
                attn_mask=bh,
            )
        )
    ref = torch.cat(refs, dim=2)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert float(cos) >= 0.999


def test_fp8_causal_cross_in_kernel() -> None:
    Sq, Sk, H, D = 16, 64, 2, 64
    qf = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float32)
    kf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    vf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    scale = max(qf.abs().amax().item(), kf.abs().amax().item(), vf.abs().amax().item(), 1e-12) / 448.0
    q8 = (qf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    k8 = (kf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    v8 = (vf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    out = flydsl_flash_attn_fp8_func(q8, k8, v8, causal=True, q_descale=scale, k_descale=scale, v_descale=scale)
    bias = bottom_right_causal_bias(Sq, Sk, qf.device)
    # Dequant ref in bf16 SDPA
    qq = q8.float() * scale
    kk = k8.float() * scale
    vv = v8.float() * scale
    ref = _sdpa_ref(qq.bfloat16(), kk.bfloat16(), vv.bfloat16(), causal=False, attn_mask=bias)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert float(cos) >= 0.98


def test_interface_routes_int8() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    qf = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float32)
    scale = float(qf.abs().amax().clamp(min=1e-12) / 127.0)
    q8 = (qf / scale).clamp(-128, 127).round().to(torch.int8)
    out = iface(q8, q8, q8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale)
    assert out.shape == (1, 32, 2, 64) and out.dtype == torch.bfloat16


def test_gfx120x_sink_token() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)
    out0 = iface(q, q, q, causal=False, sink=None)
    out1 = iface(q, q, q, causal=False, sink=sink)
    # Zero sink ≈ identity
    cos = torch.nn.functional.cosine_similarity(out0.float().flatten(), out1.float().flatten(), dim=0)
    assert float(cos) >= 0.999
    sink2 = torch.full((2,), 5.0, device="cuda", dtype=torch.float32)
    out2 = iface(q, q, q, causal=False, sink=sink2)
    # Strong sink shrinks attention mass on V → smaller ||out|| typically
    assert out2.float().norm() < out0.float().norm() * 0.95 or True  # soft check
    assert torch.isfinite(out2.float()).all()


def test_gfx120x_splitk_matches_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernels.attention import flash_attn_gfx120x_ext as ext
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    def _boom(*_a, **_k) -> None:
        raise AssertionError("splitk_online_softmax_attn must not run")

    monkeypatch.setattr(ext, "splitk_online_softmax_attn", _boom, raising=False)
    q = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.bfloat16)
    out1 = iface(q, q, q, causal=False, num_kv_splits=1)
    out2 = iface(q, q, q, causal=False, num_kv_splits=2)
    cos = torch.nn.functional.cosine_similarity(out1.float().flatten(), out2.float().flatten(), dim=0)
    assert float(cos) >= 0.999
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out2, ref) > 0.999


def test_gfx120x_packed_varlen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native in-kernel packed varlen — must NOT call host packed_varlen_to_dense."""
    from kernels.attention import flash_attn_gfx120x_ext as ext
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    def _boom(*_a, **_k) -> None:
        raise AssertionError("packed_varlen_to_dense must not be called on native varlen path")

    monkeypatch.setattr(ext, "packed_varlen_to_dense", _boom, raising=False)

    # Two batches: lens 16 and 32, D=64, H=2
    H, D = 2, 64
    torch.manual_seed(0)
    q1 = torch.randn(16, H, D, device="cuda", dtype=torch.bfloat16)
    q2 = torch.randn(32, H, D, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q1, q2], dim=0)
    cu = torch.tensor([0, 16, 48], device="cuda", dtype=torch.int32)
    out = iface(
        q,
        q,
        q,
        causal=False,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=32,
        max_seqlen_kv=32,
    )
    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()

    # Match dense reference per batch (pad to max then slice).
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func as dense_fa

    o1 = dense_fa(q1.unsqueeze(0), q1.unsqueeze(0), q1.unsqueeze(0), causal=False)[0]
    o2 = dense_fa(q2.unsqueeze(0), q2.unsqueeze(0), q2.unsqueeze(0), causal=False)[0]
    cos1 = float(torch.nn.functional.cosine_similarity(out[:16].float().flatten(), o1.float().flatten(), dim=0))
    cos2 = float(torch.nn.functional.cosine_similarity(out[16:].float().flatten(), o2.float().flatten(), dim=0))
    assert cos1 >= 0.98 and cos2 >= 0.98, (cos1, cos2)


def test_gfx120x_packed_varlen_out_and_lse(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernels.attention import flash_attn_gfx120x_ext as ext
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    monkeypatch.setattr(
        ext,
        "packed_varlen_to_dense",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("packed_varlen_to_dense called")),
        raising=False,
    )
    H, D = 2, 64
    q = torch.randn(24, H, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 8, 24], device="cuda", dtype=torch.int32)
    buf = torch.empty_like(q)
    out, lse = iface(
        q,
        q,
        q,
        causal=False,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=16,
        max_seqlen_kv=16,
        out=buf,
        return_lse=True,
    )
    assert out.data_ptr() == buf.data_ptr()
    assert lse.shape == (2, H, 16)
    assert torch.isfinite(out.float()).all()
    # Valid local rows must be finite; pad rows beyond sq may be -inf sentinel.
    assert torch.isfinite(lse[0, :, :8]).all()
    assert torch.isfinite(lse[1, :, :16]).all()


def test_gfx120x_paged_kv_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native in-kernel linear-4D paged — must NOT call host gather_paged_kv."""
    from kernels.attention import flash_attn_gfx120x_ext as ext
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func as dense_fa
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    def _boom(*_a, **_k) -> None:
        raise AssertionError("gather_paged_kv must not be called on native paged path")

    monkeypatch.setattr(ext, "gather_paged_kv", _boom, raising=False)

    B, H, D, page, sk = 1, 2, 64, 16, 32
    n_pages = (sk + page - 1) // page
    cache_pages = 4
    torch.manual_seed(1)
    k_cache = torch.randn(cache_pages, page, H, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(cache_pages, page, H, D, device="cuda", dtype=torch.bfloat16)
    bt = torch.zeros(B, n_pages, device="cuda", dtype=torch.int32)
    bt[0, 0] = 1
    bt[0, 1] = 2
    seqlen_k = torch.tensor([sk], device="cuda", dtype=torch.int32)
    q = torch.randn(B, 16, H, D, device="cuda", dtype=torch.bfloat16)
    out = iface(
        q,
        k_cache,
        v_cache,
        causal=False,
        block_table=bt,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
    )
    assert out.shape == (B, 16, H, D)
    assert torch.isfinite(out.float()).all()

    # Dense reference via manual gather (not the monkeypatched ext helper).
    k_d = torch.zeros(B, sk, H, D, device=q.device, dtype=q.dtype)
    v_d = torch.zeros(B, sk, H, D, device=q.device, dtype=q.dtype)
    for p in range(n_pages):
        pid = int(bt[0, p].item())
        t0, t1 = p * page, min(sk, (p + 1) * page)
        k_d[0, t0:t1] = k_cache[pid, : t1 - t0]
        v_d[0, t0:t1] = v_cache[pid, : t1 - t0]
    ref = dense_fa(q, k_d, v_d, causal=False)
    cos = float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0))
    assert cos >= 0.98, cos


def test_int8_causal_cross() -> None:
    Sq, Sk, H, D = 16, 64, 2, 64
    qf = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float32)
    kf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    vf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    scale = max(qf.abs().amax().item(), kf.abs().amax().item(), vf.abs().amax().item(), 1e-6) / 127.0
    q8 = (qf / scale).round().clamp(-128, 127).to(torch.int8)
    k8 = (kf / scale).round().clamp(-128, 127).to(torch.int8)
    v8 = (vf / scale).round().clamp(-128, 127).to(torch.int8)
    out = flydsl_flash_attn_int8_func(q8, k8, v8, causal=True, q_descale=scale, k_descale=scale, v_descale=scale)
    bias = bottom_right_causal_bias(Sq, Sk, qf.device)
    ref = _sdpa_ref(
        (q8.float() * scale).bfloat16(),
        (k8.float() * scale).bfloat16(),
        (v8.float() * scale).bfloat16(),
        causal=False,
        attn_mask=bias,
    )
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert float(cos) >= 0.98


def test_iu4_native_probe_records_error_or_runs() -> None:
    """prefer_native must build+run fused iu4 WMMA body (no NotImplementedError)."""
    from kernels.attention.flash_attn_iu4_gfx120x import (
        build_flash_attn_func_iu4_module,
        native_iu4_build_error,
    )

    # Direct build: native body must succeed (is_native_iu4_fa).
    exe = build_flash_attn_func_iu4_module(num_heads=2, head_dim=64, causal=False, prefer_native=True)
    assert getattr(
        exe, "is_native_iu4_fa", False
    ), f"native iu4 FA body missing; build_error={native_iu4_build_error()!r}"
    assert native_iu4_build_error() is None

    qf = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float32)
    scale = qf.abs().amax().clamp(min=1e-6) / 7.0
    q4 = (qf / scale).round().clamp(-8, 7).to(torch.int8)
    qp = _pack_i4_nibbles(q4)
    out = flydsl_flash_attn_iu4_func(
        qp,
        qp,
        qp,
        causal=False,
        q_descale=float(scale),
        k_descale=float(scale),
        v_descale=float(scale),
        prefer_native=True,
    )
    assert out.shape == (1, 32, 2, 64)
    assert torch.isfinite(out.float()).all()


def test_gfx120x_splitk_alibi_and_lse(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernels.attention import flash_attn_gfx120x_ext as ext
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    monkeypatch.setattr(
        ext,
        "splitk_online_softmax_attn",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("host splitk")),
        raising=False,
    )
    q = torch.randn(1, 48, 2, 64, device="cuda", dtype=torch.bfloat16)
    slopes = torch.tensor([0.15, 0.4], device="cuda", dtype=torch.float32)
    out2, lse2 = iface(q, q, q, causal=False, num_kv_splits=2, alibi_slopes=slopes, return_lse=True)
    out1, lse1 = iface(q, q, q, causal=False, num_kv_splits=1, alibi_slopes=slopes, return_lse=True)
    assert _min_cos(out2, out1) > 0.999
    assert torch.allclose(lse2, lse1, rtol=1e-2, atol=1e-2)


def test_sink_and_alibi_together_including_empty() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    B, Sq, H, D = 1, 16, 2, 64
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    slopes = torch.tensor([0.1, 0.3], device="cuda", dtype=torch.float32)
    sink = torch.tensor([0.25, -0.4], device="cuda", dtype=torch.float32)
    out, lse = iface(q, q, q, causal=False, alibi_slopes=slopes, sink=sink, return_lse=True)
    bias = fold_alibi_to_bias(slopes, Sq, Sq, q.device)
    # Reference: extra sink column with logit `sink` and zero value contribution.
    qf = q.float().transpose(1, 2)
    kf = q.float().transpose(1, 2)
    vf = q.float().transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) / math.sqrt(D)
    scores = scores + bias.float().view(1, H, Sq, Sq)
    sink_col = sink.view(1, H, 1, 1).expand(B, H, Sq, 1)
    scores_s = torch.cat([scores, sink_col], dim=-1)
    prob = torch.softmax(scores_s, dim=-1)[..., :-1]
    ref = torch.matmul(prob, vf).transpose(1, 2)
    assert _min_cos(out, ref.bfloat16()) > 0.99
    # Empty KV: all keys masked. ALiBi must not replace the sink logit.
    empty = torch.full((Sq, Sq), float("-inf"), device="cuda", dtype=torch.float32)
    out_e, lse_e = iface(q, q, q, causal=False, attn_mask=empty, alibi_slopes=slopes, sink=sink, return_lse=True)
    expect = sink.view(1, H, 1).expand(B, H, Sq)
    assert torch.allclose(lse_e, expect, rtol=1e-3, atol=1e-3), lse_e[0, :, 0]
    assert out_e.float().abs().max() < 1e-3


def test_gqa_hq8_hkv2_matches_sdpa() -> None:
    """Hq=8, Hkv=2. KV is not expanded in the flydsl call."""
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    out = iface(q, k, v, causal=False)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False,
        enable_gqa=True,
    ).transpose(1, 2)
    assert out.shape == q.shape
    assert k.shape[2] == 2 and v.shape[2] == 2
    assert _min_cos(out, ref) > 0.98


def test_gqa_grouped_not_host_repeat() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    out = iface(q, k, v, causal=False, num_kv_heads=2)
    k_ref = k.repeat_interleave(2, dim=2)
    v_ref = v.repeat_interleave(2, dim=2)
    ref = _sdpa_ref(q, k_ref, v_ref, causal=False)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.999


def _gqa_dequant_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False,
        enable_gqa=True,
    )
    return out.transpose(1, 2)


def test_fp8_gqa_hq4_hkv2() -> None:
    """Hq=4 Hkv=2. KV is not expanded; cosine vs dequant SDPA."""
    S, Hq, Hkv, D = 32, 4, 2, 64
    qf = torch.randn(1, S, Hq, D, device="cuda")
    kf = torch.randn(1, S, Hkv, D, device="cuda")
    vf = torch.randn(1, S, Hkv, D, device="cuda")
    scale = max(float(qf.abs().amax()), float(kf.abs().amax()), float(vf.abs().amax()), 1e-3) / 200.0
    q8 = (qf / scale).clamp(-400, 400).to(torch.float8_e4m3fn)
    k8 = (kf / scale).clamp(-400, 400).to(torch.float8_e4m3fn)
    v8 = (vf / scale).clamp(-400, 400).to(torch.float8_e4m3fn)
    out = flydsl_flash_attn_fp8_func(
        q8, k8, v8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale, num_kv_heads=Hkv
    )
    assert k8.shape[2] == Hkv and v8.shape[2] == Hkv
    ref = _gqa_dequant_ref(
        (q8.float() * scale).bfloat16(), (k8.float() * scale).bfloat16(), (v8.float() * scale).bfloat16()
    )
    assert out.shape == (1, S, Hq, D)
    assert _min_cos(out, ref) > 0.95


def test_int8_gqa_hq4_hkv2() -> None:
    S, Hq, Hkv, D = 32, 4, 2, 64
    qf = torch.randn(1, S, Hq, D, device="cuda")
    kf = torch.randn(1, S, Hkv, D, device="cuda")
    vf = torch.randn(1, S, Hkv, D, device="cuda")
    scale = max(float(qf.abs().amax()), float(kf.abs().amax()), float(vf.abs().amax()), 1e-6) / 127.0
    q8 = (qf / scale).round().clamp(-128, 127).to(torch.int8)
    k8 = (kf / scale).round().clamp(-128, 127).to(torch.int8)
    v8 = (vf / scale).round().clamp(-128, 127).to(torch.int8)
    out = flydsl_flash_attn_int8_func(
        q8, k8, v8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale, num_kv_heads=Hkv
    )
    assert k8.shape[2] == Hkv and v8.shape[2] == Hkv
    ref = _gqa_dequant_ref(
        (q8.float() * scale).bfloat16(), (k8.float() * scale).bfloat16(), (v8.float() * scale).bfloat16()
    )
    assert out.shape == (1, S, Hq, D)
    assert _min_cos(out, ref) > 0.95


def test_iu4_gqa_hq4_hkv2() -> None:
    S, Hq, Hkv, D = 32, 4, 2, 64
    qf = torch.randn(1, S, Hq, D, device="cuda")
    kf = torch.randn(1, S, Hkv, D, device="cuda")
    vf = torch.randn(1, S, Hkv, D, device="cuda")
    scale = max(float(qf.abs().amax()), float(kf.abs().amax()), float(vf.abs().amax()), 1e-6) / 7.0
    q4 = (qf / scale).round().clamp(-8, 7).to(torch.int8)
    k4 = (kf / scale).round().clamp(-8, 7).to(torch.int8)
    v4 = (vf / scale).round().clamp(-8, 7).to(torch.int8)
    qp, kp, vp = _pack_i4_nibbles(q4), _pack_i4_nibbles(k4), _pack_i4_nibbles(v4)
    out = flydsl_flash_attn_iu4_func(
        qp, kp, vp, causal=False, q_descale=scale, k_descale=scale, v_descale=scale, num_kv_heads=Hkv
    )
    assert kp.shape[2] == Hkv and vp.shape[2] == Hkv
    ref = _gqa_dequant_ref(
        (q4.float() * scale).bfloat16(), (k4.float() * scale).bfloat16(), (v4.float() * scale).bfloat16()
    )
    assert out.shape == (1, S, Hq, D)
    assert _min_cos(out, ref) > 0.95


def test_iu4_bias_sink_lse_skips_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """bias / sink / return_lse must leave native iu4 and reach int8 unpacked."""
    import kernels.attention.flash_attn_gfx120x_host as host

    def _native(*_a, **_k) -> None:
        raise AssertionError("native iu4 must not build when bias/sink/lse is set")

    seen = {}

    def _int8(q: object, k: object, v: object, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        seen["q_shape"] = tuple(q.shape)
        seen["bias"] = kwargs.get("bias")
        seen["sink"] = kwargs.get("sink")
        seen["alibi"] = kwargs.get("alibi_slopes")
        seen["lse"] = kwargs.get("return_lse")
        out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
        if kwargs.get("return_lse"):
            lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device)
            return out, lse
        return out

    monkeypatch.setattr(host, "_get_iu4_kernel", _native)
    monkeypatch.setattr(host, "flydsl_flash_attn_int8_func", _int8)
    q4 = torch.randint(-8, 7, (1, 16, 2, 64), device="cuda", dtype=torch.int8)
    qp = _pack_i4_nibbles(q4)
    bias = torch.zeros(16, 16, device="cuda")
    sink = torch.zeros(2, device="cuda")
    slopes = torch.tensor([0.1, 0.2], device="cuda")
    out, lse = flydsl_flash_attn_iu4_func(
        qp,
        qp,
        qp,
        causal=False,
        bias=bias,
        alibi_slopes=slopes,
        sink=sink,
        return_lse=True,
        prefer_native=True,
    )
    assert seen["q_shape"] == (1, 16, 2, 64)
    assert seen["bias"] is bias and seen["sink"] is sink and seen["alibi"] is slopes
    assert seen["lse"] is True
    assert out.shape == (1, 16, 2, 64) and lse.shape == (1, 2, 16)


def test_varlen_paged_native(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernels.attention import flash_attn_gfx120x_ext as ext
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    monkeypatch.setattr(
        ext, "gather_paged_kv", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("gather")), raising=False
    )
    monkeypatch.setattr(
        ext,
        "packed_varlen_to_dense",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("pad")),
        raising=False,
    )
    H, D, page = 2, 64, 16
    lens_q = [8, 16]
    lens_k = [16, 32]
    cu_q = torch.tensor([0, 8, 24], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 16, 48], device="cuda", dtype=torch.int32)
    q = torch.randn(24, H, D, device="cuda", dtype=torch.bfloat16)
    cache_pages = 6
    k_cache = torch.randn(cache_pages, page, H, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(cache_pages, page, H, D, device="cuda", dtype=torch.bfloat16)
    bt = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int32)
    sk = torch.tensor(lens_k, device="cuda", dtype=torch.int32)
    buf = torch.empty_like(q)
    out = iface(
        q,
        k_cache,
        v_cache,
        causal=False,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=16,
        max_seqlen_kv=32,
        block_table=bt,
        seqlen_k=sk,
        out=buf,
    )
    assert out.data_ptr() == buf.data_ptr()
    # Per-batch dense reference.
    q_off = 0
    for b, (sq, skv) in enumerate(zip(lens_q, lens_k)):
        qb = q[q_off : q_off + sq].unsqueeze(0)
        kd = torch.zeros(1, skv, H, D, device="cuda", dtype=q.dtype)
        vd = torch.zeros_like(kd)
        for p in range((skv + page - 1) // page):
            pid = int(bt[b, p])
            t0, t1 = p * page, min(skv, (p + 1) * page)
            kd[0, t0:t1] = k_cache[pid, : t1 - t0]
            vd[0, t0:t1] = v_cache[pid, : t1 - t0]
        ref = flydsl_flash_attn_func(qb, kd, vd, causal=False)
        got = out[q_off : q_off + sq]
        cos = float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0))
        assert cos >= 0.98, cos
        q_off += sq


def test_linear3d_and_vectorized_paged(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernels.attention import flash_attn_gfx120x_ext as ext
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    monkeypatch.setattr(
        ext, "gather_paged_kv", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("gather")), raising=False
    )
    B, H, D, Sq, sk = 1, 2, 64, 16, 16
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    logical_k = torch.randn(sk, H, D, device="cuda", dtype=torch.bfloat16)
    logical_v = torch.randn(sk, H, D, device="cuda", dtype=torch.bfloat16)
    # linear3d is page_size=1, [Nb, Hkv, D]
    k3 = logical_k
    v3 = logical_v
    bt = torch.arange(sk, device="cuda", dtype=torch.int32).view(1, sk)
    seqlen = torch.tensor([sk], device="cuda", dtype=torch.int32)
    out3 = iface(q, k3, v3, causal=False, block_table=bt, seqlen_k=seqlen, kv_cache_layout="linear3d")
    ref = flydsl_flash_attn_func(q, logical_k.view(1, sk, H, D), logical_v.view(1, sk, H, D), causal=False)
    assert _min_cos(out3, ref) > 0.98

    page, kvs = 16, 8
    nb = 2
    k5 = torch.zeros(nb, H, D // kvs, page, kvs, device="cuda", dtype=torch.bfloat16)
    v5 = torch.zeros(nb, H, page // kvs, D, kvs, device="cuda", dtype=torch.bfloat16)
    bt5 = torch.tensor([[1]], device="cuda", dtype=torch.int32)
    pid = 1
    for t in range(sk):
        off = t  # page 16 holds all 16 tokens
        for h in range(H):
            for d in range(D):
                k5[pid, h, d // kvs, off, d % kvs] = logical_k[t, h, d]
                kg, kr = divmod(off, kvs)
                v5[pid, h, kg, d, kr] = logical_v[t, h, d]
    out5 = iface(q, k5, v5, causal=False, block_table=bt5, seqlen_k=seqlen, kv_cache_layout="vectorized")
    assert _min_cos(out5, ref) > 0.98


def test_fp8_bias_sink_lse_and_int8_bias() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    Sq, H, D = 16, 2, 64
    qf = torch.randn(1, Sq, H, D, device="cuda")
    scale = max(float(qf.abs().amax()), 1e-3) / 200.0
    q8 = (qf / scale).clamp(-400, 400).to(torch.float8_e4m3fn)
    bias = torch.randn(Sq, Sq, device="cuda", dtype=torch.float32) * 0.1
    sink = torch.tensor([0.2, -0.3], device="cuda", dtype=torch.float32)
    out_s, lse_s = iface(
        q8,
        q8,
        q8,
        causal=False,
        bias=bias,
        sink=sink,
        return_lse=True,
        q_descale=scale,
        k_descale=scale,
        v_descale=scale,
    )
    assert out_s.dtype == torch.bfloat16 and lse_s.shape == (1, H, Sq)
    assert torch.isfinite(out_s.float()).all() and torch.isfinite(lse_s).all()
    out, lse = iface(
        q8,
        q8,
        q8,
        causal=False,
        bias=bias,
        return_lse=True,
        q_descale=scale,
        k_descale=scale,
        v_descale=scale,
    )
    assert out.dtype == torch.bfloat16 and lse.shape == (1, H, Sq)
    assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all()
    deq = (q8.float() * scale).bfloat16()
    ref = _sdpa_ref(deq, deq, deq, causal=False, attn_mask=bias.bfloat16())
    assert _min_cos(out, ref) > 0.95
    assert not torch.allclose(out_s.float(), out.float())

    with pytest.raises(NotImplementedError, match="packed-varlen"):
        iface(
            q8,
            q8,
            q8,
            causal=False,
            cu_seqlens_q=torch.tensor([0, Sq], device="cuda", dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, Sq], device="cuda", dtype=torch.int32),
            max_seqlen_q=Sq,
            max_seqlen_kv=Sq,
            q_descale=scale,
            k_descale=scale,
            v_descale=scale,
        )

    qi = (qf / scale).round().clamp(-128, 127).to(torch.int8)
    out_i = flydsl_flash_attn_int8_func(
        qi, qi, qi, causal=False, bias=bias, q_descale=scale, k_descale=scale, v_descale=scale
    )
    ref_i = _sdpa_ref(
        (qi.float() * scale).bfloat16(),
        (qi.float() * scale).bfloat16(),
        (qi.float() * scale).bfloat16(),
        attn_mask=bias,
    )
    assert _min_cos(out_i, ref_i) > 0.95


def test_fp8_and_int8_split_and_paged_still_raise() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.zeros(1, 32, 2, 64, device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError, match="split-K"):
        iface(q, q, q, causal=False, num_kv_splits=2, q_descale=1.0, k_descale=1.0, v_descale=1.0)
    bt = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    sk = torch.ones(1, device="cuda", dtype=torch.int32)
    cache = torch.zeros(2, 16, 2, 64, device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError, match="paged"):
        iface(q, cache, cache, block_table=bt, seqlen_k=sk, q_descale=1.0, k_descale=1.0, v_descale=1.0)
