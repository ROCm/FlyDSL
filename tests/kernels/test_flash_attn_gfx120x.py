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

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_gfx120x_host import (  # noqa: E402
    flydsl_flash_attn_fp8_func,
    flydsl_flash_attn_func,
    flydsl_flash_attn_int8_func,
    fold_alibi_to_bias,
    mask_is_noop,
    normalize_attn_mask,
)
from tests.kernels.oracles import bottom_right_causal_bias  # noqa: E402


def _arch() -> str:
    return (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]


pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not str(get_rocm_arch() or "").startswith("gfx120"),
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
    # ROCm efficient attention warns, then casts, unless the mask is bool or q's dtype.
    if attn_mask is not None and attn_mask.is_floating_point() and attn_mask.dtype != qq.dtype:
        attn_mask = attn_mask.to(dtype=qq.dtype)
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
    # Dense FA soft-pads head_dim up to LDS-safe 480; above that still raises.
    q = torch.randn(1, 32, 2, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim=512"):
        flydsl_flash_attn_func(q, q, q, causal=False)


def test_noop_mask_hits() -> None:
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    # Shape-only noop: nonempty bool is not noop (no host .all().item() probe).
    # All-True still converts on device to zeros bias and matches unmasked SDPA.
    mask = torch.ones(64, 64, device="cuda", dtype=torch.bool)
    assert not mask_is_noop(mask)
    assert mask_is_noop(None)
    assert mask_is_noop(torch.empty(0, device="cuda", dtype=torch.bool))
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
    # Shared mask as (1,1,Sq,Skv) — multi-slice expand is rejected (no host equal probe).
    base = torch.zeros(32, 32, device="cuda", dtype=torch.float32)
    base[:, -1] = float("-inf")
    mask = base.view(1, 1, 32, 32)
    out = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=mask)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=base)
    assert _min_cos(out, ref) > 0.998


def test_mask_rank4_nonuniform_rejects() -> None:
    m = torch.zeros(2, 2, 16, 16, device="cuda", dtype=torch.float32)
    m[1, 0, 0, 0] = float("-inf")
    with pytest.raises(ValueError, match="single \\[Sq, Skv\\] slice"):
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


def test_bf16_odd_head_and_short_kv_stay_unpadded() -> None:
    """D=80 and K=40 are not host-cloned. The kernel masks the tail."""
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    stream = torch.cuda.Stream()
    torch.manual_seed(0)
    with torch.cuda.stream(stream):
        q80 = torch.randn(1, 64, 2, 80, device="cuda", dtype=torch.bfloat16)
        out80 = iface(q80, q80, q80, causal=False, stream=stream)
        ref80 = _sdpa_ref(q80, q80, q80, causal=False)
        q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 40, 2, 64, device="cuda", dtype=torch.bfloat16)
        out = iface(q, k, k, causal=False, stream=stream)
        ref = _sdpa_ref(q, k, k, causal=False)
        # Odd D and a short sequence together: the tail load must not read past Sk.
        q65 = torch.randn(1, 33, 2, 65, device="cuda", dtype=torch.bfloat16)
        k65 = torch.randn(1, 33, 2, 65, device="cuda", dtype=torch.bfloat16)
        v65 = torch.randn(1, 33, 2, 65, device="cuda", dtype=torch.bfloat16)
        out65 = iface(q65, k65, v65, causal=False, stream=stream)
        ref65 = _sdpa_ref(q65, k65, v65, causal=False)
    stream.synchronize()
    assert out80.shape[-1] == 80
    assert k.shape[1] == 40
    assert out65.shape == q65.shape
    assert _min_cos(out80, ref80) > 0.999
    assert _min_cos(out, ref) > 0.999
    assert _min_cos(out65, ref65) > 0.999


def test_dense_empty_kv_seq_len_zero_zeros_out() -> None:
    """Sk=0 soft-pads to BLOCK_N zeros; output is finite zeros (no OOB / reject)."""
    torch.manual_seed(2)
    B, Sq, Sk, H, D = 1, 8, 0, 2, 64
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(B, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(B, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    torch.cuda.synchronize()
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), torch.zeros_like(out.float()), atol=1e-3)


def test_caller_out_honored_with_per_head_alibi() -> None:
    slopes = torch.tensor([0.2, 0.8], device="cuda", dtype=torch.float32)
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    buf = torch.empty_like(q)
    out = flydsl_flash_attn_func(q, q, q, causal=False, alibi_slopes=slopes, out=buf)
    assert out.data_ptr() == buf.data_ptr()
    bias = fold_alibi_to_bias(slopes, 16, 16, q.device)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=bias)
    assert _min_cos(out, ref) > 0.99


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

    torch.manual_seed(0)
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)
    out0 = iface(q, q, q, causal=False, sink=None)
    out1 = iface(q, q, q, causal=False, sink=sink)
    # Zero sink ≈ identity
    cos = torch.nn.functional.cosine_similarity(out0.float().flatten(), out1.float().flatten(), dim=0)
    assert float(cos) >= 0.999
    # Strong sink dominates softmax mass → ||out|| shrinks vs no-sink.
    sink2 = torch.full((2,), 8.0, device="cuda", dtype=torch.float32)
    out2 = iface(q, q, q, causal=False, sink=sink2)
    assert out2.float().norm() < out0.float().norm() * 0.99
    assert torch.isfinite(out2.float()).all()


def test_gfx120x_splitk_matches_dense() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.bfloat16)
    out1 = iface(q, q, q, causal=False, num_kv_splits=1)
    out2 = iface(q, q, q, causal=False, num_kv_splits=2)
    cos = torch.nn.functional.cosine_similarity(out1.float().flatten(), out2.float().flatten(), dim=0)
    assert float(cos) >= 0.999
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out2, ref) > 0.999


def test_gfx120x_packed_varlen() -> None:
    """Native in-kernel packed varlen — dense reference per batch."""
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

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


def test_gfx120x_packed_varlen_out_and_lse() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

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


def test_varlen_odd_max_bias_is_not_padded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A max KV that is not a tile stays on the caller's bias width."""
    import kernels.common.gfx120x_pad as padmod
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func as dense_fa
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    def _boom(*_args, **_kwargs):
        raise AssertionError("device_pad")

    monkeypatch.setattr(padmod, "device_pad", _boom)
    H, D = 2, 64
    torch.manual_seed(4)
    q1 = torch.randn(20, H, D, device="cuda", dtype=torch.bfloat16)
    q2 = torch.randn(40, H, D, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q1, q2], dim=0)
    cu = torch.tensor([0, 20, 60], device="cuda", dtype=torch.int32)
    bias = torch.randn(40, 40, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = iface(
            q,
            q,
            q,
            causal=False,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=40,
            max_seqlen_kv=40,
            bias=bias,
            stream=stream,
        )
        ref1 = dense_fa(
            q1.unsqueeze(0), q1.unsqueeze(0), q1.unsqueeze(0), causal=False, bias=bias[:20, :20], stream=stream
        )
        ref2 = dense_fa(q2.unsqueeze(0), q2.unsqueeze(0), q2.unsqueeze(0), causal=False, bias=bias, stream=stream)
    stream.synchronize()
    cos1 = float(torch.nn.functional.cosine_similarity(out[:20].float().flatten(), ref1[0].float().flatten(), dim=0))
    cos2 = float(torch.nn.functional.cosine_similarity(out[20:].float().flatten(), ref2[0].float().flatten(), dim=0))
    assert out.shape == q.shape
    assert cos1 >= 0.98 and cos2 >= 0.98, (cos1, cos2)


def test_paged_odd_kv_bias_is_not_padded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paged KV whose length is not a tile does not widen the bias."""
    import kernels.common.gfx120x_pad as padmod
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func as dense_fa
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    def _boom(*_args, **_kwargs):
        raise AssertionError("device_pad")

    monkeypatch.setattr(padmod, "device_pad", _boom)
    B, H, D, page, sq, sk = 1, 2, 64, 16, 16, 40
    n_pages = (sk + page - 1) // page
    torch.manual_seed(5)
    k_cache = torch.randn(4, page, H, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(4, page, H, D, device="cuda", dtype=torch.bfloat16)
    bt = torch.tensor([[1, 2, 0]], device="cuda", dtype=torch.int32)
    seqlen_k = torch.tensor([sk], device="cuda", dtype=torch.int32)
    q = torch.randn(B, sq, H, D, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(sq, sk, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = iface(
            q,
            k_cache,
            v_cache,
            causal=False,
            block_table=bt,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            bias=bias,
            stream=stream,
        )
    stream.synchronize()
    k_d = torch.zeros(B, sk, H, D, device=q.device, dtype=q.dtype)
    v_d = torch.zeros(B, sk, H, D, device=q.device, dtype=q.dtype)
    for p in range(n_pages):
        pid = int(bt[0, p].item())
        t0, t1 = p * page, min(sk, (p + 1) * page)
        k_d[0, t0:t1] = k_cache[pid, : t1 - t0]
        v_d[0, t0:t1] = v_cache[pid, : t1 - t0]
    ref = dense_fa(q, k_d, v_d, causal=False, bias=bias)
    cos = float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0))
    assert out.shape == (B, sq, H, D)
    assert cos >= 0.98, cos


def test_gfx120x_paged_kv_gather() -> None:
    """Native in-kernel linear-4D paged vs dense reference via manual gather."""
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func as dense_fa
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

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

    # Dense reference via manual gather.
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


def test_gfx120x_splitk_alibi_and_lse() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

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


def test_varlen_paged_native() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

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


def test_linear3d_and_vectorized_paged() -> None:
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

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

    cu = torch.tensor([0, Sq], device="cuda", dtype=torch.int32)
    packed = q8[0]
    var = iface(
        packed,
        packed,
        packed,
        causal=False,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=Sq,
        max_seqlen_kv=Sq,
        q_descale=scale,
        k_descale=scale,
        v_descale=scale,
    )
    assert var.shape == packed.shape
    dense = flydsl_flash_attn_fp8_func(q8, q8, q8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale)
    assert _min_cos(var.unsqueeze(0), dense) > 0.98

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


def _quant_pair(kind: str, qf: torch.Tensor):
    scale = qf.abs().amax().clamp(min=1e-6)
    if kind == "fp8":
        scale = scale / 448
        q8 = (qf / scale).to(torch.float8_e4m3fn)
    else:
        scale = scale / 127
        q8 = (qf / scale).round().clamp(-128, 127).to(torch.int8)
    return q8, scale


@pytest.mark.parametrize("dtype_kind", ["fp8", "int8"])
def test_quant_splitk_matches_dense(dtype_kind: str) -> None:
    """Same shape as test_gfx120x_splitk_matches_dense. Partials stay on device."""
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        qf = torch.randn(1, 64, 2, 64, device="cuda")
        q8, scale = _quant_pair(dtype_kind, qf)
        out1 = iface(
            q8, q8, q8, causal=False, num_kv_splits=1, q_descale=scale, k_descale=scale, v_descale=scale, stream=stream
        )
        out2 = iface(
            q8, q8, q8, causal=False, num_kv_splits=2, q_descale=scale, k_descale=scale, v_descale=scale, stream=stream
        )
    stream.synchronize()
    cos = torch.nn.functional.cosine_similarity(out1.float().flatten(), out2.float().flatten(), dim=0)
    assert float(cos) >= 0.999
    assert torch.isfinite(out2.float()).all()


@pytest.mark.parametrize("dtype_kind", ["fp8", "int8"])
def test_quant_paged_matches_dense(dtype_kind: str) -> None:
    """Linear paged cache, same shape as test_gfx120x_paged_kv_gather. No host gather."""
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    B, Sq, Sk, H, D, page = 1, 16, 32, 2, 64, 16
    with torch.cuda.stream(stream):
        qf = torch.randn(B, Sq, H, D, device="cuda")
        kf = torch.randn(B, Sk, H, D, device="cuda")
        vf = torch.randn(B, Sk, H, D, device="cuda")
        q8, sq = _quant_pair(dtype_kind, qf)
        k8, sk = _quant_pair(dtype_kind, kf)
        v8, sv = _quant_pair(dtype_kind, vf)
        n_pages = Sk // page
        cache_k = torch.empty(n_pages, page, H, D, device="cuda", dtype=k8.dtype)
        cache_v = torch.empty(n_pages, page, H, D, device="cuda", dtype=v8.dtype)
        for p in range(n_pages):
            cache_k[p].copy_(k8[0, p * page : (p + 1) * page])
            cache_v[p].copy_(v8[0, p * page : (p + 1) * page])
        bt = torch.arange(n_pages, device="cuda", dtype=torch.int32).view(1, n_pages)
        seqlen = torch.tensor([Sk], device="cuda", dtype=torch.int32)
        dense = iface(q8, k8, v8, causal=False, q_descale=sq, k_descale=sk, v_descale=sv, stream=stream)
        paged = iface(
            q8,
            cache_k,
            cache_v,
            causal=False,
            block_table=bt,
            seqlen_k=seqlen,
            q_descale=sq,
            k_descale=sk,
            v_descale=sv,
            stream=stream,
        )
    stream.synchronize()
    assert paged.shape == dense.shape
    assert _min_cos(paged, dense) > 0.999


@pytest.mark.parametrize("dtype_kind", ["fp8", "int8"])
def test_quant_paged_linear3d_and_varlen(dtype_kind: str) -> None:
    """linear3d is one token per block. Varlen Q is packed; the cache is not copied."""
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    torch.manual_seed(1)
    stream = torch.cuda.Stream()
    H, D = 2, 64
    with torch.cuda.stream(stream):
        qf = torch.randn(1, 8, H, D, device="cuda")
        kf = torch.randn(1, 8, H, D, device="cuda")
        q8, sq = _quant_pair(dtype_kind, qf)
        k8, sk = _quant_pair(dtype_kind, kf)
        cache_k = k8[0]
        cache_v = k8[0]
        bt = torch.arange(8, device="cuda", dtype=torch.int32).view(1, 8)
        seqlen = torch.tensor([8], device="cuda", dtype=torch.int32)
        dense = iface(q8, k8, k8, causal=False, q_descale=sq, k_descale=sk, v_descale=sk, stream=stream)
        linear3d = iface(
            q8,
            cache_k,
            cache_v,
            causal=False,
            block_table=bt,
            seqlen_k=seqlen,
            kv_cache_layout="linear3d",
            q_descale=sq,
            k_descale=sk,
            v_descale=sk,
            stream=stream,
        )
        cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        cache_p = k8.reshape(1, 8, H, D)
        bt_p = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        varlen = iface(
            q8[0],
            cache_p,
            cache_p,
            causal=False,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=8,
            max_seqlen_kv=8,
            block_table=bt_p,
            seqlen_k=seqlen,
            q_descale=sq,
            k_descale=sk,
            v_descale=sk,
            stream=stream,
        )
    stream.synchronize()
    assert _min_cos(linear3d, dense) > 0.999
    assert _min_cos(varlen.unsqueeze(0), dense) > 0.999


def test_quant_short_and_empty_kv_no_clone() -> None:
    """Sk not a multiple of the tile, and Sk=0. K and V stay at the caller shape."""
    from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_fp8_func

    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        qf = torch.randn(1, 32, 2, 64, device="cuda")
        kf = torch.randn(1, 40, 2, 64, device="cuda")
        scale = (qf.abs().amax() / 448).clamp(min=1e-6)
        q8 = (qf / scale).to(torch.float8_e4m3fn)
        k8 = (kf / scale).to(torch.float8_e4m3fn)
        out = flydsl_flash_attn_fp8_func(
            q8, k8, k8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale, stream=stream
        )
        deq_q = (q8.float() * scale).bfloat16()
        deq_k = (k8.float() * scale).bfloat16()
        ref = _sdpa_ref(deq_q, deq_k, deq_k, causal=False)
        empty_k = torch.empty(1, 0, 2, 64, device="cuda", dtype=q8.dtype)
        empty = flydsl_flash_attn_fp8_func(
            q8, empty_k, empty_k, causal=False, q_descale=scale, k_descale=scale, v_descale=scale, stream=stream
        )
    stream.synchronize()
    assert k8.shape[1] == 40
    assert _min_cos(out, ref) > 0.98
    assert torch.isfinite(empty).all()
    assert empty.float().abs().max() < 1e-3


def _window_mask(sq: int, sk: int, left: int, right: int, causal: bool, device: torch.device) -> torch.Tensor:
    """aiter (left, right) band, plus bottom-right causal when requested."""
    rows = torch.arange(sq, device=device)[:, None]
    cols = torch.arange(sk, device=device)[None, :]
    keep = (cols >= rows - left) & (cols <= rows + right)
    if causal:
        keep = keep & (cols <= rows + (sk - sq))
    zero = torch.zeros((), device=device, dtype=torch.float32)
    neg = torch.full((), float("-inf"), device=device, dtype=torch.float32)
    return torch.where(keep, zero, neg)


@pytest.mark.parametrize("dtype_kind", ["fp8", "int8"])
@pytest.mark.parametrize("causal", [False, True])
def test_quant_sliding_window_matches_sdpa(dtype_kind: str, causal: bool) -> None:
    """Same (left, right) band as swa_gfx950, on the shapes the bf16 smokes use."""
    B, Sq, H, D = 1, 64, 4, 64
    left, right = 16, 8
    torch.manual_seed(0)
    # A non-default stream: the default stream's cuda_stream handle is 0, so a
    # read on any other stream can observe the buffer before the kernel stores.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        qf = torch.randn(B, Sq, H, D, device="cuda")
        scale = qf.abs().amax().clamp(min=1e-6)
        mask = _window_mask(Sq, Sq, left, right, causal, qf.device)
        if dtype_kind == "fp8":
            scale = scale / 448
            q8 = (qf / scale).to(torch.float8_e4m3fn)
            out = flydsl_flash_attn_fp8_func(
                q8,
                q8,
                q8,
                causal=causal,
                sliding_window=(left, right),
                q_descale=scale,
                k_descale=scale,
                v_descale=scale,
                stream=stream,
            )
            deq = (q8.float() * scale).bfloat16()
        else:
            scale = scale / 127
            q8 = (qf / scale).round().clamp(-128, 127).to(torch.int8)
            out = flydsl_flash_attn_int8_func(
                q8,
                q8,
                q8,
                causal=causal,
                sliding_window=(left, right),
                q_descale=scale,
                k_descale=scale,
                v_descale=scale,
                stream=stream,
            )
            deq = (q8.float() * scale).bfloat16()
        ref = _sdpa_ref(deq, deq, deq, causal=False, attn_mask=mask)
    stream.synchronize()
    assert out.shape == (B, Sq, H, D)
    assert _min_cos(out, ref) > 0.98
