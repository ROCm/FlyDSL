# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""L0 tests for the SageAttention2 G0 numerical reference (spec §12)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.kernels.sageattn2_layout import (
    LOG2E,
    c_frags_to_tile,
    dequant_c_i32,
    pack_int4,
    preshuffle_b_fp8,
    unpack_int4,
    unshuffle_b_fp8,
)
from tests.kernels.sageattn2_ref import (
    E4M3_MAX,
    EPS,
    INT4_QMAX,
    INT8_QMAX,
    S_FP8_OFFSET,
    causal_mask,
    dequant_scores,
    pack_e4m3,
    quant_per_token,
    quant_v_e4m3,
    round_half_away,
    sageattn2_ref,
    softmax_update,
)
from tests.kernels.sageattn2_ref import LOG2E as REF_LOG2E

pytestmark = [pytest.mark.l0_backend_agnostic]

_HAS_E4M3 = hasattr(torch, "float8_e4m3fn")


def test_check2_round_half_away_fixture():
    x = np.array([0.5, -0.5, 1.5, -1.5, 2.5, 0.49, -0.49, 0.0], dtype=np.float32)
    got = round_half_away(x)
    want = np.array([1, -1, 2, -2, 3, 0, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(got, want)


def test_check2_torch_round_differs_on_half_to_even_ties():
    x = np.array([0.5, -0.5, 2.5, -2.5], dtype=np.float32)
    ha = round_half_away(x)
    tr = torch.round(torch.from_numpy(x)).numpy()
    np.testing.assert_array_equal(ha, np.array([1, -1, 3, -3], dtype=np.float32))
    assert not np.array_equal(ha, tr)


def test_check1_int4_scale_amax_over_qmax_plus_eps():
    x = np.array([[-3.5, 7.0]], dtype=np.float32)
    q, scale = quant_per_token(x, INT4_QMAX)
    assert scale.shape == (1,)
    assert scale.dtype == np.float32
    assert scale[0] == np.float32(7.0 / 7.0 + EPS)
    assert q.dtype == np.int8
    assert q.min() >= -8 and q.max() <= 7


@pytest.mark.parametrize("d", [64, 128])
def test_check1_int4_pack_roundtrip_and_s_i32_identity(d):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, d)).astype(np.float32)
    q, scale = quant_per_token(x, INT4_QMAX)
    stored = pack_int4(q)
    assert stored.shape == (5, d // 8)
    assert stored.dtype == np.int32
    np.testing.assert_array_equal(unpack_int4(stored), q)
    k = rng.standard_normal((7, d)).astype(np.float32)
    kq, _ = quant_per_token(k, INT4_QMAX)
    s1 = q.astype(np.int32) @ kq.astype(np.int32).T
    s2 = unpack_int4(pack_int4(q)).astype(np.int32) @ unpack_int4(pack_int4(kq)).astype(np.int32).T
    np.testing.assert_array_equal(s1, s2)
    assert s1.dtype == np.int32


def test_log2e_is_layout_object():
    assert REF_LOG2E is LOG2E
    assert LOG2E == 1.4426950408889634


def _tile_to_c_frags(p: np.ndarray, arch: str) -> np.ndarray:
    """Inverse of G-I c_frags_to_tile. Test-only; not part of sageattn2_ref."""
    c = np.empty((32, 8), dtype=p.dtype)
    for lane in range(32):
        l16, g = lane % 16, lane // 16
        for v in range(8):
            if arch != "gfx120":
                raise ValueError(arch)
            k = 8 * g + v
            c[lane, v] = p[l16, k]
    return c


@pytest.mark.parametrize("arch", ("gfx120",))
def test_check4_dequant_matches_gi_fragment_form(arch):
    rng = np.random.default_rng(4)
    q_i = rng.integers(-8, 8, size=(16, 64), dtype=np.int8)
    k_i = rng.integers(-8, 8, size=(16, 64), dtype=np.int8)
    q_scale = (rng.random(16) + 0.05).astype(np.float32)
    k_scale = (rng.random(16) + 0.05).astype(np.float32)
    sm_scale = 0.125
    s_i32 = q_i.astype(np.int32) @ k_i.astype(np.int32).T
    s_whole = dequant_scores(s_i32, q_scale, k_scale, sm_scale)
    c = _tile_to_c_frags(s_i32, arch)
    s_frags = dequant_c_i32(c, q_scale, k_scale, sm_scale, 0, 0, arch)
    s_gi = c_frags_to_tile(s_frags, arch)
    np.testing.assert_allclose(s_whole, s_gi, rtol=0.0, atol=1e-6)


def test_causal_mask_queries_at_end_when_sq_lt_sk():
    s = np.zeros((4, 8), dtype=np.float32)
    got = causal_mask(s, 4, 8)
    q = np.arange(4)[:, None]
    k = np.arange(8)[None, :]
    want_mask = k > q + (8 - 4)
    assert np.all(np.isneginf(got[want_mask]))
    assert np.all(got[~want_mask] == 0.0)
    tril = torch.ones(4, 8, dtype=torch.bool).tril(diagonal=8 - 4)
    assert np.array_equal(want_mask, (~tril).numpy())


@pytest.mark.skipif(not _HAS_E4M3, reason="torch.float8_e4m3fn required")
def test_check9_pack_e4m3_bytes():
    x = np.array([0.0, 448.0, -448.0, float("nan"), 449.0, -449.0], dtype=np.float32)
    got = pack_e4m3(x)
    assert got.dtype == np.uint8
    assert got[0] == 0x00
    assert got[1] == 0x7E
    assert got[2] == 0xFE
    assert got[3] == 0x7F
    assert got[4] == 0x7E
    assert got[5] == 0xFE


@pytest.mark.skipif(not _HAS_E4M3, reason="torch.float8_e4m3fn required")
def test_check15_v_e4m3_preshuffle_roundtrip():
    rng = np.random.default_rng(15)
    v = rng.standard_normal((32, 64)).astype(np.float32)
    v_q, v_scale = quant_v_e4m3(v)
    assert v_q.shape == (32, 64) and v_q.dtype == np.uint8
    assert v_scale.shape == (64,) and v_scale.dtype == np.float32
    amax = np.max(np.abs(v), axis=0)
    np.testing.assert_allclose(v_scale, amax / np.float32(E4M3_MAX) + np.float32(EPS), rtol=0, atol=0)
    stored = preshuffle_b_fp8(v_q)
    np.testing.assert_array_equal(unshuffle_b_fp8(stored), v_q)


def test_softmax_update_first_tile_no_offset():
    s = np.array([[0.0, -1.0, -8.0, -8.0] + [-20.0] * 12], dtype=np.float32)
    m = np.full((1,), np.float32(-np.inf))
    l = np.zeros((1,), dtype=np.float32)  # noqa: E741
    m1, l1, alpha, p = softmax_update(s, m, l, apply_fp8_offset=False)
    assert alpha.shape == (1,) and p.shape == (1, 16)
    assert alpha[0] == np.float32(0.0)
    np.testing.assert_allclose(m1[0], 0.0)
    np.testing.assert_allclose(p[0, 0], 1.0, rtol=0, atol=1e-6)
    assert np.all(np.isfinite(p))


def test_check14_g5_offset_on_this_tile_rowmax_not_running_m():
    s0 = np.full((1, 16), np.float32(-20.0))
    s0[0, 0] = np.float32(4.0)
    s1 = np.full((1, 16), np.float32(-20.0))
    s1[0, 0] = np.float32(1.0)  # tile1 max < tile0 max
    m = np.full((1,), np.float32(-np.inf))
    l = np.zeros((1,), dtype=np.float32)  # noqa: E741
    m0, l0, a0, p0 = softmax_update(s0, m, l, apply_fp8_offset=True)
    np.testing.assert_allclose(m0[0], np.float32(4.0) - S_FP8_OFFSET, rtol=0, atol=1e-5)
    assert np.max(p0) > 1.0  # scaled into e4m3 range, not (0, 1]
    m1, l1, a1, p1 = softmax_update(s1, m0, l0, apply_fp8_offset=True)
    np.testing.assert_allclose(m1, m0, rtol=0, atol=1e-6)
    s_empty = np.full((1, 16), np.float32(-np.inf))
    m2, l2, a2, p2 = softmax_update(s_empty, m1, l1, apply_fp8_offset=True)
    np.testing.assert_allclose(m2, m1, rtol=0, atol=1e-6)
    np.testing.assert_allclose(a2[0], 1.0, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(p2, 0.0)
    assert np.all(np.isfinite(p2)) and not np.any(np.isnan(l2))


def test_softmax_update_all_masked_first_tile_no_nan():
    s = np.full((2, 16), np.float32(-np.inf))
    m = np.full((2,), np.float32(-np.inf))
    l = np.zeros((2,), dtype=np.float32)  # noqa: E741
    m1, l1, alpha, p = softmax_update(s, m, l, apply_fp8_offset=True)
    assert np.all(np.isneginf(m1))
    np.testing.assert_array_equal(alpha, 0.0)
    np.testing.assert_array_equal(p, 0.0)
    np.testing.assert_array_equal(l1, 0.0)


def _randn_f32(*shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def test_rejects_mixed_dtypes_and_f32_with_int4():
    q = torch.randn(1, 1, 8, 64, dtype=torch.float16)
    k = q.clone()
    v = q.clone().float()
    with pytest.raises(ValueError, match="dtype"):
        sageattn2_ref(q, k, v)
    qf = q.float()
    with pytest.raises(ValueError, match="qk"):
        sageattn2_ref(qf, qf, qf, qk="int4")


def test_check3_smooth_k_identity_qk_none_f32():
    q = _randn_f32(1, 1, 32, 64, seed=3)
    k = _randn_f32(1, 1, 32, 64, seed=4)
    v = _randn_f32(1, 1, 32, 64, seed=5)
    o_on = sageattn2_ref(q, k, v, qk="none", pv="f32", smooth_k=True, is_causal=False)
    o_off = sageattn2_ref(q, k, v, qk="none", pv="f32", smooth_k=False, is_causal=False)
    assert o_on.dtype == torch.float32
    torch.testing.assert_close(o_on, o_off, atol=1e-5, rtol=0)
    q2 = q.clone()
    k2 = k.clone()
    v2 = v.clone()
    sageattn2_ref(q2, k2, v2, qk="none", pv="f32")
    torch.testing.assert_close(q2, q)
    torch.testing.assert_close(k2, k)


def test_numpy_in_torch_out():
    q = _randn_f32(1, 1, 16, 64, seed=6).numpy()
    k = _randn_f32(1, 1, 16, 64, seed=7).numpy()
    v = _randn_f32(1, 1, 16, 64, seed=8).numpy()
    o = sageattn2_ref(q, k, v, qk="none", pv="f32", smooth_k=False)
    assert isinstance(o, torch.Tensor) and o.dtype == torch.float32
    assert o.shape == (1, 1, 16, 64)


def test_check5_online_tk16_vs_full_row():
    q = _randn_f32(1, 1, 32, 64, seed=50)
    k = _randn_f32(1, 1, 32, 64, seed=51)
    v = _randn_f32(1, 1, 32, 64, seed=52)
    o = sageattn2_ref(q, k, v, qk="none", pv="f32", smooth_k=False, is_causal=False)
    sm = 1.0 / math.sqrt(64)
    s = (q @ k.transpose(-1, -2)) * sm * float(LOG2E)
    m = s.amax(dim=-1, keepdim=True)
    p = torch.exp2(s - m)
    want = p @ v / p.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(o, want, rtol=1e-4, atol=1e-5)


def test_check6_causal_mask_and_sq_ne_sk():
    q = _randn_f32(1, 1, 8, 64, seed=60)
    k = _randn_f32(1, 1, 8, 64, seed=61)
    v = _randn_f32(1, 1, 8, 64, seed=62)
    sm = 1.0 / math.sqrt(64)
    s = (q @ k.transpose(-1, -2)) * sm
    s = s.masked_fill(torch.ones(8, 8, dtype=torch.bool).triu(diagonal=1), float("-inf"))
    p = torch.softmax(s, dim=-1)
    want = p @ v
    o = sageattn2_ref(q, k, v, qk="none", pv="f32", smooth_k=False, is_causal=True)
    torch.testing.assert_close(o, want, atol=1e-5, rtol=0)
    q4 = _randn_f32(1, 1, 4, 64, seed=63)
    k8 = _randn_f32(1, 1, 8, 64, seed=64)
    v8 = _randn_f32(1, 1, 8, 64, seed=65)
    o48 = sageattn2_ref(q4, k8, v8, qk="none", pv="f32", smooth_k=False, is_causal=True)
    s48 = (q4 @ k8.transpose(-1, -2)) * sm
    tril = torch.ones(4, 8, dtype=torch.bool).tril(diagonal=8 - 4)
    s48 = s48.masked_fill(~tril, float("-inf"))
    want48 = torch.softmax(s48, dim=-1) @ v8
    torch.testing.assert_close(o48, want48, atol=1e-5, rtol=0)


def test_check7_gqa_repeat_after_equals_pre_repeat():
    q = _randn_f32(1, 4, 16, 64, seed=70)
    k = _randn_f32(1, 2, 16, 64, seed=71)
    v = _randn_f32(1, 2, 16, 64, seed=72)
    o = sageattn2_ref(q, k, v, qk="none", pv="f32", smooth_k=True, is_causal=False)
    o2 = sageattn2_ref(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        qk="none",
        pv="f32",
        smooth_k=True,
        is_causal=False,
    )
    torch.testing.assert_close(o, o2, atol=0.0, rtol=0.0)


def test_check12_lse_and_fully_masked_row():
    q = _randn_f32(1, 1, 16, 64, seed=120)
    k = _randn_f32(1, 1, 16, 64, seed=121)
    v = _randn_f32(1, 1, 16, 64, seed=122)
    sm = 1.0 / math.sqrt(64)
    o, lse = sageattn2_ref(q, k, v, qk="none", pv="f16", smooth_k=True, is_causal=False, return_lse=True)
    assert o.dtype == torch.float32  # q is f32
    km = k.mean(dim=2, keepdim=True)
    k_p = k - km
    scores = (q @ k_p.transpose(-1, -2)) * sm
    lse_want = torch.logsumexp(scores, dim=-1) + (q * km).sum(-1) * sm
    torch.testing.assert_close(lse, lse_want, rtol=1e-5, atol=0)
    q_m = _randn_f32(1, 1, 4, 64, seed=123)
    k_m = _randn_f32(1, 1, 2, 64, seed=124)
    v_m = _randn_f32(1, 1, 2, 64, seed=125)
    o_m, lse_m = sageattn2_ref(q_m, k_m, v_m, qk="none", pv="f32", smooth_k=True, is_causal=True, return_lse=True)
    assert torch.all(o_m[:, :, 0, :] == 0)
    assert torch.all(torch.isneginf(lse_m[:, :, 0]))


def test_check11_int8_not_nibble_packed_and_records_vs_int4():
    g = torch.Generator().manual_seed(11)
    q = torch.randn(1, 1, 32, 64, generator=g, dtype=torch.float32).to(torch.float16)
    k = torch.randn(1, 1, 32, 64, generator=g, dtype=torch.float32).to(torch.float16)
    v = torch.randn(1, 1, 32, 64, generator=g, dtype=torch.float32).to(torch.float16)
    o4 = sageattn2_ref(q, k, v, qk="int4", pv="f16", smooth_k=True, is_causal=False)
    o8 = sageattn2_ref(q, k, v, qk="int8", pv="f16", smooth_k=True, is_causal=False)
    assert o4.dtype == torch.float16 and o8.dtype == torch.float16
    x = np.full((1, 64), 10.0, dtype=np.float32)
    q8, _ = quant_per_token(x, INT8_QMAX)
    assert int(np.max(np.abs(q8))) > 7
    oracle = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=1.0 / math.sqrt(64)
    )

    def _cos(a, b):
        a = a.float().reshape(-1)
        b = b.float().reshape(-1)
        den = a.norm() * b.norm()
        if float(den) == 0.0:
            return 1.0 if float(a.norm() + b.norm()) == 0.0 else 0.0
        return float((a * b).sum() / den)

    c4, c8 = _cos(o4, oracle), _cos(o8, oracle)
    print(f"check11 cosine INT4={c4:.6f} INT8={c8:.6f}")
    # record only; do not gate G1–G5 on this cell


def test_check13_bf16_v_cast_to_fp16_before_g4_pv():
    g = torch.Generator().manual_seed(13)
    q = torch.randn(1, 1, 32, 64, generator=g, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(1, 1, 32, 64, generator=g, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(1, 1, 32, 64, generator=g, dtype=torch.float32).to(torch.bfloat16)
    o = sageattn2_ref(q, k, v, qk="none", pv="f16", smooth_k=False, is_causal=False)
    assert o.dtype == torch.bfloat16
    o_ref = sageattn2_ref(
        q.float(),
        k.float(),
        v.to(torch.float16).float(),
        qk="none",
        pv="f16",
        smooth_k=False,
        is_causal=False,
    )
    torch.testing.assert_close(o.float(), o_ref.to(torch.bfloat16).float(), atol=0.0, rtol=0.0)


def test_int4_default_qk_runs_and_is_not_identity_of_none():
    g = torch.Generator().manual_seed(64)
    q = torch.randn(1, 1, 16, 64, generator=g).to(torch.float16)
    k = torch.randn(1, 1, 16, 64, generator=g).to(torch.float16)
    v = torch.randn(1, 1, 16, 64, generator=g).to(torch.float16)
    o4 = sageattn2_ref(q, k, v, pv="f16", smooth_k=True)
    o_n = sageattn2_ref(q, k, v, qk="none", pv="f16", smooth_k=True)
    assert o4.shape == o_n.shape
    assert not torch.equal(o4.float(), o_n.float())


def test_check7_gqa_int4_repeat_after_quant():
    g = torch.Generator().manual_seed(71)
    q = torch.randn(1, 4, 16, 64, generator=g).to(torch.float16)
    k = torch.randn(1, 2, 16, 64, generator=g).to(torch.float16)
    v = torch.randn(1, 2, 16, 64, generator=g).to(torch.float16)
    o = sageattn2_ref(q, k, v, qk="int4", pv="f16", smooth_k=True, is_causal=False)
    o2 = sageattn2_ref(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        qk="int4",
        pv="f16",
        smooth_k=True,
        is_causal=False,
    )
    torch.testing.assert_close(o.float(), o2.float(), atol=0.0, rtol=0.0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    den = float(a.norm() * b.norm())
    if den == 0.0:
        return 1.0 if float(a.norm() + b.norm()) == 0.0 else 0.0
    return float((a * b).sum() / den)


@pytest.mark.skipif(not _HAS_E4M3, reason="torch.float8_e4m3fn required")
@pytest.mark.parametrize(
    "shape,seed",
    (
        ((1, 1, 32, 64), 100),
        ((1, 1, 128, 128), 101),
    ),
)
def test_check10_fp8_offset_cancels_in_o_over_l(shape, seed):
    b, h, s, d = shape
    g = torch.Generator().manual_seed(seed)
    q = (torch.rand(b, h, s, d, generator=g) * 2 - 1).to(torch.float16)
    k = (torch.rand(b, h, s, d, generator=g) * 2 - 1).to(torch.float16)
    v = (torch.rand(b, h, s, d, generator=g) * 2 - 1).to(torch.float16)
    o5, lse5 = sageattn2_ref(q, k, v, qk="int4", pv="e4m3", smooth_k=True, is_causal=False, return_lse=True)
    o_f32 = sageattn2_ref(
        q,
        k,
        v,
        qk="int4",
        pv="f32",
        apply_fp8_offset=False,
        smooth_k=True,
        is_causal=False,
    )
    assert _cosine(o5, o_f32) >= 0.999
    o4, lse4 = sageattn2_ref(q, k, v, qk="int4", pv="f16", smooth_k=True, is_causal=False, return_lse=True)
    torch.testing.assert_close(lse5, lse4, atol=1e-4, rtol=0)


@pytest.mark.skipif(not _HAS_E4M3, reason="torch.float8_e4m3fn required")
def test_check14_g5_causal_empty_tail_finite():
    g = torch.Generator().manual_seed(14)
    q = torch.randn(1, 1, 16, 64, generator=g).to(torch.float16)
    k = torch.randn(1, 1, 256, 64, generator=g).to(torch.float16)
    v = torch.randn(1, 1, 256, 64, generator=g).to(torch.float16)
    o, lse = sageattn2_ref(q, k, v, qk="none", pv="e4m3", smooth_k=False, is_causal=True, return_lse=True)
    assert torch.isfinite(o.float()).all()
    assert not torch.isnan(lse).any()


@pytest.mark.skipif(not _HAS_E4M3, reason="torch.float8_e4m3fn required")
def test_check13_bf16_v_cast_to_fp16_before_g5_quant():
    g = torch.Generator().manual_seed(130)
    q = torch.randn(1, 1, 32, 64, generator=g).to(torch.bfloat16)
    k = torch.randn(1, 1, 32, 64, generator=g).to(torch.bfloat16)
    v = torch.randn(1, 1, 32, 64, generator=g).to(torch.bfloat16)
    o = sageattn2_ref(q, k, v, qk="none", pv="e4m3", smooth_k=False, is_causal=False)
    assert o.dtype == torch.bfloat16
    o_ref = sageattn2_ref(
        q.float(),
        k.float(),
        v.to(torch.float16).float(),
        qk="none",
        pv="e4m3",
        smooth_k=False,
        is_causal=False,
    )
    torch.testing.assert_close(o.float(), o_ref.to(q.dtype).float(), atol=0.0, rtol=0.0)


@pytest.mark.skipif(not _HAS_E4M3, reason="torch.float8_e4m3fn required")
def test_gqa_e4m3_repeat_v_q():
    g = torch.Generator().manual_seed(75)
    q = torch.randn(1, 4, 16, 64, generator=g).to(torch.float16)
    k = torch.randn(1, 2, 16, 64, generator=g).to(torch.float16)
    v = torch.randn(1, 2, 16, 64, generator=g).to(torch.float16)
    o = sageattn2_ref(q, k, v, qk="int4", pv="e4m3", smooth_k=True, is_causal=False)
    o2 = sageattn2_ref(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        qk="int4",
        pv="e4m3",
        smooth_k=True,
        is_causal=False,
    )
    torch.testing.assert_close(o.float(), o2.float(), atol=0.0, rtol=0.0)


_FEASIBILITY_CELLS = [
    # cell_id, name, Hq, Hkv, Sq, Sk, D, causal, dist
    (0, "A1F", 1, 1, 32, 32, 64, False, 1),
    (1, "A1T", 1, 1, 32, 32, 64, True, 1),
    (2, "A2F", 1, 1, 32, 32, 64, False, 2),
    (3, "A2T", 1, 1, 32, 32, 64, True, 2),
    (4, "B1F", 1, 1, 128, 128, 128, False, 1),
    (5, "B1T", 1, 1, 128, 128, 128, True, 1),
    (6, "B2F", 1, 1, 128, 128, 128, False, 2),
    (7, "B2T", 1, 1, 128, 128, 128, True, 2),
    (8, "C1F", 8, 2, 128, 128, 64, False, 1),
    (9, "C1T", 8, 2, 128, 128, 64, True, 1),
    (10, "C2F", 8, 2, 128, 128, 64, False, 2),
    (11, "C2T", 8, 2, 128, 128, 64, True, 2),
    (12, "D1T", 1, 1, 512, 512, 128, True, 1),
]


def _draw(shape, dist, generator):
    if dist == 1:
        return torch.randn(*shape, generator=generator)
    return torch.rand(*shape, generator=generator) * 2 - 1


def _min_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    out = []
    for i in range(a.shape[0]):
        out.append(_cosine(a[i], b[i]))
    return float(min(out))


@pytest.mark.parametrize("cell_id,name,hq,hkv,sq,sk,d,causal,dist", _FEASIBILITY_CELLS)
def test_check8_int4_feasibility_gate(cell_id, name, hq, hkv, sq, sk, d, causal, dist):
    gen = torch.Generator().manual_seed(20260917 + cell_id)
    q = _draw((1, hq, sq, d), dist, gen).to(torch.float16)
    k = _draw((1, hkv, sk, d), dist, gen).to(torch.float16)
    v = _draw((1, hkv, sk, d), dist, gen).to(torch.float16)
    sm = 1.0 / math.sqrt(d)
    o = sageattn2_ref(q, k, v, is_causal=causal, sm_scale=sm, smooth_k=True, qk="int4", pv="f16")
    k_o = k if hq == hkv else k.repeat_interleave(hq // hkv, dim=1)
    v_o = v if hq == hkv else v.repeat_interleave(hq // hkv, dim=1)
    oracle = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k_o.float(), v_o.float(), is_causal=causal, scale=sm
    )
    of, oo = o.float(), oracle.float()
    cos = _cosine(of, oo)
    max_abs = float((of - oo).abs().max())
    mean_abs = float((of - oo).abs().mean())
    rmse = float(torch.sqrt(((of - oo) ** 2).mean()))
    min_row = _min_row_cosine(of, oo)
    print(
        f"| {name[0]} | {dist} | {name[-1]} | {cos:.6f} | {max_abs:.4e} | {mean_abs:.4e} | {rmse:.4e} | min_row={min_row:.6f} |"
    )
    floor = 0.99 if dist == 2 else 0.98
    assert cos >= floor, f"{name} cosine {cos:.6f} < {floor}"


def test_record_only_extra_metrics():
    """Spec §11.2 extras. Print; do not gate G1–G5."""
    # Smooth-K off, INT8, pv=e4m3 vs f16, outlier 1% K ×10 on shape B.
    gen = torch.Generator().manual_seed(20260917 + 4)  # B1F draw
    q = _draw((1, 1, 128, 128), 1, gen).to(torch.float16)
    k = _draw((1, 1, 128, 128), 1, gen).to(torch.float16)
    v = _draw((1, 1, 128, 128), 1, gen).to(torch.float16)
    sm = 1.0 / math.sqrt(128)
    oracle = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=False, scale=sm
    )
    o_off = sageattn2_ref(q, k, v, qk="int4", pv="f16", smooth_k=False, is_causal=False)
    o_on = sageattn2_ref(q, k, v, qk="int4", pv="f16", smooth_k=True, is_causal=False)
    o8 = sageattn2_ref(q, k, v, qk="int8", pv="f16", smooth_k=True, is_causal=False)
    print("smoothk_off", _cosine(o_off, oracle), "smoothk_on", _cosine(o_on, oracle), "int8", _cosine(o8, oracle))
    if _HAS_E4M3:
        o5 = sageattn2_ref(q, k, v, qk="int4", pv="e4m3", smooth_k=True, is_causal=False)
        print("e4m3_vs_f16", _cosine(o5, o_on), "e4m3_vs_oracle", _cosine(o5, oracle))
    k_out = k.clone()
    n = max(1, int(round(0.01 * 128)))
    k_out[:, :, :n, :] = k_out[:, :, :n, :] * 10
    o_out_on = sageattn2_ref(q, k_out, v, qk="int4", pv="f16", smooth_k=True, is_causal=False)
    o_out_off = sageattn2_ref(q, k_out, v, qk="int4", pv="f16", smooth_k=False, is_causal=False)
    oracle_out = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k_out.float(), v.float(), is_causal=False, scale=sm
    )
    print("outlier_on", _cosine(o_out_on, oracle_out), "outlier_off", _cosine(o_out_off, oracle_out))
