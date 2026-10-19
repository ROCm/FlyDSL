# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""L0 tests for the SageAttention2 G-I layout golden (spec §6)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tests.kernels.sageattn2_layout import (
    LOG2E,
    alpha_to_pv_axis,
    c_frags_to_tile,
    dequant_c_i32,
    gather_qk_iu4_k16,
    gather_qk_iu4_k32,
    pack_a_e4m3_bytes,
    pack_int4,
    preshuffle_b_fp8,
    scale_indices,
    tile_to_a_frags,
    unpack_int4,
    unshuffle_b_fp8,
)

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ARCHS = ("gfx120",)


def test_pack_int4_roundtrip_signed_range():
    rng = np.random.default_rng(1)
    logical = rng.integers(-8, 8, size=(5, 64), dtype=np.int8)
    stored = pack_int4(logical)
    assert stored.shape == (5, 8)
    assert stored.dtype == np.int32
    np.testing.assert_array_equal(unpack_int4(stored), logical)


def test_pack_int4_little_endian_nibble_order():
    logical = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int8)
    stored = pack_int4(logical)
    assert stored.shape == (1, 1)
    assert stored.view(np.uint32)[0, 0] == np.uint32(0x76543210)


def test_pack_int4_two_complement_nibble_for_neg8():
    logical = np.array([[-8, 7, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    stored = pack_int4(logical)
    assert stored.view(np.uint32)[0, 0] == np.uint32(0x00000078)
    np.testing.assert_array_equal(unpack_int4(stored), logical)


def test_pack_int4_matches_rdna3_wmma_atom_fixture():
    rng = np.random.default_rng(2)
    logical = rng.integers(-8, 8, size=(4, 16), dtype=np.int8)
    shifts = (np.arange(8, dtype=np.int64) * 4).reshape(1, 1, 8)
    fixture = ((logical.astype(np.int64) & 0xF).reshape(4, 2, 8) << shifts).sum(axis=-1).astype(np.int32)
    np.testing.assert_array_equal(pack_int4(logical), fixture)


def test_pack_int4_rejects_d_not_multiple_of_8():
    with pytest.raises(ValueError, match="multiple of 8"):
        pack_int4(np.zeros((1, 12), dtype=np.int8))


def test_preshuffle_b_fp8_roundtrip():
    rng = np.random.default_rng(3)
    logical = rng.integers(0, 256, size=(32, 64), dtype=np.uint8)
    stored = preshuffle_b_fp8(logical)
    assert stored.shape == (4, 2, 2, 16, 8)
    np.testing.assert_array_equal(unshuffle_b_fp8(stored), logical)


def test_preshuffle_b_fp8_element_identity():
    rng = np.random.default_rng(4)
    sk, d = 32, 64
    logical = rng.integers(0, 256, size=(sk, d), dtype=np.uint8)
    stored = preshuffle_b_fp8(logical)
    for db in range(d // 16):
        for kb in range(sk // 16):
            for lane in range(32):
                for val in range(8):
                    want = logical[kb * 16 + (lane // 16) * 8 + val, db * 16 + (lane % 16)]
                    got = stored[db, kb, lane // 16, lane % 16, val]
                    assert got == want


def test_preshuffle_b_fp8_matches_gemm_source_permute():
    src = (_REPO / "kernels/gemm/rdna_fp8_preshuffle_gemm.py").read_text()
    assert "B_reshaped.permute(3, 0, 1, 4, 2)" in src
    rng = np.random.default_rng(5)
    logical = rng.integers(0, 256, size=(16, 32), dtype=np.uint8)
    k0, n0 = 1, 2
    cited = np.ascontiguousarray(logical.reshape(k0, 2, 8, n0, 16).transpose(3, 0, 1, 4, 2))
    np.testing.assert_array_equal(preshuffle_b_fp8(logical), cited)


def test_g5_pv_b_is_preshuffle_fragment():
    rng = np.random.default_rng(6)
    logical = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    stored = preshuffle_b_fp8(logical)
    k0, d0 = 0, 0
    for lane in range(32):
        l16, g = lane % 16, lane // 16
        frag = stored[d0 // 16, k0 // 16, g, l16, :]
        want = logical[k0 + g * 8 : k0 + g * 8 + 8, d0 + l16]
        np.testing.assert_array_equal(frag, want)


def test_preshuffle_b_fp8_rejects_unaligned_sk_or_d():
    with pytest.raises(ValueError, match="multiples of 16"):
        preshuffle_b_fp8(np.zeros((15, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="multiples of 16"):
        preshuffle_b_fp8(np.zeros((16, 15), dtype=np.uint8))


def _m_of_v(arch: str, lane: int, v: int) -> int:
    if arch != "gfx120":
        raise ValueError(arch)
    g = lane // 16
    return 8 * g + v


@pytest.mark.parametrize("arch", _ARCHS)
def test_c_frags_to_tile_places_s_at_q_l16_k_mv(arch):
    c = np.arange(32 * 8, dtype=np.int32).reshape(32, 8)
    p = c_frags_to_tile(c, arch)
    assert p.shape == (16, 16)
    for lane in range(32):
        l16 = lane % 16
        for v in range(8):
            assert p[l16, _m_of_v(arch, lane, v)] == c[lane, v]


def test_gfx120_c_to_a_is_identity():
    rng = np.random.default_rng(7)
    c = rng.integers(-100, 100, size=(32, 8), dtype=np.int32)
    a = tile_to_a_frags(c_frags_to_tile(c, "gfx120"), "gfx120")
    assert a.shape == (32, 8)
    np.testing.assert_array_equal(a, c)


def test_pack_a_e4m3_bytes_little_endian():
    a = np.arange(32 * 8, dtype=np.uint8).reshape(32, 8)
    packed = pack_a_e4m3_bytes(a)
    assert packed.shape == (32, 2)
    assert packed.dtype == np.int32
    blob = packed.view(np.uint8).reshape(32, 8)
    np.testing.assert_array_equal(blob, a)


def test_c_frags_to_tile_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\[32, 8\]"):
        c_frags_to_tile(np.zeros((32, 16)), "gfx120")


@pytest.mark.parametrize("arch", _ARCHS)
def test_scale_indices_one_q_and_eight_k(arch):
    q0, k0 = 32, 64
    for lane in range(32):
        q_idx, k_idxs = scale_indices(lane, q0, k0, arch)
        assert q_idx == q0 + (lane % 16)
        assert len(k_idxs) == 8
        assert k_idxs == [k0 + _m_of_v(arch, lane, v) for v in range(8)]


def test_gfx120_k_scale_indices_are_contiguous():
    _, cols = scale_indices(lane=17, q0=0, k0=0, arch="gfx120")
    assert cols == [8, 9, 10, 11, 12, 13, 14, 15]


def test_log2e_matches_kernels_common_source():
    src = (_REPO / "kernels/common/kernels_common.py").read_text()
    assert "LOG2E = 1.4426950408889634" in src
    assert LOG2E == 1.4426950408889634


@pytest.mark.parametrize("arch", _ARCHS)
def test_dequant_c_i32_per_token_formula(arch):
    q0, k0 = 0, 0
    c = np.ones((32, 8), dtype=np.int32)
    q_scale = np.arange(16, dtype=np.float32) + 1.0
    k_scale = np.arange(16, dtype=np.float32) + 0.5
    sm_scale = 0.125
    got = dequant_c_i32(c, q_scale, k_scale, sm_scale, q0, k0, arch)
    for lane in range(32):
        row, cols = scale_indices(lane, q0, k0, arch)
        alpha = np.float32(q_scale[row] * sm_scale * LOG2E)
        for v in range(8):
            want = np.float32(1.0) * np.float32(k_scale[cols[v]]) * alpha
            assert got[lane, v] == want


@pytest.mark.parametrize("arch", _ARCHS)
def test_alpha_to_pv_axis_indexes_m_of_v(arch):
    alpha_q = np.arange(16, dtype=np.float32)
    out = alpha_to_pv_axis(alpha_q, arch)
    assert out.shape == (32, 8)
    for lane in range(32):
        for v in range(8):
            assert out[lane, v] == alpha_q[_m_of_v(arch, lane, v)]


def _place_ab_k16(Q_i4, K_i4, q0, k0, d0, arch):
    a_mat = np.zeros((16, 16), dtype=np.int64)
    b_mat = np.zeros((16, 16), dtype=np.int64)
    for lane in range(32):
        a_w, b_w = gather_qk_iu4_k16(Q_i4, K_i4, lane, q0, k0, d0, arch)
        assert a_w.shape == (1,) and b_w.shape == (1,)
        a_n = unpack_int4(a_w.reshape(1, -1))[0].astype(np.int64)
        b_n = unpack_int4(b_w.reshape(1, -1))[0].astype(np.int64)
        l16, g = lane % 16, lane // 16
        a_mat[l16, g * 8 : g * 8 + 8] = a_n
        b_mat[l16, g * 8 : g * 8 + 8] = b_n
    return a_mat, b_mat


def _c_frags_from_hw(c_hw, arch):
    c = np.empty((32, 8), dtype=np.int64)
    for lane in range(32):
        l16 = lane % 16
        for v in range(8):
            c[lane, v] = c_hw[_m_of_v(arch, lane, v), l16]
    return c


def simulate_qk_s_tile_k16(Q, K, q0, k0, arch):
    """Reconstruct S[16, 16] = Q @ K.T for one 16×16 tile via gather + ThrVal + §3.5."""
    Q_i4 = pack_int4(Q)
    K_i4 = pack_int4(K)
    d = Q.shape[1]
    c_hw = np.zeros((16, 16), dtype=np.int64)
    for d0 in range(0, d, 16):
        a_mat, b_mat = _place_ab_k16(Q_i4, K_i4, q0, k0, d0, arch)
        c_hw += a_mat @ b_mat.T
    return c_frags_to_tile(_c_frags_from_hw(c_hw, arch), arch)


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize("d", [16, 64])
def test_check5_k16_gather_matches_q_k_t(arch, d):
    rng = np.random.default_rng(9)
    Q = rng.integers(-8, 8, size=(16, d), dtype=np.int8)
    K = rng.integers(-8, 8, size=(16, d), dtype=np.int8)
    Q[0, 0] = -8
    K[1, 0] = 7
    got = simulate_qk_s_tile_k16(Q, K, q0=0, k0=0, arch=arch)
    want = Q.astype(np.int64) @ K.astype(np.int64).T
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("arch", _ARCHS)
def test_check5_k16_swapped_operands_do_not_match(arch):
    rng = np.random.default_rng(10)
    Q = rng.integers(-8, 8, size=(16, 16), dtype=np.int8)
    K = rng.integers(-8, 8, size=(16, 16), dtype=np.int8)
    Q_i4, K_i4 = pack_int4(Q), pack_int4(K)
    c_hw = np.zeros((16, 16), dtype=np.int64)
    a_mat = np.zeros((16, 16), dtype=np.int64)
    b_mat = np.zeros((16, 16), dtype=np.int64)
    for lane in range(32):
        a_w, b_w = gather_qk_iu4_k16(K_i4, Q_i4, lane, 0, 0, 0, arch)
        a_n = unpack_int4(a_w.reshape(1, -1))[0].astype(np.int64)
        b_n = unpack_int4(b_w.reshape(1, -1))[0].astype(np.int64)
        l16, g = lane % 16, lane // 16
        a_mat[l16, g * 8 : g * 8 + 8] = a_n
        b_mat[l16, g * 8 : g * 8 + 8] = b_n
    c_hw = a_mat @ b_mat.T
    got = c_frags_to_tile(_c_frags_from_hw(c_hw, arch), arch)
    want = Q.astype(np.int64) @ K.astype(np.int64).T
    assert not np.array_equal(got, want)


def test_g4_pv_b_f16_identity():
    rng = np.random.default_rng(12)
    V = rng.standard_normal((16, 16)).astype(np.float32)
    k0, d0 = 0, 0
    for lane in range(32):
        l16, g = lane % 16, lane // 16
        gfx120 = V[k0 + g * 8 : k0 + g * 8 + 8, d0 + l16]
        assert gfx120.shape == (8,)
        np.testing.assert_array_equal(V[k0 : k0 + 16, d0 + l16][g * 8 : g * 8 + 8], gfx120)


def test_gather_qk_iu4_k16_rejects_bad_d0_and_arch():
    Q_i4 = pack_int4(np.zeros((16, 16), dtype=np.int8))
    K_i4 = pack_int4(np.zeros((16, 16), dtype=np.int8))
    with pytest.raises(ValueError, match="multiple of 16"):
        gather_qk_iu4_k16(Q_i4, K_i4, 0, 0, 0, 8, "gfx120")
    with pytest.raises(ValueError):
        gather_qk_iu4_k16(Q_i4, K_i4, 0, 0, 0, 0, "gfx1250")


def _gfx1250_wrong_k32_gather(Q_i4, K_i4, lane, q0, k0, d0):
    """MI355 numBlocks==2 pairing: word index 2*w+g instead of ISA 2*g+w. Do not ship."""
    l16, g = lane % 16, lane // 16
    words = []
    for w in range(2):
        col = d0 // 8 + 2 * w + g
        words.append((K_i4[k0 + l16, col], Q_i4[q0 + l16, col]))
    a = np.array([words[0][0], words[1][0]], dtype=K_i4.dtype)
    b = np.array([words[0][1], words[1][1]], dtype=Q_i4.dtype)
    return a, b


def _place_ab_k32(Q_i4, K_i4, q0, k0, d0, gather_fn):
    a_mat = np.zeros((16, 32), dtype=np.int64)
    b_mat = np.zeros((16, 32), dtype=np.int64)
    for lane in range(32):
        a_w, b_w = gather_fn(Q_i4, K_i4, lane, q0, k0, d0)
        assert a_w.shape == (2,) and b_w.shape == (2,)
        a_n = unpack_int4(a_w.reshape(1, -1))[0].astype(np.int64)
        b_n = unpack_int4(b_w.reshape(1, -1))[0].astype(np.int64)
        l16, g = lane % 16, lane // 16
        a_mat[l16, g * 16 : g * 16 + 16] = a_n
        b_mat[l16, g * 16 : g * 16 + 16] = b_n
    return a_mat, b_mat


def test_k32_isa_k_map_disagrees_with_gfx1250_at_g0_v8():
    g, v = 0, 8
    isa_k = g * 16 + v
    gfx1250_k = (v // 8) * 16 + g * 8 + (v % 8)
    assert isa_k == 8
    assert gfx1250_k == 16
    assert isa_k != gfx1250_k


def _k32_qk_pair():
    Q = ((np.arange(16 * 32).reshape(16, 32) % 8) - 4).astype(np.int8)
    K = ((np.arange(16 * 32).reshape(16, 32) % 7) - 3).astype(np.int8)
    return Q, K


def test_check5_k32_isa_gather_matches_q_k_t():
    Q, K = _k32_qk_pair()
    Q_i4, K_i4 = pack_int4(Q), pack_int4(K)
    a_mat, b_mat = _place_ab_k32(Q_i4, K_i4, 0, 0, 0, gather_qk_iu4_k32)
    c_hw = a_mat @ b_mat.T
    got = c_frags_to_tile(_c_frags_from_hw(c_hw, "gfx120"), "gfx120")
    want = Q.astype(np.int64) @ K.astype(np.int64).T
    np.testing.assert_array_equal(got, want)


def test_check5_k32_gfx1250_gather_does_not_match():
    _, K = _k32_qk_pair()
    Q = ((np.arange(16 * 32).reshape(16, 32) % 5) - 2).astype(np.int8)
    Q_i4, K_i4 = pack_int4(Q), pack_int4(K)
    lane = 20  # l16=4, g=1; 2*g+w != 2*w+g
    a_isa, b_isa = gather_qk_iu4_k32(Q_i4, K_i4, lane, 0, 0, 0)
    a_wrong, b_wrong = _gfx1250_wrong_k32_gather(Q_i4, K_i4, lane, 0, 0, 0)
    assert not np.array_equal(a_isa, a_wrong)
    assert not np.array_equal(b_isa, b_wrong)
    a_isa_mat, _ = _place_ab_k32(Q_i4, K_i4, 0, 0, 0, gather_qk_iu4_k32)
    a_wrong_mat, _ = _place_ab_k32(Q_i4, K_i4, 0, 0, 0, _gfx1250_wrong_k32_gather)
    assert not np.array_equal(a_isa_mat, a_wrong_mat)


def test_gather_qk_iu4_k32_rejects_d0_not_multiple_of_32():
    Q_i4 = pack_int4(np.zeros((16, 32), dtype=np.int8))
    K_i4 = pack_int4(np.zeros((16, 32), dtype=np.int8))
    with pytest.raises(ValueError, match="multiple of 32"):
        gather_qk_iu4_k32(Q_i4, K_i4, 0, 0, 0, 16)


def test_gather_qk_iu4_k32_word_pair_is_2g_plus_w():
    logical_k = np.arange(16 * 32, dtype=np.int32).reshape(16, 32) % 8
    logical_q = (np.arange(16 * 32, dtype=np.int32).reshape(16, 32) % 7) - 3
    K_i4, Q_i4 = pack_int4(logical_k.astype(np.int8)), pack_int4(logical_q.astype(np.int8))
    lane = 20  # l16=4, g=1
    a_w, b_w = gather_qk_iu4_k32(Q_i4, K_i4, lane, 0, 0, 0)
    np.testing.assert_array_equal(a_w, K_i4[4, 2:4])
    np.testing.assert_array_equal(b_w, Q_i4[4, 2:4])
