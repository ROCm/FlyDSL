# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""G-I golden. NumPy only. G0/G2 tests import or copy these functions.

gfx120 layout golden for SageAttention2. M(v) = 8*g + v.
"""

from __future__ import annotations

import numpy as np

from tests.kernels.int4_pack import pack_int4, unpack_int4  # noqa: F401

LOG2E = 1.4426950408889634


def preshuffle_b_fp8(logical: np.ndarray) -> np.ndarray:
    """logical V [S_k, D] uint8/e4m3 bytes → [D/16, S_k/16, 2, 16, 8]."""
    sk, d = logical.shape
    if sk % 16 != 0 or d % 16 != 0:
        raise ValueError(f"S_k and D must be multiples of 16, got {sk}, {d}")
    k0, n0 = sk // 16, d // 16
    return np.ascontiguousarray(logical.reshape(k0, 2, 8, n0, 16).transpose(3, 0, 1, 4, 2))


def unshuffle_b_fp8(stored: np.ndarray) -> np.ndarray:
    """Inverse of preshuffle_b_fp8 → [S_k, D]."""
    n0, k0, two, sixteen, eight = stored.shape
    assert two == 2 and sixteen == 16 and eight == 8
    logical = stored.transpose(1, 2, 4, 0, 3).reshape(k0 * 16, n0 * 16)
    return np.ascontiguousarray(logical)


def c_frags_to_tile(c: np.ndarray, arch: str) -> np.ndarray:
    """QK C fragments c [32, 8] -> P tile [16, 16] indexed [q, k].

    QK computes S^T, so a lane owns one q (= l16) across 8 k (= M(v)).
    """
    if c.shape != (32, 8):
        raise ValueError(f"C fragment must be [32, 8], got {c.shape}")
    p = np.zeros((16, 16), dtype=c.dtype)
    for lane in range(32):
        l16, g = lane % 16, lane // 16
        for v in range(8):
            if arch != "gfx120":
                raise ValueError(arch)
            k = 8 * g + v
            p[l16, k] = c[lane, v]
    return p


def tile_to_a_frags(p: np.ndarray, arch: str) -> np.ndarray:
    """P [16, 16] ([q, k]) -> gfx120 PV A fragments [32, 8].

    This is the identity on the QK C fragment: a_frags == c.
    """
    if arch != "gfx120":
        raise ValueError(arch)
    rows = []
    for lane in range(32):
        l16, g = lane % 16, lane // 16
        rows.append(p[l16, g * 8 : g * 8 + 8])
    return np.stack(rows, axis=0)


def pack_a_e4m3_bytes(a_e4m3: np.ndarray) -> np.ndarray:
    """gfx120 A: [32, 8] uint8 → [32, 2] int32 little-endian."""
    if a_e4m3.shape != (32, 8):
        raise ValueError(f"expected [32, 8] bytes, got {a_e4m3.shape}")
    blob = np.ascontiguousarray(a_e4m3, dtype=np.uint8)
    return blob.view(np.uint32).reshape(32, 2).view(np.int32)


def alpha_to_pv_axis(alpha_q: np.ndarray, arch: str) -> np.ndarray:
    """Re-axis the softmax rescale factor for the O accumulator.

    alpha_q [16] is indexed by q; the PV C fragment wants q = M(v) per lane.
    Returns [32, 8].
    """
    if alpha_q.shape != (16,):
        raise ValueError(f"alpha must be [16], got {alpha_q.shape}")
    out = np.empty((32, 8), dtype=alpha_q.dtype)
    for lane in range(32):
        g = lane // 16
        for v in range(8):
            if arch != "gfx120":
                raise ValueError(arch)
            out[lane, v] = alpha_q[8 * g + v]
    return out


def scale_indices(lane: int, q0: int, k0: int, arch: str) -> tuple[int, list[int]]:
    """Return (1 q_scale index, 8 k_scale indices) for a QK C fragment lane."""
    l16, g = lane % 16, lane // 16
    if arch != "gfx120":
        raise ValueError(arch)
    cols = [k0 + 8 * g + v for v in range(8)]
    return q0 + l16, cols


def dequant_c_i32(
    c_i32: np.ndarray,
    q_scale: np.ndarray,
    k_scale: np.ndarray,
    sm_scale: float,
    q0: int,
    k0: int,
    arch: str,
) -> np.ndarray:
    """c_i32 [32, 8] → S_f32 [32, 8] (exp2 domain)."""
    out = np.empty_like(c_i32, dtype=np.float32)
    for lane in range(32):
        row, cols = scale_indices(lane, q0, k0, arch)
        alpha = np.float32(q_scale[row] * sm_scale * LOG2E)
        for v in range(8):
            out[lane, v] = np.float32(c_i32[lane, v]) * np.float32(k_scale[cols[v]]) * alpha
    return out


def gather_qk_iu4_k16(
    Q_i4: np.ndarray,
    K_i4: np.ndarray,
    lane: int,
    q0: int,
    k0: int,
    d0: int,
    arch: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Packed gather for one 16×16×16 IU4 WMMA. Returns (A_words, B_words).

    A is K, B is Q (§1.1). d0 % 16 == 0.
    gfx120: one i32; wave half selects the word.
    """
    if d0 % 16 != 0:
        raise ValueError(f"d0 must be a multiple of 16, got {d0}")
    if arch != "gfx120":
        raise ValueError(arch)
    l16, g = lane % 16, lane // 16
    return (
        np.array([K_i4[k0 + l16, d0 // 8 + g]], dtype=K_i4.dtype),
        np.array([Q_i4[q0 + l16, d0 // 8 + g]], dtype=Q_i4.dtype),
    )


def gather_qk_iu4_k32(
    Q_i4: np.ndarray,
    K_i4: np.ndarray,
    lane: int,
    q0: int,
    k0: int,
    d0: int,
) -> tuple[np.ndarray, np.ndarray]:
    """gfx120 16×16×32 IU4 gather. Returns two-word A and B.

    Authority: RDNA4 ISA §7.12.2 (`K = g*16 + v`). Not gfx1250 / MI355.
    d0 % 32 == 0. A is K, B is Q.
    """
    if d0 % 32 != 0:
        raise ValueError(f"d0 must be a multiple of 32, got {d0}")
    l16, g = lane % 16, lane // 16
    lo = d0 // 8 + 2 * g
    return K_i4[k0 + l16, lo : lo + 2].copy(), Q_i4[q0 + l16, lo : lo + 2].copy()
