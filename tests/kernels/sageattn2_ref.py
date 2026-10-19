# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host numerical reference for SageAttention2. gfx120 fragment layout."""

from __future__ import annotations

import math

import numpy as np
import torch

from tests.kernels.sageattn2_layout import LOG2E

EPS = 1e-7
INT4_QMAX = 7
INT8_QMAX = 127
E4M3_MAX = 448.0
S_FP8_OFFSET = np.float32(8.807)
S_FP8_OFFSET_EXP = np.float32(6680.8477)
S_FP8_OFFSET_EXP_INV = np.float32(0.0022326917)
TK = 16


def round_half_away(x: np.ndarray) -> np.ndarray:
    """SageAttn Triton: y + 0.5*sign(y), then trunc toward 0."""
    x = np.asarray(x, dtype=np.float32)
    return np.trunc(x + np.float32(0.5) * np.sign(x))


def quant_per_token(x: np.ndarray, qmax: int) -> tuple[np.ndarray, np.ndarray]:
    """x [..., S, D] fp32 → (q int8, scale [..., S] f32). amax along D."""
    x = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(x), axis=-1)
    scale = amax / np.float32(qmax) + np.float32(EPS)
    q = round_half_away(x / scale[..., None])
    lo, hi = (-8, 7) if qmax == INT4_QMAX else (-128, 127)
    q = np.clip(q, lo, hi).astype(np.int8)
    return q, scale


def dequant_scores(
    s_i32: np.ndarray,
    q_scale: np.ndarray,
    k_scale: np.ndarray,
    sm_scale: float,
) -> np.ndarray:
    """s_i32 [..., Sq, Sk], q_scale [..., Sq], k_scale [..., Sk] → exp2-domain S.

    Association matches G-I: the product q_scale * sm_scale * LOG2E stays in q_scale's dtype (G-I / NEP 50), then astype f32.
    """
    q_scale = np.asarray(q_scale)  # keep dtype; G-I is np.float32(q_scale[q] * sm_scale * LOG2E)
    k_scale = np.asarray(k_scale, dtype=np.float32)
    alpha = (q_scale * float(sm_scale) * float(LOG2E)).astype(np.float32)
    return s_i32.astype(np.float32) * k_scale[..., np.newaxis, :] * alpha[..., np.newaxis]


def causal_mask(s: np.ndarray, sq: int, sk: int) -> np.ndarray:
    """s [..., Sq, Sk]. Queries at the end of KV when Sq != Sk."""
    q = np.arange(sq)[:, None]
    k = np.arange(sk)[None, :]
    return np.where(k > q + (sk - sq), np.float32(-np.inf), s)


def pack_e4m3(x: np.ndarray) -> np.ndarray:
    """f32 → uint8 e4m3fn. Normative via torch.float8_e4m3fn."""
    t = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    return t.to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def quant_v_e4m3(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v [..., Sk, D] fp32 → (uint8 e4m3 same shape, v_scale [..., D])."""
    v = np.asarray(v, dtype=np.float32)
    amax = np.max(np.abs(v), axis=-2)
    scale = amax / np.float32(E4M3_MAX) + np.float32(EPS)
    return pack_e4m3(v / scale[..., None, :]), scale


def softmax_update(
    S_t: np.ndarray,
    m: np.ndarray,
    l: np.ndarray,  # noqa: E741
    *,
    apply_fp8_offset: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One K-tile. S_t [Sq, Tk] unoffset exp2-domain. m,l [Sq].

    Returns (m, l, alpha, P) with P [Sq, Tk], alpha [Sq]. No V.
    """
    S_t = np.asarray(S_t, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    l = np.asarray(l, dtype=np.float32)  # noqa: E741
    m_tile = np.max(S_t, axis=-1)
    offset = S_FP8_OFFSET if apply_fp8_offset else np.float32(0)
    m_cand = np.where(np.isfinite(m_tile), m_tile - offset, m_tile)
    m_ij = np.maximum(m, m_cand)
    empty = ~np.isfinite(m_ij)
    m_ij_safe = np.where(empty, np.float32(0.0), m_ij)
    m_safe = np.where(empty, np.float32(0.0), m)
    p = np.exp2(S_t - m_ij_safe[:, None])
    p = np.where(empty[:, None], np.float32(0.0), p)
    alpha = np.exp2(m_safe - m_ij_safe)
    alpha = np.where(empty, np.float32(0.0), alpha)
    l = l * alpha + p.sum(axis=-1)  # noqa: E741
    return m_ij, l, alpha, p


def _to_cpu_owned(x) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.device.type != "cpu":
        t = t.cpu()
    return t.detach().clone()


def _decode_e4m3(u8: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(u8, dtype=np.uint8))
    return t.view(torch.float8_e4m3fn).to(torch.float32).numpy()


def _pv_accumulate(p: np.ndarray, v_t: np.ndarray, pv: str) -> np.ndarray:
    """P [Sq, Tk], V_t [Tk, D] → [Sq, D] f32. For pv=e4m3, V_t is uint8."""
    if pv == "f32":
        return p @ v_t
    if pv == "f16":
        ph = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32)).to(torch.float16).to(torch.float32)
        vh = torch.from_numpy(np.ascontiguousarray(v_t, dtype=np.float32)).to(torch.float16).to(torch.float32)
        return (ph @ vh).numpy()
    if pv == "e4m3":
        return _decode_e4m3(pack_e4m3(p)) @ _decode_e4m3(v_t)
    raise ValueError(pv)


def sageattn2_ref(
    q,
    k,
    v,
    *,
    is_causal: bool = False,
    sm_scale: float | None = None,
    smooth_k: bool = True,
    qk: str = "int4",
    pv: str = "f16",
    apply_fp8_offset: bool | None = None,
    return_lse: bool = False,
):
    q_t = _to_cpu_owned(q)
    k_t = _to_cpu_owned(k)
    v_t = _to_cpu_owned(v)
    if q_t.dtype != k_t.dtype or q_t.dtype != v_t.dtype:
        raise ValueError("q, k, v must share dtype")
    if qk not in ("int4", "int8", "none"):
        raise ValueError(f"qk={qk}")
    if pv not in ("f16", "e4m3", "f32"):
        raise ValueError(f"pv={pv}")
    if q_t.dtype == torch.float32 and qk != "none":
        raise ValueError("f32 inputs require qk='none'")
    if q_t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(q_t.dtype)
    if q_t.ndim != 4 or k_t.ndim != 4 or v_t.ndim != 4:
        raise ValueError("expected [B, H, S, D]")
    b, hq, sq, d = q_t.shape
    bk, hkv, sk, dk = k_t.shape
    bv, hv, sv, dv = v_t.shape
    if (bk, bv) != (b, b) or dk != d or dv != d or sv != sk or hv != hkv:
        raise ValueError("shape mismatch")
    if hq % hkv != 0:
        raise ValueError("Hq must be a multiple of Hkv")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    if apply_fp8_offset is None:
        apply_fp8_offset = pv == "e4m3"

    q_f = q_t.float().numpy()
    k_f = k_t.float().numpy()
    v_q = None
    v_scale = None
    if pv == "f32":
        v_pv = v_t.float().numpy()
    else:
        v_f16 = v_t.to(torch.float16).float().numpy()
        if pv == "f16":
            v_pv = v_f16
        else:
            v_q, v_scale = quant_v_e4m3(v_f16)
            v_pv = None

    km = None
    if smooth_k:
        km = k_f.mean(axis=2, keepdims=True)
        k_store = torch.from_numpy((k_f - km).astype(np.float32)).to(q_t.dtype)
        k_prime = k_store.float().numpy()
    else:
        k_prime = k_f

    if qk == "none":
        q_i = q_f
        k_i = k_prime
        q_scale = k_scale = None
    else:
        qmax = INT4_QMAX if qk == "int4" else INT8_QMAX
        q_i, q_scale = quant_per_token(q_f, qmax)
        k_i, k_scale = quant_per_token(k_prime, qmax)

    g = hq // hkv
    if g != 1:
        k_i = np.repeat(k_i, g, axis=1)
        if k_scale is not None:
            k_scale = np.repeat(k_scale, g, axis=1)
        if v_pv is not None:
            v_pv = np.repeat(v_pv, g, axis=1)
        if v_q is not None:
            v_q = np.repeat(v_q, g, axis=1)
        if v_scale is not None:
            v_scale = np.repeat(v_scale, g, axis=1)
        km_b = None if km is None else np.repeat(km, g, axis=1)
    else:
        km_b = km

    if qk == "none":
        s = np.matmul(q_i, np.swapaxes(k_i, -1, -2)).astype(np.float32)
        s = s * np.float32(sm_scale) * np.float32(LOG2E)
    else:
        s_i32 = np.matmul(q_i.astype(np.int32), np.swapaxes(k_i.astype(np.int32), -1, -2))
        s = dequant_scores(s_i32, q_scale, k_scale, sm_scale)
    if is_causal:
        s = causal_mask(s, sq, sk)

    o_acc = np.zeros((b, hq, sq, d), dtype=np.float32)
    m = np.full((b, hq, sq), np.float32(-np.inf))
    ell = np.zeros((b, hq, sq), dtype=np.float32)
    for bi in range(b):
        for hi in range(hq):
            m_row = m[bi, hi]
            l_row = ell[bi, hi]
            o_row = o_acc[bi, hi]
            for k0 in range(0, sk, TK):
                s_t = s[bi, hi, :, k0 : k0 + TK]
                m_row, l_row, alpha, p = softmax_update(s_t, m_row, l_row, apply_fp8_offset=apply_fp8_offset)
                sl = s_t.shape[-1]
                if pv == "e4m3":
                    v_t = v_q[bi, hi, k0 : k0 + sl, :]
                else:
                    v_t = v_pv[bi, hi, k0 : k0 + sl, :]
                o_row = o_row * alpha[:, None] + _pv_accumulate(p, v_t, pv)
            m[bi, hi] = m_row
            ell[bi, hi] = l_row
            o_acc[bi, hi] = o_row

    if pv == "e4m3":
        o_acc = o_acc * v_scale[:, :, None, :]

    o = np.where(
        ell[..., None] == 0,
        np.float32(0.0),
        o_acc / np.where(ell[..., None] == 0, 1.0, ell[..., None]),
    )
    out = torch.tensor(np.ascontiguousarray(o), dtype=torch.float32).to(q_t.dtype)
    if not return_lse:
        return out

    masked = (ell == 0) | ~np.isfinite(m)
    lse = (m + np.log2(np.where(masked, np.float32(1.0), ell))) / np.float32(LOG2E)
    if smooth_k:
        lse = lse + np.sum(q_f * km_b, axis=-1) * np.float32(sm_scale)
    lse = np.where(masked, np.float32(-np.inf), lse)
    return out, torch.tensor(np.ascontiguousarray(lse), dtype=torch.float32)
