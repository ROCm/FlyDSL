# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx120 wave32 helpers used by the SageAttention2 prefill kernel."""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import get_warp_size
from kernels.common.tensor_shim import _to_raw as _raw

S_FP8_OFFSET = 8.807


def _reaxis_qk_to_pv(x):
    """One QK-axis f32 (indexed by lane's l16) -> 8 PV-axis f32.

    gfx120: M(v) = 8*g + v. Uses the wave-relative lane (fx.gpu.lane_id()).
    Callers in a multi-wave CTA must not substitute a block-global id.
    """
    lane = fx.gpu.lane_id()
    g = lane // fx.Int32(16)
    src_i32 = fx.Float32(x).bitcast(fx.Int32)
    out = []
    for v in range_constexpr(8):
        src_lane = g * fx.Int32(8) + fx.Int32(v)
        byte_idx = src_lane * fx.Int32(4)
        peer_i32 = fx.Int32(fx.rocdl.ds_bpermute(T.i32, byte_idx, src_i32))
        out.append(peer_i32.bitcast(fx.Float32))
    return out


def _softmax_update_frag(
    S,
    m,
    l,  # noqa: E741
    *,
    apply_fp8_offset: bool,
):
    """One 16x16 k-tile on this wave, gfx120 only.

    Returns (P, m_ij, l_new, alpha_qk, alpha_pv):
      P, alpha_pv: 8 x fx.Float32
      m_ij, l_new, alpha_qk: fx.Float32

    Caller must enter with m[lane] == m[lane^16] and l[lane] == l[lane^16].
    Non-finite running maxes are flushed to zero.
    """
    m_lane = S[0]
    for v in range_constexpr(1, 8):
        m_lane = fx.max(m_lane, S[v])
    m_tile = fx.max(m_lane, m_lane.shuffle_xor(fx.Int32(16), 32))

    offset = fx.Float32(S_FP8_OFFSET) if apply_fp8_offset else fx.Float32(0.0)
    m_tile_finite = fx.isfinite(m_tile)
    m_cand = m_tile_finite.select(m_tile - offset, m_tile)
    m_ij = fx.max(m, m_cand)

    is_fin = fx.isfinite(m_ij)
    m_ij_safe = is_fin.select(m_ij, fx.Float32(0.0))
    m_safe = is_fin.select(m, fx.Float32(0.0))

    P = []
    p_lane = fx.Float32(0.0)
    for v in range_constexpr(8):
        p_v = fx.Float32(fx.rocdl.exp2(T.f32, _raw(S[v] - m_ij_safe)))
        p_v = is_fin.select(p_v, fx.Float32(0.0))
        P.append(p_v)
        p_lane = p_lane + p_v

    alpha_qk = fx.Float32(fx.rocdl.exp2(T.f32, _raw(m_safe - m_ij_safe)))
    alpha_qk = is_fin.select(alpha_qk, fx.Float32(0.0))

    p_sum = p_lane + p_lane.shuffle_xor(fx.Int32(16), 32)
    l_new = l * alpha_qk + p_sum  # noqa: E741

    alpha_pv = _reaxis_qk_to_pv(alpha_qk)

    return P, m_ij, l_new, alpha_qk, alpha_pv


@flyc.kernel
def softmax_update_tile_kernel(
    S: fx.Tensor,
    M: fx.Tensor,
    L: fx.Tensor,
    P: fx.Tensor,
    M_out: fx.Tensor,
    L_out: fx.Tensor,
    AlphaQK: fx.Tensor,
    AlphaPV: fx.Tensor,
    apply_fp8_offset: fx.Constexpr[bool],
):
    lane = fx.Int32(gpu.thread_id("x"))

    s2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(S)), fx.make_layout((32, 8), (8, 1)))
    m1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(M)), fx.make_layout((32,), (1,)))
    l1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(L)), fx.make_layout((32,), (1,)))
    p2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(P)), fx.make_layout((32, 8), (8, 1)))
    m_out1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(M_out)), fx.make_layout((32,), (1,)))
    l_out1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(L_out)), fx.make_layout((32,), (1,)))
    alpha_qk1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(AlphaQK)), fx.make_layout((32,), (1,)))
    alpha_pv2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(AlphaPV)), fx.make_layout((32, 8), (8, 1)))

    S_in = [s2d[lane, fx.Int32(v)] for v in range_constexpr(8)]
    m_in = m1d[lane]
    l_in = l1d[lane]

    P_out, m_ij, l_new, alpha_qk, alpha_pv = _softmax_update_frag(
        S_in, m_in, l_in, apply_fp8_offset=apply_fp8_offset
    )

    for v in range_constexpr(8):
        p2d[lane, fx.Int32(v)] = P_out[v]
        alpha_pv2d[lane, fx.Int32(v)] = alpha_pv[v]
    m_out1d[lane] = m_ij
    l_out1d[lane] = l_new
    alpha_qk1d[lane] = alpha_qk


@flyc.jit
def _launch_softmax_update_tile(
    S: fx.Tensor,
    M: fx.Tensor,
    L: fx.Tensor,
    P: fx.Tensor,
    M_out: fx.Tensor,
    L_out: fx.Tensor,
    AlphaQK: fx.Tensor,
    AlphaPV: fx.Tensor,
    apply_fp8_offset: fx.Constexpr[bool],
    stream: fx.Stream = fx.Stream(None),
):
    softmax_update_tile_kernel(S, M, L, P, M_out, L_out, AlphaQK, AlphaPV, apply_fp8_offset).launch(
        grid=(1,),
        block=(32,),
        stream=stream,
    )


def _require_gfx120() -> None:
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx120"):
        raise RuntimeError(f"softmax_update_tile requires gfx120x, got {arch!r}")


def _as_stream(stream: torch.cuda.Stream | None) -> torch.cuda.Stream:
    if stream is None:
        return torch.cuda.current_stream()
    return stream


def _maybe_contiguous(t: torch.Tensor) -> torch.Tensor:
    if not t.is_contiguous():
        return t.contiguous()
    return t


def _softmax_update_tile(
    S: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,  # noqa: E741
    *,
    apply_fp8_offset: bool,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-wave launch of ``_softmax_update_frag`` for tests.

    S: [32, 8] f32 CUDA, C fragment. m, l: [32] f32 CUDA.

    Returns (P, m_out, l_out, alpha_qk, alpha_pv).
    P, alpha_pv: [32, 8] f32; m_out, l_out, alpha_qk: [32] f32.
    """
    _require_gfx120()
    if not (S.is_cuda and m.is_cuda and l.is_cuda):
        raise ValueError("S, m, l must be CUDA tensors")
    if tuple(S.shape) != (32, 8) or S.dtype != torch.float32:
        raise ValueError(f"S must be [32, 8] float32, got {tuple(S.shape)} {S.dtype}")
    if tuple(m.shape) != (32,) or m.dtype != torch.float32:
        raise ValueError(f"m must be [32] float32, got {tuple(m.shape)} {m.dtype}")
    if tuple(l.shape) != (32,) or l.dtype != torch.float32:
        raise ValueError(f"l must be [32] float32, got {tuple(l.shape)} {l.dtype}")
    assert get_warp_size() == 32, "softmax_update_tile assumes a 32-wide wave"
    stream = _as_stream(stream)
    with torch.cuda.stream(stream):
        S = _maybe_contiguous(S)
        m = _maybe_contiguous(m)
        l = _maybe_contiguous(l)  # noqa: E741
        P = torch.empty(32, 8, dtype=torch.float32, device=S.device)
        m_out = torch.empty(32, dtype=torch.float32, device=S.device)
        l_out = torch.empty(32, dtype=torch.float32, device=S.device)
        alpha_qk = torch.empty(32, dtype=torch.float32, device=S.device)
        alpha_pv = torch.empty(32, 8, dtype=torch.float32, device=S.device)
        _launch_softmax_update_tile(
            S,
            m,
            l,
            P,
            m_out,
            l_out,
            alpha_qk,
            alpha_pv,
            apply_fp8_offset,
            stream=fx.Stream(stream),
        )
        return P, m_out, l_out, alpha_qk, alpha_pv
