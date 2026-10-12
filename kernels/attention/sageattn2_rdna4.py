# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""SageAttention2 RDNA4 (gfx120x) dense prefill: INT4 QK + FP8 PV.

One implementation: b128 loads, k_scale hoisted for the stage, 4 waves
(CTA_Q=64), K=32 WMMA, finite running-max, one block rescale per stage,
and key-tail peeling on non-causal sequences long enough for a masked tail.
"""

from __future__ import annotations

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.sageattn2_softmax import S_FP8_OFFSET, _reaxis_qk_to_pv
from kernels.common.kernels_common import LOG2E as _LOG2E
from kernels.common.tensor_shim import _to_raw as _raw

WARP_Q = 16
WAVES = 4
CTA_Q = WARP_Q * WAVES  # 64
THREADS = 32 * WAVES  # 128
STAGE_K = 64
KSUB = STAGE_K // 16  # 4
KW = 32
M_SENTINEL = -3.4028234663852886e38  # -FLT_MAX

_FINITE_M0_SCORE_BOUND = 1e30


@flyc.kernel
def sageattn2_rdna4_kernel(
    Qi4: fx.Tensor,
    Qs: fx.Tensor,
    Ki4: fx.Tensor,
    Ks: fx.Tensor,
    Ve4: fx.Tensor,
    Vs: fx.Tensor,
    Out: fx.Tensor,
    B: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    Sq: fx.Int32,
    Skp: fx.Int32,
    Sk: fx.Int32,
    sm_scale: fx.Float32,
    D: fx.Constexpr[int],
    is_causal: fx.Constexpr[bool],
    V_IN_LDS: fx.Constexpr[bool],
    KEY_PEEL: fx.Constexpr[bool],
    out_elem: fx.Constexpr[str],
):
    """One dense-prefill CTA. The host flattens every tensor over batch, head, and seq."""
    assert not (KEY_PEEL and is_causal), "key peel is non-causal only"
    DT = D // 16
    assert D % 32 == 0, "V staging requires D // 16 to be even"
    DW = D // 8
    if const_expr(out_elem == "f16"):
        out_dtype = fx.Float16
    elif const_expr(out_elem == "bf16"):
        out_dtype = fx.BFloat16
    else:
        out_dtype = fx.Float32

    tid = fx.Int32(gpu.thread_id("x"))
    q_tile = fx.Int32(gpu.block_id("x"))
    hq = fx.Int32(gpu.block_id("y"))
    b = fx.Int32(gpu.block_id("z"))

    wave = tid // fx.Int32(32)
    lane = fx.gpu.lane_id()
    l16 = lane % fx.Int32(16)
    g = lane // fx.Int32(16)

    hkv = (hq * Hkv) // Hq
    q_cta0 = q_tile * fx.Int32(CTA_Q)
    q0 = q_cta0 + wave * fx.Int32(WARP_Q)

    bh_q = b * Hq + hq
    bh_k = b * Hkv + hkv
    q_row0 = bh_q * Sq
    k_row0 = bh_k * Skp
    Skp_div16 = Skp // fx.Int32(16)

    q2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Qi4)), fx.make_layout((B * Hq * Sq, DW), (DW, 1)))
    k2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Ki4)), fx.make_layout((B * Hkv * Skp, DW), (DW, 1)))
    qs1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Qs)), fx.make_layout((B * Hq * Sq,), (1,)))
    ks1d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Ks)), fx.make_layout((B * Hkv * Skp,), (1,)))
    v_words2d = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(Ve4)),
        fx.make_layout((B * Hkv, DT * Skp_div16 * 64), (DT * Skp_div16 * 64, 1)),
    )
    vs2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Vs)), fx.make_layout((B * Hkv, D), (D, 1)))
    out2d = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Out)), fx.make_layout((B * Hq * Sq, D), (D, 1)))

    q_local_safe = fx.min(q0 + l16, Sq - fx.Int32(1))
    q_row = q_row0 + q_local_safe
    q_words = [
        fx.Vector.from_elements(
            [
                q2d[q_row, fx.Int32(4 * d) + fx.Int32(2) * g],
                q2d[q_row, fx.Int32(4 * d) + fx.Int32(2) * g + fx.Int32(1)],
            ],
            fx.Int32,
        )
        for d in range_constexpr(DT // 2)
    ]
    q_alpha = qs1d[q_row] * sm_scale * fx.Float32(_LOG2E)
    v_s = [vs2d[bh_k, fx.Int32(n * 16) + l16] for n in range_constexpr(DT)]

    qk_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, KW, fx.Int4, fx.Int32, sign_a=True, sign_b=True))
    pv_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Float8E4M3FN, fx.Float32))

    if const_expr(V_IN_LDS):

        @fx.struct
        class _VStage:
            data: fx.Array[fx.Uint8, DT * 1024, 16]

        v_lds_storage = fx.SharedAllocator().allocate(_VStage).peek()
        v_lds = fx.make_view(fx.recast_iter(fx.Int32, v_lds_storage.data.ptr), fx.make_layout((DT * 256,), (1,)))
    else:
        v_lds = None

    Sk_minus_Sq = Sk - Sq
    if const_expr(is_causal):
        k_hi = fx.min(Sk, q_cta0 + fx.Int32(CTA_Q) + Sk_minus_Sq)
    else:
        k_hi = Skp
    n_stage = (k_hi + fx.Int32(STAGE_K - 1)) // fx.Int32(STAGE_K)

    init_state = [fx.Float32(M_SENTINEL), fx.Float32(0.0)] + [fx.Float32(0.0) for _ in range(8 * DT)]

    def _stage_body(state, stage_idx, *, masked):
        # masked=False is the peeled main loop, keys [0, tail_base). Physical
        # Skp clamps are identities there. The tail stage, and every stage when
        # peel does not apply, stays masked.
        stage_idx = fx.Int32(stage_idx)
        m = state[0]
        l = state[1]  # noqa: E741
        O = [[state[2 + n * 8 + v] for v in range_constexpr(8)] for n in range_constexpr(DT)]  # noqa: E741

        k_base = stage_idx * fx.Int32(STAGE_K)
        kb0 = stage_idx * fx.Int32(KSUB)

        if const_expr(V_IN_LDS):
            gpu.barrier()
            vec = 4
            n_words = DT * 256
            n_iters = n_words // (vec * THREADS)
            v_row_iter = fx.get_iter(fx.slice(v_words2d, (bh_k, None)))
            for it in range_constexpr(n_iters):
                i0 = (fx.Int32(it * THREADS) + tid) * fx.Int32(vec)
                db = i0 // fx.Int32(256)
                kb_delta = (i0 % fx.Int32(256)) // fx.Int32(64)
                word_off = i0 % fx.Int32(64)
                kb_idx = kb0 + kb_delta
                if const_expr(masked):
                    kb_safe = fx.min(kb_idx, Skp_div16 - fx.Int32(1))
                else:
                    kb_safe = kb_idx
                addr = (db * Skp_div16 + kb_safe) * fx.Int32(64) + word_off
                raw_vec = fx.make_view(fx.add_offset(v_row_iter, addr), fx.make_layout(4, 1)).load()
                for j in range_constexpr(4):
                    if const_expr(masked):
                        val = (kb_idx < Skp_div16).select(raw_vec[j], fx.Int32(0))
                    else:
                        val = raw_vec[j]
                    v_lds[i0 + fx.Int32(j)] = val
            gpu.barrier()

        ks_stage = [None] * KSUB
        for kt_pf in range_constexpr(KSUB):
            k0_pf = k_base + fx.Int32(kt_pf * 16)
            if const_expr(masked):
                k_col_base_pf = fx.min(k0_pf + fx.Int32(8) * g, Skp - fx.Int32(8))
            else:
                k_col_base_pf = k0_pf + fx.Int32(8) * g
            ks_ptr_pf = fx.add_offset(fx.get_iter(ks1d), k_row0 + k_col_base_pf)
            ks_stage[kt_pf] = (
                fx.make_view(ks_ptr_pf, fx.make_layout(4, 1)).load(),
                fx.make_view(fx.add_offset(ks_ptr_pf, fx.Int32(4)), fx.make_layout(4, 1)).load(),
            )

        def _qk_score_tile(kt):
            ks_lo, ks_hi = ks_stage[kt]
            k0 = k_base + fx.Int32(kt * 16)
            k_row_logical = k0 + l16
            if const_expr(masked):
                k_row_safe = fx.min(k_row_logical, Skp - fx.Int32(1))
            else:
                k_row_safe = k_row_logical
            k_row = k_row0 + k_row_safe

            acc = fx.Vector.filled(8, 0, fx.Int32)
            for d in range_constexpr(DT // 2):
                k_col0 = fx.Int32(4 * d) + fx.Int32(2) * g
                k_ptr = fx.add_offset(fx.get_iter(k2d), k_row * fx.Int32(DW) + k_col0)
                a_vec = fx.make_view(k_ptr, fx.make_layout(2, 1)).load().bitcast(fx.Int4)
                b_vec = q_words[d].bitcast(fx.Int4)
                acc = fx.Vector(
                    fly.mma_atom_call_ssa(
                        [fx.Vector.make_type(8, fx.Int32)],
                        qk_atom,
                        a_vec.ir_value(),
                        b_vec.ir_value(),
                        acc.ir_value(),
                    )
                )

            S = []
            for v in range_constexpr(8):
                k_col_logical = k0 + fx.Int32(8) * g + fx.Int32(v)
                k_s = ks_lo[v] if v < 4 else ks_hi[v - 4]
                s_val = acc[v].to(fx.Float32) * k_s * q_alpha
                if const_expr(masked):
                    bad = k_col_logical >= Sk
                    if const_expr(is_causal):
                        bad = bad | (k_col_logical > ((q0 + l16) + Sk_minus_Sq))
                    s_val = bad.select(fx.Float32(float("-inf")), s_val)
                S.append(s_val)
            return S

        # One block max and one O rescale per stage. m is the finite sentinel,
        # so this path does not use an isfinite guard.
        S_all = [_qk_score_tile(kt) for kt in range_constexpr(KSUB)]

        m_lane = S_all[0][0]
        for kt in range_constexpr(KSUB):
            for v in range_constexpr(8):
                if not (kt == 0 and v == 0):
                    m_lane = fx.max(m_lane, S_all[kt][v])
        m_tile = fx.max(m_lane, m_lane.shuffle_xor(fx.Int32(16), 32))
        m_cand = m_tile - fx.Float32(S_FP8_OFFSET)
        m_ij = fx.max(m, m_cand)
        alpha_qk = fx.Float32(fx.rocdl.exp2(T.f32, _raw(m - m_ij)))
        alpha_pv = _reaxis_qk_to_pv(alpha_qk)
        m = m_ij  # noqa: E741

        O = [[O[n][v] * alpha_pv[v] for v in range_constexpr(8)] for n in range_constexpr(DT)]  # noqa: E741
        l = l * alpha_qk  # noqa: E741

        for kt in range_constexpr(KSUB):
            P = []
            p_lane = fx.Float32(0.0)
            for v in range_constexpr(8):
                p_v = fx.Float32(fx.rocdl.exp2(T.f32, _raw(S_all[kt][v] - m_ij)))
                P.append(p_v)
                p_lane = p_lane + p_v
            p_sum = p_lane + p_lane.shuffle_xor(fx.Int32(16), 32)
            l = l + p_sum  # noqa: E741

            p0 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=P[0], src_b=P[1], old=fx.Int32(0), word_sel=False))
            p0 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=P[2], src_b=P[3], old=p0, word_sel=True))
            p1 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=P[4], src_b=P[5], old=fx.Int32(0), word_sel=False))
            p1 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=P[6], src_b=P[7], old=p1, word_sel=True))
            pf = fx.Vector.from_elements([p0, p1], fx.Int32)

            for n in range_constexpr(DT):
                if const_expr(V_IN_LDS):
                    lds_idx = fx.Int32(n * 256 + kt * 64) + g * fx.Int32(32) + l16 * fx.Int32(2)
                    vf = fx.Vector.from_elements([v_lds[lds_idx], v_lds[lds_idx + fx.Int32(1)]], fx.Int32)
                else:
                    kb_idx = kb0 + fx.Int32(kt)
                    if const_expr(masked):
                        kb_safe = fx.min(kb_idx, Skp_div16 - fx.Int32(1))
                    else:
                        kb_safe = kb_idx
                    addr = (fx.Int32(n) * Skp_div16 + kb_safe) * fx.Int32(64) + (g * fx.Int32(16) + l16) * fx.Int32(2)
                    vf = fx.Vector.from_elements(
                        [v_words2d[bh_k, addr], v_words2d[bh_k, addr + fx.Int32(1)]], fx.Int32
                    )
                acc_o = fx.Vector.from_elements(O[n], fx.Float32)
                result = fx.Vector(
                    fly.mma_atom_call_ssa(
                        [fx.Vector.make_type(8, fx.Float32)],
                        pv_atom,
                        pf.ir_value(),
                        vf.ir_value(),
                        acc_o.ir_value(),
                    )
                )
                O[n] = [result[v] for v in range_constexpr(8)]

        return [m, l] + [O[n][v] for n in range_constexpr(DT) for v in range_constexpr(8)]

    if const_expr(KEY_PEEL):
        for stage_idx, state in range(0, n_stage - 1, 1, init=init_state):
            state = yield _stage_body(state, stage_idx, masked=False)
        state = _stage_body(state, n_stage - fx.Int32(1), masked=True)
    else:
        for stage_idx, state in range(0, n_stage, 1, init=init_state):
            state = yield _stage_body(state, stage_idx, masked=True)

    l_final = state[1]
    O_final = [[state[2 + n * 8 + v] for v in range_constexpr(8)] for n in range_constexpr(DT)]
    l_pv = _reaxis_qk_to_pv(l_final)

    for n in range_constexpr(DT):
        for v in range_constexpr(8):
            q_logical = q0 + fx.Int32(8) * g + fx.Int32(v)
            denom = l_pv[v]
            zero = denom == fx.Float32(0.0)
            safe_denom = zero.select(fx.Float32(1.0), denom)
            out_val = O_final[n][v] * v_s[n] / safe_denom
            out_val = zero.select(fx.Float32(0.0), out_val)
            in_range = q_logical < Sq
            if in_range:
                out2d[q_row0 + q_logical, fx.Int32(n * 16) + l16] = out_val.to(out_dtype)


@flyc.jit
def _sageattn2_rdna4_launch(
    Qi4: fx.Tensor,
    Qs: fx.Tensor,
    Ki4: fx.Tensor,
    Ks: fx.Tensor,
    Ve4: fx.Tensor,
    Vs: fx.Tensor,
    Out: fx.Tensor,
    B: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    Sq: fx.Int32,
    Skp: fx.Int32,
    Sk: fx.Int32,
    sm_scale: fx.Float32,
    D: fx.Constexpr[int],
    is_causal: fx.Constexpr[bool],
    V_IN_LDS: fx.Constexpr[bool],
    KEY_PEEL: fx.Constexpr[bool],
    out_elem: fx.Constexpr[str],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (Sq + CTA_Q - 1) // CTA_Q
    sageattn2_rdna4_kernel(
        Qi4,
        Qs,
        Ki4,
        Ks,
        Ve4,
        Vs,
        Out,
        B,
        Hq,
        Hkv,
        Sq,
        Skp,
        Sk,
        sm_scale,
        D,
        is_causal,
        V_IN_LDS,
        KEY_PEEL,
        out_elem,
    ).launch(grid=(grid_x, Hq, B), block=(THREADS,), stream=stream)


_ALLOWED_D = (64, 128)
_ALLOWED_OUT_DTYPES = {torch.float16: "f16", torch.bfloat16: "bf16"}


def _require_rdna4_arch() -> None:
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx120"):
        raise RuntimeError(f"sageattn2_rdna4_prefill requires gfx120x, got {arch!r}")


def _require_cuda_contig(name: str, t: torch.Tensor, ndim: int, dtype) -> torch.Tensor:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dim() != ndim:
        raise ValueError(f"{name} expected rank {ndim}, got shape {tuple(t.shape)}")
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}, got {t.dtype}")
    return t if t.is_contiguous() else t.contiguous()


def _use_key_peel(is_causal: bool, sk: int, Skp: int, stage_k: int = STAGE_K) -> bool:
    """Peel only when a non-causal tail stage is a real suffix of the padded keys.

    ``Skp - sk < STAGE_K`` is the wrong predicate: it is unsound for
    Skp=112, sk=49 and needlessly disables Skp=128, sk=64.
    """
    tail_base = ((Skp - 1) // stage_k) * stage_k
    return (not is_causal) and (sk >= tail_base)


def _validate_finite_scores(q_scale: torch.Tensor, k_scale: torch.Tensor, sm_scale: float, D: int) -> None:
    """Scores must stay finite. The running max is a finite sentinel, so a NaN
    scale would pass through the unguarded block softmax."""
    if not math.isfinite(sm_scale):
        raise ValueError(f"sm_scale must be finite, got {sm_scale}")
    q_scale_max = float(q_scale.abs().max())
    k_scale_max = float(k_scale.abs().max())
    if not math.isfinite(q_scale_max):
        raise ValueError("q_scale must be finite (no NaN/Inf)")
    if not math.isfinite(k_scale_max):
        raise ValueError("k_scale must be finite (no NaN/Inf)")
    worst_case_abs_s = D * 64 * abs(sm_scale) * _LOG2E * q_scale_max * k_scale_max
    if not math.isfinite(worst_case_abs_s) or worst_case_abs_s > _FINITE_M0_SCORE_BOUND:
        raise ValueError(
            f"worst-case score magnitude ({worst_case_abs_s:.3e}) must stay inside "
            f"{_FINITE_M0_SCORE_BOUND:.0e}. Reduce q_scale, k_scale, or sm_scale."
        )


def _sageattn2_rdna4_prefill(
    Q_i4: torch.Tensor,
    q_scale: torch.Tensor,
    K_i4: torch.Tensor,
    k_scale: torch.Tensor,
    V_e4m3: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    seq_len_k: int | None = None,
    is_causal: bool = False,
    sm_scale: float | None = None,
    out_dtype: torch.dtype = torch.float16,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Dense-prefill SageAttention2 on gfx120x.

    Q_i4     [B, Hq,  Sq,  D/8]                       int32
    q_scale  [B, Hq,  Sq]                              float32
    K_i4     [B, Hkv, Skp, D/8]                        int32
    k_scale  [B, Hkv, Skp]                              float32
    V_e4m3   [B, Hkv, D/16, Skp/16, 2, 16, 8]           uint8
    v_scale  [B, Hkv, D]                                 float32

    Returns O [B, Hq, Sq, D] out_dtype.
    """
    _require_rdna4_arch()
    if out_dtype not in _ALLOWED_OUT_DTYPES:
        raise ValueError(f"out_dtype must be float16 or bfloat16, got {out_dtype}")
    Q_i4 = _require_cuda_contig("Q_i4", Q_i4, 4, torch.int32)
    q_scale = _require_cuda_contig("q_scale", q_scale, 3, torch.float32)
    K_i4 = _require_cuda_contig("K_i4", K_i4, 4, torch.int32)
    k_scale = _require_cuda_contig("k_scale", k_scale, 3, torch.float32)
    V_e4m3 = _require_cuda_contig("V_e4m3", V_e4m3, 7, torch.uint8)
    v_scale = _require_cuda_contig("v_scale", v_scale, 3, torch.float32)

    B, Hq, Sq, DW = Q_i4.shape
    D = DW * 8
    if D not in _ALLOWED_D:
        raise ValueError(f"D must be 64 or 128, got {D}")
    _, Hkv, Skp, _ = K_i4.shape
    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})")
    if Skp % 16 != 0:
        raise ValueError(f"Skp must be a multiple of 16, got {Skp}")
    if K_i4.shape[0] != B:
        raise ValueError(f"K_i4 batch mismatch: got {K_i4.shape[0]}, expected {B}")
    if K_i4.shape[-1] != DW:
        raise ValueError(f"K_i4 last dim mismatch: got {K_i4.shape[-1]}, expected D/8={DW}")
    if q_scale.shape != (B, Hq, Sq):
        raise ValueError(f"q_scale shape mismatch: got {tuple(q_scale.shape)}, expected {(B, Hq, Sq)}")
    if k_scale.shape != (B, Hkv, Skp):
        raise ValueError(f"k_scale shape mismatch: got {tuple(k_scale.shape)}, expected {(B, Hkv, Skp)}")
    if V_e4m3.shape != (B, Hkv, D // 16, Skp // 16, 2, 16, 8):
        raise ValueError(f"V_e4m3 shape mismatch: got {tuple(V_e4m3.shape)}")
    if v_scale.shape != (B, Hkv, D):
        raise ValueError(f"v_scale shape mismatch: got {tuple(v_scale.shape)}")

    sk = Skp if seq_len_k is None else int(seq_len_k)
    if not (0 < sk <= Skp):
        raise ValueError(f"seq_len_k must be in (0, {Skp}], got {sk}")
    scale = (1.0 / (D**0.5)) if sm_scale is None else float(sm_scale)
    _validate_finite_scores(q_scale, k_scale, scale, D)
    key_peel = _use_key_peel(is_causal, sk, Skp)

    stream = torch.cuda.current_stream() if stream is None else stream
    with torch.cuda.stream(stream):
        out = torch.empty(B * Hq * Sq, D, dtype=out_dtype, device=Q_i4.device)
        _sageattn2_rdna4_launch(
            Q_i4.view(B * Hq * Sq, DW),
            q_scale.view(-1),
            K_i4.view(B * Hkv * Skp, DW),
            k_scale.view(-1),
            V_e4m3.view(B * Hkv, -1).view(torch.int32),
            v_scale.view(B * Hkv, D),
            out,
            B,
            Hq,
            Hkv,
            Sq,
            Skp,
            sk,
            scale,
            D,
            is_causal,
            D >= 128,
            key_peel,
            _ALLOWED_OUT_DTYPES[out_dtype],
            stream=fx.Stream(stream),
        )
        return out.view(B, Hq, Sq, D)
