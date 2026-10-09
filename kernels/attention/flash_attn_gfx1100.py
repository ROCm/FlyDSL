# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Dense multi-head Flash Attention for gfx1100 (RDNA3).

Supports f16/bf16, head_dim 64/128/256, dense prefill, and dense decode
(``seq_q <= 16``). It intentionally does not cover GQA/MQA, varlen, paged KV,
or general cross-attention.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from kernels.attention.flash_attn_gfx1100_autotune import pick_tile, tile_fits
from kernels.attention.flash_attn_utils import daz_denormal_attr
from kernels.common import buffer_ops
from kernels.common.kernels_common import LOG2E, dtype_to_elem_type

WMMA_M = WMMA_N = WMMA_K = 16
WAVE_SIZE = 32
XOR_HALF = 16
NEG_BIG = -1.0e30
# Must be lower than NEG_BIG; otherwise a fully masked first block contributes
# exp2(0) before the row has seen any valid key.
MASK_NEG = -3.0e30

LOAD_VEC = 8
LDS_PAD = 8

_INTERLEAVE = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]

_VP_SEL_LO = 0x05040100
_VP_SEL_HI = 0x07060302
_VP_SEL_LO_REV = 0x01000504
_VP_SEL_HI_REV = 0x03020706
# Keeps the high half of both source dwords, which is the f32 -> bf16
# truncation for two values at once.
_VP_SEL_BF16 = 0x07060302
# Round-to-nearest bias applied to the f32 bit pattern before that truncation.
_BF16_RND = 0x8000
_CONCAT16 = list(range(16))

SUPPORTED_HEAD_DIMS = (64, 128, 256)

_ELEM_CLS = {"f16": fx.Float16, "bf16": fx.BFloat16}
_OUT_CLS = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}


def _strides(layout: str, n_heads: int, seq: int, head_dim: int):
    """(batch, head, seq) element strides for one of the supported layouts."""
    if layout == "bhsd":
        return n_heads * seq * head_dim, seq * head_dim, head_dim
    if layout == "bshd":
        return seq * n_heads * head_dim, head_dim, n_heads * head_dim
    raise ValueError(f"layout must be 'bhsd' or 'bshd', got {layout!r}")


def build_flash_attn_func_module_primary(
    batch,
    num_heads,
    seq_q,
    seq_kv,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    layout="bshd",
    out_dtype=None,
    waves_per_eu=None,
    flat_work_group_size=None,
    daz=False,
    unsafe_fp_math=False,
    fast_fp_math=False,
):
    """Build a dense f16/bf16 flash-attention launcher for gfx1100.

    ``batch``, ``seq_q`` and ``seq_kv`` are compile-time constants: they choose
    the tile and the KV loop. The tile comes from ``pick_tile``. Q/K/V/O are
    BSHD under ``layout='bshd'``. This is multi-head attention; K/V use the same
    head count as Q. Prefill is ``seq_q == seq_kv``; decode is ``seq_q <= 16``.
    """
    if out_dtype is None:
        out_dtype = dtype_str
    in_dtype = dtype_str
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {list(SUPPORTED_HEAD_DIMS)}, got {head_dim}")
    num_waves, q_tiles, block_n, vt_rows, k_from_gmem = pick_tile(head_dim, seq_q, causal, batch * num_heads)
    prefetch = seq_q <= WMMA_M and head_dim == 128 and batch * num_heads <= 128
    hoist_q = head_dim <= 128 or seq_kv >= 8192

    block_m = WMMA_M * num_waves * q_tiles
    threads = num_waves * WAVE_SIZE
    tile = (num_waves, q_tiles, block_n, vt_rows, k_from_gmem)
    if not tile_fits(head_dim, tile):
        raise ValueError(f"tile {tile} does not fit head_dim={head_dim}")

    if in_dtype not in _ELEM_CLS:
        raise ValueError(f"dtype_str must be one of {sorted(_ELEM_CLS)}, got {dtype_str!r}")
    if out_dtype not in _OUT_CLS:
        raise ValueError(f"out_dtype must be one of {sorted(_OUT_CLS)}, got {out_dtype!r}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    score_scale = float(sm_scale) * LOG2E

    is_bf16 = in_dtype == "bf16"
    elem_cls = _ELEM_CLS[in_dtype]
    out_cls = _OUT_CLS[out_dtype]
    use_perm_p = seq_q > WMMA_M and not causal and seq_q < 65536
    use_vector_store = head_dim <= 64 and seq_q > WMMA_M and not causal and seq_q < 65536
    fm = arith.FastMathFlags.fast

    q_stride_b, q_stride_h, q_stride_s = _strides(layout, num_heads, seq_q, head_dim)
    kv_stride_b, kv_stride_h, kv_stride_s = _strides(layout, num_heads, seq_kv, head_dim)
    o_stride_b, o_stride_h, o_stride_s = q_stride_b, q_stride_h, q_stride_s

    n_q_blocks = (seq_q + block_m - 1) // block_m
    q_aligned = seq_q % block_m == 0
    kv_aligned = seq_kv % block_n == 0
    n_kv_sub = block_n // WMMA_K
    n_d_tiles = head_dim // WMMA_K
    _TILE_STATE = 2 + n_d_tiles

    # Bottom-right causal alignment: query i sees up to i + (seq_kv - seq_q).
    causal_delta = seq_kv - seq_q

    k_row = head_dim + LDS_PAD
    vt_row = block_n + LDS_PAD
    k_elems = 0 if k_from_gmem else block_n * k_row
    vt_elems = head_dim * vt_row
    one_buf = k_elems + vt_elems

    chunks_per_row = head_dim // LOAD_VEC
    k_steps = 0 if k_from_gmem else block_n * chunks_per_row // threads
    v_total_chunks = (block_n // vt_rows) * chunks_per_row
    v_partial = v_total_chunks < threads
    use_cooperative_v = v_partial and vt_rows == 8 and threads == 2 * v_total_chunks
    v_steps = 1 if v_partial else v_total_chunks // threads

    def _wmma(a_vec, b_vec, acc):
        if is_bf16:
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a_vec.bitcast(fx.Int16), b_vec.bitcast(fx.Int16), acc).result
        return rocdl.wmma_f32_16x16x16_f16(acc.type, a_vec, b_vec, acc).result

    @fx.struct
    class _SharedStorage:
        lds: fx.Array[elem_cls, one_buf, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def flash_attn_func_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        pid_m = fx.Int32(fx.block_idx.x)
        head = fx.Int32(fx.block_idx.y)
        bat = fx.Int32(fx.block_idx.z)

        wave_id = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        l16 = lane % 16
        lhalf = lane // 16
        is_lo = lhalf == 0

        q_base = pid_m * block_m + wave_id * (WMMA_M * q_tiles) + l16
        q_idx = [q_base + t * WMMA_M for t in range_constexpr(q_tiles)]
        q_idx_safe = q_idx if const_expr(q_aligned) else [fx.min(qi, fx.Int32(seq_q - 1)) for qi in q_idx]

        if const_expr(causal):
            score_limit = [fx.min(fx.Int32(seq_kv), qi + fx.Int32(causal_delta + 1)) for qi in q_idx]
        else:
            score_limit = [fx.Int32(seq_kv) for _ in range_constexpr(q_tiles)]

        smem = fx.SharedAllocator().allocate(_SharedStorage).peek()
        lds_ptr = smem.lds.ptr

        elem_bytes = elem_cls.width // 8
        kv_origin = bat * kv_stride_b + head * kv_stride_h
        q_rsrc = buffer_ops.create_buffer_resource(
            Q,
            base_byte_offset=as_mlir_value(fx.Index(bat * q_stride_b + head * q_stride_h) * fx.Index(elem_bytes)),
        )
        k_rsrc = buffer_ops.create_buffer_resource(
            K, base_byte_offset=as_mlir_value(fx.Index(kv_origin) * fx.Index(elem_bytes))
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, base_byte_offset=as_mlir_value(fx.Index(kv_origin) * fx.Index(elem_bytes))
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            Out,
            base_byte_offset=as_mlir_value(
                fx.Index(bat * o_stride_b + head * o_stride_h) * fx.Index(out_cls.width // 8)
            ),
        )

        def _exp2(x):
            return fmath.exp2(x, fastmath=fm)

        def _recip(x):
            return fx.Float32(rocdl.rcp(T.f32, as_mlir_value(fx.Float32(x))))

        def _gmem_v8(rsrc, elem_off):
            return Vec(
                buffer_ops.buffer_load(rsrc, elem_off, vec_width=LOAD_VEC, dtype=elem_cls),
                (8,),
                elem_cls,
            )

        def _lds_view(elem_off, n):
            p = fx.add_offset(lds_ptr, fx.make_int_tuple(elem_off))
            return fx.make_view(fx.recast_iter(elem_cls, p), fx.make_layout(n, 1))

        def _lds_v8(elem_off):
            return _lds_view(elem_off, LOAD_VEC).load()

        def _lds_store_v8(elem_off, vec):
            _lds_view(elem_off, LOAD_VEC).store(as_mlir_value(vec))

        def _lds_operand(base):
            lo = _lds_v8(base)
            hi = _lds_v8(base + LOAD_VEC)
            return Vec(lo, (8,), elem_cls).shuffle(Vec(hi, (8,), elem_cls), _CONCAT16)

        def _gmem_operand(rsrc, elem_off):
            return _gmem_v8(rsrc, elem_off).shuffle(_gmem_v8(rsrc, elem_off + LOAD_VEC), _CONCAT16)

        k_stage = [fx.make_rmem_tensor(LOAD_VEC, elem_cls) for _ in range_constexpr(k_steps)]
        v_stage_rows = vt_rows // 2 if use_cooperative_v else vt_rows
        v_stage = [
            [fx.make_rmem_tensor(LOAD_VEC, elem_cls) for _ in range_constexpr(v_stage_rows)]
            for _ in range_constexpr(v_steps)
        ]

        kv_last = fx.Int32(seq_kv - 1)

        def _gmem_fetch(kv_base, clamp_rows):
            for st in range_constexpr(k_steps):
                c = fx.Int32(tid) + st * threads
                row = c // chunks_per_row
                col = (c % chunks_per_row) * LOAD_VEC
                src = fx.min(kv_base + row, kv_last) if const_expr(clamp_rows) else kv_base + row
                fx.memref_store_vec(_gmem_v8(k_rsrc, src * kv_stride_s + col), k_stage[st])
            for st in range_constexpr(v_steps):
                c = fx.Int32(tid) + st * threads
                if const_expr(use_cooperative_v):
                    chunk = c // 2
                    half = c % 2
                    row = (chunk // chunks_per_row) * vt_rows + half * v_stage_rows
                    col = (chunk % chunks_per_row) * LOAD_VEC
                else:
                    c = (c < v_total_chunks).select(c, fx.Int32(0)) if const_expr(v_partial) else c
                    row = (c // chunks_per_row) * vt_rows
                    col = (c % chunks_per_row) * LOAD_VEC
                for r in range_constexpr(v_stage_rows):
                    src = fx.min(kv_base + row + r, kv_last) if const_expr(clamp_rows) else kv_base + row + r
                    fx.memref_store_vec(_gmem_v8(v_rsrc, src * kv_stride_s + col), v_stage[st][r])

        def _publish_v_chunk(st, c):
            row = (c // chunks_per_row) * vt_rows
            col = (c % chunks_per_row) * LOAD_VEC
            vecs = [fx.memref_load_vec(v_stage[st][r]) for r in range_constexpr(vt_rows)]
            for i in range_constexpr(LOAD_VEC):
                run = Vec.from_elements([vecs[r][i] for r in range_constexpr(vt_rows)], elem_cls)
                _lds_view(k_elems + (col + i) * vt_row + row, vt_rows).store(as_mlir_value(run))

        def _publish_v_cooperative(st, c):
            chunk = c // 2
            pair_half = c % 2
            row = (chunk // chunks_per_row) * vt_rows
            col = (chunk % chunks_per_row) * LOAD_VEC
            vecs = [fx.memref_load_vec(v_stage[st][r]) for r in range_constexpr(v_stage_rows)]

            def _make_cooperative_run(i):
                own_dw = Vec.from_elements([vecs[r][i] for r in range_constexpr(v_stage_rows)], elem_cls).bitcast(
                    fx.Int32
                )
                peer_dw = Vec.from_elements(
                    [fx.Int32(own_dw[j]).shuffle_xor(1, WAVE_SIZE) for j in range_constexpr(v_stage_rows // 2)],
                    fx.Int32,
                )
                return own_dw.shuffle(peer_dw, list(range(vt_rows // 2))).bitcast(elem_cls)

            for base_i in range_constexpr(0, LOAD_VEC, 2):
                # Both halves of the pair must reach the shuffle_xor inside
                # _make_cooperative_run, so build the runs before branching.
                runs = [_make_cooperative_run(base_i + j) for j in range_constexpr(2)]
                if pair_half == 0:
                    for j in range_constexpr(2):
                        _lds_view(k_elems + (col + base_i + j) * vt_row + row, vt_rows).store(as_mlir_value(runs[j]))

        def _lds_publish_v():
            for st in range_constexpr(v_steps):
                c = fx.Int32(tid) + st * threads
                if const_expr(use_cooperative_v):
                    _publish_v_cooperative(st, c)
                else:
                    c = (c < v_total_chunks).select(c, fx.Int32(0)) if const_expr(v_partial) else c
                    _publish_v_chunk(st, c)

        def _lds_publish_k_and_v0():
            for st in range_constexpr(k_steps):
                c = fx.Int32(tid) + st * threads
                row = c // chunks_per_row
                col = (c % chunks_per_row) * LOAD_VEC
                _lds_store_v8(row * k_row + col, fx.memref_load_vec(k_stage[st]))
            _lds_publish_v()

        def _to_half8(p_vec):
            if const_expr(is_bf16):
                # gfx1100 has no f32 -> bf16 instruction, so the generic lowering
                # spends a shift/and/bfe sequence per element. A bf16 is the high
                # half of the f32, which v_perm_b32 extracts for two values at
                # once once the rounding bias has been folded into the mantissa.
                raw = p_vec.bitcast(fx.Int32)
                biased = [fx.Int32(raw[v]) + fx.Int32(_BF16_RND) for v in range_constexpr(8)]
                return Vec.from_elements(
                    [
                        fx.Int32(rocdl.perm_b32(biased[2 * j + 1], biased[2 * j], fx.Int32(_VP_SEL_BF16)))
                        for j in range_constexpr(4)
                    ],
                    fx.Int32,
                ).bitcast(elem_cls)
            pk_ty = ir.VectorType.get([2], dtype_to_elem_type(in_dtype).ir_type)
            pairs = [
                llvm.call_intrinsic(
                    pk_ty,
                    "llvm.amdgcn.cvt.pkrtz",
                    [
                        as_mlir_value(fx.Float32(p_vec[2 * j])),
                        as_mlir_value(fx.Float32(p_vec[2 * j + 1])),
                    ],
                    [],
                    [],
                )
                for j in range_constexpr(4)
            ]
            lo = Vec(pairs[0], (2,), elem_cls).shuffle(Vec(pairs[1], (2,), elem_cls), [0, 1, 2, 3])
            hi = Vec(pairs[2], (2,), elem_cls).shuffle(Vec(pairs[3], (2,), elem_cls), [0, 1, 2, 3])
            return lo.shuffle(hi, list(range(8)))

        def _build_p_frag(p_vec):
            own_dw = _to_half8(p_vec).bitcast(fx.Int32)
            peer_dw = [fx.Int32(own_dw[j]).shuffle_xor(XOR_HALF, WAVE_SIZE) for j in range_constexpr(4)]
            if const_expr(use_perm_p):
                sel_lo = is_lo.select(fx.Int32(_VP_SEL_LO), fx.Int32(_VP_SEL_LO_REV))
                sel_hi = is_lo.select(fx.Int32(_VP_SEL_HI), fx.Int32(_VP_SEL_HI_REV))
                return Vec.from_elements(
                    [
                        fx.Int32(
                            rocdl.perm_b32(
                                peer_dw[i // 2],
                                own_dw[i // 2],
                                sel_lo if i % 2 == 0 else sel_hi,
                            )
                        )
                        for i in range_constexpr(8)
                    ],
                    fx.Int32,
                ).bitcast(elem_cls)
            even = Vec.from_elements(
                [is_lo.select(own_dw[j], peer_dw[j]) for j in range_constexpr(4)],
                fx.Int32,
            ).bitcast(elem_cls)
            odd = Vec.from_elements(
                [is_lo.select(peer_dw[j], own_dw[j]) for j in range_constexpr(4)],
                fx.Int32,
            ).bitcast(elem_cls)
            return even.shuffle(odd, _INTERLEAVE)

        def _q_frag(qs, kd):
            return _gmem_operand(q_rsrc, qs * q_stride_s + kd * WMMA_K)

        q_frags = (
            [[_q_frag(qs, kd) for kd in range_constexpr(n_d_tiles)] for qs in q_idx_safe]
            if const_expr(hoist_q)
            else None
        )

        zero_acc = fx.full(8, 0.0, fx.Float32)
        mask_neg = fx.Float32(MASK_NEG)

        def _row_max(v):
            m = v.reduce(ReductionOp.MAX)
            peer = m.shuffle_xor(XOR_HALF, WAVE_SIZE)
            return fx.maxnumf(m, peer)

        def _row_sum(v):
            s = v.reduce(ReductionOp.ADD)
            return s + s.shuffle_xor(XOR_HALF, WAVE_SIZE)

        def _consume(kv_base, m_run, l_run, o_run, masked):
            scores = [[] for _ in range_constexpr(q_tiles)]

            def _scale_and_mask(acc, t, sub):
                s = Vec(acc, (8,), fx.Float32) * score_scale
                if const_expr(masked):
                    base = kv_base + (sub * WMMA_K + lhalf)
                    s = Vec.from_elements(
                        [
                            (base + 2 * v < score_limit[t]).select(fx.Float32(s[v]), mask_neg)
                            for v in range_constexpr(8)
                        ],
                        fx.Float32,
                    )
                return s

            if const_expr(hoist_q):
                for sub in range_constexpr(n_kv_sub):
                    s_accs = [zero_acc for _ in range_constexpr(q_tiles)]
                    k_row_idx = (
                        fx.min(kv_base + (sub * WMMA_K + l16), kv_last)
                        if const_expr(masked)
                        else kv_base + (sub * WMMA_K + l16)
                    )
                    k_gbase = k_row_idx * kv_stride_s
                    for kd in range_constexpr(n_d_tiles):
                        if const_expr(k_from_gmem):
                            k_frag = _gmem_operand(k_rsrc, k_gbase + kd * WMMA_K)
                        else:
                            k_frag = _lds_operand((sub * WMMA_K + l16) * k_row + kd * WMMA_K)
                        for t in range_constexpr(q_tiles):
                            s_accs[t] = _wmma(k_frag, q_frags[t][kd], s_accs[t])
                    for t in range_constexpr(q_tiles):
                        scores[t].append(_scale_and_mask(s_accs[t], t, sub))
            else:
                s_accs = [[zero_acc for _ in range_constexpr(n_kv_sub)] for _ in range_constexpr(q_tiles)]
                for kd in range_constexpr(n_d_tiles):
                    q_f = [_q_frag(qs, kd) for qs in q_idx_safe]
                    for sub in range_constexpr(n_kv_sub):
                        if const_expr(k_from_gmem):
                            k_row_idx = (
                                fx.min(kv_base + (sub * WMMA_K + l16), kv_last)
                                if const_expr(masked)
                                else kv_base + (sub * WMMA_K + l16)
                            )
                            k_frag = _gmem_operand(k_rsrc, k_row_idx * kv_stride_s + kd * WMMA_K)
                        else:
                            k_frag = _lds_operand((sub * WMMA_K + l16) * k_row + kd * WMMA_K)
                        for t in range_constexpr(q_tiles):
                            s_accs[t][sub] = _wmma(k_frag, q_f[t], s_accs[t][sub])
                for t in range_constexpr(q_tiles):
                    for sub in range_constexpr(n_kv_sub):
                        scores[t].append(_scale_and_mask(s_accs[t][sub], t, sub))

            m_new, l_new, probs = [], [], []
            for t in range_constexpr(q_tiles):
                m_blk = _row_max(scores[t][0])
                for sub in range_constexpr(1, n_kv_sub):
                    sub_max = _row_max(scores[t][sub])
                    m_blk = fx.maxnumf(m_blk, sub_max)
                mt = fx.maxnumf(m_run[t], m_blk)

                corr = _exp2(m_run[t] - mt)
                pt = [_exp2(s - mt) for s in scores[t]]

                l_blk = _row_sum(pt[0])
                for sub in range_constexpr(1, n_kv_sub):
                    l_blk = l_blk + _row_sum(pt[sub])
                m_new.append(mt)
                l_new.append(fmath.fma(l_run[t], corr, l_blk, fastmath=fm))
                probs.append(pt)
                o_run[t] = [o * corr for o in o_run[t]]

            for sub in range_constexpr(n_kv_sub):
                p_frags = [_build_p_frag(probs[t][sub]) for t in range_constexpr(q_tiles)]
                for dm in range_constexpr(n_d_tiles):
                    vt_frag = _lds_operand(k_elems + (dm * WMMA_K + l16) * vt_row + sub * WMMA_K)
                    for t in range_constexpr(q_tiles):
                        o_run[t][dm] = Vec(
                            _wmma(vt_frag, p_frags[t], as_mlir_value(o_run[t][dm])),
                            (8,),
                            fx.Float32,
                        )
            return m_new, l_new, o_run

        if const_expr(causal):
            q_lo = pid_m * block_m
            q_hi = fx.min(fx.Int32(seq_q - 1), q_lo + (block_m - 1))
            kv_end = fx.min(fx.Int32(seq_kv), q_hi + fx.Int32(causal_delta + 1))
            n_visit = fx.max(fx.Int32(0), (kv_end + (block_n - 1)) // block_n)
            n_full = fx.min(
                fx.Int32(seq_kv // block_n),
                fx.max(fx.Int32(0), (q_lo + fx.Int32(causal_delta + 1)) // block_n),
            )
            n_full = fx.min(n_full, n_visit)
        else:
            n_visit = fx.Int32((seq_kv + block_n - 1) // block_n)
            n_full = fx.Int32(seq_kv // block_n)

        def _phase(start, stop, init_state, masked):
            for ib, state in range(start, stop, 1, init=init_state):
                m_run = [fx.Float32(state[t * _TILE_STATE]) for t in range_constexpr(q_tiles)]
                l_run = [fx.Float32(state[t * _TILE_STATE + 1]) for t in range_constexpr(q_tiles)]
                o_run = [
                    [Vec(state[t * _TILE_STATE + 2 + dm], (8,), fx.Float32) for dm in range_constexpr(n_d_tiles)]
                    for t in range_constexpr(q_tiles)
                ]
                kv_base = fx.Int32(ib) * block_n

                if const_expr(prefetch):
                    _gmem_fetch(kv_base + block_n, True)
                    m_run, l_run, o_run = _consume(kv_base, m_run, l_run, o_run, masked)
                    gpu.barrier()
                    _lds_publish_k_and_v0()
                    gpu.barrier()
                else:
                    gpu.barrier()
                    _gmem_fetch(kv_base, masked or not kv_aligned)
                    _lds_publish_k_and_v0()
                    gpu.barrier()
                    m_run, l_run, o_run = _consume(kv_base, m_run, l_run, o_run, masked)

                packed = []
                for t in range_constexpr(q_tiles):
                    packed += [m_run[t], l_run[t]] + [as_mlir_value(o) for o in o_run[t]]
                results = yield packed
            return results

        init_state = []
        for _t in range_constexpr(q_tiles):
            init_state += [fx.Float32(NEG_BIG), fx.Float32(0.0)] + [zero_acc for _ in range_constexpr(n_d_tiles)]
        if const_expr(prefetch):
            _gmem_fetch(fx.Int32(0), not kv_aligned)
            _lds_publish_k_and_v0()
            gpu.barrier()

        state = _phase(fx.Int32(0), n_full, init_state, False)
        state = _phase(n_full, n_visit, list(state), True)

        def _store_output_tile(qi, dm, o):
            if const_expr(use_vector_store):
                peer = [fx.Float32(o[v]).shuffle_xor(XOR_HALF, WAVE_SIZE) for v in range_constexpr(8)]
                if is_lo:
                    for j in range_constexpr(4):
                        vals = Vec.from_elements(
                            [o[2 * j], peer[2 * j], o[2 * j + 1], peer[2 * j + 1]],
                            fx.Float32,
                        )
                        packed = vals if const_expr(out_cls is fx.Float32) else vals.to(out_cls)
                        buffer_ops.buffer_store(
                            as_mlir_value(packed),
                            o_rsrc,
                            as_mlir_value(fx.Int32(qi * o_stride_s + dm * WMMA_K + 4 * j)),
                        )
            else:
                for v in range_constexpr(8):
                    d = dm * WMMA_K + 2 * v + lhalf
                    val = fx.Float32(o[v]) if const_expr(out_cls is fx.Float32) else fx.Float32(o[v]).to(out_cls)
                    buffer_ops.buffer_store(
                        as_mlir_value(val),
                        o_rsrc,
                        as_mlir_value(fx.Int32(qi * o_stride_s + d)),
                    )

        # Fully masked rows must write zeros, not NaN from 1/0.
        for t in range_constexpr(q_tiles):
            l_sum = fx.Float32(state[t * _TILE_STATE + 1])
            inv_l = (l_sum > fx.Float32(0.0)).select(_recip(l_sum), fx.Float32(0.0))

            if const_expr(q_aligned):
                for dm in range_constexpr(n_d_tiles):
                    o = Vec(state[t * _TILE_STATE + 2 + dm], (8,), fx.Float32) * inv_l
                    _store_output_tile(q_idx[t], dm, o)
            else:
                if q_idx[t] < seq_q:
                    for dm in range_constexpr(n_d_tiles):
                        o = Vec(state[t * _TILE_STATE + 2 + dm], (8,), fx.Float32) * inv_l
                        _store_output_tile(q_idx[t], dm, o)

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Out: fx.Tensor,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        q_n = batch * num_heads * seq_q * head_dim
        kv_n = batch * num_heads * seq_kv * head_dim
        Qf = fx.make_view(fx.get_iter(Q), fx.make_layout(q_n, 1))
        Kf = fx.make_view(fx.get_iter(K), fx.make_layout(kv_n, 1))
        Vf = fx.make_view(fx.get_iter(V), fx.make_layout(kv_n, 1))
        Of = fx.make_view(fx.get_iter(Out), fx.make_layout(q_n, 1))

        wpe = None
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                wpe = _wpe
        fwgs = None
        if const_expr(flat_work_group_size is not None):
            _fwgs = int(flat_work_group_size)
            if const_expr(_fwgs >= 1):
                fwgs = f"{_fwgs},{_fwgs}"
        passthrough_entries = (
            [
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_func_kernel(
            Qf,
            Kf,
            Vf,
            Of,
            value_attrs={
                "rocdl.waves_per_eu": wpe,
                "rocdl.flat_work_group_size": fwgs,
                "passthrough": passthrough_entries,
                "llvm.denormal_fpenv": daz_denormal_attr() if const_expr(daz) else None,
            },
        ).launch(grid=(n_q_blocks, num_heads, batch), block=(threads, 1, 1), stream=stream)

    launch_flash_attn_func.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }
    return launch_flash_attn_func


build_flash_attn_func_module = build_flash_attn_func_module_primary
