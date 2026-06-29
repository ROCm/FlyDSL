# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

This is a *correctness-first* reimplementation of the decode math that the
production ``pa_decode_ps_kernel`` (``pa_decode_fp8.py``) computes, written with
FlyDSL's high-level tile/layout API (``make_buffer_tensor`` + ``zipped_divide`` +
``make_tiled_mma`` + ``fx.gemm`` + tiled copies) instead of raw buffer
intrinsics and hand-scheduled MFMA.  It deliberately trades performance for
clarity and is scoped to a single configuration:

* per-tensor fp8 K/V quantization (scalar ``key_scale`` / ``value_scale``),
* ``query_length == 1`` (pure decode, one query token per sequence),
* no kv-varlen, no sliding window.

Like ``pa_decode_ps_kernel`` it uses a **cross-CTA partition split**: the context
is split across ``grid.z`` partitions (one CTA each) that write partial
(max, sum, numerator) results, and a small reduce kernel flash-combines them into
the output.  This spreads low-batch / long-context work across many CUs instead of
serializing it on one.  The host picks the partition count from CU count vs
batch×kv_heads (only split when the GPU isn't already filled).

fp8: K/V are stored as fp8 (e4m3 **FNUZ** — the format gfx942 fp8 MFMA consumes)
and fed straight into fp8 ``mfma_f32_16x16x32_fp8_fp8`` MMAs.  Q (bf16/f16 input)
is quantized to fp8 in-kernel with a per-row scale, and the softmax probabilities
P are quantized to fp8 for the P·V matmul.  All scales fold out of the matmuls:
``q_scale`` (per row) and ``key_scale`` into the QK score scaling; ``value_scale``
and the constant P dequant (1/FP8_MAX) into the epilogue.  The softmax max/sum are
kept in f32, so the denominator stays accurate; only the matmul operands are fp8.

Layouts (simple / logical, NOT the production preshuffle layout):

* ``query``        ``[num_seqs, num_q_heads, head_dim]``                 f16/bf16
* ``key_cache``    ``[num_blocks, num_kv_heads, block_size, head_dim]``   fp8 e4m3fnuz
* ``value_cache``  ``[num_blocks, num_kv_heads, head_dim, block_size]``   fp8 e4m3fnuz
                   (V stored transposed so it is already the ``[N, K]`` operand
                    the MMA wants for ``P @ V``)
* ``block_tables`` ``[num_seqs, max_blocks_per_seq]``                    int32
* ``context_lengths`` ``[num_seqs]``                                     int32
* ``output``       ``[num_seqs, num_q_heads, head_dim]``                 f16

Algorithm: one CTA (4 waves / 256 threads, like ``pa_decode_ps_kernel``) per
``(seq, kv_head)`` runs a flash-style online softmax over the context in
256-token compute blocks (= ``KV_COMPUTE_BLOCK``), gathering each block's page
through ``block_tables``.  Following the production warp layout, the 4 waves
**split the tokens** for Q·Kᵀ (each wave owns 64 of the 256 tokens) and
**split the head-dim output** for P·V (each wave owns HEAD/4 dims); an LDS
round-trip on the probabilities transposes that warp ownership between the two
MMAs.  Both MMAs use a 16×16×32 fp8 atom via ``fx.gemm`` with a 4-warp tiled
layout ``(1,4,1)``.

The softmax is **distributed across the 4 waves**: each wave reduces over its own
64-token slice (local max, then exp + local sum), and the per-wave partials are
merged through small LDS scratch (``sLmax``/``sLsum``[16,4]) into the shared
running ``(m, l)`` — so all 4 waves stay busy instead of one wave doing the whole
row.  Running ``(m, l, acc)`` state is shared across the 4 waves in LDS, so it
does not multiply with the wave count.

Per-tensor scales are folded *out* of the inner loop: the scalar
``softmax_scale * key_scale`` is applied to the whole score tile and
``value_scale`` is applied once to the final output.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.vector import ReductionOp


def _global_ptr(tensor):
    """Aligned addrspace(1) pointer for raw scalar loads from a global tensor."""
    from flydsl._mlir.dialects import fly as _fly

    raw = tensor.ir_value() if hasattr(tensor, "ir_value") and not isinstance(tensor, ir.Value) else tensor
    return _fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)


def _load_i32(global_ptr, elem_offset_i32):
    """Load one *signless* i32 element. Plain tensor indexing of an int32 memref
    yields si32, which neither composes with signless arith ops nor lowers
    through llvm.load — so int metadata is read with this raw load instead."""
    byte_off = fx.Int64(elem_offset_i32) * fx.Int64(4)
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=byte_off, elem_type=T.i8)
    return llvm.LoadOp(T.i32, ptr, alignment=4).result


MFMA_MNK = 16  # M = N = 16 for the MMA atom
MFMA_K = 32  # fp8 MFMA contracts K = 32 per instruction (mfma_f32_16x16x32_fp8_fp8)
WAVE = 64
LOG2E = 1.4426950408889634
FP8 = fx.Float8E4M3FNUZ  # gfx942 fp8 MFMA uses the FNUZ format (not e4m3fn)
FP8_MAX = 240.0  # max representable magnitude of e4m3fnuz


@functools.lru_cache(maxsize=None)
def compile_pa_decode_tile(
    *,
    head_dim: int,
    query_group_size: int,
    num_partitions: int = 1,
    softmax_scale: float | None = None,
):
    """Build the tile-programming PA-decode kernel + launch wrapper.

    Returns a dict with ``launch`` and ``kernel`` entries, mirroring the
    ``compile_*`` factories in ``pa_decode_fp8.py``.
    """
    assert head_dim % MFMA_MNK == 0, f"head_dim {head_dim} must be a multiple of {MFMA_MNK}"
    HEAD = head_dim
    GS = query_group_size
    M = MFMA_MNK  # query rows handled per CTA (padded to 16)
    NWARP = 4  # 4 waves / CTA (matches pa_decode_ps_kernel)
    TOK_PER_WARP = 64  # tokens each warp owns per compute block (matches production KV_COMPUTE_BLOCK)
    TILE_TOK = NWARP * TOK_PER_WARP  # 256 tokens / compute block
    assert HEAD % (NWARP * MFMA_MNK) == 0, "head_dim must split across the 4 warps for PV"

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD**0.5)
    _softmax_scale = float(softmax_scale)
    NP = int(num_partitions)  # context partitions (grid.z); compile-time constant

    BLOCK_THREADS = NWARP * WAVE  # 256

    # ── LDS layout (shared across the 4 warps; running state is NOT per-warp) ──
    # sQ  : fp8[16,HEAD]      staged + quantized query tile
    # sS  : f32[16,TILE_TOK]  QK score tile      (warp w writes its token slice)
    # sP  : fp8[16,TILE_TOK]  quantized softmax probs (re-read by all warps for P·V)
    # sOp : f32[16,HEAD]      P·V partial out    (warp w writes its HEAD/4 slice)
    # sO  : f32[16,HEAD]      running output accumulator
    # sM/sL/sCorr/sQscale : f32[16]   sLmax/sLsum : f32[16,NWARP]
    f32 = 4
    sQ_off = 0
    sQ_bytes = M * HEAD * 1  # fp8
    sS_off = sQ_off + sQ_bytes
    sS_bytes = M * TILE_TOK * f32
    sP_off = sS_off + sS_bytes
    sP_bytes = M * TILE_TOK * 1  # fp8
    # sOp kept separate from sScore (not aliased): that lets us drop the barrier
    # after the accumulate (the next iter's QK/PV-stage barriers order sOp reuse),
    # trading 8KB LDS for one fewer barrier — barriers, not LDS, bound this kernel.
    sOp_off = sP_off + sP_bytes
    sOp_bytes = M * HEAD * f32
    sO_off = sOp_off + sOp_bytes
    sO_bytes = M * HEAD * f32
    sM_off = sO_off + sO_bytes
    sL_off = sM_off + M * f32
    sCorr_off = sL_off + M * f32
    sQscale_off = sCorr_off + M * f32  # per-row query dequant scale
    # cross-warp reduction scratch: per (query row, warp) local max / local sum
    sLmax_off = sQscale_off + M * f32
    sLsum_off = sLmax_off + M * NWARP * f32
    total_bytes = sLsum_off + M * NWARP * f32

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_tile_kernel(
        output_ptr: fx.Tensor,  # [num_seqs, num_q_heads, HEAD]  (written directly when NP==1)
        # per-partition partial outputs (combined by the reduce kernel when NP>1):
        pmax_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS]   row max
        psum_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS]   row sum
        pout_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS, HEAD] numerator
        query_ptr: fx.Tensor,  # [num_seqs, num_q_heads, HEAD]
        key_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, block_size, HEAD]
        value_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, HEAD, block_size]
        block_tables_ptr: fx.Tensor,  # [num_seqs, max_blocks_per_seq]
        context_lengths_ptr: fx.Tensor,  # [num_seqs]
        key_scale: fx.Float32,
        value_scale: fx.Float32,
        block_size: Int32,
        max_blocks_per_seq: Int32,
        num_q_heads: Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = tid // arith.constant(WAVE, type=T.i32)  # 0..NWARP-1
        lane = tid - warp * arith.constant(WAVE, type=T.i32)  # 0..63
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        part = fx.Int32(gpu.block_id("z"))  # context partition handled by this CTA
        n_kv = num_q_heads // arith.constant(GS, type=T.i32)  # num_kv_heads

        context_len = fx.Int32(_load_i32(_global_ptr(context_lengths_ptr), seq))
        bt_gp = _global_ptr(block_tables_ptr)

        # ── per-CTA scalar constants ──
        # softmax_scale and key_scale fold into the score tile; log2e folds into
        # the exp2 used for softmax.
        scale_qk = fx.Float32(_softmax_scale * LOG2E) * fx.Float32(key_scale)
        v_scale_f = fx.Float32(value_scale)
        NEG_INF = fx.Float32(float("-inf"))
        ZERO_F = fx.Float32(0.0)
        # float iota over a warp's token slice (float compares avoid int-signedness issues)
        iota_w = Vec.from_elements([arith.constant(float(c), type=T.f32) for c in range_constexpr(TOK_PER_WARP)])

        # ── LDS views ──
        # One i8 blob carved into typed views via byte-offset pointers.  The
        # same view tensor is used both as a tiled-copy partition target and for
        # direct .load()/.store() of whole rows by the row-owner lanes.
        lds_base = fx.SharedAllocator().allocate(total_bytes).peek().ptr  # i8 base pointer

        def _view(byte_off, elem_ty, layout, esz):
            p = fx.add_offset(lds_base, fx.make_int_tuple(byte_off))
            ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Shared, esz)
            return fx.Tensor(fx.make_view(fx.recast_iter(ptr_ty, p), layout))

        def _row(byte_off, m_idx, width, elem_ty, esz):
            off = arith.constant(byte_off, type=T.i32) + m_idx * arith.constant(width * esz, type=T.i32)
            return _view(off, elem_ty, fx.make_layout(width, 1), esz)

        def _ld1(byte_off, m_idx):
            return _row(byte_off, m_idx, 1, fx.Float32, 4).load()[0]

        def _st1(byte_off, m_idx, val):
            _row(byte_off, m_idx, 1, fx.Float32, 4).store(Vec.from_elements([val], dtype=fx.Float32))

        def _ld_row(byte_off, m_idx, width):
            return _row(byte_off, m_idx, width, fx.Float32, 4).load()

        def _st_row(byte_off, m_idx, vec_val):
            _row(byte_off, m_idx, vec_val.shape[0], fx.Float32, 4).store(vec_val)

        # f32[16, NWARP] cross-warp scratch: scalar write at (row, warp), vec read of a row
        def _st_lw(base_off, row, w, val):
            off = arith.constant(base_off, type=T.i32) + (row * arith.constant(NWARP, type=T.i32) + w) * arith.constant(
                4, type=T.i32
            )
            _view(off, fx.Float32, fx.make_layout(1, 1), 4).store(Vec.from_elements([val], dtype=fx.Float32))

        def _ld_lw_row(base_off, row):
            off = arith.constant(base_off, type=T.i32) + row * arith.constant(NWARP * 4, type=T.i32)
            return _view(off, fx.Float32, fx.make_layout(NWARP, 1), 4).load()

        def _f32_to_fp8_words(vf32):
            # f32 -> fp8 must use the hw cvt (arith.truncf to fp8 is not lowerable);
            # pack 4 f32 -> 1 i32 (4 fp8) via two cvt_pk_fp8_f32 calls. Returns the
            # i32 words so the result can be stored to LDS as plain i32.
            n = vf32.shape[0]
            words = []
            for i in range_constexpr(n // 4):
                b = i * 4
                lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b], vf32[b + 1], arith.constant(0, type=T.i32), False)
                words.append(fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b + 2], vf32[b + 3], lo, True))
            return Vec.from_elements(words, dtype=fx.Int32)

        def _st_words(byte_off, words):
            _view(byte_off, fx.Int32, fx.make_layout(words.shape[0], 1), 4).store(words)

        # whole-tile views (for tiled copies)
        sQ_v = _view(arith.constant(sQ_off, type=T.i32), FP8, fx.make_layout((M, HEAD), (HEAD, 1)), 1)
        sS_v = _view(arith.constant(sS_off, type=T.i32), fx.Float32, fx.make_layout((M, TILE_TOK), (TILE_TOK, 1)), 4)
        sP_v = _view(arith.constant(sP_off, type=T.i32), FP8, fx.make_layout((M, TILE_TOK), (TILE_TOK, 1)), 1)

        # ── MMA atoms: two 4-warp fp8 tiled MMAs (layout (1,4,1) = 4 warps along N) ──
        # QK splits the TILE_TOK tokens across the 4 warps (N = token);
        # PV splits the HEAD output dims across the 4 warps (N = head_dim).
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_MNK, MFMA_MNK, MFMA_K, FP8))
        tiled_mma_qk = fx.make_tiled_mma(mma_atom, fx.make_layout((1, NWARP, 1), (0, 1, 0)))
        tiled_mma_pv = fx.make_tiled_mma(mma_atom, fx.make_layout((1, NWARP, 1), (0, 1, 0)))
        thr_mma_qk = tiled_mma_qk.thr_slice(tid)

        # ── Stage 0: stage this (seq, kv_head)'s GS query rows into LDS sQ.
        # Row tid (< GS) holds q-head kv_h*GS + tid; rows [GS, 16) are zero
        # padding (the MMA atom is 16 wide; padded rows are never stored).  LDS
        # staging handles any group size cleanly (a 16-row global window would
        # not be tile-aligned when GS < 16).
        if tid < arith.constant(M, type=T.i32):
            if tid < arith.constant(GS, type=T.i32):
                qh0 = kv_h * arith.constant(GS, type=T.i32) + tid
                row_elem = (seq * num_q_heads + qh0) * arith.constant(HEAD, type=T.i32)
                q_iter = fx.add_offset(fx.get_iter(query_ptr), fx.make_int_tuple(row_elem))
                q_row = fx.Tensor(fx.make_view(q_iter, fx.make_layout(HEAD, 1))).load().to(fx.Float32)
                # per-row symmetric fp8 quantization: q_scale = absmax / FP8_MAX
                absmax = q_row.maximumf(Vec.filled(HEAD, 0.0, fx.Float32) - q_row).reduce(ReductionOp.MAX)
                q_scale = absmax * fx.Float32(1.0 / FP8_MAX)
                inv = fx.Float32(1.0) / q_scale.maximumf(fx.Float32(1e-20))
                q_scaled = q_row * Vec.from_elements([inv], dtype=fx.Float32).broadcast_to(HEAD)
                # store fp8 bytes as i32 words (the fp8 tiled-copy reads them back)
                _st_words(
                    arith.constant(sQ_off, type=T.i32) + tid * arith.constant(HEAD, type=T.i32),
                    _f32_to_fp8_words(q_scaled),
                )
                _st1(sQscale_off, tid, q_scale)
            else:
                _st_words(
                    arith.constant(sQ_off, type=T.i32) + tid * arith.constant(HEAD, type=T.i32),
                    Vec.filled(HEAD // 4, 0, fx.Int32),
                )
                _st1(sQscale_off, tid, ZERO_F)
        gpu.barrier()

        copy_q = fx.make_copy_atom(fx.UniversalCopy64b(), FP8)
        thr_copy_q = fx.make_tiled_copy_A(copy_q, tiled_mma_qk).get_slice(tid)
        tmpl_Q = fx.make_rmem_tensor(fx.make_layout((M, HEAD), (HEAD, 1)), FP8)
        frag_Q = thr_mma_qk.make_fragment_A(tmpl_Q)
        fx.copy(copy_q, thr_copy_q.partition_S(sQ_v), thr_copy_q.retile(frag_Q), pred=None)

        # ── init running softmax state (row-owner lanes 0..15) ──
        if tid < arith.constant(M, type=T.i32):
            _st1(sM_off, tid, NEG_INF)
            _st1(sL_off, tid, ZERO_F)
            _st_row(sO_off, tid, Vec.filled(HEAD, 0.0, fx.Float32))
        gpu.barrier()

        # This CTA only walks its partition's slice of the TILE_TOK-blocks, so the
        # context is parallelized across grid.z CTAs (more CUs for low batch).
        num_tiles = (context_len + arith.constant(TILE_TOK - 1, type=T.i32)) // arith.constant(TILE_TOK, type=T.i32)
        tiles_per_part = (num_tiles + arith.constant(NP - 1, type=T.i32)) // arith.constant(NP, type=T.i32)
        part_start = part * tiles_per_part
        part_end_raw = part_start + tiles_per_part
        part_end = arith.select(part_end_raw < num_tiles, part_end_raw, num_tiles)
        loop_start = fx.Index(arith.unwrap(part_start))
        loop_end = fx.Index(arith.unwrap(part_end))

        key_buf = fx.rocdl.make_buffer_tensor(key_cache_ptr)
        val_buf = fx.rocdl.make_buffer_tensor(value_cache_ptr)

        copy_kv = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), FP8)  # fp8 K/V global -> reg
        copy_p = fx.make_copy_atom(fx.UniversalCopy64b(), FP8)  # fp8 P LDS -> reg
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tcopy_k = fx.make_tiled_copy_B(copy_kv, tiled_mma_qk)  # K: token-split across warps
        tcopy_p = fx.make_tiled_copy_A(copy_p, tiled_mma_pv)  # P: replicated across warps
        tcopy_v = fx.make_tiled_copy_B(copy_kv, tiled_mma_pv)  # V: head-dim-split across warps
        tcopy_s = fx.make_tiled_copy_C(copy_c, tiled_mma_qk)  # scores -> sS
        tcopy_o = fx.make_tiled_copy_C(copy_c, tiled_mma_pv)  # PV out -> sOp

        # Shape templates (default addrspace) for the MMA fragments; only their
        # layout is read by make_fragment_*, no real storage is consumed.
        tmpl_S = fx.make_rmem_tensor(fx.make_layout((M, TILE_TOK), (TILE_TOK, 1)), fx.Float32)
        tmpl_P = fx.make_rmem_tensor(fx.make_layout((M, TILE_TOK), (TILE_TOK, 1)), FP8)
        # P·V is loop-tiled over head-dim (like the production VHELOOP): each step
        # computes O[:, vh*VHE_SIZE : +VHE_SIZE], shrinking the live V operand and
        # PV accumulator instead of materializing the full [16, HEAD] at once.
        VHE_CHUNKS = 2
        VHE_SIZE = HEAD // VHE_CHUNKS
        tmpl_Op = fx.make_rmem_tensor(fx.make_layout((M, VHE_SIZE), (VHE_SIZE, 1)), fx.Float32)

        c_TILE_TOK = arith.constant(TILE_TOK, type=T.i32)

        # (phys page, sub-tile index, token offset) for tile index `tt_i32`.
        def _tile_coords(tt_i32):
            tok0 = tt_i32 * c_TILE_TOK
            page = tok0 // block_size
            within = tok0 - page * block_size
            phys = fx.Int32(_load_i32(bt_gp, seq * max_blocks_per_seq + page))
            return phys, within // c_TILE_TOK, tok0

        for tt in range(loop_start, loop_end, arith.index(1)):
            tt_i32 = fx.Int32(arith.index_cast(T.i32, tt))
            phys, within_tile, tok0 = _tile_coords(tt_i32)

            # Thread slices must be (re)created inside the loop: the ThrCopy/
            # ThrMma python subclass is stripped back to its TiledCopy/TiledMma
            # base when a value is captured across the scf.for region boundary.
            thr_mma_qk_l = tiled_mma_qk.get_slice(tid)
            thr_mma_pv_l = tiled_mma_pv.get_slice(tid)
            thr_copy_k = tcopy_k.get_slice(tid)
            thr_copy_p = tcopy_p.get_slice(tid)
            thr_copy_v = tcopy_v.get_slice(tid)
            thr_copy_s = tcopy_s.get_slice(tid)
            thr_copy_o = tcopy_o.get_slice(tid)

            # ---- K tile [TILE_TOK, HEAD] from page `phys` (4 warps split tokens) ----
            k_page = fx.slice(key_buf, (phys, kv_h, None, None))  # [block_size, HEAD]
            bK = fx.slice(fx.zipped_divide(k_page, (TILE_TOK, HEAD)), (None, within_tile))
            frag_K = thr_mma_qk_l.make_fragment_B(bK)
            fx.copy(copy_kv, thr_copy_k.partition_S(bK), thr_copy_k.retile(frag_K), pred=None)

            # ---- prefetch V (overlap its global DMA with QK + softmax compute) ----
            # V depends only on (phys, kv_h), known here, so issue its loads now and
            # let the latency hide behind the QK MMA + softmax that follow (the
            # production kernel software-pipelines K/V loads the same way).
            v_page = fx.slice(val_buf, (phys, kv_h, None, None))  # [HEAD, block_size]
            v_hsplit = fx.zipped_divide(v_page, (VHE_SIZE, TILE_TOK))  # [(VHE,TILE), (HEAD/VHE, blk/TILE)]
            frag_Vs = []
            for vh in range_constexpr(VHE_CHUNKS):
                bV = fx.slice(v_hsplit, (None, (vh, within_tile)))  # [VHE_SIZE, TILE_TOK]
                fV = thr_mma_pv_l.make_fragment_B(bV)
                fx.copy(copy_kv, thr_copy_v.partition_S(bV), thr_copy_v.retile(fV), pred=None)
                frag_Vs.append(fV)

            # ---- QK: S = Q . Kᵀ  -> scores[16, TILE_TOK] (per-warp [16,64]) ----
            frag_S = thr_mma_qk_l.make_fragment_C(tmpl_S)
            frag_S.fill(0.0)
            fx.gemm(tiled_mma_qk, frag_S, frag_Q, frag_K, frag_S)

            # softmax bookkeeping shared by the max reduction and phase 2.
            # lane l (<16) of warp w owns query row l, token cols [w*TPW : +TPW].
            col0 = warp * arith.constant(TOK_PER_WARP, type=T.i32)
            n_valid_loc = (context_len - tok0 - col0).to(fx.Float32)  # valid cols in this warp's slice
            mask = iota_w < Vec.from_elements([n_valid_loc], dtype=fx.Float32).broadcast_to(TOK_PER_WARP)
            neg_inf_w = Vec.filled(TOK_PER_WARP, float("-inf"), fx.Float32)
            # byte offset of this warp's score / prob slice for query row = lane
            s_slice_off = arith.constant(sS_off, type=T.i32) + (
                lane * arith.constant(TILE_TOK, type=T.i32) + col0
            ) * arith.constant(4, type=T.i32)
            p_slice_off = arith.constant(sP_off, type=T.i32) + (
                lane * arith.constant(TILE_TOK, type=T.i32) + col0
            ) * arith.constant(1, type=T.i32)

            # ---- register-resident per-warp max (DPP), no score read-back ----
            # Probed 4-warp C-fragment layout: vec index v of lane L (warp w) holds
            #   row m = (L%64//16)*4 + v%4,  GLOBAL token n = (v//4)*64 + w*16 + L%16.
            # The 4 warps own interleaved 16-token strips (NOT contiguous blocks),
            # so the mask uses the true global token index vs the tile's valid count
            # (context_len - tok0).  Reduce over a warp's tokens without staging
            # scores back through LDS: register-max across the 4 N-atoms (v//4),
            # then shuffle_xor(1,2,4,8) across the 16-lane group (L%16).  The 4
            # warps' partials are unioned by the cross-warp max in phase 2.  The
            # positive per-row scale (scale_qk * q_scale[row]) is applied after.
            c16 = arith.constant(16, type=T.i32)
            sv = frag_S.load()
            lane16 = lane - (lane // c16) * c16
            tok_base_f = fx.Int32(warp * c16 + lane16).to(fx.Float32)
            n_valid_tile = (context_len - tok0).to(fx.Float32)
            const_n = Vec.from_elements([arith.constant(float((v // 4) * 64), type=T.f32) for v in range_constexpr(16)])
            local_n = const_n + Vec.from_elements([tok_base_f], dtype=fx.Float32).broadcast_to(16)
            nvalid_b = Vec.from_elements([n_valid_tile], dtype=fx.Float32).broadcast_to(16)
            sv_masked = (local_n < nvalid_b).select(sv, Vec.filled(16, float("-inf"), fx.Float32))
            c_w = arith.constant(WAVE, type=T.i32)
            pm = []
            for r in range_constexpr(4):
                pmr = sv_masked[r]
                for a in range_constexpr(1, 4):
                    pmr = pmr.maximumf(sv_masked[r + a * 4])
                for sh in (1, 2, 4, 8):
                    pmr = pmr.maximumf(pmr.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
                pm.append(pmr)
            # one lane per 16-group writes its group's 4 rows' scaled max
            m_base = (lane // c16) * arith.constant(4, type=T.i32)
            if lane16 == arith.constant(0, type=T.i32):
                for r in range_constexpr(4):
                    rr = m_base + arith.constant(r, type=T.i32)
                    _st_lw(sLmax_off, rr, warp, pm[r] * scale_qk * _ld1(sQscale_off, rr))

            # ---- stage scores to sS for the fp8 P pack (phase 2 reads them) ----
            fx.copy(copy_c, thr_copy_s.retile(frag_S), thr_copy_s.partition_D(sS_v), pred=None)
            gpu.barrier()

            # phase 2: global max -> exp -> per-warp local sum, write fp8 P slice.
            # P (in [0,1]) is quantized to fp8 by *FP8_MAX (dequant 1/FP8_MAX folds
            # into the epilogue).  The local sum uses the f32 probs, so the softmax
            # denominator stays accurate.
            if lane < arith.constant(M, type=T.i32):
                row_scale = scale_qk * _ld1(sQscale_off, lane)
                m_old = _ld1(sM_off, lane)
                tile_max = _ld_lw_row(sLmax_off, lane).reduce(ReductionOp.MAX)
                m_new = m_old.maximumf(tile_max)
                s_w = _view(s_slice_off, fx.Float32, fx.make_layout(TOK_PER_WARP, 1), 4).load() * row_scale
                s_w = mask.select(s_w, neg_inf_w)
                p_w = (s_w - Vec.from_elements([m_new], dtype=fx.Float32).broadcast_to(TOK_PER_WARP)).exp2()
                _st_words(p_slice_off, _f32_to_fp8_words(p_w * Vec.filled(TOK_PER_WARP, FP8_MAX, fx.Float32)))
                _st_lw(sLsum_off, lane, warp, p_w.reduce(ReductionOp.ADD))
                if warp == arith.constant(0, type=T.i32):
                    _st1(sM_off, lane, m_new)
                    _st1(sCorr_off, lane, Vec.from_elements([m_old - m_new], dtype=fx.Float32).exp2()[0])
            gpu.barrier()

            # phase 3: merge per-warp sums into the running denominator
            if tid < arith.constant(M, type=T.i32):
                gsum = _ld_lw_row(sLsum_off, tid).reduce(ReductionOp.ADD)
                _st1(sL_off, tid, _ld1(sL_off, tid) * _ld1(sCorr_off, tid) + gsum)

            # ---- load P back as the A operand for P·V (replicated across warps) ----
            frag_P = thr_mma_pv_l.make_fragment_A(tmpl_P)
            fx.copy(copy_p, thr_copy_p.partition_S(sP_v), thr_copy_p.retile(frag_P), pred=None)

            # ---- PV, loop-tiled over head-dim chunks (V was prefetched above) ----
            for vh in range_constexpr(VHE_CHUNKS):
                frag_Op = thr_mma_pv_l.make_fragment_C(tmpl_Op)  # [16, VHE_SIZE]
                frag_Op.fill(0.0)
                fx.gemm(tiled_mma_pv, frag_Op, frag_P, frag_Vs[vh], frag_Op)
                # write this head-dim chunk into sOp[:, vh*VHE_SIZE : +VHE_SIZE]
                sOp_chunk = _view(
                    arith.constant(sOp_off + vh * VHE_SIZE * 4, type=T.i32),
                    fx.Float32,
                    fx.make_layout((M, VHE_SIZE), (HEAD, 1)),
                    4,
                )
                fx.copy(copy_c, thr_copy_o.retile(frag_Op), thr_copy_o.partition_D(sOp_chunk), pred=None)
            gpu.barrier()

            # ---- accumulate into running output (row-owner): O = O*corr + Op ----
            # No barrier after this: sOp/sScore are not aliased, and the next
            # iteration's QK-stage + PV-stage barriers order any reuse of sO/sOp.
            if tid < arith.constant(M, type=T.i32):
                corr = _ld1(sCorr_off, tid)
                o_v = _ld_row(sO_off, tid, HEAD)
                op_v = _ld_row(sOp_off, tid, HEAD)
                _st_row(sO_off, tid, o_v * Vec.from_elements([corr], dtype=fx.Float32).broadcast_to(HEAD) + op_v)

        # ── epilogue (row-owner, real query rows) ──
        if tid < arith.constant(GS, type=T.i32):
            o_v = _ld_row(sO_off, tid, HEAD)
            if const_expr(NP == 1):
                # single partition: normalize and write the output directly (no
                # partials / reduce round-trip).  Fold value_scale and 1/FP8_MAX.
                qh = kv_h * arith.constant(GS, type=T.i32) + tid
                inv_l = (fx.Float32(1.0) / _ld1(sL_off, tid)) * v_scale_f * fx.Float32(1.0 / FP8_MAX)
                o_out = (o_v * Vec.from_elements([inv_l], dtype=fx.Float32).broadcast_to(HEAD)).to(fx.Float16)
                for d in range_constexpr(HEAD):
                    output_ptr[seq, qh, d] = o_out[d]
            else:
                # multi-partition: write this partition's (m_p, l_p, numerator O_p);
                # the reduce kernel flash-combines them (value_scale/1/FP8_MAX there).
                base = ((seq * n_kv + kv_h) * arith.constant(NP, type=T.i32) + part) * arith.constant(
                    GS, type=T.i32
                ) + tid
                pmax_ptr[base] = _ld1(sM_off, tid)
                psum_ptr[base] = _ld1(sL_off, tid)
                o_base = base * arith.constant(HEAD, type=T.i32)
                for d in range_constexpr(HEAD):
                    pout_ptr[o_base + d] = o_v[d]

    # ── reduce kernel: flash-combine the NP partition partials -> output ──
    # grid (num_seqs, num_kv_heads, GS): one CTA per query row, so the combine is
    # spread across GS× more CUs (critical for low batch, where grid (seqs,kv) is
    # otherwise just 1 CTA on 1 CU). Each thread d owns one head-dim element.
    RED_THREADS = HEAD

    @flyc.kernel(known_block_size=(RED_THREADS, 1, 1))
    def pa_decode_tile_reduce_kernel(
        output_ptr: fx.Tensor,  # [num_seqs, num_q_heads, HEAD]
        pmax_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, NP, GS]
        psum_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, NP, GS]
        pout_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, NP, GS, HEAD]
        value_scale: fx.Float32,
        num_q_heads: Int32,
    ):
        d = fx.Int32(gpu.thread_id("x"))  # head-dim element
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        row = fx.Int32(gpu.block_id("z"))  # query row within the kv-head group
        n_kv = num_q_heads // arith.constant(GS, type=T.i32)
        out_scale = fx.Float32(value_scale) * fx.Float32(1.0 / FP8_MAX)
        c_GS = arith.constant(GS, type=T.i32)
        # element index of (seq, kv_h, partition 0, this row); + p*GS, then *HEAD + d
        base = (seq * n_kv + kv_h) * arith.constant(NP * GS, type=T.i32) + row

        def _exp2s(x):
            return Vec.from_elements([x], dtype=fx.Float32).exp2()[0]

        # pass 1: global max over partitions
        gmax = fx.Float32(float("-inf"))
        for p in range_constexpr(NP):
            gmax = gmax.maximumf(pmax_ptr[base + arith.constant(p, type=T.i32) * c_GS])
        # pass 2: weighted numerator (this thread's head-dim d) / denominator
        num = fx.Float32(0.0)
        den = fx.Float32(0.0)
        for p in range_constexpr(NP):
            idx = base + arith.constant(p, type=T.i32) * c_GS
            w = _exp2s(pmax_ptr[idx] - gmax)
            den = den + psum_ptr[idx] * w
            num = num + pout_ptr[idx * arith.constant(HEAD, type=T.i32) + d] * w
        qh = kv_h * c_GS + row
        output_ptr[seq, qh, d] = (num / den * out_scale).to(fx.Float16)

    @flyc.jit
    def pa_decode_tile_launch(
        output: fx.Tensor,
        pmax: fx.Tensor,
        psum: fx.Tensor,
        pout: fx.Tensor,
        query: fx.Tensor,
        key_cache: fx.Tensor,
        value_cache: fx.Tensor,
        block_tables: fx.Tensor,
        context_lengths: fx.Tensor,
        key_scale: fx.Float32,
        value_scale: fx.Float32,
        block_size: Int32,
        max_blocks_per_seq: Int32,
        num_q_heads: Int32,
        num_seqs: Int32,
        num_kv_heads: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        pa_decode_tile_kernel(
            output,
            pmax,
            psum,
            pout,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            block_size,
            max_blocks_per_seq,
            num_q_heads,
        ).launch(grid=(num_seqs, num_kv_heads, NP), block=(BLOCK_THREADS, 1, 1), stream=stream)
        if const_expr(NP > 1):
            pa_decode_tile_reduce_kernel(output, pmax, psum, pout, value_scale, num_q_heads).launch(
                grid=(num_seqs, num_kv_heads, GS), block=(RED_THREADS, 1, 1), stream=stream
            )

    return {"launch": pa_decode_tile_launch, "kernel": pa_decode_tile_kernel}


def pa_decode_tile(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: float,
    value_scale: float,
    softmax_scale: float | None = None,
    stream=None,
) -> None:
    """Host entry point. See module docstring for the expected tensor layouts."""
    num_seqs, num_q_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _hd = key_cache.shape
    assert _hd == head_dim
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]

    # Choose the number of context partitions (grid.z).  Split only enough to fill
    # the GPU: when batch*kv_heads already covers the CUs, extra partitions just add
    # partial-write/read traffic and hurt (so high batch -> few partitions).  Also
    # don't split finer than ~512 tokens.  Bucket to powers of two to bound JIT
    # recompiles, and bound by block-table capacity (no device sync).
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        num_cus = int(get_cu_num())
    except Exception:
        num_cus = 304
    ctx_max = max_blocks_per_seq * block_size
    base_ctas = max(1, num_seqs * num_kv_heads)
    npart_by_occ = (num_cus + base_ctas - 1) // base_ctas
    npart_by_ctx = max(1, ctx_max // 256)  # don't split finer than ~1 compute block
    npart_want = max(1, min(npart_by_occ, npart_by_ctx))
    num_partitions = 1
    for cand in (1, 2, 4, 8, 16, 32, 64):
        if cand <= npart_want:
            num_partitions = cand
    GS = query_group_size

    compiled = compile_pa_decode_tile(
        head_dim=head_dim,
        query_group_size=query_group_size,
        num_partitions=num_partitions,
        softmax_scale=softmax_scale,
    )
    dev = query.device
    if num_partitions == 1:
        # NP==1 fast path writes output directly; partials are unused (dead code).
        dummy = torch.empty(1, dtype=torch.float32, device=dev)
        pmax = psum = pout = dummy
    else:
        pmax = torch.empty(num_seqs, num_kv_heads, num_partitions, GS, dtype=torch.float32, device=dev)
        psum = torch.empty(num_seqs, num_kv_heads, num_partitions, GS, dtype=torch.float32, device=dev)
        pout = torch.empty(num_seqs, num_kv_heads, num_partitions, GS, head_dim, dtype=torch.float32, device=dev)
    s = stream or torch.cuda.current_stream()
    compiled["launch"](
        output,
        pmax.view(-1),
        psum.view(-1),
        pout.view(-1),
        query,
        key_cache,
        value_cache,
        block_tables.to(torch.int32),
        context_lengths.to(torch.int32),
        float(key_scale),
        float(value_scale),
        int(block_size),
        int(max_blocks_per_seq),
        int(num_q_heads),
        int(num_seqs),
        int(num_kv_heads),
        s,
    )
