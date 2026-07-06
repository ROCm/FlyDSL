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

Layouts (simple / logical, NOT the production preshuffle layout). ``block_size``
is a **compile-time constant restricted to 16 or 64** (see
``compile_pa_decode_tile``) -- a 256-token compute tile spans multiple pages at
these sizes, so the K/V gather addressing unrolls a fixed page fan-out per
tile at trace time; it cannot take an arbitrary runtime ``block_size``.

* ``query``        ``[num_seqs, num_q_heads, head_dim]``                 f16/bf16
* ``key_cache``    ``[num_blocks, num_kv_heads, head_dim // 32, block_size, 32]``  fp8 e4m3fnuz
                   (K stored with the head-quarter as the outer axis and token
                    as the next-innermost, so that a wave's raw dwordx4 loads --
                    which put adjacent lanes on adjacent tokens, not adjacent
                    head-dim elements, per the MFMA A-operand's fixed lane roles
                    -- land on contiguous, coalesced addresses instead of a
                    ``head_dim``-byte stride per token)
* ``value_cache``  ``[num_blocks, num_kv_heads, head_dim // 16, 16, block_size]``  fp8 e4m3fnuz
                   (V blocked the same way as K -- ``block_size`` innermost --
                    so adjacent lanes -- which own adjacent HEAD values for PV's
                    B operand -- are a contiguous ``block_size``-byte apart
                    instead of a ``block_size``-element (whole page) stride,
                    while each lane's own ``block_size``-token run stays
                    contiguous within its page)
* ``block_tables`` ``[num_seqs, max_blocks_per_seq]``                    int32
                   (``max_blocks_per_seq`` must cover
                    ``ceil(context_len / 256) * 256 / block_size`` pages, i.e.
                    the context length rounded UP to the 256-token compute-tile
                    granularity, not just ``ceil(context_len / block_size)`` --
                    the last tile issues K/V loads for its full 256-token span
                    even when only partially valid (masked out in softmax, not
                    skipped), so under-allocating causes an out-of-bounds
                    block-table read)
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
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.vector import ReductionOp
from kernels.utils import (
    exp2_amdgcn_scalar,
    exp2_f32_fast,
    extract_global_ptr,
    global_load_i32,
    global_load_i64x2,
    global_store,
)

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
    block_size: int,
    num_partitions: int = 1,
    softmax_scale: float | None = None,
):
    """Build the tile-programming PA-decode kernel + launch wrapper.

    Returns a dict with ``launch`` and ``kernel`` entries, mirroring the
    ``compile_*`` factories in ``pa_decode_fp8.py``.

    ``block_size`` is a compile-time constant (not a kernel argument): the K/V
    paged-gather addressing unrolls a fixed number of block-table page lookups
    per compute tile at trace time (``PAGES_PER_CHUNK`` below), so it cannot be
    a runtime value. Only 16 and 64 are supported -- both divide the 64-token
    per-warp chunk evenly, so a chunk maps to either one page (64) or four
    pages (16); see the module docstring / kernel body for the addressing
    derivation. Each distinct ``block_size`` gets its own compiled kernel via
    this function's ``lru_cache``.
    """
    assert head_dim % MFMA_MNK == 0, f"head_dim {head_dim} must be a multiple of {MFMA_MNK}"
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    HEAD = head_dim
    GS = query_group_size
    M = MFMA_MNK  # query rows handled per CTA (padded to 16)
    NWARP = 4  # 4 waves / CTA (matches pa_decode_ps_kernel)
    TOK_PER_WARP = 64  # tokens each warp owns per compute block (matches production KV_COMPUTE_BLOCK)
    TILE_TOK = NWARP * TOK_PER_WARP  # 256 tokens / compute block
    PAGES_PER_CHUNK = TOK_PER_WARP // block_size  # pages spanned by one 64-token warp-chunk: 1 (bs=64) or 4 (bs=16)
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
    # No sS: with the token=M orientation the softmax reduces over M (tokens) with
    # cheap shuffle_xor(16,32), so scores stay in the QK C-fragment.  Only the fp8
    # probabilities sP are staged to LDS — stored transposed to [qhead, token] for
    # the PV A operand.
    sP_off = sQ_off + sQ_bytes
    sP_bytes = M * TILE_TOK * 1  # fp8
    # The running output is register-resident (loop-carried PV C-fragment), so
    # there is no per-tile sOp partial in LDS; sO is only epilogue scratch (the
    # final accumulator is staged here once for the row-major output write).
    sO_off = sP_off + sP_bytes
    sO_bytes = M * HEAD * f32
    sM_off = sO_off + sO_bytes
    sL_off = sM_off + M * f32
    sCorr_off = sL_off + M * f32
    sQscale_off = sCorr_off + M * f32  # per-row query dequant scale
    # cross-warp reduction scratch: per (query row, warp) local max / local sum.
    # Row stride is padded to NWARP+1 (not NWARP) floats: with the plain
    # NWARP=4-float (16-byte = 4-bank) stride, 16 rows wrap the 32-bank LDS
    # twice, so all 16 lanes writing/reading their own row every tile hit a
    # 2-way bank conflict (row r and r+8 always land on the same bank). A
    # stride coprime with 32 banks (5 is) makes all 16 rows land on distinct
    # banks -- same fix as `pa_decode_ps_kernel`'s `PROB_ROW_STRIDE_BYTES`
    # (32 data + 8 padding) for its own LDS row layout.
    NWARP_PAD = NWARP + 1
    sLmax_off = sQscale_off + M * f32
    sLsum_off = sLmax_off + M * NWARP_PAD * f32
    # V page-table prefetch staging: warp w's PAGES_PER_CHUNK-wide row (fetched
    # via one scalar wide load) is broadcast here for all 4 warps to read (V's
    # page depends on `rgroup`, which is shared across warps -- see `_v_page`).
    sVPage_off = sLsum_off + M * NWARP_PAD * f32
    sVPage_bytes = NWARP * PAGES_PER_CHUNK * 4  # i32
    total_bytes = sVPage_off + sVPage_bytes

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_tile_kernel(
        output_ptr: fx.Tensor,  # [num_seqs, num_q_heads, HEAD]  (written directly when NP==1)
        # per-partition partial outputs (combined by the reduce kernel when NP>1):
        pmax_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS]   row max
        psum_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS]   row sum
        pout_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS, HEAD] numerator
        query_ptr: fx.Tensor,  # [num_seqs, num_q_heads, HEAD]
        key_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, HEAD//32, block_size, 32] (blocked, see module docstring)
        value_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, HEAD//16, 16, block_size] (blocked)
        block_tables_ptr: fx.Tensor,  # [num_seqs, max_blocks_per_seq]
        context_lengths_ptr: fx.Tensor,  # [num_seqs]
        key_scale: fx.Float32,
        value_scale: fx.Float32,
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

        context_len = fx.Int32(global_load_i32(extract_global_ptr(context_lengths_ptr), seq))
        bt_gp = extract_global_ptr(block_tables_ptr)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)

        # This CTA only walks its partition's slice of the TILE_TOK-blocks, so the
        # context is parallelized across grid.z CTAs (more CUs for low batch).
        # Computed here (pure arithmetic on context_len/part, no memory access,
        # no dependency on anything below) so the K prefetch that needs it can
        # also be hoisted before the Q-quantization barrier -- see the K-ops
        # section below for why.
        num_tiles = (context_len + arith.constant(TILE_TOK - 1, type=T.i32)) // arith.constant(TILE_TOK, type=T.i32)
        tiles_per_part = (num_tiles + arith.constant(NP - 1, type=T.i32)) // arith.constant(NP, type=T.i32)
        part_start = part * tiles_per_part
        part_end_raw = part_start + tiles_per_part
        part_end = arith.select(part_end_raw < num_tiles, part_end_raw, num_tiles)
        loop_start = fx.Index(arith.unwrap(part_start))
        loop_end = fx.Index(arith.unwrap(part_end))

        kg = extract_global_ptr(key_cache_ptr)  # raw addrspace(1) ptr for dwordx4 K loads
        vg = extract_global_ptr(value_cache_ptr)  # raw addrspace(1) ptr for dwordx4 V loads
        og = extract_global_ptr(output_ptr)  # raw addrspace(1) ptr for the epilogue's vectorized f16 store
        pg = extract_global_ptr(pout_ptr)  # raw addrspace(1) ptr for the epilogue's vectorized f32 partial store

        # ── LDS views ──
        # One i8 blob carved into typed views via byte-offset pointers.  The
        # same view tensor is used both as a tiled-copy partition target and for
        # direct .load()/.store() of whole rows by the row-owner lanes. Defined
        # here (rather than down with the other LDS helpers) so the V
        # page-prefetch helpers below -- issued before the Q-quant barrier,
        # alongside K's own prologue prefetch -- can use it too.
        lds_base = fx.SharedAllocator().allocate(total_bytes).peek().ptr  # i8 base pointer

        def _view(byte_off, elem_ty, layout, esz):
            p = fx.add_offset(lds_base, fx.make_int_tuple(byte_off))
            ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Shared, esz)
            return fx.Tensor(fx.make_view(fx.recast_iter(ptr_ty, p), layout))

        c16 = arith.constant(16, type=T.i32)
        lane16 = lane - (lane // c16) * c16  # 0..15: this row's head-dim chunk index
        rgroup = lane // c16  # 0..3: which quarter-wave (paired with warp -> query row)

        TOK_CHUNK = NWARP * MFMA_MNK  # 64
        NCHUNK = TILE_TOK // TOK_CHUNK  # 4
        c_TILE_TOK = arith.constant(TILE_TOK, type=T.i32)
        c_block_size = arith.constant(block_size, type=T.i32)

        # A compute tile always starts exactly on a page boundary: TILE_TOK
        # (256) is a multiple of block_size for both supported values (16, 64),
        # so there is no "within-page" remainder to track (unlike the old
        # design, which supported block_size >= TILE_TOK and needed a
        # within-tile sub-page index).
        def _tile_tok0_and_page(tt_i32):
            tok0 = tt_i32 * c_TILE_TOK
            return tok0, tok0 // c_block_size

        def _load_phys(page):
            return fx.Int32(global_load_i32(bt_gp, seq * max_blocks_per_seq + page))

        def _load_phys_scalar(page):
            # ONLY valid when `page` is wave-uniform (same value for all 64
            # lanes of the calling warp) -- routes through the scalar/SMEM
            # cache and lands the result directly in an SGPR via
            # llvm.amdgcn.s.buffer.load, instead of a per-lane VMEM
            # global_load_i32 + its `s_waitcnt vmcnt(0)` drain. This is the
            # same fix pa_decode_ps_kernel applies to this exact block-table
            # lookup (see `_pa_small_block_stage_phys_blocks` in
            # pa_decode_fp8.py) -- confirmed via ATT trace there too ("was 25%
            # of all kernel stalls"). K's page depends only on (a, warp), both
            # wave-uniform, so it qualifies directly. V's page depends on
            # `rgroup` (NOT wave-uniform -- varies within a warp), so V can't
            # use this per-warp path for its OWN consumption -- but see
            # `_v_page_fetch_and_stage` below, which uses this same scalar
            # mechanism to *fetch* (not consume) V's pages, one warp per
            # `rgroup` row, broadcasting the result to the other warps via LDS.
            return fx.Int32(
                buffer_ops.buffer_load(bt_rsrc, seq * max_blocks_per_seq + page, vec_width=1, is_scalar=True)
            )

        def _v_page_fetch_and_stage(tt_i32):
            # V's page depends on `rgroup`, which is shared across all 4 warps
            # (the P/V transpose means every warp needs the same 256-token V
            # range for its own head-dim slice) -- so instead of each warp
            # redundantly re-deriving all PAGES_PER_CHUNK pages itself, warp
            # `w` fetches (only) the row for `rgroup == w` via one scalar,
            # wave-uniform wide load (mirroring `_load_phys_scalar`, just
            # vec_width=PAGES_PER_CHUNK instead of 1), and broadcasts it to
            # LDS for every warp to read back (see `_v_page_read_row`). This
            # is prefetched one tile ahead, with the store issued before --
            # and the read-back after -- an already-existing barrier (see the
            # main loop), so no new barrier is added for it.
            _, base_page = _tile_tok0_and_page(tt_i32)
            fetch_off = seq * max_blocks_per_seq + base_page + warp * arith.constant(PAGES_PER_CHUNK, type=T.i32)
            fetched = buffer_ops.buffer_load(bt_rsrc, fetch_off, vec_width=PAGES_PER_CHUNK, is_scalar=True)
            if lane == arith.constant(0, type=T.i32):
                if const_expr(PAGES_PER_CHUNK == 1):
                    _view(
                        arith.constant(sVPage_off, type=T.i32) + warp * arith.constant(4, type=T.i32),
                        fx.Int32,
                        fx.make_layout(1, 1),
                        4,
                    ).store(Vec.from_elements([fx.Int32(fetched)], dtype=fx.Int32))
                else:
                    _view(
                        arith.constant(sVPage_off, type=T.i32) + warp * arith.constant(PAGES_PER_CHUNK * 4, type=T.i32),
                        fx.Int32,
                        fx.make_layout(PAGES_PER_CHUNK, 1),
                        4,
                    ).store(fx.Vector(fetched))

        def _v_page_read_row():
            off = arith.constant(sVPage_off, type=T.i32) + rgroup * arith.constant(PAGES_PER_CHUNK * 4, type=T.i32)
            row = _view(off, fx.Int32, fx.make_layout(PAGES_PER_CHUNK, 1), 4).load()
            return [row[sub] for sub in range_constexpr(PAGES_PER_CHUNK)]

        # ── raw dwordx4 K load (A operand), BLOCKED layout ──
        # Lane (lane16, rgroup) feeds A row m=lane16 = token (a*64 + warp*16 +
        # lane16) with head[rgroup*32 : +32] — the same contiguous slice / head→
        # k_step permutation as q_ops, loaded straight to registers as two i64x2
        # (128-bit) transactions instead of the make_tiled_copy_A fragment the
        # MFMA atom layout caps at 64-bit.
        #
        # key_cache_ptr uses a BLOCKED layout, NOT the plain PA
        # [num_blocks,num_kv_heads,block_size,HEAD] layout:
        # [num_blocks, num_kv_heads, HEAD//32, block_size, 32] (fp8, 1B/elem) — the
        # 32-byte head-quarter is the innermost/contiguous run per token, and
        # consecutive tokens for a FIXED head-quarter are 32B apart. This exists
        # because the plain layout puts head_dim (128B) innermost, so adjacent
        # lanes (which own adjacent TOKENS, not adjacent head-dim slices, per the
        # MFMA-fixed lane roles below) land 128B apart per global_load_i64x2 --
        # confirmed via ATT trace + address-pattern analysis to be the dominant
        # stall (~58% of all cycles) from poor cross-lane coalescing, matching
        # production's own preshuffled/blocked K cache
        # (`key_cache.permute(0,1,3,2,4)` in test_pa.py's small-block harness,
        # which achieves a 16B adjacent-lane stride). Re-laying the axes so the
        # head-quarter is outermost and token is next-innermost makes adjacent
        # lanes (adjacent tokens, fixed head-quarter) exactly 32B apart --
        # contiguous, coalesced multi-lane loads -- while keeping each lane's own
        # 32B slice contiguous for the two i64x2 (w0/w1) reads below.
        #
        # block_size < TILE_TOK means a tile's 64-token warp-chunk `a` can span
        # multiple pages: `local_tok = warp*16+lane16` (0..63) decomposes into
        # `page = base_page + a*PAGES_PER_CHUNK + local_tok//block_size` and
        # `within_page_tok = local_tok % block_size`. At block_size=64,
        # local_tok//64 is always 0 (page depends only on `a`, shared by the
        # whole warp); at block_size=16, local_tok//16 == warp (page depends on
        # (a, warp), shared by all 64 lanes of that warp) -- either way this is
        # one `_load_phys` per (thread's own) warp per `a`, not per-lane.
        c32 = arith.constant(32, type=T.i32)
        c_nhgroup = arith.constant(HEAD // 32, type=T.i32)
        local_tok = warp * c16 + lane16  # 0..63: this thread's token within a 64-token chunk

        def _k_page(base_page, a):
            return base_page + arith.constant(a * PAGES_PER_CHUNK, type=T.i32) + local_tok // c_block_size

        def _k_ops(phys):
            within_page_tok = local_tok % c_block_size
            base = (((phys * n_kv + kv_h) * c_nhgroup + rgroup) * c_block_size + within_page_tok) * c32
            w0 = fx.Vector(global_load_i64x2(kg, base))  # head[rgroup*32 : +16] -> k_step 0,1
            w1 = fx.Vector(global_load_i64x2(kg, base + arith.constant(16, type=T.i32)))  # +16:+32 -> k_step 2,3
            return [w0[0], w0[1], w1[0], w1[1]]

        # All NCHUNK chunks' K as one flat list of NCHUNK*4 i64 operands — carried
        # through the scf.for iter_args so tile tt+1's K prefetch overlaps tt's
        # softmax + PV (cross-iteration pipelining, like pa_decode_ps_kernel).
        NKOPS = NCHUNK * 4
        assert NKOPS == 16, "the k_next skip-on-last-tile unpack below hardcodes 16 names for NKOPS"

        def _k_ops_flat(tt_i32):
            _, base_page = _tile_tok0_and_page(tt_i32)
            # Issue all NCHUNK block-table lookups up front, before any of them
            # is consumed: `page` -> `phys` is a genuine ~600cyc dependent load
            # (block_size < TILE_TOK means up to NCHUNK distinct pages per
            # tile, vs. one shared page in the old >=TILE_TOK design), and
            # computing the K address immediately after each individual
            # `_load_phys` call forces a full `s_waitcnt vmcnt(0)` per lookup --
            # confirmed via ATT trace to be the dominant stall (~49% of all
            # cycles) once block_size shrank below TILE_TOK. Batching the
            # loads lets their latencies overlap instead of serializing.
            phys_list = [_load_phys_scalar(_k_page(base_page, a)) for a in range_constexpr(NCHUNK)]
            flat = []
            for a in range_constexpr(NCHUNK):
                flat.extend(_k_ops(phys_list[a]))
            return flat

        # ── prologue: prefetch the first tile's K ──
        # Issued here -- before Q-quantization and its barrier below -- rather
        # than right before the main loop, so K's global loads (and their
        # block-table lookups) are independent of, and can be scheduled
        # concurrently with, Q's own global load: Q-quant's barrier is a hard
        # reordering boundary, so anything issued *after* it can no longer
        # overlap with anything issued *before* it. The slowest wave's
        # critical path (the one doing real Q-quant work: global load +
        # absmax + pack) pays both latencies back-to-back either way, but
        # issuing them together lets the memory subsystem process them with
        # overlapping latency (memory-level parallelism) instead of fully
        # serial latency-then-latency.
        num_tiles_m1 = num_tiles - arith.constant(1, type=T.i32)
        start_safe = arith.select(part_start < num_tiles, part_start, num_tiles_m1)
        k_pf0 = _k_ops_flat(start_safe)
        # Prologue V page-index prefetch: issued here too (before Q-quant's
        # barrier) so the fetch overlaps Q-quant the same way K's does; the
        # LDS write is then visible after crossing that same barrier below,
        # so `_v_page_read_row` (once `rgroup` is in scope) needs no barrier
        # of its own -- see `_v_page_fetch_and_stage`'s comment.
        _v_page_fetch_and_stage(start_safe)

        # ── per-CTA scalar constants ──
        # softmax_scale and key_scale fold into the score tile; log2e folds into
        # the exp2 used for softmax.
        scale_qk = fx.Float32(_softmax_scale * LOG2E) * fx.Float32(key_scale)
        v_scale_f = fx.Float32(value_scale)
        NEG_INF = fx.Float32(float("-inf"))
        ZERO_F = fx.Float32(0.0)

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

        # f32[16, NWARP] cross-warp scratch (row stride padded to NWARP_PAD to
        # avoid the 2-way LDS bank conflict -- see `sLmax_off`'s comment):
        # scalar write at (row, warp), vec read of a row's NWARP valid slots.
        def _st_lw(base_off, row, w, val):
            off = arith.constant(base_off, type=T.i32) + (
                row * arith.constant(NWARP_PAD, type=T.i32) + w
            ) * arith.constant(4, type=T.i32)
            _view(off, fx.Float32, fx.make_layout(1, 1), 4).store(Vec.from_elements([val], dtype=fx.Float32))

        def _ld_lw_row(base_off, row):
            off = arith.constant(base_off, type=T.i32) + row * arith.constant(NWARP_PAD * 4, type=T.i32)
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

        # sP holds the fp8 probabilities as [qhead, token] (qhead stride TILE_TOK,
        # token stride 1); each lane writes its own (qhead, token) slice directly
        # (raw i32-word store, see the softmax loop) and the raw PV P read (p_ops)
        # reads it back with the same layout.

        # ── MMA atoms (production token=M orientation) ──
        # QK: D[token, qhead] = K(A) · Q(B)ᵀ, tiled (NWARP,1,1) splits tokens (M)
        # across the 4 warps — so the softmax reduces over M (tokens) cheaply.
        # PV: O[qhead, head_dim], tiled (1,NWARP,1) splits head_dim (N).
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_MNK, MFMA_MNK, MFMA_K, FP8))
        tiled_mma_pv = fx.make_tiled_mma(mma_atom, fx.make_layout((1, NWARP, 1), (0, 1, 0)))

        # ── Stage 0: stage this (seq, kv_head)'s GS query rows into LDS sQ.
        # Spread the per-row absmax quantization across all 256 threads instead
        # of GS (<=16) lanes: (warp, rgroup) selects one of the 16 query rows
        # (qh_local = warp*4 + rgroup) and lane16 selects that row's own
        # QCHUNK=8-element (128-bit) head-dim slice, so every thread loads,
        # converts, and packs exactly one chunk -- matching pa_decode_ps_kernel's
        # `_finish_q_fragments` lane-per-chunk layout. The row's absmax is then a
        # butterfly reduction over the 16 lanes sharing (warp, rgroup) via
        # shuffle_xor(width=16), with no LDS/barrier needed for the reduction
        # itself (only the store below needs the barrier after it). This
        # replaces the old design where a single lane serially handled a whole
        # row's 16 chunks while the other ~240 threads idled at the barrier --
        # that idle wait was confirmed via ATT trace to cost ~7-8% of all
        # kernel stall cycles at bs=128/ctx=16384.
        qg = extract_global_ptr(query_ptr)
        QCHUNK = 8  # f16 elements per 128-bit chunk
        NQCHUNK = HEAD // QCHUNK
        assert NQCHUNK == 16, "Q-quant lane assignment requires HEAD == 16 * QCHUNK (128)"

        qh_local = warp * arith.constant(4, type=T.i32) + rgroup  # 0..15: this thread's query row

        if qh_local < arith.constant(GS, type=T.i32):
            qh0 = kv_h * arith.constant(GS, type=T.i32) + qh_local
            row_byte0 = (seq * num_q_heads + qh0) * arith.constant(HEAD * 2, type=T.i32)  # f16 = 2B/elem
            chunk_off = row_byte0 + lane16 * arith.constant(QCHUNK * 2, type=T.i32)
            q_chunk_f16 = fx.Vector(global_load_i64x2(qg, chunk_off)).bitcast(fx.Float16)

            # local absmax over this thread's own 8 elements, then butterfly
            # reduce over the 16 lanes owning this same row (fixed warp/rgroup,
            # lane16 varies) -- fp16->fp32 is monotonic for finite values, so
            # comparing in f16 and only widening the final scalar to f32 avoids
            # feeding a full-vector fpext into a reduce (which would otherwise
            # force the backend to scalarize into per-element v_cvt_f32_f16_sdwa
            # instead of packed v_cvt_pk_f32_f16).
            local_absmax_f16 = fmath.absf(q_chunk_f16).reduce(ReductionOp.MAX)
            for sh in (1, 2, 4, 8):
                local_absmax_f16 = local_absmax_f16.maximumf(
                    local_absmax_f16.shuffle_xor(arith.constant(sh, type=T.i32), c16)
                )
            absmax = local_absmax_f16.to(fx.Float32)
            # per-row symmetric fp8 quantization: q_scale = absmax / FP8_MAX
            q_scale = absmax * fx.Float32(1.0 / FP8_MAX)
            inv = fx.Float32(1.0) / q_scale.maximumf(fx.Float32(1e-20))
            inv_b = Vec.from_elements([inv], dtype=fx.Float32).broadcast_to(QCHUNK)

            q_scaled_chunk = q_chunk_f16.to(fx.Float32) * inv_b
            _st_words(
                arith.constant(sQ_off, type=T.i32)
                + qh_local * arith.constant(HEAD, type=T.i32)
                + lane16 * arith.constant(QCHUNK, type=T.i32),
                _f32_to_fp8_words(q_scaled_chunk),
            )
            if lane16 == arith.constant(0, type=T.i32):
                _st1(sQscale_off, qh_local, q_scale)
        else:
            _st_words(
                arith.constant(sQ_off, type=T.i32)
                + qh_local * arith.constant(HEAD, type=T.i32)
                + lane16 * arith.constant(QCHUNK, type=T.i32),
                Vec.filled(QCHUNK // 4, 0, fx.Int32),
            )
            if lane16 == arith.constant(0, type=T.i32):
                _st1(sQscale_off, qh_local, ZERO_F)

        # ── init running softmax state (row-owner lanes 0..15) ──
        # The output accumulator O is register-resident (loop-carried), so only
        # the scalar m/l running state lives in LDS here. Done here (rather
        # than after the Q-quant barrier below) so its writes share the SAME
        # barrier as the Q-quant writes above instead of needing a second one
        # -- the two are independent (different LDS regions, no ordering
        # requirement between them), so merging saves a full barrier's worth
        # of fixed per-CTA sync overhead. This matters most for small
        # batch/short-context shapes, where ATT trace showed barrier-adjacent
        # LDS/SMEM-wait stalls dominating ~52% of all cycles (there is very
        # little real per-tile work to amortize fixed sync cost against).
        if tid < arith.constant(M, type=T.i32):
            _st1(sM_off, tid, NEG_INF)
            _st1(sL_off, tid, ZERO_F)
        gpu.barrier()

        # First tile's V page-index row, now visible after the barrier above
        # (the fetch+LDS-store was issued earlier, alongside k_pf0).
        v_page_pf0 = _v_page_read_row()

        # Q is the B operand (D[token,qhead] = K·Qᵀ), read raw from sQ as fp8 i64
        # operands for the raw-MFMA QK below (replicated across warps, constant
        # across tiles → read once, held in registers).  Lane (lane16, rgroup)
        # feeds MMA column n=lane16 (qhead) with the K=32 contraction quarter
        # rgroup; QK sums over the full head_dim, so we are free to pick the
        # head→k_step permutation that makes each lane's slice head[rgroup*32:+32]
        # contiguous (one dwordx4 pair) as long as K uses the same mapping.
        q_ops = _view(
            arith.constant(sQ_off, type=T.i32)
            + lane16 * arith.constant(HEAD, type=T.i32)
            + rgroup * arith.constant(32, type=T.i32),
            fx.Int64,
            fx.make_layout(4, 1),
            8,
        ).load()  # 4 fp8 i64 operands = head[rgroup*32 : +32] of qhead=lane16

        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tcopy_o = fx.make_tiled_copy_C(copy_c, tiled_mma_pv)  # PV out -> sO (epilogue)

        # QK in TLOOP chunks of TOK_CHUNK tokens: each small fx.gemm yields a f32x4
        # C-fragment, so the softmax processes 4 scores at a time (scores stay in
        # AGPR, VGPR peak low) — matching pa_decode_ps_kernel's TLOOP.
        # compile-time per-chunk token offsets (token = a*TOK_CHUNK + base + r)
        _ct = [
            Vec.from_elements([arith.constant(float(a * TOK_CHUNK + r), type=T.f32) for r in range_constexpr(4)])
            for a in range_constexpr(NCHUNK)
        ]
        # P·V is loop-tiled over head-dim (like the production VHELOOP): each step
        # computes O[:, vh*VHE_SIZE : +VHE_SIZE], shrinking the live V operand and
        # PV accumulator instead of materializing the full [16, HEAD] at once.
        VHE_CHUNKS = 2
        VHE_SIZE = HEAD // VHE_CHUNKS
        tmpl_Op = fx.make_rmem_tensor(fx.make_layout((M, VHE_SIZE), (VHE_SIZE, 1)), fx.Float32)
        OP_ELEMS = M * VHE_SIZE // (NWARP * WAVE)  # PV C-fragment elements/lane/chunk (probed = 4)

        # ── raw dwordx4 V load (B operand), BLOCKED layout ──
        # PV contracts over token, so (like QK's head permutation) the token→k_step
        # mapping is free as long as V and P (p_ops) agree: lane (rgroup) takes the
        # contiguous token slice [rgroup*64 : +64] for its head (vh*VHE_SIZE +
        # warp*16 + lane16), loaded as 4× i64x2 (128-bit) = 8 k_step operands.
        #
        # value_cache_ptr uses a BLOCKED layout (same coalescing motivation as K,
        # see its comment above): [num_blocks, num_kv_heads, HEAD//16, 16,
        # block_size] (fp8, 1B/elem) -- block_size innermost, mirroring K --
        # instead of the plain transposed [num_blocks,num_kv_heads,HEAD,block_size]:
        # adjacent lanes own adjacent HEAD values (lane16), so the plain layout's
        # block_size-byte stride per lane is worse than K's. head_group = vh*4+warp
        # needs no runtime div/mod (VHE_SIZE=64 and 16 both divide warp*16 and
        # vh*VHE_SIZE evenly).
        #
        # A rgroup's 64-contiguous-token PV operand run can itself span multiple
        # block_size-sized pages: outer loop `sub` (0..PAGES_PER_CHUNK-1) picks
        # the page, inner loop `step` (0..block_size//16-1) walks that page's own
        # block_size-token run in 16-token (one i64x2) increments. This collapses
        # to today's single-page/4-step behavior at block_size=64 and becomes
        # 4 pages/1 step at block_size=16 -- both total NVOPS=8 i64 operands.
        NVOPS = TILE_TOK // MFMA_K  # 8 PV k_steps (256 tokens / K=32)
        c_nhgroup16 = arith.constant(HEAD // 16, type=T.i32)
        STEPS_PER_PAGE = block_size // 16

        # V's page depends only on (rgroup, sub) -- NOT on `warp` -- so all 4
        # warps (256 threads) want the exact same V_PAGE_COUNT pages every
        # tile. See `_v_page_fetch_and_stage`/`_v_page_read_row` (above, near
        # `_load_phys_scalar`) for how the page *index* is fetched once
        # (scalar, warp-partitioned, LDS-broadcast, prefetched one tile ahead
        # reusing an existing barrier) instead of redundantly re-derived per
        # lane every tile.
        def _v_ops(phys_row, vh):
            head_group = arith.constant((vh * VHE_SIZE) // 16, type=T.i32) + warp
            ops = []
            for sub in range_constexpr(PAGES_PER_CHUNK):
                page_base = (((phys_row[sub] * n_kv + kv_h) * c_nhgroup16 + head_group) * c16 + lane16) * c_block_size
                for step in range_constexpr(STEPS_PER_PAGE):
                    w = fx.Vector(global_load_i64x2(vg, page_base + arith.constant(step * 16, type=T.i32)))
                    ops.extend([w[0], w[1]])
            return ops  # NVOPS i64, the 64-token contiguous run for this head

        def _v_ops_from_phys_row(phys_row):
            return [_v_ops(phys_row, vh) for vh in range_constexpr(VHE_CHUNKS)]

        # O is carried in registers across tiles (one VHE_CHUNKS-list of PV
        # C-fragment vectors), rescaled by the softmax correction each tile.
        o_zero = Vec.filled(OP_ELEMS, 0.0, fx.Float32)
        for tt, ostate in range(loop_start, loop_end, arith.index(1), init=[o_zero, o_zero, *k_pf0, *v_page_pf0]):
            o_acc = [ostate[0], ostate[1]]
            k_cur = [ostate[2 + i] for i in range_constexpr(NKOPS)]  # this tile's prefetched K
            v_page_cur = [ostate[2 + NKOPS + i] for i in range_constexpr(PAGES_PER_CHUNK)]  # this tile's V pages
            tt_i32 = fx.Int32(arith.index_cast(T.i32, tt))
            tok0, _ = _tile_tok0_and_page(tt_i32)

            # ---- hoist raw dwordx4 V loads BEFORE QK so their DMA hides behind QK
            # + softmax (production loads V early too); consumed by the PV MMA.
            # `v_page_cur` was prefetched last iteration (see below). ----
            v_cur = _v_ops_from_phys_row(v_page_cur)

            # ---- QK in TLOOP chunks: NCHUNK raw MFMAs -> f32x4/lane ----
            # Each chunk accumulates the 4 head-quarter k_steps (this tile's
            # prefetched k_cur) into one f32x4 C-fragment (D[token, qhead]); the raw
            # dwordx4 K feeds the same MFMA the old fx.gemm wrapped, so the C layout
            # — and thus the softmax / P-pack / PV below — is unchanged.
            frag_Ss = []
            for a in range_constexpr(NCHUNK):
                acc = arith.constant_vector(0.0, T.f32x4)
                for s in range_constexpr(4):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k_cur[a * 4 + s], q_ops[s], acc, 0, 0, 0])
                frag_Ss.append(fx.Vector(acc))

            # prefetch next tile's K (the MFMAs above consumed k_cur); the loads
            # overlap the softmax + PV below and become next iter's state. At
            # these small block sizes the backing page(s) almost always change
            # tile-to-tile, so (unlike the old >=TILE_TOK block_size design)
            # there is no same-page case worth special-casing -- every tile
            # re-derives its own page(s) fresh via `_k_ops_flat`.
            #
            # (V is deliberately NOT carried the same way: an earlier attempt
            # at prefetching v_next one tile ahead, mirroring k_next, pushed
            # VGPR usage up ~20% -- 136 -> 164 -- and measured *slower* despite
            # the extra hiding time, so V stays hoisted at tile-start instead.)
            #
            # On a partition's LAST tile there is no next tile to prefetch, so
            # skip the load with a real conditional (not `arith.select`, which
            # only clamps the *index* -- the K global-loads and their page
            # lookups would still execute and be thrown away). This matters
            # most exactly when a partition has few tiles (e.g. NP chosen so
            # each partition covers just 1 tile): unconditionally prefetching
            # there was pure waste, confirmed via kernel-level profiling to be
            # a meaningful fraction of the main kernel's time for that shape.
            # The placeholder value assigned when skipped is never read: the
            # loop doesn't continue, so nothing consumes `k_next` in that case.
            #
            # NKOPS is a fixed compile-time constant (NCHUNK=4 chunks x 4
            # i64/chunk = 16); this DSL's dynamic-`if` variable-reassignment
            # tracking requires each state variable to be an individual
            # MLIR-backed scalar (not a Python list), hence the explicit
            # unpack/repack instead of assigning `k_next` as one list.
            tt1 = tt_i32 + arith.constant(1, type=T.i32)
            (
                kn0,
                kn1,
                kn2,
                kn3,
                kn4,
                kn5,
                kn6,
                kn7,
                kn8,
                kn9,
                kn10,
                kn11,
                kn12,
                kn13,
                kn14,
                kn15,
            ) = k_cur
            if tt1 < part_end:
                (
                    kn0,
                    kn1,
                    kn2,
                    kn3,
                    kn4,
                    kn5,
                    kn6,
                    kn7,
                    kn8,
                    kn9,
                    kn10,
                    kn11,
                    kn12,
                    kn13,
                    kn14,
                    kn15,
                ) = _k_ops_flat(tt1)
            k_next = [kn0, kn1, kn2, kn3, kn4, kn5, kn6, kn7, kn8, kn9, kn10, kn11, kn12, kn13, kn14, kn15]

            # Prefetch next tile's V page-index row the same way (see
            # `_v_page_fetch_and_stage`): issue the scalar fetch + LDS store
            # here, before the softmax barrier below; read it back (into
            # `v_page_next`) right after that barrier, reusing it instead of
            # adding a new one.
            if tt1 < part_end:
                _v_page_fetch_and_stage(tt1)

            # ---- register-resident softmax over M = token, 4 scores at a time ----
            # Each lane owns ONE qhead (= lane%16); reduce its tokens with a register
            # reduce + shuffle_xor(16,32) (2 offsets).  Scores stay in AGPR (chunk
            # fragments); the mask is a cheap scalar threshold (token = a*64 + base + r
            # < n_valid  <=>  (a*64+r) < n_valid - base).
            c16 = arith.constant(16, type=T.i32)
            c_w = arith.constant(WAVE, type=T.i32)
            qh = lane - (lane // c16) * c16  # qhead = lane % 16
            l16g = lane // c16  # 0..3 lane-group within the warp
            scale = scale_qk * _ld1(sQscale_off, qh)  # per-qhead positive score scale
            n_valid_tile = (context_len - tok0).to(fx.Float32)
            base_tok_f = fx.Int32(warp * c16 + l16g * arith.constant(4, type=T.i32)).to(fx.Float32)
            thr = Vec.from_elements([n_valid_tile - base_tok_f], dtype=fx.Float32).broadcast_to(4)
            neg4 = Vec.filled(4, float("-inf"), fx.Float32)

            # Computed once and reused in pass 2 below (was previously
            # recomputed from scratch in both passes -- the compare + select
            # doesn't depend on anything pass 1 produces, so this halves the
            # mask compare/select instruction count for no cost: the values
            # just stay VGPR-resident across the barrier, same as any other
            # loop-local state).
            masked_chunks = [(_ct[a] < thr).select(frag_Ss[a], neg4) for a in range_constexpr(NCHUNK)]

            # pass 1: per-warp max for this qhead
            pm = fx.Float32(float("-inf"))
            for a in range_constexpr(NCHUNK):
                pm = pm.maximumf(masked_chunks[a].reduce(ReductionOp.MAX))
            for sh in (16, 32):
                pm = pm.maximumf(pm.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            if l16g == arith.constant(0, type=T.i32):  # one lane per (qhead, warp)
                _st_lw(sLmax_off, qh, warp, pm * scale)
            gpu.barrier()

            # Read back next tile's V page-index row now that the barrier
            # above has made the store from `_v_page_fetch_and_stage` (issued
            # earlier this iteration) visible. Same DSL constraint as k_next:
            # if-reassigned loop state must be individual scalars, not a
            # Python list, so branch on the (compile-time) row width instead
            # of writing one generic path.
            if const_expr(PAGES_PER_CHUNK == 4):
                vp0, vp1, vp2, vp3 = v_page_cur
                if tt1 < part_end:
                    vp0, vp1, vp2, vp3 = _v_page_read_row()
                v_page_next = [vp0, vp1, vp2, vp3]
            else:
                assert PAGES_PER_CHUNK == 1
                (vp0,) = v_page_cur
                if tt1 < part_end:
                    (vp0,) = _v_page_read_row()
                v_page_next = [vp0]

            # pass 2: global max over warps -> exp -> fp8 P pack (-> sP) -> sum
            m_old = _ld1(sM_off, qh)
            m_new = m_old.maximumf(_ld_lw_row(sLmax_off, qh).reduce(ReductionOp.MAX))
            m_new_b = Vec.from_elements([m_new], dtype=fx.Float32).broadcast_to(4)
            ls = fx.Float32(0.0)
            # raw i32-word store straight to sP[qhead][token_base:+4] (fp8, 1B/elem):
            # the packed word's 4 fp8 lanes are exactly the 4 consecutive tokens this
            # lane owns in chunk `a` (token_base = a*TOK_CHUNK + warp*16 + l16g*4), so
            # a direct store replaces the make_fragment_C/tiled_copy_C round trip.
            base4 = arith.constant(4, type=T.i32)
            for a in range_constexpr(NCHUNK):
                # HW exp2 intrinsic (exp2_f32_fast) instead of MLIR's generic
                # math.exp2 (a polynomial approximation whose edge-case handling
                # costs ~32 extra v_cndmask here, measured via ISA source-line
                # attribution) -- matches ps's own exp2_f32_fast usage.
                Pa = fx.Vector(exp2_f32_fast(masked_chunks[a] * scale - m_new_b))
                ls = ls + Pa.reduce(ReductionOp.ADD)
                word = _f32_to_fp8_words(Pa * Vec.filled(4, FP8_MAX, fx.Float32))[0]
                token_base = arith.constant(a * TOK_CHUNK, type=T.i32) + warp * c16 + l16g * base4
                p_off = arith.constant(sP_off, type=T.i32) + qh * c_TILE_TOK + token_base
                _view(p_off, fx.Int32, fx.make_layout(1, 1), 4).store(Vec.from_elements([word], dtype=fx.Int32))
            for sh in (16, 32):
                ls = ls + ls.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
            if l16g == arith.constant(0, type=T.i32):
                _st_lw(sLsum_off, qh, warp, ls)
                if warp == arith.constant(0, type=T.i32):
                    _st1(sM_off, qh, m_new)
                    _st1(sCorr_off, qh, fx.Float32(exp2_amdgcn_scalar(m_old - m_new)))
            gpu.barrier()

            # phase 3: merge per-warp sums into the running denominator
            if tid < arith.constant(M, type=T.i32):
                gsum = _ld_lw_row(sLsum_off, tid).reduce(ReductionOp.ADD)
                _st1(sL_off, tid, _ld1(sL_off, tid) * _ld1(sCorr_off, tid) + gsum)

            # ---- read P back as the A operand for P·V, raw (replicated across
            # warps) — lane reads sP[qhead=lane16][token rgroup*64:+64] as NVOPS i64,
            # the same permuted token slice v_ops uses so the raw PV MMA matches. ----
            p_ops = _view(
                arith.constant(sP_off, type=T.i32) + lane16 * c_TILE_TOK + rgroup * arith.constant(64, type=T.i32),
                fx.Int64,
                fx.make_layout(NVOPS, 1),
                8,
            ).load()

            # ---- PV with register-resident O accumulate (no LDS round-trip) ----
            # O_new = O_old * corr + P·V per head-dim chunk; corr = exp2(m_old-m_new)
            # is per-row.  Probed PV C-fragment: vec element v of lane L holds row
            # m = (L%64//16)*4 + v, so corr_s[v] = corr[m_base + v].  Done element-
            # wise (Vec*Vec broadcasts to an outer product here).  No barrier after
            # PV: O is in registers and the next iter's QK/phase2 barriers order
            # any sS/sP reuse (sOp is gone).  Raw PV MMA: NVOPS k_steps accumulate
            # into one f32x4 (this warp's [16 qhead, 16 head] output atom).
            m_base_pv = (lane // c16) * arith.constant(4, type=T.i32)
            # OP_ELEMS contiguous f32 (indices m_base_pv..+OP_ELEMS-1) in one
            # vectorized LDS read instead of OP_ELEMS separate scalar reads.
            corr_off = arith.constant(sCorr_off, type=T.i32) + m_base_pv * arith.constant(4, type=T.i32)
            corr_vec = _view(corr_off, fx.Float32, fx.make_layout(OP_ELEMS, 1), 4).load()
            corr_s = [corr_vec[v] for v in range_constexpr(OP_ELEMS)]
            for vh in range_constexpr(VHE_CHUNKS):
                acc = arith.constant_vector(0.0, T.f32x4)
                for s in range_constexpr(NVOPS):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [p_ops[s], v_cur[vh][s], acc, 0, 0, 0])
                op = fx.Vector(acc)
                oo = o_acc[vh]
                o_acc[vh] = Vec.from_elements(
                    [oo[v] * corr_s[v] + op[v] for v in range_constexpr(OP_ELEMS)], dtype=fx.Float32
                )
            results = yield [o_acc[0], o_acc[1], *k_next, *v_page_next]
        o_final = results

        # ── stage the register-resident O accumulator to sO (row-major) so the
        # epilogue can read whole rows and write the output as before ──
        thr_copy_o_e = tcopy_o.get_slice(tid)
        for vh in range_constexpr(VHE_CHUNKS):
            frag_Oe = tiled_mma_pv.get_slice(tid).make_fragment_C(tmpl_Op)
            frag_Oe.store(o_final[vh])
            sO_chunk = _view(
                arith.constant(sO_off + vh * VHE_SIZE * 4, type=T.i32),
                fx.Float32,
                fx.make_layout((M, VHE_SIZE), (HEAD, 1)),
                4,
            )
            fx.copy(copy_c, thr_copy_o_e.retile(frag_Oe), thr_copy_o_e.partition_D(sO_chunk), pred=None)
        gpu.barrier()

        # ── epilogue: spread the row -> global write across ALL 256 threads
        # (THREADS_PER_ROW threads per query row, each owning a contiguous
        # ELEMS_PER_THREAD-wide slice) instead of only the GS row-owner lanes
        # looping over all HEAD elements themselves. The old row-owner-only
        # version had only 8-16 of 256 lanes active, each doing a HEAD-long
        # unrolled scalar loop (HEAD/4 sequential LDS reads + HEAD sequential
        # global stores) -- this version does the same total work as 1-2
        # vectorized LDS reads + 1-2 vectorized global stores per lane, fully
        # using the wave and cutting the epilogue's static instruction count
        # (measured ds_read: 45 -> matching ps's 15-ish range).
        assert BLOCK_THREADS % GS == 0, "epilogue requires BLOCK_THREADS to divide evenly by GS"
        THREADS_PER_ROW = BLOCK_THREADS // GS
        ELEMS_PER_THREAD = HEAD // THREADS_PER_ROW
        assert ELEMS_PER_THREAD * THREADS_PER_ROW == HEAD, "epilogue requires HEAD % (BLOCK_THREADS // GS) == 0"
        assert ELEMS_PER_THREAD % 4 == 0, "epilogue vectorizes global stores in f32x4/f16x4 units"

        c_tpr = arith.constant(THREADS_PER_ROW, type=T.i32)
        row_e = tid // c_tpr
        sub_e = tid - row_e * c_tpr
        col_e = sub_e * arith.constant(ELEMS_PER_THREAD, type=T.i32)
        row_off = (
            arith.constant(sO_off, type=T.i32)
            + row_e * arith.constant(HEAD * 4, type=T.i32)
            + col_e * arith.constant(4, type=T.i32)
        )
        o_v = _view(row_off, fx.Float32, fx.make_layout(ELEMS_PER_THREAD, 1), 4).load()

        if const_expr(NP == 1):
            # single partition: normalize and write the output directly (no
            # partials / reduce round-trip).  Fold value_scale and 1/FP8_MAX.
            qh = kv_h * arith.constant(GS, type=T.i32) + row_e
            inv_l = (fx.Float32(1.0) / _ld1(sL_off, row_e)) * v_scale_f * fx.Float32(1.0 / FP8_MAX)
            o_out = (o_v * Vec.from_elements([inv_l], dtype=fx.Float32).broadcast_to(ELEMS_PER_THREAD)).to(fx.Float16)
            out_byte_off = fx.Int64((seq * num_q_heads + qh) * arith.constant(HEAD * 2, type=T.i32)) + fx.Int64(
                col_e
            ) * fx.Int64(2)
            for c in range_constexpr(ELEMS_PER_THREAD // 4):
                chunk = Vec.from_elements([o_out[c * 4 + i] for i in range_constexpr(4)], dtype=fx.Float16)
                packed = chunk.bitcast(fx.Int64)
                global_store(og, out_byte_off + fx.Int64(c * 8), packed[0], alignment=8)
        else:
            # multi-partition: write this partition's (m_p, l_p, numerator O_p);
            # the reduce kernel flash-combines them (value_scale/1/FP8_MAX there).
            base = ((seq * n_kv + kv_h) * arith.constant(NP, type=T.i32) + part) * arith.constant(
                GS, type=T.i32
            ) + row_e
            if sub_e == arith.constant(0, type=T.i32):
                pmax_ptr[base] = _ld1(sM_off, row_e)
                psum_ptr[base] = _ld1(sL_off, row_e)
            o_base = base * arith.constant(HEAD, type=T.i32) + col_e
            pout_byte_off = fx.Int64(o_base) * fx.Int64(4)
            for c in range_constexpr(ELEMS_PER_THREAD // 4):
                chunk = Vec.from_elements([o_v[c * 4 + i] for i in range_constexpr(4)], dtype=fx.Float32)
                global_store(pg, pout_byte_off + fx.Int64(c * 16), chunk.ir_value(), alignment=16)

    # ── reduce kernel: flash-combine the NP partition partials -> output ──
    # grid (num_seqs, num_kv_heads, GS): one CTA per query row, so the combine is
    # spread across GS× more CUs (critical for low batch, where grid (seqs,kv) is
    # otherwise just 1 CTA on 1 CU). Each thread d owns one head-dim element.
    #
    # (Tried collapsing grid.z=GS into one CTA per (seq,kv_head) with an
    # internal GS-row loop, on the theory that GS tiny CTAs cost more in
    # per-CTA dispatch overhead than they gain in breadth -- measured
    # dramatically *slower* (batch=3, ctx=1027: 18.2 -> 28.2us): the lost
    # GS-way CU parallelism far outweighs any dispatch-overhead saving.
    # Reverted; the per-row grid.z is load-bearing, not incidental.)
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
            return fx.Float32(exp2_amdgcn_scalar(x))

        # pmax/psum are indexed only by (seq, kv_h, row, p) -- the same address
        # for all RED_THREADS=HEAD threads in this CTA, since neither depends
        # on `d`. Reading them via the tensor's per-lane vector load makes
        # every thread redundantly issue its own global load for an address
        # that's uniform across the whole block; route through a scalar
        # buffer load instead so it lands once in an SGPR and broadcasts for
        # free, the same fix applied to the main kernel's block-table lookup
        # (see `_load_phys_scalar`).
        pmax_rsrc = buffer_ops.create_buffer_resource(pmax_ptr, max_size=True)
        psum_rsrc = buffer_ops.create_buffer_resource(psum_ptr, max_size=True)

        def _ld_scalar_f32(rsrc, idx):
            return fx.Int32(buffer_ops.buffer_load(rsrc, idx, vec_width=1, is_scalar=True)).bitcast(fx.Float32)

        # pass 1: global max over partitions
        gmax = fx.Float32(float("-inf"))
        for p in range_constexpr(NP):
            gmax = gmax.maximumf(_ld_scalar_f32(pmax_rsrc, base + arith.constant(p, type=T.i32) * c_GS))
        # pass 2: weighted numerator (this thread's head-dim d) / denominator
        num = fx.Float32(0.0)
        den = fx.Float32(0.0)
        for p in range_constexpr(NP):
            idx = base + arith.constant(p, type=T.i32) * c_GS
            w = _exp2s(_ld_scalar_f32(pmax_rsrc, idx) - gmax)
            den = den + _ld_scalar_f32(psum_rsrc, idx) * w
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
    num_blocks, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape
    assert num_hgroups * hgroup_width == head_dim and hgroup_width == 32
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]

    # Choose the number of context partitions (grid.z). This kernel launches a
    # *static* one-CTA-per-partition grid (unlike production's *persistent*
    # kernel, which dynamically rebalances work across a fixed set of launched
    # thread blocks and is what `get_recommended_splits` is tuned for), so it
    # needs its own NP heuristic. Two regimes, both measured directly on an
    # 80-CU MI308X:
    #
    #   - CU-STARVED (num_seqs * num_kv_heads < device CU count): batch alone
    #     doesn't even touch every CU, so activating more CUs matters more
    #     than amortizing each partition's fixed cost -- push NP up to
    #     `cu_fill_np`, uncapped by tiles-per-partition (thin partitions are
    #     fine when the alternative is idle CUs). Measured: batch=1,
    #     ctx=16384 -> NP=8 (only 8 CTAs) to a CU-filling NP=32 is 2.5x
    #     faster; batch=3, ctx=1027 [5 tiles] -> NP=5 (= the tile count
    #     itself) beats NP=1 by ~30%.
    #   - NOT CU-STARVED (batch already >= CU count): further CTAs don't add
    #     breadth (CUs are already all touched), only per-CU occupancy depth,
    #     and splitting into more, smaller per-partition tile ranges adds
    #     reduce-kernel + prologue overhead for each extra partition. Cap
    #     tiles-per-partition to at least MIN_TILES_PER_PARTITION here.
    #     Measured: batch=81, ctx=16384 [64 tiles] -> NP=5..8 all close, ~6-8%
    #     faster than NP=4; but batch=81, ctx=1027 [5 tiles] -> NP=8 (an
    #     uncapped CU-fill formula's pick) measured 30% *slower* than NP=2 --
    #     confirming the same total CTAs (648 vs 162) is *not* what mattered;
    #     tiles-per-partition (1 vs 3) was.
    #
    # TARGET_CTAS_PER_CU=8 and MIN_TILES_PER_PARTITION=2 were picked by a
    # direct sweep on that hardware (not tightly derived from occupancy
    # theory) to land at or very near every measured sweet spot above.
    from kernels.pa_decode_fp8 import get_recommended_splits

    TARGET_CTAS_PER_CU = 8
    MIN_TILES_PER_PARTITION = 2
    device_cus = torch.cuda.get_device_properties(query.device).multi_processor_count
    cu_fill_np = -(-(TARGET_CTAS_PER_CU * device_cus) // (num_seqs * num_kv_heads))  # ceil div
    # Bounded by `max_blocks_per_seq * block_size / TILE_TOK` (an upper bound
    # on the number of 256-token compute tiles any sequence could need), NOT
    # by the actual per-sequence context length: reading `context_lengths` on
    # the host (e.g. via `.max().item()`) forces a GPU sync, which is illegal
    # during CUDA graph capture and broke it outright when tried. This bound
    # is exact when callers size `block_tables` to the actual context length
    # (as this repo's own tests/benchmarks do); callers that over-allocate
    # `block_tables` for a larger max-sequence-length than any single call
    # uses will see a looser (but still correct) bound here.
    tile_tok = 256
    max_possible_tiles = -(-(max_blocks_per_seq * block_size) // tile_tok)  # ceil div, no host sync
    cu_starved = (num_seqs * num_kv_heads) < device_cus
    tiles_np_cap = max_possible_tiles if cu_starved else max(1, max_possible_tiles // MIN_TILES_PER_PARTITION)
    base_np = get_recommended_splits(num_seqs, num_kv_heads)
    num_partitions = max(1, min(max(base_np, cu_fill_np), tiles_np_cap))
    GS = query_group_size

    compiled = compile_pa_decode_tile(
        head_dim=head_dim,
        query_group_size=query_group_size,
        block_size=int(block_size),
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
        int(max_blocks_per_seq),
        int(num_q_heads),
        int(num_seqs),
        int(num_kv_heads),
        s,
    )
