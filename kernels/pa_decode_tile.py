# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

This is a *correctness-first* reimplementation of the decode math that the
production ``pa_decode_ps_kernel`` (``pa_decode_fp8.py``) computes, written with
FlyDSL's high-level tile/layout API (``make_buffer_tensor`` + ``zipped_divide`` +
``make_tiled_mma`` + ``fx.gemm`` + tiled copies) instead of raw buffer
intrinsics and hand-scheduled MFMA.  It deliberately trades performance for
clarity and is scoped to a single configuration:

* per-tensor fp8 K/V quantization (``key_scale`` / ``value_scale``, 1-element
  device tensors read in-kernel via a scalar buffer load, like
  ``pa_decode_ps_kernel``),
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
``head_dim`` is a compile-time constant restricted to a multiple of 64 (64 or
128; not 32) -- the same floor ``pa_decode_ps_kernel`` imposes, since the raw
dwordx4-load-based addressing below has a 64-element minimum granularity.

* ``query``        ``[num_seqs, num_q_heads, head_dim]``                 f16/bf16
* ``key_cache``    ``[num_blocks, num_kv_heads, head_dim // 16, block_size, 16]``  fp8 e4m3fnuz
                   (SAME layout ``pa_decode_ps_kernel`` uses -- see
                    ``_pa_small_block_load_k_flat`` in ``pa_decode_fp8.py`` --
                    so both kernels share one cache-prep path. K stored with
                    the 16-element head-chunk as the outer axis and token as
                    the next-innermost, so that a wave's raw dwordx4 loads --
                    which put adjacent lanes on adjacent tokens, not adjacent
                    head-dim elements, per the MFMA A-operand's fixed lane roles
                    -- land on contiguous, coalesced addresses instead of a
                    ``head_dim``-byte stride per token. See
                    ``RGROUP_QUARTERS``/``QKHE_LOOP`` in
                    ``compile_pa_decode_tile``)
* ``value_cache``  ``[num_blocks, num_kv_heads, block_size // 16, head_dim, 16]``  fp8 e4m3fnuz
                   (SAME "trans_v" layout ``pa_decode_ps_kernel`` uses -- see
                    ``_pa_small_block_load_v_trans`` in ``pa_decode_fp8.py`` --
                    so both kernels share one cache-prep path. Unlike K, V is
                    TOKEN-vectorized: 16 CONSECUTIVE TOKENS are innermost,
                    head_dim is the (unchunked) middle axis, and the
                    ``block_size // 16`` token-subblock index is outermost --
                    adjacent lanes, which own adjacent HEAD values for PV's B
                    operand, are a contiguous 16-byte apart instead of a
                    ``block_size``-element (whole page) stride)
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
from flydsl.expr.vector import ReductionOp
from kernels import dpp_utils
from kernels.utils import (
    cdiv,
    exp2_amdgcn_scalar,
    exp2_f32_fast,
    rcp_f32,
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
    query_dtype: str = "f16",
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

    ``head_dim`` must be a multiple of 64 (64 or 128; not 32) -- same floor as
    ``pa_decode_ps_kernel``'s own head_dim constraint, since the raw
    dwordx4-load addressing this kernel uses has a 64-element minimum
    granularity.

    ``query_dtype`` selects the query tensor's 16-bit float element type
    (``"f16"`` or ``"bf16"``), matching ``pa_decode_ps_kernel``'s own
    ``query_input_dtype`` flag. It's a compile-time constant (not a kernel
    argument) since it picks the load's element type at trace time; each
    distinct value gets its own compiled kernel via this function's
    ``lru_cache``.
    """
    assert head_dim % MFMA_MNK == 0, f"head_dim {head_dim} must be a multiple of {MFMA_MNK}"
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    assert query_dtype in (
        "f16",
        "bf16",
    ), f"pa_decode_tile only supports query_dtype in ('f16', 'bf16'), got {query_dtype}"
    Q_DTYPE = fx.BFloat16 if query_dtype == "bf16" else fx.Float16
    # head_dim must be a multiple of 64: same floor as pa_decode_ps_kernel's own
    # head_dim constraint (a raw dwordx4 fp8 fetch is 16B = 16 elements; the
    # smallest per-lane-group unit this kernel's addressing scheme divides
    # head_dim into is 64 elements -- see QKHE_LOOP/N_SUBCHUNKS/VHE_CHUNKS
    # below), so head_dim=32 would need a genuinely smaller load granularity,
    # not just different formulas. Only 64 and 128 are supported.
    assert head_dim % 64 == 0, f"pa_decode_tile only supports head_dim that's a multiple of 64, got {head_dim}"
    HEAD = head_dim
    GS = query_group_size
    M = MFMA_MNK  # query rows handled per CTA (padded to 16)
    NWARP = 4  # 4 waves / CTA (matches pa_decode_ps_kernel)
    TOK_PER_WARP = 64  # tokens each warp owns per compute block (matches production KV_COMPUTE_BLOCK)
    TILE_TOK = NWARP * TOK_PER_WARP  # 256 tokens / compute block
    PAGES_PER_CHUNK = TOK_PER_WARP // block_size  # pages spanned by one 64-token warp-chunk: 1 (bs=64) or 4 (bs=16)
    assert HEAD % (NWARP * MFMA_MNK) == 0, "head_dim must split across the 4 warps for PV"

    # ── head_dim-derived QK contraction chunking (matches pa_decode_ps_kernel's
    # own K/Q addressing exactly -- see _pa_small_block_load_k_flat /
    # _finish_q_fragments in pa_decode_fp8.py) ──
    # The fp8 mfma_f32_16x16x32 atom's A/B operand is M*K/WAVE = 16*32/64 = 8
    # fp8 elements (one i64) per lane per instruction. head_dim is chunked in
    # TWO levels: a fixed 16-element chunk (`QK_CHUNK_ELEMS`, one dwordx4
    # load), 4 of which (`RGROUP_QUARTERS = WAVE // MFMA_MNK`, architecture
    # fixed, this kernel's `rgroup` == production's `rowid`) make up one
    # 64-element "fetch group"; head_dim's outer fetch-group count
    # (`QKHE_LOOP`) is what actually scales with head_dim:
    #   QKHE_LOOP = HEAD // (RGROUP_QUARTERS * QK_CHUNK_ELEMS)  (2 for HEAD=128, 1 for 64)
    #   N_SUBCHUNKS = QKHE_LOOP * (QK_CHUNK_ELEMS // 8)          (4 for HEAD=128, 2 for 64)
    # head_dim element for a given (qkhe, rgroup, qkr) triple:
    #   (qkhe * RGROUP_QUARTERS + rgroup) * QK_CHUNK_ELEMS + qkr * 8
    # N_SUBCHUNKS replaces the QK inner loop's old hardcoded `4`.
    RGROUP_QUARTERS = 4
    QK_CHUNK_ELEMS = 16
    QKHE_LOOP = HEAD // (RGROUP_QUARTERS * QK_CHUNK_ELEMS)
    assert QKHE_LOOP >= 1, f"head_dim {head_dim} must be at least {RGROUP_QUARTERS * QK_CHUNK_ELEMS}"
    N_SUBCHUNKS = QKHE_LOOP * (QK_CHUNK_ELEMS // 8)

    # Q-quant chunk width: a SEPARATE, independent axis from the QK
    # contraction chunking above. NQCHUNK (the number of QCHUNK-wide slices
    # per row) must stay fixed at 16 -- it's tied to `lane16`'s role as the
    # width of the per-row absmax butterfly reduction (itself MFMA_MNK-tied),
    # not to head_dim -- so QCHUNK is what scales with head_dim instead.
    NQCHUNK = 16
    QCHUNK = HEAD // NQCHUNK  # f16 elements per lane's load chunk (8 for HEAD=128, 4 for HEAD=64)

    # PV output-dim chunking: VHE_CHUNKS * NWARP * MFMA_MNK covers HEAD (like
    # RGROUP_QUARTERS above, but for PV's N-dimension instead of QK's
    # K-dimension), so VHE_CHUNKS scales with head_dim while NWARP/MFMA_MNK
    # stay architecture-fixed.
    VHE_CHUNKS = HEAD // (NWARP * MFMA_MNK)  # 2 for HEAD=128, 1 for HEAD=64

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
        key_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, HEAD//16, block_size, 16] (blocked, see module docstring)
        value_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, block_size//16, HEAD, 16] (blocked, see module docstring)
        block_tables_ptr: fx.Tensor,  # [num_seqs, max_blocks_per_seq]
        context_lengths_ptr: fx.Tensor,  # [num_seqs]
        key_scale_ptr: fx.Tensor,  # [1] per-tensor fp8 K dequant scale
        value_scale_ptr: fx.Tensor,  # [1] per-tensor fp8 V dequant scale
        max_blocks_per_seq: Int32,
        num_q_heads: Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = tid // WAVE  # 0..NWARP-1
        lane = tid - warp * WAVE  # 0..63
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        part = fx.Int32(gpu.block_id("z"))  # context partition handled by this CTA
        n_kv = num_q_heads // GS  # num_kv_heads

        # EXPERIMENTAL: fx.copy-based K/V/Q/context_len loaders (see
        # fp8_gemm_utils.py's G2SLoader/StoreC pattern), indexed by the same
        # raw byte/element offset the raw-pointer loaders already compute
        # (fp8 is 1B/elem, so byte offset == element index).
        # `logical_divide(..., make_layout(1, 1))` + `slice` only picks the
        # start element; the copy atom's own width determines how much
        # actually gets copied per call.
        #
        # K/V/context_len use UniversalCopy128b/32b over a plain raw
        # addrspace(1) pointer (same `llvm.load` CopyOpUniversalCopyType
        # emits, matching the global_load_i64x2/i32 this replaces) -- no
        # buffer-resource descriptor. Q keeps the buffer-resource
        # BufferCopy128b path. Which of the two `copy_op` is (a
        # CopyOpCDNA3BufferCopyType or a plain CopyOpUniversalCopyType)
        # decides whether the source needs a buffer-resource descriptor.
        def _make_flat_loader(tensor_ptr, elem_ty, reg_width, copy_op):
            use_buffer_resource = isinstance(copy_op, fx.rocdl.CopyOpCDNA3BufferCopyType)
            copy_atom = fx.make_copy_atom(copy_op, elem_ty)
            reg = fx.make_rmem_tensor(fx.make_layout(reg_width, 1), elem_ty)
            base_iter = (
                fx.get_iter(fx.rocdl.make_buffer_tensor(tensor_ptr, max_size=True))
                if use_buffer_resource
                else fx.get_iter(tensor_ptr)
            )
            flat = fx.Tensor(fx.make_view(base_iter, fx.make_layout(1 << 30, 1)))
            div = fx.logical_divide(flat, fx.make_layout(1, 1))

            def _load(elem_idx):
                fx.copy(copy_atom, fx.slice(div, (None, elem_idx)), reg)
                return fx.Vector(fx.memref_load_vec(reg))

            return _load

        _k_load_fp8x16 = _make_flat_loader(key_cache_ptr, FP8, 16, fx.UniversalCopy128b())
        _v_load_fp8x16 = _make_flat_loader(value_cache_ptr, FP8, 16, fx.UniversalCopy128b())
        # QCHUNK 16-bit float (f16 or bf16, per Q_DTYPE) elements = QCHUNK*16
        # bits per lane's load (128 bits for QCHUNK=8 at HEAD=128, 64 bits for
        # QCHUNK=4 at HEAD=64).
        _q_copy_op = fx.rocdl.BufferCopy128b() if QCHUNK == 8 else fx.rocdl.BufferCopy64b()
        _q_load_chunk = _make_flat_loader(query_ptr, Q_DTYPE, QCHUNK, _q_copy_op)
        _ctxlen_load = _make_flat_loader(context_lengths_ptr, fx.Int32, 1, fx.rocdl.BufferCopy32b())

        def _k_load16(byte_off):
            return _k_load_fp8x16(byte_off).bitcast(fx.Int64)

        def _v_load16(byte_off):
            return _v_load_fp8x16(byte_off).bitcast(fx.Int64)

        context_len = fx.Int32(_ctxlen_load(seq)[0])
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        # Per-tensor K/V scales are uniform across the whole launch (same
        # address for every lane) -- route through a scalar buffer load so it
        # lands once in an SGPR and broadcasts for free, like
        # `_load_phys_scalar`'s block-table lookup below, and matching how
        # `pa_decode_ps_kernel` reads its own per-tensor `key_scale`/
        # `value_scale` tensors.
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        key_scale = fx.Int32(
            buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True)
        ).bitcast(fx.Float32)
        value_scale = fx.Int32(
            buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True)
        ).bitcast(fx.Float32)

        # This CTA only walks its partition's slice of the TILE_TOK-blocks, so the
        # context is parallelized across grid.z CTAs (more CUs for low batch).
        # Computed here (pure arithmetic on context_len/part, no memory access,
        # no dependency on anything below) so the K prefetch that needs it can
        # also be hoisted before the Q-quantization barrier -- see the K-ops
        # section below for why.
        num_tiles = cdiv(context_len, TILE_TOK)
        tiles_per_part = cdiv(num_tiles, NP)
        part_start = part * tiles_per_part
        part_end_raw = part_start + tiles_per_part
        part_end = arith.select(part_end_raw < num_tiles, part_end_raw, num_tiles)
        loop_start = fx.Index(part_start)
        loop_end = fx.Index(part_end)

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

        c16 = 16
        lane16 = lane - (lane // c16) * c16  # 0..15: this row's head-dim chunk index
        rgroup = lane // c16  # 0..3: which quarter-wave (paired with warp -> query row)

        TOK_CHUNK = NWARP * MFMA_MNK  # 64
        NCHUNK = TILE_TOK // TOK_CHUNK  # 4

        # A compute tile always starts exactly on a page boundary: TILE_TOK
        # (256) is a multiple of block_size for both supported values (16, 64),
        # so there is no "within-page" remainder to track (unlike the old
        # design, which supported block_size >= TILE_TOK and needed a
        # within-tile sub-page index).
        def _tile_tok0_and_page(tt_i32):
            tok0 = tt_i32 * TILE_TOK
            return tok0, tok0 // block_size

        def _load_phys_scalar(page, vec_width=1):
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
            # `_v_page_fetch_and_stage` below, which reuses this same scalar
            # mechanism (vec_width=PAGES_PER_CHUNK) to *fetch* (not consume)
            # V's pages, one warp per `rgroup` row, broadcasting the result to
            # the other warps via LDS.
            result = buffer_ops.buffer_load(
                bt_rsrc, seq * max_blocks_per_seq + page, vec_width=vec_width, is_scalar=True
            )
            return fx.Int32(result) if vec_width == 1 else result

        def _v_page_fetch_and_stage(tt_i32):
            # V's page depends on `rgroup`, which is shared across all 4 warps
            # (the P/V transpose means every warp needs the same 256-token V
            # range for its own head-dim slice) -- so instead of each warp
            # redundantly re-deriving all PAGES_PER_CHUNK pages itself, warp
            # `w` fetches (only) the row for `rgroup == w` via one scalar,
            # wave-uniform wide load (`_load_phys_scalar` with
            # vec_width=PAGES_PER_CHUNK instead of 1), and broadcasts it to
            # LDS for every warp to read back (see `_v_page_read_row`). This
            # is prefetched one tile ahead, with the store issued before --
            # and the read-back after -- an already-existing barrier (see the
            # main loop), so no new barrier is added for it.
            _, base_page = _tile_tok0_and_page(tt_i32)
            fetched = _load_phys_scalar(base_page + warp * PAGES_PER_CHUNK, PAGES_PER_CHUNK)
            if lane == 0:
                # buffer_load(vec_width=1) returns a bare scalar, not a Vector --
                # wrap it. vec_width>1 already returns a Vector directly.
                fetched_vec = (
                    fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
                    if const_expr(PAGES_PER_CHUNK == 1)
                    else fx.Vector(fetched)
                )
                _view(
                    sVPage_off + warp * (PAGES_PER_CHUNK * 4),
                    fx.Int32,
                    fx.make_layout(PAGES_PER_CHUNK, 1),
                    4,
                ).store(fetched_vec)

        def _v_page_read_row():
            off = sVPage_off + rgroup * (PAGES_PER_CHUNK * 4)
            row = _view(off, fx.Int32, fx.make_layout(PAGES_PER_CHUNK, 1), 4).load()
            return [row[sub] for sub in range_constexpr(PAGES_PER_CHUNK)]

        # ── raw dwordx4 K load (A operand), pa_decode_ps_kernel's layout ──
        # Lane (lane16, rgroup) feeds A row m=lane16 = token (a*64 + warp*16 +
        # lane16). head_dim is chunked in two levels (see QKHE_LOOP's
        # comment): a fixed 16-element chunk index `qkhe*RGROUP_QUARTERS +
        # rgroup` (`rgroup` plays exactly production's `rowid` role here),
        # each chunk loaded as one dwordx4 (16B = QK_CHUNK_ELEMS fp8) --
        # SAME formula pa_decode_ps_kernel's `_pa_small_block_load_k_flat`
        # uses (`k_he_off_dw = rowid*c_he_stride_dw + qkhe*4*c_he_stride_dw`).
        #
        # key_cache_ptr uses pa_decode_ps_kernel's own BLOCKED layout, NOT the
        # plain PA [num_blocks,num_kv_heads,block_size,HEAD] layout:
        # [num_blocks, num_kv_heads, HEAD // 16, block_size, 16] (fp8, 1B/elem)
        # -- the 16-byte head-chunk is the innermost/contiguous run per token,
        # and consecutive tokens for a FIXED head-chunk are 16 bytes apart.
        # This exists because the plain layout puts head_dim (HEAD bytes)
        # innermost, so adjacent lanes (which own adjacent TOKENS, not
        # adjacent head-dim slices, per the MFMA-fixed lane roles below) land
        # HEAD bytes apart per global_load_i64x2 -- confirmed via ATT trace +
        # address-pattern analysis to be the dominant stall (~58% of all
        # cycles) from poor cross-lane coalescing. Re-laying the axes so the
        # head-chunk is outermost and token is next-innermost makes adjacent
        # lanes (adjacent tokens, fixed head-chunk) exactly 16 bytes apart --
        # contiguous, coalesced multi-lane loads.
        #
        # block_size < TILE_TOK means a tile's 64-token warp-chunk `a` can span
        # multiple pages: `local_tok = warp*16+lane16` (0..63) decomposes into
        # `page = base_page + a*PAGES_PER_CHUNK + local_tok//block_size` and
        # `within_page_tok = local_tok % block_size`. At block_size=64,
        # local_tok//64 is always 0 (page depends only on `a`, shared by the
        # whole warp); at block_size=16, local_tok//16 == warp (page depends on
        # (a, warp), shared by all 64 lanes of that warp) -- either way this is
        # one `_load_phys` per (thread's own) warp per `a`, not per-lane.
        # == HEAD // 16 (same formula as QCHUNK -- reused rather than recomputed).
        c_nhgroup = QCHUNK  # total 16-element head-chunks in the cache layout
        local_tok = warp * c16 + lane16  # 0..63: this thread's token within a 64-token chunk

        def _k_page(base_page, a):
            return base_page + (a * PAGES_PER_CHUNK) + local_tok // block_size

        def _k_ops(phys):
            within_page_tok = local_tok % block_size
            ops = []
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                base = (((phys * n_kv + kv_h) * c_nhgroup + he_idx) * block_size + within_page_tok) * QK_CHUNK_ELEMS
                w = _k_load16(base)  # head[he_idx*16 : +16] -> k_step 2*qkhe, 2*qkhe+1
                ops.extend([w[0], w[1]])
            return ops  # N_SUBCHUNKS i64 operands

        # All NCHUNK chunks' K as one flat (NCHUNK*N_SUBCHUNKS,) i64 vector —
        # carried as a SINGLE loop-carried value through the scf.for iter_args
        # (instead of NCHUNK*N_SUBCHUNKS individual scalars) so tile tt+1's K
        # prefetch overlaps tt's softmax + PV (cross-iteration pipelining,
        # like pa_decode_ps_kernel).
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
            #
            # At block_size=64 (PAGES_PER_CHUNK=1), `_k_page(base_page, a) ==
            # base_page + a` for every `a` -- NCHUNK consecutive block-table
            # entries -- so one wide `vec_width=NCHUNK` scalar load fetches all
            # of them in a single s_buffer_load_dwordx4 instead of NCHUNK
            # separate s_buffer_load_dword instructions (confirmed via LLVM
            # IR), matching pa_decode_ps_kernel's own
            # `_pa_small_block_stage_phys_blocks`, which does the same
            # single-wide-load batching. At block_size=16 (PAGES_PER_CHUNK=4),
            # `_k_page`'s per-`a` addresses are PAGES_PER_CHUNK apart (not
            # consecutive), so they can't be folded into one contiguous wide
            # load this way; keep the per-`a` loop there.
            if const_expr(PAGES_PER_CHUNK == 1):
                phys_vec = fx.Vector(_load_phys_scalar(base_page, NCHUNK))
                phys_list = [fx.Int32(phys_vec[a]) for a in range_constexpr(NCHUNK)]
            else:
                phys_list = [_load_phys_scalar(_k_page(base_page, a)) for a in range_constexpr(NCHUNK)]
            flat = []
            for a in range_constexpr(NCHUNK):
                flat.extend(_k_ops(phys_list[a]))
            return fx.Vector.from_elements(flat, dtype=fx.Int64)

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
        num_tiles_m1 = num_tiles - 1
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
            off = byte_off + m_idx * (width * esz)
            return _view(off, elem_ty, fx.make_layout(width, 1), esz)

        def _ld1(byte_off, m_idx):
            return _row(byte_off, m_idx, 1, fx.Float32, 4).load()[0]

        def _st1(byte_off, m_idx, val):
            _row(byte_off, m_idx, 1, fx.Float32, 4).store(fx.Vector.from_elements([val], dtype=fx.Float32))

        # f32[16, NWARP] cross-warp scratch (row stride padded to NWARP_PAD to
        # avoid the 2-way LDS bank conflict -- see `sLmax_off`'s comment):
        # scalar write at (row, warp), vec read of a row's NWARP valid slots.
        def _st_lw(base_off, row, w, val):
            off = base_off + (row * NWARP_PAD + w) * 4
            _view(off, fx.Float32, fx.make_layout(1, 1), 4).store(fx.Vector.from_elements([val], dtype=fx.Float32))

        def _ld_lw_row(base_off, row):
            off = base_off + row * (NWARP_PAD * 4)
            return _view(off, fx.Float32, fx.make_layout(NWARP, 1), 4).load()

        def _f32_to_fp8_words(vf32):
            # f32 -> fp8 must use the hw cvt (arith.truncf to fp8 is not lowerable);
            # pack 4 f32 -> 1 i32 (4 fp8) via two cvt_pk_fp8_f32 calls. Returns the
            # i32 words so the result can be stored to LDS as plain i32.
            n = vf32.shape[0]
            words = []
            for i in range_constexpr(n // 4):
                b = i * 4
                lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b], vf32[b + 1], 0, False)
                words.append(fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b + 2], vf32[b + 3], lo, True))
            return fx.Vector.from_elements(words, dtype=fx.Int32)

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
        # QCHUNK-element head-dim slice, so every thread loads,
        # converts, and packs exactly one chunk -- matching pa_decode_ps_kernel's
        # `_finish_q_fragments` lane-per-chunk layout. The row's absmax is then a
        # butterfly reduction over the 16 lanes sharing (warp, rgroup) via
        # shuffle_xor(width=16), with no LDS/barrier needed for the reduction
        # itself (only the store below needs the barrier after it). This
        # replaces the old design where a single lane serially handled a whole
        # row's 16 chunks while the other ~240 threads idled at the barrier --
        # that idle wait was confirmed via ATT trace to cost ~7-8% of all
        # kernel stall cycles at bs=128/ctx=16384.

        qh_local = warp * 4 + rgroup  # 0..15: this thread's query row

        if qh_local < GS:
            qh0 = kv_h * GS + qh_local
            row_byte0 = (seq * num_q_heads + qh0) * (HEAD * 2)  # 16-bit float = 2B/elem
            chunk_off = row_byte0 + lane16 * (QCHUNK * 2)
            q_chunk = _q_load_chunk(chunk_off // 2)  # byte offset -> element index

            # local absmax over this thread's own QCHUNK elements (kept in
            # Q_DTYPE: widening to f32 is monotonic for finite values in both
            # f16 and bf16, so comparing at the native width and only
            # widening the scalar result to f32 avoids feeding a full-vector
            # fpext into a reduce, which would otherwise force the backend to
            # scalarize into per-element conversions instead of a packed one
            # -- for f16 specifically, packed v_cvt_pk_f32_f16 vs scalarized
            # v_cvt_f32_f16_sdwa; bf16->f32 widening is even cheaper, a plain
            # left-shift, so this structure is at worst neutral there), then a
            # butterfly reduce over the 16 lanes owning this same row (fixed
            # warp/rgroup, lane16 varies).
            #
            # The cross-lane reduce uses `dpp_utils.dpp_xor_f32` (raw
            # `llvm.amdgcn.update.dpp.i32`), matching pa_decode_ps_kernel's own
            # analogous per-row Q-absmax reduction exactly (see
            # `_finish_q_fragments` in pa_decode_fp8.py) -- not the DSL's
            # generic `shuffle_xor`. DPP executes in the VALU crossbar with no
            # LDS/DS-unit involvement at all, one level cheaper than even the
            # `ds_swizzle` path `shuffle_xor(sh, WAVE)` gets for a 16-lane XOR
            # (confirmed via LLVM IR: `shuffle_xor` at width=c16 emits 4x
            # mbcnt+ds_bpermute; at width=WAVE it emits ds_swizzle; dpp_xor_f32
            # emits update.dpp directly, no ds_swizzle/ds_bpermute/mbcnt at all).
            local_absmax = fmath.absf(q_chunk).reduce(ReductionOp.MAX)
            absmax = local_absmax.to(fx.Float32)
            for sh in (8, 4, 2, 1):
                absmax = absmax.maximumf(dpp_utils.dpp_xor_f32(absmax, sh))
            # per-row symmetric fp8 quantization: q_scale = absmax / FP8_MAX.
            # HW reciprocal approximation (rcp_f32, same helper
            # pa_decode_ps_kernel uses for this exact quantity -- see
            # `inv_query_scale` in `_finish_q_fragments`) instead of a full
            # IEEE division: its ULP-level error is negligible next to the
            # fp8 quantization noise this kernel already carries.
            q_scale = absmax * fx.Float32(1.0 / FP8_MAX)
            inv = fx.Float32(rcp_f32(q_scale.maximumf(fx.Float32(1e-20))))
            inv_b = fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(QCHUNK)

            q_scaled_chunk = q_chunk.to(fx.Float32) * inv_b
            _st_words(
                sQ_off + qh_local * HEAD + lane16 * QCHUNK,
                _f32_to_fp8_words(q_scaled_chunk),
            )
            if lane16 == 0:
                _st1(sQscale_off, qh_local, q_scale)
        else:
            _st_words(
                sQ_off + qh_local * HEAD + lane16 * QCHUNK,
                fx.Vector.filled(QCHUNK // 4, 0, fx.Int32),
            )
            if lane16 == 0:
                _st1(sQscale_off, qh_local, ZERO_F)

        # ── init running softmax state ──
        # The output accumulator O is register-resident (loop-carried); the
        # running max `m` and running denominator `l` are ALSO both
        # register-resident now (loop-carried per-thread, see
        # `m_prev`/`m_new` and `l_prev`/`l_new` in the main loop below) --
        # both are read/written via `tid`-indexing inside the loop and later
        # via a DIFFERENT `row_e`-indexing in the epilogue, but every thread
        # already holds its own cross-warp-combined value after computing it
        # each tile, so the per-tile LDS store+load is unneeded; only a
        # single post-loop bridge store into `sM`/`sL` remains, for the
        # epilogue's differently-indexed threads to read (see the post-loop
        # bridge-write comments below).
        gpu.barrier()

        # First tile's V page-index row, now visible after the barrier above
        # (the fetch+LDS-store was issued earlier, alongside k_pf0).
        v_page_pf0 = _v_page_read_row()

        # Q is the B operand (D[token,qhead] = K·Qᵀ), read raw from sQ as fp8 i64
        # operands for the raw-MFMA QK below (replicated across warps, constant
        # across tiles → read once, held in registers).  Lane (lane16, rgroup)
        # feeds MMA column n=lane16 (qhead); this MUST use the exact same
        # (qkhe, rgroup, qkr) -> head_dim permutation as K's `_k_ops` (same
        # formula pa_decode_ps_kernel's `_finish_q_fragments` derives for its
        # own Q reader: "thread (rowid R, lane16id L) consumes, for
        # k_step = qkhe*2+qkr, Q[head=L][hd=(qkhe*4+R)*16+qkr*8+0..7]").
        q_ops = []
        for qkhe in range_constexpr(QKHE_LOOP):
            he_idx = qkhe * RGROUP_QUARTERS + rgroup
            chunk = _view(sQ_off + lane16 * HEAD + he_idx * QK_CHUNK_ELEMS, fx.Int64, fx.make_layout(2, 1), 8).load()
            q_ops.extend([chunk[0], chunk[1]])
        # q_ops[s] for s=0..N_SUBCHUNKS-1, s = qkhe*2+qkr, = head[he_idx*16+qkr*8 : +8] of qhead=lane16

        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tcopy_o = fx.make_tiled_copy_C(copy_c, tiled_mma_pv)  # PV out -> sO (epilogue)

        # QK in TLOOP chunks of TOK_CHUNK tokens: each small fx.gemm yields a f32x4
        # C-fragment, so the softmax processes 4 scores at a time (scores stay in
        # AGPR, VGPR peak low) — matching pa_decode_ps_kernel's TLOOP.
        # compile-time per-chunk token offsets (token = a*TOK_CHUNK + base + r)
        _ct = [
            fx.Vector.from_elements([float(a * TOK_CHUNK + r) for r in range_constexpr(4)])
            for a in range_constexpr(NCHUNK)
        ]
        # P·V is loop-tiled over head-dim (like the production VHELOOP): each step
        # computes O[:, vh*VHE_SIZE : +VHE_SIZE], shrinking the live V operand and
        # PV accumulator instead of materializing the full [16, HEAD] at once.
        VHE_SIZE = HEAD // VHE_CHUNKS
        tmpl_Op = fx.make_rmem_tensor(fx.make_layout((M, VHE_SIZE), (VHE_SIZE, 1)), fx.Float32)
        OP_ELEMS = M * VHE_SIZE // (NWARP * WAVE)  # PV C-fragment elements/lane/chunk (probed = 4)

        # ── raw dwordx4 V load (B operand), pa_decode_ps_kernel's layout ──
        # PV contracts over token, so (like QK's head permutation) the token→k_step
        # mapping is free as long as V and P (p_ops) agree: lane (rgroup) takes the
        # contiguous token slice [rgroup*64 : +64] for its head (vh*VHE_SIZE +
        # warp*16 + lane16), loaded as 4× i64x2 (128-bit) = 8 k_step operands.
        #
        # value_cache_ptr uses pa_decode_ps_kernel's own "trans_v" BLOCKED
        # layout (same formula `_pa_small_block_load_v_trans` uses):
        # [num_blocks, num_kv_heads, block_size // 16, head_dim, 16] (fp8,
        # 1B/elem) -- 16 CONSECUTIVE TOKENS innermost (V is token-vectorized,
        # not head-dim-vectorized like K), head_dim in the middle, and the
        # block_size//16 token-subblock index outermost. head_group = vh*4+warp
        # needs no runtime div/mod (VHE_SIZE=64 and 16 both divide warp*16 and
        # vh*VHE_SIZE evenly).
        #
        # A rgroup's 64-contiguous-token PV operand run can itself span multiple
        # block_size-sized pages: outer loop `sub` (0..PAGES_PER_CHUNK-1) picks
        # the page, inner loop `step` (0..block_size//16-1) walks that page's own
        # block_size-token run in 16-token (one i64x2) increments -- `step` is
        # exactly production's token-subblock index. This collapses to today's
        # single-page/4-step behavior at block_size=64 and becomes 4 pages/1
        # step at block_size=16 -- both total NVOPS=8 i64 operands.
        NVOPS = TILE_TOK // MFMA_K  # 8 PV k_steps (256 tokens / K=32)
        STEPS_PER_PAGE = block_size // 16

        # V's page depends only on (rgroup, sub) -- NOT on `warp` -- so all 4
        # warps (256 threads) want the exact same V_PAGE_COUNT pages every
        # tile. See `_v_page_fetch_and_stage`/`_v_page_read_row` (above, near
        # `_load_phys_scalar`) for how the page *index* is fetched once
        # (scalar, warp-partitioned, LDS-broadcast, prefetched one tile ahead
        # reusing an existing barrier) instead of redundantly re-derived per
        # lane every tile.
        def _v_ops(phys_row, vh):
            head_group = ((vh * VHE_SIZE) // 16) + warp
            head_element = head_group * 16 + lane16
            ops = []
            for sub in range_constexpr(PAGES_PER_CHUNK):
                for step in range_constexpr(STEPS_PER_PAGE):
                    base = (((phys_row[sub] * n_kv + kv_h) * STEPS_PER_PAGE + step) * HEAD + head_element) * 16
                    w = _v_load16(base)
                    ops.extend([w[0], w[1]])
            return ops  # NVOPS i64, the 64-token contiguous run for this head

        # O is carried in registers across tiles (one VHE_CHUNKS-list of PV
        # C-fragment vectors), rescaled by the softmax correction each tile.
        # `m` (the running row max) is ALSO carried in registers now -- every
        # thread already computes the cross-warp-combined max for its own
        # `qh` each tile (see `m_new` below), so carrying it forward directly
        # eliminates a per-tile LDS store+load (see the init comment above).
        o_zero = fx.Vector.filled(OP_ELEMS, 0.0, fx.Float32)
        for tt, ostate in range(
            loop_start, loop_end, arith.index(1), init=[o_zero, o_zero, k_pf0, *v_page_pf0, NEG_INF, ZERO_F]
        ):
            o_acc = [ostate[0], ostate[1]]
            k_cur = ostate[2]  # this tile's prefetched K, as one (NCHUNK*N_SUBCHUNKS,) i64 vector
            v_page_cur = [ostate[3 + i] for i in range_constexpr(PAGES_PER_CHUNK)]  # this tile's V pages
            m_prev = ostate[3 + PAGES_PER_CHUNK]  # this thread's own running max, carried from last tile
            l_prev = ostate[4 + PAGES_PER_CHUNK]  # this thread's own running denom, carried from last tile
            tt_i32 = fx.Int32(tt)
            tok0, _ = _tile_tok0_and_page(tt_i32)

            # ---- QK in TLOOP chunks: NCHUNK raw MFMAs -> f32x4/lane ----
            # Each chunk accumulates the N_SUBCHUNKS head-quarter k_steps (this
            # tile's prefetched k_cur) into one f32x4 C-fragment (D[token, qhead]);
            # the raw dwordx4 K feeds the same MFMA the old fx.gemm wrapped, so the
            # C layout — and thus the softmax / P-pack / PV below — is unchanged.
            frag_Ss = []
            for a in range_constexpr(NCHUNK):
                acc = arith.constant_vector(0.0, T.f32x4)
                for s in range_constexpr(N_SUBCHUNKS):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [k_cur[a * N_SUBCHUNKS + s], q_ops[s], acc, 0, 0, 0]
                    )
                frag_Ss.append(fx.Vector(acc))

            # prefetch next tile's K (the MFMAs above consumed k_cur); the loads
            # overlap the softmax + PV below and become next iter's state. At
            # these small block sizes the backing page(s) almost always change
            # tile-to-tile, so (unlike the old >=TILE_TOK block_size design)
            # there is no same-page case worth special-casing -- every tile
            # re-derives its own page(s) fresh via `_k_ops_flat`.
            #
            # (Issuing this before the QK MFMAs instead -- so the load overlaps
            # QK too, not just softmax+PV -- was tried: `_k_ops_flat` has no
            # dependency on the QK MFMAs' output, so it's legal, but it
            # extended k_next's live range back across the whole QK MFMA loop.
            # The compiler responded by flipping to AGPR-form MFMA with far
            # higher combined register pressure (VGPR 16 + AGPR 128 = 144 vs.
            # 112 here), dropping occupancy 4 -> 3 waves/SIMD, and measured
            # ~9.5% *slower* end-to-end despite the extra hiding window --
            # same failure mode as V's analogous attempt below. Reverted.)
            #
            # (V is deliberately NOT carried the same way: an earlier attempt
            # at prefetching v_next one tile ahead, mirroring k_next, pushed
            # VGPR usage up ~20% -- 136 -> 164 -- and measured *slower* despite
            # the extra hiding time. V's own load is now deferred to right
            # before its PV use instead -- see the PV loop below -- which
            # cuts peak VGPR far more (132 -> 112, crossing the 512//132==3
            # -> 512//112==4 waves/SIMD occupancy boundary) and measured
            # ~10% *faster* despite losing the QK+softmax hiding window V's
            # raw loads used to get from being hoisted up here.)
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
            # This DSL's dynamic-`if` variable-reassignment tracking requires
            # each reassigned state variable to resolve to a single
            # MLIR-backed value; k_cur/k_next are now one
            # (NCHUNK*N_SUBCHUNKS,) i64 Vector (a single such value) instead
            # of NCHUNK*N_SUBCHUNKS individual scalars, so a plain
            # reassignment works directly -- no per-element unpack.
            tt1 = tt_i32 + 1
            k_next = k_cur
            if tt1 < part_end:
                k_next = _k_ops_flat(tt1)

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
            c16 = 16
            qh = lane - (lane // c16) * c16  # qhead = lane % 16
            l16g = lane // c16  # 0..3 lane-group within the warp
            scale = scale_qk * _ld1(sQscale_off, qh)  # per-qhead positive score scale
            n_valid_tile = (context_len - tok0).to(fx.Float32)
            base_tok_f = fx.Int32(warp * c16 + l16g * 4).to(fx.Float32)
            thr = fx.Vector.from_elements([n_valid_tile - base_tok_f], dtype=fx.Float32).broadcast_to(4)
            neg4 = fx.Vector.filled(4, float("-inf"), fx.Float32)

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
                pm = pm.maximumf(pm.shuffle_xor(sh, WAVE))
            # pa_decode_ps_kernel's equivalent store (`_cross_warp_softmax_and_prob_pack`)
            # is unconditional -- all 4 lanes sharing this qhead (l16g=0..3) already
            # hold the identical post-shuffle_xor `pm`, so this is a harmless
            # same-value redundant write, not a race.
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
            m_old = m_prev
            m_new = m_old.maximumf(_ld_lw_row(sLmax_off, qh).reduce(ReductionOp.MAX))
            m_new_b = fx.Vector.from_elements([m_new], dtype=fx.Float32).broadcast_to(4)
            ls = fx.Float32(0.0)
            # raw i32-word store straight to sP[qhead][token_base:+4] (fp8, 1B/elem):
            # the packed word's 4 fp8 lanes are exactly the 4 consecutive tokens this
            # lane owns in chunk `a` (token_base = a*TOK_CHUNK + warp*16 + l16g*4), so
            # a direct store replaces the make_fragment_C/tiled_copy_C round trip.
            base4 = 4
            words = []
            for a in range_constexpr(NCHUNK):
                # HW exp2 intrinsic (exp2_f32_fast) instead of MLIR's generic
                # math.exp2 (a polynomial approximation whose edge-case handling
                # costs ~32 extra v_cndmask here, measured via ISA source-line
                # attribution) -- matches ps's own exp2_f32_fast usage.
                Pa = fx.Vector(exp2_f32_fast(masked_chunks[a] * scale - m_new_b))
                ls = ls + Pa.reduce(ReductionOp.ADD)
                words.append(_f32_to_fp8_words(Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32))[0])
            # One strided vector store (stride = TOK_CHUNK bytes = TOK_CHUNK/4
            # Int32 elements, matching consecutive `a`'s token_base spacing)
            # instead of NCHUNK separate 4-byte stores -- the backend packs
            # the 4 logical stores into 2 `ds_write2_b32` instructions instead
            # of 3 (1 solo + 1 paired + 1 solo) for the old per-`a` stores,
            # with no VGPR cost (confirmed via ISA dump: 132 either way).
            p_off0 = sP_off + qh * TILE_TOK + warp * c16 + l16g * base4
            _view(p_off0, fx.Int32, fx.make_layout(NCHUNK, TOK_CHUNK // 4), 4).store(
                fx.Vector.from_elements(words, dtype=fx.Int32)
            )
            for sh in (16, 32):
                ls = ls + ls.shuffle_xor(sh, WAVE)
            if l16g == 0:
                _st_lw(sLsum_off, qh, warp, ls)
                if warp == 0:
                    _st1(sCorr_off, qh, fx.Float32(exp2_amdgcn_scalar(m_old - m_new)))
            gpu.barrier()

            # phase 3: merge per-warp sums into the running denominator. `tid <
            # M` is exactly the `(l16g == 0 and warp == 0)` thread set above,
            # so this thread's own correction factor is already sitting in
            # its `m_old`/`m_new` registers from earlier this tile -- no need
            # to read the `sCorr_off` this same thread just wrote, or the
            # `sL_off` this same thread wrote last tile; both round-trip
            # through registers instead (`l_new` below feeds next tile's
            # `l_prev`, mirroring `m_new`/`m_prev`).
            l_new = l_prev
            if tid < M:
                gsum = _ld_lw_row(sLsum_off, tid).reduce(ReductionOp.ADD)
                corr_reg = fx.Float32(exp2_amdgcn_scalar(m_old - m_new))
                l_new = l_prev * corr_reg + gsum

            # ---- read P back as the A operand for P·V, raw (replicated across
            # warps) — lane reads sP[qhead=lane16][token rgroup*64:+64] as NVOPS i64,
            # the same permuted token slice v_ops uses so the raw PV MMA matches. ----
            p_ops = _view(
                sP_off + lane16 * TILE_TOK + rgroup * 64,
                fx.Int64,
                fx.make_layout(NVOPS, 1),
                8,
            ).load()

            # ---- PV with register-resident O accumulate (no LDS round-trip) ----
            # O_new = O_old * corr + P·V per head-dim chunk; corr = exp2(m_old-m_new)
            # is per-row.  Probed PV C-fragment: vec element v of lane L holds row
            # m = (L%64//16)*4 + v, so corr_s[v] = corr[m_base + v].  Done element-
            # wise (fx.Vector*fx.Vector broadcasts to an outer product here).  No barrier after
            # PV: O is in registers and the next iter's QK/phase2 barriers order
            # any sS/sP reuse (sOp is gone).  Raw PV MMA: NVOPS k_steps accumulate
            # into one f32x4 (this warp's [16 qhead, 16 head] output atom).
            m_base_pv = (lane // c16) * 4
            # OP_ELEMS contiguous f32 (indices m_base_pv..+OP_ELEMS-1) in one
            # vectorized LDS read instead of OP_ELEMS separate scalar reads.
            corr_off = sCorr_off + m_base_pv * 4
            corr_vec = _view(corr_off, fx.Float32, fx.make_layout(OP_ELEMS, 1), 4).load()
            corr_s = [corr_vec[v] for v in range_constexpr(OP_ELEMS)]
            for vh in range_constexpr(VHE_CHUNKS):
                # Load THIS vh's V data right before its own MFMA use instead
                # of hoisting the full VHE_CHUNKS set before QK: only NVOPS
                # i64 (one vh's worth) needs to be live at a time instead of
                # VHE_CHUNKS*NVOPS simultaneously, trading some of the
                # QK+softmax latency-hiding window for lower peak VGPR --
                # worth trying since VGPR is the kernel's binding occupancy
                # constraint (512//132 == 512//128 + 1 wave/SIMD boundary is
                # close by).
                v_vh = _v_ops(v_page_cur, vh)
                acc = arith.constant_vector(0.0, T.f32x4)
                for s in range_constexpr(NVOPS):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [p_ops[s], v_vh[s], acc, 0, 0, 0])
                op = fx.Vector(acc)
                oo = o_acc[vh]
                o_acc[vh] = fx.Vector.from_elements(
                    [oo[v] * corr_s[v] + op[v] for v in range_constexpr(OP_ELEMS)], dtype=fx.Float32
                )
            results = yield [o_acc[0], o_acc[1], k_next, *v_page_next, m_new, l_new]
        o_final = results
        m_final = o_final[3 + PAGES_PER_CHUNK]
        l_final = o_final[4 + PAGES_PER_CHUNK]

        # One-time bridge write of the final running max/denom from
        # `qh`-indexing (this loop's thread role) to `sM_off`/`sL_off`, so the
        # epilogue's DIFFERENT `row_e`-indexed threads can read them below
        # (`sM_off` only matters for NP>1 -- the NP==1 epilogue never reads
        # it; `sL_off` is read by both). This replaces the old per-tile
        # store+load of `m`/`l` through LDS (see the init comment above the
        # loop) with a single store after the loop. `qh`/`l16g` inside the
        # loop body are scoped to that scf.for region (don't dominate this
        # post-loop use), so recompute them here from `lane`/`c16` (both
        # outer-scope, defined before the loop) -- cheap pure arithmetic, no
        # LDS/memory access.
        qh_post = lane - (lane // c16) * c16
        l16g_post = lane // c16
        if l16g_post == 0 and warp == 0:
            if const_expr(NP > 1):
                _st1(sM_off, qh_post, m_final)
            _st1(sL_off, qh_post, l_final)

        # ── stage the register-resident O accumulator to sO (row-major) so the
        # epilogue can read whole rows and write the output as before ──
        thr_copy_o_e = tcopy_o.get_slice(tid)
        for vh in range_constexpr(VHE_CHUNKS):
            frag_Oe = tiled_mma_pv.get_slice(tid).make_fragment_C(tmpl_Op)
            frag_Oe.store(o_final[vh])
            sO_chunk = _view(
                (sO_off + vh * VHE_SIZE * 4),
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

        c_tpr = THREADS_PER_ROW
        row_e = tid // c_tpr
        sub_e = tid - row_e * c_tpr
        col_e = sub_e * ELEMS_PER_THREAD
        row_off = sO_off + row_e * (HEAD * 4) + col_e * 4
        o_v = _view(row_off, fx.Float32, fx.make_layout(ELEMS_PER_THREAD, 1), 4).load()
        # value_scale and the P dequant (1/FP8_MAX) are true per-CTA constants
        # (this kernel only supports per-tensor K/V scale, unlike
        # pa_decode_ps_kernel's `_pv_mfma`, which folds value_scale into every
        # PV tile because that code path is shared with its per-token_kv mode)
        # -- fold them in exactly once here, before the NP branch, so both
        # paths below write an already-scaled numerator and the reduce kernel
        # (NP>1) needs no value_scale of its own, matching pa_decode_ps_kernel's
        # property that its reduce step only flash-combines partitions.
        o_scale = v_scale_f * fx.Float32(1.0 / FP8_MAX)
        o_v = o_v * fx.Vector.from_elements([o_scale], dtype=fx.Float32).broadcast_to(ELEMS_PER_THREAD)

        if const_expr(NP == 1):
            # single partition: normalize and write the output directly (no
            # partials / reduce round-trip).
            qh = kv_h * GS + row_e
            # HW reciprocal approximation, matching pa_decode_ps_kernel's own
            # softmax-denominator reciprocal (`inv_sum = rcp_f32(safe_sum)`).
            inv_l = fx.Float32(rcp_f32(_ld1(sL_off, row_e)))
            o_out = (o_v * fx.Vector.from_elements([inv_l], dtype=fx.Float32).broadcast_to(ELEMS_PER_THREAD)).to(
                fx.Float16
            )
            # tile-programming store: divide this (seq, qh) row's HEAD axis
            # into ELEMS_PER_THREAD-wide chunks and pick this lane's chunk,
            # instead of manually computing a byte offset for global_store.
            out_row = output_ptr[seq, qh, None]
            out_chunk = fx.slice(fx.logical_divide(out_row, fx.make_layout(ELEMS_PER_THREAD, 1)), (None, sub_e))
            out_chunk.store(o_out)
        else:
            # multi-partition: write this partition's (m_p, l_p, already-scaled
            # numerator O_p); the reduce kernel only flash-combines partitions.
            base = ((seq * n_kv + kv_h) * NP + part) * GS + row_e
            if sub_e == 0:
                pmax_ptr[base] = _ld1(sM_off, row_e)
                psum_ptr[base] = _ld1(sL_off, row_e)
            pout_div = fx.logical_divide(pout_ptr, fx.make_layout(ELEMS_PER_THREAD, 1))
            pout_chunk = fx.slice(pout_div, (None, base * THREADS_PER_ROW + sub_e))
            pout_chunk.store(o_v)

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
        num_q_heads: Int32,
    ):
        d = fx.Int32(gpu.thread_id("x"))  # head-dim element
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        row = fx.Int32(gpu.block_id("z"))  # query row within the kv-head group
        n_kv = num_q_heads // GS
        # element index of (seq, kv_h, partition 0, this row); + p*GS, then *HEAD + d
        base = (seq * n_kv + kv_h) * (NP * GS) + row

        def _exp2s(x):
            return fx.Float32(exp2_amdgcn_scalar(x))

        # pmax/psum are indexed only by (seq, kv_h, row, p) -- the same
        # address for all RED_THREADS=HEAD threads in this CTA, since neither
        # depends on `d`. Reading them via the tensor's per-lane vector load
        # makes every thread redundantly issue its own global load for an
        # address that's uniform across the whole block; route through a
        # scalar buffer load instead so it lands once in an SGPR and
        # broadcasts for free, the same fix applied to the main kernel's
        # block-table lookup (see `_load_phys_scalar`). `pout` is already
        # scaled by value_scale/1/FP8_MAX (folded in once by the main
        # kernel's epilogue), so this reduce step only flash-combines
        # partitions -- no scale of its own, matching pa_decode_ps_kernel's
        # reduce step.
        pmax_rsrc = buffer_ops.create_buffer_resource(pmax_ptr, max_size=True)
        psum_rsrc = buffer_ops.create_buffer_resource(psum_ptr, max_size=True)

        def _ld_scalar_f32(rsrc, idx):
            return fx.Int32(buffer_ops.buffer_load(rsrc, idx, vec_width=1, is_scalar=True)).bitcast(fx.Float32)

        # pass 1: global max over partitions
        gmax = fx.Float32(float("-inf"))
        for p in range_constexpr(NP):
            gmax = gmax.maximumf(_ld_scalar_f32(pmax_rsrc, base + p * GS))
        # pass 2: weighted numerator (this thread's head-dim d) / denominator
        num = fx.Float32(0.0)
        den = fx.Float32(0.0)
        for p in range_constexpr(NP):
            idx = base + p * GS
            w = _exp2s(_ld_scalar_f32(pmax_rsrc, idx) - gmax)
            den = den + _ld_scalar_f32(psum_rsrc, idx) * w
            num = num + pout_ptr[idx * HEAD + d] * w
        qh = kv_h * GS + row
        # HW reciprocal approximation instead of a per-lane division (same
        # rationale as the main kernel's `inv`/`inv_l`).
        output_ptr[seq, qh, d] = (num * fx.Float32(rcp_f32(den))).to(fx.Float16)

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
        key_scale: fx.Tensor,  # [1] per-tensor fp8 K dequant scale
        value_scale: fx.Tensor,  # [1] per-tensor fp8 V dequant scale
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
            pa_decode_tile_reduce_kernel(output, pmax, psum, pout, num_q_heads).launch(
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
    key_scale: float | torch.Tensor,
    value_scale: float | torch.Tensor,
    softmax_scale: float | None = None,
    stream=None,
) -> None:
    """Host entry point. See module docstring for the expected tensor layouts."""
    num_seqs, num_q_heads, head_dim = query.shape
    num_blocks, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape
    # K's blocked layout matches pa_decode_ps_kernel's own K cache layout: a
    # fixed 16-element head-chunk width (see compile_pa_decode_tile's
    # QKHE_LOOP comment).
    assert num_hgroups == head_dim // 16 and hgroup_width == 16
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    _, _, v_subblocks, v_head_dim, v_width = value_cache.shape
    # V's "trans_v" layout matches pa_decode_ps_kernel's own V cache layout:
    # 16-consecutive-token chunks, head_dim unchunked, block_size//16 subblocks.
    assert v_head_dim == head_dim and v_width == 16 and v_subblocks == block_size // 16
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]
    if query.dtype == torch.bfloat16:
        query_dtype = "bf16"
    elif query.dtype == torch.float16:
        query_dtype = "f16"
    else:
        raise ValueError(f"pa_decode_tile only supports f16/bf16 query, got {query.dtype}")

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
    cu_fill_np = cdiv(TARGET_CTAS_PER_CU * device_cus, num_seqs * num_kv_heads)
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
    max_possible_tiles = cdiv(max_blocks_per_seq * block_size, tile_tok)  # no host sync (see below)
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
        query_dtype=query_dtype,
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
    # Per-tensor K/V scales are read in-kernel via buffer_load (matching
    # `pa_decode_ps_kernel`), not baked in as a host scalar kernarg -- so a
    # plain float still works (wrapped into a 1-element tensor here) but a
    # caller-owned device tensor (e.g. computed on-device) is also accepted
    # and passed straight through.
    key_scale_t = key_scale if isinstance(key_scale, torch.Tensor) else torch.tensor([float(key_scale)], device=dev)
    value_scale_t = (
        value_scale if isinstance(value_scale, torch.Tensor) else torch.tensor([float(value_scale)], device=dev)
    )
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
        key_scale_t.to(dtype=torch.float32, device=dev),
        value_scale_t.to(dtype=torch.float32, device=dev),
        int(max_blocks_per_seq),
        int(num_q_heads),
        int(num_seqs),
        int(num_kv_heads),
        s,
    )
