# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

Correctness-first reimplementation of the decode math that production
``pa_decode_ps_kernel`` (``pa_decode_fp8.py``) computes, using FlyDSL's
tile/layout API instead of raw buffer intrinsics and hand-scheduled MFMA.
Trades performance for clarity; scoped to per-tensor fp8 K/V quantization
(``key_scale``/``value_scale``, 1-element device tensors read via a scalar
buffer load), ``query_length == 1``, no kv-varlen, no sliding window.

Like ``pa_decode_ps_kernel``, splits the context across ``grid.z`` partitions
(one CTA each) writing partial (max, sum, numerator) results, combined by a
small reduce kernel -- spreads low-batch/long-context work across more CUs.

fp8: K/V stored as e4m3 FNUZ, fed straight into ``mfma_f32_16x16x32_fp8_fp8``.
Q (bf16/f16) is quantized to fp8 with a per-row scale; softmax probabilities P
are quantized to fp8 for P·V. Scales fold out of the matmuls: ``q_scale`` and
``key_scale`` into the QK score; ``value_scale`` and the P dequant (1/FP8_MAX)
into the epilogue. Softmax max/sum stay f32.

Layouts are simple/logical (NOT production's preshuffle layout). ``block_size``
is a compile-time constant, 16 or 64 only (the K/V gather unrolls a fixed page
fan-out per 256-token compute tile at trace time). ``head_dim`` must be a
multiple of 64 (64 or 128), matching production's own floor.

* ``query``        [num_seqs, num_q_heads, head_dim]                 f16/bf16
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8 e4m3fnuz
                   (SAME layout as ``_pa_small_block_load_k_flat`` in
                    ``pa_decode_fp8.py``: 16-element head-chunk outer, token
                    next-innermost, for coalesced dwordx4 loads -- see
                    ``QKHE_LOOP`` below)
* ``value_cache``  [num_blocks, num_kv_heads, block_size//16, head_dim, 16]  fp8 e4m3fnuz
                   (SAME "trans_v" layout as ``_pa_small_block_load_v_trans``:
                    16 consecutive tokens innermost, unlike K)
* ``block_tables`` [num_seqs, max_blocks_per_seq]                    int32
                   (must cover ceil(context_len/256)*256/block_size pages --
                    rounded UP to the 256-token tile granularity, not just
                    ceil(context_len/block_size), since the last tile always
                    issues a full-span load)
* ``context_lengths`` [num_seqs]                                     int32
* ``output``       [num_seqs, num_q_heads, head_dim]                 f16

Algorithm: one CTA (4 waves/256 threads) per (seq, kv_head) runs a flash-style
online softmax over 256-token compute blocks gathered via ``block_tables``.
The 4 waves split tokens for Q·Kᵀ and split head-dim for P·V; an LDS
round-trip on the probabilities transposes ownership between the two MMAs
(both 16x16x32 fp8, tiled (1,4,1)).

Softmax is distributed across the 4 waves (local max/sum per wave, merged via
LDS scratch into the shared running (m, l)) so all 4 stay busy instead of one
wave doing the whole row.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_size_of
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

    ``block_size``, ``head_dim``, and ``query_dtype`` are all compile-time
    constants, not kernel arguments -- each distinct value gets its own
    compiled kernel via this function's ``lru_cache``.

    ``block_size`` (16 or 64 only) sets the K/V paged-gather's fixed
    block-table page fan-out per compute tile (``PAGES_PER_CHUNK`` below);
    both values divide the 64-token per-warp chunk evenly (one page for 64,
    four for 16).

    ``head_dim`` must be a multiple of 64 (64 or 128) -- same floor as
    pa_decode_ps_kernel's own, since the raw dwordx4-load addressing has a
    64-element minimum granularity.

    ``query_dtype`` (``"f16"`` or ``"bf16"``) selects the query tensor's
    16-bit float element type, matching pa_decode_ps_kernel's
    ``query_input_dtype`` flag.
    """
    assert head_dim % MFMA_MNK == 0, f"head_dim {head_dim} must be a multiple of {MFMA_MNK}"
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    assert query_dtype in (
        "f16",
        "bf16",
    ), f"pa_decode_tile only supports query_dtype in ('f16', 'bf16'), got {query_dtype}"
    Q_DTYPE = fx.BFloat16 if query_dtype == "bf16" else fx.Float16
    # Same floor as pa_decode_ps_kernel: the addressing below (QKHE_LOOP/
    # N_SUBCHUNKS/VHE_CHUNKS) divides head_dim into 64-element units, so 32
    # would need a smaller load granularity, not just different formulas.
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
    # own K/Q addressing -- see _pa_small_block_load_k_flat /
    # _finish_q_fragments in pa_decode_fp8.py) ──
    # The fp8 mfma_f32_16x16x32 atom's A/B operand is 8 fp8 elements (one i64)
    # per lane per instruction. head_dim is chunked in two levels: a fixed
    # 16-element chunk (QK_CHUNK_ELEMS, one dwordx4 load), 4 of which
    # (RGROUP_QUARTERS = WAVE // MFMA_MNK, `rgroup` == production's `rowid`)
    # make one 64-element fetch group; QKHE_LOOP (the outer fetch-group count)
    # is what scales with head_dim:
    #   QKHE_LOOP = HEAD // (RGROUP_QUARTERS * QK_CHUNK_ELEMS)  (2 for HEAD=128, 1 for 64)
    #   N_SUBCHUNKS = QKHE_LOOP * (QK_CHUNK_ELEMS // 8)          (4 for HEAD=128, 2 for 64)
    # head_dim element for (qkhe, rgroup, qkr): (qkhe*RGROUP_QUARTERS+rgroup)*QK_CHUNK_ELEMS + qkr*8
    RGROUP_QUARTERS = 4
    QK_CHUNK_ELEMS = 16
    QKHE_LOOP = HEAD // (RGROUP_QUARTERS * QK_CHUNK_ELEMS)
    assert QKHE_LOOP >= 1, f"head_dim {head_dim} must be at least {RGROUP_QUARTERS * QK_CHUNK_ELEMS}"
    N_SUBCHUNKS = QKHE_LOOP * (QK_CHUNK_ELEMS // 8)

    # Q-quant chunk width: independent of the QK chunking above. NQCHUNK (the
    # number of QCHUNK-wide slices per row) stays fixed at 16 -- tied to
    # `lane16`'s role as the per-row absmax butterfly width -- so QCHUNK is
    # what scales with head_dim instead.
    NQCHUNK = 16
    QCHUNK = HEAD // NQCHUNK  # f16 elements per lane's load chunk (8 for HEAD=128, 4 for HEAD=64)

    # PV output-dim chunking: VHE_CHUNKS*NWARP*MFMA_MNK covers HEAD (like
    # RGROUP_QUARTERS above, but for PV's N-dimension), so VHE_CHUNKS scales
    # with head_dim while NWARP/MFMA_MNK stay architecture-fixed.
    VHE_CHUNKS = HEAD // (NWARP * MFMA_MNK)  # 2 for HEAD=128, 1 for HEAD=64

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD**0.5)
    _softmax_scale = float(softmax_scale)
    NP = int(num_partitions)  # context partitions (grid.z); compile-time constant

    BLOCK_THREADS = NWARP * WAVE  # 256

    # ── LDS layout (shared across the 4 warps; running state is NOT per-warp) ──
    # sQ  : fp8[16,HEAD]      staged + quantized query tile
    # sP  : fp8[16,TILE_TOK]  quantized softmax probs (re-read by all warps for P·V)
    # sO  : f32[16,HEAD]      running output accumulator (epilogue staging only)
    # sM/sL/sCorr/sQscale : f32[16]   sLmax/sLsum : f32[16,NWARP]
    # No sS/sOp: QK scores stay in the C-fragment (token=M orientation lets the
    # softmax reduce over M via cheap shuffle_xor) and PV's output accumulator
    # is register-resident/loop-carried, not per-tile LDS.
    f32 = 4
    sQ_off = 0
    sQ_bytes = M * HEAD * 1  # fp8
    sP_off = sQ_off + sQ_bytes
    # sP's per-qhead row is padded 16 bytes past its TILE_TOK data bytes: an
    # unpadded TILE_TOK=256B stride is a multiple of the 32-bank*4B=128B LDS
    # wrap, so every (qh, l16g) write in the P-pack store below lands on the
    # SAME bank across all 16 qh values -- a 16-way bank conflict, verified
    # by direct bank-index computation (`(qh*TILE_TOK//4 + l16g) % 32` is
    # identical for every qh, since TILE_TOK//4=64 is a multiple of 32).
    # +16B is the SMALLEST padding that both cuts the worst-case conflict to
    # 2-way AND keeps the row stride a multiple of 16B -- required for the PV
    # read's ds_read_b128 (128-bit) vectorized loads to stay aligned; +8B
    # also cuts the conflict to 2-way but breaks that 16B alignment, forcing
    # narrower/more LDS load instructions and measuring ~3.5% *slower*
    # despite the fixed bank conflict (confirmed via interleaved A/B).
    SP_ROW_BYTES = TILE_TOK + 16
    sP_bytes = M * SP_ROW_BYTES  # fp8, padded rows (only the first TILE_TOK bytes/row hold real data)
    sO_off = sP_off + sP_bytes
    sO_bytes = M * HEAD * f32
    sM_off = sO_off + sO_bytes
    sL_off = sM_off + M * f32
    sCorr_off = sL_off + M * f32
    sQscale_off = sCorr_off + M * f32  # per-row query dequant scale
    # Cross-warp reduction scratch: per (query row, warp) local max/sum. Row
    # stride is padded to NWARP+1 (not NWARP): a plain 4-float/16-byte stride
    # wraps the 32-bank LDS twice, so all 16 rows writing/reading every tile
    # hit a 2-way bank conflict (row r and r+8 share a bank); 5 is coprime
    # with 32 banks, so every row lands on a distinct bank -- same fix as
    # pa_decode_ps_kernel's PROB_ROW_STRIDE_BYTES padding.
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
        pout_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS, HEAD] bf16, normalized O_p/l_p
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

        # fx.copy-based K/V/Q/context_len loaders (see fp8_gemm_utils.py's
        # G2SLoader/StoreC pattern), indexed by the same raw byte/element
        # offset the raw-pointer loaders already compute (fp8 is 1B/elem, so
        # byte offset == element index).
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
        # QCHUNK 16-bit elements (f16 or bf16, per Q_DTYPE) per lane's load:
        # 128 bits for QCHUNK=8 (HEAD=128), 64 bits for QCHUNK=4 (HEAD=64).
        _q_copy_op = fx.rocdl.BufferCopy128b() if QCHUNK == 8 else fx.rocdl.BufferCopy64b()
        _q_load_chunk = _make_flat_loader(query_ptr, Q_DTYPE, QCHUNK, _q_copy_op)
        _ctxlen_load = _make_flat_loader(context_lengths_ptr, fx.Int32, 1, fx.rocdl.BufferCopy32b())

        def _k_load16(byte_off):
            return _k_load_fp8x16(byte_off).bitcast(fx.Int64)

        def _v_load16(byte_off):
            return _v_load_fp8x16(byte_off).bitcast(fx.Int64)

        context_len = fx.Int32(_ctxlen_load(seq)[0])
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        # Per-tensor K/V scales are uniform across the whole launch -- route
        # through a scalar buffer load so it lands once in an SGPR and
        # broadcasts for free (like `_load_phys_scalar` below), matching how
        # pa_decode_ps_kernel reads its own key_scale/value_scale tensors.
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        key_scale = fx.Int32(
            buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True)
        ).bitcast(fx.Float32)
        value_scale = fx.Int32(
            buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True)
        ).bitcast(fx.Float32)

        # This CTA only walks its partition's slice of the TILE_TOK-blocks
        # (context parallelized across grid.z CTAs). Pure arithmetic, no
        # memory access, so it can be computed before the Q-quant barrier,
        # letting the K prefetch that needs it hoist above that barrier too.
        num_tiles = cdiv(context_len, TILE_TOK)
        tiles_per_part = cdiv(num_tiles, NP)
        part_start = part * tiles_per_part
        part_end_raw = part_start + tiles_per_part
        part_end = arith.select(part_end_raw < num_tiles, part_end_raw, num_tiles)
        loop_start = fx.Index(part_start)
        loop_end = fx.Index(part_end)

        # ── LDS views ──
        # One i8 blob carved into typed views via byte-offset pointers, used
        # both as a tiled-copy partition target and for direct .load()/.store()
        # by row-owner lanes. Defined here (not further down) so the V
        # page-prefetch helpers, issued before the Q-quant barrier alongside
        # K's own prologue prefetch, can use it too.
        lds_base = fx.SharedAllocator().allocate(total_bytes).peek().ptr  # i8 base pointer

        def _view(byte_off, elem_ty, layout):
            p = fx.add_offset(lds_base, fx.make_int_tuple(byte_off))
            ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Shared, dsl_size_of(elem_ty))
            return fx.Tensor(fx.make_view(fx.recast_iter(ptr_ty, p), layout))

        c16 = 16
        lane16 = lane - (lane // c16) * c16  # 0..15: this row's head-dim chunk index
        rgroup = lane // c16  # 0..3: which quarter-wave (paired with warp -> query row)

        TOK_CHUNK = NWARP * MFMA_MNK  # 64
        NCHUNK = TILE_TOK // TOK_CHUNK  # 4

        # A compute tile always starts exactly on a page boundary: TILE_TOK
        # (256) is a multiple of block_size for both supported values (16, 64),
        # so there's no "within-page" remainder to track.
        def _tile_tok0_and_page(tt_i32):
            tok0 = tt_i32 * TILE_TOK
            return tok0, tok0 // block_size

        def _load_phys_scalar(page, vec_width=1):
            # ONLY valid when `page` is wave-uniform (same for all 64 lanes) --
            # routes through the scalar/SMEM cache via llvm.amdgcn.s.buffer.load
            # instead of a per-lane VMEM load, the same fix pa_decode_ps_kernel
            # applies to this lookup (`_pa_small_block_stage_phys_blocks` in
            # pa_decode_fp8.py; was 25% of all kernel stalls there). K's page
            # depends only on (a, warp), both wave-uniform. V's page depends on
            # `rgroup` (NOT wave-uniform), so V can't consume this path
            # directly -- but `_v_page_fetch_and_stage` below reuses it
            # (vec_width=PAGES_PER_CHUNK) to *fetch* V's pages, one warp per
            # `rgroup` row, broadcasting to the other warps via LDS.
            result = buffer_ops.buffer_load(
                bt_rsrc, seq * max_blocks_per_seq + page, vec_width=vec_width, is_scalar=True
            )
            return fx.Int32(result) if vec_width == 1 else result

        def _v_page_fetch_and_stage(tt_i32):
            # V's page depends on `rgroup`, shared across all 4 warps (the P/V
            # transpose means every warp needs the same 256-token V range) --
            # so instead of each warp re-deriving all PAGES_PER_CHUNK pages,
            # warp `w` fetches only the row for `rgroup == w` (one scalar wide
            # load, vec_width=PAGES_PER_CHUNK) and broadcasts it to LDS for
            # every warp to read back (`_v_page_read_row`). Prefetched one
            # tile ahead, with the store before and read-back after an
            # already-existing barrier, so no new barrier is needed.
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
                ).store(fetched_vec)

        def _v_page_read_row():
            # Returns one PAGES_PER_CHUNK-wide Vector (not a list): like
            # k_cur/k_next, a single MLIR-backed value lets the loop-carried
            # `if tt1 < part_end: v_page_next = _v_page_read_row()` below
            # reassign it directly, no per-element unpack.
            off = sVPage_off + rgroup * (PAGES_PER_CHUNK * 4)
            return _view(off, fx.Int32, fx.make_layout(PAGES_PER_CHUNK, 1)).load()

        # ── raw dwordx4 K load (A operand), pa_decode_ps_kernel's layout ──
        # CONTIGUOUS-per-warp token assignment: token = warp*TOK_CHUNK +
        # a*c16 + lane16 (each warp owns one contiguous 64-token run
        # [warp*64:+64], matching pa_decode_ps_kernel's own per-warp token
        # ownership -- unlike the earlier interleaved scheme, where warp
        # `warp`'s tokens were four separate 16-token windows spread across
        # the whole 256-token tile: a*64+warp*16+lane16). Softmax's mask
        # (_ct/base_tok_f below) and the P-pack write position must encode
        # this SAME formula for correctness; V's own addressing partitions
        # tokens by `rgroup` (a different, already-contiguous split) and sP
        # is addressed by actual token value either way, so neither needs to
        # change.
        #
        # head_dim chunk index = qkhe*RGROUP_QUARTERS+rgroup (`rgroup` ==
        # production's `rowid`), each chunk one dwordx4 (16B) -- same formula
        # as `_pa_small_block_load_k_flat`.
        #
        # key_cache_ptr uses pa_decode_ps_kernel's own BLOCKED layout
        # [num_blocks, num_kv_heads, HEAD//16, block_size, 16] (fp8), NOT the
        # plain [num_blocks,num_kv_heads,block_size,HEAD] layout: the 16-byte
        # head-chunk is innermost/contiguous per token, so adjacent lanes
        # (adjacent TOKENS, per the MFMA-fixed lane roles) land 16 bytes
        # apart -- coalesced. The plain layout instead puts adjacent lanes
        # HEAD bytes apart (confirmed via ATT trace to be the dominant stall,
        # ~58% of all cycles, from poor coalescing).
        #
        # A warp's own 64-token run spans PAGES_PER_CHUNK pages (1 at
        # block_size=64, 4 at block_size=16): `page_within_warp = (a*c16) //
        # block_size`, `within_page_tok = (a*c16 + lane16) % block_size`,
        # `physical_page = base_page + warp*PAGES_PER_CHUNK + page_within_warp`
        # -- and because a warp's PAGES_PER_CHUNK pages are now always
        # CONSECUTIVE (unlike the old interleaved formula, which was only
        # consecutive across `a` at block_size=64), `_k_ops_flat` below folds
        # every case into one wide scalar load, not just block_size=64.
        c_nhgroup = QCHUNK  # total 16-element head-chunks in the cache layout (== HEAD // 16)

        def _k_ops(phys, a):
            within_page_tok = (a * c16 + lane16) % block_size
            ops = []
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                base = (((phys * n_kv + kv_h) * c_nhgroup + he_idx) * block_size + within_page_tok) * QK_CHUNK_ELEMS
                w = _k_load16(base)  # head[he_idx*16 : +16] -> k_step 2*qkhe, 2*qkhe+1
                ops.extend([w[0], w[1]])
            return ops  # N_SUBCHUNKS i64 operands

        # All NCHUNK chunks' K as one flat (NCHUNK*N_SUBCHUNKS,) i64 vector --
        # one loop-carried value (not NCHUNK*N_SUBCHUNKS scalars) so tile
        # tt+1's K prefetch overlaps tt's softmax+PV, like pa_decode_ps_kernel.
        def _k_ops_flat(tt_i32):
            _, base_page = _tile_tok0_and_page(tt_i32)
            # This warp's PAGES_PER_CHUNK physical pages are consecutive,
            # starting at base_page + warp*PAGES_PER_CHUNK -- one wide
            # `vec_width=PAGES_PER_CHUNK` scalar load replaces the old
            # per-`a` (or, at PAGES_PER_CHUNK==1, per-NCHUNK) lookups, same
            # `_load_phys_scalar`-into-LDS-broadcast-free pattern as
            # `pa_decode_ps_kernel`'s own `_pa_small_block_stage_phys_blocks`.
            fetched = _load_phys_scalar(base_page + warp * PAGES_PER_CHUNK, PAGES_PER_CHUNK)
            phys_vec = (
                fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
                if const_expr(PAGES_PER_CHUNK == 1)
                else fx.Vector(fetched)
            )
            flat = []
            for a in range_constexpr(NCHUNK):
                phys = fx.Int32(phys_vec[(a * c16) // block_size])
                flat.extend(_k_ops(phys, a))
            if const_expr(HEAD == 64):
                fx.rocdl.sched_vmem(len(flat) // 2)
            return fx.Vector.from_elements(flat, dtype=fx.Int64)

        # ── prologue: prefetch the first tile's K ──
        # Issued before Q-quant's barrier (not right before the main loop), so
        # K's global loads and block-table lookups can be scheduled
        # concurrently with Q's own global load -- a barrier is a hard
        # reordering boundary, so issuing them together lets the memory
        # subsystem overlap their latency instead of paying both serially.
        num_tiles_m1 = num_tiles - 1
        start_safe = arith.select(part_start < num_tiles, part_start, num_tiles_m1)
        k_pf0 = _k_ops_flat(start_safe)
        # V page-index prefetch, issued here too so it overlaps Q-quant the
        # same way K's does; the LDS write is visible after the barrier below,
        # so `_v_page_read_row` needs no barrier of its own.
        _v_page_fetch_and_stage(start_safe)

        # ── per-CTA scalar constants ──
        # softmax_scale and key_scale fold into the score tile; log2e folds into
        # the exp2 used for softmax.
        scale_qk = fx.Float32(_softmax_scale * LOG2E) * fx.Float32(key_scale)
        v_scale_f = fx.Float32(value_scale)
        NEG_INF = fx.Float32(float("-inf"))
        ZERO_F = fx.Float32(0.0)

        def _row(byte_off, m_idx, width, elem_ty):
            off = byte_off + m_idx * (width * dsl_size_of(elem_ty))
            return _view(off, elem_ty, fx.make_layout(width, 1))

        def _ld1(byte_off, m_idx):
            return _row(byte_off, m_idx, 1, fx.Float32).load()[0]

        def _st1(byte_off, m_idx, val):
            _row(byte_off, m_idx, 1, fx.Float32).store(fx.Vector.from_elements([val], dtype=fx.Float32))

        # f32[16, NWARP] cross-warp scratch (row stride padded to NWARP_PAD to
        # avoid the 2-way LDS bank conflict -- see `sLmax_off`'s comment):
        # scalar write at (row, warp), vec read of a row's NWARP valid slots.
        def _st_lw(base_off, row, w, val):
            off = base_off + (row * NWARP_PAD + w) * 4
            _view(off, fx.Float32, fx.make_layout(1, 1)).store(fx.Vector.from_elements([val], dtype=fx.Float32))

        def _ld_lw_row(base_off, row):
            off = base_off + row * (NWARP_PAD * 4)
            return _view(off, fx.Float32, fx.make_layout(NWARP, 1)).load()

        def _f32_to_fp8_words(vf32):
            # f32 -> fp8 must use the HW cvt (arith.truncf to fp8 doesn't lower);
            # pack 4 f32 -> 1 i32 (4 fp8) via two cvt_pk_fp8_f32 calls.
            n = vf32.shape[0]
            words = []
            for i in range_constexpr(n // 4):
                b = i * 4
                lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b], vf32[b + 1], 0, False)
                words.append(fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b + 2], vf32[b + 3], lo, True))
            return fx.Vector.from_elements(words, dtype=fx.Int32)

        def _st_words(byte_off, words):
            _view(byte_off, fx.Int32, fx.make_layout(words.shape[0], 1)).store(words)

        # sP holds fp8 probabilities as [qhead, token] (qhead stride
        # SP_ROW_BYTES, padded past TILE_TOK to avoid an LDS bank conflict --
        # see SP_ROW_BYTES's comment; token stride 1); each lane writes its
        # (qhead, token) slice directly and the PV P read (p_ops) reads it
        # back with the same layout.

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
        # QCHUNK-element head-dim slice, so every thread loads, converts, and
        # packs exactly one chunk -- matching pa_decode_ps_kernel's
        # `_finish_q_fragments` lane-per-chunk layout. The row's absmax is then
        # a DPP butterfly reduction over the 16 lanes sharing (warp, rgroup)
        # (see below), no LDS/barrier needed for the reduction itself. A
        # single-lane-per-row design here previously idled ~240 threads at the
        # barrier, costing ~7-8% of all kernel stall cycles at bs=128/ctx=16384.

        qh_local = warp * 4 + rgroup  # 0..15: this thread's query row

        if qh_local < GS:
            qh0 = kv_h * GS + qh_local
            row_byte0 = (seq * num_q_heads + qh0) * (HEAD * 2)  # 16-bit float = 2B/elem
            chunk_off = row_byte0 + lane16 * (QCHUNK * 2)
            q_chunk = _q_load_chunk(chunk_off // 2)  # byte offset -> element index

            # Local absmax over this thread's own QCHUNK elements, kept in
            # Q_DTYPE (widening to f32 is monotonic for both f16/bf16, so
            # comparing at native width avoids a full-vector fpext forcing
            # scalarized conversions), then a butterfly reduce over the 16
            # lanes owning this row (fixed warp/rgroup, lane16 varies).
            #
            # The cross-lane reduce uses `dpp_utils.dpp_xor_f32` (raw
            # llvm.amdgcn.update.dpp.i32), matching pa_decode_ps_kernel's own
            # analogous reduction (`_finish_q_fragments`) instead of the DSL's
            # generic `shuffle_xor`: DPP runs in the VALU crossbar with no
            # LDS/DS-unit involvement, cheaper than even shuffle_xor's
            # ds_swizzle path for a 16-lane XOR (confirmed via LLVM IR).
            local_absmax = fmath.absf(q_chunk).reduce(ReductionOp.MAX)
            absmax = local_absmax.to(fx.Float32)
            for sh in (8, 4, 2, 1):
                absmax = absmax.maximumf(dpp_utils.dpp_xor_f32(absmax, sh))
            # q_scale = absmax / FP8_MAX; rcp_f32 (HW reciprocal, matching
            # pa_decode_ps_kernel's `inv_query_scale`) instead of a full IEEE
            # division -- its error is negligible next to fp8 quant noise.
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
        # O, the running max `m`, and running denom `l` are all loop-carried
        # registers (see m_prev/m_new, l_prev/l_new below): every thread
        # already holds its own cross-warp-combined value each tile, so no
        # per-tile LDS round-trip is needed -- just a single post-loop bridge
        # store into sM/sL for the epilogue's differently-indexed threads.
        gpu.barrier()

        # First tile's V page-index row, now visible after the barrier above
        # (the fetch+LDS-store was issued earlier, alongside k_pf0).
        v_page_pf0 = _v_page_read_row()

        # Q is the B operand (D[token,qhead] = K·Qᵀ), read raw from sQ as fp8
        # i64 operands (constant across tiles -> read once, held in
        # registers). Lane (lane16, rgroup) feeds MMA column n=lane16 (qhead);
        # MUST use the exact same (qkhe, rgroup, qkr) -> head_dim permutation
        # as K's `_k_ops` (matches pa_decode_ps_kernel's `_finish_q_fragments`).
        q_ops = []
        for qkhe in range_constexpr(QKHE_LOOP):
            he_idx = qkhe * RGROUP_QUARTERS + rgroup
            chunk = _view(sQ_off + lane16 * HEAD + he_idx * QK_CHUNK_ELEMS, fx.Int64, fx.make_layout(2, 1)).load()
            q_ops.extend([chunk[0], chunk[1]])
        # q_ops[s] for s=0..N_SUBCHUNKS-1, s = qkhe*2+qkr, = head[he_idx*16+qkr*8 : +8] of qhead=lane16

        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tcopy_o = fx.make_tiled_copy_C(copy_c, tiled_mma_pv)  # PV out -> sO (epilogue)

        # QK in TLOOP chunks of TOK_CHUNK tokens: each chunk yields a f32x4
        # C-fragment, so softmax processes 4 scores at a time (low VGPR peak)
        # -- matching pa_decode_ps_kernel's TLOOP.
        # compile-time per-chunk token offsets (token = base_tok_f + a*c16 + r,
        # matching K's own contiguous-per-warp formula above: base_tok_f
        # supplies warp*TOK_CHUNK + l16g*4)
        _ct = [
            fx.Vector.from_elements([float(a * c16 + r) for r in range_constexpr(4)]) for a in range_constexpr(NCHUNK)
        ]
        # P·V is loop-tiled over head-dim (like production's VHELOOP): each
        # step computes O[:, vh*VHE_SIZE:+VHE_SIZE] instead of materializing
        # the full [16, HEAD] at once.
        VHE_SIZE = HEAD // VHE_CHUNKS
        tmpl_Op = fx.make_rmem_tensor(fx.make_layout((M, VHE_SIZE), (VHE_SIZE, 1)), fx.Float32)
        OP_ELEMS = M * VHE_SIZE // (NWARP * WAVE)  # PV C-fragment elements/lane/chunk (probed = 4)

        # ── raw dwordx4 V load (B operand), pa_decode_ps_kernel's layout ──
        # PV contracts over token, so the token->k_step mapping is free as
        # long as V and P (p_ops) agree: lane (rgroup) takes the contiguous
        # token slice [rgroup*64:+64] for its head (vh*VHE_SIZE+warp*16+
        # lane16), loaded as 4x i64x2 (128-bit) = 8 k_step operands.
        #
        # value_cache_ptr uses pa_decode_ps_kernel's own "trans_v" BLOCKED
        # layout [num_blocks, num_kv_heads, block_size//16, head_dim, 16]
        # (fp8): 16 CONSECUTIVE TOKENS innermost (V is token-vectorized, not
        # head-dim-vectorized like K), block_size//16 token-subblock outermost.
        #
        # A rgroup's 64-token run can span multiple block_size pages: outer
        # loop `sub` picks the page, inner loop `step` walks that page's own
        # token run in 16-token increments (production's token-subblock
        # index) -- collapses to single-page/4-step at block_size=64 and
        # 4-pages/1-step at block_size=16, both totaling NVOPS=8 i64 operands.
        NVOPS = TILE_TOK // MFMA_K  # 8 PV k_steps (256 tokens / K=32)
        STEPS_PER_PAGE = block_size // 16

        # V's page depends only on (rgroup, sub), not `warp`, so all 4 warps
        # want the same pages every tile -- see `_v_page_fetch_and_stage`/
        # `_v_page_read_row` above for the once-per-tile fetch+broadcast.
        def _v_ops(phys_row, vh):
            head_group = ((vh * VHE_SIZE) // 16) + warp
            head_element = head_group * 16 + lane16
            ops = []
            for sub in range_constexpr(PAGES_PER_CHUNK):
                for step in range_constexpr(STEPS_PER_PAGE):
                    base = (((phys_row[sub] * n_kv + kv_h) * STEPS_PER_PAGE + step) * HEAD + head_element) * 16
                    w = _v_load16(base)
                    ops.extend([w[0], w[1]])
            # `sched_group_barrier(mask_vmem_rd, NVOPS, 0)` groups all NVOPS
            # raw V loads above into one scheduling unit -- a scheduling-only
            # intrinsic with no result register and no VGPR-live-range cost,
            # unlike every register-carrying V-hiding attempt above. (A
            # first attempt inserting this hint after EACH individual load
            # instead of once for the whole group measured ~1% *slower*,
            # t=-3.49 -- it forced per-load serialization instead of letting
            # the scheduler batch all NVOPS loads together for memory-level
            # parallelism; reverted in favor of this single whole-group form.)
            #
            # Measured shape-dependent: ~2.2% faster at head_dim=64/ctx=16384
            # (t=8.35, 16-sample interleaved A/B) but ~3.2% *slower* at
            # head_dim=128/ctx=32768 (t=-2.14, VHE_CHUNKS=2 there calls this
            # hint twice per tile instead of once, apparently over-
            # constraining the scheduler across the two vh chunks). Gated to
            # HEAD==64 only, matching this file's existing head_dim compile-
            # time specialization elsewhere (QKHE_LOOP/VHE_CHUNKS/NCHUNK).
            if const_expr(HEAD == 64):
                fx.rocdl.sched_vmem(len(ops) // 2)
            return ops  # NVOPS i64, the 64-token contiguous run for this head

        # O is carried in registers across tiles (one VHE_CHUNKS-list of PV
        # C-fragment vectors), rescaled by the softmax correction each tile.
        # `m` is carried the same way (see the init comment above).
        o_zero = fx.Vector.filled(OP_ELEMS, 0.0, fx.Float32)
        for tt, ostate in range(
            loop_start, loop_end, arith.index(1), init=[o_zero, o_zero, k_pf0, v_page_pf0, NEG_INF, ZERO_F]
        ):
            o_acc = [ostate[0], ostate[1]]
            k_cur = ostate[2]  # this tile's prefetched K, as one (NCHUNK*N_SUBCHUNKS,) i64 vector
            v_page_cur = ostate[3]  # this tile's V pages, as one PAGES_PER_CHUNK-wide i32 vector
            m_prev = ostate[4]  # this thread's own running max, carried from last tile
            l_prev = ostate[5]  # this thread's own running denom, carried from last tile
            tt_i32 = fx.Int32(tt)
            tok0, _ = _tile_tok0_and_page(tt_i32)

            # ---- QK in TLOOP chunks: NCHUNK raw MFMAs -> f32x4/lane ----
            # Each chunk accumulates the N_SUBCHUNKS head-quarter k_steps
            # (this tile's prefetched k_cur) into one f32x4 C-fragment
            # (D[token, qhead]).
            frag_Ss = []
            for a in range_constexpr(NCHUNK):
                acc = arith.constant_vector(0.0, T.f32x4)
                for s in range_constexpr(N_SUBCHUNKS):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [k_cur[a * N_SUBCHUNKS + s], q_ops[s], acc, 0, 0, 0]
                    )
                frag_Ss.append(fx.Vector(acc))

            # Prefetch next tile's K; loads overlap softmax+PV below. Backing
            # pages almost always change tile-to-tile at these small block
            # sizes, so every tile re-derives its page(s) fresh.
            #
            # (Issuing this before the QK MFMAs instead was tried -- legal,
            # since `_k_ops_flat` has no dependency on QK's output, but it
            # extended k_next's live range across the whole QK MFMA loop. The
            # compiler responded with AGPR-form MFMA and far higher register
            # pressure (VGPR 16+AGPR 128=144 vs. 112), dropping occupancy
            # 4->3 waves/SIMD and measuring ~9.5% *slower* despite the extra
            # hiding window -- same failure mode as V's attempt below. Reverted.)
            #
            # (V is deliberately NOT carried the same way: prefetching v_next
            # one tile ahead, mirroring k_next, pushed VGPR up ~20% (136->164)
            # and measured slower. V's load is deferred to right before its PV
            # use instead -- see the PV loop below -- cutting peak VGPR far
            # more (132->112, crossing the 3->4 waves/SIMD boundary) and
            # measuring ~10% *faster* despite losing that hiding window.)
            #
            # On a partition's last tile there's no next tile, so skip with a
            # real conditional (not `arith.select`, which only clamps the
            # index while the loads still execute) -- unconditional
            # prefetching there was a measurable waste for thin partitions.
            #
            # k_cur/k_next are one (NCHUNK*N_SUBCHUNKS,) i64 Vector (a single
            # MLIR-backed value, not NCHUNK*N_SUBCHUNKS scalars), so this
            # if-reassignment works directly, no per-element unpack.
            tt1 = tt_i32 + 1
            k_next = k_cur
            if tt1 < part_end:
                k_next = _k_ops_flat(tt1)

            # Prefetch next tile's V page-index row the same way: fetch +
            # LDS store here, before the softmax barrier; read back into
            # `v_page_next` right after, reusing that barrier.
            if tt1 < part_end:
                _v_page_fetch_and_stage(tt1)

            # ---- register-resident softmax over M = token, 4 scores at a time ----
            # Each lane owns ONE qhead (= lane%16); reduce its tokens with a
            # register reduce + shuffle_xor(16,32). Mask is a cheap scalar
            # threshold (token = warp*TOK_CHUNK+a*c16+l16g*4+r < n_valid,
            # matching K's contiguous-per-warp formula above).
            c16 = 16
            qh = lane - (lane // c16) * c16  # qhead = lane % 16
            l16g = lane // c16  # 0..3 lane-group within the warp
            scale = scale_qk * _ld1(sQscale_off, qh)  # per-qhead positive score scale
            n_valid_tile = (context_len - tok0).to(fx.Float32)
            base_tok_f = fx.Int32(warp * TOK_CHUNK + l16g * 4).to(fx.Float32)
            thr = fx.Vector.from_elements([n_valid_tile - base_tok_f], dtype=fx.Float32).broadcast_to(4)
            neg4 = fx.Vector.filled(4, float("-inf"), fx.Float32)

            # Computed once and reused in pass 2 below (doesn't depend on pass
            # 1's output) instead of recomputing from scratch in both passes,
            # halving the mask compare/select instruction count for free.
            masked_chunks = [(_ct[a] < thr).select(frag_Ss[a], neg4) for a in range_constexpr(NCHUNK)]

            # pass 1: per-warp max for this qhead
            pm = fx.Float32(float("-inf"))
            for a in range_constexpr(NCHUNK):
                pm = pm.maximumf(masked_chunks[a].reduce(ReductionOp.MAX))
            for sh in (16, 32):
                pm = pm.maximumf(pm.shuffle_xor(sh, WAVE))
            # Unconditional store (matches pa_decode_ps_kernel's own
            # `_cross_warp_softmax_and_prob_pack`): all 4 lanes sharing this
            # qhead already hold the identical post-shuffle_xor `pm`, so this
            # is a harmless same-value redundant write, not a race.
            _st_lw(sLmax_off, qh, warp, pm * scale)
            gpu.barrier()

            # Read back next tile's V page-index row now that the barrier
            # above has made `_v_page_fetch_and_stage`'s store visible.
            v_page_next = v_page_cur
            if tt1 < part_end:
                v_page_next = _v_page_read_row()

            # Re-test of an idea previously reverted at head_dim=128 (see the
            # PV loop's own comment below, "three variants ... none produced
            # a verified win"): `v_page_cur` has no dependency on anything
            # computed THIS tile (it's this tile's own already-prefetched
            # page row, available at loop entry), so its raw V loads can
            # legally be issued here -- right after the pass-1 barrier,
            # before pass 2's exp/pack/store work -- instead of right before
            # PV with nothing to hide behind. The earlier attempts predate
            # the fastmath + LDS bank-conflict fixes elsewhere in this file,
            # which dropped baseline VGPR enough (84+4=88, now equal to
            # production's own combined VGPR at this shape) that this costs
            # only +8 VGPR (92+4=96) -- NOT enough to cross the 5->4
            # waves/SIMD occupancy boundary (512//96 still floors to 5) the
            # way the old attempts did. Measured ~4.9% faster (t=27.4,
            # 16-sample interleaved A/B) at head_dim=64/ctx=16384 -- this is
            # what closed the remaining gap to pa_decode_ps_kernel and then
            # some (tile now ~1.2% *faster*, t=-8.3). Only applied at
            # HEAD==64 (VHE_CHUNKS==1: this loads exactly the one vh chunk
            # PV will use, not "the full VHE_CHUNKS set" the reverted
            # head_dim=128 attempts hoisted) -- not re-verified at HEAD==128.
            v_vh_early = None
            if const_expr(HEAD == 64):
                v_vh_early = _v_ops(v_page_cur, 0)

            # pass 2: global max over warps -> exp -> fp8 P pack (-> sP) -> sum
            m_old = m_prev
            m_new = m_old.maximumf(_ld_lw_row(sLmax_off, qh).reduce(ReductionOp.MAX))
            m_new_b = fx.Vector.from_elements([m_new], dtype=fx.Float32).broadcast_to(4)
            ls = fx.Float32(0.0)
            # Raw i32-word store straight to sP[qhead][token_base:+4] (fp8,
            # 1B/elem): the packed word's 4 fp8 lanes are exactly the 4
            # consecutive tokens this lane owns in chunk `a`.
            base4 = 4
            words = []
            for a in range_constexpr(NCHUNK):
                # HW exp2 intrinsic instead of MLIR's generic math.exp2 (a
                # polynomial approximation costing ~32 extra v_cndmask here)
                # -- matches pa_decode_ps_kernel's own exp2_f32_fast usage.
                Pa = fx.Vector(exp2_f32_fast(masked_chunks[a] * scale - m_new_b))
                ls = ls + Pa.reduce(ReductionOp.ADD)
                words.append(_f32_to_fp8_words(Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32))[0])
            # One strided vector store instead of NCHUNK separate 4-byte
            # stores -- the backend packs it into 2 ds_write2_b32 instructions
            # instead of 3, with no VGPR cost. sP is addressed by actual
            # token value ([qhead][token]), so this write position must
            # match K's contiguous-per-warp token formula above: word `a`'s 4
            # packed tokens start at warp*TOK_CHUNK+a*c16+l16g*4, i.e. a
            # per-`a` stride of c16 bytes (not TOK_CHUNK) now that `a` is the
            # WITHIN-warp offset instead of the across-tile one. Row stride
            # is SP_ROW_BYTES (TILE_TOK+8 padding), not TILE_TOK -- see
            # SP_ROW_BYTES's comment above (bank-conflict avoidance).
            p_off0 = sP_off + qh * SP_ROW_BYTES + warp * TOK_CHUNK + l16g * base4
            _view(p_off0, fx.Int32, fx.make_layout(NCHUNK, c16 // 4)).store(
                fx.Vector.from_elements(words, dtype=fx.Int32)
            )
            if const_expr(HEAD == 64):
                # `sched_group_barrier(mask_dswr, NCHUNK, 0)`: same
                # scheduling-only, zero-VGPR-cost mechanism as the
                # `sched_vmem` hints on K/V's raw loads above (see `_v_ops`),
                # now grouping this LDS write instead. Measured ~0.68%
                # faster (t=2.82, 16-sample interleaved A/B) on top of the
                # K/V sched_vmem wins, no regression at head_dim=128 (not
                # applied there). A matching `sched_dsrd` hint on the P·V
                # read side below (`p_ops`) was tried and measured a severe
                # ~10% *slower* result instead -- that read is already one
                # fused vectorized LDS load, not a multi-instruction loop
                # like this store, so grouping it fought the scheduler
                # rather than helping it.
                fx.rocdl.sched_dswr(NCHUNK)
            for sh in (16, 32):
                ls = ls.addf(ls.shuffle_xor(sh, WAVE), fastmath=arith.FastMathFlags.contract)
            if l16g == 0:
                _st_lw(sLsum_off, qh, warp, ls)
                if warp == 0:
                    _st1(sCorr_off, qh, fx.Float32(exp2_amdgcn_scalar(m_old - m_new)))
            gpu.barrier()

            # phase 3: merge per-warp sums into the running denominator. `tid <
            # M` is exactly the `(l16g==0 and warp==0)` thread set above, so
            # its correction factor is already in `m_old`/`m_new` registers --
            # no need to read back what it just wrote to sCorr_off/sL_off.
            l_new = l_prev
            if tid < M:
                gsum = _ld_lw_row(sLsum_off, tid).reduce(ReductionOp.ADD)
                corr_reg = fx.Float32(exp2_amdgcn_scalar(m_old - m_new))
                accum_sum = fx.Float32(
                    arith.mulf(arith.unwrap(l_prev), arith.unwrap(corr_reg), fastmath=arith.FastMathFlags.contract)
                )
                l_new = accum_sum.addf(gsum, fastmath=arith.FastMathFlags.contract)

            # ---- read P back as the A operand for P·V: lane reads
            # sP[qhead=lane16][token rgroup*64:+64], the same permuted token
            # slice v_ops uses so the raw PV MMA matches. Row stride is
            # SP_ROW_BYTES (see the P-pack store above), not TILE_TOK. ----
            p_ops = _view(
                sP_off + lane16 * SP_ROW_BYTES + rgroup * 64,
                fx.Int64,
                fx.make_layout(NVOPS, 1),
            ).load()

            # ---- PV with register-resident O accumulate (no LDS round-trip) ----
            # O_new = O_old*corr + P·V per head-dim chunk; corr = exp2(m_old-m_new)
            # is per-row (PV C-fragment: vec element v of lane L holds row
            # m = (L%64//16)*4+v, so corr_s[v] = corr[m_base+v]). No barrier
            # after PV: O is in registers and next iter's barriers order any
            # sP reuse. Raw PV MMA: NVOPS k_steps accumulate into one f32x4.
            m_base_pv = (lane // c16) * 4
            # OP_ELEMS contiguous f32 in one vectorized LDS read instead of
            # OP_ELEMS separate scalar reads.
            corr_off = sCorr_off + m_base_pv * 4
            corr_vec = _view(corr_off, fx.Float32, fx.make_layout(OP_ELEMS, 1)).load()
            corr_s = [corr_vec[v] for v in range_constexpr(OP_ELEMS)]
            for vh in range_constexpr(VHE_CHUNKS):
                # At HEAD==64, `v_vh` is actually `v_vh_early` (loaded right
                # after the pass-1 barrier -- see that assignment's own
                # comment) rather than hoisted here right before its MFMA
                # use. This overturns the finding right below: at the time
                # it was written (predating the fastmath + LDS bank-conflict
                # fixes elsewhere in this file), issuing V any earlier than
                # right-before-use always cost enough VGPR to drop occupancy
                # and lose net. Once baseline VGPR fell to 88 (equal to
                # production's), the SAME early-issue idea cost only +8 VGPR
                # -- not enough to cross the occupancy boundary -- and
                # measured ~4.9% faster (t=27.4). At HEAD==128 (VHE_CHUNKS==
                # 2, not re-verified under the current baseline) this
                # right-before-use load is still used, trading hiding window
                # for lower peak VGPR (the kernel's binding occupancy
                # constraint -- see the K-prefetch comment above for the
                # measured win this bought there). Re-confirmed at
                # large tiles-per-partition (bs=128, ctx up to 65536) with
                # THREE variants -- full-tile ping-pong issued right after the
                # pass-1 barrier, issued right after this tile's own PV MFMA
                # (mirroring pa_decode_ps_kernel's placement), and a smaller
                # one-vh-ahead software pipeline -- none produced a verified
                # win: the full-tile variants measured ~15-18% *slower*
                # (AGPR-heavy 3-waves/SIMD occupancy loss); the one-vh
                # pipeline kept occupancy unchanged but left the ATT stall
                # profile statistically unchanged too (VMEM-wait 38.06% vs
                # baseline 37.49%) and its wall-clock delta was noise-level
                # across repeated runs. Root cause (confirmed via ISA-level,
                # exec-count-normalized ATT analysis): tile's per-tile
                # cross-warp softmax reduction (`shuffle_xor`-based, ~742/781
                # below) cost ~3x more cycles per instance than
                # pa_decode_ps_kernel's structurally-identical reduction, for
                # reasons rooted in inter-warp LDS/DS-unit contention, not
                # source-level coding or scheduling differences -- so tile
                # couldn't afford to trade a wave of occupancy for V-hiding
                # the way production can, and reordering V's load within the
                # existing occupancy budget didn't move the needle since the
                # compiler's own scheduler already extracts what overlap is
                # available regardless of source-level call placement. Fixed
                # not by further V-prefetch tuning but by the contiguous-
                # per-warp token reassignment above (K's addressing / mask /
                # P-pack): measured (15-sample A/B, since the true ~3-5%
                # effect is smaller than single-digit-sample noise) ~3.1%
                # faster at ctx=32768 (t=2.42) and ~4.9% faster at ctx=65536
                # (t=5.16) at bs=128, block_size=64, head_dim=128. A second,
                # independent win: pa_decode_ps_kernel applies
                # `fastmath=contract` (FMA fusion) to its own cross-warp sum
                # combine and PV correction-scale (`_cross_warp_softmax_and_
                # prob_pack`/`_pv_mfma`); tile used plain `+`/`*` there with
                # no fastmath at all. Matching it (see `ls.addf`/`l_new`
                # combine/`o_acc[vh]` update below) dropped VGPR 92->84 at
                # head_dim=64 (fewer fused instructions, not a VGPR/occupancy
                # trade) and measured ~1.1% faster at head_dim=64/ctx=16384
                # (t=3.81, 8-sample interleaved A/B), with no regression at
                # head_dim=128/ctx=32768 (t=-0.28, noise-level).
                #
                # A fourth idea specifically targeting V's exposed HBM latency
                # (the ~7% residual gap at head_dim=64/ctx=16384 after the LDS
                # bank-conflict fix elsewhere in this file): `rocdl.
                # global_prefetch` (llvm.amdgcn.global.prefetch) issues a
                # fire-and-forget L2 cache-warm hint with NO destination
                # register, so unlike every attempt above it can't cost VGPR/
                # occupancy at all -- fetch next tile's V bytes via this
                # intrinsic right after `v_page_next` becomes available, no
                # register-carried state needed. Untried avenue on paper, but
                # it hard-crashes ("Fatal Python error: Aborted") during the
                # compile pipeline on gfx942 -- confirming this file's own
                # source comment in tdm_ops.py's `l2_prefetch_tile`: its LLVM
                # ISel pattern was written for gfx1250's global_prefetch_b8
                # and is not merely "silently dropped" elsewhere, it's
                # unsupported outright on gfx942. Not usable here.
                v_vh = v_vh_early if const_expr(HEAD == 64) else _v_ops(v_page_cur, vh)
                acc = arith.constant_vector(0.0, T.f32x4)
                for s in range_constexpr(NVOPS):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [p_ops[s], v_vh[s], acc, 0, 0, 0])
                op = fx.Vector(acc)
                oo = o_acc[vh]
                fm_contract = arith.FastMathFlags.contract
                o_acc[vh] = fx.Vector.from_elements(
                    [
                        fx.Float32(
                            arith.addf(
                                arith.mulf(arith.unwrap(oo[v]), arith.unwrap(corr_s[v]), fastmath=fm_contract),
                                arith.unwrap(op[v]),
                                fastmath=fm_contract,
                            )
                        )
                        for v in range_constexpr(OP_ELEMS)
                    ],
                    dtype=fx.Float32,
                )
            results = yield [o_acc[0], o_acc[1], k_next, v_page_next, m_new, l_new]
        o_final = results
        m_final = o_final[4]
        l_final = o_final[5]

        # One-time bridge write of the final running max/denom from `qh`
        # (this loop's indexing) to sM_off/sL_off, so the epilogue's
        # DIFFERENT `row_e`-indexed threads can read them (sM_off only
        # matters for NP>1). `qh`/`l16g` are scoped to the loop body, so
        # recompute them here from `lane`/`c16` -- cheap, no memory access.
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
            )
            fx.copy(copy_c, thr_copy_o_e.retile(frag_Oe), thr_copy_o_e.partition_D(sO_chunk), pred=None)
        gpu.barrier()

        # ── epilogue: spread the row -> global write across ALL 256 threads
        # (THREADS_PER_ROW threads per query row, each owning a contiguous
        # ELEMS_PER_THREAD-wide slice) instead of only the GS row-owner lanes
        # looping over all HEAD elements -- fully uses the wave and cuts the
        # epilogue's static instruction count (measured ds_read: 45 -> ~15,
        # matching pa_decode_ps_kernel's range).
        assert BLOCK_THREADS % GS == 0, "epilogue requires BLOCK_THREADS to divide evenly by GS"
        THREADS_PER_ROW = BLOCK_THREADS // GS
        ELEMS_PER_THREAD = HEAD // THREADS_PER_ROW
        assert ELEMS_PER_THREAD * THREADS_PER_ROW == HEAD, "epilogue requires HEAD % (BLOCK_THREADS // GS) == 0"

        c_tpr = THREADS_PER_ROW
        row_e = tid // c_tpr
        sub_e = tid - row_e * c_tpr
        col_e = sub_e * ELEMS_PER_THREAD
        row_off = sO_off + row_e * (HEAD * 4) + col_e * 4
        o_v = _view(row_off, fx.Float32, fx.make_layout(ELEMS_PER_THREAD, 1)).load()
        # value_scale and the P dequant (1/FP8_MAX) are true per-CTA constants
        # here (unlike pa_decode_ps_kernel's `_pv_mfma`, which folds
        # value_scale into every PV tile since that path is shared with
        # per-token_kv) -- fold them in exactly once, before the NP branch, so
        # both paths write an already-scaled numerator and the reduce kernel
        # needs no value_scale of its own.
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
            # Divide this (seq, qh) row's HEAD axis into ELEMS_PER_THREAD-wide
            # chunks and pick this lane's chunk (no manual byte offset).
            out_row = output_ptr[seq, qh, None]
            out_chunk = fx.slice(fx.logical_divide(out_row, fx.make_layout(ELEMS_PER_THREAD, 1)), (None, sub_e))
            out_chunk.store(o_out)
        else:
            # multi-partition: write this partition's (m_p, l_p, already-
            # normalized + scaled O_p/l_p in Q_DTYPE) -- reused pa_decode_sw_reduce_kernel
            # (kernels/pa_decode_swa.py, pa_decode_ps_kernel's own reduce)
            # does a weighted blend of these pre-normalized partials, not a
            # raw-numerator/raw-denominator combine like the old custom reduce.
            base = ((seq * n_kv + kv_h) * NP + part) * GS + row_e
            if sub_e == 0:
                pmax_ptr[base] = _ld1(sM_off, row_e)
                psum_ptr[base] = _ld1(sL_off, row_e)
            inv_l_p = fx.Float32(rcp_f32(_ld1(sL_off, row_e)))
            o_norm = (o_v * fx.Vector.from_elements([inv_l_p], dtype=fx.Float32).broadcast_to(ELEMS_PER_THREAD)).to(
                Q_DTYPE
            )
            pout_div = fx.logical_divide(pout_ptr, fx.make_layout(ELEMS_PER_THREAD, 1))
            pout_chunk = fx.slice(pout_div, (None, base * THREADS_PER_ROW + sub_e))
            pout_chunk.store(o_norm)

    # NP>1 reduce: reuses pa_decode_ps_kernel's own `pa_decode_sw_reduce_kernel`
    # (kernels/pa_decode_swa.py) instead of a custom implementation -- see
    # `pa_decode_tile()` below, where it's compiled and launched from the host
    # (matching how pa_decode_ps_launch itself calls it).

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
    # K's blocked layout matches pa_decode_ps_kernel's own (fixed 16-element
    # head-chunk width -- see compile_pa_decode_tile's QKHE_LOOP comment).
    assert num_hgroups == head_dim // 16 and hgroup_width == 16
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    _, _, v_subblocks, v_head_dim, v_width = value_cache.shape
    # V's "trans_v" layout matches pa_decode_ps_kernel's own.
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
    # kernel), so it needs its own NP heuristic. Two regimes, measured
    # directly on an 80-CU MI308X:
    #
    #   - CU-STARVED (num_seqs*num_kv_heads < device CU count): push NP up to
    #     `cu_fill_np`, uncapped by tiles-per-partition (idle CUs are worse
    #     than thin partitions). Measured: batch=1, ctx=16384 -> NP=8 to a
    #     CU-filling NP=32 is 2.5x faster; batch=3, ctx=1027 [5 tiles] ->
    #     NP=5 beats NP=1 by ~30%.
    #   - NOT CU-STARVED: further CTAs only add occupancy depth and reduce-
    #     kernel/prologue overhead, so cap tiles-per-partition to at least
    #     MIN_TILES_PER_PARTITION. Measured: batch=81, ctx=16384 [64 tiles] ->
    #     NP=5..8 all close, ~6-8% faster than NP=4; batch=81, ctx=1027
    #     [5 tiles] -> uncapped NP=8 measured 30% *slower* than NP=2 (same
    #     total CTAs, 648 vs 162, wasn't what mattered -- tiles-per-partition,
    #     1 vs 3, was).
    #
    # TARGET_CTAS_PER_CU=8 and MIN_TILES_PER_PARTITION=2 were picked by a
    # direct sweep on that hardware to land near every sweet spot above.
    from kernels.pa_decode_fp8 import get_recommended_splits

    TARGET_CTAS_PER_CU = 8
    MIN_TILES_PER_PARTITION = 2
    device_cus = torch.cuda.get_device_properties(query.device).multi_processor_count
    cu_fill_np = cdiv(TARGET_CTAS_PER_CU * device_cus, num_seqs * num_kv_heads)
    # Bounded by max_blocks_per_seq*block_size/TILE_TOK, NOT the actual
    # context length: reading `context_lengths` on the host forces a GPU
    # sync, illegal during CUDA graph capture. Exact when callers size
    # `block_tables` to the actual context length; looser (but still
    # correct) if they over-allocate for a larger max-sequence-length.
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
        # Pre-normalized (O_p/l_p) partials, matching what the reused
        # pa_decode_sw_reduce_kernel expects (see the main kernel's NP>1
        # store) -- kept in query's own dtype (f16/bf16) rather than forced
        # to bf16, since tile never has an fp8 query needing that precision
        # tradeoff.
        pout_dtype = torch.bfloat16 if query_dtype == "bf16" else torch.float16
        pout = torch.empty(num_seqs, num_kv_heads, num_partitions, GS, head_dim, dtype=pout_dtype, device=dev)
    s = stream or torch.cuda.current_stream()
    # K/V scales are read in-kernel via buffer_load (matching
    # pa_decode_ps_kernel), not a host kernarg -- a plain float still works
    # (wrapped into a 1-element tensor) or a caller-owned device tensor.
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
    if num_partitions > 1:
        # Reuse pa_decode_ps_kernel's own reduce kernel (kernels/pa_decode_swa.py)
        # instead of a custom implementation, called exactly the way
        # pa_decode_ps_launch itself calls it: raw data_ptr()s + explicit
        # strides, host-side (not from within pa_decode_tile_launch's own
        # trace, since this reduce kernel's calling convention is
        # raw-fx.Int64-pointer based, not fx.Tensor based like this file's
        # own kernels).
        from kernels.pa_decode_swa import compile_pa_decode_sw_reduce

        reduce_compiled = compile_pa_decode_sw_reduce(
            max_context_partition_num=num_partitions,
            query_seq_len=1,
            query_group_size=GS,
            head_size=head_dim,
            output_dtype_str="f16",
            logits_dtype_str=query_dtype,
        )
        reduce_compiled["launch"](
            output.data_ptr(),
            psum.data_ptr(),  # exp_sums
            pmax.data_ptr(),  # max_logits
            pout.data_ptr(),  # logits (already-normalized bf16 partials)
            output.stride(0),
            0,  # stride_output_len: unused (query_length==1, query_idx always 0)
            GS * output.stride(1),
            output.stride(1),
            pmax.stride(0),
            pmax.stride(1),
            pmax.stride(2),
            pout.stride(0),
            pout.stride(1),
            pout.stride(2),
            pout.stride(3),
            int(num_seqs),
            int(num_kv_heads),
            s,
        )
