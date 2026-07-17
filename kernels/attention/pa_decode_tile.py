# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

fp8: K/V stored as e4m3 (FNUZ on gfx942, OCP e4m3fn on gfx950 -- see ``FP8``
in ``compile_pa_decode_tile``), fed straight into ``mfma_f32_16x16x32_fp8_fp8``.
Q (bf16/f16) is quantized to fp8 with a per-row scale; softmax probabilities P
are quantized to fp8 for P·V. Scales fold out of the matmuls: ``q_scale`` and
``key_scale`` into the QK score; ``value_scale`` and the P dequant (1/FP8_MAX)
into the epilogue. Softmax max/sum stay f32.

``key_scale``/``value_scale`` support two modes, chosen by tensor rank (see
``pa_decode_tile()``): a ``[1]`` per-tensor scalar (default), or a
``[num_blocks, num_kv_heads, block_size]`` per-token tensor, matching
``pa_decode_ps_kernel``'s own ``per_token_kv`` mode. Per-token K-scale folds
into the QK score before the softmax max-reduce (not a positive constant, so
it can't be deferred like the per-tensor case); per-token V-scale is folded
into the softmax probabilities before fp8-packing, normalized by the tile's
own max V-scale to stay within fp8 range, and undone via a per-tile
correction factor afterwards.

Layouts are simple/logical (NOT production's preshuffle layout). ``block_size``
is a compile-time constant, 16 or 64 only. ``head_dim`` must be a multiple of
64 (64 or 128), matching production's own floor.

* ``query``        [num_seqs, num_q_heads, head_dim]                 f16/bf16
                   (rows/heads may be strided -- e.g. a slice of a combined
                    qkv tensor -- but head_dim must be contiguous)
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8 (see ``FP8``)
                   (SAME layout as ``_pa_small_block_load_k_flat`` in
                    ``pa_decode_fp8.py``: 16-element head-chunk outer, token
                    next-innermost, for coalesced dwordx4 loads)
* ``value_cache``  fp8 (see ``FP8``), either layout (detected from rank):
                   [num_blocks, num_kv_heads, block_size//16, head_dim, 16]
                   ("trans_v", SAME layout as
                   ``_pa_small_block_load_v_trans``) or the plain, un-shuffled
                   [num_blocks, num_kv_heads, head_dim, block_size]
* ``block_tables`` [num_seqs, max_blocks_per_seq]                    int32
                   (must cover ceil(context_len/256)*256/block_size pages,
                    not just ceil(context_len/block_size) -- the last tile
                    always issues a full 256-token-span load)
* ``context_lengths`` [num_seqs]                                     int32
* ``output``       [num_seqs, num_q_heads, head_dim]                 same dtype as ``query`` (f16/bf16)

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
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, vector
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.vector import ReductionOp
from flydsl.runtime.device import get_rocm_arch
from kernels.common import dpp_utils
from kernels.common.tensor_shim import _run_compiled
from kernels.common.utils import (
    cdiv,
    exp2_amdgcn_scalar,
    exp2_f32_fast,
    rcp_f32,
)

MFMA_MNK = 16  # M = N = 16 for the MMA atom; also query rows handled per CTA (padded to 16)
MFMA_K = 32  # fp8 MFMA contracts K = 32 per instruction (mfma_f32_16x16x32_fp8_fp8)
WAVE = 64
LOG2E = 1.4426950408889634


@functools.lru_cache(maxsize=None)
def compile_pa_decode_tile(
    *,
    head_dim: int,
    query_group_size: int,
    block_size: int,
    num_partitions: int = 1,
    softmax_scale: float | None = None,
    query_dtype: str = "f16",
    per_token_kv: bool = False,
    query_length: int = 1,
    trans_v: bool = True,
):
    """Build the tile-programming PA-decode kernel + launch wrapper.

    Returns a dict with ``launch`` and ``kernel`` entries, mirroring the
    ``compile_*`` factories in ``pa_decode_fp8.py``. ``block_size``,
    ``head_dim``, and ``query_dtype`` are all compile-time constants, not
    kernel arguments -- each distinct value gets its own compiled kernel via
    this function's ``lru_cache``.

    ``block_size`` (16 or 64) sets the K/V paged-gather's fixed block-table
    page fan-out per compute tile (``PAGES_PER_CHUNK`` below). ``head_dim``
    must be a multiple of 64. ``query_dtype`` (``"f16"``/``"bf16"``) selects
    the query element type. ``per_token_kv`` selects per-token (vs.
    per-tensor) K/V dequant scales -- see the module docstring. ``trans_v``
    selects the V-cache layout (see the module docstring); both layouts are
    one dwordx4 raw load per (16-token sub-block, head_dim element), only
    ``_v_ops``'s offset formula differs.

    ``query_length`` (MTP) and ``query_group_size > 16`` (wide GQA) both
    flatten into one ``TOTAL_ROWS = query_length * query_group_size``
    query-row axis, tiled into ``M_TILES = ceil(TOTAL_ROWS / 16)``
    independent 16-row MFMA tiles -- same mechanism ``pa_decode_ps_kernel``
    uses (its ``_mtp_groups``). Every configuration goes through this one
    row-tiled code path; each extra M-tile duplicates a full set of
    loop-carried softmax/output state, so VGPR/LDS/occupancy scale roughly
    linearly with ``M_TILES``.
    """
    is_gfx950 = "gfx95" in get_rocm_arch()
    FP8 = fx.Float8E4M3FN if is_gfx950 else fx.Float8E4M3FNUZ
    FP8_MAX = 448.0 if is_gfx950 else 240.0  # max representable magnitude of the format above

    assert head_dim % MFMA_MNK == 0, f"head_dim {head_dim} must be a multiple of {MFMA_MNK}"
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    assert query_dtype in (
        "f16",
        "bf16",
    ), f"pa_decode_tile only supports query_dtype in ('f16', 'bf16'), got {query_dtype}"
    Q_DTYPE = fx.BFloat16 if query_dtype == "bf16" else fx.Float16

    assert head_dim % 64 == 0, f"pa_decode_tile only supports head_dim that's a multiple of 64, got {head_dim}"
    assert query_length >= 1, f"query_length must be >= 1, got {query_length}"
    # Flattened query-row axis (MTP position outer, GQA head inner -- same
    # row-major convention as pa_decode_ps_kernel's own `_mtp_groups`),
    # tiled into independent 16-row MFMA M-tiles -- see this function's own
    # docstring.
    TOTAL_ROWS = query_length * query_group_size
    M_TILES = cdiv(TOTAL_ROWS, MFMA_MNK)
    ROWS_PADDED = M_TILES * MFMA_MNK
    # Single PV layout for every config: V=A, P=B -> output [head-dim (row),
    # query-row (col=lane16)] + direct-store epilogue. Generalizes over any
    # head_dim (multiple of 64) via the VHE_CHUNKS loop -- OP_ELEMS is always 4
    # (MFMA_MNK*VHE_SIZE/(NWARP*WAVE), VHE_SIZE==64 for all head_dim). There is
    # no alternate PV layout anymore.
    NWARP = 4  # 4 waves / CTA
    TOK_PER_WARP = 64  # tokens each warp owns per compute block (matches production KV_COMPUTE_BLOCK)
    TILE_TOK = NWARP * TOK_PER_WARP  # 256 tokens / compute block
    PAGES_PER_CHUNK = TOK_PER_WARP // block_size  # pages spanned by one 64-token warp-chunk: 1 (bs=64) or 4 (bs=16)
    assert head_dim % (NWARP * MFMA_MNK) == 0, "head_dim must split across the 4 warps for PV"

    # head_dim splits into a fixed 16-element chunk (QK_CHUNK_ELEMS, one
    # dwordx4 load), 4 of which (RGROUP_QUARTERS, `rgroup` == production's
    # `rowid`) make one 64-element fetch group; QKHE_LOOP is the fetch-group
    # count, scaling with head_dim.
    RGROUP_QUARTERS = 4
    QK_CHUNK_ELEMS = 16
    QKHE_LOOP = head_dim // (RGROUP_QUARTERS * QK_CHUNK_ELEMS)
    assert QKHE_LOOP >= 1, f"head_dim {head_dim} must be at least {RGROUP_QUARTERS * QK_CHUNK_ELEMS}"
    N_SUBCHUNKS = QKHE_LOOP * (QK_CHUNK_ELEMS // 8)

    # Q-quant chunk width: NQCHUNK stays fixed at 16 (tied to `lane16`'s role
    # as the absmax butterfly width); QCHUNK scales with head_dim instead.
    NQCHUNK = 16
    QCHUNK = head_dim // NQCHUNK  # f16 elements per lane's load chunk (8 for head_dim=128, 4 for head_dim=64)

    VHE_CHUNKS = head_dim // (NWARP * MFMA_MNK)  # 2 for head_dim=128, 1 for head_dim=64

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    NP = int(num_partitions)  # context partitions (grid.z); compile-time constant

    BLOCK_THREADS = NWARP * WAVE  # 256

    # ── LDS layout (shared across the 4 warps) ──
    # sQ: fp8[ROWS_PADDED,head_dim] staged+quantized query, all M-tiles.
    # sP: fp8[16,TILE_TOK] quantized probs, reused across M-tiles/KV-tiles
    # (each write is fully consumed before the next, via an existing barrier).
    # sQscale: f32[ROWS_PADDED]. sLmax/sLsum: transient cross-warp scratch,
    # reused across M-tiles. sVPage: V page-index broadcast. sKScale/sVScale/
    # sVScaleMax: per-token K/V scale staging. No sO/sM/sL/sCorr: the PV output
    # is register-resident/loop-carried and stored straight to global (V=A/P=B
    # swap), so no output staging, per-row correction, or reduce scratch is
    # needed in LDS.
    f32 = 4
    sQ_bytes = ROWS_PADDED * head_dim * 1  # fp8
    sP_off = sQ_bytes
    # +16B row padding: an unpadded 256B stride is a multiple of the
    # 32-bank*4B LDS wrap, so every (qh, l16g) P-pack write would hit the
    # same bank across all 16 qh. +16B is the smallest padding that breaks
    # the conflict while keeping the row 16B-aligned for PV's ds_read_b128.
    SP_ROW_BYTES = TILE_TOK + 16
    sP_bytes = MFMA_MNK * SP_ROW_BYTES  # fp8, padded rows
    sQscale_off = sP_off + sP_bytes
    # NWARP_PAD = NWARP+1 (not NWARP): a plain 16B stride wraps the 32-bank
    # LDS twice (row r and r+8 share a bank); 5 is coprime with 32 banks.
    NWARP_PAD = NWARP + 1
    # Phase-split for every M_TILES>1 shape (all block_size / head_dim): sLmax
    # gets a per-M-tile slice so all M-tiles' pass-1 (QK+mask+max-reduce) writes
    # share ONE barrier instead of M_TILES separate ones (see the phase-1/
    # phase-2 split below). M_TILES==1 has nothing to merge, so PHASE1_MTILES==1
    # naturally routes it to the single-slice per-m path.
    PHASE1_MTILES = M_TILES
    sLmax_off = sQscale_off + ROWS_PADDED * f32
    sLsum_off = sLmax_off + PHASE1_MTILES * MFMA_MNK * NWARP_PAD * f32
    # V page-table prefetch staging: warp w's row is broadcast here for all
    # 4 warps to read (V's page depends on `rgroup`, shared across warps).
    sVPage_off = sLsum_off + MFMA_MNK * NWARP_PAD * f32
    sVPage_bytes = NWARP * PAGES_PER_CHUNK * 4  # i32
    # Per-token K/V scale staging (per_token_kv only).
    #
    # DOUBLE-BUFFER (per_token phase-split only): the tt+1 prefetch stages
    # into the OTHER buffer (tt&1 ping-pong), so the current tile's v_scale
    # survives to Phase B and can be re-read there instead of held in
    # registers across the whole M-tile loop. That lets Phase A hold only
    # k_scale (16 VGPR) and Phase B only v_scale (16), vs the old single-
    # buffer path that had to hoist both (32) before the prefetch clobbered
    # them -- the 16 VGPR freed drops per_token M_TILES=4 under the 256/2-wave
    # cliff. Costs one extra buffer of LDS.
    KV_DOUBLE_BUF = per_token_kv and M_TILES > 1
    KV_BUFS = 2 if KV_DOUBLE_BUF else 1
    KV_BUF_STRIDE = 2 * NWARP * TOK_PER_WARP * f32  # k-region + v-region, one buffer
    sKScale_off = sVPage_off + sVPage_bytes
    sVScale_off = sKScale_off + NWARP * TOK_PER_WARP * f32
    sKVScale_bytes = KV_BUFS * KV_BUF_STRIDE if per_token_kv else 0
    sVScaleMax_off = sKScale_off + sKVScale_bytes
    # pv_max is m-independent (see the hoisted compute in the phase-split
    # path), so a single NWARP-wide cross-warp slot suffices.
    sVScaleMax_bytes = NWARP_PAD * f32 if per_token_kv else 0
    total_bytes = sVScaleMax_off + sVScaleMax_bytes

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_tile_kernel(
        output_ptr: fx.Tensor,  # [num_seqs*query_length, num_q_heads, head_dim]  (written directly when NP==1)
        # per-partition partial outputs (combined by the reduce kernel when NP>1):
        pmax_ptr: fx.Tensor,  # [num_seqs*query_length, num_kv_heads, num_partitions, query_group_size]   row max
        psum_ptr: fx.Tensor,  # [num_seqs*query_length, num_kv_heads, num_partitions, query_group_size]   row sum
        pout_ptr: fx.Tensor,  # [num_seqs*query_length, num_kv_heads, num_partitions, query_group_size, head_dim] Q_DTYPE, normalized O_p/l_p
        query_ptr: fx.Tensor,  # [num_seqs*query_length, num_q_heads, head_dim] -- row = seq*query_length + qi (MTP position)
        key_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, head_dim//16, block_size, 16] (blocked, see module docstring)
        value_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, block_size//16, head_dim, 16] (blocked, see module docstring)
        block_tables_ptr: fx.Tensor,  # [num_seqs, max_blocks_per_seq]
        context_lengths_ptr: fx.Tensor,  # [num_seqs]
        key_scale_ptr: fx.Tensor,  # [1] per-tensor OR [num_blocks, num_kv_heads, block_size] per-token
        value_scale_ptr: fx.Tensor,  # same shape as key_scale_ptr
        max_blocks_per_seq: fx.Int32,
        num_q_heads: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = tid // WAVE  # 0..NWARP-1
        lane = tid - warp * WAVE  # 0..63
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        part = fx.Int32(gpu.block_id("z"))  # context partition handled by this CTA
        n_kv = num_q_heads // query_group_size  # num_kv_heads

        # fx.copy-based K/V/Q/context_len loaders (fp8 is 1B/elem, so byte
        # offset == element index). K/V/context_len use UniversalCopy128b/32b
        # over a raw pointer; Q keeps the buffer-resource BufferCopy128b path.
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
        # 128 bits for QCHUNK=8 (head_dim=128), 64 bits for QCHUNK=4 (head_dim=64).
        _q_copy_op = fx.rocdl.BufferCopy128b() if QCHUNK == 8 else fx.rocdl.BufferCopy64b()
        _q_load_chunk = _make_flat_loader(query_ptr, Q_DTYPE, QCHUNK, _q_copy_op)
        _ctxlen_load = _make_flat_loader(context_lengths_ptr, fx.Int32, 1, fx.rocdl.BufferCopy32b())

        def _k_load16(byte_off):
            return _k_load_fp8x16(byte_off).bitcast(fx.Int64)

        def _v_load16(byte_off):
            return _v_load_fp8x16(byte_off).bitcast(fx.Int64)

        context_len = fx.Int32(_ctxlen_load(seq)[0])
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        # Per-tensor: a single global scale, read once. Per-token: read
        # per-token instead (see _kv_scale_ops/_stage_kv_scale_to_lds below).
        if const_expr(not per_token_kv):
            key_scale = fx.Int32(
                buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True)
            ).bitcast(fx.Float32)
            value_scale = fx.Int32(
                buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True)
            ).bitcast(fx.Float32)

        num_tiles = cdiv(context_len, TILE_TOK)
        tiles_per_part = cdiv(num_tiles, NP)
        part_start = part * tiles_per_part
        part_end_raw = part_start + tiles_per_part
        part_end = arith.select(part_end_raw < num_tiles, part_end_raw, num_tiles)

        # One i8 blob carved into typed views via byte-offset pointers.
        # Defined here so the V page-prefetch helpers below (issued before
        # the Q-quant barrier, alongside K's own prologue prefetch) can use it.
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
            result = buffer_ops.buffer_load(
                bt_rsrc, seq * max_blocks_per_seq + page, vec_width=vec_width, is_scalar=True
            )
            return fx.Int32(result) if vec_width == 1 else result

        def _v_page_fetch_and_stage(tt_i32):
            # V's page depends on `rgroup`, shared across all 4 warps, so
            # warp `w` fetches only the row for `rgroup == w` and broadcasts
            # it to LDS for every warp to read back (`_v_page_read_row`).
            # Prefetched one tile ahead; store/read-back straddle an
            # already-existing barrier, so no new barrier is needed.
            _, base_page = _tile_tok0_and_page(tt_i32)
            fetched = _load_phys_scalar(base_page + warp * PAGES_PER_CHUNK, PAGES_PER_CHUNK)
            if lane == 0:
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
            # One PAGES_PER_CHUNK-wide Vector (not a list) so the loop-carried
            # `if tt1 < part_end: v_page_next = _v_page_read_row()` below can
            # reassign it directly.
            off = sVPage_off + rgroup * (PAGES_PER_CHUNK * 4)
            return _view(off, fx.Int32, fx.make_layout(PAGES_PER_CHUNK, 1)).load()

        def _kv_buf_off(tt_val):
            # ping-pong buffer byte offset for the double-buffered KV-scale
            # staging (0 when single-buffered -- compile-time constant then).
            if const_expr(KV_DOUBLE_BUF):
                return (tt_val & fx.Int32(1)) * KV_BUF_STRIDE
            return 0

        def _stage_kv_scale_to_lds(phys_vec, buf_off=0):
            if const_expr(block_size == 64):
                phys = fx.Int32(phys_vec[0])
                base_tok = lane16 * NCHUNK
                scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + base_tok
                k_scale_vec = fx.Vector(buffer_ops.buffer_load(ks_rsrc, scale_idx, vec_width=NCHUNK, dtype=fx.Float32))
                v_scale_vec = fx.Vector(buffer_ops.buffer_load(vs_rsrc, scale_idx, vec_width=NCHUNK, dtype=fx.Float32))
                slot = (warp * TOK_PER_WARP + base_tok) * f32
                _view(sKScale_off + buf_off + slot, fx.Float32, fx.make_layout(NCHUNK, 1)).store(k_scale_vec)
                _view(sVScale_off + buf_off + slot, fx.Float32, fx.make_layout(NCHUNK, 1)).store(v_scale_vec)
            else:
                # block_size==16: NCHUNK=4 separate physical pages/tokens are
                # owned by this warp's 64 lanes, one per `rgroup` (0..3) x
                # `lane16` (0..15) -- previously this looped a compile-time
                # `a` over all 4 sub-blocks with EVERY lane executing all 4
                # iterations redundantly (the per-lane token/page only
                # depends on `a`/`lane16`, never `rgroup`), so all 4
                # `rgroup`-groups computed and wrote the identical 4 values
                # 4x over. Using `rgroup` itself (a dynamic per-thread lane
                # id, matching production's own `rowid`-selects-page
                # mechanism) to pick this thread's ONE page/token instead
                # covers all 4 sub-blocks in parallel across the 4
                # rgroup-groups, with a single load/store each instead of a
                # 4-iteration loop -- cuts both the redundant global
                # loads and the LDS store instruction count 4x -> 1x here.
                phys = fx.Int32(
                    vector.extract(arith.unwrap(phys_vec), static_position=[], dynamic_position=[fx.Index(rgroup)])
                )
                scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + lane16
                k_scale_scalar = fx.Float32(buffer_ops.buffer_load(ks_rsrc, scale_idx, vec_width=1, dtype=fx.Float32))
                v_scale_scalar = fx.Float32(buffer_ops.buffer_load(vs_rsrc, scale_idx, vec_width=1, dtype=fx.Float32))
                fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                slot = (warp * TOK_PER_WARP + rgroup * c16 + lane16) * f32
                _view(sKScale_off + buf_off + slot, fx.Float32, fx.make_layout(1, 1)).store(
                    fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32)
                )
                _view(sVScale_off + buf_off + slot, fx.Float32, fx.make_layout(1, 1)).store(
                    fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32)
                )

        def _load_scale_vec(base_off, a, buf_off=0):
            # This lane's 4 per-token scales for chunk `a` from an LDS scale
            # region (sKScale_off or sVScale_off), at ping-pong buffer buf_off.
            # Reads (not holds) let Phase A/B keep only one side live at a time.
            slot = (warp * TOK_CHUNK + a * c16 + rgroup * 4) * f32
            return _view(base_off + buf_off + slot, fx.Float32, fx.make_layout(4, 1)).load()

        def _load_kv_scale_vecs(a, buf_off=0):
            return _load_scale_vec(sKScale_off, a, buf_off), _load_scale_vec(sVScale_off, a, buf_off)

        # ── raw dwordx4 K load (A operand) ──
        # Contiguous-per-warp token assignment: token = warp*TOK_CHUNK +
        # a*c16 + lane16. Softmax's mask (_ct/base_tok_f below) and the
        # P-pack write position must encode this same formula.
        def _k_ops(phys, a):
            within_page_tok = (a * c16 + lane16) % block_size
            ops = []
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                base = (((phys * n_kv + kv_h) * QCHUNK + he_idx) * block_size + within_page_tok) * QK_CHUNK_ELEMS
                w = _k_load16(base)  # head[he_idx*16 : +16] -> k_step 2*qkhe, 2*qkhe+1
                if const_expr(block_size == 16):
                    # block_size==16 gathers up to PAGES_PER_CHUNK=4 separate
                    # physical pages per warp-chunk (vs. 1 for block_size==64)
                    # -- this hint helps the scheduler pipeline/overlap those
                    # extra loads, matching production's own small-block K
                    # loader for this same block_size.
                    fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                ops.extend([w[0], w[1]])
            return ops  # N_SUBCHUNKS i64 operands

        def _k_ops_flat(tt_i32):
            _, base_page = _tile_tok0_and_page(tt_i32)
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
            if const_expr(head_dim == 64):
                fx.rocdl.sched_vmem(len(flat) // 2)

            return fx.Vector.from_elements(flat, dtype=fx.Int64), phys_vec

        # ── prologue: prefetch the first tile's K ──
        num_tiles_m1 = num_tiles - 1
        start_safe = arith.select(part_start < num_tiles, part_start, num_tiles_m1)
        k_pf0, phys_vec0 = _k_ops_flat(start_safe)
        # V page-index prefetch, issued here too for the same overlap; the
        # LDS write is visible after the barrier below.
        _v_page_fetch_and_stage(start_safe)
        if const_expr(per_token_kv):
            _stage_kv_scale_to_lds(phys_vec0, _kv_buf_off(fx.Int32(start_safe)))

        # per_token_kv has no single global key_scale/value_scale: scale_qk
        # drops the key_scale factor (folded in per-token, see masked_chunks
        # below) and v_scale_f is unused (replaced by v_max_scaled).
        if const_expr(per_token_kv):
            scale_qk = fx.Float32(softmax_scale * LOG2E)
        else:
            scale_qk = fx.Float32(softmax_scale * LOG2E) * fx.Float32(key_scale)
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

        # f32[16, NWARP] cross-warp scratch: scalar write at (row, warp), vec read of a row's NWARP valid slots.
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

        qh_local = warp * 4 + rgroup  # 0..15: this thread's query row within an M-tile

        # Each M-tile quantizes 16 rows of the flattened (MTP position, GQA
        # head) axis: `flat_idx = m*16 + qh_local`, decomposed as
        # `qi = flat_idx // query_group_size`, `gs_head = flat_idx %
        # query_group_size` (same convention as `_mtp_groups`). No
        # cross-M-tile dependency, so no barriers needed between iterations.
        def _quant_q_row(m, qi, gs_head, q_row_off):
            qh0 = kv_h * query_group_size + gs_head
            row_byte0 = ((seq * query_length + qi) * stride_q_row + qh0 * stride_q_head) * 2  # 16-bit float = 2B/elem
            chunk_off = row_byte0 + lane16 * (QCHUNK * 2)
            q_chunk = _q_load_chunk(chunk_off // 2)  # byte offset -> element index

            local_absmax = fmath.absf(q_chunk).reduce(ReductionOp.MAX)
            absmax = local_absmax.to(fx.Float32)
            for sh in (8, 4, 2, 1):
                absmax = arith.maxnumf(absmax, dpp_utils.dpp_xor_f32(absmax, sh))

            q_scale = absmax * fx.Float32(1.0 / FP8_MAX)
            inv = fx.Float32(rcp_f32(arith.maxnumf(q_scale, fx.Float32(1e-20))))
            inv_b = fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(QCHUNK)

            q_scaled_chunk = q_chunk.to(fx.Float32) * inv_b
            _st_words(
                q_row_off + qh_local * head_dim + lane16 * QCHUNK,
                _f32_to_fp8_words(q_scaled_chunk),
            )
            if lane16 == 0:
                # Transposed [qh][m] (not [m][qh]) so the whole M_TILES-wide
                # row for a fixed qh is contiguous, letting the KV-loop read
                # it back in one wide load instead of M_TILES separate
                # narrow ones -- see the read site below.
                _st1(sQscale_off, qh_local * M_TILES + m, q_scale)

        for m in range_constexpr(M_TILES):
            flat_idx = m * MFMA_MNK + qh_local
            qi = flat_idx // query_group_size
            gs_head = flat_idx - qi * query_group_size
            q_row_off = m * MFMA_MNK * head_dim
            # `flat_idx < TOTAL_ROWS` is statically true for every lane except
            # possibly on the last M-tile (only it can be a partial tile), so
            # skip the runtime branch (and its EXEC-mask overhead) entirely
            # for every other M-tile.
            if const_expr((m + 1) * MFMA_MNK <= TOTAL_ROWS):
                _quant_q_row(m, qi, gs_head, q_row_off)
            elif flat_idx < TOTAL_ROWS:
                _quant_q_row(m, qi, gs_head, q_row_off)
            else:
                _st_words(
                    q_row_off + qh_local * head_dim + lane16 * QCHUNK,
                    fx.Vector.filled(QCHUNK // 4, 0, fx.Int32),
                )
                if lane16 == 0:
                    _st1(sQscale_off, qh_local * M_TILES + m, ZERO_F)

        gpu.barrier()

        # First tile's V page-index row, now visible after the barrier above
        # (the fetch+LDS-store was issued earlier, alongside k_pf0).
        v_page_pf0 = _v_page_read_row()

        # Q is the B operand, read raw from sQ once per M-tile and held in
        # registers. MUST use the exact same (qkhe, rgroup, qkr) -> head_dim
        # permutation as K's `_k_ops`.
        q_ops_all = []
        for m in range_constexpr(M_TILES):
            q_row_off = m * MFMA_MNK * head_dim
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                chunk = _view(
                    q_row_off + lane16 * head_dim + he_idx * QK_CHUNK_ELEMS, fx.Int64, fx.make_layout(2, 1)
                ).load()
                q_ops_all.extend([chunk[0], chunk[1]])
        # q_ops_all[m*N_SUBCHUNKS+s] for s=0..N_SUBCHUNKS-1, s = qkhe*2+qkr,
        # = M-tile m's head[he_idx*16+qkr*8 : +8] of qhead=lane16

        # QK in NCHUNK chunks of 4 tokens: each chunk yields a f32x4
        # C-fragment, so softmax processes 4 scores at a time (low VGPR peak).
        _ct = [
            fx.Vector.from_elements([float(a * c16 + r) for r in range_constexpr(4)]) for a in range_constexpr(NCHUNK)
        ]
        # P·V is loop-tiled over head-dim (like production's VHELOOP): each
        # step computes O[:, vh*VHE_SIZE:+VHE_SIZE] instead of materializing
        # the full [16, head_dim] at once.
        VHE_SIZE = head_dim // VHE_CHUNKS
        OP_ELEMS = MFMA_MNK * VHE_SIZE // (NWARP * WAVE)  # PV C-fragment elements/lane/chunk (probed = 4)

        # ── raw dwordx4 V load (B operand) ──
        # lane (rgroup) takes the contiguous token slice [rgroup*64:+64] for
        # its head. Either V layout keeps 16 tokens contiguous for a fixed
        # (page, head, head_elem), so both are one dwordx4 load per
        # (16-token sub-block, head_elem) -- only the offset formula differs
        # (trans_v=True has the sub-block index ahead of head_dim; False has
        # it as a `step*16` offset within head_elem's own block_size row).
        # A rgroup's 64-token run can span multiple block_size pages, so
        # `sub`/`step` below walk pages/16-token sub-blocks either way.
        NVOPS = TILE_TOK // MFMA_K  # 8 PV k_steps (256 tokens / K=32)
        STEPS_PER_PAGE = block_size // 16

        def _v_ops(phys_row, vh):
            head_group = ((vh * VHE_SIZE) // 16) + warp
            head_element = head_group * 16 + lane16
            ops = []
            for sub in range_constexpr(PAGES_PER_CHUNK):
                for step in range_constexpr(STEPS_PER_PAGE):
                    if const_expr(trans_v):
                        base = (((phys_row[sub] * n_kv + kv_h) * STEPS_PER_PAGE + step) * head_dim + head_element) * 16
                    else:
                        base = ((phys_row[sub] * n_kv + kv_h) * head_dim + head_element) * block_size + step * 16
                    w = _v_load16(base)
                    if const_expr(block_size == 16):
                        # Same rationale as `_k_ops`'s block_size==16 hint:
                        # `sub` walks PAGES_PER_CHUNK=4 separate physical
                        # pages per warp-chunk, so help the scheduler
                        # pipeline/overlap those loads.
                        fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                    ops.extend([w[0], w[1]])
            if const_expr(head_dim == 64):
                fx.rocdl.sched_vmem(len(ops) // 2)
            return ops  # NVOPS i64, the 64-token contiguous run for this head

        # query_length==1 implies qi_of_row==0 for every row (covers wide-GQA
        # too, not just M_TILES==1); plain `context_len` instead of a
        # `lane16`-derived expr keeps this wave-uniform -- `lane16` is
        # per-lane, so folding it in forces a live per-lane VGPR even though
        # it's compile-time-constant here (measured ~20 VGPR cost, 122->142
        # at head_dim=128, for no behavioral difference).
        if const_expr(query_length == 1):
            causal_bound = [context_len for _m in range_constexpr(M_TILES)]
        else:
            causal_bound = [
                context_len - (query_length - 1) + (m * MFMA_MNK + lane16) // query_group_size
                for m in range_constexpr(M_TILES)
            ]

        K_SLOT, V_SLOT = 0, 1

        def _o0_slot(m):
            return 2 + 4 * m

        def _o1_slot(m):
            return 2 + 4 * m + 1

        def _m_slot(m):
            return 2 + 4 * m + 2

        def _l_slot(m):
            return 2 + 4 * m + 3

        o_zero = fx.Vector.filled(OP_ELEMS, 0.0, fx.Float32)
        init_state = [k_pf0, v_page_pf0]
        for _m in range_constexpr(M_TILES):
            init_state.extend([o_zero, o_zero, NEG_INF, ZERO_F])
        for tt, ostate in range(part_start, part_end, 1, init=init_state):
            k_cur = ostate[K_SLOT]  # this tile's prefetched K, as one (NCHUNK*N_SUBCHUNKS,) i64 vector
            v_page_cur = ostate[V_SLOT]  # this tile's V pages, as one PAGES_PER_CHUNK-wide i32 vector
            tt = fx.Int32(tt)
            tok0, _ = _tile_tok0_and_page(tt)

            tt1 = tt + 1

            next_state = [None, None]  # slots 0/1 (K_SLOT/V_SLOT) filled in at m==0 below

            # k/v-scale-per-token doesn't depend on `m` (only on `warp`/`a`), so
            # the phase-split hoists it once (via KV_DOUBLE_BUF: k in Phase A,
            # v re-read in Phase B from the un-clobbered ping-pong buffer) instead
            # of re-reading per M-tile. The non-phase-split path (head64 / M1)
            # reads per-m directly (see below).
            cur_kv_buf = _kv_buf_off(tt)

            # V pages/data for this KV-tile iteration don't depend on `m`
            # either; hoist once for the phase-split (M_TILES>1, head_dim!=64,
            # any block_size) instead of reloading per M-tile (V loads are raw
            # global loads, so redundant reloads are costlier than LDS re-reads).
            v_vh_shared = None
            if const_expr(M_TILES > 1):
                v_vh_shared = [_v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)]

            # q_scale doesn't depend on `m` either; read this lane's whole
            # M_TILES-wide row in one shot (contiguous thanks to the
            # transposed [qh][m] sQscale layout) instead of M_TILES separate
            # narrow re-reads, one per m-tile's own pass below (phase-split).
            q_scale_vec = None
            if const_expr(M_TILES > 1):
                qh_for_scale = lane - (lane // c16) * c16
                q_scale_vec = _view(
                    sQscale_off + qh_for_scale * (M_TILES * f32), fx.Float32, fx.make_layout(M_TILES, 1)
                ).load()

            def _lmax_off_m(m):
                return sLmax_off + (m * MFMA_MNK * NWARP_PAD * f32 if const_expr(PHASE1_MTILES > 1) else 0)

            # Phase-split (PHASE1_MTILES==M_TILES>1, i.e. any M_TILES>1 shape):
            # split pass-1 (QK + mask + per-warp max-reduce) into its own loop
            # over every M-tile, writing each M-tile's own LDS slice, so all
            # M-tiles share ONE barrier instead of M_TILES separate ones
            # (fewer barrier-adjacent ds_read/ds_write sequences). Only
            # M_TILES==1 (nothing to merge) takes the single-barrier-per-m
            # loop below.
            if const_expr(PHASE1_MTILES > 1):
                masked_chunks_saved = [None] * M_TILES
                scale_saved = [None] * M_TILES

                # Double-buffered per_token: pv_max is m-independent (see the
                # unmask above), so compute it ONCE here from a brief v_scale
                # read (released before Phase A), then hold only k_scale across
                # the Phase A M-tile loop. v_scale is re-read after the barrier
                # for Phase B (the tt+1 prefetch wrote the OTHER buffer, so the
                # current tile's scales survive). Peak scale liveness 16, not 32.
                k_scale_shared = None
                if const_expr(per_token_kv):
                    v_scale_A = [_load_scale_vec(sVScale_off, a, cur_kv_buf) for a in range_constexpr(NCHUNK)]
                    pv_max = fx.Float32(0.0)
                    for a in range_constexpr(NCHUNK):
                        pv_max = arith.maxnumf(pv_max, v_scale_A[a].reduce(ReductionOp.MAX))
                    for sh in (16, 32):
                        pv_max = arith.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
                    _st_lw(sVScaleMax_off, 0, warp, pv_max)
                    k_scale_shared = [_load_scale_vec(sKScale_off, a, cur_kv_buf) for a in range_constexpr(NCHUNK)]

                for m in range_constexpr(M_TILES):
                    frag_Ss = []
                    for a in range_constexpr(NCHUNK):
                        acc = arith.constant_vector(0.0, T.f32x4)
                        for s in range_constexpr(N_SUBCHUNKS):
                            acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [k_cur[a * N_SUBCHUNKS + s], q_ops_all[m * N_SUBCHUNKS + s], acc, 0, 0, 0]
                            )
                        frag_Ss.append(fx.Vector(acc))

                    qh = lane - (lane // c16) * c16
                    l16g = lane // c16
                    scale = scale_qk * fx.Float32(q_scale_vec[m])
                    n_valid_tile = (causal_bound[m] - tok0).to(fx.Float32)
                    base_tok_f = fx.Int32(warp * TOK_CHUNK + l16g * 4).to(fx.Float32)
                    thr = fx.Vector.from_elements([n_valid_tile - base_tok_f], dtype=fx.Float32).broadcast_to(4)
                    neg4 = fx.Vector.filled(4, -1e30, fx.Float32)

                    if const_expr(per_token_kv):
                        scaled_frags = [frag_Ss[a] * k_scale_shared[a] for a in range_constexpr(NCHUNK)]
                    else:
                        scaled_frags = frag_Ss

                    masked_chunks = [(_ct[a] < thr).select(scaled_frags[a], neg4) for a in range_constexpr(NCHUNK)]

                    pm = fx.Float32(float("-inf"))
                    for a in range_constexpr(NCHUNK):
                        pm = arith.maxnumf(pm, masked_chunks[a].reduce(ReductionOp.MAX))
                    for sh in (16, 32):
                        pm = arith.maxnumf(pm, pm.shuffle_xor(sh, WAVE))
                    _st_lw(_lmax_off_m(m), qh, warp, pm * scale)

                    masked_chunks_saved[m] = masked_chunks
                    scale_saved[m] = scale

                # K/V/scale prefetch doesn't depend on the query row; issued
                # here (once, not per-m) so the V-page read below can reuse
                # this one barrier instead of a dedicated one.
                k_next = k_cur
                if tt1 < part_end:
                    k_next, phys_vec1 = _k_ops_flat(tt1)
                    _v_page_fetch_and_stage(tt1)
                    if const_expr(per_token_kv):
                        # Stage tt+1 into the OTHER ping-pong buffer so the
                        # current tile's v_scale (re-read below) is not clobbered.
                        _stage_kv_scale_to_lds(phys_vec1, _kv_buf_off(tt1))
                next_state[K_SLOT] = k_next

                gpu.barrier()

                v_page_next = v_page_cur
                if tt1 < part_end:
                    v_page_next = _v_page_read_row()
                next_state[V_SLOT] = v_page_next

                # Phase B v_scale: M_TILES>=4 re-reads per chunk (keeps one
                # 4-vec live) to stay under the 256/2-wave cliff; smaller
                # M_TILES are already occ 2, so they hold all NCHUNK (fewer
                # LDS reads). Both read the un-clobbered current buffer.
                v_scale_shared = None
                if const_expr(per_token_kv and M_TILES < 4):
                    v_scale_shared = [_load_scale_vec(sVScale_off, a, cur_kv_buf) for a in range_constexpr(NCHUNK)]

                for m in range_constexpr(M_TILES):
                    o_acc = [ostate[_o0_slot(m)], ostate[_o1_slot(m)]]
                    m_prev = ostate[_m_slot(m)]  # this thread's own running max, carried from last tile
                    l_prev = ostate[_l_slot(m)]  # this thread's own running denom, carried from last tile

                    qh = lane - (lane // c16) * c16
                    l16g = lane // c16
                    masked_chunks = masked_chunks_saved[m]
                    scale = scale_saved[m]

                    v_max_scaled = None
                    norm_factor_b = None
                    if const_expr(per_token_kv):
                        v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(ReductionOp.MAX)
                        v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                        v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                        norm_factor = fx.Float32(rcp_f32(v_max_safe))
                        norm_factor_b = fx.Vector.from_elements([norm_factor], dtype=fx.Float32).broadcast_to(4)

                    m_new = arith.maxnumf(m_prev, _ld_lw_row(_lmax_off_m(m), qh).reduce(ReductionOp.MAX))
                    m_new_b = fx.Vector.from_elements([m_new], dtype=fx.Float32).broadcast_to(4)
                    ls = fx.Float32(0.0)
                    words = []
                    zero4_p = fx.Vector.filled(4, 0.0, fx.Float32)
                    for a in range_constexpr(NCHUNK):
                        # Re-mask Pa itself (not just the pre-exp score): a tile
                        # with zero valid tokens for this row has every
                        # masked_chunks[a] lane at the -1e30 sentinel AND m_new
                        # derived from that same sentinel, so
                        # exp2(score*scale - m_new) cancels to exp2(0)==1 rather
                        # than ~0. Invalid lanes are exactly the sentinel ones
                        # (real scaled scores are >> -1e29), so force them to a
                        # true 0 contribution. Matters most for NP>1 empty
                        # partitions (an early MTP row whose causal window ends
                        # before this partition's tiles).
                        valid_a = masked_chunks[a] > fx.Vector.filled(4, -1e29, fx.Float32)
                        Pa = valid_a.select(fx.Vector(exp2_f32_fast(masked_chunks[a] * scale - m_new_b)), zero4_p)
                        ls = ls + Pa.reduce(ReductionOp.ADD)
                        if const_expr(per_token_kv):
                            v_sc = (
                                _load_scale_vec(sVScale_off, a, cur_kv_buf)
                                if const_expr(M_TILES >= 4)
                                else v_scale_shared[a]
                            )
                            p_scaled = Pa * v_sc * norm_factor_b
                        else:
                            p_scaled = Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32)
                        words.append(_f32_to_fp8_words(p_scaled)[0])

                    p_off0 = sP_off + qh * SP_ROW_BYTES + warp * TOK_CHUNK + l16g * 4
                    _view(p_off0, fx.Int32, fx.make_layout(NCHUNK, c16 // 4)).store(
                        fx.Vector.from_elements(words, dtype=fx.Int32)
                    )
                    for sh in (16, 32):
                        ls = ls.addf(ls.shuffle_xor(sh, WAVE), fastmath=arith.FastMathFlags.contract)
                    fm_contract = arith.FastMathFlags.contract
                    # PV output is [head-dim (row), query-row (col=lane16)]
                    # after the operand swap below, so the softmax
                    # correction and running denominator are single
                    # per-lane scalars keyed on query-row=lane16 -- no
                    # sCorr LDS round-trip, no per-output-row gather.
                    corr_reg = fx.Float32(exp2_amdgcn_scalar(m_prev - m_new))
                    if l16g == 0:
                        _st_lw(sLsum_off, qh, warp, ls)
                    gpu.barrier()
                    # Every lane needs the denominator for its own
                    # query-row=lane16 (redundant across warp/rgroup, but
                    # register-cheap and avoids the tid<16 special-case).
                    gsum = _ld_lw_row(sLsum_off, lane16).reduce(ReductionOp.ADD)
                    l_new = fx.Float32(
                        arith.mulf(arith.unwrap(l_prev), arith.unwrap(corr_reg), fastmath=fm_contract)
                    ).addf(gsum, fastmath=fm_contract)

                    p_ops = _view(
                        sP_off + lane16 * SP_ROW_BYTES + rgroup * 64,
                        fx.Int64,
                        fx.make_layout(NVOPS, 1),
                    ).load()

                    corr_b = fx.Vector.from_elements([corr_reg], dtype=fx.Float32).broadcast_to(OP_ELEMS)
                    for vh in range_constexpr(VHE_CHUNKS):
                        v_vh = v_vh_shared[vh]
                        acc = arith.constant_vector(0.0, T.f32x4)
                        for s in range_constexpr(NVOPS):
                            # SWAPPED operands (V=A, P=B): output row =
                            # head-dim, output col = query-row=lane16.
                            acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v_vh[s], p_ops[s], acc, 0, 0, 0])
                        op = fx.Vector(acc)
                        if const_expr(per_token_kv):
                            op = op * fx.Vector.from_elements([v_max_scaled], dtype=fx.Float32).broadcast_to(OP_ELEMS)
                        o_acc[vh] = o_acc[vh] * corr_b + op
                    next_state.extend([o_acc[0], o_acc[1], m_new, l_new])
                    # Force full retirement of this M-tile's masked_chunks/
                    # scale (saved across the phase-1/phase-2 barrier above,
                    # so already a real per-M-tile live range) before the
                    # next M-tile's own chain starts, instead of letting the
                    # scheduler interleave multiple M-tiles' chains for ILP
                    # at the cost of peak concurrent liveness.
                    if const_expr(m < M_TILES - 1):
                        fx.rocdl.sched_barrier(0)
            else:
                # M_TILES==1 single tile: loop-carried state is [K=0, V=1, o0=2,
                # o1=3, running-max=4, running-denom=5] (tile-0 of _o0_slot(m)=2+4*m,
                # _o1_slot=3+4*m, _m_slot=4+4*m, _l_slot=5+4*m).
                o_acc = [ostate[2], ostate[3]]
                m_prev = ostate[4]  # this thread's own running max, carried from last tile
                l_prev = ostate[5]  # this thread's own running denom, carried from last tile
                # QK: each NCHUNK chunk accumulates N_SUBCHUNKS k_steps into
                # one f32x4 C-fragment (D[token, qhead]), using this M-tile's
                # own Q operand.
                frag_Ss = []
                for a in range_constexpr(NCHUNK):
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for s in range_constexpr(N_SUBCHUNKS):
                        acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [k_cur[a * N_SUBCHUNKS + s], q_ops_all[s], acc, 0, 0, 0]
                        )
                    frag_Ss.append(fx.Vector(acc))
                # K/V/scale prefetch for tt+1, issued here so the V-page read
                # below can reuse the upcoming pass-1 barrier instead of a
                # dedicated one.
                k_next = k_cur
                if tt1 < part_end:
                    k_next, phys_vec1 = _k_ops_flat(tt1)
                    _v_page_fetch_and_stage(tt1)
                    if const_expr(per_token_kv):
                        _stage_kv_scale_to_lds(phys_vec1)
                next_state[K_SLOT] = k_next
                # Softmax: each lane owns ONE qhead (= lane%16); reduce its
                # tokens with a register reduce + shuffle_xor(16,32). Mask is
                # a scalar threshold (token < n_valid).
                qh = lane - (lane // c16) * c16  # qhead = lane % 16
                l16g = lane // c16  # 0..3 lane-group within the warp
                # This path only runs for M_TILES==1 (the phase-split covers
                # all M_TILES>1), so q_scale is read per-m directly -- the
                # hoisted q_scale_vec is phase-split-only.
                scale = scale_qk * _ld1(sQscale_off, qh)  # per-qhead positive score scale (M_TILES==1)
                n_valid_tile = (causal_bound[0] - tok0).to(fx.Float32)
                base_tok_f = fx.Int32(warp * TOK_CHUNK + l16g * 4).to(fx.Float32)
                thr = fx.Vector.from_elements([n_valid_tile - base_tok_f], dtype=fx.Float32).broadcast_to(4)
                neg4 = fx.Vector.filled(4, -1e30, fx.Float32)
                # per_token_kv: K-scale varies per token, so (unlike the
                # per-tensor `scale`, a positive constant applied AFTER the
                # max-reduce) it must be folded in BEFORE masking/max-reduce.
                v_scale_vecs = None
                if const_expr(per_token_kv):
                    v_scale_vecs = []
                    scaled_frags = []
                    for a in range_constexpr(NCHUNK):
                        k_scale_vec, v_scale_vec = _load_kv_scale_vecs(a)
                        v_scale_vecs.append(v_scale_vec)
                        scaled_frags.append(frag_Ss[a] * k_scale_vec)
                else:
                    scaled_frags = frag_Ss
                # Reused in pass 2 below, halving the mask instruction count.
                masked_chunks = [(_ct[a] < thr).select(scaled_frags[a], neg4) for a in range_constexpr(NCHUNK)]
                # pass 1: per-warp max for this qhead
                pm = fx.Float32(float("-inf"))
                for a in range_constexpr(NCHUNK):
                    pm = arith.maxnumf(pm, masked_chunks[a].reduce(ReductionOp.MAX))
                for sh in (16, 32):
                    pm = arith.maxnumf(pm, pm.shuffle_xor(sh, WAVE))
                # All 4 lanes sharing this qhead hold the identical
                # post-shuffle_xor `pm`, so this write is harmlessly redundant.
                _st_lw(sLmax_off, qh, warp, pm * scale)
                # per_token_kv: this warp's max V-scale, used only as a
                # per-tile fp8 normalization constant (P divided by it,
                # O multiplied back) -- any positive value is correct, so
                # skip the causal-mask select (real out-of-context v_scales
                # are normal-magnitude cache entries) and just max over all
                # NCHUNK v_scales. Reuses the pass-1 barrier below.
                if const_expr(per_token_kv):
                    pv_max = fx.Float32(0.0)
                    for a in range_constexpr(NCHUNK):
                        pv_max = arith.maxnumf(pv_max, v_scale_vecs[a].reduce(ReductionOp.MAX))
                    for sh in (16, 32):
                        pv_max = arith.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
                    _st_lw(sVScaleMax_off, 0, warp, pv_max)
                gpu.barrier()
                # Read back next tile's V page-index row now that the barrier
                # above made `_v_page_fetch_and_stage`'s store visible.
                v_page_next = v_page_cur
                if tt1 < part_end:
                    v_page_next = _v_page_read_row()
                next_state[V_SLOT] = v_page_next
                # per_token_kv: combine all 4 warps' max V-scale into this
                # tile's normalization factor; `v_max_scaled` also doubles as
                # the PV correction factor below, replacing the per-tensor
                # path's single `v_scale_f`.
                v_max_scaled = None
                norm_factor_b = None
                if const_expr(per_token_kv):
                    v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(ReductionOp.MAX)
                    v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                    v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                    norm_factor = fx.Float32(rcp_f32(v_max_safe))
                    norm_factor_b = fx.Vector.from_elements([norm_factor], dtype=fx.Float32).broadcast_to(4)
                # pass 2: global max over warps -> exp -> fp8 P pack (-> sP) -> sum
                m_new = arith.maxnumf(m_prev, _ld_lw_row(sLmax_off, qh).reduce(ReductionOp.MAX))
                m_new_b = fx.Vector.from_elements([m_new], dtype=fx.Float32).broadcast_to(4)
                ls = fx.Float32(0.0)
                # Raw i32-word store straight to sP[qhead][token_base:+4] (fp8,
                # 1B/elem): the packed word's 4 fp8 lanes are exactly the 4
                # consecutive tokens this lane owns in chunk `a`.
                words = []
                zero4_p = fx.Vector.filled(4, 0.0, fx.Float32)
                for a in range_constexpr(NCHUNK):
                    # See the phase-split path: re-mask Pa so a fully-masked
                    # chunk contributes exactly 0 despite the score/max
                    # sentinel cancellation to exp2(0)==1.
                    valid_a = masked_chunks[a] > fx.Vector.filled(4, -1e29, fx.Float32)
                    Pa = valid_a.select(fx.Vector(exp2_f32_fast(masked_chunks[a] * scale - m_new_b)), zero4_p)
                    ls = ls + Pa.reduce(ReductionOp.ADD)
                    if const_expr(per_token_kv):
                        v_scale_this = (
                            _load_scale_vec(sVScale_off, a) if const_expr(head_dim == 64) else v_scale_vecs[a]
                        )
                        p_scaled = Pa * v_scale_this * norm_factor_b
                    else:
                        p_scaled = Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32)
                    words.append(_f32_to_fp8_words(p_scaled)[0])
                p_off0 = sP_off + qh * SP_ROW_BYTES + warp * TOK_CHUNK + l16g * 4
                _view(p_off0, fx.Int32, fx.make_layout(NCHUNK, c16 // 4)).store(
                    fx.Vector.from_elements(words, dtype=fx.Int32)
                )
                if const_expr(head_dim == 64):
                    fx.rocdl.sched_dswr(NCHUNK)
                for sh in (16, 32):
                    ls = ls.addf(ls.shuffle_xor(sh, WAVE), fastmath=arith.FastMathFlags.contract)
                # PV (V=A, P=B) -> output [head-dim, query-row=lane16], so
                # the correction and denominator are single per-lane scalars
                # keyed on query-row=lane16 (no sCorr LDS round-trip), and
                # the epilogue stores O straight to global. Same layout as
                # the phase-split path; this non-split path serves block64
                # (any M) and M_TILES==1.
                corr_reg = fx.Float32(exp2_amdgcn_scalar(m_prev - m_new))
                if l16g == 0:
                    _st_lw(sLsum_off, qh, warp, ls)
                gpu.barrier()
                gsum = _ld_lw_row(sLsum_off, lane16).reduce(ReductionOp.ADD)
                l_new = fx.Float32(
                    arith.mulf(arith.unwrap(l_prev), arith.unwrap(corr_reg), fastmath=arith.FastMathFlags.contract)
                ).addf(gsum, fastmath=arith.FastMathFlags.contract)
                p_ops = _view(
                    sP_off + lane16 * SP_ROW_BYTES + rgroup * 64,
                    fx.Int64,
                    fx.make_layout(NVOPS, 1),
                ).load()
                corr_b = fx.Vector.from_elements([corr_reg], dtype=fx.Float32).broadcast_to(OP_ELEMS)
                # Single tile: no sibling M-tile chain to hide the V-load latency
                # behind, so batch both vh's V loads upfront (the 2nd hides behind
                # the 1st's MFMA chain).
                v_vh_batch = [_v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)]
                for vh in range_constexpr(VHE_CHUNKS):
                    v_vh = v_vh_batch[vh]
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for s in range_constexpr(NVOPS):
                        acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [v_vh[s], p_ops[s], acc, 0, 0, 0])
                    op = fx.Vector(acc)
                    if const_expr(per_token_kv):
                        op = op * fx.Vector.from_elements([v_max_scaled], dtype=fx.Float32).broadcast_to(OP_ELEMS)
                    o_acc[vh] = o_acc[vh] * corr_b + op
                next_state.extend([o_acc[0], o_acc[1], m_new, l_new])
            results = yield next_state
        o_final = results

        # Direct-store epilogue (no sO staging / no epilogue barrier):
        # after the PV operand swap each lane already holds
        # O[head-dim = vh*64 + warp*16 + rgroup*4 + v, query-row = lane16]
        # for one query-row, so it writes its own 4 head-dim values
        # straight to global -- exactly production's _store_partition_results.
        inv_fp8 = fx.Float32(1.0 / FP8_MAX)
        for m in range_constexpr(M_TILES):
            row = m * MFMA_MNK + lane16  # flat (mtp, gqa) query-row for this lane
            row_ok = (m + 1) * MFMA_MNK <= TOTAL_ROWS  # last tile may be partial
            l_row = o_final[_l_slot(m)]
            safe_l = arith.select(l_row > ZERO_F, l_row, fx.Float32(1.0))
            inv_l = fx.Float32(rcp_f32(safe_l))
            if const_expr(per_token_kv):
                o_scale = inv_l
            else:
                o_scale = fx.Float32(
                    arith.mulf(
                        arith.unwrap(inv_l),
                        arith.unwrap(v_scale_f * inv_fp8),
                        fastmath=arith.FastMathFlags.contract,
                    )
                )
            o_scale_b = fx.Vector.from_elements([o_scale], dtype=fx.Float32).broadcast_to(OP_ELEMS)
            qi_e = row // query_group_size
            gs_head_e = row - qi_e * query_group_size
            qh = kv_h * query_group_size + gs_head_e

            def _emit(o_norm, sub):
                if const_expr(NP == 1):
                    out_row = output_ptr[seq * query_length + qi_e, qh, None]
                    out_chunk = fx.slice(fx.logical_divide(out_row, fx.make_layout(OP_ELEMS, 1)), (None, sub))
                    out_chunk.store(o_norm)
                else:
                    base = ((seq * n_kv + kv_h) * NP + part) * TOTAL_ROWS + row
                    pout_div = fx.logical_divide(pout_ptr, fx.make_layout(OP_ELEMS, 1))
                    pout_chunk = fx.slice(pout_div, (None, base * (head_dim // OP_ELEMS) + sub))
                    pout_chunk.store(o_norm)

            for vh in range_constexpr(VHE_CHUNKS):
                o_slot = _o0_slot(m) if vh == 0 else _o1_slot(m)
                o_norm = (o_final[o_slot] * o_scale_b).to(Q_DTYPE)
                head_base = vh * (NWARP * MFMA_MNK) + warp * MFMA_MNK + rgroup * OP_ELEMS
                sub = head_base // OP_ELEMS
                if row_ok:
                    _emit(o_norm, sub)
                else:
                    if row < TOTAL_ROWS:
                        _emit(o_norm, sub)

            if const_expr(NP > 1):
                if warp == 0 and rgroup == 0:
                    base = ((seq * n_kv + kv_h) * NP + part) * TOTAL_ROWS + row
                    if row_ok:
                        pmax_ptr[base] = o_final[_m_slot(m)]
                        psum_ptr[base] = l_row
                    else:
                        if row < TOTAL_ROWS:
                            pmax_ptr[base] = o_final[_m_slot(m)]
                            psum_ptr[base] = l_row

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
        key_scale: fx.Tensor,  # [1] per-tensor OR [num_blocks, num_kv_heads, block_size] per-token
        value_scale: fx.Tensor,  # same shape as key_scale
        max_blocks_per_seq: fx.Int32,
        num_q_heads: fx.Int32,
        num_seqs: fx.Int32,
        num_kv_heads: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
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
            stride_ks_block,
            stride_ks_head,
            stride_q_row,
            stride_q_head,
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
    *,
    num_partitions: int | None = None,
    pmax: torch.Tensor | None = None,
    psum: torch.Tensor | None = None,
    pout: torch.Tensor | None = None,
) -> None:
    """Host entry point. See module docstring for the expected tensor layouts.

    ``num_partitions``/``pmax``/``psum``/``pout`` are optional overrides for
    callers (like ``pa_decode_ps_launch``) that manage their own partition
    count and intermediate buffers -- e.g. to keep them consistent across a
    CUDA graph capture and its replays, where nothing may be allocated
    on-the-fly. ``pmax``/``psum`` are ``[num_seqs, num_kv_heads,
    num_partitions, query_length*query_group_size]`` float32, and ``pout``
    is the same shape plus a trailing ``head_dim`` axis, dtype matching
    ``output``. When omitted, this function picks/allocates them itself
    exactly as before (and will refuse to do so mid-capture).
    """
    num_seqs = context_lengths.shape[0]
    total_q_rows, num_q_heads, head_dim = query.shape
    assert (
        total_q_rows % num_seqs == 0
    ), f"query.shape[0] ({total_q_rows}) must be a multiple of context_lengths.shape[0] ({num_seqs})"
    query_length = total_q_rows // num_seqs
    _, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape

    assert num_hgroups == head_dim // 16 and hgroup_width == 16
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"

    trans_v = value_cache.dim() == 5
    if trans_v:
        _, v_num_kv_heads, v_subblocks, v_head_dim, v_width = value_cache.shape
        assert (
            v_head_dim == head_dim and v_width == 16 and v_subblocks == block_size // 16
        ), f"value_cache shape {tuple(value_cache.shape)} doesn't match block_size={block_size}, head_dim={head_dim}"
    else:
        _, v_num_kv_heads, v_head_dim, v_block_size = value_cache.shape
        assert v_head_dim == head_dim and v_block_size == block_size, (
            f"value_cache shape {tuple(value_cache.shape)} doesn't match "
            f"block_size={block_size}, head_dim={head_dim}"
        )
    assert v_num_kv_heads == num_kv_heads
    assert block_tables.dtype == torch.int32, f"block_tables must be int32, got {block_tables.dtype}"
    assert context_lengths.dtype == torch.int32, f"context_lengths must be int32, got {context_lengths.dtype}"
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]
    if query.dtype == torch.bfloat16:
        query_dtype = "bf16"
    elif query.dtype == torch.float16:
        query_dtype = "f16"
    else:
        raise ValueError(f"pa_decode_tile only supports f16/bf16 query, got {query.dtype}")
    assert (
        output.dtype == query.dtype
    ), f"pa_decode_tile requires output.dtype == query.dtype, got {output.dtype} vs {query.dtype}"

    assert query.stride(2) == 1, f"pa_decode_tile requires a contiguous head_dim axis, got strides {query.stride()}"

    dev = query.device
    key_scale_t = key_scale if isinstance(key_scale, torch.Tensor) else torch.tensor([float(key_scale)], device=dev)
    value_scale_t = (
        value_scale if isinstance(value_scale, torch.Tensor) else torch.tensor([float(value_scale)], device=dev)
    )
    per_token_kv = key_scale_t.dim() > 1
    if per_token_kv:
        assert value_scale_t.dim() > 1, "value_scale must also be per-token (dim>1) when key_scale is per-token"
        assert (
            key_scale_t.shape == value_scale_t.shape
        ), f"key_scale/value_scale shape mismatch: {tuple(key_scale_t.shape)} vs {tuple(value_scale_t.shape)}"
        assert key_scale_t.shape == (key_cache.shape[0], num_kv_heads, block_size), (
            "per-token key_scale/value_scale must be [num_blocks, num_kv_heads, block_size] "
            f"matching the KV cache, got {tuple(key_scale_t.shape)}"
        )
        stride_ks_block = int(key_scale_t.stride(0))
        stride_ks_head = int(key_scale_t.stride(1))
    else:
        stride_ks_block = 0
        stride_ks_head = 0
    assert (
        key_scale_t.dtype == torch.float32 and key_scale_t.device == dev
    ), f"key_scale tensor must be float32 on {dev}, got {key_scale_t.dtype} on {key_scale_t.device}"
    assert (
        value_scale_t.dtype == torch.float32 and value_scale_t.device == dev
    ), f"value_scale tensor must be float32 on {dev}, got {value_scale_t.dtype} on {value_scale_t.device}"

    if num_partitions is None:
        from kernels.attention.pa_decode_fp8 import get_recommended_splits

        num_partitions = get_recommended_splits(
            num_seqs, num_kv_heads, max_blocks_per_seq=max_blocks_per_seq, block_size=block_size, device=dev
        )

    compiled = compile_pa_decode_tile(
        head_dim=head_dim,
        query_group_size=query_group_size,
        block_size=int(block_size),
        num_partitions=num_partitions,
        softmax_scale=softmax_scale,
        query_dtype=query_dtype,
        per_token_kv=per_token_kv,
        query_length=query_length,
        trans_v=trans_v,
    )
    from kernels.attention.pa_decode_fp8 import _is_current_stream_capturing

    is_graph_capturing = _is_current_stream_capturing()
    if num_partitions == 1:
        # NP==1 fast path writes output directly; partials are unused (dead
        # code), so caller-provided pmax/psum/pout (if any) are ignored.
        if pmax is None:
            if is_graph_capturing:
                raise ValueError(
                    "CUDA graph capture requires preallocated `pmax`/`psum`/`pout` "
                    "even when num_partitions==1 (nothing may be allocated mid-capture)."
                )
            pmax = psum = pout = torch.empty(1, dtype=torch.float32, device=dev)
    else:
        total_rows = query_length * query_group_size
        expected_scalar_shape = (num_seqs, num_kv_heads, num_partitions, total_rows)
        if pmax is None or psum is None or pout is None:
            if is_graph_capturing:
                raise ValueError(
                    "CUDA graph capture requires preallocated `pmax`/`psum`/`pout` "
                    "for num_partitions>1 (nothing may be allocated mid-capture)."
                )
            pmax = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
            psum = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
            pout = torch.empty(*expected_scalar_shape, head_dim, dtype=output.dtype, device=dev)
        else:
            assert pmax.shape == expected_scalar_shape, f"pmax shape {tuple(pmax.shape)} != {expected_scalar_shape}"
            assert psum.shape == expected_scalar_shape, f"psum shape {tuple(psum.shape)} != {expected_scalar_shape}"
            assert pout.shape == (
                *expected_scalar_shape,
                head_dim,
            ), f"pout shape {tuple(pout.shape)} != {(*expected_scalar_shape, head_dim)}"
    s = stream or torch.cuda.current_stream()

    _run_compiled(
        compiled["launch"],
        output,
        pmax.view(-1),
        psum.view(-1),
        pout.view(-1),
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        key_scale_t,
        value_scale_t,
        int(max_blocks_per_seq),
        int(num_q_heads),
        int(num_seqs),
        int(num_kv_heads),
        stride_ks_block,
        stride_ks_head,
        int(query.stride(0)),
        int(query.stride(1)),
        s,
    )
    if num_partitions > 1:
        from kernels.attention.pa_decode_fp8 import _get_output_dtype_str
        from kernels.attention.pa_decode_swa import compile_pa_decode_sw_reduce

        reduce_compiled = compile_pa_decode_sw_reduce(
            max_context_partition_num=num_partitions,
            query_seq_len=query_length,
            query_group_size=query_group_size,
            head_size=head_dim,
            output_dtype_str=_get_output_dtype_str(output),
            logits_dtype_str=_get_output_dtype_str(pout),
        )
        _run_compiled(
            reduce_compiled["launch"],
            output.data_ptr(),
            psum.data_ptr(),  # exp_sums
            pmax.data_ptr(),  # max_logits
            pout.data_ptr(),  # logits (already-normalized query_dtype partials)
            query_length * output.stride(0),  # stride_output_bs: per TRUE seq, spans all query_length rows
            output.stride(0),  # stride_output_len: per MTP position (query_length==1: unused, multiplied by 0)
            query_group_size * output.stride(1),
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
