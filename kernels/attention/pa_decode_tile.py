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
``[num_blocks, num_kv_heads, block_size]`` per-token tensor -- one dequant
scale per physical KV token, matching ``pa_decode_ps_kernel``'s own
``per_token_kv`` mode. Per-token K-scale folds into the QK score before the
softmax max-reduce (it isn't a positive constant, so it can't be deferred
like the per-tensor case); per-token V-scale can't be factored out of the PV
sum after MFMA, so it's instead folded into the softmax probabilities before
they're fp8-packed, normalized by the tile's own max V-scale so the packed
values stay within fp8 range, and undone via a per-tile correction factor
afterwards.

Layouts are simple/logical (NOT production's preshuffle layout). ``block_size``
is a compile-time constant, 16 or 64 only (the K/V gather unrolls a fixed page
fan-out per 256-token compute tile at trace time). ``head_dim`` must be a
multiple of 64 (64 or 128), matching production's own floor.

* ``query``        [num_seqs, num_q_heads, head_dim]                 f16/bf16
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8 (see ``FP8``)
                   (SAME layout as ``_pa_small_block_load_k_flat`` in
                    ``pa_decode_fp8.py``: 16-element head-chunk outer, token
                    next-innermost, for coalesced dwordx4 loads -- see
                    ``QKHE_LOOP`` below)
* ``value_cache``  [num_blocks, num_kv_heads, block_size//16, head_dim, 16]  fp8 (see ``FP8``)
                   (SAME "trans_v" layout as ``_pa_small_block_load_v_trans``:
                    16 consecutive tokens innermost, unlike K)
* ``block_tables`` [num_seqs, max_blocks_per_seq]                    int32
                   (must cover ceil(context_len/256)*256/block_size pages --
                    rounded UP to the 256-token tile granularity, not just
                    ceil(context_len/block_size), since the last tile always
                    issues a full-span load)
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
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
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
    since the raw dwordx4-load addressing has a 64-element minimum granularity.

    ``query_dtype`` (``"f16"`` or ``"bf16"``) selects the query tensor's
    16-bit float element type.

    ``per_token_kv`` selects per-token (vs. per-tensor) K/V dequant scales;
    see the module docstring and ``pa_decode_tile()``'s own docstring for the
    expected ``key_scale``/``value_scale`` tensor shape in that mode.
    """
    # The native fp8 MFMA operand format differs by chip generation: gfx942
    # (CDNA3) uses e4m3 FNUZ, gfx950 (CDNA4) uses OCP e4m3fn -- same convention
    # as preshuffle_gemm.py/qk_norm_rope_quant.py. Feeding FNUZ-encoded bits
    # into gfx950's fp8 MFMA (which decodes them as OCP) silently produces
    # wrong results rather than NaN.
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
    HEAD = head_dim
    GS = query_group_size
    NWARP = 4  # 4 waves / CTA
    TOK_PER_WARP = 64  # tokens each warp owns per compute block (matches production KV_COMPUTE_BLOCK)
    TILE_TOK = NWARP * TOK_PER_WARP  # 256 tokens / compute block
    PAGES_PER_CHUNK = TOK_PER_WARP // block_size  # pages spanned by one 64-token warp-chunk: 1 (bs=64) or 4 (bs=16)
    assert HEAD % (NWARP * MFMA_MNK) == 0, "head_dim must split across the 4 warps for PV"

    # head_dim-derived QK chunking: the fp8 MFMA operand is 8 fp8 elements (one i64) per lane per instruction. head_dim splits into a fixed 16-element chunk (QK_CHUNK_ELEMS, one dwordx4 load), 4 of which (RGROUP_QUARTERS, `rgroup` == production's `rowid`) make one 64-element fetch group; QKHE_LOOP is the fetch-group count and scales with head_dim. head_dim
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

    VHE_CHUNKS = HEAD // (NWARP * MFMA_MNK)  # 2 for HEAD=128, 1 for HEAD=64

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD**0.5)
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
    sQ_bytes = MFMA_MNK * HEAD * 1  # fp8
    sP_off = sQ_bytes
    # sP's per-qhead row is padded 16B past TILE_TOK: an unpadded 256B stride
    # is a multiple of the 32-bank*4B LDS wrap, so every (qh, l16g) P-pack
    # write hits the same bank across all 16 qh values. +16B is the smallest
    # padding that both breaks the conflict and keeps the row 16B-aligned for
    # the PV read's ds_read_b128 loads (+8B also breaks it but misaligns
    # those loads, measuring slower overall).
    SP_ROW_BYTES = TILE_TOK + 16
    sP_bytes = MFMA_MNK * SP_ROW_BYTES  # fp8, padded rows (only the first TILE_TOK bytes/row hold real data)
    sO_off = sP_off + sP_bytes
    sO_bytes = MFMA_MNK * HEAD * f32
    sM_off = sO_off + sO_bytes
    sL_off = sM_off + MFMA_MNK * f32
    sCorr_off = sL_off + MFMA_MNK * f32
    sQscale_off = sCorr_off + MFMA_MNK * f32  # per-row query dequant scale
    # Cross-warp reduction scratch: per (query row, warp) local max/sum. Row
    # stride padded to NWARP+1 (not NWARP), since a plain 16-byte stride
    # wraps the 32-bank LDS twice (row r and r+8 share a bank); 5 is coprime
    # with 32 banks, avoiding the conflict.
    NWARP_PAD = NWARP + 1
    sLmax_off = sQscale_off + MFMA_MNK * f32
    sLsum_off = sLmax_off + MFMA_MNK * NWARP_PAD * f32
    # V page-table prefetch staging: warp w's PAGES_PER_CHUNK-wide row (fetched
    # via one scalar wide load) is broadcast here for all 4 warps to read (V's
    # page depends on `rgroup`, which is shared across warps -- see `_v_page`).
    sVPage_off = sLsum_off + MFMA_MNK * NWARP_PAD * f32
    sVPage_bytes = NWARP * PAGES_PER_CHUNK * 4  # i32
    total_bytes = sVPage_off + sVPage_bytes
    # Per-token K/V dequant scale staging (per_token_kv only): NWARP*
    # TOK_PER_WARP floats each for K and V, laid out by (warp,
    # token-in-warp) -- the same addressing K/V data itself uses -- then
    # re-read as 4-wide per-(a, l16g) vectors matching the QK/P token
    # layout (_ct's own contiguous-per-warp formula). K/V share one
    # physical-page lookup (same token, different scale tensor).
    sKScale_off = sO_off
    sVScale_off = sKScale_off + NWARP * TOK_PER_WARP * f32
    sKVScale_bytes = 2 * NWARP * TOK_PER_WARP * f32 if per_token_kv else 0
    # Cross-warp v_scale-max reduction scratch (per_token_kv only): one
    # f32 per warp (v_scale doesn't depend on qhead), reusing
    # _st_lw/_ld_lw_row's existing NWARP_PAD padding/bank-conflict fix.
    sVScaleMax_off = sKScale_off + sKVScale_bytes
    sVScaleMax_bytes = NWARP_PAD * f32 if per_token_kv else 0
    assert sVScaleMax_off + sVScaleMax_bytes <= sO_off + sO_bytes, (
        "per-token K/V scale staging (aliased into sO's LDS range) overflows sO_bytes -- "
        f"needs {sVScaleMax_off + sVScaleMax_bytes - sO_off} bytes, sO only has {sO_bytes}"
    )

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_tile_kernel(
        output_ptr: fx.Tensor,  # [num_seqs, num_q_heads, HEAD]  (written directly when NP==1)
        # per-partition partial outputs (combined by the reduce kernel when NP>1):
        pmax_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS]   row max
        psum_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS]   row sum
        pout_ptr: fx.Tensor,  # [num_seqs, num_kv_heads, num_partitions, GS, HEAD] Q_DTYPE, normalized O_p/l_p
        query_ptr: fx.Tensor,  # [num_seqs, num_q_heads, HEAD]
        key_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, HEAD//16, block_size, 16] (blocked, see module docstring)
        value_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, block_size//16, HEAD, 16] (blocked, see module docstring)
        block_tables_ptr: fx.Tensor,  # [num_seqs, max_blocks_per_seq]
        context_lengths_ptr: fx.Tensor,  # [num_seqs]
        key_scale_ptr: fx.Tensor,  # [1] per-tensor OR [num_blocks, num_kv_heads, block_size] per-token
        value_scale_ptr: fx.Tensor,  # same shape as key_scale_ptr
        max_blocks_per_seq: fx.Int32,
        num_q_heads: fx.Int32,
        # Per-token K/V scale strides (per_token_kv only), layout
        # [num_blocks, num_kv_heads, block_size]; both 0 for per-tensor.
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = tid // WAVE  # 0..NWARP-1
        lane = tid - warp * WAVE  # 0..63
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        part = fx.Int32(gpu.block_id("z"))  # context partition handled by this CTA
        n_kv = num_q_heads // GS  # num_kv_heads

        # fx.copy-based K/V/Q/context_len loaders, indexed by the same raw
        # byte/element offset the raw-pointer loaders already compute (fp8 is
        # 1B/elem, so byte offset == element index). K/V/context_len use
        # UniversalCopy128b/32b over a plain raw pointer (no buffer-resource
        # descriptor); Q keeps the buffer-resource BufferCopy128b path --
        # `copy_op`'s type decides which.
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
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        # Per-tensor: a single global scale, read once here. Per-token: no
        # single global value exists -- key_scale/value_scale are read
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
        loop_start = fx.Int64(part_start)
        loop_end = fx.Int64(part_end)

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

        # ── per-token K/V dequant scale (per_token_kv only) ──
        # Same physical-page lookup as K's own `_k_ops_flat` (a fresh scalar
        # buffer_load, not shared with K's own fetch, to keep this self
        # contained); `within_page_tok` matches `_k_ops`'s own formula so
        # this lane's scale corresponds to the SAME token its own K load
        # covers. key_scale/value_scale share one [num_blocks, num_kv_heads,
        # block_size] layout, hence one shared `scale_idx`.
        def _kv_scale_ops(phys, a):
            within_page_tok = (a * c16 + lane16) % block_size
            scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + within_page_tok
            k_scale_scalar = fx.Float32(buffer_ops.buffer_load(ks_rsrc, scale_idx, vec_width=1, dtype=fx.Float32))
            v_scale_scalar = fx.Float32(buffer_ops.buffer_load(vs_rsrc, scale_idx, vec_width=1, dtype=fx.Float32))
            return k_scale_scalar, v_scale_scalar

        def _stage_kv_scale_to_lds(phys_vec):
            if const_expr(block_size == 64):
                phys = fx.Int32(phys_vec[0])
                base_tok = lane16 * NCHUNK
                scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + base_tok
                k_scale_vec = fx.Vector(buffer_ops.buffer_load(ks_rsrc, scale_idx, vec_width=NCHUNK, dtype=fx.Float32))
                v_scale_vec = fx.Vector(buffer_ops.buffer_load(vs_rsrc, scale_idx, vec_width=NCHUNK, dtype=fx.Float32))
                slot = (warp * TOK_PER_WARP + base_tok) * f32
                _view(sKScale_off + slot, fx.Float32, fx.make_layout(NCHUNK, 1)).store(k_scale_vec)
                _view(sVScale_off + slot, fx.Float32, fx.make_layout(NCHUNK, 1)).store(v_scale_vec)
            else:
                loaded = [
                    _kv_scale_ops(fx.Int32(phys_vec[(a * c16) // block_size]), a) for a in range_constexpr(NCHUNK)
                ]
                for a in range_constexpr(NCHUNK):
                    k_scale_scalar, v_scale_scalar = loaded[a]
                    slot = (warp * TOK_PER_WARP + a * c16 + lane16) * f32
                    _view(sKScale_off + slot, fx.Float32, fx.make_layout(1, 1)).store(
                        fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32)
                    )
                    _view(sVScale_off + slot, fx.Float32, fx.make_layout(1, 1)).store(
                        fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32)
                    )

        def _load_kv_scale_vecs(a):
            slot = (warp * TOK_CHUNK + a * c16 + rgroup * 4) * f32
            k_scale_vec = _view(sKScale_off + slot, fx.Float32, fx.make_layout(4, 1)).load()
            v_scale_vec = _view(sVScale_off + slot, fx.Float32, fx.make_layout(4, 1)).load()
            return k_scale_vec, v_scale_vec

        def _load_v_scale_vec(a):
            # V-only re-read of `_load_kv_scale_vecs`'s own LDS slot, used by
            # the P-pack loop far below instead of holding NCHUNK 4-wide
            # `v_scale_vecs` live across the whole intervening pass-1
            # barrier/V-page-read/v_max_scaled section -- trades one cheap
            # LDS read per chunk for a much shorter register live range
            # (was costing 16 VGPR held idle across that whole span).
            slot = (warp * TOK_CHUNK + a * c16 + rgroup * 4) * f32
            return _view(sVScale_off + slot, fx.Float32, fx.make_layout(4, 1)).load()

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
            if const_expr(HEAD == 64):
                # Groups the raw K loads into one scheduling unit for the
                # post-RA scheduler (zero VGPR cost); measured faster at
                # head_dim=64, not re-verified at head_dim=128.
                fx.rocdl.sched_vmem(len(flat) // 2)
            # `phys_vec` is also returned so per_token_kv's own
            # `_stage_kv_scale_to_lds` (same tile, same page formula) can
            # reuse it instead of re-issuing the block-table lookup.
            return fx.Vector.from_elements(flat, dtype=fx.Int64), phys_vec

        # ── prologue: prefetch the first tile's K ──
        num_tiles_m1 = num_tiles - 1
        start_safe = arith.select(part_start < num_tiles, part_start, num_tiles_m1)
        k_pf0, phys_vec0 = _k_ops_flat(start_safe)
        # V page-index prefetch, issued here too for the same overlap; the
        # LDS write is visible after the barrier below.
        _v_page_fetch_and_stage(start_safe)
        if const_expr(per_token_kv):
            _stage_kv_scale_to_lds(phys_vec0)

        # ── per-CTA scalar constants ──
        # softmax_scale and key_scale fold into the score tile; log2e folds into
        # the exp2 used for softmax. per_token_kv has no single global
        # key_scale/value_scale -- scale_qk drops the key_scale factor (it's
        # folded in per-token instead, see masked_chunks below) and v_scale_f
        # is unused (replaced by the per-tile v_max_scaled correction).
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

        # QK: D[token, qhead] = K(A) · Q(B)ᵀ, tiled (NWARP,1,1) splits tokens (M)
        # across the 4 warps — so the softmax reduces over M (tokens) cheaply.
        # PV: O[qhead, head_dim], tiled (1,NWARP,1) splits head_dim (N).
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_MNK, MFMA_MNK, MFMA_K, FP8))
        tiled_mma_pv = fx.make_tiled_mma(mma_atom, fx.make_layout((1, NWARP, 1), (0, 1, 0)))

        qh_local = warp * 4 + rgroup  # 0..15: this thread's query row

        if qh_local < GS:
            qh0 = kv_h * GS + qh_local
            row_byte0 = (seq * num_q_heads + qh0) * (HEAD * 2)  # 16-bit float = 2B/elem
            chunk_off = row_byte0 + lane16 * (QCHUNK * 2)
            q_chunk = _q_load_chunk(chunk_off // 2)  # byte offset -> element index

            local_absmax = fmath.absf(q_chunk).reduce(ReductionOp.MAX)
            absmax = local_absmax.to(fx.Float32)
            for sh in (8, 4, 2, 1):
                absmax = absmax.maximumf(dpp_utils.dpp_xor_f32(absmax, sh))

            q_scale = absmax * fx.Float32(1.0 / FP8_MAX)
            inv = fx.Float32(rcp_f32(q_scale.maximumf(fx.Float32(1e-20))))
            inv_b = fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(QCHUNK)

            q_scaled_chunk = q_chunk.to(fx.Float32) * inv_b
            _st_words(
                qh_local * HEAD + lane16 * QCHUNK,
                _f32_to_fp8_words(q_scaled_chunk),
            )
            if lane16 == 0:
                _st1(sQscale_off, qh_local, q_scale)
        else:
            _st_words(
                qh_local * HEAD + lane16 * QCHUNK,
                fx.Vector.filled(QCHUNK // 4, 0, fx.Int32),
            )
            if lane16 == 0:
                _st1(sQscale_off, qh_local, ZERO_F)

        gpu.barrier()

        # First tile's V page-index row, now visible after the barrier above
        # (the fetch+LDS-store was issued earlier, alongside k_pf0).
        v_page_pf0 = _v_page_read_row()

        # Q is the B operand (D[token,qhead] = K·Qᵀ), read raw from sQ as fp8
        # i64 operands (constant across tiles -> read once, held in
        # registers). Lane (lane16, rgroup) feeds MMA column n=lane16 (qhead);
        # MUST use the exact same (qkhe, rgroup, qkr) -> head_dim permutation
        # as K's `_k_ops`.
        q_ops = []
        for qkhe in range_constexpr(QKHE_LOOP):
            he_idx = qkhe * RGROUP_QUARTERS + rgroup
            chunk = _view(lane16 * HEAD + he_idx * QK_CHUNK_ELEMS, fx.Int64, fx.make_layout(2, 1)).load()
            q_ops.extend([chunk[0], chunk[1]])
        # q_ops[s] for s=0..N_SUBCHUNKS-1, s = qkhe*2+qkr, = head[he_idx*16+qkr*8 : +8] of qhead=lane16

        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tcopy_o = fx.make_tiled_copy_C(copy_c, tiled_mma_pv)  # PV out -> sO (epilogue)

        # QK in TLOOP chunks of TOK_CHUNK tokens: each chunk yields a f32x4
        # C-fragment, so softmax processes 4 scores at a time (low VGPR peak)
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
        tmpl_Op = fx.make_rmem_tensor(fx.make_layout((MFMA_MNK, VHE_SIZE), (VHE_SIZE, 1)), fx.Float32)
        OP_ELEMS = MFMA_MNK * VHE_SIZE // (NWARP * WAVE)  # PV C-fragment elements/lane/chunk (probed = 4)

        # ── raw dwordx4 V load (B operand) ──
        # lane (rgroup) takes the contiguous token slice [rgroup*64:+64] for
        # its head (vh*VHE_SIZE+warp*16+lane16). value_cache_ptr's "trans_v"
        # layout [num_blocks, num_kv_heads, block_size//16, head_dim, 16]
        # keeps 16 consecutive tokens innermost (V is token-vectorized,
        # unlike K); a rgroup's 64-token run can span multiple block_size
        # pages, so `sub`/`step` below walk pages/16-token sub-blocks.
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
            if const_expr(HEAD == 64):
                # Groups the NVOPS raw V loads into one scheduling unit for
                # the post-RA scheduler (zero VGPR cost); measured faster at
                # head_dim=64, slower at head_dim=128 (not applied there).
                fx.rocdl.sched_vmem(len(ops) // 2)
            return ops  # NVOPS i64, the 64-token contiguous run for this head

        # O is carried in registers across tiles (one VHE_CHUNKS-list of PV
        # C-fragment vectors), rescaled by the softmax correction each tile.
        # `m` is carried the same way (see the init comment above).
        o_zero = fx.Vector.filled(OP_ELEMS, 0.0, fx.Float32)
        for tt, ostate in range(loop_start, loop_end, 1, init=[o_zero, o_zero, k_pf0, v_page_pf0, NEG_INF, ZERO_F]):
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

            tt1 = tt_i32 + 1
            k_next = k_cur
            # K's prefetch, V's page-index prefetch, and (per_token_kv only)
            # the K/V scale stage all share the same `tt1 < part_end` guard,
            # so they're issued together in one branch -- this also lets
            # `_stage_kv_scale_to_lds` reuse `_k_ops_flat`'s own `phys_vec`
            # (same tile, same page formula) instead of re-fetching the
            # block-table lookup, without threading it out through a
            # None-initialized variable across a dynamic if.
            if tt1 < part_end:
                k_next, phys_vec1 = _k_ops_flat(tt1)
                _v_page_fetch_and_stage(tt1)
                # Per-token K/V scale (per_token_kv only): staged here too,
                # one tile ahead, so it's already barrier-visible (via
                # either barrier below) by the time next tile's iteration
                # reads it back at the very top -- same cadence as
                # k_next/K's own prefetch.
                if const_expr(per_token_kv):
                    _stage_kv_scale_to_lds(phys_vec1)

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

            # per_token_kv: K-scale varies per token, so (unlike the
            # per-tensor `scale` above, a positive constant that commutes
            # with max and can be applied AFTER the max-reduce below) it
            # must be folded into the raw score BEFORE masking/max-reduce.
            # V-scale doesn't affect QK at all -- it's carried alongside here
            # only because it's read from the same LDS round-trip.
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

            # Computed once and reused in pass 2 below (doesn't depend on pass
            # 1's output) instead of recomputing from scratch in both passes,
            # halving the mask compare/select instruction count for free.
            masked_chunks = [(_ct[a] < thr).select(scaled_frags[a], neg4) for a in range_constexpr(NCHUNK)]

            # pass 1: per-warp max for this qhead
            pm = fx.Float32(float("-inf"))
            for a in range_constexpr(NCHUNK):
                pm = pm.maximumf(masked_chunks[a].reduce(ReductionOp.MAX))
            for sh in (16, 32):
                pm = pm.maximumf(pm.shuffle_xor(sh, WAVE))
            # Unconditional store: all 4 lanes sharing this qhead already hold
            # the identical post-shuffle_xor `pm`, so this is a harmless
            # same-value redundant write, not a race.
            _st_lw(sLmax_off, qh, warp, pm * scale)

            # per_token_kv: this warp's own max V-scale over its 256/4 owned
            # tokens (masked to 0, not -inf, for out-of-range tokens -- a
            # max reduce ignores 0 as long as real scales are positive,
            # matching production's `_store_vmax_warp`). v_scale doesn't
            # depend on qhead, so this is naturally redundant across the 16
            # qh-differing lanes sharing (warp, l16g) -- harmless, since
            # they all compute the identical value. Reuses the pass-1
            # barrier below (no new barrier needed), same as `pm`/sLmax.
            if const_expr(per_token_kv):
                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                pv_max = fx.Float32(0.0)
                for a in range_constexpr(NCHUNK):
                    pv_max = pv_max.maximumf((_ct[a] < thr).select(v_scale_vecs[a], zero4).reduce(ReductionOp.MAX))
                for sh in (16, 32):
                    pv_max = pv_max.maximumf(pv_max.shuffle_xor(sh, WAVE))
                _st_lw(sVScaleMax_off, 0, warp, pv_max)
            gpu.barrier()

            # Read back next tile's V page-index row now that the barrier
            # above has made `_v_page_fetch_and_stage`'s store visible.
            v_page_next = v_page_cur
            if tt1 < part_end:
                v_page_next = _v_page_read_row()

            # per_token_kv: combine all 4 warps' max V-scale (staged just
            # above, now barrier-visible) into this tile's normalization
            # factor -- mirrors production's `v_max_scaled`/`norm_factor`.
            # `v_max_scaled` also doubles as this tile's PV correction
            # factor (applied to `op` in the PV loop below), replacing the
            # single per-CTA `v_scale_f` the per-tensor path uses instead.
            v_max_scaled = None
            norm_factor_b = None
            if const_expr(per_token_kv):
                v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(ReductionOp.MAX)
                v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                norm_factor = fx.Float32(rcp_f32(v_max_safe))
                norm_factor_b = fx.Vector.from_elements([norm_factor], dtype=fx.Float32).broadcast_to(4)

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
                Pa = fx.Vector(exp2_f32_fast(masked_chunks[a] * scale - m_new_b))
                ls = ls + Pa.reduce(ReductionOp.ADD)
                if const_expr(per_token_kv):
                    v_scale_this = _load_v_scale_vec(a) if const_expr(HEAD == 64) else v_scale_vecs[a]
                    p_scaled = Pa * v_scale_this * norm_factor_b
                else:
                    p_scaled = Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32)
                words.append(_f32_to_fp8_words(p_scaled)[0])

            p_off0 = sP_off + qh * SP_ROW_BYTES + warp * TOK_CHUNK + l16g * base4
            _view(p_off0, fx.Int32, fx.make_layout(NCHUNK, c16 // 4)).store(
                fx.Vector.from_elements(words, dtype=fx.Int32)
            )
            if const_expr(HEAD == 64):
                # Same scheduling-hint mechanism as the `sched_vmem` calls
                # above, grouping this LDS store instead; measured faster.
                # A matching `sched_dsrd` hint on the P·V read below (already
                # one fused vectorized load, not a multi-instruction store)
                # measured much slower and isn't used.
                fx.rocdl.sched_dswr(NCHUNK)
            for sh in (16, 32):
                ls = ls.addf(ls.shuffle_xor(sh, WAVE), fastmath=arith.FastMathFlags.contract)
            if l16g == 0:
                _st_lw(sLsum_off, qh, warp, ls)
                if warp == 0:
                    _st1(sCorr_off, qh, fx.Float32(exp2_amdgcn_scalar(m_old - m_new)))
            gpu.barrier()

            # pass 3: merge per-warp sums into the running denominator. `tid <
            # M` is exactly the `(l16g==0 and warp==0)` thread set above, so
            # its correction factor is already in `m_old`/`m_new` registers --
            # no need to read back what it just wrote to sCorr_off/sL_off.
            l_new = l_prev
            if tid < MFMA_MNK:
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
            corr_off = sCorr_off + m_base_pv * 4
            corr_vec = _view(corr_off, fx.Float32, fx.make_layout(OP_ELEMS, 1)).load()
            corr_s = [corr_vec[v] for v in range_constexpr(OP_ELEMS)]
            for vh in range_constexpr(VHE_CHUNKS):
                v_vh = v_vh_early if const_expr(HEAD == 64) else _v_ops(v_page_cur, vh)
                acc = arith.constant_vector(0.0, T.f32x4)
                for s in range_constexpr(NVOPS):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [p_ops[s], v_vh[s], acc, 0, 0, 0])
                op = fx.Vector(acc)
                if const_expr(per_token_kv):
                    # This tile's V-scale correction (undoes the P*v_scale
                    # normalization applied before the fp8 pack above) --
                    # replaces the per-tensor path's single per-CTA
                    # `v_scale_f`, folded in once at the very end instead
                    # (see the epilogue's `o_scale` below).
                    v_corr_b = fx.Vector.from_elements([v_max_scaled], dtype=fx.Float32).broadcast_to(OP_ELEMS)
                    op = op * v_corr_b
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
                fx.make_layout((MFMA_MNK, VHE_SIZE), (HEAD, 1)),
            )
            fx.copy(copy_c, thr_copy_o_e.retile(frag_Oe), thr_copy_o_e.partition_D(sO_chunk), pred=None)
        gpu.barrier()

        # ── epilogue: spread the row -> global write across ALL 256 threads
        # (THREADS_PER_ROW threads per query row, each owning a contiguous
        # ELEMS_PER_THREAD-wide slice) instead of only the GS row-owner lanes
        # looping over all HEAD elements -- fully uses the wave and cuts the
        # epilogue's static instruction count (measured ds_read: 45 -> ~15,
        # matching range).
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
        # Per-tensor packs P unscaled by v_scale (`Pa*FP8_MAX`, see the
        # P-pack loop above), so v_scale and the P dequant (1/FP8_MAX) are
        # true per-CTA constants deferred to exactly here, once, before the
        # NP branch. per_token_kv instead packs `Pa*v_scale*norm_factor`
        # (V-scale can't be deferred -- see that same loop) and already
        # fully undid that scaling with the per-tile `v_max_scaled`
        # correction in the main loop above, so no further scale is needed
        # here.
        if const_expr(per_token_kv):
            o_scale = fx.Float32(1.0)
        else:
            o_scale = v_scale_f * fx.Float32(1.0 / FP8_MAX)
        o_v = o_v * fx.Vector.from_elements([o_scale], dtype=fx.Float32).broadcast_to(ELEMS_PER_THREAD)

        if const_expr(NP == 1):
            # single partition: normalize and write the output directly (no
            # partials / reduce round-trip).
            qh = kv_h * GS + row_e

            l_row = _ld1(sL_off, row_e)
            safe_l = arith.select(l_row > ZERO_F, l_row, fx.Float32(1.0))
            inv_l = fx.Float32(rcp_f32(safe_l))
            o_out = (o_v * fx.Vector.from_elements([inv_l], dtype=fx.Float32).broadcast_to(ELEMS_PER_THREAD)).to(
                Q_DTYPE
            )
            # Divide this (seq, qh) row's HEAD axis into ELEMS_PER_THREAD-wide
            # chunks and pick this lane's chunk (no manual byte offset).
            out_row = output_ptr[seq, qh, None]
            out_chunk = fx.slice(fx.logical_divide(out_row, fx.make_layout(ELEMS_PER_THREAD, 1)), (None, sub_e))
            out_chunk.store(o_out)
        else:
            base = ((seq * n_kv + kv_h) * NP + part) * GS + row_e
            if sub_e == 0:
                pmax_ptr[base] = _ld1(sM_off, row_e)
                psum_ptr[base] = _ld1(sL_off, row_e)
            l_p_row = _ld1(sL_off, row_e)
            safe_l_p = arith.select(l_p_row > ZERO_F, l_p_row, fx.Float32(1.0))
            inv_l_p = fx.Float32(rcp_f32(safe_l_p))
            o_norm = (o_v * fx.Vector.from_elements([inv_l_p], dtype=fx.Float32).broadcast_to(ELEMS_PER_THREAD)).to(
                Q_DTYPE
            )
            pout_div = fx.logical_divide(pout_ptr, fx.make_layout(ELEMS_PER_THREAD, 1))
            pout_chunk = fx.slice(pout_div, (None, base * THREADS_PER_ROW + sub_e))
            pout_chunk.store(o_norm)

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
        ).launch(grid=(num_seqs, num_kv_heads, NP), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {"launch": pa_decode_tile_launch, "kernel": pa_decode_tile_kernel}


def _choose_num_partitions(
    num_seqs: int,
    num_kv_heads: int,
    max_blocks_per_seq: int,
    block_size: int,
    device: torch.device,
    *,
    target_ctas_per_cu: int = 8,
    min_tiles_per_partition: int = 2,
    tile_tok: int = 256,
) -> int:
    """Choose the number of context partitions (grid.z) for pa_decode_tile.

    This kernel launches a *static* one-CTA-per-partition grid (unlike
    production's *persistent* kernel), so it needs its own NP heuristic,
    tuned via a direct sweep on an 80-CU MI308X. Two regimes:
      - CU-STARVED (num_seqs*num_kv_heads < device CU count): push NP up to
        `cu_fill_np`, uncapped by tiles-per-partition -- idle CUs are worse
        than thin partitions.
      - NOT CU-STARVED: further CTAs only add occupancy depth and reduce-
        kernel/prologue overhead, so cap tiles-per-partition to at least
        `min_tiles_per_partition`.

    ``target_ctas_per_cu``, ``min_tiles_per_partition``, and ``tile_tok``
    default to the values this heuristic was tuned with; override only to
    re-sweep or adapt to a different device.
    """
    from kernels.attention.pa_decode_fp8 import get_recommended_splits

    device_cus = torch.cuda.get_device_properties(device).multi_processor_count
    cu_fill_np = cdiv(target_ctas_per_cu * device_cus, num_seqs * num_kv_heads)
    # Bounded by max_blocks_per_seq*block_size/tile_tok, NOT the actual
    # context length: reading `context_lengths` on the host forces a GPU
    # sync, illegal during CUDA graph capture. Exact when callers size
    # `block_tables` to the actual context length; looser (but still
    # correct) if they over-allocate for a larger max-sequence-length.
    max_possible_tiles = cdiv(max_blocks_per_seq * block_size, tile_tok)
    cu_starved = (num_seqs * num_kv_heads) < device_cus
    tiles_np_cap = max_possible_tiles if cu_starved else max(1, max_possible_tiles // min_tiles_per_partition)
    base_np = get_recommended_splits(num_seqs, num_kv_heads)
    return max(1, min(max(base_np, cu_fill_np), tiles_np_cap))


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
    _, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape

    assert num_hgroups == head_dim // 16 and hgroup_width == 16
    assert block_size in (16, 64), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    _, _, v_subblocks, v_head_dim, v_width = value_cache.shape

    assert v_head_dim == head_dim and v_width == 16 and v_subblocks == block_size // 16
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

    dev = query.device
    # per_token_kv: key_scale/value_scale are [num_blocks, num_kv_heads,
    # block_size] tensors (one dequant scale per physical KV token) instead
    # of a single per-tensor scalar -- detected purely from tensor rank,
    # matching pa_decode_ps_kernel's own `per_token_kv = key_scale.ndim > 1`.
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

    num_partitions = _choose_num_partitions(num_seqs, num_kv_heads, max_blocks_per_seq, block_size, dev)
    GS = query_group_size

    compiled = compile_pa_decode_tile(
        head_dim=head_dim,
        query_group_size=query_group_size,
        block_size=int(block_size),
        num_partitions=num_partitions,
        softmax_scale=softmax_scale,
        query_dtype=query_dtype,
        per_token_kv=per_token_kv,
    )
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
        s,
    )
    if num_partitions > 1:
        from kernels.attention.pa_decode_swa import compile_pa_decode_sw_reduce

        reduce_compiled = compile_pa_decode_sw_reduce(
            max_context_partition_num=num_partitions,
            query_seq_len=1,
            query_group_size=GS,
            head_size=head_dim,
            output_dtype_str=query_dtype,
            logits_dtype_str=query_dtype,
        )
        _run_compiled(
            reduce_compiled["launch"],
            output.data_ptr(),
            psum.data_ptr(),  # exp_sums
            pmax.data_ptr(),  # max_logits
            pout.data_ptr(),  # logits (already-normalized query_dtype partials)
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
