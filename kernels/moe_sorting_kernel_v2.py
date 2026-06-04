# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE token sorting kernels (FlyDSL).

Counting sort that reorganises router top-k selections by expert for batched
expert GEMM. Output contract: packed ``(slot << 24) | token``, sentinel
``(topk << 24) | M``, ``num_valid_ids = [num_tokens_post_pad, M]``, with the
``moe_buf`` accumulator zeroed in the same launch.

Three variants, selected by padded token count
(one thread per expert, W =``workgroup_width(experts)`` threads per block):

- ``mesh_flatfill`` -- low token. One block stages a ``token x expert`` LDS mesh
  (slot+1 per routed pair), histograms each expert column, flat-fills sentinels,
  then scatters. JIT-unrolls over tokens, hence the token ceiling.
- ``count_sort`` -- mid token. K1 counts pairs per expert; K2 prefix-sums the
  counts to expert bases and scatters pairs in pair-index (stable) order.
- ``count_sort_paired`` -- high token. K1 per-block histogram, K2 column-scans to
  expert bases, K3 writes block/sentinel/tail structure, K4 scatters by block+rank.

Shared machinery: per-expert counts are padded up to ``unit_size``, then an
inclusive ``ds_bpermute`` warp scan (folded across waves through LDS) turns them
into expert base offsets; under ``has_mask`` the dense expert index is packed above
the padded count so one scan carries both. The ``moe_buf`` zero is folded onto each
variant's trailing grid blocks as a vectorized grid-stride memset.

``unit_size`` (B) is the padding granularity / GEMM tile-M;
``workgroup_size`` (W)``= workgroup_width(experts)`` is the thread-block width.
"""

import functools
import math
import warnings
from typing import Literal

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

# ===========================================================================
# Constants.
# ===========================================================================

UNIT_SIZE = 32  # default padding granularity (GEMM tile-M)
MAX_THREADS_PER_BLOCK = 1024  # AMD workgroup cap; one-thread-per-expert -> expert cap
LDS_BYTES_PER_BLOCK = 64 * 1024
WARP_SIZE_FALLBACK = 64  # CDNA wave64, for the GPU-free diagnostics
MAX_LOW_TOKEN_COUNT = 256  # mesh_flatfill JIT-unroll ceiling

# mesh_flatfill <-> count_sort boundary: tokens* = M / (topk + T0). mesh cost is
# const_cost + tokens*(a*topk + b); T0 folds the topk-independent per-token term b (the
# mesh histogram) into the denominator so the boundary tracks it -- plain M/topk
# mispredicts at low topk.
_MESH_FLATFILL_RECOMMENDED_TOPK_OFFSET = 7  # T0, the topk-independent per-token term
_MESH_FLATFILL_RECOMMENDED_M = {"gfx950": 320, "gfx942": 256}
_MESH_FLATFILL_RECOMMENDED_M_FALLBACK = 256

# count_sort <-> count_sort_paired boundary as a per-block chunk count K:
# tokens* = K * workgroup_size / topk. count_sort streams ceil(num_pairs/W) chunks
# (one W-wide scan+barrier pass over the pairs each) per block; count_sort_paired spreads
# pairs across blocks and wins once that chunk count crosses the fitted K.
_COUNT_SORT_CROSSOVER_CHUNKS = {"gfx950": 10, "gfx942": 8}
_COUNT_SORT_CROSSOVER_CHUNKS_FALLBACK = 8

# mesh_flatfill register-cache token ceiling (VGPR budget).
_MESH_FLATFILL_CACHE_COLUMN_MAX_TOKENS = {"gfx950": 64, "gfx942": 64}
_MESH_FLATFILL_CACHE_COLUMN_MAX_TOKENS_FALLBACK = 64

# arch -> (store_vec_width, zero_target_occupancy) for the moe_buf zero launch.
_STORE_VEC_OCCUPANCY = {"gfx950": (4, 2), "gfx942": (4, 2)}
_STORE_VEC_OCCUPANCY_FALLBACK = (4, 2)


# ===========================================================================
# Feasibility / support helpers.
# ===========================================================================


def workgroup_width(experts: int, warp_size: int = WARP_SIZE_FALLBACK) -> int:
    """Thread-block width (W): experts rounded up to whole waves, capped at the
    workgroup limit. Decoupled from ``unit_size`` (B)."""
    waves = (experts + warp_size - 1) // warp_size
    return min(MAX_THREADS_PER_BLOCK, waves * warp_size)


def mesh_lds_bytes(experts: int, tokens: int) -> int:
    """LDS the mesh family allocates (the feasibility-binding term): a
    ``width x tokens`` int32 mesh plus a ``width`` column."""
    width = workgroup_width(experts)
    return (width * tokens + width) * 4


def _structural_unsupported(experts: int) -> str | None:
    """Reason a one-thread-per-expert kernel can't run this many experts, or None."""
    workgroup_size = workgroup_width(experts)
    if experts > workgroup_size:
        return f"experts ({experts}) > workgroup_size ({workgroup_size})"
    if workgroup_size > MAX_THREADS_PER_BLOCK:
        return f"workgroup_size ({workgroup_size}) exceeds the {MAX_THREADS_PER_BLOCK}-work-item workgroup limit"
    return None


def mesh_flatfill_unsupported(experts: int, tokens: int) -> str | None:
    """Reason mesh_flatfill can't run this shape, or None. It JIT-unrolls over tokens and
    stages a token x expert LDS mesh, so it has a token ceiling and an LDS cap."""
    reason = _structural_unsupported(experts)
    if reason is not None:
        return reason
    if tokens > MAX_LOW_TOKEN_COUNT:
        return (
            f"mesh_flatfill unrolls over tokens; tokens ({tokens}) exceeds the "
            f"{MAX_LOW_TOKEN_COUNT}-token ceiling -- use count_sort"
        )
    used = mesh_lds_bytes(experts, tokens)
    if used > LDS_BYTES_PER_BLOCK:
        return (
            f"mesh_flatfill needs {used // 1024} KiB LDS, over the "
            f"{LDS_BYTES_PER_BLOCK // 1024} KiB cap -- use count_sort"
        )
    return None


def _require_supported(reason: str | None) -> None:
    if reason is not None:
        raise ValueError(reason)


# ===========================================================================
# Arch-keyed tuning. Arch detection is best-effort (conservative fallback) so
# the diagnostics stay GPU-free.
# ===========================================================================


def _value_for_arch(table: dict, fallback, arch: str | None = None):
    """First ``table`` value whose arch-substring key matches the (detected) arch,
    else ``fallback``. Arch detection is best-effort so the tuning stays GPU-free."""
    if arch is None:
        try:
            arch = get_rocm_arch()
        except Exception:
            arch = None
    arch = (arch or "").lower()
    return next((v for k, v in table.items() if k in arch), fallback)


def count_sort_crossover_tokens(workgroup_size: int, topk: int, arch: str | None = None) -> int:
    """Token count where count_sort_paired overtakes count_sort: ``K * workgroup_size / topk``.

    ``K`` is the per-block chunk count at the crossover (one chunk = one
    ``workgroup_size``-wide pass over the ``tokens * topk`` pairs); see the
    ``_COUNT_SORT_CROSSOVER_CHUNKS`` note.
    """
    chunks = _value_for_arch(
        _COUNT_SORT_CROSSOVER_CHUNKS,
        _COUNT_SORT_CROSSOVER_CHUNKS_FALLBACK,
        arch,
    )
    return max(1, chunks * workgroup_size // topk)


def mesh_flatfill_recommended_max_tokens(topk: int, arch: str | None = None) -> int:
    """Token count where count_sort overtakes mesh_flatfill: ``M / (topk + T0)``.

    ``T0`` (``_MESH_FLATFILL_RECOMMENDED_TOPK_OFFSET``) is mesh_flatfill's topk-independent
    per-token cost, added to ``topk`` so the boundary tracks both cost terms.
    """
    m = _value_for_arch(
        _MESH_FLATFILL_RECOMMENDED_M,
        _MESH_FLATFILL_RECOMMENDED_M_FALLBACK,
        arch,
    )
    return max(1, m // (topk + _MESH_FLATFILL_RECOMMENDED_TOPK_OFFSET))


def mesh_flatfill_cache_column_max_tokens(arch: str | None = None) -> int:
    return _value_for_arch(
        _MESH_FLATFILL_CACHE_COLUMN_MAX_TOKENS,
        _MESH_FLATFILL_CACHE_COLUMN_MAX_TOKENS_FALLBACK,
        arch,
    )


def default_store_vec_width(arch: str | None = None) -> int:
    return _value_for_arch(
        _STORE_VEC_OCCUPANCY,
        _STORE_VEC_OCCUPANCY_FALLBACK,
        arch,
    )[0]


def default_zero_target_occupancy(arch: str | None = None) -> int:
    return _value_for_arch(
        _STORE_VEC_OCCUPANCY,
        _STORE_VEC_OCCUPANCY_FALLBACK,
        arch,
    )[1]


def _get_warp_size(arch=None) -> int:
    """CDNA wave64, RDNA wave32. Resolved at JIT time."""
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


def count_sort_paired_scratch_elems(tokens: int, topk: int, experts: int) -> tuple[int, int]:
    """``(hist_elems, side_elems)``: ``hist`` is ``n_pair_blocks * experts`` int32;
    ``counts`` and ``estart`` are ``experts`` int32 each."""
    num_pairs = tokens * topk
    workgroup_size = workgroup_width(experts)
    n_pair_blocks = (num_pairs + workgroup_size - 1) // workgroup_size
    return n_pair_blocks * experts, experts


def _raw(v):
    """Unwrap an fx value to its raw MLIR ``ir.Value`` (for raw ODS op builders)."""
    return v.ir_value() if hasattr(v, "ir_value") else v


# ===========================================================================
# mesh_flatfill -- low-token mesh + flat-fill sort.
# ===========================================================================


def compile_mesh_flatfill(
    *,
    experts: int,
    tokens: int,
    topk: int,
    unit_size: int,
    has_mask: bool = False,
    has_padding: bool = False,
    store_vec_width: int | None = None,
):
    """Build the mesh_flatfill launcher (sort on block 0; zero ``moe_buf`` on blocks 1..N)."""
    _require_supported(mesh_flatfill_unsupported(experts, tokens))

    WARP_SIZE = _get_warp_size()
    workgroup_size = workgroup_width(experts, WARP_SIZE)

    vec_width = store_vec_width
    if vec_width is None:
        vec_width = default_store_vec_width()
    assert vec_width in (2, 4), f"store_vec_width ({vec_width}) must be 2 or 4"

    assert workgroup_size % WARP_SIZE == 0
    assert unit_size % vec_width == 0, f"unit_size ({unit_size}) must be a multiple of store_vec_width ({vec_width})"
    n_waves = workgroup_size // WARP_SIZE
    assert n_waves <= WARP_SIZE
    log2_warp = int(math.log2(WARP_SIZE))

    num_pairs = tokens * topk
    n_mesh = workgroup_size * tokens
    n_load_iters = (num_pairs + workgroup_size - 1) // workgroup_size
    max_num_tokens_padded = num_pairs + experts * unit_size - topk
    max_num_m_blocks = (max_num_tokens_padded + unit_size - 1) // unit_size
    sentinel_id = (topk << 24) | tokens

    needs_cross_wave_scan = experts > WARP_SIZE

    # has_mask packs ``present`` (high bits) above ``padded`` (low bits) in one i32, so a
    # single scan carries both: low bits give the padded base offset, and
    # ``(high bits >> pack_shift) - present`` is the exclusive prefix of present experts =
    # the dense (compacted) expert index. No extra scan / LDS.
    pack_shift = max_num_tokens_padded.bit_length()
    assert (not has_mask) or pack_shift + max(1, experts.bit_length()) <= 31
    pack_lo_mask = (1 << pack_shift) - 1

    # Cache the mesh column in registers across histogram + scatter (gated by VGPR budget).
    cache_column = tokens <= mesh_flatfill_cache_column_max_tokens()

    if needs_cross_wave_scan:

        @fx.struct  # ty: ignore[too-many-positional-arguments]
        class SharedStorage:
            s_mesh: fx.Array[fx.Int32, n_mesh]
            s_wave: fx.Array[fx.Int32, n_waves]
            s_total: fx.Array[fx.Int32, 1]
    else:

        @fx.struct  # ty: ignore[too-many-positional-arguments]
        class SharedStorage:
            s_mesh: fx.Array[fx.Int32, n_mesh]
            s_total: fx.Array[fx.Int32, 1]

    @flyc.jit
    def _emit_sort(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
    ):
        tid = fx.thread_idx.x

        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids)
        topk_weights_rsrc = buffer_ops.create_buffer_resource(topk_weights)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_ids)
        sorted_weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights)
        sorted_expert_ids_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids)
        num_valid_ids_rsrc = buffer_ops.create_buffer_resource(num_valid_ids)

        expert_id = fx.Int32(tid)
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        neg_one_i32 = fx.Int32(-1)
        m_i32 = fx.Int32(tokens)
        topk_i32 = fx.Int32(topk)
        tokens_i32 = fx.Int32(tokens)
        unit_size_i32 = fx.Int32(unit_size)
        unit_size_minus_one = fx.Int32(unit_size - 1)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        workgroup_size_minus_one = fx.Int32(workgroup_size - 1)
        max_num_m_blocks_i32 = fx.Int32(max_num_m_blocks)
        num_pairs_i32 = fx.Int32(num_pairs)
        experts_i32 = fx.Int32(experts)
        pack_shift_i32 = fx.Int32(pack_shift)
        pack_lo_mask_i32 = fx.Int32(pack_lo_mask)

        if const_expr(has_mask):
            expert_mask_rsrc = buffer_ops.create_buffer_resource(expert_mask)
            in_experts = expert_id < experts_i32
            safe_e = in_experts.select(expert_id, zero_i32)
            mask_val = buffer_ops.buffer_load(expert_mask_rsrc, safe_e, vec_width=1, dtype=fx.Int32)
            present = (in_experts & (mask_val != zero_i32)).select(one_i32, zero_i32)
        else:
            present = one_i32

        if const_expr(has_padding):
            num_local_tokens_rsrc = buffer_ops.create_buffer_resource(num_local_tokens)
            n_local = buffer_ops.buffer_load(num_local_tokens_rsrc, zero_i32, vec_width=1, dtype=fx.Int32)
            pair_bound = n_local * topk_i32
        else:
            n_local = m_i32
            pair_bound = num_pairs_i32

        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_mesh = lds.s_mesh.view(fx.make_layout(n_mesh, 1))
        s_total = lds.s_total.view(fx.make_layout(1, 1))

        # Clear the mesh.
        for i in range_constexpr(0, n_mesh, workgroup_size):
            fx.memref_store(zero_i32, s_mesh, fx.Int32(i) + tid)
        gpu.barrier()

        # Fill: one thread per pair writes slot+1 into mesh[expert, token].
        for j in range_constexpr(n_load_iters):
            p = fx.Int32(j * workgroup_size) + tid
            if p < pair_bound:
                ex = buffer_ops.buffer_load(topk_ids_rsrc, p, vec_width=1, dtype=fx.Int32)
                token = p // topk_i32
                slot = p % topk_i32
                fx.memref_store(slot + one_i32, s_mesh, ex * tokens_i32 + token)
        gpu.barrier()

        # Histogram this expert's mesh column (cache the cells for the scatter).
        col_base = expert_id * tokens_i32
        count = zero_i32
        col_vals: list = []
        for t in range_constexpr(tokens):
            mv = fx.memref_load(s_mesh, col_base + fx.Int32(t))
            if const_expr(cache_column):
                col_vals.append(mv)
            count = count + (mv != zero_i32).select(one_i32, zero_i32)

        if const_expr(has_mask):
            count = (present != zero_i32).select(count, zero_i32)

        padded = ((count + unit_size_minus_one) // unit_size_i32) * unit_size_i32

        if const_expr(has_mask):
            scan_val = (present << pack_shift_i32) | padded
        else:
            scan_val = padded

        # Inclusive scan of padded counts -> exclusive expert base offsets (intra-wave).
        scanned = scan_val
        for step in range_constexpr(log2_warp):
            off = 1 << step
            in_range = lane >= off
            peer_lane = in_range.select(lane - off, zero_i32)
            byte_addr = peer_lane * 4
            peer_raw = fx.rocdl.ds_bpermute(T.i32, byte_addr, scanned)
            peer = fx.Int32(peer_raw)
            peer_safe = in_range.select(peer, zero_i32)
            scanned = scanned + peer_safe

        # Fold the per-wave totals across waves (only when experts span >1 wave).
        if const_expr(needs_cross_wave_scan):
            s_wave = lds.s_wave.view(fx.make_layout(n_waves, 1))

            if lane == WARP_SIZE - 1:
                fx.memref_store(scanned, s_wave, wave)
            gpu.barrier()

            if wave == 0:
                in_n_waves = lane < n_waves
                safe_idx = in_n_waves.select(lane, zero_i32)
                loaded = fx.memref_load(s_wave, safe_idx)
                val = in_n_waves.select(loaded, zero_i32)

                for step in range_constexpr(log2_warp):
                    off = 1 << step
                    in_range = lane >= off
                    peer_lane = in_range.select(lane - off, zero_i32)
                    byte_addr = peer_lane * 4
                    peer_raw = fx.rocdl.ds_bpermute(T.i32, byte_addr, val)
                    peer = fx.Int32(peer_raw)
                    peer_safe = in_range.select(peer, zero_i32)
                    val = val + peer_safe

                # Publish the grand total on this barrier (skips the broadcast barrier).
                if lane == fx.Int32(n_waves - 1):
                    fx.memref_store(val, s_total, zero_i32)

                not_lane_0 = lane > 0
                peer_lane = not_lane_0.select(lane - 1, zero_i32)
                byte_addr = peer_lane * 4
                prev_raw = fx.rocdl.ds_bpermute(T.i32, byte_addr, val)
                prev = fx.Int32(prev_raw)
                exclusive_wave = not_lane_0.select(prev, zero_i32)

                if lane < n_waves:
                    fx.memref_store(exclusive_wave, s_wave, lane)
            gpu.barrier()

            wave_prefix = fx.memref_load(s_wave, wave)
            inclusive_scan = scanned + wave_prefix
            total_scan = fx.memref_load(s_total, zero_i32)
        else:
            inclusive_scan = scanned
            if tid == fx.Int32(experts - 1):
                fx.memref_store(inclusive_scan, s_total, zero_i32)
            gpu.barrier()
            total_scan = fx.memref_load(s_total, zero_i32)

        if const_expr(has_mask):
            inclusive_padded = inclusive_scan & pack_lo_mask_i32
            total_padded = total_scan & pack_lo_mask_i32
            local_idx = (inclusive_scan >> pack_shift_i32) - present
        else:
            inclusive_padded = inclusive_scan
            total_padded = total_scan
            local_idx = expert_id

        # Flat vectorized sentinel fill of [0, total_padded); scatter overwrites valid slots.
        cw_i32 = fx.Int32(vec_width)
        sentinel_vec = fx.Vector.filled(vec_width, sentinel_id, fx.Int32)
        n_vec = total_padded // cw_i32
        n_fill = (n_vec + workgroup_size_minus_one) // workgroup_size_i32
        for k in range(zero_i32, n_fill):
            vidx = k * workgroup_size_i32 + tid
            if vidx < n_vec:
                buffer_ops.buffer_store(sentinel_vec, sorted_ids_rsrc, vidx * cw_i32)
        gpu.barrier()

        # Scatter this expert's valid pairs + write its per-block expert ids.
        exclusive_padded = inclusive_padded - padded
        block_offset = exclusive_padded // unit_size_i32
        n_blocks = padded // unit_size_i32

        if tid < experts:
            # Prefetch all weights before any store so the loads pipeline.
            w_vals: list = []
            for t in range_constexpr(tokens):
                mv = col_vals[t] if const_expr(cache_column) else fx.memref_load(s_mesh, col_base + fx.Int32(t))
                safe_slot = (mv != zero_i32).select(mv - one_i32, zero_i32)
                w_vals.append(
                    buffer_ops.buffer_load(
                        topk_weights_rsrc,
                        fx.Int32(t) * topk_i32 + safe_slot,
                        vec_width=1,
                        dtype=fx.Float32,
                    )
                )

            pos = zero_i32
            for t in range_constexpr(tokens):
                if const_expr(cache_column):
                    mv = col_vals[t]
                else:
                    mv = fx.memref_load(s_mesh, col_base + fx.Int32(t))
                is_match = mv != zero_i32
                if const_expr(has_mask):
                    is_match = is_match & (present != zero_i32)
                slot = mv - one_i32
                write_pos = exclusive_padded + pos
                token_i = fx.Int32(t)
                packed = (slot << fx.Int32(24)) | token_i
                if is_match:
                    buffer_ops.buffer_store(packed, sorted_ids_rsrc, write_pos)
                    buffer_ops.buffer_store(w_vals[t], sorted_weights_rsrc, write_pos)
                pos = pos + is_match.select(one_i32, zero_i32)

            for b in range(zero_i32, n_blocks):
                buffer_ops.buffer_store(local_idx, sorted_expert_ids_rsrc, block_offset + b)

        # Cooperative -1 fill of the trailing sorted_expert_ids blocks.
        n_valid_blocks = total_padded // unit_size_i32
        n_tail_iters = (max_num_m_blocks_i32 - n_valid_blocks + workgroup_size_minus_one) // workgroup_size_i32
        for k in range(zero_i32, n_tail_iters):
            b = n_valid_blocks + k * workgroup_size_i32 + tid
            if b < max_num_m_blocks_i32:
                buffer_ops.buffer_store(neg_one_i32, sorted_expert_ids_rsrc, b)

        # Publish num_valid_ids = [num_tokens_post_pad, M].
        if tid == fx.Int32(experts - 1):
            buffer_ops.buffer_store(inclusive_padded, num_valid_ids_rsrc, zero_i32)
            buffer_ops.buffer_store(n_local, num_valid_ids_rsrc, one_i32)

    @flyc.jit
    def _emit_zero(moe_buf: fx.Tensor, i32_moe_buf_elems: fx.Int32):
        """Zero moe_buf on blocks 1..N via a vectorized grid-stride loop."""
        tid = fx.thread_idx.x
        bid = gpu.block_idx.x
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        cw_i32 = fx.Int32(vec_width)
        workgroup_size_i32 = fx.Int32(workgroup_size)

        moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf)
        c_zero_vec = fx.Vector.filled(vec_width, 0, fx.Int32)
        total_vec = i32_moe_buf_elems // cw_i32
        num_zero_blocks = gpu.grid_dim.x - one_i32
        gid_vec = (bid - one_i32) * workgroup_size_i32 + tid
        stride_vec = num_zero_blocks * workgroup_size_i32
        n_iters = (total_vec + stride_vec - one_i32) // stride_vec
        for z in range(zero_i32, n_iters):
            idx = gid_vec + z * stride_vec
            if idx < total_vec:
                buffer_ops.buffer_store(c_zero_vec, moe_buf_rsrc, idx * cw_i32)

        # First zero block mops up the scalar tail.
        tail_start = total_vec * cw_i32
        if bid == one_i32:
            ti = tail_start + tid
            if ti < i32_moe_buf_elems:
                buffer_ops.buffer_store(zero_i32, moe_buf_rsrc, ti)

    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = gpu.block_idx.x
        if bid != fx.Int32(0):
            _emit_zero(moe_buf, i32_moe_buf_elems)
        if bid == fx.Int32(0):
            _emit_sort(
                topk_ids,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                expert_mask,
                num_local_tokens,
            )

    @flyc.jit
    def launcher(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_moe_buf_elems: fx.Int32,
        n_grid: fx.Int32,
        stream: fx.Stream,
    ):
        block = (workgroup_size, 1, 1)
        launcher = kernel(  # ty: ignore[call-non-callable]
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            expert_mask,
            num_local_tokens,
            moe_buf,
            i32_moe_buf_elems,
        )
        launcher.launch(grid=(n_grid, 1, 1), block=block, stream=stream)

    return launcher


@functools.cache
def get_moe_sorting_mesh_flatfill_kernel(
    *,
    experts: int,
    tokens: int,
    topk: int,
    unit_size: int,
    has_mask: bool = False,
    has_padding: bool = False,
    store_vec_width: int | None = None,
):
    return compile_mesh_flatfill(
        experts=experts,
        tokens=tokens,
        topk=topk,
        unit_size=unit_size,
        has_mask=has_mask,
        has_padding=has_padding,
        store_vec_width=store_vec_width,
    )


# ===========================================================================
# count_sort -- mid-token two-kernel counting sort.
# ===========================================================================


def compile_count_sort(
    *,
    experts: int,
    tokens: int,
    topk: int,
    unit_size: int,
    has_mask: bool = False,
    has_padding: bool = False,
    store_vec_width: int | None = None,
):
    """Build the count_sort launcher: K1 (count) + K2 (scatter), with the ``moe_buf``
    zero folded onto K2's trailing blocks."""
    _require_supported(_structural_unsupported(experts))  # count_sort: only the expert/workgroup bound

    vec_width = store_vec_width
    if vec_width is None:
        vec_width = default_store_vec_width()
    assert vec_width in (2, 4), f"store_vec_width ({vec_width}) must be 2 or 4"

    WARP_SIZE = _get_warp_size()
    workgroup_size = workgroup_width(experts, WARP_SIZE)
    assert workgroup_size % WARP_SIZE == 0
    n_waves = workgroup_size // WARP_SIZE
    assert n_waves <= WARP_SIZE
    log2_warp = int(math.log2(WARP_SIZE))

    num_pairs = tokens * topk
    max_num_tokens_padded = num_pairs + experts * unit_size - topk
    max_num_m_blocks = (max_num_tokens_padded + unit_size - 1) // unit_size
    sentinel_id = (topk << 24) | tokens

    # Trip counts are compile-time but the loops are runtime scf.for, so JIT is O(1) in tokens.
    n_chunks = (num_pairs + workgroup_size - 1) // workgroup_size
    n_tail_iters = (max_num_m_blocks + workgroup_size - 1) // workgroup_size
    n_pad_iters = (unit_size + workgroup_size - 1) // workgroup_size

    # has_mask packs ``present`` (high bits) above ``padded`` (low bits) in one i32, so a
    # single scan carries both: low bits give the base offset, ``(scan >> pack_shift) -
    # present`` is the dense (compacted) expert index.
    pack_shift = max_num_tokens_padded.bit_length()
    assert (not has_mask) or pack_shift + max(1, experts.bit_length()) <= 31
    pack_lo_mask = (1 << pack_shift) - 1

    @fx.struct  # ty: ignore[too-many-positional-arguments]
    class ScanStorage:
        # Double-buffered (2 * n_waves): the scatter alternates banks by chunk parity so
        # it can drop the per-chunk reuse barrier (same-bank reuse is two chunks apart).
        s_wave: fx.Array[fx.Int32, 2 * n_waves]
        s_excl: fx.Array[fx.Int32, workgroup_size]

    # ------------------------------------------------------------------ K1
    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def count_kernel(
        topk_ids: fx.Tensor,
        counts: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
    ):
        """One block per expert: count pairs routed to ``block_idx`` -> counts[e]."""
        tid = fx.thread_idx.x
        expert_bid = gpu.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        num_pairs_i32 = fx.Int32(num_pairs)

        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids)
        counts_rsrc = buffer_ops.create_buffer_resource(counts)

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource(num_local_tokens),
                zero_i32,
                vec_width=1,
                dtype=fx.Int32,
            )
            pair_bound = n_local * fx.Int32(topk)
        else:
            pair_bound = num_pairs_i32

        lds = fx.SharedAllocator().allocate(ScanStorage).peek()
        s_wave = lds.s_wave.view(fx.make_layout(n_waves, 1))

        for _c, state in range(  # ty: ignore[no-matching-overload, not-iterable]
            fx.Index(0), fx.Index(n_chunks), fx.Index(1), init=[zero_i32]
        ):
            acc = state[0]
            p = fx.Int32(_c) * workgroup_size_i32 + tid
            valid = p < pair_bound
            safe_p = valid.select(p, zero_i32)
            ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
            is_mine = valid & (ex == expert_bid)
            results = yield [acc + is_mine.select(one_i32, zero_i32)]
        local = results

        scanned = local
        for step in range_constexpr(log2_warp):
            off = 1 << step
            in_range = lane >= off
            peer_lane = in_range.select(lane - off, zero_i32)
            peer = fx.Int32(fx.rocdl.ds_bpermute(T.i32, peer_lane * 4, scanned))
            scanned = scanned + in_range.select(peer, zero_i32)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()

        if tid == zero_i32:
            total = zero_i32
            for w in range_constexpr(n_waves):
                total = total + fx.memref_load(s_wave, fx.Int32(w))
            if const_expr(has_mask):
                em_val = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource(expert_mask),
                    expert_bid,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                total = (em_val != zero_i32).select(total, zero_i32)
            buffer_ops.buffer_store(total, counts_rsrc, expert_bid)

    # ------------------------------------------------------------------ K2
    @flyc.jit
    def _scatter_pairs(
        tid,
        expert_bid,
        lane,
        wave,
        my_start,
        present_block,
        pair_bound,
        topk_ids_rsrc,
        topk_weights_rsrc,
        sorted_ids_rsrc,
        sorted_weights_rsrc,
        s_wave,
    ):
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        topk_i32 = fx.Int32(topk)
        workgroup_size_i32 = fx.Int32(workgroup_size)

        for _c, state in range(  # ty: ignore[no-matching-overload, not-iterable]
            fx.Index(0), fx.Index(n_chunks), fx.Index(1), init=[my_start]
        ):
            position = state[0]
            bank = (fx.Int32(_c) & one_i32) * fx.Int32(n_waves)
            p = fx.Int32(_c) * workgroup_size_i32 + tid
            valid = p < pair_bound
            safe_p = valid.select(p, zero_i32)
            ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
            is_mine = valid & (ex == expert_bid) & (present_block != zero_i32)
            v = is_mine.select(one_i32, zero_i32)
            # Load early so its latency hides behind the scan.
            w_val = buffer_ops.buffer_load(topk_weights_rsrc, safe_p, vec_width=1, dtype=fx.Float32)

            scan = v
            for step in range_constexpr(log2_warp):
                off = 1 << step
                in_range = lane >= off
                peer_lane = in_range.select(lane - off, zero_i32)
                peer = fx.Int32(fx.rocdl.ds_bpermute(T.i32, peer_lane * 4, scan))
                scan = scan + in_range.select(peer, zero_i32)
            if lane == WARP_SIZE - 1:
                fx.memref_store(scan, s_wave, bank + wave)
            gpu.barrier()
            wcross = zero_i32
            chunk_total = zero_i32
            for w in range_constexpr(n_waves):
                wv = fx.memref_load(s_wave, bank + fx.Int32(w))
                wcross = (wave > fx.Int32(w)).select(wcross + wv, wcross)
                chunk_total = chunk_total + wv
            inclusive = scan + wcross
            dst = position + inclusive - v

            if is_mine:
                token = safe_p // topk_i32
                slot = safe_p % topk_i32
                packed = (slot << fx.Int32(24)) | token
                buffer_ops.buffer_store(packed, sorted_ids_rsrc, dst)
                buffer_ops.buffer_store(w_val, sorted_weights_rsrc, dst)

            results = yield [position + chunk_total]
        return results

    @flyc.jit
    def _emit_scatter(
        tid,
        expert_bid,
        lane,
        wave,
        topk_ids_rsrc,
        topk_weights_rsrc,
        sorted_ids_rsrc,
        sorted_weights_rsrc,
        sorted_expert_ids_rsrc,
        num_valid_ids_rsrc,
        counts_rsrc,
        expert_mask_rsrc,
        num_local_tokens_rsrc,
        s_wave,
        s_excl,
    ):
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        neg_one_i32 = fx.Int32(-1)
        sentinel_i32 = fx.Int32(sentinel_id)
        tokens_i32 = fx.Int32(tokens)
        unit_size_i32 = fx.Int32(unit_size)
        unit_size_minus_one = fx.Int32(unit_size - 1)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        workgroup_size_minus_one = fx.Int32(workgroup_size - 1)
        experts_i32 = fx.Int32(experts)
        max_num_m_blocks_i32 = fx.Int32(max_num_m_blocks)
        pack_shift_i32 = fx.Int32(pack_shift)
        pack_lo_mask_i32 = fx.Int32(pack_lo_mask)

        # Step 1: padded-count prefix sum -> each expert's start offset + grand total.
        in_experts = tid < experts_i32
        safe_tid = in_experts.select(tid, zero_i32)
        cnt = in_experts.select(
            buffer_ops.buffer_load(counts_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
            zero_i32,
        )
        if const_expr(has_mask):
            em_val = in_experts.select(
                buffer_ops.buffer_load(expert_mask_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
                zero_i32,
            )
            present = (em_val != zero_i32).select(one_i32, zero_i32)
        else:
            present = one_i32
        padded = ((cnt + unit_size_minus_one) // unit_size_i32) * unit_size_i32

        if const_expr(has_mask):
            scan_val = (present << pack_shift_i32) | padded
        else:
            scan_val = padded

        scanned = scan_val
        for step in range_constexpr(log2_warp):
            off = 1 << step
            in_range = lane >= off
            peer_lane = in_range.select(lane - off, zero_i32)
            peer = fx.Int32(fx.rocdl.ds_bpermute(T.i32, peer_lane * 4, scanned))
            scanned = scanned + in_range.select(peer, zero_i32)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()
        cross = zero_i32
        total_scan = zero_i32
        for w in range_constexpr(n_waves):
            wv = fx.memref_load(s_wave, fx.Int32(w))
            cross = (wave > fx.Int32(w)).select(cross + wv, cross)
            total_scan = total_scan + wv
        inclusive_scan = scanned + cross

        if const_expr(has_mask):
            inclusive_padded = inclusive_scan & pack_lo_mask_i32
            total_padded = total_scan & pack_lo_mask_i32
            local_idx = (inclusive_scan >> pack_shift_i32) - present
        else:
            inclusive_padded = inclusive_scan
            total_padded = total_scan
            local_idx = tid
        exclusive_padded = inclusive_padded - padded

        # Publish (dense idx, start offset) per expert so each block reads its own from s_excl.
        if const_expr(has_mask):
            packed_excl = (local_idx << pack_shift_i32) | exclusive_padded
        else:
            packed_excl = exclusive_padded
        fx.memref_store(packed_excl, s_excl, tid)
        gpu.barrier()
        my_packed = fx.memref_load(s_excl, expert_bid)
        if const_expr(has_mask):
            my_start = my_packed & pack_lo_mask_i32
            my_local = my_packed >> pack_shift_i32
        else:
            my_start = my_packed
            my_local = expert_bid

        my_count = buffer_ops.buffer_load(counts_rsrc, expert_bid, vec_width=1, dtype=fx.Int32)
        if const_expr(has_mask):
            present_block = buffer_ops.buffer_load(expert_mask_rsrc, expert_bid, vec_width=1, dtype=fx.Int32)
        else:
            present_block = one_i32
        my_padded = ((my_count + unit_size_minus_one) // unit_size_i32) * unit_size_i32
        my_end_valid = my_start + my_count
        block_offset = my_start // unit_size_i32
        n_blocks = my_padded // unit_size_i32

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(num_local_tokens_rsrc, zero_i32, vec_width=1, dtype=fx.Int32)
            pair_bound = n_local * fx.Int32(topk)
        else:
            pair_bound = fx.Int32(num_pairs)

        # Step 2: sorted_expert_ids blocks.
        n_blk_iters = (n_blocks + workgroup_size_minus_one) // workgroup_size_i32
        for _jb in range(fx.Index(0), ArithValue(n_blk_iters).index_cast(T.index), fx.Index(1)):
            b = fx.Int32(_jb) * workgroup_size_i32 + tid
            if b < n_blocks:
                buffer_ops.buffer_store(my_local, sorted_expert_ids_rsrc, block_offset + b)

        # Step 3: scatter this expert's pairs in ascending pair-index order.
        _scatter_pairs(
            tid,
            expert_bid,
            lane,
            wave,
            my_start,
            present_block,
            pair_bound,
            topk_ids_rsrc,
            topk_weights_rsrc,
            sorted_ids_rsrc,
            sorted_weights_rsrc,
            s_wave,
        )

        # Step 4: sentinel padding tail.
        pad_amount = my_padded - my_count
        for _jp in range_constexpr(n_pad_iters):
            pi = fx.Int32(_jp) * workgroup_size_i32 + tid
            if pi < pad_amount:
                buffer_ops.buffer_store(sentinel_i32, sorted_ids_rsrc, my_end_valid + pi)

        # Step 5: num_valid_ids (block 0) + the -1 block tail (idempotent in all blocks).
        if (expert_bid == zero_i32) & (tid == zero_i32):
            buffer_ops.buffer_store(total_padded, num_valid_ids_rsrc, zero_i32)
            if const_expr(has_padding):
                nvi1 = buffer_ops.buffer_load(num_local_tokens_rsrc, zero_i32, vec_width=1, dtype=fx.Int32)
            else:
                nvi1 = tokens_i32
            buffer_ops.buffer_store(nvi1, num_valid_ids_rsrc, one_i32)
        total_blocks = total_padded // unit_size_i32
        for _jt in range(fx.Index(0), fx.Index(n_tail_iters), fx.Index(1)):
            b = total_blocks + fx.Int32(_jt) * workgroup_size_i32 + tid
            if b < max_num_m_blocks_i32:
                buffer_ops.buffer_store(neg_one_i32, sorted_expert_ids_rsrc, b)

    @flyc.jit
    def _emit_zero(tid, bid, moe_buf_rsrc, i32_moe_buf_elems):
        """Zero moe_buf on blocks experts.. via a vectorized grid-stride loop."""
        zero_i32 = fx.Int32(0)
        cw_i32 = fx.Int32(vec_width)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        experts_i32 = fx.Int32(experts)
        c_zero_vec = fx.Vector.filled(vec_width, 0, fx.Int32)

        total_vec = i32_moe_buf_elems // cw_i32
        num_zero_blocks = gpu.grid_dim.x - experts_i32
        gid_vec = (bid - experts_i32) * workgroup_size_i32 + tid
        stride_vec = num_zero_blocks * workgroup_size_i32
        n_iters = (total_vec + stride_vec - fx.Int32(1)) // stride_vec
        for _z in range(fx.Index(0), ArithValue(n_iters).index_cast(T.index), fx.Index(1)):
            idx = gid_vec + fx.Int32(_z) * stride_vec
            if idx < total_vec:
                buffer_ops.buffer_store(c_zero_vec, moe_buf_rsrc, idx * cw_i32)
        tail_start = total_vec * cw_i32
        if bid == experts_i32:
            ti = tail_start + tid
            if ti < i32_moe_buf_elems:
                buffer_ops.buffer_store(zero_i32, moe_buf_rsrc, ti)

    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def scatter_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
        counts: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_moe_buf_elems: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = gpu.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        lds = fx.SharedAllocator().allocate(ScanStorage).peek()
        s_wave = lds.s_wave.view(fx.make_layout(2 * n_waves, 1))
        s_excl = lds.s_excl.view(fx.make_layout(workgroup_size, 1))

        if bid >= fx.Int32(experts):
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf)
            _emit_zero(tid, bid, moe_buf_rsrc, i32_moe_buf_elems)
        if bid < fx.Int32(experts):
            _emit_scatter(
                tid,
                bid,
                lane,
                wave,
                buffer_ops.create_buffer_resource(topk_ids),
                buffer_ops.create_buffer_resource(topk_weights),
                buffer_ops.create_buffer_resource(sorted_ids),
                buffer_ops.create_buffer_resource(sorted_weights),
                buffer_ops.create_buffer_resource(sorted_expert_ids),
                buffer_ops.create_buffer_resource(num_valid_ids),
                buffer_ops.create_buffer_resource(counts),
                buffer_ops.create_buffer_resource(expert_mask),
                buffer_ops.create_buffer_resource(num_local_tokens),
                s_wave,
                s_excl,
            )

    @flyc.jit
    def launcher(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
        counts: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_moe_buf_elems: fx.Int32,
        n_zero_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        block = (workgroup_size, 1, 1)
        k1 = count_kernel(  # ty: ignore[call-non-callable]
            topk_ids, counts, expert_mask, num_local_tokens
        )
        k1.launch(grid=(experts, 1, 1), block=block, stream=stream)
        k2 = scatter_kernel(  # ty: ignore[call-non-callable]
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            expert_mask,
            num_local_tokens,
            counts,
            moe_buf,
            i32_moe_buf_elems,
        )
        k2.launch(
            grid=(fx.Int32(experts) + n_zero_blocks, 1, 1),
            block=block,
            stream=stream,
        )

    return launcher


@functools.cache
def get_moe_sorting_count_sort_kernel(
    *,
    experts: int,
    tokens: int,
    topk: int,
    unit_size: int,
    has_mask: bool = False,
    has_padding: bool = False,
    store_vec_width: int | None = None,
):
    return compile_count_sort(
        experts=experts,
        tokens=tokens,
        topk=topk,
        unit_size=unit_size,
        has_mask=has_mask,
        has_padding=has_padding,
        store_vec_width=store_vec_width,
    )


# ===========================================================================
# count_sort_paired -- high-token partition-by-pairs counting sort.
# ===========================================================================


def compile_count_sort_paired(
    *,
    experts: int,
    tokens: int,
    topk: int,
    unit_size: int,
    has_mask: bool = False,
    has_padding: bool = False,
    store_vec_width: int | None = None,
):
    """Build the count_sort_paired launcher: K1 (hist) -> K2 (colscan) -> K3 (structure) ->
    K4 (scatter), with the ``moe_buf`` zero folded onto K4's trailing blocks."""
    _require_supported(_structural_unsupported(experts))  # count_sort_paired: only the expert/workgroup bound

    vec_width = store_vec_width
    if vec_width is None:
        vec_width = default_store_vec_width()
    assert vec_width in (2, 4), f"store_vec_width ({vec_width}) must be 2 or 4"

    WARP_SIZE = _get_warp_size()
    # W is the block dim AND the pairs-per-block count (one thread per pair).
    workgroup_size = workgroup_width(experts, WARP_SIZE)
    assert workgroup_size % WARP_SIZE == 0
    n_waves = workgroup_size // WARP_SIZE
    assert n_waves <= WARP_SIZE
    log2_warp = int(math.log2(WARP_SIZE))
    # K4 rank: ballot rounds to fully discriminate an expert id (one per bit).
    nbits = max(1, (experts - 1).bit_length())

    num_pairs = tokens * topk
    max_num_tokens_padded = num_pairs + experts * unit_size - topk
    max_num_m_blocks = (max_num_tokens_padded + unit_size - 1) // unit_size
    sentinel_id = (topk << 24) | tokens
    n_pair_blocks = (num_pairs + workgroup_size - 1) // workgroup_size

    n_tail_iters = (max_num_m_blocks + workgroup_size - 1) // workgroup_size
    n_pad_iters = (unit_size + workgroup_size - 1) // workgroup_size

    # has_mask packs ``present`` (high bits) above ``padded`` (low bits) in one i32, so a
    # single scan carries both: low bits give the base offset, ``(scan >> pack_shift) -
    # present`` is the dense (compacted) expert index.
    pack_shift = max_num_tokens_padded.bit_length()
    assert (not has_mask) or pack_shift + max(1, experts.bit_length()) <= 31
    pack_lo_mask = (1 << pack_shift) - 1

    @fx.struct  # ty: ignore[too-many-positional-arguments]
    class ScanStorage:
        s_wave: fx.Array[fx.Int32, n_waves]
        s_excl: fx.Array[fx.Int32, workgroup_size]

    @fx.struct  # ty: ignore[too-many-positional-arguments]
    class RankStorage:
        # Per-expert accumulator shared by K1 (block histogram) and K4 (scatter rank).
        run: fx.Array[fx.Int32, experts]

    def _same_key_wave_rank(ex, is_expert, zero_i32, one_i32):
        """``(rank_wave, wave_total)`` for same-expert lanes, via match_any emulation.

        Each round ANDs in the lanes that agree with this lane on bit ``b`` (``set_b`` if
        this lane's bit is set, else its complement), so after ``nbits`` rounds ``match``
        is the same-expert lane mask: mbcnt(match) -> rank within the group, ctpop -> size.
        """

        def _bit_is_set(b):
            return ((ex >> fx.Int32(b)) & one_i32) == one_i32

        if WARP_SIZE == 64:
            active = fx.Int64(fx.rocdl.ballot(T.i64, _raw(is_expert)))
            match = active
            for _b in range_constexpr(nbits):
                set_b = fx.Int64(fx.rocdl.ballot(T.i64, _raw(is_expert & _bit_is_set(_b))))
                match = _bit_is_set(_b).select(match & set_b, match & (active & ~set_b))
            match_lo = fx.Int32(ArithValue(match).trunci(T.i32))
            match_hi = fx.Int32(ArithValue(match >> fx.Int64(32)).trunci(T.i32))
            lo = fx.Int32(fx.rocdl.mbcnt_lo(T.i32, _raw(match_lo), _raw(zero_i32)))
            rank_wave = fx.Int32(fx.rocdl.mbcnt_hi(T.i32, _raw(match_hi), _raw(lo)))
            wave_total = fx.Int32(ArithValue(fx.Int64(_llvm.intr_ctpop(_raw(match), results=[T.i64]))).trunci(T.i32))
        else:
            active = fx.Int32(fx.rocdl.ballot(T.i32, _raw(is_expert)))
            match = active
            for _b in range_constexpr(nbits):
                set_b = fx.Int32(fx.rocdl.ballot(T.i32, _raw(is_expert & _bit_is_set(_b))))
                match = _bit_is_set(_b).select(match & set_b, match & (active & ~set_b))
            rank_wave = fx.Int32(fx.rocdl.mbcnt_lo(T.i32, _raw(match), _raw(zero_i32)))
            wave_total = fx.Int32(_llvm.intr_ctpop(_raw(match), results=[T.i32]))
        return rank_wave, wave_total

    # ------------------------------------------------------------------ K1
    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def hist_kernel(
        topk_ids: fx.Tensor,
        hist: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
    ):
        """One block per pair-slice: histogram its pairs -> hist[block, :experts]."""
        tid = fx.thread_idx.x
        block_bid = gpu.block_idx.x
        wave = tid // fx.Int32(WARP_SIZE)

        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        num_pairs_i32 = fx.Int32(num_pairs)
        experts_i32 = fx.Int32(experts)

        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids)
        hist_rsrc = buffer_ops.create_buffer_resource(hist)

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource(num_local_tokens),
                zero_i32,
                vec_width=1,
                dtype=fx.Int32,
            )
            pair_bound = n_local * fx.Int32(topk)
        else:
            pair_bound = num_pairs_i32

        lds = fx.SharedAllocator().allocate(RankStorage).peek()
        run = lds.run.view(fx.make_layout(experts, 1))

        # Out-of-range / masked lanes fall out of every ballot (matches K4's drop).
        p = block_bid * workgroup_size_i32 + tid
        valid = p < pair_bound
        safe_p = valid.select(p, zero_i32)
        ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
        is_expert = valid & (ex >= zero_i32) & (ex < experts_i32)
        if const_expr(has_mask):
            mask_key = is_expert.select(ex, zero_i32)
            em_val = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource(expert_mask),
                mask_key,
                vec_width=1,
                dtype=fx.Int32,
            )
            is_expert = is_expert & (em_val != zero_i32)
        safe_key = is_expert.select(ex, zero_i32)

        rank_wave, wave_total = _same_key_wave_rank(ex, is_expert, zero_i32, one_i32)

        if tid < experts_i32:
            fx.memref_store(zero_i32, run, tid)
        gpu.barrier()

        # Wave w's lowest matching lane per expert folds that wave's total in.
        for _w in range_constexpr(n_waves):
            if (wave == fx.Int32(_w)) & is_expert & (rank_wave == zero_i32):
                cur = fx.memref_load(run, safe_key)
                fx.memref_store(cur + wave_total, run, safe_key)
            gpu.barrier()

        if tid < experts_i32:
            buffer_ops.buffer_store(fx.memref_load(run, tid), hist_rsrc, block_bid * experts_i32 + tid)

    # ------------------------------------------------------------------ K2
    # n_col_strip = num_pairs / W^2 is O(1) (a handful), so its constexpr unroll is fine.
    n_col_strip = (n_pair_blocks + workgroup_size - 1) // workgroup_size

    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def colscan_kernel(
        hist: fx.Tensor,
        counts: fx.Tensor,
    ):
        """One block per expert: cooperative exclusive prefix down its hist column."""
        tid = fx.thread_idx.x
        e = gpu.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        zero_i32 = fx.Int32(0)
        experts_i32 = fx.Int32(experts)
        n_pair_blocks_i32 = fx.Int32(n_pair_blocks)
        base_b = tid * fx.Int32(n_col_strip)

        hist_rsrc = buffer_ops.create_buffer_resource(hist)
        counts_rsrc = buffer_ops.create_buffer_resource(counts)
        lds = fx.SharedAllocator().allocate(ScanStorage).peek()
        s_wave = lds.s_wave.view(fx.make_layout(n_waves, 1))

        strip_total = zero_i32
        for j in range_constexpr(n_col_strip):
            b = base_b + fx.Int32(j)
            if b < n_pair_blocks_i32:
                strip_total = strip_total + buffer_ops.buffer_load(
                    hist_rsrc, b * experts_i32 + e, vec_width=1, dtype=fx.Int32
                )

        scanned = strip_total
        for step in range_constexpr(log2_warp):
            off = 1 << step
            in_range = lane >= off
            peer_lane = in_range.select(lane - off, zero_i32)
            peer = fx.Int32(fx.rocdl.ds_bpermute(T.i32, peer_lane * 4, scanned))
            scanned = scanned + in_range.select(peer, zero_i32)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()
        cross = zero_i32
        total = zero_i32
        for w in range_constexpr(n_waves):
            wv = fx.memref_load(s_wave, fx.Int32(w))
            cross = (wave > fx.Int32(w)).select(cross + wv, cross)
            total = total + wv
        strip_excl = scanned + cross - strip_total

        run = strip_excl
        for j in range_constexpr(n_col_strip):
            b = base_b + fx.Int32(j)
            if b < n_pair_blocks_i32:
                idx = b * experts_i32 + e
                v = buffer_ops.buffer_load(hist_rsrc, idx, vec_width=1, dtype=fx.Int32)
                buffer_ops.buffer_store(run, hist_rsrc, idx)
                run = run + v

        if tid == zero_i32:
            buffer_ops.buffer_store(total, counts_rsrc, e)

    # ------------------------------------------------------------------ K3
    @flyc.jit
    def _emit_structure(
        tid,
        expert_bid,
        lane,
        wave,
        sorted_ids_rsrc,
        sorted_expert_ids_rsrc,
        num_valid_ids_rsrc,
        counts_rsrc,
        estart_rsrc,
        expert_mask_rsrc,
        num_local_tokens_rsrc,
        s_wave,
        s_excl,
    ):
        """Padded-count prefix -> expert starts + sorted_expert_ids + sentinel tail +
        num_valid_ids + -1 tail. K4 fills the valid slots."""
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        neg_one_i32 = fx.Int32(-1)
        sentinel_i32 = fx.Int32(sentinel_id)
        tokens_i32 = fx.Int32(tokens)
        unit_size_i32 = fx.Int32(unit_size)
        unit_size_minus_one = fx.Int32(unit_size - 1)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        workgroup_size_minus_one = fx.Int32(workgroup_size - 1)
        experts_i32 = fx.Int32(experts)
        max_num_m_blocks_i32 = fx.Int32(max_num_m_blocks)
        pack_shift_i32 = fx.Int32(pack_shift)
        pack_lo_mask_i32 = fx.Int32(pack_lo_mask)

        in_experts = tid < experts_i32
        safe_tid = in_experts.select(tid, zero_i32)
        cnt = in_experts.select(
            buffer_ops.buffer_load(counts_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
            zero_i32,
        )
        if const_expr(has_mask):
            em_val = in_experts.select(
                buffer_ops.buffer_load(expert_mask_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
                zero_i32,
            )
            present = (em_val != zero_i32).select(one_i32, zero_i32)
        else:
            present = one_i32
        padded = ((cnt + unit_size_minus_one) // unit_size_i32) * unit_size_i32

        if const_expr(has_mask):
            scan_val = (present << pack_shift_i32) | padded
        else:
            scan_val = padded

        scanned = scan_val
        for step in range_constexpr(log2_warp):
            off = 1 << step
            in_range = lane >= off
            peer_lane = in_range.select(lane - off, zero_i32)
            peer = fx.Int32(fx.rocdl.ds_bpermute(T.i32, peer_lane * 4, scanned))
            scanned = scanned + in_range.select(peer, zero_i32)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()
        cross = zero_i32
        total_scan = zero_i32
        for w in range_constexpr(n_waves):
            wv = fx.memref_load(s_wave, fx.Int32(w))
            cross = (wave > fx.Int32(w)).select(cross + wv, cross)
            total_scan = total_scan + wv
        inclusive_scan = scanned + cross

        if const_expr(has_mask):
            inclusive_padded = inclusive_scan & pack_lo_mask_i32
            total_padded = total_scan & pack_lo_mask_i32
            local_idx = (inclusive_scan >> pack_shift_i32) - present
        else:
            inclusive_padded = inclusive_scan
            total_padded = total_scan
            local_idx = tid
        exclusive_padded = inclusive_padded - padded

        # Block 0 publishes every expert's start; the dense index rides s_excl's high bits.
        if (expert_bid == zero_i32) & in_experts:
            buffer_ops.buffer_store(exclusive_padded, estart_rsrc, tid)
        if const_expr(has_mask):
            packed_excl = (local_idx << pack_shift_i32) | exclusive_padded
        else:
            packed_excl = exclusive_padded
        fx.memref_store(packed_excl, s_excl, tid)
        gpu.barrier()
        my_packed = fx.memref_load(s_excl, expert_bid)
        if const_expr(has_mask):
            my_start = my_packed & pack_lo_mask_i32
            my_local = my_packed >> pack_shift_i32
        else:
            my_start = my_packed
            my_local = expert_bid

        my_count = buffer_ops.buffer_load(counts_rsrc, expert_bid, vec_width=1, dtype=fx.Int32)
        my_padded = ((my_count + unit_size_minus_one) // unit_size_i32) * unit_size_i32
        block_offset = my_start // unit_size_i32
        n_blocks = my_padded // unit_size_i32

        n_blk_iters = (n_blocks + workgroup_size_minus_one) // workgroup_size_i32
        for _jb in range(fx.Index(0), ArithValue(n_blk_iters).index_cast(T.index), fx.Index(1)):
            b = fx.Int32(_jb) * workgroup_size_i32 + tid
            if b < n_blocks:
                buffer_ops.buffer_store(my_local, sorted_expert_ids_rsrc, block_offset + b)

        my_end_valid = my_start + my_count
        pad_amount = my_padded - my_count
        for _jp in range_constexpr(n_pad_iters):
            pi = fx.Int32(_jp) * workgroup_size_i32 + tid
            if pi < pad_amount:
                buffer_ops.buffer_store(sentinel_i32, sorted_ids_rsrc, my_end_valid + pi)

        if (expert_bid == zero_i32) & (tid == zero_i32):
            buffer_ops.buffer_store(total_padded, num_valid_ids_rsrc, zero_i32)
            if const_expr(has_padding):
                nvi1 = buffer_ops.buffer_load(num_local_tokens_rsrc, zero_i32, vec_width=1, dtype=fx.Int32)
            else:
                nvi1 = tokens_i32
            buffer_ops.buffer_store(nvi1, num_valid_ids_rsrc, one_i32)
        total_blocks = total_padded // unit_size_i32
        for _jt in range(fx.Index(0), fx.Index(n_tail_iters), fx.Index(1)):
            b = total_blocks + fx.Int32(_jt) * workgroup_size_i32 + tid
            if b < max_num_m_blocks_i32:
                buffer_ops.buffer_store(neg_one_i32, sorted_expert_ids_rsrc, b)
        return exclusive_padded

    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def structure_kernel(
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        counts: fx.Tensor,
        estart: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid = gpu.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        lds = fx.SharedAllocator().allocate(ScanStorage).peek()
        s_wave = lds.s_wave.view(fx.make_layout(n_waves, 1))
        s_excl = lds.s_excl.view(fx.make_layout(workgroup_size, 1))
        _emit_structure(
            tid,
            bid,
            lane,
            wave,
            buffer_ops.create_buffer_resource(sorted_ids),
            buffer_ops.create_buffer_resource(sorted_expert_ids),
            buffer_ops.create_buffer_resource(num_valid_ids),
            buffer_ops.create_buffer_resource(counts),
            buffer_ops.create_buffer_resource(estart),
            buffer_ops.create_buffer_resource(expert_mask),
            buffer_ops.create_buffer_resource(num_local_tokens),
            s_wave,
            s_excl,
        )

    # ------------------------------------------------------------------ K4
    @flyc.jit
    def _emit_pair_scatter(
        tid,
        block_bid,
        topk_ids_rsrc,
        topk_weights_rsrc,
        sorted_ids_rsrc,
        sorted_weights_rsrc,
        hist_rsrc,
        estart_rsrc,
        expert_mask_rsrc,
        num_local_tokens_rsrc,
        run,
    ):
        """Scatter this block's pairs once to estart[e] + hist[block,e] + rank."""
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        topk_i32 = fx.Int32(topk)
        warp_i32 = fx.Int32(WARP_SIZE)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        num_pairs_i32 = fx.Int32(num_pairs)
        experts_i32 = fx.Int32(experts)
        wave = tid // warp_i32

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(num_local_tokens_rsrc, zero_i32, vec_width=1, dtype=fx.Int32)
            pair_bound = n_local * topk_i32
        else:
            pair_bound = num_pairs_i32

        p = block_bid * workgroup_size_i32 + tid
        valid = p < pair_bound
        safe_p = valid.select(p, zero_i32)
        ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
        is_expert = valid & (ex >= zero_i32) & (ex < experts_i32)
        if const_expr(has_mask):
            mask_key = is_expert.select(ex, zero_i32)
            em_val = buffer_ops.buffer_load(expert_mask_rsrc, mask_key, vec_width=1, dtype=fx.Int32)
            is_expert = is_expert & (em_val != zero_i32)
        safe_key = is_expert.select(ex, zero_i32)

        # Within-wave rank + a per-expert running count carried in wave order keeps
        # the rank monotone in pair index (the stable order).
        rank_wave, wave_total = _same_key_wave_rank(ex, is_expert, zero_i32, one_i32)

        if tid < experts_i32:
            fx.memref_store(zero_i32, run, tid)
        gpu.barrier()

        cross = zero_i32
        for _w in range_constexpr(n_waves):
            rv = fx.memref_load(run, safe_key)
            take = (wave == fx.Int32(_w)) & is_expert
            cross = take.select(rv, cross)
            gpu.barrier()
            if take & (rank_wave == zero_i32):
                cur = fx.memref_load(run, safe_key)
                fx.memref_store(cur + wave_total, run, safe_key)
            gpu.barrier()

        if is_expert:
            rank = cross + rank_wave
            block_base = buffer_ops.buffer_load(
                hist_rsrc, block_bid * experts_i32 + safe_key, vec_width=1, dtype=fx.Int32
            )
            base = buffer_ops.buffer_load(estart_rsrc, safe_key, vec_width=1, dtype=fx.Int32)
            dst = base + block_base + rank
            token = safe_p // topk_i32
            slot = safe_p % topk_i32
            packed = (slot << fx.Int32(24)) | token
            buffer_ops.buffer_store(packed, sorted_ids_rsrc, dst)
            w_val = buffer_ops.buffer_load(topk_weights_rsrc, safe_p, vec_width=1, dtype=fx.Float32)
            buffer_ops.buffer_store(w_val, sorted_weights_rsrc, dst)

    @flyc.jit
    def _emit_zero(tid, bid, moe_buf_rsrc, i32_moe_buf_elems):
        """Zero moe_buf on blocks n_pair_blocks.. via a vectorized grid-stride loop."""
        zero_i32 = fx.Int32(0)
        cw_i32 = fx.Int32(vec_width)
        workgroup_size_i32 = fx.Int32(workgroup_size)
        n_pair_blocks_i32 = fx.Int32(n_pair_blocks)
        c_zero_vec = fx.Vector.filled(vec_width, 0, fx.Int32)

        total_vec = i32_moe_buf_elems // cw_i32
        num_zero_blocks = gpu.grid_dim.x - n_pair_blocks_i32
        gid_vec = (bid - n_pair_blocks_i32) * workgroup_size_i32 + tid
        stride_vec = num_zero_blocks * workgroup_size_i32
        n_iters = (total_vec + stride_vec - fx.Int32(1)) // stride_vec
        for _z in range(fx.Index(0), ArithValue(n_iters).index_cast(T.index), fx.Index(1)):
            idx = gid_vec + fx.Int32(_z) * stride_vec
            if idx < total_vec:
                buffer_ops.buffer_store(c_zero_vec, moe_buf_rsrc, idx * cw_i32)
        tail_start = total_vec * cw_i32
        if bid == n_pair_blocks_i32:
            ti = tail_start + tid
            if ti < i32_moe_buf_elems:
                buffer_ops.buffer_store(zero_i32, moe_buf_rsrc, ti)

    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def scatter_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        hist: fx.Tensor,
        estart: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_moe_buf_elems: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = gpu.block_idx.x
        lds = fx.SharedAllocator().allocate(RankStorage).peek()
        run = lds.run.view(fx.make_layout(experts, 1))
        if bid >= fx.Int32(n_pair_blocks):
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf)
            _emit_zero(tid, bid, moe_buf_rsrc, i32_moe_buf_elems)
        if bid < fx.Int32(n_pair_blocks):
            _emit_pair_scatter(
                tid,
                bid,
                buffer_ops.create_buffer_resource(topk_ids),
                buffer_ops.create_buffer_resource(topk_weights),
                buffer_ops.create_buffer_resource(sorted_ids),
                buffer_ops.create_buffer_resource(sorted_weights),
                buffer_ops.create_buffer_resource(hist),
                buffer_ops.create_buffer_resource(estart),
                buffer_ops.create_buffer_resource(expert_mask),
                buffer_ops.create_buffer_resource(num_local_tokens),
                run,
            )

    @flyc.jit
    def launcher(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
        hist: fx.Tensor,
        counts: fx.Tensor,
        estart: fx.Tensor,
        moe_buf: fx.Tensor,
        i32_moe_buf_elems: fx.Int32,
        n_zero_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        block = (workgroup_size, 1, 1)
        k1 = hist_kernel(  # ty: ignore[call-non-callable]
            topk_ids, hist, expert_mask, num_local_tokens
        )
        k1.launch(grid=(n_pair_blocks, 1, 1), block=block, stream=stream)
        k2 = colscan_kernel(hist, counts)  # ty: ignore[call-non-callable]
        k2.launch(grid=(experts, 1, 1), block=block, stream=stream)
        k3 = structure_kernel(  # ty: ignore[call-non-callable]
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            counts,
            estart,
            expert_mask,
            num_local_tokens,
        )
        k3.launch(grid=(experts, 1, 1), block=block, stream=stream)
        k4 = scatter_kernel(  # ty: ignore[call-non-callable]
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            hist,
            estart,
            expert_mask,
            num_local_tokens,
            moe_buf,
            i32_moe_buf_elems,
        )
        k4.launch(
            grid=(fx.Int32(n_pair_blocks) + n_zero_blocks, 1, 1),
            block=block,
            stream=stream,
        )

    return launcher


@functools.cache
def get_moe_sorting_count_sort_paired_kernel(
    *,
    experts: int,
    tokens: int,
    topk: int,
    unit_size: int,
    has_mask: bool = False,
    has_padding: bool = False,
    store_vec_width: int | None = None,
):
    return compile_count_sort_paired(
        experts=experts,
        tokens=tokens,
        topk=topk,
        unit_size=unit_size,
        has_mask=has_mask,
        has_padding=has_padding,
        store_vec_width=store_vec_width,
    )


# ===========================================================================
# Selection + feasibility.
# ===========================================================================

MoESortingVariant = Literal["mesh_flatfill", "count_sort", "count_sort_paired"]
VARIANTS: tuple[MoESortingVariant, ...] = ("mesh_flatfill", "count_sort", "count_sort_paired")


def _variant_unsupported(variant: MoESortingVariant, *, experts: int, tokens: int) -> str | None:
    """Reason ``variant`` can't run this shape, or None. mesh_flatfill has the token/LDS
    ceiling; count_sort/count_sort_paired only have the structural (expert/workgroup) bound."""
    if variant == "mesh_flatfill":
        return mesh_flatfill_unsupported(experts, tokens)
    return _structural_unsupported(experts)


def mesh_flatfill_is_feasible(*, experts: int, tokens: int, topk: int, unit_size: int, arch: str | None = None) -> bool:
    """True iff mesh_flatfill can run this shape correctly (not merely fast)."""
    return mesh_flatfill_unsupported(experts, tokens) is None


def select_moe_sorting_kernel(
    *, num_experts: int, tokens: int, topk: int, unit_size: int, arch: str | None = None
) -> MoESortingVariant:
    """Fastest feasible variant for this shape (GPU-free). ``tokens`` is the padded
    count, so the crossover keys on ``workgroup_width(num_experts)``."""
    if mesh_flatfill_is_feasible(
        experts=num_experts, tokens=tokens, topk=topk, unit_size=unit_size, arch=arch
    ) and tokens <= mesh_flatfill_recommended_max_tokens(topk, arch):
        return "mesh_flatfill"
    workgroup_size = workgroup_width(num_experts)
    if tokens <= count_sort_crossover_tokens(workgroup_size, topk, arch):
        return "count_sort"
    return "count_sort_paired"


def _slow_reason(variant: MoESortingVariant, *, tokens: int, topk: int, experts: int, arch: str | None) -> str | None:
    """If ``variant`` is feasible but not the fastest for this shape, why; else None."""
    workgroup_size = workgroup_width(experts)
    if variant == "mesh_flatfill":
        rmax = mesh_flatfill_recommended_max_tokens(topk, arch)
        if tokens > rmax:
            return f"mesh_flatfill is tuned for tokens <= {rmax}; count_sort is faster at {tokens}"
    elif variant == "count_sort":
        crossover = count_sort_crossover_tokens(workgroup_size, topk, arch)
        if tokens > crossover:
            return f"above ~{crossover} tokens count_sort_paired is faster"
    elif variant == "count_sort_paired":
        crossover = count_sort_crossover_tokens(workgroup_size, topk, arch)
        if tokens < crossover:
            return f"below ~{crossover} tokens count_sort is faster"
    return None


def verify_moe_sorting_args(
    *,
    num_experts: int,
    tokens: int,
    topk: int,
    unit_size: int,
    arch: str | None = None,
    kernel: MoESortingVariant | None = None,
) -> bool:
    """True iff these args can run. Infeasible -> False; feasible-but-slow -> True
    after a ``warnings.warn``. ``kernel`` checks a forced variant, else the selected one."""
    variant = kernel or select_moe_sorting_kernel(
        num_experts=num_experts, tokens=tokens, topk=topk, unit_size=unit_size, arch=arch
    )
    if variant not in VARIANTS:
        raise ValueError(f"unknown kernel {variant!r}; expected one of {list(VARIANTS)}")
    if _variant_unsupported(variant, experts=num_experts, tokens=tokens) is not None:
        return False
    slow = _slow_reason(variant, tokens=tokens, topk=topk, experts=num_experts, arch=arch)
    if slow is not None:
        warnings.warn(
            f"moe_sorting {variant} supports (tokens={tokens}, experts={num_experts}, "
            f"topk={topk}, unit_size={unit_size}) but is not the fastest variant: {slow}",
            stacklevel=2,
        )
    return True


# ===========================================================================
# moe_buf zero grid + host entry.
# ===========================================================================


def moe_buf_zero_block_count(
    i32_moe_buf_elems: int,
    unit_size: int,
    num_cu: int,
    occupancy: int | None = None,
) -> int:
    """Zero-block count: one per ``unit_size`` chunk, capped at ``num_cu * occupancy``."""
    if occupancy is None:
        occupancy = default_zero_target_occupancy()
    return min((i32_moe_buf_elems + unit_size - 1) // unit_size, num_cu * occupancy)


def moe_buf_zero_grid(
    moe_buf: torch.Tensor, *, unit_size: int, zero_target_occupancy: int | None = None
) -> tuple[torch.Tensor, int, int]:
    """``(moe_buf_i32, i32_moe_buf_elems, n_grid)``: block 0 sorts, blocks 1..N zero."""
    moe_buf_i32 = moe_buf.view(torch.int32)
    moe_buf_elems = moe_buf_i32.numel()
    num_cu = torch.cuda.get_device_properties(moe_buf.device).multi_processor_count
    n_zero_blocks = moe_buf_zero_block_count(moe_buf_elems, unit_size, num_cu, zero_target_occupancy)
    return moe_buf_i32, moe_buf_elems, 1 + n_zero_blocks


def moe_sorting_flydsl(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    unit_size: int = UNIT_SIZE,
    expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    *,
    kernel: MoESortingVariant | None = None,
    store_vec_width: int | None = None,
    zero_target_occupancy: int | None = None,
    stream=None,
):
    """Self-dispatching MoE sort: pick the variant for the shape, then run it.

    All output tensors must be pre-allocated by the caller; the kernels write their
    own sentinels and fold the ``moe_buf`` zero into the launch. Selection keys on
    the padded token count ``M = topk_ids.shape[0]``, so ``num_local_tokens`` only
    bounds the valid work, never which variant runs. ``expert_mask`` /
    ``num_local_tokens`` being non-None flips the compile-time ``has_mask`` /
    ``has_padding`` flags; ``kernel=`` forces a variant; ``workspace`` is accepted
    for API parity but unused (each variant self-allocates its scratch).

    Returns ``(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)``.
    """
    del workspace  # variants self-allocate scratch
    device = topk_ids.device
    M = int(topk_ids.shape[0])
    topk = int(topk_ids.shape[1])

    variant = kernel or select_moe_sorting_kernel(num_experts=num_experts, tokens=M, topk=topk, unit_size=unit_size)
    # has_mask / has_padding are compile-time flags (part of the factory cache key).
    build_kw = dict(
        experts=num_experts,
        tokens=M,
        topk=topk,
        unit_size=unit_size,
        has_mask=expert_mask is not None,
        has_padding=num_local_tokens is not None,
        store_vec_width=store_vec_width,
    )

    # Read only under has_mask / has_padding; identity placeholders otherwise.
    if expert_mask is None:
        expert_mask = torch.ones(num_experts, dtype=torch.int32, device=device)
    if num_local_tokens is None:
        num_local_tokens = torch.tensor([M], dtype=torch.int32, device=device)

    if stream is None:
        stream = torch.cuda.current_stream()

    moe_buf_i32, moe_buf_elems, n_grid = moe_buf_zero_grid(
        moe_buf, unit_size=unit_size, zero_target_occupancy=zero_target_occupancy
    )

    if variant == "mesh_flatfill":
        launcher = get_moe_sorting_mesh_flatfill_kernel(**build_kw)
        launcher(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            expert_mask,
            num_local_tokens,
            moe_buf_i32,
            moe_buf_elems,
            n_grid,
            stream,
        )
    elif variant == "count_sort":
        launcher = get_moe_sorting_count_sort_kernel(**build_kw)
        counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        launcher(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            expert_mask,
            num_local_tokens,
            counts,
            moe_buf_i32,
            moe_buf_elems,
            n_grid - 1,
            stream,
        )
    elif variant == "count_sort_paired":
        launcher = get_moe_sorting_count_sort_paired_kernel(**build_kw)
        hist_elems, side_elems = count_sort_paired_scratch_elems(M, topk, num_experts)
        hist = torch.empty(hist_elems, dtype=torch.int32, device=device)
        counts = torch.empty(side_elems, dtype=torch.int32, device=device)
        estart = torch.empty(side_elems, dtype=torch.int32, device=device)
        launcher(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            expert_mask,
            num_local_tokens,
            hist,
            counts,
            estart,
            moe_buf_i32,
            moe_buf_elems,
            n_grid - 1,
            stream,
        )
    else:
        raise ValueError(f"unknown kernel {variant!r}; expected one of {list(VARIANTS)}")

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def compile_moe_sorting(
    *,
    num_experts: int,
    topk: int,
    max_tokens: int = 128,
    unit_size: int = UNIT_SIZE,
    has_mask: bool = False,
):
    """Compile and return the three variant launchers ``(mesh_flatfill, count_sort, count_sort_paired)``
    for a shape (``moe_sorting_flydsl`` selects between them per call)."""
    build_kw = dict(
        experts=num_experts,
        tokens=max_tokens,
        topk=topk,
        unit_size=unit_size,
        has_mask=has_mask,
        has_padding=False,
        store_vec_width=None,
    )
    return (
        get_moe_sorting_mesh_flatfill_kernel(**build_kw),
        get_moe_sorting_count_sort_kernel(**build_kw),
        get_moe_sorting_count_sort_paired_kernel(**build_kw),
    )
