# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Standalone FlyDSL MoE token sorting kernels.

Reorders router top-k pairs by expert into unit_size-aligned runs for batched
expert GEMM. Each path also zeros moe_buf through trailing grid blocks.

Inputs/outputs:
  - topk_ids: [tokens, topk] int32 expert ids; topk_weights: [tokens, topk] f32.
  - sorted_ids: packed token ids; padding slots hold the sentinel.
  - sorted_weights: gathered f32 weights for real pairs.
  - sorted_expert_ids: dense expert id per unit_size block; -1 after valid blocks.
  - num_valid_ids: [total_padded_pairs, num_local_tokens].

Packed token id: (topk_slot << 24) | token_id
  - upper 8 bits: topk slot.
  - lower 24 bits: token index.
  - sentinel: (topk << 24) | tokens.

Paths:
  - mesh_flatfill (low token):
        one sort block, expert x token LDS mesh, constexpr token loop.
  - count_sort (mid token):
        K1 counts per expert; K2 scans, scatters, and zeros moe_buf.
  - count_sort_paired (high token):
        K1 block histograms; K2 column scan; K3 structure; K4 scatter.

Constraints:
  - one thread per expert in scan/structure phases.
  - mesh_flatfill: tokens <= 256 and mesh LDS fits LDS_BYTES_PER_BLOCK.
  - mesh_flatfill assumes distinct routed experts within each token row.
  - store_vec_width in (2, 4); unit_size % store_vec_width == 0.
"""

import functools
import math
from typing import Literal

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

UNIT_SIZE = 32
MAX_THREADS_PER_BLOCK = 1024
LDS_BYTES_PER_BLOCK = 64 * 1024
WARP_SIZE_FALLBACK = 64
DEFAULT_STORE_VEC_WIDTH = 4


def _get_warp_size(arch=None) -> int:
    """Wave size for the active ROCm arch."""
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


@functools.lru_cache(maxsize=None)
def get_num_cu(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# -- host helpers --


def workgroup_width(experts: int, warp_size: int = WARP_SIZE_FALLBACK) -> int:
    waves = (experts + warp_size - 1) // warp_size
    return min(MAX_THREADS_PER_BLOCK, waves * warp_size)


def mesh_lds_bytes(experts: int, tokens: int) -> int:
    width = workgroup_width(experts)
    return (width * tokens + width) * 4


def count_sort_paired_scratch_elems(tokens: int, topk: int, experts: int) -> tuple[int, int]:
    """Scratch layout: hist[n_pair_blocks, experts], counts[experts], estart[experts]."""
    num_pairs = tokens * topk
    workgroup_size = workgroup_width(experts)
    n_pair_blocks = (num_pairs + workgroup_size - 1) // workgroup_size
    return n_pair_blocks * experts, experts


def _unwrap_val(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


DPP_ROW_SHR_1 = 0x111
DPP_ROW_SHR_2 = 0x112
DPP_ROW_SHR_4 = 0x114
DPP_ROW_SHR_8 = 0x118
DPP_ROW_MASK = 0xF
DPP_BANK_MASK = 0xF


def _dpp_intra_wave_prefix_sum(val, lane, log2_warp):
    """Wave-local inclusive sum: 4 DPP row-shifts then ds_bpermute cross-row carries."""
    warp_size = 1 << log2_warp

    val_raw = _unwrap_val(val)
    zero_raw = _unwrap_val(fx.Int32(0))

    for shift, dpp_op, threshold in [
        (1, DPP_ROW_SHR_1, 1),
        (2, DPP_ROW_SHR_2, 2),
        (4, DPP_ROW_SHR_4, 4),
        (8, DPP_ROW_SHR_8, 8),
    ]:
        remote = fx.rocdl.update_dpp(T.i32, zero_raw, val_raw, dpp_op, DPP_ROW_MASK, DPP_BANK_MASK, True)
        val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
        val_raw = _unwrap_val(val)

    src_lane_16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fx.rocdl.ds_bpermute(T.i32, src_lane_16 * fx.Int32(4), val)
    val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote16), val)

    if warp_size > 32:
        src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
        remote32 = fx.rocdl.ds_bpermute(T.i32, src_lane_32 * fx.Int32(4), val)
        val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote32), val)

    return val


def _wave_cross_and_total(s_wave, wave, n_waves):
    """Exclusive wave prefix and block total from per-wave totals in LDS."""
    cross = fx.Int32(0)
    total = fx.Int32(0)
    for w in range_constexpr(n_waves):
        wv = fx.memref_load(s_wave, fx.Int32(w))
        cross = (wave > fx.Int32(w)).select(cross + wv, cross)
        total = total + wv
    return cross, total


# -- shared device helpers --


@flyc.jit
def _zero_moe_buf_grid_stride(
    tid: fx.Int32,
    bid: fx.Int32,
    moe_buf_rsrc,
    i32_moe_buf_elems: fx.Int32,
    block_offset: fx.Int32,
    vec_width: fx.Constexpr[int],
    workgroup_size: fx.Constexpr[int],
):
    """Grid-stride moe_buf zero with vector body and scalar tail."""
    c_zero_i32 = fx.Int32(0)
    c_cw_i32 = fx.Int32(vec_width)
    c_workgroup_size_i32 = fx.Int32(workgroup_size)
    c_zero_vec = fx.Vector.filled(vec_width, 0, fx.Int32)

    total_vec = i32_moe_buf_elems // c_cw_i32
    num_zero_blocks = gpu.grid_dim.x - block_offset
    gid_vec = (bid - block_offset) * c_workgroup_size_i32 + tid
    stride_vec = num_zero_blocks * c_workgroup_size_i32
    n_iters = (total_vec + stride_vec - fx.Int32(1)) // stride_vec
    for _z in range(fx.Index(0), ArithValue(n_iters).index_cast(T.index), fx.Index(1)):
        idx = gid_vec + fx.Int32(_z) * stride_vec
        if idx < total_vec:
            buffer_ops.buffer_store(c_zero_vec, moe_buf_rsrc, idx * c_cw_i32)

    # Keep the scalar tail single-owner after vectorized stores.
    tail_start = total_vec * c_cw_i32
    if bid == block_offset:
        ti = tail_start + tid
        if ti < i32_moe_buf_elems:
            buffer_ops.buffer_store(c_zero_i32, moe_buf_rsrc, ti)


@flyc.jit
def _write_expert_id_blocks(
    tid: fx.Int32,
    my_local: fx.Int32,
    block_offset: fx.Int32,
    n_blocks: fx.Int32,
    sorted_expert_ids_rsrc,
    workgroup_size: fx.Constexpr[int],
):
    """Fill a contiguous sorted_expert_ids block run."""
    c_workgroup_size_i32 = fx.Int32(workgroup_size)
    n_blk_iters = (n_blocks + fx.Int32(workgroup_size - 1)) // c_workgroup_size_i32
    for _jb in range(fx.Index(0), ArithValue(n_blk_iters).index_cast(T.index), fx.Index(1)):
        b = fx.Int32(_jb) * c_workgroup_size_i32 + tid
        if b < n_blocks:
            buffer_ops.buffer_store(my_local, sorted_expert_ids_rsrc, block_offset + b)


@flyc.jit
def _fill_sentinel_pad(
    tid: fx.Int32,
    my_end_valid: fx.Int32,
    pad_amount: fx.Int32,
    sorted_ids_rsrc,
    sentinel_id: fx.Constexpr[int],
    n_pad_iters: fx.Constexpr[int],
    workgroup_size: fx.Constexpr[int],
):
    """Fill one expert's padding slots with the packed-id sentinel."""
    c_sentinel_i32 = fx.Int32(sentinel_id)
    c_workgroup_size_i32 = fx.Int32(workgroup_size)
    for _jp in range_constexpr(n_pad_iters):
        pi = fx.Int32(_jp) * c_workgroup_size_i32 + tid
        if pi < pad_amount:
            buffer_ops.buffer_store(c_sentinel_i32, sorted_ids_rsrc, my_end_valid + pi)


# -- mesh_flatfill: low-token LDS mesh path --


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
    """Build the mesh_flatfill path.

    Contract:
      - tokens is padded M; the per-token loop is constexpr-unrolled.
      - each token row must route to distinct experts (one mesh slot per expert).
      - has_padding bounds valid pairs by num_local_tokens.
      - has_mask drops masked experts and emits dense expert ids.

    Grid:
      - block 0 sorts through an expert x token LDS mesh.
      - blocks > 0 zero moe_buf.
    """
    assert moe_sorting_supported("mesh_flatfill", experts=experts, tokens=tokens), "mesh_flatfill: unsupported shape"

    WARP_SIZE = _get_warp_size()
    workgroup_size = workgroup_width(experts, WARP_SIZE)

    vec_width = store_vec_width
    if vec_width is None:
        vec_width = DEFAULT_STORE_VEC_WIDTH
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

    # Under has_mask, one scan carries both dense expert id and padded offset.
    pack_shift = max_num_tokens_padded.bit_length()
    assert (not has_mask) or pack_shift + max(1, experts.bit_length()) <= 31
    pack_lo_mask = (1 << pack_shift) - 1

    # Keep the mesh column in registers only below the VGPR budget ceiling.
    cache_column = tokens <= 64

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
        tid = gpu.thread_idx.x

        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        topk_weights_rsrc = buffer_ops.create_buffer_resource(topk_weights, max_size=True)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        sorted_weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        sorted_expert_ids_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
        num_valid_ids_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)

        expert_id = fx.Int32(tid)
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_neg_one_i32 = fx.Int32(-1)
        c_m_i32 = fx.Int32(tokens)
        c_topk_i32 = fx.Int32(topk)
        c_tokens_i32 = fx.Int32(tokens)
        c_unit_size_i32 = fx.Int32(unit_size)
        c_unit_size_minus_one = fx.Int32(unit_size - 1)
        c_workgroup_size_i32 = fx.Int32(workgroup_size)
        c_workgroup_size_minus_one = fx.Int32(workgroup_size - 1)
        c_max_num_m_blocks_i32 = fx.Int32(max_num_m_blocks)
        c_num_pairs_i32 = fx.Int32(num_pairs)
        c_experts_i32 = fx.Int32(experts)
        c_pack_shift_i32 = fx.Int32(pack_shift)
        c_pack_lo_mask_i32 = fx.Int32(pack_lo_mask)

        if const_expr(has_mask):
            expert_mask_rsrc = buffer_ops.create_buffer_resource(expert_mask, max_size=True)
            in_experts = expert_id < c_experts_i32
            safe_e = in_experts.select(expert_id, c_zero_i32)
            mask_val = buffer_ops.buffer_load(expert_mask_rsrc, safe_e, vec_width=1, dtype=fx.Int32)
            present = (in_experts & (mask_val != c_zero_i32)).select(c_one_i32, c_zero_i32)
        else:
            present = c_one_i32

        if const_expr(has_padding):
            num_local_tokens_rsrc = buffer_ops.create_buffer_resource(num_local_tokens, max_size=True)
            n_local = buffer_ops.buffer_load(num_local_tokens_rsrc, c_zero_i32, vec_width=1, dtype=fx.Int32)
            pair_bound = n_local * c_topk_i32
        else:
            n_local = c_m_i32
            pair_bound = c_num_pairs_i32

        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_mesh = lds.s_mesh.view(fx.make_layout(n_mesh, 1))
        s_total = lds.s_total.view(fx.make_layout(1, 1))

        # -- mesh fill --
        # Fill stores slot+1; zero remains the unrouted sentinel.
        for i in range_constexpr(0, n_mesh, workgroup_size):
            fx.memref_store(c_zero_i32, s_mesh, fx.Int32(i) + tid)
        gpu.barrier()

        for j in range_constexpr(n_load_iters):
            p = fx.Int32(j * workgroup_size) + tid
            if p < pair_bound:
                ex = buffer_ops.buffer_load(topk_ids_rsrc, p, vec_width=1, dtype=fx.Int32)
                token = p // c_topk_i32
                slot = p % c_topk_i32
                fx.memref_store(slot + c_one_i32, s_mesh, ex * c_tokens_i32 + token)
        gpu.barrier()

        # -- count / prefix --
        col_base = expert_id * c_tokens_i32
        count = c_zero_i32
        col_vals: list = []
        for t in range_constexpr(tokens):
            mv = fx.memref_load(s_mesh, col_base + fx.Int32(t))
            if const_expr(cache_column):
                col_vals.append(mv)
            count = count + (mv != c_zero_i32).select(c_one_i32, c_zero_i32)

        if const_expr(has_mask):
            count = (present != c_zero_i32).select(count, c_zero_i32)

        padded = ((count + c_unit_size_minus_one) // c_unit_size_i32) * c_unit_size_i32

        if const_expr(has_mask):
            scan_val = (present << c_pack_shift_i32) | padded
        else:
            scan_val = padded

        scanned = _dpp_intra_wave_prefix_sum(scan_val, lane, log2_warp)

        # Cross-wave prefix uses LDS only when one wave cannot cover all experts.
        if const_expr(needs_cross_wave_scan):
            s_wave = lds.s_wave.view(fx.make_layout(n_waves, 1))

            if lane == WARP_SIZE - 1:
                fx.memref_store(scanned, s_wave, wave)
            gpu.barrier()

            if wave == 0:
                in_n_waves = lane < n_waves
                safe_idx = in_n_waves.select(lane, c_zero_i32)
                loaded = fx.memref_load(s_wave, safe_idx)
                val = in_n_waves.select(loaded, c_zero_i32)

                val = _dpp_intra_wave_prefix_sum(val, lane, log2_warp)

                # Publish the scan total
                if lane == fx.Int32(n_waves - 1):
                    fx.memref_store(val, s_total, c_zero_i32)

                not_lane_0 = lane > 0
                peer_lane = not_lane_0.select(lane - 1, c_zero_i32)
                byte_addr = peer_lane * 4
                prev_raw = fx.rocdl.ds_bpermute(T.i32, byte_addr, val)
                prev = fx.Int32(prev_raw)
                exclusive_wave = not_lane_0.select(prev, c_zero_i32)

                if lane < n_waves:
                    fx.memref_store(exclusive_wave, s_wave, lane)
            gpu.barrier()

            wave_prefix = fx.memref_load(s_wave, wave)
            inclusive_scan = scanned + wave_prefix
            total_scan = fx.memref_load(s_total, c_zero_i32)
        else:
            inclusive_scan = scanned
            if tid == fx.Int32(experts - 1):
                fx.memref_store(inclusive_scan, s_total, c_zero_i32)
            gpu.barrier()
            total_scan = fx.memref_load(s_total, c_zero_i32)

        if const_expr(has_mask):
            inclusive_padded = inclusive_scan & c_pack_lo_mask_i32
            total_padded = total_scan & c_pack_lo_mask_i32
            local_idx = (inclusive_scan >> c_pack_shift_i32) - present
        else:
            inclusive_padded = inclusive_scan
            total_padded = total_scan
            local_idx = expert_id

        # -- scatter / tails --
        # Pre-fill the padded range; scatter overwrites valid slots.
        c_cw_i32 = fx.Int32(vec_width)
        sentinel_vec = fx.Vector.filled(vec_width, sentinel_id, fx.Int32)
        n_vec = total_padded // c_cw_i32
        n_fill = (n_vec + c_workgroup_size_minus_one) // c_workgroup_size_i32
        for _k in range(fx.Index(0), ArithValue(n_fill).index_cast(T.index), fx.Index(1)):
            vidx = fx.Int32(_k) * c_workgroup_size_i32 + tid
            if vidx < n_vec:
                buffer_ops.buffer_store(sentinel_vec, sorted_ids_rsrc, vidx * c_cw_i32)
        gpu.barrier()

        # Scatter valid pairs, then write this expert's output block ids.
        exclusive_padded = inclusive_padded - padded
        block_offset = exclusive_padded // c_unit_size_i32
        n_blocks = padded // c_unit_size_i32

        if tid < experts:
            # Issue weight loads before stores to give memory latency room.
            w_vals: list = []
            for t in range_constexpr(tokens):
                mv = col_vals[t] if const_expr(cache_column) else fx.memref_load(s_mesh, col_base + fx.Int32(t))
                safe_slot = (mv != c_zero_i32).select(mv - c_one_i32, c_zero_i32)
                w_vals.append(
                    buffer_ops.buffer_load(
                        topk_weights_rsrc,
                        fx.Int32(t) * c_topk_i32 + safe_slot,
                        vec_width=1,
                        dtype=fx.Float32,
                    )
                )

            pos = c_zero_i32
            for t in range_constexpr(tokens):
                if const_expr(cache_column):
                    mv = col_vals[t]
                else:
                    mv = fx.memref_load(s_mesh, col_base + fx.Int32(t))
                is_match = mv != c_zero_i32
                if const_expr(has_mask):
                    is_match = is_match & (present != c_zero_i32)
                slot = mv - c_one_i32
                write_pos = exclusive_padded + pos
                token_i = fx.Int32(t)
                packed = (slot << fx.Int32(24)) | token_i
                if is_match:
                    buffer_ops.buffer_store(packed, sorted_ids_rsrc, write_pos)
                    buffer_ops.buffer_store(w_vals[t], sorted_weights_rsrc, write_pos)
                pos = pos + is_match.select(c_one_i32, c_zero_i32)

            for _b in range(fx.Index(0), ArithValue(n_blocks).index_cast(T.index), fx.Index(1)):
                buffer_ops.buffer_store(local_idx, sorted_expert_ids_rsrc, block_offset + fx.Int32(_b))

        # Mark expert blocks beyond total_padded as invalid.
        n_valid_blocks = total_padded // c_unit_size_i32
        n_tail_iters = (c_max_num_m_blocks_i32 - n_valid_blocks + c_workgroup_size_minus_one) // c_workgroup_size_i32
        for _k in range(fx.Index(0), ArithValue(n_tail_iters).index_cast(T.index), fx.Index(1)):
            b = n_valid_blocks + fx.Int32(_k) * c_workgroup_size_i32 + tid
            if b < c_max_num_m_blocks_i32:
                buffer_ops.buffer_store(c_neg_one_i32, sorted_expert_ids_rsrc, b)

        # num_valid_ids contract: [post-pad pair count, local token count].
        if tid == fx.Int32(experts - 1):
            buffer_ops.buffer_store(inclusive_padded, num_valid_ids_rsrc, c_zero_i32)
            buffer_ops.buffer_store(n_local, num_valid_ids_rsrc, c_one_i32)

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
            tid = gpu.thread_idx.x
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
            _zero_moe_buf_grid_stride(
                tid,
                bid,
                moe_buf_rsrc,
                i32_moe_buf_elems,
                fx.Int32(1),
                vec_width,
                workgroup_size,
            )
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


# -- count_sort: mid-token two-kernel path --


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
    """Build the count_sort path.

    Phases:
      - K1: one block per expert, counts matching pairs.
      - K2: scans padded counts, scatters stable pair order, fills tails, zeros moe_buf.

    Constraints:
      - repeated experts in a token row are supported.
      - pair loops are runtime scf.for loops; trace size is independent of tokens.
    """
    assert moe_sorting_supported("count_sort", experts=experts, tokens=tokens), "count_sort: unsupported shape"

    vec_width = store_vec_width
    if vec_width is None:
        vec_width = DEFAULT_STORE_VEC_WIDTH
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

    n_chunks = (num_pairs + workgroup_size - 1) // workgroup_size
    n_tail_iters = (max_num_m_blocks + workgroup_size - 1) // workgroup_size
    n_pad_iters = (unit_size + workgroup_size - 1) // workgroup_size

    # Under has_mask, one scan carries both dense expert id and padded offset.
    pack_shift = max_num_tokens_padded.bit_length()
    assert (not has_mask) or pack_shift + max(1, experts.bit_length()) <= 31
    pack_lo_mask = (1 << pack_shift) - 1

    @fx.struct  # ty: ignore[too-many-positional-arguments]
    class ScanStorage:
        # Chunk parity selects a wave-total bank; same-bank reuse is two chunks apart.
        s_wave: fx.Array[fx.Int32, 2 * n_waves]
        s_excl: fx.Array[fx.Int32, workgroup_size]

    # -- K1 --
    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def count_kernel(
        topk_ids: fx.Tensor,
        counts: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
    ):
        """K1: one block per expert; writes counts[e]."""
        tid = gpu.thread_idx.x
        expert_bid = gpu.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_workgroup_size_i32 = fx.Int32(workgroup_size)
        c_num_pairs_i32 = fx.Int32(num_pairs)

        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        counts_rsrc = buffer_ops.create_buffer_resource(counts, max_size=True)

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource(num_local_tokens, max_size=True),
                c_zero_i32,
                vec_width=1,
                dtype=fx.Int32,
            )
            pair_bound = n_local * fx.Int32(topk)
        else:
            pair_bound = c_num_pairs_i32

        lds = fx.SharedAllocator().allocate(ScanStorage).peek()
        s_wave = lds.s_wave.view(fx.make_layout(n_waves, 1))

        for _c, state in range(  # ty: ignore[no-matching-overload, not-iterable]
            fx.Index(0),
            fx.Index(n_chunks),
            fx.Index(1),
            init=[c_zero_i32],
        ):
            acc = state[0]
            p = fx.Int32(_c) * c_workgroup_size_i32 + tid
            valid = p < pair_bound
            safe_p = valid.select(p, c_zero_i32)
            ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
            is_mine = valid & (ex == expert_bid)
            results = yield [acc + is_mine.select(c_one_i32, c_zero_i32)]
        local = results

        scanned = _dpp_intra_wave_prefix_sum(local, lane, log2_warp)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()

        if tid == c_zero_i32:
            total = c_zero_i32
            for w in range_constexpr(n_waves):
                total = total + fx.memref_load(s_wave, fx.Int32(w))
            if const_expr(has_mask):
                em_val = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource(expert_mask, max_size=True),
                    expert_bid,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                total = (em_val != c_zero_i32).select(total, c_zero_i32)
            buffer_ops.buffer_store(total, counts_rsrc, expert_bid)

    # -- K2 --
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
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_topk_i32 = fx.Int32(topk)
        c_workgroup_size_i32 = fx.Int32(workgroup_size)

        for _c, state in range(  # ty: ignore[no-matching-overload, not-iterable]
            fx.Index(0),
            fx.Index(n_chunks),
            fx.Index(1),
            init=[my_start],
        ):
            position = state[0]
            bank = (fx.Int32(_c) & c_one_i32) * fx.Int32(n_waves)
            p = fx.Int32(_c) * c_workgroup_size_i32 + tid
            valid = p < pair_bound
            safe_p = valid.select(p, c_zero_i32)
            ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
            is_mine = valid & (ex == expert_bid) & (present_block != c_zero_i32)
            v = is_mine.select(c_one_i32, c_zero_i32)
            # Load before scan so weight latency overlaps rank calculation.
            w_val = buffer_ops.buffer_load(topk_weights_rsrc, safe_p, vec_width=1, dtype=fx.Float32)

            scan = _dpp_intra_wave_prefix_sum(v, lane, log2_warp)
            if lane == WARP_SIZE - 1:
                fx.memref_store(scan, s_wave, bank + wave)
            gpu.barrier()
            wcross = c_zero_i32
            chunk_total = c_zero_i32
            for w in range_constexpr(n_waves):
                wv = fx.memref_load(s_wave, bank + fx.Int32(w))
                wcross = (wave > fx.Int32(w)).select(wcross + wv, wcross)
                chunk_total = chunk_total + wv
            inclusive = scan + wcross
            dst = position + inclusive - v

            if is_mine:
                token = safe_p // c_topk_i32
                slot = safe_p % c_topk_i32
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
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_neg_one_i32 = fx.Int32(-1)
        c_tokens_i32 = fx.Int32(tokens)
        c_unit_size_i32 = fx.Int32(unit_size)
        c_unit_size_minus_one = fx.Int32(unit_size - 1)
        c_workgroup_size_i32 = fx.Int32(workgroup_size)
        c_experts_i32 = fx.Int32(experts)
        c_max_num_m_blocks_i32 = fx.Int32(max_num_m_blocks)
        c_pack_shift_i32 = fx.Int32(pack_shift)
        c_pack_lo_mask_i32 = fx.Int32(pack_lo_mask)

        # -- prefix / structure --
        in_experts = tid < c_experts_i32
        safe_tid = in_experts.select(tid, c_zero_i32)
        cnt = in_experts.select(
            buffer_ops.buffer_load(counts_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
            c_zero_i32,
        )
        if const_expr(has_mask):
            em_val = in_experts.select(
                buffer_ops.buffer_load(expert_mask_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
                c_zero_i32,
            )
            present = (em_val != c_zero_i32).select(c_one_i32, c_zero_i32)
        else:
            present = c_one_i32
        padded = ((cnt + c_unit_size_minus_one) // c_unit_size_i32) * c_unit_size_i32

        if const_expr(has_mask):
            scan_val = (present << c_pack_shift_i32) | padded
        else:
            scan_val = padded

        scanned = _dpp_intra_wave_prefix_sum(scan_val, lane, log2_warp)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()
        cross, total_scan = _wave_cross_and_total(s_wave, wave, n_waves)
        inclusive_scan = scanned + cross

        if const_expr(has_mask):
            inclusive_padded = inclusive_scan & c_pack_lo_mask_i32
            total_padded = total_scan & c_pack_lo_mask_i32
            local_idx = (inclusive_scan >> c_pack_shift_i32) - present
        else:
            inclusive_padded = inclusive_scan
            total_padded = total_scan
            local_idx = tid
        exclusive_padded = inclusive_padded - padded

        # Publish per-expert start; dense id rides the high bits under has_mask.
        if const_expr(has_mask):
            packed_excl = (local_idx << c_pack_shift_i32) | exclusive_padded
        else:
            packed_excl = exclusive_padded
        fx.memref_store(packed_excl, s_excl, tid)
        gpu.barrier()
        my_packed = fx.memref_load(s_excl, expert_bid)
        if const_expr(has_mask):
            my_start = my_packed & c_pack_lo_mask_i32
            my_local = my_packed >> c_pack_shift_i32
        else:
            my_start = my_packed
            my_local = expert_bid

        my_count = buffer_ops.buffer_load(counts_rsrc, expert_bid, vec_width=1, dtype=fx.Int32)
        if const_expr(has_mask):
            present_block = buffer_ops.buffer_load(expert_mask_rsrc, expert_bid, vec_width=1, dtype=fx.Int32)
        else:
            present_block = c_one_i32
        my_padded = ((my_count + c_unit_size_minus_one) // c_unit_size_i32) * c_unit_size_i32
        my_end_valid = my_start + my_count
        block_offset = my_start // c_unit_size_i32
        n_blocks = my_padded // c_unit_size_i32

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(num_local_tokens_rsrc, c_zero_i32, vec_width=1, dtype=fx.Int32)
            pair_bound = n_local * fx.Int32(topk)
        else:
            pair_bound = fx.Int32(num_pairs)

        # -- scatter / tails --
        _write_expert_id_blocks(tid, my_local, block_offset, n_blocks, sorted_expert_ids_rsrc, workgroup_size)

        # Stable scatter in ascending pair-index order.
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

        pad_amount = my_padded - my_count
        _fill_sentinel_pad(tid, my_end_valid, pad_amount, sorted_ids_rsrc, sentinel_id, n_pad_iters, workgroup_size)

        # Block 0 publishes num_valid_ids; all blocks may idempotently fill -1 tail.
        if (expert_bid == c_zero_i32) & (tid == c_zero_i32):
            buffer_ops.buffer_store(total_padded, num_valid_ids_rsrc, c_zero_i32)
            if const_expr(has_padding):
                nvi1 = buffer_ops.buffer_load(num_local_tokens_rsrc, c_zero_i32, vec_width=1, dtype=fx.Int32)
            else:
                nvi1 = c_tokens_i32
            buffer_ops.buffer_store(nvi1, num_valid_ids_rsrc, c_one_i32)
        total_blocks = total_padded // c_unit_size_i32
        for _jt in range(fx.Index(0), fx.Index(n_tail_iters), fx.Index(1)):
            b = total_blocks + fx.Int32(_jt) * c_workgroup_size_i32 + tid
            if b < c_max_num_m_blocks_i32:
                buffer_ops.buffer_store(c_neg_one_i32, sorted_expert_ids_rsrc, b)

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
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        lds = fx.SharedAllocator().allocate(ScanStorage).peek()
        s_wave = lds.s_wave.view(fx.make_layout(2 * n_waves, 1))
        s_excl = lds.s_excl.view(fx.make_layout(workgroup_size, 1))

        if bid >= fx.Int32(experts):
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
            _zero_moe_buf_grid_stride(
                tid,
                bid,
                moe_buf_rsrc,
                i32_moe_buf_elems,
                fx.Int32(experts),
                vec_width,
                workgroup_size,
            )
        if bid < fx.Int32(experts):
            _emit_scatter(
                tid,
                bid,
                lane,
                wave,
                buffer_ops.create_buffer_resource(topk_ids, max_size=True),
                buffer_ops.create_buffer_resource(topk_weights, max_size=True),
                buffer_ops.create_buffer_resource(sorted_ids, max_size=True),
                buffer_ops.create_buffer_resource(sorted_weights, max_size=True),
                buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True),
                buffer_ops.create_buffer_resource(num_valid_ids, max_size=True),
                buffer_ops.create_buffer_resource(counts, max_size=True),
                buffer_ops.create_buffer_resource(expert_mask, max_size=True),
                buffer_ops.create_buffer_resource(num_local_tokens, max_size=True),
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
            topk_ids,
            counts,
            expert_mask,
            num_local_tokens,
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


# -- count_sort_paired: high-token pair-partitioned path --


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
    """Build the count_sort_paired path.
    Repeated experts in a token row are supported.
    Same-expert rank uses per-bit ballot match_any emulation.

    Phases:
      - K1: one block per pair slice, writes hist[block, expert].
      - K2: one block per expert, exclusive scan down hist[:, expert].
      - K3: expert starts, block ids, padding sentinels, num_valid_ids.
      - K4: scatter to estart[e] + hist[block,e] + same-block rank; zero moe_buf.
    """
    assert moe_sorting_supported("count_sort_paired", experts=experts, tokens=tokens), (
        "count_sort_paired: unsupported shape"
    )

    vec_width = store_vec_width
    if vec_width is None:
        vec_width = DEFAULT_STORE_VEC_WIDTH
    assert vec_width in (2, 4), f"store_vec_width ({vec_width}) must be 2 or 4"

    WARP_SIZE = _get_warp_size()
    # workgroup_size is both block size and pairs per pair block.
    workgroup_size = workgroup_width(experts, WARP_SIZE)
    assert workgroup_size % WARP_SIZE == 0
    n_waves = workgroup_size // WARP_SIZE
    assert n_waves <= WARP_SIZE
    log2_warp = int(math.log2(WARP_SIZE))
    # Same-expert rank uses one ballot-discrimination round per expert-id bit.
    nbits = max(1, (experts - 1).bit_length())

    num_pairs = tokens * topk
    max_num_tokens_padded = num_pairs + experts * unit_size - topk
    max_num_m_blocks = (max_num_tokens_padded + unit_size - 1) // unit_size
    sentinel_id = (topk << 24) | tokens
    n_pair_blocks = (num_pairs + workgroup_size - 1) // workgroup_size

    n_tail_iters = (max_num_m_blocks + workgroup_size - 1) // workgroup_size
    n_pad_iters = (unit_size + workgroup_size - 1) // workgroup_size

    # Under has_mask, one scan carries both dense expert id and padded offset.
    pack_shift = max_num_tokens_padded.bit_length()
    assert (not has_mask) or pack_shift + max(1, experts.bit_length()) <= 31
    pack_lo_mask = (1 << pack_shift) - 1

    @fx.struct  # ty: ignore[too-many-positional-arguments]
    class ScanStorage:
        s_wave: fx.Array[fx.Int32, n_waves]
        s_excl: fx.Array[fx.Int32, workgroup_size]

    @fx.struct  # ty: ignore[too-many-positional-arguments]
    class RankStorage:
        # Per-expert accumulator for block histograms and per-block scatter rank.
        run: fx.Array[fx.Int32, experts]

    def _same_key_wave_rank(ex, is_expert, c_zero_i32, c_one_i32):
        """Wave rank among lanes with the same expert id."""

        def _bit_is_set(b):
            return ((ex >> fx.Int32(b)) & c_one_i32) == c_one_i32

        if WARP_SIZE == 64:
            active = fx.Int64(fx.rocdl.ballot(T.i64, _unwrap_val(is_expert)))
            match = active
            for _b in range_constexpr(nbits):
                set_b = fx.Int64(fx.rocdl.ballot(T.i64, _unwrap_val(is_expert & _bit_is_set(_b))))
                match = _bit_is_set(_b).select(match & set_b, match & (active & ~set_b))
            match_lo = fx.Int32(ArithValue(match).trunci(T.i32))
            match_hi = fx.Int32(ArithValue(match >> fx.Int64(32)).trunci(T.i32))
            lo = fx.Int32(fx.rocdl.mbcnt_lo(T.i32, _unwrap_val(match_lo), _unwrap_val(c_zero_i32)))
            rank_wave = fx.Int32(fx.rocdl.mbcnt_hi(T.i32, _unwrap_val(match_hi), _unwrap_val(lo)))
            wave_total = fx.Int32(
                ArithValue(fx.Int64(_llvm.intr_ctpop(_unwrap_val(match), results=[T.i64]))).trunci(T.i32)
            )
        else:
            active = fx.Int32(fx.rocdl.ballot(T.i32, _unwrap_val(is_expert)))
            match = active
            for _b in range_constexpr(nbits):
                set_b = fx.Int32(fx.rocdl.ballot(T.i32, _unwrap_val(is_expert & _bit_is_set(_b))))
                match = _bit_is_set(_b).select(match & set_b, match & (active & ~set_b))
            rank_wave = fx.Int32(fx.rocdl.mbcnt_lo(T.i32, _unwrap_val(match), _unwrap_val(c_zero_i32)))
            wave_total = fx.Int32(_llvm.intr_ctpop(_unwrap_val(match), results=[T.i32]))
        return rank_wave, wave_total

    # -- K1 --
    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def hist_kernel(
        topk_ids: fx.Tensor,
        hist: fx.Tensor,
        expert_mask: fx.Tensor,
        num_local_tokens: fx.Tensor,
    ):
        """K1: one block per pair slice, writes hist[block, expert]."""
        tid = gpu.thread_idx.x
        block_bid = gpu.block_idx.x
        wave = tid // fx.Int32(WARP_SIZE)

        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_workgroup_size_i32 = fx.Int32(workgroup_size)
        c_num_pairs_i32 = fx.Int32(num_pairs)
        c_experts_i32 = fx.Int32(experts)

        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        hist_rsrc = buffer_ops.create_buffer_resource(hist, max_size=True)

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource(num_local_tokens, max_size=True),
                c_zero_i32,
                vec_width=1,
                dtype=fx.Int32,
            )
            pair_bound = n_local * fx.Int32(topk)
        else:
            pair_bound = c_num_pairs_i32

        lds = fx.SharedAllocator().allocate(RankStorage).peek()
        run = lds.run.view(fx.make_layout(experts, 1))

        # -- pair histogram --
        # Invalid or masked lanes are excluded from every same-key ballot.
        p = block_bid * c_workgroup_size_i32 + tid
        valid = p < pair_bound
        safe_p = valid.select(p, c_zero_i32)
        ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
        is_expert = valid & (ex >= c_zero_i32) & (ex < c_experts_i32)
        if const_expr(has_mask):
            mask_key = is_expert.select(ex, c_zero_i32)
            em_val = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource(expert_mask, max_size=True),
                mask_key,
                vec_width=1,
                dtype=fx.Int32,
            )
            is_expert = is_expert & (em_val != c_zero_i32)
        safe_key = is_expert.select(ex, c_zero_i32)

        rank_wave, wave_total = _same_key_wave_rank(ex, is_expert, c_zero_i32, c_one_i32)

        if tid < c_experts_i32:
            fx.memref_store(c_zero_i32, run, tid)
        gpu.barrier()

        # One lane per expert per wave contributes that wave's group total.
        for _w in range_constexpr(n_waves):
            if (wave == fx.Int32(_w)) & is_expert & (rank_wave == c_zero_i32):
                cur = fx.memref_load(run, safe_key)
                fx.memref_store(cur + wave_total, run, safe_key)
            gpu.barrier()

        if tid < c_experts_i32:
            buffer_ops.buffer_store(fx.memref_load(run, tid), hist_rsrc, block_bid * c_experts_i32 + tid)

    # -- K2 --
    # n_col_strip is small enough to constexpr-unroll.
    n_col_strip = (n_pair_blocks + workgroup_size - 1) // workgroup_size

    @flyc.kernel(known_block_size=[workgroup_size, 1, 1])
    def colscan_kernel(
        hist: fx.Tensor,
        counts: fx.Tensor,
    ):
        """K2: one block per expert, exclusive scan down hist[:, expert]."""
        tid = gpu.thread_idx.x
        e = gpu.block_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        c_zero_i32 = fx.Int32(0)
        c_experts_i32 = fx.Int32(experts)
        c_n_pair_blocks_i32 = fx.Int32(n_pair_blocks)
        base_b = tid * fx.Int32(n_col_strip)

        hist_rsrc = buffer_ops.create_buffer_resource(hist, max_size=True)
        counts_rsrc = buffer_ops.create_buffer_resource(counts, max_size=True)
        lds = fx.SharedAllocator().allocate(ScanStorage).peek()
        s_wave = lds.s_wave.view(fx.make_layout(n_waves, 1))

        # -- column prefix --
        strip_total = c_zero_i32
        for j in range_constexpr(n_col_strip):
            b = base_b + fx.Int32(j)
            if b < c_n_pair_blocks_i32:
                strip_total = strip_total + buffer_ops.buffer_load(
                    hist_rsrc,
                    b * c_experts_i32 + e,
                    vec_width=1,
                    dtype=fx.Int32,
                )

        scanned = _dpp_intra_wave_prefix_sum(strip_total, lane, log2_warp)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()
        cross, total = _wave_cross_and_total(s_wave, wave, n_waves)
        strip_excl = scanned + cross - strip_total

        run = strip_excl
        for j in range_constexpr(n_col_strip):
            b = base_b + fx.Int32(j)
            if b < c_n_pair_blocks_i32:
                idx = b * c_experts_i32 + e
                v = buffer_ops.buffer_load(hist_rsrc, idx, vec_width=1, dtype=fx.Int32)
                buffer_ops.buffer_store(run, hist_rsrc, idx)
                run = run + v

        if tid == c_zero_i32:
            buffer_ops.buffer_store(total, counts_rsrc, e)

    # -- K3 --
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
        """K3: expert starts, block ids, padding sentinels, num_valid_ids."""
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_neg_one_i32 = fx.Int32(-1)
        c_tokens_i32 = fx.Int32(tokens)
        c_unit_size_i32 = fx.Int32(unit_size)
        c_unit_size_minus_one = fx.Int32(unit_size - 1)
        c_workgroup_size_i32 = fx.Int32(workgroup_size)
        c_experts_i32 = fx.Int32(experts)
        c_max_num_m_blocks_i32 = fx.Int32(max_num_m_blocks)
        c_pack_shift_i32 = fx.Int32(pack_shift)
        c_pack_lo_mask_i32 = fx.Int32(pack_lo_mask)

        in_experts = tid < c_experts_i32
        safe_tid = in_experts.select(tid, c_zero_i32)
        cnt = in_experts.select(
            buffer_ops.buffer_load(counts_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
            c_zero_i32,
        )
        if const_expr(has_mask):
            em_val = in_experts.select(
                buffer_ops.buffer_load(expert_mask_rsrc, safe_tid, vec_width=1, dtype=fx.Int32),
                c_zero_i32,
            )
            present = (em_val != c_zero_i32).select(c_one_i32, c_zero_i32)
        else:
            present = c_one_i32
        padded = ((cnt + c_unit_size_minus_one) // c_unit_size_i32) * c_unit_size_i32

        if const_expr(has_mask):
            scan_val = (present << c_pack_shift_i32) | padded
        else:
            scan_val = padded

        scanned = _dpp_intra_wave_prefix_sum(scan_val, lane, log2_warp)
        if lane == WARP_SIZE - 1:
            fx.memref_store(scanned, s_wave, wave)
        gpu.barrier()
        cross, total_scan = _wave_cross_and_total(s_wave, wave, n_waves)
        inclusive_scan = scanned + cross

        if const_expr(has_mask):
            inclusive_padded = inclusive_scan & c_pack_lo_mask_i32
            total_padded = total_scan & c_pack_lo_mask_i32
            local_idx = (inclusive_scan >> c_pack_shift_i32) - present
        else:
            inclusive_padded = inclusive_scan
            total_padded = total_scan
            local_idx = tid
        exclusive_padded = inclusive_padded - padded

        # -- expert structure --
        # Block 0 publishes estart; s_excl carries dense id when has_mask is set.
        if (expert_bid == c_zero_i32) & in_experts:
            buffer_ops.buffer_store(exclusive_padded, estart_rsrc, tid)
        if const_expr(has_mask):
            packed_excl = (local_idx << c_pack_shift_i32) | exclusive_padded
        else:
            packed_excl = exclusive_padded
        fx.memref_store(packed_excl, s_excl, tid)
        gpu.barrier()
        my_packed = fx.memref_load(s_excl, expert_bid)
        if const_expr(has_mask):
            my_start = my_packed & c_pack_lo_mask_i32
            my_local = my_packed >> c_pack_shift_i32
        else:
            my_start = my_packed
            my_local = expert_bid

        my_count = buffer_ops.buffer_load(counts_rsrc, expert_bid, vec_width=1, dtype=fx.Int32)
        my_padded = ((my_count + c_unit_size_minus_one) // c_unit_size_i32) * c_unit_size_i32
        block_offset = my_start // c_unit_size_i32
        n_blocks = my_padded // c_unit_size_i32

        _write_expert_id_blocks(tid, my_local, block_offset, n_blocks, sorted_expert_ids_rsrc, workgroup_size)

        my_end_valid = my_start + my_count
        pad_amount = my_padded - my_count
        _fill_sentinel_pad(tid, my_end_valid, pad_amount, sorted_ids_rsrc, sentinel_id, n_pad_iters, workgroup_size)

        if (expert_bid == c_zero_i32) & (tid == c_zero_i32):
            buffer_ops.buffer_store(total_padded, num_valid_ids_rsrc, c_zero_i32)
            if const_expr(has_padding):
                nvi1 = buffer_ops.buffer_load(num_local_tokens_rsrc, c_zero_i32, vec_width=1, dtype=fx.Int32)
            else:
                nvi1 = c_tokens_i32
            buffer_ops.buffer_store(nvi1, num_valid_ids_rsrc, c_one_i32)
        total_blocks = total_padded // c_unit_size_i32
        for _jt in range(fx.Index(0), fx.Index(n_tail_iters), fx.Index(1)):
            b = total_blocks + fx.Int32(_jt) * c_workgroup_size_i32 + tid
            if b < c_max_num_m_blocks_i32:
                buffer_ops.buffer_store(c_neg_one_i32, sorted_expert_ids_rsrc, b)
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
        tid = gpu.thread_idx.x
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
            buffer_ops.create_buffer_resource(sorted_ids, max_size=True),
            buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True),
            buffer_ops.create_buffer_resource(num_valid_ids, max_size=True),
            buffer_ops.create_buffer_resource(counts, max_size=True),
            buffer_ops.create_buffer_resource(estart, max_size=True),
            buffer_ops.create_buffer_resource(expert_mask, max_size=True),
            buffer_ops.create_buffer_resource(num_local_tokens, max_size=True),
            s_wave,
            s_excl,
        )

    # -- K4 --
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
        """K4: scatter pairs to estart[e] + hist[block,e] + same-block rank."""
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_topk_i32 = fx.Int32(topk)
        c_warp_i32 = fx.Int32(WARP_SIZE)
        c_workgroup_size_i32 = fx.Int32(workgroup_size)
        c_num_pairs_i32 = fx.Int32(num_pairs)
        c_experts_i32 = fx.Int32(experts)
        wave = tid // c_warp_i32

        if const_expr(has_padding):
            n_local = buffer_ops.buffer_load(num_local_tokens_rsrc, c_zero_i32, vec_width=1, dtype=fx.Int32)
            pair_bound = n_local * c_topk_i32
        else:
            pair_bound = c_num_pairs_i32

        p = block_bid * c_workgroup_size_i32 + tid
        valid = p < pair_bound
        safe_p = valid.select(p, c_zero_i32)
        ex = buffer_ops.buffer_load(topk_ids_rsrc, safe_p, vec_width=1, dtype=fx.Int32)
        is_expert = valid & (ex >= c_zero_i32) & (ex < c_experts_i32)
        if const_expr(has_mask):
            mask_key = is_expert.select(ex, c_zero_i32)
            em_val = buffer_ops.buffer_load(expert_mask_rsrc, mask_key, vec_width=1, dtype=fx.Int32)
            is_expert = is_expert & (em_val != c_zero_i32)
        safe_key = is_expert.select(ex, c_zero_i32)

        # -- pair scatter --
        # Wave-order accumulator preserves stable pair-index order.
        rank_wave, wave_total = _same_key_wave_rank(ex, is_expert, c_zero_i32, c_one_i32)

        if tid < c_experts_i32:
            fx.memref_store(c_zero_i32, run, tid)
        gpu.barrier()

        cross = c_zero_i32
        for _w in range_constexpr(n_waves):
            rv = fx.memref_load(run, safe_key)
            take = (wave == fx.Int32(_w)) & is_expert
            cross = take.select(rv, cross)
            gpu.barrier()
            if take & (rank_wave == c_zero_i32):
                cur = fx.memref_load(run, safe_key)
                fx.memref_store(cur + wave_total, run, safe_key)
            gpu.barrier()

        if is_expert:
            rank = cross + rank_wave
            block_base = buffer_ops.buffer_load(
                hist_rsrc,
                block_bid * c_experts_i32 + safe_key,
                vec_width=1,
                dtype=fx.Int32,
            )
            base = buffer_ops.buffer_load(estart_rsrc, safe_key, vec_width=1, dtype=fx.Int32)
            dst = base + block_base + rank
            token = safe_p // c_topk_i32
            slot = safe_p % c_topk_i32
            packed = (slot << fx.Int32(24)) | token
            buffer_ops.buffer_store(packed, sorted_ids_rsrc, dst)
            w_val = buffer_ops.buffer_load(topk_weights_rsrc, safe_p, vec_width=1, dtype=fx.Float32)
            buffer_ops.buffer_store(w_val, sorted_weights_rsrc, dst)

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
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        lds = fx.SharedAllocator().allocate(RankStorage).peek()
        run = lds.run.view(fx.make_layout(experts, 1))
        if bid >= fx.Int32(n_pair_blocks):
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
            _zero_moe_buf_grid_stride(
                tid,
                bid,
                moe_buf_rsrc,
                i32_moe_buf_elems,
                fx.Int32(n_pair_blocks),
                vec_width,
                workgroup_size,
            )
        if bid < fx.Int32(n_pair_blocks):
            _emit_pair_scatter(
                tid,
                bid,
                buffer_ops.create_buffer_resource(topk_ids, max_size=True),
                buffer_ops.create_buffer_resource(topk_weights, max_size=True),
                buffer_ops.create_buffer_resource(sorted_ids, max_size=True),
                buffer_ops.create_buffer_resource(sorted_weights, max_size=True),
                buffer_ops.create_buffer_resource(hist, max_size=True),
                buffer_ops.create_buffer_resource(estart, max_size=True),
                buffer_ops.create_buffer_resource(expert_mask, max_size=True),
                buffer_ops.create_buffer_resource(num_local_tokens, max_size=True),
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
            topk_ids,
            hist,
            expert_mask,
            num_local_tokens,
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


# -- selection and feasibility --

MoESortingVariant = Literal["mesh_flatfill", "count_sort", "count_sort_paired"]
VARIANTS: tuple[MoESortingVariant, ...] = ("mesh_flatfill", "count_sort", "count_sort_paired")


def moe_sorting_supported(variant: MoESortingVariant, *, experts: int, tokens: int) -> bool:
    """Shape gate for per-expert workgroups and mesh_flatfill LDS/unroll limits."""
    workgroup_size = workgroup_width(experts)

    if experts > workgroup_size:
        return False
    if workgroup_size > MAX_THREADS_PER_BLOCK:
        return False

    if variant == "mesh_flatfill":
        if tokens > 256:
            return False

        used = mesh_lds_bytes(experts, tokens)
        if used > LDS_BYTES_PER_BLOCK:
            return False
    return True


def select_moe_sorting_kernel(
    *, num_experts: int, tokens: int, topk: int, arch: str | None = None
) -> MoESortingVariant:
    """Select mesh_flatfill, count_sort, or count_sort_paired from token regime."""
    meshff_is_supported = moe_sorting_supported("mesh_flatfill", experts=num_experts, tokens=tokens)

    arch = arch or get_rocm_arch()
    if arch.startswith("gfx94"):
        m = 256
    elif arch.startswith("gfx95"):
        m = 320
    else:
        m = 256

    # m = num_pairs budget (tokens * topk)
    # Convert the pair-budget to tokens.
    # 7 was empirically found to be a good fit.
    meshff_max_tokens = max(1, m // (topk + 7))
    if meshff_is_supported and tokens <= meshff_max_tokens:
        return "mesh_flatfill"

    if arch.startswith("gfx94"):
        chunks = 8
    elif arch.startswith("gfx95"):
        chunks = 10
    else:
        chunks = 8

    workgroup_size = workgroup_width(num_experts)
    count_sort_max_tokens = max(1, chunks * workgroup_size // topk)
    if tokens <= count_sort_max_tokens:
        return "count_sort"

    return "count_sort_paired"


# -- moe_buf zero grid and host entry --


def moe_buf_zero_block_count(
    i32_moe_buf_elems: int,
    unit_size: int,
    num_cu: int,
    occupancy: int | None = None,
) -> int:
    """Number of trailing zero blocks capped by moe_buf coverage and occupancy."""
    occupancy = occupancy or 2
    return min((i32_moe_buf_elems + unit_size - 1) // unit_size, num_cu * occupancy)


def moe_buf_zero_grid(
    moe_buf: torch.Tensor, *, unit_size: int, zero_target_occupancy: int | None = None
) -> tuple[torch.Tensor, int, int]:
    """Return i32 moe_buf view, element count, and one-plus-zero-block count."""
    moe_buf_i32 = moe_buf.view(torch.int32)
    moe_buf_elems = moe_buf_i32.numel()
    num_cu = get_num_cu(moe_buf.device.index)
    n_zero_blocks = moe_buf_zero_block_count(moe_buf_elems, unit_size, num_cu, zero_target_occupancy)
    return moe_buf_i32, moe_buf_elems, 1 + n_zero_blocks


def moe_sorting_get_workspace_size(
    M: int,
    num_experts: int,
    topk: int,
    unit_size: int = UNIT_SIZE,
    *,
    kernel: MoESortingVariant | None = None,
) -> int:
    """Scratch size in i32 elements for the selected sorting path."""
    variant = kernel or select_moe_sorting_kernel(num_experts=num_experts, tokens=M, topk=topk)
    if variant == "mesh_flatfill":
        return 0
    if variant == "count_sort":
        return num_experts
    if variant == "count_sort_paired":
        hist_elems, side_elems = count_sort_paired_scratch_elems(M, topk, num_experts)
        return hist_elems + 2 * side_elems
    raise ValueError(f"unknown moe_sorting kernel {variant!r}")


_moe_sorting_cf_cache: dict = {}


def _launch_cached(cache, key, launch_fn, args, stream):
    """AOT-compiled dispatch: cache keyed by constexpr values."""
    cf = cache.get(key)
    stream_arg = fx.Stream(stream)
    if cf is not None:
        cf(*args, stream_arg)
    else:
        launch_fn(*args, stream=stream)
        cf = flyc.compile(launch_fn, *args, stream_arg)
        cache[key] = cf


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
    num_local_tokens: torch.Tensor | int | float | None = None,
    workspace: torch.Tensor | None = None,
    *,
    kernel: MoESortingVariant | None = None,
    store_vec_width: int | None = None,
    zero_target_occupancy: int | None = None,
    stream=None,
):
    """Host entry for standalone FlyDSL MoE sorting.

    - output tensors are caller-allocated.
    - expert_mask enables dense local expert ids for the EP path.
    - num_local_tokens bounds valid top-k pairs when input M is padded.
    - returns sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf.
    """
    device = topk_ids.device
    M = int(topk_ids.shape[0])
    topk = int(topk_ids.shape[1])

    variant = kernel or select_moe_sorting_kernel(num_experts=num_experts, tokens=M, topk=topk)

    ws_elems = moe_sorting_get_workspace_size(M, num_experts, topk, unit_size, kernel=variant)
    if ws_elems:
        if workspace is None:
            workspace = torch.empty(ws_elems, dtype=torch.int32, device=device)
        workspace = workspace.view(torch.int32).reshape(-1)
        if workspace.numel() < ws_elems:
            raise ValueError(f"workspace too small: {variant} needs {ws_elems} int32 elems, got {workspace.numel()}")

    build_kw = dict(
        experts=num_experts,
        tokens=M,
        topk=topk,
        unit_size=unit_size,
        has_mask=expert_mask is not None,
        has_padding=num_local_tokens is not None,
        store_vec_width=store_vec_width,
    )

    if expert_mask is None:
        expert_mask = torch.ones(num_experts, dtype=torch.int32, device=device)
    if num_local_tokens is None:
        num_local_tokens = torch.tensor([M], dtype=torch.int32, device=device)
    elif not isinstance(num_local_tokens, torch.Tensor):
        num_local_tokens = torch.tensor([int(num_local_tokens)], dtype=torch.int32, device=device)
    if stream is None:
        stream = torch.cuda.current_stream()

    moe_buf_i32, moe_buf_elems, n_grid = moe_buf_zero_grid(
        moe_buf,
        unit_size=unit_size,
        zero_target_occupancy=zero_target_occupancy,
    )

    # Cache only on constexprs; moe_buf size and zero-grid count are runtime args.
    cf_key = (
        variant,
        num_experts,
        M,
        topk,
        unit_size,
        build_kw["has_mask"],
        build_kw["has_padding"],
        store_vec_width,
        device.index,
    )

    if variant == "mesh_flatfill":
        launcher = get_moe_sorting_mesh_flatfill_kernel(**build_kw)
        args = (
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
        )
        _launch_cached(_moe_sorting_cf_cache, cf_key, launcher, args, stream)
    elif variant == "count_sort":
        launcher = get_moe_sorting_count_sort_kernel(**build_kw)
        assert workspace is not None  # ws_elems > 0 for count_sort
        counts = workspace[:num_experts]
        args = (
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
        )
        _launch_cached(_moe_sorting_cf_cache, cf_key, launcher, args, stream)
    elif variant == "count_sort_paired":
        launcher = get_moe_sorting_count_sort_paired_kernel(**build_kw)
        assert workspace is not None  # ws_elems > 0 for count_sort_paired
        hist_elems, side_elems = count_sort_paired_scratch_elems(M, topk, num_experts)
        hist = workspace[:hist_elems]
        counts = workspace[hist_elems : hist_elems + side_elems]
        estart = workspace[hist_elems + side_elems : hist_elems + 2 * side_elems]
        args = (
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
        )
        _launch_cached(_moe_sorting_cf_cache, cf_key, launcher, args, stream)
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
    """Build all sorting launchers for one static shape.

    max_tokens sizes the mesh_flatfill constexpr token loop.
    """
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
