# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# Shared helpers for the MegaMoE fused group-GEMM1 closure (gemm1).

import types
from collections import namedtuple
from contextlib import contextmanager

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import T as epkT
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemPtr
from kernels.common.layout_utils import get as layout_get
from kernels.common.layout_utils import idx2crd
from kernels.common.mma.mfma_preshuffle_pipeline import make_preshuffle_b_layout, make_preshuffle_scale_layout

# compile-time config / layout / LDS + IR view builders
# Deep-pipeline K-loop schedule tables (per-phase op lists + n_phases) consumed by KLoop.
PipeSchedule = namedtuple("PipeSchedule", ["pp_mfma", "pp_a_reads", "pp_b_loads", "pp_has_scale", "n_phases"])


def build_pipe_schedule(*, tile_m, tile_k_bytes, pack_K, pack_N, n_per_wave, gate_up_interleave):
    """Build the frozen unified K-loop scheduling tables consumed by KLoop. Returns a PipeSchedule."""
    pipe_m_repeat = tile_m // 16
    pipe_k_unroll = tile_k_bytes // 128
    pipe_k_unroll_packed = pipe_k_unroll // pack_K
    pipe_num_acc_n = n_per_wave // 16

    # A ds_read groups: group by mi (same mi, all k values together)
    pipe_a_groups = []
    for mi in range(pipe_m_repeat):
        grp = []
        for k in range(pipe_k_unroll):
            grp.append((k, mi))
            if len(grp) == 2:
                pipe_a_groups.append(grp)
                grp = []
        if grp:
            pipe_a_groups.append(grp)

    # B VMEM loads: individual gate/up loads
    pipe_b_loads = []
    for ku in range(pipe_k_unroll):
        for ni in range(pipe_num_acc_n):
            pipe_b_loads.append(("gate", ku, ni))
            if not gate_up_interleave:
                pipe_b_loads.append(("up", ku, ni))

    # MFMA order: B-major (fix B from VMEM, cycle all A tiles from LDS before next B).
    pipe_num_acc_n_packed = pipe_num_acc_n // pack_N
    pipe_all_mfma = []
    for ku128 in range(pipe_k_unroll_packed):
        for ni_packed in range(pipe_num_acc_n_packed):
            for ikxdl in range(pack_K):
                for inxdl in range(pack_N):
                    k_idx = ku128 * pack_K + ikxdl
                    ni_idx = ni_packed * pack_N + inxdl
                    pipe_all_mfma.append((k_idx, ni_idx, ikxdl, inxdl, ku128))

    # Group MFMAs per scheduling phase (wider M -> more MFMAs per phase)
    pipe_mfma_per_phase = max(1, len(pipe_all_mfma) // 4)
    pipe_n_phases = len(pipe_all_mfma) // pipe_mfma_per_phase

    # Build unified phase descriptors
    a_groups_per_phase = (len(pipe_a_groups) + pipe_n_phases - 1) // pipe_n_phases
    pipe_phases = []
    mfma_i = 0
    a_i = 0
    for p in range(pipe_n_phases):
        a_reads = []
        for _ in range(a_groups_per_phase):
            if a_i < len(pipe_a_groups):
                a_reads.extend(pipe_a_groups[a_i])
                a_i += 1
        phase = {
            "mfma": pipe_all_mfma[mfma_i : mfma_i + pipe_mfma_per_phase],
            "a_reads": a_reads,
            "b_loads": [],
            "has_scale": (p == 0),
        }
        mfma_i += pipe_mfma_per_phase
        pipe_phases.append(phase)

    # Distribute B loads evenly across phases 1..n-1 (phase 0 has scales)
    bi = 0
    for p in range(1, pipe_n_phases):
        rem_b = len(pipe_b_loads) - bi
        rem_p = pipe_n_phases - p
        n_b = (rem_b + rem_p - 1) // rem_p if rem_p > 0 else 0
        for _ in range(n_b):
            if bi < len(pipe_b_loads):
                pipe_phases[p]["b_loads"].append(pipe_b_loads[bi])
                bi += 1

    # Extract flat lists for kernel access (avoids dict access in AST rewriter)
    return PipeSchedule(
        pp_mfma=[p["mfma"] for p in pipe_phases],
        pp_a_reads=[p["a_reads"] for p in pipe_phases],
        pp_b_loads=[p["b_loads"] for p in pipe_phases],
        pp_has_scale=[p["has_scale"] for p in pipe_phases],
        n_phases=pipe_n_phases,
    )


def make_layouts(
    arith,
    *,
    w_experts,
    inter_dim,
    k_in,
    pack_K,
    kpack_bytes,
    b_elem_bytes,
    size_expert_ids_in,
    sort_block_m,
    model_dim,
):
    """Preshuffle B / A-scale / B-scale logical data layouts.

    Returns (layout_b, layout_a_scale, layout_b_scale, sorted_m); sorted_m (the sorted A-scale
    row count) is also consumed by the caller's a-scale buffer sizing.
    """
    # B preshuffle layout [E*2*inter_dim, model_dim]; expert e gate rows at [e*2*inter_dim, +inter_dim).
    c_n_total = arith.constant(w_experts * (2 * inter_dim), index=True)
    b_layout = make_preshuffle_b_layout(
        arith,
        c_n=c_n_total,
        c_k=k_in // pack_K,
        kpack_bytes=kpack_bytes,
        elem_bytes=b_elem_bytes,
        # k_major=True,
    )
    layout_b = b_layout.layout_b

    # A-scale [sorted_size, K/32]: indexed by sorted_row position, not token_id.
    sorted_m = size_expert_ids_in * arith.constant(sort_block_m, index=True)
    layout_a_scale = make_preshuffle_scale_layout(arith, c_mn=sorted_m, c_k=arith.constant(model_dim, index=True))
    # B-scale: [E*2*inter_dim, K/32]
    layout_b_scale = make_preshuffle_scale_layout(arith, c_mn=c_n_total, c_k=arith.constant(model_dim, index=True))

    return layout_b, layout_a_scale, layout_b_scale, sorted_m


WaveTiling = namedtuple("WaveTiling", ["num_waves", "total_waves", "total_threads", "pack_M", "n_per_wave", "pack_N"])


def resolve_wave_tiling(*, tile_m, tile_n, slice_k=1):
    """Wave / tile decomposition (capped by tile_n // 32, each wave owns >= 32 of N).

    slice_k>1 gives slice_k wave groups each summing a distinct K-slice of the same (M,N) tile
    (total_waves = num_waves * slice_k); slice_k==1 is the identity.
    """
    num_waves = min(4, tile_n // 32)
    total_waves = num_waves * slice_k
    total_threads = total_waves * 64
    pack_M = 1 if tile_m < 32 else 2
    n_per_wave = tile_n // num_waves
    pack_N = min(2, n_per_wave // 16)
    return WaveTiling(num_waves, total_waves, total_threads, pack_M, n_per_wave, pack_N)


# LDS arena plan: ping/pong X + cshuffle-out + tid + raw_a_scale staging (+ optional compact-allgather histogram).
LdsPlan = namedtuple(
    "LdsPlan",
    [
        "pong_offset",
        "tid_offset_pong",
        "scale_offset_pong",
        "ping_offset",
        "sc_cp_vec",
        "sc_cp_iters",
        "cnt_lds_off",
        "split_lds_out",
        "scale_lds_n_i32",
        "data_bytes",
        "scale_bytes",
        "total_bytes",
        "input_elems",
        "slice_k",
    ],
)


def plan_lds(
    allocator_pong,
    allocator_ping,
    *,
    gpu_arch,
    tile_m,
    tile_n,
    num_waves,
    pack_M,
    model_dim,
    total_threads,
    waves_per_eu,
    lds_stride,
    a_elem_bytes,
    a_elem_vec_pack,
    use_async_copy,
    compact_ag,
    fz_total_experts,
    slice_k=1,
):
    """Lay out the per-block LDS and bump the pong/ping SmemAllocator arenas; returns an LdsPlan.

    Allocation order: X pong -> tid -> raw_a_scale -> (compact count) -> X ping -> waves_per_eu
    occupancy floor. slice_k>1 gives each K-slice group its own ping/pong X and unions the
    cshuffle-out reduction scratch with pong (no split-LDS); slice_k==1 is the identity.
    """
    cshuffle_elem_bytes = 4
    group_threads = total_threads // slice_k
    # fp4 A packs in LDS (vec_pack=2) under async-copy -> size ping/pong at the packed stride.
    eff_lds_stride_alloc = lds_stride // a_elem_vec_pack if (use_async_copy and a_elem_vec_pack > 1) else lds_stride
    single_x_bytes = int(tile_m) * int(eff_lds_stride_alloc) * int(a_elem_bytes)
    x_pong_bytes = slice_k * single_x_bytes  # per-group X buffers (slice_k==1 -> single_x)
    x_ping_bytes = slice_k * single_x_bytes
    lds_out_bytes = cshuffle_elem_bytes * int(tile_m) * int(tile_n)
    lds_tid_bytes = int(tile_m) * 4
    input_elems = single_x_bytes if a_elem_bytes == 1 else (single_x_bytes // 2)

    # pong=max(input,lds_out)+tid, ping=input; overflow splits lds_out across pong/ping (slice_k>1 forces no-split).
    GLOBAL_ALIGN = 1024
    std_pong = max(x_pong_bytes, lds_out_bytes) + lds_tid_bytes
    std_ping = x_ping_bytes
    std_pong_aligned = allocator_pong._align(std_pong, 128)
    std_total = allocator_pong._align(std_pong_aligned, GLOBAL_ALIGN) + allocator_pong._align(std_ping, 128)
    lds_limit = {"gfx950": 163840, "gfx942": 65536}.get(gpu_arch, 0)
    split_lds_out = lds_limit > 0 and lds_out_bytes > 0 and std_total > lds_limit and num_waves >= 2 and slice_k == 1

    if split_lds_out:
        half_out_bytes = cshuffle_elem_bytes * int(tile_m) * (int(tile_n) // 2)
        pong_buffer_bytes = max(single_x_bytes, half_out_bytes)
        ping_buffer_bytes = max(single_x_bytes, half_out_bytes)
    else:
        pong_buffer_bytes = max(x_pong_bytes, lds_out_bytes)
        ping_buffer_bytes = x_ping_bytes

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + pong_buffer_bytes
    lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = lds_tid_offset_pong + lds_tid_bytes

    # raw_a_scale: stage this tile's full A-scale once in the prologue (K-loop reads it coalesced); taken from pong before the waves_per_eu pad.
    raw_sni_lds = model_dim // 128  # i32 scale cols per scale_em row
    scale_lds_rows = ((tile_m // 16) // pack_M) * 32  # m_repeat_packed * 32 rows
    scale_lds_n_i32 = scale_lds_rows * raw_sni_lds
    scale_lds_offset_pong = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = scale_lds_offset_pong + scale_lds_n_i32 * 4
    # coalesced cooperative copy: largest vec in {4,2,1} that evenly tiles over group_threads = total_threads // slice_k.
    sc_cp_vec = 4
    while sc_cp_vec > 1 and (scale_lds_n_i32 % (group_threads * sc_cp_vec)) != 0:
        sc_cp_vec //= 2
    assert scale_lds_n_i32 % (group_threads * sc_cp_vec) == 0, (
        f"raw_a_scale LDS staging: {scale_lds_n_i32} i32 not tileable by "
        f"{group_threads} group threads x {sc_cp_vec}"
    )
    sc_cp_iters = scale_lds_n_i32 // (group_threads * sc_cp_vec)

    # compact-allgather PHASE-0 count histogram (2*te i32), reused from pong (disjoint from GEMM).
    if compact_ag:
        cnt_lds_off = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = cnt_lds_off + 2 * fz_total_experts * 4
    else:
        cnt_lds_off = 0

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + ping_buffer_bytes

    # Real per-block DATA LDS (incl. scale staging) BEFORE the waves_per_eu occupancy floor.
    lds_data_bytes = allocator_pong._align(allocator_pong.ptr, 128) + allocator_ping._align(allocator_ping.ptr, 128)
    lds_scale_bytes = scale_lds_n_i32 * 4

    if waves_per_eu is not None and waves_per_eu >= 1:
        total_cu_lds = 160 * 1024
        min_lds = total_cu_lds // (waves_per_eu + 1) + 1
        pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        cur_lds = pong_sz + ping_sz
        if cur_lds < min_lds:
            allocator_ping.ptr += min_lds - cur_lds

    # Real per-block LDS AFTER the floor -- this bounds LDS occupancy.
    lds_total_bytes = allocator_pong._align(allocator_pong.ptr, 128) + allocator_ping._align(allocator_ping.ptr, 128)

    # slice_k>1 must still fit per-CU LDS (else occupancy drops below 1 block/CU and the dispatch grid barrier deadlocks).
    if slice_k > 1 and lds_limit > 0:
        assert lds_total_bytes <= lds_limit, (
            f"slice_k={slice_k} LDS {lds_total_bytes} exceeds {lds_limit} on {gpu_arch}; "
            f"use a smaller tile_m/tile_n or a smaller slice_k."
        )

    return LdsPlan(
        pong_offset=lds_pong_offset,
        tid_offset_pong=lds_tid_offset_pong,
        scale_offset_pong=scale_lds_offset_pong,
        ping_offset=lds_ping_offset,
        sc_cp_vec=sc_cp_vec,
        sc_cp_iters=sc_cp_iters,
        cnt_lds_off=cnt_lds_off,
        split_lds_out=split_lds_out,
        scale_lds_n_i32=scale_lds_n_i32,
        data_bytes=lds_data_bytes,
        scale_bytes=lds_scale_bytes,
        total_bytes=lds_total_bytes,
        input_elems=input_elems,
        slice_k=slice_k,
    )


LdsViews = namedtuple("LdsViews", ["x_pong", "x_ping", "out", "out_B", "tid", "a_scale"])


def build_lds_views(allocator_pong, allocator_ping, *, lds, tile_m, tile_n):
    """Materialise per-block LDS memref views (X ping/pong, cshuffle-out [+split B], tid, a-scale)
    from the two arenas + LdsPlan offsets. Called once in the prologue before the persistent
    scf.for (no loop-carried dominance concern); returns LdsViews.
    """
    base_pong = allocator_pong.get_base()
    base_ping = allocator_ping.get_base()
    # slice_k>1: X views span all slice_k per-group buffers (K-loop adds wid_k*input_elems); slice_k==1 keeps shape=(input_elems,).
    x_elems = lds.slice_k * lds.input_elems
    x_pong = SmemPtr(base_pong, lds.pong_offset, T.f8, shape=(x_elems,)).get()
    x_ping = SmemPtr(base_ping, lds.ping_offset, T.f8, shape=(x_elems,)).get()
    out_t = T.f32
    if const_expr(lds.split_lds_out):
        half = int(tile_m) * (int(tile_n) // 2)
        out = SmemPtr(base_pong, lds.pong_offset, out_t, shape=(half,)).get()
        out_B = SmemPtr(base_ping, lds.ping_offset, out_t, shape=(half,)).get()
    else:
        out = SmemPtr(base_pong, lds.pong_offset, out_t, shape=(tile_m * tile_n,)).get()
        out_B = None
    tid = SmemPtr(base_pong, lds.tid_offset_pong, T.i32, shape=(tile_m,)).get()
    a_scale = SmemPtr(base_pong, lds.scale_offset_pong, T.i32, shape=(lds.scale_lds_n_i32,)).get()
    return LdsViews(x_pong, x_ping, out, out_B, tid, a_scale)


# ---- Kernel-side raw-MLIR wrappers (keep the kernel body free of bare ir./scf./llvm. calls) ----


def tensor_base_addr_index(t):
    """Base address of a tensor as an index value (raw device-pointer extraction)."""
    ptr = fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr"), t)
    return arith.index_cast(T.index, llvm.ptrtoint(T.i64, ptr))


def finalize_lds(allocator_pong, allocator_ping):
    """Emit the LDS globals for both SmemAllocator arenas into the gpu module (launcher-side)."""
    allocator_pong.finalized = False
    allocator_ping.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        allocator_pong.finalize()
        allocator_ping.finalize()


@contextmanager
def persistent_tile_loop(lo, hi, step):
    """scf.for over the persistent tile range: yields the induction variable, closes the body with YieldOp.
    No loop-carried values (round-robin state lives on the TileScheduler)."""
    for_op = scf.ForOp(lo, hi, step)
    ip = ir.InsertionPoint(for_op.body)
    ip.__enter__()
    try:
        yield for_op.induction_variable
        scf.YieldOp([])
    finally:
        ip.__exit__(None, None, None)


# loop-invariant address atoms
class AGatherAddresser:
    """expert-major contiguous A row-base indices (K-invariant).

    Production stage1 is always contiguous_io: A row r IS the physical row (bx_m + row_local),
    no sorted_token_ids gather. row_bases() returns the three lists the K-loop reads.
    """

    def __init__(
        self,
        *,
        num_x_loads,
        x_tile_chunk_coord_i32,  # callable(i) -> (row_local, col_local_i32)
        c_k_div4,
    ):
        self.num_x_loads = num_x_loads
        self._x_tile_chunk_coord_i32 = x_tile_chunk_coord_i32
        self.c_k_div4 = c_k_div4

    def row_bases(self, bx_m):
        """Per-tile: -> (x_row_base_div4, x_col_local_i32, x_row_local).  K-invariant."""
        x_row_base_div4 = []
        x_col_local_i32 = []
        x_row_local = []
        for i in range_constexpr(self.num_x_loads):
            row_local, col_local_i32 = self._x_tile_chunk_coord_i32(i)
            x_row_local.append(row_local)
            x_col_local_i32.append(col_local_i32)

            # expert-major contiguous: A row r IS physical row (bx_m+row_local), no sorted_token_ids gather.
            sorted_row_i = bx_m + row_local
            x_row_base_div4.append(sorted_row_i * self.c_k_div4)
        return x_row_base_div4, x_col_local_i32, x_row_local


class ExpertOperandBases:
    """Per-expert, per-tile N-block / scale base indices for gate & up weights (K-invariant).

    gate_up_interleave selects whether the up lists are populated (SEPARATED) or folded into
    interleaved slots (INTERLEAVE).
    """

    def __init__(
        self,
        *,
        gate_up_interleave,
        # N-tile block lists
        num_acc_n,
        n_tile_base,
        lane_mod_16,
        w_experts,
        inter_dim,
        pack_N,
        # scale bases
        lane_div_16,
        layout_b_scale,
        layout_a_scale,
        sort_block_m,
        scale_mn_pack,
    ):
        self.gate_up_interleave = gate_up_interleave
        self.num_acc_n = num_acc_n
        self.n_tile_base = n_tile_base
        self.lane_mod_16 = lane_mod_16
        self.w_experts = w_experts
        self.inter_dim = inter_dim
        self.pack_N = pack_N
        self.lane_div_16 = lane_div_16
        self.layout_b_scale = layout_b_scale
        self.layout_a_scale = layout_a_scale
        self.sort_block_m = sort_block_m
        self.scale_mn_pack = scale_mn_pack

    def gate_up_n_blocks(self, expert_off_idx, by_n):
        """Per-tile N-tile block/intra lists for B weight gather. Returns (gate_n_blk_list,
        gate_n_intra_list, up_n_blk_list, up_n_intra_list, col_g_list, inter_idx); inter_idx is
        also consumed by downstream inline code (bias / interleave).
        """
        gate_n_intra_list = []
        gate_n_blk_list = []
        up_n_intra_list = []
        up_n_blk_list = []
        col_g_list = []
        c_n0_static = self.w_experts * (2 * self.inter_dim) // 16
        layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
        inter_idx = arith.constant(self.inter_dim, index=True)

        for i in range_constexpr(self.num_acc_n):
            offset = i * 16
            c_offset = arith.constant(offset, index=True)
            if const_expr(not self.gate_up_interleave):
                col_g = by_n + self.n_tile_base + c_offset + self.lane_mod_16
                col_g_list.append(col_g)

            global_n = by_n + self.n_tile_base + c_offset + self.lane_mod_16
            # Gate/interleave: rows [expert_off, expert_off + 2*inter_dim)
            gate_row_w = expert_off_idx + global_n
            gate_coord = idx2crd(gate_row_w, layout_n_blk_intra)
            gate_n_blk_list.append(layout_get(gate_coord, 0))
            gate_n_intra_list.append(layout_get(gate_coord, 1))
            if const_expr(not self.gate_up_interleave):
                up_row_w = gate_row_w + inter_idx
                up_coord = idx2crd(up_row_w, layout_n_blk_intra)
                up_n_blk_list.append(layout_get(up_coord, 0))
                up_n_intra_list.append(layout_get(up_coord, 1))

        if const_expr(self.gate_up_interleave):
            gui_num_acc_n_out = self.num_acc_n // self.pack_N
            for gui_i in range_constexpr(gui_num_acc_n_out):
                gui_offset = gui_i * 16
                gui_c_offset = arith.constant(gui_offset, index=True)
                gui_col_g = (by_n + self.n_tile_base) // arith.constant(2, index=True) + gui_c_offset + self.lane_mod_16
                col_g_list.append(gui_col_g)

        return gate_n_blk_list, gate_n_intra_list, up_n_blk_list, up_n_intra_list, col_g_list, inter_idx

    def scale_lane_elem(self):
        """Loop-invariant per-lane scale element offset (shared by A & B scale bases)."""
        return self.lane_div_16 * self.layout_b_scale.stride_klane + self.lane_mod_16

    def gate_up_scale_bases(self, expert_off_idx, by_n, inter_idx, scale_lane_elem, num_acc_n_packed):
        """Per-tile B micro-scale base element indices.  Returns (gate, up) lists."""
        gate_scale_bases = []
        up_scale_bases = []
        for ni in range_constexpr(num_acc_n_packed):
            col_base = by_n + self.n_tile_base + arith.constant(ni * 16 * self.pack_N, index=True)
            gate_mni = (expert_off_idx + col_base) // arith.constant(32, index=True)
            gate_scale_bases.append(gate_mni * self.layout_b_scale.stride_n0 + scale_lane_elem)
            if const_expr(not self.gate_up_interleave):
                up_mni = (expert_off_idx + inter_idx + col_base) // arith.constant(32, index=True)
                up_scale_bases.append(up_mni * self.layout_b_scale.stride_n0 + scale_lane_elem)
        return gate_scale_bases, up_scale_bases


def make_rearrange_scales(*, pack_M, pack_N, scale_mn_pack, bx_m, n_tile_base):
    """Build the pure `_rearrange_a_scale` / `_rearrange_b_scale` byte-twiddle helpers.

    Emitted as ONE builder (mirroring the original single inline block) so the shared
    `_c16_idx`/`_c2_idx`/`_scale_mask_lo` constants and the m/n-half shift constants are
    emitted exactly once, in the original op-emission order.  Returns the two callables;
    each is identity (no ops) when its pack axis >= scale_mn_pack.  Loop-invariant.
    """
    c16_idx = arith.constant(16, index=True)
    c2_idx = arith.constant(2, index=True)
    scale_mask_lo = arith.constant(0xFF, type=T.i32)

    m_half_idx = arith.constant(0, type=T.i32)
    m_half_i32 = arith.constant(0, type=T.i32)
    scale_shift = arith.constant(0, type=T.i32)
    scale_shift_hi = arith.constant(0, type=T.i32)
    n_half_idx = arith.constant(0, type=T.i32)
    n_half_i32 = arith.constant(0, type=T.i32)
    bscale_shift = arith.constant(0, type=T.i32)
    bscale_shift_hi = arith.constant(0, type=T.i32)
    if const_expr(pack_M < scale_mn_pack):
        m_half_idx = (bx_m // c16_idx) % c2_idx
        m_half_i32 = arith.index_cast(T.i32, m_half_idx)
        scale_shift = m_half_i32 * arith.constant(8, type=T.i32)
        scale_shift_hi = scale_shift + arith.constant(16, type=T.i32)

    if const_expr(pack_N < scale_mn_pack):
        n_half_idx = (n_tile_base // c16_idx) % c2_idx
        n_half_i32 = arith.index_cast(T.i32, n_half_idx)
        bscale_shift = n_half_i32 * arith.constant(8, type=T.i32)
        bscale_shift_hi = bscale_shift + arith.constant(16, type=T.i32)

    def rearrange_a_scale(raw_i32):
        """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
        if const_expr(pack_M >= scale_mn_pack):
            return raw_i32
        b_k0 = arith.andi(arith.shrui(raw_i32, scale_shift), scale_mask_lo)
        b_k1 = arith.andi(arith.shrui(raw_i32, scale_shift_hi), scale_mask_lo)
        return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

    def rearrange_b_scale(raw_i32):
        """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
        if const_expr(pack_N >= scale_mn_pack):
            return raw_i32
        b_k0 = arith.andi(arith.shrui(raw_i32, bscale_shift), scale_mask_lo)
        b_k1 = arith.andi(arith.shrui(raw_i32, bscale_shift_hi), scale_mask_lo)
        return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

    return rearrange_a_scale, rearrange_b_scale


# persistent tile scheduler
class TileCoord:
    """Per-tile decode result read by the GEMM body, epilogue, and srcmap-emit.

    expert_idx is the LOCAL expert id (indexes w1/scale); expert_global is the GLOBAL id
    (validity + overlap-gate); cnt_ef/ef/kf are P-static readlane registers, None for sparse/dense.
    """

    __slots__ = ("bx_m", "expert_idx", "expert_global", "blk_valid", "exp_valid", "bx", "cnt_ef", "ef", "kf")

    def __init__(self, *, bx_m, expert_idx, expert_global, blk_valid, exp_valid, bx, cnt_ef, ef, kf):
        self.bx_m = bx_m
        self.expert_idx = expert_idx
        self.expert_global = expert_global
        self.blk_valid = blk_valid
        self.exp_valid = exp_valid
        self.bx = bx
        self.cnt_ef = cnt_ef
        self.ef = ef
        self.kf = kf


class PersistState:
    """Loop-invariant register state produced by `setup_persistent()` (P-static only)."""

    __slots__ = ("st_ll", "st_nt", "st_pre")

    def __init__(self, *, st_ll, st_nt, st_pre):
        self.st_ll = st_ll
        self.st_nt = st_nt
        self.st_pre = st_pre


class TileScheduler:
    """grid -> (expert, m-tile, sorted-row-base), persistent round-robin + XCD swizzle."""

    def __init__(
        self,
        *,
        mode,  # "static" | "sparse"
        # XCD swizzle
        xcd_swizzle,
        gate_up_interleave,
        n_in,
        tile_n,
        # persistent loop
        sort_block_m,
        skip_gemm,
        # decode: global->local expert + tile geometry
        rank,
        experts_per_rank,
        experts,
        fz_epr,
        fz_cap,
        fz_tile_m,
        fz_rank,
        # backing address (per-expert expected-real count, for the P-static prefix scan)
        addr_expected_real,
        # per-warp lane thread id for the P-static prefix scan
        tx,
        # readlane namespace (passed in to avoid a circular import)
        epk,
    ):
        self.mode = mode
        self.xcd_swizzle = xcd_swizzle
        self.gate_up_interleave = gate_up_interleave
        self.n_in = n_in
        self.tile_n = tile_n
        self.sort_block_m = sort_block_m
        self.skip_gemm = skip_gemm
        self.rank = rank
        self.experts_per_rank = experts_per_rank
        self.experts = experts
        self.fz_epr = fz_epr
        self.fz_cap = fz_cap
        self.fz_tile_m = fz_tile_m
        self.fz_rank = fz_rank
        self.addr_expected_real = addr_expected_real
        self.tx = tx
        self._epk = epk
        # P-static decode predicate; when fz_epr > 64 the static build falls through to SPARSE.
        self._p_static = mode == "static" and fz_epr <= 64
        # mode is always "static" or "sparse" (dense fixed-slot layout is unused).
        # scratch stashed across the method boundaries.
        self._c0_p = None
        self._c1_p = None
        self._c_cu_p = None
        self._total_m_tiles = None
        self._num_valid_i32 = None

    def swizzled_wg(self, bx_persist, by):
        """XCD-swizzle WG remap.  Uses the ACTUAL launched grid_dim.y."""
        if const_expr(self.xcd_swizzle > 0):
            NUM_XCDS_S1 = 8
            c1_sw = arith.constant(1, index=True)
            c_tn_sw = arith.constant(self.tile_n, index=True)
            if const_expr(self.gate_up_interleave):
                gx = (self.n_in + c_tn_sw - c1_sw) / c_tn_sw
            else:
                c2_sw = arith.constant(2, index=True)
                gx = (self.n_in + c2_sw * c_tn_sw - c1_sw) / c_tn_sw / c2_sw
            # gy MUST be the ACTUAL launched grid_dim.y, NOT a hardcoded cu_num (else _num_wgs inflates and the XCD remap skips most tiles).
            gy = arith.index_cast(T.index, gpu.grid_dim.y)
            linear_id = bx_persist * gx + by
            num_wgs = gx * gy

            c_xcds = arith.constant(NUM_XCDS_S1, index=True)
            wgs_per_xcd = num_wgs / c_xcds
            wgid = (linear_id % c_xcds) * wgs_per_xcd + (linear_id / c_xcds)

            WGM_S1 = self.xcd_swizzle
            c_wgm = arith.constant(WGM_S1, index=True)
            num_wgid_in_group = c_wgm * gx
            group_id = wgid / num_wgid_in_group
            first_pid_m = group_id * c_wgm
            remaining_m = gy - first_pid_m
            cmp_m = arith.cmpi(CmpIPredicate.ult, remaining_m, c_wgm)
            group_size_m = arith.select(cmp_m, remaining_m, c_wgm)

            wgid_in_group = wgid % num_wgid_in_group
            bx_persist = first_pid_m + (wgid_in_group % group_size_m)
            by = wgid_in_group / group_size_m

        return bx_persist, by

    def tiles_per_block(self, num_valid_i32):
        """Persistent round-robin bound: ceil(ceil(num_valid/sort_block_m)/grid_y).
        Emits _c0_p/_c1_p/_c_cu_p and stashes _total_m_tiles for decode.
        """
        c0_p = arith.constant(0, index=True)
        c1_p = arith.constant(1, index=True)
        # persistent round-robin: grid_y CTAs each take a strided chunk of M-tiles, bounded by device num_valid.
        c_cu_p = arith.index_cast(T.index, gpu.grid_dim.y)
        c_sbm_p = arith.constant(self.sort_block_m, index=True)
        num_valid_idx = arith.index_cast(T.index, num_valid_i32)
        total_m_tiles = (num_valid_idx + c_sbm_p - c1_p) / c_sbm_p
        tiles_per_block = (total_m_tiles + c_cu_p - c1_p) / c_cu_p
        if const_expr(self.skip_gemm):
            tiles_per_block = arith.constant(0, index=True)  # DIAG: skip consumer GEMM (time prologue+producer)
        self._c0_p = c0_p
        self._c1_p = c1_p
        self._c_cu_p = c_cu_p
        self._total_m_tiles = total_m_tiles
        self._num_valid_i32 = num_valid_i32
        return c0_p, tiles_per_block, c1_p

    def setup_persistent(self):
        """P-static: per-warp ll_count prefix scan in registers (loop-invariant).
        Returns the register state consumed by decode, or None for sparse/dense.
        """
        if const_expr(self._p_static):
            # per-warp ll_count prefix in registers (lane le holds ll_count[le], ntiles[le], exclusive prefix); each tile reads (expert,k) via readlane.
            st_llr = buffer_ops.create_buffer_resource_from_addr(self.addr_expected_real)
            st_lane = arith.andi(arith.index_cast(T.i32, self.tx), arith.constant(63, type=T.i32))
            st_ok = arith.cmpi(CmpIPredicate.slt, st_lane, arith.constant(self.fz_epr, type=T.i32))
            st_ll = arith.select(
                st_ok,
                buffer_ops.buffer_load(st_llr, arith.index_cast(T.index, st_lane), vec_width=1, dtype=T.i32),
                arith.constant(0),
            )
            st_ctm = arith.constant(self.fz_tile_m, type=T.i32)
            st_ctm1 = arith.constant(self.fz_tile_m - 1, type=T.i32)
            st_nt = (st_ll + st_ctm1) // st_ctm
            st_pre = arith.constant(0, type=T.i32)
            for stj in range_constexpr(self.fz_epr):
                st_sj = self._epk._readlane(st_nt, stj)
                st_pre = arith.select(
                    arith.cmpi(CmpIPredicate.sgt, st_lane, arith.constant(stj, type=T.i32)), st_pre + st_sj, st_pre
                )
            return PersistState(st_ll=st_ll, st_nt=st_nt, st_pre=st_pre)
        return None

    def decode(self, bx_persist, mi_p, persist_state, *, sorted_rsrc, expert_rsrc):
        """Per-tile 3-way decode -> TileCoord (static / sparse / dense const_expr branch).

        sorted_rsrc/expert_rsrc are passed here (not at construction) because they are created
        after the XCD-swizzle emission point.
        """
        # Strided round-robin: CTA k does tiles {k, k+grid_y, ...} (adjacent CTAs -> per-wave B L2 reuse).
        bx = bx_persist + mi_p * self._c_cu_p
        if const_expr(self._p_static):
            return self._decode_static(bx, persist_state)
        return self._decode_sparse_dense(bx, sorted_rsrc, expert_rsrc)

    def _decode_static(self, bx, persist_state):
        st_ll = persist_state.st_ll
        st_nt = persist_state.st_nt
        st_pre = persist_state.st_pre
        # P-static: derive (expert, k) for compact tile bx via readlane prefix-scan (no se/trb).
        st_bx = arith.index_cast(T.i32, bx)
        ef = arith.constant(0, type=T.i32)
        kf = arith.constant(0, type=T.i32)
        cnt_ef = arith.constant(0, type=T.i32)
        for ste in range_constexpr(self.fz_epr):
            st_pe = self._epk._readlane(st_pre, ste)
            st_nte = self._epk._readlane(st_nt, ste)
            st_in = arith.andi(
                arith.cmpi(CmpIPredicate.sge, st_bx, st_pe), arith.cmpi(CmpIPredicate.slt, st_bx, st_pe + st_nte)
            )
            ef = arith.select(st_in, arith.constant(ste, type=T.i32), ef)
            kf = arith.select(st_in, st_bx - st_pe, kf)
            # capture ll_count[_ef] so the epilogue can mask rows with position-in-expert >= count (no host srcmap sentinel).
            cnt_ef = arith.select(st_in, self._epk._readlane(st_ll, ste), cnt_ef)
        bx_m_i32 = ef * arith.constant(self.fz_cap, type=T.i32) + kf * arith.constant(self.fz_tile_m, type=T.i32)
        bx_m = arith.index_cast(T.index, bx_m_i32)
        blk_valid = arith.cmpi(CmpIPredicate.ult, st_bx, arith.index_cast(T.i32, self._total_m_tiles))
        expert_i32 = ef + arith.constant(self.fz_rank * self.fz_epr, type=T.i32)
        # w1/scale indexed by the LOCAL expert id (_ef in [0,epr)).
        e_w_i32 = ef
        expert_idx = arith.index_cast(T.index, e_w_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(self.experts, type=T.i32))
        return TileCoord(
            bx_m=bx_m,
            expert_idx=expert_idx,
            expert_global=expert_i32,
            blk_valid=blk_valid,
            exp_valid=exp_valid,
            bx=bx,
            cnt_ef=cnt_ef,
            ef=ef,
            kf=kf,
        )

    def _decode_sparse_dense(self, bx, sorted_rsrc, expert_rsrc):
        # sparse fixed-slot: per-tile row base = tile_row_base[bx] (carried in arg_sorted_token_ids), rows contiguous.
        trb = buffer_ops.buffer_load(sorted_rsrc, bx, vec_width=1, dtype=T.i32)
        bx_m = arith.index_cast(T.index, trb)
        # Block validity: sparse compares tile index vs tile count (bx_m is a sparse base).
        blk_valid = arith.cmpi(
            CmpIPredicate.ult, arith.index_cast(T.i32, bx), arith.index_cast(T.i32, self._total_m_tiles)
        )
        expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
        # sorted_expert_ids carries GLOBAL ids -> index w1/scale by LOCAL id (global - rank*epr).
        e_w_i32 = arith.subi(expert_i32, arith.constant(self.fz_rank * self.fz_epr, type=T.i32))
        expert_idx = arith.index_cast(T.index, e_w_i32)
        exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(self.experts, type=T.i32))
        return TileCoord(
            bx_m=bx_m,
            expert_idx=expert_idx,
            expert_global=expert_i32,
            blk_valid=blk_valid,
            exp_valid=exp_valid,
            bx=bx,
            cnt_ef=None,
            ef=None,
            kf=None,
        )


# low-level device primitives (raw llvm / inline asm)
def epk_to_i64(v):
    return arith.extui(epkT.i64(), arith.unwrap(v))


def epk_to_ptr_global(v):
    return llvm.IntToPtrOp(llvm.PointerType.get(address_space=1), arith.unwrap(v)).result


def epk_store_i32_system(addr_i64, offset, val):
    base = arith.unwrap(addr_i64)
    off = arith.unwrap(offset)
    val_ = arith.unwrap(val)
    i64 = ir.IntegerType.get_signless(64)
    i32 = ir.IntegerType.get_signless(32)
    nuw = ir.Attribute.parse("#llvm.overflow<none>")
    off64 = llvm.ZExtOp(i64, off).res if off.type == i32 else off
    byte_off = llvm.MulOp(off64, llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 4)).result, nuw).result
    addr = llvm.AddOp(base, byte_off, nuw).result
    gptr = llvm.IntToPtrOp(llvm.PointerType.get(address_space=1), addr).result
    llvm.StoreOp(val_, gptr, alignment=4, ordering=llvm.AtomicOrdering.release, syncscope="one-as")


def epk_fence_system_acquire():
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="one-as")


def epk_fence_system_release():
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="one-as")


# Agent (device) scope: L2 flush for same-GPU cross-block visibility (cheaper than the system/xGMI flush).
def epk_fence_agent_acquire():
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="agent-one-as")


def epk_fence_agent_release():
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="agent-one-as")


def epk_atomic_add_agent(addr_i64, val):
    ptr = epk_to_ptr_global(addr_i64)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, ptr, arith.unwrap(val), llvm.AtomicOrdering.monotonic, syncscope="agent"
    ).res


def epk_atomic_add_system(addr_i64, val):
    ptr = epk_to_ptr_global(addr_i64)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, ptr, arith.unwrap(val), llvm.AtomicOrdering.monotonic, syncscope="one-as"
    ).res


def epk_readlane0(val_i32):
    return rocdl.readlane(epkT.i32(), arith.unwrap(val_i32), arith.unwrap(arith.constant(0)))


def epk_readlane(val_i32, lane):
    # broadcast val_i32 from a UNIFORM (compile-const) lane to all lanes (v_readlane needs SGPR lane).
    return rocdl.readlane(epkT.i32(), arith.unwrap(val_i32), arith.unwrap(arith.constant(lane)))


epk = types.SimpleNamespace(
    _to_i64=epk_to_i64,
    store_i32_system=epk_store_i32_system,
    fence_system_acquire=epk_fence_system_acquire,
    fence_system_release=epk_fence_system_release,
    fence_agent_acquire=epk_fence_agent_acquire,
    fence_agent_release=epk_fence_agent_release,
    atomic_add_agent=epk_atomic_add_agent,
    atomic_add_system=epk_atomic_add_system,
    _readlane=epk_readlane,
    _readlane0=epk_readlane0,
)


def barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm (bypasses LLVM SIInsertWaitcnts' conservative vmcnt(0) lgkmcnt(0) before every S_BARRIER)."""
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )
