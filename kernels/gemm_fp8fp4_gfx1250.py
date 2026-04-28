"""Unified MXFP4/MXFP8/A8W4 GEMM kernel for gfx1250.

Supports FP4 (E2M1), FP8 (E4M3) and A8W4 (FP8 activation + FP4 weight)
data with E8M0 block scales via V_WMMA_SCALE instructions.
Select precision with ``data_format="fp4"|"fp8"|"a8w4"``.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.named_barrier_allocator import NamedBarrierAllocator
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx, get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
    pipeline_fence_signal_named, pipeline_fence_wait_named,
    pipeline_fence_signal_named_multi, pipeline_fence_wait_named_multi,
    init_named_barriers,
    store_acc_vec8_to_buffer, store_acc_vec8_to_lds,
)
from kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

# Common constants
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
SCALE_BLOCK = 32
SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK  # 4

LDS_PAD_A_BYTES = 16
LDS_PAD_D_BYTES = 16


def compile_mxscale_gemm(
    *,
    data_format: str = "fp4",
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    use_tdm_store: bool = True,
    out_dtype: str = "f32",
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    split_k: int = 1,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
):
    """Compile an MXFP4/MXFP8/A8W4 GEMM kernel with TDM async copy.

    Args:
        data_format: "fp4" for FP4/E2M1, "fp8" for FP8/E4M3,
            or "a8w4" for FP8 activations with FP4 weights.
        M, N, K: compile-time problem dimensions. M and N must be the
            tile-aligned padded dimensions passed to the launch function; this
            kernel does not predicate tail tiles.

    Data layout (both formats):
        A: [M, K_packed] uint8 (FP4: K_packed=K//2, FP8: K_packed=K)
        B: [N, K_packed] uint8, preshuffled (16x16 byte tiles)
        scale_A: [M, K//32] uint8 E8M0 (preshuffled)
        scale_B: [N, K//32] uint8 E8M0 (preshuffled)

    Returns a JitFunction:
        launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, stream)
    """
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}")

    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"

    if out_dtype not in ("f32", "bf16", "f16"):
        raise ValueError(f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}")
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4

    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"M, N, and K must be positive, got M={M}, N={N}, K={K}")
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
        raise ValueError(
            f"tile_m, tile_n, and tile_k must be positive, "
            f"got {tile_m}, {tile_n}, {tile_k}")
    if m_warp <= 0 or n_warp <= 0:
        raise ValueError(f"m_warp and n_warp must be positive, got {m_warp}, {n_warp}")
    if cluster_m <= 0 or cluster_n <= 0:
        raise ValueError(
            f"cluster_m and cluster_n must be positive, got {cluster_m}, {cluster_n}")
    if waves_per_eu is not None and waves_per_eu < 1:
        raise ValueError(f"waves_per_eu must be >= 1 when set, got {waves_per_eu}")
    if M % tile_m != 0:
        raise ValueError(
            f"M must be divisible by tile_m={tile_m}; pad M before compile, got M={M}")
    if N % tile_n != 0:
        raise ValueError(
            f"N must be divisible by tile_n={tile_n}; pad N before compile, got N={N}")

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 2

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    if block_threads > 1024:
        raise ValueError(
            f"block_threads must be <= 1024, got {block_threads}")

    if wave_specialized_tdm and num_warps != 4:
        raise ValueError(
            f"wave_specialized_tdm requires exactly 4 waves, got {num_warps}")

    # ── Format-dependent compile-time constants ──
    # A8W4: activation is FP8 (PACK_FACTOR_A=1), weight is FP4 (PACK_FACTOR_B=2)
    if is_a8w4:
        PACK_FACTOR_A = 1   # FP8 activation
        PACK_FACTOR_B = 2   # FP4 weight
    elif is_fp4:
        PACK_FACTOR_A = 2
        PACK_FACTOR_B = 2
    else:
        PACK_FACTOR_A = 1
        PACK_FACTOR_B = 1

    WMMA_N_EFF = 32 if is_fp4 else 16   # N-cols covered per WMMA instruction
    ACC_VEC_SIZE = 16 if is_fp4 else 8   # accumulator vector width
    DS_LOADS_PER_A_FRAG = 2 if is_fp4 else 4

    packed_tile_k_a = tile_k // PACK_FACTOR_A
    packed_tile_k_b = tile_k // PACK_FACTOR_B
    scale_k_per_tile = tile_k // SCALE_BLOCK
    K_packed_a = K // PACK_FACTOR_A
    K_packed_b = K // PACK_FACTOR_B
    K_scale = K // SCALE_BLOCK
    split_k_chunk = K // split_k

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if K % split_k != 0:
        raise ValueError(f"K must be divisible by split_k={split_k}, got K={K}")
    if split_k_chunk % tile_k != 0:
        raise ValueError(
            f"K/split_k must be divisible by tile_k={tile_k}, got {split_k_chunk}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if packed_tile_k_a % 4 != 0:
        raise ValueError(f"packed_tile_k_a must be a multiple of 4, got {packed_tile_k_a}")
    if packed_tile_k_b % 4 != 0:
        raise ValueError(f"packed_tile_k_b must be a multiple of 4, got {packed_tile_k_b}")
    if scale_k_per_tile % 4 != 0:
        raise ValueError(
            f"scale_k_per_tile must be a multiple of 4 (tile_k >= 128), got {scale_k_per_tile}")

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N_EFF != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N_EFF}")

    if split_k > 1 and use_tdm_store:
        raise ValueError("split_k > 1 currently requires use_tdm_store=False")

    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles}")

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    n_accs = wmma_m_rep * wmma_n_rep
    # FP4 A/B swap: BScale rep derived from WMMA_M, not WMMA_N_EFF
    b_scale_load_rep = warp_tile_n // WMMA_M if is_fp4 else wmma_n_rep
    _enable_split_data_scale_async = os.environ.get(
        "FLYDSL_GEMM_SPLIT_DATA_SCALE_ASYNC", "1") != "0"
    split_data_tdm_scale_async = (
        _enable_split_data_scale_async
        and wave_specialized_tdm
        and not use_cluster
        and (wmma_m_rep * scale_k_per_tile) % WAVE_SIZE == 0
        and (b_scale_load_rep * scale_k_per_tile) % WAVE_SIZE == 0
    )

    _b_frag_loads_per_wn = 2 if is_a8w4 else 4
    _bs_ds_loads = (
        wmma_n_rep * _b_frag_loads_per_wn
        + (b_scale_load_rep + 3) // 4
        + (wmma_m_rep + 3) // 4
    )

    lds_a_stride_bytes = packed_tile_k_a + LDS_PAD_A_BYTES

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * packed_tile_k_b
    _scale_guard_bytes = 16
    lds_a_scale_bytes = tile_m * scale_k_per_tile + _scale_guard_bytes
    lds_b_scale_bytes = tile_n * scale_k_per_tile + _scale_guard_bytes

    _MIN_TDM_DIM0_BYTES = 256

    def _scale_tdm_group(total_reps, warp_reps):
        g = min(total_reps, max(warp_reps, _MIN_TDM_DIM0_BYTES // scale_k_per_tile))
        if g % warp_reps != 0:
            g = warp_reps
        return g

    _as_total_m_reps = tile_m // WMMA_M
    as_tdm_group_m = (
        wmma_m_rep if split_data_tdm_scale_async
        else _scale_tdm_group(_as_total_m_reps, wmma_m_rep)
    )
    interleaved_scale_cols_a = as_tdm_group_m * scale_k_per_tile

    _bs_total_n_reps = tile_n // WMMA_M
    bs_tdm_group_n = (
        b_scale_load_rep if split_data_tdm_scale_async
        else _scale_tdm_group(_bs_total_n_reps, b_scale_load_rep)
    )
    interleaved_scale_cols_b = bs_tdm_group_n * scale_k_per_tile

    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    # TDM descriptors partition a tile cooperatively across ``num_warps`` by
    # deriving per-wave offsets from ``wave_id``. In wave-specialized mode we
    # dedicate one loader wave to each tensor (A/B/A_scale/B_scale), so each
    # active loader wave must issue a full-tile descriptor by itself.
    tdm_desc_num_warps = 1 if wave_specialized_tdm else num_warps

    # All pipeline stages share the same intra-stage layout. Keep that layout
    # unchanged and only remap each logical stage to a physical base inside one
    # LDS arena so TDM epilogue can alias the dead prefix of the arena.
    stage_layout = SmemAllocator(
        None, arch=gpu_arch, global_sym_name=f"mxscale_{data_format}_layout")
    stage_a_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_data_rel_off + lds_a_data_bytes
    stage_b_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_data_rel_off + lds_b_data_bytes
    stage_a_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_scale_rel_off + lds_a_scale_bytes
    stage_b_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_scale_rel_off + lds_b_scale_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    _last_compute_stage = _base_tail_plan[-1][1]

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"mxscale_{data_format}_{tile_m}x{tile_n}x{tile_k}_"
            f"{m_warp}x{n_warp}_{num_buffers}buf_arena"),
    )

    # Named-barrier infrastructure (opt-in via FLYDSL_GEMM_NAMED_BAR=1) — an
    # alternative to the default ``s_barrier_signal -1`` (workgroup-barrier)
    # pipeline fence. A wave may be joined to at most one named barrier at
    # a time so a single workgroup-wide barrier is used.
    # See :func:`init_named_barriers` for the INIT+JOIN protocol.
    _named_bar_enabled = (
        wave_specialized_tdm and not split_data_tdm_scale_async
        and os.environ.get("FLYDSL_GEMM_NAMED_BAR", "0") == "1"
    )
    if _named_bar_enabled:
        nbar_alloc = NamedBarrierAllocator(
            sym_prefix=(
                f"mxscale_{data_format}_{tile_m}x{tile_n}x{tile_k}_"
                f"{m_warp}x{n_warp}_{num_buffers}buf_nbar"))
        nbar_sync = nbar_alloc.alloc(member_count=num_warps, name_hint="sync")
    else:
        nbar_alloc = None
        nbar_sync = None

    stage_phys_order = [i for i in range(num_buffers) if i != _last_compute_stage]
    stage_phys_order.append(_last_compute_stage)
    stage_base_off = [0] * num_buffers
    for phys_i, logical_i in enumerate(stage_phys_order):
        stage_base_off[logical_i] = phys_i * stage_pitch_bytes
    arena_alloc.ptr = stage_pitch_bytes * num_buffers
    arena_total_bytes = arena_alloc.ptr
    epilogue_fence_threshold_bytes = tdm_epilogue_fence_threshold_bytes(
        stage_base_off=stage_base_off,
        tail_plan=_base_tail_plan,
        loop_iters=loop_iters,
        extra=extra,
    )

    stage_a_data_off = [
        stage_base_off[i] + stage_a_data_rel_off for i in range(num_buffers)
    ]
    stage_b_data_off = [
        stage_base_off[i] + stage_b_data_rel_off for i in range(num_buffers)
    ]
    stage_a_scale_off = [
        stage_base_off[i] + stage_a_scale_rel_off for i in range(num_buffers)
    ]
    stage_b_scale_off = [
        stage_base_off[i] + stage_b_scale_rel_off for i in range(num_buffers)
    ]

    if use_tdm_store:
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        d_need_epilogue_fence = total_d_bytes > epilogue_fence_threshold_bytes
        if total_d_bytes > arena_total_bytes:
            arena_total_bytes = total_d_bytes
            arena_alloc.ptr = total_d_bytes
    check_smem_capacity(arena_total_bytes, gpu_arch)

    # TENSORcnt is tracked per-wave in hardware. The regular path issues four
    # tensor ops per wave per K-stage, while the wave-specialized path issues
    # only one tensor op from each dedicated loader wave.
    TDM_LOADS_PER_STEP = 1 if wave_specialized_tdm else 4
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    # Pre-compute epilogue sub-tile layout (unified for FP4 vec16 and FP8 vec8)
    _sub_tiles = []
    for _wm in range(wmma_m_rep):
        for _wn in range(wmma_n_rep):
            if is_fp4:
                # vec<16,f32>: split into 2 × 8 elements (2 × 16-col halves)
                for _half in range(2):
                    acc_idx = _wm * wmma_n_rep + _wn
                    vec_base = _half * 8
                    m_off = _wm * WMMA_M
                    n_sub = _wn * 2 + _half
                    _sub_tiles.append((acc_idx, vec_base, m_off, n_sub))
            else:
                # vec<8,f32>: single 8-element block
                acc_idx = _wm * wmma_n_rep + _wn
                m_off = _wm * WMMA_M
                n_sub = _wn
                _sub_tiles.append((acc_idx, 0, m_off, n_sub))

    COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING = "row_major_streaming"
    COMPUTE_SCHEDULE_FP4_COL_BAND = "fp4_col_band"
    COMPUTE_SCHEDULE_COL_BAND = "col_band"

    def _pick_compute_schedule_kind():
        # The col-band (quadrant) schedule splits B loads into left/right
        # halves and processes four quadrants (top-left, bottom-left,
        # top-right, bottom-right).  This overlaps B-right loading with
        # quadrant-1 WMMA compute, dramatically reducing the initial load
        # burst and improving xDL utilization.
        if wmma_m_rep % 2 != 0 or wmma_n_rep % 2 != 0:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        if n_accs < 8:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        if is_fp4:
            return COMPUTE_SCHEDULE_FP4_COL_BAND
        return COMPUTE_SCHEDULE_COL_BAND

    compute_schedule_kind = _pick_compute_schedule_kind()
    use_fp4_bank_friendly_schedule = (
        compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND
    )
    use_col_band_schedule = (
        compute_schedule_kind in (COMPUTE_SCHEDULE_FP4_COL_BAND,
                                  COMPUTE_SCHEDULE_COL_BAND)
    )

    if use_col_band_schedule:
        _bank_half_wm = wmma_m_rep // 2
        _bank_half_wn = wmma_n_rep // 2
        _bank_group_size = _bank_half_wm * _bank_half_wn
        _bank_half_b_scale_rep = b_scale_load_rep // 2
        _bank_group_to_row_major = []
        for _wm in range(_bank_half_wm):
            for _wn in range(_bank_half_wn):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm, wmma_m_rep):
            for _wn in range(_bank_half_wn):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm):
            for _wn in range(_bank_half_wn, wmma_n_rep):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm, wmma_m_rep):
            for _wn in range(_bank_half_wn, wmma_n_rep):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        # Enable back-to-back WMMA issue (SCHED_MODE bit[4] = DISABLE_VALU_STALL)
        rocdl.disable_xdl_arb_stall()

        if inst_prefetch:
            from flydsl._mlir.dialects import llvm as llvm_dialect
            if arith.cmpi(arith.CmpIPredicate.eq, rocdl.wave_id(),
                          arith.constant(0, type=T.i32)):
                _prefetch_lines = ["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 8, 1), 1"]
                for _pg in range_constexpr(10):
                    _prefetch_lines.append(
                        f"s_prefetch_inst_pc_rel {_pg * 4096}, s0, 31")
                llvm_dialect.inline_asm(
                    None, [],
                    "\n".join(_prefetch_lines),
                    "", has_side_effects=True,
                )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = fx.Index(gpu.block_idx.z) if split_k > 1 else arith.index(0)

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        split_k_base = bz * arith.index(split_k_chunk)

        if use_cluster:
            local_x, local_y = gpu.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                local_x, local_y, cluster_m, cluster_n)
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1),
            fx.get(thr_coord, 2), fx.get(thr_coord, 3))

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_stride = arith.index(N)
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)
        zero_i32 = arith.constant(0, type=T.i32)

        def make_desc_a(memref, k_base):
            k_packed_off = k_base / arith.index(PACK_FACTOR_A)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=memref,
                global_offset=(blk_m, k_packed_off),
                tensor_shape=(tile_m, packed_tile_k_a),
                strides=(K_packed_a, 1),
                tile_shape=(tile_m, packed_tile_k_a),
                elem_bytes=1,
                pad_interval=packed_tile_k_a, pad_amount=LDS_PAD_A_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        def make_desc_b(memref, k_base):
            k_packed_off = k_base / arith.index(PACK_FACTOR_B)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=memref,
                global_offset=(blk_n / arith.index(16),
                               k_packed_off * arith.index(16)),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(tile_n // 16, packed_tile_k_b * 16),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        def make_desc_a_half(memref, k_base, half_idx: int):
            k_packed_off = k_base / arith.index(PACK_FACTOR_A)
            half_m = tile_m // 2
            half_m_off = arith.index(half_idx * half_m)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=memref,
                global_offset=(blk_m + half_m_off, k_packed_off),
                tensor_shape=(half_m, packed_tile_k_a),
                strides=(K_packed_a, 1),
                tile_shape=(half_m, packed_tile_k_a),
                elem_bytes=1,
                pad_interval=packed_tile_k_a, pad_amount=LDS_PAD_A_BYTES,
                num_warps=1,
                lds_byte_offset=arith.index(
                    half_idx * half_m * lds_a_stride_bytes),
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        def make_desc_b_half(memref, k_base, half_idx: int):
            k_packed_off = k_base / arith.index(PACK_FACTOR_B)
            half_b_rows = (tile_n // 16) // 2
            half_b_row_off = arith.index(half_idx * half_b_rows)
            half_b_bytes = half_b_rows * packed_tile_k_b * 16
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=memref,
                global_offset=(blk_n / arith.index(16) + half_b_row_off,
                               k_packed_off * arith.index(16)),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(half_b_rows, packed_tile_k_b * 16),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=1,
                lds_byte_offset=arith.index(half_idx * half_b_bytes),
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        def make_desc_as(memref, k_base):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_m / arith.index(as_tdm_group_m)
            inner_off = k_scale_off * arith.index(as_tdm_group_m)
            _as_tdm_outer = WMMA_M * _as_total_m_reps // as_tdm_group_m
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a_scale, lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(_as_tdm_outer, interleaved_scale_cols_a),
                strides=(as_tdm_group_m * K_scale, 1),
                tile_shape=(_as_tdm_outer, interleaved_scale_cols_a),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        def make_desc_bs(memref, k_base):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_n / arith.index(bs_tdm_group_n)
            inner_off = k_scale_off * arith.index(bs_tdm_group_n)
            _bs_tdm_outer = WMMA_M * _bs_total_n_reps // bs_tdm_group_n
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b_scale, lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(_bs_tdm_outer, interleaved_scale_cols_b),
                strides=(bs_tdm_group_n * K_scale, 1),
                tile_shape=(_bs_tdm_outer, interleaved_scale_cols_b),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        if wave_specialized_tdm:
            tdm_wave_id = rocdl.wave_id()
            tdm_wave_is_a = arith.cmpi(
                arith.CmpIPredicate.eq, tdm_wave_id,
                arith.constant(0, type=T.i32))
            tdm_wave_is_b = arith.cmpi(
                arith.CmpIPredicate.eq, tdm_wave_id,
                arith.constant(1, type=T.i32))
            tdm_wave_is_as = arith.cmpi(
                arith.CmpIPredicate.eq, tdm_wave_id,
                arith.constant(2, type=T.i32))

            def _select_wave_tdm_value(a_value, b_value, as_value, bs_value):
                result = arith.select(tdm_wave_is_as, as_value, bs_value)
                result = arith.select(tdm_wave_is_b, b_value, result)
                return arith.select(tdm_wave_is_a, a_value, result)

        elem_ty_lds = T.f16

        def _precompute_a_lane_bases(lds_ptr):
            """Precompute per-wm A fragment lane base addresses (byte offsets)."""
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            # K-dimension interleaving: kgrp0/kgrp1 read alternating 128-bit chunks
            # All formats: kgrp offset = 16 bytes (one ds_load_b128 width)
            k_half_off = lane_kgrp * arith.index(16)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
                bases.append(base)
            return lds_ptr, bases


        def load_a_frag(lds_buffer, a_lane_base, ks):
            """Load one A-fragment from LDS.

            FP4: vec<8xi32> via 2 × ds_load_b128 (32 bytes per lane).
            FP8/A8W4: vec<16xi32> via 4 × ds_load_b128 (64 bytes per lane).
              Interleaved K layout:
              kgrp0 reads bytes [0:15],[32:47],[64:79],[96:111] (stride=32)
              kgrp1 reads bytes [16:31],[48:63],[80:95],[112:127] (stride=32)
            """
            k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR_A)
            byte_off = a_lane_base + k_byte_off
            v0 = lds_load_b128_raw(lds_buffer, byte_off)
            if is_fp4:
                # Interleaved stride=32: +0, +32
                v1 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))
                return vector.shuffle(v0, v1, list(range(8)))
            else:
                # Interleaved stride=32: +0, +32, +64, +96
                v1 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))
                v2 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(64))
                v3 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(96))
                v01 = vector.shuffle(v0, v1, list(range(8)))
                v23 = vector.shuffle(v2, v3, list(range(8)))
                return vector.shuffle(v01, v23, list(range(16)))

        def _precompute_b_lane_bases(lds_ptr):
            """Precompute per-wn B fragment lane base addresses (byte offsets).

            FP4: 2 bases per wn (32-col WMMA = 2 N-groups of 16).
            FP8: 1 base per wn (16-col WMMA = 1 N-group).
            A8W4: 1 base per wn (16-col WMMA, FP4 packed weight).

            K-dimension interleaving for FP8/A8W4:
              kgrp0 and kgrp1 read alternating 16x16 tiles (stride = 2 tiles).
              kgrp offset = 1 tile = 256 bytes.
            """
            _ngroup_stride = packed_tile_k_b * 16
            _n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
            row_off = lane16 * arith.index(16)
            # All formats: interleaved — kgrp offset = 1 tile = 256 bytes
            k_tile_off = lane_kgrp * arith.index(256)
            bases = []
            if is_fp4:
                for wn_half in range_constexpr(wmma_n_rep * 2):
                    ngroup_off = _n_group_base * arith.index(_ngroup_stride) \
                        + arith.index(wn_half * _ngroup_stride)
                    bases.append(ngroup_off + row_off + k_tile_off)
            else:
                # FP8 and A8W4: 1 base per wn (16-col WMMA)
                for wn in range_constexpr(wmma_n_rep):
                    ngroup_off = _n_group_base * arith.index(_ngroup_stride) \
                        + arith.index(wn * _ngroup_stride)
                    bases.append(ngroup_off + row_off + k_tile_off)
            return lds_ptr, bases

        def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
            """Load one B-fragment from preshuffled LDS.

            FP4: 32x128 → vec<16xi32> from 2 N-groups (bases[wn*2], bases[wn*2+1]).
            FP8: 16x128 → vec<16xi32> from 1 N-group (bases[wn]).
            A8W4: 16x128 FP4 → vec<8xi32> from 1 N-group (bases[wn]).

            K-dimension interleaving (FP8/A8W4):
              Stride = 2 tiles = 512 bytes between loads.
              kgrp0 reads tiles 0,2,4,6; kgrp1 reads tiles 1,3,5,7.
            """
            if is_fp4:
                # FP4: 2 N-groups per wn, 4 tiles per N-group
                # Interleaved stride=512 (2 tiles): kgrp0→tiles 0,2; kgrp1→tiles 1,3
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 4 tiles total per N-group
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn * 2] + k_subtile_off
                v0 = lds_load_b128_raw(lds_buffer, base0)
                v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(512))
                base1 = b_lane_bases[wn * 2 + 1] + k_subtile_off
                v2 = lds_load_b128_raw(lds_buffer, base1)
                v3 = lds_load_b128_raw(lds_buffer, base1 + arith.index(512))
                v01 = vector.shuffle(v0, v1, list(range(8)))
                v23 = vector.shuffle(v2, v3, list(range(8)))
                return vector.shuffle(v01, v23, list(range(16)))
            elif is_a8w4:
                # A8W4: FP4 weight, 4 tiles per N-group
                # Interleaved stride=512: kgrp0→tiles 0,2; kgrp1→tiles 1,3
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 4 tiles total
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn] + k_subtile_off
                v0 = lds_load_b128_raw(lds_buffer, base0)
                v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(512))
                return vector.shuffle(v0, v1, list(range(8)))
            else:
                # FP8: 8 tiles per N-group
                # Interleaved stride=512: kgrp0→tiles 0,2,4,6; kgrp1→tiles 1,3,5,7
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 8 tiles total
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn] + k_subtile_off
                v0 = lds_load_b128_raw(lds_buffer, base0)
                v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(512))
                v2 = lds_load_b128_raw(lds_buffer, base0 + arith.index(1024))
                v3 = lds_load_b128_raw(lds_buffer, base0 + arith.index(1536))
                v01 = vector.shuffle(v0, v1, list(range(8)))
                v23 = vector.shuffle(v2, v3, list(range(8)))
                return vector.shuffle(v01, v23, list(range(16)))

        def _precompute_scale_lane_bases(lds_ptr, warp_base, reps,
                                         interleaved_cols, tdm_group=0):
            """Precompute scale lane bases (byte offsets).

            When *tdm_group* > *reps*, the preshuffle packs
            tdm_group WMMA reps per output row to widen TDM dim0
            for Direct Copy.  Each wave reads only its own
            *reps*-wide slice within the wider row.
            """
            if tdm_group > reps:
                _elems_per_scale = WMMA_M // SCALES_PER_WMMA
                warp_rep_off = warp_base / arith.index(_elems_per_scale)
                base = lane16 * arith.index(interleaved_cols) + warp_rep_off
            else:
                warp_lds_row = warp_base / arith.index(reps) + lane16
                base = warp_lds_row * arith.index(interleaved_cols)
            base = base + lane_kgrp * arith.index(SCALES_PER_WMMA)
            return lds_ptr, [base]

        def issue_scale_loads_raw(lds_buffer, scale_base, reps, ks=0,
                                  ks_group=0):
            """Issue ds_load_b128 for scale data WITHOUT extracting.

            Returns raw vec4 values that must later be passed to
            extract_scales_from_raw().  Splitting load from extract
            lets the caller insert independent work (A-frag loads)
            in between to hide LDS latency.
            """
            _ks_reps = ks_group if ks_group > 0 else reps
            ks_byte_off = ks * _ks_reps * SCALES_PER_WMMA
            eff_base = scale_base if ks_byte_off == 0 \
                else scale_base + arith.index(ks_byte_off)
            num_loads = (reps + 3) // 4
            vecs = []
            for ld in range_constexpr(num_loads):
                off = eff_base if ld == 0 else eff_base + arith.index(ld * 16)
                vecs.append(lds_load_b128_raw(lds_buffer, off))
            return vecs

        def extract_scales_from_raw(raw_vecs, reps):
            """Extract individual scale values from raw ds_load_b128 vec4s."""
            results = []
            for i in range_constexpr(reps):
                vi = vector.extract(raw_vecs[i // 4],
                                    static_position=[i % 4], dynamic_position=[])
                results.append(vi)
            return results

        def load_scale_b128(lds_buffer, scale_base, reps, ks=0,
                            ks_group=0):
            """Load all wmma_rep scales via ds_load_b128(s) for K-subtile *ks*.

            When *ks_group* > 0 the LDS row packs ks_group reps
            per K-subtile, so the ks stride is ks_group * SCALES_PER_WMMA
            instead of reps * SCALES_PER_WMMA.
            """
            raw = issue_scale_loads_raw(lds_buffer, scale_base, reps,
                                        ks, ks_group)
            return extract_scales_from_raw(raw, reps)

        def load_scale_slice_b128(lds_buffer, scale_base, full_reps,
                                  rep_start, rep_count, ks=0,
                                  ks_group=0):
            """Load a contiguous slice of packed scale VGPRs for one K-subtile."""
            raw_vecs = issue_scale_slice_loads_raw(
                lds_buffer, scale_base, full_reps, rep_start, rep_count, ks,
                ks_group)
            results = []
            for i in range_constexpr(rep_count):
                vi = vector.extract(raw_vecs[i // 4],
                                    static_position=[i % 4], dynamic_position=[])
                results.append(vi)
            return results

        def issue_scale_slice_loads_raw(lds_buffer, scale_base, full_reps,
                                        rep_start, rep_count, ks=0,
                                        ks_group=0):
            """Issue raw ds_load_b128 for a contiguous scale slice."""
            _ks_reps = ks_group if ks_group > 0 else full_reps
            ks_byte_off = (ks * _ks_reps + rep_start) * SCALES_PER_WMMA
            eff_base = scale_base if ks_byte_off == 0 \
                else scale_base + arith.index(ks_byte_off)
            num_loads = (rep_count + 3) // 4
            vecs = []
            for ld in range_constexpr(num_loads):
                off = eff_base if ld == 0 else eff_base + arith.index(ld * 16)
                vecs.append(lds_load_b128_raw(lds_buffer, off))
            return vecs

        def _issue_b_and_scale_loads(b_buf, b_bases, bs_buf, bs_bases,
                                     as_buf, as_bases, ks):
            """Load B frags + issue scale ds_loads (raw, no extract).

            Returns (b_frags, bs_raw_vecs, as_raw_vecs).  The caller
            must pass the raw vecs through _extract_and_filter_scales()
            before consuming them in WMMA.
            """
            b_frags = [load_b_frag(b_buf, b_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
            bs_raw = issue_scale_loads_raw(bs_buf, bs_bases[0],
                                           b_scale_load_rep, ks,
                                           ks_group=bs_tdm_group_n)
            as_raw = issue_scale_loads_raw(as_buf, as_bases[0],
                                           wmma_m_rep, ks,
                                           ks_group=as_tdm_group_m)
            return b_frags, bs_raw, as_raw

        def _extract_and_filter_scales(bs_raw, as_raw):
            """Extract + filter scales from raw LDS vec4 values.

            Op_sel packing: each scale VGPR holds two adjacent WMMA reps
            (lo/hi 16-bit halves), so we stride-2 to get one VGPR per
            pair and use scaleType op_sel to select the half.
            FP4 32x16 WMMA: only A-scales use op_sel (via scaleBType);
            B-scales are indexed directly (scaleAType always 0).
            FP8/A8W4 16x16 WMMA: both A and B scales use op_sel.
            """
            b_scales_all = extract_scales_from_raw(bs_raw, b_scale_load_rep)
            a_scales_all = extract_scales_from_raw(as_raw, wmma_m_rep)
            a_scales = a_scales_all[::2]
            if is_fp4:
                b_scales = b_scales_all
            else:
                b_scales = b_scales_all[::2]
            return b_scales, a_scales

        def _emit_wmma(accs, wm, wn, a_frag, b_frags, a_scales, b_scales):
            """Emit one WMMA instruction (format-specific).

            Scale op_sel packing: adjacent WMMA reps share one scale VGPR,
            selecting lo (op_sel=0) or hi (op_sel=1) 16-bit half.
            A-scale op_sel (via scaleBType) is used by all formats.
            B-scale op_sel (via scaleAType) is used by FP8/A8W4 only;
            FP4 32x16 WMMA indexes B-scales directly.
            """
            idx = wm * wmma_n_rep + wn
            a_scale_idx = wm // 2
            a_opsel = wm % 2

            if is_fp4:
                accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                    T.vec(16, T.f32),
                    b_frags[wn], a_frag, accs[idx],
                    b_scales[wn * 2], a_scales[a_scale_idx],
                    scaleAType=0,
                    scaleBType=a_opsel,
                )
            else:
                b_scale_idx = wn // 2
                b_opsel = wn % 2
                accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                    T.vec(8, T.f32),
                    b_frags[wn], a_frag, accs[idx],
                    b_scales[b_scale_idx], a_scales[a_scale_idx],
                    fmtA=4 if is_a8w4 else 0, fmtB=0,
                    scaleAType=b_opsel, scaleBType=a_opsel,
                )

        def _a_streaming_compute(accs, a_buf, a_bases, b_frags, bs_raw,
                                 as_raw, ks, emit_filler=None,
                                 next_bs_info=None,
                                 mid_compute_callback=None):
            """Half-based A-streaming with deferred scale extraction.

            Scale ds_loads are issued in _issue_b_and_scale_loads but
            the vector.extract is deferred until AFTER A-frag loads,
            giving the LDS pipeline time to retire scale data before
            the extract forces a wait.
            """
            next_result = None
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm

            a_frags_front = [load_a_frag(a_buf, a_bases[wm], ks)
                             for wm in range_constexpr(_front_wm)]

            b_scales, a_scales = _extract_and_filter_scales(bs_raw, as_raw)

            def _emit_rows(start_wm, a_frags):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    is_last = (wm == wmma_m_rep - 1)
                    if is_last and emit_filler is not None:
                        rocdl.sched_barrier(0)
                        emit_filler()
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        _emit_wmma(accs, wm, wn, a_frags[frag_i], b_frags,
                                   a_scales, b_scales)

            _use_partial_drain = (
                next_bs_info is not None
                and _front_wm * wmma_n_rep >= 4
            )

            if _use_partial_drain:
                nb_buf, nb_bases, nbs_buf, nbs_bases, \
                    nas_buf, nas_bases, n_ks = next_bs_info
                next_result = _issue_b_and_scale_loads(
                    nb_buf, nb_bases, nbs_buf, nbs_bases,
                    nas_buf, nas_bases, n_ks)
                rocdl.s_wait_dscnt(_bs_ds_loads)
            else:
                rocdl.s_wait_dscnt(0)

            _emit_rows(0, a_frags_front)

            if mid_compute_callback is not None:
                rocdl.sched_barrier(0)
                mid_compute_callback()

            if _back_wm > 0:
                a_frags_back = [load_a_frag(a_buf, a_bases[_front_wm + h], ks)
                                for h in range_constexpr(_back_wm)]
                _back_drain = _bs_ds_loads if _use_partial_drain else 0
                rocdl.s_wait_dscnt(_back_drain)
                _emit_rows(_front_wm, a_frags_back)

            if _use_partial_drain:
                return accs, next_result
            if next_bs_info is not None:
                nb_buf, nb_bases, nbs_buf, nbs_bases, \
                    nas_buf, nas_bases, n_ks = next_bs_info
                next_result = _issue_b_and_scale_loads(
                    nb_buf, nb_bases, nbs_buf, nbs_bases,
                    nas_buf, nas_bases, n_ks)
                return accs, next_result
            return accs

        # ── Compute on one LDS buffer ──
        def compute_tile(accs_in, lds_a, lds_b, lds_as, lds_bs,
                         emit_filler=None, mid_compute_callback=None):
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            as_buf, as_bases = _precompute_scale_lane_bases(
                lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a,
                tdm_group=as_tdm_group_m)
            bs_buf, bs_bases = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b,
                tdm_group=bs_tdm_group_n)

            if k_wmma_steps == 1:
                b_frags, bs_raw, as_raw = _issue_b_and_scale_loads(
                    b_buf, b_bases, bs_buf, bs_bases, as_buf, as_bases, 0)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, b_frags, bs_raw,
                    as_raw, 0, emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback)
            else:
                prev_b, prev_bs_raw, prev_as_raw = \
                    _issue_b_and_scale_loads(
                        b_buf, b_bases, bs_buf, bs_bases,
                        as_buf, as_bases, 0)
                for ks in range_constexpr(k_wmma_steps - 1):
                    _mid_cb = mid_compute_callback if ks == 0 else None
                    current_accs, (prev_b, prev_bs_raw, prev_as_raw) = \
                        _a_streaming_compute(
                            current_accs, a_buf, a_bases,
                            prev_b, prev_bs_raw, prev_as_raw, ks,
                            next_bs_info=(b_buf, b_bases, bs_buf, bs_bases,
                                          as_buf, as_bases, ks + 1),
                            mid_compute_callback=_mid_cb)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases,
                    prev_b, prev_bs_raw, prev_as_raw,
                    k_wmma_steps - 1, emit_filler=emit_filler)
            return current_accs

        def compute_tile_fp4_bank_friendly(
            accs_in, lds_a, lds_b, lds_as, lds_bs, emit_filler=None,
            mid_compute_callback=None,
        ):
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            as_buf, as_bases = _precompute_scale_lane_bases(
                lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a,
                tdm_group=as_tdm_group_m)
            bs_buf, bs_bases = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b,
                tdm_group=bs_tdm_group_n)
            _b_half_scale_loads = (_bank_half_b_scale_rep + 3) // 4

            def _fp4_get_a_scale_and_opsel(a_scales_all, wm_idx):
                return a_scales_all[(wm_idx // 2) * 2], wm_idx % 2

            def _load_a_group(wm_base, wm_count, ks):
                return [
                    load_a_frag(a_buf, a_bases[wm_base + wm_local], ks)
                    for wm_local in range_constexpr(wm_count)
                ]

            def _load_b_half(wn_base, ks):
                return [
                    load_b_frag(b_buf, b_bases, wn_base + wn_local, ks)
                    for wn_local in range_constexpr(_bank_half_wn)
                ]

            def _load_b_half_bundle(wn_base, rep_start, ks):
                b_frags = _load_b_half(wn_base, ks)
                b_scales = load_scale_slice_b128(
                    bs_buf, bs_bases[0],
                    b_scale_load_rep, rep_start,
                    _bank_half_b_scale_rep, ks,
                    ks_group=bs_tdm_group_n)
                return b_frags, b_scales

            def _emit_group_rows(group_base, wm_base, a_frags, b_frags, a_scales,
                                 b_scales, row_start, row_count,
                                 emit_filler_now=False):
                if emit_filler_now and emit_filler is not None:
                    rocdl.sched_barrier(0)
                    emit_filler()
                for row_offset in range_constexpr(row_count):
                    wm_local = row_start + row_offset
                    a_frag = a_frags[wm_local]
                    global_wm = wm_base + wm_local
                    a_scale, a_opsel = _fp4_get_a_scale_and_opsel(
                        a_scales, global_wm)
                    row_base = group_base + wm_local * _bank_half_wn
                    for wn_local in range_constexpr(_bank_half_wn):
                        idx = row_base + wn_local
                        current_accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                            T.vec(16, T.f32),
                            b_frags[wn_local], a_frag, current_accs[idx],
                            b_scales[wn_local * 2], a_scale,
                            scaleAType=0,
                            scaleBType=a_opsel,
                        )

            def _emit_group(group_base, wm_base, a_frags, b_frags, a_scales,
                            b_scales, emit_filler_now=False):
                _emit_group_rows(
                    group_base, wm_base,
                    a_frags, b_frags,
                    a_scales, b_scales,
                    0, _bank_half_wm,
                    emit_filler_now=emit_filler_now,
                )

            b_left_frags, b_left_scales = _load_b_half_bundle(0, 0, 0)

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales_all = load_scale_b128(as_buf, as_bases[0],
                                               wmma_m_rep, ks,
                                               ks_group=as_tdm_group_m)

                a_top_frags = _load_a_group(0, _bank_half_wm, ks)
                a_bottom_frags = _load_a_group(_bank_half_wm, _bank_half_wm, ks)

                # Wait for bottom-A loads; top-A stays in flight during Q1.
                rocdl.s_wait_dscnt(_bank_half_wm * DS_LOADS_PER_A_FRAG)

                _emit_group(
                    0, 0,
                    a_top_frags,
                    b_left_frags,
                    a_scales_all,
                    b_left_scales,
                )

                if ks == 0 and mid_compute_callback is not None:
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                b_right_frags, b_right_scales = _load_b_half_bundle(
                    _bank_half_wn, _bank_half_b_scale_rep, ks)

                # Hold only the next B half outstanding while the second
                # quadrant consumes the current left-half fragments.
                rocdl.s_wait_dscnt(_bank_half_wn * 4 + _b_half_scale_loads)

                _emit_group(
                    _bank_group_size, _bank_half_wm,
                    a_bottom_frags,
                    b_left_frags,
                    a_scales_all,
                    b_left_scales,
                )

                if not is_last_ks:
                    next_left_frags, next_left_scales = _load_b_half_bundle(
                        0, 0, ks + 1)
                    # Older right-half loads must be ready before consuming
                    # them, while the next ks left-half preload can remain in
                    # flight under the final two quadrants.
                    rocdl.s_wait_dscnt(
                        _bank_half_wn * 4 + _b_half_scale_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_group(
                    _bank_group_size * 2, 0,
                    a_top_frags,
                    b_right_frags,
                    a_scales_all,
                    b_right_scales,
                )
                _emit_group(
                    _bank_group_size * 3, _bank_half_wm,
                    a_bottom_frags,
                    b_right_frags,
                    a_scales_all,
                    b_right_scales,
                    emit_filler_now=is_last_ks,
                )

                if not is_last_ks:
                    b_left_frags = next_left_frags
                    b_left_scales = next_left_scales

            return current_accs

        def compute_tile_col_band(accs_in, lds_a, lds_b, lds_as, lds_bs,
                                  emit_filler=None,
                                  mid_compute_callback=None,
                                  prefetched_b_left_head=None,
                                  emit_prefetch_next=None):
            """Column-band (quadrant) compute for FP8/A8W4.

            Splits B loads into left (wn<half_wn) and right (wn>=half_wn)
            halves, processing four quadrants:
              Q1: top-A × B-left
              Q2: bottom-A × B-left  (B-right loading overlaps)
              Q3: top-A × B-right
              Q4: bottom-A × B-right (next-ks B-left prefetch overlaps)

            This cuts the initial ds_load burst from ~76 to ~26 before
            first WMMA, dramatically improving xDL utilization.
            """
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            as_buf, as_bases = _precompute_scale_lane_bases(
                lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a,
                tdm_group=as_tdm_group_m)
            bs_buf, bs_bases = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b,
                tdm_group=bs_tdm_group_n)

            _half_wn = wmma_n_rep // 2
            _half_wm = wmma_m_rep // 2
            _half_b_scale_rep = b_scale_load_rep // 2
            _half_b_scale_loads = (_half_b_scale_rep + 3) // 4

            def _load_b_half(wn_start, ks):
                return [load_b_frag(b_buf, b_bases, wn_start + wn_local, ks)
                        for wn_local in range_constexpr(_half_wn)]

            def _load_b_half_with_scales(wn_start, scale_rep_start, ks):
                frags = _load_b_half(wn_start, ks)
                scales_all = load_scale_slice_b128(
                    bs_buf, bs_bases[0],
                    b_scale_load_rep, scale_rep_start,
                    _half_b_scale_rep, ks,
                    ks_group=bs_tdm_group_n)
                # Op_sel packing: adjacent reps share one VGPR;
                # stride-2 gives one per pair (same as _extract_and_filter_scales).
                scales = scales_all[::2]
                return frags, scales

            def _load_a_group(wm_start, count, ks):
                return [load_a_frag(a_buf, a_bases[wm_start + i], ks)
                        for i in range_constexpr(count)]

            def _emit_q_rows(a_frags, b_frags, a_scales, b_scales,
                             wm_base, wn_base, group_base, row_start=0):
                fmt_a = 4 if is_a8w4 else 0
                for row_off in range_constexpr(len(a_frags)):
                    wm_local = row_start + row_off
                    wm = wm_base + wm_local
                    a_scale_idx = wm // 2
                    a_opsel = wm % 2
                    for wn_local in range_constexpr(_half_wn):
                        wn = wn_base + wn_local
                        idx = group_base + wm_local * _half_wn + wn_local
                        b_scale_idx = wn_local // 2
                        b_opsel = wn_local % 2
                        current_accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                            T.vec(8, T.f32),
                            b_frags[wn_local], a_frags[wm_local],
                            current_accs[idx],
                            b_scales[b_scale_idx], a_scales[a_scale_idx],
                            fmtA=fmt_a, fmtB=0,
                            scaleAType=b_opsel, scaleBType=a_opsel,
                        )

            def _emit_q(a_frags, b_frags, a_scales, b_scales,
                        wm_base, wn_base, group_base):
                # Fence the Q-block on both sides so LLVM cannot
                # interleave ds_load between v_wmma_scale instructions.
                rocdl.sched_barrier(0)
                _emit_q_rows(
                    a_frags, b_frags, a_scales, b_scales,
                    wm_base, wn_base, group_base, row_start=0)
                rocdl.sched_barrier(0)

            _q_size = _half_wm * _half_wn  # group size for quadrant indexing

            # Load left B half + all A-scales (A-scales are small, 1 load)
            if prefetched_b_left_head is None:
                b_left_frags, b_left_scales = _load_b_half_with_scales(0, 0, 0)
            else:
                # Use prefetched head frags (from prev buf's Q4);
                # only load the remaining tail frags + scales here.
                _pref_count = len(prefetched_b_left_head)
                _remain_count = _half_wn - _pref_count
                b_left_back = [
                    load_b_frag(b_buf, b_bases, _pref_count + i, 0)
                    for i in range_constexpr(_remain_count)
                ]
                b_left_scales_all = load_scale_slice_b128(
                    bs_buf, bs_bases[0],
                    b_scale_load_rep, 0,
                    _half_b_scale_rep, 0,
                    ks_group=bs_tdm_group_n)
                b_left_frags = list(prefetched_b_left_head) + b_left_back
                b_left_scales = b_left_scales_all[::2]

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1

                # k_wmma_steps>1 (e.g., tile_k=256/2-buf): the Q1↔Q2 mid-callback
                # slot is too short to hide TDM (~960 cy overlap window).
                # Issue TDM at the start of ks=0 to widen the window by +128 cy
                # k_wmma_steps==1 keeps the original Q1↔Q2 site (baseline).
                _mid_early = (ks == 0 and mid_compute_callback is not None
                              and k_wmma_steps > 1)
                if _mid_early:
                    mid_compute_callback()

                # Load A-scales (shared by all quadrants)
                a_scales_all = load_scale_b128(
                    as_buf, as_bases[0], wmma_m_rep, ks,
                    ks_group=as_tdm_group_m)
                a_scales = a_scales_all[::2]

                # Load A top first (Q1 needs only a_top + b_left + a_scales).
                a_top = _load_a_group(0, _half_wm, ks)

                # Wait for a_top + a_scales to be ready (small drain).
                rocdl.s_wait_dscnt(0)

                # Fuse Q1 WMMA + Q2 loads + Q2 WMMA into a single
                # scheduling region so sched_group_barrier hints can organize
                # the "each WMMA paired with <=4 ds_load".
                # Leading barrier keeps prologue loads out of the region.
                rocdl.sched_barrier(0)
                _emit_q_rows(a_top, b_left_frags, a_scales, b_left_scales,
                             0, 0, 0, row_start=0)

                # TDM prefetch stays inside the fused region;
                # TDM has an independent dispatch path and is not classified
                # by sched_group_barrier (mfma/dsrd/dswr/vmem_rd masks), so
                # the scheduler is free to interleave it alongside Q1 WMMAs.
                if ks == 0 and mid_compute_callback is not None and not _mid_early:
                    mid_compute_callback()

                # Q2 loads join the fused region (no s_wait_dscnt: LLVM will
                # auto-insert one driven by VGPR dependency to Q2 WMMAs).
                a_bottom = _load_a_group(_half_wm, _half_wm, ks)
                b_right_frags, b_right_scales = _load_b_half_with_scales(
                    _half_wn, _half_b_scale_rep, ks)

                _emit_q_rows(a_bottom, b_left_frags, a_scales, b_left_scales,
                             _half_wm, 0, _q_size, row_start=0)

                # Pipeline hint — fused_wmma form:
                #   [mfma x 1][dsrd x per_wmma] repeated for N WMMAs, then the
                #   remaining Q2 WMMAs clustered at the tail.
                # Q1 WMMAs are dep-free w.r.t. Q2 loads (their operands are
                # already in VGPR from the pre-region s_wait_dscnt(0)), so the
                # scheduler can freely interleave them with Q2 ds_load. Q2
                # WMMAs carry deps on Q2 loads and naturally tail-cluster.
                _n_q_wmma_s3a = _half_wm * _half_wn
                _q2_dsrd_s3a = (
                    _half_wm * DS_LOADS_PER_A_FRAG
                    + _half_wn * _b_frag_loads_per_wn
                    + _half_b_scale_loads
                )
                _per_wmma_dsrd_s3a = max(1, min(4, _q2_dsrd_s3a // _n_q_wmma_s3a))
                for _ in range_constexpr(_n_q_wmma_s3a):
                    rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
                    rocdl.sched_group_barrier(
                        rocdl.mask_dsrd, _per_wmma_dsrd_s3a, 0)
                rocdl.sched_group_barrier(rocdl.mask_mfma, _n_q_wmma_s3a, 0)
                rocdl.sched_barrier(0)

                # Prefetch next-ks B-left (overlaps with Q3/Q4 WMMA)
                if not is_last_ks:
                    next_left_frags, next_left_scales = \
                        _load_b_half_with_scales(0, 0, ks + 1)
                    rocdl.s_wait_dscnt(
                        _half_wn * _b_frag_loads_per_wn + _half_b_scale_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                # Q3: top-A × B-right
                _emit_q(a_top, b_right_frags, a_scales, b_right_scales,
                        0, _half_wn, _q_size * 2)

                if is_last_ks and emit_filler is not None:
                    rocdl.sched_barrier(0)
                    emit_filler()

                # Q4: bottom-A × B-right
                _emit_q(a_bottom, b_right_frags, a_scales, b_right_scales,
                        _half_wm, _half_wn, _q_size * 3)

                if not is_last_ks:
                    b_left_frags = next_left_frags
                    b_left_scales = next_left_scales

            # Prefetch next buf's b_left_head at Q4 end.
            _next_b_left_head = None
            if emit_prefetch_next is not None:
                rocdl.sched_barrier(0)
                _next_b_left_head = emit_prefetch_next()

            return current_accs, _next_b_left_head

        # Cross-buf b_left prefetch is only valid for the FP8/A8W4 col-band
        # schedule, where compute_tile_scheduled returns the prefetched frags.
        # Row-major and FP4 bank-friendly schedules return None in that slot.
        # Prefetch up to four columns, capped by the current B-left half width.
        #
        # num_buffers<=3 race: emit_prefetch_next reads next_buf_idx =
        # (buf_idx+1)%N, mid_callback writes load_stage = (buf_idx+N-1)%N.
        # For N=2 these collide on the same LDS slot. For N=3, the prefetch can
        # still reach a stage loaded in the previous step while the pipeline
        # fence intentionally leaves one TDM step outstanding, so it may observe
        # stale/in-flight LDS data. Keep this prefetch to 4-stage buffering.
        _col_band_prefetch_enabled = (
            num_buffers >= 4
            and compute_schedule_kind == COMPUTE_SCHEDULE_COL_BAND
        )
        _col_band_prefetch_cols = min(4, wmma_n_rep // 2)

        def _prefetch_b_left_head_for_stage(lds_b_idx):
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_idx)
            return [load_b_frag(b_buf, b_bases, wn_local, 0)
                    for wn_local in range_constexpr(_col_band_prefetch_cols)]

        def hot_loop_scheduler():
            _half_wm = wmma_m_rep // 2
            _b_loads_per_frag = 2 if is_a8w4 else 4
            _a_front = _half_wm * DS_LOADS_PER_A_FRAG   # 8
            _b_total = wmma_n_rep * _b_loads_per_frag    # 32

            for _ks in range_constexpr(k_wmma_steps):
                if _ks == 0:
                    # k-step 0: 42 loads (B=32 + scale=2 + A_front=8) + 32 WMMA
                    # Issue B loads in groups of 8, starting WMMA as soon as
                    # first B data is likely ready (~3 WMMA ≈ 24cy > LDS latency)
                    rocdl.sched_dsrd(8)      # B chunk 1
                    rocdl.sched_dsrd(8)      # B chunk 2
                    rocdl.sched_dsrd(8)      # B chunk 3
                    rocdl.sched_mfma(4)      # 4 WMMA — B chunk 1 data ready
                    rocdl.sched_dsrd(8)      # B chunk 4
                    rocdl.sched_mfma(4)      # 4 WMMA
                    rocdl.sched_dsrd(2)      # scale loads
                    rocdl.sched_dsrd(_a_front)  # A front (8)
                    rocdl.sched_mfma(4)      # 4 WMMA
                    rocdl.sched_mfma(4)      # 4 WMMA (front half done: 16)
                else:
                    # k-step 1+: A_front(8) + next-B prefetch overlapped with 16 WMMA
                    # The next-B loads from k_step 0 are already in flight;
                    # just need A_front for this k-step
                    rocdl.sched_dsrd(_a_front)  # A front (8)
                    rocdl.sched_mfma(4)
                    rocdl.sched_mfma(4)
                    rocdl.sched_mfma(4)
                    rocdl.sched_mfma(4)      # front half done: 16

                # A_back(8) interleaved with back-half WMMA(16)
                rocdl.sched_dsrd(4)      # A back first half
                rocdl.sched_mfma(4)
                rocdl.sched_dsrd(4)      # A back second half
                rocdl.sched_mfma(4)
                rocdl.sched_mfma(4)
                rocdl.sched_mfma(4)      # back half done: 16

                if _ks < k_wmma_steps - 1:
                    # Prefetch next k-step B+scale(34) interleaved in tail
                    # These overlap with subsequent k-step's WMMA
                    rocdl.sched_dsrd(8)
                    rocdl.sched_dsrd(8)
                    rocdl.sched_dsrd(8)
                    rocdl.sched_dsrd(10)

            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp4_bank_friendly():
            _a_all_loads = wmma_m_rep * DS_LOADS_PER_A_FRAG
            _a_scale_loads = (wmma_m_rep + 3) // 4
            _b_half_loads = _bank_half_wn * 4
            _b_half_scale_loads = (_bank_half_b_scale_rep + 3) // 4
            _group_wmma = _bank_group_size
            _right_half_loads = _b_half_loads + _b_half_scale_loads

            for _ks in range_constexpr(k_wmma_steps):
                if _ks == 0:
                    rocdl.sched_dsrd(
                        _a_all_loads + _a_scale_loads
                        + _b_half_loads + _b_half_scale_loads)
                else:
                    rocdl.sched_dsrd(_a_all_loads + _a_scale_loads)
                rocdl.sched_mfma(_group_wmma)
                rocdl.sched_dsrd(_right_half_loads)
                rocdl.sched_mfma(_group_wmma)
                if _ks < k_wmma_steps - 1:
                    rocdl.sched_dsrd(_right_half_loads)
                rocdl.sched_mfma(_group_wmma)
                rocdl.sched_mfma(_group_wmma)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_col_band():
            """Sched hints matching compute_tile_col_band quadrant order.

            Uses sched_barrier(0) only — the col-band code restructuring
            provides sufficient data-dependency structure for LLVM to schedule
            well on its own. Empirically, adding fine-grained sched_group_barrier
            hints is at best neutral and at worst slightly harmful.
            """
            rocdl.sched_barrier(0)

        def compute_tile_scheduled(accs_in, lds_a, lds_b, lds_as, lds_bs,
                                   emit_filler=None,
                                   mid_compute_callback=None,
                                   prefetched_b_left_head=None,
                                   emit_prefetch_next=None):
            # col_band returns (accs, prefetched_or_None);
            # all other schedules return (accs, None) so call sites unify.
            if compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND:
                return compute_tile_fp4_bank_friendly(
                    accs_in, lds_a, lds_b, lds_as, lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback), None
            if compute_schedule_kind == COMPUTE_SCHEDULE_COL_BAND:
                return compute_tile_col_band(
                    accs_in, lds_a, lds_b, lds_as, lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    prefetched_b_left_head=prefetched_b_left_head,
                    emit_prefetch_next=emit_prefetch_next)
            return compute_tile(
                accs_in, lds_a, lds_b, lds_as, lds_bs,
                emit_filler=emit_filler,
                mid_compute_callback=mid_compute_callback), None

        def hot_loop_scheduler_scheduled():
            if compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND:
                hot_loop_scheduler_fp4_bank_friendly()
            elif compute_schedule_kind == COMPUTE_SCHEDULE_COL_BAND:
                hot_loop_scheduler_col_band()
            else:
                hot_loop_scheduler()

        # ── Epilogue (unified via _sub_tiles) ──
        def _get_acc_sub8(accs, acc_idx, vec_base):
            """Extract 8-element sub-vector from accumulator."""
            if ACC_VEC_SIZE == 8:
                return accs[acc_idx]
            indices = [vec_base + i for i in range_constexpr(8)]
            return vector.shuffle(accs[acc_idx], accs[acc_idx], indices)

        def epilogue_prepare_addrs():
            addrs = []
            _bf16_out = out_dtype in ("bf16", "f16")
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                col_base = (blk_n + warp_n_base + arith.index(wn * WMMA_N)
                            + lane_kgrp * arith.index(8))
                if _bf16_out:
                    c_off_bytes = (row * n_stride + col_base) \
                        * arith.index(elem_bytes_d)
                    addrs.append(c_off_bytes)
                else:
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        c_off = row * n_stride + col
                        addrs.append(c_off)
            return addrs

        _bf16_out = out_dtype in ("bf16", "f16")
        _out_elem_local = T.bf16 if out_dtype == "bf16" else \
            (T.f16 if out_dtype == "f16" else None)

        def epilogue_stores(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                if _bf16_out:
                    addr_idx += store_acc_vec8_to_buffer(
                        sub8, c_rsrc, addrs[addr_idx],
                        out_elem=_out_elem_local, offset_is_bytes=True)
                else:
                    addr_idx += store_acc_vec8_to_buffer(
                        sub8, c_rsrc, addrs[addr_idx:addr_idx + 2])

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                imm = m_off * _lds_d_stride_elems + wn * _n_col_d_elems
                store_acc_vec8_to_lds(d_buf, d_base, imm, sub8,
                                      out_elem=_out_elem_local)

        def _atomic_add_acc_vec8_to_buffer(acc_vec8, addr):
            if _bf16_out:
                h_vec = arith.trunc_f(T.vec(8, _out_elem_local), acc_vec8)
                pair_ty = T.vec(2, _out_elem_local)
                for pair in range_constexpr(4):
                    e0 = vector.extract(
                        h_vec, static_position=[pair * 2], dynamic_position=[])
                    e1 = vector.extract(
                        h_vec, static_position=[pair * 2 + 1], dynamic_position=[])
                    pair_vec = vector.from_elements(pair_ty, [e0, e1])
                    byte_off = arith.index_cast(T.i32, addr + arith.index(pair * 4))
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        pair_vec, c_rsrc, byte_off, zero_i32, zero_i32)
                return 1

            for half in range_constexpr(2):
                base_addr = addr[half] if isinstance(addr, (list, tuple)) else addr
                for vi in range_constexpr(4):
                    val = vector.extract(
                        acc_vec8,
                        static_position=[half * 4 + vi],
                        dynamic_position=[],
                    )
                    byte_off = arith.index_cast(
                        T.i32, (base_addr + arith.index(vi)) * arith.index(4))
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val, c_rsrc, byte_off, zero_i32, zero_i32)
            return 2

        def epilogue_atomic_adds(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                if _bf16_out:
                    addr_idx += _atomic_add_acc_vec8_to_buffer(
                        sub8, addrs[addr_idx])
                else:
                    addr_idx += _atomic_add_acc_vec8_to_buffer(
                        sub8, addrs[addr_idx:addr_idx + 2])

        def grouped_accs_to_row_major(accs_grouped):
            row_major = [None] * n_accs
            for group_idx in range_constexpr(n_accs):
                row_major[_bank_group_to_row_major[group_idx]] = accs_grouped[group_idx]
            return row_major

        def finalize_acc_layout(accs_in):
            if use_col_band_schedule:
                return grouped_accs_to_row_major(accs_in)
            return accs_in

        _effective_l2_pf = l2_prefetch_distance
        if use_cluster and l2_prefetch_distance > 0:
            _effective_l2_pf = max(1, l2_prefetch_distance - 1)

        def _l2_prefetch(k_base):
            if _effective_l2_pf <= 0:
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            pf_k_packed_a = pf_k / arith.index(PACK_FACTOR_A)
            pf_k_packed_b = pf_k / arith.index(PACK_FACTOR_B)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, pf_k_packed_a),
                (tile_m, packed_tile_k_a), (K_packed_a, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b,
                (blk_n / arith.index(16), pf_k_packed_b * arith.index(16)),
                (tile_n // 16, packed_tile_k_b * 16),
                (K_packed_b * 16, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        accs = [acc_zero] * n_accs

        lds_a_data_f16 = lds_a_data_bytes // 2
        lds_b_data_f16 = lds_b_data_bytes // 2
        lds_a_scale_f16 = lds_a_scale_bytes // 2
        lds_b_scale_f16 = lds_b_scale_bytes // 2

        arena_base_ptr = arena_alloc.get_base()

        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_data_off[i], elem_ty_lds,
                    shape=(lds_a_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_data_off[i], elem_ty_lds,
                    shape=(lds_b_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_as = [
            SmemPtr(arena_base_ptr, stage_a_scale_off[i], elem_ty_lds,
                    shape=(lds_a_scale_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_bs = [
            SmemPtr(arena_base_ptr, stage_b_scale_off[i], elem_ty_lds,
                    shape=(lds_b_scale_f16,))
            for i in range_constexpr(num_buffers)
        ]

        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_as_mem = [stages_as[i].get() for i in range_constexpr(num_buffers)]
        stages_bs_mem = [stages_bs[i].get() for i in range_constexpr(num_buffers)]

        stages_a_idx = [extract_lds_base_idx(stages_a[i])
                        for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i])
                        for i in range_constexpr(num_buffers)]
        stages_as_idx = [extract_lds_base_idx(stages_as[i])
                         for i in range_constexpr(num_buffers)]
        stages_bs_idx = [extract_lds_base_idx(stages_bs[i])
                         for i in range_constexpr(num_buffers)]

        if split_data_tdm_scale_async:
            a_scale_base_idx = buffer_ops.extract_base_index(
                arg_a_scale, address_space=1)
            b_scale_base_idx = buffer_ops.extract_base_index(
                arg_b_scale, address_space=1)
            a_scale_panel_cols = interleaved_scale_cols_a
            b_scale_panel_cols = interleaved_scale_cols_b
            a_scale_global_rows = M // as_tdm_group_m
            b_scale_global_rows = N // bs_tdm_group_n
            a_scale_k_tile_bytes = a_scale_global_rows * a_scale_panel_cols
            b_scale_k_tile_bytes = b_scale_global_rows * b_scale_panel_cols
            a_scale_tile_bytes = tile_m * scale_k_per_tile
            b_scale_tile_bytes = tile_n * scale_k_per_tile
            a_scale_tile_chunks = a_scale_tile_bytes // 16
            b_scale_tile_chunks = b_scale_tile_bytes // 16
            scale_tile_chunks = a_scale_tile_chunks + b_scale_tile_chunks
            scale_bulk_copy_iters = scale_tile_chunks // block_threads
            scale_side_copy_iters = (
                scale_tile_chunks + block_threads - 1) // block_threads
            scale_async_loads_per_stage = scale_side_copy_iters
            scale_async_buffered_outstanding = (
                scale_async_loads_per_stage * (pre_loaded - 1))

            def _copy_scale_chunk(scale_base_idx, lds_memref, global_row_base,
                                  global_k_tile_base, panel_cols, chunk):
                from flydsl._mlir.dialects import memref as memref_dialect

                local_byte_off = chunk * arith.index(16)
                global_byte_off = (
                    global_k_tile_base
                    + global_row_base * arith.index(panel_cols)
                    + local_byte_off)
                global_ptr = buffer_ops.create_llvm_ptr(
                    scale_base_idx + global_byte_off, address_space=1)
                lds_base_idx = memref_dialect.extract_aligned_pointer_as_index(
                    lds_memref)
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_base_idx + local_byte_off, address_space=3)
                rocdl.global_load_async_to_lds_b128(
                    global_ptr, lds_ptr, 0, 0)

            def issue_scale_async(stage_idx: int, k_base):
                from flydsl._mlir.dialects import memref as memref_dialect

                k_tile_idx = k_base / arith.index(tile_k)
                a_global_row_base = blk_m / arith.index(as_tdm_group_m)
                b_global_row_base = blk_n / arith.index(bs_tdm_group_n)
                a_global_k_tile_base = (
                    k_tile_idx * arith.index(a_scale_k_tile_bytes))
                b_global_k_tile_base = (
                    k_tile_idx * arith.index(b_scale_k_tile_bytes))
                a_global_panel_base_idx = (
                    a_scale_base_idx
                    + a_global_k_tile_base
                    + a_global_row_base * arith.index(a_scale_panel_cols))
                b_global_panel_base_idx = (
                    b_scale_base_idx
                    + b_global_k_tile_base
                    + b_global_row_base * arith.index(b_scale_panel_cols))
                a_lds_base_idx = (
                    memref_dialect.extract_aligned_pointer_as_index(
                        stages_as_mem[stage_idx]))
                b_lds_base_idx = (
                    memref_dialect.extract_aligned_pointer_as_index(
                        stages_bs_mem[stage_idx]))

                def _issue_when(pred, body):
                    if_op = scf.IfOp(pred)
                    with ir.InsertionPoint(if_op.then_block):
                        body()
                        scf.YieldOp([])

                def _copy_scale_chunk_selected(linear_chunk):
                    is_b = arith.cmpi(
                        arith.CmpIPredicate.uge, linear_chunk,
                        arith.index(a_scale_tile_chunks))
                    b_chunk = linear_chunk - arith.index(a_scale_tile_chunks)
                    local_chunk = arith.select(is_b, b_chunk, linear_chunk)
                    local_byte_off = local_chunk * arith.index(16)

                    global_panel_base_idx = arith.select(
                        is_b, b_global_panel_base_idx,
                        a_global_panel_base_idx)
                    lds_base_idx = arith.select(
                        is_b, b_lds_base_idx, a_lds_base_idx)

                    global_ptr = buffer_ops.create_llvm_ptr(
                        global_panel_base_idx + local_byte_off,
                        address_space=1)
                    lds_ptr = buffer_ops.create_llvm_ptr(
                        lds_base_idx + local_byte_off, address_space=3)
                    rocdl.global_load_async_to_lds_b128(
                        global_ptr, lds_ptr, 0, 0)

                for it in range_constexpr(scale_bulk_copy_iters):
                    linear_chunk = tx + arith.index(it * block_threads)
                    _copy_scale_chunk_selected(linear_chunk)

                for tail_it in range_constexpr(
                        scale_side_copy_iters - scale_bulk_copy_iters):
                    it = scale_bulk_copy_iters + tail_it
                    linear_chunk = tx + arith.index(it * block_threads)
                    in_a = arith.cmpi(
                        arith.CmpIPredicate.ult, linear_chunk,
                        arith.index(a_scale_tile_chunks))
                    b_lower = arith.cmpi(
                        arith.CmpIPredicate.uge, linear_chunk,
                        arith.index(a_scale_tile_chunks))
                    b_upper = arith.cmpi(
                        arith.CmpIPredicate.ult, linear_chunk,
                        arith.index(scale_tile_chunks))
                    in_b = arith.andi(b_lower, b_upper)

                    def _copy_a():
                        _copy_scale_chunk(
                            a_scale_base_idx, stages_as_mem[stage_idx],
                            a_global_row_base, a_global_k_tile_base,
                            a_scale_panel_cols, linear_chunk)

                    def _copy_b():
                        b_chunk = linear_chunk - arith.index(a_scale_tile_chunks)
                        _copy_scale_chunk(
                            b_scale_base_idx, stages_bs_mem[stage_idx],
                            b_global_row_base, b_global_k_tile_base,
                            b_scale_panel_cols, b_chunk)

                    _issue_when(in_a, _copy_a)
                    _issue_when(in_b, _copy_b)
        else:
            def issue_scale_async(stage_idx: int, k_base):
                return
            scale_async_buffered_outstanding = 0

        if use_tdm_store:
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(d_lds_base_ptr, d_output_off, elem_ty_lds,
                             shape=(d_lds_f16_count,))
            d_lds_buffer = get_lds_memref(d_smem)
            warp_lds_off = (wave_m_idx * arith.index(n_warp) + wave_n_idx) \
                * arith.index(_warp_d_elems)
            d_lane_base = (warp_lds_off
                           + lane16 * arith.index(_lds_d_stride_elems)
                           + lane_kgrp * arith.index(4 * elem_bytes_d))
            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr = wave_id_idx * arith.index(warp_d_bytes) \
                + arith.index(d_output_off)
            warp_m_off_sgpr = (wave_id_idx / arith.index(n_warp)) \
                * arith.index(warp_tile_m)
            warp_n_off_sgpr = (wave_id_idx % arith.index(n_warp)) \
                * arith.index(warp_tile_n)
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_c,
                lds_memref=d_lds_base_ptr,
                global_offset=(blk_m + warp_m_off_sgpr,
                               blk_n + warp_n_off_sgpr),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(N, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
            )

        # Precompute LDS addresses for TDM descriptor switching
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        stages_as_lds_addr = []
        stages_bs_lds_addr = []
        stages_a1_lds_addr = []
        stages_b1_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(vector.extract(
                (make_desc_a_half(stages_a_mem[i], arith.index(0), 0)
                 if split_data_tdm_scale_async
                 else make_desc_a(stages_a_mem[i], arith.index(0))).dgroup0,
                static_position=[1], dynamic_position=[]))
            stages_b_lds_addr.append(vector.extract(
                (make_desc_b_half(stages_b_mem[i], arith.index(0), 0)
                 if split_data_tdm_scale_async
                 else make_desc_b(stages_b_mem[i], arith.index(0))).dgroup0,
                static_position=[1], dynamic_position=[]))
            if split_data_tdm_scale_async:
                stages_a1_lds_addr.append(vector.extract(
                    make_desc_a_half(stages_a_mem[i], arith.index(0), 1).dgroup0,
                    static_position=[1], dynamic_position=[]))
                stages_b1_lds_addr.append(vector.extract(
                    make_desc_b_half(stages_b_mem[i], arith.index(0), 1).dgroup0,
                    static_position=[1], dynamic_position=[]))
            else:
                stages_as_lds_addr.append(vector.extract(
                    make_desc_as(stages_as_mem[i], arith.index(0)).dgroup0,
                    static_position=[1], dynamic_position=[]))
                stages_bs_lds_addr.append(vector.extract(
                    make_desc_bs(stages_bs_mem[i], arith.index(0)).dgroup0,
                    static_position=[1], dynamic_position=[]))

        desc_a_init = (
            make_desc_a_half(stages_a_mem[0], split_k_base, 0)
            if split_data_tdm_scale_async
            else make_desc_a(stages_a_mem[0], split_k_base))
        desc_b_init = (
            make_desc_b_half(stages_b_mem[0], split_k_base, 0)
            if split_data_tdm_scale_async
            else make_desc_b(stages_b_mem[0], split_k_base))
        desc_as_init = (
            make_desc_a_half(stages_a_mem[0], split_k_base, 1)
            if split_data_tdm_scale_async
            else make_desc_as(stages_as_mem[0], split_k_base))
        desc_bs_init = (
            make_desc_b_half(stages_b_mem[0], split_k_base, 1)
            if split_data_tdm_scale_async
            else make_desc_bs(stages_bs_mem[0], split_k_base))

        adv_a_i32 = arith.constant(tile_k // PACK_FACTOR_A, type=T.i32)
        adv_b_i32 = arith.constant(packed_tile_k_b * 16, type=T.i32)
        adv_as_i32 = arith.constant(tile_k // SCALE_BLOCK * as_tdm_group_m, type=T.i32)
        adv_bs_i32 = arith.constant(tile_k // SCALE_BLOCK * bs_tdm_group_n, type=T.i32)

        pred_const = arith.constant(1, type=T.i32)

        if wave_specialized_tdm:
            active_stage_lds_addr = [
                _select_wave_tdm_value(
                    stages_a_lds_addr[i],
                    stages_b_lds_addr[i],
                    (stages_a1_lds_addr[i] if split_data_tdm_scale_async
                     else stages_as_lds_addr[i]),
                    (stages_b1_lds_addr[i] if split_data_tdm_scale_async
                     else stages_bs_lds_addr[i]),
                )
                for i in range_constexpr(num_buffers)
            ]
            active_addr_lo = _select_wave_tdm_value(
                vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[]),
                vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[]),
                vector.extract(desc_as_init.dgroup0, static_position=[2], dynamic_position=[]),
                vector.extract(desc_bs_init.dgroup0, static_position=[2], dynamic_position=[]),
            )
            active_addr_hi = _select_wave_tdm_value(
                vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[]),
                vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[]),
                vector.extract(desc_as_init.dgroup0, static_position=[3], dynamic_position=[]),
                vector.extract(desc_bs_init.dgroup0, static_position=[3], dynamic_position=[]),
            )
            active_dgroup1 = _select_wave_tdm_value(
                desc_a_init.dgroup1,
                desc_b_init.dgroup1,
                desc_as_init.dgroup1,
                desc_bs_init.dgroup1,
            )
            active_adv_i32 = (
                _select_wave_tdm_value(
                    adv_a_i32, adv_b_i32, adv_a_i32, adv_b_i32)
                if split_data_tdm_scale_async
                else _select_wave_tdm_value(
                    adv_a_i32, adv_b_i32, adv_as_i32, adv_bs_i32)
            )
        else:
            addr_lo_a = vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[])
            addr_hi_a = vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[])
            addr_lo_b = vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[])
            addr_hi_b = vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[])
            addr_lo_as = vector.extract(desc_as_init.dgroup0, static_position=[2], dynamic_position=[])
            addr_hi_as = vector.extract(desc_as_init.dgroup0, static_position=[3], dynamic_position=[])
            addr_lo_bs = vector.extract(desc_bs_init.dgroup0, static_position=[2], dynamic_position=[])
            addr_hi_bs = vector.extract(desc_bs_init.dgroup0, static_position=[3], dynamic_position=[])

            dgroup1_a = desc_a_init.dgroup1
            dgroup1_b = desc_b_init.dgroup1
            dgroup1_as = desc_as_init.dgroup1
            dgroup1_bs = desc_bs_init.dgroup1

        # Initialize named barriers ONCE before any TDM/sync activity.
        # Inits the memberCnt (wave 0 only, followed by a WG barrier) AND
        # JOINs every wave to the barrier so later s_barrier_wait actually
        # blocks — see :func:`init_named_barriers.
        if nbar_alloc is not None:
            init_named_barriers([nbar_sync])

        def _pipeline_fence(outstanding=0, async_outstanding=0):
            if split_data_tdm_scale_async:
                tdm_ops.tensor_wait(outstanding)
                rocdl.s_wait_loadcnt(0)
                rocdl.s_wait_dscnt(0)
                rocdl.s_wait_asynccnt(async_outstanding)
                gpu.barrier()
            else:
                pipeline_fence(outstanding=outstanding, use_cluster=use_cluster)

        def _pipeline_fence_signal(outstanding=0, async_outstanding=0):
            if split_data_tdm_scale_async:
                tdm_ops.tensor_wait(outstanding)
                rocdl.s_wait_loadcnt(0)
                rocdl.s_wait_dscnt(0)
                rocdl.s_wait_asynccnt(async_outstanding)
                rocdl.s_barrier_signal(-1)
            else:
                pipeline_fence_signal(
                    outstanding=outstanding, use_cluster=use_cluster)

        def _pipeline_fence_wait():
            if split_data_tdm_scale_async:
                rocdl.s_barrier_wait(-1)
            else:
                pipeline_fence_wait(use_cluster=use_cluster)

        # Prologue
        if wave_specialized_tdm:
            for i in range_constexpr(pre_loaded):
                if split_data_tdm_scale_async:
                    issue_scale_async(
                        i, split_k_base + arith.index(i * tile_k))
                dg0 = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, active_stage_lds_addr[i],
                    active_addr_lo, active_addr_hi])
                tdm_ops.tensor_load_2d(
                    tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                active_addr_lo = arith.addi(active_addr_lo, active_adv_i32)
        else:
            for i in range_constexpr(pre_loaded):
                dg0_a = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_a_lds_addr[i], addr_lo_a, addr_hi_a])
                dg0_b = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_b_lds_addr[i], addr_lo_b, addr_hi_b])
                dg0_as = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_as_lds_addr[i], addr_lo_as, addr_hi_as])
                dg0_bs = vector.from_elements(T.vec(4, T.i32), [
                    pred_const, stages_bs_lds_addr[i], addr_lo_bs, addr_hi_bs])

                issue_tdm_loads(
                    tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                    tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                    tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                    tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                    wave_specialized=wave_specialized_tdm)

                addr_lo_a = arith.addi(addr_lo_a, adv_a_i32)
                addr_lo_b = arith.addi(addr_lo_b, adv_b_i32)
                addr_lo_as = arith.addi(addr_lo_as, adv_as_i32)
                addr_lo_bs = arith.addi(addr_lo_bs, adv_bs_i32)

        _pipeline_fence(
            outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2),
            async_outstanding=scale_async_buffered_outstanding)

        # Main loop — acc_mixed style: fence at top, TDM_load mid-compute.
        # This overlaps TDM DMA with the remaining WMMA instructions,
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

        tail_leftover_prefetched = None

        if loop_iters > 0:
            if wave_specialized_tdm:
                if _col_band_prefetch_enabled:
                    rocdl.sched_barrier(0)
                    first_prefetched = _prefetch_b_left_head_for_stage(
                        stages_b_idx[0])
                    init_args = list(accs) + [active_addr_lo] + first_prefetched
                else:
                    init_args = list(accs) + [active_addr_lo]

                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_addr_lo = state[n_accs]
                    if _col_band_prefetch_enabled:
                        prefetched = list(
                            state[n_accs + 1
                                  : n_accs + 1 + _col_band_prefetch_cols])

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers
                        next_buf_idx = (buf_idx + 1) % num_buffers
                        scale_load_k = (
                            split_k_base
                            + loop_iter * arith.index(num_buffers * tile_k)
                            + arith.index((buf_idx + pre_loaded) * tile_k)
                        )

                        addr_box = [cur_addr_lo]

                        def _mid_tdm_ws(
                            _ls=load_stage,
                            _ab=addr_box,
                            _k_off=(split_k_base
                                    + loop_iter * arith.index(num_buffers * tile_k)
                                    + arith.index(buf_idx * tile_k)),
                        ):
                            dg0 = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, active_stage_lds_addr[_ls],
                                _ab[0], active_addr_hi])
                            tdm_ops.tensor_load_2d(
                                tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                            _ab[0] = arith.addi(_ab[0], active_adv_i32)
                            _l2_prefetch(_k_off)

                        rocdl.sched_barrier(0)
                        if split_data_tdm_scale_async:
                            issue_scale_async(load_stage, scale_load_k)
                        if _col_band_prefetch_enabled:
                            def _prefetch_next(_nb=next_buf_idx):
                                return _prefetch_b_left_head_for_stage(
                                    stages_b_idx[_nb])
                            accs_in, prefetched = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                mid_compute_callback=_mid_tdm_ws,
                                prefetched_b_left_head=prefetched,
                                emit_prefetch_next=_prefetch_next)
                        else:
                            accs_in, _ = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                mid_compute_callback=_mid_tdm_ws)
                        cur_addr_lo = addr_box[0]
                        hot_loop_scheduler_scheduled()

                        if nbar_alloc is not None:
                            _nbars = [nbar_sync]
                            pipeline_fence_signal_named_multi(
                                _nbars,
                                outstanding=_fence_outstanding,
                                use_cluster=use_cluster)
                            pipeline_fence_wait_named_multi(
                                _nbars, use_cluster=use_cluster)
                        else:
                            _pipeline_fence_signal(
                                outstanding=_fence_outstanding,
                                async_outstanding=scale_async_buffered_outstanding)
                            _pipeline_fence_wait()

                    if _col_band_prefetch_enabled:
                        _yield_values = (
                            list(accs_in) + [cur_addr_lo] + prefetched)
                    else:
                        _yield_values = list(accs_in) + [cur_addr_lo]
                    results = yield _yield_values

                accs = list(results[:n_accs])
                active_addr_lo = results[n_accs]
                if _col_band_prefetch_enabled:
                    tail_leftover_prefetched = list(
                        results[n_accs + 1
                                : n_accs + 1 + _col_band_prefetch_cols])
            else:
                if _col_band_prefetch_enabled:
                    rocdl.sched_barrier(0)
                    first_prefetched = _prefetch_b_left_head_for_stage(
                        stages_b_idx[0])
                    init_args = list(accs) + [
                        addr_lo_a, addr_lo_b, addr_lo_as, addr_lo_bs
                    ] + first_prefetched
                else:
                    init_args = list(accs) + [addr_lo_a, addr_lo_b, addr_lo_as, addr_lo_bs]

                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_lo_a = state[n_accs]
                    cur_lo_b = state[n_accs + 1]
                    cur_lo_as = state[n_accs + 2]
                    cur_lo_bs = state[n_accs + 3]
                    if _col_band_prefetch_enabled:
                        prefetched = list(
                            state[n_accs + 4
                                  : n_accs + 4 + _col_band_prefetch_cols])

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers
                        next_buf_idx = (buf_idx + 1) % num_buffers

                        addr_boxes = [[cur_lo_a], [cur_lo_b],
                                      [cur_lo_as], [cur_lo_bs]]

                        def _mid_tdm_nws(
                            _ls=load_stage,
                            _ab=addr_boxes,
                            _k_off=(split_k_base
                                    + loop_iter * arith.index(num_buffers * tile_k)
                                    + arith.index(buf_idx * tile_k)),
                        ):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0][0], addr_hi_a])
                            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_b_lds_addr[_ls],
                                _ab[1][0], addr_hi_b])
                            dg0_as = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_as_lds_addr[_ls],
                                _ab[2][0], addr_hi_as])
                            dg0_bs = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_bs_lds_addr[_ls],
                                _ab[3][0], addr_hi_bs])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                                tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                                wave_specialized=wave_specialized_tdm)
                            _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                            _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)
                            _ab[2][0] = arith.addi(_ab[2][0], adv_as_i32)
                            _ab[3][0] = arith.addi(_ab[3][0], adv_bs_i32)
                            _l2_prefetch(_k_off)

                        rocdl.sched_barrier(0)
                        if _col_band_prefetch_enabled:
                            def _prefetch_next(_nb=next_buf_idx):
                                return _prefetch_b_left_head_for_stage(
                                    stages_b_idx[_nb])
                            accs_in, prefetched = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                mid_compute_callback=_mid_tdm_nws,
                                prefetched_b_left_head=prefetched,
                                emit_prefetch_next=_prefetch_next)
                        else:
                            accs_in, _ = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                mid_compute_callback=_mid_tdm_nws)
                        cur_lo_a = addr_boxes[0][0]
                        cur_lo_b = addr_boxes[1][0]
                        cur_lo_as = addr_boxes[2][0]
                        cur_lo_bs = addr_boxes[3][0]
                        hot_loop_scheduler_scheduled()

                        _pipeline_fence_signal(
                            outstanding=_fence_outstanding)
                        _pipeline_fence_wait()

                    if _col_band_prefetch_enabled:
                        _yield_values = list(accs_in) + [
                            cur_lo_a, cur_lo_b, cur_lo_as, cur_lo_bs
                        ] + prefetched
                    else:
                        _yield_values = list(accs_in) + [
                            cur_lo_a, cur_lo_b, cur_lo_as, cur_lo_bs]
                    results = yield _yield_values

                accs = list(results[:n_accs])
                addr_lo_a = results[n_accs]
                addr_lo_b = results[n_accs + 1]
                addr_lo_as = results[n_accs + 2]
                addr_lo_bs = results[n_accs + 3]
                if _col_band_prefetch_enabled:
                    tail_leftover_prefetched = list(
                        results[n_accs + 4
                                : n_accs + 4 + _col_band_prefetch_cols])

        # Tail — same acc_mixed pattern: fence at top, TDM mid-compute.
        if loop_iters > 0:
            _pipeline_fence(outstanding=0)
        elif use_cluster:
            gpu.cluster_barrier()
        epi_addrs_box = [None]
        _tail_had_load = False
        _tail_prefetch_box = [tail_leftover_prefetched]
        _tail_scale_k_box = [
            split_k_base + arith.index((_tail_start + pre_loaded) * tile_k)
        ]

        def _consume_tail_prefetched():
            v = _tail_prefetch_box[0]
            _tail_prefetch_box[0] = None
            return v

        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if _outstanding == -1:
                if _tail_had_load:
                    _pipeline_fence(outstanding=0)
                if use_tdm_store:
                    accs, _ = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage],
                        prefetched_b_left_head=_consume_tail_prefetched())
                else:
                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    accs, _ = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage],
                        emit_filler=_emit_epi_addrs,
                        prefetched_b_left_head=_consume_tail_prefetched())
            else:
                _pipeline_fence_signal(outstanding=_outstanding)
                _pipeline_fence_wait()

                _tail_mid_cb = None
                if _load_stage is not None:
                    _tail_had_load = True
                    if wave_specialized_tdm:
                        _tail_addr_box = [active_addr_lo]

                        def _tail_mid_ws(
                            _ls=_load_stage,
                            _ab=_tail_addr_box,
                        ):
                            dg0 = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, active_stage_lds_addr[_ls],
                                _ab[0], active_addr_hi])
                            tdm_ops.tensor_load_2d(
                                tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                            _ab[0] = arith.addi(_ab[0], active_adv_i32)

                        _tail_mid_cb = _tail_mid_ws
                    else:
                        _tail_ab = [[addr_lo_a], [addr_lo_b],
                                    [addr_lo_as], [addr_lo_bs]]

                        def _tail_mid_nws(_ls=_load_stage, _ab=_tail_ab):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0][0], addr_hi_a])
                            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_b_lds_addr[_ls],
                                _ab[1][0], addr_hi_b])
                            dg0_as = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_as_lds_addr[_ls],
                                _ab[2][0], addr_hi_as])
                            dg0_bs = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_bs_lds_addr[_ls],
                                _ab[3][0], addr_hi_bs])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                                tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                                wave_specialized=wave_specialized_tdm)
                            _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                            _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)
                            _ab[2][0] = arith.addi(_ab[2][0], adv_as_i32)
                            _ab[3][0] = arith.addi(_ab[3][0], adv_bs_i32)

                        _tail_mid_cb = _tail_mid_nws

                rocdl.sched_barrier(0)
                if (
                    split_data_tdm_scale_async
                    and wave_specialized_tdm
                    and _load_stage is not None
                ):
                    issue_scale_async(_load_stage, _tail_scale_k_box[0])
                    _tail_scale_k_box[0] = (
                        _tail_scale_k_box[0] + arith.index(tile_k))
                accs, _ = compute_tile_scheduled(
                    accs,
                    stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                    stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage],
                    mid_compute_callback=_tail_mid_cb,
                    prefetched_b_left_head=_consume_tail_prefetched())

                if _load_stage is not None:
                    if wave_specialized_tdm:
                        active_addr_lo = _tail_addr_box[0]
                    else:
                        addr_lo_a = _tail_ab[0][0]
                        addr_lo_b = _tail_ab[1][0]
                        addr_lo_as = _tail_ab[2][0]
                        addr_lo_bs = _tail_ab[3][0]

                hot_loop_scheduler_scheduled()

        accs = finalize_acc_layout(accs)

        if use_tdm_store:
            if d_need_epilogue_fence:
                _pipeline_fence(outstanding=0)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            rocdl.sched_barrier(0)
            if epi_addrs_box[0] is None:
                epi_addrs_box[0] = epilogue_prepare_addrs()
            if split_k > 1:
                epilogue_atomic_adds(accs, epi_addrs_box[0])
            else:
                epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (data_format, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, compute_schedule_kind,
                 effective_waves_per_eu, l2_prefetch_distance,
                 cluster_m, cluster_n, use_tdm_store,
                 out_dtype, inst_prefetch, wave_specialized_tdm, split_k,
                 expert_sched_mode, atomic_barrier_enable)

    @flyc.jit
    def launch_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()
            if nbar_alloc is not None:
                nbar_alloc.finalized = False
                nbar_alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))
        gz = split_k

        launcher = kernel_mxscale_gemm(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, i32_m, i32_n)
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                if effective_waves_per_eu is not None:
                    _wpe = int(effective_waves_per_eu)
                    if _wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
                if use_cluster:
                    op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                        f"{cluster_m},{cluster_n},1")
        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        launcher.launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    if expert_sched_mode:
        launch_mxscale_gemm.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_mxscale_gemm


compile_mxfp4_gemm = lambda **kw: compile_mxscale_gemm(data_format="fp4", **kw)
compile_mxfp8_gemm = lambda **kw: compile_mxscale_gemm(data_format="fp8", **kw)
compile_a8w4_gemm = lambda **kw: compile_mxscale_gemm(data_format="a8w4", **kw)

__all__ = ["compile_mxscale_gemm", "compile_mxfp4_gemm", "compile_mxfp8_gemm",
           "compile_a8w4_gemm"]
