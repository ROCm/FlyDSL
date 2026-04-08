"""TDM async copy WMMA GEMM kernel for gfx1250.

Supports double-buffer (2-stage) and triple-buffer (3-stage) pipelining
with TDM (Tensor Data Mover) hardware async copy for both A and B tiles.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity

from flydsl.expr import idx2crd
from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx, get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw, lds_transpose_load_raw,
    pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
    store_acc_vec8_to_buffer, store_acc_vec8_to_lds,
)
from kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32
DS_LOADS_PER_A_FRAG = 2
DS_LOADS_PER_B_FRAG = 2

LDS_PAD_A = 8
LDS_PAD_B = 8
LDS_PAD_D_BYTES = 16

_make_tail_plan = make_tail_plan


def compile_wmma_gemm_tdm(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    in_dtype: str = "fp16",
    out_dtype: str = None,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    use_tdm_store: bool = True,
    cluster_m: int = 1,
    cluster_n: int = 1,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    expert_sched_mode: bool = True,
):
    """Compile a WMMA GEMM kernel with TDM async copy and multi-stage buffering.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, M, N, stream)

    Args:
        out_dtype: Output element type ("f16", "bf16", "f32").
                   Default (None) = matches input type.
        num_buffers: Number of LDS buffers (2=double, 3=triple, 4=quad buffering).
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
        l2_prefetch_distance: Number of k-tiles ahead to prefetch into L2.
                              0 = disabled, 2 = typical value.
        use_tdm_store: Use TDM store epilogue via LDS (True) or buffer_store (False).
        cluster_m: Cluster dimension along M (WG rows per cluster, 1=disabled).
        cluster_n: Cluster dimension along N (WG cols per cluster, 1=disabled).
        inst_prefetch: Enable instruction prefetch via s_set_inst_prefetch_distance.
        wave_specialized_tdm: Each wave handles one TDM descriptor direction
                              (wave 0 → A, wave 1 → B, others compute-only).
        expert_sched_mode: Enable AMDGPU expert scheduling mode.
    """
    _ = (M, N)
    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3 or 4, got {num_buffers}")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")
    is_f16 = in_dtype == "fp16"
    if out_dtype is None:
        out_dtype = "f16" if is_f16 else "bf16"
    if out_dtype not in ("f32", "f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f32', 'f16', or 'bf16', got {out_dtype!r}")
    elem_bytes = 2
    elem_bytes_d = 2 if out_dtype in ("f16", "bf16") else 4

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}={cluster_m * cluster_n}")
        if cluster_m < 1 or cluster_n < 1:
            raise ValueError(f"cluster dims must be >= 1, got ({cluster_m}, {cluster_n})")
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        # Cluster mode can deadlock if a workgroup is split and only a subset
        # of its waves are resident while hitting early workgroup barriers.
        # Use conservative occupancy by default for cluster-enabled kernels.
        effective_waves_per_eu = 1

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE

    if wave_specialized_tdm and num_warps < 2:
        raise ValueError(
            f"wave_specialized_tdm requires at least 2 waves, got {num_warps}")

    TDM_LOADS_PER_STEP = 1 if wave_specialized_tdm else 2

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2 for TDM async copy, got {tile_k}")

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles} (K={K}, tile_k={tile_k})")

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16
    k_wmma_steps = tile_k // WMMA_K

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep

    lds_a_stride = tile_k + LDS_PAD_A
    lds_b_stride = tile_n + LDS_PAD_B
    lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    lds_b_elems = tile_k * lds_b_stride + LDS_PAD_B

    # --- LDS allocation (B-first: B at offset 0 for smaller ds_load offsets) ---
    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    # Keep per-stage LDS layout unchanged; only remap logical stages to
    # physical stage bases inside one arena to enable safe epilogue aliasing.
    stage_layout = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_layout")
    stage_b_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_rel_off + lds_b_elems * elem_bytes
    stage_a_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_rel_off + lds_a_elems * elem_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    # Compile-time pipeline parameters
    pre_loaded = num_buffers - 1        # stages pre-loaded in prologue
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers  # index of first un-computed tile in tail
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = _make_tail_plan(num_buffers, pre_loaded, extra)
    _last_compute_stage = _base_tail_plan[-1][1]
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"wmma_tdm_{in_dtype}_{out_dtype}_{tile_m}x{tile_n}x{tile_k}_"
            f"{m_warp}x{n_warp}_{num_buffers}buf_arena"),
    )
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

    stage_b_offsets = [
        stage_base_off[i] + stage_b_rel_off for i in range(num_buffers)
    ]
    stage_a_offsets = [
        stage_base_off[i] + stage_a_rel_off for i in range(num_buffers)
    ]
    if use_tdm_store:
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        # Element-based versions (f16 = 2 bytes) for vector LDS store path
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        d_need_epilogue_fence = total_d_bytes > epilogue_fence_threshold_bytes
        if total_d_bytes > arena_total_bytes:
            arena_total_bytes = total_d_bytes
            arena_alloc.ptr = total_d_bytes
    check_smem_capacity(arena_total_bytes, gpu_arch)

    @flyc.kernel
    def kernel_wmma_gemm_tdm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        # --- Cluster MCAST setup ---
        if use_cluster:
            local_x, local_y = gpu.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                local_x, local_y, cluster_m, cluster_n)
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        # --- Thread/wave decomposition ---
        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3))

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        elem_ty = _elem_type()

        # --- Epilogue setup ---
        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_stride = arith.index(N)
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        # --- TDM async copy helpers (MCAST-aware) ---
        def make_desc_a(lds_a_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=lds_a_mem_ref,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k), strides=(K, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_A,
                num_warps=num_warps,
                workgroup_mask=a_mcast_mask)

        def make_desc_b(lds_b_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=lds_b_mem_ref,
                global_offset=(k_base, blk_n),
                tensor_shape=(tile_k, tile_n), strides=(N, 1),
                tile_shape=(tile_k, tile_n), elem_bytes=elem_bytes,
                pad_interval=tile_n, pad_amount=LDS_PAD_B,
                num_warps=num_warps,
                workgroup_mask=b_mcast_mask)

        # --- LDS load helpers ---
        def _precompute_a_lane_bases(lds_base_idx):
            """Precompute per-wm A fragment lane base addresses.

            Returns (lds_buffer, bases) where bases[wm] =
              (warp_m_base + wm*WMMA_M + lane16) * lds_a_stride + lane_kgrp * 8
            """
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride * elem_bytes)
            k_lane_off = lane_kgrp * arith.index(8 * elem_bytes)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = (
                    row_stride_off
                    + arith.index(wm * WMMA_M * lds_a_stride * elem_bytes)
                    + k_lane_off
                )
                bases.append(a_base)
            return lds_base_idx, bases

        def load_wmma_frag(a_lds_base_idx, a_lane_base, ks):
            """Load one 16x32 WMMA fragment from LDS using vectorized 128-bit loads.

            a_lane_base is precomputed by _precompute_a_lane_bases.
            ks is the K-subtile index (compile-time constant).
            """
            vec8_ty = ir.VectorType.get([8], elem_ty)

            k_byte_off = arith.index(ks * WMMA_K * elem_bytes)
            off0 = a_lane_base + k_byte_off
            off1 = a_lane_base + k_byte_off + arith.index(32)

            raw0 = lds_load_b128_raw(a_lds_base_idx, off0)
            raw1 = lds_load_b128_raw(a_lds_base_idx, off1)
            v0 = vector.bitcast(vec8_ty, raw0)
            v1 = vector.bitcast(vec8_ty, raw1)

            return vector.shuffle(v0, v1, list(range(16)))

        def _precompute_b_lane_bases(lds_base_idx):
            """Precompute per-wn B fragment lane base addresses.

            Returns a list of (lds_buffer, b_lane_base) for each wn.
            b_lane_base = (lane_kgrp*8 + lane8) * lds_b_stride
                        + (warp_n_base + wn*WMMA_N + lane_ngrp*8)
            where lane8 = lane16 % 8, lane_ngrp = lane16 / 8.

            After precompute, lane8/lane_ngrp are dead → frees VGPRs.
            """
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (
                (lane_kgrp * arith.index(8) + lane8)
                * arith.index(lds_b_stride * elem_bytes)
            )
            n_lane_off = lane_ngrp * arith.index(8 * elem_bytes)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = (
                    (warp_n_base + arith.index(wn * WMMA_N))
                    * arith.index(elem_bytes)
                    + n_lane_off
                )
                b_base = k_lane_off + n_col
                bases.append(b_base)
            return lds_base_idx, bases

        def load_wmma_frag_tr(lds_base_idx, b_lane_base, ks):
            """Load one 16x32 WMMA B fragment using ds_load_tr16_b128.

            b_lane_base is precomputed by _precompute_b_lane_bases.
            ks is the K-subtile index (compile-time constant from range_constexpr).
            The K offset is folded into a compile-time constant multiplication.
            """
            vec8_ty = ir.VectorType.get([8], elem_ty)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride * elem_bytes
                elem_off = b_lane_base + arith.index(k_row_off)
                v = lds_transpose_load_raw(vec8_ty, lds_base_idx, elem_off)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        # --- K-subtile compute (A-streaming pipeline) ---
        def _load_b_frags(b_lds_buffer, b_bases, ks):
            """Load all B fragments for one K-subtile (no wait)."""
            return [load_wmma_frag_tr(b_lds_buffer, b_bases[wn], ks)
                    for wn in range_constexpr(wmma_n_rep)]

        use_half_streaming_schedule = (
            (wmma_m_rep % 2) == 0 and wmma_m_rep > 1
        )

        def _emit_wmma_row(accs, wm, a_frag, b_frags):
            for wn in range_constexpr(wmma_n_rep):
                idx = wm * wmma_n_rep + wn
                accs[idx] = wmma_op(
                    T.vec(8, T.f32),
                    b_frags[wn], a_frag, accs[idx],
                    signA=False, signB=False, modC=0,
                    reuseA=False, reuseB=False,
                ).result

        def _a_streaming_compute_per_wm(accs, a_buf, a_bases, b_frags, ks,
                                        emit_filler=None,
                                        mid_compute_callback=None,
                                        next_b_info=None):
            """Stream A fragments per-wm group, interleaved with WMMA.

            mid_compute_callback: called mid-compute (after first half of wm
                groups) to issue TDM loads / L2 prefetch overlapped with WMMA.
            """
            next_b_frags = None
            a_frag = load_wmma_frag(a_buf, a_bases[0], ks)
            for wm in range_constexpr(wmma_m_rep):
                is_last = (wm == wmma_m_rep - 1)
                if not is_last:
                    a_next = load_wmma_frag(a_buf, a_bases[wm + 1], ks)
                if is_last:
                    rocdl.s_wait_dscnt(0)
                    if emit_filler is not None:
                        rocdl.sched_barrier(0)
                        emit_filler()
                    if next_b_info is not None:
                        nb_buf, nb_bases, nb_ks = next_b_info
                        next_b_frags = _load_b_frags(nb_buf, nb_bases, nb_ks)
                else:
                    rocdl.s_wait_dscnt(DS_LOADS_PER_A_FRAG)
                _emit_wmma_row(accs, wm, a_frag, b_frags)
                if not is_last:
                    a_frag = a_next

            if mid_compute_callback is not None:
                rocdl.sched_barrier(0)
                mid_compute_callback()

            if next_b_info is not None:
                return accs, next_b_frags
            return accs

        def _a_streaming_compute_half(accs, a_buf, a_bases, b_frags, ks,
                                      emit_filler=None,
                                      mid_compute_callback=None,
                                      next_b_info=None):
            """Half-based A-streaming with mid-compute callback."""
            next_b_frags = None
            half_wm = wmma_m_rep // 2
            half_wait = (half_wm - 1) * DS_LOADS_PER_A_FRAG

            a_frags_h0 = [load_wmma_frag(a_buf, a_bases[wm], ks)
                          for wm in range_constexpr(half_wm)]
            rocdl.s_wait_dscnt(half_wait)

            if mid_compute_callback is not None:
                rocdl.sched_barrier(0)
                mid_compute_callback()

            for wm in range_constexpr(half_wm):
                _emit_wmma_row(accs, wm, a_frags_h0[wm], b_frags)

            a_frags_h1 = [load_wmma_frag(a_buf, a_bases[half_wm + h], ks)
                          for h in range_constexpr(half_wm)]
            rocdl.s_wait_dscnt(half_wait)
            for h in range_constexpr(half_wm):
                wm = half_wm + h
                if wm == wmma_m_rep - 1 and emit_filler is not None:
                    rocdl.sched_barrier(0)
                    emit_filler()
                _emit_wmma_row(accs, wm, a_frags_h1[h], b_frags)

            if next_b_info is not None:
                nb_buf, nb_bases, nb_ks = next_b_info
                next_b_frags = _load_b_frags(nb_buf, nb_bases, nb_ks)
                return accs, next_b_frags
            return accs

        def _a_streaming_compute(accs, a_buf, a_bases, b_frags, ks,
                                 emit_filler=None,
                                 mid_compute_callback=None,
                                 next_b_info=None):
            if use_half_streaming_schedule:
                return _a_streaming_compute_half(
                    accs, a_buf, a_bases, b_frags, ks,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    next_b_info=next_b_info)
            return _a_streaming_compute_per_wm(
                accs, a_buf, a_bases, b_frags, ks,
                emit_filler=emit_filler,
                mid_compute_callback=mid_compute_callback,
                next_b_info=next_b_info)

        # --- Compute on one LDS buffer (A-streaming K-subtile pipeline) ---
        def compute_tile(accs_in, lds_a_idx, lds_b_idx,
                         emit_filler=None, mid_compute_callback=None):
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_idx)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_idx)

            if k_wmma_steps == 1:
                b_frags = _load_b_frags(b_buf, b_bases, 0)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, b_frags, 0,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback)
            else:
                prev_b = _load_b_frags(b_buf, b_bases, 0)
                for ks in range_constexpr(k_wmma_steps - 1):
                    _mid_cb = mid_compute_callback if ks == 0 else None
                    current_accs, prev_b = _a_streaming_compute(
                        current_accs, a_buf, a_bases, prev_b, ks,
                        mid_compute_callback=_mid_cb,
                        next_b_info=(b_buf, b_bases, ks + 1))
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, prev_b,
                    k_wmma_steps - 1,
                    emit_filler=emit_filler)

            return current_accs

        # --- Scheduling ---
        def hot_loop_scheduler():
            if not use_half_streaming_schedule:
                rocdl.sched_barrier(0)
                return

            half_wm = wmma_m_rep // 2
            half_wmma = half_wm * wmma_n_rep
            a_half_loads = half_wm * DS_LOADS_PER_A_FRAG
            b_full_loads = wmma_n_rep * DS_LOADS_PER_B_FRAG

            for ks in range_constexpr(k_wmma_steps):
                if ks == 0:
                    rocdl.sched_dsrd(b_full_loads + a_half_loads)
                else:
                    rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                if ks < k_wmma_steps - 1:
                    rocdl.sched_dsrd(b_full_loads)
            rocdl.sched_barrier(0)

        # --- Epilogue helpers ---
        _half_out = out_dtype in ("f16", "bf16")
        _out_elem = T.f16 if out_dtype == "f16" else (T.bf16 if out_dtype == "bf16" else None)

        def epilogue_prepare_addrs():
            """Precompute all epilogue store addresses (VALU only, no stores)."""
            addrs = []
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = (blk_n + warp_n_base + arith.index(wn * WMMA_N)
                                + lane_kgrp * arith.index(8))
                    if _half_out:
                        c_off_bytes = (row * n_stride + col_base) * arith.index(elem_bytes_d)
                        addrs.append(c_off_bytes)
                    else:
                        for half in range_constexpr(2):
                            col = col_base + arith.index(half * 4)
                            c_off = row * n_stride + col
                            addrs.append(c_off)
            return addrs

        def epilogue_stores(final_accs, addrs):
            """Execute buffer_store using precomputed addresses."""
            addr_idx = 0
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    if _half_out:
                        addr_idx += store_acc_vec8_to_buffer(
                            final_accs[idx], c_rsrc, addrs[addr_idx],
                            out_elem=_out_elem, offset_is_bytes=True)
                    else:
                        addr_idx += store_acc_vec8_to_buffer(
                            final_accs[idx], c_rsrc, addrs[addr_idx:addr_idx + 2])

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            """Write accumulators to D output LDS via lds_store_b128."""
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    imm = wm * WMMA_M * _lds_d_stride_elems + wn * _n_col_d_elems
                    store_acc_vec8_to_lds(
                        d_buf, d_base, imm, final_accs[idx], out_elem=_out_elem)

        _effective_l2_pf = l2_prefetch_distance
        if use_cluster and l2_prefetch_distance > 0:
            _effective_l2_pf = max(1, l2_prefetch_distance - 1)

        def _l2_prefetch(k_base):
            if _effective_l2_pf <= 0:
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, pf_k), (tile_m, tile_k), (K, 1),
                elem_bytes=elem_bytes, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b, (pf_k, blk_n), (tile_k, tile_n), (N, 1),
                elem_bytes=elem_bytes, thread_id=tx, block_threads=block_threads)

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        # Build per-stage SmemPtrs (all stages share one arena base)
        arena_base_ptr = arena_alloc.get_base()
        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_offsets[i], elem_ty, shape=(lds_a_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_offsets[i], elem_ty, shape=(lds_b_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_a_idx = [extract_lds_base_idx(stages_a[i])
                        for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i])
                        for i in range_constexpr(num_buffers)]

        # D output LDS setup for TDM store epilogue
        if use_tdm_store:
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // elem_bytes
            d_smem = SmemPtr(d_lds_base_ptr, d_output_off, elem_ty,
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

        # --- TDM descriptor addr_lo management (FP4-style) ---
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(vector.extract(
                make_desc_a(stages_a_mem[i], arith.index(0)).dgroup0,
                static_position=[1], dynamic_position=[]))
            stages_b_lds_addr.append(vector.extract(
                make_desc_b(stages_b_mem[i], arith.index(0)).dgroup0,
                static_position=[1], dynamic_position=[]))

        desc_a_init = make_desc_a(stages_a_mem[0], arith.index(0))
        desc_b_init = make_desc_b(stages_b_mem[0], arith.index(0))

        adv_a_i32 = arith.constant(tile_k * elem_bytes, type=T.i32)
        adv_b_i32 = arith.constant(tile_k * N * elem_bytes, type=T.i32)
        pred_const = arith.constant(1, type=T.i32)

        if wave_specialized_tdm:
            tdm_wave_id = rocdl.wave_id()
            tdm_wave_is_a = arith.cmpi(
                arith.CmpIPredicate.eq, tdm_wave_id,
                arith.constant(0, type=T.i32))

            def _select_wave_tdm_value(a_value, b_value):
                return arith.select(tdm_wave_is_a, a_value, b_value)

            active_stage_lds_addr = [
                _select_wave_tdm_value(
                    stages_a_lds_addr[i], stages_b_lds_addr[i])
                for i in range_constexpr(num_buffers)
            ]
            active_addr_lo = _select_wave_tdm_value(
                vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[]),
                vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[]))
            active_addr_hi = _select_wave_tdm_value(
                vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[]),
                vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[]))
            active_dgroup1 = _select_wave_tdm_value(
                desc_a_init.dgroup1, desc_b_init.dgroup1)
            active_adv_i32 = _select_wave_tdm_value(adv_a_i32, adv_b_i32)
        else:
            addr_lo_a = vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[])
            addr_hi_a = vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[])
            addr_lo_b = vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[])
            addr_hi_b = vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[])
            dgroup1_a = desc_a_init.dgroup1
            dgroup1_b = desc_b_init.dgroup1

        # --- Prologue ---
        if wave_specialized_tdm:
            for i in range_constexpr(pre_loaded):
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
                issue_tdm_loads(
                    tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                    tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                    wave_specialized=wave_specialized_tdm)
                addr_lo_a = arith.addi(addr_lo_a, adv_a_i32)
                addr_lo_b = arith.addi(addr_lo_b, adv_b_i32)

        pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2),
                       use_cluster=use_cluster)

        # --- Main loop (acc_mixed: fence at top, TDM mid-compute) ---
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

        if loop_iters > 0:
            if wave_specialized_tdm:
                init_args = list(accs) + [active_addr_lo]

                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_addr_lo = state[n_accs]

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers

                        pipeline_fence_signal(
                            outstanding=_fence_outstanding,
                            use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        addr_box = [cur_addr_lo]

                        def _mid_tdm_ws(
                            _ls=load_stage,
                            _ab=addr_box,
                            _k_off=(loop_iter * arith.index(num_buffers * tile_k)
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
                        accs_in = compute_tile(
                            accs_in,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            mid_compute_callback=_mid_tdm_ws)
                        cur_addr_lo = addr_box[0]
                        hot_loop_scheduler()

                    results = yield list(accs_in) + [cur_addr_lo]

                accs = list(results[:n_accs])
                active_addr_lo = results[n_accs]
            else:
                init_args = list(accs) + [addr_lo_a, addr_lo_b]

                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_lo_a = state[n_accs]
                    cur_lo_b = state[n_accs + 1]

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers

                        pipeline_fence_signal(
                            outstanding=_fence_outstanding,
                            use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        addr_boxes = [[cur_lo_a], [cur_lo_b]]

                        def _mid_tdm_nws(
                            _ls=load_stage,
                            _ab=addr_boxes,
                            _k_off=(loop_iter * arith.index(num_buffers * tile_k)
                                    + arith.index(buf_idx * tile_k)),
                        ):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0][0], addr_hi_a])
                            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_b_lds_addr[_ls],
                                _ab[1][0], addr_hi_b])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                wave_specialized=wave_specialized_tdm)
                            _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                            _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)
                            _l2_prefetch(_k_off)

                        rocdl.sched_barrier(0)
                        accs_in = compute_tile(
                            accs_in,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            mid_compute_callback=_mid_tdm_nws)
                        cur_lo_a = addr_boxes[0][0]
                        cur_lo_b = addr_boxes[1][0]
                        hot_loop_scheduler()

                    results = yield list(accs_in) + [cur_lo_a, cur_lo_b]

                accs = list(results[:n_accs])
                addr_lo_a = results[n_accs]
                addr_lo_b = results[n_accs + 1]

        # --- Tail ---
        # The main loop's last mid-compute TDM load needs to be fenced
        # before the tail starts reading newly loaded LDS data.
        if loop_iters > 0:
            pipeline_fence(outstanding=0, use_cluster=use_cluster)
        elif use_cluster:
            gpu.cluster_barrier()
        epi_addrs_box = [None]
        _tail_had_load = False
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if _outstanding == -1:
                if _tail_had_load:
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                if use_tdm_store:
                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage])
                else:
                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        emit_filler=_emit_epi_addrs)
            else:
                pipeline_fence_signal(outstanding=_outstanding,
                                      use_cluster=use_cluster)
                pipeline_fence_wait(use_cluster=use_cluster)

                _tail_mid_cb = None
                if _load_stage is not None:
                    _tail_had_load = True
                    if wave_specialized_tdm:
                        _tail_addr_box = [active_addr_lo]

                        def _tail_mid_ws(_ls=_load_stage, _ab=_tail_addr_box):
                            dg0 = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, active_stage_lds_addr[_ls],
                                _ab[0], active_addr_hi])
                            tdm_ops.tensor_load_2d(
                                tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                            _ab[0] = arith.addi(_ab[0], active_adv_i32)

                        _tail_mid_cb = _tail_mid_ws
                    else:
                        _tail_ab = [[addr_lo_a], [addr_lo_b]]

                        def _tail_mid_nws(_ls=_load_stage, _ab=_tail_ab):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0][0], addr_hi_a])
                            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_b_lds_addr[_ls],
                                _ab[1][0], addr_hi_b])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                wave_specialized=wave_specialized_tdm)
                            _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                            _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)

                        _tail_mid_cb = _tail_mid_nws

                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs,
                    stages_a_idx[_compute_stage],
                    stages_b_idx[_compute_stage],
                    mid_compute_callback=_tail_mid_cb)
                hot_loop_scheduler()

                if _load_stage is not None:
                    if wave_specialized_tdm:
                        active_addr_lo = _tail_addr_box[0]
                    else:
                        addr_lo_a = _tail_ab[0][0]
                        addr_lo_b = _tail_ab[1][0]

        # --- Epilogue ---
        if use_tdm_store:
            if d_need_epilogue_fence:
                pipeline_fence(outstanding=0, use_cluster=use_cluster)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            rocdl.sched_barrier(0)
            epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (in_dtype, out_dtype, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, effective_waves_per_eu, l2_prefetch_distance,
                 use_tdm_store, cluster_m, cluster_n,
                 wave_specialized_tdm, inst_prefetch, expert_sched_mode)

    @flyc.jit
    def launch_wmma_gemm_tdm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        launcher = kernel_wmma_gemm_tdm(arg_c, arg_a, arg_b, i32_m, i32_n)
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
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

        llvm_opts = {}
        if expert_sched_mode:
            llvm_opts["amdgpu-expert-scheduling-mode"] = True
        if inst_prefetch:
            llvm_opts["amdgpu-inst-prefetch-distance"] = 8
        if llvm_opts:
            launch_wmma_gemm_tdm.compile_hints["llvm_options"] = llvm_opts

    return launch_wmma_gemm_tdm


__all__ = ["compile_wmma_gemm_tdm"]
