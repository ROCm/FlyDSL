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
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl.expr import idx2crd
from kernels.gemm_common_gfx1250 import (
    get_lds_memref, pipeline_fence,
    store_acc_vec8_to_buffer, store_acc_vec8_to_lds,
)
from kernels.pipeline_utils import make_tail_plan

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32

LDS_PAD_A = 8
LDS_PAD_B = 8
LDS_PAD_D_BYTES = 16

_STAGE_NAMES = ("ping", "pong", "pang")


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
):
    """Compile a WMMA GEMM kernel with TDM async copy and multi-stage buffering.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, M, N, stream)

    Args:
        out_dtype: Output element type ("f16", "bf16", "f32").
                   Default (None) = matches input type.
        num_buffers: Number of LDS buffers (2=double, 3=triple buffering).
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
        l2_prefetch_distance: Number of k-tiles ahead to prefetch into L2.
                              0 = disabled, 2 = typical value.
        use_tdm_store: Use TDM store epilogue via LDS (True) or buffer_store (False).
        cluster_m: Cluster dimension along M (WG rows per cluster, 1=disabled).
        cluster_n: Cluster dimension along N (WG cols per cluster, 1=disabled).
    """
    _ = (M, N)
    if num_buffers not in (2, 3):
        raise ValueError(f"num_buffers must be 2 or 3, got {num_buffers}")
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

    block_threads = m_warp * n_warp * WAVE_SIZE

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

    buf_size_elems = lds_a_elems + lds_b_elems

    # --- LDS allocation (B-first: B at offset 0 for smaller ds_load offsets) ---
    num_warps = m_warp * n_warp

    stage_allocators = []
    stage_a_offsets = []
    stage_b_offsets = []
    for i in range(num_buffers):
        name = _STAGE_NAMES[i]
        alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"wmma_tdm_{name}")
        off = alloc._align(alloc.ptr, 16)
        alloc.ptr = off + buf_size_elems * elem_bytes
        stage_allocators.append(alloc)
        stage_b_offsets.append(off)
        stage_a_offsets.append(off + lds_b_elems * elem_bytes)

    # Compile-time pipeline parameters
    pre_loaded = num_buffers - 1        # stages pre-loaded in prologue
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers  # index of first un-computed tile in tail
    extra = num_k_tiles - _tail_start - pre_loaded
    tail_plan = _make_tail_plan(num_buffers, pre_loaded, extra)

    if use_tdm_store:
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        # Element-based versions (f16 = 2 bytes) for vector LDS store path
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        _last_compute_stage = tail_plan[-1][1]
        d_reuse_stage = 1 if _last_compute_stage == 0 else 0
        if total_d_bytes > stage_allocators[d_reuse_stage].ptr:
            stage_allocators[d_reuse_stage].ptr = total_d_bytes

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
        def _precompute_a_lane_bases(lds_ptr):
            """Precompute per-wm A fragment lane base addresses.

            Returns (lds_buffer, bases) where bases[wm] =
              (warp_m_base + wm*WMMA_M + lane16) * lds_a_stride + lane_kgrp * 8
            """
            lds_buffer = get_lds_memref(lds_ptr)
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride)
            k_lane_off = lane_kgrp * arith.index(8)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = row_stride_off + arith.index(wm * WMMA_M * lds_a_stride) + k_lane_off
                bases.append(a_base)
            return lds_buffer, bases

        def load_wmma_frag(a_lds_buffer, a_lane_base, ks):
            """Load one 16x32 WMMA fragment from LDS using vectorized 128-bit loads.

            a_lane_base is precomputed by _precompute_a_lane_bases.
            ks is the K-subtile index (compile-time constant).
            """
            vec8_ty = ir.VectorType.get([8], elem_ty)

            off0 = a_lane_base + arith.index(ks * WMMA_K)
            off1 = a_lane_base + arith.index(ks * WMMA_K + 16)

            v0 = vector.load_op(vec8_ty, a_lds_buffer, [off0])
            v1 = vector.load_op(vec8_ty, a_lds_buffer, [off1])

            return vector.shuffle(v0, v1, list(range(16)))

        def _precompute_b_lane_bases(lds_ptr):
            """Precompute per-wn B fragment lane base addresses.

            Returns a list of (lds_buffer, b_lane_base) for each wn.
            b_lane_base = (lane_kgrp*8 + lane8) * lds_b_stride
                        + (warp_n_base + wn*WMMA_N + lane_ngrp*8)
            where lane8 = lane16 % 8, lane_ngrp = lane16 / 8.

            After precompute, lane8/lane_ngrp are dead → frees VGPRs.
            """
            lds_buffer = get_lds_memref(lds_ptr)
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(lds_b_stride)
            n_lane_off = lane_ngrp * arith.index(8)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = warp_n_base + arith.index(wn * WMMA_N) + n_lane_off
                b_base = k_lane_off + n_col
                bases.append(b_base)
            return lds_buffer, bases

        def load_wmma_frag_tr(lds_buffer, b_lane_base, ks):
            """Load one 16x32 WMMA B fragment using ds_load_tr16_b128.

            b_lane_base is precomputed by _precompute_b_lane_bases.
            ks is the K-subtile index (compile-time constant from range_constexpr).
            The K offset is folded into a compile-time constant multiplication.
            """
            vec8_ty = ir.VectorType.get([8], elem_ty)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride
                elem_off = b_lane_base + arith.index(k_row_off)
                v = rocdl.lds_transpose_load(vec8_ty, lds_buffer, elem_off, elem_bytes)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        # --- K-subtile compute (A-streaming pipeline) ---
        def _load_b_frags(b_lds_buffer, b_bases, ks):
            """Load all B fragments for one K-subtile (no wait)."""
            return [load_wmma_frag_tr(b_lds_buffer, b_bases[wn], ks)
                    for wn in range_constexpr(wmma_n_rep)]

        def _a_streaming_compute(accs, a_buf, a_bases, b_frags, ks,
                                 emit_filler=None, next_b_info=None):
            """Stream A fragments per-wm group, interleaved with WMMA.

            B frags are pre-loaded and reused across all wm groups.
            Next A frag is pre-loaded during current wm's WMMA to hide
            ds_load latency behind WMMA execution.

            next_b_info: optional (b_buf, b_bases, next_ks) — issues B loads
                         for the next K-subtile during last wm's WMMAs.
                         When set, returns (accs, next_b_frags).
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
                    rocdl.s_wait_dscnt(2)
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    accs[idx] = wmma_op(
                        T.vec(8, T.f32),
                        b_frags[wn], a_frag, accs[idx],
                        signA=False, signB=False, modC=0,
                        reuseA=False, reuseB=False,
                    ).result
                if not is_last:
                    a_frag = a_next
            if next_b_info is not None:
                return accs, next_b_frags
            return accs

        # --- Compute on one LDS buffer (A-streaming K-subtile pipeline) ---
        def compute_tile(accs_in, lds_a_ptr, lds_b_ptr, emit_filler=None):
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_ptr)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_ptr)

            if k_wmma_steps == 1:
                b_frags = _load_b_frags(b_buf, b_bases, 0)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, b_frags, 0,
                    emit_filler=emit_filler)
            else:
                prev_b = _load_b_frags(b_buf, b_bases, 0)
                for ks in range_constexpr(k_wmma_steps - 1):
                    current_accs, prev_b = _a_streaming_compute(
                        current_accs, a_buf, a_bases, prev_b, ks,
                        next_b_info=(b_buf, b_bases, ks + 1))
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, prev_b,
                    k_wmma_steps - 1, emit_filler=emit_filler)

            return current_accs

        # --- Scheduling ---
        def hot_loop_scheduler():
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

        # Build per-stage SmemPtrs (one per pipeline stage)
        base_ptrs = [sa.get_base() for sa in stage_allocators]
        stages_a = [
            SmemPtr(base_ptrs[i], stage_a_offsets[i], elem_ty, shape=(lds_a_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(base_ptrs[i], stage_b_offsets[i], elem_ty, shape=(lds_b_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]

        # D output LDS setup for TDM store epilogue
        if use_tdm_store:
            d_lds_base_ptr = base_ptrs[d_reuse_stage]
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

        # Precompute LDS addresses for all stages
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(vector.extract(make_desc_a(stages_a_mem[i], arith.index(0)).dgroup0, static_position=[1], dynamic_position=[]))
            stages_b_lds_addr.append(vector.extract(make_desc_b(stages_b_mem[i], arith.index(0)).dgroup0, static_position=[1], dynamic_position=[]))

        # Create initial descriptors (pointing to tile 0)
        desc_a_init = make_desc_a(stages_a_mem[0], arith.index(0))
        desc_b_init = make_desc_b(stages_b_mem[0], arith.index(0))
        
        dgroup0_a = desc_a_init.dgroup0
        dgroup0_b = desc_b_init.dgroup0
        
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1
        
        adv_a_bytes = arith.index(tile_k * elem_bytes)
        adv_b_bytes = arith.index(tile_k * N * elem_bytes)

        # Prologue: load first (num_buffers - 1) tiles into stages 0..(num_buffers-2)
        for i in range_constexpr(pre_loaded):
            dgroup0_a = vector.insert(stages_a_lds_addr[i], dgroup0_a, static_position=[1], dynamic_position=[])
            dgroup0_b = vector.insert(stages_b_lds_addr[i], dgroup0_b, static_position=[1], dynamic_position=[])
            
            desc_a = tdm_ops.TDMDescriptor2D(dgroup0_a, dgroup1_a)
            desc_b = tdm_ops.TDMDescriptor2D(dgroup0_b, dgroup1_b)
            
            tdm_ops.tensor_load_2d(desc_a)
            tdm_ops.tensor_load_2d(desc_b)
            
            dgroup0_a = tdm_ops.advance_tdm_descriptor(desc_a, adv_a_bytes).dgroup0
            dgroup0_b = tdm_ops.advance_tdm_descriptor(desc_b, adv_b_bytes).dgroup0
            
        pipeline_fence(outstanding=2 * (num_buffers - 2), use_cluster=use_cluster)

        def _select_base(bases_list, stage_idx):
            if len(bases_list) == 1:
                return bases_list[0]
            if len(bases_list) == 2:
                is_zero = arith.cmpi(arith.CmpIPredicate.eq, stage_idx, arith.index(0))
                return arith.select(is_zero, bases_list[0], bases_list[1])
            result = bases_list[-1]
            _n = len(bases_list)
            for ii in range_constexpr(_n - 1):
                i = _n - 2 - ii
                is_i = arith.cmpi(arith.CmpIPredicate.eq, stage_idx, arith.index(i))
                result = arith.select(is_i, bases_list[i], result)
            return result

        # Main loop: each iteration covers num_buffers K-tiles
        main_end = loop_iters * num_buffers * tile_k

        if loop_iters > 0:
            init_args = list(accs) + [dgroup0_a, dgroup0_b]
            
            for iv, state in range(0, main_end, num_buffers * tile_k, init=init_args):
                accs_in = list(state[:n_accs])
                cur_dgroup0_a = state[n_accs]
                cur_dgroup0_b = state[n_accs + 1]
                
                for s in range_constexpr(num_buffers):
                    _load_stage = (s + num_buffers - 1) % num_buffers
                    
                    lds_a = _select_base(stages_a_lds_addr, arith.index(_load_stage))
                    lds_b = _select_base(stages_b_lds_addr, arith.index(_load_stage))
                    
                    next_dgroup0_a = vector.insert(lds_a, cur_dgroup0_a, static_position=[1], dynamic_position=[])
                    next_dgroup0_b = vector.insert(lds_b, cur_dgroup0_b, static_position=[1], dynamic_position=[])
                    
                    desc_a = tdm_ops.TDMDescriptor2D(next_dgroup0_a, dgroup1_a)
                    desc_b = tdm_ops.TDMDescriptor2D(next_dgroup0_b, dgroup1_b)
                    
                    tdm_ops.tensor_load_2d(desc_a)
                    tdm_ops.tensor_load_2d(desc_b)
                    
                    cur_dgroup0_a = tdm_ops.advance_tdm_descriptor(desc_a, adv_a_bytes).dgroup0
                    cur_dgroup0_b = tdm_ops.advance_tdm_descriptor(desc_b, adv_b_bytes).dgroup0

                    _l2_prefetch(iv + arith.index(s * tile_k))
                    rocdl.sched_barrier(0)
                    accs_in = compute_tile(accs_in, stages_a[s], stages_b[s])
                    hot_loop_scheduler()
                    pipeline_fence(outstanding=2 * (num_buffers - 2),
                                   use_cluster=use_cluster)
                results = yield list(accs_in) + [cur_dgroup0_a, cur_dgroup0_b]
            
            accs = list(results[:n_accs])
            dgroup0_a = results[n_accs]
            dgroup0_b = results[n_accs + 1]

        # Tail: handle remaining tiles using the compile-time plan
        if loop_iters == 0 and use_cluster:
            gpu.cluster_barrier()
        _extra_j = 0
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if _load_stage is not None:
                dgroup0_a = vector.insert(stages_a_lds_addr[_load_stage], dgroup0_a, static_position=[1], dynamic_position=[])
                dgroup0_b = vector.insert(stages_b_lds_addr[_load_stage], dgroup0_b, static_position=[1], dynamic_position=[])
                
                desc_a = tdm_ops.TDMDescriptor2D(dgroup0_a, dgroup1_a)
                desc_b = tdm_ops.TDMDescriptor2D(dgroup0_b, dgroup1_b)
                
                tdm_ops.tensor_load_2d(desc_a)
                tdm_ops.tensor_load_2d(desc_b)
                
                dgroup0_a = tdm_ops.advance_tdm_descriptor(desc_a, adv_a_bytes).dgroup0
                dgroup0_b = tdm_ops.advance_tdm_descriptor(desc_b, adv_b_bytes).dgroup0
                
                _extra_j += 1
            if _outstanding == -1:
                if use_tdm_store:
                    accs = compute_tile(
                        accs, stages_a[_compute_stage], stages_b[_compute_stage])
                else:
                    epi_addrs_box = [None]

                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    accs = compute_tile(
                        accs, stages_a[_compute_stage], stages_b[_compute_stage],
                        emit_filler=_emit_epi_addrs)
            else:
                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs, stages_a[_compute_stage], stages_b[_compute_stage])
                hot_loop_scheduler()
                pipeline_fence(outstanding=_outstanding,
                               use_cluster=use_cluster)

        if use_tdm_store:
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (in_dtype, out_dtype, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, effective_waves_per_eu, l2_prefetch_distance,
                 use_tdm_store, cluster_m, cluster_n)

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
            for alloc in stage_allocators:
                alloc.finalized = False
            for alloc in stage_allocators:
                alloc.finalize()

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

    return launch_wmma_gemm_tdm


__all__ = ["compile_wmma_gemm_tdm"]
