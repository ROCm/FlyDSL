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
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

from kernels.layout_utils import crd2idx, idx2crd

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32

LDS_PAD_A = 8
LDS_PAD_B = 8

_STAGE_NAMES = ("ping", "pong", "pang")


def _make_tail_plan(num_buffers, pre_loaded, extra):
    """Compute a compile-time tail execution plan for the N-stage pipeline.

    Returns a list of (load_stage, compute_stage, outstanding) tuples, one per
    tail step.  outstanding=-1 means "last step, use compute_tile (no barrier)".

    Args:
        num_buffers: total number of pipeline stages.
        pre_loaded:  stages already loaded and ready to compute (= num_buffers - 1).
        extra:       additional tiles that must be loaded in the tail.
    """
    steps = pre_loaded + extra
    plan = []
    for i in range(steps):
        compute_stage = (
            i if i < pre_loaded
            else (i - pre_loaded + num_buffers - 1) % num_buffers
        )
        load_stage = (
            (i + num_buffers - 1) % num_buffers if i < extra
            else None
        )
        is_last = (i == steps - 1)
        if is_last:
            outstanding = -1
        else:
            j = i + 1
            next_compute = (
                j if j < pre_loaded
                else (j - pre_loaded + num_buffers - 1) % num_buffers
            )
            outstanding = (
                2 if (load_stage is not None and load_stage != next_compute)
                else 0
            )
        plan.append((load_stage, compute_stage, outstanding))
    return plan


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
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
):
    """Compile a WMMA GEMM kernel with TDM async copy and multi-stage buffering.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, M, N, stream)

    Args:
        num_buffers: Number of LDS buffers (2=double, 3=triple buffering).
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
        l2_prefetch_distance: Number of k-tiles ahead to prefetch into L2.
                              0 = disabled, 2 = typical value.
    """
    _ = (M, N)
    if num_buffers not in (2, 3):
        raise ValueError(f"num_buffers must be 2 or 3, got {num_buffers}")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")
    is_f16 = in_dtype == "fp16"
    elem_bytes = 2

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

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16
    k_wmma_steps = tile_k // WMMA_K

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep

    lds_a_stride = tile_k + LDS_PAD_A
    lds_b_stride = tile_k + LDS_PAD_B
    lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    lds_b_elems = tile_n * lds_b_stride + LDS_PAD_A

    buf_size_elems = lds_a_elems + lds_b_elems

    # --- LDS allocation ---
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
        stage_a_offsets.append(off)
        stage_b_offsets.append(off + lds_a_elems * elem_bytes)

    # Compile-time pipeline parameters
    pre_loaded = num_buffers - 1        # stages pre-loaded in prologue
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers  # index of first un-computed tile in tail
    extra = num_k_tiles - _tail_start - pre_loaded
    tail_plan = _make_tail_plan(num_buffers, pre_loaded, extra)

    @flyc.kernel
    def kernel_wmma_gemm_tdm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        # --- Thread/wave decomposition ---
        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            thr_coord[0], thr_coord[1], thr_coord[2], thr_coord[3])

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        elem_ty = _elem_type()

        # --- Epilogue setup ---
        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_stride = arith.index(N)
        c_nrec = m_idx * n_stride * arith.index(4)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        # --- TDM async copy helpers ---
        def copy_a_to_lds(k_base, lds_a_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=lds_a_mem_ref,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k), strides=(K, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_A,
                num_warps=num_warps)
            tdm_ops.tensor_load_2d(desc)

        def copy_b_to_lds(k_base, lds_b_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=lds_b_mem_ref,
                global_offset=(blk_n, k_base),
                tensor_shape=(tile_n, tile_k), strides=(K, 1),
                tile_shape=(tile_n, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_B,
                num_warps=num_warps)
            tdm_ops.tensor_load_2d(desc)

        layout_smem_a = fx.make_layout((tile_m, lds_a_stride), (lds_a_stride, 1))
        layout_smem_b = fx.make_layout((tile_n, lds_b_stride), (lds_b_stride, 1))

        # --- LDS load helpers ---
        from flydsl._mlir.dialects import vector as vec_d

        FRAG_K_ELEMS = 8

        def _get_lds_memref(lds_ptr):
            """Get the raw memref value from SmemPtr or raw memref."""
            if isinstance(lds_ptr, SmemPtr):
                return get_op_result_or_value(lds_ptr.get())
            return get_op_result_or_value(lds_ptr)

        def load_wmma_frag(lds_ptr, row_base, k_base, lds_layout):
            """Load one 16x32 WMMA fragment from LDS using vectorized 128-bit loads.

            Uses vector.load to read 8 contiguous fp16 elements at once,
            avoiding scalar load + v_perm/v_alignbit overhead.
            Two 128-bit loads per fragment (2 × 8 fp16 = 16 fp16 values).
            """
            raw_memref = _get_lds_memref(lds_ptr)
            row = row_base + lane16
            vec8_ty = ir.VectorType.get([8], elem_ty)

            # Two K-groups per fragment:
            # Group 0 (values 0-7):  k = k_base + lane_kgrp * 8
            # Group 1 (values 8-15): k = k_base + (2 + lane_kgrp) * 8
            k0 = k_base + lane_kgrp * arith.index(8)
            k1 = k_base + (arith.index(2) + lane_kgrp) * arith.index(8)

            off0 = crd2idx((row, k0), lds_layout)
            off1 = crd2idx((row, k1), lds_layout)

            idx0 = [get_op_result_or_value(off0)]
            idx1 = [get_op_result_or_value(off1)]

            v0 = vec_d.load(vec8_ty, raw_memref, idx0)
            v1 = vec_d.load(vec8_ty, raw_memref, idx1)

            # Concatenate two vec<8> into vec<16> via vector.shuffle
            mask = ir.DenseI64ArrayAttr.get(list(range(16)))
            return vec_d.shuffle(v0, v1, mask)

        # --- K-subtile load/compute helpers ---
        # Number of ds_load_b128 per K-subtile:
        # B frags: wmma_n_rep * 2, A frags: wmma_m_rep * 2
        LOADS_PER_SUBTILE = (wmma_m_rep + wmma_n_rep) * 2

        def load_k_subtile_frags(lds_a_ptr, lds_b_ptr, ks):
            """Batch-load all A and B fragments for one K-subtile (no wait)."""
            k_off = arith.index(ks * WMMA_K)

            b_frags = [load_wmma_frag(
                lds_b_ptr, warp_n_base + arith.index(wn * WMMA_N),
                k_off, layout_smem_b)
                for wn in range_constexpr(wmma_n_rep)]

            a_frags = [load_wmma_frag(
                lds_a_ptr, warp_m_base + arith.index(wm * WMMA_M),
                k_off, layout_smem_a)
                for wm in range_constexpr(wmma_m_rep)]

            return a_frags, b_frags

        def do_k_subtile_wmma(a_frags, b_frags, accs):
            """Execute all WMMAs for one K-subtile using pre-loaded fragments."""
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    accs[idx] = wmma_op(
                        T.vec(8, T.f32),
                        b_frags[wn], a_frags[wm],
                        accs[idx],
                        signA=False, signB=False, modC=0,
                        reuseA=False, reuseB=False,
                    ).result
            return accs

        # --- Compute on one LDS buffer (K-subtile pipelined) ---
        def compute_tile(accs_in, lds_a_ptr, lds_b_ptr, emit_filler=None):
            current_accs = list(accs_in)

            if k_wmma_steps == 1:
                a_frags, b_frags = load_k_subtile_frags(lds_a_ptr, lds_b_ptr, 0)
                rocdl.s_wait_dscnt(0)
                if emit_filler is not None:
                    emit_filler()
                current_accs = do_k_subtile_wmma(a_frags, b_frags, current_accs)
            else:
                # Prologue: batch-load K-subtile 0
                prev_a, prev_b = load_k_subtile_frags(lds_a_ptr, lds_b_ptr, 0)

                # Main K-loop: overlap load[ks+1] with compute[ks]
                for ks in range_constexpr(k_wmma_steps - 1):
                    next_a, next_b = load_k_subtile_frags(
                        lds_a_ptr, lds_b_ptr, ks + 1)
                    rocdl.s_wait_dscnt(LOADS_PER_SUBTILE)
                    current_accs = do_k_subtile_wmma(prev_a, prev_b, current_accs)
                    prev_a, prev_b = next_a, next_b

                rocdl.s_wait_dscnt(0)
                if emit_filler is not None:
                    rocdl.sched_barrier(0)
                    emit_filler()
                current_accs = do_k_subtile_wmma(prev_a, prev_b, current_accs)

            return current_accs

        # --- Scheduling ---
        def hot_loop_scheduler():
            rocdl.sched_barrier(0)

        # --- Epilogue: vectorized buffer_store_b128 ---
        def epilogue_prepare_addrs():
            """Precompute all epilogue store addresses (VALU only, no stores). """
            addrs = []
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = (blk_n + warp_n_base + arith.index(wn * WMMA_N)
                                + lane_kgrp * arith.index(8))
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
                    for half in range_constexpr(2):
                        vals = [vector.extract(
                            final_accs[idx],
                            static_position=[half * 4 + vi],
                            dynamic_position=[])
                            for vi in range_constexpr(4)]
                        vec4 = vector.from_elements(T.vec(4, T.f32), vals)
                        buffer_ops.buffer_store(vec4, c_rsrc, addrs[addr_idx])
                        addr_idx += 1

        # --- Pipeline helpers ---
        def wait_and_barrier(outstanding=0):
            tdm_ops.tensor_wait(outstanding)
            gpu.barrier()

        def _compute_and_schedule(accs_in, lds_a, lds_b):
            rocdl.sched_barrier(0)
            accs_out = compute_tile(accs_in, lds_a, lds_b)
            hot_loop_scheduler()
            return accs_out

        def _l2_prefetch(k_base):
            if l2_prefetch_distance <= 0:
                return
            pf_k = k_base + arith.index(l2_prefetch_distance * tile_k)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, pf_k), (tile_m, tile_k), (K, 1),
                elem_bytes=elem_bytes, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b, (blk_n, pf_k), (tile_n, tile_k), (K, 1),
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

        # Prologue: load first (num_buffers - 1) tiles into stages 0..(num_buffers-2)
        for i in range_constexpr(pre_loaded):
            copy_a_to_lds(arith.index(i * tile_k), stages_a_mem[i])
            copy_b_to_lds(arith.index(i * tile_k), stages_b_mem[i])
        # Wait until stage[0] is ready; allow later-stage loads to still be in flight
        # outstanding = 2 * (num_buffers - 2): 0 for double-buffer, 2 for triple-buffer
        wait_and_barrier(outstanding=2 * (num_buffers - 2))

        # Main loop: each iteration covers (num_buffers) K-tiles
        # Sub-phase s: load the "next" tile, compute the "current" tile, then barrier
        #   load_stage  = (s + num_buffers - 1) % num_buffers
        #   load_offset = iv + (s + num_buffers - 1) * tile_k
        #   compute stage[s]
        main_end = loop_iters * num_buffers * tile_k

        if loop_iters > 0:
            for iv, state in range(0, main_end, num_buffers * tile_k, init=list(accs)):
                accs_in = list(state)
                for s in range_constexpr(num_buffers):
                    _load_stage = (s + num_buffers - 1) % num_buffers
                    _load_k_off = (s + num_buffers - 1) * tile_k
                    copy_a_to_lds(iv + arith.index(_load_k_off), stages_a_mem[_load_stage])
                    copy_b_to_lds(iv + arith.index(_load_k_off), stages_b_mem[_load_stage])
                    _l2_prefetch(iv + arith.index(s * tile_k))
                    accs_in = _compute_and_schedule(accs_in, stages_a[s], stages_b[s])
                    wait_and_barrier(outstanding=2)
                results = yield list(accs_in)
            accs = list(results)

        # Tail: handle remaining tiles using the compile-time plan
        # Each plan step: optionally load one tile, compute one stage, then wait.
        # outstanding=-1 → last step: use compute_tile (no barrier).
        _extra_j = 0
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if _load_stage is not None:
                _k_off = (_tail_start + pre_loaded + _extra_j) * tile_k
                copy_a_to_lds(arith.index(_k_off), stages_a_mem[_load_stage])
                copy_b_to_lds(arith.index(_k_off), stages_b_mem[_load_stage])
                _extra_j += 1
            if _outstanding == -1:
                epi_addrs_box = [None]

                def _emit_epi_addrs():
                    epi_addrs_box[0] = epilogue_prepare_addrs()

                accs = compute_tile(
                    accs, stages_a[_compute_stage], stages_b[_compute_stage],
                    emit_filler=_emit_epi_addrs)
            else:
                accs = _compute_and_schedule(
                    accs, stages_a[_compute_stage], stages_b[_compute_stage])
                wait_and_barrier(outstanding=_outstanding)

        epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (in_dtype, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, waves_per_eu, l2_prefetch_distance)

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
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
        launcher.launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_wmma_gemm_tdm


__all__ = ["compile_wmma_gemm_tdm"]
