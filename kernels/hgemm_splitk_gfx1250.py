"""Split-K HGEMM kernel for gfx1250 using WMMA + TDM async copy.

Combines the split-K reduction strategy from hgemm_splitk.py with
the gfx1250 TDM pipeline infrastructure from wmma_gemm_gfx1250.py.

Supports double-buffer (2-stage) and triple-buffer (3-stage) pipelining
with TDM hardware async copy for both A and B tiles.

Matrix layouts:
  A: M x K (row-major)
  B: K x N (row-major)
  C: M x N (row-major)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity

from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx,
    get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    lds_transpose_load_raw,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    store_acc_vec8_to_buffer,
    store_acc_vec8_to_lds,
)
from kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32
DS_LOADS_PER_A_FRAG = 2
DS_LOADS_PER_B_FRAG = 2

LDS_PAD_A = 8
LDS_PAD_B = 8
LDS_PAD_D_BYTES = 16

SPLIT_K_COUNTER_MAX_LEN = 512

_make_tail_plan = make_tail_plan


def compile_hgemm_splitk_gfx1250(
    *,
    N: int,
    K: int,
    tile_m: int = 128,
    tile_n: int = 256,
    tile_k: int = 128,
    SPLIT_K: int = 1,
    m_warp: int = 2,
    n_warp: int = 4,
    in_dtype: str = "fp16",
    out_dtype: str = None,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    use_tdm_store: bool = True,
    inst_prefetch: bool = False,
    expert_sched_mode: bool = True,
):
    """Compile a split-K WMMA HGEMM kernel with TDM async copy for gfx1250.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, M, N, COUNTER, signal_state, stream)

    Args:
        N: Number of columns in B / C.
        K: Reduction dimension (before split-K division).
        tile_m: Tile size along M.
        tile_n: Tile size along N.
        tile_k: Tile size along K (per iteration).
        SPLIT_K: Number of K-slices. 1 = no split-K.
        m_warp: Number of warps along M.
        n_warp: Number of warps along N.
        in_dtype: Input element type ("fp16" or "bf16").
        out_dtype: Output element type (None = matches input).
        num_buffers: Number of LDS buffers (2 or 3).
        waves_per_eu: Occupancy hint.
        l2_prefetch_distance: K-tiles ahead to prefetch into L2.
        use_tdm_store: Use TDM store epilogue (only when SPLIT_K == 1).
        inst_prefetch: Enable instruction prefetch.
        expert_sched_mode: Enable AMDGPU expert scheduling mode.
    """
    IS_SPLIT_K = SPLIT_K > 1
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

    # Split-K forces buffer_store epilogue with atomic fadd
    if IS_SPLIT_K:
        use_tdm_store = False

    assert (K % SPLIT_K == 0) and (K // SPLIT_K >= 1)
    ks = K // SPLIT_K  # per-slice K dimension

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    effective_waves_per_eu = waves_per_eu

    TDM_LOADS_PER_STEP = 2  # always load both A and B

    if ks % tile_k != 0:
        raise ValueError(f"K/SPLIT_K must be divisible by tile_k={tile_k}, got ks={ks}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2, got {tile_k}")

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    num_k_tiles = ks // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles} (ks={ks}, tile_k={tile_k})")

    gpu_arch = str(get_rocm_arch())
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

    # --- LDS allocation ---
    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    stage_layout = SmemAllocator(None, arch=gpu_arch, global_sym_name="splitk_tdm_layout")
    stage_b_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_rel_off + lds_b_elems * elem_bytes
    stage_a_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_rel_off + lds_a_elems * elem_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    # Pipeline parameters
    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
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
            f"splitk_tdm_{in_dtype}_{out_dtype}_{tile_m}x{tile_n}x{tile_k}_"
            f"{m_warp}x{n_warp}_{num_buffers}buf_spk{SPLIT_K}_arena"),
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

    # TDM store epilogue setup (only for non-split-K)
    if use_tdm_store and not IS_SPLIT_K:
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

    # For split-K atomic epilogue, we need LDS for staging accumulators
    if IS_SPLIT_K:
        # C tile in LDS for cooperative store
        _splitk_c_elems = tile_m * tile_n
        _splitk_c_bytes = _splitk_c_elems * elem_bytes
        if _splitk_c_bytes > arena_total_bytes:
            arena_total_bytes = _splitk_c_bytes
            arena_alloc.ptr = _splitk_c_bytes

    check_smem_capacity(arena_total_bytes, gpu_arch)

    # --- Epilogue store helpers for split-K atomic ---
    LDG_VEC_SIZE = 8
    DTYPE_BYTES = 2
    LDG_C_X_THREADS = tile_n // LDG_VEC_SIZE
    BLOCK_MN_SIZE = tile_m * tile_n
    BLOCK_VECS = LDG_VEC_SIZE * block_threads
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert LDG_REG_C_COUNT >= 1
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0

    @flyc.kernel
    def kernel_hgemm_splitk(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        COUNTER: fx.Tensor,
        signal_state: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = gpu.block_id("z")  # split-K slice index

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        # K-slice offset
        ks_begin = bz * arith.index(ks)

        # No cluster for split-K
        a_mcast_mask = 0
        b_mcast_mask = 0

        # --- Thread/wave decomposition ---
        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1),
            fx.get(thr_coord, 2), fx.get(thr_coord, 3))

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        elem_ty = _elem_type()

        # --- Global memory setup ---
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        _i64_type = T.i64
        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_stride = arith.index(N)

        # Buffer resource for C
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        # Split-K counter setup
        if IS_SPLIT_K:
            tid_i32 = arith.index_cast(T.i32, tx)
            sig_i32 = arith.ArithValue(signal_state.ir_value())
            counter_idx = (
                sig_i32
                * arith.constant(SPLIT_K_COUNTER_MAX_LEN, type=T.i32)
                + arith.index_cast(T.i32, bx)
                * arith.constant(N // tile_n, type=T.i32)
                + arith.index_cast(T.i32, by)
            )

        # --- Zero-C and semaphore prologue for split-K ---
        if IS_SPLIT_K:
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, bz, arith.index(0))

            # Zero C output (runtime loop to avoid excessive unrolling)
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                c_zero = arith.constant(0.0, type=elem_ty)
                zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, elem_ty), c_zero)
                _zc_lo = _raw(arith.index(0))
                _zc_hi = _raw(arith.index(LDG_REG_C_COUNT))
                _zc_step = _raw(arith.index(1))
                _zc_loop = scf.ForOp(_zc_lo, _zc_hi, _zc_step, [])
                with ir.InsertionPoint(_zc_loop.body):
                    _zc_iv = arith.ArithValue(_zc_loop.induction_variable)
                    global_tid = _zc_iv * arith.index(block_threads) + tx
                    m_local_idx = global_tid / arith.index(LDG_C_X_THREADS)
                    n_local_idx = (global_tid % arith.index(LDG_C_X_THREADS)) * arith.index(LDG_VEC_SIZE)
                    row_idx = blk_m + m_local_idx
                    cond_boundary = arith.cmpi(arith.CmpIPredicate.ult, row_idx, m_idx)
                    cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                    with ir.InsertionPoint(cond_boundary_if.then_block):
                        c_off = row_idx * n_stride + blk_n + n_local_idx
                        c_off_i32 = arith.index_cast(T.i32, c_off)
                        buffer_ops.buffer_store(zero_vec, c_rsrc, c_off_i32)
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()

            # Write flag (thread 0 of ks_idx == 0)
            cond_ks0_if2 = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if2.then_block):
                is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tx, arith.index(0))
                is_t0_if = scf.IfOp(is_t0, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_if.then_block):
                    counter_base_ptr = fly.extract_aligned_pointer_as_index(
                        _ptr_type, fly_values(COUNTER)[0])
                    counter_base_int = llvm.PtrToIntOp(_i64_type, counter_base_ptr).result
                    counter_byte_off = arith.index_cast(
                        T.i64,
                        arith.index_cast(T.index, counter_idx) * arith.index(4))
                    counter_ptr = llvm.IntToPtrOp(
                        _ptr_type,
                        llvm.AddOp(counter_base_int, counter_byte_off,
                                   llvm.IntegerOverflowFlags(0)).result).result
                    counter_ptr_v = counter_ptr._value if hasattr(counter_ptr, "_value") else counter_ptr
                    llvm.InlineAsmOp(
                        None, [counter_ptr_v, arith.constant(1, type=T.i32)],
                        "global_store_b32 $0, $1, off scope:SCOPE_DEV", "v,v",
                        has_side_effects=True)
                    rocdl.s_wait_storecnt(0)
                    scf.YieldOp([])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()

            # Clean old signal — vec4 store: 128 threads × 4 dwords = 512 entries
            # Per CU: 128 × 16B = 2048B; per device: bm×bn × 2048B
            cond_ks0_if3 = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if3.then_block):
                sig_val = arith.ArithValue(signal_state.ir_value())
                clean_state = (sig_val + arith.constant(2, type=T.i32)) % arith.constant(3, type=T.i32)
                clean_base = clean_state * arith.constant(SPLIT_K_COUNTER_MAX_LEN, type=T.i32)
                clean_idx = clean_base + arith.index_cast(T.i32, tx) * arith.constant(4, type=T.i32)
                counter_base_ptr2 = fly.extract_aligned_pointer_as_index(
                    _ptr_type, fly_values(COUNTER)[0])
                counter_base_int2 = llvm.PtrToIntOp(_i64_type, counter_base_ptr2).result
                clean_byte_off = arith.index_cast(
                    T.i64,
                    arith.index_cast(T.index, clean_idx) * arith.index(4))
                clean_ptr = llvm.IntToPtrOp(
                    _ptr_type,
                    llvm.AddOp(counter_base_int2, clean_byte_off,
                               llvm.IntegerOverflowFlags(0)).result).result
                clean_ptr_v = clean_ptr._value if hasattr(clean_ptr, "_value") else clean_ptr
                zero_vec4 = vector.broadcast(T.vec(4, T.i32), arith.constant(0, type=T.i32))
                zero_vec4_v = zero_vec4._value if hasattr(zero_vec4, "_value") else zero_vec4
                llvm.InlineAsmOp(
                    None, [clean_ptr_v, zero_vec4_v],
                    "global_store_b128 $0, $1, off scope:SCOPE_DEV", "v,v",
                    has_side_effects=True)
                rocdl.s_wait_storecnt(0)
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()

        # --- TDM async copy helpers ---
        def make_desc_a(lds_a_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=lds_a_mem_ref,
                global_offset=(blk_m, ks_begin + k_base),
                tensor_shape=(tile_m, tile_k), strides=(K, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_A,
                num_warps=num_warps,
                workgroup_mask=a_mcast_mask)

        def make_desc_b(lds_b_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=lds_b_mem_ref,
                global_offset=(ks_begin + k_base, blk_n),
                tensor_shape=(tile_k, tile_n), strides=(N, 1),
                tile_shape=(tile_k, tile_n), elem_bytes=elem_bytes,
                pad_interval=tile_n, pad_amount=LDS_PAD_B,
                num_warps=num_warps,
                workgroup_mask=b_mcast_mask)

        # --- LDS load helpers ---
        def _precompute_a_lane_bases(lds_base_idx):
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
            vec8_ty = ir.VectorType.get([8], elem_ty)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride * elem_bytes
                elem_off = b_lane_base + arith.index(k_row_off)
                v = lds_transpose_load_raw(vec8_ty, lds_base_idx, elem_off)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        # --- K-subtile compute ---
        def _load_b_frags(b_lds_buffer, b_bases, ks):
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

        def compute_tile(accs_in, lds_a_idx, lds_b_idx,
                         emit_filler=None, mid_compute_callback=None,
                         fence_wait_fn=None):
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_idx)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_idx)
            if fence_wait_fn is not None:
                fence_wait_fn()

            if k_wmma_steps == 1:
                b_frags = _load_b_frags(b_buf, b_bases, 0)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, b_frags, 0,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback)
            else:
                prev_b = _load_b_frags(b_buf, b_bases, 0)
                for ks_step in range_constexpr(k_wmma_steps - 1):
                    _mid_cb = mid_compute_callback if ks_step == 0 else None
                    current_accs, prev_b = _a_streaming_compute(
                        current_accs, a_buf, a_bases, prev_b, ks_step,
                        mid_compute_callback=_mid_cb,
                        next_b_info=(b_buf, b_bases, ks_step + 1))
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, prev_b,
                    k_wmma_steps - 1,
                    emit_filler=emit_filler)

            return current_accs

        # --- Scheduling ---
        def hot_loop_scheduler():
            return
            if not use_half_streaming_schedule:
                rocdl.sched_barrier(0)
                return

            half_wm = wmma_m_rep // 2
            half_wmma = half_wm * wmma_n_rep
            a_half_loads = half_wm * DS_LOADS_PER_A_FRAG
            b_full_loads = wmma_n_rep * DS_LOADS_PER_B_FRAG

            for ks_step in range_constexpr(k_wmma_steps):
                if ks_step == 0:
                    rocdl.sched_dsrd(b_full_loads + a_half_loads)
                else:
                    rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                if ks_step < k_wmma_steps - 1:
                    rocdl.sched_dsrd(b_full_loads)
            rocdl.sched_barrier(0)

        # --- Epilogue helpers ---
        _half_out = out_dtype in ("f16", "bf16")
        _out_elem = T.f16 if out_dtype == "f16" else (T.bf16 if out_dtype == "bf16" else None)

        def epilogue_prepare_addrs():
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

        if use_tdm_store and not IS_SPLIT_K:
            def epilogue_lds_stores(final_accs, d_buf, d_base):
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        imm = wm * WMMA_M * _lds_d_stride_elems + wn * _n_col_d_elems
                        store_acc_vec8_to_lds(
                            d_buf, d_base, imm, final_accs[idx], out_elem=_out_elem)

        # --- L2 prefetch ---
        _effective_l2_pf = l2_prefetch_distance

        def _l2_prefetch(k_base):
            if _effective_l2_pf <= 0:
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, ks_begin + pf_k), (tile_m, tile_k), (K, 1),
                elem_bytes=elem_bytes, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b, (ks_begin + pf_k, blk_n), (tile_k, tile_n), (N, 1),
                elem_bytes=elem_bytes, thread_id=tx, block_threads=block_threads)

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        # Build per-stage SmemPtrs
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

        # TDM store epilogue setup (non split-K only)
        if use_tdm_store and not IS_SPLIT_K:
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

        # --- TDM descriptor addr_lo management ---
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

        addr_lo_a = vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_a = vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[])
        addr_lo_b = vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_b = vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[])
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1

        # --- Prologue ---
        for i in range_constexpr(pre_loaded):
            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                pred_const, stages_a_lds_addr[i], addr_lo_a, addr_hi_a])
            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                pred_const, stages_b_lds_addr[i], addr_lo_b, addr_hi_b])
            issue_tdm_loads(
                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                wave_specialized=False)
            addr_lo_a = arith.addi(addr_lo_a, adv_a_i32)
            addr_lo_b = arith.addi(addr_lo_b, adv_b_i32)

        pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2),
                       use_cluster=False)

        # --- Main loop ---
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

        if loop_iters > 0:
            init_args = list(accs) + [addr_lo_a, addr_lo_b]

            for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                accs_in = list(state[:n_accs])
                cur_lo_a = state[n_accs]
                cur_lo_b = state[n_accs + 1]

                for buf_idx in range_constexpr(num_buffers):
                    load_stage = (buf_idx + num_buffers - 1) % num_buffers

                    pipeline_fence_signal(
                        outstanding=_fence_outstanding,
                        use_cluster=False)

                    addr_boxes = [[cur_lo_a], [cur_lo_b]]

                    def _mid_tdm(
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
                            wave_specialized=False)
                        _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                        _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)
                        _l2_prefetch(_k_off)

                    def _fence_wait():
                        pipeline_fence_wait(use_cluster=False)

                    rocdl.sched_barrier(0)
                    accs_in = compute_tile(
                        accs_in,
                        stages_a_idx[buf_idx],
                        stages_b_idx[buf_idx],
                        mid_compute_callback=_mid_tdm,
                        fence_wait_fn=_fence_wait)
                    cur_lo_a = addr_boxes[0][0]
                    cur_lo_b = addr_boxes[1][0]
                    hot_loop_scheduler()

                results = yield list(accs_in) + [cur_lo_a, cur_lo_b]

            accs = list(results[:n_accs])
            addr_lo_a = results[n_accs]
            addr_lo_b = results[n_accs + 1]

        # --- Tail ---
        _already_fenced = False
        if loop_iters > 0:
            pipeline_fence(outstanding=0, use_cluster=False)
            _already_fenced = True
        epi_addrs_box = [None]
        _tail_had_load = False
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if _outstanding == -1:
                if _tail_had_load:
                    pipeline_fence(outstanding=0, use_cluster=False)
                elif not _already_fenced:
                    pipeline_fence(outstanding=0, use_cluster=False)
                if (use_tdm_store and not IS_SPLIT_K):
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
                if _already_fenced:
                    _already_fenced = False
                else:
                    pipeline_fence_signal(outstanding=_outstanding,
                                          use_cluster=False)
                    pipeline_fence_wait(use_cluster=False)

                _tail_mid_cb = None
                if _load_stage is not None:
                    _tail_had_load = True
                    _tail_ab = [[addr_lo_a], [addr_lo_b]]

                    def _tail_mid(_ls=_load_stage, _ab=_tail_ab):
                        dg0_a = vector.from_elements(T.vec(4, T.i32), [
                            pred_const, stages_a_lds_addr[_ls],
                            _ab[0][0], addr_hi_a])
                        dg0_b = vector.from_elements(T.vec(4, T.i32), [
                            pred_const, stages_b_lds_addr[_ls],
                            _ab[1][0], addr_hi_b])
                        issue_tdm_loads(
                            tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                            tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                            wave_specialized=False)
                        _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                        _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)

                    _tail_mid_cb = _tail_mid

                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs,
                    stages_a_idx[_compute_stage],
                    stages_b_idx[_compute_stage],
                    mid_compute_callback=_tail_mid_cb)
                hot_loop_scheduler()

                if _load_stage is not None:
                    addr_lo_a = _tail_ab[0][0]
                    addr_lo_b = _tail_ab[1][0]

        # --- Epilogue ---
        if IS_SPLIT_K:
            # Split-K epilogue: stage accumulators to LDS, then cooperative
            # atomic fadd to global using a runtime loop (avoids excessive
            # instruction count from full unrolling).

            # Step 1: Write accumulators to LDS (unrolled per-warp, compact)
            #   WMMA 16x16x32 output layout per lane:
            #     lane16 -> M-row within 16x16 tile
            #     lane_kgrp -> N-column group (0=cols 0-7, 1=cols 8-15)
            #     8 consecutive N-elements per lane
            c_lds_smem = SmemPtr(arena_base_ptr, 0, elem_ty,
                                 shape=(_splitk_c_elems,))
            c_lds_memref = c_lds_smem.get()
            gpu.barrier()
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    acc_vec = accs[idx]  # vec<8, f32>
                    # Truncate f32 -> half
                    half_elems = []
                    for _e in range_constexpr(8):
                        ef = vector.extract(acc_vec, static_position=[_e],
                                            dynamic_position=[])
                        half_elems.append(ef.truncf(elem_ty))
                    half_vec = vector.from_elements(T.vec(8, elem_ty), half_elems)
                    # Compute LDS offset
                    lds_m = warp_m_base + arith.index(wm * WMMA_M) + lane16
                    lds_n = (warp_n_base + arith.index(wn * WMMA_N)
                             + lane_kgrp * arith.index(8))
                    lds_off = lds_m * arith.index(tile_n) + lds_n
                    vector.store(half_vec, c_lds_memref, [lds_off])
            SmemPtr._view_cache = None

            # Step 2: Split-K barrier (spin on counter until flag is set)
            gpu.barrier()
            init_cur = arith.constant(0, type=T.i32)
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq, cur,
                    arith.constant(0, type=T.i32)).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                counter_base_ptr3 = fly.extract_aligned_pointer_as_index(
                    _ptr_type, fly_values(COUNTER)[0])
                counter_base_int3 = llvm.PtrToIntOp(_i64_type, counter_base_ptr3).result
                counter_byte_off3 = arith.index_cast(
                    T.i64,
                    arith.index_cast(T.index, counter_idx) * arith.index(4))
                counter_ptr3 = llvm.IntToPtrOp(
                    _ptr_type,
                    llvm.AddOp(counter_base_int3, counter_byte_off3,
                               llvm.IntegerOverflowFlags(0)).result).result
                counter_ptr_v3 = counter_ptr3._value if hasattr(counter_ptr3, "_value") else counter_ptr3
                data = llvm.InlineAsmOp(
                    T.i32, [counter_ptr_v3],
                    "global_load_b32 $0, $1, off scope:SCOPE_DEV", "=v,v",
                    has_side_effects=True).result
                rocdl.s_wait_loadcnt(0)
                scf.YieldOp([data])
            gpu.barrier()

            # Step 3: Cooperative read from LDS + atomic fadd (runtime loop)
            out_raw = fly_values(arg_c)[0]
            out_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, out_raw)
            out_base_int = llvm.PtrToIntOp(_i64_type, out_base_ptr).result
            vec2_ty = T.vec(2, elem_ty)

            _ep_lo = _raw(arith.index(0))
            _ep_hi = _raw(arith.index(LDG_REG_C_COUNT))
            _ep_step = _raw(arith.index(1))
            _ep_loop = scf.ForOp(_ep_lo, _ep_hi, _ep_step, [])
            with ir.InsertionPoint(_ep_loop.body):
                _ep_iv = arith.ArithValue(_ep_loop.induction_variable)
                global_tid = _ep_iv * arith.index(block_threads) + tx
                m_local_idx = global_tid / arith.index(LDG_C_X_THREADS)
                n_local_idx = ((global_tid % arith.index(LDG_C_X_THREADS))
                               * arith.index(LDG_VEC_SIZE))
                m_global_idx = blk_m + m_local_idx
                n_global_idx = blk_n + n_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, m_idx)
                cond_boundary_if = scf.IfOp(
                    cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    # Load vec from LDS
                    lds_rd_off = m_local_idx * arith.index(tile_n) + n_local_idx
                    pk_val = vector.load_op(
                        T.vec(LDG_VEC_SIZE, elem_ty), c_lds_memref, [lds_rd_off])
                    base_byte_off = ((m_global_idx * n_stride + n_global_idx)
                                     * arith.index(elem_bytes))
                    # Atomic fadd in vec2 chunks (4 iterations, kept unrolled)
                    for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                        e0 = vector.extract(pk_val, static_position=[vec_idx * 2],
                                            dynamic_position=[])
                        e1 = vector.extract(pk_val, static_position=[vec_idx * 2 + 1],
                                            dynamic_position=[])
                        pair = vector.from_elements(vec2_ty, [e0, e1])
                        pair_byte = (base_byte_off
                                     + arith.index(vec_idx * 2 * elem_bytes))
                        pair_i64 = arith.index_cast(T.i64, pair_byte)
                        pair_addr = llvm.AddOp(
                            out_base_int, pair_i64,
                            llvm.IntegerOverflowFlags(0)).result
                        pair_ptr = llvm.IntToPtrOp(_ptr_type, pair_addr).result
                        pair_ptr_v = (pair_ptr._value
                                      if hasattr(pair_ptr, "_value") else pair_ptr)
                        pair_v = (pair._value
                                  if hasattr(pair, "_value") else pair)
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            pair_ptr_v,
                            pair_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )
                    scf.YieldOp([])
                scf.YieldOp([])

        elif use_tdm_store:
            # TDM store epilogue (non split-K)
            if d_need_epilogue_fence:
                pipeline_fence(outstanding=0, use_cluster=False)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            # Direct buffer_store epilogue (non split-K)
            rocdl.sched_barrier(0)
            epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (in_dtype, out_dtype, N, K, tile_m, tile_n, tile_k, SPLIT_K,
                 m_warp, n_warp, num_buffers, effective_waves_per_eu,
                 l2_prefetch_distance, use_tdm_store, inst_prefetch,
                 expert_sched_mode)

    @flyc.jit
    def launch_hgemm_splitk(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        COUNTER: fx.Tensor,
        signal_state: fx.Int32,
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

        launcher = kernel_hgemm_splitk(
            arg_c, arg_a, arg_b, i32_m, i32_n, COUNTER, signal_state)
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                if effective_waves_per_eu is not None:
                    _wpe = int(effective_waves_per_eu)
                    if _wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
        launcher.launch(
            grid=(gx, gy, SPLIT_K),
            block=(block_threads, 1, 1),
            stream=stream,
        )

        llvm_opts = {}
        if expert_sched_mode:
            llvm_opts["amdgpu-expert-scheduling-mode"] = True
        if inst_prefetch:
            llvm_opts["amdgpu-inst-prefetch-distance"] = 8
        if llvm_opts:
            launch_hgemm_splitk.compile_hints["llvm_options"] = llvm_opts

    return launch_hgemm_splitk


SPLIT_K_GLOBAL_SEMAPHORE = {}
SPLIT_K_GLOBAL_SEMAPHORE_STATE = {}


def hgemm_splitk_gfx1250_(
    c,
    a,
    b,
    hgemm_kwargs=None,
    stream=None,
):
    """Convenience wrapper for the gfx1250 split-K HGEMM kernel.

    Args:
        c: Output tensor (M x N), dtype f16 or bf16.
        a: Input A tensor (M x K), dtype f16 or bf16.
        b: Input B tensor (K x N), dtype f16 or bf16.
        hgemm_kwargs: Dict of compile options (tile_m, tile_n, tile_k, SPLIT_K, etc).
        stream: CUDA/HIP stream.
    """
    import torch

    global SPLIT_K_GLOBAL_SEMAPHORE
    global SPLIT_K_GLOBAL_SEMAPHORE_STATE

    if hgemm_kwargs is None:
        hgemm_kwargs = {}
    if stream is None:
        stream = torch.cuda.current_stream()

    M, K_ = a.shape[-2], a.shape[-1]
    K_b, N = b.shape[-2], b.shape[-1]
    assert K_ == K_b, f"A.shape[-1]={K_} != B.shape[-2]={K_b}"

    a = a.contiguous().view(-1, K_)
    M = a.shape[0]
    b = b.contiguous()
    c = c.contiguous().view(-1, N)
    assert c.shape[0] == M

    if a.dtype == torch.half:
        in_dtype = "fp16"
    elif a.dtype == torch.bfloat16:
        in_dtype = "bf16"
    else:
        raise NotImplementedError(f"Unsupported dtype: {a.dtype}")

    SPLIT_K = hgemm_kwargs.get('SPLIT_K', 1)

    # Semaphore management
    if SPLIT_K_GLOBAL_SEMAPHORE.get(stream, None) is None:
        SPLIT_K_GLOBAL_SEMAPHORE[stream] = torch.zeros(
            (3 * SPLIT_K_COUNTER_MAX_LEN,), dtype=torch.int32, device=a.device)
        SPLIT_K_GLOBAL_SEMAPHORE_STATE[stream] = 0
    signal_state = SPLIT_K_GLOBAL_SEMAPHORE_STATE[stream]
    semaphore = SPLIT_K_GLOBAL_SEMAPHORE[stream]

    tile_m = hgemm_kwargs.get('tile_m', 128)
    tile_n = hgemm_kwargs.get('tile_n', 256)
    tile_k = hgemm_kwargs.get('tile_k', 128)

    # Pad M, N to tile multiples
    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n

    launch_fn = compile_hgemm_splitk_gfx1250(
        N=npad,
        K=K_,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        SPLIT_K=SPLIT_K,
        m_warp=hgemm_kwargs.get('m_warp', 2),
        n_warp=hgemm_kwargs.get('n_warp', 4),
        in_dtype=in_dtype,
        out_dtype=hgemm_kwargs.get('out_dtype', None),
        num_buffers=hgemm_kwargs.get('num_buffers', 2),
        waves_per_eu=hgemm_kwargs.get('waves_per_eu', None),
        l2_prefetch_distance=hgemm_kwargs.get('l2_prefetch_distance', 2),
        use_tdm_store=hgemm_kwargs.get('use_tdm_store', True),
        inst_prefetch=hgemm_kwargs.get('inst_prefetch', False),
        expert_sched_mode=hgemm_kwargs.get('expert_sched_mode', True),
    )

    # Pad input tensors if needed
    if M != mpad or N != npad:
        a_pad = torch.zeros((mpad, K_), dtype=a.dtype, device=a.device)
        a_pad[:M, :] = a
        b_pad = torch.zeros((K_, npad), dtype=b.dtype, device=b.device)
        b_pad[:, :N] = b
        c_pad = torch.zeros((mpad, npad), dtype=c.dtype, device=c.device)
    else:
        a_pad = a
        b_pad = b
        c_pad = c

    if SPLIT_K > 1:
        bm = (mpad + tile_m - 1) // tile_m
        bn = npad // tile_n
        assert bm * bn <= SPLIT_K_COUNTER_MAX_LEN

    launch_fn(
        c_pad.view(-1), a_pad.view(-1), b_pad.view(-1),
        mpad, npad, semaphore, signal_state, stream)

    if M != mpad or N != npad:
        c[:M, :N] = c_pad[:M, :N]

    if SPLIT_K > 1:
        SPLIT_K_GLOBAL_SEMAPHORE_STATE[stream] = (signal_state + 1) % 3


__all__ = ["compile_hgemm_splitk_gfx1250", "hgemm_splitk_gfx1250_"]
