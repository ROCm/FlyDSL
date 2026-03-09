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
from flydsl._mlir.dialects import memref as memref_d
from flydsl.expr.gpu import lds_space
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value
from flydsl._mlir.extras import types as mlir_T

from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32

LDS_PAD_A = 8
LDS_PAD_B = 8


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
    l2_prefetch_distance: int = 0,
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
    use_triple_buffer = num_buffers == 3
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

    if use_triple_buffer:
        # Triple-buffer: 3 separate allocators (ping/pong/pang)
        allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_ping")
        allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_pong")
        allocator_pang = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_pang")

        ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = ping_offset + buf_size_elems * elem_bytes
        pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = pong_offset + buf_size_elems * elem_bytes
        pang_offset = allocator_pang._align(allocator_pang.ptr, 16)
        allocator_pang.ptr = pang_offset + buf_size_elems * elem_bytes

        lds_a_offset_ping = ping_offset
        lds_b_offset_ping = ping_offset + lds_a_elems * elem_bytes
        lds_a_offset_pong = pong_offset
        lds_b_offset_pong = pong_offset + lds_a_elems * elem_bytes
        lds_a_offset_pang = pang_offset
        lds_b_offset_pang = pang_offset + lds_a_elems * elem_bytes

        allocator_dbuf = None
    else:
        # Double-buffer: unified allocator with dynamic buffer selection
        allocator_ping = None
        allocator_pong = None
        allocator_pang = None

        allocator_dbuf = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_dbuf")
        _dbuf0_off = allocator_dbuf._align(allocator_dbuf.ptr, 16)
        allocator_dbuf.ptr = _dbuf0_off + buf_size_elems * elem_bytes
        _dbuf1_off = allocator_dbuf._align(allocator_dbuf.ptr, 16)
        allocator_dbuf.ptr = _dbuf1_off + buf_size_elems * elem_bytes

        lds_a_off_b0 = _dbuf0_off
        lds_b_off_b0 = _dbuf0_off + lds_a_elems * elem_bytes
        lds_a_off_b1 = _dbuf1_off
        lds_b_off_b1 = _dbuf1_off + lds_a_elems * elem_bytes

    # --- Scheduling hints ---
    DS_COALESCE = 8
    total_wmma_insts = k_wmma_steps * wmma_m_rep * wmma_n_rep
    total_lds_reads = k_wmma_steps * (wmma_m_rep + wmma_n_rep) * 16

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

        n_stride = arith.index_cast(T.index, i32_n.ir_value())
        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        # --- Thread/wave decomposition ---
        layout_thr = fx.make_layout(
            (m_warp * n_warp, WAVE_SIZE), (WAVE_SIZE, 1))
        layout_lane = fx.make_layout((2, 16), (16, 1))

        thr = idx2crd(tx, layout_thr)
        wave_id = layout_get(thr, 0)
        lane = layout_get(thr, 1)

        wave_m = wave_id / arith.index(n_warp)
        wave_n = wave_id % arith.index(n_warp)

        lc = idx2crd(lane, layout_lane)
        lane_kgrp = layout_get(lc, 0)
        lane16 = layout_get(lc, 1)

        warp_m_off = wave_m * arith.index(warp_tile_m)
        warp_n_off = wave_n * arith.index(warp_tile_n)

        elem_ty = _elem_type()

        # --- Buffer resources ---
        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        a_nrec = m_idx * arith.index(K * elem_bytes)
        b_nrec = n_stride * arith.index(K * elem_bytes)
        c_nrec = m_idx * n_stride * arith.index(4)  # f32 output
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, num_records_bytes=a_nrec)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, num_records_bytes=b_nrec)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        # --- TDM async copy helpers ---
        def copy_a_to_lds(k_base, lds_a_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=lds_a_mem_ref,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k), strides=(K, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_A,
                num_warps=num_warps, wave_id=wave_id)
            tdm_ops.tensor_load_2d(desc)

        def copy_b_to_lds(k_base, lds_b_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=lds_b_mem_ref,
                global_offset=(blk_n, k_base),
                tensor_shape=(tile_n, tile_k), strides=(K, 1),
                tile_shape=(tile_n, tile_k), elem_bytes=elem_bytes,
                pad_interval=tile_k, pad_amount=LDS_PAD_B,
                num_warps=num_warps, wave_id=wave_id)
            tdm_ops.tensor_load_2d(desc)

        # --- LDS load helper ---
        def _lds_load(ptr_or_memref, indices):
            """Load a single element from LDS (SmemPtr or raw memref)."""
            if isinstance(ptr_or_memref, SmemPtr):
                return ptr_or_memref.load(indices)
            idx_vals = [get_op_result_or_value(i) for i in indices]
            return memref_d.load(get_op_result_or_value(ptr_or_memref), idx_vals)

        # --- WMMA fragment loader ---
        def load_wmma_frag(lds_ptr, row_offset, k_offset, lds_stride):
            """Load one 16x32 WMMA fragment from LDS.

            Each lane loads 16 elements in 2 groups of 8 consecutive elements.
            Row = row_offset + lane16, Col = k_offset + (k0*2 + lane_kgrp)*8 + k1
            """
            vals = []
            for k0 in range_constexpr(2):
                for k1 in range_constexpr(8):
                    kk = k_offset + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8) + arith.index(k1)
                    off = (row_offset + lane16) * arith.index(lds_stride) + kk
                    vals.append(_lds_load(lds_ptr, [off]))
            return vector.from_elements(T.vec(16, elem_ty), vals)

        # --- Compute on one LDS buffer ---
        def compute_tile(accs_in, lds_a_ptr, lds_b_ptr):
            rocdl.sched_barrier(0)
            current_accs = list(accs_in)
            for ks in range_constexpr(k_wmma_steps):
                k_off = arith.index(ks * WMMA_K)

                b_frags = []
                for wn in range_constexpr(wmma_n_rep):
                    b_frags.append(load_wmma_frag(
                        lds_b_ptr,
                        warp_n_off + arith.index(wn * WMMA_N),
                        k_off, lds_b_stride))

                for wm in range_constexpr(wmma_m_rep):
                    a_frag = load_wmma_frag(
                        lds_a_ptr,
                        warp_m_off + arith.index(wm * WMMA_M),
                        k_off, lds_a_stride)

                    for wn in range_constexpr(wmma_n_rep):
                        acc_idx = wm * wmma_n_rep + wn
                        current_accs[acc_idx] = wmma_op(
                            T.vec(8, T.f32),
                            b_frags[wn], a_frag,
                            current_accs[acc_idx],
                            signA=False, signB=False, modC=0,
                            reuseA=False, reuseB=False,
                        ).result
            return current_accs

        # --- Scheduling ---
        _n_dsrd_machine = total_lds_reads // DS_COALESCE
        _n_dsrd_eff = max(0, _n_dsrd_machine - (16 // DS_COALESCE))
        _n_wmma = total_wmma_insts

        def _build_schedule(numer, denom):
            if denom <= 0:
                return []
            if numer <= 0:
                return [0] * denom
            out, prev = [], 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        _dsrd_preload = min(2, _n_dsrd_eff)
        _dsrd_remaining = _n_dsrd_eff - _dsrd_preload
        _dsrd_sched = _build_schedule(_dsrd_remaining, _n_wmma) if _n_wmma > 0 else []

        def hot_loop_scheduler():
            if _n_wmma <= 0:
                rocdl.sched_barrier(0)
                return
            if _dsrd_preload > 0:
                rocdl.sched_group_barrier(0x100, _dsrd_preload, 0)
            for i in range_constexpr(_n_wmma):
                rocdl.sched_group_barrier(0x08, 1, 0)
                n_d = _dsrd_sched[i]
                if n_d > 0:
                    rocdl.sched_group_barrier(0x100, n_d, 0)
            rocdl.sched_barrier(0)

        # --- Epilogue: vectorized buffer_store_b128 ---
        def epilogue(final_accs):
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    acc_idx = wm * wmma_n_rep + wn
                    row = blk_m + warp_m_off + arith.index(wm * WMMA_M) + lane16
                    col_base = (blk_n + warp_n_off + arith.index(wn * WMMA_N)
                                + lane_kgrp * arith.index(8))
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        c_off = row * n_stride + col
                        vals = []
                        for vi in range_constexpr(4):
                            vals.append(vector.extract(
                                final_accs[acc_idx],
                                static_position=[half * 4 + vi],
                                dynamic_position=[],
                            ))
                        vec4 = vector.from_elements(T.vec(4, T.f32), vals)
                        buffer_ops.buffer_store(vec4, c_rsrc, c_off)

        # --- Pipeline helpers ---
        def wait_and_barrier(outstanding=0):
            tdm_ops.tensor_wait(outstanding)
            gpu.barrier()

        def _compute_and_schedule(accs_in, lds_a, lds_b):
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

        if use_triple_buffer:
            # ====== Triple-buffer 3-stage pipeline ======
            base_ptr_ping = allocator_ping.get_base()
            base_ptr_pong = allocator_pong.get_base()
            base_ptr_pang = allocator_pang.get_base()

            lds_a_ping = SmemPtr(base_ptr_ping, lds_a_offset_ping, elem_ty, shape=(lds_a_elems,))
            lds_a_pong = SmemPtr(base_ptr_pong, lds_a_offset_pong, elem_ty, shape=(lds_a_elems,))
            lds_a_pang = SmemPtr(base_ptr_pang, lds_a_offset_pang, elem_ty, shape=(lds_a_elems,))
            lds_b_ping = SmemPtr(base_ptr_ping, lds_b_offset_ping, elem_ty, shape=(lds_b_elems,))
            lds_b_pong = SmemPtr(base_ptr_pong, lds_b_offset_pong, elem_ty, shape=(lds_b_elems,))
            lds_b_pang = SmemPtr(base_ptr_pang, lds_b_offset_pang, elem_ty, shape=(lds_b_elems,))

            lds_a_ping_mem = lds_a_ping.get()
            lds_a_pong_mem = lds_a_pong.get()
            lds_a_pang_mem = lds_a_pang.get()
            lds_b_ping_mem = lds_b_ping.get()
            lds_b_pong_mem = lds_b_pong.get()
            lds_b_pang_mem = lds_b_pang.get()

            # Prologue: load first 2 tiles
            copy_b_to_lds(arith.index(0), lds_b_pong_mem)
            copy_a_to_lds(arith.index(0), lds_a_pong_mem)
            copy_b_to_lds(arith.index(tile_k), lds_b_ping_mem)
            copy_a_to_lds(arith.index(tile_k), lds_a_ping_mem)
            wait_and_barrier(outstanding=2)

            _safe_iters = max(0, num_k_tiles - 2) // 3
            _tiles_in_loop = _safe_iters * 3
            _tail_start = _tiles_in_loop
            _n_tail = num_k_tiles - _tail_start
            safe_loop_bound = _safe_iters * 3 * tile_k

            if _safe_iters > 0:
                for iv, state in range(0, safe_loop_bound, tile_k * 3, init=list(accs)):
                    accs_in = list(state)

                    copy_a_to_lds(iv + arith.index(tile_k * 2), lds_a_pang_mem)
                    copy_b_to_lds(iv + arith.index(tile_k * 2), lds_b_pang_mem)
                    _l2_prefetch(iv)
                    accs_in = _compute_and_schedule(accs_in, lds_a_pong, lds_b_pong)
                    wait_and_barrier(outstanding=2)

                    copy_a_to_lds(iv + arith.index(tile_k * 3), lds_a_pong_mem)
                    copy_b_to_lds(iv + arith.index(tile_k * 3), lds_b_pong_mem)
                    _l2_prefetch(iv + arith.index(tile_k))
                    accs_in = _compute_and_schedule(accs_in, lds_a_ping, lds_b_ping)
                    wait_and_barrier(outstanding=2)

                    copy_a_to_lds(iv + arith.index(tile_k * 4), lds_a_ping_mem)
                    copy_b_to_lds(iv + arith.index(tile_k * 4), lds_b_ping_mem)
                    _l2_prefetch(iv + arith.index(tile_k * 2))
                    accs_in = _compute_and_schedule(accs_in, lds_a_pang, lds_b_pang)
                    wait_and_barrier(outstanding=2)

                    results = yield list(accs_in)
                accs = list(results)

            t0 = _tail_start
            if _n_tail == 2:
                accs = _compute_and_schedule(accs, lds_a_pong, lds_b_pong)
                wait_and_barrier(outstanding=0)
                accs = compute_tile(accs, lds_a_ping, lds_b_ping)
            elif _n_tail == 3:
                copy_a_to_lds(arith.index((t0 + 2) * tile_k), lds_a_pang_mem)
                copy_b_to_lds(arith.index((t0 + 2) * tile_k), lds_b_pang_mem)
                accs = _compute_and_schedule(accs, lds_a_pong, lds_b_pong)
                wait_and_barrier(outstanding=2)
                accs = _compute_and_schedule(accs, lds_a_ping, lds_b_ping)
                wait_and_barrier(outstanding=0)
                accs = compute_tile(accs, lds_a_pang, lds_b_pang)
            elif _n_tail == 4:
                copy_a_to_lds(arith.index((t0 + 2) * tile_k), lds_a_pang_mem)
                copy_b_to_lds(arith.index((t0 + 2) * tile_k), lds_b_pang_mem)
                accs = _compute_and_schedule(accs, lds_a_pong, lds_b_pong)
                wait_and_barrier(outstanding=2)
                copy_a_to_lds(arith.index((t0 + 3) * tile_k), lds_a_pong_mem)
                copy_b_to_lds(arith.index((t0 + 3) * tile_k), lds_b_pong_mem)
                accs = _compute_and_schedule(accs, lds_a_ping, lds_b_ping)
                wait_and_barrier(outstanding=2)
                accs = _compute_and_schedule(accs, lds_a_pang, lds_b_pang)
                wait_and_barrier(outstanding=0)
                accs = compute_tile(accs, lds_a_pong, lds_b_pong)
            else:
                raise RuntimeError(f"Unexpected _n_tail={_n_tail}")

        else:
            # ====== Double-buffer 2-stage SCF pipeline ======
            base_uni = allocator_dbuf.get_base()
            _a_vtype = mlir_T.memref(lds_a_elems, elem_ty, memory_space=lds_space())
            _b_vtype = mlir_T.memref(lds_b_elems, elem_ty, memory_space=lds_space())

            def _mk_a_view(off_val):
                return memref_d.view(_a_vtype, base_uni, off_val, sizes=[])
            def _mk_b_view(off_val):
                return memref_d.view(_b_vtype, base_uni, off_val, sizes=[])

            c_a0 = arith.index(lds_a_off_b0)
            c_b0 = arith.index(lds_b_off_b0)
            c_a1 = arith.index(lds_a_off_b1)
            c_b1 = arith.index(lds_b_off_b1)

            # Prologue: load k=0 → buf0
            copy_a_to_lds(arith.index(0), _mk_a_view(c_a0))
            copy_b_to_lds(arith.index(0), _mk_b_view(c_b0))
            wait_and_barrier()

            # Main loop: each iteration loads NEXT tile, computes CURRENT
            main_end = (num_k_tiles - 1) * tile_k
            init_st = list(accs) + [arith.index(0)]

            for iv, state in range(0, main_end, tile_k, init=init_st):
                accs_in = list(state[:n_accs])
                buf_flag = state[n_accs]
                is_buf0 = arith.cmpi(arith.CmpIPredicate.eq, buf_flag, arith.index(0))

                comp_a = arith.select(is_buf0, c_a0, c_a1)
                comp_b = arith.select(is_buf0, c_b0, c_b1)
                load_a = arith.select(is_buf0, c_a1, c_a0)
                load_b = arith.select(is_buf0, c_b1, c_b0)

                next_k = iv + arith.index(tile_k)
                copy_a_to_lds(next_k, _mk_a_view(load_a))
                copy_b_to_lds(next_k, _mk_b_view(load_b))
                _l2_prefetch(iv)

                accs_in = compute_tile(accs_in, _mk_a_view(comp_a), _mk_b_view(comp_b))
                hot_loop_scheduler()
                wait_and_barrier(outstanding=2)

                next_flag = arith.select(is_buf0, arith.index(1), arith.index(0))
                results = yield list(accs_in) + [next_flag]

            accs = list(results[:n_accs])
            last_flag = results[n_accs]

            # Tail: compute the last loaded tile
            is_last_b0 = arith.cmpi(arith.CmpIPredicate.eq, last_flag, arith.index(0))
            tail_a = arith.select(is_last_b0, c_a0, c_a1)
            tail_b = arith.select(is_last_b0, c_b0, c_b1)
            accs = compute_tile(accs, _mk_a_view(tail_a), _mk_b_view(tail_b))

        epilogue(accs)

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
            if use_triple_buffer:
                allocator_ping.finalized = False
                allocator_pong.finalized = False
                allocator_pang.finalized = False
                allocator_ping.finalize()
                allocator_pong.finalize()
                allocator_pang.finalize()
            else:
                allocator_dbuf.finalized = False
                allocator_dbuf.finalize()

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
