"""TDM async copy WMMA GEMM kernel for gfx1250."""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32

# LDS padding to avoid bank conflicts on 64-bank LDS (gfx1250).
# CK computes: pad_amount = get_n_dwords_per_128b() - 1 = 3 (dwords)
# PaddingDataAmount = (3+1) * 4 / sizeof(f16) = 8 elements.
LDS_PAD_A = 8  # elements, for A[tile_m, tile_k]
LDS_PAD_B = 8  # elements, for B[tile_k, tile_n]

# TDM padding params in elements (converted to descriptor encoding by tdm_ops).
TDM_PAD_INTERVAL = 64  # elements (= tile_k for fp16, 64-bank LDS)
TDM_PAD_AMOUNT = 8     # elements (= LDS_PAD_A)


def preshuffle_b_weight(b_tensor: torch.Tensor, tile_k: int = 64) -> torch.Tensor:
    """Pre-arrange B matrix [K, N] for WMMA fp16 consumption.

    The preshuffled layout groups each tile_k chunk of K into the WMMA
    fragment access pattern:
      B_preshuffle[N//16, K//tile_k, tile_k//16, 16, 16]
        where the last dims are [k_group_of_16, n_within_16, k_within_16]

    This allows direct loads from global memory in the WMMA fragment order.

    Args:
        b_tensor: [K, N] weight matrix in fp16 or bf16
        tile_k: K-dimension tile size (must divide K)
    Returns:
        Preshuffled tensor with same total elements, contiguous
    """
    K, N = b_tensor.shape
    assert K % tile_k == 0, f"K={K} must be divisible by tile_k={tile_k}"
    assert N % 16 == 0, f"N={N} must be divisible by 16"
    assert tile_k % 16 == 0, f"tile_k={tile_k} must be divisible by 16"

    # Reshape: [K, N] -> [K//tile_k, tile_k, N//16, 16]
    b = b_tensor.reshape(K // tile_k, tile_k, N // 16, 16)
    # -> [N//16, K//tile_k, tile_k//16, 16, 16]  (n_blk, k_blk, k_grp, k_inner, n_inner)
    b = b.reshape(K // tile_k, tile_k // 16, 16, N // 16, 16)
    b = b.permute(3, 0, 1, 4, 2)  # [N//16, K//tile_k, tile_k//16, 16(n), 16(k)]
    return b.contiguous().reshape(-1)


def compile_wmma_gemm_tdm(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    m_warp: int = 2,
    n_warp: int = 4,
    in_dtype: str = "fp16",
    use_double_buffer: bool = True,
    use_cshuffle: bool = True,
    use_async_copy: bool = False,
    use_preshuffle: bool = False,
):
    """Compile a WMMA GEMM kernel with CK TDM V1 tile config.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, M, N, stream)

    Args:
        use_double_buffer: Enable ping-pong double buffering for LDS.
        use_cshuffle: Enable CShuffle epilogue for coalesced global writes.
        use_async_copy: Use TDM GLOBAL_LOAD_ASYNC_TO_LDS for A tile transfer.
        use_preshuffle: B matrix is pre-shuffled; load directly from global
                        to VGPR in WMMA fragment order (bypasses LDS for B).
    """
    _ = (M, N)
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

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(
            f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}"
        )
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(
            f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}"
        )

    num_k_tiles = K // tile_k
    if use_double_buffer and num_k_tiles < 2:
        use_double_buffer = False

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16
    k_wmma_steps = tile_k // WMMA_K  # 64/32 = 2

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    wmma_m_rep = warp_tile_m // WMMA_M  # 64/16 = 4
    wmma_n_rep = warp_tile_n // WMMA_N  # 32/16 = 2
    n_accs = wmma_m_rep * wmma_n_rep     # 4*2 = 8

    # Padded LDS row strides (in elements)
    lds_a_stride = tile_k + LDS_PAD_A   # 72
    lds_b_stride = tile_n + LDS_PAD_B   # 136

    lds_a_elems = tile_m * lds_a_stride  # 9216

    if use_preshuffle:
        lds_b_elems = 0  # B bypasses LDS
    else:
        lds_b_elems = tile_k * lds_b_stride  # 8704

    lds_single_bytes = (lds_a_elems + lds_b_elems) * elem_bytes
    lds_cshuffle_elems = tile_m * tile_n if use_cshuffle else 0

    if use_double_buffer:
        allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_ping")
        allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_pong")

        buf_size_bytes = max(lds_single_bytes, lds_cshuffle_elems * elem_bytes)
        buf_size_elems = buf_size_bytes // elem_bytes

        ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = ping_offset + buf_size_elems * elem_bytes

        pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = pong_offset + buf_size_elems * elem_bytes

        lds_a_offset_ping = ping_offset
        lds_b_offset_ping = ping_offset + lds_a_elems * elem_bytes
        lds_a_offset_pong = pong_offset
        lds_b_offset_pong = pong_offset + lds_a_elems * elem_bytes
    else:
        allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_smem")
        allocator_pong = allocator_ping

        total_bytes = max(lds_single_bytes, lds_cshuffle_elems * elem_bytes)
        total_elems = total_bytes // elem_bytes

        single_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = single_offset + total_elems * elem_bytes

        lds_a_offset_ping = single_offset
        lds_b_offset_ping = single_offset + lds_a_elems * elem_bytes
        lds_a_offset_pong = lds_a_offset_ping
        lds_b_offset_pong = lds_b_offset_ping

        pong_offset = single_offset
        ping_offset = single_offset

    # Vectorized copy config
    total_vec_a = tile_m * (tile_k // 4)
    if total_vec_a % block_threads != 0:
        raise ValueError(f"vec copy A: {total_vec_a} not divisible by {block_threads}")
    vec_iters_a = total_vec_a // block_threads

    if not use_preshuffle:
        total_vec_b = tile_k * (tile_n // 4)
        if total_vec_b % block_threads != 0:
            raise ValueError(f"vec copy B: {total_vec_b} not divisible by {block_threads}")
        vec_iters_b = total_vec_b // block_threads
    else:
        vec_iters_b = 0

    num_warps = m_warp * n_warp

    # Scheduling hints
    total_wmma_insts = k_wmma_steps * wmma_m_rep * wmma_n_rep
    total_lds_reads = k_wmma_steps * (wmma_m_rep * 16 + (0 if use_preshuffle else wmma_n_rep * 16))
    if total_wmma_insts >= total_lds_reads:
        wmma_per_lds = max(1, (total_wmma_insts + total_lds_reads - 1) // max(1, total_lds_reads))
        lds_per_wmma = 1
    else:
        wmma_per_lds = 1
        lds_per_wmma = max(1, (total_lds_reads + total_wmma_insts - 1) // max(1, total_wmma_insts))

    # CShuffle epilogue constants
    CSHUFFLE_NLANE = 32
    CSHUFFLE_EVEC = 2
    cshuffle_mlane = block_threads // CSHUFFLE_NLANE
    cshuffle_m_reps = tile_m // cshuffle_mlane
    cshuffle_n_reps = tile_n // (CSHUFFLE_NLANE * CSHUFFLE_EVEC)

    # Preshuffle B layout constants
    ps_k_groups = tile_k // 16
    ps_b_tile_stride = ps_k_groups * 16 * 16

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

        layout_thr = fx.make_layout(
            (m_warp * n_warp, WAVE_SIZE), (WAVE_SIZE, 1)
        )
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

        layout_vec_a = fx.make_layout((tile_m, tile_k // 4), (tile_k // 4, 1))
        if not use_preshuffle:
            layout_vec_b = fx.make_layout((tile_k, tile_n // 4), (tile_n // 4, 1))

        elem_ty = _elem_type()
        vec4_elem_ty = T.vec(4, elem_ty)

        base_ptr_ping = allocator_ping.get_base()
        base_ptr_pong = allocator_pong.get_base()

        lds_a_ping = SmemPtr(base_ptr_ping, lds_a_offset_ping, elem_ty, shape=(lds_a_elems,))
        lds_a_pong = SmemPtr(base_ptr_pong, lds_a_offset_pong, elem_ty, shape=(lds_a_elems,))
        lds_a_ping_mem = lds_a_ping.get()
        lds_a_pong_mem = lds_a_pong.get()

        if not use_preshuffle:
            lds_b_ping = SmemPtr(base_ptr_ping, lds_b_offset_ping, elem_ty, shape=(lds_b_elems,))
            lds_b_pong = SmemPtr(base_ptr_pong, lds_b_offset_pong, elem_ty, shape=(lds_b_elems,))
            lds_b_ping_mem = lds_b_ping.get()
            lds_b_pong_mem = lds_b_pong.get()

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)

        # --- Copy A tile to LDS (VGPR path) ---
        def copy_a_to_lds_vgpr(k_base, lds_a_mem_ref):
            for t in range_constexpr(vec_iters_a):
                vec_idx = tx + arith.index(t * block_threads)
                a_crd = idx2crd(vec_idx, layout_vec_a)
                a_m = layout_get(a_crd, 0)
                a_kv = layout_get(a_crd, 1)
                a_k = a_kv * arith.index(4)

                g_off = (blk_m + a_m) * arith.index(K) + (k_base + a_k)
                v_i16 = buffer_ops.buffer_load(a_rsrc, g_off, vec_width=4, dtype=T.i16)
                v = vector.bitcast(vec4_elem_ty, v_i16)
                lds_off = a_m * arith.index(lds_a_stride) + a_k
                vector.store(v, lds_a_mem_ref, [lds_off])

        # --- Copy A tile to LDS (TDM descriptor async path) ---
        def copy_a_to_lds_async(k_base, lds_a_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a,
                lds_memref=lds_a_mem_ref,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=elem_bytes,
                pad_interval=TDM_PAD_INTERVAL,
                pad_amount=TDM_PAD_AMOUNT,
                num_warps=num_warps,
                wave_id=wave_id,
            )
            tdm_ops.tensor_load_2d(desc)

        def copy_a_to_lds(k_base, lds_a_mem_ref):
            if use_async_copy:
                copy_a_to_lds_async(k_base, lds_a_mem_ref)
            else:
                copy_a_to_lds_vgpr(k_base, lds_a_mem_ref)

        # --- Copy B tile to LDS ---
        def copy_b_to_lds(k_base, lds_b_mem_ref):
            for t in range_constexpr(vec_iters_b):
                vec_idx = tx + arith.index(t * block_threads)
                b_crd = idx2crd(vec_idx, layout_vec_b)
                b_k = layout_get(b_crd, 0)
                b_nv = layout_get(b_crd, 1)
                b_n = b_nv * arith.index(4)

                g_off = (k_base + b_k) * n_stride + (blk_n + b_n)
                v_i16 = buffer_ops.buffer_load(b_rsrc, g_off, vec_width=4, dtype=T.i16)
                v = vector.bitcast(vec4_elem_ty, v_i16)
                lds_off = b_k * arith.index(lds_b_stride) + b_n
                vector.store(v, lds_b_mem_ref, [lds_off])

        # --- Load B fragment from preshuffled global ---
        def load_b_frag_preshuffle(kblk_idx, ks, wn):
            n_blk = (blk_n + warp_n_off + arith.index(wn * WMMA_N)) / arith.index(16)
            num_k_tiles_c = arith.index(K // tile_k)

            vals = []
            for k0 in range_constexpr(2):
                for k1 in range_constexpr(8):
                    kk = arith.index(k0 * 2) + lane_kgrp
                    kk = kk * arith.index(8) + arith.index(k1)
                    k_in_tile = arith.index(ks * WMMA_K) + kk
                    k_grp = k_in_tile / arith.index(16)
                    k_within = k_in_tile % arith.index(16)

                    idx = (n_blk * num_k_tiles_c * arith.index(ps_b_tile_stride)
                           + arith.index(kblk_idx) * arith.index(ps_b_tile_stride)
                           + k_grp * arith.index(256)
                           + lane16 * arith.index(16)
                           + k_within)
                    v_i16 = buffer_ops.buffer_load(b_rsrc, idx, vec_width=1, dtype=T.i16)
                    v_elem = vector.bitcast(T.vec(1, elem_ty),
                                            vector.from_elements(T.vec(1, T.i16), [v_i16]))
                    vals.append(vector.extract(v_elem, static_position=[0], dynamic_position=[]))
            return vector.from_elements(T.vec(16, elem_ty), vals)

        # --- Compute on one LDS buffer ---
        def compute_tile(accs_in, lds_a_ptr, lds_b_ptr, kblk_idx=0):
            current_accs = list(accs_in)
            for ks in range_constexpr(k_wmma_steps):
                k_step = arith.index(ks * WMMA_K)

                b_frags = []
                for wn in range_constexpr(wmma_n_rep):
                    if use_preshuffle:
                        b_frags.append(load_b_frag_preshuffle(kblk_idx, ks, wn))
                    else:
                        n_off = warp_n_off + arith.index(wn * WMMA_N)
                        vals = []
                        for k0 in range_constexpr(2):
                            for k1 in range_constexpr(8):
                                kk = (k_step
                                      + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                                      + arith.index(k1))
                                off = kk * arith.index(lds_b_stride) + (n_off + lane16)
                                vals.append(lds_b_ptr.load([off]))
                        b_frags.append(vector.from_elements(T.vec(16, elem_ty), vals))

                for wm in range_constexpr(wmma_m_rep):
                    m_off = warp_m_off + arith.index(wm * WMMA_M)
                    a_vals = []
                    for k0 in range_constexpr(2):
                        for k1 in range_constexpr(8):
                            kk = (k_step
                                  + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                                  + arith.index(k1))
                            off = (m_off + lane16) * arith.index(lds_a_stride) + kk
                            a_vals.append(lds_a_ptr.load([off]))
                    a_frag = vector.from_elements(T.vec(16, elem_ty), a_vals)

                    for wn in range_constexpr(wmma_n_rep):
                        acc_idx = wm * wmma_n_rep + wn
                        current_accs[acc_idx] = wmma_op(
                            T.vec(8, T.f32),
                            a_frag,
                            b_frags[wn],
                            current_accs[acc_idx],
                            signA=False, signB=False, modC=0,
                            reuseA=False, reuseB=False,
                        ).result
            return current_accs

        # --- Scheduling hints ---
        def hot_loop_scheduler():
            n_lds = total_lds_reads
            n_wmma = total_wmma_insts
            if n_wmma >= n_lds and n_lds > 0:
                for _ in range_constexpr(n_lds):
                    rocdl.sched_group_barrier(0x20, 1, 0)
                    rocdl.sched_group_barrier(0x08, wmma_per_lds, 0)
            elif n_lds > 0:
                for _ in range_constexpr(n_wmma):
                    rocdl.sched_group_barrier(0x20, lds_per_wmma, 0)
                    rocdl.sched_group_barrier(0x08, 1, 0)
            rocdl.sched_barrier(0)

        # --- Epilogue ---
        def direct_epilogue(final_accs):
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    acc_idx = wm * wmma_n_rep + wn
                    m_base = blk_m + warp_m_off + arith.index(wm * WMMA_M)
                    n_base = blk_n + warp_n_off + arith.index(wn * WMMA_N)
                    for mi in range_constexpr(8):
                        row = m_base + lane_kgrp * arith.index(8) + arith.index(mi)
                        col = n_base + lane16
                        c_off = row * n_stride + col
                        c_val = vector.extract(
                            final_accs[acc_idx],
                            static_position=[mi], dynamic_position=[],
                        )
                        fx.memref_store(c_val, arg_c, c_off)

        def cshuffle_epilogue(final_accs):
            lds_out = SmemPtr(base_ptr_pong, pong_offset, T.f16,
                              shape=(tile_m * tile_n,)).get()
            gpu.barrier()
            tile_n_idx = arith.index(tile_n)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    acc_idx = wm * wmma_n_rep + wn
                    m_base_local = warp_m_off + arith.index(wm * WMMA_M)
                    n_base_local = warp_n_off + arith.index(wn * WMMA_N)
                    for mi in range_constexpr(8):
                        row_local = m_base_local + lane_kgrp * arith.index(8) + arith.index(mi)
                        col_local = n_base_local + lane16
                        lds_idx = row_local * tile_n_idx + col_local
                        c_val = vector.extract(
                            final_accs[acc_idx],
                            static_position=[mi], dynamic_position=[],
                        )
                        val_f16 = arith.trunc_f(T.f16, c_val)
                        v1 = vector.from_elements(T.vec(1, T.f16), [val_f16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)
            gpu.barrier()

            c_nlane = arith.index(CSHUFFLE_NLANE)
            c_evec = arith.index(CSHUFFLE_EVEC)
            m_lane = tx / c_nlane
            n_lane = tx % c_nlane
            vec2_f16 = T.vec(CSHUFFLE_EVEC, T.f16)
            for mr in range_constexpr(cshuffle_m_reps):
                row_local = arith.index(mr * cshuffle_mlane) + m_lane
                row_global = blk_m + row_local
                for nr in range_constexpr(cshuffle_n_reps):
                    col_local = arith.index(nr * CSHUFFLE_NLANE * CSHUFFLE_EVEC) + n_lane * c_evec
                    lds_idx = row_local * tile_n_idx + col_local
                    frag = vector.load_op(vec2_f16, lds_out, [lds_idx])
                    col_global = blk_n + col_local
                    for ev in range_constexpr(CSHUFFLE_EVEC):
                        val_f16 = vector.extract(frag, static_position=[ev], dynamic_position=[])
                        val_f32 = arith.extf(T.f32, val_f16)
                        c_off = row_global * n_stride + (col_global + arith.index(ev))
                        fx.memref_store(val_f32, arg_c, c_off)

        # ====== Main pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        rocdl.sched_barrier(0)

        def wait_and_barrier(outstanding=0):
            if use_async_copy:
                tdm_ops.tensor_wait(outstanding)
            gpu.barrier()

        if use_double_buffer and num_k_tiles >= 2:
            # --- Double-buffer pipeline ---
            copy_a_to_lds(arith.index(0), lds_a_pong_mem)
            if not use_preshuffle:
                copy_b_to_lds(arith.index(0), lds_b_pong_mem)
            wait_and_barrier()

            for kblk in range_constexpr(num_k_tiles):
                is_even = (kblk % 2) == 0

                if kblk < num_k_tiles - 1:
                    next_k = arith.index((kblk + 1) * tile_k)
                    if is_even:
                        copy_a_to_lds(next_k, lds_a_ping_mem)
                        if not use_preshuffle:
                            copy_b_to_lds(next_k, lds_b_ping_mem)
                    else:
                        copy_a_to_lds(next_k, lds_a_pong_mem)
                        if not use_preshuffle:
                            copy_b_to_lds(next_k, lds_b_pong_mem)

                lds_b_cur = None if use_preshuffle else (lds_b_pong if is_even else lds_b_ping)
                accs = compute_tile(
                    accs,
                    lds_a_pong if is_even else lds_a_ping,
                    lds_b_cur,
                    kblk_idx=kblk,
                )
                hot_loop_scheduler()
                wait_and_barrier()
        else:
            # --- Single-buffer pipeline ---
            for kblk in range_constexpr(num_k_tiles):
                k_base = arith.index(kblk * tile_k)
                copy_a_to_lds(k_base, lds_a_pong_mem)
                if not use_preshuffle:
                    copy_b_to_lds(k_base, lds_b_pong_mem)
                wait_and_barrier()
                accs = compute_tile(
                    accs,
                    lds_a_pong,
                    None if use_preshuffle else lds_b_pong,
                    kblk_idx=kblk,
                )
                hot_loop_scheduler()
                gpu.barrier()

        # --- Epilogue ---
        if use_cshuffle:
            cshuffle_epilogue(accs)
        else:
            direct_epilogue(accs)

    cache_tag = (in_dtype, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 use_double_buffer, use_cshuffle, use_async_copy, use_preshuffle)

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
        allocator_ping.finalized = False
        if allocator_pong is not allocator_ping:
            allocator_pong.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_ping.finalize()
            if allocator_pong is not allocator_ping:
                allocator_pong.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        kernel_wmma_gemm_tdm(arg_c, arg_a, arg_b, i32_m, i32_n).launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_wmma_gemm_tdm


__all__ = ["compile_wmma_gemm_tdm", "preshuffle_b_weight"]
