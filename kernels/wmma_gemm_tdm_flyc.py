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
from flydsl._mlir.dialects import memref as memref_d
from flydsl.expr.gpu import lds_space
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value
from flydsl._mlir.extras import types as mlir_T

from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32

# LDS padding to avoid bank conflicts on 64-bank LDS (gfx1250).
# CK computes: pad_amount = get_n_dwords_per_128b() - 1 = 3 (dwords)
# PaddingDataAmount = (3+1) * 4 / sizeof(f16) = 8 elements.
LDS_PAD_A = 8  # elements, for A[tile_m, tile_k]
LDS_PAD_B = 8  # elements, for B[tile_k, tile_n]


def preshuffle_b_weight(b_tensor: torch.Tensor, tile_k: int = 64) -> torch.Tensor:
    """Pre-arrange B matrix [N, K] for WMMA fp16 consumption.

    The preshuffled layout groups each tile_k chunk of K into the WMMA
    fragment access pattern:
      B_preshuffle[N//16, K//tile_k, tile_k//16, 16, 16]
        where the last dims are [k_group_of_16, n_within_16, k_within_16]

    This allows direct loads from global memory in the WMMA fragment order.

    Args:
        b_tensor: [N, K] weight matrix in fp16 or bf16
        tile_k: K-dimension tile size (must divide K)
    Returns:
        Preshuffled tensor with same total elements, contiguous
    """
    N, K = b_tensor.shape
    assert K % tile_k == 0, f"K={K} must be divisible by tile_k={tile_k}"
    assert N % 16 == 0, f"N={N} must be divisible by 16"
    assert tile_k % 16 == 0, f"tile_k={tile_k} must be divisible by 16"

    # Reshape: [N, K] -> [N//16, 16, K//tile_k, tile_k//16, 16]
    #           dims:      [n_blk, n_inner, k_blk, k_grp, k_inner]
    b = b_tensor.reshape(N // 16, 16, K // tile_k, tile_k // 16, 16)
    b = b.permute(0, 2, 3, 1, 4)  # [N//16, K//tile_k, tile_k//16, 16(n), 16(k)]
    return b.contiguous().reshape(-1)


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
    use_cshuffle: bool = True,
    use_preshuffle: bool = False,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
):
    """Compile a WMMA GEMM kernel with TDM async copy and multi-stage buffering.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, M, N, stream)

    Args:
        use_cshuffle: Enable CShuffle epilogue for coalesced global writes.
        use_preshuffle: B matrix is pre-shuffled; load directly from global
                        to VGPR in WMMA fragment order (bypasses LDS for B).
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
    swap_operands = not use_cshuffle
    out_elem_bytes = 4  # always f32 output

    block_threads = m_warp * n_warp * WAVE_SIZE

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    # TDM async copy requires power-of-2 tile_k for pad encoding.
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2 for TDM async copy, got {tile_k}")

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
    min_k_tiles = num_buffers
    if num_k_tiles < min_k_tiles:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {min_k_tiles}, "
            f"got {num_k_tiles} (K={K}, tile_k={tile_k})"
        )

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16
    k_wmma_steps = tile_k // WMMA_K

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep

    # Padded LDS row strides (in elements)
    lds_a_stride = tile_k + LDS_PAD_A
    lds_b_stride = tile_k + LDS_PAD_B

    # TDM async copy guard padding
    lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A

    if use_preshuffle:
        lds_b_elems = 0
    else:
        lds_b_elems = tile_n * lds_b_stride + LDS_PAD_A

    lds_single_bytes = (lds_a_elems + lds_b_elems) * elem_bytes
    lds_cshuffle_bytes = tile_m * tile_n * out_elem_bytes if use_cshuffle else 0

    # Multi-stage LDS allocation (ping/pong[/pang])
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_ping")
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_pong")

    buf_size_bytes = max(lds_single_bytes, lds_cshuffle_bytes)
    buf_size_elems = buf_size_bytes // elem_bytes

    ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = ping_offset + buf_size_elems * elem_bytes

    pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = pong_offset + buf_size_elems * elem_bytes

    lds_a_offset_ping = ping_offset
    lds_b_offset_ping = ping_offset + lds_a_elems * elem_bytes
    lds_a_offset_pong = pong_offset
    lds_b_offset_pong = pong_offset + lds_a_elems * elem_bytes

    if use_triple_buffer:
        allocator_pang = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_pang")
        pang_offset = allocator_pang._align(allocator_pang.ptr, 16)
        allocator_pang.ptr = pang_offset + buf_size_elems * elem_bytes
        lds_a_offset_pang = pang_offset
        lds_b_offset_pang = pang_offset + lds_a_elems * elem_bytes
    else:
        allocator_pang = None

    # Unified double-buffer LDS allocation for non-preshuffle SCF loop path.
    # Both buffers live in a single global, enabling dynamic offset selection.
    allocator_dbuf = None
    if not use_triple_buffer and not use_preshuffle and not use_cshuffle:
        allocator_dbuf = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_dbuf")
        _dbuf0_off = allocator_dbuf._align(allocator_dbuf.ptr, 16)
        allocator_dbuf.ptr = _dbuf0_off + buf_size_elems * elem_bytes
        _dbuf1_off = allocator_dbuf._align(allocator_dbuf.ptr, 16)
        allocator_dbuf.ptr = _dbuf1_off + buf_size_elems * elem_bytes
        lds_a_off_b0 = _dbuf0_off
        lds_b_off_b0 = _dbuf0_off + lds_a_elems * elem_bytes
        lds_a_off_b1 = _dbuf1_off
        lds_b_off_b1 = _dbuf1_off + lds_a_elems * elem_bytes

    num_warps = m_warp * n_warp

    # Scheduling hints — all counts are MACHINE instruction counts.
    # The LLVM backend coalesces 8 contiguous scalar ds_load_b16 → 1 ds_load_b128,
    # so each 16-element fragment becomes 2 machine ds_load_b128 instructions.
    DS_COALESCE = 8
    total_wmma_insts = k_wmma_steps * wmma_m_rep * wmma_n_rep
    total_lds_reads = k_wmma_steps * (wmma_m_rep * 16 + (0 if use_preshuffle else wmma_n_rep * 16))

    # CShuffle epilogue constants
    CSHUFFLE_NLANE = 32
    CSHUFFLE_EVEC = 4 if (tile_n % (32 * 4)) == 0 else 2
    cshuffle_mlane = block_threads // CSHUFFLE_NLANE
    cshuffle_m_reps = tile_m // cshuffle_mlane
    cshuffle_n_reps = tile_n // (CSHUFFLE_NLANE * CSHUFFLE_EVEC)

    # Preshuffle B layout constants
    ps_k_groups = tile_k // 16
    ps_b_tile_stride = ps_k_groups * 16 * 16

    # B fragment count for preshuffle cross-iteration prefetch.
    n_b_frags = k_wmma_steps * wmma_n_rep if use_preshuffle else 0

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

        elem_ty = _elem_type()

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
        else:
            lds_b_ping = lds_b_pong = None
            lds_b_ping_mem = lds_b_pong_mem = None

        if use_triple_buffer:
            base_ptr_pang = allocator_pang.get_base()
            lds_a_pang = SmemPtr(base_ptr_pang, lds_a_offset_pang, elem_ty, shape=(lds_a_elems,))
            lds_a_pang_mem = lds_a_pang.get()
            if not use_preshuffle:
                lds_b_pang = SmemPtr(base_ptr_pang, lds_b_offset_pang, elem_ty, shape=(lds_b_elems,))
                lds_b_pang_mem = lds_b_pang.get()
            else:
                lds_b_pang = None
                lds_b_pang_mem = None

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        a_nrec = m_idx * arith.index(K * elem_bytes)
        b_nrec = n_stride * arith.index(K * elem_bytes)
        c_nrec = m_idx * n_stride * arith.index(out_elem_bytes)
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, num_records_bytes=a_nrec)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, num_records_bytes=b_nrec)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        # --- TDM async copy: A tile → LDS ---
        def copy_a_to_lds(k_base, lds_a_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a,
                lds_memref=lds_a_mem_ref,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=elem_bytes,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A,
                num_warps=num_warps,
                wave_id=wave_id,
            )
            tdm_ops.tensor_load_2d(desc)

        # --- TDM async copy: B tile → LDS ---
        def copy_b_to_lds(k_base, lds_b_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b,
                lds_memref=lds_b_mem_ref,
                global_offset=(blk_n, k_base),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=elem_bytes,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_B,
                num_warps=num_warps,
                wave_id=wave_id,
            )
            tdm_ops.tensor_load_2d(desc)

        # --- Load B fragment from preshuffled global (vectorized: 2×dwordx4) ---
        def load_b_frag_preshuffle(kblk_idx, ks, wn):
            n_blk = (blk_n + warp_n_off + arith.index(wn * WMMA_N)) / arith.index(16)
            num_k_tiles_c = arith.index(K // tile_k)
            base_off = (n_blk * num_k_tiles_c * arith.index(ps_b_tile_stride)
                        + kblk_idx * arith.index(ps_b_tile_stride))

            halves = []
            for k0 in range_constexpr(2):
                kk_start = (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                k_in_tile = arith.index(ks * WMMA_K) + kk_start
                k_grp = k_in_tile / arith.index(16)
                k_within_base = k_in_tile % arith.index(16)

                elem_idx = (base_off + k_grp * arith.index(256)
                            + lane16 * arith.index(16) + k_within_base)
                dw_idx = elem_idx / arith.index(2)
                v_i32x4 = buffer_ops.buffer_load(b_rsrc, dw_idx, vec_width=4, dtype=T.i32)
                halves.append(vector.bitcast(T.vec(8, elem_ty), v_i32x4))

            vals = []
            for h in range_constexpr(2):
                for i in range_constexpr(8):
                    vals.append(vector.extract(halves[h], static_position=[i],
                                               dynamic_position=[]))
            return vector.from_elements(T.vec(16, elem_ty), vals)

        # --- Prefetch first A fragment (ks=0, wm=0) for latency hiding ---
        def prefetch_a0_frag(lds_a_ptr):
            m_off = warp_m_off
            vals = []
            for k0 in range_constexpr(2):
                for k1 in range_constexpr(8):
                    kk = ((arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                          + arith.index(k1))
                    off = (m_off + lane16) * arith.index(lds_a_stride) + kk
                    vals.append(_lds_load(lds_a_ptr, [off]))
            return vector.from_elements(T.vec(16, elem_ty), vals)

        # --- Pre-load all B fragments for a k-tile (preshuffle cross-iter prefetch) ---
        def prefetch_b_tile_ps(kblk_idx):
            frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wn in range_constexpr(wmma_n_rep):
                    frags.append(load_b_frag_preshuffle(kblk_idx, ks, wn))
            return frags

        # --- LDS load helper: works with both SmemPtr and raw memref views ---
        def _lds_load(ptr_or_memref, indices):
            if isinstance(ptr_or_memref, SmemPtr):
                return ptr_or_memref.load(indices)
            idx_vals = [get_op_result_or_value(i) for i in indices]
            return memref_d.load(get_op_result_or_value(ptr_or_memref), idx_vals)

        # --- Compute on one LDS buffer ---
        def compute_tile(accs_in, lds_a_ptr, lds_b_ptr=None, kblk_idx=None,
                         a0_prefetch=None, b_tile_prefetch=None):
            rocdl.sched_barrier(0)
            current_accs = list(accs_in)
            for ks in range_constexpr(k_wmma_steps):
                k_step = arith.index(ks * WMMA_K)

                b_frags = []
                for wn in range_constexpr(wmma_n_rep):
                    if b_tile_prefetch is not None:
                        b_frags.append(b_tile_prefetch[ks * wmma_n_rep + wn])
                    elif use_preshuffle:
                        b_frags.append(load_b_frag_preshuffle(kblk_idx, ks, wn))
                    else:
                        n_off = warp_n_off + arith.index(wn * WMMA_N)
                        vals = []
                        for k0 in range_constexpr(2):
                            for k1 in range_constexpr(8):
                                kk = (k_step
                                      + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                                      + arith.index(k1))
                                off = (n_off + lane16) * arith.index(lds_b_stride) + kk
                                vals.append(_lds_load(lds_b_ptr, [off]))
                        b_frags.append(vector.from_elements(T.vec(16, elem_ty), vals))

                for wm in range_constexpr(wmma_m_rep):
                    if a0_prefetch is not None and ks == 0 and wm == 0:
                        a_frag = a0_prefetch
                    else:
                        m_off = warp_m_off + arith.index(wm * WMMA_M)
                        a_vals = []
                        for k0 in range_constexpr(2):
                            for k1 in range_constexpr(8):
                                kk = (k_step
                                      + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                                      + arith.index(k1))
                                off = (m_off + lane16) * arith.index(lds_a_stride) + kk
                                a_vals.append(_lds_load(lds_a_ptr, [off]))
                        a_frag = vector.from_elements(T.vec(16, elem_ty), a_vals)

                    for wn in range_constexpr(wmma_n_rep):
                        acc_idx = wm * wmma_n_rep + wn
                        src0 = b_frags[wn] if swap_operands else a_frag
                        src1 = a_frag if swap_operands else b_frags[wn]
                        current_accs[acc_idx] = wmma_op(
                            T.vec(8, T.f32),
                            src0,
                            src1,
                            current_accs[acc_idx],
                            signA=False, signB=False, modC=0,
                            reuseA=False, reuseB=False,
                        ).result
            return current_accs

        # --- Scheduling hints (machine instruction counts) ---
        _a0_prefetch_machine = 16 // DS_COALESCE  # = 2 ds_load_b128
        _n_dsrd_machine = total_lds_reads // DS_COALESCE
        _n_dsrd_eff = max(0, _n_dsrd_machine - _a0_prefetch_machine)
        _n_wmma = total_wmma_insts

        def _build_schedule(numer, denom):
            if denom <= 0:
                return []
            if numer <= 0:
                return [0] * denom
            out = []
            prev = 0
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

        # --- Epilogue (all paths output f32) ---

        def direct_epilogue(final_accs):
            # Operand-swap mode: D' = B × A^T = C^T, stored in D register as
            # VGPR → N dimension, Lane → M dimension.
            # 4 consecutive VGPRs map to 4 consecutive N positions → buffer_store_b128.
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    acc_idx = wm * wmma_n_rep + wn
                    row = blk_m + warp_m_off + arith.index(wm * WMMA_M) + lane16
                    col_base = (blk_n + warp_n_off + arith.index(wn * WMMA_N)
                                + lane_kgrp * arith.index(8))
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        byte_off = (row * n_stride + col) * arith.index(4)
                        vals = []
                        for vi in range_constexpr(4):
                            vals.append(vector.extract(
                                final_accs[acc_idx],
                                static_position=[half * 4 + vi],
                                dynamic_position=[],
                            ))
                        vec4 = vector.from_elements(T.vec(4, T.f32), vals)
                        vec4_i32 = vector.bitcast(T.vec(4, T.i32), vec4)
                        buffer_ops.buffer_store(vec4_i32, c_rsrc, byte_off,
                                                offset_is_bytes=True)

        def cshuffle_epilogue(final_accs):
            lds_out = SmemPtr(base_ptr_pong, pong_offset, T.f32,
                              shape=(tile_m * tile_n,)).get()
            gpu.barrier()
            tile_n_idx = arith.index(tile_n)
            vec1_f32 = T.vec(1, T.f32)
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
                        v1 = vector.from_elements(vec1_f32, [c_val])
                        vector.store(v1, lds_out, [lds_idx], alignment=4)
            gpu.barrier()

            c_nlane = arith.index(CSHUFFLE_NLANE)
            c_evec = arith.index(CSHUFFLE_EVEC)
            m_lane = tx / c_nlane
            n_lane = tx % c_nlane
            vec_f32 = T.vec(CSHUFFLE_EVEC, T.f32)
            for mr in range_constexpr(cshuffle_m_reps):
                row_local = arith.index(mr * cshuffle_mlane) + m_lane
                row_global = blk_m + row_local
                for nr in range_constexpr(cshuffle_n_reps):
                    col_local = arith.index(nr * CSHUFFLE_NLANE * CSHUFFLE_EVEC) + n_lane * c_evec
                    lds_idx = row_local * tile_n_idx + col_local
                    frag = vector.load_op(vec_f32, lds_out, [lds_idx])
                    col_global = blk_n + col_local
                    idx_out = row_global * n_stride + col_global
                    byte_off = idx_out * arith.index(4)
                    frag_i32 = vector.bitcast(T.vec(CSHUFFLE_EVEC, T.i32), frag)
                    buffer_ops.buffer_store(frag_i32, c_rsrc, byte_off,
                                            offset_is_bytes=True)

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        def _pack_state_ps(accs_list, a0_pf, b_tile=None):
            st = list(accs_list) + [a0_pf]
            if b_tile is not None:
                st.extend(b_tile)
            return st

        def _unpack_state_ps(state):
            a_out = list(state[:n_accs])
            a0_out = state[n_accs]
            bt_out = list(state[n_accs + 1 : n_accs + 1 + n_b_frags])
            return a_out, a0_out, bt_out

        def _initial_b_load(k_base, kblk_idx, lds_b_mem_ref):
            if use_preshuffle:
                return prefetch_b_tile_ps(kblk_idx)
            else:
                copy_b_to_lds(k_base, lds_b_mem_ref)
                return None

        def wait_and_barrier(outstanding=0):
            tdm_ops.tensor_wait(outstanding)
            gpu.barrier()

        # Helper: issue TDM loads for a k-tile to given buffer
        def _issue_loads(k_base, lds_a_mem, lds_b_mem):
            copy_a_to_lds(k_base, lds_a_mem)
            if not use_preshuffle:
                copy_b_to_lds(k_base, lds_b_mem)

        # Helper: issue L2 prefetch for a k-tile
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

        # Helper: compute on a buffer with scheduling
        def _compute_and_schedule_ps(accs_in, lds_a, kblk_idx, a0_pf, b_tile_pf):
            accs_out = compute_tile(
                accs_in, lds_a, None,
                kblk_idx=kblk_idx, a0_prefetch=a0_pf,
                b_tile_prefetch=b_tile_pf,
            )
            hot_loop_scheduler()
            return accs_out

        # Helper: compute on a buffer with scheduling
        def _compute_and_schedule(accs_in, lds_a, lds_b):
            accs_out = compute_tile(accs_in, lds_a, lds_b)
            hot_loop_scheduler()
            return accs_out

        if use_triple_buffer:
            # ====== Triple-buffer 3-stage pipeline ======
            # Buffers: pong (buf0), ping (buf1), pang (buf2)
            # Pre-fill: load k=0 → pong, load k=1 → ping
            # Loop: step by 3, each iteration processes pong→ping→pang
            #   and pre-loads for the NEXT iteration (k+3→pong, k+4→ping).
            # After the loop, the remaining 2-4 tiles are handled statically.

            _tdm_loads_per_buf = 1 if use_preshuffle else 2

            # Pre-fill buf0 (pong): k=0
            _initial_b_load(arith.index(0), arith.index(0), lds_b_pong_mem)
            copy_a_to_lds(arith.index(0), lds_a_pong_mem)

            # Pre-fill buf1 (ping): k=1
            _initial_b_load(arith.index(tile_k), arith.index(1), lds_b_ping_mem)
            copy_a_to_lds(arith.index(tile_k), lds_a_ping_mem)

            if use_preshuffle:
                rocdl.s_wait_loadcnt(0)
            # Wait for buf0 — allow buf1 loads to stay in-flight
            wait_and_barrier(outstanding=_tdm_loads_per_buf)

            _safe_iters = max(0, num_k_tiles - 2) // 3
            _tiles_in_loop = _safe_iters * 3
            _tail_start = _tiles_in_loop
            _n_tail = num_k_tiles - _tail_start

            safe_loop_bound = _safe_iters * 3 * tile_k

            if use_preshuffle:
                # --- Preshuffle triple-buffer pipeline ---
                a0_pf_init = prefetch_a0_frag(lds_a_pong)
                b_tile_buf0 = prefetch_b_tile_ps(arith.index(0))

                if _safe_iters > 0:
                    init_state = _pack_state_ps(accs, a0_pf_init, b_tile_buf0)
                    for iv, state in range(0, safe_loop_bound, tile_k * 3, init=init_state):
                        accs_in, a0_pf_in, b_tile_pong_in = _unpack_state_ps(state)
                        kblk_pong = iv / arith.index(tile_k)
                        kblk_ping = kblk_pong + arith.index(1)
                        kblk_pang = kblk_pong + arith.index(2)

                        b_tile_pang_pf = prefetch_b_tile_ps(kblk_pang)
                        next_k2 = iv + arith.index(tile_k * 2)
                        _issue_loads(next_k2, lds_a_pang_mem, lds_b_pang_mem)
                        accs_in = _compute_and_schedule_ps(
                            accs_in, lds_a_pong, kblk_pong, a0_pf_in, b_tile_pong_in)
                        rocdl.s_wait_loadcnt(0)
                        wait_and_barrier(outstanding=_tdm_loads_per_buf)
                        a0_pf_ping = prefetch_a0_frag(lds_a_ping)

                        kblk_next_pong = kblk_pong + arith.index(3)
                        b_tile_next_pong = prefetch_b_tile_ps(kblk_next_pong)
                        b_tile_ping_pf = prefetch_b_tile_ps(kblk_ping)
                        next_k3 = iv + arith.index(tile_k * 3)
                        _issue_loads(next_k3, lds_a_pong_mem, lds_b_pong_mem)
                        accs_in = _compute_and_schedule_ps(
                            accs_in, lds_a_ping, kblk_ping, a0_pf_ping, b_tile_ping_pf)
                        rocdl.s_wait_loadcnt(0)
                        wait_and_barrier(outstanding=_tdm_loads_per_buf)
                        a0_pf_pang = prefetch_a0_frag(lds_a_pang)

                        next_k4 = iv + arith.index(tile_k * 4)
                        _issue_loads(next_k4, lds_a_ping_mem, lds_b_ping_mem)
                        accs_in = _compute_and_schedule_ps(
                            accs_in, lds_a_pang, kblk_pang, a0_pf_pang, b_tile_pang_pf)
                        rocdl.s_wait_loadcnt(0)
                        wait_and_barrier(outstanding=_tdm_loads_per_buf)
                        a0_pf_pong_new = prefetch_a0_frag(lds_a_pong)

                        results = yield _pack_state_ps(
                            accs_in, a0_pf_pong_new, b_tile_next_pong)

                    accs, a0_pf, b_tile_pong_cur = _unpack_state_ps(results)
                else:
                    a0_pf = a0_pf_init
                    b_tile_pong_cur = b_tile_buf0

                # Tail (preshuffle)
                t0 = _tail_start
                if _n_tail == 2:
                    accs = _compute_and_schedule_ps(
                        accs, lds_a_pong, arith.index(t0), a0_pf, b_tile_pong_cur)
                    rocdl.s_wait_loadcnt(0)
                    wait_and_barrier(outstanding=0)
                    a0_pf_1 = prefetch_a0_frag(lds_a_ping)
                    bt_last = prefetch_b_tile_ps(arith.index(t0 + 1))
                    accs = compute_tile(accs, lds_a_ping, None,
                                        kblk_idx=arith.index(t0 + 1),
                                        a0_prefetch=a0_pf_1, b_tile_prefetch=bt_last)
                elif _n_tail == 3:
                    _issue_loads(arith.index((t0 + 2) * tile_k), lds_a_pang_mem, lds_b_pang_mem)
                    accs = _compute_and_schedule_ps(
                        accs, lds_a_pong, arith.index(t0), a0_pf, b_tile_pong_cur)
                    rocdl.s_wait_loadcnt(0)
                    wait_and_barrier(outstanding=_tdm_loads_per_buf)
                    a0_pf_1 = prefetch_a0_frag(lds_a_ping)
                    bt_1 = prefetch_b_tile_ps(arith.index(t0 + 1))
                    accs = _compute_and_schedule_ps(
                        accs, lds_a_ping, arith.index(t0 + 1), a0_pf_1, bt_1)
                    rocdl.s_wait_loadcnt(0)
                    wait_and_barrier(outstanding=0)
                    a0_pf_2 = prefetch_a0_frag(lds_a_pang)
                    bt_2 = prefetch_b_tile_ps(arith.index(t0 + 2))
                    accs = compute_tile(accs, lds_a_pang, None,
                                        kblk_idx=arith.index(t0 + 2),
                                        a0_prefetch=a0_pf_2, b_tile_prefetch=bt_2)
                elif _n_tail == 4:
                    _issue_loads(arith.index((t0 + 2) * tile_k), lds_a_pang_mem, lds_b_pang_mem)
                    accs = _compute_and_schedule_ps(
                        accs, lds_a_pong, arith.index(t0), a0_pf, b_tile_pong_cur)
                    rocdl.s_wait_loadcnt(0)
                    wait_and_barrier(outstanding=_tdm_loads_per_buf)
                    a0_pf_1 = prefetch_a0_frag(lds_a_ping)
                    _issue_loads(arith.index((t0 + 3) * tile_k), lds_a_pong_mem, lds_b_pong_mem)
                    bt_1 = prefetch_b_tile_ps(arith.index(t0 + 1))
                    accs = _compute_and_schedule_ps(
                        accs, lds_a_ping, arith.index(t0 + 1), a0_pf_1, bt_1)
                    rocdl.s_wait_loadcnt(0)
                    wait_and_barrier(outstanding=_tdm_loads_per_buf)
                    a0_pf_2 = prefetch_a0_frag(lds_a_pang)
                    bt_2 = prefetch_b_tile_ps(arith.index(t0 + 2))
                    accs = _compute_and_schedule_ps(
                        accs, lds_a_pang, arith.index(t0 + 2), a0_pf_2, bt_2)
                    rocdl.s_wait_loadcnt(0)
                    wait_and_barrier(outstanding=0)
                    a0_pf_3 = prefetch_a0_frag(lds_a_pong)
                    bt_3 = prefetch_b_tile_ps(arith.index(t0 + 3))
                    accs = compute_tile(accs, lds_a_pong, None,
                                        kblk_idx=arith.index(t0 + 3),
                                        a0_prefetch=a0_pf_3, b_tile_prefetch=bt_3)
                else:
                    raise RuntimeError(f"Unexpected _n_tail={_n_tail}")

            else:
                # --- Non-preshuffle triple-buffer pipeline ---
                # Clean producer/consumer: no a0_prefetch, no b_tile_prefetch.
                # Loop state = accs only. Wait count = 2 (1 A + 1 B in flight).

                if _safe_iters > 0:
                    for iv, state in range(0, safe_loop_bound, tile_k * 3, init=list(accs)):
                        accs_in = list(state)

                        # Tile on pong (buf0): load k+2→pang, compute k
                        next_k2 = iv + arith.index(tile_k * 2)
                        _issue_loads(next_k2, lds_a_pang_mem, lds_b_pang_mem)
                        _l2_prefetch(iv)
                        accs_in = _compute_and_schedule(accs_in, lds_a_pong, lds_b_pong)
                        wait_and_barrier(outstanding=_tdm_loads_per_buf)

                        # Tile on ping (buf1): load k+3→pong, compute k+1
                        next_k1 = iv + arith.index(tile_k)
                        next_k3 = iv + arith.index(tile_k * 3)
                        _issue_loads(next_k3, lds_a_pong_mem, lds_b_pong_mem)
                        _l2_prefetch(next_k1)
                        accs_in = _compute_and_schedule(accs_in, lds_a_ping, lds_b_ping)
                        wait_and_barrier(outstanding=_tdm_loads_per_buf)

                        # Tile on pang (buf2): load k+4→ping, compute k+2
                        next_k4 = iv + arith.index(tile_k * 4)
                        _issue_loads(next_k4, lds_a_ping_mem, lds_b_ping_mem)
                        _l2_prefetch(next_k2)
                        accs_in = _compute_and_schedule(accs_in, lds_a_pang, lds_b_pang)
                        wait_and_barrier(outstanding=_tdm_loads_per_buf)

                        results = yield list(accs_in)

                    accs = list(results)
                # else: accs unchanged from init

                # Tail (non-preshuffle)
                t0 = _tail_start
                if _n_tail == 2:
                    accs = _compute_and_schedule(accs, lds_a_pong, lds_b_pong)
                    wait_and_barrier(outstanding=0)
                    accs = compute_tile(accs, lds_a_ping, lds_b_ping)
                elif _n_tail == 3:
                    _issue_loads(arith.index((t0 + 2) * tile_k), lds_a_pang_mem, lds_b_pang_mem)
                    accs = _compute_and_schedule(accs, lds_a_pong, lds_b_pong)
                    wait_and_barrier(outstanding=_tdm_loads_per_buf)
                    accs = _compute_and_schedule(accs, lds_a_ping, lds_b_ping)
                    wait_and_barrier(outstanding=0)
                    accs = compute_tile(accs, lds_a_pang, lds_b_pang)
                elif _n_tail == 4:
                    _issue_loads(arith.index((t0 + 2) * tile_k), lds_a_pang_mem, lds_b_pang_mem)
                    accs = _compute_and_schedule(accs, lds_a_pong, lds_b_pong)
                    wait_and_barrier(outstanding=_tdm_loads_per_buf)
                    _issue_loads(arith.index((t0 + 3) * tile_k), lds_a_pong_mem, lds_b_pong_mem)
                    accs = _compute_and_schedule(accs, lds_a_ping, lds_b_ping)
                    wait_and_barrier(outstanding=_tdm_loads_per_buf)
                    accs = _compute_and_schedule(accs, lds_a_pang, lds_b_pang)
                    wait_and_barrier(outstanding=0)
                    accs = compute_tile(accs, lds_a_pong, lds_b_pong)
                else:
                    raise RuntimeError(f"Unexpected _n_tail={_n_tail}")

        else:
            # ====== Double-buffer 2-stage pipeline ======

            if use_preshuffle:
                # Prologue for preshuffle: load first tile → pong
                _initial_b_load(arith.index(0), arith.index(0), lds_b_pong_mem)
                copy_a_to_lds(arith.index(0), lds_a_pong_mem)
                rocdl.s_wait_loadcnt(0)
                wait_and_barrier()

            if use_preshuffle:
                # --- Preshuffle double-buffer pipeline ---
                a0_pf_init = prefetch_a0_frag(lds_a_pong)
                b_tile_init = prefetch_b_tile_ps(arith.index(0))

                if num_k_tiles >= 3:
                    if (num_k_tiles % 2) == 1:
                        main_loop_bound = (num_k_tiles - 1) * tile_k
                    else:
                        main_loop_bound = (num_k_tiles - 2) * tile_k

                    init_state = _pack_state_ps(accs, a0_pf_init, b_tile_init)
                    for iv, state in range(0, main_loop_bound, tile_k * 2, init=init_state):
                        accs_in, a0_pf_in, b_tile_pong_in = _unpack_state_ps(state)
                        kblk_pong = iv / arith.index(tile_k)
                        kblk_ping = kblk_pong + arith.index(1)

                        b_tile_ping = prefetch_b_tile_ps(kblk_ping)
                        next_k1 = iv + arith.index(tile_k)
                        copy_a_to_lds(next_k1, lds_a_ping_mem)
                        accs_in = _compute_and_schedule_ps(
                            accs_in, lds_a_pong, kblk_pong, a0_pf_in, b_tile_pong_in)
                        rocdl.s_wait_loadcnt(0)
                        wait_and_barrier()
                        a0_pf_ping = prefetch_a0_frag(lds_a_ping)

                        kblk_next_pong = kblk_pong + arith.index(2)
                        b_tile_next_pong = prefetch_b_tile_ps(kblk_next_pong)
                        next_k2 = iv + arith.index(tile_k * 2)
                        copy_a_to_lds(next_k2, lds_a_pong_mem)
                        accs_in = _compute_and_schedule_ps(
                            accs_in, lds_a_ping, kblk_ping, a0_pf_ping, b_tile_ping)
                        rocdl.s_wait_loadcnt(0)
                        wait_and_barrier()
                        a0_pf_pong_new = prefetch_a0_frag(lds_a_pong)

                        results = yield _pack_state_ps(
                            accs_in, a0_pf_pong_new, b_tile_next_pong)

                    accs, a0_pf, b_tile_pong_final = _unpack_state_ps(results)

                    if (num_k_tiles % 2) == 1:
                        last_kblk = arith.index(num_k_tiles - 1)
                        accs = compute_tile(accs, lds_a_pong, None,
                                            kblk_idx=last_kblk, a0_prefetch=a0_pf,
                                            b_tile_prefetch=b_tile_pong_final)
                    else:
                        last_pong_kblk = arith.index(num_k_tiles - 2)
                        last_ping_k = arith.index((num_k_tiles - 1) * tile_k)
                        last_ping_kblk = arith.index(num_k_tiles - 1)

                        b_tile_last_ping = prefetch_b_tile_ps(last_ping_kblk)
                        copy_a_to_lds(last_ping_k, lds_a_ping_mem)
                        accs = _compute_and_schedule_ps(
                            accs, lds_a_pong, last_pong_kblk, a0_pf, b_tile_pong_final)
                        rocdl.s_wait_loadcnt(0)
                        wait_and_barrier()
                        a0_pf_ping = prefetch_a0_frag(lds_a_ping)
                        accs = compute_tile(accs, lds_a_ping, None,
                                            kblk_idx=last_ping_kblk,
                                            a0_prefetch=a0_pf_ping,
                                            b_tile_prefetch=b_tile_last_ping)
                else:
                    # num_k_tiles == 2
                    b_tile_1 = prefetch_b_tile_ps(arith.index(1))
                    copy_a_to_lds(arith.index(tile_k), lds_a_ping_mem)
                    accs = _compute_and_schedule_ps(
                        accs, lds_a_pong, arith.index(0), a0_pf_init, b_tile_init)
                    rocdl.s_wait_loadcnt(0)
                    wait_and_barrier()
                    a0_pf_ping = prefetch_a0_frag(lds_a_ping)
                    accs = compute_tile(accs, lds_a_ping, None,
                                        kblk_idx=arith.index(1),
                                        a0_prefetch=a0_pf_ping,
                                        b_tile_prefetch=b_tile_1)

            elif not use_cshuffle:
                # --- Non-preshuffle, non-cshuffle double-buffer SCF pipeline ---
                # Single-step K-loop with dynamic buffer selection via unified LDS.
                # Loop state = accs + [buf_flag]. buf_flag selects compute buffer.

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

                # Prologue: load k=0 → buf0, wait for completion
                copy_a_to_lds(arith.index(0), _mk_a_view(c_a0))
                copy_b_to_lds(arith.index(0), _mk_b_view(c_b0))
                wait_and_barrier()

                # Main loop: (num_k_tiles - 1) iterations, step by tile_k
                main_end = (num_k_tiles - 1) * tile_k
                init_st = list(accs) + [arith.index(0)]

                for iv, state in range(0, main_end, tile_k, init=init_st):
                    accs_in = list(state[:n_accs])
                    buf_flag = state[n_accs]

                    is_buf0 = arith.cmpi(arith.CmpIPredicate.eq, buf_flag, arith.index(0))

                    # Dynamic buffer offset selection
                    comp_a = arith.select(is_buf0, c_a0, c_a1)
                    comp_b = arith.select(is_buf0, c_b0, c_b1)
                    load_a = arith.select(is_buf0, c_a1, c_a0)
                    load_b = arith.select(is_buf0, c_b1, c_b0)

                    # Issue TDM loads for NEXT k-tile → load buffer
                    next_k = iv + arith.index(tile_k)
                    copy_a_to_lds(next_k, _mk_a_view(load_a))
                    copy_b_to_lds(next_k, _mk_b_view(load_b))
                    _l2_prefetch(iv)

                    # Compute on CURRENT buffer
                    accs_in = compute_tile(accs_in, _mk_a_view(comp_a), _mk_b_view(comp_b))
                    hot_loop_scheduler()
                    wait_and_barrier(outstanding=2)

                    # Flip buffer flag
                    next_flag = arith.select(is_buf0, arith.index(1), arith.index(0))
                    results = yield list(accs_in) + [next_flag]

                accs = list(results[:n_accs])
                last_flag = results[n_accs]

                # Tail: compute the last loaded tile (no more TDM loads)
                is_last_b0 = arith.cmpi(arith.CmpIPredicate.eq, last_flag, arith.index(0))
                tail_a = arith.select(is_last_b0, c_a0, c_a1)
                tail_b = arith.select(is_last_b0, c_b0, c_b1)
                accs = compute_tile(accs, _mk_a_view(tail_a), _mk_b_view(tail_b))

            else:
                # --- Non-preshuffle, cshuffle double-buffer pipeline ---
                # Uses original ping/pong allocators with tile_k*2 stepping.
                copy_b_to_lds(arith.index(0), lds_b_pong_mem)
                copy_a_to_lds(arith.index(0), lds_a_pong_mem)
                wait_and_barrier()
                a0_pf_init = prefetch_a0_frag(lds_a_pong)

                def _pack_state_cs(accs_list, a0_pf):
                    return list(accs_list) + [a0_pf]

                def _unpack_state_cs(state):
                    return list(state[:n_accs]), state[n_accs]

                if num_k_tiles >= 3:
                    if (num_k_tiles % 2) == 1:
                        main_loop_bound = (num_k_tiles - 1) * tile_k
                    else:
                        main_loop_bound = (num_k_tiles - 2) * tile_k

                    init_state = _pack_state_cs(accs, a0_pf_init)
                    for iv, state in range(0, main_loop_bound, tile_k * 2, init=init_state):
                        accs_in, a0_pf_in = _unpack_state_cs(state)
                        # --- Pong tile: prefetch next→ping, compute on pong ---
                        next_k1 = iv + arith.index(tile_k)
                        copy_a_to_lds(next_k1, lds_a_ping_mem)
                        copy_b_to_lds(next_k1, lds_b_ping_mem)
                        accs_in = compute_tile(accs_in, lds_a_pong, lds_b_pong,
                                               a0_prefetch=a0_pf_in)
                        hot_loop_scheduler()
                        wait_and_barrier()
                        a0_pf_ping = prefetch_a0_frag(lds_a_ping)

                        # --- Ping tile: prefetch next→pong, compute on ping ---
                        next_k2 = iv + arith.index(tile_k * 2)
                        copy_a_to_lds(next_k2, lds_a_pong_mem)
                        copy_b_to_lds(next_k2, lds_b_pong_mem)
                        accs_in = compute_tile(accs_in, lds_a_ping, lds_b_ping,
                                               a0_prefetch=a0_pf_ping)
                        hot_loop_scheduler()
                        wait_and_barrier()
                        a0_pf_pong_new = prefetch_a0_frag(lds_a_pong)

                        results = yield _pack_state_cs(accs_in, a0_pf_pong_new)

                    accs, a0_pf = _unpack_state_cs(results)

                    if (num_k_tiles % 2) == 1:
                        accs = compute_tile(accs, lds_a_pong, lds_b_pong,
                                            a0_prefetch=a0_pf)
                    else:
                        last_ping_k = arith.index((num_k_tiles - 1) * tile_k)
                        copy_a_to_lds(last_ping_k, lds_a_ping_mem)
                        copy_b_to_lds(last_ping_k, lds_b_ping_mem)
                        accs = compute_tile(accs, lds_a_pong, lds_b_pong,
                                            a0_prefetch=a0_pf)
                        wait_and_barrier()
                        a0_pf_ping = prefetch_a0_frag(lds_a_ping)
                        accs = compute_tile(accs, lds_a_ping, lds_b_ping,
                                            a0_prefetch=a0_pf_ping)
                else:
                    # num_k_tiles == 2
                    copy_a_to_lds(arith.index(tile_k), lds_a_ping_mem)
                    copy_b_to_lds(arith.index(tile_k), lds_b_ping_mem)
                    accs = compute_tile(accs, lds_a_pong, lds_b_pong,
                                        a0_prefetch=a0_pf_init)
                    wait_and_barrier()
                    a0_pf_ping = prefetch_a0_frag(lds_a_ping)
                    accs = compute_tile(accs, lds_a_ping, lds_b_ping,
                                        a0_prefetch=a0_pf_ping)

        # --- Epilogue ---
        if use_cshuffle:
            cshuffle_epilogue(accs)
        else:
            direct_epilogue(accs)

    cache_tag = (in_dtype, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 use_cshuffle, use_preshuffle, num_buffers, waves_per_eu,
                 l2_prefetch_distance)

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
        allocator_pong.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_ping.finalize()
            allocator_pong.finalize()
            if use_triple_buffer:
                allocator_pang.finalized = False
                allocator_pang.finalize()
            if allocator_dbuf is not None:
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


__all__ = ["compile_wmma_gemm_tdm", "preshuffle_b_weight"]
