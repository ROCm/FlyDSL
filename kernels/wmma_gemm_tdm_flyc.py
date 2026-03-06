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
    waves_per_eu: int = None,
):
    """Compile a WMMA GEMM kernel with CK TDM V1 tile config.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, M, N, stream)

    Args:
        use_double_buffer: Enable ping-pong double buffering for LDS.
        use_cshuffle: Enable CShuffle epilogue for coalesced global writes.
        use_async_copy: Use TDM GLOBAL_LOAD_ASYNC_TO_LDS for A tile transfer.
        use_preshuffle: B matrix is pre-shuffled; load directly from global
                        to VGPR in WMMA fragment order (bypasses LDS for B).
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
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
    # TDM padding: pad_interval is set to tile_k at descriptor creation time
    # so that TDM inserts one pad per row, matching lds_stride = tile_k + LDS_PAD.
    # Constraint: tile_k must be a power-of-2 for TDM pad encoding (3-bit field).
    # Max pad_interval for f16: 512 elements (enc=7).
    if use_async_copy and (tile_k & (tile_k - 1)) != 0:
        use_async_copy = False

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
    # A in LDS: [tile_m, tile_k + pad]  (A is row-major [M, K])
    # B in LDS: [tile_n, tile_k + pad]  (B is row-major [N, K])
    lds_a_stride = tile_k + LDS_PAD_A   # 72
    lds_b_stride = tile_k + LDS_PAD_B   # 72

    _tdm_guard = LDS_PAD_A if use_async_copy else 0
    lds_a_elems = tile_m * lds_a_stride + _tdm_guard

    if use_preshuffle:
        lds_b_elems = 0  # B bypasses LDS
    else:
        lds_b_elems = tile_n * lds_b_stride + _tdm_guard

    lds_single_bytes = (lds_a_elems + lds_b_elems) * elem_bytes
    lds_cshuffle_elems = tile_m * tile_n if use_cshuffle else 0
    lds_cshuffle_bytes = lds_cshuffle_elems * 4 if use_cshuffle else 0

    if use_double_buffer:
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
    else:
        allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_tdm_smem")
        allocator_pong = allocator_ping

        total_bytes = max(lds_single_bytes, lds_cshuffle_bytes)
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
        total_vec_b = tile_n * (tile_k // 4)
        if total_vec_b % block_threads != 0:
            raise ValueError(f"vec copy B: {total_vec_b} not divisible by {block_threads}")
        vec_iters_b = total_vec_b // block_threads
    else:
        vec_iters_b = 0

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
    # Each WMMA k-step × n-repeat produces one vec<16, elem_ty> fragment.
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

        layout_vec_a = fx.make_layout((tile_m, tile_k // 4), (tile_k // 4, 1))
        if not use_preshuffle:
            layout_vec_b = fx.make_layout((tile_n, tile_k // 4), (tile_k // 4, 1))

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
        else:
            lds_b_ping = lds_b_pong = None
            lds_b_ping_mem = lds_b_pong_mem = None

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        a_nrec = m_idx * arith.index(K * elem_bytes)
        b_nrec = n_stride * arith.index(K * elem_bytes)
        c_nrec = m_idx * n_stride * arith.index(4)
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, num_records_bytes=a_nrec)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, num_records_bytes=b_nrec)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

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
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A,
                num_warps=num_warps,
                wave_id=wave_id,
            )
            tdm_ops.tensor_load_2d(desc)

        def copy_a_to_lds(k_base, lds_a_mem_ref):
            if use_async_copy:
                copy_a_to_lds_async(k_base, lds_a_mem_ref)
            else:
                copy_a_to_lds_vgpr(k_base, lds_a_mem_ref)

        # --- Copy B tile to LDS (VGPR path) ---
        def copy_b_to_lds_vgpr(k_base, lds_b_mem_ref):
            for t in range_constexpr(vec_iters_b):
                vec_idx = tx + arith.index(t * block_threads)
                b_crd = idx2crd(vec_idx, layout_vec_b)
                b_n = layout_get(b_crd, 0)
                b_kv = layout_get(b_crd, 1)
                b_k = b_kv * arith.index(4)

                g_off = (blk_n + b_n) * arith.index(K) + (k_base + b_k)
                v_i16 = buffer_ops.buffer_load(b_rsrc, g_off, vec_width=4, dtype=T.i16)
                v = vector.bitcast(vec4_elem_ty, v_i16)
                lds_off = b_n * arith.index(lds_b_stride) + b_k
                vector.store(v, lds_b_mem_ref, [lds_off])

        # --- Copy B tile to LDS (TDM descriptor async path) ---
        def copy_b_to_lds_async(k_base, lds_b_mem_ref):
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

        def copy_b_to_lds(k_base, lds_b_mem_ref):
            if use_async_copy:
                copy_b_to_lds_async(k_base, lds_b_mem_ref)
            else:
                copy_b_to_lds_vgpr(k_base, lds_b_mem_ref)

        # --- Load B fragment from preshuffled global (vectorized: 2×dwordx4) ---
        def load_b_frag_preshuffle(kblk_idx, ks, wn):
            """Load B fragment from preshuffled global memory.

            Each fragment is 16 contiguous-in-groups-of-8 f16 elements.
            We load each group of 8 as a single dwordx4 (16 bytes).

            Args:
                kblk_idx: K-tile index, MLIR index value.
                ks: K-WMMA step (Python int from range_constexpr).
                wn: N-warp repeat index (Python int from range_constexpr).
            """
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
            """Pre-load A fragment for (ks=0, wm=0) from LDS into VGPRs.

            Issued right after barrier so that LDS read latency is overlapped
            with B-tile loads / copy-next-tile work.
            """
            m_off = warp_m_off  # wm=0
            vals = []
            for k0 in range_constexpr(2):
                for k1 in range_constexpr(8):
                    kk = ((arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                          + arith.index(k1))
                    off = (m_off + lane16) * arith.index(lds_a_stride) + kk
                    vals.append(lds_a_ptr.load([off]))
            return vector.from_elements(T.vec(16, elem_ty), vals)

        # --- Pre-load all B fragments for a k-tile (preshuffle cross-iter prefetch) ---
        def prefetch_b_tile_ps(kblk_idx):
            frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wn in range_constexpr(wmma_n_rep):
                    frags.append(load_b_frag_preshuffle(kblk_idx, ks, wn))
            return frags

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
                                vals.append(lds_b_ptr.load([off]))
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

        # --- Scheduling hints (machine instruction counts) ---
        # LLVM coalesces 8 scalar ds_load_b16 into 1 ds_load_b128.
        # a0_prefetch (1 fragment = 2 ds_load_b128) is outside the scheduling region.
        # B preshuffle VMEM loads happen before sched_barrier(0) in compute_tile,
        # so they are NOT in this scheduling region.
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
                        buffer_ops.buffer_store(c_val, c_rsrc, c_off)

        def cshuffle_epilogue(final_accs):
            lds_out = SmemPtr(base_ptr_pong, pong_offset, T.f32,
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
                        v1 = vector.from_elements(T.vec(1, T.f32), [c_val])
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
                    frag_f32 = vector.load_op(vec_f32, lds_out, [lds_idx])
                    col_global = blk_n + col_local
                    idx_out = row_global * n_stride + col_global
                    byte_off = idx_out * arith.index(4)
                    buffer_ops.buffer_store(frag_f32, c_rsrc, byte_off,
                                            offset_is_bytes=True)

        # ====== Main pipeline (SCF for loop) ======
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        def wait_and_barrier(outstanding=0):
            if use_async_copy:
                tdm_ops.tensor_wait(outstanding)
            gpu.barrier()

        # --- State pack/unpack for SCF for (carries accs + a0_pf + b_tile) ---
        def _pack_state(accs_list, a0_pf, b_tile=None):
            st = list(accs_list) + [a0_pf]
            if use_preshuffle and b_tile is not None:
                st.extend(b_tile)
            return st

        def _unpack_state(state):
            a_out = list(state[:n_accs])
            a0_out = state[n_accs]
            bt_out = list(state[n_accs + 1 : n_accs + 1 + n_b_frags]) if use_preshuffle else None
            return a_out, a0_out, bt_out

        def _initial_b_load(k_base, kblk_idx, lds_b_mem_ref):
            """Issue initial B data load (preshuffle→VGPR, else→LDS)."""
            if use_preshuffle:
                return prefetch_b_tile_ps(kblk_idx)
            else:
                copy_b_to_lds(k_base, lds_b_mem_ref)
                return None

        if use_double_buffer and num_k_tiles >= 2:
            # --- Double-buffer 2-stage pipeline (SCF for) ---
            # Prefetch first tile → pong
            b_tile_init = _initial_b_load(arith.index(0), arith.index(0), lds_b_pong_mem)
            copy_a_to_lds(arith.index(0), lds_a_pong_mem)
            if use_preshuffle:
                rocdl.s_wait_loadcnt(0)
            wait_and_barrier()
            a0_pf_init = prefetch_a0_frag(lds_a_pong)

            if num_k_tiles >= 3:
                if (num_k_tiles % 2) == 1:
                    main_loop_bound = (num_k_tiles - 1) * tile_k
                else:
                    main_loop_bound = (num_k_tiles - 2) * tile_k

                init_state = _pack_state(accs, a0_pf_init, b_tile_init)
                for iv, state in range(0, main_loop_bound, tile_k * 2, init=init_state):
                    accs_in, a0_pf_in, b_tile_pong_in = _unpack_state(state)
                    kblk_pong = iv / arith.index(tile_k)
                    kblk_ping = kblk_pong + arith.index(1)

                    # --- Pong tile: prefetch next→ping, compute on pong ---
                    if use_preshuffle:
                        b_tile_ping = prefetch_b_tile_ps(kblk_ping)
                    next_k1 = iv + arith.index(tile_k)
                    copy_a_to_lds(next_k1, lds_a_ping_mem)
                    if not use_preshuffle:
                        copy_b_to_lds(next_k1, lds_b_ping_mem)
                    accs_in = compute_tile(
                        accs_in, lds_a_pong,
                        None if use_preshuffle else lds_b_pong,
                        kblk_idx=kblk_pong, a0_prefetch=a0_pf_in,
                        b_tile_prefetch=b_tile_pong_in,
                    )
                    hot_loop_scheduler()
                    if use_preshuffle:
                        rocdl.s_wait_loadcnt(0)
                    wait_and_barrier()
                    a0_pf_ping = prefetch_a0_frag(lds_a_ping)

                    # --- Ping tile: prefetch next→pong, compute on ping ---
                    kblk_next_pong = kblk_pong + arith.index(2)
                    if use_preshuffle:
                        b_tile_next_pong = prefetch_b_tile_ps(kblk_next_pong)
                    next_k2 = iv + arith.index(tile_k * 2)
                    copy_a_to_lds(next_k2, lds_a_pong_mem)
                    if not use_preshuffle:
                        copy_b_to_lds(next_k2, lds_b_pong_mem)
                    accs_in = compute_tile(
                        accs_in, lds_a_ping,
                        None if use_preshuffle else lds_b_ping,
                        kblk_idx=kblk_ping, a0_prefetch=a0_pf_ping,
                        b_tile_prefetch=b_tile_ping if use_preshuffle else None,
                    )
                    hot_loop_scheduler()
                    if use_preshuffle:
                        rocdl.s_wait_loadcnt(0)
                    wait_and_barrier()
                    a0_pf_pong_new = prefetch_a0_frag(lds_a_pong)

                    results = yield _pack_state(
                        accs_in, a0_pf_pong_new,
                        b_tile_next_pong if use_preshuffle else None,
                    )

                accs, a0_pf, b_tile_pong_final = _unpack_state(results)

                if (num_k_tiles % 2) == 1:
                    # Odd: one remaining pong tile (last tile, no prefetch)
                    last_kblk = arith.index(num_k_tiles - 1)
                    accs = compute_tile(
                        accs, lds_a_pong,
                        None if use_preshuffle else lds_b_pong,
                        kblk_idx=last_kblk, a0_prefetch=a0_pf,
                        b_tile_prefetch=b_tile_pong_final,
                    )
                else:
                    # Even: last pong+ping pair outside loop
                    last_pong_kblk = arith.index(num_k_tiles - 2)
                    last_ping_k = arith.index((num_k_tiles - 1) * tile_k)
                    last_ping_kblk = arith.index(num_k_tiles - 1)

                    if use_preshuffle:
                        b_tile_last_ping = prefetch_b_tile_ps(last_ping_kblk)
                    copy_a_to_lds(last_ping_k, lds_a_ping_mem)
                    if not use_preshuffle:
                        copy_b_to_lds(last_ping_k, lds_b_ping_mem)
                    accs = compute_tile(
                        accs, lds_a_pong,
                        None if use_preshuffle else lds_b_pong,
                        kblk_idx=last_pong_kblk, a0_prefetch=a0_pf,
                        b_tile_prefetch=b_tile_pong_final,
                    )
                    hot_loop_scheduler()
                    if use_preshuffle:
                        rocdl.s_wait_loadcnt(0)
                    wait_and_barrier()
                    a0_pf_ping = prefetch_a0_frag(lds_a_ping)

                    accs = compute_tile(
                        accs, lds_a_ping,
                        None if use_preshuffle else lds_b_ping,
                        kblk_idx=last_ping_kblk, a0_prefetch=a0_pf_ping,
                        b_tile_prefetch=b_tile_last_ping if use_preshuffle else None,
                    )
            else:
                # num_k_tiles == 2: just 1 pair, no loop needed
                if use_preshuffle:
                    b_tile_1 = prefetch_b_tile_ps(arith.index(1))
                copy_a_to_lds(arith.index(tile_k), lds_a_ping_mem)
                if not use_preshuffle:
                    copy_b_to_lds(arith.index(tile_k), lds_b_ping_mem)
                accs = compute_tile(
                    accs, lds_a_pong,
                    None if use_preshuffle else lds_b_pong,
                    kblk_idx=arith.index(0), a0_prefetch=a0_pf_init,
                    b_tile_prefetch=b_tile_init,
                )
                hot_loop_scheduler()
                if use_preshuffle:
                    rocdl.s_wait_loadcnt(0)
                wait_and_barrier()
                a0_pf_ping = prefetch_a0_frag(lds_a_ping)

                accs = compute_tile(
                    accs, lds_a_ping,
                    None if use_preshuffle else lds_b_ping,
                    kblk_idx=arith.index(1), a0_prefetch=a0_pf_ping,
                    b_tile_prefetch=b_tile_1 if use_preshuffle else None,
                )
        else:
            # --- Single-buffer 1-stage pipeline (SCF for) ---
            if num_k_tiles >= 2:
                b_tile_0 = _initial_b_load(arith.index(0), arith.index(0), lds_b_pong_mem)
                copy_a_to_lds(arith.index(0), lds_a_pong_mem)
                if use_preshuffle:
                    rocdl.s_wait_loadcnt(0)
                wait_and_barrier()
                a0_pf_init = prefetch_a0_frag(lds_a_pong)

                init_state = _pack_state(accs, a0_pf_init, b_tile_0)
                for iv, state in range(0, K - tile_k, tile_k, init=init_state):
                    accs_in, a0_pf_in, b_tile_in = _unpack_state(state)
                    kblk = iv / arith.index(tile_k)

                    accs_in = compute_tile(
                        accs_in, lds_a_pong,
                        None if use_preshuffle else lds_b_pong,
                        kblk_idx=kblk, a0_prefetch=a0_pf_in,
                        b_tile_prefetch=b_tile_in,
                    )
                    hot_loop_scheduler()
                    gpu.barrier()

                    next_k = iv + arith.index(tile_k)
                    next_kblk = kblk + arith.index(1)
                    if use_preshuffle:
                        b_tile_next = prefetch_b_tile_ps(next_kblk)
                    else:
                        copy_b_to_lds(next_k, lds_b_pong_mem)
                        b_tile_next = None
                    copy_a_to_lds(next_k, lds_a_pong_mem)
                    if use_preshuffle:
                        rocdl.s_wait_loadcnt(0)
                    wait_and_barrier()
                    a0_pf_next = prefetch_a0_frag(lds_a_pong)

                    results = yield _pack_state(accs_in, a0_pf_next, b_tile_next)

                accs, a0_pf, b_tile_last = _unpack_state(results)
                last_kblk = arith.index(num_k_tiles - 1)
                accs = compute_tile(
                    accs, lds_a_pong,
                    None if use_preshuffle else lds_b_pong,
                    kblk_idx=last_kblk, a0_prefetch=a0_pf,
                    b_tile_prefetch=b_tile_last,
                )
            else:
                b_tile_0 = _initial_b_load(arith.index(0), arith.index(0), lds_b_pong_mem)
                copy_a_to_lds(arith.index(0), lds_a_pong_mem)
                if use_preshuffle:
                    rocdl.s_wait_loadcnt(0)
                wait_and_barrier()
                a0_pf = prefetch_a0_frag(lds_a_pong)
                accs = compute_tile(
                    accs, lds_a_pong,
                    None if use_preshuffle else lds_b_pong,
                    kblk_idx=arith.index(0), a0_prefetch=a0_pf,
                    b_tile_prefetch=b_tile_0,
                )

        # --- Epilogue ---
        if use_cshuffle:
            cshuffle_epilogue(accs)
        else:
            direct_epilogue(accs)

    cache_tag = (in_dtype, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 use_double_buffer, use_cshuffle, use_async_copy, use_preshuffle,
                 waves_per_eu)

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
