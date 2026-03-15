"""MXFP4 GEMM kernel for gfx1250.

Uses V_WMMA_SCALE_F32_16X16X128_F8F6F4 with FP4 (E2M1) data and E8M0 block scales.
Supports N-stage buffering (2/3/4), TDM async copy, cluster MCAST.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl._mlir.dialects import llvm as llvm_dialect, memref as memref_dialect
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

from kernels.layout_utils import idx2crd
from kernels.pipeline_utils import make_tail_plan

# WMMA tile dimensions for MXFP4
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
PACK_FACTOR = 2        # 2 FP4 elements per byte
SCALE_BLOCK = 32       # 32 FP4 elements per E8M0 scale
SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK  # 4

# LDS padding in bytes (4 DWORDs = 16 bytes, matches SP3)
LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16

_STAGE_NAMES = ("ping", "pong", "pang", "pung")


def compile_mxfp4_gemm(
    *,
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
):
    """Compile an MXFP4 GEMM kernel with TDM async copy and multi-stage buffering.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, stream)
    """
    _ = (M, N)
    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 1

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE

    packed_tile_k = tile_k // PACK_FACTOR  # bytes along K in LDS per row
    scale_k_per_tile = tile_k // SCALE_BLOCK
    K_packed = K // PACK_FACTOR
    K_scale = K // SCALE_BLOCK

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if packed_tile_k % 4 != 0:
        raise ValueError(f"packed_tile_k must be a multiple of 4, got {packed_tile_k}")
    if scale_k_per_tile % 4 != 0:
        raise ValueError(
            f"scale_k_per_tile must be a multiple of 4 (tile_k >= 128), got {scale_k_per_tile}")

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
            f"got {num_k_tiles}")

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep

    lds_a_stride_bytes = packed_tile_k + LDS_PAD_A_BYTES
    lds_b_stride_bytes = packed_tile_k + LDS_PAD_B_BYTES

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * lds_b_stride_bytes
    lds_a_scale_bytes = tile_m * scale_k_per_tile
    lds_b_scale_bytes = tile_n * scale_k_per_tile

    stage_allocators = []
    stage_a_data_off = []
    stage_b_data_off = []
    stage_a_scale_off = []
    stage_b_scale_off = []

    for i in range(num_buffers):
        name = _STAGE_NAMES[i]
        alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"mxfp4_{name}")

        off = alloc._align(alloc.ptr, 16)
        stage_a_data_off.append(off)
        alloc.ptr = off + lds_a_data_bytes

        off = alloc._align(alloc.ptr, 16)
        stage_b_data_off.append(off)
        alloc.ptr = off + lds_b_data_bytes

        off = alloc._align(alloc.ptr, 16)
        stage_a_scale_off.append(off)
        alloc.ptr = off + lds_a_scale_bytes

        off = alloc._align(alloc.ptr, 16)
        stage_b_scale_off.append(off)
        alloc.ptr = off + lds_b_scale_bytes

        stage_allocators.append(alloc)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _raw_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    # Number of TDM loads per step: A_data + B_data + A_scale + B_scale = 4
    TDM_LOADS_PER_STEP = 4

    # Scale tail plan outstanding values: make_tail_plan uses 2 (for fp16's A+B),
    # but MXFP4 has 4 loads per step (A_data + B_data + A_scale + B_scale).
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _raw_tail_plan
    ]

    # Number of LDS loads per K-subtile (for s_wait_dscnt):
    # A frag: wmma_m_rep * 2 ds_load_b128
    # B frag: wmma_n_rep * 2 ds_load_b128
    # A scale: wmma_m_rep ds_load_b32
    # B scale: wmma_n_rep ds_load_b32
    LOADS_PER_SUBTILE = wmma_m_rep * 2 + wmma_n_rep * 2 + wmma_m_rep + wmma_n_rep

    @flyc.kernel
    def kernel_mxfp4_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        # Disable VALU stall for back-to-back WMMA
        llvm_dialect.inline_asm(
            None, [],
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "", has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

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
            thr_coord[0], thr_coord[1], thr_coord[2], thr_coord[3])

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_idx = arith.index_cast(T.index, i32_n.ir_value())
        n_stride = arith.index(N)
        c_nrec = m_idx * n_stride * arith.index(4)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        def _get_lds_memref(lds_ptr):
            if isinstance(lds_ptr, SmemPtr):
                return get_op_result_or_value(lds_ptr.get())
            return get_op_result_or_value(lds_ptr)

        def copy_a_data_to_lds(k_base, lds_mem_ref):
            k_packed_off = k_base / arith.index(PACK_FACTOR)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=lds_mem_ref,
                global_offset=(blk_m, k_packed_off),
                tensor_shape=(tile_m, packed_tile_k),
                strides=(K_packed, 1),
                tile_shape=(tile_m, packed_tile_k),
                elem_bytes=1,
                pad_interval=packed_tile_k, pad_amount=LDS_PAD_A_BYTES,
                num_warps=num_warps,
                workgroup_mask=a_mcast_mask)
            tdm_ops.tensor_load_2d(desc)

        def copy_b_data_to_lds(k_base, lds_mem_ref):
            k_packed_off = k_base / arith.index(PACK_FACTOR)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=lds_mem_ref,
                global_offset=(blk_n, k_packed_off),
                tensor_shape=(tile_n, packed_tile_k),
                strides=(K_packed, 1),
                tile_shape=(tile_n, packed_tile_k),
                elem_bytes=1,
                pad_interval=packed_tile_k, pad_amount=LDS_PAD_B_BYTES,
                num_warps=num_warps,
                workgroup_mask=b_mcast_mask)
            tdm_ops.tensor_load_2d(desc)

        def copy_a_scale_to_lds(k_base, lds_mem_ref):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a_scale, lds_memref=lds_mem_ref,
                global_offset=(blk_m, k_scale_off),
                tensor_shape=(tile_m, scale_k_per_tile),
                strides=(K_scale, 1),
                tile_shape=(tile_m, scale_k_per_tile),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=num_warps,
                workgroup_mask=a_mcast_mask)
            tdm_ops.tensor_load_2d(desc)

        def copy_b_scale_to_lds(k_base, lds_mem_ref):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b_scale, lds_memref=lds_mem_ref,
                global_offset=(blk_n, k_scale_off),
                tensor_shape=(tile_n, scale_k_per_tile),
                strides=(K_scale, 1),
                tile_shape=(tile_n, scale_k_per_tile),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=num_warps,
                workgroup_mask=b_mcast_mask)
            tdm_ops.tensor_load_2d(desc)

        def issue_all_tdm_loads(k_base, a_mem, b_mem, as_mem, bs_mem):
            copy_a_data_to_lds(k_base, a_mem)
            copy_b_data_to_lds(k_base, b_mem)
            copy_a_scale_to_lds(k_base, as_mem)
            copy_b_scale_to_lds(k_base, bs_mem)

        elem_ty_lds = T.f16

        def _precompute_a_lane_bases(lds_ptr):
            """Precompute per-wm A fragment lane base addresses (in BYTES).

            Each lane loads 32 bytes = 64 FP4 (one K-half).
            lane16 → M-row, lane_kgrp → K-half (0 or 1).
            """
            lds_buffer = _get_lds_memref(lds_ptr)
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            k_half_off = lane_kgrp * arith.index(32)  # 32 bytes = 64 FP4
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
                bases.append(base)
            return lds_buffer, bases

        def _lds_load_b128(lds_buffer, byte_offset):
            """Load 16 bytes from LDS at given byte offset via ds_load_b128."""
            from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
            lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
            raw_memref = arith.unwrap(lds_buffer)
            lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)
            from flydsl.expr.arith import ArithValue as _AV
            total_byte = _AV(lds_base) + byte_offset
            addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
            ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)
            vec4_i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
            return llvm_dialect.load(vec4_i32_ty, ptr_val)

        def load_a_frag(lds_buffer, a_lane_base, ks):
            """Load one 16x128 FP4 A-fragment from LDS.

            Returns vector<8xi32> (8 VGPRs, 64 FP4 per lane).
            2 x ds_load_b128 via direct LDS pointer load.
            """
            k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR)  # bytes per K-subtile
            byte_off = a_lane_base + k_byte_off
            v0 = _lds_load_b128(lds_buffer, byte_off)
            v1 = _lds_load_b128(lds_buffer, byte_off + arith.index(16))
            return vector.shuffle(v0, v1, list(range(8)))

        def _precompute_b_lane_bases(lds_ptr):
            """Precompute per-wn B fragment lane base addresses (in BYTES).

            B stored as [tile_n, packed_tile_k + pad] in LDS.
            lane16 → N-row, lane_kgrp → K-half.
            """
            lds_buffer = _get_lds_memref(lds_ptr)
            row_base = (warp_n_base + lane16) * arith.index(lds_b_stride_bytes)
            k_half_off = lane_kgrp * arith.index(32)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                base = row_base + arith.index(wn * WMMA_N * lds_b_stride_bytes) + k_half_off
                bases.append(base)
            return lds_buffer, bases

        def load_b_frag(lds_buffer, b_lane_base, ks):
            """Load one 128x16 FP4 B-fragment from LDS. Same pattern as A."""
            k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR)
            byte_off = b_lane_base + k_byte_off
            v0 = _lds_load_b128(lds_buffer, byte_off)
            v1 = _lds_load_b128(lds_buffer, byte_off + arith.index(16))
            return vector.shuffle(v0, v1, list(range(8)))

        def _precompute_scale_lane_bases(lds_ptr, warp_base, reps):
            """Precompute scale lane bases (in BYTES).

            Scale LDS layout: [tile_m_or_n, scale_k_per_tile] bytes.
            lane16 → M/N row.
            """
            lds_buffer = _get_lds_memref(lds_ptr)
            row_base = (warp_base + lane16) * arith.index(scale_k_per_tile)
            bases = []
            for w in range_constexpr(reps):
                base = row_base + arith.index(w * WMMA_M * scale_k_per_tile)
                bases.append(base)
            return lds_buffer, bases

        def _shuffle_scale_i32(val):
            """Swap bytes 1 and 2 of an i32 scale value via v_perm_b32.

            FP4 data VGPR layout splits K=128 as:
              V0-V3 lanes 0-15: K=0..31,   V4-V7 lanes 0-15: K=64..95
              V0-V3 lanes 16-31: K=32..63,  V4-V7 lanes 16-31: K=96..127
            The WMMA_SCALE hardware processes data in VGPR-group-first order,
            so the scale i32 byte-to-K-block mapping is [0, 2, 1, 3]:
              byte0 → K=0..31, byte1 → K=64..95, byte2 → K=32..63, byte3 → K=96..127
            Memory stores scales sequentially [K0,K1,K2,K3], so we swap bytes 1↔2
            to produce [K0,K2,K1,K3] using a single v_perm_b32 instruction.
            """
            i32_ty = ir.IntegerType.get_signless(32)
            return llvm_dialect.inline_asm(
                i32_ty, [_raw(val) if not isinstance(val, ir.Value) else val],
                "v_perm_b32 $0, $1, $1, 0x03010200",
                "=v,v", has_side_effects=False,
            )

        def load_scale(lds_buffer, scale_base, ks):
            """Load scale for one 16x128 WMMA from LDS.

            Returns i32 (1 VGPR) containing 4 packed E8M0 scale values,
            shuffled to match the WMMA_SCALE instruction's byte-to-K-block
            mapping: [K0, K2, K1, K3].
            ds_load_b32 via direct LDS pointer load.
            """
            from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
            lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
            raw_memref = arith.unwrap(lds_buffer)
            lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)
            # scale_k_per_tile bytes per row, ks-th group = ks * SCALES_PER_WMMA bytes
            byte_off = scale_base + arith.index(ks * SCALES_PER_WMMA)
            from flydsl.expr.arith import ArithValue as _AV
            total_byte = _AV(lds_base) + byte_off
            addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
            ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)
            i32_ty = ir.IntegerType.get_signless(32)
            raw_scale = llvm_dialect.load(i32_ty, ptr_val)
            return _shuffle_scale_i32(raw_scale)

        def load_k_subtile_frags(a_buf, a_bases, b_buf, b_bases,
                                  as_buf, as_bases, bs_buf, bs_bases, ks):
            """Batch-load all A/B fragments and scales for one K-subtile."""
            # Load B frags first (gives more time for A frags to arrive)
            b_frags = [load_b_frag(b_buf, b_bases[wn], ks)
                       for wn in range_constexpr(wmma_n_rep)]
            a_frags = [load_a_frag(a_buf, a_bases[wm], ks)
                       for wm in range_constexpr(wmma_m_rep)]
            # Load scales
            b_scales = [load_scale(bs_buf, bs_bases[wn], ks)
                        for wn in range_constexpr(wmma_n_rep)]
            a_scales = [load_scale(as_buf, as_bases[wm], ks)
                        for wm in range_constexpr(wmma_m_rep)]
            return a_frags, b_frags, a_scales, b_scales

        def do_k_subtile_wmma(a_frags, b_frags, a_scales, b_scales, accs):
            """Execute all WMMAs for one K-subtile with scales.

            Uses wmma_scale_f32_16x16x128_f8f6f4 (gfx1250 wave32) with:
              fmtA=4 (FP4/E2M1), fmtB=4 (FP4/E2M1),
              scaleAType=0 (opsel lo), scaleBType=0 (opsel lo).
              fmtScaleA/B defaults to 0 (E8M0).

            Operands are passed as (B, A) instead of (A, B) to compensate
            for the WMMA output VGPR layout where lane16→col and
            lane_kgrp→row_group.  Swapping computes C^T, making the output
            match the epilogue's row-major store pattern.
            """
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_frags[wn], a_frags[wm], accs[idx],
                        b_scales[wn], a_scales[wm],
                        fmtA=4, fmtB=4,
                        scaleAType=0, scaleBType=0,
                    )
            return accs

        def compute_tile(accs_in, lds_a, lds_b, lds_as, lds_bs, emit_filler=None):
            current_accs = list(accs_in)

            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            as_buf, as_bases = _precompute_scale_lane_bases(lds_as, warp_m_base, wmma_m_rep)
            bs_buf, bs_bases = _precompute_scale_lane_bases(lds_bs, warp_n_base, wmma_n_rep)

            if k_wmma_steps == 1:
                frags = load_k_subtile_frags(
                    a_buf, a_bases, b_buf, b_bases,
                    as_buf, as_bases, bs_buf, bs_bases, 0)
                rocdl.s_wait_dscnt(0)
                if emit_filler is not None:
                    emit_filler()
                current_accs = do_k_subtile_wmma(*frags, current_accs)
            else:
                prev = load_k_subtile_frags(
                    a_buf, a_bases, b_buf, b_bases,
                    as_buf, as_bases, bs_buf, bs_bases, 0)

                # Main K-loop: overlap load[ks+1] with compute[ks]
                for ks in range_constexpr(k_wmma_steps - 1):
                    next_frags = load_k_subtile_frags(
                        a_buf, a_bases, b_buf, b_bases,
                        as_buf, as_bases, bs_buf, bs_bases, ks + 1)
                    rocdl.s_wait_dscnt(LOADS_PER_SUBTILE)
                    current_accs = do_k_subtile_wmma(*prev, current_accs)
                    prev = next_frags

                # Epilogue
                rocdl.s_wait_dscnt(0)
                if emit_filler is not None:
                    rocdl.sched_barrier(0)
                    emit_filler()
                current_accs = do_k_subtile_wmma(*prev, current_accs)

            return current_accs

        def hot_loop_scheduler():
            rocdl.sched_barrier(0)

        # --- Epilogue: vectorized buffer_store_b128 ---
        # WMMA output VGPR layout (wave32, 16x16 tile):
        #   lane16  (lane_id % 16) → N column
        #   lane_kgrp (lane_id / 16) → M row group (0=rows 0-7, 1=rows 8-15)
        #   element[i] → M row offset within group
        # We compensate by swapping A/B operands in the WMMA call (see
        # do_k_subtile_wmma) so the WMMA effectively computes C^T, making
        # the output VGPR layout match this epilogue's store pattern:
        #   lane16 → M row, lane_kgrp*8 + ele → N column group.
        def epilogue_prepare_addrs():
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

        def wait_and_barrier(outstanding=0):
            tdm_ops.tensor_wait(outstanding)
            gpu.barrier()

        def wait_and_cluster_barrier(outstanding=0):
            tdm_ops.tensor_wait(outstanding)
            if use_cluster:
                gpu.cluster_barrier()
            else:
                gpu.barrier()

        def _compute_and_schedule(accs_in, a, b, a_s, b_s):
            rocdl.sched_barrier(0)
            accs_out = compute_tile(accs_in, a, b, a_s, b_s)
            hot_loop_scheduler()
            return accs_out

        _effective_l2_pf = l2_prefetch_distance
        if use_cluster and l2_prefetch_distance > 0:
            _effective_l2_pf = max(1, l2_prefetch_distance - 1)

        def _l2_prefetch(k_base):
            if _effective_l2_pf <= 0:
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            pf_k_packed = pf_k / arith.index(PACK_FACTOR)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, pf_k_packed), (tile_m, packed_tile_k), (K_packed, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b, (blk_n, pf_k_packed), (tile_n, packed_tile_k), (K_packed, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        # Build per-stage SmemPtrs using f16 element type for addressing.
        # FP4 packed data (1 byte = 2 FP4) + scale (1 byte E8M0) both
        # addressed in f16 units (2 bytes). This matches the fp16 kernel's
        # proven vector.load_op pattern.
        lds_a_data_f16 = lds_a_data_bytes // 2
        lds_b_data_f16 = lds_b_data_bytes // 2
        lds_a_scale_f16 = lds_a_scale_bytes // 2
        lds_b_scale_f16 = lds_b_scale_bytes // 2

        base_ptrs = [sa.get_base() for sa in stage_allocators]

        stages_a = [
            SmemPtr(base_ptrs[i], stage_a_data_off[i], elem_ty_lds, shape=(lds_a_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(base_ptrs[i], stage_b_data_off[i], elem_ty_lds, shape=(lds_b_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_as = [
            SmemPtr(base_ptrs[i], stage_a_scale_off[i], elem_ty_lds, shape=(lds_a_scale_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_bs = [
            SmemPtr(base_ptrs[i], stage_b_scale_off[i], elem_ty_lds, shape=(lds_b_scale_f16,))
            for i in range_constexpr(num_buffers)
        ]

        # Get memrefs for TDM (raw memref values)
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_as_mem = [stages_as[i].get() for i in range_constexpr(num_buffers)]
        stages_bs_mem = [stages_bs[i].get() for i in range_constexpr(num_buffers)]

        # Prologue: load first (num_buffers - 1) tiles
        for i in range_constexpr(pre_loaded):
            issue_all_tdm_loads(
                arith.index(i * tile_k),
                stages_a_mem[i], stages_b_mem[i],
                stages_as_mem[i], stages_bs_mem[i])
        # Wait for all but the last batch of TDM loads
        wait_and_barrier(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

        # Main loop
        main_end = loop_iters * num_buffers * tile_k

        if loop_iters > 0:
            for iv, state in range(0, main_end, num_buffers * tile_k, init=list(accs)):
                accs_in = list(state)
                for s in range_constexpr(num_buffers):
                    _load_stage = (s + num_buffers - 1) % num_buffers
                    _load_k_off = (s + num_buffers - 1) * tile_k
                    issue_all_tdm_loads(
                        iv + arith.index(_load_k_off),
                        stages_a_mem[_load_stage], stages_b_mem[_load_stage],
                        stages_as_mem[_load_stage], stages_bs_mem[_load_stage])
                    _l2_prefetch(iv + arith.index(s * tile_k))
                    accs_in = _compute_and_schedule(
                        accs_in,
                        stages_a[s], stages_b[s],
                        stages_as[s], stages_bs[s])
                    if s == num_buffers - 1:
                        wait_and_cluster_barrier(outstanding=TDM_LOADS_PER_STEP)
                    else:
                        wait_and_barrier(outstanding=TDM_LOADS_PER_STEP)
                results = yield list(accs_in)
            accs = list(results)

        # Tail
        if loop_iters == 0 and use_cluster:
            gpu.cluster_barrier()
        _extra_j = 0
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if _load_stage is not None:
                _k_off = (_tail_start + pre_loaded + _extra_j) * tile_k
                issue_all_tdm_loads(
                    arith.index(_k_off),
                    stages_a_mem[_load_stage], stages_b_mem[_load_stage],
                    stages_as_mem[_load_stage], stages_bs_mem[_load_stage])
                _extra_j += 1
            if _outstanding == -1:
                epi_addrs_box = [None]

                def _emit_epi_addrs():
                    epi_addrs_box[0] = epilogue_prepare_addrs()

                accs = compute_tile(
                    accs,
                    stages_a[_compute_stage], stages_b[_compute_stage],
                    stages_as[_compute_stage], stages_bs[_compute_stage],
                    emit_filler=_emit_epi_addrs)
            else:
                accs = _compute_and_schedule(
                    accs,
                    stages_a[_compute_stage], stages_b[_compute_stage],
                    stages_as[_compute_stage], stages_bs[_compute_stage])
                if use_cluster and _load_stage is not None:
                    wait_and_cluster_barrier(outstanding=_outstanding)
                else:
                    wait_and_barrier(outstanding=_outstanding)

        epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, effective_waves_per_eu, l2_prefetch_distance,
                 cluster_m, cluster_n)

    @flyc.jit
    def launch_mxfp4_gemm(
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
            for alloc in stage_allocators:
                alloc.finalized = False
            for alloc in stage_allocators:
                alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        launcher = kernel_mxfp4_gemm(
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
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    return launch_mxfp4_gemm


__all__ = ["compile_mxfp4_gemm"]
