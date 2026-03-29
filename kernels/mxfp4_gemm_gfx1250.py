"""MXFP4 GEMM kernel for gfx1250.

Uses V_WMMA_SCALE_F32_32X16X128_F4 with FP4 (E2M1) data and E8M0 block scales.
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
from kernels.gemm_gfx1250_common import (
    extract_lds_base_idx, get_lds_memref,
    lds_load_b128_raw,
    pipeline_fence, store_acc_vec8_to_buffer, store_acc_vec8_to_lds,
)
from kernels.pipeline_utils import make_tail_plan

# WMMA tile dimensions for MXFP4 (32x16x128 instruction)
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WMMA_N_EFF = 32   # Each 32x16 WMMA covers 32 N-cols (SRC0 after A/B swap)
WAVE_SIZE = 32
PACK_FACTOR = 2        # 2 FP4 elements per byte
SCALE_BLOCK = 32       # 32 FP4 elements per E8M0 scale
SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK  # 4

LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 0
LDS_PAD_D_BYTES = 16

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
    use_tdm_store: bool = True,
    out_dtype: str = "f32",
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    use_scale_opsel: bool = False,
):
    """Compile an MXFP4 GEMM kernel with TDM async copy and multi-stage buffering.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, stream)
    """
    _ = (M, N)
    if out_dtype not in ("f32", "bf16", "f16"):
        raise ValueError(f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}")
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4

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

    if wave_specialized_tdm and num_warps < 4:
        raise ValueError(
            f"wave_specialized_tdm requires num_warps >= 4, got {num_warps}")

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
    if warp_tile_n % WMMA_N_EFF != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N_EFF}")

    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles}")

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    n_accs = wmma_m_rep * wmma_n_rep
    b_scale_load_rep = warp_tile_n // WMMA_M

    lds_a_stride_bytes = packed_tile_k + LDS_PAD_A_BYTES
    lds_b_stride_bytes = packed_tile_k + LDS_PAD_B_BYTES

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * lds_b_stride_bytes
    _scale_guard_bytes = 16
    lds_a_scale_bytes = tile_m * scale_k_per_tile + _scale_guard_bytes
    lds_b_scale_bytes = tile_n * scale_k_per_tile + _scale_guard_bytes
    # Interleaved scale layout: [WMMA_M * m_or_n_warp, load_rep * scale_k_per_tile]
    interleaved_scale_cols_a = wmma_m_rep * scale_k_per_tile
    interleaved_scale_cols_b = b_scale_load_rep * scale_k_per_tile

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
    _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    if use_tdm_store:
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        _last_compute_stage = _base_tail_plan[-1][1]
        d_reuse_stage = 1 if _last_compute_stage == 0 else 0
        if total_d_bytes > stage_allocators[d_reuse_stage].ptr:
            stage_allocators[d_reuse_stage].ptr = total_d_bytes

    # Number of TDM loads per step: A_data + B_data + A_scale + B_scale = 4
    TDM_LOADS_PER_STEP = 4

    # Scale tail plan outstanding values: make_tail_plan returns (num_buffers-2)*2,
    # MXFP4 scales by TDM_LOADS_PER_STEP/2 for 4 loads per step.
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    # Pre-compute sub-tile layout
    _sub_tiles = []
    for _wm in range(wmma_m_rep):           # M direction, 16-row steps
        for _wn in range(wmma_n_rep):       # N direction, 32-col steps
            for _half in range(2):          # split 32-col into 2×16-col
                acc_idx = _wm * wmma_n_rep + _wn
                vec_base = _half * 8        # first/second 8 elements of vec16
                m_off = _wm * WMMA_M        # M offset (16-row granularity)
                n_sub = _wn * 2 + _half     # N sub-index (16-col granularity)
                _sub_tiles.append((acc_idx, vec_base, m_off, n_sub))

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
        # Enable back-to-back WMMA issue (SCHED_MODE bit[4] = DISABLE_VALU_STALL)
        rocdl.disable_xdl_arb_stall()

        # Wave 0 warms I-cache (~40 KB, 10 pages)
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
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3))

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_idx = arith.index_cast(T.index, i32_n.ir_value())
        n_stride = arith.index(N)
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        def make_desc_a(memref, k_base):
            k_packed_off = k_base / arith.index(PACK_FACTOR)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=memref,
                global_offset=(blk_m, k_packed_off),
                tensor_shape=(tile_m, packed_tile_k),
                strides=(K_packed, 1),
                tile_shape=(tile_m, packed_tile_k),
                elem_bytes=1,
                pad_interval=packed_tile_k, pad_amount=LDS_PAD_A_BYTES,
                num_warps=num_warps,
                workgroup_mask=a_mcast_mask)

        def make_desc_b(memref, k_base):
            k_packed_off = k_base / arith.index(PACK_FACTOR)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=memref,
                global_offset=(blk_n, k_packed_off),
                tensor_shape=(tile_n, packed_tile_k),
                strides=(K_packed, 1),
                tile_shape=(tile_n, packed_tile_k),
                elem_bytes=1,
                pad_interval=packed_tile_k, pad_amount=LDS_PAD_B_BYTES,
                num_warps=num_warps,
                workgroup_mask=b_mcast_mask)

        def make_desc_as(memref, k_base):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_m / arith.index(wmma_m_rep)
            inner_off = k_scale_off * arith.index(wmma_m_rep)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a_scale, lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(WMMA_M * m_warp, interleaved_scale_cols_a),
                strides=(wmma_m_rep * K_scale, 1),
                tile_shape=(WMMA_M * m_warp, interleaved_scale_cols_a),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=num_warps,
                workgroup_mask=a_mcast_mask)

        def make_desc_bs(memref, k_base):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_n / arith.index(b_scale_load_rep)
            inner_off = k_scale_off * arith.index(b_scale_load_rep)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b_scale, lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(WMMA_M * n_warp, interleaved_scale_cols_b),
                strides=(b_scale_load_rep * K_scale, 1),
                tile_shape=(WMMA_M * n_warp, interleaved_scale_cols_b),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=num_warps,
                workgroup_mask=b_mcast_mask)

        def issue_all_tdm_loads(desc_a, desc_b, desc_as, desc_bs):
            if wave_specialized_tdm:
                wid = rocdl.wave_id()
                wid_mod4 = arith.remui(wid, arith.constant(4, type=T.i32))
                if arith.cmpi(arith.CmpIPredicate.eq, wid_mod4, arith.constant(0, type=T.i32)):
                    tdm_ops.tensor_load_2d(desc_a)
                if arith.cmpi(arith.CmpIPredicate.eq, wid_mod4, arith.constant(1, type=T.i32)):
                    tdm_ops.tensor_load_2d(desc_b)
                if arith.cmpi(arith.CmpIPredicate.eq, wid_mod4, arith.constant(2, type=T.i32)):
                    tdm_ops.tensor_load_2d(desc_as)
                if arith.cmpi(arith.CmpIPredicate.eq, wid_mod4, arith.constant(3, type=T.i32)):
                    tdm_ops.tensor_load_2d(desc_bs)
            else:
                tdm_ops.tensor_load_2d(desc_a)
                tdm_ops.tensor_load_2d(desc_b)
                tdm_ops.tensor_load_2d(desc_as)
                tdm_ops.tensor_load_2d(desc_bs)

        elem_ty_lds = T.f16

        def _precompute_a_lane_bases(lds_ptr):
            """Precompute per-wm A fragment lane base addresses (byte offsets)."""
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            k_half_off = lane_kgrp * arith.index(32)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
                bases.append(base)
            return lds_ptr, bases

        def _select_base(bases_list, stage_idx):
            """Runtime select among pre-extracted LDS bases (2/3/4 stages)."""
            if len(bases_list) == 1:
                return bases_list[0]
            if len(bases_list) == 2:
                is_zero = arith.cmpi(
                    arith.CmpIPredicate.eq, stage_idx, arith.index(0))
                return arith.select(is_zero, bases_list[0], bases_list[1])
            result = bases_list[-1]
            _n = len(bases_list)
            for ii in range_constexpr(_n - 1):
                i = _n - 2 - ii
                is_i = arith.cmpi(
                    arith.CmpIPredicate.eq, stage_idx, arith.index(i))
                result = arith.select(is_i, bases_list[i], result)
            return result

        def load_a_frag(lds_buffer, a_lane_base, ks):
            """Load one 16x128 FP4 A-fragment from LDS.

            Returns vector<8xi32> (8 VGPRs, 64 FP4 per lane).
            2 x ds_load_b128.
            """
            k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR)
            byte_off = a_lane_base + k_byte_off
            v0 = lds_load_b128_raw(lds_buffer, byte_off)
            v1 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(16))
            return vector.shuffle(v0, v1, list(range(8)))

        def _precompute_b_lane_bases(lds_ptr):
            """Precompute per-wn B fragment lane base addresses (byte offsets).

            B stored as [tile_n, packed_tile_k + pad] in LDS.
            lane16 -> N-row, lane_kgrp -> K-half.
            Each 32-col WMMA needs 2 groups of 16 N-rows -> 2 bases per wn.
            """
            row_base = (warp_n_base + lane16) * arith.index(lds_b_stride_bytes)
            k_half_off = lane_kgrp * arith.index(32)
            bases = []
            for wn_half in range_constexpr(wmma_n_rep * 2):
                base = row_base + arith.index(wn_half * WMMA_N * lds_b_stride_bytes) + k_half_off
                bases.append(base)
            return lds_ptr, bases

        def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
            """Load one 32x128 FP4 B-fragment (SRC0 after A/B swap).

            Returns vector<16xi32> via 4 x ds_load_b128.
            b_lane_bases: list of 2*wmma_n_rep pre-computed bases.
            """
            k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR)
            base0 = b_lane_bases[wn * 2] + k_byte_off
            v0 = lds_load_b128_raw(lds_buffer, base0)
            v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(16))
            base1 = b_lane_bases[wn * 2 + 1] + k_byte_off
            v2 = lds_load_b128_raw(lds_buffer, base1)
            v3 = lds_load_b128_raw(lds_buffer, base1 + arith.index(16))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        def _precompute_scale_lane_bases(lds_ptr, warp_base, reps,
                                        interleaved_cols):
            """Precompute scale lane bases for preshuffled interleaved layout (byte offsets)."""
            warp_lds_row = warp_base / arith.index(reps) + lane16
            base = warp_lds_row * arith.index(interleaved_cols) \
                + lane_kgrp * arith.index(SCALES_PER_WMMA)
            return lds_ptr, [base]

        def load_scale_b128(lds_buffer, scale_base, reps, ks=0):
            """Load all wmma_rep scales via ds_load_b128(s) for K-subtile *ks*."""
            ks_byte_off = ks * reps * SCALES_PER_WMMA
            eff_base = scale_base if ks_byte_off == 0 else scale_base + arith.index(ks_byte_off)
            num_loads = (reps + 3) // 4
            vecs = []
            for ld in range_constexpr(num_loads):
                off = eff_base if ld == 0 else eff_base + arith.index(ld * 16)
                vecs.append(lds_load_b128_raw(lds_buffer, off))
            results = []
            for i in range_constexpr(reps):
                vi = vector.extract(vecs[i // 4], static_position=[i % 4], dynamic_position=[])
                results.append(vi)
            return results

        # --- K-subtile compute (A-streaming pipeline) ---
        def _load_b_and_scales(b_buf, b_bases, bs_buf, bs_bases,
                               as_buf, as_bases, ks):
            """Load B frags + all scales for one K-subtile (no wait).

            B frags (vec16i32, 32 N-rows) and ALL scales are loaded upfront
            because they are reused across all wm groups.
            When use_scale_opsel, only wmma_m_rep//2 AScale VGPRs are kept --
            each covers two adjacent wm groups via lane half-select (opsel).
            """
            b_frags = [load_b_frag(b_buf, b_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
            b_scales = load_scale_b128(bs_buf, bs_bases[0], b_scale_load_rep, ks)
            a_scales_all = load_scale_b128(as_buf, as_bases[0], wmma_m_rep, ks)
            if use_scale_opsel:
                # Keep only even-indexed scales: each VGPR already holds
                # a (wm_2i, wm_2i+1) pair via lane_kgrp offset.
                a_scales = a_scales_all[::2]
            else:
                a_scales = a_scales_all
            return b_frags, b_scales, a_scales

        def _emit_wmma(accs, wm, wn, a_frag, b_frags, a_scales, b_scales):
            """Emit one 32x16 WMMA (A/B swapped).

            SRC0=B_frag(vec16i32), SRC1=A_frag(vec8i32)
            Scale0=BScale, Scale1=AScale
            """
            idx = wm * wmma_n_rep + wn
            if use_scale_opsel:
                a_scale_idx = wm // 2
                a_opsel = wm % 2
            else:
                a_scale_idx = wm
                a_opsel = 0
            accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                T.vec(16, T.f32),
                b_frags[wn],           # SRC0: vec16i32 (32 N-rows)
                a_frag,                # SRC1: vec8i32  (16 M-rows)
                accs[idx],
                b_scales[wn * 2],      # Scale0: BScale (all 32 lanes)
                a_scales[a_scale_idx], # Scale1: AScale
                scaleAType=0,          # BScale: not applicable for 32x16
                scaleBType=a_opsel,    # AScale: lane half-select
            )

        def _a_streaming_compute(accs, a_buf, a_bases, b_frags, b_scales,
                                 a_scales, ks, emit_filler=None,
                                 next_bs_info=None):
            """Half-based A-streaming: 4-WMMA bursts, 2 waits per K-subtile.

            Splits wmma_m_rep iterations into two halves:
              Half 0 (wm=0..half-1): pre-load 2 A-frags, wait, 4 WMMAs
              Half 1 (wm=half..end): load during Half 0 coexec, wait, 4 WMMAs

            next_bs_info: optional (b_buf, b_bases, bs_buf, bs_bases,
                          as_buf, as_bases, next_ks) -- issues B+scale loads
                          after Half 1 for K-subtile overlap.
                          When set, returns (accs, next_b, next_bs, next_as).
            """
            next_result = None
            _half_wm = wmma_m_rep // 2

            # ── Pre-load Half 0 A-frags (2 frags = 4 ds_load) ──
            a_frags_h0 = [load_a_frag(a_buf, a_bases[wm], ks)
                          for wm in range_constexpr(_half_wm)]

            # Wait for B+scales (loaded by caller) + Half 0 A-frags
            rocdl.s_wait_dscnt(0)

            # ── Half 0: 4 consecutive WMMAs (wm=0..half_wm-1) ──
            for wm in range_constexpr(_half_wm):
                for wn_raw in range_constexpr(wmma_n_rep):
                    wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                    _emit_wmma(accs, wm, wn, a_frags_h0[wm], b_frags,
                               a_scales, b_scales)

            # Load Half 1 A-frags (interleaved in Half 0 WMMA coexec)
            a_frags_h1 = [load_a_frag(a_buf, a_bases[_half_wm + h], ks)
                          for h in range_constexpr(_half_wm)]

            # Wait for Half 1 A-frags (expected near-zero stall:
            # 4 WMMAs above provide ~16-32 cycles of hiding time)
            rocdl.s_wait_dscnt(0)

            # ── Half 1: 4 consecutive WMMAs (wm=half_wm..wmma_m_rep-1) ──
            for h in range_constexpr(_half_wm):
                wm = _half_wm + h
                is_last = (wm == wmma_m_rep - 1)

                if is_last and emit_filler is not None:
                    rocdl.sched_barrier(0)
                    emit_filler()

                for wn_raw in range_constexpr(wmma_n_rep):
                    wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                    _emit_wmma(accs, wm, wn, a_frags_h1[h], b_frags,
                               a_scales, b_scales)

            # Transition: next K-subtile B+scale prefetch
            if next_bs_info is not None:
                nb_buf, nb_bases, nbs_buf, nbs_bases, \
                    nas_buf, nas_bases, n_ks = next_bs_info
                next_result = _load_b_and_scales(
                    nb_buf, nb_bases, nbs_buf, nbs_bases,
                    nas_buf, nas_bases, n_ks)
                return accs, next_result
            return accs

        # --- Compute on one LDS buffer (A-streaming K-subtile pipeline) ---
        def compute_tile(accs_in, lds_a, lds_b, lds_as, lds_bs,
                         emit_filler=None):
            current_accs = list(accs_in)

            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            as_buf, as_bases = _precompute_scale_lane_bases(
                lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
            bs_buf, bs_bases = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b)

            if k_wmma_steps == 1:
                b_frags, b_scales, a_scales = _load_b_and_scales(
                    b_buf, b_bases, bs_buf, bs_bases, as_buf, as_bases, 0)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, b_frags, b_scales,
                    a_scales, 0, emit_filler=emit_filler)
            else:
                prev_b, prev_bs, prev_as = _load_b_and_scales(
                    b_buf, b_bases, bs_buf, bs_bases, as_buf, as_bases, 0)
                for ks in range_constexpr(k_wmma_steps - 1):
                    current_accs, (prev_b, prev_bs, prev_as) = \
                        _a_streaming_compute(
                            current_accs, a_buf, a_bases, prev_b, prev_bs,
                            prev_as, ks,
                            next_bs_info=(b_buf, b_bases, bs_buf, bs_bases,
                                          as_buf, as_bases, ks + 1))
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, prev_b, prev_bs, prev_as,
                    k_wmma_steps - 1, emit_filler=emit_filler)

            return current_accs

        def hot_loop_scheduler():
            _half_wm = wmma_m_rep // 2
            _half_wmma = _half_wm * wmma_n_rep  # WMMAs per half

            for _ks in range_constexpr(k_wmma_steps):
                if _ks == 0:
                    # B+scales(10) + Half 0 A-frags(half_wm * 2)
                    rocdl.sched_dsrd(wmma_n_rep * 4 + 2 + _half_wm * 2)
                else:
                    # Only Half 0 A-frags (B+scales already counted)
                    rocdl.sched_dsrd(_half_wm * 2)
                rocdl.sched_mfma(_half_wmma)         # Half 0 WMMAs
                rocdl.sched_dsrd(_half_wm * 2)       # Half 1 A-frag loads
                rocdl.sched_mfma(_half_wmma)         # Half 1 WMMAs
                if _ks < k_wmma_steps - 1:
                    rocdl.sched_dsrd(wmma_n_rep * 4 + 2)  # next ks B+scales

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
        def _get_acc_sub8(accs, acc_idx, vec_base):
            """Extract an 8-element sub-vector from vec16 accumulator at vec_base."""
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
                    c_off_bytes = (row * n_stride + col_base) * arith.index(elem_bytes_d)
                    addrs.append(c_off_bytes)
                else:
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        c_off = row * n_stride + col
                        addrs.append(c_off)
            return addrs

        def epilogue_stores(final_accs, addrs):
            _bf16_out = out_dtype in ("bf16", "f16")
            _out_elem_local = T.bf16 if out_dtype == "bf16" else (T.f16 if out_dtype == "f16" else None)
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
            """Write accumulators to D output LDS via ds_store_b128."""
            _out_elem_local = T.bf16 if out_dtype == "bf16" else (T.f16 if out_dtype == "f16" else None)
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                imm = m_off * _lds_d_stride_elems + wn * _n_col_d_elems
                store_acc_vec8_to_lds(d_buf, d_base, imm, sub8,
                                      out_elem=_out_elem_local)

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

        acc_zero = arith.constant_vector(0.0, T.vec(16, T.f32))
        accs = [acc_zero] * n_accs

        # Build per-stage SmemPtrs (f16 element type for memref construction;
        # actual loads use raw byte offsets via lds_load_b128_raw).
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

        # Pre-extracted LDS base indices for SCF loop compute_tile (byte offsets)
        stages_a_idx = [extract_lds_base_idx(stages_a[i])
                        for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i])
                        for i in range_constexpr(num_buffers)]
        stages_as_idx = [extract_lds_base_idx(stages_as[i])
                         for i in range_constexpr(num_buffers)]
        stages_bs_idx = [extract_lds_base_idx(stages_bs[i])
                         for i in range_constexpr(num_buffers)]

        # D output LDS setup for TDM store epilogue
        if use_tdm_store:
            d_lds_base_ptr = base_ptrs[d_reuse_stage]
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(d_lds_base_ptr, d_output_off, elem_ty_lds,
                             shape=(d_lds_f16_count,))
            d_lds_buffer = get_lds_memref(d_smem)

            # Per-lane LDS base address (VGPR) in f16 element offsets
            warp_lds_off = (wave_m_idx * arith.index(n_warp) + wave_n_idx) \
                * arith.index(_warp_d_elems)
            d_lane_base = (warp_lds_off
                           + lane16 * arith.index(_lds_d_stride_elems)
                           + lane_kgrp * arith.index(4 * elem_bytes_d))

            # Per-warp TDM descriptor for D store (all addresses SGPR via wave_id)
            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr = wave_id_idx * arith.index(warp_d_bytes) \
                + arith.index(d_output_off)

            # Decompose wave_id into (m, n) coords in SGPR pipeline
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
                pad_interval=warp_tile_n, pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
            )

        # Precompute LDS addresses for all stages
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        stages_as_lds_addr = []
        stages_bs_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(vector.extract(make_desc_a(stages_a_mem[i], arith.index(0)).dgroup0, static_position=[1], dynamic_position=[]))
            stages_b_lds_addr.append(vector.extract(make_desc_b(stages_b_mem[i], arith.index(0)).dgroup0, static_position=[1], dynamic_position=[]))
            stages_as_lds_addr.append(vector.extract(make_desc_as(stages_as_mem[i], arith.index(0)).dgroup0, static_position=[1], dynamic_position=[]))
            stages_bs_lds_addr.append(vector.extract(make_desc_bs(stages_bs_mem[i], arith.index(0)).dgroup0, static_position=[1], dynamic_position=[]))

        # Create initial descriptors (pointing to tile 0)
        desc_a_init = make_desc_a(stages_a_mem[0], arith.index(0))
        desc_b_init = make_desc_b(stages_b_mem[0], arith.index(0))
        desc_as_init = make_desc_as(stages_as_mem[0], arith.index(0))
        desc_bs_init = make_desc_bs(stages_bs_mem[0], arith.index(0))
        
        dgroup0_a = desc_a_init.dgroup0
        dgroup0_b = desc_b_init.dgroup0
        dgroup0_as = desc_as_init.dgroup0
        dgroup0_bs = desc_bs_init.dgroup0
        
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1
        dgroup1_as = desc_as_init.dgroup1
        dgroup1_bs = desc_bs_init.dgroup1
        
        adv_a_bytes = arith.index(tile_k // PACK_FACTOR)
        adv_b_bytes = arith.index(tile_k // PACK_FACTOR)
        adv_as_bytes = arith.index(tile_k // SCALE_BLOCK * wmma_m_rep)
        adv_bs_bytes = arith.index(tile_k // SCALE_BLOCK * b_scale_load_rep)

        # Prologue: load first (num_buffers - 1) tiles
        for i in range_constexpr(pre_loaded):
            dgroup0_a = vector.insert(stages_a_lds_addr[i], dgroup0_a, static_position=[1], dynamic_position=[])
            dgroup0_b = vector.insert(stages_b_lds_addr[i], dgroup0_b, static_position=[1], dynamic_position=[])
            dgroup0_as = vector.insert(stages_as_lds_addr[i], dgroup0_as, static_position=[1], dynamic_position=[])
            dgroup0_bs = vector.insert(stages_bs_lds_addr[i], dgroup0_bs, static_position=[1], dynamic_position=[])
            
            desc_a = tdm_ops.TDMDescriptor2D(dgroup0_a, dgroup1_a)
            desc_b = tdm_ops.TDMDescriptor2D(dgroup0_b, dgroup1_b)
            desc_as = tdm_ops.TDMDescriptor2D(dgroup0_as, dgroup1_as)
            desc_bs = tdm_ops.TDMDescriptor2D(dgroup0_bs, dgroup1_bs)
            
            issue_all_tdm_loads(desc_a, desc_b, desc_as, desc_bs)
            
            dgroup0_a = tdm_ops.advance_tdm_descriptor(desc_a, adv_a_bytes).dgroup0
            dgroup0_b = tdm_ops.advance_tdm_descriptor(desc_b, adv_b_bytes).dgroup0
            dgroup0_as = tdm_ops.advance_tdm_descriptor(desc_as, adv_as_bytes).dgroup0
            dgroup0_bs = tdm_ops.advance_tdm_descriptor(desc_bs, adv_bs_bytes).dgroup0
            
        pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2),
                               use_cluster=use_cluster)

        # Main loop (SCF for — single body, runtime stage select)
        if loop_iters > 0:
            total_main_tiles = loop_iters * num_buffers
            init_args = list(accs) + [dgroup0_a, dgroup0_b, dgroup0_as, dgroup0_bs]
            
            for tile_idx, state in range(0, total_main_tiles, 1, init=init_args):
                accs_in = list(state[:n_accs])
                cur_dgroup0_a = state[n_accs]
                cur_dgroup0_b = state[n_accs + 1]
                cur_dgroup0_as = state[n_accs + 2]
                cur_dgroup0_bs = state[n_accs + 3]

                # K-offset for L2 prefetch (tile_idx maps to global k-tile index)
                k_off = tile_idx * arith.index(tile_k)

                # Compute stage index
                s_idx = tile_idx % arith.index(num_buffers)

                # TDM load for the next-to-fill stage
                load_s_idx = (tile_idx + arith.index(num_buffers - 1)) % arith.index(num_buffers)
                
                lds_a = _select_base(stages_a_lds_addr, load_s_idx)
                lds_b = _select_base(stages_b_lds_addr, load_s_idx)
                lds_as = _select_base(stages_as_lds_addr, load_s_idx)
                lds_bs = _select_base(stages_bs_lds_addr, load_s_idx)
                
                next_dgroup0_a = vector.insert(lds_a, cur_dgroup0_a, static_position=[1], dynamic_position=[])
                next_dgroup0_b = vector.insert(lds_b, cur_dgroup0_b, static_position=[1], dynamic_position=[])
                next_dgroup0_as = vector.insert(lds_as, cur_dgroup0_as, static_position=[1], dynamic_position=[])
                next_dgroup0_bs = vector.insert(lds_bs, cur_dgroup0_bs, static_position=[1], dynamic_position=[])
                
                desc_a = tdm_ops.TDMDescriptor2D(next_dgroup0_a, dgroup1_a)
                desc_b = tdm_ops.TDMDescriptor2D(next_dgroup0_b, dgroup1_b)
                desc_as = tdm_ops.TDMDescriptor2D(next_dgroup0_as, dgroup1_as)
                desc_bs = tdm_ops.TDMDescriptor2D(next_dgroup0_bs, dgroup1_bs)
                
                issue_all_tdm_loads(desc_a, desc_b, desc_as, desc_bs)
                
                next_dgroup0_a = tdm_ops.advance_tdm_descriptor(desc_a, adv_a_bytes).dgroup0
                next_dgroup0_b = tdm_ops.advance_tdm_descriptor(desc_b, adv_b_bytes).dgroup0
                next_dgroup0_as = tdm_ops.advance_tdm_descriptor(desc_as, adv_as_bytes).dgroup0
                next_dgroup0_bs = tdm_ops.advance_tdm_descriptor(desc_bs, adv_bs_bytes).dgroup0

                _l2_prefetch(k_off)

                # Select compute bases at runtime (single compute_tile body)
                cur_a = _select_base(stages_a_idx, s_idx)
                cur_b = _select_base(stages_b_idx, s_idx)
                cur_as = _select_base(stages_as_idx, s_idx)
                cur_bs = _select_base(stages_bs_idx, s_idx)

                rocdl.sched_barrier(0)
                accs_in = compute_tile(accs_in, cur_a, cur_b, cur_as, cur_bs)
                hot_loop_scheduler()
                pipeline_fence(
                    outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2),
                    use_cluster=use_cluster)
                results = yield list(accs_in) + [next_dgroup0_a, next_dgroup0_b, next_dgroup0_as, next_dgroup0_bs]
            
            accs = list(results[:n_accs])
            dgroup0_a = results[n_accs]
            dgroup0_b = results[n_accs + 1]
            dgroup0_as = results[n_accs + 2]
            dgroup0_bs = results[n_accs + 3]

        # Tail
        if loop_iters == 0 and use_cluster:
            gpu.cluster_barrier()
        _extra_j = 0
        epi_addrs_box = [None]
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if _load_stage is not None:
                dgroup0_a = vector.insert(stages_a_lds_addr[_load_stage], dgroup0_a, static_position=[1], dynamic_position=[])
                dgroup0_b = vector.insert(stages_b_lds_addr[_load_stage], dgroup0_b, static_position=[1], dynamic_position=[])
                dgroup0_as = vector.insert(stages_as_lds_addr[_load_stage], dgroup0_as, static_position=[1], dynamic_position=[])
                dgroup0_bs = vector.insert(stages_bs_lds_addr[_load_stage], dgroup0_bs, static_position=[1], dynamic_position=[])
                
                desc_a = tdm_ops.TDMDescriptor2D(dgroup0_a, dgroup1_a)
                desc_b = tdm_ops.TDMDescriptor2D(dgroup0_b, dgroup1_b)
                desc_as = tdm_ops.TDMDescriptor2D(dgroup0_as, dgroup1_as)
                desc_bs = tdm_ops.TDMDescriptor2D(dgroup0_bs, dgroup1_bs)
                
                issue_all_tdm_loads(desc_a, desc_b, desc_as, desc_bs)
                
                dgroup0_a = tdm_ops.advance_tdm_descriptor(desc_a, adv_a_bytes).dgroup0
                dgroup0_b = tdm_ops.advance_tdm_descriptor(desc_b, adv_b_bytes).dgroup0
                dgroup0_as = tdm_ops.advance_tdm_descriptor(desc_as, adv_as_bytes).dgroup0
                dgroup0_bs = tdm_ops.advance_tdm_descriptor(desc_bs, adv_bs_bytes).dgroup0
                
                _extra_j += 1
            if _outstanding == -1:
                if use_tdm_store:
                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage])
                else:
                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage],
                        emit_filler=_emit_epi_addrs)
            else:
                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs,
                    stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                    stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage])
                hot_loop_scheduler()
                pipeline_fence(outstanding=_outstanding,
                               use_cluster=use_cluster)

        if use_tdm_store:
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            rocdl.sched_barrier(0)
            epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, effective_waves_per_eu, l2_prefetch_distance,
                 cluster_m, cluster_n, use_tdm_store,
                 out_dtype, inst_prefetch, wave_specialized_tdm,
                 use_scale_opsel)

    @flyc.jit(compile_hints={"expert_scheduling_mode": True})
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
