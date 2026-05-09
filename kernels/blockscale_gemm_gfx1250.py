"""DeepSeek blockscale FP8 GEMM kernel for gfx1250.

E4M3 elements with FP32 block scales (1×128 activation, 128×128 weight).
Two-stage accumulator: raw V_WMMA_F32_16X16X128_FP8_FP8 -> block partial ->
math.fma(partial, combined_scale, global_acc).

Scale path differs from mxscale:
  - Scales are FP32, loaded via ``buffer_load`` directly into VGPRs (no LDS, no TDM).
  - A scale is preshuffled by the host into a wave32-friendly layout
    (see ``tests/kernels/utils/fp4_utils.py:preshuffle_a_scale_for_wmma``).
  - B scale stays in DeepSeek-native ``[N/scale_block_n, K/scale_block_k]`` row-major.
"""

import math as py_math
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, idx2crd, math, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.typing import T, Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx,
    get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    store_acc_vec8_to_buffer,
    store_acc_vec8_to_lds,
    validate_tdm_descriptor_2d,
)
from kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

# Common constants
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32

LDS_PAD_A_BYTES = 16
LDS_PAD_D_BYTES = 16


def compile_blockscale_gemm(
    *,
    data_format: str = "fp8",
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 2,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    use_tdm_store: bool = True,
    out_dtype: str = "bf16",
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    split_k: int = 1,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
):
    """Compile DeepSeek-style blockscale GEMM for gfx1250.

    Args:
        data_format: Element format. v1 only supports ``"fp8"`` (E4M3).
            Future formats (``"bf8"``, ``"fp4"``, mixed) reserve this slot.

    Data layout (kernel-side, all 1-D flattened tensors):
        A: [M, K] fp8 e4m3 (uint8), row-major
        B: [N, K] fp8 e4m3 (uint8), preshuffled in 16x16 byte tiles
           (use ``fp4_utils.preshuffle_b_16x16``)
        a_scale: preshuffled fp32, total M * K/scale_block_k elements.
                 Logical layout:
                 [N_wg_m, N_tile_k, m_warp, 16(lane16), scale_k_per_tile, wmma_m_rep]
                 (use ``fp4_utils.preshuffle_a_scale_for_wmma``)
        b_scale: fp32, no preshuffle.
                 [N/scale_block_n, K/scale_block_k] row-major.
        out:     [M, N] bf16/fp16/f32

    Returns a JitFunction:
        launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, stream)
    """
    if data_format != "fp8":
        raise ValueError(
            f"data_format={data_format!r} not supported in v1 (only 'fp8' E4M3)")

    if out_dtype not in ("f32", "bf16", "f16"):
        raise ValueError(f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}")
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4

    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if scale_block_k != WMMA_K:
        raise ValueError(
            f"scale_block_k must equal WMMA_K={WMMA_K} for v1; got {scale_block_k}")
    if scale_block_n % WMMA_N != 0:
        raise ValueError(
            f"scale_block_n={scale_block_n} must be a multiple of WMMA_N={WMMA_N}")

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
    effective_waves_per_eu = waves_per_eu

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    if block_threads > 1024:
        raise ValueError(
            f"block_threads must be <= 1024, got {block_threads}")

    if wave_specialized_tdm and num_warps < 2:
        raise ValueError(
            f"wave_specialized_tdm requires >= 2 waves (one each for A, B), "
            f"got {num_warps}")

    # Compile-time constants
    PACK_FACTOR_A = 1
    PACK_FACTOR_B = 1
    WMMA_N_EFF = 16   # N-cols covered per WMMA instruction (16x16x128 FP8)
    ACC_VEC_SIZE = 8   # vec<8,f32> accumulator
    DS_LOADS_PER_A_FRAG = 4

    packed_tile_k_a = tile_k
    packed_tile_k_b = tile_k
    scale_k_per_tile = tile_k // scale_block_k
    K_packed_a = K
    K_packed_b = K
    K_scale = K // scale_block_k
    scale_n = N // scale_block_n
    split_k_chunk = K // split_k

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if K % split_k != 0:
        raise ValueError(f"K must be divisible by split_k={split_k}, got K={K}")
    if split_k_chunk % tile_k != 0:
        raise ValueError(
            f"K/split_k must be divisible by tile_k={tile_k}, got {split_k_chunk}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if tile_k % scale_block_k != 0:
        raise ValueError(
            f"tile_k={tile_k} must be a multiple of scale_block_k={scale_block_k}")
    if N % scale_block_n != 0:
        raise ValueError(
            f"N={N} must be a multiple of scale_block_n={scale_block_n}")

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N_EFF != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N_EFF}")

    if split_k > 1 and use_tdm_store:
        raise ValueError("split_k > 1 currently requires use_tdm_store=False")

    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles}")

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    n_accs = wmma_m_rep * wmma_n_rep
    # B-scale rep counts how many wn-blocks share a single n-block within a warp
    # (only used for fma loop; not for ds_loads since we bypass LDS)
    sb_per_tile = scale_k_per_tile  # 1 scale block per WMMA K-step
    assert k_wmma_steps == sb_per_tile, (
        f"v1 requires k_wmma_steps == sb_per_tile (1 WMMA per scale block); "
        f"got k_wmma_steps={k_wmma_steps}, sb_per_tile={sb_per_tile}")

    # Number of A-scale fp32 fetched per lane per K-tile = scale_k_per_tile * wmma_m_rep
    a_scale_fp32_per_lane = scale_k_per_tile * wmma_m_rep
    # Split into vec_width=4 chunks (b128 max). Build a compile-time chunk plan.
    _a_scale_chunk_plan = []
    _rem = a_scale_fp32_per_lane
    _off = 0
    while _rem > 0:
        _vw = 4 if _rem >= 4 else (2 if _rem >= 2 else 1)
        _a_scale_chunk_plan.append((_vw, _off))
        _off += _vw
        _rem -= _vw

    # Number of distinct n-blocks covered by this workgroup tile
    n_blocks_per_tile = tile_n // scale_block_n
    if n_blocks_per_tile < 1:
        # tile_n < scale_block_n: workgroup straddles 1 n-block, but multiple
        # workgroups share it. Treat as 1 n-block per workgroup.
        n_blocks_per_tile = 1
    # Fast path: every warp's N-span either fits exactly inside one scale-N
    # block or covers an integer number of scale-N blocks. This preserves the
    # existing constant-index scale selection for common tile_n=128/256 layouts.
    _b_scale_fast_path = (
        warp_tile_n % scale_block_n == 0
        or scale_block_n % warp_tile_n == 0
    )
    # General path: warp N-spans can straddle scale-N row boundaries. Load the
    # maximum number of scale rows any warp can touch, accounting for the fact
    # that warp starts advance by warp_tile_n.
    if _b_scale_fast_path:
        n_blocks_per_warp = max(1, warp_tile_n // scale_block_n)
    else:
        _n_start_mod_step = py_math.gcd(warp_tile_n, scale_block_n)
        _max_start_mod = scale_block_n - _n_start_mod_step
        n_blocks_per_warp = max(
            1,
            (_max_start_mod + warp_tile_n + scale_block_n - 1)
            // scale_block_n,
        )
    b_scale_fp32_per_warp = n_blocks_per_warp * scale_k_per_tile

    # Approximate ds_load count for hot-loop scheduler hints (B frags only,
    # since scale path uses buffer_load not ds_load).
    _b_frag_loads_per_wn = 4   # FP8: 4 ds_load_b128 per B fragment
    _bs_ds_loads = wmma_n_rep * _b_frag_loads_per_wn

    lds_a_stride_bytes = packed_tile_k_a + LDS_PAD_A_BYTES

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * packed_tile_k_b
    # Scales bypass LDS entirely (direct buffer_load -> VGPR).
    lds_a_scale_bytes = 0
    lds_b_scale_bytes = 0

    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    # TDM descriptors partition a tile cooperatively across ``num_warps`` by
    # deriving per-wave offsets from ``wave_id``. In wave-specialized mode we
    # dedicate one loader wave to each tensor (A/B), so each active loader wave
    # must issue a full-tile descriptor by itself. (Scales bypass TDM entirely.)
    tdm_desc_num_warps = 1 if wave_specialized_tdm else num_warps

    # Pre-flight TDM descriptor bound checks
    # 16-bit per-warp tile_d limit + 32-bit stride0 limit.  Catches over-large
    # tile_k/tile_m/tile_n/K early instead of letting the kernel emit a corrupt
    # descriptor that produces silent OOB / wrong results.
    validate_tdm_descriptor_2d(
        "A",
        tile_shape=(tile_m, packed_tile_k_a),
        strides=(K_packed_a, 1),
        tensor_shape=(tile_m, packed_tile_k_a),
        num_warps=tdm_desc_num_warps,
        elem_bytes=1,
    )
    validate_tdm_descriptor_2d(
        "B",
        tile_shape=(tile_n // 16, packed_tile_k_b * 16),
        strides=(K_packed_b * 16, 1),
        tensor_shape=(tile_n // 16, packed_tile_k_b * 16),
        num_warps=tdm_desc_num_warps,
        elem_bytes=1,
    )

    # All pipeline stages share the same intra-stage layout. Only A and B data
    # sit in LDS — scales go through buffer_load directly.
    stage_layout = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="blockscale_fp8_layout")
    stage_a_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_data_rel_off + lds_a_data_bytes
    stage_b_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_data_rel_off + lds_b_data_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    _last_compute_stage = _base_tail_plan[-1][1]

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"blockscale_fp8_{tile_m}x{tile_n}x{tile_k}_"
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

    stage_a_data_off = [
        stage_base_off[i] + stage_a_data_rel_off for i in range(num_buffers)
    ]
    stage_b_data_off = [
        stage_base_off[i] + stage_b_data_rel_off for i in range(num_buffers)
    ]

    if use_tdm_store:
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
    check_smem_capacity(arena_total_bytes, gpu_arch)

    # TENSORcnt is tracked per-wave in hardware. The regular path issues two
    # tensor ops per wave per K-stage (A + B), while wave-specialized issues
    # only one tensor op from each dedicated loader wave.
    TDM_LOADS_PER_STEP = 1 if wave_specialized_tdm else 2
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    # Epilogue sub-tile layout: vec<8,f32>, single 8-element block per (wm, wn).
    _sub_tiles = []
    for _wm in range(wmma_m_rep):
        for _wn in range(wmma_n_rep):
            acc_idx = _wm * wmma_n_rep + _wn
            m_off = _wm * WMMA_M
            n_sub = _wn
            _sub_tiles.append((acc_idx, 0, m_off, n_sub))

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_blockscale_gemm(
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

        if const_expr(inst_prefetch):
            from flydsl._mlir.dialects import llvm as llvm_dialect
            if rocdl.wave_id() == fx.Int32(0):
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
        bz = fx.Index(gpu.block_idx.z) if split_k > 1 else arith.index(0)

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        split_k_base = bz * arith.index(split_k_chunk)

        if const_expr(use_cluster):
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
            fx.get(thr_coord, 0), fx.get(thr_coord, 1),
            fx.get(thr_coord, 2), fx.get(thr_coord, 3))

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = fx.Index(i32_m)
        n_stride = arith.index(N)
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)
        zero_i32 = fx.Int32(0)

        # Scale buffer resources: fp32, accessed via direct buffer_load (no LDS).
        # arg_a_scale is preshuffled into:
        #   [N_wg_m, N_tile_k, m_warp, 16, scale_k_per_tile, wmma_m_rep] flat fp32.
        # Total size = M * K_scale fp32 = M * K_scale * 4 bytes.
        sa_nbytes = m_idx * arith.index(K_scale * 4)
        scale_a_rsrc = buffer_ops.create_buffer_resource(
            arg_a_scale, max_size=False, num_records_bytes=sa_nbytes)
        # arg_b_scale is row-major [N/scale_block_n, K/scale_block_k] fp32.
        # Static size — use max_size=True (compile-time constant).
        scale_b_rsrc = buffer_ops.create_buffer_resource(
            arg_b_scale, max_size=True)

        def make_desc_a(memref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=memref,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, packed_tile_k_a),
                strides=(K_packed_a, 1),
                tile_shape=(tile_m, packed_tile_k_a),
                elem_bytes=1,
                pad_interval=packed_tile_k_a, pad_amount=LDS_PAD_A_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        def make_desc_b(memref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=memref,
                global_offset=(blk_n / arith.index(16),
                               k_base * arith.index(16)),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(tile_n // 16, packed_tile_k_b * 16),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable)

        elem_ty_lds = T.f16

        def _precompute_a_lane_bases(lds_ptr):
            """Precompute per-wm A fragment lane base addresses (byte offsets)."""
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            # K-dimension interleaving: kgrp0/kgrp1 read alternating 128-bit chunks
            # All formats: kgrp offset = 16 bytes (one ds_load_b128 width)
            k_half_off = lane_kgrp * arith.index(16)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
                bases.append(base)
            return lds_ptr, bases


        def load_a_frag(lds_buffer, a_lane_base, ks):
            """Load one A-fragment (FP8) from LDS as vec<16xi32>.

            4 × ds_load_b128 (64 bytes per lane), interleaved K stride=32:
              kgrp0 reads bytes [0:15],[32:47],[64:79],[96:111]
              kgrp1 reads bytes [16:31],[48:63],[80:95],[112:127]
            """
            k_byte_off = arith.index(ks * WMMA_K)
            byte_off = a_lane_base + k_byte_off
            v0 = lds_load_b128_raw(lds_buffer, byte_off)
            v1 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))
            v2 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(64))
            v3 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(96))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        def _precompute_b_lane_bases(lds_ptr):
            """Precompute per-wn B fragment lane base addresses (byte offsets).

            FP8: 1 base per wn (16-col WMMA = 1 N-group).
            K-dimension interleaving: kgrp0/kgrp1 read alternating 16x16 tiles
            (stride = 2 tiles, kgrp offset = 1 tile = 256 bytes).
            """
            _ngroup_stride = packed_tile_k_b * 16
            _n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
            row_off = lane16 * arith.index(16)
            k_tile_off = lane_kgrp * arith.index(256)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                ngroup_off = _n_group_base * arith.index(_ngroup_stride) \
                    + arith.index(wn * _ngroup_stride)
                bases.append(ngroup_off + row_off + k_tile_off)
            return lds_ptr, bases

        def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
            """Load one FP8 B-fragment from preshuffled LDS as vec<16xi32>.

            16x128 from 1 N-group: 4 × ds_load_b128 with stride=512 (2 tiles):
              kgrp0 reads tiles 0,2,4,6; kgrp1 reads tiles 1,3,5,7.
            """
            _num_tiles = WMMA_K // 16  # 8 tiles per N-group for FP8 (K=128)
            k_subtile_off = arith.index(ks * _num_tiles * 256)
            base0 = b_lane_bases[wn] + k_subtile_off
            v0 = lds_load_b128_raw(lds_buffer, base0)
            v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(512))
            v2 = lds_load_b128_raw(lds_buffer, base0 + arith.index(1024))
            v3 = lds_load_b128_raw(lds_buffer, base0 + arith.index(1536))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        def _load_b_frags(b_buf, b_bases, ks):
            """Load all wmma_n_rep B fragments for one K-subtile *ks*."""
            return [load_b_frag(b_buf, b_bases, wn, ks)
                    for wn in range_constexpr(wmma_n_rep)]

        # FP32 scale loading via direct buffer_load
        # Compile-time strides into preshuffled scale_a layout
        # [N_wg_m, N_tile_k, m_warp, 16, scale_k_per_tile, wmma_m_rep] (fp32 elems)
        _sa_stride_lane16 = scale_k_per_tile * wmma_m_rep
        _sa_stride_warp_m = 16 * _sa_stride_lane16
        _sa_stride_kt = m_warp * _sa_stride_warp_m   # = tile_m * scale_k_per_tile
        _sa_stride_wg_m = (K_scale // scale_k_per_tile) * _sa_stride_kt

        # B scale: [N/scale_block_n, K/scale_block_k] row-major fp32
        _sb_stride_n_block = K_scale   # row stride in fp32 elements

        def load_a_scales(k_tile_global_idx):
            """Load wmma_m_rep * scale_k_per_tile fp32 A-scales for one K-tile.

            Returns flat list ``a_scales[wm + sb * wmma_m_rep]`` of fp32 scalars
            (one per lane, distinct values across lane16/wave_m_idx).
            ``k_tile_global_idx`` is an Index-typed value.
            """
            sa_base_lane = (
                bx * arith.index(_sa_stride_wg_m)
                + k_tile_global_idx * arith.index(_sa_stride_kt)
                + wave_m_idx * arith.index(_sa_stride_warp_m)
                + lane16 * arith.index(_sa_stride_lane16)
            )
            results = []
            for vw, chunk_off in _a_scale_chunk_plan:
                eff_base = sa_base_lane if chunk_off == 0 \
                    else sa_base_lane + arith.index(chunk_off)
                v = buffer_ops.buffer_load(
                    scale_a_rsrc, eff_base, vec_width=vw, dtype=T.f32)
                if const_expr(vw == 1):
                    results.append(v)
                else:
                    v_wrapped = Vec(v)
                    for i in range_constexpr(vw):
                        results.append(v_wrapped[i])
            return results

        def load_b_scales(k_tile_global_idx):
            """Load b_scale_fp32_per_warp B-scales for one K-tile.

            Returns nested list ``b_scales[nb][sb]`` of fp32 scalars (broadcast,
            one value per (n_block, scale_block) for whole warp).
            """
            warp_n_block_base = (by * arith.index(tile_n)
                                 + wave_n_idx * arith.index(warp_tile_n)
                                 ) // arith.index(scale_block_n)
            kt_scale_base = k_tile_global_idx * arith.index(scale_k_per_tile)
            results = []
            for nb in range_constexpr(n_blocks_per_warp):
                row_idx = warp_n_block_base + arith.index(nb)
                row_off = row_idx * arith.index(_sb_stride_n_block)
                row_scales = []
                for sb in range_constexpr(scale_k_per_tile):
                    sb_idx = row_off + kt_scale_base + arith.index(sb)
                    v = buffer_ops.buffer_load(
                        scale_b_rsrc, sb_idx, vec_width=1, dtype=T.f32)
                    row_scales.append(v)
                results.append(row_scales)
            return results

        def combine_scales(a_scales_flat, b_scales):
            """Build combined[sb][wm][wn]: vec<8,f32> = a_scale * b_scale.

            ``a_scales_flat[sb * wmma_m_rep + wm]`` (per-lane scalar).
            Fast path uses the compile-time ``wn`` bucket. General path selects
            by absolute N position so a warp may cross ``scale_block_n`` rows.
            """
            wn_per_n_block = max(1, scale_block_n // WMMA_N)
            warp_n_abs_base = None
            warp_base_scale_row = None
            if not _b_scale_fast_path:
                warp_n_abs_base = (
                    by * arith.index(tile_n)
                    + wave_n_idx * arith.index(warp_tile_n)
                )
                warp_base_scale_row = warp_n_abs_base // arith.index(scale_block_n)

            def _select_b_scale(wn, sb):
                if const_expr(_b_scale_fast_path):
                    return b_scales[wn // wn_per_n_block][sb]

                wn_scale_row = (
                    warp_n_abs_base + arith.index(wn * WMMA_N)
                ) // arith.index(scale_block_n)
                row_rel = wn_scale_row - warp_base_scale_row
                selected = b_scales[0][sb]
                for nb in range_constexpr(1, n_blocks_per_warp):
                    is_nb = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        row_rel,
                        arith.index(nb),
                    )
                    selected = arith.select(is_nb, b_scales[nb][sb], selected)
                return selected

            combined = []
            for sb in range_constexpr(scale_k_per_tile):
                sb_combined = []
                for wm in range_constexpr(wmma_m_rep):
                    wm_combined = []
                    a_s = a_scales_flat[sb * wmma_m_rep + wm]
                    for wn in range_constexpr(wmma_n_rep):
                        b_s = _select_b_scale(wn, sb)
                        wm_combined.append(
                            Vec.filled(8, a_s * b_s, fx.Float32))
                    sb_combined.append(wm_combined)
                combined.append(sb_combined)
            return combined

        def _emit_wmma_blockscale(block_accs, wm, wn, a_frag, b_frags):
            """Emit one raw FP8 WMMA (no scale) into ``block_accs``.

            Uses A/B fragment swap (SRC0=B, SRC1=A) so the C accumulator layout
            matches the row/col scheme used by the epilogue — no CShuffle needed.
            """
            idx = wm * wmma_n_rep + wn
            block_accs[idx] = rocdl.wmma_f32_16x16x128_fp8_fp8(
                T.vec(8, T.f32),
                [b_frags[wn], a_frag, block_accs[idx]],
            )

        def _a_streaming_compute(block_accs, a_buf, a_bases, b_frags, ks,
                                 emit_filler=None,
                                 next_b_info=None,
                                 mid_compute_callback=None):
            """Half-based A-streaming with zigzag wn ordering.

            When *next_b_info* is provided, the next K-subtile's B-frag loads
            are issued BEFORE the s_wait_dscnt so they overlap with the current
            WMMA execution (partial drain pattern).
            """
            next_result = None
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm

            def _emit_rows(start_wm, a_frags):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    is_last = (wm == wmma_m_rep - 1)
                    if const_expr(is_last and emit_filler is not None):
                        rocdl.sched_barrier(0)
                        emit_filler()
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        _emit_wmma_blockscale(block_accs, wm, wn,
                                              a_frags[frag_i], b_frags)

            a_frags_front = [load_a_frag(a_buf, a_bases[wm], ks)
                             for wm in range_constexpr(_front_wm)]

            _use_partial_drain = (
                next_b_info is not None
                and _front_wm * wmma_n_rep >= 4
            )

            if const_expr(_use_partial_drain):
                nb_buf, nb_bases, n_ks = next_b_info
                next_result = _load_b_frags(nb_buf, nb_bases, n_ks)
                rocdl.s_wait_dscnt(_bs_ds_loads)
            else:
                rocdl.s_wait_dscnt(0)

            _emit_rows(0, a_frags_front)

            if const_expr(mid_compute_callback is not None):
                rocdl.sched_barrier(0)
                mid_compute_callback()

            if const_expr(_back_wm > 0):
                a_frags_back = [load_a_frag(a_buf, a_bases[_front_wm + h], ks)
                                for h in range_constexpr(_back_wm)]
                _back_drain = _bs_ds_loads if _use_partial_drain else 0
                rocdl.s_wait_dscnt(_back_drain)
                _emit_rows(_front_wm, a_frags_back)

            if const_expr(_use_partial_drain):
                return block_accs, next_result
            if const_expr(next_b_info is not None):
                nb_buf, nb_bases, n_ks = next_b_info
                next_result = _load_b_frags(nb_buf, nb_bases, n_ks)
                return block_accs, next_result
            return block_accs

        # Compute on one LDS buffer (one whole tile_k worth of K)
        # Two-stage accumulator:
        #   block_accs   — fp32 partial for ONE scale_block (1 WMMA per (wm,wn))
        #   global_accs  — persistent fp32 outer acc, math.fma after each block
        acc_zero_local = Vec.filled(ACC_VEC_SIZE, 0.0, fx.Float32)

        def compute_tile(global_accs, lds_a, lds_b, k_tile_global_idx,
                         emit_filler=None, mid_compute_callback=None):
            current_global = list(global_accs)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)

            # Pre-load all FP32 scales for this K-tile (cheap: fits in regs).
            a_scales_flat = load_a_scales(k_tile_global_idx)
            b_scales_nested = load_b_scales(k_tile_global_idx)
            combined = combine_scales(a_scales_flat, b_scales_nested)

            # One scale block == one WMMA-K step (scale_block_k == WMMA_K == 128).
            # Iterate over scale blocks (== k_wmma_steps).
            for sb in range_constexpr(sb_per_tile):
                block_accs = [acc_zero_local] * n_accs
                _is_last_sb = sb == sb_per_tile - 1
                _emit_filler_for_this_sb = emit_filler if _is_last_sb else None
                _mid_cb_for_this_sb = mid_compute_callback if sb == 0 else None
                if const_expr(sb_per_tile == 1):
                    # No prefetch needed: only one WMMA-K step.
                    b_frags = _load_b_frags(b_buf, b_bases, sb)
                    block_accs = _a_streaming_compute(
                        block_accs, a_buf, a_bases, b_frags, sb,
                        emit_filler=_emit_filler_for_this_sb,
                        mid_compute_callback=_mid_cb_for_this_sb)
                else:
                    if const_expr(sb == 0):
                        b_frags = _load_b_frags(b_buf, b_bases, 0)
                    if const_expr(_is_last_sb):
                        block_accs = _a_streaming_compute(
                            block_accs, a_buf, a_bases, b_frags, sb,
                            emit_filler=_emit_filler_for_this_sb,
                            mid_compute_callback=_mid_cb_for_this_sb)
                    else:
                        block_accs, b_frags = _a_streaming_compute(
                            block_accs, a_buf, a_bases, b_frags, sb,
                            next_b_info=(b_buf, b_bases, sb + 1),
                            mid_compute_callback=_mid_cb_for_this_sb)

                # Apply combined scale: global_accs[i] = fma(block, combined, global)
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        current_global[idx] = math.fma(
                            block_accs[idx],
                            combined[sb][wm][wn],
                            current_global[idx],
                        )

            return current_global

        def hot_loop_scheduler():
            _half_wm = wmma_m_rep // 2
            _half_wmma = _half_wm * wmma_n_rep
            _b_loads_per_frag = 4   # FP8: 4 ds_load_b128 per B fragment

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag
                                     + _half_wm * DS_LOADS_PER_A_FRAG)
                else:
                    rocdl.sched_dsrd(_half_wm * DS_LOADS_PER_A_FRAG)
                rocdl.sched_mfma(_half_wmma)
                rocdl.sched_dsrd(_half_wm * DS_LOADS_PER_A_FRAG)
                rocdl.sched_mfma(_half_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag)
            rocdl.sched_barrier(0)

        def compute_tile_scheduled(accs_in, lds_a, lds_b, k_tile_global_idx,
                                   emit_filler=None,
                                   mid_compute_callback=None):
            return compute_tile(
                accs_in, lds_a, lds_b, k_tile_global_idx,
                emit_filler=emit_filler,
                mid_compute_callback=mid_compute_callback)

        def hot_loop_scheduler_scheduled():
            hot_loop_scheduler()

        # Epilogue (unified via _sub_tiles)
        def _get_acc_sub8(accs, acc_idx, vec_base):
            """Extract 8-element sub-vector from accumulator."""
            if const_expr(ACC_VEC_SIZE == 8):
                return accs[acc_idx]
            indices = [vec_base + i for i in range_constexpr(8)]
            return vector.shuffle(accs[acc_idx], accs[acc_idx], indices)

        def epilogue_prepare_addrs():
            addrs = []
            _bf16_out = out_dtype in ("bf16", "f16")
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                col_base = (blk_n + warp_n_base + arith.index(wn * WMMA_N)
                            + lane_kgrp * arith.index(8))
                if const_expr(_bf16_out):
                    c_off_bytes = (row * n_stride + col_base) \
                        * arith.index(elem_bytes_d)
                    addrs.append(c_off_bytes)
                else:
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        c_off = row * n_stride + col
                        addrs.append(c_off)
            return addrs

        _bf16_out = out_dtype in ("bf16", "f16")
        _out_elem_local = T.bf16 if out_dtype == "bf16" else \
            (T.f16 if out_dtype == "f16" else None)

        def epilogue_stores(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                if const_expr(_bf16_out):
                    addr_idx += store_acc_vec8_to_buffer(
                        sub8, c_rsrc, addrs[addr_idx],
                        out_elem=_out_elem_local, offset_is_bytes=True)
                else:
                    addr_idx += store_acc_vec8_to_buffer(
                        sub8, c_rsrc, addrs[addr_idx:addr_idx + 2])

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                imm = m_off * _lds_d_stride_elems + wn * _n_col_d_elems
                store_acc_vec8_to_lds(d_buf, d_base, imm, sub8,
                                      out_elem=_out_elem_local)

        def _atomic_add_acc_vec8_to_buffer(acc_vec8, addr):
            if const_expr(_bf16_out):
                h_vec = Vec(acc_vec8).truncf(T.vec(8, _out_elem_local))
                h_vec_w = Vec(h_vec)
                pair_ty = T.vec(2, _out_elem_local)
                for pair in range_constexpr(4):
                    pair_vec = vector.from_elements(
                        pair_ty, [h_vec_w[pair * 2], h_vec_w[pair * 2 + 1]])
                    byte_off = arith.index_cast(T.i32, addr + arith.index(pair * 4))
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        pair_vec, c_rsrc, byte_off, zero_i32, zero_i32)
                return 1

            acc_w = Vec(acc_vec8)
            for half in range_constexpr(2):
                base_addr = addr[half] if isinstance(addr, (list, tuple)) else addr
                for vi in range_constexpr(4):
                    val = acc_w[half * 4 + vi]
                    byte_off = arith.index_cast(
                        T.i32, (base_addr + arith.index(vi)) * arith.index(4))
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val, c_rsrc, byte_off, zero_i32, zero_i32)
            return 2

        def epilogue_atomic_adds(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                if const_expr(_bf16_out):
                    addr_idx += _atomic_add_acc_vec8_to_buffer(
                        sub8, addrs[addr_idx])
                else:
                    addr_idx += _atomic_add_acc_vec8_to_buffer(
                        sub8, addrs[addr_idx:addr_idx + 2])

        _effective_l2_pf = l2_prefetch_distance
        if const_expr(use_cluster and l2_prefetch_distance > 0):
            _effective_l2_pf = max(1, l2_prefetch_distance - 1)

        def _l2_prefetch(k_base):
            if const_expr(_effective_l2_pf <= 0):
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            pf_k_packed_a = pf_k / arith.index(PACK_FACTOR_A)
            pf_k_packed_b = pf_k / arith.index(PACK_FACTOR_B)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, pf_k_packed_a),
                (tile_m, packed_tile_k_a), (K_packed_a, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b,
                (blk_n / arith.index(16), pf_k_packed_b * arith.index(16)),
                (tile_n // 16, packed_tile_k_b * 16),
                (K_packed_b * 16, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)

        # ====== Multi-stage pipeline ======
        acc_zero = Vec.filled(ACC_VEC_SIZE, 0.0, fx.Float32)
        accs = [acc_zero] * n_accs

        lds_a_data_f16 = lds_a_data_bytes // 2
        lds_b_data_f16 = lds_b_data_bytes // 2

        arena_base_ptr = arena_alloc.get_base()

        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_data_off[i], elem_ty_lds,
                    shape=(lds_a_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_data_off[i], elem_ty_lds,
                    shape=(lds_b_data_f16,))
            for i in range_constexpr(num_buffers)
        ]

        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]

        stages_a_idx = [extract_lds_base_idx(stages_a[i])
                        for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i])
                        for i in range_constexpr(num_buffers)]

        if const_expr(use_tdm_store):
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(d_lds_base_ptr, d_output_off, elem_ty_lds,
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

        # Precompute LDS addresses for TDM descriptor switching
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(
                Vec(make_desc_a(stages_a_mem[i], arith.index(0)).dgroup0)[1])
            stages_b_lds_addr.append(
                Vec(make_desc_b(stages_b_mem[i], arith.index(0)).dgroup0)[1])

        desc_a_init = make_desc_a(stages_a_mem[0], split_k_base)
        desc_b_init = make_desc_b(stages_b_mem[0], split_k_base)

        adv_a_i32 = fx.Int32(tile_k)
        adv_b_i32 = fx.Int32(packed_tile_k_b * 16)

        pred_const = fx.Int32(1)

        # Track A and B addr_lo separately. With wave_specialized_tdm=True, all
        # waves still hold both A and B addr_lo SGPRs (small overhead: 1 extra
        # SGPR vs single active_addr_lo) — but the TDM op itself is gated by
        # wave_id inside ``issue_tdm_loads`` so only wave 0 issues A and wave 1
        # issues B; other waves skip the tensor_load_2d entirely.
        dg0_a_init = Vec(desc_a_init.dgroup0)
        dg0_b_init = Vec(desc_b_init.dgroup0)
        addr_lo_a = dg0_a_init[2]
        addr_hi_a = dg0_a_init[3]
        addr_lo_b = dg0_b_init[2]
        addr_hi_b = dg0_b_init[3]

        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1

        # Prologue
        for i in range_constexpr(pre_loaded):
            dg0_a = Vec.from_elements(
                [pred_const, stages_a_lds_addr[i], addr_lo_a, addr_hi_a], fx.Int32)
            dg0_b = Vec.from_elements(
                [pred_const, stages_b_lds_addr[i], addr_lo_b, addr_hi_b], fx.Int32)

            issue_tdm_loads(
                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                wave_specialized=wave_specialized_tdm)

            addr_lo_a = addr_lo_a + adv_a_i32
            addr_lo_b = addr_lo_b + adv_b_i32

        pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2),
                       use_cluster=use_cluster)

        # Main loop — acc_mixed style: fence at top, TDM_load mid-compute.
        # This overlaps TDM DMA with the remaining WMMA instructions,
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

        if const_expr(loop_iters > 0):
            init_args = list(accs) + [addr_lo_a, addr_lo_b]

            for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                accs_in = list(state[:n_accs])
                cur_lo_a = state[n_accs]
                cur_lo_b = state[n_accs + 1]

                for buf_idx in range_constexpr(num_buffers):
                    load_stage = (buf_idx + num_buffers - 1) % num_buffers
                    kt_idx = (loop_iter * arith.index(num_buffers)
                              + arith.index(buf_idx))
                    kt_idx = kt_idx + bz * arith.index(num_k_tiles)

                    pipeline_fence_signal(
                        outstanding=_fence_outstanding,
                        use_cluster=use_cluster)
                    pipeline_fence_wait(use_cluster=use_cluster)

                    addr_boxes = [[cur_lo_a], [cur_lo_b]]

                    def _mid_tdm(
                        _ls=load_stage,
                        _ab=addr_boxes,
                        _k_off=(split_k_base
                                + loop_iter * arith.index(num_buffers * tile_k)
                                + arith.index(buf_idx * tile_k)),
                    ):
                        dg0_a = Vec.from_elements(
                            [pred_const, stages_a_lds_addr[_ls],
                             _ab[0][0], addr_hi_a], fx.Int32)
                        dg0_b = Vec.from_elements(
                            [pred_const, stages_b_lds_addr[_ls],
                             _ab[1][0], addr_hi_b], fx.Int32)
                        issue_tdm_loads(
                            tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                            tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                            wave_specialized=wave_specialized_tdm)
                        _ab[0][0] = _ab[0][0] + adv_a_i32
                        _ab[1][0] = _ab[1][0] + adv_b_i32
                        _l2_prefetch(_k_off)

                    rocdl.sched_barrier(0)
                    accs_in = compute_tile_scheduled(
                        accs_in,
                        stages_a_idx[buf_idx],
                        stages_b_idx[buf_idx],
                        kt_idx,
                        mid_compute_callback=_mid_tdm)
                    cur_lo_a = addr_boxes[0][0]
                    cur_lo_b = addr_boxes[1][0]
                    hot_loop_scheduler_scheduled()

                results = yield list(accs_in) + [cur_lo_a, cur_lo_b]

            accs = list(results[:n_accs])
            addr_lo_a = results[n_accs]
            addr_lo_b = results[n_accs + 1]

        # Tail — same acc_mixed pattern: fence at top, TDM mid-compute.
        if const_expr(loop_iters > 0):
            pipeline_fence(outstanding=0, use_cluster=use_cluster)
        elif const_expr(use_cluster):
            gpu.cluster_barrier()
        epi_addrs_box = [None]
        _tail_had_load = False
        _tail_kt_base = loop_iters * num_buffers
        for _step_i, (_load_stage, _compute_stage, _outstanding) in enumerate(tail_plan):
            _kt_idx = (arith.index(_tail_kt_base + _step_i)
                       + bz * arith.index(num_k_tiles))
            if const_expr(_outstanding == -1):
                if const_expr(_tail_had_load):
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                if const_expr(use_tdm_store):
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                        _kt_idx)
                else:
                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                        _kt_idx,
                        emit_filler=_emit_epi_addrs)
            else:
                pipeline_fence_signal(outstanding=_outstanding,
                                      use_cluster=use_cluster)
                pipeline_fence_wait(use_cluster=use_cluster)

                _tail_mid_cb = None
                if const_expr(_load_stage is not None):
                    _tail_had_load = True
                    _tail_ab = [[addr_lo_a], [addr_lo_b]]

                    def _tail_mid(_ls=_load_stage, _ab=_tail_ab):
                        dg0_a = Vec.from_elements(
                            [pred_const, stages_a_lds_addr[_ls],
                             _ab[0][0], addr_hi_a], fx.Int32)
                        dg0_b = Vec.from_elements(
                            [pred_const, stages_b_lds_addr[_ls],
                             _ab[1][0], addr_hi_b], fx.Int32)
                        issue_tdm_loads(
                            tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                            tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                            wave_specialized=wave_specialized_tdm)
                        _ab[0][0] = _ab[0][0] + adv_a_i32
                        _ab[1][0] = _ab[1][0] + adv_b_i32

                    _tail_mid_cb = _tail_mid

                rocdl.sched_barrier(0)
                accs = compute_tile_scheduled(
                    accs,
                    stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                    _kt_idx,
                    mid_compute_callback=_tail_mid_cb)

                if const_expr(_load_stage is not None):
                    addr_lo_a = _tail_ab[0][0]
                    addr_lo_b = _tail_ab[1][0]

                hot_loop_scheduler_scheduled()

        if const_expr(use_tdm_store):
            if const_expr(d_need_epilogue_fence):
                pipeline_fence(outstanding=0, use_cluster=use_cluster)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            rocdl.sched_barrier(0)
            if const_expr(epi_addrs_box[0] is None):
                epi_addrs_box[0] = epilogue_prepare_addrs()
            if const_expr(split_k > 1):
                epilogue_atomic_adds(accs, epi_addrs_box[0])
            else:
                epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (data_format, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, scale_block_k, scale_block_n,
                 effective_waves_per_eu, l2_prefetch_distance,
                 cluster_m, cluster_n, use_tdm_store,
                 out_dtype, inst_prefetch, wave_specialized_tdm, split_k,
                 expert_sched_mode, atomic_barrier_enable)

    @flyc.jit
    def launch_blockscale_gemm(
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
            arena_alloc.finalized = False
            arena_alloc.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = (i32_n + (tile_n - 1)) // tile_n
        gz = split_k

        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        kernel_blockscale_gemm(
            arg_c,
            arg_a,
            arg_b,
            arg_a_scale,
            arg_b_scale,
            i32_m,
            i32_n,
            value_attrs={
                "rocdl.waves_per_eu": effective_waves_per_eu,
                "rocdl.cluster_dims": f"{cluster_m},{cluster_n},1" if const_expr(use_cluster) else None,
            },
        ).launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    if expert_sched_mode:
        launch_blockscale_gemm.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_blockscale_gemm


__all__ = ["compile_blockscale_gemm"]
