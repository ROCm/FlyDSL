"""Unified MXFP4/MXFP8/A8W4 GEMM kernel for gfx1250.

Supports FP4 (E2M1), FP8 (E4M3) and A8W4 (FP8 activation + FP4 weight),
selected via ``data_format="fp4"|"fp8"|"a8w4"``. Scales are either E8M0
block scales applied in-MMA (``scale_mode="mxscale"``) or per-token/
per-channel fp32 scales applied in the epilogue (``scale_mode="ptpc"``).
"""

import functools
import inspect
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from kernels.gemm_common_gfx1250 import (
    WGP_BARRIER_ID,
    extract_lds_base_idx,
    get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    store_acc_vec8_to_buffer,
    store_acc_vec8_to_lds,
)
from kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes


def _s_prefetch_inst_burst(num_pages: int, page_bytes: int = 4096):
    """gfx1250: prefetch ``num_pages`` × 4 KB of instructions ahead of PC.

    Caller must keep ``num_pages * page_bytes`` within shader bounds; over-reach
    page-faults.
    """
    from flydsl._mlir.dialects import llvm as _llvm

    lines = [f"s_prefetch_inst_pc_rel {pg * page_bytes}, null, 31" for pg in range(num_pages)]
    _llvm.inline_asm(None, [], "\n".join(lines), "", has_side_effects=True)


# compatible with no early_timeout descriptor
_TDM_HAS_EARLY_TIMEOUT = "early_timeout" in inspect.signature(tdm_ops.make_tensor_descriptor_2d).parameters


def _make_tdm_desc(*, early_timeout=False, **kwargs):
    """Build a TDM descriptor, applying early_timeout only when supports it."""
    if _TDM_HAS_EARLY_TIMEOUT:
        kwargs["early_timeout"] = early_timeout
    return tdm_ops.make_tensor_descriptor_2d(**kwargs)


# Common constants
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
SCALE_BLOCK = 32
SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK  # 4

LDS_PAD_A_BYTES = 16
LDS_PAD_D_BYTES = 16
# Tile-contiguous B (a8w4): one 16x16 byte tile = 256B. LDS may pad each tile to
# break bank conflicts; read side (load_b_frag) and write side (make_desc_b) both
# use B_TILE_BYTES + pad. 0 = no pad. Sweepable via FLYDSL_B_TILE_PAD.
B_TILE_BYTES = 256
LDS_PAD_B_TILE_BYTES = int(os.environ.get("FLYDSL_B_TILE_PAD", "16"))
LDS_SEGMENT_BYTES = 64 * 1024
LDS_GFX1250_MAX_BYTES = 5 * LDS_SEGMENT_BYTES


@functools.lru_cache(maxsize=256)
def compile_fp8fp4_gemm(
    *,
    data_format: str = "fp4",
    scale_mode: str = "mxscale",
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
    split_k: int = 1,
    use_scale_opsel: bool = False,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
    b_streaming: bool = False,
    scale_load_path: str = "tdm",
    fp8_schedule: str = "auto",
    pf_depth_wmma: int = None,
):
    """Compile an FP4/FP8/A8W4 GEMM kernel with TDM async copy.

    Args:
        data_format: "fp4" (E2M1), "fp8" (E4M3), or "a8w4" (FP8 act + FP4 weight).
        scale_mode: "mxscale" (E8M0 block scale via V_WMMA_SCALE) or "ptpc"
            (per-token sa[M] / per-channel sb[N] fp32, applied in the epilogue).

    Data layout:
        A: [M, K_packed] uint8 (FP4: K_packed=K//2, FP8: K_packed=K)
        B: [N, K_packed] uint8, preshuffled (16x16 byte tiles)
        mxscale: scale_A [M, K//32], scale_B [N, K//32] uint8 E8M0 (preshuffled)
        ptpc:    scale_A [M], scale_B [N] fp32

    Returns a JitFunction:
        launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, lda, ldc, stream)
    where lda/ldc are A/C runtime leading-dim strides in elements (dense: lda=K, ldc=N).
    """
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}")
    if scale_mode not in ("mxscale", "ptpc"):
        raise ValueError(f"scale_mode must be 'mxscale' or 'ptpc', got {scale_mode!r}")
    if scale_mode == "ptpc" and data_format not in ("fp8", "a8w4"):
        raise ValueError("scale_mode='ptpc' currently only supports data_format='fp8' or 'a8w4'")

    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"
    is_ptpc = scale_mode == "ptpc"

    if out_dtype not in ("f32", "bf16", "f16"):
        raise ValueError(f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}")
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4
    # scale_load_path: "tdm" = TDM->LDS (default); "vgpr" = buffer_load->VGPR,
    # off the LDS/TDM/barrier path; "vgpr_ab_split" = "vgpr" plus repurposing the
    # idle scale waves 2,3 to load the second A/B halves.
    scale_load_paths = ("tdm", "vgpr", "vgpr_ab_split")
    if scale_load_path not in scale_load_paths:
        raise ValueError(f"scale_load_path must be one of {scale_load_paths}, got {scale_load_path!r}")
    fp8_schedule_modes = ("auto", "quadrant", "deep-pipeline")
    if fp8_schedule not in fp8_schedule_modes:
        raise ValueError(f"fp8_schedule must be one of {fp8_schedule_modes}, got {fp8_schedule!r}")
    if fp8_schedule != "auto" and data_format != "fp8":
        raise ValueError(f"fp8_schedule={fp8_schedule!r} is only valid for data_format='fp8'")
    if fp8_schedule != "auto" and b_streaming:
        raise ValueError("fp8_schedule cannot be combined with b_streaming=True")
    effective_expert_sched_mode = bool(expert_sched_mode)

    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
    effective_waves_per_eu = waves_per_eu

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    if block_threads > 1024:
        raise ValueError(f"block_threads must be <= 1024, got {block_threads}")

    _min_wave_spec_warps = 2 if is_ptpc else 4
    if wave_specialized_tdm and num_warps < _min_wave_spec_warps:
        raise ValueError(f"wave_specialized_tdm requires at least {_min_wave_spec_warps} waves, got {num_warps}")

    # ── Format-dependent compile-time constants ──
    # A8W4: activation is FP8 (PACK_FACTOR_A=1), weight is FP4 (PACK_FACTOR_B=2)
    if is_a8w4:
        PACK_FACTOR_A = 1  # FP8 activation
        PACK_FACTOR_B = 2  # FP4 weight
    elif is_fp4:
        PACK_FACTOR_A = 2
        PACK_FACTOR_B = 2
    else:
        PACK_FACTOR_A = 1
        PACK_FACTOR_B = 1

    WMMA_N_EFF = 32 if is_fp4 else 16  # N-cols covered per WMMA instruction
    ACC_VEC_SIZE = 16 if is_fp4 else 8  # accumulator vector width
    DS_LOADS_PER_A_FRAG = 2 if is_fp4 else 4

    packed_tile_k_a = tile_k // PACK_FACTOR_A
    packed_tile_k_b = tile_k // PACK_FACTOR_B
    scale_k_per_tile = tile_k // SCALE_BLOCK
    K_packed_a = K // PACK_FACTOR_A
    K_packed_b = K // PACK_FACTOR_B
    K_scale = K // SCALE_BLOCK
    split_k_chunk = K // split_k

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if K % split_k != 0:
        raise ValueError(f"K must be divisible by split_k={split_k}, got K={K}")
    if split_k_chunk % tile_k != 0:
        raise ValueError(f"K/split_k must be divisible by tile_k={tile_k}, got {split_k_chunk}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if packed_tile_k_a % 4 != 0:
        raise ValueError(f"packed_tile_k_a must be a multiple of 4, got {packed_tile_k_a}")
    if packed_tile_k_b % 4 != 0:
        raise ValueError(f"packed_tile_k_b must be a multiple of 4, got {packed_tile_k_b}")
    if scale_k_per_tile % 4 != 0:
        raise ValueError(f"scale_k_per_tile must be a multiple of 4 (tile_k >= 128), got {scale_k_per_tile}")

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
        raise ValueError(f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, " f"got {num_k_tiles}")

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    n_accs = wmma_m_rep * wmma_n_rep
    # FP4 A/B swap: BScale rep derived from WMMA_M, not WMMA_N_EFF
    b_scale_load_rep = warp_tile_n // WMMA_M if is_fp4 else wmma_n_rep

    _b_frag_loads_per_wn = 2 if is_a8w4 else 4
    _a_frag_loads_per_wm = 2 if is_fp4 else 4
    _scale_ds_loads = (wmma_m_rep + 3) // 4 + (b_scale_load_rep + 3) // 4
    _bs_ds_loads = wmma_n_rep * _b_frag_loads_per_wn + _scale_ds_loads
    _as_ds_loads = wmma_m_rep * _a_frag_loads_per_wm + _scale_ds_loads

    lds_a_stride_bytes = packed_tile_k_a + LDS_PAD_A_BYTES
    if scale_load_path == "vgpr_ab_split":
        if tile_m % 2 != 0:
            raise ValueError(f"scale_load_path='vgpr_ab_split' requires even tile_m, got {tile_m}")
        if tile_n % 32 != 0:
            raise ValueError(f"scale_load_path='vgpr_ab_split' requires tile_n divisible by 32, got {tile_n}")

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    # Tile-contiguous B (a8w4): per-16x16-tile LDS pad. _b_pad=0 (non-a8w4) makes
    # these byte-identical to the original (packed_tile_k_b*16) layout.
    _b_pad = LDS_PAD_B_TILE_BYTES if is_a8w4 else 0
    _b_blk_stride = B_TILE_BYTES + _b_pad  # padded 16×16-tile byte step in LDS
    _b_ngroup_stride = (packed_tile_k_b // 16) * _b_blk_stride
    _b_num_blocks = (tile_n // 16) * (packed_tile_k_b // 16)
    # ks-major within-tile block order (a8w4 only): k-block outer, n-group inner, so
    # a ks-step is a contiguous block range (enables pad-free TDM K-split). The read
    # side uses _b_blk_stride as the n-group step and _b_kstep as the k-block step.
    # B_NGMAJOR=1 reverts to the verified n_group-major layout for comparison.
    _b_ksmajor = is_a8w4 and os.environ.get("B_NGMAJOR", "0") != "1"
    _b_kstep = (tile_n // 16) * _b_blk_stride  # k-block step in ks-major LDS
    if _b_pad:
        lds_b_data_bytes = _b_num_blocks * _b_blk_stride
    else:
        lds_b_data_bytes = tile_n * packed_tile_k_b  # HEAD formula
    ab_split_a_rows = tile_m // 2
    ab_split_b_groups = tile_n // 32
    _scale_guard_bytes = 16
    lds_a_scale_bytes = 0 if is_ptpc else tile_m * scale_k_per_tile + _scale_guard_bytes
    lds_b_scale_bytes = 0 if is_ptpc else tile_n * scale_k_per_tile + _scale_guard_bytes
    interleaved_scale_cols_a = wmma_m_rep * scale_k_per_tile
    interleaved_scale_cols_b = b_scale_load_rep * scale_k_per_tile

    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    # TDM descriptors partition a tile cooperatively across ``num_warps`` by
    # deriving per-wave offsets from ``wave_id``. In wave-specialized mode we
    # dedicate one loader wave to each tensor (A/B/A_scale/B_scale), so each
    # active loader wave must issue a full-tile descriptor by itself.
    tdm_desc_num_warps = 1 if wave_specialized_tdm else num_warps

    # All pipeline stages share the same intra-stage layout in the generic
    # arena path. The active gfx1250 FP8 TDM tile uses a separate reference
    # pool layout below.
    stage_layout = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"mxscale_{data_format}_layout")
    stage_a_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_data_rel_off + lds_a_data_bytes
    stage_b_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_data_rel_off + lds_b_data_bytes
    stage_a_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_scale_rel_off + lds_a_scale_bytes
    stage_b_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_scale_rel_off + lds_b_scale_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    _full_pf_req = os.environ.get("PF_FULL_PREFETCH", "0") == "1"
    pre_loaded = num_buffers if _full_pf_req else (num_buffers - 1)
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra, full_prefetch=_full_pf_req)

    _last_compute_stage = _base_tail_plan[-1][1]

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"mxscale_{data_format}_{tile_m}x{tile_n}x{tile_k}_" f"{m_warp}x{n_warp}_{num_buffers}buf_arena"
        ),
    )

    use_ref_segmented_lds_layout = (
        data_format == "fp8"
        and tile_m == 256
        and tile_n == 256
        and tile_k == 128
        and m_warp == 2
        and n_warp == 2
        and num_buffers == 4
        and split_k == 1
        and wave_specialized_tdm
        and not use_scale_opsel
    )

    # "vgpr"/"vgpr_ab_split": load scale global->VGPR via buffer_load, bypassing
    # TDM+LDS entirely. Requires the reference segmented LDS layout.
    use_buffer_vgpr_scale = scale_load_path in ("vgpr", "vgpr_ab_split")
    if use_buffer_vgpr_scale and not use_ref_segmented_lds_layout:
        raise ValueError(
            f"scale_load_path={scale_load_path!r} requires the reference segmented "
            "LDS layout (not active for this tile/format configuration)"
        )
    # Scale prefetch depth (K-tiles ahead) for the buffer->VGPR path. D=1 is the
    # sweet spot; D=2 doubles scale VGPRs -> spill + ~18% regression.
    _bvs_D = max(1, int(os.environ.get("FLYDSL_BUFFER_VGPR_SCALE_DEPTH", "1")))
    # ab_half_split: repurpose the (under "vgpr") idle scale waves 2,3 as the
    # second halves of A/B, so all 4 waves share the A/B TDM (wave0=A0, wave1=B0,
    # wave2=A1, wave3=B1). Measured wall-neutral.
    use_ab_half_split = scale_load_path == "vgpr_ab_split"
    # The buffer_load->VGPR scale ring is built only when scale is actually loaded.
    _bvs_active = use_buffer_vgpr_scale

    if use_ref_segmented_lds_layout:
        # The A/B data pools are no longer packed into the same per-stage
        # 64KiB segment window. Scale pools keep the reference 0x800 stride so
        # every TDM LDS target remains 2KiB-aligned.
        ref_a_stage_stride = 0x9000
        ref_b_stage_stride = 0x8000
        ref_scale_stage_stride = 0x800
        if lds_a_data_bytes > ref_a_stage_stride:
            raise RuntimeError(
                "reference segmented LDS layout requires A stage <= 0x9000 bytes, " f"got {lds_a_data_bytes}"
            )
        if lds_b_data_bytes > ref_b_stage_stride:
            raise RuntimeError(
                "reference segmented LDS layout requires B stage <= 0x8000 bytes, " f"got {lds_b_data_bytes}"
            )
        if lds_a_scale_bytes > ref_scale_stage_stride or lds_b_scale_bytes > ref_scale_stage_stride:
            raise RuntimeError(
                "reference segmented LDS layout requires scale stage <= 0x800 bytes, "
                f"got A={lds_a_scale_bytes} B={lds_b_scale_bytes}"
            )

        stage_a_data_off = [0x00000, 0x09000, 0x16000, 0x1F000]
        stage_a_scale_off = [0x12000 + i * ref_scale_stage_stride for i in range(num_buffers)]
        stage_b_scale_off = [0x28000 + i * ref_scale_stage_stride for i in range(num_buffers)]
        stage_b_data_off = [0x30000 + i * ref_b_stage_stride for i in range(num_buffers)]
        arena_alloc.ptr = LDS_GFX1250_MAX_BYTES
        arena_total_bytes = arena_alloc.ptr

        # The epilogue may reuse the prefix only after all main/tail TDM traffic
        # is fully fenced. This is outside the hot loop and avoids assuming a
        # single monotonic per-stage base for the segmented pool layout.
        epilogue_fence_threshold_bytes = 0
    else:
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

        stage_a_data_off = [stage_base_off[i] + stage_a_data_rel_off for i in range(num_buffers)]
        stage_b_data_off = [stage_base_off[i] + stage_b_data_rel_off for i in range(num_buffers)]
        stage_a_scale_off = [stage_base_off[i] + stage_a_scale_rel_off for i in range(num_buffers)]
        stage_b_scale_off = [stage_base_off[i] + stage_b_scale_rel_off for i in range(num_buffers)]

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

    # TENSORcnt is tracked per-wave in hardware. Wave-specialized TDM issues one
    # tensor_load per wave per step; otherwise all 4 (A/B/A_scale/B_scale).
    if wave_specialized_tdm:
        TDM_LOADS_PER_STEP = 1
    else:
        TDM_LOADS_PER_STEP = 4
    tail_plan = [(ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o) for ls, cs, o in _base_tail_plan]

    # Pre-compute epilogue sub-tile layout (unified for FP4 vec16 and FP8 vec8)
    _sub_tiles = []
    for _wm in range(wmma_m_rep):
        for _wn in range(wmma_n_rep):
            if is_fp4:
                # vec<16,f32>: split into 2 × 8 elements (2 × 16-col halves)
                for _half in range(2):
                    acc_idx = _wm * wmma_n_rep + _wn
                    vec_base = _half * 8
                    m_off = _wm * WMMA_M
                    n_sub = _wn * 2 + _half
                    _sub_tiles.append((acc_idx, vec_base, m_off, n_sub))
            else:
                # vec<8,f32>: single 8-element block
                acc_idx = _wm * wmma_n_rep + _wn
                m_off = _wm * WMMA_M
                n_sub = _wn
                _sub_tiles.append((acc_idx, 0, m_off, n_sub))

    COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING = "row_major_streaming"
    COMPUTE_SCHEDULE_FP4_COL_BAND = "fp4_col_band"
    COMPUTE_SCHEDULE_FP8_QUADRANT = "fp8_quadrant"
    COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE = "fp8_deep_pipeline"
    COMPUTE_SCHEDULE_B_STREAMING = "b_streaming"

    fp8_deep_pipeline_eligible = (
        data_format in ("fp8", "a8w4")
        and tile_m == 256
        and tile_n == 256
        and tile_k == 128
        and m_warp == 2
        and n_warp == 2
        and num_buffers == 4
        and wave_specialized_tdm
        and out_dtype == "bf16"
        and not use_scale_opsel
    )
    if fp8_schedule == "deep-pipeline" and not fp8_deep_pipeline_eligible:
        raise ValueError(
            "fp8_schedule='deep-pipeline' requires fp8 256x256x128, "
            "m_warp=n_warp=2, num_buffers=4, wave_specialized_tdm=True, "
            "out_dtype='bf16', and use_scale_opsel=False"
        )

    def _pick_compute_schedule_kind():
        if b_streaming:
            return COMPUTE_SCHEDULE_B_STREAMING
        # PF_FORCE_RMS=1: force the row-major-streaming schedule even for n_accs>=8 shapes
        # (normally FP8_QUADRANT), so the WMMA-driven sub-depth prefetch pipeline
        # (_use_lds_pf, gated on RMS) engages for the larger quadrant-eligible tiles too.
        if os.environ.get("PF_FORCE_RMS", "0") == "1":
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        if wmma_m_rep % 2 != 0 or wmma_n_rep % 2 != 0 or n_accs < 8:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        # Quadrant schedules split B into left/right halves and compute
        # top-left, bottom-left, top-right, bottom-right. FP4 additionally
        # changes accumulator layout for bank friendliness; FP8 keeps row-major
        # accumulators and uses the split to increase LDS-load-to-WMMA distance.
        if is_fp4:
            return COMPUTE_SCHEDULE_FP4_COL_BAND
        # A8W4 (FP8 act + FP4 weight) shares FP8's accumulator layout and operand
        # path, so it reuses the FP8 schedules.
        if data_format in ("fp8", "a8w4"):
            if fp8_schedule == "deep-pipeline" or (fp8_schedule == "auto" and fp8_deep_pipeline_eligible):
                return COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
            return COMPUTE_SCHEDULE_FP8_QUADRANT
        return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING

    compute_schedule_kind = _pick_compute_schedule_kind()
    use_fp4_bank_friendly_schedule = compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND
    use_fp8_quadrant_schedule = compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT
    use_fp8_deep_pipeline_schedule = compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
    use_b_streaming_schedule = compute_schedule_kind == COMPUTE_SCHEDULE_B_STREAMING
    if use_buffer_vgpr_scale and not use_fp8_deep_pipeline_schedule:
        raise ValueError(f"scale_load_path={scale_load_path!r} is only supported with the FP8 deep-pipeline schedule")
    use_ws_tdm_split_signal_overlap = (
        wave_specialized_tdm
        and (use_fp8_quadrant_schedule or use_fp8_deep_pipeline_schedule)
        and num_buffers == 4
        and use_cluster
    )
    if use_b_streaming_schedule:
        print(
            f"[b_streaming] {data_format} tile=({tile_m},{tile_n},{tile_k}) " f"M_r={wmma_m_rep} N_r={wmma_n_rep}",
            flush=True,
        )

    if use_fp4_bank_friendly_schedule:
        _bank_half_wm = wmma_m_rep // 2
        _bank_half_wn = wmma_n_rep // 2
        _bank_group_size = _bank_half_wm * _bank_half_wn
        _bank_half_b_scale_rep = b_scale_load_rep // 2
        _bank_group_to_row_major = []
        for _wm in range(_bank_half_wm):
            for _wn in range(_bank_half_wn):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm, wmma_m_rep):
            for _wn in range(_bank_half_wn):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm):
            for _wn in range(_bank_half_wn, wmma_n_rep):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm, wmma_m_rep):
            for _wn in range(_bank_half_wn, wmma_n_rep):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)

    if use_fp8_quadrant_schedule or use_fp8_deep_pipeline_schedule:
        _fp8_half_wm = wmma_m_rep // 2
        _fp8_half_wn = wmma_n_rep // 2
        _fp8_group_size = _fp8_half_wm * _fp8_half_wn
        _fp8_b_scale_loads = 0 if is_ptpc else (b_scale_load_rep + 3) // 4
    if use_fp8_deep_pipeline_schedule:
        _fp8_pair_wm = 2
        _fp8_pair_wn = 2
        _fp8_wm_pairs = wmma_m_rep // _fp8_pair_wm
        _fp8_wn_pairs = wmma_n_rep // _fp8_pair_wn
        _fp8_pair_a_loads = _fp8_pair_wm * DS_LOADS_PER_A_FRAG
        _fp8_pair_b_loads = _fp8_pair_wn * _b_frag_loads_per_wn
        _fp8_scale_loads = 0 if is_ptpc else (wmma_m_rep + 3) // 4 + (b_scale_load_rep + 3) // 4

    # ── Compile-time LDS prefetch offset table ────────────────────────────────
    # Built here (outside @flyc.kernel) so plain Python loops are safe.
    # Consumed inside the kernel as a closed-over Python constant.
    # Tags: 0=A, 1=B, 2=AS, 3=BS
    _PF_TAG_A, _PF_TAG_B, _PF_TAG_AS, _PF_TAG_BS = 0, 1, 2, 3
    # Order-graft: the carry/lazy-assembly pipeline is order-agnostic (compute_tile just
    # iterates _pf_pos and loads _pf_wmma_plan_flat[i]), so the SAME machinery serves
    # FP8_QUADRANT by building the table in QUADRANT emit-order instead of RMS snake.
    # Gated by PF_QUADRANT=1 + tile_m<=64 (tile_m=128 needs a separate design); default
    # off = zero change to existing behavior.
    _pf_allow_quadrant = (
        os.environ.get("PF_QUADRANT", "0") == "1"
        and compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT
        and tile_m <= 64
    )
    _use_lds_pf_outer = (
        wave_specialized_tdm
        and (compute_schedule_kind == COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING or _pf_allow_quadrant)
        and not _bvs_active
    )
    if _use_lds_pf_outer:
        _pf_bs_stride = 2 if (use_scale_opsel and not is_fp4) else 1
        _pf_num_tiles_b = WMMA_K // PACK_FACTOR_B // 16

        # Per-tile WMMA emit order: list of (wm, wn). The table, _pf_pos, and
        # _pf_wmma_plan_flat are all built in THIS order, so compute_tile emits in it.
        if compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT:
            # Quadrant: top-left rows, bottom-left rows, top-right cols, bottom-right
            # cols — mirrors compute_tile_fp8_quadrant's _emit_group / _emit_group_col.
            _hwm, _hwn = wmma_m_rep // 2, wmma_n_rep // 2
            _pf_order = (
                [(_wm, _wn) for _wm in range(_hwm) for _wn in range(_hwn)]            # TL
                + [(_wm, _wn) for _wm in range(_hwm, wmma_m_rep) for _wn in range(_hwn)]  # BL
                + [(_wm, _wn) for _wn in range(_hwn, wmma_n_rep) for _wm in range(_hwm)]  # TR (col-major)
                + [(_wm, _wn) for _wn in range(_hwn, wmma_n_rep) for _wm in range(_hwm, wmma_m_rep)]  # BR
            )
        else:
            # RMS: snake wn for odd wm (the original order).
            _pf_order = [
                (_wm, (wmma_n_rep - 1 - _wnr) if (_wm % 2 == 1) else _wnr)
                for _wm in range(wmma_m_rep)
                for _wnr in range(wmma_n_rep)
            ]

        _pf_load_table = []      # [(tag, imm), ...] in WMMA emit order
        _pf_plans = []           # per ks: (as_start, as_n, a_groups[wm], b_groups[wn], bs_groups)
        _pf_pos = []             # per emitted WMMA: (ks, wm, wn)
        _pf_wmma_plan_flat = []  # per emitted WMMA: (table_start, count)

        # Per-operand load emitters (identical bytes to the legacy build). Each appends
        # to _pf_load_table and returns its (start, count) group descriptor. The A-imm
        # formula is uniform: wm=0 reduces wm*WMMA_M*stride to 0 = the old wm==0 case.
        def _emit_a_load(wm, ks):
            _a_imm = wm * WMMA_M * lds_a_stride_bytes + ks * (WMMA_K // PACK_FACTOR_A)
            _start = len(_pf_load_table)
            for _sub in range(DS_LOADS_PER_A_FRAG):
                _pf_load_table.append((_PF_TAG_A, _a_imm + _sub * 32))
            return (_start, DS_LOADS_PER_A_FRAG)

        def _emit_b_load(wn, ks):
            _start = len(_pf_load_table)
            if is_fp4:
                _imm0 = wn * 2 * _b_ngroup_stride + ks * _pf_num_tiles_b * 256
                _imm1 = (wn * 2 + 1) * _b_ngroup_stride + ks * _pf_num_tiles_b * 256
                _pf_load_table.extend([(_PF_TAG_B, _imm0), (_PF_TAG_B, _imm0 + 512),
                                       (_PF_TAG_B, _imm1), (_PF_TAG_B, _imm1 + 512)])
                return (_start, 4)
            if is_a8w4:
                # Mirror load_b_frag's a8w4 LDS layout EXACTLY (ks-major default vs
                # n-group-major) so raw sub-depth B reads hit the assembled offsets.
                if _b_ksmajor:
                    _imm = wn * _b_blk_stride + ks * _pf_num_tiles_b * _b_kstep
                    _bstep = 2 * _b_kstep
                else:
                    _imm = wn * _b_ngroup_stride + ks * _pf_num_tiles_b * _b_blk_stride
                    _bstep = 2 * _b_blk_stride
                _pf_load_table.extend([(_PF_TAG_B, _imm), (_PF_TAG_B, _imm + _bstep)])
                return (_start, 2)
            _imm = wn * _b_ngroup_stride + ks * _pf_num_tiles_b * 256
            for _so in (0, 512, 1024, 1536):
                _pf_load_table.append((_PF_TAG_B, _imm + _so))
            return (_start, 4)

        def _emit_as_load(ks):
            _as_n = (wmma_m_rep + 3) // 4
            _as_off = ks * wmma_m_rep * SCALES_PER_WMMA
            _start = len(_pf_load_table)
            for _ld in range(_as_n):
                _pf_load_table.append((_PF_TAG_AS, _as_off + _ld * 16))
            return (_start, _as_n)

        def _emit_bs_group_load(bsg, ks):
            _rep_start = bsg * _pf_bs_stride
            _gs = min(_pf_bs_stride, b_scale_load_rep - _rep_start)
            _bs_n = (_gs + 3) // 4
            _bs_off = (ks * b_scale_load_rep + _rep_start) * SCALES_PER_WMMA
            _start = len(_pf_load_table)
            for _ld in range(_bs_n):
                _pf_load_table.append((_PF_TAG_BS, _bs_off + _ld * 16))
            return (_start, _bs_n, _gs)

        for _pf_ks in range(k_wmma_steps):
            # Load each operand at its FIRST use in emit order (AS once per ks at the
            # first WMMA; BS once per group; A once per wm; B once per wn). Per-WMMA
            # slice order is AS,BS,A,B — for snake this reproduces the legacy table
            # byte-for-byte; for quadrant it lays loads out in quadrant order.
            _a_seen, _b_seen, _bs_seen, _recs = {}, {}, {}, {}
            _as_rec = None
            for (_wm, _wn) in _pf_order:
                _wmma_start = len(_pf_load_table)
                if _as_rec is None:
                    _as_rec = _emit_as_load(_pf_ks)
                if not is_ptpc:
                    _bsg = _wn // _pf_bs_stride
                    if _bsg not in _bs_seen:
                        _bs_seen[_bsg] = _emit_bs_group_load(_bsg, _pf_ks)
                if _wm not in _a_seen:
                    _a_seen[_wm] = _emit_a_load(_wm, _pf_ks)
                if _wn not in _b_seen:
                    _b_seen[_wn] = _emit_b_load(_wn, _pf_ks)
                _recs[(_wm, _wn)] = (_wmma_start, len(_pf_load_table) - _wmma_start)
            _as_start, _as_n = _as_rec
            _a_groups = [_a_seen[_wm] for _wm in range(wmma_m_rep)]   # indexed by wm
            _b_groups = [_b_seen[_wn] for _wn in range(wmma_n_rep)]   # indexed by wn
            _bs_groups = [_bs_seen[_g] for _g in sorted(_bs_seen.keys())]
            _pf_plans.append((_as_start, _as_n, _a_groups, _b_groups, _bs_groups))
            for (_wm, _wn) in _pf_order:
                _pf_pos.append((_pf_ks, _wm, _wn))
                _pf_wmma_plan_flat.append(_recs[(_wm, _wn)])

        # Per-ks scale SOURCE maps for the lazy (sub-ks) assembler: which
        # (table_idx, sub) each PRE-opsel a_scales[r] / b_scales[f] entry reads from
        # the raw carry. Mirrors _assemble_pf_ks's indexing exactly so the lazy path
        # can materialize a SINGLE b_scale entry when its WMMA needs it — a shallow
        # carry (_pf_D < wmma_n_rep) does not yet hold all of a ks-step's BS loads.
        _pf_as_src = []   # _pf_as_src[ks][r]  -> (table_idx, sub)  (a_scales, pre-opsel)
        _pf_bs_src = []   # _pf_bs_src[ks][f]  -> (table_idx, sub)  (b_scales, pre-opsel)
        for _sks in range(k_wmma_steps):
            _as_start, _as_n, _a_g, _b_g, _bs_groups = _pf_plans[_sks]
            _pf_as_src.append([(_as_start + r // 4, r % 4) for r in range(wmma_m_rep)])
            _bs_map = []
            for _bs_start, _bs_n, _gs in _bs_groups:
                for _i in range(_gs):
                    _bs_map.append((_bs_start + _i // 4, _i % 4))
            _pf_bs_src.append(_bs_map)

    # ── pf_depth_wmma: WMMA-granular prefetch depth ───────────────────────────
    # The cross-loop carry holds operands for the first _pf_D WMMAs of a tile.
    # In compute_tile, WMMA i consumes position i (carry if i<D, else cur_raw) and
    # issues the ds_load for position j=i+D (cur_raw "remain" if j<A, else the next
    # tile's carry "nxt_raw").  Two boundaries — i=D (operand source flips) and
    # i=A-D (load target flips remain→next-pf) — reproduce the 1/2/3-phase cases.
    # pf_depth_wmma=None means full depth (_pf_D == A): the original assembled-carry
    # path is kept verbatim; only D<A uses the raw-vector pipeline below.
    if _use_lds_pf_outer:
        _pf_wpks = wmma_m_rep * wmma_n_rep            # WMMAs per ks-step
        _A_wmma = k_wmma_steps * _pf_wpks             # total WMMAs in a tile
        _raw_D = _A_wmma if pf_depth_wmma is None else int(pf_depth_wmma)
        _pf_D = max(1, min(_A_wmma, _raw_D))
        # Full-prefetch backfills the buffer being consumed NOW, so the carry/look-
        # ahead depth must align with the pf/remain TDM SEGMENT boundary (a whole
        # ks-step). A sub-ks carry (D < _pf_wpks) makes the look-ahead read the
        # remain region before the pf seg is consumed, so cb_pf cannot find a clean
        # point to overwrite the live buffer's pf seg (it is either still being read
        # or its carry is still in flight) -> data race on buffer reuse. Snap D UP to
        # a ks multiple here so each phase maps 1:1 to a TDM segment (the design the
        # WMMA-driven pipeline assumes). Legacy (non-full-prefetch) keeps sub-ks.
        if _full_pf_req and os.environ.get("PF_PIPELINE", "0") == "1":
            _pf_D = min(_A_wmma, ((_pf_D + _pf_wpks - 1) // _pf_wpks) * _pf_wpks)
        _pf_full_depth = _pf_D == _A_wmma
        # Sub-depth reads the tile's "remain" operands from its own LDS buffer
        # mid-compute. That is only safe when the buffer is never reused: once a
        # buffer is recycled, its reuse TDM overwrites the slot between the carry
        # read (one tile ahead) and the remain read (this tile), so the remain
        # gets the next occupant's data. Reuse happens iff num_k_tiles > num_buffers.
        # Fall back to full prefetch (always correct) when reuse is possible.
        # PF_PIPELINE=1 implements the deferred-remain WMMA-driven pipeline that makes
        # the reuse case safe (remain TDM issued only after this tile's remain is read),
        # so skip the fallback there.
        if not _pf_full_depth and num_k_tiles > num_buffers and os.environ.get("PF_PIPELINE", "0") != "1":
            import warnings as _pf_warn
            _pf_warn.warn(
                f"pf_depth_wmma={pf_depth_wmma} ignored: num_k_tiles({num_k_tiles}) > "
                f"num_buffers({num_buffers}) causes LDS buffer reuse, which the sub-depth "
                f"path cannot read safely; using full prefetch."
            )
            _pf_D = _A_wmma
            _pf_full_depth = True
        # Sub-ks-step depth IS supported: the carry (_pf_split_idx below) is per-WMMA,
        # so _pf_D can be any 1.._A_wmma. The only quantity that must stay ks-aligned is
        # the TDM seg split (_pf_Dk), because a ks-step's WMMAs all share the same K=128
        # range (WMMA_K) — the DRAM->LDS TDM cannot split finer than a ks-step. We round
        # _pf_Dk UP (ceil) so the carry always lies WITHIN the pf seg and never spills
        # into the deferred remain region (which would read it before its TDM is issued).
        # So a sub-ks _pf_D (e.g. 4 of 8) gives a finer CARRY depth while the seg split
        # stays at the first ks boundary (pf=ks0, remain=ks1+).
        # (_pf_pos and _pf_wmma_plan_flat were built in emit order in the table loop.)
        # Table entries [0:_pf_split_idx) are the carry (first _pf_D WMMAs).
        _pf_split_idx = (
            _pf_wmma_plan_flat[_pf_D][0] if _pf_D < _A_wmma else len(_pf_load_table)
        )
        # ── K-split (pf/remain) quantities for WMMA-driven TDM prefetch ───────
        # The tile's K is split at the pf/remain boundary (Dk ks-steps) so pf and
        # remain can be loaded by SEPARATE TDMs; the remain TDM is deferred until
        # the current tile's remain has been read, so a reused buffer's remain
        # region is not overwritten under the read (fixes the reuse data race).
        # All operands are K-contiguous (A: per M-row, B: per n-group, scales:
        # ks-major), so the split is a contiguous sub-range of the inner K dim.
        # ceil so the pf seg fully covers the carry: _pf_D=4,_pf_wpks=8 -> _pf_Dk=1
        # (pf seg = ks0, carry = first 4 WMMAs ⊆ ks0). Backward compatible: when _pf_D
        # is a ks-multiple, ceil == floor, so existing depths are unchanged.
        _pf_Dk = (_pf_D + _pf_wpks - 1) // _pf_wpks       # ks-steps spanned by pf seg
        # _pf_tdm_Dk = ks-boundary at which the per-tile TDM is split into pf/remain.
        # PF_FORCE_DK lets us validate the K-split descriptors on the full-depth path.
        _pf_force_dk = int(os.environ.get("PF_FORCE_DK", "0"))
        _pf_tdm_Dk = _pf_force_dk if _pf_force_dk > 0 else _pf_Dk
        _pf_tdm_split = 0 < _pf_tdm_Dk < k_wmma_steps
        # WMMA index of the pf/remain segment boundary: the next_pf TDM (cb_pf)
        # overwrites the pf seg (ks0.._pf_tdm_Dk-1), which is fully CONSUMED only at
        # this WMMA. When the cb backfills the buffer being consumed NOW (full-
        # prefetch), cb_pf MUST fire here, not at the (smaller) carry depth _pf_D —
        # else it clobbers the still-unread pf-seg tail [_pf_D : _pf_seg_wmmas).
        _pf_seg_wmmas = _pf_tdm_Dk * _pf_wpks
        # DEBUG: PF_TDM_SPLIT_OFF=1 forces the single full TDM path (prologue + tail)
        # even for sub-depth, to isolate the sub-depth COMPUTE from the K-split TDM
        # wiring. (Temporary diagnostic — remove after Stage C.)
        if os.environ.get("PF_TDM_SPLIT_OFF", "0") == "1":
            _pf_tdm_split = False
        # Per-operand split boundary (in ks-steps) for isolation debugging.
        # PF_SPLIT_OP=a|b|as|bs splits ONLY that operand at _pf_tdm_Dk; the others
        # load the full tile in BOTH seg slots (idempotent re-write, always correct),
        # so a NaN pins the bug to the single split operand. Default "all" splits all.
        _pf_split_op = os.environ.get("PF_SPLIT_OP", "all")

        def _op_dk(op):
            return _pf_tdm_Dk if _pf_split_op in ("all", op) else k_wmma_steps

        _dk_a, _dk_b = _op_dk("a"), _op_dk("b")
        # Scales (AS/BS) are K-split at the SAME pf/remain boundary as A/B. They MUST
        # be split: loading the full scale tile in BOTH segs writes the same LDS scale
        # region (offset 0) from cb_pf AND cb_rem. In the deep WMMA-driven pipeline both
        # TDMs are in flight at once, so a tile reusing the buffer reads a torn/raced
        # scale value -> nondeterministic ~0.1% wrong outputs (constant scales hid it).
        # Splitting makes pf write ks0 scales and remain write ks1 scales into disjoint
        # LDS, identical to A/B (which are race-free), so the scale read is safe at full
        # pipeline depth. _seg_cols falls back to full-in-both when the scale tile is too
        # small to split (pk >= full), which stays correct (single small region).
        _dk_as, _dk_bs = _op_dk("as"), _op_dk("bs")

        # Per-operand (col_off, col_len) for seg 0 (pf) and seg 1 (remain), computed
        # at config scope (pure Python) so the traced make_desc_*_seg only index a
        # tuple — no in-kernel branch for the AST rewriter to mangle. A non-split
        # operand (dk == k_wmma_steps) loads the full inner extent in BOTH segs.
        def _seg_cols(dk, full, unit):
            pk = dk * unit
            if pk >= full:
                return [(0, full), (0, full)]
            return [(0, pk), (pk, full - pk)]

        _cols_a = _seg_cols(_dk_a, packed_tile_k_a, WMMA_K // PACK_FACTOR_A)
        _cols_b = _seg_cols(_dk_b, packed_tile_k_b, WMMA_K // PACK_FACTOR_B)
        _cols_as = _seg_cols(_dk_as, interleaved_scale_cols_a, wmma_m_rep * SCALES_PER_WMMA)
        _cols_bs = _seg_cols(_dk_bs, interleaved_scale_cols_b, b_scale_load_rep * SCALES_PER_WMMA)
        # ks-major B: K-split is a contiguous BLOCK range. (blk_off, blk_len) per seg,
        # in 256B blocks; a ks-step spans _b_nblk_per_ks*ng_count blocks.
        _b_nblk_per_ks = (WMMA_K // PACK_FACTOR_B // 16) * (tile_n // 16)
        _cols_b_blk = _seg_cols(_dk_b, _b_num_blocks, _b_nblk_per_ks)
    else:
        _pf_load_table, _pf_plans, _pf_wmma_plan = [], [], []
        _pf_as_src, _pf_bs_src = [], []
        _pf_wpks = 0
        _A_wmma = 0
        _pf_D = 0
        _pf_full_depth = True
        _pf_wmma_plan_flat = []
        _pf_pos = []
        _pf_split_idx = 0
        _pf_tdm_Dk = 0
        _pf_tdm_split = False
        _pf_pk_a = _pf_pk_b = _pf_sc_a = _pf_sc_b = 0

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        # Enable back-to-back WMMA issue (SCHED_MODE bit[4] = DISABLE_VALU_STALL)
        rocdl.disable_xdl_arb_stall()

        if const_expr(inst_prefetch):
            if rocdl.wave_id() == fx.Int32(0):
                _s_prefetch_inst_burst(num_pages=4)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = fx.Index(gpu.block_idx.z) if split_k > 1 else arith.index(0)

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        split_k_base = bz * arith.index(split_k_chunk)

        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        # The FP8 deep pipeline runs cleaner when adjacent wave ids advance M
        # first; keep the default mapping for the other schedules.
        if const_expr(use_fp8_deep_pipeline_schedule):
            layout_thr = fx.make_layout((m_warp, n_warp, 2, 16), (WAVE_SIZE, m_warp * WAVE_SIZE, 16, 1))
        else:
            layout_thr = fx.make_layout((m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0),
            fx.get(thr_coord, 1),
            fx.get(thr_coord, 2),
            fx.get(thr_coord, 3),
        )

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        if const_expr(use_buffer_vgpr_scale):
            # Direct global->VGPR scale load (no TDM/LDS). Coalesced lane-major
            # host layout [M_block(128), K_tile, group(2), lane16(16), 4 i32], so
            # each buffer_load_b128's 16 lanes read 256 contiguous bytes:
            #   i32_off(group) = (mb*Kt + kt)*128 + group*64 + lane16*4
            _bvs_a_rsrc = buffer_ops.create_buffer_resource(arg_a_scale, max_size=False)
            _bvs_b_rsrc = buffer_ops.create_buffer_resource(arg_b_scale, max_size=False)
            _bvs_Kt = K // tile_k  # total K-tiles
            _bvs_mb_a = blk_m / arith.index(128) + wave_m_idx
            _bvs_mb_b = blk_n / arith.index(128) + wave_n_idx
            _bvs_lane4 = lane16 * arith.index(4)

            def _bvs_load_scales(rsrc, mb, rep, k_base):
                kt = k_base / arith.index(tile_k)
                tile_i32 = (mb * arith.index(_bvs_Kt) + kt) * arith.index(128)
                vals = []
                for ld in range_constexpr(rep // 4):  # rep=8 -> 2 groups of 4 i32
                    off = arith.index_cast(T.i32, tile_i32 + arith.index(ld * 64) + _bvs_lane4)
                    v = fx.Vector(buffer_ops.buffer_load(rsrc, off, vec_width=4, dtype=T.i32))
                    for j in range_constexpr(4):
                        vals.append(v[j])
                return vals

            def _bvs_prefetch(k_base):
                # Issue scale buffer_load for one K-tile; returns (a[8], b[8]) VGPR.
                a = _bvs_load_scales(_bvs_a_rsrc, _bvs_mb_a, wmma_m_rep, k_base)
                b = _bvs_load_scales(_bvs_b_rsrc, _bvs_mb_b, b_scale_load_rep, k_base)
                return a, b

        m_idx = fx.Index(i32_m)
        # Runtime leading-dim strides (strided A/C). Dense callers pass lda == K,
        # ldc == N for byte-identical addressing. A's stride is in packed elements.
        if const_expr(PACK_FACTOR_A == 1):
            lda_packed = fx.Index(i32_lda)
        else:
            lda_packed = fx.Index(i32_lda) / arith.index(PACK_FACTOR_A)
        n_stride = fx.Index(i32_ldc)
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)
        c_global_ptr_type = ir.Type.parse("!llvm.ptr<1>")
        c_global_base_i64 = llvm.PtrToIntOp(
            T.i64, fly.extract_aligned_pointer_as_index(c_global_ptr_type, arg_c.__extract_to_ir_values__()[0])
        ).result

        def make_desc_a(memref, k_base):
            k_packed_off = k_base / arith.index(PACK_FACTOR_A)
            return _make_tdm_desc(
                global_ptr=arg_a,
                lds_memref=memref,
                global_offset=(blk_m, k_packed_off),
                tensor_shape=(tile_m, packed_tile_k_a),
                strides=(lda_packed, 1),
                tile_shape=(tile_m, packed_tile_k_a),
                elem_bytes=1,
                pad_interval=packed_tile_k_a,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=i32_m,
            )

        def make_desc_b(memref, k_base):
            if const_expr(is_a8w4):
                # Tile-contiguous B (preshuffle_b_16x16_tiled): global is a flat
                # array of 256B blocks ordered [n_tile, k_tile, ng_loc, kt_loc, 256B],
                # so one GEMM tile's _b_num_blocks blocks are contiguous. TDM dim0 =
                # 256B block, tile_shape dim0 = whole-tile block count -> one
                # contiguous DMA per tile. pad_interval/pad_amount add the LDS-write
                # padding to break bank conflicts (global side flat).
                _kt = K_packed_b // packed_tile_k_b  # K-direction GEMM tiles
                bn = blk_n / arith.index(tile_n)
                bk = k_base / arith.index(tile_k)
                tile_blk0 = (bn * arith.index(_kt) + bk) * arith.index(_b_num_blocks)
                return _make_tdm_desc(
                    global_ptr=arg_b,
                    lds_memref=memref,
                    global_offset=(tile_blk0, arith.index(0)),
                    tensor_shape=(N * K_packed_b // B_TILE_BYTES, B_TILE_BYTES),
                    strides=(B_TILE_BYTES, 1),
                    tile_shape=(_b_num_blocks, B_TILE_BYTES),
                    elem_bytes=1,
                    pad_interval=B_TILE_BYTES if _b_pad else 0,
                    pad_amount=_b_pad,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=b_mcast_mask,
                    atomic_barrier_enable=atomic_barrier_enable,
                    early_timeout=True,
                )
            k_packed_off = k_base / arith.index(PACK_FACTOR_B)
            return _make_tdm_desc(
                global_ptr=arg_b,
                lds_memref=memref,
                global_offset=(blk_n / arith.index(16), k_packed_off * arith.index(16)),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(tile_n // 16, packed_tile_k_b * 16),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        def make_desc_a_half(memref, k_base, m_half: int):
            row_start = m_half * ab_split_a_rows
            k_packed_off = k_base / arith.index(PACK_FACTOR_A)
            return _make_tdm_desc(
                global_ptr=arg_a,
                lds_memref=memref,
                global_offset=(blk_m + arith.index(row_start), k_packed_off),
                tensor_shape=(tile_m, packed_tile_k_a),
                strides=(lda_packed, 1),
                tile_shape=(ab_split_a_rows, packed_tile_k_a),
                elem_bytes=1,
                pad_interval=packed_tile_k_a,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=1,
                workgroup_mask=a_mcast_mask,
                lds_byte_offset=arith.index(row_start * lds_a_stride_bytes),
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=i32_m,
            )

        def make_desc_b_half(memref, k_base, n_half: int):
            group_start = n_half * ab_split_b_groups
            k_packed_off = k_base / arith.index(PACK_FACTOR_B)
            return _make_tdm_desc(
                global_ptr=arg_b,
                lds_memref=memref,
                global_offset=(blk_n / arith.index(16) + arith.index(group_start), k_packed_off * arith.index(16)),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(ab_split_b_groups, packed_tile_k_b * 16),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=1,
                workgroup_mask=b_mcast_mask,
                lds_byte_offset=arith.index(group_start * packed_tile_k_b * 16),
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        def make_desc_as(memref, k_base):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_m / arith.index(wmma_m_rep)
            inner_off = k_scale_off * arith.index(wmma_m_rep)
            return _make_tdm_desc(
                global_ptr=arg_a_scale,
                lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(WMMA_M * m_warp, interleaved_scale_cols_a),
                strides=(wmma_m_rep * K_scale, 1),
                tile_shape=(WMMA_M * m_warp, interleaved_scale_cols_a),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        def make_desc_bs(memref, k_base):
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_n / arith.index(b_scale_load_rep)
            inner_off = k_scale_off * arith.index(b_scale_load_rep)
            return _make_tdm_desc(
                global_ptr=arg_b_scale,
                lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(WMMA_M * n_warp, interleaved_scale_cols_b),
                strides=(b_scale_load_rep * K_scale, 1),
                tile_shape=(WMMA_M * n_warp, interleaved_scale_cols_b),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        # ── K-split (pf/remain) descriptor variants ──────────────────────────
        # seg 0 = pf [0, boundary), seg 1 = remain [boundary, end), splitting the
        # inner (K / scale-col) dimension. Each writes a sub-range of every row /
        # n-group and pads to the full row stride so the LDS layout is identical
        # to the single-TDM load.
        def make_desc_a_seg(memref, k_base, seg):
            col_off, col_len = _cols_a[seg]
            k_packed_off = k_base / arith.index(PACK_FACTOR_A) + arith.index(col_off)
            return _make_tdm_desc(
                global_ptr=arg_a,
                lds_memref=memref,
                global_offset=(blk_m, k_packed_off),
                tensor_shape=(tile_m, col_len),
                strides=(lda_packed, 1),
                tile_shape=(tile_m, col_len),
                elem_bytes=1,
                pad_interval=col_len,
                pad_amount=lds_a_stride_bytes - col_len,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                lds_byte_offset=arith.index(col_off),
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=i32_m,
            )

        def make_desc_b_seg(memref, k_base, seg):
            if const_expr(_b_ksmajor):
                # ks-major tile-contiguous: K-split = contiguous 256B-block sub-range.
                blk_off, blk_len = _cols_b_blk[seg]
                _kt = K_packed_b // packed_tile_k_b
                bn = blk_n / arith.index(tile_n)
                bk = k_base / arith.index(tile_k)
                tile_blk0 = (bn * arith.index(_kt) + bk) * arith.index(_b_num_blocks) + arith.index(blk_off)
                return _make_tdm_desc(
                    global_ptr=arg_b,
                    lds_memref=memref,
                    global_offset=(tile_blk0, arith.index(0)),
                    tensor_shape=(N * K_packed_b // B_TILE_BYTES, B_TILE_BYTES),
                    strides=(B_TILE_BYTES, 1),
                    tile_shape=(blk_len, B_TILE_BYTES),
                    elem_bytes=1,
                    pad_interval=B_TILE_BYTES if _b_pad else 0,
                    pad_amount=_b_pad,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=b_mcast_mask,
                    lds_byte_offset=arith.index(blk_off * _b_blk_stride),
                    atomic_barrier_enable=atomic_barrier_enable,
                    early_timeout=True,
                )
            col_off, col_len = _cols_b[seg]
            k_packed_off = k_base / arith.index(PACK_FACTOR_B) + arith.index(col_off)
            return _make_tdm_desc(
                global_ptr=arg_b,
                lds_memref=memref,
                global_offset=(blk_n / arith.index(16), k_packed_off * arith.index(16)),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(tile_n // 16, col_len * 16),
                elem_bytes=1,
                pad_interval=col_len * 16,
                pad_amount=_b_ngroup_stride - col_len * 16,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                lds_byte_offset=arith.index(col_off * 16),
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        def make_desc_as_seg(memref, k_base, seg):
            col_off, col_len = _cols_as[seg]
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_m / arith.index(wmma_m_rep)
            inner_off = k_scale_off * arith.index(wmma_m_rep) + arith.index(col_off)
            return _make_tdm_desc(
                global_ptr=arg_a_scale,
                lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(WMMA_M * m_warp, col_len),
                strides=(wmma_m_rep * K_scale, 1),
                tile_shape=(WMMA_M * m_warp, col_len),
                elem_bytes=1,
                pad_interval=col_len,
                pad_amount=interleaved_scale_cols_a - col_len,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                lds_byte_offset=arith.index(col_off),
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        def make_desc_bs_seg(memref, k_base, seg):
            col_off, col_len = _cols_bs[seg]
            k_scale_off = k_base / arith.index(SCALE_BLOCK)
            outer_off = blk_n / arith.index(b_scale_load_rep)
            inner_off = k_scale_off * arith.index(b_scale_load_rep) + arith.index(col_off)
            return _make_tdm_desc(
                global_ptr=arg_b_scale,
                lds_memref=memref,
                global_offset=(outer_off, inner_off),
                tensor_shape=(WMMA_M * n_warp, col_len),
                strides=(b_scale_load_rep * K_scale, 1),
                tile_shape=(WMMA_M * n_warp, col_len),
                elem_bytes=1,
                pad_interval=col_len,
                pad_amount=interleaved_scale_cols_b - col_len,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                lds_byte_offset=arith.index(col_off),
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        if const_expr(wave_specialized_tdm):
            tdm_wave_id = rocdl.wave_id()
            tdm_wave_is_a = tdm_wave_id == fx.Int32(0)
            tdm_wave_is_b = tdm_wave_id == fx.Int32(1)
            tdm_wave_is_as = tdm_wave_id == fx.Int32(2)

            def _select_wave_tdm_value(a_value, b_value, as_value, bs_value):
                result = arith.select(tdm_wave_is_as, as_value, bs_value)
                result = arith.select(tdm_wave_is_b, b_value, result)
                return arith.select(tdm_wave_is_a, a_value, result)

        elem_ty_lds = T.f16

        def _precompute_a_lane_bases(lds_ptr):
            """Single lane-dependent A base; per-wm and per-ks offsets go into ds_load imm."""
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            k_half_off = lane_kgrp * arith.index(16)
            base = row_base + k_half_off
            return lds_ptr, base

        def load_a_frag(lds_buffer, a_lane_base, wm, ks):
            """Load one A-fragment from LDS.

            wm and ks are folded into ds_load immediates — no VALU offset arithmetic.
            FP4: vec<8xi32> via 2 × ds_load_b128.
            FP8/A8W4: vec<16xi32> via 4 × ds_load_b128.
            """
            imm = wm * WMMA_M * lds_a_stride_bytes + ks * (WMMA_K // PACK_FACTOR_A)
            v0 = fx.Vector(lds_load_b128_raw(lds_buffer, a_lane_base, imm))
            if const_expr(is_fp4):
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, a_lane_base, imm + 32))
                return v0.shuffle(v1, list(range(8)))
            else:
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, a_lane_base, imm + 32))
                v2 = fx.Vector(lds_load_b128_raw(lds_buffer, a_lane_base, imm + 64))
                v3 = fx.Vector(lds_load_b128_raw(lds_buffer, a_lane_base, imm + 96))
                v01 = v0.shuffle(v1, list(range(8)))
                v23 = v2.shuffle(v3, list(range(8)))
                return v01.shuffle(v23, list(range(16)))

        def _precompute_b_lane_bases(lds_ptr):
            """Precompute per-wn B fragment lane base addresses (byte offsets).

            FP4: 2 bases per wn (32-col WMMA = 2 N-groups of 16).
            FP8: 1 base per wn (16-col WMMA = 1 N-group).
            A8W4: 1 base per wn (16-col WMMA, FP4 packed weight).

            K-dimension interleaving for FP8/A8W4:
              kgrp0 and kgrp1 read alternating 16x16 tiles (stride = 2 tiles).
              kgrp offset = 1 tile = 256 bytes.
            """
            # Single lane-dependent B base; per-wn and per-ks offsets go into ds_load imm.
            _n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
            row_off = lane16 * arith.index(16)
            if const_expr(_b_ksmajor):
                # ks-major: n-group step = _b_blk_stride, k-block step = _b_kstep.
                k_tile_off = lane_kgrp * arith.index(_b_kstep)
                base = _n_group_base * arith.index(_b_blk_stride) + row_off + k_tile_off
            else:
                k_tile_off = lane_kgrp * arith.index(_b_blk_stride)  # padded 16×16-tile step
                base = _n_group_base * arith.index(_b_ngroup_stride) + row_off + k_tile_off
            return lds_ptr, base

        def load_b_frag(lds_buffer, b_lane_base, wn, ks):
            """Load one B-fragment from preshuffled LDS.

            wn and ks are folded into ds_load immediates — no VALU offset arithmetic.
            FP4: 32x128 → vec<16xi32> from 2 N-groups.
            FP8: 16x128 → vec<16xi32> from 1 N-group.
            A8W4: 16x128 FP4 → vec<8xi32> from 1 N-group.
            """
            if const_expr(is_fp4):
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16
                imm0 = wn * 2 * _b_ngroup_stride + ks * _num_tiles * 256
                imm1 = (wn * 2 + 1) * _b_ngroup_stride + ks * _num_tiles * 256
                v0 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm0))
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm0 + 512))
                v2 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm1))
                v3 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm1 + 512))
                v01 = v0.shuffle(v1, list(range(8)))
                v23 = v2.shuffle(v3, list(range(8)))
                return v01.shuffle(v23, list(range(16)))
            elif const_expr(is_a8w4):
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16
                if const_expr(_b_ksmajor):
                    # ks-major: n-group step = _b_blk_stride, k-block step = _b_kstep.
                    imm = wn * _b_blk_stride + ks * _num_tiles * _b_kstep
                    v0 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm))
                    v1 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm + 2 * _b_kstep))
                else:
                    imm = wn * _b_ngroup_stride + ks * _num_tiles * _b_blk_stride
                    v0 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm))
                    v1 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm + 2 * _b_blk_stride))
                return v0.shuffle(v1, list(range(8)))
            else:
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16
                imm = wn * _b_ngroup_stride + ks * _num_tiles * 256
                v0 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm))
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm + 512))
                v2 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm + 1024))
                v3 = fx.Vector(lds_load_b128_raw(lds_buffer, b_lane_base, imm + 1536))
                v01 = v0.shuffle(v1, list(range(8)))
                v23 = v2.shuffle(v3, list(range(8)))
                return v01.shuffle(v23, list(range(16)))

        def _precompute_scale_lane_bases(lds_ptr, warp_base, reps, interleaved_cols):
            """Single lane-dependent scale base; ks/rep offsets go into ds_load imm."""
            warp_lds_row = warp_base / arith.index(reps) + lane16
            base = warp_lds_row * arith.index(interleaved_cols)
            if const_expr(is_fp4 or is_a8w4):
                base = base + lane_kgrp * arith.index(SCALES_PER_WMMA)
            else:
                if const_expr(use_scale_opsel):
                    base = base + lane_kgrp * arith.index(SCALES_PER_WMMA)
            return lds_ptr, base

        def load_scale_b128(lds_buffer, scale_base, reps, ks=0):
            """Load all wmma_rep scales via ds_load_b128(s) for K-subtile *ks*.

            ks offset is a pure Python integer folded into ds_load imm — no VALU.
            """
            ks_byte_off = ks * reps * SCALES_PER_WMMA
            num_loads = (reps + 3) // 4
            vecs = []
            for ld in range_constexpr(num_loads):
                vecs.append(fx.Vector(lds_load_b128_raw(lds_buffer, scale_base, ks_byte_off + ld * 16)))
            results = []
            for i in range_constexpr(reps):
                results.append(vecs[i // 4][i % 4])
            return results

        def load_scale_slice_b128(lds_buffer, scale_base, full_reps, rep_start, rep_count, ks=0):
            """Load a contiguous slice of packed scale VGPRs for one K-subtile.

            All offsets are pure Python integers folded into ds_load imm — no VALU.
            """
            ks_byte_off = (ks * full_reps + rep_start) * SCALES_PER_WMMA
            num_loads = (rep_count + 3) // 4
            vecs = []
            for ld in range_constexpr(num_loads):
                vecs.append(fx.Vector(lds_load_b128_raw(lds_buffer, scale_base, ks_byte_off + ld * 16)))
            results = []
            for i in range_constexpr(rep_count):
                results.append(vecs[i // 4][i % 4])
            return results

        def _scales_for_emit(as_buf, as_base, bs_buf, bs_base, ks):
            """Load both scale tensors and apply op_sel downsampling per format.

            FP4 BScale has no op_sel (scaleAType=0 fixed); only AScale halves.
            FP8/A8W4 16x16 supports op_sel on both.
            """
            if const_expr(is_ptpc):
                return None, None
            a_all = load_scale_b128(as_buf, as_base, wmma_m_rep, ks)
            b_all = load_scale_b128(bs_buf, bs_base, b_scale_load_rep, ks)
            if const_expr(use_scale_opsel):
                a = a_all[::2]
                b = b_all if const_expr(is_fp4) else b_all[::2]
            else:
                a, b = a_all, b_all
            return a, b

        def _load_b_and_scales(b_buf, b_base, bs_buf, bs_base, as_buf, as_base, ks):
            b_frags = [load_b_frag(b_buf, b_base, wn, ks) for wn in range_constexpr(wmma_n_rep)]
            a_scales, b_scales = _scales_for_emit(as_buf, as_base, bs_buf, bs_base, ks)
            return b_frags, b_scales, a_scales

        def _load_a_and_scales(a_buf, a_base, as_buf, as_base, bs_buf, bs_base, ks):
            a_frags = [load_a_frag(a_buf, a_base, wm, ks) for wm in range_constexpr(wmma_m_rep)]
            a_scales, b_scales = _scales_for_emit(as_buf, as_base, bs_buf, bs_base, ks)
            return a_frags, a_scales, b_scales

        def _emit_wmma(accs, wm, wn, a_frag, b_frag, a_scales, b_scales):
            """Emit one WMMA instruction (format-specific)."""
            idx = wm * wmma_n_rep + wn
            if const_expr(is_ptpc):
                if const_expr(is_a8w4):
                    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_frag,
                        a_frag,
                        accs[idx],
                        0x7F7F7F7F,
                        0x7F7F7F7F,
                        fmtA=4,
                        fmtB=0,
                    )
                else:
                    # PTPC-FP8 needs no per-K scaling. We emit the scaled f8f6f4 op
                    # with an identity E8M0 scale (0x7F = 2^0 = 1.0) for toolchain
                    # compatibility; it is numerically equivalent to the dedicated
                    # no-scale op. Future: switch to the equivalent no-scale wmma:
                    #   accs[idx] = rocdl.wmma_f32_16x16x128_fp8_fp8(T.vec(8, T.f32), b_frag, a_frag, accs[idx])
                    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_frag,
                        a_frag,
                        accs[idx],
                        0x7F7F7F7F,
                        0x7F7F7F7F,
                        fmtA=0,
                        fmtB=0,
                    )
                return
            if const_expr(use_scale_opsel):
                a_scale_idx = wm // 2
                a_opsel = wm % 2
            else:
                a_scale_idx = wm
                a_opsel = 0

            if const_expr(is_fp4):
                # 32x16 WMMA with A/B swap: SRC0=B, SRC1=A
                accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                    T.vec(16, T.f32),
                    b_frag,
                    a_frag,
                    accs[idx],
                    b_scales[wn * 2],
                    a_scales[a_scale_idx],
                    scaleAType=0,
                    scaleBType=a_opsel,
                )
            else:
                # 16x16x128 WMMA: A8W4 (fmtA=FP4) or FP8 (fmtA=FP8)
                if const_expr(use_scale_opsel):
                    b_scale_idx = wn // 2
                    b_opsel = wn % 2
                else:
                    b_scale_idx = wn
                    b_opsel = 0
                accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                    T.vec(8, T.f32),
                    b_frag,
                    a_frag,
                    accs[idx],
                    b_scales[b_scale_idx],
                    a_scales[a_scale_idx],
                    fmtA=4 if is_a8w4 else 0,
                    fmtB=0,
                    scaleAType=b_opsel,
                    scaleBType=a_opsel,
                )

        def _a_streaming_compute(
            accs,
            a_buf,
            a_base,
            b_frags,
            b_scales,
            a_scales,
            ks,
            emit_filler=None,
            next_bs_info=None,
            mid_compute_callback=None,
            pf_a_frags=None,
            skip_wait=False,
        ):
            """Half-based A-streaming with zigzag wn ordering.

            When *next_bs_info* is provided, the next K-subtile's B+scale
            loads are issued BEFORE the s_wait_dscnt so they overlap with
            the current WMMA execution (partial drain pattern).

            When *pf_a_frags* is provided the A ds_loads for this k-step are
            skipped — the pre-fetched VGPR values from the previous tile's
            post-fetch are used directly, so s_wait_dscnt(0) sees fewer
            (or zero) pending loads.
            """
            next_result = None
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm

            def _emit_rows(start_wm, a_frags):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    is_last = wm == wmma_m_rep - 1
                    if const_expr(is_last and emit_filler is not None):
                        rocdl.sched_barrier(0)
                        emit_filler()
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        _emit_wmma(accs, wm, wn, a_frags[frag_i], b_frags[wn], a_scales, b_scales)

            if const_expr(pf_a_frags is not None):
                a_frags_front = pf_a_frags[:_front_wm]
            else:
                a_frags_front = [load_a_frag(a_buf, a_base, wm, ks) for wm in range_constexpr(_front_wm)]

            _use_partial_drain = next_bs_info is not None and _front_wm * wmma_n_rep >= 4

            if const_expr(_use_partial_drain):
                nb_buf, nb_base, nbs_buf, nbs_base, nas_buf, nas_base, n_ks = next_bs_info
                next_result = _load_b_and_scales(nb_buf, nb_base, nbs_buf, nbs_base, nas_buf, nas_base, n_ks)
                rocdl.s_wait_dscnt(_bs_ds_loads)
            elif const_expr(skip_wait):
                pass  # operands are loop-carried VGPRs; no ds_loads pending
            else:
                rocdl.s_wait_dscnt(0)

            _emit_rows(0, a_frags_front)

            if const_expr(mid_compute_callback is not None):
                rocdl.sched_barrier(0)
                mid_compute_callback()

            if const_expr(_back_wm > 0):
                if const_expr(pf_a_frags is not None):
                    a_frags_back = pf_a_frags[_front_wm : _front_wm + _back_wm]
                else:
                    a_frags_back = [load_a_frag(a_buf, a_base, _front_wm + h, ks) for h in range_constexpr(_back_wm)]
                if const_expr(skip_wait):
                    pass
                else:
                    _back_drain = _bs_ds_loads if _use_partial_drain else 0
                    rocdl.s_wait_dscnt(_back_drain)
                _emit_rows(_front_wm, a_frags_back)

            if const_expr(_use_partial_drain):
                return accs, next_result
            if const_expr(next_bs_info is not None):
                nb_buf, nb_base, nbs_buf, nbs_base, nas_buf, nas_base, n_ks = next_bs_info
                next_result = _load_b_and_scales(nb_buf, nb_base, nbs_buf, nbs_base, nas_buf, nas_base, n_ks)
                return accs, next_result
            return accs

        def _b_streaming_compute(
            accs,
            b_buf,
            b_base,
            a_frags,
            a_scales,
            b_scales,
            ks,
            emit_filler=None,
            next_info=None,
            mid_compute_callback=None,
        ):
            """B-streaming counterpart to _a_streaming_compute (A held, B streamed)."""
            next_result = None
            _front_wn = (wmma_n_rep + 1) // 2
            _back_wn = wmma_n_rep - _front_wn

            def _emit_cols(start_wn, b_frags_chunk):
                for frag_i in range_constexpr(len(b_frags_chunk)):
                    wn = start_wn + frag_i
                    if const_expr(wn == wmma_n_rep - 1 and emit_filler is not None):
                        rocdl.sched_barrier(0)
                        emit_filler()
                    for wm_raw in range_constexpr(wmma_m_rep):
                        wm = (wmma_m_rep - 1 - wm_raw) if (wn % 2 == 1) else wm_raw
                        _emit_wmma(accs, wm, wn, a_frags[wm], b_frags_chunk[frag_i], a_scales, b_scales)

            b_frags_front = [load_b_frag(b_buf, b_base, wn, ks) for wn in range_constexpr(_front_wn)]
            _use_partial_drain = next_info is not None and _front_wn * wmma_m_rep >= 4

            if const_expr(_use_partial_drain):
                next_result = _load_a_and_scales(*next_info)
                rocdl.s_wait_dscnt(_as_ds_loads)
            else:
                rocdl.s_wait_dscnt(0)

            _emit_cols(0, b_frags_front)

            if const_expr(mid_compute_callback is not None):
                rocdl.sched_barrier(0)
                mid_compute_callback()

            if const_expr(_back_wn > 0):
                b_frags_back = [load_b_frag(b_buf, b_base, _front_wn + h, ks) for h in range_constexpr(_back_wn)]
                rocdl.s_wait_dscnt(_as_ds_loads if _use_partial_drain else 0)
                _emit_cols(_front_wn, b_frags_back)

            if const_expr(_use_partial_drain):
                return accs, next_result
            if const_expr(next_info is not None):
                return accs, _load_a_and_scales(*next_info)
            return accs

        def _assemble_pf_ks(raw, ks):
            """Assemble one ks-step's (a_frags, b_frags, b_scales, a_scales) from raw
            b128 vectors indexed by ABSOLUTE table index (raw[t] == table entry t).
            Used by the sub-depth raw-carry pipeline; shares _pf_plans with the
            full-depth assembled-carry path."""
            _as_start, _as_n, _a_groups, _b_groups, _bs_groups = _pf_plans[ks]
            if const_expr(not is_ptpc):
                _as_vecs = raw[_as_start : _as_start + _as_n]
                _as_all = [_as_vecs[r // 4][r % 4] for r in range_constexpr(wmma_m_rep)]
                _as_ks = _as_all[::2] if const_expr(use_scale_opsel) else _as_all
            else:
                _as_ks = None
            _a_ks = []
            for _a_start, _a_n in _a_groups:
                _r = raw[_a_start : _a_start + _a_n]
                if const_expr(is_fp4):
                    _a_ks.append(_r[0].shuffle(_r[1], list(range(8))))
                else:
                    _v01 = _r[0].shuffle(_r[1], list(range(8)))
                    _v23 = _r[2].shuffle(_r[3], list(range(8)))
                    _a_ks.append(_v01.shuffle(_v23, list(range(16))))
            _b_ks = []
            for _b_start, _b_n in _b_groups:
                _r = raw[_b_start : _b_start + _b_n]
                if const_expr(is_a8w4):
                    _b_ks.append(_r[0].shuffle(_r[1], list(range(8))))
                else:
                    _v01 = _r[0].shuffle(_r[1], list(range(8)))
                    _v23 = _r[2].shuffle(_r[3], list(range(8)))
                    _b_ks.append(_v01.shuffle(_v23, list(range(16))))
            _bs_ks = []
            for _bs_start, _bs_n, _gs in _bs_groups:
                _bs_vecs = raw[_bs_start : _bs_start + _bs_n]
                for _i in range(_gs):
                    _bs_ks.append(_bs_vecs[_i // 4][_i % 4])
            if const_expr(use_scale_opsel and not is_fp4):
                _bs_ks = _bs_ks[::2]
            return _a_ks, _b_ks, _bs_ks, _as_ks

        # Per-WMMA LAZY assembly for the sub-ks carry pipeline (_pf_D < _pf_wpks).
        # _assemble_pf_ks assembles a WHOLE ks-step's frags at once, which needs all of
        # that ks-step's raw b128 loads present when called. A sub-ks carry has fewer
        # than one ks-step's WMMAs in flight, so ks-0 is incomplete at the ks boundary
        # and the bulk assemble indexes past the end. Instead, assemble each
        # A[wm,ks]/B[wn,ks]/scale[ks] JUST BEFORE its WMMA is emitted: the i->i+D
        # ds_load pipeline guarantees WMMA i's raw loads (and any earlier WMMA's, e.g.
        # B[wn] owned by wm=0) are in `raw` by iteration i. The `cache` dict (passed in,
        # one per tile) keeps each frag assembled exactly once (A reused across wn,
        # scales across the ks-step), so the shuffle count matches the bulk path. `raw`
        # is the growing cur_all list, indexed by ABSOLUTE table index (raw[t]==entry t).
        # NB: caches are explicit args, not closure cells — FlyDSL's AST rewriter treats
        # an enclosing-scope `d[k]=v` as a local rebind (UnboundLocalError), so a captured
        # mutable dict cannot be used here.
        def _pf_lazy_a(raw, cache, ks, wm):
            _k = (ks, wm)
            if _k not in cache:
                _s, _n = _pf_plans[ks][2][wm]   # a_groups[wm]
                _r = raw[_s : _s + _n]
                if const_expr(is_fp4):
                    cache[_k] = _r[0].shuffle(_r[1], list(range(8)))
                else:
                    _v01 = _r[0].shuffle(_r[1], list(range(8)))
                    _v23 = _r[2].shuffle(_r[3], list(range(8)))
                    cache[_k] = _v01.shuffle(_v23, list(range(16)))
            return cache[_k]

        def _pf_lazy_b(raw, cache, ks, wn):
            _k = (ks, wn)
            if _k not in cache:
                _s, _n = _pf_plans[ks][3][wn]   # b_groups[wn]
                _r = raw[_s : _s + _n]
                if const_expr(is_a8w4):
                    cache[_k] = _r[0].shuffle(_r[1], list(range(8)))
                else:
                    _v01 = _r[0].shuffle(_r[1], list(range(8)))
                    _v23 = _r[2].shuffle(_r[3], list(range(8)))
                    cache[_k] = _v01.shuffle(_v23, list(range(16)))
            return cache[_k]

        def _pf_lazy_scales(raw, cache, ks, wm, wn):
            """Lazy scale assembly for the sub-ks carry path. Returns (a_scales,
            b_scales) post-opsel lists with AT LEAST the entries _emit_wmma reads for
            (wm,wn) materialized. a_scales (AS) is owned by the ks-step's first WMMA so
            it is always present -> assembled in full once. b_scales (BS) entries are
            owned by wm=0's successive wn WMMAs, so for a shallow carry the later ones
            arrive only as the loop advances -> materialize each on demand. cache[ks]
            persists the (a_scales, partial b_scales) so each entry is built once."""
            if ks not in cache:
                if const_expr(is_ptpc):
                    _as_ks = None
                else:
                    _as_pre = [raw[_ti][_sub] for (_ti, _sub) in _pf_as_src[ks]]
                    _as_ks = _as_pre[::2] if const_expr(use_scale_opsel) else _as_pre
                _bs_pre_len = len(_pf_bs_src[ks])
                if const_expr(use_scale_opsel and not is_fp4):
                    _bs_len = (_bs_pre_len + 1) // 2
                else:
                    _bs_len = _bs_pre_len
                cache[ks] = (_as_ks, [None] * _bs_len)
            _as_ks, _bs_list = cache[ks]
            if const_expr(not is_ptpc):
                # b_scale index _emit_wmma will read for this wn (matches its logic).
                if const_expr(is_fp4):
                    _bidx = wn * 2
                elif const_expr(use_scale_opsel):
                    _bidx = wn // 2
                else:
                    _bidx = wn
                if _bs_list[_bidx] is None:
                    _pre = _bidx * 2 if const_expr(use_scale_opsel and not is_fp4) else _bidx
                    _ti, _sub = _pf_bs_src[ks][_pre]
                    _bs_list[_bidx] = raw[_ti][_sub]
            return _as_ks, _bs_list

        # ── Compute on one LDS buffer ──
        def compute_tile(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            pf_all_ks=None,
            next_lds_a=None,
            next_lds_b=None,
            next_lds_bs=None,
            next_lds_as=None,
            tdm_cb_pf=None,
            tdm_cb_remain=None,
        ):
            """Compute one LDS buffer's WMMA tile.

            pf_all_ks: list[(a_frags, b_frags, b_scales, a_scales)] per k-step,
              loop-carried VGPRs from the PREVIOUS tile's interleaved prefetch.
              Current-tile ds_loads are skipped; all s_wait_dscnt(0) = instant.

            next_lds_{a,b,bs,as}: when provided, issue NEXT tile's ds_loads for each
              k-step interleaved with the corresponding current-tile WMMA group.
              The sched_dsrd/sched_mfma hints in hot_loop_scheduler guide LLVM to
              schedule the ds_loads into the WMMA latency slots so they complete by
              the time the next tile starts. Returns (accs, nxt_pf_all_ks).
            """
            current_accs = list(accs_in)
            a_buf, a_base = _precompute_a_lane_bases(lds_a)
            b_buf, b_base = _precompute_b_lane_bases(lds_b)
            as_buf, as_base = _precompute_scale_lane_bases(lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
            bs_buf, bs_base = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b
            )

            _has_pf = pf_all_ks is not None
            _has_next = next_lds_b is not None

            if const_expr(_has_next):
                nxt_a_buf, nxt_a_base = _precompute_a_lane_bases(next_lds_a)
                nxt_b_buf, nxt_b_base = _precompute_b_lane_bases(next_lds_b)
                nxt_as_buf, nxt_as_base = _precompute_scale_lane_bases(
                    next_lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a
                )
                nxt_bs_buf, nxt_bs_base = _precompute_scale_lane_bases(
                    next_lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b
                )

            if const_expr(_has_pf and _has_next):
                if const_expr(_pf_full_depth):
                    # ── D=A: committed baseline path (high-level loaders) ───────────
                    # Inline WMMAs at _emit_wmma granularity.  Next-tile ds_loads
                    # are ordered per WMMA:  as → bs[wn] → a[wm] → b[wn] → WMMA.
                    _pf_front_wm = (wmma_m_rep + 1) // 2
                    if const_expr(use_scale_opsel and not is_fp4):
                        _bs_stride = 2
                    else:
                        _bs_stride = 1
                    nxt_pf = []
                    for ks in range_constexpr(k_wmma_steps):
                        pf_a_ks, pf_b_ks, pf_bs_ks, pf_as_ks = pf_all_ks[ks]
                        _mid_cb = mid_compute_callback if ks == 0 else None
                        _filler = emit_filler if ks == k_wmma_steps - 1 else None
                        if const_expr(not is_ptpc):
                            _nxt_as_all = load_scale_b128(nxt_as_buf, nxt_as_base, wmma_m_rep, ks)
                            if const_expr(use_scale_opsel):
                                nxt_as_ks = _nxt_as_all[::2]
                            else:
                                nxt_as_ks = _nxt_as_all
                        else:
                            nxt_as_ks = None
                        nxt_a_ks = []
                        nxt_b_ks = []
                        nxt_bs_ks = []
                        for wm in range_constexpr(wmma_m_rep):
                            if const_expr(wm == wmma_m_rep - 1 and _filler is not None):
                                rocdl.sched_barrier(0)
                                _filler()
                            for wn_raw in range_constexpr(wmma_n_rep):
                                wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                                if const_expr(wm == 0):
                                    if const_expr(not is_ptpc):
                                        _bs_group = wn // _bs_stride
                                        _bs_in_group = wn % _bs_stride
                                        if const_expr(_bs_in_group == 0):
                                            _bs_slice = load_scale_slice_b128(
                                                nxt_bs_buf, nxt_bs_base,
                                                b_scale_load_rep, _bs_group * _bs_stride,
                                                min(_bs_stride, b_scale_load_rep - _bs_group * _bs_stride),
                                                ks,
                                            )
                                            nxt_bs_ks.extend(_bs_slice)
                                    if const_expr(wn_raw == 0):
                                        nxt_a_ks.append(load_a_frag(nxt_a_buf, nxt_a_base, wm, ks))
                                    nxt_b_ks.append(load_b_frag(nxt_b_buf, nxt_b_base, wn, ks))
                                else:
                                    if const_expr(wn_raw == 0):
                                        nxt_a_ks.append(load_a_frag(nxt_a_buf, nxt_a_base, wm, ks))
                                rocdl.sched_barrier(0)
                                _emit_wmma(
                                    current_accs, wm, wn,
                                    pf_a_ks[wm], pf_b_ks[wn], pf_as_ks, pf_bs_ks,
                                )
                                rocdl.sched_barrier(0)
                            if const_expr(wm == _pf_front_wm - 1 and _mid_cb is not None):
                                _mid_cb()
                        nxt_pf.append((nxt_a_ks, nxt_b_ks, nxt_bs_ks, nxt_as_ks))
                    return current_accs, nxt_pf

                # ── D<A: table-driven raw-carry pipeline ────────────────────────────
                # WMMA i consumes position i (carry if i<D else cur_all) and issues the
                # ds_load for position j=i+D: j<A → "remain" into cur_all (this tile);
                # j>=A → next tile's carry nxt_raw (position j-A). cur_all[t]==table[t].
                _nxt_buf_pairs = [
                    (nxt_a_buf, nxt_a_base), (nxt_b_buf, nxt_b_base),
                    (nxt_as_buf, nxt_as_base), (nxt_bs_buf, nxt_bs_base),
                ]
                _cur_buf_pairs = [(a_buf, a_base), (b_buf, b_base), (as_buf, as_base), (bs_buf, bs_base)]
                cur_all = list(pf_all_ks)
                nxt_raw = []
                # _pf_D >= _pf_wpks: carry covers whole ks-steps -> bulk assemble each
                # ks-step at its boundary (validated path, unchanged). _pf_D < _pf_wpks
                # (sub-ks carry): a ks-step is never fully present at its boundary, so
                # assemble per-WMMA, lazily, from the growing cur_all.
                if const_expr(_pf_D >= _pf_wpks):
                    _cur_ks = _assemble_pf_ks(cur_all, 0)  # ks 0 (positions 0.._pf_wpks-1)
                else:
                    _cur_ks = None
                    _lz_ac, _lz_bc, _lz_scc = {}, {}, {}   # per-tile lazy-assembly caches
                # cb_pf fires once the pf SEGMENT is fully consumed. Legacy backfills
                # an already-dead buffer, so the carry depth _pf_D suffices; full-
                # prefetch backfills the live buffer, so it must wait until the whole
                # pf seg (ks0.._pf_tdm_Dk-1) is read -> _pf_seg_wmmas (>= _pf_D).
                _cb_pf_at = _pf_seg_wmmas if _full_prefetch else _pf_D
                for i in range_constexpr(_A_wmma):
                    # ── TDM boundary 1 (pf seg consumed): wait this tile's remain TDM,
                    # issue next tile's pf TDM. See WMMA-driven prefetch plan.
                    if const_expr(i == _cb_pf_at and tdm_cb_pf is not None):
                        tdm_cb_pf()
                    ks, wm, wn = _pf_pos[i]
                    _j = i + _pf_D
                    if const_expr(_j < _A_wmma):
                        _ls, _lc = _pf_wmma_plan_flat[_j]
                        for _t, _m in _pf_load_table[_ls : _ls + _lc]:
                            _cb, _cba = _cur_buf_pairs[_t]
                            cur_all.append(fx.Vector(lds_load_b128_raw(_cb, _cba, _m)))
                    else:
                        _ls, _lc = _pf_wmma_plan_flat[_j - _A_wmma]
                        for _t, _m in _pf_load_table[_ls : _ls + _lc]:
                            _nb, _nba = _nxt_buf_pairs[_t]
                            nxt_raw.append(fx.Vector(lds_load_b128_raw(_nb, _nba, _m)))
                    if const_expr(_pf_D >= _pf_wpks and i % _pf_wpks == 0 and i > 0):
                        _cur_ks = _assemble_pf_ks(cur_all, ks)
                    rocdl.sched_barrier(0)
                    if const_expr(os.environ.get("PF_FORCE_WAIT", "0") == "1"):
                        rocdl.s_wait_dscnt(0)
                    # Scales come from the raw carry too (prefetched into VGPR like A/B):
                    # _cur_ks[3]=a_scales, _cur_ks[2]=b_scales. The _pf_load_table already
                    # carries AS/BS entries (loaded into cur_all, assembled by
                    # _assemble_pf_ks) — they were just DCE'd while we read fresh. Using
                    # them eliminates the per-WMMA fresh scale ds_load + blocking
                    # s_wait_dscnt. (Correct now: scales load full/un-split into LDS via
                    # the same _precompute_scale_lane_bases addressing.)
                    if const_expr(_pf_D >= _pf_wpks):
                        _emit_wmma(current_accs, wm, wn, _cur_ks[0][wm], _cur_ks[1][wn], _cur_ks[3], _cur_ks[2])
                    else:
                        _as_e, _bs_e = _pf_lazy_scales(cur_all, _lz_scc, ks, wm, wn)
                        _emit_wmma(current_accs, wm, wn,
                                   _pf_lazy_a(cur_all, _lz_ac, ks, wm),
                                   _pf_lazy_b(cur_all, _lz_bc, ks, wn), _as_e, _bs_e)
                    rocdl.sched_barrier(0)
                # ── TDM boundary 2 (remain WMMAs done): wait next tile's pf TDM,
                # issue next tile's remain TDM.
                if const_expr(tdm_cb_remain is not None):
                    tdm_cb_remain()
                return current_accs, nxt_raw

            if const_expr(_has_pf):
                # ── Final-tile path: carry + remain loads, NO next tile ─────────────
                if const_expr(_pf_full_depth):
                    # D=A: original path, operands all from carry, no remain loads.
                    for ks in range_constexpr(k_wmma_steps):
                        pf_a_ks, pf_b_ks, pf_bs_ks, pf_as_ks = pf_all_ks[ks]
                        _mid_cb = mid_compute_callback if ks == 0 else None
                        _filler = emit_filler if ks == k_wmma_steps - 1 else None
                        current_accs = _a_streaming_compute(
                            current_accs, a_buf, a_base,
                            pf_b_ks, pf_bs_ks, pf_as_ks, ks,
                            emit_filler=_filler, mid_compute_callback=_mid_cb,
                            pf_a_frags=pf_a_ks, skip_wait=True,
                        )
                    return current_accs

                # D<A: same i→i+D pipeline, but j>=A loads are dropped (no next tile).
                _cur_buf_pairs = [(a_buf, a_base), (b_buf, b_base), (as_buf, as_base), (bs_buf, bs_base)]
                _p_mid = ((wmma_m_rep + 1) // 2) * wmma_n_rep - 1   # after ks0 front rows
                _p_fill = _A_wmma - wmma_n_rep                       # last row of last ks
                cur_all = list(pf_all_ks)
                # Same bulk-vs-lazy split as the with-next path (see note there).
                if const_expr(_pf_D >= _pf_wpks):
                    _cur_ks = _assemble_pf_ks(cur_all, 0)
                else:
                    _cur_ks = None
                    _lz_ac, _lz_bc, _lz_scc = {}, {}, {}   # per-tile lazy-assembly caches
                for i in range_constexpr(_A_wmma):
                    # TDM boundary 1 for the final tile: wait this tile's remain TDM
                    # before the remain WMMAs (the caller's cb issues NO next tile).
                    if const_expr(i == _pf_D and tdm_cb_pf is not None):
                        tdm_cb_pf()
                    ks, wm, wn = _pf_pos[i]
                    _j = i + _pf_D
                    if const_expr(_j < _A_wmma):
                        _ls, _lc = _pf_wmma_plan_flat[_j]
                        for _t, _m in _pf_load_table[_ls : _ls + _lc]:
                            _cb, _cba = _cur_buf_pairs[_t]
                            cur_all.append(fx.Vector(lds_load_b128_raw(_cb, _cba, _m)))
                    if const_expr(_pf_D >= _pf_wpks and i % _pf_wpks == 0 and i > 0):
                        _cur_ks = _assemble_pf_ks(cur_all, ks)
                    if const_expr(i == _p_fill and emit_filler is not None):
                        rocdl.sched_barrier(0)
                        emit_filler()
                    rocdl.sched_barrier(0)
                    if const_expr(os.environ.get("PF_FORCE_WAIT", "0") == "1"):
                        rocdl.s_wait_dscnt(0)
                    # Scales from the raw carry (prefetched, like A/B): _cur_ks[3]=a_scales,
                    # [2]=b_scales. See the with-next path note.
                    if const_expr(_pf_D >= _pf_wpks):
                        _emit_wmma(current_accs, wm, wn, _cur_ks[0][wm], _cur_ks[1][wn], _cur_ks[3], _cur_ks[2])
                    else:
                        _as_e, _bs_e = _pf_lazy_scales(cur_all, _lz_scc, ks, wm, wn)
                        _emit_wmma(current_accs, wm, wn,
                                   _pf_lazy_a(cur_all, _lz_ac, ks, wm),
                                   _pf_lazy_b(cur_all, _lz_bc, ks, wn), _as_e, _bs_e)
                    rocdl.sched_barrier(0)
                    if const_expr(i == _p_mid and mid_compute_callback is not None):
                        rocdl.sched_barrier(0)
                        mid_compute_callback()
                return current_accs

            # ── Original code path (no pre-fetch) ──
            if const_expr(k_wmma_steps == 1):
                b_frags, b_scales, a_scales = _load_b_and_scales(
                    b_buf, b_base, bs_buf, bs_base, as_buf, as_base, 0
                )
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_base, b_frags, b_scales, a_scales, 0,
                    emit_filler=emit_filler, mid_compute_callback=mid_compute_callback,
                )
            else:
                prev_b, prev_bs, prev_as = _load_b_and_scales(
                    b_buf, b_base, bs_buf, bs_base, as_buf, as_base, 0
                )
                for ks in range_constexpr(k_wmma_steps - 1):
                    _mid_cb = mid_compute_callback if ks == 0 else None
                    current_accs, (prev_b, prev_bs, prev_as) = _a_streaming_compute(
                        current_accs, a_buf, a_base, prev_b, prev_bs, prev_as, ks,
                        next_bs_info=(b_buf, b_base, bs_buf, bs_base, as_buf, as_base, ks + 1),
                        mid_compute_callback=_mid_cb,
                    )
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_base, prev_b, prev_bs, prev_as,
                    k_wmma_steps - 1, emit_filler=emit_filler,
                )
            return current_accs

        def compute_tile_fp4_bank_friendly(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
        ):
            current_accs = list(accs_in)
            a_buf, a_base = _precompute_a_lane_bases(lds_a)
            b_buf, b_base = _precompute_b_lane_bases(lds_b)
            as_buf, as_base = _precompute_scale_lane_bases(lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
            bs_buf, bs_base = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b
            )
            _b_half_scale_loads = (_bank_half_b_scale_rep + 3) // 4

            def _fp4_get_a_scale_and_opsel(a_scales_all, wm_idx):
                if const_expr(use_scale_opsel):
                    return a_scales_all[(wm_idx // 2) * 2], wm_idx % 2
                return a_scales_all[wm_idx], 0

            def _load_a_group(wm_base, wm_count, ks):
                return [load_a_frag(a_buf, a_base, wm_base + wm_local, ks) for wm_local in range_constexpr(wm_count)]

            def _load_b_half(wn_base, ks):
                return [
                    load_b_frag(b_buf, b_base, wn_base + wn_local, ks) for wn_local in range_constexpr(_bank_half_wn)
                ]

            def _load_b_half_bundle(wn_base, rep_start, ks):
                b_frags = _load_b_half(wn_base, ks)
                b_scales = load_scale_slice_b128(
                    bs_buf, bs_base, b_scale_load_rep, rep_start, _bank_half_b_scale_rep, ks
                )
                return b_frags, b_scales

            def _emit_group_rows(
                group_base, wm_base, a_frags, b_frags, a_scales, b_scales, row_start, row_count, emit_filler_now=False
            ):
                if const_expr(emit_filler_now and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()
                for row_offset in range_constexpr(row_count):
                    wm_local = row_start + row_offset
                    a_frag = a_frags[wm_local]
                    global_wm = wm_base + wm_local
                    a_scale, a_opsel = _fp4_get_a_scale_and_opsel(a_scales, global_wm)
                    row_base = group_base + wm_local * _bank_half_wn
                    for wn_local in range_constexpr(_bank_half_wn):
                        idx = row_base + wn_local
                        current_accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                            T.vec(16, T.f32),
                            b_frags[wn_local],
                            a_frag,
                            current_accs[idx],
                            b_scales[wn_local * 2],
                            a_scale,
                            scaleAType=0,
                            scaleBType=a_opsel,
                        )

            def _emit_group(group_base, wm_base, a_frags, b_frags, a_scales, b_scales, emit_filler_now=False):
                _emit_group_rows(
                    group_base,
                    wm_base,
                    a_frags,
                    b_frags,
                    a_scales,
                    b_scales,
                    0,
                    _bank_half_wm,
                    emit_filler_now=emit_filler_now,
                )

            b_left_frags, b_left_scales = _load_b_half_bundle(0, 0, 0)

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales_all = load_scale_b128(as_buf, as_base, wmma_m_rep, ks)

                a_top_frags = _load_a_group(0, _bank_half_wm, ks)
                a_bottom_frags = _load_a_group(_bank_half_wm, _bank_half_wm, ks)

                # Wait for bottom-A loads; top-A stays in flight during Q1.
                rocdl.s_wait_dscnt(_bank_half_wm * DS_LOADS_PER_A_FRAG)

                _emit_group(
                    0,
                    0,
                    a_top_frags,
                    b_left_frags,
                    a_scales_all,
                    b_left_scales,
                )

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                b_right_frags, b_right_scales = _load_b_half_bundle(_bank_half_wn, _bank_half_b_scale_rep, ks)

                # Hold only the next B half outstanding while the second
                # quadrant consumes the current left-half fragments.
                rocdl.s_wait_dscnt(_bank_half_wn * 4 + _b_half_scale_loads)

                _emit_group(
                    _bank_group_size,
                    _bank_half_wm,
                    a_bottom_frags,
                    b_left_frags,
                    a_scales_all,
                    b_left_scales,
                )

                if const_expr(not is_last_ks):
                    next_left_frags, next_left_scales = _load_b_half_bundle(0, 0, ks + 1)
                    # Older right-half loads must be ready before consuming
                    # them, while the next ks left-half preload can remain in
                    # flight under the final two quadrants.
                    rocdl.s_wait_dscnt(_bank_half_wn * 4 + _b_half_scale_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_group(
                    _bank_group_size * 2,
                    0,
                    a_top_frags,
                    b_right_frags,
                    a_scales_all,
                    b_right_scales,
                )
                _emit_group(
                    _bank_group_size * 3,
                    _bank_half_wm,
                    a_bottom_frags,
                    b_right_frags,
                    a_scales_all,
                    b_right_scales,
                    emit_filler_now=is_last_ks,
                )

                if const_expr(not is_last_ks):
                    b_left_frags = next_left_frags
                    b_left_scales = next_left_scales

            return current_accs

        def compute_tile_fp8_quadrant(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
        ):
            current_accs = list(accs_in)
            a_buf, a_base = _precompute_a_lane_bases(lds_a)
            b_buf, b_base = _precompute_b_lane_bases(lds_b)
            as_buf, as_base = _precompute_scale_lane_bases(lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
            bs_buf, bs_base = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b
            )
            _b_half_loads = _fp8_half_wn * _b_frag_loads_per_wn
            _b_left_bundle_loads = _b_half_loads + _fp8_b_scale_loads

            def _load_a_group(wm_base, wm_count, ks):
                return [load_a_frag(a_buf, a_base, wm_base + wm_local, ks) for wm_local in range_constexpr(wm_count)]

            def _load_b_half(wn_base, ks):
                return [
                    load_b_frag(b_buf, b_base, wn_base + wn_local, ks) for wn_local in range_constexpr(_fp8_half_wn)
                ]

            def _load_a_scales(ks):
                if const_expr(is_ptpc):
                    return None  # PTPC: scale applied in epilogue, not in K-loop
                a_scales = load_scale_b128(as_buf, as_base, wmma_m_rep, ks)
                if const_expr(use_scale_opsel):
                    return a_scales[::2]
                return a_scales

            def _load_b_scales(ks):
                if const_expr(is_ptpc):
                    return None  # PTPC: scale applied in epilogue, not in K-loop
                b_scales = load_scale_b128(bs_buf, bs_base, b_scale_load_rep, ks)
                if const_expr(use_scale_opsel):
                    return b_scales[::2]
                return b_scales

            def _load_b_left_bundle(ks):
                return _load_b_half(0, ks), _load_b_scales(ks)

            def _emit_group_rows(
                wm_base,
                wn_base,
                a_frags,
                b_frags,
                a_scales,
                b_scales,
                row_start,
                row_count,
                emit_filler_now=False,
            ):
                if const_expr(emit_filler_now and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()
                for row_offset in range_constexpr(row_count):
                    wm_local = row_start + row_offset
                    global_wm = wm_base + wm_local
                    for wn_local in range_constexpr(_fp8_half_wn):
                        global_wn = wn_base + wn_local
                        _emit_wmma(
                            current_accs,
                            global_wm,
                            global_wn,
                            a_frags[wm_local],
                            b_frags[wn_local],
                            a_scales,
                            b_scales,
                        )

            def _emit_group(wm_base, wn_base, a_frags, b_frags, a_scales, b_scales, emit_filler_now=False):
                _emit_group_rows(
                    wm_base,
                    wn_base,
                    a_frags,
                    b_frags,
                    a_scales,
                    b_scales,
                    0,
                    _fp8_half_wm,
                    emit_filler_now=emit_filler_now,
                )

            def _emit_group_col(wm_base, wn_base, a_frags, b_frags, a_scales, b_scales, wn_local):
                global_wn = wn_base + wn_local
                for wm_local in range_constexpr(_fp8_half_wm):
                    global_wm = wm_base + wm_local
                    _emit_wmma(
                        current_accs,
                        global_wm,
                        global_wn,
                        a_frags[wm_local],
                        b_frags[wn_local],
                        a_scales,
                        b_scales,
                    )

            b_left_frags, b_scales = _load_b_left_bundle(0)
            _first_top_row_keep = max((_fp8_half_wm - 1) * DS_LOADS_PER_A_FRAG - _fp8_b_scale_loads, 0)
            _bottom_left_keep = max(_b_half_loads - DS_LOADS_PER_A_FRAG, 0)

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales = _load_a_scales(ks)

                a_top_frags = _load_a_group(0, _fp8_half_wm, ks)

                # Consume the first top-left row before issuing bottom-A.
                # The barriers only constrain LLVM scheduling; they are not
                # hardware synchronization points.
                rocdl.s_wait_dscnt(_first_top_row_keep)
                rocdl.sched_barrier(0)
                _emit_group_rows(0, 0, a_top_frags, b_left_frags, a_scales, b_scales, 0, 1)
                rocdl.sched_barrier(0)

                a_bottom_frags = _load_a_group(_fp8_half_wm, _fp8_half_wm, ks)
                if const_expr(_fp8_half_wm > 1):
                    _emit_group_rows(
                        0,
                        0,
                        a_top_frags,
                        b_left_frags,
                        a_scales,
                        b_scales,
                        1,
                        _fp8_half_wm - 1,
                    )
                b_right_frags = _load_b_half(_fp8_half_wn, ks)

                # Drain bottom-A while keeping most right-half B in flight.
                rocdl.s_wait_dscnt(_bottom_left_keep)

                _emit_group(_fp8_half_wm, 0, a_bottom_frags, b_left_frags, a_scales, b_scales)

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if const_expr(not is_last_ks):
                    next_left_frags, next_b_scales = _load_b_left_bundle(ks + 1)

                for wn_local in range_constexpr(_fp8_half_wn):
                    if const_expr(not is_last_ks):
                        _right_keep = _b_left_bundle_loads + (_fp8_half_wn - wn_local - 1) * _b_frag_loads_per_wn
                    else:
                        _right_keep = (_fp8_half_wn - wn_local - 1) * _b_frag_loads_per_wn
                    rocdl.s_wait_dscnt(_right_keep)
                    _emit_group_col(0, _fp8_half_wn, a_top_frags, b_right_frags, a_scales, b_scales, wn_local)

                if const_expr(is_last_ks and late_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    late_compute_callback()

                if const_expr(is_last_ks and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()

                for wn_local in range_constexpr(_fp8_half_wn):
                    _emit_group_col(
                        _fp8_half_wm,
                        _fp8_half_wn,
                        a_bottom_frags,
                        b_right_frags,
                        a_scales,
                        b_scales,
                        wn_local,
                    )

                if const_expr(not is_last_ks):
                    b_left_frags = next_left_frags
                    b_scales = next_b_scales

            return current_accs

        def compute_tile_fp8_deep_pipeline(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            a0_prefetch=None,
            scale_k_base=None,
            pf_a_scales=None,
            pf_b_scales=None,
        ):
            current_accs = list(accs_in)
            a_buf, a_base = _precompute_a_lane_bases(lds_a)
            b_buf, b_base = _precompute_b_lane_bases(lds_b)
            as_buf, as_base = _precompute_scale_lane_bases(lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
            bs_buf, bs_base = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b
            )

            def load_a_pair(wm_pair, ks):
                wm_base = wm_pair * _fp8_pair_wm
                return [
                    load_a_frag(a_buf, a_base, wm_base + wm_local, ks) for wm_local in range_constexpr(_fp8_pair_wm)
                ]

            def load_b_pair(wn_pair, ks):
                wn_base = wn_pair * _fp8_pair_wn
                return [
                    load_b_frag(b_buf, b_base, wn_base + wn_local, ks) for wn_local in range_constexpr(_fp8_pair_wn)
                ]

            def _load_a_scales(ks):
                if const_expr(is_ptpc):
                    return None  # PTPC: scale applied in epilogue, not in K-loop
                if const_expr(use_buffer_vgpr_scale):
                    if const_expr(pf_a_scales is not None):
                        return pf_a_scales  # prefetched (issued in the prior compute tile)
                    return _bvs_load_scales(_bvs_a_rsrc, _bvs_mb_a, wmma_m_rep, scale_k_base)
                return load_scale_b128(as_buf, as_base, wmma_m_rep, ks)

            def _load_b_scales(ks):
                if const_expr(is_ptpc):
                    return None  # PTPC: scale applied in epilogue, not in K-loop
                if const_expr(use_buffer_vgpr_scale):
                    if const_expr(pf_b_scales is not None):
                        return pf_b_scales
                    return _bvs_load_scales(_bvs_b_rsrc, _bvs_mb_b, b_scale_load_rep, scale_k_base)
                return load_scale_b128(bs_buf, bs_base, b_scale_load_rep, ks)

            def emit_panel_2x2(
                wm_pair,
                wn_pair,
                a_pair,
                b_pair,
                scale_pair,
                prefetch_after_first_row=None,
            ):
                a_scales, b_scales = scale_pair
                wm_base = wm_pair * _fp8_pair_wm
                wn_base = wn_pair * _fp8_pair_wn
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base,
                        wn_base + wn_local,
                        a_pair[0],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )
                if const_expr(prefetch_after_first_row is not None):
                    prefetch_after_first_row()
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base + 1,
                        wn_base + wn_local,
                        a_pair[1],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )

            def emit_panel_2x2_row(wm_pair, wn_pair, row_local, a_pair, b_pair, scale_pair):
                a_scales, b_scales = scale_pair
                wm_base = wm_pair * _fp8_pair_wm
                wn_base = wn_pair * _fp8_pair_wn
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base + row_local,
                        wn_base + wn_local,
                        a_pair[row_local],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )

            _pair_loads = _fp8_pair_a_loads
            _two_pair_loads = _fp8_pair_a_loads + _fp8_pair_b_loads

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales = _load_a_scales(ks)
                b_scales = _load_b_scales(ks)
                scale_pair = (a_scales, b_scales)

                b0 = load_b_pair(0, ks)
                if const_expr(ks == 0 and a0_prefetch is not None and len(a0_prefetch) == _fp8_pair_wm):
                    a0 = list(a0_prefetch)
                elif const_expr(ks == 0 and a0_prefetch is not None):
                    a0 = [a0_prefetch[0], load_a_frag(a_buf, a_base, 1, ks)]
                else:
                    a0 = load_a_pair(0, ks)
                b1 = load_b_pair(1, ks)
                b2 = load_b_pair(2, ks)

                a1_box = [None]
                b3_box = [None]
                a2_box = [None]
                a3_box = [None]

                def _prefetch_a1():
                    a1_box[0] = load_a_pair(1, ks)

                first_wait_keep = _two_pair_loads + 3
                if const_expr(ks == 0 and a0_prefetch is not None):
                    first_wait_keep += DS_LOADS_PER_A_FRAG * len(a0_prefetch)
                rocdl.s_wait_dscnt(first_wait_keep)
                emit_panel_2x2(0, 0, a0, b0, scale_pair, prefetch_after_first_row=_prefetch_a1)

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                def _prefetch_b3():
                    b3_box[0] = load_b_pair(3, ks)

                def _prefetch_a3():
                    a3_box[0] = load_a_pair(3, ks)

                rocdl.s_wait_dscnt(_pair_loads + _fp8_pair_b_loads)
                emit_panel_2x2(0, 1, a0, b1, scale_pair, prefetch_after_first_row=_prefetch_b3)

                rocdl.s_wait_dscnt(_fp8_pair_b_loads + 2)
                emit_panel_2x2(1, 0, a1_box[0], b0, scale_pair, prefetch_after_first_row=_prefetch_a3)

                def _prefetch_a2():
                    a2_box[0] = load_a_pair(2, ks)

                emit_panel_2x2(1, 1, a1_box[0], b1, scale_pair)

                emit_panel_2x2(0, 2, a0, b2, scale_pair, prefetch_after_first_row=_prefetch_a2)
                emit_panel_2x2_row(1, 2, 0, a1_box[0], b2, scale_pair)
                emit_panel_2x2_row(1, 2, 1, a1_box[0], b2, scale_pair)
                rocdl.s_wait_dscnt(_pair_loads)
                emit_panel_2x2(0, 3, a0, b3_box[0], scale_pair)
                emit_panel_2x2(1, 3, a1_box[0], b3_box[0], scale_pair)

                emit_panel_2x2(2, 0, a2_box[0], b0, scale_pair)
                if const_expr(is_last_ks and late_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    late_compute_callback()
                emit_panel_2x2(2, 1, a2_box[0], b1, scale_pair)

                rocdl.s_wait_dscnt(0)
                emit_panel_2x2(3, 0, a3_box[0], b0, scale_pair)
                emit_panel_2x2(3, 1, a3_box[0], b1, scale_pair)

                if const_expr(is_last_ks and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()

                emit_panel_2x2(2, 2, a2_box[0], b2, scale_pair)
                emit_panel_2x2(2, 3, a2_box[0], b3_box[0], scale_pair)
                emit_panel_2x2(3, 2, a3_box[0], b2, scale_pair)
                emit_panel_2x2(3, 3, a3_box[0], b3_box[0], scale_pair)

            return current_accs

        def compute_tile_b_streaming(
            accs_in, lds_a, lds_b, lds_as, lds_bs, emit_filler=None, mid_compute_callback=None
        ):
            """compute_tile counterpart with A held and B streamed."""
            current_accs = list(accs_in)
            a_buf, a_base = _precompute_a_lane_bases(lds_a)
            b_buf, b_base = _precompute_b_lane_bases(lds_b)
            as_buf, as_base = _precompute_scale_lane_bases(lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
            bs_buf, bs_base = _precompute_scale_lane_bases(
                lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b
            )
            load_args = (a_buf, a_base, as_buf, as_base, bs_buf, bs_base)

            if const_expr(k_wmma_steps == 1):
                a_frags, a_scales, b_scales = _load_a_and_scales(*load_args, 0)
                return _b_streaming_compute(
                    current_accs,
                    b_buf,
                    b_base,
                    a_frags,
                    a_scales,
                    b_scales,
                    0,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                )

            prev_a, prev_as, prev_bs = _load_a_and_scales(*load_args, 0)
            for ks in range_constexpr(k_wmma_steps - 1):
                current_accs, (prev_a, prev_as, prev_bs) = _b_streaming_compute(
                    current_accs,
                    b_buf,
                    b_base,
                    prev_a,
                    prev_as,
                    prev_bs,
                    ks,
                    next_info=load_args + (ks + 1,),
                    mid_compute_callback=mid_compute_callback if ks == 0 else None,
                )
            return _b_streaming_compute(
                current_accs,
                b_buf,
                b_base,
                prev_a,
                prev_as,
                prev_bs,
                k_wmma_steps - 1,
                emit_filler=emit_filler,
            )

        def hot_loop_scheduler():
            _half_wm = wmma_m_rep // 2
            _half_wmma = _half_wm * wmma_n_rep
            _front_wm = (wmma_m_rep + 1) // 2
            _b_loads_per_frag = 2 if is_a8w4 else 4
            _scale_dsrd = 0 if is_ptpc else 2

            if const_expr(_use_lds_pf):
                # Interleaved full-tile prefetch: distribute next-tile ds_loads evenly
                # across WMMAs so each WMMA group is preceded by roughly equal loads.
                # floor(total/n_wmma) loads per WMMA; remainder goes to the first group.
                _next_ks_loads = wmma_n_rep * _b_loads_per_frag + _scale_dsrd + wmma_m_rep * DS_LOADS_PER_A_FRAG
                _n_wmma_per_ks = wmma_n_rep * wmma_m_rep
                _loads_per_wmma = _next_ks_loads // _n_wmma_per_ks
                _loads_extra = _next_ks_loads % _n_wmma_per_ks
                for _ks in range_constexpr(k_wmma_steps):
                    for _wn in range_constexpr(_n_wmma_per_ks):
                        # Extra loads go to the first WMMA slot in each k-step.
                        _this_loads = _loads_per_wmma + (1 if _wn < _loads_extra else 0)
                        rocdl.sched_dsrd(_this_loads)
                        rocdl.sched_mfma(1)
                rocdl.sched_barrier(0)
                return

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag + _scale_dsrd + _half_wm * DS_LOADS_PER_A_FRAG)
                else:
                    rocdl.sched_dsrd(_half_wm * DS_LOADS_PER_A_FRAG)
                rocdl.sched_mfma(_half_wmma)
                rocdl.sched_dsrd(_half_wm * DS_LOADS_PER_A_FRAG)
                rocdl.sched_mfma(_half_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag + _scale_dsrd)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp4_bank_friendly():
            _a_all_loads = wmma_m_rep * DS_LOADS_PER_A_FRAG
            _a_scale_loads = (wmma_m_rep + 3) // 4
            _b_half_loads = _bank_half_wn * 4
            _b_half_scale_loads = (_bank_half_b_scale_rep + 3) // 4
            _group_wmma = _bank_group_size
            _right_half_loads = _b_half_loads + _b_half_scale_loads

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(_a_all_loads + _a_scale_loads + _b_half_loads + _b_half_scale_loads)
                else:
                    rocdl.sched_dsrd(_a_all_loads + _a_scale_loads)
                rocdl.sched_mfma(_group_wmma)
                rocdl.sched_dsrd(_right_half_loads)
                rocdl.sched_mfma(_group_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(_right_half_loads)
                rocdl.sched_mfma(_group_wmma)
                rocdl.sched_mfma(_group_wmma)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp8_quadrant():
            _a_scale_loads = 0 if is_ptpc else (wmma_m_rep + 3) // 4
            _a_top_loads = _fp8_half_wm * DS_LOADS_PER_A_FRAG
            _a_bottom_loads = _a_top_loads
            _b_half_loads = _fp8_half_wn * _b_frag_loads_per_wn
            _b_left_bundle_loads = _b_half_loads + _fp8_b_scale_loads
            _group_wmma = _fp8_group_size
            _first_row_wmma = _fp8_half_wn
            _remaining_top_left_wmma = (_fp8_half_wm - 1) * _fp8_half_wn

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(_b_left_bundle_loads + _a_scale_loads + _a_top_loads)
                else:
                    rocdl.sched_dsrd(_a_scale_loads + _a_top_loads)
                rocdl.sched_mfma(_first_row_wmma)
                rocdl.sched_dsrd(_a_bottom_loads)
                if const_expr(_remaining_top_left_wmma > 0):
                    rocdl.sched_mfma(_remaining_top_left_wmma)
                rocdl.sched_dsrd(_b_half_loads)
                rocdl.sched_mfma(_group_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(_b_left_bundle_loads)
                for _wn_local in range_constexpr(_fp8_half_wn):
                    rocdl.sched_mfma(_fp8_half_wm)
                for _wn_local in range_constexpr(_fp8_half_wn):
                    rocdl.sched_mfma(_fp8_half_wm)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp8_deep_pipeline():
            def _sched_panel_2x2(prefetch_loads=0):
                if const_expr(prefetch_loads > 0):
                    rocdl.sched_mfma(_fp8_pair_wn)
                    rocdl.sched_dsrd(prefetch_loads)
                    rocdl.sched_mfma(_fp8_pair_wn)
                else:
                    rocdl.sched_mfma(_fp8_pair_wm * _fp8_pair_wn)

            def _sched_panel_row():
                rocdl.sched_mfma(_fp8_pair_wn)

            _initial_loads = _fp8_scale_loads + _fp8_pair_b_loads * 3 + _fp8_pair_a_loads

            for _ks in range_constexpr(k_wmma_steps):
                _ks_initial_loads = _initial_loads
                if const_expr(_ks == 0):
                    _ks_initial_loads -= _fp8_pair_a_loads
                rocdl.sched_dsrd(_ks_initial_loads)
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_2x2(_fp8_pair_b_loads)
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_2x2()
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_row()
                _sched_panel_row()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
            rocdl.sched_barrier(0)

        def compute_tile_scheduled(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            a0_prefetch=None,
            scale_k_base=None,
            pf_a_scales=None,
            pf_b_scales=None,
            pf_all_ks=None,
            next_lds_a=None,
            next_lds_b=None,
            next_lds_bs=None,
            next_lds_as=None,
            tdm_cb_pf=None,
            tdm_cb_remain=None,
        ):
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_B_STREAMING):
                return compute_tile_b_streaming(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                )
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND):
                return compute_tile_fp4_bank_friendly(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                )
            # Order-graft: when PF_QUADRANT routes a quadrant shape onto the pf pipeline
            # (_pf_allow_quadrant), fall through to compute_tile (which now emits in
            # quadrant order via _pf_pos). Otherwise use the legacy hand-tuned quadrant.
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT and not _pf_allow_quadrant):
                return compute_tile_fp8_quadrant(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    late_compute_callback=late_compute_callback,
                )
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE):
                return compute_tile_fp8_deep_pipeline(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    late_compute_callback=late_compute_callback,
                    a0_prefetch=a0_prefetch,
                    scale_k_base=scale_k_base,
                    pf_a_scales=pf_a_scales,
                    pf_b_scales=pf_b_scales,
                )
            return compute_tile(
                accs_in,
                lds_a,
                lds_b,
                lds_as,
                lds_bs,
                emit_filler=emit_filler,
                mid_compute_callback=mid_compute_callback,
                pf_all_ks=pf_all_ks,
                next_lds_a=next_lds_a,
                next_lds_b=next_lds_b,
                next_lds_bs=next_lds_bs,
                next_lds_as=next_lds_as,
                tdm_cb_pf=tdm_cb_pf,
                tdm_cb_remain=tdm_cb_remain,
            )

        def hot_loop_scheduler_b_streaming():
            """hot_loop_scheduler counterpart for B-streaming."""
            _front_wn = (wmma_n_rep + 1) // 2
            _back_wn = wmma_n_rep - _front_wn
            _a_loads_total = wmma_m_rep * DS_LOADS_PER_A_FRAG
            _front_b_loads = _front_wn * _b_frag_loads_per_wn
            _back_b_loads = _back_wn * _b_frag_loads_per_wn
            _next_ks_loads = _a_loads_total + _scale_ds_loads

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(_next_ks_loads + _front_b_loads)
                else:
                    rocdl.sched_dsrd(_front_b_loads)
                rocdl.sched_mfma(_front_wn * wmma_m_rep)
                if const_expr(_back_wn > 0):
                    rocdl.sched_dsrd(_back_b_loads)
                    rocdl.sched_mfma(_back_wn * wmma_m_rep)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(_next_ks_loads)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_scheduled():
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_B_STREAMING):
                hot_loop_scheduler_b_streaming()
            elif const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND):
                hot_loop_scheduler_fp4_bank_friendly()
            elif const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE):
                hot_loop_scheduler_fp8_deep_pipeline()
            elif const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT):
                hot_loop_scheduler_fp8_quadrant()
            else:
                hot_loop_scheduler()

        def prefetch_fp8_deep_a0_frags(lds_a):
            a_buf, a_base = _precompute_a_lane_bases(lds_a)
            return [load_a_frag(a_buf, a_base, wm_local, 0) for wm_local in range_constexpr(_fp8_pair_wm)]

        def maybe_prefetch_fp8_deep_a0(lds_a):
            # Call only after the TDM fence for this stage; pre-fence LDS reads can race multicast delivery.
            if const_expr(use_fp8_deep_pipeline_schedule):
                return prefetch_fp8_deep_a0_frags(lds_a)
            return None

        # ── Epilogue (unified via _sub_tiles) ──
        def _get_acc_sub8(accs, acc_idx, vec_base):
            """Extract 8-element sub-vector from accumulator."""
            if const_expr(ACC_VEC_SIZE == 8):
                return accs[acc_idx]
            indices = [vec_base + i for i in range_constexpr(8)]
            acc = fx.Vector(accs[acc_idx])
            return acc.shuffle(acc, indices)

        def epilogue_prepare_addrs():
            addrs = []
            _bf16_out = out_dtype in ("bf16", "f16")
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                col_base = blk_n + warp_n_base + arith.index(wn * WMMA_N) + lane_kgrp * arith.index(8)
                if const_expr(_bf16_out):
                    c_off_bytes = (row * n_stride + col_base) * arith.index(elem_bytes_d)
                    addrs.append(c_off_bytes)
                else:
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        c_off = row * n_stride + col
                        addrs.append(c_off)
            return addrs

        _bf16_out = out_dtype in ("bf16", "f16")
        _out_elem_local = T.bf16 if out_dtype == "bf16" else (T.f16 if out_dtype == "f16" else None)

        def epilogue_stores(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                if const_expr(_bf16_out):
                    addr_idx += store_acc_vec8_to_buffer(
                        sub8, c_rsrc, addrs[addr_idx], out_elem=_out_elem_local, offset_is_bytes=True
                    )
                else:
                    addr_idx += store_acc_vec8_to_buffer(sub8, c_rsrc, addrs[addr_idx : addr_idx + 2])

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                imm = m_off * _lds_d_stride_elems + wn * _n_col_d_elems
                store_acc_vec8_to_lds(d_buf, d_base, imm, sub8, out_elem=_out_elem_local)

        def _atomic_fadd_global(val, byte_off):
            # Device-scoped, relaxed atomic add into C at c_global_base_i64 + byte_off.
            addr_i64 = llvm.AddOp(
                c_global_base_i64, arith.index_cast(T.i64, byte_off), llvm.IntegerOverflowFlags(0)
            ).result
            ptr = llvm.IntToPtrOp(c_global_ptr_type, addr_i64).result
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                ptr,
                val.ir_value(),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

        def _atomic_add_acc_vec8_to_buffer(acc_vec8, addr):
            if const_expr(_bf16_out):
                h_vec = fx.Vector(arith.trunc_f(T.vec(8, _out_elem_local), acc_vec8))
                for pair in range_constexpr(4):
                    pair_vec = fx.Vector.from_elements([h_vec[pair * 2], h_vec[pair * 2 + 1]])
                    byte_off = addr + arith.index(pair * 4)
                    _atomic_fadd_global(pair_vec, byte_off)
                return 1

            acc_vec = fx.Vector(acc_vec8)
            for half in range_constexpr(2):
                base_addr = addr[half] if isinstance(addr, (list, tuple)) else addr
                for vi in range_constexpr(4):
                    val = acc_vec[half * 4 + vi]
                    byte_off = (base_addr + arith.index(vi)) * arith.index(4)
                    _atomic_fadd_global(val, byte_off)
            return 2

        def epilogue_atomic_adds(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                n_slots = 1 if _bf16_out else 2
                addr_arg = addrs[addr_idx] if _bf16_out else addrs[addr_idx : addr_idx + 2]
                # Atomics use a raw global ptr (no num_records clip), so predicate
                # per-lane to skip rows >= M.
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                if_op = scf.IfOp(row < m_idx, [], has_else=False)
                with ir.InsertionPoint(if_op.then_block):
                    _atomic_add_acc_vec8_to_buffer(sub8, addr_arg)
                    scf.YieldOp([])
                addr_idx += n_slots

        def grouped_accs_to_row_major(accs_grouped):
            row_major = [None] * n_accs
            for group_idx in range_constexpr(n_accs):
                row_major[_bank_group_to_row_major[group_idx]] = accs_grouped[group_idx]
            return row_major

        def finalize_acc_layout(accs_in):
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND):
                return grouped_accs_to_row_major(accs_in)
            return accs_in

        def epilogue_load_ptpc_scales():
            # PTPC scales: sa[M] per-token (scalar per wm), sb[N] per-channel
            # (8 contiguous N cols per wn). Both fp32, constant along K.
            # The scale memrefs are dynamically shaped, so max_size=False would fall
            # back to a max-sized descriptor and disable hardware OOB. Derive
            # num_records from runtime M / compile-time N (fp32 = 4 bytes) so the
            # partial last M-tile clips rows >= M (and cols >= N) to 0.
            sa_rsrc = buffer_ops.create_buffer_resource(arg_a_scale, num_records_bytes=m_idx * arith.index(4))
            sb_rsrc = buffer_ops.create_buffer_resource(arg_b_scale, num_records_bytes=N * 4)
            sa = []
            for wm in range_constexpr(wmma_m_rep):
                row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                sv = buffer_ops.buffer_load(sa_rsrc, arith.index_cast(T.i32, row), vec_width=1, dtype=T.f32)
                sa.append(fx.Vector.from_elements([sv] * 8))
            sb = []
            for wn in range_constexpr(wmma_n_rep):
                col_base = blk_n + warp_n_base + arith.index(wn * WMMA_N) + lane_kgrp * arith.index(8)
                # buffer_load vec_width is capped at 4: read 8 cols as 2x vec4.
                lo = fx.Vector(
                    buffer_ops.buffer_load(sb_rsrc, arith.index_cast(T.i32, col_base), vec_width=4, dtype=T.f32)
                )
                hi = fx.Vector(
                    buffer_ops.buffer_load(
                        sb_rsrc, arith.index_cast(T.i32, col_base + arith.index(4)), vec_width=4, dtype=T.f32
                    )
                )
                sb.append(fx.Vector.from_elements([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]))
            return sa, sb

        def epilogue_apply_ptpc_scale(accs_in, sa, sb):
            out = list(accs_in)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    out[idx] = (fx.Vector(out[idx]) * sb[wn] * sa[wm]).ir_value()
            return out

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
                arg_a,
                (blk_m, pf_k_packed_a),
                (tile_m, packed_tile_k_a),
                (K_packed_a, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )
            tdm_ops.l2_prefetch_tile(
                arg_b,
                (blk_n / arith.index(16), pf_k_packed_b * arith.index(16)),
                (tile_n // 16, packed_tile_k_b * 16),
                (K_packed_b * 16, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )

        def _l2_prefetch_b_at(k_base):
            # Cold-start B-only L2 warm at the EXACT k-tile (no look-ahead
            # offset), used in the prologue to cover the prologue B tiles before
            # their TDM loads. Pure L2 hint — no correctness role.
            if const_expr(_effective_l2_pf <= 0):
                return
            pf_k_packed_b = k_base / arith.index(PACK_FACTOR_B)
            tdm_ops.l2_prefetch_tile(
                arg_b,
                (blk_n / arith.index(16), pf_k_packed_b * arith.index(16)),
                (tile_n // 16, packed_tile_k_b * 16),
                (K_packed_b * 16, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        accs = [acc_zero] * n_accs

        lds_a_data_f16 = lds_a_data_bytes // 2
        lds_b_data_f16 = lds_b_data_bytes // 2
        lds_a_scale_f16 = lds_a_scale_bytes // 2
        lds_b_scale_f16 = lds_b_scale_bytes // 2

        arena_base_ptr = arena_alloc.get_base()

        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_data_off[i], elem_ty_lds, shape=(lds_a_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_data_off[i], elem_ty_lds, shape=(lds_b_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        if const_expr(is_ptpc):
            # PTPC applies sa*sb in the epilogue from global memory: no scale LDS.
            # Alias the scale stage handles to A/B so the shared plumbing stays
            # valid; for PTPC they are never written (no scale TDM) or read.
            stages_as = stages_a
            stages_bs = stages_b
        else:
            stages_as = [
                SmemPtr(arena_base_ptr, stage_a_scale_off[i], elem_ty_lds, shape=(lds_a_scale_f16,))
                for i in range_constexpr(num_buffers)
            ]
            stages_bs = [
                SmemPtr(arena_base_ptr, stage_b_scale_off[i], elem_ty_lds, shape=(lds_b_scale_f16,))
                for i in range_constexpr(num_buffers)
            ]

        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_as_mem = [stages_as[i].get() for i in range_constexpr(num_buffers)]
        stages_bs_mem = [stages_bs[i].get() for i in range_constexpr(num_buffers)]

        stages_a_idx = [extract_lds_base_idx(stages_a[i]) for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i]) for i in range_constexpr(num_buffers)]
        stages_as_idx = [extract_lds_base_idx(stages_as[i]) for i in range_constexpr(num_buffers)]
        stages_bs_idx = [extract_lds_base_idx(stages_bs[i]) for i in range_constexpr(num_buffers)]

        if const_expr(use_tdm_store):
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(d_lds_base_ptr, d_output_off, elem_ty_lds, shape=(d_lds_f16_count,))
            d_lds_buffer = get_lds_memref(d_smem)
            warp_lds_off = (wave_m_idx * arith.index(n_warp) + wave_n_idx) * arith.index(_warp_d_elems)
            d_lane_base = (
                warp_lds_off + lane16 * arith.index(_lds_d_stride_elems) + lane_kgrp * arith.index(4 * elem_bytes_d)
            )
            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            # Match the TDM-store descriptor offsets to the compute wave mapping.
            if const_expr(use_fp8_deep_pipeline_schedule):
                wave_m_sgpr = wave_id_idx % arith.index(m_warp)
                wave_n_sgpr = wave_id_idx / arith.index(m_warp)
            else:
                wave_m_sgpr = wave_id_idx / arith.index(n_warp)
                wave_n_sgpr = wave_id_idx % arith.index(n_warp)
            d_warp_linear_sgpr = wave_m_sgpr * arith.index(n_warp) + wave_n_sgpr
            d_warp_off_sgpr = d_warp_linear_sgpr * arith.index(warp_d_bytes) + arith.index(d_output_off)
            warp_m_off_sgpr = wave_m_sgpr * arith.index(warp_tile_m)
            warp_n_off_sgpr = wave_n_sgpr * arith.index(warp_tile_n)
            d_desc = _make_tdm_desc(
                global_ptr=arg_c,
                lds_memref=d_lds_base_ptr,
                global_offset=(blk_m + warp_m_off_sgpr, blk_n + warp_n_off_sgpr),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(n_stride, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
                oob_outer_bound=i32_m,
            )

        # TDM descriptor lane layout: dgroup0 = [predicate, lds_addr, addr_lo, addr_hi].
        def _dg0_lane(desc, lane):
            return fx.Vector(desc.dgroup0)[lane]

        def _pack_dg0(pred, lds_addr, addr_lo, addr_hi):
            # Build as two vec2 halves joined by shuffle so LLVM cannot
            # reuse a nearby vec4 template (which causes a dead s_mov_b64
            # for the upper pair that is immediately overwritten).
            lo = fx.Vector.from_elements([pred, lds_addr], fx.Int32)
            hi = fx.Vector.from_elements([addr_lo, addr_hi], fx.Int32)
            return lo.shuffle(hi, [0, 1, 2, 3])

        # Precompute LDS addresses for TDM descriptor switching
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        stages_as_lds_addr = []
        stages_bs_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(_dg0_lane(make_desc_a(stages_a_mem[i], arith.index(0)), 1))
            stages_b_lds_addr.append(_dg0_lane(make_desc_b(stages_b_mem[i], arith.index(0)), 1))
            if const_expr(not is_ptpc):
                stages_as_lds_addr.append(_dg0_lane(make_desc_as(stages_as_mem[i], arith.index(0)), 1))
                stages_bs_lds_addr.append(_dg0_lane(make_desc_bs(stages_bs_mem[i], arith.index(0)), 1))

        desc_a_init = make_desc_a(stages_a_mem[0], split_k_base)
        desc_b_init = make_desc_b(stages_b_mem[0], split_k_base)
        if const_expr(is_ptpc):
            # No scale TDM for PTPC: alias the scale descriptors/addresses to A/B.
            # Scale waves are predicated off, so these selections are never issued.
            stages_as_lds_addr = stages_a_lds_addr
            stages_bs_lds_addr = stages_b_lds_addr
            desc_as_init = desc_a_init
            desc_bs_init = desc_b_init
        else:
            desc_as_init = make_desc_as(stages_as_mem[0], split_k_base)
            desc_bs_init = make_desc_bs(stages_bs_mem[0], split_k_base)
        if const_expr(use_ab_half_split):
            stages_a0_lds_addr = []
            stages_b0_lds_addr = []
            stages_a1_lds_addr = []
            stages_b1_lds_addr = []
            for i in range_constexpr(num_buffers):
                stages_a0_lds_addr.append(_dg0_lane(make_desc_a_half(stages_a_mem[i], arith.index(0), 0), 1))
                stages_b0_lds_addr.append(_dg0_lane(make_desc_b_half(stages_b_mem[i], arith.index(0), 0), 1))
                stages_a1_lds_addr.append(_dg0_lane(make_desc_a_half(stages_a_mem[i], arith.index(0), 1), 1))
                stages_b1_lds_addr.append(_dg0_lane(make_desc_b_half(stages_b_mem[i], arith.index(0), 1), 1))

            desc_a0_init = make_desc_a_half(stages_a_mem[0], split_k_base, 0)
            desc_b0_init = make_desc_b_half(stages_b_mem[0], split_k_base, 0)
            desc_a1_init = make_desc_a_half(stages_a_mem[0], split_k_base, 1)
            desc_b1_init = make_desc_b_half(stages_b_mem[0], split_k_base, 1)

        adv_a_i32 = fx.Int32(tile_k // PACK_FACTOR_A)
        # Tile-contiguous a8w4 advances one full tile (_b_num_blocks 256B blocks) per
        # K-step; the interleaved layout advances packed_tile_k_b*16 bytes.
        adv_b_i32 = fx.Int32(_b_num_blocks * B_TILE_BYTES if is_a8w4 else packed_tile_k_b * 16)
        adv_as_i32 = fx.Int32(tile_k // SCALE_BLOCK * wmma_m_rep)
        adv_bs_i32 = fx.Int32(tile_k // SCALE_BLOCK * b_scale_load_rep)

        pred_const = fx.Int32(1)
        if const_expr(wave_specialized_tdm):
            _drop_scale_waves = is_ptpc or (use_buffer_vgpr_scale and not use_ab_half_split)
            _active_wave_limit = 2 if _drop_scale_waves else 4
            active_pred_const = arith.select(tdm_wave_id < fx.Int32(_active_wave_limit), fx.Int32(1), fx.Int32(0))

            def _select4(values):
                return _select_wave_tdm_value(values[0], values[1], values[2], values[3])

            def _desc_lanes(descs, lane):
                return [_dg0_lane(desc, lane) for desc in descs]

            def _select_active_tdm(stage_lds_addrs, descs, advs):
                active_stages = [
                    _select_wave_tdm_value(
                        stage_lds_addrs[0][i],
                        stage_lds_addrs[1][i],
                        stage_lds_addrs[2][i],
                        stage_lds_addrs[3][i],
                    )
                    for i in range_constexpr(num_buffers)
                ]
                return (
                    active_stages,
                    _select4(_desc_lanes(descs, 2)),
                    _select4(_desc_lanes(descs, 3)),
                    _select4([desc.dgroup1 for desc in descs]),
                    _select4(advs),
                )

        else:
            active_pred_const = pred_const

        if const_expr(use_ab_half_split):
            # All 4 waves load A/B halves: wave0=A0, wave1=B0, wave2=A1, wave3=B1.
            # Both halves of A share adv_a (same K-step); both halves of B share adv_b.
            active_stage_lds_addr, active_addr_lo, active_addr_hi, active_dgroup1, active_adv_i32 = _select_active_tdm(
                (stages_a0_lds_addr, stages_b0_lds_addr, stages_a1_lds_addr, stages_b1_lds_addr),
                (desc_a0_init, desc_b0_init, desc_a1_init, desc_b1_init),
                (adv_a_i32, adv_b_i32, adv_a_i32, adv_b_i32),
            )
        elif const_expr(wave_specialized_tdm):
            active_stage_lds_addr, active_addr_lo, active_addr_hi, active_dgroup1, active_adv_i32 = _select_active_tdm(
                (stages_a_lds_addr, stages_b_lds_addr, stages_as_lds_addr, stages_bs_lds_addr),
                (desc_a_init, desc_b_init, desc_as_init, desc_bs_init),
                (adv_a_i32, adv_b_i32, adv_as_i32, adv_bs_i32),
            )
            if const_expr(_pf_tdm_split):
                # pf/remain K-split active descriptors (sub-depth WMMA-driven TDM).
                def _seg_descs(seg):
                    _as = (make_desc_as_seg(stages_as_mem[0], split_k_base, seg)
                           if not is_ptpc else make_desc_a_seg(stages_a_mem[0], split_k_base, seg))
                    _bs = (make_desc_bs_seg(stages_bs_mem[0], split_k_base, seg)
                           if not is_ptpc else make_desc_b_seg(stages_b_mem[0], split_k_base, seg))
                    return (
                        make_desc_a_seg(stages_a_mem[0], split_k_base, seg),
                        make_desc_b_seg(stages_b_mem[0], split_k_base, seg),
                        _as, _bs,
                    )

                _advs = (adv_a_i32, adv_b_i32, adv_as_i32, adv_bs_i32)
                _stg = (stages_a_lds_addr, stages_b_lds_addr, stages_as_lds_addr, stages_bs_lds_addr)
                _, _seg_alo_pf, _seg_ahi_pf, _seg_dg_pf, _ = _select_active_tdm(_stg, _seg_descs(0), _advs)
                _, _seg_alo_rem, _seg_ahi_rem, _seg_dg_rem, _ = _select_active_tdm(_stg, _seg_descs(1), _advs)
        else:
            addr_lo_a = _dg0_lane(desc_a_init, 2)
            addr_hi_a = _dg0_lane(desc_a_init, 3)
            addr_lo_b = _dg0_lane(desc_b_init, 2)
            addr_hi_b = _dg0_lane(desc_b_init, 3)
            addr_lo_as = _dg0_lane(desc_as_init, 2)
            addr_hi_as = _dg0_lane(desc_as_init, 3)
            addr_lo_bs = _dg0_lane(desc_bs_init, 2)
            addr_hi_bs = _dg0_lane(desc_bs_init, 3)

            dgroup1_a = desc_a_init.dgroup1
            dgroup1_b = desc_b_init.dgroup1
            dgroup1_as = desc_as_init.dgroup1
            dgroup1_bs = desc_bs_init.dgroup1

        def _pipeline_fence(outstanding=0):
            pipeline_fence(outstanding=outstanding, use_cluster=use_cluster)

        def _pipeline_fence_signal(outstanding=0):
            pipeline_fence_signal(outstanding=outstanding, use_cluster=use_cluster)

        if const_expr(wave_specialized_tdm):
            # Hoist uniform dg0 components to SGPR once.  readfirstlane marks each
            # value as "provably uniform" to LLVM's divergence analysis, so when
            # these are later packed into a dgroup0 vector LLVM allocates that
            # vector in SGPR space and emits s_mov rather than v_readfirstlane
            # per tile.
            _pred_sgpr = rocdl.readfirstlane(T.i32, active_pred_const)
            _addr_hi_sgpr = rocdl.readfirstlane(T.i32, active_addr_hi)
            _stage_lds_sgprs = [rocdl.readfirstlane(T.i32, v) for v in active_stage_lds_addr]

            def _issue_active_tdm(load_stage, addr_lo, k_prefetch=None):
                """Issue TDM using pre-computed addr_lo (SGPR).

                Caller is responsible for advancing addr_lo AFTER this call
                (via active_adv_i32) so the s_add_co_i32 happens in the
                compute window, not on the critical path just before TDM.
                """
                dg0 = _pack_dg0(_pred_sgpr, _stage_lds_sgprs[load_stage], addr_lo, _addr_hi_sgpr)
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                if k_prefetch is not None:
                    _l2_prefetch(k_prefetch)

            if const_expr(_pf_tdm_split):
                _seg_ahi_pf_sgpr = rocdl.readfirstlane(T.i32, _seg_ahi_pf)
                _seg_ahi_rem_sgpr = rocdl.readfirstlane(T.i32, _seg_ahi_rem)
                # The remain seg writes at LDS offset col_off (per operand); _pack_dg0
                # below rebuilds dg0 from the per-stage LDS base, so we must re-add that
                # offset here (it was baked into the seg descriptor's dropped dg0).
                _b_rem_lds_off = _cols_b_blk[1][0] * _b_blk_stride if _b_ksmajor else _cols_b[1][0] * 16
                _seg_lds_off_rem = _select_wave_tdm_value(
                    arith.constant(_cols_a[1][0], type=T.i32),
                    arith.constant(_b_rem_lds_off, type=T.i32),
                    arith.constant(_cols_as[1][0], type=T.i32),
                    arith.constant(_cols_bs[1][0], type=T.i32),
                )
                _stage_lds_rem_sgprs = [
                    rocdl.readfirstlane(T.i32, v + _seg_lds_off_rem) for v in active_stage_lds_addr
                ]
                def _issue_active_tdm_seg(load_stage, addr_lo, seg, k_prefetch=None):
                    """Issue one K-segment (seg 0=pf, 1=remain) of the per-tile TDM."""
                    _dg1 = _seg_dg_pf if seg == 0 else _seg_dg_rem
                    _ahi = _seg_ahi_pf_sgpr if seg == 0 else _seg_ahi_rem_sgpr
                    _lds = _stage_lds_sgprs[load_stage] if seg == 0 else _stage_lds_rem_sgprs[load_stage]
                    dg0 = _pack_dg0(_pred_sgpr, _lds, addr_lo, _ahi)
                    tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, _dg1))
                    if k_prefetch is not None:
                        _l2_prefetch(k_prefetch)

        # Main loop — acc_mixed style: TDM issued at tile start, WMMA interleaved with
        # next-tile LDS prefetch. Active only for ROW_MAJOR_STREAMING on ws-tdm path.
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)
        _use_lds_pf = (
            wave_specialized_tdm
            and (compute_schedule_kind == COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING or _pf_allow_quadrant)
            and not _bvs_active
        )

        # WMMA-driven prefetch PIPELINE (Stage C): real sub-depth with pf/remain TDM
        # split issued at WMMA phase boundaries (tdm_cb_pf @ i==_pf_D, tdm_cb_remain
        # @ tile end). Gated on PF_PIPELINE=1 (default off → legacy path). Requires
        # real sub-depth (D<A) with K-split active.
        _pf_pipeline = (
            _use_lds_pf
            and not _pf_full_depth
            and _pf_tdm_split
            and os.environ.get("PF_PIPELINE", "0") == "1"
        )
        _full_prefetch = _full_pf_req and _pf_pipeline
        # Overlapped prologue startup (full-prefetch with a main loop): drain tile 0
        # alone first, then issue the remaining N-1 tiles' TDMs and tile 0's carry
        # ds_load together so the ds_load overlaps the in-flight TDMs instead of
        # serializing after a single all-tiles drain.
        _pf_overlap_prologue = _full_prefetch and loop_iters > 0
        # Pipeline TDM-wait count. Each phase boundary issues 1 TDM then fences.
        # Full-prefetch: cb_pf + cb_rem each fence to 2*(N-1). After issuing 1 TDM
        # the FIFO has at most 2*(N-1)+1 entries; draining to 2*(N-1) retires the
        # oldest — which is exactly the same-type TDM from N-1 tiles ago (the one
        # the NEXT phase will read). Legacy: same analysis with N-1 look-ahead.
        # NOTE: cmodel treats TDM as synchronous — NOT cmodel-validatable.
        if const_expr(_full_prefetch):
            _pf_keep = 2 * (num_buffers - 1)
        else:
            _pf_keep = max(0, 2 * (num_buffers - 1) - 2)
        _pf_keep_cb_rem = _pf_keep
        if _pf_pipeline:
            _fence_outstanding = _pf_keep

        # Compute per-k-step prefetch flat-list layout (trace-time constants).
        if const_expr(_use_lds_pf):
            _n_pf_a_per_ks = wmma_m_rep
            _n_pf_b_per_ks = wmma_n_rep
            if const_expr(is_ptpc):
                _n_pf_bs_per_ks = 0
                _n_pf_as_per_ks = 0
            elif const_expr(use_scale_opsel):
                _n_pf_bs_per_ks = (b_scale_load_rep + 1) // 2
                _n_pf_as_per_ks = (wmma_m_rep + 1) // 2
            else:
                _n_pf_bs_per_ks = b_scale_load_rep
                _n_pf_as_per_ks = wmma_m_rep
            _n_pf_per_ks = _n_pf_a_per_ks + _n_pf_b_per_ks + _n_pf_bs_per_ks + _n_pf_as_per_ks
            # Full depth carries assembled frag-tuples per ks-step; sub-depth carries
            # the raw b128 LDS loads for the first _pf_D WMMAs (_pf_split_idx vectors).
            if const_expr(_pf_full_depth):
                _n_pf_total = k_wmma_steps * _n_pf_per_ks
            else:
                _n_pf_total = _pf_split_idx

            def _pf_all_ks_to_flat(pf_all_ks):
                if const_expr(not _pf_full_depth):
                    return list(pf_all_ks)  # already a flat list of raw b128 vectors
                flat = []
                for (a_frags, b_frags, b_scales, a_scales) in pf_all_ks:
                    flat.extend(a_frags)
                    flat.extend(b_frags)
                    if const_expr(not is_ptpc):
                        flat.extend(b_scales)
                        flat.extend(a_scales)
                return flat

            def _flat_to_pf_all_ks(state, base):
                if const_expr(not _pf_full_depth):
                    return list(state[base : base + _pf_split_idx])
                result = []
                idx = base
                for _ in range_constexpr(k_wmma_steps):
                    a_frags = list(state[idx : idx + _n_pf_a_per_ks])
                    idx += _n_pf_a_per_ks
                    b_frags = list(state[idx : idx + _n_pf_b_per_ks])
                    idx += _n_pf_b_per_ks
                    if const_expr(not is_ptpc):
                        b_scales = list(state[idx : idx + _n_pf_bs_per_ks])
                        idx += _n_pf_bs_per_ks
                        a_scales = list(state[idx : idx + _n_pf_as_per_ks])
                        idx += _n_pf_as_per_ks
                    else:
                        b_scales, a_scales = None, None
                    result.append((a_frags, b_frags, b_scales, a_scales))
                return result

            def _issue_pf_all_ks(lds_a, lds_b, lds_bs, lds_as):
                """Emit the carry's ds_loads. Sub-depth returns the raw b128 vectors
                (assembled lazily in compute_tile); full depth returns assembled
                per-ks frag-tuples."""
                _a_buf, _a_base = _precompute_a_lane_bases(lds_a)
                _b_buf, _b_base = _precompute_b_lane_bases(lds_b)
                _as_buf, _as_base = _precompute_scale_lane_bases(
                    lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a
                )
                _bs_buf, _bs_base = _precompute_scale_lane_bases(
                    lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b
                )
                _buf_pairs = [(_a_buf, _a_base), (_b_buf, _b_base), (_as_buf, _as_base), (_bs_buf, _bs_base)]

                if const_expr(not _pf_full_depth):
                    # Sub-depth: raw carry = first _pf_D WMMAs' table entries.
                    _raw = []
                    for _tag, _imm in _pf_load_table[:_pf_split_idx]:
                        _rbuf, _rbase = _buf_pairs[_tag]
                        _raw.append(fx.Vector(lds_load_b128_raw(_rbuf, _rbase, _imm)))
                    return _raw

                # Full depth: assembled per-ks frag-tuples (committed baseline path).
                pf = []
                for ks in range_constexpr(k_wmma_steps):
                    _b_ks, _bs_ks, _as_ks = _load_b_and_scales(
                        _b_buf, _b_base, _bs_buf, _bs_base, _as_buf, _as_base, ks
                    )
                    _a_ks = [load_a_frag(_a_buf, _a_base, wm, ks) for wm in range_constexpr(wmma_m_rep)]
                    pf.append((_a_ks, _b_ks, _bs_ks, _as_ks))
                return pf

        # Prologue
        if const_expr(wave_specialized_tdm):
            # Cold-start: warm the prologue B tiles into L2 BEFORE their TDM loads.
            # All waves participate (tx spans 0..block_threads-1), so the whole
            # B tile's cache lines are covered. Issued up-front so the HBM round
            # trip overlaps the prologue's TDM setup instead of being a serial
            # 1500cc cold miss on the first tensor_load.
            if const_expr(_effective_l2_pf > 0):
                for i in range_constexpr(pre_loaded):
                    _l2_prefetch_b_at(split_k_base + arith.index(i * tile_k))
            if const_expr(_pf_tdm_split):
                # K-split prologue: load pf segment + remain segment per stage.
                # (Step-1c validation: two partial-K TDMs must fill the buffer
                # identically to the single full-K TDM.)
                _seg_alo_rem_pl = _seg_alo_rem
                for i in range_constexpr(pre_loaded):
                    _issue_active_tdm_seg(i, active_addr_lo, 0)
                    _issue_active_tdm_seg(i, _seg_alo_rem_pl, 1)
                    active_addr_lo = active_addr_lo + active_adv_i32
                    _seg_alo_rem_pl = _seg_alo_rem_pl + active_adv_i32
                    # Overlapped startup: drain tile 0 alone right after issuing it so
                    # its carry ds_load (below) overlaps tiles 1..N-1's TDMs.
                    if const_expr(_pf_overlap_prologue and i == 0):
                        _pipeline_fence(outstanding=0)
            else:
                for i in range_constexpr(pre_loaded):
                    _issue_active_tdm(i, active_addr_lo)
                    # Advance right after TDM: s_add_co_i32 runs during the
                    # subsequent compute window, not on the next TDM's critical path.
                    active_addr_lo = active_addr_lo + active_adv_i32
        else:
            for i in range_constexpr(pre_loaded):
                dg0_a = _pack_dg0(pred_const, stages_a_lds_addr[i], addr_lo_a, addr_hi_a)
                dg0_b = _pack_dg0(pred_const, stages_b_lds_addr[i], addr_lo_b, addr_hi_b)
                dg0_as = _pack_dg0(pred_const, stages_as_lds_addr[i], addr_lo_as, addr_hi_as)
                dg0_bs = _pack_dg0(pred_const, stages_bs_lds_addr[i], addr_lo_bs, addr_hi_bs)
                issue_tdm_loads(
                    tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                    tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                    tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                    tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                    wave_specialized=wave_specialized_tdm,
                    cache_policies=_tdm_cache_policies,
                    enabled=_tdm_enabled,
                )

                addr_lo_a = addr_lo_a + adv_a_i32
                addr_lo_b = addr_lo_b + adv_b_i32
                addr_lo_as = addr_lo_as + adv_as_i32
                addr_lo_bs = addr_lo_bs + adv_bs_i32

        if const_expr(_bvs_active):
            # Prologue: prefetch the first _bvs_D K-tiles (global->VGPR). Carried as
            # FLAT lists of i32 (list-of-tuples can't be loop-carried).
            _bvs_pf = [_bvs_prefetch(split_k_base + arith.index(_d * tile_k)) for _d in range(_bvs_D)]
            _bvs_ra = [_v for (_a, _b) in _bvs_pf for _v in _a]
            _bvs_rb = [_v for (_a, _b) in _bvs_pf for _v in _b]

        _tail_pf_all_ks = None  # set from main-loop results if _use_lds_pf

        # WMMA-driven prefetch PIPELINE: pf/remain TDM base addresses carried from the
        # prologue (which primed tile[pre_loaded-1]) through the main loop into the tail.
        # Mutable boxes so the boundary callbacks (fired inside compute_tile) can advance
        # them. Both step by active_adv_i32 per tile.
        if const_expr(_pf_pipeline):
            _pl_addr_pf_box = [active_addr_lo]
            _pl_addr_rem_box = [_seg_alo_rem_pl]

        # Pipeline prologue: DRAIN all pre-loaded TDMs (tensorcnt 0) + barrier, so the
        # first (num_buffers-1) tiles are consumed from a fully-ready/visible state. The
        # steady-state in-flight count then builds up to _pf_keep over the first loop
        # tiles (matches the user's deep-pipeline prologue: prefetch-all then tensorcnt 0).
        if const_expr(_pf_overlap_prologue):
            # tile 0 already drained + barriered in the prologue. Issue its carry
            # ds_load NOW (overlaps tiles 1..N-1's in-flight TDMs), then drain those
            # TDMs and finally the carry ds_load.
            _pf_init = _issue_pf_all_ks(
                stages_a_idx[0], stages_b_idx[0], stages_bs_idx[0], stages_as_idx[0]
            )
            _pf_init_flat = _pf_all_ks_to_flat(_pf_init)
            _pipeline_fence(outstanding=0)
            rocdl.s_wait_dscnt(0)
        elif const_expr(_pf_pipeline):
            _pipeline_fence(outstanding=0)
        else:
            _pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

        if const_expr(loop_iters > 0 and use_ws_tdm_split_signal_overlap and not _full_prefetch):
            _pipeline_fence_signal(outstanding=_fence_outstanding)

        if const_expr(loop_iters > 0):
            if const_expr(wave_specialized_tdm):
                if const_expr(_pf_overlap_prologue):
                    pass  # _pf_init / _pf_init_flat already computed above
                elif const_expr(_use_lds_pf):
                    _pf_init = _issue_pf_all_ks(
                        stages_a_idx[0], stages_b_idx[0], stages_bs_idx[0], stages_as_idx[0]
                    )
                    from flydsl.expr.rocdl import tdm_ops as _pf_tdm_main
                    _pf_tdm_main.tensor_wait(0)
                    rocdl.s_wait_dscnt(0)
                    _pf_init_flat = _pf_all_ks_to_flat(_pf_init)
                else:
                    _pf_init_flat = []

                # Pipeline carries TWO TDM addresses (pf + remain); legacy carries one.
                if const_expr(_pf_pipeline):
                    init_args = list(accs) + [_pl_addr_pf_box[0], _pl_addr_rem_box[0]] + _pf_init_flat
                else:
                    init_args = list(accs) + [active_addr_lo] + _pf_init_flat
                if const_expr(_bvs_active):
                    init_args = init_args + _bvs_ra + _bvs_rb

                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    if const_expr(_pf_pipeline):
                        _addr_pf_box = [state[n_accs]]
                        _addr_rem_box = [state[n_accs + 1]]
                        _naddr = 2
                    else:
                        cur_addr_lo = state[n_accs]
                        _naddr = 1

                    if const_expr(_use_lds_pf):
                        _pf_base = n_accs + _naddr
                        _cur_pf_all_ks = _flat_to_pf_all_ks(state, _pf_base)
                        _pf_end = _pf_base + _n_pf_total
                    else:
                        _cur_pf_all_ks = None
                        _pf_end = n_accs + _naddr

                    if const_expr(_bvs_active):
                        _ra0 = _pf_end
                        _ring_a = list(state[_ra0 : _ra0 + _bvs_D * wmma_m_rep])
                        _rb0 = _ra0 + _bvs_D * wmma_m_rep
                        _ring_b = list(state[_rb0 : _rb0 + _bvs_D * b_scale_load_rep])

                    for buf_idx in range_constexpr(num_buffers):
                        if const_expr(_full_prefetch):
                            load_stage = buf_idx
                        else:
                            load_stage = (buf_idx + num_buffers - 1) % num_buffers
                        next_buf = (buf_idx + 1) % num_buffers

                        _tdm_cb_pf = None
                        _tdm_cb_rem = None
                        if const_expr(not _pf_pipeline):
                            # TDM issued at START of each tile (before fence) for maximum
                            # DMA overlap time.  addr_lo is pre-computed (advanced from the
                            # PREVIOUS tile's post-TDM window), so no s_delay_alu needed.
                            _k_off_tdm = (
                                split_k_base
                                + loop_iter * arith.index(num_buffers * tile_k)
                                + arith.index(buf_idx * tile_k)
                            )
                            _issue_active_tdm(load_stage, cur_addr_lo, k_prefetch=_k_off_tdm)
                            # Advance addr right after TDM: s_add_co_i32 runs during the
                            # fence + barrier window, arriving long before the next TDM call.
                            cur_addr_lo = cur_addr_lo + active_adv_i32
                        else:
                            # WMMA-driven pipeline: issue the DEEP-prefetch tile's pf/remain
                            # seg TDMs at the WMMA phase boundaries (fired inside
                            # compute_tile). The prefetch target is the just-freed buffer
                            # (load_stage = num_buffers-1 ahead), same as the non-pipeline
                            # path -- NOT next_buf (1 ahead), which would clobber an in-use
                            # buffer for num_buffers>2. _addr_pf/rem_box already track the
                            # (n-1)-ahead tile's global address (primed past the prologue's
                            # pre_loaded tiles, advanced per tile).
                            def _tdm_cb_pf(_nb=load_stage):
                                _issue_active_tdm_seg(_nb, _addr_pf_box[0], 0)
                                _addr_pf_box[0] = _addr_pf_box[0] + active_adv_i32
                                _pipeline_fence(outstanding=_pf_keep)

                            def _tdm_cb_rem(_nb=load_stage, _bi=buf_idx):
                                _issue_active_tdm_seg(_nb, _addr_rem_box[0], 1)
                                _addr_rem_box[0] = _addr_rem_box[0] + active_adv_i32
                                if const_expr(_full_prefetch):
                                    _pipeline_fence(outstanding=_pf_keep_cb_rem)
                                elif const_expr(_bi == num_buffers - 1):
                                    _pipeline_fence(outstanding=_pf_keep_cb_rem)

                        if const_expr(not use_ws_tdm_split_signal_overlap):
                            _pipeline_fence_signal(outstanding=_fence_outstanding)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        _late_tdm_ws_fence_signal = None
                        if const_expr(use_ws_tdm_split_signal_overlap and not _full_prefetch):

                            def _late_tdm_ws_split_signal():
                                _pipeline_fence_signal(outstanding=_fence_outstanding)

                            _late_tdm_ws_fence_signal = _late_tdm_ws_split_signal

                        a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[buf_idx])
                        rocdl.sched_barrier(0)
                        if const_expr(_bvs_active):
                            _cur_a = _ring_a[:wmma_m_rep]
                            _cur_b = _ring_b[:b_scale_load_rep]
                            _next_kb = (
                                split_k_base
                                + loop_iter * arith.index(num_buffers * tile_k)
                                + arith.index((buf_idx + _bvs_D) * tile_k)
                            )
                            _na, _nb2 = _bvs_prefetch(_next_kb)
                            _ring_a = _ring_a[wmma_m_rep:] + list(_na)
                            _ring_b = _ring_b[b_scale_load_rep:] + list(_nb2)
                        else:
                            _cur_a = None
                            _cur_b = None

                        if const_expr(_use_lds_pf):
                            # Pass current tile's pre-fetched operands and next tile's LDS.
                            # compute_tile emits next-tile ds_loads interleaved with current
                            # WMMAs, then returns nxt_pf for the next loop-carry.
                            compute_result = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                late_compute_callback=_late_tdm_ws_fence_signal,
                                a0_prefetch=a0_prefetch,
                                pf_a_scales=_cur_a,
                                pf_b_scales=_cur_b,
                                pf_all_ks=_cur_pf_all_ks,
                                next_lds_a=stages_a_idx[next_buf],
                                next_lds_b=stages_b_idx[next_buf],
                                next_lds_bs=stages_bs_idx[next_buf],
                                next_lds_as=stages_as_idx[next_buf],
                                tdm_cb_pf=_tdm_cb_pf,
                                tdm_cb_remain=_tdm_cb_rem,
                            )
                            accs_in = compute_result[0]
                            _cur_pf_all_ks = compute_result[1]
                        else:
                            accs_in = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                mid_compute_callback=None,
                                late_compute_callback=_late_tdm_ws_fence_signal,
                                a0_prefetch=a0_prefetch,
                                pf_a_scales=_cur_a,
                                pf_b_scales=_cur_b,
                            )

                        hot_loop_scheduler_scheduled()

                    if const_expr(_use_lds_pf):
                        _pf_yield = _pf_all_ks_to_flat(_cur_pf_all_ks)
                    else:
                        _pf_yield = []
                    if const_expr(_bvs_active):
                        _bvs_yield = _ring_a + _ring_b
                    else:
                        _bvs_yield = []
                    if const_expr(_pf_pipeline):
                        _addr_yield = [_addr_pf_box[0], _addr_rem_box[0]]
                    else:
                        _addr_yield = [cur_addr_lo]
                    results = yield list(accs_in) + _addr_yield + _pf_yield + _bvs_yield

                accs = list(results[:n_accs])
                if const_expr(_pf_pipeline):
                    # Hand the advanced pf/remain addresses to the tail (via the boxes).
                    _pl_addr_pf_box[0] = results[n_accs]
                    _pl_addr_rem_box[0] = results[n_accs + 1]
                    active_addr_lo = results[n_accs]
                    _pf_base = n_accs + 2
                else:
                    active_addr_lo = results[n_accs]
                    _pf_base = n_accs + 1
                # Extract pf_all_ks for the first tail tile (loop-carry from last iter).
                if const_expr(_use_lds_pf):
                    _tail_pf_all_ks = _flat_to_pf_all_ks(results, _pf_base)
                else:
                    _tail_pf_all_ks = None
            else:
                init_args = list(accs) + [addr_lo_a, addr_lo_b, addr_lo_as, addr_lo_bs]

                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_lo_a = state[n_accs]
                    cur_lo_b = state[n_accs + 1]
                    cur_lo_as = state[n_accs + 2]
                    cur_lo_bs = state[n_accs + 3]

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers

                        _pipeline_fence_signal(outstanding=_fence_outstanding)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        addr_boxes = [[cur_lo_a], [cur_lo_b], [cur_lo_as], [cur_lo_bs]]

                        def _mid_tdm_nws(
                            _ls=load_stage,
                            _ab=addr_boxes,
                            _k_off=(
                                split_k_base
                                + loop_iter * arith.index(num_buffers * tile_k)
                                + arith.index(buf_idx * tile_k)
                            ),
                        ):
                            dg0_a = _pack_dg0(pred_const, stages_a_lds_addr[_ls], _ab[0][0], addr_hi_a)
                            dg0_b = _pack_dg0(pred_const, stages_b_lds_addr[_ls], _ab[1][0], addr_hi_b)
                            dg0_as = _pack_dg0(pred_const, stages_as_lds_addr[_ls], _ab[2][0], addr_hi_as)
                            dg0_bs = _pack_dg0(pred_const, stages_bs_lds_addr[_ls], _ab[3][0], addr_hi_bs)
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                                tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                                wave_specialized=wave_specialized_tdm,
                            )
                            _ab[0][0] = _ab[0][0] + adv_a_i32
                            _ab[1][0] = _ab[1][0] + adv_b_i32
                            _ab[2][0] = _ab[2][0] + adv_as_i32
                            _ab[3][0] = _ab[3][0] + adv_bs_i32
                            _l2_prefetch(_k_off)

                        a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[buf_idx])
                        rocdl.sched_barrier(0)
                        accs_in = compute_tile_scheduled(
                            accs_in,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            stages_as_idx[buf_idx],
                            stages_bs_idx[buf_idx],
                            mid_compute_callback=_mid_tdm_nws,
                            a0_prefetch=a0_prefetch,
                        )
                        cur_lo_a = addr_boxes[0][0]
                        cur_lo_b = addr_boxes[1][0]
                        cur_lo_as = addr_boxes[2][0]
                        cur_lo_bs = addr_boxes[3][0]
                        hot_loop_scheduler_scheduled()

                    results = yield list(accs_in) + [cur_lo_a, cur_lo_b, cur_lo_as, cur_lo_bs]

                accs = list(results[:n_accs])
                addr_lo_a = results[n_accs]
                addr_lo_b = results[n_accs + 1]
                addr_lo_as = results[n_accs + 2]
                addr_lo_bs = results[n_accs + 3]

        # When loop_iters == 0 (few K-tiles, e.g. large tile_k) the main loop never
        # runs, so _tail_pf_all_ks is never seeded and the whole GEMM falls back to
        # the non-prefetched tail (every WMMA preceded by s_wait_dscnt(0)). Seed it
        # here exactly like the main-loop prologue: drain prologue TDMs, then issue
        # the first compute stage's ds_loads so the tail's interleaved-prefetch path
        # (_use_tail_pf) carries operands across tail tiles. Tail plan entry 0 always
        # has compute_stage == 0 (see make_tail_plan), which the prologue preloaded.
        if const_expr(loop_iters == 0 and _use_lds_pf):
            _tail_pf_all_ks = _issue_pf_all_ks(
                stages_a_idx[0], stages_b_idx[0], stages_bs_idx[0], stages_as_idx[0]
            )
            from flydsl.expr.rocdl import tdm_ops as _pf_tdm_tail
            _pf_tdm_tail.tensor_wait(0)
            rocdl.s_wait_dscnt(0)

        # Tail — same acc_mixed pattern: fence at top, TDM mid-compute.
        # if const_expr(loop_iters > 0 and use_ws_tdm_split_signal_overlap):
        #     pipeline_fence_wait(use_cluster=use_cluster)
        # if const_expr(loop_iters > 0):
        #     _pipeline_fence(outstanding=0)
        # elif const_expr(use_cluster):
        #     cluster.cluster_barrier()
        epi_addrs_box = [None]
        _ptpc_scale_box = [None]

        def _load_ptpc_scales_once():
            if const_expr(is_ptpc and _ptpc_scale_box[0] is None):
                _ptpc_scale_box[0] = epilogue_load_ptpc_scales()

        _tail_had_load = False
        # Tail K-tile index, so the VGPR-path scale buffer_load uses the right k_base.
        _bvs_tail_kt = [loop_iters * num_buffers]

        def _bvs_tail_kb():
            if const_expr(not _bvs_active):
                return None
            kb = split_k_base + arith.index(_bvs_tail_kt[0] * tile_k)
            _bvs_tail_kt[0] += 1
            return kb

        # Pre-expand tail_plan: augment each entry with the NEXT active compute stage
        # so we can do pf chaining without dict lookups inside the traced for loop.
        _tail_plan_ext = []  # (load_stage, compute_stage, outstanding, next_compute_stage)
        _tail_plan_active_cs = [cs for ls, cs, out in tail_plan]
        for _tei, (ls, cs, out) in enumerate(tail_plan):
            _next_cs = None
            if out != -1:
                _apos = _tail_plan_active_cs.index(cs)
                if _apos + 1 < len(_tail_plan_active_cs):
                    _next_cs = _tail_plan_active_cs[_apos + 1]
            _tail_plan_ext.append((ls, cs, out, _next_cs))

        # Tail TDM drain countdown (split pipeline). At the main->tail boundary the
        # last cb_rem fence leaves at most _pf_keep_cb_rem TDMs in flight. The tail
        # issues no new deep lookahead, so the in-flight bound must wind DOWN to 0
        # across the tail — one step per cb fence (cb_pf, cb_rem).
        # The final tile forces 0 (every TDM must complete before the epilogue).
        # Trace-time counter (the tail loop is fully unrolled).
        _tail_tcnt_box = [_pf_keep_cb_rem]

        def _tail_drain_next():
            _tail_tcnt_box[0] = max(0, _tail_tcnt_box[0] - 1)
            return _tail_tcnt_box[0]

        for _load_stage, _compute_stage, _outstanding, _tail_next_cs in _tail_plan_ext:
            _entry_kb = _bvs_tail_kb()
            if const_expr(_pf_pipeline):
                # ── WMMA-driven prefetch PIPELINE tail ──────────────────────────────
                # No upfront full TDM. The pf/remain TDM for the NEXT tile is issued by
                # the boundary callbacks fired inside compute_tile:
                #   cb_pf  (i==_pf_D): tensor_wait(0) [this tile's remain TDM] + issue
                #                      next tile's pf TDM into buffer _tail_next_cs
                #   cb_rem (tile end): tensor_wait(0) [next tile's pf TDM] + issue
                #                      next tile's remain TDM into buffer _tail_next_cs
                # Final tile (no next): cb_pf only waits its own remain TDM, issues none.
                # Two DISTINCT buffers (decoupled, like the non-pipeline tail):
                #  - _pl_next  = _tail_next_cs : NEXT compute stage -> nxt_raw carry read.
                #  - _pl_load  = _load_stage   : tail-plan's load buffer -> cb TDM fill.
                # For num_buffers==2 these coincide; for >2 they differ and conflating them
                # (old bug) made the cb clobber a pre-loaded tile -> NaN.
                _pl_next = _tail_next_cs
                _pl_has_next = _tail_next_cs is not None
                _pl_load = _load_stage
                _pl_has_load = _load_stage is not None

                def _cb_pf(_n=_pl_load, _has=_pl_has_load, _final=not _pl_has_next):
                    if const_expr(_has):
                        _issue_active_tdm_seg(_n, _pl_addr_pf_box[0], 0)
                        _pl_addr_pf_box[0] = _pl_addr_pf_box[0] + active_adv_i32
                    # wait+barrier (bound = _pipeline_fence) makes the drained TDMs
                    # cross-wave visible. Wind the in-flight bound down by 1 per cb fence;
                    # the final tile forces 0 so all TDMs complete before the epilogue.
                    _pipeline_fence(outstanding=0 if const_expr(_final) else _tail_drain_next())

                def _cb_rem(_n=_pl_load, _has=_pl_has_load):
                    if const_expr(_has):
                        _issue_active_tdm_seg(_n, _pl_addr_rem_box[0], 1)
                        _pl_addr_rem_box[0] = _pl_addr_rem_box[0] + active_adv_i32
                    _pipeline_fence(outstanding=_tail_drain_next())

                a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[_compute_stage])
                rocdl.sched_barrier(0)
                if const_expr(_pl_has_next):
                    compute_result = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_all_ks=_tail_pf_all_ks,
                        next_lds_a=stages_a_idx[_pl_next],
                        next_lds_b=stages_b_idx[_pl_next],
                        next_lds_bs=stages_bs_idx[_pl_next],
                        next_lds_as=stages_as_idx[_pl_next],
                        tdm_cb_pf=_cb_pf,
                        tdm_cb_remain=_cb_rem,
                    )
                    accs = compute_result[0]
                    _tail_pf_all_ks = compute_result[1]
                else:
                    # Final tile: carry + own remain, no next tile. cb_pf waits the
                    # final tile's own remain TDM (issued at the prev tile's cb_rem).
                    if const_expr(not use_tdm_store):
                        def _emit_epi_addrs_pl():
                            epi_addrs_box[0] = epilogue_prepare_addrs()
                            _load_ptpc_scales_once()
                        _pl_filler = _emit_epi_addrs_pl
                    else:
                        _pl_filler = _load_ptpc_scales_once if is_ptpc else None
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        emit_filler=_pl_filler,
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_all_ks=_tail_pf_all_ks,
                        tdm_cb_pf=_cb_pf,
                    )
                hot_loop_scheduler_scheduled()
            elif const_expr(_outstanding == -1):
                # The final tail tile has no next tile to overlap, but if a prior
                # tile pre-fetched this tile's operands (chained via _tail_pf_all_ks),
                # pass them through so compute_tile uses skip_wait partial drains
                # instead of a serial s_wait_dscnt(0) before every WMMA.
                _last_pf = _tail_pf_all_ks if (_use_lds_pf and _tail_pf_all_ks is not None) else None
                if const_expr(use_tdm_store):
                    a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[_compute_stage])
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        emit_filler=(_load_ptpc_scales_once if is_ptpc else None),
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_all_ks=_last_pf,
                    )
                else:

                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()
                        _load_ptpc_scales_once()

                    a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[_compute_stage])
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        emit_filler=_emit_epi_addrs,
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_all_ks=_last_pf,
                    )
            else:
                # Issue TDM BEFORE fence (same as main loop) so the DMA starts
                # early and the addr advancement happens in the fence window.
                if const_expr(_load_stage is not None):
                    _tail_had_load = True
                    if const_expr(wave_specialized_tdm):
                        _issue_active_tdm(_load_stage, active_addr_lo)
                        active_addr_lo = active_addr_lo + active_adv_i32
                        # Pin TDM before the fence scheduling region so LLVM cannot
                        # sink tensor_load_to_lds to after the barrier.
                        rocdl.sched_barrier(0)
                    else:
                        dg0_a = _pack_dg0(pred_const, stages_a_lds_addr[_load_stage], addr_lo_a, addr_hi_a)
                        dg0_b = _pack_dg0(pred_const, stages_b_lds_addr[_load_stage], addr_lo_b, addr_hi_b)
                        dg0_as = _pack_dg0(pred_const, stages_as_lds_addr[_load_stage], addr_lo_as, addr_hi_as)
                        dg0_bs = _pack_dg0(pred_const, stages_bs_lds_addr[_load_stage], addr_lo_bs, addr_hi_bs)
                        issue_tdm_loads(
                            tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                            tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                            tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                            tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                            wave_specialized=wave_specialized_tdm,
                        )
                        addr_lo_a = addr_lo_a + adv_a_i32
                        addr_lo_b = addr_lo_b + adv_b_i32
                        addr_lo_as = addr_lo_as + adv_as_i32
                        addr_lo_bs = addr_lo_bs + adv_bs_i32
                    # When a TDM was just issued, use at least TDM_LOADS_PER_STEP as
                    # the fence outstanding so the just-issued TDM stays in-flight
                    # during compute (avoids s_wait_tensorcnt(0) immediately after issue).
                    # DEBUG PF_TAIL_WAIT=1: force wait-all (0) when sub-depth reads the
                    # just-issued buffer this tile (probes whether cmodel models TDM latency).
                    if const_expr(
                        os.environ.get("PF_TAIL_WAIT", "0") == "1"
                        and _use_lds_pf and not _pf_full_depth
                        and _tail_next_cs is not None and _load_stage == _tail_next_cs
                    ):
                        _tail_fence_out = 0
                    else:
                        _tail_fence_out = max(_outstanding, TDM_LOADS_PER_STEP)
                else:
                    _tail_fence_out = _outstanding

                _pipeline_fence_signal(outstanding=_tail_fence_out)
                pipeline_fence_wait(use_cluster=use_cluster)

                a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[_compute_stage])
                rocdl.sched_barrier(0)

                _use_tail_pf = _use_lds_pf and _tail_pf_all_ks is not None
                _has_tail_next = _tail_next_cs is not None

                if _use_tail_pf and _has_tail_next:
                    compute_result = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_all_ks=_tail_pf_all_ks,
                        next_lds_a=stages_a_idx[_tail_next_cs],
                        next_lds_b=stages_b_idx[_tail_next_cs],
                        next_lds_bs=stages_bs_idx[_tail_next_cs],
                        next_lds_as=stages_as_idx[_tail_next_cs],
                    )
                    accs = compute_result[0]
                    _tail_pf_all_ks = compute_result[1]
                elif _use_tail_pf:
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_all_ks=_tail_pf_all_ks,
                    )
                    _tail_pf_all_ks = None
                else:
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                    )

                hot_loop_scheduler_scheduled()

        accs = finalize_acc_layout(accs)

        if const_expr(is_ptpc):
            _load_ptpc_scales_once()
            _ptpc_sa, _ptpc_sb = _ptpc_scale_box[0]
            accs = epilogue_apply_ptpc_scale(accs, _ptpc_sa, _ptpc_sb)

        def _emit_tdm_store():
            if const_expr(d_need_epilogue_fence):
                _pipeline_fence(outstanding=0)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)

        def _emit_buffer_store():
            rocdl.sched_barrier(0)
            if const_expr(epi_addrs_box[0] is None):
                epi_addrs_box[0] = epilogue_prepare_addrs()
            if const_expr(split_k > 1):
                epilogue_atomic_adds(accs, epi_addrs_box[0])
            else:
                epilogue_stores(accs, epi_addrs_box[0])

        if const_expr(use_tdm_store):
            full_tile = (blk_m + arith.index(tile_m)) <= m_idx
            if_op = scf.IfOp(full_tile, [], has_else=True)
            with ir.InsertionPoint(if_op.then_block):
                _emit_tdm_store()
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                _emit_buffer_store()
                scf.YieldOp([])
        else:
            _emit_buffer_store()

    cache_tag = (
        data_format,
        scale_mode,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        compute_schedule_kind,
        effective_waves_per_eu,
        l2_prefetch_distance,
        cluster_m,
        cluster_n,
        use_tdm_store,
        out_dtype,
        inst_prefetch,
        wave_specialized_tdm,
        split_k,
        use_scale_opsel,
        expert_sched_mode,
        atomic_barrier_enable,
        b_streaming,
        scale_load_path,
        fp8_schedule,
    )

    @flyc.jit
    def launch_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
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

        if const_expr(use_cluster):
            # Cluster launch needs a cluster-divisible grid
            gx = ((gx + (cluster_m - 1)) // cluster_m) * cluster_m

        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        kernel_mxscale_gemm(
            arg_c,
            arg_a,
            arg_b,
            arg_a_scale,
            arg_b_scale,
            i32_m,
            i32_n,
            i32_lda,
            i32_ldc,
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

    if effective_expert_sched_mode:
        launch_mxscale_gemm.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_mxscale_gemm


def compile_mxscale_gemm(**kw):
    """Backward-compatible wrapper: MX block-scale (E8M0) GEMM."""
    return compile_fp8fp4_gemm(scale_mode="mxscale", **kw)


def compile_mxfp4_gemm(**kw):
    return compile_fp8fp4_gemm(data_format="fp4", scale_mode="mxscale", **kw)


def compile_mxfp8_gemm(**kw):
    return compile_fp8fp4_gemm(data_format="fp8", scale_mode="mxscale", **kw)


def compile_a8w4_gemm(**kw):
    return compile_fp8fp4_gemm(data_format="a8w4", scale_mode="mxscale", **kw)


def compile_ptpc_gemm(
    *,
    N: int = 0,
    K: int,
    data_format: str = "fp8",
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 4,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
    cluster_m: int = 1,
    cluster_n: int = 1,
    out_dtype: str = "bf16",
    inst_prefetch: bool = False,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
    split_k: int = 1,
):
    """Compile a PTPC (per-token per-channel) GEMM kernel.

    A scale is per-token (sa[M], fp32), B scale is per-channel (sb[N], fp32),
    both constant along K. The K-loop runs the WMMA unscaled (FP8) or with an
    identity E8M0 scale (A8W4, which has no non-scale op); sa*sb is applied in
    the epilogue in fp32. split_k>1 is supported (atomic add path).

    data_format: "fp8" (FP8 act + FP8 weight) or "a8w4" (FP8 act + FP4 weight).
    wave_specialized_tdm=True requires m_warp*n_warp >= 2.
    """
    return compile_fp8fp4_gemm(
        data_format=data_format,
        scale_mode="ptpc",
        b_streaming=False,
        wave_specialized_tdm=True,
        use_scale_opsel=False,
        fp8_schedule="auto",
        scale_load_path="tdm",
        use_tdm_store=(split_k == 1),
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        out_dtype=out_dtype,
        inst_prefetch=inst_prefetch,
        expert_sched_mode=expert_sched_mode,
        atomic_barrier_enable=atomic_barrier_enable,
        split_k=split_k,
    )


__all__ = [
    "compile_fp8fp4_gemm",
    "compile_mxscale_gemm",
    "compile_mxfp4_gemm",
    "compile_mxfp8_gemm",
    "compile_a8w4_gemm",
    "compile_ptpc_gemm",
]
