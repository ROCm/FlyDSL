"""Batched WMMA GEMM a16w8 kernel for gfx1250.

C[m,b,n] = A[m,b,k] @ B_dequant[b,n,k]^T
  A: bf16 (activations) in [M, B, K] layout (M-major, B=groups middle, K inner)
  B: fp8_e4m3fn weights in [B, N, K] layout (K-inner, wo_a format)
  C: bf16 in [M, B, N] layout

Scale modes (compile-time):
  no_scale=True       fp8→bf16 via V_CVT_SCALE_PK8_BF16_FP8 with E8M0=127 (=1.0 constant).
                      arg_scale is unused; pass any 1-element tensor as placeholder.
  use_e8m0_scale=True fp8→bf16 with per-block uint8 E8M0 from arg_scale (O2 path).
  default             fp8→bf16 via cvt_pk_f32_fp8 + residual V_PK_MUL_BF16 (fp32 scale).

DeepSeek-V4 use case: einsum("sgd,grd->sgr", o, wo_a)
  B=G=16, M=T (runtime), K=D=4096, N=R=1024
  A (o) layout:    [T, G, D] = [M, B, K] with K innermost
  B (wo_a) layout: [G, R, D] = [B, N, K] with K innermost (unchanged)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity

from flydsl.expr import idx2crd
from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx, get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
    store_acc_vec8_to_buffer, store_acc_vec8_to_lds,
)


def _lds_ptr_raw(lds_base_idx, byte_offset):
    """Materialize an LLVM LDS (addrspace 3) pointer from a byte-base index."""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.arith import ArithValue as _AV
    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    total_byte = _AV(lds_base_idx) + byte_offset
    addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
    return _llvm.inttoptr(lds_ptr_ty, addr_i32)
from kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32
DS_LOADS_PER_A_FRAG = 2

LDS_PAD_A = 10  # stride=138 elem=276B=69DW; GCD(69,32)=1 (no bank conflict); was 12 (2-way), 8 (4-way)
LDS_PAD_B_FP8 = 12  # stride=140 fp8=35DW; GCD(35,32)=1 (no bank conflict); was 8 (2-way)
LDS_PAD_D_BYTES = 16

ELEM_BYTES_A = 2    # bf16
ELEM_BYTES_B = 1    # fp8
ELEM_BYTES_SCALE = 4  # fp32

_make_tail_plan = make_tail_plan


def compile_bmm_a16w8_gfx1250(
    *,
    B: int = 16,
    M: int = 0,
    N: int = 1024,
    K: int = 4096,
    group_k: int = 128,
    group_n: int = 128,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    out_dtype: str = "bf16",
    num_buffers: int = 3,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    use_tdm_store: bool = True,
    expert_sched_mode: bool = True,
    inst_prefetch: bool = False,
    use_e8m0_scale: bool = False,
    no_scale: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
):
    """Compile an a16w8 (bf16 act, fp8 weight, per-block scale) batched GEMM.

    use_e8m0_scale=True: arg_scale is uint8 E8M0 (1 byte/block) instead of fp32 (4 bytes/block).
    Eliminates 4×V_PK_MUL_BF16 per 8 fp8 elements; scale must be pre-rounded to nearest 2^n.

    Returns a JitFunction: launch_fn(arg_c, arg_a, arg_b, arg_scale, M, stream)

    Tensors (all flat 1-D):
      arg_a:     A[M, B, K]               bf16  (M-major: tokens × groups × K)
      arg_b:     B[B, N, K]               fp8_e4m3 (K-inner, wo_a format)
      arg_c:     C[M, B, N]               bf16 or f32  (M-major: tokens × groups × N)
      arg_scale: scale[B, K//gk, N//gn]   fp32

    Constraints:
      tile_k == group_k (one K-block per K-tile → one scale per tile)
      tile_n == group_n (one N-block per N-tile → one scale per N column)
    """
    _ = M
    if B < 1:
        raise ValueError(f"B must be >= 1, got {B}")
    if N <= 0 or K <= 0:
        raise ValueError(f"N and K must be positive, got N={N}, K={K}")
    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3 or 4, got {num_buffers}")
    if out_dtype not in ("f32", "f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f32', 'f16', or 'bf16', got {out_dtype!r}")

    if tile_k != group_k:
        raise ValueError(
            f"tile_k ({tile_k}) must equal group_k ({group_k}) for simple scale indexing")
    if tile_n != group_n:
        raise ValueError(
            f"tile_n ({tile_n}) must equal group_n ({group_n}) for simple scale indexing")
    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if N % tile_n != 0:
        raise ValueError(f"N must be divisible by tile_n={tile_n}, got N={N}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2, got {tile_k}")
    if use_e8m0_scale and no_scale:
        raise ValueError("use_e8m0_scale and no_scale are mutually exclusive")
    if cluster_m < 1 or cluster_n < 1:
        raise ValueError(f"cluster dims must be >= 1, got ({cluster_m}, {cluster_n})")
    if cluster_m * cluster_n > 16:
        raise ValueError(
            f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}={cluster_m * cluster_n}")

    _effective_e8m0 = use_e8m0_scale or no_scale
    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster and waves_per_eu is None:
        waves_per_eu = 1  # prevent cluster barrier deadlock with multiple waves
    _effective_l2_pf = max(1, l2_prefetch_distance - 1) if use_cluster else l2_prefetch_distance

    elem_bytes_d = 2 if out_dtype in ("f16", "bf16") else 4
    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE

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

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_bf16
    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep

    n_k_blocks = K // group_k    # scale K dimension
    n_n_blocks = N // group_n    # scale N dimension

    # A LDS layout (bf16)
    lds_a_stride = tile_k + LDS_PAD_A       # in bf16 elements
    lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    lds_a_bytes = lds_a_elems * ELEM_BYTES_A

    # B LDS layout (fp8 — [N,K] K-inner; each N-row is K+pad fp8 elements wide)
    lds_b_stride_fp8 = tile_k + LDS_PAD_B_FP8   # K+pad per N-row, in fp8 elements
    lds_b_elems_fp8 = tile_n * lds_b_stride_fp8 + LDS_PAD_B_FP8
    lds_b_bytes = lds_b_elems_fp8 * ELEM_BYTES_B

    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    # Per-stage layout: A (bf16) + B (fp8)
    stage_layout = SmemAllocator(None, arch=gpu_arch,
                                  global_sym_name="bmm_a16w8_tdm_layout")
    stage_b_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_rel_off + lds_b_bytes
    stage_a_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_rel_off + lds_a_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = _make_tail_plan(num_buffers, pre_loaded, extra)
    _last_compute_stage = _base_tail_plan[-1][1]
    tail_plan = _base_tail_plan

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None, arch=gpu_arch,
        global_sym_name=(
            f"bmm_a16w8_B{B}_K{K}_N{N}_{tile_m}x{tile_n}x{tile_k}_"
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

    stage_b_offsets = [stage_base_off[i] + stage_b_rel_off for i in range(num_buffers)]
    stage_a_offsets = [stage_base_off[i] + stage_a_rel_off for i in range(num_buffers)]

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

    _half_out = out_dtype in ("f16", "bf16")
    gy_compile = N // tile_n

    @flyc.kernel
    def kernel_bmm_a16w8_gfx1250(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale: fx.Tensor,
        i32_m: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = gpu.block_id("z")  # batch index

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        # --- Cluster MCAST setup ---
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
            fx.get(thr_coord, 0), fx.get(thr_coord, 1),
            fx.get(thr_coord, 2), fx.get(thr_coord, 3))

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        # --- Batch offsets ---
        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        # A is [M, B, K]: batch bz shifts the K inner dimension by bz*K elements
        a_batch_inner_off = bz * arith.index(K)   # inner K offset for A [M,B,K]
        b_batch_off = bz * arith.index(N)         # [B*N, K] row offset for B (unchanged)
        # C is [M, B, N]: batch bz shifts the N inner dimension by bz*N elements
        c_batch_inner_off = bz * arith.index(N)   # inner N offset for C [M,B,N]

        # --- Epilogue setup ---
        c_nrec = arith.index(B) * m_idx * arith.index(N * elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        # --- Scale resource ---
        if not no_scale:
            scale_total_elems = arith.index(B * n_k_blocks * n_n_blocks)
            elem_bytes_scale_eff = 1 if use_e8m0_scale else ELEM_BYTES_SCALE
            scale_nrec = scale_total_elems * arith.index(elem_bytes_scale_eff)
            scale_rsrc = buffer_ops.create_buffer_resource(arg_scale, num_records_bytes=scale_nrec)
            # n_block = by (since tile_n = group_n)
            by_i32 = arith.index_cast(T.i32, by)
            bz_i32 = arith.index_cast(T.i32, bz)
            scale_batch_off_i32 = bz_i32 * arith.constant(n_k_blocks * n_n_blocks, type=T.i32)
            scale_n_stride_i32 = arith.constant(n_n_blocks, type=T.i32)

        # --- TDM descriptor factories ---
        def make_desc_a(lds_a_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, lds_memref=lds_a_mem_ref,
                global_offset=(blk_m, a_batch_inner_off + k_base),
                tensor_shape=(tile_m, tile_k), strides=(B * K, 1),
                tile_shape=(tile_m, tile_k), elem_bytes=ELEM_BYTES_A,
                pad_interval=tile_k, pad_amount=LDS_PAD_A,
                num_warps=num_warps, workgroup_mask=a_mcast_mask)

        def make_desc_b(lds_b_mem_ref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=lds_b_mem_ref,
                global_offset=(b_batch_off + blk_n, k_base),
                tensor_shape=(tile_n, tile_k), strides=(K, 1),
                tile_shape=(tile_n, tile_k), elem_bytes=ELEM_BYTES_B,
                pad_interval=tile_k, pad_amount=LDS_PAD_B_FP8,
                num_warps=num_warps, workgroup_mask=b_mcast_mask)

        # --- A LDS load helpers (bf16, unchanged from bf16 BMM) ---
        def _precompute_a_lane_bases(lds_base_idx):
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride * ELEM_BYTES_A)
            k_lane_off = lane_kgrp * arith.index(8 * ELEM_BYTES_A)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = (
                    row_stride_off
                    + arith.index(wm * WMMA_M * lds_a_stride * ELEM_BYTES_A)
                    + k_lane_off
                )
                bases.append(a_base)
            return lds_base_idx, bases

        def load_wmma_frag_a(a_lds_base_idx, a_lane_base, ks):
            vec8_ty = ir.VectorType.get([8], T.bf16)
            k_byte_off = arith.index(ks * WMMA_K * ELEM_BYTES_A)
            off0 = a_lane_base + k_byte_off
            off1 = a_lane_base + k_byte_off + arith.index(32)
            raw0 = lds_load_b128_raw(a_lds_base_idx, off0)
            raw1 = lds_load_b128_raw(a_lds_base_idx, off1)
            v0 = vector.bitcast(vec8_ty, raw0)
            v1 = vector.bitcast(vec8_ty, raw1)
            return vector.shuffle(v0, v1, list(range(16)))

        # --- B LDS load helpers (fp8 → bf16 with per-block scale) ---
        def _precompute_b_lane_bases_fp8(lds_base_idx):
            # B^T [N,K] K-inner LDS: each lane selects an N-row (like A's M-row),
            # K is the offset within the row. Symmetric with _precompute_a_lane_bases.
            row_stride_off = (warp_n_base + lane16) * arith.index(lds_b_stride_fp8 * ELEM_BYTES_B)
            k_lane_off = lane_kgrp * arith.index(8 * ELEM_BYTES_B)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                b_base = (
                    row_stride_off
                    + arith.index(wn * WMMA_N * lds_b_stride_fp8 * ELEM_BYTES_B)
                    + k_lane_off
                )
                bases.append(b_base)
            return lds_base_idx, bases

        ty_2xi32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
        ty_2xf32 = ir.VectorType.get([2], ir.F32Type.get())
        ty_8xf32 = ir.VectorType.get([8], ir.F32Type.get())
        ty_8xbf16 = ir.VectorType.get([8], T.bf16)

        def _decompose_scale(scale_f32):
            """Decompose fp32 scale → (e8m0_byte, residual_bf16). Call once per K-tile.

            scale_f32 = 2^(E-127) × residual, residual ∈ [1.0, 2.0)
            E8M0 = biased exponent byte (bits 30:23 of fp32 bit pattern).
            residual = fp32 with exponent forced to 127.
            """
            scale_i32 = arith.bitcast(T.i32, scale_f32)
            e8m0_byte = arith.andi(
                arith.shrui(scale_i32, arith.constant(23, type=T.i32)),
                arith.constant(0xFF, type=T.i32))
            mantissa = arith.andi(scale_i32, arith.constant(0x807FFFFF, type=T.i32))
            residual_f32 = arith.bitcast(T.f32,
                arith.ori(mantissa, arith.constant(0x3F800000, type=T.i32)))
            residual_bf16 = arith.trunc_f(T.bf16, residual_f32)
            return e8m0_byte, residual_bf16

        def _load_frag_b_fp8_parts(lds_base_idx, b_lane_base, ks,
                                    e8m0_byte, residual_bf16):
            """Load one 16×32 bf16 WMMA B fragment from K-inner fp8 LDS (A-symmetric: lds_load_b128_raw × 2).

            B^T [N,K] layout: each lane holds one N-row; K offset within row.
            Two 128-bit loads at 0 and +16 bytes cover 32 fp8 = one WMMA_K group.
            Lower 64 bits (2×i32) extracted per load → 8 fp8 each.
            """
            k_byte_off = arith.index(ks * WMMA_K * ELEM_BYTES_B)
            off0 = b_lane_base + k_byte_off
            off1 = b_lane_base + k_byte_off + arith.index(16)
            raw0_128 = lds_load_b128_raw(lds_base_idx, off0)
            raw1_128 = lds_load_b128_raw(lds_base_idx, off1)
            raw0 = vector.shuffle(raw0_128, raw0_128, [0, 1])  # 2×i32 = 8 fp8
            raw1 = vector.shuffle(raw1_128, raw1_128, [0, 1])
            bf16_0 = rocdl.cvt_scale_pk8_bf16_fp8(ty_8xbf16, raw0, _raw(e8m0_byte), 0)
            bf16_1 = rocdl.cvt_scale_pk8_bf16_fp8(ty_8xbf16, raw1, _raw(e8m0_byte), 0)
            res0 = arith.mulf(bf16_0, vector.broadcast(ty_8xbf16, residual_bf16))
            res1 = arith.mulf(bf16_1, vector.broadcast(ty_8xbf16, residual_bf16))
            return vector.shuffle(res0, res1, list(range(16)))

        def _load_frag_b_fp8_parts_e8m0(lds_base_idx, b_lane_base, ks, e8m0_byte):
            """E8M0-only path: fp8 → bf16 with integer E8M0 scale, no residual mulf.

            Eliminates 4×V_PK_MUL_BF16 vs the fp32 path. Same A-symmetric lds_load_b128_raw layout.
            """
            k_byte_off = arith.index(ks * WMMA_K * ELEM_BYTES_B)
            off0 = b_lane_base + k_byte_off
            off1 = b_lane_base + k_byte_off + arith.index(16)
            raw0_128 = lds_load_b128_raw(lds_base_idx, off0)
            raw1_128 = lds_load_b128_raw(lds_base_idx, off1)
            raw0 = vector.shuffle(raw0_128, raw0_128, [0, 1])
            raw1 = vector.shuffle(raw1_128, raw1_128, [0, 1])
            bf16_0 = rocdl.cvt_scale_pk8_bf16_fp8(ty_8xbf16, raw0, _raw(e8m0_byte), 0)
            bf16_1 = rocdl.cvt_scale_pk8_bf16_fp8(ty_8xbf16, raw1, _raw(e8m0_byte), 0)
            return vector.shuffle(bf16_0, bf16_1, list(range(16)))

        def load_wmma_frag_b_fp8(lds_base_idx, b_lane_base, ks, scale_val):
            """Load one 16×32 bf16 WMMA B fragment.

            scale_val: fp32 scalar when _effective_e8m0=False (decomposed on-the-fly),
                       i32 E8M0 byte when _effective_e8m0=True (used directly;
                       constant 127 when no_scale=True).
            """
            if _effective_e8m0:
                return _load_frag_b_fp8_parts_e8m0(lds_base_idx, b_lane_base, ks, scale_val)
            else:
                e8m0_byte, residual_bf16 = _decompose_scale(scale_val)
                return _load_frag_b_fp8_parts(lds_base_idx, b_lane_base, ks,
                                              e8m0_byte, residual_bf16)

        # --- K-subtile compute ---
        def _load_b_frags_fp8(b_lds_buffer, b_bases, ks, scale_f32):
            return [load_wmma_frag_b_fp8(b_lds_buffer, b_bases[wn], ks, scale_f32)
                    for wn in range_constexpr(wmma_n_rep)]

        use_half_streaming_schedule = (wmma_m_rep % 2) == 0 and wmma_m_rep > 1

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
                                         emit_filler=None, mid_compute_callback=None,
                                         next_b_info=None):
            next_b_frags = None
            a_frag = load_wmma_frag_a(a_buf, a_bases[0], ks)
            for wm in range_constexpr(wmma_m_rep):
                is_last = (wm == wmma_m_rep - 1)
                if not is_last:
                    a_next = load_wmma_frag_a(a_buf, a_bases[wm + 1], ks)
                if is_last:
                    rocdl.s_wait_dscnt(0)
                    if emit_filler is not None:
                        rocdl.sched_barrier(0)
                        emit_filler()
                    if next_b_info is not None:
                        nb_buf, nb_bases, nb_ks, nb_scale = next_b_info
                        next_b_frags = _load_b_frags_fp8(nb_buf, nb_bases, nb_ks, nb_scale)
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
                                       emit_filler=None, mid_compute_callback=None,
                                       next_b_info=None):
            next_b_frags = None
            half_wm = wmma_m_rep // 2
            half_wait = (half_wm - 1) * DS_LOADS_PER_A_FRAG
            a_frags_h0 = [load_wmma_frag_a(a_buf, a_bases[wm], ks)
                          for wm in range_constexpr(half_wm)]
            rocdl.s_wait_dscnt(half_wait)
            if mid_compute_callback is not None:
                rocdl.sched_barrier(0)
                mid_compute_callback()
            for wm in range_constexpr(half_wm):
                _emit_wmma_row(accs, wm, a_frags_h0[wm], b_frags)
            a_frags_h1 = [load_wmma_frag_a(a_buf, a_bases[half_wm + h], ks)
                          for h in range_constexpr(half_wm)]
            rocdl.s_wait_dscnt(half_wait)
            for h in range_constexpr(half_wm):
                wm = half_wm + h
                if wm == wmma_m_rep - 1 and emit_filler is not None:
                    rocdl.sched_barrier(0)
                    emit_filler()
                _emit_wmma_row(accs, wm, a_frags_h1[h], b_frags)
            if next_b_info is not None:
                nb_buf, nb_bases, nb_ks, nb_scale = next_b_info
                next_b_frags = _load_b_frags_fp8(nb_buf, nb_bases, nb_ks, nb_scale)
                return accs, next_b_frags
            return accs

        def _a_streaming_compute(accs, a_buf, a_bases, b_frags, ks,
                                  emit_filler=None, mid_compute_callback=None,
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

        def compute_tile(accs_in, lds_a_idx, lds_b_idx, scale_f32,
                         emit_filler=None, mid_compute_callback=None):
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_idx)
            b_buf, b_bases = _precompute_b_lane_bases_fp8(lds_b_idx)

            if k_wmma_steps == 1:
                b_frags = _load_b_frags_fp8(b_buf, b_bases, 0, scale_f32)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, b_frags, 0,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback)
            else:
                prev_b = _load_b_frags_fp8(b_buf, b_bases, 0, scale_f32)
                for ks in range_constexpr(k_wmma_steps - 1):
                    _mid_cb = mid_compute_callback if ks == 0 else None
                    current_accs, prev_b = _a_streaming_compute(
                        current_accs, a_buf, a_bases, prev_b, ks,
                        mid_compute_callback=_mid_cb,
                        next_b_info=(b_buf, b_bases, ks + 1, scale_f32))
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, prev_b,
                    k_wmma_steps - 1,
                    emit_filler=emit_filler)
            return current_accs

        def hot_loop_scheduler():
            if not use_half_streaming_schedule:
                rocdl.sched_barrier(0)
                return
            half_wm = wmma_m_rep // 2
            half_wmma = half_wm * wmma_n_rep
            # fp8 B: ds_load_tr8_b64 x2 per K-subtile → DS_LOADS_PER_B_FRAG=2
            b_full_loads = wmma_n_rep * 2
            a_half_loads = half_wm * DS_LOADS_PER_A_FRAG
            for ks in range_constexpr(k_wmma_steps):
                if ks == 0:
                    rocdl.sched_dsrd(b_full_loads + a_half_loads)
                else:
                    rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                if ks < k_wmma_steps - 1:
                    rocdl.sched_dsrd(b_full_loads)
            rocdl.sched_barrier(0)

        # --- Epilogue helpers ---
        _out_elem = T.f16 if out_dtype == "f16" else (T.bf16 if out_dtype == "bf16" else None)

        def epilogue_prepare_addrs():
            addrs = []
            bn_stride = arith.index(B * N)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = (c_batch_inner_off + blk_n + warp_n_base
                                + arith.index(wn * WMMA_N) + lane_kgrp * arith.index(8))
                    if _half_out:
                        c_off_bytes = (row * bn_stride + col_base) * arith.index(elem_bytes_d)
                        addrs.append(c_off_bytes)
                    else:
                        for half in range_constexpr(2):
                            col = col_base + arith.index(half * 4)
                            c_off = row * bn_stride + col
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

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    imm = wm * WMMA_M * _lds_d_stride_elems + wn * _n_col_d_elems
                    store_acc_vec8_to_lds(d_buf, d_base, imm, final_accs[idx], out_elem=_out_elem)

        def _l2_prefetch(k_base):
            if _effective_l2_pf <= 0:
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, a_batch_inner_off + pf_k), (tile_m, tile_k), (B * K, 1),
                elem_bytes=ELEM_BYTES_A, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b, (b_batch_off + blk_n, pf_k), (tile_n, tile_k), (K, 1),
                elem_bytes=ELEM_BYTES_B, thread_id=tx, block_threads=block_threads)

        def _load_scale(k_tile_i32):
            """Load scale[bz, k_tile_i32, n_block].

            no_scale=True:        constant E8M0=127 (scale=1.0, plain fp8→bf16 conversion).
            use_e8m0_scale=True:  uint8 E8M0 zero-extended to i32 from HBM.
            default:              fp32 scalar from HBM.
            """
            if no_scale:
                return arith.constant(127, type=T.i32)
            scale_elem_off = (scale_batch_off_i32
                              + k_tile_i32 * scale_n_stride_i32
                              + by_i32)
            if use_e8m0_scale:
                i8_val = buffer_ops.buffer_load(
                    scale_rsrc, scale_elem_off, vec_width=1, dtype=T.i8)
                return arith.extui(T.i32, i8_val)
            return buffer_ops.buffer_load(
                scale_rsrc, scale_elem_off, vec_width=1, dtype=T.f32)

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        arena_base_ptr = arena_alloc.get_base()

        # A stages: bf16 SmemPtr
        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_offsets[i], T.bf16, shape=(lds_a_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_a_idx = [extract_lds_base_idx(stages_a[i])
                        for i in range_constexpr(num_buffers)]

        # B stages: fp8 (1 byte/elem). Use T.bf16 with half-size shape so the
        # SmemPtr covers the right byte range (lds_b_elems_fp8 bytes total).
        # extract_lds_base_idx returns a byte base; _lds_ptr_raw uses byte offsets.
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_offsets[i], T.bf16,
                    shape=(lds_b_elems_fp8 // 2,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i])
                        for i in range_constexpr(num_buffers)]

        if use_tdm_store:
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // ELEM_BYTES_A
            d_smem = SmemPtr(d_lds_base_ptr, d_output_off, T.bf16,
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
            warp_m_off_sgpr = (wave_id_idx / arith.index(n_warp)) * arith.index(warp_tile_m)
            warp_n_off_sgpr = (wave_id_idx % arith.index(n_warp)) * arith.index(warp_tile_n)
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_c,
                lds_memref=d_lds_base_ptr,
                global_offset=(blk_m + warp_m_off_sgpr,
                               c_batch_inner_off + blk_n + warp_n_off_sgpr),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(B * N, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
            )

        # --- TDM addr_lo management ---
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

        addr_lo_a = vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_a = vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[])
        addr_lo_b = vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[])
        addr_hi_b = vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[])
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1

        adv_a_i32 = arith.constant(tile_k * ELEM_BYTES_A, type=T.i32)
        adv_b_i32 = arith.constant(tile_k * ELEM_BYTES_B, type=T.i32)
        pred_const = arith.constant(1, type=T.i32)

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
            old_lo_b = addr_lo_b
            addr_lo_b = arith.addi(addr_lo_b, adv_b_i32)
            carry = arith.cmpi(arith.CmpIPredicate.ult, addr_lo_b, old_lo_b)
            addr_hi_b = arith.addi(addr_hi_b, arith.extui(T.i32, carry))

        pipeline_fence(outstanding=2 * (num_buffers - 2), use_cluster=use_cluster)

        # --- Main loop ---
        _fence_outstanding = 2 * (num_buffers - 2)

        # Scale prefetch: overlap fp32 scale HBM load with WMMA by issuing it inside
        # mid_compute_callback.  No-op for e8m0/no_scale (scale is cheap: 1 byte or
        # constant).  The prefetched value is loop-carried so each iteration starts
        # with scale already in registers.
        _do_scale_prefetch = loop_iters > 0 and not _effective_e8m0

        if loop_iters > 0:
            if _do_scale_prefetch:
                # Scale prefetch path: each iteration uses the previously-prefetched
                # scale and issues the next K-tile's scale load mid-WMMA via callback.
                _prefetch_scale_0 = _load_scale(arith.constant(0, type=T.i32))
                _sp_init = list(accs) + [addr_lo_a, addr_lo_b, addr_hi_b,
                                         _prefetch_scale_0]
                for loop_iter, state in range(0, loop_iters, 1, init=_sp_init):
                    accs_in = list(state[:n_accs])
                    cur_lo_a = state[n_accs]
                    cur_lo_b = state[n_accs + 1]
                    cur_hi_b = state[n_accs + 2]
                    scale_box = [state[n_accs + 3]]

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers

                        pipeline_fence_signal(outstanding=_fence_outstanding, use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        scale_val = scale_box[0]
                        _next_k_i32 = arith.index_cast(
                            T.i32,
                            loop_iter * arith.index(num_buffers) + arith.index(buf_idx + 1))

                        addr_boxes = [[cur_lo_a], [cur_lo_b], [cur_hi_b]]

                        def _mid_tdm_sp(
                            _ls=load_stage,
                            _ab=addr_boxes,
                            _k_off=(loop_iter * arith.index(num_buffers * tile_k)
                                    + arith.index(buf_idx * tile_k)),
                            _nk=_next_k_i32,
                        ):
                            dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_a_lds_addr[_ls],
                                _ab[0][0], addr_hi_a])
                            dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                pred_const, stages_b_lds_addr[_ls],
                                _ab[1][0], _ab[2][0]])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                wave_specialized=False)
                            _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                            old_lo_b = _ab[1][0]
                            new_lo_b = arith.addi(old_lo_b, adv_b_i32)
                            carry = arith.cmpi(arith.CmpIPredicate.ult, new_lo_b, old_lo_b)
                            _ab[1][0] = new_lo_b
                            _ab[2][0] = arith.addi(_ab[2][0], arith.extui(T.i32, carry))
                            scale_box[0] = _load_scale(_nk)
                            _l2_prefetch(_k_off)

                        rocdl.sched_barrier(0)
                        accs_in = compute_tile(
                            accs_in,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            scale_val,
                            mid_compute_callback=_mid_tdm_sp)
                        cur_lo_a = addr_boxes[0][0]
                        cur_lo_b = addr_boxes[1][0]
                        cur_hi_b = addr_boxes[2][0]
                        hot_loop_scheduler()

                    results = yield list(accs_in) + [cur_lo_a, cur_lo_b, cur_hi_b,
                                                     scale_box[0]]

                accs = list(results[:n_accs])
                addr_lo_a = results[n_accs]
                addr_lo_b = results[n_accs + 1]
                addr_hi_b = results[n_accs + 2]
            else:
                init_args = list(accs) + [addr_lo_a, addr_lo_b, addr_hi_b]

                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_lo_a = state[n_accs]
                    cur_lo_b = state[n_accs + 1]
                    cur_hi_b = state[n_accs + 2]

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers

                        pipeline_fence_signal(outstanding=_fence_outstanding, use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        # Load scale for K-tile being computed (loop_iter*num_buffers + buf_idx)
                        k_tile_compute_i32 = arith.index_cast(
                            T.i32,
                            loop_iter * arith.index(num_buffers) + arith.index(buf_idx))
                        scale_val = _load_scale(k_tile_compute_i32)

                        addr_boxes = [[cur_lo_a], [cur_lo_b], [cur_hi_b]]

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
                                _ab[1][0], _ab[2][0]])
                            issue_tdm_loads(
                                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                wave_specialized=False)
                            _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                            # Advance addr_lo_b with carry propagation into addr_hi_b.
                            # adv_b_i32 = tile_k*ELEM_BYTES_B bytes (K-inner stride);
                            # small value but carry guard kept for correctness at large K.
                            old_lo_b = _ab[1][0]
                            new_lo_b = arith.addi(old_lo_b, adv_b_i32)
                            carry = arith.cmpi(arith.CmpIPredicate.ult, new_lo_b, old_lo_b)
                            _ab[1][0] = new_lo_b
                            _ab[2][0] = arith.addi(_ab[2][0], arith.extui(T.i32, carry))
                            _l2_prefetch(_k_off)

                        rocdl.sched_barrier(0)
                        accs_in = compute_tile(
                            accs_in,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            scale_val,
                            mid_compute_callback=_mid_tdm)
                        cur_lo_a = addr_boxes[0][0]
                        cur_lo_b = addr_boxes[1][0]
                        cur_hi_b = addr_boxes[2][0]
                        hot_loop_scheduler()

                    results = yield list(accs_in) + [cur_lo_a, cur_lo_b, cur_hi_b]

                accs = list(results[:n_accs])
                addr_lo_a = results[n_accs]
                addr_lo_b = results[n_accs + 1]
                addr_hi_b = results[n_accs + 2]

        # --- Tail ---
        if loop_iters > 0:
            pipeline_fence(outstanding=0, use_cluster=use_cluster)
        elif use_cluster:
            gpu.cluster_barrier()
        epi_addrs_box = [None]
        _tail_had_load = False
        _tail_compute_counter = [0]

        for _load_stage, _compute_stage, _outstanding in tail_plan:
            _tail_k_tile = loop_iters * num_buffers + _tail_compute_counter[0]
            _tail_scale_i32 = arith.constant(_tail_k_tile, type=T.i32)
            _tail_scale = _load_scale(_tail_scale_i32)
            _tail_compute_counter[0] += 1

            if _outstanding == -1:
                if _tail_had_load:
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                if use_tdm_store:
                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        _tail_scale)
                else:
                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        _tail_scale,
                        emit_filler=_emit_epi_addrs)
            else:
                pipeline_fence_signal(outstanding=_outstanding, use_cluster=use_cluster)
                pipeline_fence_wait(use_cluster=use_cluster)

                _tail_mid_cb = None
                if _load_stage is not None:
                    _tail_had_load = True
                    _tail_ab = [[addr_lo_a], [addr_lo_b], [addr_hi_b]]

                    def _tail_mid_nws(_ls=_load_stage, _ab=_tail_ab):
                        dg0_a = vector.from_elements(T.vec(4, T.i32), [
                            pred_const, stages_a_lds_addr[_ls],
                            _ab[0][0], addr_hi_a])
                        dg0_b = vector.from_elements(T.vec(4, T.i32), [
                            pred_const, stages_b_lds_addr[_ls],
                            _ab[1][0], _ab[2][0]])
                        issue_tdm_loads(
                            tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                            tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                            wave_specialized=False)
                        _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                        old_lo_b = _ab[1][0]
                        new_lo_b = arith.addi(old_lo_b, adv_b_i32)
                        carry = arith.cmpi(arith.CmpIPredicate.ult, new_lo_b, old_lo_b)
                        _ab[1][0] = new_lo_b
                        _ab[2][0] = arith.addi(_ab[2][0], arith.extui(T.i32, carry))

                    _tail_mid_cb = _tail_mid_nws

                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs,
                    stages_a_idx[_compute_stage],
                    stages_b_idx[_compute_stage],
                    _tail_scale,
                    mid_compute_callback=_tail_mid_cb)
                hot_loop_scheduler()

                if _load_stage is not None:
                    addr_lo_a = _tail_ab[0][0]
                    addr_lo_b = _tail_ab[1][0]
                    addr_hi_b = _tail_ab[2][0]

        # --- Epilogue ---
        if use_tdm_store:
            if d_need_epilogue_fence:
                pipeline_fence(outstanding=0, use_cluster=use_cluster)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            rocdl.sched_barrier(0)
            epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (B, K, N, group_k, group_n, tile_m, tile_n, tile_k,
                 m_warp, n_warp, num_buffers, out_dtype, waves_per_eu,
                 l2_prefetch_distance, use_tdm_store, inst_prefetch, expert_sched_mode,
                 use_e8m0_scale, no_scale, cluster_m, cluster_n)

    @flyc.jit
    def launch_bmm_a16w8_gfx1250(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale: fx.Tensor,
        i32_m: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))

        launcher = kernel_bmm_a16w8_gfx1250(arg_c, arg_a, arg_b, arg_scale, i32_m)
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                if waves_per_eu is not None:
                    _wpe = int(waves_per_eu)
                    if _wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
                if use_cluster:
                    op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                        f"{cluster_m},{cluster_n},1")
        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        launcher.launch(
            grid=(gx, gy_compile, B),
            block=(block_threads, 1, 1),
            cluster=cluster_arg,
            stream=stream,
        )

        llvm_opts = {}
        if expert_sched_mode:
            llvm_opts["amdgpu-expert-scheduling-mode"] = True
        if inst_prefetch:
            llvm_opts["amdgpu-inst-prefetch-distance"] = 8
        if llvm_opts:
            launch_bmm_a16w8_gfx1250.compile_hints["llvm_options"] = llvm_opts

    return launch_bmm_a16w8_gfx1250


__all__ = ["compile_bmm_a16w8_gfx1250"]
