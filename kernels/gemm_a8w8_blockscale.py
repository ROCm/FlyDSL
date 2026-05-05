# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W8 FP8 blockscale GEMM for gfx1250.

Computes Y = X @ W^T with per-K-block f32 scales.
Supports reg_preload / no_op_preload variants and optional TDM-store output.

Variants:
  - reg_preload    : default. Operand frags loop-carried across K-tiles.
                       * W-scales: bulk-load K-tiles' W-scales into VGPRs
                         (each buffer_load_b32 covers up to 32 K-blocks).
                         scale_k <= 32 → one bulk load at kernel entry +
                         per-K-tile v_readlane. scale_k > 32 → a cur/prefetch
                         chunk chain in the loop carry. Requires
                         w_is_wave_uniform.
                       * X-scales: TDM-staged into LDS (num_buffers stages,
                         aligned with X+W tile stages), then ds_read_b32 into
                         VGPRs in lane16 layout.
  - no_op_preload  : operand frags loaded fresh per K-tile (lower VGPR cost).
                     Uses the legacy per-K-block buffer_load scale path.
"""

from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
    tdm_ops,
    vector,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_mlir_type_size

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
FRAG_VGPRS = 16
DS_LOADS_PER_FRAG = 4
LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16
LDS_PAD_D_BYTES = 16
_STAGE_NAMES = ("ping", "pong", "pang", "pung")


def _lds_vec_type(memref, total_bits):
    """Return a vector type of the right shape to hold `total_bits` of the
    memref's element type (used to size ds_load_bNNN reads)."""
    raw_mr = arith.unwrap(memref)
    elem_type = ir.MemRefType(raw_mr.type).element_type
    elem_bits = get_mlir_type_size(elem_type) * 8
    n = total_bits // elem_bits
    return ir.VectorType.get([n], elem_type)


def lds_load_b128(memref, elem_off):
    """ds_load_b128: load 16 bytes from LDS into a vector<4xi32>."""
    vec_ty = _lds_vec_type(memref, 128)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    return vector.bitcast(
        ir.VectorType.get([4], ir.IntegerType.get_signless(32)), loaded
    )


def lds_load_b32_f32(memref, elem_off):
    """ds_load_b32: load 4 bytes from LDS as a single f32. The memref's
    element type may be smaller (e.g. f8 for byte-addressed staging); we
    read the right number of element units to cover 32 bits and bitcast."""
    vec_ty = _lds_vec_type(memref, 32)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    as_f32_vec = vector.bitcast(
        ir.VectorType.get([1], ir.F32Type.get()), loaded
    )
    return vector.extract(as_f32_vec, static_position=[0], dynamic_position=[])


def lds_store_b128(memref, elem_off, data):
    """ds_store_b128: store 16 bytes to LDS, bitcast to match the memref element type."""
    vec_ty = _lds_vec_type(memref, 128)
    typed_vec = vector.bitcast(vec_ty, data)
    vector.store(typed_vec, memref, [elem_off])


def store_acc_vec8_to_lds(memref, base_elem_off, imm_elem_off, acc_vec8, out_elem=None):
    """Write a vec<8 f32> accumulator to LDS for TDM-store epilogue.

    Half output (out_elem = T.bf16/T.f16): trunc_f → bitcast(vec<4xi32>) → 1
    ds_store_b128 (16 bytes covering all 8 elements).
    f32 output (out_elem = None): two ds_store_b128 calls writing 4 f32 each;
    second store offset by 8 LDS elements (the LDS memref is 16-bit-typed even
    for f32 output, so 8 elems = 16 bytes = 4 f32).
    """
    off = base_elem_off + arith.index(imm_elem_off)
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
        lds_store_b128(memref, off, i32_vec)
    else:
        for half in range(2):
            vals = [
                vector.extract(
                    acc_vec8, static_position=[half * 4 + vi], dynamic_position=[]
                )
                for vi in range(4)
            ]
            vec4 = vector.from_elements(T.vec(4, T.f32), vals)
            lds_store_b128(memref, off + arith.index(half * 8), vec4)


def store_acc_vec8_to_buffer(
    acc_vec8, c_rsrc, addr, out_elem=None, offset_is_bytes=False
):
    """Write a vec<8xf32> accumulator to global via buffer_store.

    If `out_elem` is a half-precision type (bf16/fp16), truncate f32→half and
    emit a single 16-byte buffer_store of a vec<4xi32>.
    If `out_elem` is None (f32 out), emit two vec<4xf32> stores (one per half).
    """
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
        buffer_ops.buffer_store(i32_vec, c_rsrc, addr, offset_is_bytes=offset_is_bytes)
        return 1
    for half in range(2):
        vals = [
            vector.extract(
                acc_vec8, static_position=[half * 4 + vi], dynamic_position=[]
            )
            for vi in range(4)
        ]
        vec4 = vector.from_elements(T.vec(4, T.f32), vals)
        if isinstance(addr, (list, tuple)):
            buffer_ops.buffer_store(vec4, c_rsrc, addr[half])
        else:
            buffer_ops.buffer_store(vec4, c_rsrc, addr)
    return 2


def compile_gemm_a8w8_blockscale(
    *,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
    out_dtype: str = "bf16",
    variant: str = "reg_preload",
    N: int = 0,
    use_tdm_store: bool = False,
    loop_carried_load_percent: Optional[int] = None,
    kernarg_preload: bool = False,
):
    if variant not in ("reg_preload", "no_op_preload"):
        raise ValueError(
            f"variant must be 'reg_preload' or 'no_op_preload', got {variant!r}"
        )
    if const_expr(variant == "reg_preload"):
        _w_is_wave_uniform = (tile_n // n_warp) <= scale_block_n
        if not _w_is_wave_uniform:
            raise ValueError(
                f"variant='reg_preload' requires warp_tile_n ({tile_n // n_warp}) "
                f"<= scale_block_n ({scale_block_n}) (W-scale must be wave-uniform)"
            )
        # scale_k > 32 → multi-chunk prefetch chain (issued at entry chunks 0+1,
        # advanced per iteration in the main loop). No upper bound.
    if out_dtype not in ("bf16", "fp16", "f32"):
        raise ValueError(
            f"out_dtype must be 'bf16', 'fp16', or 'f32', got {out_dtype!r}"
        )
    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_k % scale_block_k != 0:
        raise ValueError(
            f"tile_k ({tile_k}) must be a multiple of scale_block_k ({scale_block_k})"
        )
    if scale_block_k % WMMA_K != 0:
        raise ValueError(
            f"scale_block_k ({scale_block_k}) must be a multiple of {WMMA_K}"
        )
    if K % tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by tile_k ({tile_k})")
    if K % scale_block_k != 0:
        raise ValueError(
            f"K ({K}) must be divisible by scale_block_k ({scale_block_k})"
        )
    if use_tdm_store:
        if N <= 0:
            raise ValueError(
                "use_tdm_store=True requires N > 0 (compile-time row stride)"
            )
        if N % tile_n != 0:
            raise ValueError(
                f"use_tdm_store=True requires N ({N}) to be a multiple of tile_n ({tile_n})"
            )

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    wmma_m_rep = warp_tile_m // WMMA_M  # WMMA tiles per warp along M
    wmma_n_rep = warp_tile_n // WMMA_N  # WMMA tiles per warp along N
    n_accs = wmma_m_rep * wmma_n_rep  # global accumulators per warp
    k_wmma_steps = tile_k // WMMA_K  # WMMAs per K-tile along K
    scales_per_tile = tile_k // scale_block_k  # scale blocks per K-tile
    wmma_steps_per_scale = scale_block_k // WMMA_K
    wmma_pipeline_depth = min(n_accs, 2)
    acc_coords = [
        (wm, wn, wm * wmma_n_rep + wn)
        for wm in range(wmma_m_rep)
        for wn in range(wmma_n_rep)
    ]

    num_k_tiles = K // tile_k
    scale_k = K // scale_block_k

    # W-scale chunking: 1 buffer_load_b32 covers 32 K-blocks; lazy chain when scale_k > 32.
    NUM_W_CHUNKS = (scale_k + 31) // 32 if variant == "reg_preload" else 1
    USES_W_CHUNK_PREFETCH = variant == "reg_preload" and NUM_W_CHUNKS > 1

    if num_k_tiles < num_buffers - 1:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers - 1}, "
            f"got {num_k_tiles}"
        )

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    elem_bytes_d = 2 if out_dtype in ("bf16", "fp16") else 4
    effective_waves_per_eu = waves_per_eu

    lds_a_stride_bytes = tile_k + LDS_PAD_A_BYTES
    lds_b_stride_bytes = tile_k + LDS_PAD_B_BYTES
    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * lds_b_stride_bytes

    # X-scale LDS (TDM-staged): tile_m rows × scales_per_tile × 4B per stage.
    USES_X_SCALE_TDM = variant == "reg_preload"
    lds_x_scale_row_bytes = scales_per_tile * 4
    lds_x_scale_data_bytes = tile_m * lds_x_scale_row_bytes if USES_X_SCALE_TDM else 0

    stage_allocators = []
    stage_a_data_off = []
    stage_b_data_off = []
    stage_x_scale_off = []

    for i in range(num_buffers):
        alloc = SmemAllocator(
            None, arch=gpu_arch, global_sym_name=f"a8w8bs_{_STAGE_NAMES[i]}"
        )
        off = alloc._align(alloc.ptr, 16)
        stage_a_data_off.append(off)
        alloc.ptr = off + lds_a_data_bytes
        off = alloc._align(alloc.ptr, 16)
        stage_b_data_off.append(off)
        alloc.ptr = off + lds_b_data_bytes
        if const_expr(USES_X_SCALE_TDM):
            off = alloc._align(alloc.ptr, 16)
            stage_x_scale_off.append(off)
            alloc.ptr = off + lds_x_scale_data_bytes
        stage_allocators.append(alloc)

    if use_tdm_store:
        lds_d_row_stride_bytes = tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        total_d_bytes = tile_m * lds_d_row_stride_bytes
        _lds_d_stride_elems_d = lds_d_row_stride_bytes // 2
        _n_col_d_elems_d = WMMA_N * elem_bytes_d // 2

        d_lds_allocator = SmemAllocator(
            None,
            arch=gpu_arch,
            global_sym_name="a8w8bs_d_out",
        )
        d_lds_allocator.ptr = total_d_bytes

    prologue_tiles = num_buffers - 1
    main_loop_iters = (num_k_tiles - prologue_tiles) // num_buffers
    extra_tiles = num_k_tiles - main_loop_iters * num_buffers - prologue_tiles
    drain_iters = num_buffers - 2

    # TDMs per tile: 2 (X+W) for no_op_preload, 3 (X+W+scale) for reg_preload.
    _TDMS_PER_TILE_EXP = 3 if USES_X_SCALE_TDM else 2
    MAIN_TDM_OUTSTANDING = (num_buffers - 2) * 2
    MAIN_TDM_OUTSTANDING_EXPERIMENTAL = (num_buffers - 2) * _TDMS_PER_TILE_EXP

    @flyc.kernel
    def kernel_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx = fx.get(thr_coord, 0)
        wave_n_idx = fx.get(thr_coord, 1)
        lane_kgrp = fx.get(thr_coord, 2)
        lane16 = fx.get(thr_coord, 3)

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_idx = arith.index_cast(T.index, i32_n.ir_value())
        n_stride = n_idx

        y_total_bytes = m_idx * n_stride * arith.index(elem_bytes_d)
        y_buf = buffer_ops.create_buffer_resource(
            arg_y, num_records_bytes=y_total_bytes
        )

        x_scale_total_bytes = m_idx * arith.index(scale_k) * arith.index(4)
        x_scale_buf = buffer_ops.create_buffer_resource(
            arg_x_scale, num_records_bytes=x_scale_total_bytes
        )

        num_n_scale_blocks = (n_idx + arith.index(scale_block_n - 1)) / arith.index(
            scale_block_n
        )
        w_scale_total_bytes = num_n_scale_blocks * arith.index(scale_k) * arith.index(4)
        w_scale_buf = buffer_ops.create_buffer_resource(
            arg_w_scale, num_records_bytes=w_scale_total_bytes
        )

        scale_zero = arith.constant(0.0, type=T.f32)

        stages_a = [
            SmemPtr(
                stage_allocators[i].get_base(),
                stage_a_data_off[i],
                T.f8,
                shape=(lds_a_data_bytes,),
            )
            for i in range(num_buffers)
        ]
        stages_b = [
            SmemPtr(
                stage_allocators[i].get_base(),
                stage_b_data_off[i],
                T.f8,
                shape=(lds_b_data_bytes,),
            )
            for i in range(num_buffers)
        ]
        stages_a_mem = [p.get() for p in stages_a]
        stages_b_mem = [p.get() for p in stages_b]

        # TDM descriptors built once at entry; GROUP1 + addr_hi stay in SGPRs, lo32 advances per K-tile.
        def _make_desc_x(lds_mem, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_x,
                lds_memref=lds_mem,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=num_warps,
            )

        def _make_desc_w(lds_mem, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w,
                lds_memref=lds_mem,
                global_offset=(blk_n, k_base),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_B_BYTES,
                num_warps=num_warps,
            )

        _desc_x_init = _make_desc_x(stages_a_mem[0], arith.index(0))
        _desc_w_init = _make_desc_w(stages_b_mem[0], arith.index(0))

        dgroup1_x = _desc_x_init.dgroup1
        dgroup1_w = _desc_w_init.dgroup1
        addr_hi_x = vector.extract(
            _desc_x_init.dgroup0, static_position=[3], dynamic_position=[]
        )
        addr_hi_w = vector.extract(
            _desc_w_init.dgroup0, static_position=[3], dynamic_position=[]
        )
        addr_lo_x_init = vector.extract(
            _desc_x_init.dgroup0, static_position=[2], dynamic_position=[]
        )
        addr_lo_w_init = vector.extract(
            _desc_w_init.dgroup0, static_position=[2], dynamic_position=[]
        )

        # Per-stage LDS i32 addresses (pre-extracted, selected by buf_idx).
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(
                vector.extract(
                    _make_desc_x(stages_a_mem[i], arith.index(0)).dgroup0,
                    static_position=[1],
                    dynamic_position=[],
                )
            )
            stages_b_lds_addr.append(
                vector.extract(
                    _make_desc_w(stages_b_mem[i], arith.index(0)).dgroup0,
                    static_position=[1],
                    dynamic_position=[],
                )
            )

        # K-axis innermost (stride=1, elem=1B): per-tile lo32 advance = tile_k bytes.
        adv_x_i32 = arith.constant(tile_k, type=T.i32)
        adv_w_i32 = arith.constant(tile_k, type=T.i32)
        pred_const = arith.constant(1, type=T.i32)

        def issue_tdm_loads(buf_idx, lo_x, lo_w):
            """Issue X+W TDMs for one K-tile into LDS stage `buf_idx`. Returns advanced (lo_x, lo_w)."""
            # rocdl.s_setprio(2)
            dg0_x = vector.from_elements(
                T.vec(4, T.i32),
                [pred_const, stages_a_lds_addr[buf_idx], lo_x, addr_hi_x],
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_x, dgroup1_x))
            dg0_w = vector.from_elements(
                T.vec(4, T.i32),
                [pred_const, stages_b_lds_addr[buf_idx], lo_w, addr_hi_w],
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_w, dgroup1_w))
            # rocdl.s_setprio(0)
            return arith.addi(lo_x, adv_x_i32), arith.addi(lo_w, adv_w_i32)

        # ── X-scale TDM descriptor + LDS staging (hoisted) ──────────────────
        if const_expr(USES_X_SCALE_TDM):
            # Single block so stages_x_scale_mem and downstream helpers
            # all live in the same scope, no cross-closure references.
            stages_x_scale = [
                SmemPtr(
                    stage_allocators[i].get_base(),
                    stage_x_scale_off[i],
                    T.f8,
                    shape=(lds_x_scale_data_bytes,),
                )
                for i in range(num_buffers)
            ]
            stages_x_scale_mem = [p.get() for p in stages_x_scale]

            def _make_desc_x_scale(lds_mem, kb_base):
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_x_scale,
                    lds_memref=lds_mem,
                    global_offset=(blk_m, kb_base),
                    tensor_shape=(tile_m, scales_per_tile),
                    strides=(scale_k, 1),
                    tile_shape=(tile_m, scales_per_tile),
                    elem_bytes=4,
                    pad_interval=scales_per_tile,
                    pad_amount=0,
                    num_warps=num_warps,
                )

            _desc_x_scale_init = _make_desc_x_scale(
                stages_x_scale_mem[0], arith.index(0)
            )
            dgroup1_x_scale = _desc_x_scale_init.dgroup1
            addr_hi_x_scale = vector.extract(
                _desc_x_scale_init.dgroup0, static_position=[3], dynamic_position=[]
            )
            addr_lo_x_scale_init = vector.extract(
                _desc_x_scale_init.dgroup0, static_position=[2], dynamic_position=[]
            )

            stages_x_scale_lds_addr = []
            for i in range_constexpr(num_buffers):
                stages_x_scale_lds_addr.append(
                    vector.extract(
                        _make_desc_x_scale(
                            stages_x_scale_mem[i], arith.index(0)
                        ).dgroup0,
                        static_position=[1],
                        dynamic_position=[],
                    )
                )

            # K-axis advance per K-tile: scales_per_tile K-blocks × 4 B/elem.
            adv_x_scale_i32 = arith.constant(scales_per_tile * 4, type=T.i32)

            def issue_x_scale_tdm(buf_idx, lo_x_scale):
                """Issue one x_scale K-tile TDM load into LDS stage `buf_idx`.
                Returns the lo32 advanced by scales_per_tile*4 bytes."""
                dg0 = vector.from_elements(
                    T.vec(4, T.i32),
                    [
                        pred_const,
                        stages_x_scale_lds_addr[buf_idx],
                        lo_x_scale,
                        addr_hi_x_scale,
                    ],
                )
                tdm_ops.tensor_load_2d(
                    tdm_ops.TDMDescriptor2D(dg0, dgroup1_x_scale)
                )
                return arith.addi(lo_x_scale, adv_x_scale_i32)

            def ds_read_x_scales(buf_idx):
                """Read this warp's x_scales for one K-tile from LDS stage
                `buf_idx`. Returns flat list with the same indexing convention
                as reg_preload's x_raw:
                    out[sc * wmma_m_rep + wm] = x_scale[row=lane16, sc, wm]
                Per-lane: ds_read_b32 at byte offset
                    (warp_m_base + wm*WMMA_M + lane16) * scales_per_tile*4 + sc*4
                Lanes 0..15 fetch 16 distinct rows; lanes 16..31 hit the same
                addresses (LDS broadcast — free). No cross-lane ops."""
                lds = stages_x_scale_mem[buf_idx]
                out = []
                row_stride_bytes = scales_per_tile * 4
                for sc in range_constexpr(scales_per_tile):
                    for wm in range_constexpr(wmma_m_rep):
                        row = warp_m_base + arith.index(wm * WMMA_M) + lane16
                        off = row * arith.index(row_stride_bytes) + arith.index(sc * 4)
                        out.append(lds_load_b32_f32(lds, off))
                return out

        w_is_wave_uniform = warp_tile_n <= scale_block_n
        # Hoisted out of `if const_expr(w_is_wave_uniform):` so closures wrapped by the
        # AST rewriter for downstream branches can resolve it.
        wave_n_block = (blk_n + warp_n_base) / arith.index(scale_block_n)

        # Bulk W-scale load: 1 buffer_load_b32 covers 32 K-blocks; chunk-prefetch chain when scale_k > 32.
        if const_expr(variant == "reg_preload"):
            lane_id_full = lane_kgrp * arith.index(16) + lane16

            def _issue_w_chunk_const(chunk_i):
                """Issue one bulk W-scale load for compile-time chunk_i."""
                offset = arith.index(chunk_i * 32)
                idx = wave_n_block * arith.index(scale_k) + lane_id_full + offset
                return buffer_ops.buffer_load(
                    w_scale_buf, idx, vec_width=1, dtype=T.f32
                )

            def _issue_w_chunk_runtime(chunk_idx_i32):
                """Issue one bulk W-scale load for runtime chunk_idx_i32.
                Index is clamped to NUM_W_CHUNKS-1 so out-of-range issues are
                cache-cheap re-loads of the last chunk (never readlane'd)."""
                clamped_i32 = arith.minui(
                    chunk_idx_i32,
                    arith.constant(NUM_W_CHUNKS - 1, type=T.i32),
                )
                offset_i32 = arith.muli(
                    clamped_i32, arith.constant(32, type=T.i32)
                )
                offset = arith.index_cast(T.index, offset_i32)
                idx = wave_n_block * arith.index(scale_k) + lane_id_full + offset
                return buffer_ops.buffer_load(
                    w_scale_buf, idx, vec_width=1, dtype=T.f32
                )

            # Deferred to prologue; _w_readlane resolves these at call time.
            bulk_w_cur = None
            bulk_w_prefetch = None
            cur_chunk_idx_i32 = arith.constant(0, type=T.i32)

        def _w_readlane(kb_i32):
            """Fetch w_scale[wave_n_block, kb] for the experimental variant.
            Single-chunk: direct readlane from bulk_w_cur. Multi-chunk:
            picks bulk_w_cur or bulk_w_prefetch based on kb's chunk index
            (vs. the loop-carried cur_chunk_idx_i32), then readlanes."""
            if const_expr(NUM_W_CHUNKS == 1):
                return rocdl.readlane(T.f32, bulk_w_cur, kb_i32)
            kb_chunk_i32 = arith.shrui(
                kb_i32, arith.constant(5, type=T.i32)
            )
            lane_in_chunk_i32 = arith.andi(
                kb_i32, arith.constant(31, type=T.i32)
            )
            is_cur = arith.cmpi(
                arith.CmpIPredicate.eq, kb_chunk_i32, cur_chunk_idx_i32
            )
            chosen = arith.select(is_cur, bulk_w_cur, bulk_w_prefetch)
            return rocdl.readlane(T.f32, chosen, lane_in_chunk_i32)

        def issue_raw_scales(k_base):
            """Fire buffer_loads for one K-tile's x_scale + w_scale values —
            no multiply. Returns (x_raw, w_raw) flat lists.

            reg_preload / no_op_preload variant: lane16-strided x_scale loads
            (one per wm, 16 unique rows replicated across the upper-half lanes)
            and a direct buffer_load per K-block for w_scale.

            Indexing:
                x_raw[sc * wmma_m_rep + wm] = x_scale[row=wm, kb=sc]
                w_raw[sc * wmma_n_rep + wn] = w_scale[n_block=wn, kb=sc]
                                              (all same value if w_is_wave_uniform)
            """
            kb_base = k_base / arith.index(scale_block_k)
            x_raw = []
            w_raw = []
            for sc in range_constexpr(scales_per_tile):
                kb = kb_base + arith.index(sc)
                for wm in range_constexpr(wmma_m_rep):
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    idx = row * arith.index(scale_k) + kb
                    x_raw.append(
                        buffer_ops.buffer_load(
                            x_scale_buf, idx, vec_width=1, dtype=T.f32
                        )
                    )
                if const_expr(w_is_wave_uniform):
                    idx = wave_n_block * arith.index(scale_k) + kb
                    w_val = buffer_ops.buffer_load(
                        w_scale_buf, idx, vec_width=1, dtype=T.f32
                    )
                    for wn in range_constexpr(wmma_n_rep):
                        w_raw.append(w_val)
                else:
                    for wn in range_constexpr(wmma_n_rep):
                        col = (
                            blk_n
                            + warp_n_base
                            + arith.index(wn * WMMA_N)
                            + lane_kgrp * arith.index(8)
                        )
                        n_block = col / arith.index(scale_block_n)
                        idx = n_block * arith.index(scale_k) + kb
                        w_raw.append(
                            buffer_ops.buffer_load(
                                w_scale_buf, idx, vec_width=1, dtype=T.f32
                            )
                        )
            return x_raw, w_raw

        def issue_raw_scales_for_tile(tile_idx):
            """Issue raw scales for a compile-time tile index (reg_preload)."""
            return issue_raw_scales(arith.index(tile_idx * tile_k))

        def issue_raw_scales_for_future_tile_rt(future_tile_rt):
            """Runtime-safe raw-scale prefetch for dynamic main-loop tiles
            (reg_preload). If `future_tile_rt` is out of range, issue a safe
            in-range load and mask results to zero.
            """
            future_tile_i32 = arith.index_cast(T.i32, future_tile_rt)
            valid_future = arith.cmpi(
                arith.CmpIPredicate.ult,
                future_tile_i32,
                arith.constant(num_k_tiles, type=T.i32),
            )
            safe_tile_i32 = arith.select(
                valid_future, future_tile_i32, arith.constant(0, type=T.i32)
            )
            safe_tile_idx = arith.index_cast(T.index, safe_tile_i32)
            safe_k_base = safe_tile_idx * arith.index(tile_k)
            raw_x, raw_w = issue_raw_scales(safe_k_base)
            masked_x = [arith.select(valid_future, v, scale_zero) for v in raw_x]
            masked_w = [arith.select(valid_future, v, scale_zero) for v in raw_w]
            return masked_x, masked_w

        # W-scale issue: chunk-cached readlane (wave-uniform) or per-(wn) buffer_load.
        def issue_w_raw_scales_experimental(k_base):
            """Returns w_raw flat list, indexed [sc * wmma_n_rep + wn] =
            w_scale[n_block=wn, kb=sc]. All wn entries equal when
            w_is_wave_uniform."""
            kb_base = k_base / arith.index(scale_block_k)
            w_raw = []
            for sc in range_constexpr(scales_per_tile):
                kb = kb_base + arith.index(sc)
                if const_expr(w_is_wave_uniform):
                    kb_i32 = arith.index_cast(T.i32, kb)
                    w_val = _w_readlane(kb_i32)
                    for wn in range_constexpr(wmma_n_rep):
                        w_raw.append(w_val)
                else:
                    for wn in range_constexpr(wmma_n_rep):
                        col = (
                            blk_n
                            + warp_n_base
                            + arith.index(wn * WMMA_N)
                            + lane_kgrp * arith.index(8)
                        )
                        n_block = col / arith.index(scale_block_n)
                        idx = n_block * arith.index(scale_k) + kb
                        w_raw.append(
                            buffer_ops.buffer_load(
                                w_scale_buf, idx, vec_width=1, dtype=T.f32
                            )
                        )
            return w_raw

        def issue_w_raw_scales_for_tile_experimental(tile_idx):
            """W-scales for a compile-time tile index (experimental)."""
            return issue_w_raw_scales_experimental(arith.index(tile_idx * tile_k))

        def issue_w_raw_scales_for_future_tile_rt_experimental(future_tile_rt):
            """Runtime-safe W-scale prefetch for dynamic main-loop tiles
            (experimental). Out-of-range future tiles get zero-masked."""
            future_tile_i32 = arith.index_cast(T.i32, future_tile_rt)
            valid_future = arith.cmpi(
                arith.CmpIPredicate.ult,
                future_tile_i32,
                arith.constant(num_k_tiles, type=T.i32),
            )
            safe_tile_i32 = arith.select(
                valid_future, future_tile_i32, arith.constant(0, type=T.i32)
            )
            safe_tile_idx = arith.index_cast(T.index, safe_tile_i32)
            safe_k_base = safe_tile_idx * arith.index(tile_k)
            raw_w = issue_w_raw_scales_experimental(safe_k_base)
            masked_w = [arith.select(valid_future, v, scale_zero) for v in raw_w]
            return masked_w

        # lane_kgrp selects K-half: kgrp=0 → bytes [0..63], kgrp=1 → [64..127].
        k_half_byte_offset = lane_kgrp * arith.index(64)

        def _compute_lane_bases(warp_base, stride_bytes, num_reps, rep_stride_elems):
            """Compute per-lane LDS byte offsets for loading `num_reps` WMMA
            frags along M or N. Returns a list of base offsets indexed by rep."""
            row_base_bytes = (warp_base + lane16) * arith.index(stride_bytes)
            bases = []
            for rep in range_constexpr(num_reps):
                base = (
                    row_base_bytes
                    + arith.index(rep * rep_stride_elems * stride_bytes)
                    + k_half_byte_offset
                )
                bases.append(base)
            return bases

        def _load_frag(lds_memref, lane_base, ks):
            """Load one WMMA frag (16 × b128) from LDS into a vector<16xi32>
            per lane, starting at byte offset (lane_base + ks * WMMA_K)."""
            k_sub_off = arith.index(ks * WMMA_K)
            off = lane_base + k_sub_off
            v0 = lds_load_b128(lds_memref, off)
            v1 = lds_load_b128(lds_memref, off + arith.index(16))
            v2 = lds_load_b128(lds_memref, off + arith.index(32))
            v3 = lds_load_b128(lds_memref, off + arith.index(48))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        a_lane_bases = _compute_lane_bases(
            warp_m_base, lds_a_stride_bytes, wmma_m_rep, WMMA_M
        )
        b_lane_bases = _compute_lane_bases(
            warp_n_base, lds_b_stride_bytes, wmma_n_rep, WMMA_N
        )

        def load_operand_frags(buffer_idx):
            """Load all A/B frags for one K-tile from LDS stage `buffer_idx`.

            Returns (a_frags, b_frags) with indexing:
                a_frags[ks * wmma_m_rep + wm]
                b_frags[ks * wmma_n_rep + wn]

            Fast (ds_read ~100 cyc per b128) but cannot cross the K-loop
            boundary — frags are loaded once per tile.
            """
            a_lds = stages_a_mem[buffer_idx]
            b_lds = stages_b_mem[buffer_idx]
            a_frags = []
            b_frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_frags.append(_load_frag(a_lds, a_lane_bases[wm], ks))
                for wn in range_constexpr(wmma_n_rep):
                    b_frags.append(_load_frag(b_lds, b_lane_bases[wn], ks))
            return a_frags, b_frags

        # ═══════════════════════════════════════════════════════════════════
        # HELPERS: WMMA compute + scale FMA
        # ═══════════════════════════════════════════════════════════════════

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))

        def compute_wmma_with_frags(global_accs, a_frags, b_frags, x_raw, w_raw):
            """2-deep rolling WMMA/FMA pipeline (reg_preload / no_op_preload).

            Non-transposed WMMA: ISA operand order (B, A, C). Output layout is
            lane→(row=lane16, col_band=lane_kgrp*8+0..7), so each lane's vec<8>
            shares one row → x_scale broadcasts as a scalar.

            Pattern per scale block:
              - seed temp0/temp1 (or just temp0 when n_accs == 1),
              - fold temp0 and issue one new temp each step,
              - fold the remaining temps at the end.
            """

            def issue_wmma_temp(sc, wm, wn):
                temp = acc_zero
                for ks_inner in range_constexpr(wmma_steps_per_scale):
                    ks = sc * wmma_steps_per_scale + ks_inner
                    a_frag = a_frags[ks * wmma_m_rep + wm]
                    b_frag = b_frags[ks * wmma_n_rep + wn]
                    # ISA operand order: (B, A, C), reversed from math.
                    temp = rocdl.wmma_f32_16x16x128_fp8_fp8(
                        T.vec(8, T.f32),
                        b_frag,
                        a_frag,
                        temp,
                    ).result
                return temp

            def compute_scale(wm, wn, sc_x_base, sc_w_base):
                return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

            def wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base):
                scale = compute_scale(wm, wn, sc_x_base, sc_w_base)
                scale_vec = vector.broadcast(T.vec(8, T.f32), scale)
                global_accs[idx] = math_dialect.fma(temp, scale_vec, global_accs[idx])

            for sc in range_constexpr(scales_per_tile):
                sc_x_base = sc * wmma_m_rep
                sc_w_base = sc * wmma_n_rep

                wm0, wn0, idx0 = acc_coords[0]
                # rocdl.s_setprio(2)
                # hold onto a temp wmma to prevent the next instr from using fma on same vgpr (vnop issue)
                temp0 = issue_wmma_temp(sc, wm0, wn0)
                if const_expr(n_accs > 1):
                    wm1, wn1, idx1 = acc_coords[1]
                    temp1 = issue_wmma_temp(sc, wm1, wn1)
                # Might not need this since dscnt 0 is gone
                # rocdl.s_setprio(0)

                if const_expr(n_accs == 1):
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                else:
                    for i in range_constexpr(n_accs - wmma_pipeline_depth):
                        wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                        wm0, wn0, idx0 = wm1, wn1, idx1
                        temp0 = temp1
                        wm1, wn1, idx1 = acc_coords[i + wmma_pipeline_depth]
                        temp1 = issue_wmma_temp(sc, wm1, wn1)
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                    wmma_with_scale(temp1, wm1, wn1, idx1, sc_x_base, sc_w_base)

            return global_accs

        def compute_wmma_with_frags_experimental(
            global_accs, a_frags, b_frags, x_raw, w_raw
        ):
            """2-deep rolling WMMA/FMA pipeline (experimental variant).

            Non-transposed WMMA: ISA operand order (B, A, C). Same WMMA
            output layout as reg_preload — each lane's vec<8> shares one row,
            so the per-row x_scale broadcasts as a scalar at FMA time.

            Pattern per scale block matches the reg_preload version. Kept as
            a separate function so the experimental path can diverge from
            reg_preload independently (e.g., scale-apply rewrites or
            instruction scheduling experiments).
            """

            def issue_wmma_temp(sc, wm, wn):
                temp = acc_zero
                for ks_inner in range_constexpr(wmma_steps_per_scale):
                    ks = sc * wmma_steps_per_scale + ks_inner
                    a_frag = a_frags[ks * wmma_m_rep + wm]
                    b_frag = b_frags[ks * wmma_n_rep + wn]
                    # ISA operand order: (B, A, C), reversed from math.
                    temp = rocdl.wmma_f32_16x16x128_fp8_fp8(
                        T.vec(8, T.f32),
                        b_frag,
                        a_frag,
                        temp,
                    ).result
                return temp

            def compute_scale(wm, wn, sc_x_base, sc_w_base):
                return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

            def wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base):
                scale = compute_scale(wm, wn, sc_x_base, sc_w_base)
                scale_vec = vector.broadcast(T.vec(8, T.f32), scale)
                global_accs[idx] = math_dialect.fma(temp, scale_vec, global_accs[idx])

            for sc in range_constexpr(scales_per_tile):
                sc_x_base = sc * wmma_m_rep
                sc_w_base = sc * wmma_n_rep

                wm0, wn0, idx0 = acc_coords[0]
                # rocdl.s_setprio(2)
                temp0 = issue_wmma_temp(sc, wm0, wn0)
                if const_expr(n_accs > 1):
                    wm1, wn1, idx1 = acc_coords[1]
                    temp1 = issue_wmma_temp(sc, wm1, wn1)
                # rocdl.s_setprio(0)

                if const_expr(n_accs == 1):
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                else:
                    for i in range_constexpr(n_accs - wmma_pipeline_depth):
                        wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                        wm0, wn0, idx0 = wm1, wn1, idx1
                        temp0 = temp1
                        wm1, wn1, idx1 = acc_coords[i + wmma_pipeline_depth]
                        temp1 = issue_wmma_temp(sc, wm1, wn1)
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                    wmma_with_scale(temp1, wm1, wn1, idx1, sc_x_base, sc_w_base)

            return global_accs

        # N_ACCS         — global accumulators (always carried)
        # N_A_FRAGS      — cur_a operand frags (reg_preload only)
        # N_B_FRAGS      — cur_b operand frags (reg_preload only)
        # N_CUR_X_RAW    — current tile raw x_scales
        # N_CUR_W_RAW    — current tile raw w_scales
        # N_PREFETCH_X   — next tile raw x_scales (prefetched)
        # N_PREFETCH_W   — next tile raw w_scales (prefetched)
        N_ACCS = n_accs
        N_A_FRAGS = wmma_m_rep * k_wmma_steps
        N_B_FRAGS = wmma_n_rep * k_wmma_steps
        # Carry shape for x_raw is wmma_m_rep entries per sc across all
        # variants — experimental's TDM scheme materializes from LDS to the
        # same lane16 layout that reg_preload / no_op_preload already use.
        N_CUR_X_RAW = scales_per_tile * wmma_m_rep
        N_CUR_W_RAW = scales_per_tile * wmma_n_rep
        N_PREFETCH_X = N_CUR_X_RAW
        N_PREFETCH_W = N_CUR_W_RAW
        zero_x_raw = [scale_zero] * N_CUR_X_RAW
        zero_w_raw = [scale_zero] * N_CUR_W_RAW

        # This packing/unpacking just sends our vars to the next iteration, stores them cleanly kinda
        def _pack_state_reg_preload(accs_, a_, b_, cur_x_, cur_w_, px, pw):
            return (
                list(accs_)
                + list(a_)
                + list(b_)
                + list(cur_x_)
                + list(cur_w_)
                + list(px)
                + list(pw)
            )

        def _unpack_state_reg_preload(state):
            i = 0
            accs_ = list(state[i : i + N_ACCS])
            i += N_ACCS
            a_ = list(state[i : i + N_A_FRAGS])
            i += N_A_FRAGS
            b_ = list(state[i : i + N_B_FRAGS])
            i += N_B_FRAGS
            cur_x_ = list(state[i : i + N_CUR_X_RAW])
            i += N_CUR_X_RAW
            cur_w_ = list(state[i : i + N_CUR_W_RAW])
            i += N_CUR_W_RAW
            px = list(state[i : i + N_PREFETCH_X])
            i += N_PREFETCH_X
            pw = list(state[i : i + N_PREFETCH_W])
            i += N_PREFETCH_W
            return accs_, a_, b_, cur_x_, cur_w_, px, pw

        def _pack_state_no_op_preload(accs_, cur_x_, cur_w_, px, pw):
            return list(accs_) + list(cur_x_) + list(cur_w_) + list(px) + list(pw)

        def _unpack_state_no_op_preload(state):
            i = 0
            accs_ = list(state[i : i + N_ACCS])
            i += N_ACCS
            cur_x_ = list(state[i : i + N_CUR_X_RAW])
            i += N_CUR_X_RAW
            cur_w_ = list(state[i : i + N_CUR_W_RAW])
            i += N_CUR_W_RAW
            px = list(state[i : i + N_PREFETCH_X])
            i += N_PREFETCH_X
            pw = list(state[i : i + N_PREFETCH_W])
            i += N_PREFETCH_W
            return accs_, cur_x_, cur_w_, px, pw

        # Experimental loop carry: same as reg_preload minus prefetch_x_raw
        # (X-scale prefetch lives in LDS, not VGPRs).
        def _pack_state_experimental(accs_, a_, b_, cur_x_, cur_w_, pw):
            return (
                list(accs_)
                + list(a_)
                + list(b_)
                + list(cur_x_)
                + list(cur_w_)
                + list(pw)
            )

        def _unpack_state_experimental(state):
            i = 0
            accs_ = list(state[i : i + N_ACCS])
            i += N_ACCS
            a_ = list(state[i : i + N_A_FRAGS])
            i += N_A_FRAGS
            b_ = list(state[i : i + N_B_FRAGS])
            i += N_B_FRAGS
            cur_x_ = list(state[i : i + N_CUR_X_RAW])
            i += N_CUR_X_RAW
            cur_w_ = list(state[i : i + N_CUR_W_RAW])
            i += N_CUR_W_RAW
            pw = list(state[i : i + N_PREFETCH_W])
            i += N_PREFETCH_W
            return accs_, a_, b_, cur_x_, cur_w_, pw

        # Prologue: issue TDMs for the first prologue_tiles, X-scale interleaved per-tile.
        lo_x = addr_lo_x_init
        lo_w = addr_lo_w_init
        if const_expr(USES_X_SCALE_TDM):
            lo_x_scale = addr_lo_x_scale_init

        # Boost wave priority for the TDM issue burst to compress wave-dispatch skew.
        rocdl.s_setprio(2)
        for i in range_constexpr(prologue_tiles):
            lo_x, lo_w = issue_tdm_loads(i, lo_x, lo_w)
            if const_expr(USES_X_SCALE_TDM):
                lo_x_scale = issue_x_scale_tdm(i, lo_x_scale)
        rocdl.s_setprio(0)

        accs = [acc_zero] * n_accs

        # MAIN LOOP
        # asm is unrolled in flydsl for range_constexpr, so we see more in asm

        if const_expr(variant == "reg_preload"):
            # Bulk W-scale buffer_load deferred to here (after prologue TDM issues).
            bulk_w_cur = _issue_w_chunk_const(0)
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_prefetch = _issue_w_chunk_const(1)
            else:
                bulk_w_prefetch = bulk_w_cur

            # Single wait: retires tile-0 X+W+S; leaves just-issued tiles pending.
            tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
            gpu.barrier()

            cur_a, cur_b = load_operand_frags(0)
            cur_x_raw = ds_read_x_scales(0)

            # sched_barrier pins readlane after wait so vmcnt is near zero.
            rocdl.sched_barrier(0)
            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            main_loop_end_k = main_loop_iters * num_buffers * tile_k
            if const_expr(main_loop_iters > 0):
                init_state = _pack_state_experimental(
                    accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                )
                if const_expr(USES_W_CHUNK_PREFETCH):
                    init_state = init_state + [
                        bulk_w_cur,
                        bulk_w_prefetch,
                        cur_chunk_idx_i32,
                    ]
                init_state = init_state + [lo_x, lo_w, lo_x_scale]
                for iter_k_base, state in range(
                    0, main_loop_end_k, num_buffers * tile_k, init=init_state
                ):
                    cur_lo_x_scale = state[-1]
                    cur_lo_w = state[-2]
                    cur_lo_x = state[-3]
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-4]
                        bulk_w_prefetch = state[-5]
                        bulk_w_cur = state[-6]
                        _reg_state = state[:-6]
                    else:
                        _reg_state = state[:-3]
                    (
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    ) = _unpack_state_experimental(_reg_state)
                    tile_idx_rt = iter_k_base / arith.index(tile_k)

                    for substage in range_constexpr(num_buffers):
                        # Substage body: issue X+W+scale → wait → ds_read next scale → WMMA (uses cur) → ds_load next frags → rotate.
                        load_buffer = (substage + num_buffers - 1) % num_buffers
                        cur_lo_x, cur_lo_w = issue_tdm_loads(
                            load_buffer, cur_lo_x, cur_lo_w
                        )
                        cur_lo_x_scale = issue_x_scale_tdm(
                            load_buffer, cur_lo_x_scale
                        )

                        # Wait retires t_(k+1) X+W+S; leaves just-issued tile k+nb-1 pending.
                        tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                        gpu.barrier()

                        # Prefetch next substage's scales into VGPRs (overlaps with WMMA via lgkmcnt).
                        next_buffer = (substage + 1) % num_buffers
                        next_x_raw = ds_read_x_scales(next_buffer)

                        cur_accs = compute_wmma_with_frags_experimental(
                            cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                        )

                        # Load next substage's frags + rotate cur_x_raw.
                        cur_a, cur_b = load_operand_frags(next_buffer)
                        cur_x_raw = next_x_raw

                        cur_w_raw = prefetch_w_raw
                        future_tile_rt = tile_idx_rt + arith.index(substage + 2)
                        prefetch_w_raw = (
                            issue_w_raw_scales_for_future_tile_rt_experimental(
                                future_tile_rt
                            )
                        )

                    # End-of-iteration W-scale chunk advance 
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        next_iter_first_tile = tile_idx_rt + arith.index(num_buffers)
                        next_iter_first_kb = (
                            next_iter_first_tile * arith.index(scales_per_tile)
                        )
                        next_iter_first_kb_i32 = arith.index_cast(
                            T.i32, next_iter_first_kb
                        )
                        next_chunk_i32 = arith.shrui(
                            next_iter_first_kb_i32,
                            arith.constant(5, type=T.i32),
                        )
                        need_advance = arith.cmpi(
                            arith.CmpIPredicate.ne,
                            next_chunk_i32,
                            cur_chunk_idx_i32,
                        )
                        new_bulk_w_cur = arith.select(
                            need_advance, bulk_w_prefetch, bulk_w_cur
                        )
                        new_cur_chunk_idx_i32 = next_chunk_i32
                        target_chunk_i32 = arith.addi(
                            new_cur_chunk_idx_i32,
                            arith.constant(1, type=T.i32),
                        )
                        new_bulk_w_prefetch = _issue_w_chunk_runtime(
                            target_chunk_i32
                        )
                        bulk_w_cur = new_bulk_w_cur
                        bulk_w_prefetch = new_bulk_w_prefetch
                        cur_chunk_idx_i32 = new_cur_chunk_idx_i32

                    _new_state = _pack_state_experimental(
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    )
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        _new_state = _new_state + [
                            bulk_w_cur,
                            bulk_w_prefetch,
                            cur_chunk_idx_i32,
                        ]
                    _new_state = _new_state + [cur_lo_x, cur_lo_w, cur_lo_x_scale]
                    results = yield _new_state
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-4]
                    bulk_w_prefetch = results[-5]
                    bulk_w_cur = results[-6]
                    _reg_results = results[:-6]
                else:
                    _reg_results = results[:-3]
                (
                    accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                ) = _unpack_state_experimental(_reg_results)
                lo_x = results[-3]
                lo_w = results[-2]
                lo_x_scale = results[-1]
            else:
                accs = list(accs)

            # Extra tiles — same early-prefetch pattern as the main loop
            # substage: issue X+W+S, wait, ds_read NEXT scale, WMMA, ds_load
            # NEXT frags, rotate cur=next.
            extra_base_tile = main_loop_iters * num_buffers
            for step in range_constexpr(extra_tiles):
                load_tile = extra_base_tile + step + num_buffers - 1
                load_buffer = load_tile % num_buffers
                lo_x, lo_w = issue_tdm_loads(load_buffer, lo_x, lo_w)
                lo_x_scale = issue_x_scale_tdm(load_buffer, lo_x_scale)

                tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                gpu.barrier()

                next_tile = extra_base_tile + step + 1
                next_buffer = next_tile % num_buffers
                next_x_raw = ds_read_x_scales(next_buffer)

                accs = compute_wmma_with_frags_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                )

                cur_a, cur_b = load_operand_frags(next_buffer)
                cur_x_raw = next_x_raw

                next_cur_w_raw = prefetch_w_raw
                future_tile = extra_base_tile + step + 2
                if const_expr(future_tile < num_k_tiles):
                    next_prefetch_w = issue_w_raw_scales_for_tile_experimental(
                        future_tile
                    )
                else:
                    next_prefetch_w = zero_w_raw

                cur_w_raw = next_cur_w_raw
                prefetch_w_raw = next_prefetch_w

            # Drain — no new TDM issues. Wait formula `(nb - 3 - drain_i) * 3`
            # retires one tile's worth of TDMs per step.
            drain_base_tile = extra_base_tile + extra_tiles
            for drain_i in range_constexpr(drain_iters):
                outstanding = (num_buffers - 3 - drain_i) * _TDMS_PER_TILE_EXP
                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                next_tile = drain_base_tile + drain_i + 1
                next_buffer = next_tile % num_buffers
                next_x_raw = ds_read_x_scales(next_buffer)

                accs = compute_wmma_with_frags_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                )

                cur_a, cur_b = load_operand_frags(next_buffer)
                cur_x_raw = next_x_raw

                next_cur_w_raw = prefetch_w_raw
                future_tile = drain_base_tile + drain_i + 2
                if const_expr(future_tile < num_k_tiles):
                    next_prefetch_w = issue_w_raw_scales_for_tile_experimental(
                        future_tile
                    )
                else:
                    next_prefetch_w = zero_w_raw

                cur_w_raw = next_cur_w_raw
                prefetch_w_raw = next_prefetch_w

            # final wmma — uses the last rotated cur values
            accs = compute_wmma_with_frags_experimental(
                accs, cur_a, cur_b, cur_x_raw, cur_w_raw
            )

        else:  # variant 1 (no_op_preload), not tested a lot
            # Initial scale preload (shares the reg_preload helpers — operand
            # frags are loaded fresh per K-tile, but scales still use the
            # lane16-strided layout).
            cur_x_raw, cur_w_raw = issue_raw_scales_for_tile(0)
            if const_expr(num_k_tiles > 1):
                prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(1)
            else:
                prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

            main_loop_end_k = main_loop_iters * num_buffers * tile_k
            if const_expr(main_loop_iters > 0):
                init_state = _pack_state_no_op_preload(
                    accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw
                ) + [lo_x, lo_w]
                for iter_k_base, state in range(
                    0, main_loop_end_k, num_buffers * tile_k, init=init_state
                ):
                    cur_lo_x = state[-2]
                    cur_lo_w = state[-1]
                    cur_accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                        _unpack_state_no_op_preload(state[:-2])
                    )
                    tile_idx_rt = iter_k_base / arith.index(tile_k)

                    for substage in range_constexpr(num_buffers):
                        load_buffer = (substage + num_buffers - 1) % num_buffers
                        cur_lo_x, cur_lo_w = issue_tdm_loads(
                            load_buffer, cur_lo_x, cur_lo_w
                        )

                        compute_stage = substage % num_buffers
                        fresh_a, fresh_b = load_operand_frags(compute_stage)

                        cur_accs = compute_wmma_with_frags(
                            cur_accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw
                        )

                        tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                        gpu.barrier()

                        cur_x_raw = prefetch_x_raw
                        cur_w_raw = prefetch_w_raw

                        future_tile_rt = tile_idx_rt + arith.index(substage + 2)
                        prefetch_x_raw, prefetch_w_raw = (
                            issue_raw_scales_for_future_tile_rt(future_tile_rt)
                        )

                    results = yield _pack_state_no_op_preload(
                        cur_accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw
                    ) + [cur_lo_x, cur_lo_w]
                accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                    _unpack_state_no_op_preload(results[:-2])
                )
                lo_x = results[-2]
                lo_w = results[-1]
            else:
                accs = list(accs)

            # Extra tiles: if main loop iterations doesnt cleanly divide in the
            # const_expr loop then we need this for the final buffers
            extra_base_tile = main_loop_iters * num_buffers
            for step in range_constexpr(extra_tiles):
                load_tile = extra_base_tile + step + num_buffers - 1
                load_buffer = load_tile % num_buffers
                lo_x, lo_w = issue_tdm_loads(load_buffer, lo_x, lo_w)

                compute_stage = (extra_base_tile + step) % num_buffers
                fresh_a, fresh_b = load_operand_frags(compute_stage)

                accs = compute_wmma_with_frags(
                    accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw
                )

                tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                gpu.barrier()

                cur_x_raw = prefetch_x_raw
                cur_w_raw = prefetch_w_raw

                future_tile = extra_base_tile + step + 2
                if const_expr(future_tile < num_k_tiles):
                    prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(
                        future_tile
                    )
                else:
                    prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

            # Drain
            drain_base_tile = extra_base_tile + extra_tiles
            for drain_i in range_constexpr(drain_iters):
                outstanding = (num_buffers - 3 - drain_i) * 2
                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                compute_stage = (drain_base_tile + drain_i) % num_buffers
                fresh_a, fresh_b = load_operand_frags(compute_stage)

                accs = compute_wmma_with_frags(
                    accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw
                )

                cur_x_raw = prefetch_x_raw
                cur_w_raw = prefetch_w_raw

                future_tile = drain_base_tile + drain_i + 2
                if const_expr(future_tile < num_k_tiles):
                    prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(
                        future_tile
                    )
                else:
                    prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

            # Final wmma
            final_tile = drain_base_tile + drain_iters
            final_compute_stage = final_tile % num_buffers
            fresh_a, fresh_b = load_operand_frags(final_compute_stage)
            accs = compute_wmma_with_frags(accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw)

        # Step 4: convert f32 accs to out_dtype, buffer_store to Y.
        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)

        out_elem = (
            T.bf16 if out_dtype == "bf16" else T.f16 if out_dtype == "fp16" else None
        )
        is_half_out = out_dtype in ("bf16", "fp16")

        if use_tdm_store:
            d_lds_elem_ty = T.bf16 if out_dtype != "fp16" else T.f16
            d_lds_elems = total_d_bytes // 2
            d_smem = SmemPtr(
                d_lds_allocator.get_base(), 0, d_lds_elem_ty, shape=(d_lds_elems,)
            )
            d_lds_buffer = d_smem.get()

            row_lds = warp_m_base + lane16  # warp_m_base = wave_m_idx * warp_tile_m
            col_lds = warp_n_base + lane_kgrp * arith.index(8)  # bf16 col within row
            d_lane_base = row_lds * arith.index(_lds_d_stride_elems_d) + col_lds
            if not is_half_out:
                d_lane_base = (
                    row_lds * arith.index(_lds_d_stride_elems_d)
                    + warp_n_base * arith.index(elem_bytes_d // 2)
                    + lane_kgrp * arith.index(4 * elem_bytes_d)
                )

            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    imm = wm * WMMA_M * _lds_d_stride_elems_d + wn * _n_col_d_elems_d
                    store_acc_vec8_to_lds(
                        d_lds_buffer,
                        d_lane_base,
                        imm,
                        accs[idx],
                        out_elem=out_elem,
                    )

            rocdl.s_wait_dscnt(0)
            gpu.barrier()

            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_y,
                lds_memref=d_lds_buffer,
                global_offset=(blk_m, blk_n),
                tensor_shape=(tile_m, tile_n),
                strides=(N, 1),
                tile_shape=(tile_m, tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=num_warps,
                for_store=True,
            )
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = (
                        blk_n
                        + warp_n_base
                        + arith.index(wn * WMMA_N)
                        + lane_kgrp * arith.index(8)
                    )

                    if is_half_out:
                        c_off_bytes = (row * n_stride + col_base) * arith.index(
                            elem_bytes_d
                        )
                        store_acc_vec8_to_buffer(
                            accs[idx],
                            y_buf,
                            c_off_bytes,
                            out_elem=out_elem,
                            offset_is_bytes=True,
                        )
                    else:
                        offsets = []
                        for half in range_constexpr(2):
                            col = col_base + arith.index(half * 4)
                            offsets.append(row * n_stride + col)
                        store_acc_vec8_to_buffer(accs[idx], y_buf, offsets)

    cache_tag = (
        K,
        N,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        scale_block_k,
        scale_block_n,
        num_buffers,
        effective_waves_per_eu,
        l2_prefetch_distance,
        out_dtype,
        variant,
        use_tdm_store,
        loop_carried_load_percent,
        kernarg_preload,
    )

    @flyc.jit
    def launch_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag

        ctx = CompilationContext.get_current()
        all_allocators = list(stage_allocators)
        if use_tdm_store:
            all_allocators.append(d_lds_allocator)
        with ir.InsertionPoint(ctx.gpu_module_body):
            for alloc in all_allocators:
                alloc.finalized = False
            for alloc in all_allocators:
                alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        launcher = kernel_gemm_a8w8_blockscale(
            arg_y, arg_x, arg_w, arg_x_scale, arg_w_scale, i32_m, i32_n
        )

        if effective_waves_per_eu is not None:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    wpe = int(effective_waves_per_eu)
                    if wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), wpe
                        )

        flat_wg_attr = ir.StringAttr.get(f"{block_threads},{block_threads}")
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        # Experimental, loop_carried_load_percent
        if loop_carried_load_percent is not None:
            lcv = ir.ArrayAttr.get(
                [
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("amdgpu-loop-carried-load-percent"),
                            ir.StringAttr.get(str(int(loop_carried_load_percent))),
                        ]
                    )
                ]
            )
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["passthrough"] = lcv

        # Mark kernel args as inreg so AMDGPU preloads them into user SGPRs at dispatch.
        if kernarg_preload:
            inreg_attr = ir.UnitAttr.get()
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    num_args = len(op.regions[0].blocks[0].arguments)
                    per_arg = [
                        ir.DictAttr.get({"llvm.inreg": inreg_attr})
                        for _ in range(num_args)
                    ]
                    op.attributes["arg_attrs"] = ir.ArrayAttr.get(per_arg)

        launcher.launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm_a8w8_blockscale


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    y: torch.Tensor = None,
    dtype: torch.dtype = torch.bfloat16,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
    variant: str = "reg_preload",
    use_tdm_store: bool = False,
    loop_carried_load_percent: Optional[int] = None,
    kernarg_preload: bool = False,
):
    """Compute Y = (X @ W^T) with per-block f32 scales (A8W8 blockscale).

    variant: "reg_preload" (default) or "no_op_preload".
      - "reg_preload"   : loop-carry cur_a/cur_b operand frags across iters.
                          W-scales bulk-loaded via buffer_load_b32 + readlane,
                          X-scales TDM-staged into LDS. Requires
                          w_is_wave_uniform.
      - "no_op_preload" : operand frags loaded fresh from LDS each iter
                          (~256 VGPRs cheaper). Uses the legacy per-K-block
                          buffer_load scale path.
    """
    assert x.ndim == 2 and w.ndim == 2, "X and W must be 2D"
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"K mismatch: X has {K}, W has {K_w}"

    assert x_scale.ndim == 2 and w_scale.ndim == 2, "scales must be 2D"
    assert x_scale.shape[0] == M, f"x_scale rows {x_scale.shape[0]} != M {M}"
    scale_k_x = x_scale.shape[1]
    scale_n, scale_k_w = w_scale.shape
    assert (
        scale_k_x == scale_k_w
    ), f"scale_k mismatch: x_scale has {scale_k_x}, w_scale has {scale_k_w}"
    scale_k = scale_k_x

    def _next_pow2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    scale_block_k_derived = _next_pow2((K + scale_k - 1) // scale_k)
    scale_block_n_derived = _next_pow2((N + scale_n - 1) // scale_n)

    torch_to_str = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "f32",
    }
    assert dtype in torch_to_str, f"Unsupported output dtype {dtype}"
    out_dtype_str = torch_to_str[dtype]

    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    if K_padded != K:
        pad_size = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad_size))
        w = torch.nn.functional.pad(w, (0, pad_size))
        new_scale_k = K_padded // scale_block_k_derived
        scale_pad = new_scale_k - scale_k
        if scale_pad > 0:
            x_scale = torch.nn.functional.pad(x_scale, (0, scale_pad))
            w_scale = torch.nn.functional.pad(w_scale, (0, scale_pad))
        K = K_padded

    # Pad N up to tile_n so the kernel's WMMAs and stores land inside
    # the allocated output
    N_stride = ((N + tile_n - 1) // tile_n) * tile_n

    if y is not None:
        assert y.shape == (M, N), f"y shape {y.shape} != ({M}, {N})"
        assert y.dtype == dtype, f"y dtype {y.dtype} != {dtype}"

    if N_stride != N:
        y_buf = torch.empty((M, N_stride), dtype=dtype, device=x.device)
    elif y is not None:
        y_buf = y
    else:
        y_buf = torch.empty((M, N), dtype=dtype, device=x.device)

    launcher = compile_gemm_a8w8_blockscale(
        K=K,
        N=N_stride,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        scale_block_k=scale_block_k_derived,
        scale_block_n=scale_block_n_derived,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        out_dtype=out_dtype_str,
        variant=variant,
        use_tdm_store=use_tdm_store,
        loop_carried_load_percent=loop_carried_load_percent,
        kernarg_preload=kernarg_preload,
    )

    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    launcher(y_buf, x, w, x_scale, w_scale, M, N_stride, stream=stream)

    if N_stride != N:
        result = y_buf[:, :N]
        if y is not None:
            y.copy_(result)
            return y
        return result
    return y_buf


__all__ = [
    "compile_gemm_a8w8_blockscale",
    "gemm_a8w8_blockscale",
]
