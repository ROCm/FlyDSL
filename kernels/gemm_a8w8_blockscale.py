"""A8W8 FP8 blockscale GEMM for gfx1250."""

from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect, scf as scf_dialect
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

# ═══════════════════════════════════════════════════════════════════════════
# DEBUG: no-op ALL scheduling hints — sched_barrier, sched_group_barrier, and
# s_setprio (incl. the row/top-wall 0x6 fences and everything in _shadow_quota)
# — so the bare scheduler (coexec) runs unguided. Everything else passes through
# to the real rocdl (s_barrier_signal/wait, wmma, readlane, … stay functional).
# TO RESTORE THE HINTS — delete this whole block (or cp the .hints_backup).
# ═══════════════════════════════════════════════════════════════════════════
_real_rocdl = rocdl


class _NoSchedHints:
    sched_barrier = staticmethod(lambda *a, **k: None)
    sched_group_barrier = staticmethod(lambda *a, **k: None)
    s_setprio = staticmethod(lambda *a, **k: None)

    def __getattr__(self, name):
        return getattr(_real_rocdl, name)


rocdl = _NoSchedHints()
# ═══════════════════════════════════════════════════════════════════════════

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
FRAG_VGPRS = 16
DS_LOADS_PER_FRAG = 4
LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16
LDS_PAD_D_BYTES = 16
_STAGE_NAMES = ("ping", "pong", "pang", "pung")

def _lds_vec_type(memref, total_bits):
    raw_mr = arith.unwrap(memref)
    elem_type = ir.MemRefType(raw_mr.type).element_type
    elem_bits = get_mlir_type_size(elem_type) * 8
    n = total_bits // elem_bits
    return ir.VectorType.get([n], elem_type)

def lds_load_b128(memref, elem_off):
    vec_ty = _lds_vec_type(memref, 128)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    return vector.bitcast(
        ir.VectorType.get([4], ir.IntegerType.get_signless(32)), loaded
    )

def lds_load_b32_f32(memref, elem_off):
    vec_ty = _lds_vec_type(memref, 32)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    as_f32_vec = vector.bitcast(
        ir.VectorType.get([1], ir.F32Type.get()), loaded
    )
    return vector.extract(as_f32_vec, static_position=[0], dynamic_position=[])

def lds_store_b128(memref, elem_off, data):
    vec_ty = _lds_vec_type(memref, 128)
    typed_vec = vector.bitcast(vec_ty, data)
    vector.store(typed_vec, memref, [elem_off])

def _disable_unroll_on_enclosing_loop():
    block = ir.InsertionPoint.current.block
    op = block.owner
    if op.name != "scf.for":
        return
    anno = ir.Attribute.parse(
        "#llvm.loop_annotation<unroll = <disable = true>, "
        "disableNonforced = true>"
    )
    op.attributes["loop_annotation"] = anno

def store_acc_vec8_to_lds(memref, base_elem_off, imm_elem_off, acc_vec8, out_elem=None):
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

def _enable_expert_sched_mode():
    from flydsl._mlir.dialects import llvm as _llvm

    # hwreg encoding: ID=26(SCHED_MODE), Offset=0, Size=2 -> 26 | (1<<11) = 2074
    imm_val = arith.unwrap(arith.constant(2074, type=T.i32))
    val_val = arith.unwrap(arith.constant(2, type=T.i32))
    _llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm_val, val_val], [], [])

def _disable_xdl_arb_stall():
    from flydsl._mlir.dialects import llvm as _llvm

    imm_val = arith.unwrap(arith.constant(154, type=T.i32))
    val_val = arith.unwrap(arith.constant(1, type=T.i32))
    _llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm_val, val_val], [], [])

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
    wmma_operand_reuse: bool = False,
    use_manual_barrier: bool = False,
    a_resident_refill: bool = False,
):
    if variant not in ("reg_preload", "manual"):
        raise ValueError(
            f"variant must be 'reg_preload' or 'manual', got {variant!r}"
        )
    if a_resident_refill and variant != "manual":
        raise ValueError(
            "a_resident_refill=True is only supported for variant='manual'"
        )
    if const_expr(variant == "reg_preload"):
        _w_is_wave_uniform = (tile_n // n_warp) <= scale_block_n
        if not _w_is_wave_uniform:
            raise ValueError(
                f"variant='reg_preload' requires warp_tile_n ({tile_n // n_warp}) "
                f"<= scale_block_n ({scale_block_n}) (W-scale must be wave-uniform)"
            )
    if const_expr(variant == "manual"):
        # B-resident streaming-A is OPTIMAL at wmma_n_rep >= 2 and even (balanced
        # boot/stream B split + 2addr pairing + enough WMMA shadows to drip the
        # T+1 prefetch). wmma_n_rep == 1 (or odd) still computes CORRECTLY but
        # SUBOPTIMALLY: the prefetch tail-clumps (drained at the end-of-sweep
        # flush, not dripped) and there's no boot/stream split. Allowed for
        # small-tile / skinny-M experiments. wmma_m_rep >= 1 required.
        _warp_tile_m = tile_m // m_warp
        _warp_tile_n = tile_n // n_warp
        _wmma_m_rep = _warp_tile_m // WMMA_M
        _wmma_n_rep = _warp_tile_n // WMMA_N
        if _wmma_m_rep < 1:
            raise ValueError(
                f"variant='manual' requires wmma_m_rep ({_wmma_m_rep}) ≥ 1 "
                f"(warp_tile_m={_warp_tile_m})"
            )
        if _wmma_n_rep < 1:
            raise ValueError(
                f"variant='manual' requires wmma_n_rep ({_wmma_n_rep}) ≥ 1 "
                f"(warp_tile_n={_warp_tile_n}); note wmma_n_rep < 2 or odd is "
                f"correct but suboptimal (prefetch does not drip-overlap)"
            )
        # B-resident compute supports scales_per_tile >= 1 via outer-sc loop.
        _scales_per_tile_m = tile_k // scale_block_k
        if _scales_per_tile_m < 1:
            raise ValueError(
                f"variant='manual' requires scales_per_tile >= 1 "
                f"(tile_k={tile_k}, scale_block_k={scale_block_k} → "
                f"scales_per_tile={_scales_per_tile_m})"
            )
        if a_resident_refill and _scales_per_tile_m != 1:
            raise ValueError(
                "a_resident_refill=True requires scales_per_tile == 1 "
                f"(tile_k must equal scale_block_k); got tile_k={tile_k}, "
                f"scale_block_k={scale_block_k} → "
                f"scales_per_tile={_scales_per_tile_m}"
            )
        # Manual also assumes wave-uniform W (same readlane path as reg_preload).
        _w_is_wave_uniform_m = _warp_tile_n <= scale_block_n
        if not _w_is_wave_uniform_m:
            raise ValueError(
                f"variant='manual' requires warp_tile_n ({_warp_tile_n}) "
                f"<= scale_block_n ({scale_block_n}) (W-scale must be wave-uniform)"
            )
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

    _USES_REG_W = True
    NUM_W_CHUNKS = (scale_k + 31) // 32
    USES_W_CHUNK_PREFETCH = NUM_W_CHUNKS > 1

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
    lds_b_stride_bytes = tile_k * 16  # per-stripe size in LDS bytes
    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = (tile_n // 16) * lds_b_stride_bytes

    # X-scale LDS (TDM-staged): tile_m rows × scales_per_tile × 4B per stage.
    USES_X_SCALE_TDM = True
    lds_x_scale_row_bytes = scales_per_tile * 4
    lds_x_scale_data_bytes = tile_m * lds_x_scale_row_bytes

    unified_alloc = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="a8w8bs_unified"
    )
    unified_a_off = unified_alloc._align(unified_alloc.ptr, 16)
    unified_alloc.ptr = unified_a_off + num_buffers * lds_a_data_bytes
    unified_b_off = unified_alloc._align(unified_alloc.ptr, 16)
    unified_alloc.ptr = unified_b_off + num_buffers * lds_b_data_bytes
    if USES_X_SCALE_TDM:
        unified_x_scale_off = unified_alloc._align(unified_alloc.ptr, 16)
        unified_alloc.ptr = (
            unified_x_scale_off + num_buffers * lds_x_scale_data_bytes
        )
    else:
        unified_x_scale_off = 0

    stage_a_data_off = [
        unified_a_off + i * lds_a_data_bytes for i in range(num_buffers)
    ]
    stage_b_data_off = [
        unified_b_off + i * lds_b_data_bytes for i in range(num_buffers)
    ]
    if USES_X_SCALE_TDM:
        stage_x_scale_off = [
            unified_x_scale_off + i * lds_x_scale_data_bytes
            for i in range(num_buffers)
        ]
    else:
        stage_x_scale_off = []
    stage_allocators = [unified_alloc] * num_buffers

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
    # CHANGE 1 (reg_preload): prologue now fills ALL NB buffers, so one extra
    # tile is drained in the epilogue (NB-1 drain iters + 1 final WMMA = NB tiles).
    drain_iters = num_buffers - 1

    # 3 TDMs per tile (X + W + X-scale) for both surviving variants.
    _TDMS_PER_TILE_EXP = 3
    # CHANGE 1 (reg_preload): prologue fills ALL NB buffers and the steady-loop
    # lookahead is NB, so each wait (prologue + main loop, both below the TDM
    # issue) leaves NB-1 tiles in flight.
    MAIN_TDM_OUTSTANDING_EXPERIMENTAL = (num_buffers - 1) * _TDMS_PER_TILE_EXP

    MANUAL_NB_EFF = min(num_buffers, num_k_tiles)
    MANUAL_PROLOGUE_TILES = MANUAL_NB_EFF
    MANUAL_TDM_OUTSTANDING = (num_buffers - 1) * _TDMS_PER_TILE_EXP


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

        big_a = SmemPtr(
            unified_alloc.get_base(),
            unified_a_off,
            T.f8,
            shape=(num_buffers * lds_a_data_bytes,),
        )
        big_b = SmemPtr(
            unified_alloc.get_base(),
            unified_b_off,
            T.f8,
            shape=(num_buffers * lds_b_data_bytes,),
        )
        big_a_mem = big_a.get()
        big_b_mem = big_b.get()
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
                global_offset=(blk_n // arith.index(16), k_base * arith.index(16)),
                tensor_shape=(tile_n // 16, tile_k * 16),
                strides=(K * 16, 1),
                tile_shape=(tile_n // 16, tile_k * 16),
                elem_bytes=1,
                pad_interval=tile_k * 16,
                pad_amount=0,
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

        a_lds_base_i32 = vector.extract(
            _make_desc_x(stages_a_mem[0], arith.index(0)).dgroup0,
            static_position=[1],
            dynamic_position=[],
        )
        b_lds_base_i32 = vector.extract(
            _make_desc_w(stages_b_mem[0], arith.index(0)).dgroup0,
            static_position=[1],
            dynamic_position=[],
        )
        slot_stride_a_i32 = arith.constant(lds_a_data_bytes, type=T.i32)
        slot_stride_b_i32 = arith.constant(lds_b_data_bytes, type=T.i32)

        # Per-K-tile lo32 advance: X=tile_k bytes, W=tile_k*16 (cycle-major, 16 B per K cell).
        adv_x_i32 = arith.constant(tile_k, type=T.i32)
        adv_w_i32 = arith.constant(tile_k * 16, type=T.i32)
        pred_const = arith.constant(1, type=T.i32)

        def _buf_idx_to_i32(buf_idx):
            if const_expr(isinstance(buf_idx, int)):
                return arith.constant(buf_idx, type=T.i32)
            else:
                return buf_idx

        def issue_tdm_loads(buf_idx, lo_x, lo_w):
            buf_i32 = _buf_idx_to_i32(buf_idx)
            a_addr = arith.addi(
                a_lds_base_i32, arith.muli(buf_i32, slot_stride_a_i32)
            )
            b_addr = arith.addi(
                b_lds_base_i32, arith.muli(buf_i32, slot_stride_b_i32)
            )
            dg0_x = vector.from_elements(
                T.vec(4, T.i32), [pred_const, a_addr, lo_x, addr_hi_x]
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_x, dgroup1_x))
            dg0_w = vector.from_elements(
                T.vec(4, T.i32), [pred_const, b_addr, lo_w, addr_hi_w]
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_w, dgroup1_w))
            return arith.addi(lo_x, adv_x_i32), arith.addi(lo_w, adv_w_i32)

        # ── X-scale TDM descriptor + LDS staging (hoisted) ──────────────────
        if const_expr(USES_X_SCALE_TDM):
            # One big memref spanning all num_buffers slots.
            big_x_scale = SmemPtr(
                unified_alloc.get_base(),
                unified_x_scale_off,
                T.f8,
                shape=(num_buffers * lds_x_scale_data_bytes,),
            )
            big_x_scale_mem = big_x_scale.get()
            # Per-stage memref for descriptor construction (slot 0).
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

            x_scale_lds_base_i32 = vector.extract(
                _desc_x_scale_init.dgroup0,
                static_position=[1],
                dynamic_position=[],
            )
            slot_stride_x_scale_i32 = arith.constant(
                lds_x_scale_data_bytes, type=T.i32
            )

            # K-axis advance per K-tile: scales_per_tile K-blocks × 4 B/elem.
            adv_x_scale_i32 = arith.constant(scales_per_tile * 4, type=T.i32)

            def issue_x_scale_tdm(buf_idx, lo_x_scale):
                buf_i32 = _buf_idx_to_i32(buf_idx)
                xs_addr = arith.addi(
                    x_scale_lds_base_i32,
                    arith.muli(buf_i32, slot_stride_x_scale_i32),
                )
                dg0 = vector.from_elements(
                    T.vec(4, T.i32),
                    [pred_const, xs_addr, lo_x_scale, addr_hi_x_scale],
                )
                tdm_ops.tensor_load_2d(
                    tdm_ops.TDMDescriptor2D(dg0, dgroup1_x_scale)
                )
                return arith.addi(lo_x_scale, adv_x_scale_i32)

            def ds_read_x_scales(buf_idx):
                if const_expr(isinstance(buf_idx, int)):
                    slot_byte_off = arith.index(buf_idx * lds_x_scale_data_bytes)
                else:
                    slot_byte_off = arith.index_cast(
                        T.index,
                        arith.muli(buf_idx, slot_stride_x_scale_i32),
                    )
                out = []
                row_stride_bytes = scales_per_tile * 4
                for sc in range_constexpr(scales_per_tile):
                    for wm in range_constexpr(wmma_m_rep):
                        row = warp_m_base + arith.index(wm * WMMA_M) + lane16
                        off = (
                            slot_byte_off
                            + row * arith.index(row_stride_bytes)
                            + arith.index(sc * 4)
                        )
                        out.append(lds_load_b32_f32(big_x_scale_mem, off))
                return out

        w_is_wave_uniform = warp_tile_n <= scale_block_n
        wave_n_block = (blk_n + warp_n_base) / arith.index(scale_block_n)

        # Bulk W-scale load: 1 buffer_load_b32 covers 32 K-blocks; chunk-prefetch chain when scale_k > 32.
        if const_expr(_USES_REG_W):
            lane_id_full = lane_kgrp * arith.index(16) + lane16

            def _issue_w_chunk_const(chunk_i):
                offset = arith.index(chunk_i * 32)
                idx = wave_n_block * arith.index(scale_k) + lane_id_full + offset
                return buffer_ops.buffer_load(
                    w_scale_buf, idx, vec_width=1, dtype=T.f32
                )

            def _issue_w_chunk_runtime(chunk_idx_i32):
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
            bulk_w_chunks = None
            cur_chunk_idx_i32 = arith.constant(0, type=T.i32)

        def _w_readlane(kb_i32):
            if const_expr(NUM_W_CHUNKS == 1):
                return rocdl.readlane(T.f32, bulk_w_chunks[0], kb_i32)
            kb_chunk_i32 = arith.shrui(kb_i32, arith.constant(5, type=T.i32))
            lane_in_chunk_i32 = arith.andi(kb_i32, arith.constant(31, type=T.i32))
            chosen = bulk_w_chunks[NUM_W_CHUNKS - 1]
            for _c in range_constexpr(NUM_W_CHUNKS - 1):
                is_c = arith.cmpi(
                    arith.CmpIPredicate.eq, kb_chunk_i32,
                    arith.constant(_c, type=T.i32),
                )
                chosen = arith.select(is_c, bulk_w_chunks[_c], chosen)
            return rocdl.readlane(T.f32, chosen, lane_in_chunk_i32)

        # W-scale issue: chunk-cached readlane (wave-uniform) or per-(wn) buffer_load.

        # lane_kgrp selects K-half: kgrp=0 → bytes [0..63], kgrp=1 → [64..127].
        k_half_byte_offset = lane_kgrp * arith.index(64)

        b_k_half_byte_offset = lane_kgrp * arith.index(256)

        def _compute_lane_bases(warp_base, stride_bytes, num_reps, rep_stride_elems):
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

        def _compute_b_lane_bases_preshuffled(warp_base, num_reps):
            bases = []
            for rep in range_constexpr(num_reps):
                # warp_base is the starting N-element index; each rep advances by WMMA_N=16
                stripe_offset = (
                    warp_base + arith.index(rep * WMMA_N)
                ) / arith.index(WMMA_N) * arith.index(lds_b_stride_bytes)
                base = (
                    stripe_offset
                    + b_k_half_byte_offset
                    + lane16 * arith.index(16)
                )
                bases.append(base)
            return bases

        def _assemble_frag(parts):
            v01 = vector.shuffle(parts[0], parts[1], list(range(8)))
            v23 = vector.shuffle(parts[2], parts[3], list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        def _frag_load_thunks(lds_memref, lane_base, ks, cycle_stride_bytes=16, ks_stride_bytes=WMMA_K,
                              const_off=0):
            base_off = ks * ks_stride_bytes + const_off
            return [
                lambda: lds_load_b128(lds_memref, lane_base + arith.index(base_off)),
                lambda: lds_load_b128(lds_memref, lane_base + arith.index(base_off + cycle_stride_bytes)),
                lambda: lds_load_b128(lds_memref, lane_base + arith.index(base_off + cycle_stride_bytes * 2)),
                lambda: lds_load_b128(lds_memref, lane_base + arith.index(base_off + cycle_stride_bytes * 3)),
            ]

        def _load_frag(lds_memref, lane_base, ks, cycle_stride_bytes=16, ks_stride_bytes=WMMA_K,
                       const_off=0):
            # Eager fragment load (all 4 parts now) — byte-identical to before.
            thunks = _frag_load_thunks(
                lds_memref, lane_base, ks, cycle_stride_bytes, ks_stride_bytes, const_off
            )
            return _assemble_frag([t() for t in thunks])

        a_lane_bases = _compute_lane_bases(
            warp_m_base, lds_a_stride_bytes, wmma_m_rep, WMMA_M
        )
        b_lane_bases = _compute_b_lane_bases_preshuffled(
            warp_n_base, wmma_n_rep
        )

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))

        def issue_wmma_step(sc, wm, wn, a_frags, b_frags):
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

        def compute_scale_step(sc, wm, wn, x_raw, w_raw):
            sc_x_base = sc * wmma_m_rep
            sc_w_base = sc * wmma_n_rep
            return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

        def apply_scale(temp, scale, acc):
            scale_vec = vector.broadcast(T.vec(8, T.f32), scale)
            return math_dialect.fma(temp, scale_vec, acc)

        N_ACCS = n_accs
        N_A_FRAGS = wmma_m_rep * k_wmma_steps
        N_B_FRAGS = wmma_n_rep * k_wmma_steps
        # x_raw carry: wmma_m_rep entries per sc (lane16-strided layout).
        N_CUR_X_RAW = scales_per_tile * wmma_m_rep
        N_CUR_W_RAW = scales_per_tile * wmma_n_rep
        N_PREFETCH_X = N_CUR_X_RAW
        N_PREFETCH_W = N_CUR_W_RAW
        zero_x_raw = [scale_zero] * N_CUR_X_RAW
        zero_w_raw = [scale_zero] * N_CUR_W_RAW

        # Prologue + accs init live inside each variant branch.
        if const_expr(variant == "reg_preload"):
            # ───── reg_preload-only helpers ─────
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

            def issue_w_raw_scales_experimental(k_base):
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
                return issue_w_raw_scales_experimental(arith.index(tile_idx * tile_k))

            def issue_w_raw_scales_for_future_tile_rt_experimental(future_tile_rt):
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

            def load_operand_frags_with_xscale_interleave(buffer_idx):
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                    slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
                else:
                    slot_off_a = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                    )
                    slot_off_b = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                    )
                a_frags = []
                b_frags = []
                x_raw = None
                for ks in range_constexpr(k_wmma_steps):
                    for wm in range_constexpr(wmma_m_rep):
                        a_frags.append(
                            _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                        )
                    for wn in range_constexpr(wmma_n_rep):
                        b_frags.append(
                            _load_frag(big_b_mem, b_lane_bases[wn] + slot_off_b, ks, cycle_stride_bytes=512, ks_stride_bytes=2048)
                        )
                    # Pin X-scale ds_read right after K-step 0's frag loads.
                    if const_expr(ks == 0):
                        rocdl.sched_barrier(0)
                        x_raw = ds_read_x_scales(buffer_idx)
                        rocdl.sched_barrier(0)
                return a_frags, b_frags, x_raw

            def load_operand_frags(buffer_idx):
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                    slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
                else:
                    slot_off_a = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                    )
                    slot_off_b = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                    )
                a_frags = []
                b_frags = []
                for ks in range_constexpr(k_wmma_steps):
                    for wm in range_constexpr(wmma_m_rep):
                        a_frags.append(
                            _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                        )
                    for wn in range_constexpr(wmma_n_rep):
                        b_frags.append(
                            _load_frag(big_b_mem, b_lane_bases[wn] + slot_off_b, ks, cycle_stride_bytes=512, ks_stride_bytes=2048)
                        )
                return a_frags, b_frags

            def load_a_frags(buffer_idx):
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                else:
                    slot_off_a = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                    )
                a_frags = []
                for ks in range_constexpr(k_wmma_steps):
                    for wm in range_constexpr(wmma_m_rep):
                        a_frags.append(
                            _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                        )
                return a_frags

            def load_b_frags(buffer_idx):
                # B-only load, split out so the main loop can prefetch B into its
                # OWN vgprs (double-buffered) the moment the n+1 tensor_wait fires.
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
                else:
                    slot_off_b = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                    )
                b_frags = []
                for ks in range_constexpr(k_wmma_steps):
                    for wn in range_constexpr(wmma_n_rep):
                        b_frags.append(
                            _load_frag(big_b_mem, b_lane_bases[wn] + slot_off_b, ks, cycle_stride_bytes=512, ks_stride_bytes=2048)
                        )
                return b_frags

            def compute_wmma_with_frags_experimental(
                global_accs, a_frags, b_frags, x_raw, w_raw
            ):

                def issue_wmma_temp(sc, wm, wn):
                    return issue_wmma_step(sc, wm, wn, a_frags, b_frags)

                def compute_scale(wm, wn, sc_x_base, sc_w_base):
                    return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

                def wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base):
                    scale = compute_scale(wm, wn, sc_x_base, sc_w_base)
                    global_accs[idx] = apply_scale(temp, scale, global_accs[idx])

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

            # PROLOGUE — pre-fill state for main-loop iter 0.
            lo_x = addr_lo_x_init
            lo_w = addr_lo_w_init
            lo_x_scale = addr_lo_x_scale_init

            # Boost wave priority for the TDM issue burst to compress wave-dispatch skew.
            rocdl.s_setprio(2)
            # CHANGE 1: fill all NB buffers in the prologue (was NB-1).
            for i in range_constexpr(num_buffers):
                lo_x, lo_w = issue_tdm_loads(i, lo_x, lo_w)
                lo_x_scale = issue_x_scale_tdm(i, lo_x_scale)
            rocdl.s_setprio(0)

            bulk_w_chunks = [_issue_w_chunk_const(c) for c in range(NUM_W_CHUNKS)]
            bulk_w_cur = bulk_w_chunks[0]
            bulk_w_prefetch = bulk_w_chunks[1] if USES_W_CHUNK_PREFETCH else bulk_w_chunks[0]

            # Single wait: retires tile-0 X+W+S; leaves NB-1 tiles pending.
            tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
            gpu.barrier()

            cur_a, cur_b, cur_x_raw = load_operand_frags_with_xscale_interleave(0)

            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            accs = [acc_zero] * n_accs

            # CHANGE 1: prologue fills NB tiles, so the steady loop runs one fewer
            # iteration and the first tile it loads is tile NB (lookahead = NB).
            main_loop_iters_g = num_k_tiles - num_buffers

            load_idx_init = arith.constant(num_buffers, type=T.i32)
            compute_idx_init = arith.constant(0, type=T.i32)

            if const_expr(main_loop_iters_g > 0):
                init_state = _pack_state_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw
                )
                if const_expr(USES_W_CHUNK_PREFETCH):
                    init_state = init_state + [
                        bulk_w_cur,
                        bulk_w_prefetch,
                        cur_chunk_idx_i32,
                    ]
                init_state = init_state + [
                    lo_x, lo_w, lo_x_scale,
                    load_idx_init, compute_idx_init,
                ]

                nb_const_i32 = arith.constant(num_buffers, type=T.i32)
                one_i32 = arith.constant(1, type=T.i32)
                two_i32 = arith.constant(2, type=T.i32)

                for tile_step, state in range(
                    0, main_loop_iters_g, 1, init=init_state
                ):
                    _disable_unroll_on_enclosing_loop()
                    cur_compute_idx = state[-1]
                    cur_load_idx = state[-2]
                    cur_lo_x_scale = state[-3]
                    cur_lo_w = state[-4]
                    cur_lo_x = state[-5]
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-6]
                        bulk_w_prefetch = state[-7]
                        bulk_w_cur = state[-8]
                        _reg_state = state[:-8]
                    else:
                        _reg_state = state[:-5]
                    (
                        cur_accs, cur_a, cur_b,
                        cur_x_raw, cur_w_raw, prefetch_w_raw,
                    ) = _unpack_state_experimental(_reg_state)

                    # SSA buf indices for this iteration.
                    load_buf_i32 = arith.remui(cur_load_idx, nb_const_i32)
                    next_compute_idx = arith.addi(cur_compute_idx, one_i32)
                    next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32)

                    # WMMA on cur tile.
                    cur_accs = compute_wmma_with_frags_experimental(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                    )

                    # Issue TDMs for tile load_idx.
                    cur_lo_x, cur_lo_w = issue_tdm_loads(
                        load_buf_i32, cur_lo_x, cur_lo_w
                    )
                    cur_lo_x_scale = issue_x_scale_tdm(
                        load_buf_i32, cur_lo_x_scale
                    )

                    # Wait for tile compute_idx+1 to land in LDS.
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                    gpu.barrier()

                    # Pre-load tile compute_idx+1 into VGPRs.
                    # Double-buffer B: issue B's ds_read FIRST (as soon as the n+1
                    # wait fires) into its own vgprs, separate from cur_b, so the B
                    # load isn't WAR-blocked behind this tile's WMMAs reading cur_b.
                    # The cur_b = next_b swap at the loop tail becomes the v_movs
                    # that carry the prefetched B across the backedge.
                    next_b = load_b_frags(next_buf_i32)
                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    next_a = load_a_frags(next_buf_i32)

                    cur_a = next_a
                    cur_b = next_b
                    cur_x_raw = next_x_raw
                    cur_w_raw = prefetch_w_raw

                    future_tile_i32 = arith.addi(cur_compute_idx, two_i32)
                    future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                    prefetch_w_raw = (
                        issue_w_raw_scales_for_future_tile_rt_experimental(
                            future_tile_idx
                        )
                    )


                    new_load_idx = arith.addi(cur_load_idx, one_i32)
                    new_state = _pack_state_experimental(
                        cur_accs, cur_a, cur_b,
                        cur_x_raw, cur_w_raw, prefetch_w_raw,
                    )
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        new_state = new_state + [
                            bulk_w_cur,
                            bulk_w_prefetch,
                            cur_chunk_idx_i32,
                        ]
                    new_state = new_state + [
                        cur_lo_x, cur_lo_w, cur_lo_x_scale,
                        new_load_idx, next_compute_idx,
                    ]
                    results = yield new_state

                final_compute_idx = results[-1]
                lo_x_scale = results[-3]
                lo_w = results[-4]
                lo_x = results[-5]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-6]
                    bulk_w_prefetch = results[-7]
                    bulk_w_cur = results[-8]
                    _reg_results = results[:-8]
                else:
                    _reg_results = results[:-5]
                (
                    accs, cur_a, cur_b,
                    cur_x_raw, cur_w_raw, prefetch_w_raw,
                ) = _unpack_state_experimental(_reg_results)
            else:
                accs = list(accs)
                # No main loop ran — drain starts at compute_idx = 0.
                final_compute_idx = arith.constant(0, type=T.i32)

            # EPILOGUE — drain NB-1 iters + final WMMA; Y-store is shared after the manual branch.
            drain_compute_idx = final_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)
            two_i32_d = arith.constant(2, type=T.i32)
            if const_expr(drain_iters > 0):
                drain_state0 = _pack_state_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw
                ) + [drain_compute_idx]
                for _drain_i, dstate in range(
                    0, drain_iters, 1, init=drain_state0
                ):
                    _disable_unroll_on_enclosing_loop()
                    cur_dci = dstate[-1]
                    (
                        accs, cur_a, cur_b,
                        cur_x_raw, cur_w_raw, prefetch_w_raw,
                    ) = _unpack_state_experimental(dstate[:-1])

                    accs = compute_wmma_with_frags_experimental(
                        accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                    )

                    tdm_ops.tensor_wait(0)
                    gpu.barrier()

                    next_compute_idx = arith.addi(cur_dci, one_i32_d)
                    next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32_d)

                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    cur_a, cur_b = load_operand_frags(next_buf_i32)
                    cur_x_raw = next_x_raw

                    cur_w_raw = prefetch_w_raw
                    future_tile_i32 = arith.addi(cur_dci, two_i32_d)
                    future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                    prefetch_w_raw = (
                        issue_w_raw_scales_for_future_tile_rt_experimental(
                            future_tile_idx
                        )
                    )

                    dresults = yield _pack_state_experimental(
                        accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw
                    ) + [next_compute_idx]

                (
                    accs, cur_a, cur_b,
                    cur_x_raw, cur_w_raw, prefetch_w_raw,
                ) = _unpack_state_experimental(dresults[:-1])

            # Final WMMA on the last rotated tile.
            accs = compute_wmma_with_frags_experimental(
                accs, cur_a, cur_b, cur_x_raw, cur_w_raw
            )

        elif const_expr(variant == "manual"):
            def _compute_tile_b_resident(cur_accs_, cur_buf_i32_, w_T_list, boot=None, prefetch=None,
                                         tail_rows=0, mid_fn=None):
                _HAS_BOOT = boot is not None
                bootstrap_half = wmma_n_rep // 2
                second_half = wmma_n_rep - bootstrap_half
                prefetch_a_at = max(0, wmma_n_rep - 3)
                prefetch_xs_at = max(0, wmma_n_rep - 2)

                slot_off_a = arith.index_cast(
                    T.index, arith.muli(cur_buf_i32_, slot_stride_a_i32),
                )
                slot_off_b = arith.index_cast(
                    T.index, arith.muli(cur_buf_i32_, slot_stride_b_i32),
                )
                slot_off_xs = arith.index_cast(
                    T.index, arith.muli(cur_buf_i32_, slot_stride_x_scale_i32),
                )

                tile_base_a = a_lane_bases[0] + slot_off_a
                tile_base_b = b_lane_bases[0] + slot_off_b
                tile_base_xs = (
                    slot_off_xs
                    + (warp_m_base + lane16) * arith.index(scales_per_tile * 4)
                )

                def _load_a(wm, sc):
                    return _load_frag(
                        big_a_mem, tile_base_a, ks=sc,
                        const_off=wm * WMMA_M * lds_a_stride_bytes,
                    )

                def _load_a_thunks(wm, sc):
                    return _frag_load_thunks(
                        big_a_mem, tile_base_a, ks=sc,
                        const_off=wm * WMMA_M * lds_a_stride_bytes,
                    )

                def _load_b(wn, sc):
                    return _load_frag(
                        big_b_mem, tile_base_b, ks=sc,
                        cycle_stride_bytes=512, ks_stride_bytes=2048,
                        const_off=wn * lds_b_stride_bytes,
                    )

                def _load_xs(wm, sc):
                    # Row stride = scales_per_tile fp32s per row (const per wm).
                    return lds_load_b32_f32(
                        big_x_scale_mem,
                        tile_base_xs
                        + arith.index(wm * WMMA_M * scales_per_tile * 4 + sc * 4),
                    )

                # ── Phase B: cross-tile (T+1) prefetch, dripped into shadows ──
                _HAS_PF = prefetch is not None
                pf_flat = []        # flat FIFO of (dest_list, dest_idx, thunk)
                pf_b_parts = []     # [wn][part] collected B parts → assemble later
                pf_a_parts = []     # [wm][part] collected A parts
                pf_xs_vals = []     # [wm] collected x-scale values
                if const_expr(_HAS_PF):
                    _pf_b_jobs, _pf_a_jobs, _pf_xs_jobs = prefetch
                    pf_b_parts = [[None] * len(j) for j in _pf_b_jobs]
                    pf_a_parts = [[None] * len(j) for j in _pf_a_jobs]
                    pf_xs_vals = [None] * len(_pf_xs_jobs)
                    def _ds_fence():
                        rocdl.sched_barrier(0xE)
                        return None
                    if const_expr(a_resident_refill):
                        pf_flat = []
                    else:
                        pf_flat = (
                            [(pf_a_parts[0], pi, _pf_a_jobs[0][pi])
                             for pi in range(len(_pf_a_jobs[0]))]
                            + [(pf_xs_vals, wm, _pf_xs_jobs[wm]) for wm in range(len(_pf_xs_jobs))]
                            + [([None], 0, _ds_fence)]
                            + [(pf_b_parts[wn], pi, _pf_b_jobs[wn][pi])
                             for wn in range(len(_pf_b_jobs)) for pi in range(len(_pf_b_jobs[wn]))]
                            + [([None], 0, _ds_fence)]
                            + [(pf_a_parts[wm], pi, _pf_a_jobs[wm][pi])
                               for wm in range(1, len(_pf_a_jobs)) for pi in range(len(_pf_a_jobs[wm]))]
                        )

                _skew = (mid_fn is not None) and (tail_rows > 0)
                tail_start_m = max(1, wmma_m_rep - tail_rows) if _skew else wmma_m_rep
                _tail_shadows_left = 0

                FOLD_LAG = 2
                pending = []  # [(temp, s_cur, acc_idx)] awaiting their 4-fma fold

                PF_RATE = 2

                def _shadow_quota(k):
                    rocdl.sched_group_barrier(0x8, 1, 0)    # 1 WMMA
                    if k > 0:
                        rocdl.sched_group_barrier(0x100, k, 0)  # k DS-reads this shadow
                    rocdl.sched_group_barrier(0x2, 3, 0)    # 3 fold fmas at the 3 I-slots
                    # 4th fma dropped: a WMMA shadow has only ~3 VALU co-exec (I)
                    # slots, so pinning a 4th fma into this group can't co-issue —
                    # leave it un-grouped so the scheduler places it freely.

                # ── A in-place refill (a_resident_refill) ───────────────────
                _a_refill_out = []
                _a_refill_parts = []
                _xs_refill_out = []
                _sec_fifo = []
                if const_expr(a_resident_refill and _HAS_PF):
                    # SINGLE-BUFFER A/xs: alias the refill targets to the caller's
                    # resident panel (boot[1]/boot[2]) so the next-tile data
                    # overwrites the SAME registers the WMMAs just read — in place,
                    # one bank. The unroll hands ONE shared A/xs panel to both
                    # sub-tiles, so A/xs never take a second bank; B keeps its
                    # double-buffer. Safe: every boot[k] is read into a_cur one row
                    # before its refill (read idx m+1, refill idx m-1).
                    _a_refill_out = boot[1] if _HAS_BOOT else [None] * wmma_m_rep
                    _a_refill_parts = [[None] * 4 for _ in range(wmma_m_rep)]
                    _xs_refill_out = boot[2] if _HAS_BOOT else [None] * wmma_m_rep
                    _sec_fifo = [
                        (pf_b_parts[wn], pi, _pf_b_jobs[wn][pi])
                        for wn in range(len(_pf_b_jobs)) for pi in range(len(_pf_b_jobs[wn]))
                    ]

                for sc in range_constexpr(scales_per_tile):
                    w_T_sc = w_T_list[sc]
                    _sc_boot = _HAS_BOOT and sc == 0  # use caller's prefetch

                    # ── Bootstrap: A0 + Xs0 + full B column for this sc ──
                    if const_expr(_sc_boot):
                        b_frags = [None] * wmma_n_rep
                        for n in range_constexpr(wmma_n_rep):
                            b_frags[n] = boot[0][n]
                        a_cur = boot[1][0]
                        xs_cur = boot[2][0]
                    else:
                        a_cur = _load_a(0, sc)
                        xs_cur = _load_xs(0, sc)
                        b_frags = [None] * wmma_n_rep
                        for n in range_constexpr(bootstrap_half):
                            b_frags[n] = _load_b(n, sc)

                    scale_cur = arith.mulf(xs_cur, w_T_sc)
                    s_cur = vector.broadcast(T.vec(8, T.f32), scale_cur)

                    a_next = None
                    xs_next = None

                    xs_pre = {}
                    xs_pre_start = 2 if _sc_boot else 1

                    # ── 2,1-burst A-frag prefetch (tile T): boot carries rows
                    _resident_a = NUM_BOOT_A if _sc_boot else 1
                    _intile_a = list(range(_resident_a, wmma_m_rep))
                    _a_thunk = {r: _load_a_thunks(r, sc) for r in _intile_a}
                    _a_parts = {r: [None] * 4 for r in _intile_a}
                    _a_sched = {}
                    _ai = 0
                    for _cr in range_constexpr(wmma_m_rep):
                        _a_sched[_cr] = _intile_a[_ai:_ai + (2 if (_cr % 2 == 0) else 1)]
                        _ai += len(_a_sched[_cr])
                    _a_jobs = {
                        _cr: [(_a_parts[_row], _p, _a_thunk[_row][_p])
                              for _row in _a_sched[_cr] for _p in range(4)]
                        for _cr in range(wmma_m_rep)
                    }

                    # ── Row 0: drip compute-row-0's scheduled A-loads + (non-boot)
                    # 2nd-half B + T+1 prefetch under the WMMAs. ──
                    _jobs0 = _a_jobs[0]
                    for n in range_constexpr(wmma_n_rep):
                        _ds_k = 0.0  # b128 DS instrs under this shadow (xs b32s unmatched)
                        if const_expr((not _sc_boot) and n < second_half):
                            b_frags[bootstrap_half + n] = _load_b(bootstrap_half + n, sc)
                            _ds_k += 4
                        for _k in range_constexpr(len(_jobs0)):
                            if const_expr((wmma_n_rep * _k) // max(1, len(_jobs0)) == n):
                                _dst, _pi, _th = _jobs0[_k]
                                _dst[_pi] = _th()
                                _ds_k += 1
                        if const_expr(a_resident_refill and _HAS_PF):
                            _n_b0 = min(len(_sec_fifo), 2)
                            for _i in range_constexpr(_n_b0):
                                _dst, _pi, _th = _sec_fifo.pop(0)
                                _dst[_pi] = _th()
                            _ds_k += _n_b0
                        if const_expr(_HAS_PF and (not _skew) and sc == 0 and n != prefetch_a_at and len(pf_flat) > 0):
                            _ptl, _pti, _pth = pf_flat.pop(0)
                            _ptl[_pti] = _pth()
                            _ds_k += 1
                        temp = rocdl.wmma_f32_16x16x128_fp8_fp8(
                            T.vec(8, T.f32), b_frags[n], a_cur, acc_zero,
                        ).result
                        pending.append((temp, s_cur, 0 * wmma_n_rep + n))
                        if const_expr(len(pending) > FOLD_LAG):
                            t_p, s_p, ix_p = pending.pop(0)
                            cur_accs_[ix_p] = math_dialect.fma(t_p, s_p, cur_accs_[ix_p])
                        if const_expr(n == 0):
                            if const_expr(not (a_resident_refill and _sc_boot)):
                                for _r in range_constexpr(xs_pre_start, wmma_m_rep):
                                    xs_pre[_r] = _load_xs(_r, sc)
                            if const_expr(wmma_m_rep > 1):
                                if const_expr(_sc_boot):
                                    xs_next = boot[2][1]
                                else:
                                    xs_next = xs_pre[1]
                        _shadow_quota(int(-(-_ds_k // 1)))

                    # a_cur for row 1: boot-resident (NUM_BOOT_A>=2) or assembled.
                    if const_expr(wmma_m_rep > 1):
                        if const_expr(_sc_boot and 1 < NUM_BOOT_A):
                            a_cur = boot[1][1]
                        else:
                            a_cur = _assemble_frag(_a_parts[1])
                        xs_cur = xs_next

                    if const_expr(a_resident_refill and _HAS_PF):
                        rocdl.sched_barrier(0x6)  # pin WMMA+DS per row (see steady-sweep fence)

                    # ── Steady-state A-row sweep: m = 1 .. wmma_m_rep - 1 ──
                    for m in range_constexpr(1, wmma_m_rep):
                        if const_expr(_skew and m == tail_start_m):
                            mid_fn()
                            _tail_shadows_left = tail_rows * wmma_n_rep
                        scale_cur = arith.mulf(xs_cur, w_T_sc)
                        s_cur = vector.broadcast(T.vec(8, T.f32), scale_cur)

                        _jobsm = _a_jobs[m]
                        _row_a_q = []
                        _row_xs_q = []
                        if const_expr(a_resident_refill and _HAS_PF):
                            _row_a_q = [
                                (_a_refill_parts[m - 1], _p, _pf_a_jobs[m - 1][_p])
                                for _p in range(4)
                            ]
                            _row_xs_q = [(_xs_refill_out, m - 1, _pf_xs_jobs[m - 1])]
                        for n in range_constexpr(wmma_n_rep):
                            _ds_k = 0.0
                            for _k in range_constexpr(len(_jobsm)):
                                if const_expr((wmma_n_rep * _k) // max(1, len(_jobsm)) == n):
                                    _dst, _pi, _th = _jobsm[_k]
                                    _dst[_pi] = _th()
                                    _ds_k += 1
                            if const_expr(m + 1 < wmma_m_rep):
                                if const_expr(a_resident_refill and _sc_boot):
                                    if const_expr(n == 0):
                                        xs_next = boot[2][m + 1]  # resident x-scale
                                elif const_expr((m + 1) in xs_pre):
                                    if const_expr(n == 0):
                                        xs_next = xs_pre[m + 1]
                                elif const_expr(n == prefetch_xs_at):
                                    xs_next = _load_xs(m + 1, sc)
                            if const_expr(a_resident_refill and _HAS_PF):
                                # BURST: fire all 4 b128 of A_{m-1} (+ its xs) together
                                # in this one shadow so the frag lands FULLY in its row
                                # (4/4) instead of the per-shadow drip splitting it 3+1
                                # across the row boundary. B keeps draining 2/shadow.
                                _n_a = len(_row_a_q)
                                for _i in range_constexpr(_n_a):
                                    _dst, _pi, _th = _row_a_q.pop(0)
                                    _dst[_pi] = _th()
                                _n_xs = len(_row_xs_q)
                                for _i in range_constexpr(_n_xs):
                                    _dst, _pi, _th = _row_xs_q.pop(0)
                                    _dst[_pi] = _th()
                                _n_b = min(len(_sec_fifo), 2)
                                for _i in range_constexpr(_n_b):
                                    _dst, _pi, _th = _sec_fifo.pop(0)
                                    _dst[_pi] = _th()
                                _ds_k += _n_a + _n_xs + _n_b
                            if const_expr(_skew):
                                if const_expr(_HAS_PF and sc == 0 and m >= tail_start_m):
                                    if len(pf_flat) > 0:
                                        _take = -(-len(pf_flat) // max(1, _tail_shadows_left))
                                        for _kk in range_constexpr(_take):
                                            _ptl, _pti, _pth = pf_flat.pop(0)
                                            _ptl[_pti] = _pth()
                                        _ds_k += _take
                                    _tail_shadows_left -= 1
                            elif const_expr(_HAS_PF and sc == 0 and _ds_k == 0):
                                for _kk in range_constexpr(min(PF_RATE, len(pf_flat))):
                                    _ptl, _pti, _pth = pf_flat.pop(0)
                                    _ptl[_pti] = _pth()
                                    _ds_k += 1
                            temp = rocdl.wmma_f32_16x16x128_fp8_fp8(
                                T.vec(8, T.f32), b_frags[n], a_cur, acc_zero,
                            ).result
                            pending.append((temp, s_cur, m * wmma_n_rep + n))
                            if const_expr(len(pending) > FOLD_LAG):
                                t_p, s_p, ix_p = pending.pop(0)
                                cur_accs_[ix_p] = math_dialect.fma(t_p, s_p, cur_accs_[ix_p])
                            _shadow_quota(int(-(-_ds_k // 1)))

                        # a_cur for row m+1: boot-resident or assembled from buffer.
                        if const_expr(m + 1 < wmma_m_rep):
                            if const_expr(_sc_boot and (m + 1) < NUM_BOOT_A):
                                a_cur = boot[1][m + 1]
                            else:
                                a_cur = _assemble_frag(_a_parts[m + 1])
                            xs_cur = xs_next

                        if const_expr(a_resident_refill and _HAS_PF):
                            for _i in range_constexpr(len(_row_a_q)):
                                _dst, _pi, _th = _row_a_q.pop(0)
                                _dst[_pi] = _th()
                            for _i in range_constexpr(len(_row_xs_q)):
                                _dst, _pi, _th = _row_xs_q.pop(0)
                                _dst[_pi] = _th()
                            _a_refill_out[m - 1] = _assemble_frag(_a_refill_parts[m - 1])
                            # 0x6 (was 0xE): SALU/VALU may cross, but MFMA *and* DS are
                            # pinned — keeps this row's 8 WMMAs in their own region (with
                            # the 4 A-refill loads) instead of drifting into neighbors and
                            # stranding a 0-WMMA region (the r5 case). Trade: less cross-row
                            # WMMA pipelining; measuring whether the even distribution wins.
                            rocdl.sched_barrier(0x6)

                    if const_expr(a_resident_refill and _HAS_PF):
                        _tr = wmma_m_rep - 1
                        for _p in range_constexpr(4):
                            _a_refill_parts[_tr][_p] = _pf_a_jobs[_tr][_p]()
                        _a_refill_out[_tr] = _assemble_frag(_a_refill_parts[_tr])
                        _xs_refill_out[_tr] = _pf_xs_jobs[_tr]()
                        for _i in range_constexpr(len(_sec_fifo)):
                            _dst, _pi, _th = _sec_fifo.pop(0)
                            _dst[_pi] = _th()
                        # Pin the FINAL-frag (r7) refill right after row7's WMMAs:
                        # 0x6 (MFMA+DS pinned) stops these last A/xs/B ds_loads from
                        # sinking past the fold-drain + v_dual_movs to the loop tail.
                        rocdl.sched_barrier(0x6)

                    n_drain = len(pending)
                    for _i in range_constexpr(n_drain):
                        t_p, s_p, ix_p = pending.pop(0)
                        cur_accs_[ix_p] = math_dialect.fma(t_p, s_p, cur_accs_[ix_p])

                nxt_boot = None
                if const_expr(_HAS_PF):
                    for _pi in range_constexpr(len(pf_flat)):
                        _ptl, _pti, _pth = pf_flat.pop(0)
                        _ptl[_pti] = _pth()
                    nxt_boot = (
                        [_assemble_frag(p) for p in pf_b_parts],
                        (list(_a_refill_out) if a_resident_refill
                         else [_assemble_frag(p) for p in pf_a_parts]),
                        (list(_xs_refill_out) if a_resident_refill
                         else list(pf_xs_vals)),
                    )
                return cur_accs_, nxt_boot

            NUM_BOOT_A = wmma_m_rep if a_resident_refill else min(2, wmma_m_rep)
            NUM_BOOT_XS = wmma_m_rep if a_resident_refill else min(2, wmma_m_rep)
            NUM_BOOT_B = wmma_n_rep
            BOOT_LEN = NUM_BOOT_B + NUM_BOOT_A + NUM_BOOT_XS  # b + a_frags + xs

            _PF_JOBS = NUM_BOOT_B * 4 + NUM_BOOT_A * 4 + NUM_BOOT_XS
            _PF_SHADOWS = -(-_PF_JOBS // 2)            # ceil(jobs / 2) = WMMA shadows
            SKEW_TAIL_ROWS = max(1, min(wmma_m_rep - 1, -(-_PF_SHADOWS // wmma_n_rep)))

            def _load_bootstrap(buf_i32):
                base_a = a_lane_bases[0] + arith.index_cast(
                    T.index, arith.muli(buf_i32, slot_stride_a_i32))
                base_b = b_lane_bases[0] + arith.index_cast(
                    T.index, arith.muli(buf_i32, slot_stride_b_i32))
                base_xs = (
                    arith.index_cast(T.index, arith.muli(buf_i32, slot_stride_x_scale_i32))
                    + (warp_m_base + lane16) * arith.index(scales_per_tile * 4)
                )
                b_boot = [
                    _load_frag(
                        big_b_mem, base_b, ks=0,
                        cycle_stride_bytes=512, ks_stride_bytes=2048,
                        const_off=n * lds_b_stride_bytes,
                    )
                    for n in range(NUM_BOOT_B)
                ]
                a_boot = [
                    _load_frag(big_a_mem, base_a, ks=0,
                               const_off=m * WMMA_M * lds_a_stride_bytes)
                    for m in range(NUM_BOOT_A)
                ]
                xs_boot = [
                    lds_load_b32_f32(
                        big_x_scale_mem,
                        base_xs + arith.index(m * WMMA_M * scales_per_tile * 4),
                    )
                    for m in range(NUM_BOOT_XS)
                ]
                return b_boot, a_boot, xs_boot

            def _bootstrap_thunks(buf_i32):
                base_a = a_lane_bases[0] + arith.index_cast(
                    T.index, arith.muli(buf_i32, slot_stride_a_i32))
                base_b = b_lane_bases[0] + arith.index_cast(
                    T.index, arith.muli(buf_i32, slot_stride_b_i32))
                off_xs = arith.index_cast(T.index, arith.muli(buf_i32, slot_stride_x_scale_i32))
                b_jobs = [
                    _frag_load_thunks(
                        big_b_mem, base_b, 0,
                        cycle_stride_bytes=512, ks_stride_bytes=2048,
                        const_off=n * lds_b_stride_bytes,
                    )
                    for n in range(NUM_BOOT_B)
                ]
                a_jobs = [
                    _frag_load_thunks(big_a_mem, base_a, 0,
                                      const_off=m * WMMA_M * lds_a_stride_bytes)
                    for m in range(NUM_BOOT_A)
                ]
                xs_jobs = [
                    (lambda mm=m: lds_load_b32_f32(
                        big_x_scale_mem,
                        off_xs
                        + (warp_m_base + arith.index(mm * WMMA_M) + lane16)
                        * arith.index(scales_per_tile * 4),
                    ))
                    for m in range(NUM_BOOT_XS)
                ]
                return b_jobs, a_jobs, xs_jobs

            lo_x = addr_lo_x_init
            lo_w = addr_lo_w_init
            lo_x_scale = addr_lo_x_scale_init

            rocdl.s_setprio(2)
            for i in range_constexpr(MANUAL_PROLOGUE_TILES):
                lo_x, lo_w = issue_tdm_loads(i, lo_x, lo_w)
                lo_x_scale = issue_x_scale_tdm(i, lo_x_scale)
            rocdl.s_setprio(0)

            bulk_w_chunks = [_issue_w_chunk_const(c) for c in range(NUM_W_CHUNKS)]
            bulk_w_cur = bulk_w_chunks[0]
            bulk_w_prefetch = bulk_w_chunks[1] if USES_W_CHUNK_PREFETCH else bulk_w_chunks[0]

            nb_const_i32     = arith.constant(num_buffers, type=T.i32)         # ring-buffer modulus for buf_idx = idx % NB
            one_i32          = arith.constant(1, type=T.i32)                   # increment for compute_idx each iter
            two_i32          = arith.constant(2, type=T.i32)                   # offset for "future tile" W-scale prefetch (compute_idx + 2)
            load_idx_init    = arith.constant(num_buffers, type=T.i32)         # first tile to TDM-issue in main loop (= prologue's last + 1)
            compute_idx_init = arith.constant(0, type=T.i32)                   # first tile to compute (= the one tensor_wait just retired)
            accs = [acc_zero] * n_accs                                         # n_accs vec<8xf32> output accumulators, one per (wm, wn) cell

            main_loop_iters_g = num_k_tiles - num_buffers

            if const_expr(main_loop_iters_g > 0):
                tdm_ops.tensor_wait(MANUAL_TDM_OUTSTANDING)
                gpu.barrier()
                boot0 = _load_bootstrap(arith.constant(0, type=T.i32))

                kb0_i32 = arith.muli(
                    compute_idx_init, arith.constant(scales_per_tile, type=T.i32)
                )
                w_T0_list = [
                    _w_readlane(arith.addi(kb0_i32, arith.constant(sc, type=T.i32)))
                    for sc in range(scales_per_tile)
                ]

                init_state = (
                    list(accs)
                    + list(boot0[0]) + list(boot0[1]) + list(boot0[2])
                    + list(w_T0_list)
                )
                if const_expr(USES_W_CHUNK_PREFETCH):
                    init_state = init_state + [
                        bulk_w_cur, bulk_w_prefetch, cur_chunk_idx_i32,
                    ]
                init_state = init_state + [
                    lo_x, lo_w, lo_x_scale,
                    load_idx_init, compute_idx_init,
                ]

                def _run_one_tile(state, discrete_bootstrap=False):
                    cur_compute_idx  = state[-1]
                    cur_load_idx     = state[-2]
                    cur_lo_x_scale   = state[-3]
                    cur_lo_w         = state[-4]
                    cur_lo_x         = state[-5]
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-6]
                        bulk_w_prefetch   = state[-7]
                        bulk_w_cur        = state[-8]
                    cur_accs = list(state[:N_ACCS])
                    # Carried cross-tile prefetch for THIS tile (loaded last iter).
                    _bo = N_ACCS
                    cur_boot = (
                        list(state[_bo : _bo + NUM_BOOT_B]),
                        list(state[_bo + NUM_BOOT_B : _bo + NUM_BOOT_B + NUM_BOOT_A]),
                        list(state[_bo + NUM_BOOT_B + NUM_BOOT_A : _bo + BOOT_LEN]),
                    )
                    # Carried W-scale readlane(s) for THIS tile (prefetched last iter).
                    _wo = _bo + BOOT_LEN
                    cur_w_T_list = list(state[_wo : _wo + scales_per_tile])

                    # Ring-buffer SSA indices for this iter.
                    load_buf_i32      = arith.remui(cur_load_idx, nb_const_i32)
                    next_compute_idx  = arith.addi(cur_compute_idx, one_i32)
                    cur_buf_i32       = arith.remui(cur_compute_idx, nb_const_i32)
                    next_buf_i32      = arith.remui(next_compute_idx, nb_const_i32)

                    # Top wall: wait T+1's TDM landed; all NB slots full.
                    tdm_ops.tensor_wait(MANUAL_TDM_OUTSTANDING)
                    if const_expr(use_manual_barrier):
                        rocdl.s_barrier_signal(-1)
                        rocdl.s_barrier_wait(-1)
                    else:
                        gpu.barrier()

                    w_T_list = cur_w_T_list

                    # discrete_bootstrap (odd leftover): no dripped prefetch so
                    # cur_boot + next boot never co-exist (else odd-tile spill);
                    # next boot is loaded discretely below once cur_boot is dead.
                    if const_expr(discrete_bootstrap):
                        cur_accs, _ = _compute_tile_b_resident(
                            cur_accs, cur_buf_i32, w_T_list, boot=cur_boot,
                            prefetch=None,
                        )
                    else:
                        cur_accs, nxt_boot = _compute_tile_b_resident(
                            cur_accs, cur_buf_i32, w_T_list, boot=cur_boot,
                            prefetch=_bootstrap_thunks(next_buf_i32),
                        )

                    # End-of-tile WALL: every wave must finish reading T's LDS slot
                    # before the T+NB TDM recycles it — closes the LDS-WAR (removing
                    # this fails the UT).
                    if const_expr(use_manual_barrier):
                        rocdl.s_barrier_signal(-1)
                        rocdl.s_barrier_wait(-1)
                    else:
                        gpu.barrier()

                    cur_lo_x, cur_lo_w = issue_tdm_loads(
                        load_buf_i32, cur_lo_x, cur_lo_w
                    )
                    cur_lo_x_scale = issue_x_scale_tdm(
                        load_buf_i32, cur_lo_x_scale
                    )

                    if const_expr(discrete_bootstrap):
                        tdm_ops.tensor_wait(MANUAL_TDM_OUTSTANDING)
                        if const_expr(use_manual_barrier):
                            rocdl.s_barrier_signal(-1)
                            rocdl.s_barrier_wait(-1)
                        else:
                            gpu.barrier()
                        nxt_boot = _load_bootstrap(next_buf_i32)


                    # ── Yield updated state ──────────────────────────────────
                    kb_next_i32 = arith.muli(
                        next_compute_idx, arith.constant(scales_per_tile, type=T.i32)
                    )
                    nxt_w_T_list = [
                        _w_readlane(arith.addi(kb_next_i32, arith.constant(sc, type=T.i32)))
                        for sc in range(scales_per_tile)
                    ]
                    new_load_idx = arith.addi(cur_load_idx, one_i32)
                    new_state = (
                        list(cur_accs)
                        + list(nxt_boot[0]) + list(nxt_boot[1]) + list(nxt_boot[2])
                        + list(nxt_w_T_list)
                    )
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        new_state = new_state + [
                            bulk_w_cur, bulk_w_prefetch, cur_chunk_idx_i32,
                        ]
                    new_state = new_state + [
                        cur_lo_x, cur_lo_w, cur_lo_x_scale,
                        new_load_idx, next_compute_idx,
                    ]
                    return new_state

                # ── ROLLED main loop (NO unroll): ONE tile per scf.for iteration.
                # A frag k is read once (row k) and refilled in place — there is no
                # second sub-tile A panel to coexist, so A single-banks (low only)
                # and the loop runs v_nop-free. B is relocated each iter via ~64
                # v_dual_movs (the cost of dropping the unroll's register self-loop).
                # NOTE: unroll-by-2 (verified to self-loop B / 0 copies / no spill,
                # but adds ~48-reg A bank-1 drift + ~63 v_nops) is parked until the
                # per-iteration efficiency is cleaned up — see memory.
                results = init_state
                for tile_step, state in range(
                    0, main_loop_iters_g, 1, init=init_state
                ):
                    _disable_unroll_on_enclosing_loop()
                    state = _run_one_tile(state)
                    results = yield state
                final_compute_idx = results[-1]
                lo_x_scale = results[-3]
                lo_w = results[-4]
                lo_x = results[-5]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-6]
                    bulk_w_prefetch = results[-7]
                    bulk_w_cur = results[-8]
                accs = list(results[:N_ACCS])
            else:
                accs = list(accs)
                final_compute_idx = arith.constant(0, type=T.i32)

            # EPILOGUE — drain NB queued tiles, no new TDM issues.
            drain_compute_idx = final_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)

            _DRAIN_BOOT = main_loop_iters_g > 0
            if const_expr(num_buffers > 1 and (not USES_W_CHUNK_PREFETCH)):
                sc_per_tile_i32_d = arith.constant(scales_per_tile, type=T.i32)
                drain_state0 = list(accs)
                if const_expr(_DRAIN_BOOT):
                    drain_state0 += (
                        list(results[N_ACCS:N_ACCS + BOOT_LEN])
                    )
                drain_state0 += [drain_compute_idx]
                for _drain_i, dstate in range(
                    0, MANUAL_NB_EFF, 1, init=drain_state0
                ):
                    _disable_unroll_on_enclosing_loop()
                    cur_accs_d = list(dstate[:N_ACCS])
                    cur_boot_d = None
                    if const_expr(_DRAIN_BOOT):
                        _db = N_ACCS
                        cur_boot_d = (
                            list(dstate[_db:_db + NUM_BOOT_B]),
                            list(dstate[_db + NUM_BOOT_B:_db + NUM_BOOT_B + NUM_BOOT_A]),
                            list(dstate[_db + NUM_BOOT_B + NUM_BOOT_A:_db + BOOT_LEN]),
                        )
                    cur_dci = dstate[-1]

                    for _t in range_constexpr(MANUAL_NB_EFF):
                        _is_t = arith.cmpi(
                            arith.CmpIPredicate.eq, _drain_i,
                            arith.constant(_t, type=T.index),
                        )
                        if _is_t:
                            tdm_ops.tensor_wait(
                                max(0, MANUAL_NB_EFF - 1 - _t) * _TDMS_PER_TILE_EXP
                            )
                    gpu.barrier()

                    drain_buf_i32 = arith.remui(cur_dci, nb_const_i32_d)
                    kb_drain_i32 = arith.muli(cur_dci, sc_per_tile_i32_d)
                    w_drain_list = [
                        _w_readlane(
                            arith.addi(kb_drain_i32, arith.constant(sc, type=T.i32))
                        )
                        for sc in range(scales_per_tile)
                    ]
                    next_dci = arith.addi(cur_dci, one_i32_d)
                    next_buf_d = arith.remui(next_dci, nb_const_i32_d)
                    if const_expr(_DRAIN_BOOT):
                        cur_accs_d, nxt_boot_d = _compute_tile_b_resident(
                            cur_accs_d, drain_buf_i32, w_drain_list,
                            boot=cur_boot_d,
                            prefetch=_bootstrap_thunks(next_buf_d),
                        )
                        new_dstate = (
                            list(cur_accs_d)
                            + list(nxt_boot_d[0]) + list(nxt_boot_d[1]) + list(nxt_boot_d[2])
                            + [next_dci]
                        )
                    else:
                        cur_accs_d, _ = _compute_tile_b_resident(
                            cur_accs_d, drain_buf_i32, w_drain_list
                        )
                        new_dstate = list(cur_accs_d) + [next_dci]
                    dresults = yield new_dstate
                accs = list(dresults[:N_ACCS])
            else:
                for drain_i in range_constexpr(MANUAL_NB_EFF):
                    # Outstanding TDMs decreasing as we drain.
                    outstanding = max(0, MANUAL_NB_EFF - 1 - drain_i) * _TDMS_PER_TILE_EXP
                    tdm_ops.tensor_wait(outstanding)
                    gpu.barrier()

                    # Same compute body as main loop, just no TDM issue.
                    drain_buf_i32 = arith.remui(drain_compute_idx, nb_const_i32_d)
                    kb_drain_i32 = arith.muli(
                        drain_compute_idx,
                        arith.constant(scales_per_tile, type=T.i32),
                    )
                    w_drain_list = [
                        _w_readlane(
                            arith.addi(kb_drain_i32, arith.constant(sc, type=T.i32))
                        )
                        for sc in range(scales_per_tile)
                    ]

                    accs, _ = _compute_tile_b_resident(
                        accs, drain_buf_i32, w_drain_list
                    )


                    drain_compute_idx = arith.addi(drain_compute_idx, one_i32_d)

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
        a_resident_refill,
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

    launch_gemm_a8w8_blockscale.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
        "amdgpu-anti-hints-for-va-vdst": True,
        "amdgpu-enable-static-simulator": True,
        "amdgpu-static-sim-inline": True,
        # Co-exec scheduler (gfx1250-only): top-down MachineScheduler that packs
        # VALU/DS/SALU into WMMA co-exec I-slots. Selected via the attribute the
        # AMDGPU backend reads (getSchedStrategy); also nops the post-RA sched.
        "amdgpu-sched-strategy": "coexec",
    }

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
    wmma_operand_reuse: bool = False,
):
    assert x.ndim == 2 and w.ndim == 2, "X and W must be 2D"
    M, K = x.shape
    if getattr(w, "is_shuffled", False):
        # Preshuffled W: shape is (N // 16, K * 16). Recover logical (N, K).
        N = w.shape[0] * 16
        K_w = w.shape[1] // 16
    else:
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
        assert not getattr(w, "is_shuffled", False), (
            "K padding is not supported for preshuffled W; pass K that is a "
            "multiple of tile_k or shuffle the padded W."
        )
        pad_size = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad_size))
        w = torch.nn.functional.pad(w, (0, pad_size))
        new_scale_k = K_padded // scale_block_k_derived
        scale_pad = new_scale_k - scale_k
        if scale_pad > 0:
            x_scale = torch.nn.functional.pad(x_scale, (0, scale_pad))
            w_scale = torch.nn.functional.pad(w_scale, (0, scale_pad))
        K = K_padded

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
        wmma_operand_reuse=wmma_operand_reuse,
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
