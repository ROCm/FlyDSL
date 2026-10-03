# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
"""Native iu4 (nibble-pack) Flash Attention for gfx120x — in-kernel WMMA.

Q/K/V GMEM are nibble-packed int8 ``[..., D//2]`` (low nibble = even col),
same pack as ``rdna4_iu4_gemm``. Loads use **scalar i32** fragments only
(never i4 buffer-load — LLVM scalarizes those). WMMA atom is
``wmma_i32_16x16x16_iu4`` with A/B bitcast from i32 → Int4.

Tiling mirrors iu8 FA (BLOCK_M=128, BLOCK_N=32, head_dim>=64, %32==0).
LDS holds packed bytes (half of iu8). Softmax P is nibble-packed for PV.
Descales / online softmax / bf16 O match the iu8/FP8 contract.
"""

import math as host_math
import os

import torch  # noqa: E402

# STACK_OPT_COUNTERS (rate-limited glass-walls read these)
_stack_opt_cf_hits = 0
_stack_opt_run_compiled = 0

from collections.abc import Callable  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl.compiler.jit_argument import PointerJitArg  # noqa: E402
from flydsl.compiler.jit_function import CompiledFunction  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
from flydsl.expr import const_expr, gpu, range_constexpr  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402
from flydsl.expr.typing import Vector as Vec  # noqa: E402
from kernels.common.kernels_common import LOG2E as _LOG2E  # noqa: E402
from kernels.common.tensor_shim import _run_compiled  # noqa: E402

KERNEL_NAME = "flash_attn_func_iu4_native_gfx120x_kernel"


def build_flash_attn_func_iu4_native_body(
    num_heads: int,
    head_dim: int,
    causal: bool = True,
    dtype_str: str = "iu4",
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    flat_work_group_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    unsafe_fp_math: bool = True,
    fast_fp_math: bool = True,
    daz: bool = True,
    path_tag: str = "auto",
    num_kv_heads: int | None = None,
) -> Callable[..., None]:
    """Build gfx120x native iu4 FA (nibble-pack QKV → iu4 WMMA → bf16 O)."""

    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    ROWS_PER_WAVE = WMMA_M

    BLOCK_M = block_m if block_m is not None else 128
    BLOCK_N = block_n if block_n is not None else 32

    assert BLOCK_N % K_SUB_N == 0, f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})"
    assert BLOCK_M % ROWS_PER_WAVE == 0, f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    BLOCK_N_OUT = BLOCK_N

    NUM_PREFETCH_K = 1
    NUM_PREFETCH_V = 1

    K_STEP_QK = WMMA_K
    K_STEPS_QK = head_dim // K_STEP_QK
    WMMA_LANE_K = 8

    D_CHUNK = WMMA_N
    D_CHUNKS = head_dim // D_CHUNK

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0
    assert head_dim >= 64
    assert dtype_str in ("iu4", "int4", "i4"), f"iu4 native FA expects dtype_str iu4, got {dtype_str!r}"
    assert head_dim % 2 == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim  # logical
    HEAD_DIM_PACKED = head_dim // 2
    # Dense packed BSHD GQA: same index as bf16 gfx120x (no host KV repeat).
    NUM_KV_HEADS = int(num_heads if num_kv_heads is None else num_kv_heads)
    if NUM_KV_HEADS <= 0 or int(num_heads) % NUM_KV_HEADS != 0:
        raise ValueError(f"gfx120x FA: num_heads={int(num_heads)} must be divisible by num_kv_heads={NUM_KV_HEADS}")
    KV_GROUP = int(num_heads) // NUM_KV_HEADS
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM_PACKED  # packed bytes per token×query heads
    KV_STRIDE = NUM_KV_HEADS * HEAD_DIM_PACKED

    # LDS holds nibble-packed bytes; pad for bank conflicts + i32 align.
    K_STRIDE = HEAD_DIM_PACKED + 8
    V_STRIDE = HEAD_DIM_PACKED + 8

    # dwordx4 = 16 packed bytes = 32 logical nibbles.
    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8  # packed bytes per coop vector
    V_LOAD_WIDTH = 8  # packed bytes per V global load piece
    THREADS_PER_ROW_LOAD = HEAD_DIM_PACKED // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # Buffer loads cap at dwordx4 (16B); V pieces are V_LOAD_WIDTH elems.
    V_SUBVECS = VEC_WIDTH // V_LOAD_WIDTH
    NUM_V_VECS = NUM_BATCHES_KV * V_SUBVECS

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # Nibble-packed bytes in GMEM/LDS; iu4 WMMA via i32 frags; softmax/O in f32/bf16.
    elem_numeric_cls = fx.Int8  # storage = packed int8 bytes
    out_numeric_cls = fx.BFloat16

    @fx.struct
    class SharedStorage:  # noqa: E741
        kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_iu4_native_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        seq_len_kv_valid: fx.Int32,
        q_descale: fx.Float32,
        k_descale: fx.Float32,
        v_descale: fx.Float32,
    ) -> None:
        elem_dtype = elem_numeric_cls
        out_dtype = out_numeric_cls

        def _fadd(a: object, b: object) -> object:
            return a + b

        def _fsub(a: object, b: object) -> object:
            return a - b

        def _fmul(a: object, b: object) -> object:
            return a * b

        def _fmax(a: object, b: object) -> fx.Float32:
            return fx.max(fx.Float32(a), fx.Float32(b))

        def _as_elem_ptr(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(
                fx.PointerType.get(elem_dtype.ir_type, ptr.address_space),
                ptr,
            )

        q_elem_ptr = _as_elem_ptr(Q)
        k_elem_ptr = _as_elem_ptr(K)
        v_elem_ptr = _as_elem_ptr(V)

        def _as_out_ptr(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(
                fx.PointerType.get(out_dtype.ir_type, ptr.address_space),
                ptr,
            )

        o_elem_ptr = _as_out_ptr(O)

        def _bounds_checked_buf_ptr(ptr: fx.Pointer, num_records_bytes: fx.Int64) -> fx.Pointer:
            # OOB_SELECT=3 zero-fills accesses beyond num_records.
            flags = (7 << 12) | (4 << 15) | (1 << 24) | (3 << 28)
            buf_ptr_ty = fx.PointerType.get(
                elem_ty=ptr.element_type.ir_type,
                address_space=fx.rocdl.TargetAddressSpace.BufferDesc,
                alignment=ptr.alignment,
            )
            return fx.make_ptr(
                buf_ptr_ty,
                [
                    ptr,
                    fx.Int16(0).ir_value(),
                    fx.Int64(num_records_bytes).ir_value(),
                    fx.Int32(flags).ir_value(),
                ],
            )

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(
                WMMA_M,
                WMMA_N,
                WMMA_K,
                fx.Int4,
                fx.Int32,
                sign_a=True,
                sign_b=True,
                clamp=False,
            )
        )

        def wmma_acc_i32(a_i32: object, b_i32: object, c_v8: fx.Vector) -> fx.Vector:
            """RDNA4 Wave32 iu4 WMMA via ``fx.gemm``: scalar i32 A/B (8 nibbles), v8i32 C."""
            a_frag = fx.make_rmem_tensor(8, fx.Int4)
            b_frag = fx.make_rmem_tensor(8, fx.Int4)
            c_frag = fx.make_rmem_tensor(8, fx.Int32)
            a_frag.store(Vec(a_i32).bitcast(fx.Int4))
            b_frag.store(Vec(b_i32).bitcast(fx.Int4))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)
            return Vec(c_frag.load())

        def wmma_acc(a_i32: object, b_i32: object, c_v8_f32: object) -> fx.Vector:
            """iu4 MMA into i32, then widen to f32 and add into float accumulator."""
            c0 = Vec.filled(8, 0, fx.Int32)
            iacc = wmma_acc_i32(a_i32, b_i32, c0)
            parts = []
            for i in range_constexpr(8):
                parts.append(fx.Float32(iacc[i]) + fx.Float32(c_v8_f32[i]))
            return Vec.from_elements(parts, fx.Float32)

        # seq_len = Q/O length; seq_len_kv = K/V buffer (may be padded);
        # seq_len_kv_valid = real KV tokens (non-causal pad mask).
        seq_len_q_v = fx.Uint64(seq_len)
        seq_len_kv_v = fx.Uint64(seq_len_kv)
        seq_len_kv_valid_i32 = fx.Int32(seq_len_kv_valid)
        seq_len_v = seq_len_q_v  # alias for residual Q-side uses

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kv = lds.kv.ptr

        def lds_view(offset: fx.Int32 | fx.Int64 | int, width: int) -> fx.Tensor:
            return fx.make_view(
                lds_kv + fx.Int32(offset),
                fx.make_layout(width, 1),
            )

        def lds_load(offset: fx.Int32 | fx.Int64 | int, width: int = 1) -> object:
            return lds_view(offset, width).load()

        def lds_store(offset: fx.Int32 | fx.Int64 | int, value: fx.Vector) -> None:
            lds_view(offset, value.numel).store(value)

        block_id = fx.Uint64(gpu.block_idx.x)
        tid = fx.Uint64(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        wave_q_offset = wave_id * ROWS_PER_WAVE

        head_idx = block_id % NUM_HEADS
        kv_head_idx = head_idx // fx.Uint64(KV_GROUP)
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        _q_tile_linear = batch_q_tile_id % num_q_tiles
        if const_expr(CAUSAL):
            # Dispatch longer causal tiles first.
            q_tile_idx = num_q_tiles - fx.Uint64(1) - _q_tile_linear
        else:
            q_tile_idx = _q_tile_linear
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx_q(token_idx: fx.Int32 | fx.Int64 | int, col_packed: object) -> fx.Int64:
            token = batch_idx * seq_len_q_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM_PACKED + col_packed

        def global_idx_kv(token_idx: fx.Int32 | fx.Int64 | int, col_packed: object) -> fx.Int64:
            token = batch_idx * seq_len_kv_v + token_idx
            return token * KV_STRIDE + kv_head_idx * HEAD_DIM_PACKED + col_packed

        # Packed Q/K/V indexing.
        def global_idx(token_idx: fx.Int32 | fx.Int64 | int, col_packed: object) -> fx.Int64:
            return global_idx_q(token_idx, col_packed)

        def global_idx_o(token_idx: object, col_logical: object) -> fx.Uint64:
            """bf16 O is logical [B,S,H,D] — not nibble-packed."""
            token = batch_idx * seq_len_q_v + token_idx
            return token * fx.Uint64(NUM_HEADS * HEAD_DIM) + head_idx * HEAD_DIM + col_logical

        # Hardware OOB handling keeps the tail prefetch branch-free. A batch
        # slice must fit the descriptor's 32-bit num_records.
        ELEM_BYTES = (elem_numeric_cls.width + 7) // 8
        v_batch_elems = seq_len_kv_v * fx.Uint64(KV_STRIDE)
        v_buf_ptr = _bounds_checked_buf_ptr(
            fx.add_offset(v_elem_ptr, fx.Int64(batch_idx * v_batch_elems)),
            fx.Int64(v_batch_elems) * fx.Int64(ELEM_BYTES),
        )

        def v_idx(token_idx: fx.Int32 | fx.Int64 | int, col_packed: object) -> fx.Int64:
            return token_idx * KV_STRIDE + kv_head_idx * HEAD_DIM_PACKED + col_packed

        def _as_i32_ptr(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, ptr.address_space),
                ptr,
            )

        def _load_global_i8_vec_via_i32(
            elem_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int, width: int
        ) -> fx.Vector:
            # Nibble-pack bytes via i32 dwords (never i4 buffer-load).
            assert width % 4 == 0
            i32_ptr = _as_i32_ptr(elem_ptr)
            n_i32 = width // 4
            view = fx.make_view(
                fx.add_offset(i32_ptr, fx.Int64(base_idx) // fx.Int64(4)),
                fx.make_layout(n_i32, 1),
            )
            return Vec(view.load()).bitcast(fx.Int8)

        def _load_global_i32_frag(elem_ptr: object, byte_idx: object) -> fx.Vector:
            """One scalar i32 fragment (8 packed nibbles) — GEMM load pattern."""
            i32_ptr = _as_i32_ptr(elem_ptr)
            view = fx.make_view(
                fx.add_offset(i32_ptr, fx.Int64(byte_idx) // fx.Int64(4)),
                fx.make_layout(1, 1),
            )
            return Vec.from_elements([fx.Int32(Vec(view.load())[0])], fx.Int32)

        def _store_global_half(elem_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int, val: fx.Vector) -> None:
            # O is bf16 logical layout [B,S,H,D]; store via typed out pointer.
            view = fx.make_view(
                fx.add_offset(elem_ptr, fx.Int64(base_idx)),
                fx.make_layout(val.numel, 1),
            )
            view.store(Vec(val))

        def load_global_f8xN(base_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int) -> fx.Vector:
            return _load_global_i8_vec_via_i32(base_ptr, base_idx, VEC_WIDTH)

        def load_global_v8f8(base_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int) -> fx.Vector:
            return _load_global_i8_vec_via_i32(base_ptr, base_idx, 8)

        def _lds_i32_ptr() -> fx.Pointer:
            return _as_i32_ptr(lds_kv)

        def lds_load_i8(offset: fx.Int32 | fx.Int64 | int, width: int = 1) -> fx.Vector:
            if const_expr(width == 1):
                return lds_load(offset, 1)
            # width multiple of 4: load as i32 then bitcast
            i32_ptr = _lds_i32_ptr()
            n_i32 = width // 4
            view = fx.make_view(
                i32_ptr + fx.Int32(offset) // fx.Int32(4),
                fx.make_layout(n_i32, 1),
            )
            return Vec(view.load()).bitcast(fx.Int8)

        def lds_store_i8(offset: fx.Int32 | fx.Int64 | int, value: fx.Vector) -> None:
            # Store i8 vector via i32 dwords when width >= 4.
            width = value.numel
            if const_expr(width == 1):
                lds_store(offset, value)
                return
            i32_ptr = _lds_i32_ptr()
            i32_vec = Vec(value).bitcast(fx.Int32)
            view = fx.make_view(
                i32_ptr + fx.Int32(offset) // fx.Int32(4),
                fx.make_layout(width // 4, 1),
            )
            view.store(i32_vec)

        def _bitcast_i32(value: fx.Float32 | float) -> fx.Int32:
            return fx.Float32(value).bitcast(fx.Int32)

        def _pack_bf16_pair(lo: fx.Float32, hi: fx.Float32, shift: fx.Int32, mask: fx.Int32) -> fx.Int32:
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack_v8(f32_vals: fx.Vector) -> fx.Vector:
            """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits)."""
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            pairs = []
            for j in range_constexpr(4):
                pairs.append(_pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask))
            return Vec.from_elements(pairs, fx.Int32).bitcast(out_dtype)

        _P_I4_SCALE = 7.0  # softmax probs in ~[0,1] → signed iu4; undo in epilogue

        def iu4_pack_v8(f32_vals: object) -> fx.Vector:
            """Quantize 8 f32 probs to signed iu4 nibbles in one i32 (PV WMMA B)."""
            elems = []
            scale = fx.Float32(_P_I4_SCALE)
            half = fx.Float32(0.5)
            lo_i = fx.Int32(-8)
            hi_i = fx.Int32(7)
            for j in range_constexpr(8):
                x = fx.Float32(f32_vals[j]) * scale
                pos = x >= fx.Float32(0.0)
                adj = pos.select(x + half, x - half)
                qi = adj.to(fx.Int32)
                qi = fx.max(fx.min(qi, hi_i), lo_i)
                elems.append(qi)
            # Pack 8 signed nibbles → one i32 (layout: low nibble = even index)
            packed = (
                (elems[0] & fx.Int32(0xF))
                | ((elems[1] & fx.Int32(0xF)) << fx.Int32(4))
                | ((elems[2] & fx.Int32(0xF)) << fx.Int32(8))
                | ((elems[3] & fx.Int32(0xF)) << fx.Int32(12))
                | ((elems[4] & fx.Int32(0xF)) << fx.Int32(16))
                | ((elems[5] & fx.Int32(0xF)) << fx.Int32(20))
                | ((elems[6] & fx.Int32(0xF)) << fx.Int32(24))
                | ((elems[7] & fx.Int32(0xF)) << fx.Int32(28))
            )
            return Vec.from_elements([packed], fx.Int32)

        def k_buf_base(buf_id: fx.Int32 | int) -> fx.Int64:
            if const_expr(isinstance(buf_id, int)):
                return fx.Int64(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Int64(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id: fx.Int32 | int) -> fx.Int64:
            return fx.Int64(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        def coop_load_k(tile_start: fx.Int32 | fx.Int64 | int, buf_id: fx.Int32 | int = 0) -> None:
            tile_start = fx.Int64(tile_start)
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Int64(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                        vec = load_global_f8xN(k_elem_ptr, g_idx)
                        lds_store_i8(lds_idx, Vec(vec))
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    vec = load_global_f8xN(k_elem_ptr, g_idx)
                    lds_store_i8(lds_idx, Vec(vec))

        def _v_store_row_major(
            v_base: fx.Int64, lds_row: fx.Int32 | fx.Int64 | int, col_extra: int, vec: fx.Vector
        ) -> None:
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base + col_extra
            lds_store_i8(lds_idx, Vec(vec))

        def coop_load_v_global(tile_start: fx.Int32 | fx.Int64 | int) -> list[fx.Vector]:
            tile_start = fx.Int64(tile_start)
            # Clamp surplus loader rows; their LDS stores are discarded.
            load_row_v = load_row_in_batch
            if const_expr(KV_NEEDS_GUARD):
                row_cap = fx.Int64(BLOCK_N - 1)
                load_row_v = fx.Int64((load_row_in_batch < row_cap).select(load_row_in_batch, row_cap))
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_v + row_offset
                for sv in range_constexpr(V_SUBVECS):
                    g_idx = v_idx(row_idx, load_col_base + fx.Int64(sv * V_LOAD_WIDTH))
                    vecs.append(_load_global_i8_vec_via_i32(v_buf_ptr, g_idx, V_LOAD_WIDTH))
            return vecs

        def coop_store_v_lds(vecs: list[fx.Vector], buf_id: fx.Int32 | int = 0) -> None:
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Int64(BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        for sv in range_constexpr(V_SUBVECS):
                            _v_store_row_major(
                                v_base,
                                lds_row,
                                sv * V_LOAD_WIDTH,
                                vecs[batch * V_SUBVECS + sv],
                            )
                else:
                    lds_row = load_row_in_batch + row_offset
                    for sv in range_constexpr(V_SUBVECS):
                        _v_store_row_major(
                            v_base,
                            lds_row,
                            sv * V_LOAD_WIDTH,
                            vecs[batch * V_SUBVECS + sv],
                        )

        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)

        q_in_bounds = q_row < seq_len_v
        q_row_safe = fx.Int64(q_in_bounds.select(q_row, fx.Int64(0)))

        # First KV column fully masked for this wave (bottom-right causal).
        wave_kv_limit_i32 = fx.Int32(q_start + wave_q_offset + fx.Int64(ROWS_PER_WAVE)) + (
            seq_len_kv_valid_i32 - fx.Int32(seq_len)
        )
        # v1: no OOB select on fp8 vectors (arith.select f8<->i8 fails to
        # legalize with WMMA AB). q_row_safe zeros the address; output is
        # still gated by q_in_bounds. Revisit with i8-memory path later.
        # Q fragments: one i32 (8 nibbles) per lane K-half — nibble pack along D.
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col_packed = fx.Int64(ks * (K_STEP_QK // 2)) + klane * fx.Int64(4)
            g_idx = global_idx(q_row_safe, q_col_packed)
            q_b_packs.append(_load_global_i32_frag(q_elem_ptr, g_idx))

        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        # Raw QK WMMA logits; fold sm_scale * q_descale * k_descale into log2
        # domain (matches gfx950 DualwaveFp8KernelContext.init_descale).
        c_logit_scale = fx.Float32(sm_scale * _LOG2E) * q_descale * k_descale
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32: fx.Float32 | fx.Vector) -> fx.Float32:
            return fx.gpu.shuffle_xor(fx.Float32(v_f32), shuf_16_i32, width_i32)

        _q_end = q_start + BLOCK_M
        # Bottom-right causal: query i attends keys j <= i + (Skv_valid - Sq).
        # Equal lengths → classic causal; unequal → Dao/FA cross causal.
        causal_br_off_i32 = seq_len_kv_valid_i32 - fx.Int32(seq_len)
        if const_expr(CAUSAL):
            _last_allow = (_q_end - fx.Int64(1)) + fx.Int64(causal_br_off_i32)
            _last_allow = fx.Int64((_last_allow < fx.Int64(0)).select(fx.Int64(0), _last_allow))
            _kv_end = _last_allow + fx.Int64(1)
            kv_upper = fx.Int64((_kv_end < seq_len_kv_v).select(_kv_end, seq_len_kv_v))
        else:
            kv_upper = seq_len_kv_v

        # Non-causal carries prefetched V across iterations; causal avoids the
        # extra VGPR lifetime and loads V in the current iteration.
        PREFETCH_V_ACROSS_ITERS = not CAUSAL

        if const_expr(PREFETCH_V_ACROSS_ITERS):
            _v_vecs_init = coop_load_v_global(fx.Int64(0))

        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v8f32)
        if const_expr(PREFETCH_V_ACROSS_ITERS):
            for vi in range_constexpr(NUM_V_VECS):
                init_args.append(_v_vecs_init[vi])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(fx.Int64(0), kv_upper, fx.Int64(BLOCK_N_OUT), init=init_args):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            if const_expr(PREFETCH_V_ACROSS_ITERS):
                _v_vecs_tile = [inner_iter_args[2 + D_CHUNKS + b] for b in range_constexpr(NUM_V_VECS)]

            coop_load_k(kv_block_start, 0)
            gpu.barrier()
            k_base = k_buf_base(0)

            if const_expr(not PREFETCH_V_ACROSS_ITERS):
                # Overlap the current V load with GEMM1 and softmax.
                _v_vecs_tile = coop_load_v_global(kv_block_start)

            if const_expr(CAUSAL):
                wave_needs_kv_tile = fx.Int32(kv_block_start) < wave_kv_limit_i32
            else:
                wave_needs_kv_tile = True

            # S = K @ Q^T
            s_accs = [c_zero_v8f32 for _ in range(NUM_S_ACCS)]

            if wave_needs_kv_tile:
                for ks in range_constexpr(K_STEPS_QK):
                    # Packed byte offset for this K-step + lane half (4 bytes / i32).
                    k_col_packed = fx.Int64(ks * (K_STEP_QK // 2)) + klane * fx.Int64(4)

                    for st_idx in range_constexpr(N_SUB_TILES):
                        st_base_row = st_idx * K_SUB_N

                        k_row_a = lane16 + fx.Int64(st_base_row)
                        k_lds_a = k_base + k_row_a * K_STRIDE + k_col_packed
                        # Load one i32 frag from LDS (8 nibbles).
                        k_pack_a = Vec(lds_load_i8(k_lds_a, 4)).bitcast(fx.Int32)

                        k_row_b = lane16 + fx.Int64(st_base_row + 16)
                        k_lds_b = k_base + k_row_b * K_STRIDE + k_col_packed
                        k_pack_b = Vec(lds_load_i8(k_lds_b, 4)).bitcast(fx.Int32)

                        acc_idx_a = st_idx * 2
                        acc_idx_b = st_idx * 2 + 1
                        s_accs[acc_idx_a] = wmma_acc(k_pack_a, q_b_packs[ks], s_accs[acc_idx_a])
                        s_accs[acc_idx_b] = wmma_acc(k_pack_b, q_b_packs[ks], s_accs[acc_idx_b])

            s_raw = []
            for st in range_constexpr(NUM_S_ACCS):
                for r in range_constexpr(8):
                    s_raw.append(Vec(s_accs[st])[r])

            if const_expr(CAUSAL):
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_i32 = fx.Int32(klane)
                q_start_i32 = fx.Int32(q_start)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                q_limit_i32 = q_start_i32 + causal_br_off_i32
                tile_needs_mask = max_kv_col_i32 > q_limit_i32

                s_v0 = s_raw[0]
                s_v1 = s_raw[1]
                s_v2 = s_raw[2]
                s_v3 = s_raw[3]
                s_v4 = s_raw[4]
                s_v5 = s_raw[5]
                s_v6 = s_raw[6]
                s_v7 = s_raw[7]
                s_v8 = s_raw[8]
                s_v9 = s_raw[9]
                s_v10 = s_raw[10]
                s_v11 = s_raw[11]
                s_v12 = s_raw[12]
                s_v13 = s_raw[13]
                s_v14 = s_raw[14]
                s_v15 = s_raw[15]
                if tile_needs_mask:
                    klane_off_i32 = klane_i32 * fx.Int32(8)
                    _b0 = kv_start_i32 + fx.Int32(0) + klane_off_i32
                    s_v0 = (_b0 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v0)
                    _b1 = kv_start_i32 + fx.Int32(1) + klane_off_i32
                    s_v1 = (_b1 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v1)
                    _b2 = kv_start_i32 + fx.Int32(2) + klane_off_i32
                    s_v2 = (_b2 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v2)
                    _b3 = kv_start_i32 + fx.Int32(3) + klane_off_i32
                    s_v3 = (_b3 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v3)
                    _b4 = kv_start_i32 + fx.Int32(4) + klane_off_i32
                    s_v4 = (_b4 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v4)
                    _b5 = kv_start_i32 + fx.Int32(5) + klane_off_i32
                    s_v5 = (_b5 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v5)
                    _b6 = kv_start_i32 + fx.Int32(6) + klane_off_i32
                    s_v6 = (_b6 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v6)
                    _b7 = kv_start_i32 + fx.Int32(7) + klane_off_i32
                    s_v7 = (_b7 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v7)
                    _b8 = kv_start_i32 + fx.Int32(16) + klane_off_i32
                    s_v8 = (_b8 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v8)
                    _b9 = kv_start_i32 + fx.Int32(17) + klane_off_i32
                    s_v9 = (_b9 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v9)
                    _b10 = kv_start_i32 + fx.Int32(18) + klane_off_i32
                    s_v10 = (_b10 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v10)
                    _b11 = kv_start_i32 + fx.Int32(19) + klane_off_i32
                    s_v11 = (_b11 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v11)
                    _b12 = kv_start_i32 + fx.Int32(20) + klane_off_i32
                    s_v12 = (_b12 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v12)
                    _b13 = kv_start_i32 + fx.Int32(21) + klane_off_i32
                    s_v13 = (_b13 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v13)
                    _b14 = kv_start_i32 + fx.Int32(22) + klane_off_i32
                    s_v14 = (_b14 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v14)
                    _b15 = kv_start_i32 + fx.Int32(23) + klane_off_i32
                    s_v15 = (_b15 > (q_row_i32 + causal_br_off_i32)).select(c_neg_inf, s_v15)
                s_raw = [
                    s_v0,
                    s_v1,
                    s_v2,
                    s_v3,
                    s_v4,
                    s_v5,
                    s_v6,
                    s_v7,
                    s_v8,
                    s_v9,
                    s_v10,
                    s_v11,
                    s_v12,
                    s_v13,
                    s_v14,
                    s_v15,
                ]

            # Non-causal: mask K/V columns past real (unpadded) seq.
            if const_expr(not CAUSAL):
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_i32 = fx.Int32(klane)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                tile_needs_pad_mask = max_kv_col_i32 >= seq_len_kv_valid_i32
                s_v0 = s_raw[0]
                s_v1 = s_raw[1]
                s_v2 = s_raw[2]
                s_v3 = s_raw[3]
                s_v4 = s_raw[4]
                s_v5 = s_raw[5]
                s_v6 = s_raw[6]
                s_v7 = s_raw[7]
                s_v8 = s_raw[8]
                s_v9 = s_raw[9]
                s_v10 = s_raw[10]
                s_v11 = s_raw[11]
                s_v12 = s_raw[12]
                s_v13 = s_raw[13]
                s_v14 = s_raw[14]
                s_v15 = s_raw[15]
                if tile_needs_pad_mask:
                    klane_off_i32 = klane_i32 * fx.Int32(8)
                    _b0 = kv_start_i32 + fx.Int32(0) + klane_off_i32
                    s_v0 = (_b0 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v0)
                    _b1 = kv_start_i32 + fx.Int32(1) + klane_off_i32
                    s_v1 = (_b1 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v1)
                    _b2 = kv_start_i32 + fx.Int32(2) + klane_off_i32
                    s_v2 = (_b2 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v2)
                    _b3 = kv_start_i32 + fx.Int32(3) + klane_off_i32
                    s_v3 = (_b3 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v3)
                    _b4 = kv_start_i32 + fx.Int32(4) + klane_off_i32
                    s_v4 = (_b4 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v4)
                    _b5 = kv_start_i32 + fx.Int32(5) + klane_off_i32
                    s_v5 = (_b5 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v5)
                    _b6 = kv_start_i32 + fx.Int32(6) + klane_off_i32
                    s_v6 = (_b6 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v6)
                    _b7 = kv_start_i32 + fx.Int32(7) + klane_off_i32
                    s_v7 = (_b7 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v7)
                    _b8 = kv_start_i32 + fx.Int32(16) + klane_off_i32
                    s_v8 = (_b8 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v8)
                    _b9 = kv_start_i32 + fx.Int32(17) + klane_off_i32
                    s_v9 = (_b9 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v9)
                    _b10 = kv_start_i32 + fx.Int32(18) + klane_off_i32
                    s_v10 = (_b10 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v10)
                    _b11 = kv_start_i32 + fx.Int32(19) + klane_off_i32
                    s_v11 = (_b11 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v11)
                    _b12 = kv_start_i32 + fx.Int32(20) + klane_off_i32
                    s_v12 = (_b12 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v12)
                    _b13 = kv_start_i32 + fx.Int32(21) + klane_off_i32
                    s_v13 = (_b13 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v13)
                    _b14 = kv_start_i32 + fx.Int32(22) + klane_off_i32
                    s_v14 = (_b14 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v14)
                    _b15 = kv_start_i32 + fx.Int32(23) + klane_off_i32
                    s_v15 = (_b15 >= seq_len_kv_valid_i32).select(c_neg_inf, s_v15)
                s_raw = [
                    s_v0,
                    s_v1,
                    s_v2,
                    s_v3,
                    s_v4,
                    s_v5,
                    s_v6,
                    s_v7,
                    s_v8,
                    s_v9,
                    s_v10,
                    s_v11,
                    s_v12,
                    s_v13,
                    s_v14,
                    s_v15,
                ]

            local_max = s_raw[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, s_raw[r + 1])
            peer_max = reduction_peer(local_max)
            row_max = _fmax(local_max, peer_max)
            m_new_raw = _fmax(m_running, row_max)

            diff_m_raw = _fsub(m_running, m_new_raw)
            diff_m_scaled = _fmul(diff_m_raw, c_logit_scale)
            corr = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(diff_m_scaled).ir_value()))

            scaled_max = _fmul(c_logit_scale, m_new_raw)
            neg_scaled_max = _fsub(c_zero_f, scaled_max)

            p_vals = []
            local_sum = c_zero_f
            for r in range_constexpr(NUM_S_VALS):
                diff = fx.math.fma(
                    s_raw[r],
                    c_logit_scale,
                    neg_scaled_max,
                )
                p = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(diff).ir_value()))
                p_vals.append(p)
                local_sum = _fadd(local_sum, p)

            peer_sum = reduction_peer(local_sum)
            tile_sum = _fadd(local_sum, peer_sum)
            l_corr = _fmul(corr, l_running)
            l_new = _fadd(l_corr, tile_sum)

            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8)
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            coop_store_v_lds(_v_vecs_tile, 0)
            gpu.barrier()

            p_packs_all = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_packs_st = []
                for pks in range_constexpr(PV_K_STEPS):
                    acc_idx = st_idx * 2 + pks
                    p_base = acc_idx * 8
                    p_slice = [p_vals[p_base + j] for j in range(8)]
                    p_packs_st.append(iu4_pack_v8(p_slice))
                p_packs_all.append(p_packs_st)

            # O += V^T @ P, pipelined across V packs.
            v_base = v_buf_base(0)

            def _load_v_rowmajor(
                st_kv_base_val: fx.Int32 | fx.Int64 | int,
                pks_val: fx.Int32 | int,
                dc_val: fx.Int32 | int,
                v_base: fx.Int64 | int = v_base,
            ) -> fx.Vector:
                """Gather 8 packed nibbles along KV (PV-K) at logical D → i32 frag."""
                d_logical = fx.Int64(dc_val * D_CHUNK) + lane16
                d_byte = d_logical // fx.Int64(2)
                use_hi = (d_logical % fx.Int64(2)) != fx.Int64(0)
                nibs = []
                for k_sub in range_constexpr(8):
                    kv_row = (
                        fx.Int64(st_kv_base_val + pks_val * PV_K_STEP) + klane * fx.Int64(WMMA_LANE_K) + fx.Int64(k_sub)
                    )
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_byte
                    # Load packed byte via i32 LDS view (avoid i4).
                    raw_u8 = Vec(lds_load_i8(v_lds_idx, 1))
                    raw_b = fx.Int32(raw_u8[0].to(fx.Int32)) & fx.Int32(0xFF)
                    lo = raw_b & fx.Int32(0xF)
                    hi = (raw_b >> fx.Int32(4)) & fx.Int32(0xF)
                    nibs.append(use_hi.select(hi, lo))
                packed = (
                    (nibs[0] & fx.Int32(0xF))
                    | ((nibs[1] & fx.Int32(0xF)) << fx.Int32(4))
                    | ((nibs[2] & fx.Int32(0xF)) << fx.Int32(8))
                    | ((nibs[3] & fx.Int32(0xF)) << fx.Int32(12))
                    | ((nibs[4] & fx.Int32(0xF)) << fx.Int32(16))
                    | ((nibs[5] & fx.Int32(0xF)) << fx.Int32(20))
                    | ((nibs[6] & fx.Int32(0xF)) << fx.Int32(24))
                    | ((nibs[7] & fx.Int32(0xF)) << fx.Int32(28))
                )
                return Vec.from_elements([packed], fx.Int32)

            if wave_needs_kv_tile:
                o_tmp = list(o_accs)

                cur_v_packs = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    cur_v_packs.append(_load_v_rowmajor(st_idx * K_SUB_N, 0, 0))

                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        next_dc = dc + 1
                        next_pks = pks
                        if const_expr(next_dc >= D_CHUNKS):
                            next_dc = 0
                            next_pks = pks + 1
                        has_next = const_expr(next_pks < PV_K_STEPS)

                        next_v_packs = []
                        if const_expr(has_next):
                            for st_idx in range_constexpr(N_SUB_TILES):
                                next_v_packs.append(_load_v_rowmajor(st_idx * K_SUB_N, next_pks, next_dc))

                        for st_idx in range_constexpr(N_SUB_TILES):
                            o_tmp[dc] = wmma_acc(
                                cur_v_packs[st_idx],
                                p_packs_all[st_idx][pks],
                                o_tmp[dc],
                            )

                        if const_expr(has_next):
                            cur_v_packs = next_v_packs

                o_accs = o_tmp

            m_running = m_new_raw
            l_running = l_new

            if const_expr(PREFETCH_V_ACROSS_ITERS):
                next_kv_start = fx.Int64(kv_block_start) + fx.Int64(BLOCK_N_OUT)
                _v_vecs_tile = coop_load_v_global(next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(PREFETCH_V_ACROSS_ITERS):  # noqa: E741
                for vi in range_constexpr(NUM_V_VECS):
                    _yield_args.append(_v_vecs_tile[vi])
            loop_results = yield _yield_args

        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        # v_descale on the normalized accumulator (gfx950: inv_l *= vd_fp8).
        inv_l = (c_one_f / l_final) * v_descale * fx.Float32(1.0 / _P_I4_SCALE)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8)

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                o_trunc = Vec(o_norm_vec).to(out_dtype)
                d_col = fx.Int64(dc * D_CHUNK) + klane * 8
                o_global = global_idx_o(q_row, d_col)
                _store_global_half(o_elem_ptr, o_global, o_trunc)

    @flyc.jit
    def launch_flash_attn_iu4_func(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        seq_len_kv_valid: fx.Int32,
        q_descale: fx.Float32,
        k_descale: fx.Float32,
        v_descale: fx.Float32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ) -> None:
        ctx = CompilationContext.get_current()

        bs_idx = fx.Uint64(batch_size)
        sl_idx = fx.Uint64(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        launcher = flash_attn_func_iu4_native_kernel(
            Q,
            K,
            V,
            O,  # noqa: E741
            seq_len,
            seq_len_kv,
            seq_len_kv_valid,
            q_descale,
            k_descale,
            v_descale,
        )

        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
        if const_expr(flat_work_group_size is not None):
            _fwgs = int(flat_work_group_size)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("no-nans-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("unsafe-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    # Hoist PointerJitArg helpers once per built module (quant playbook).
    # Hot path: skip FakeTensor checks; never default to fx.Stream(None).
    _from_c_void_p = flyc.from_c_void_p
    _FX_UINT8 = fx.Uint8

    def _ptr_arg(t: object) -> PointerJitArg | object:
        """Cold/compile path: FakeTensor-safe."""
        if hasattr(t, "data_ptr"):
            type_name = type(t).__name__
            module_name = type(t).__module__
            ptr = 0 if type_name == "FakeTensor" or "fake_tensor" in module_name else t.data_ptr()
            return _from_c_void_p(_FX_UINT8, ptr)
        return t

    def _ptr_fast(t: torch.Tensor) -> PointerJitArg:
        """Hot inference: real CUDA tensors only."""
        return _from_c_void_p(_FX_UINT8, t.data_ptr())

    def _wrap_qkvo_fast(args: tuple | list, kwargs: dict) -> tuple[list, dict]:
        args = list(args)
        for idx in range(min(4, len(args))):
            a = args[idx]
            if hasattr(a, "data_ptr"):
                args[idx] = _ptr_fast(a)
        for name in ("Q", "K", "V", "O"):
            if name in kwargs:
                kwargs[name] = _ptr_fast(kwargs[name])
        return args, kwargs

    launch_flash_attn_iu4_func.compile_hints = dict(_fmha_compile_hints)
    _cached_cf = None  # closure-local CompiledFunction after first warm

    def _launch(*args, **kwargs) -> None:
        global _stack_opt_cf_hits, _stack_opt_run_compiled
        nonlocal _cached_cf  # noqa: E741
        args, kwargs = _wrap_qkvo_fast(args, kwargs)
        # Prefer torch current_stream — NEVER fx.Stream(None).
        stream = kwargs.pop("stream", None)
        if stream is None:
            stream = torch.cuda.current_stream()
        cf = _cached_cf or getattr(launch_flash_attn_iu4_func, "_cf", None)
        if cf is not None:
            _cached_cf = cf
            _stack_opt_cf_hits += 1
            cf(*args, stream)
            return
        _stack_opt_run_compiled += 1
        _run_compiled(launch_flash_attn_iu4_func, *args, stream)
        # Prefer CF attached to closed-over handle; else scan common stash.
        _cached_cf = getattr(launch_flash_attn_iu4_func, "_cf", None) or getattr(
            launch_flash_attn_iu4_func, "_last_compiled", None
        )

    def _compile(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,  # noqa: E741
        batch_size: int,
        seq_len: int,
        seq_len_kv: int | None,
        seq_len_kv_valid: int | None,
        q_descale: float = 1.0,
        k_descale: float = 1.0,
        v_descale: float = 1.0,
        stream: torch.cuda.Stream | None = None,
    ) -> CompiledFunction | None:
        if stream is None:
            stream = torch.cuda.current_stream()
        return flyc.compile(
            launch_flash_attn_iu4_func,
            _ptr_arg(Q),
            _ptr_arg(K),
            _ptr_arg(V),
            _ptr_arg(O),
            batch_size,
            seq_len,
            seq_len_kv,
            seq_len_kv_valid,
            float(q_descale),
            float(k_descale),
            float(v_descale),
            stream,
        )

    _launch.compile = _compile
    _launch.is_native_iu4_fa = True
    _launch.jit_function = launch_flash_attn_iu4_func
    return _launch
