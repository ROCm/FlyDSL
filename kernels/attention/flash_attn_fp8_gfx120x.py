# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Flash Attention FP8 (E4M3FN) forward kernel for gfx120x (RDNA4; HW may report gfx1201).

Q/K/V are Float8E4M3FN or Float8E5M2 storage;
accumulator is f32; output is bf16. Uses 16x16x16 wave32 WMMA with
``fx.rocdl.WMMA(..., elem_ty_ab=Float8E4M3FN)`` (RDNA4 floating-point WMMA
requires M=N=K=16; do NOT use gfx950 dualwave / ds_read_tr16).

Tiling mirrors the bf16 gfx120x FA kernel (BLOCK_M=128, BLOCK_N=32,
head_dim >= 64 and head_dim % 32 == 0). FP8 halves the LDS element footprint
vs bf16, so the same tiles stay well under the 64KiB LDS budget; we do not
blindly enlarge tiles in v1.

Per-tensor Q/K/V descales are one-element fp32 device buffers (gfx950-style;
host passes a pointer, kernel loads the scalar). Softmax uses ``sm_scale * q_descale * k_descale`` on the
raw QK logits (gfx950-compatible); ``v_descale`` multiplies the final
``inv_l`` normalization. Softmax P is cast f32->E4M3FN for the PV WMMA.
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
from kernels.attention.flash_attn_gfx120x_host import (  # noqa: E402
    add_alibi_scores,
    add_score_bias,
    apply_sliding_window,
    attention_inv_l,
    attention_lse,
    clear_o_if_empty,
    fold_attention_sink,
    kill_score_columns,
    online_softmax_tile,
)
from kernels.common.kernels_common import LOG2E as _LOG2E  # noqa: E402
from kernels.common.tensor_shim import _run_compiled  # noqa: E402

KERNEL_NAME = "flash_attn_func_fp8_gfx120x_kernel"


def build_flash_attn_func_fp8_module_primary(
    num_heads: int,
    head_dim: int,
    causal: bool = True,
    dtype_str: str = "fp8_e4m3fn",
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    flat_work_group_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    unsafe_fp_math: bool = True,
    fast_fp_math: bool = True,
    daz: bool = True,
    path_tag: str = "auto",  # ignored on gfx120x (API symmetry with generic FA)
    has_attn_bias: bool = False,
    has_per_head_bias: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    num_kv_heads: int | None = None,
    sign_a: bool = True,
    sign_b: bool = True,
    logical_head_dim: int | None = None,
    varlen: bool = False,
    paged: bool = False,
    page_size: int = 16,
    kv_cache_layout: str = "linear",
    split_k: bool = False,
    sliding_window: tuple[int, int] | None = None,
    bias_bottom_right: bool = False,
    has_alibi: bool = False,
    alibi_per_head: bool = False,
) -> Callable[..., None]:
    """Build the gfx120x FP8 Flash Attention kernel (QKV=E4M3FN, O=bf16)."""

    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    ROWS_PER_WAVE = WMMA_M

    BLOCK_M = block_m if block_m is not None else 128
    BLOCK_N = block_n if block_n is not None else 32

    assert BLOCK_N == 32, (
        f"gfx120x FA score-lane masks assume BLOCK_N==32 (got {BLOCK_N}); "
        "generalize N_SUB_TILES masking before using other block_n"
    )
    assert BLOCK_N % K_SUB_N == 0, f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})"
    assert BLOCK_M % ROWS_PER_WAVE == 0, f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    assert flat_work_group_size == NUM_WAVES * WARP_SIZE, (
        f"flat_work_group_size ({flat_work_group_size}) must equal NUM_WAVES*WARP_SIZE " f"({NUM_WAVES * WARP_SIZE})"
    )
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
    assert head_dim <= 480, f"gfx120x FA head_dim ({head_dim}) must be <= 480 (LDS budget at BLOCK_N=32 prefetch1)"
    # Bind cvt / clip / AB type at *build* scope (Python). Inside @flyc.kernel the
    # AST rewriter turns `a if cond else b` into scf_ifexp and breaks bf8 cvt.
    _IS_INT8 = dtype_str in ("int8", "i8")
    _FP8_E5M2 = dtype_str in ("fp8_e5m2", "e5m2")
    if _IS_INT8:
        _AB_TY = fx.Int8
        _PK_CVT = fx.rocdl.cvt_pk_fp8_f32
        _FP8_MAX = 448.0
    else:
        assert dtype_str in (
            "fp8_e4m3fn",
            "fp8",
            "fp8_e5m2",
            "e5m2",
        ), f"quant gfx120x FA expects fp8_e4m3fn, fp8_e5m2, or int8, got {dtype_str!r}"
        _AB_TY = fx.Float8E5M2 if _FP8_E5M2 else fx.Float8E4M3FN
        _PK_CVT = fx.rocdl.cvt_pk_bf8_f32 if _FP8_E5M2 else fx.rocdl.cvt_pk_fp8_f32
        _FP8_MAX = 57344.0 if _FP8_E5M2 else 448.0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    # WMMA tile is a multiple of 32. The caller's D can be shorter; global
    # strides use that length so the host does not clone Q, K, or V.
    HEAD_DIM = head_dim
    LOGICAL_D = head_dim if logical_head_dim is None else int(logical_head_dim)
    if not 1 <= LOGICAL_D <= HEAD_DIM:
        raise ValueError(f"logical_head_dim={LOGICAL_D} is outside 1..tile {HEAD_DIM}")
    # Dense BSHD GQA: same index as bf16 gfx120x (no host KV repeat).
    NUM_KV_HEADS = int(num_heads if num_kv_heads is None else num_kv_heads)
    if NUM_KV_HEADS <= 0 or int(num_heads) % NUM_KV_HEADS != 0:
        raise ValueError(f"gfx120x FA: num_heads={int(num_heads)} must be divisible by num_kv_heads={NUM_KV_HEADS}")
    KV_GROUP = int(num_heads) // NUM_KV_HEADS
    CAUSAL = causal
    HAS_ATTN_BIAS = bool(has_attn_bias) or bool(has_per_head_bias)
    HAS_PER_HEAD_BIAS = bool(has_per_head_bias)
    BIAS_BOTTOM_RIGHT = bool(bias_bottom_right)
    HAS_ALIBI = bool(has_alibi)
    ALIBI_PER_HEAD = bool(alibi_per_head)
    RETURN_LSE = bool(return_lse)
    HAS_SINK = bool(has_sink)
    VARLEN = bool(varlen)
    PAGED = bool(paged)
    SPLIT = bool(split_k)
    if sliding_window is None:
        HAS_WINDOW = False
        SWA_LEFT = 0
        SWA_RIGHT = 0
    else:
        if len(sliding_window) != 2:
            raise ValueError(f"sliding_window must be (left, right), got {sliding_window!r}")
        SWA_LEFT, SWA_RIGHT = int(sliding_window[0]), int(sliding_window[1])
        if SWA_LEFT < 0 or SWA_RIGHT < 0:
            raise ValueError(f"sliding_window (left, right) must be >= 0, got {sliding_window!r}")
        HAS_WINDOW = True
    PAGE_SIZE = int(page_size) if PAGED else 16
    if PAGED and PAGE_SIZE <= 0:
        raise ValueError(f"gfx120x quant FA: page_size must be > 0, got {PAGE_SIZE}")
    _kv_layout = kv_cache_layout or "linear"
    if _kv_layout not in ("linear", "linear3d", "vectorized"):
        raise ValueError(f"gfx120x quant FA: unknown kv_cache_layout {_kv_layout!r}")
    if PAGED and _kv_layout == "linear3d" and int(page_size) != 1:
        raise ValueError("gfx120x quant FA: linear3d paged KV requires page_size=1")
    # fp8 and int8 are 1 byte, so the aiter vector group is 16. bf16/fp16 stay at 8.
    KV_VEC = 16
    KV_VECTORIZED = bool(PAGED) and _kv_layout == "vectorized"
    if KV_VECTORIZED and (int(head_dim) % KV_VEC != 0 or int(page_size) % KV_VEC != 0):
        raise ValueError(f"gfx120x quant FA: vectorized KV needs head_dim and page_size divisible by {KV_VEC}")
    STRIDE_TOKEN = NUM_HEADS * LOGICAL_D
    KV_STRIDE = NUM_KV_HEADS * int(LOGICAL_D)

    # Padding reduces LDS bank conflicts.
    K_STRIDE = HEAD_DIM + 4
    V_STRIDE = HEAD_DIM + 4

    # FP8: buffer dwordx4 = 16 bytes = 16 E4M3FN elems. Keep VEC_WIDTH=16.
    # V rows still fetch in 8-element pieces to match WMMA lane K packing.
    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    V_LOAD_WIDTH = 8  # elements per V global load (8 bytes)
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    # Multi-batch KV loads: every BLOCK_N row must be covered. Floor division
    # silently dropped the tail when ROWS_PER_BATCH_LOAD did not divide BLOCK_N
    # (common for soft-padded head dims). Use ceil batches and guard surplus
    # LDS rows whenever coverage is partial or threads overshoot BLOCK_N.
    if ROWS_PER_BATCH_LOAD <= 0:
        raise ValueError(
            f"ROWS_PER_BATCH_LOAD must be > 0 (BLOCK_SIZE={BLOCK_SIZE}, "
            f"THREADS_PER_ROW_LOAD={THREADS_PER_ROW_LOAD})"
        )
    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = (BLOCK_N + ROWS_PER_BATCH_LOAD - 1) // ROWS_PER_BATCH_LOAD
        # Always guard when multi-batch: last ceil batch may be partial.
        KV_NEEDS_GUARD = True

    # Buffer loads cap at dwordx4 (16B); V pieces are V_LOAD_WIDTH elems.
    V_SUBVECS = VEC_WIDTH // V_LOAD_WIDTH
    NUM_V_VECS = NUM_BATCHES_KV * V_SUBVECS

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # Memory/LDS/AB fragments store E4M3FN *bits* as Int8. Using f8E4M3FN as
    # the runtime vector type makes LLVM conversion emit illegal
    # i8<->f8 materializations around WMMA (ABI is vector<8xi8>). The WMMA
    # atom itself is still typed Float8E4M3FN. Output is bf16.
    elem_numeric_cls = fx.Int8
    out_numeric_cls = fx.BFloat16

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_fp8_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        seq_len_kv_valid: fx.Int32,
        QDescale: fx.Pointer,
        KDescale: fx.Pointer,
        VDescale: fx.Pointer,
        Bias: fx.Pointer,
        Slopes: fx.Pointer,
        LSE: fx.Pointer,
        Sink: fx.Pointer,
        CuSeqlensQ: fx.Pointer,
        CuSeqlensKV: fx.Pointer,
        BlockTable: fx.Pointer,
        SeqlenK: fx.Pointer,
        block_table_stride: fx.Int32,
        num_splits: fx.Int32,
        WsM: fx.Pointer,
        WsL: fx.Pointer,
        WsO: fx.Pointer,
        ws_rows: fx.Int32,
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

        def _load_descale(ptr: fx.Pointer) -> fx.Float32:
            view = fx.make_view(
                fx.recast_iter(
                    fx.PointerType.get(fx.Float32.ir_type, ptr.address_space),
                    ptr,
                ),
                fx.make_layout(1, 1),
            )
            return fx.Float32(view.load()[0])

        q_descale = _load_descale(QDescale)
        k_descale = _load_descale(KDescale)
        v_descale = _load_descale(VDescale)

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

        if const_expr(HAS_ATTN_BIAS):
            bias_elem_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, Bias.address_space),
                Bias,
            )
            _bias_tiles = fx.Int64(seq_len) * fx.Int64(seq_len_kv)
            if const_expr(VARLEN and not HAS_PER_HEAD_BIAS):
                # Packed [total_q, Sk]. A later sequence starts at q_off >= max_q.
                bias_bytes = fx.Int64(0x7FFFFFFF)
            elif const_expr(HAS_PER_HEAD_BIAS):
                bias_bytes = _bias_tiles * fx.Int64(NUM_HEADS) * fx.Int64(4)
            else:
                bias_bytes = _bias_tiles * fx.Int64(4)
            bias_buf = _bounds_checked_buf_ptr(bias_elem_ptr, bias_bytes)
        if const_expr(HAS_ALIBI):
            slopes_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, Slopes.address_space),
                Slopes,
            )
        if const_expr(RETURN_LSE):
            lse_elem_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, LSE.address_space),
                LSE,
            )
        if const_expr(HAS_SINK):
            sink_elem_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, Sink.address_space),
                Sink,
            )

        if const_expr(_IS_INT8):
            wmma_atom = fx.make_mma_atom(
                fx.rocdl.WMMA(
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    fx.Int8,
                    fx.Int32,
                    sign_a=sign_a,
                    sign_b=sign_b,
                    clamp=False,
                )
            )

            def wmma_acc_i32(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
                a_frag = fx.make_rmem_tensor(8, fx.Int8)
                b_frag = fx.make_rmem_tensor(8, fx.Int8)
                c_frag = fx.make_rmem_tensor(8, fx.Int32)
                a_frag.store(Vec(a_v8))
                b_frag.store(Vec(b_v8))
                c_frag.store(Vec(c_v8))
                fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)
                return Vec(c_frag.load())

            def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8_f32: object) -> fx.Vector:
                # Each K step starts at i32 zero and adds the widened product
                # into the f32 carry, matching the fp8 accumulator ABI.
                c0 = Vec.filled(8, 0, fx.Int32)
                iacc = wmma_acc_i32(a_v8, b_v8, c0)
                parts = []
                for i in range_constexpr(8):
                    parts.append(fx.Float32(iacc[i]) + fx.Float32(c_v8_f32[i]))
                return Vec.from_elements(parts, fx.Float32)

        else:
            # RDNA4 FP8 WMMA: M=N=K=16, AB=E4M3FN or E5M2, acc=f32 (v8-operand ABI).
            wmma_atom = fx.make_mma_atom(
                fx.rocdl.WMMA(
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    _AB_TY,
                    fx.Float32,
                )
            )

            def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
                a_frag = fx.make_rmem_tensor(8, elem_dtype)
                b_frag = fx.make_rmem_tensor(8, elem_dtype)
                c_frag = fx.make_rmem_tensor(8, fx.Float32)
                a_frag.store(Vec(a_v8))
                b_frag.store(Vec(b_v8))
                c_frag.store(Vec(c_v8))
                fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)  # FlyDSL 0.3.4.1: a/b Sequence
                return Vec(c_frag.load())

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

        if const_expr(SPLIT):
            block_id_all = fx.Uint64(gpu.block_idx.x)
            split_idx = block_id_all % fx.Uint64(num_splits)
            block_id = block_id_all // fx.Uint64(num_splits)
        else:
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

        if const_expr(VARLEN):
            _cu_q_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, CuSeqlensQ.address_space),
                CuSeqlensQ,
            )
            q_off_i32 = fx.Int32(fx.ptr_load(_cu_q_ptr + fx.Int32(batch_idx)))
            q_end_i32 = fx.Int32(fx.ptr_load(_cu_q_ptr + fx.Int32(batch_idx) + fx.Int32(1)))
            q_off_i64 = fx.Int64(q_off_i32)
            sq_i32 = q_end_i32 - q_off_i32
        if const_expr(PAGED):
            _sk_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, SeqlenK.address_space),
                SeqlenK,
            )
            seq_len_kv_valid_i32 = fx.Int32(fx.ptr_load(_sk_ptr + fx.Int32(batch_idx)))
            k_off_i64 = fx.Int64(0)
            _bt_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, BlockTable.address_space),
                BlockTable,
            )
            _bt_stride_i32 = fx.Int32(block_table_stride)
        if const_expr(VARLEN and not PAGED):
            _cu_kv_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, CuSeqlensKV.address_space),
                CuSeqlensKV,
            )
            k_off_i32 = fx.Int32(fx.ptr_load(_cu_kv_ptr + fx.Int32(batch_idx)))
            k_end_i32 = fx.Int32(fx.ptr_load(_cu_kv_ptr + fx.Int32(batch_idx) + fx.Int32(1)))
            k_off_i64 = fx.Int64(k_off_i32)
            seq_len_kv_valid_i32 = k_end_i32 - k_off_i32

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx_q(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            if const_expr(VARLEN):
                token = q_off_i64 + fx.Int64(token_idx)
            else:
                token = batch_idx * seq_len_q_v + token_idx
            return token * STRIDE_TOKEN + head_idx * LOGICAL_D + col

        def global_idx_kv(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            if const_expr(PAGED):
                _tok = fx.Int32(token_idx)
                _page_idx = _tok // fx.Int32(PAGE_SIZE)
                _page_off = _tok % fx.Int32(PAGE_SIZE)
                _bt_idx = fx.Int32(batch_idx) * _bt_stride_i32 + _page_idx
                _pid = fx.Int32(fx.ptr_load(_bt_ptr + _bt_idx))
                return (
                    fx.Int64(_pid) * fx.Int64(PAGE_SIZE * KV_STRIDE)
                    + fx.Int64(_page_off) * fx.Int64(KV_STRIDE)
                    + kv_head_idx * LOGICAL_D
                    + col
                )
            if const_expr(VARLEN):
                token = k_off_i64 + fx.Int64(token_idx)
                return token * KV_STRIDE + kv_head_idx * LOGICAL_D + col
            token = batch_idx * seq_len_kv_v + token_idx
            return token * KV_STRIDE + kv_head_idx * LOGICAL_D + col

        # Back-compat: historical call sites meant Q indexing.
        def global_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            return global_idx_q(token_idx, col)

        # Hardware OOB handling keeps the tail prefetch branch-free. A batch
        # slice must fit the descriptor's 32-bit num_records.
        ELEM_BYTES = (elem_numeric_cls.width + 7) // 8
        if const_expr(PAGED or VARLEN):
            # Packed or paged addresses are absolute from the tensor base.
            v_buf_ptr = _bounds_checked_buf_ptr(v_elem_ptr, fx.Int64(0x7FFFFFFF))
            k_buf_ptr = _bounds_checked_buf_ptr(k_elem_ptr, fx.Int64(0x7FFFFFFF))

            def _clamp_tok(token_idx: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                _sk_i64 = fx.Int64(seq_len_kv_valid_i32)
                _last = (_sk_i64 > fx.Int64(0)).select(_sk_i64 - fx.Int64(1), fx.Int64(0))
                return (fx.Int64(token_idx) < _sk_i64).select(fx.Int64(token_idx), _last)

            def v_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                return global_idx_kv(_clamp_tok(token_idx), col)

            def k_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                return global_idx_kv(_clamp_tok(token_idx), col)

        else:
            kv_batch_elems = seq_len_kv_v * fx.Uint64(KV_STRIDE)
            v_buf_ptr = _bounds_checked_buf_ptr(
                fx.add_offset(v_elem_ptr, fx.Int64(batch_idx * kv_batch_elems)),
                fx.Int64(kv_batch_elems) * fx.Int64(ELEM_BYTES),
            )
            # Mirror V: bounds-checked K descriptor so last-tile overhang rows
            # zero-fill instead of raw OOB global reads (host may pad, but the
            # kernel ABI itself must stay safe for unpadded seq_len_kv).
            k_buf_ptr = _bounds_checked_buf_ptr(
                fx.add_offset(k_elem_ptr, fx.Int64(batch_idx * kv_batch_elems)),
                fx.Int64(kv_batch_elems) * fx.Int64(ELEM_BYTES),
            )

            def v_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                return token_idx * KV_STRIDE + kv_head_idx * LOGICAL_D + col

            def k_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                return token_idx * KV_STRIDE + kv_head_idx * LOGICAL_D + col

        def _as_i32_ptr(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, ptr.address_space),
                ptr,
            )

        def _load_global_i8_vec_via_i32(
            elem_ptr: fx.Pointer,
            base_idx: fx.Int32 | fx.Int64 | int,
            width: int,
            col: fx.Int32 | fx.Int64 | int | None = None,
        ) -> fx.Vector:
            # Buffer loads cannot return v8i8 / odd i8 vectors; transfer dwords
            # then bitcast to i8 (fp8 bits). width must be a multiple of 4.
            # Aligned D keeps this wide load with no tail select, so its ISA
            # does not pay for the odd-D path below.
            assert width % 4 == 0
            n_i32 = width // 4
            if const_expr(LOGICAL_D == HEAD_DIM):
                i32_ptr = _as_i32_ptr(elem_ptr)
                aligned = fx.make_view(
                    fx.add_offset(i32_ptr, fx.Int64(base_idx) // fx.Int64(4)),
                    fx.make_layout(n_i32, 1),
                )
                return Vec(aligned.load()).bitcast(fx.Int8)
            # Caller D is shorter than the WMMA tile. Not traced when
            # LOGICAL_D == HEAD_DIM. The pointer must be a plain element
            # pointer: selecting a buffer-descriptor address lowers to an
            # align-1 raw buffer load, and a scalar buffer load does not
            # legalize. Bytes at or past LOGICAL_D reload offset 0 and are
            # discarded, the same rule as the GEMM K tail.
            col0 = fx.Int64(col)
            if const_expr(int(LOGICAL_D) % width == 0):
                in_row = col0 + fx.Int64(width) <= fx.Int64(LOGICAL_D)
                i32_ptr = _as_i32_ptr(elem_ptr)
                safe = in_row.select(fx.Int64(base_idx), fx.Int64(0))
                wide = fx.make_view(
                    fx.add_offset(i32_ptr, safe // fx.Int64(4)),
                    fx.make_layout(n_i32, 1),
                )
                raw = Vec(wide.load())
                z = fx.Int32(0)
                words = []
                for i in range_constexpr(n_i32):
                    words.append(in_row.select(fx.Int32(raw[i]), z))
                return Vec.from_elements(words, fx.Int32).bitcast(fx.Int8)
            i8_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int8.ir_type, elem_ptr.address_space),
                elem_ptr,
            )
            bvals = []
            for b in range_constexpr(width):
                take = col0 + fx.Int64(b) < fx.Int64(LOGICAL_D)
                off = take.select(fx.Int64(base_idx) + fx.Int64(b), fx.Int64(0))
                one = fx.make_view(fx.add_offset(i8_ptr, off), fx.make_layout(1, 1))
                raw_b = fx.Int32(Vec(one.load())[0]) & fx.Int32(255)
                bvals.append(take.select(raw_b, fx.Int32(0)))
            words = []
            for w in range_constexpr(n_i32):
                acc = fx.Int32(0)
                for t in range_constexpr(4):
                    acc = acc | (bvals[w * 4 + t] << fx.Int32(8 * t))
                words.append(acc)
            return Vec.from_elements(words, fx.Int32).bitcast(fx.Int8)

        def _store_global_half(elem_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int, val: fx.Vector) -> None:
            # O is bf16 -> None: store as-is via typed out pointer (not i8 path).
            view = fx.make_view(
                fx.add_offset(elem_ptr, fx.Int64(base_idx)),
                fx.make_layout(val.numel, 1),
            )
            view.store(Vec(val))

        def load_global_f8xN(
            base_ptr: fx.Pointer,
            base_idx: fx.Int32 | fx.Int64 | int,
            col: fx.Int32 | fx.Int64 | int | None = None,
        ) -> fx.Vector:
            return _load_global_i8_vec_via_i32(base_ptr, base_idx, VEC_WIDTH, col)

        def load_global_v8f8(
            base_ptr: fx.Pointer,
            base_idx: fx.Int32 | fx.Int64 | int,
            col: fx.Int32 | fx.Int64 | int | None = None,
        ) -> fx.Vector:
            return _load_global_i8_vec_via_i32(base_ptr, base_idx, 8, col)

        def _col_in_alloc(row_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            # A KV row past seq_len_kv must not touch that address. Forcing the
            # column to LOGICAL_D makes every byte of the vector a discarded
            # reload of offset 0, which stays inside a non-empty allocation.
            row_ok = fx.Uint64(row_idx) < seq_len_kv_v
            return row_ok.select(fx.Int64(col), fx.Int64(LOGICAL_D))

        def _zero_i8(width: int) -> fx.Vector:
            return Vec.from_elements([fx.Int8(0) for _ in range_constexpr(width)], fx.Int8)

        def _paged_pid_off(token_idx: fx.Int32 | fx.Int64 | int) -> tuple[fx.Int32, fx.Int32]:
            _tok = fx.Int32(_clamp_tok(token_idx))
            _page_idx = _tok // fx.Int32(PAGE_SIZE)
            _page_off = _tok % fx.Int32(PAGE_SIZE)
            _bt_idx = fx.Int32(batch_idx) * _bt_stride_i32 + _page_idx
            _pid = fx.Int32(fx.ptr_load(_bt_ptr + _bt_idx))
            return _pid, _page_off

        def _vec_k_byte(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            """Byte of K[block, head, dim/16, page_off, 16]. ``col`` is the dim."""
            _pid, _poff = _paged_pid_off(token_idx)
            group = fx.Int64(col) // fx.Int64(KV_VEC)
            inner = fx.Int64(col) % fx.Int64(KV_VEC)
            return (
                (
                    (fx.Int64(_pid) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * fx.Int64(HEAD_DIM // KV_VEC)
                    + group
                )
                * fx.Int64(PAGE_SIZE)
                + fx.Int64(_poff)
            ) * fx.Int64(KV_VEC) + inner

        def _vec_v_byte(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            """Byte of V[block, head, page_off/16, dim, 16]."""
            _pid, _poff = _paged_pid_off(token_idx)
            kg = fx.Int32(_poff) // fx.Int32(KV_VEC)
            kr = fx.Int32(_poff) % fx.Int32(KV_VEC)
            return (
                (
                    (fx.Int64(_pid) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * fx.Int64(PAGE_SIZE // KV_VEC)
                    + fx.Int64(kg)
                )
                * fx.Int64(HEAD_DIM)
                + fx.Int64(col)
            ) * fx.Int64(KV_VEC) + fx.Int64(kr)

        def _byte_from_dword(word: fx.Int32, which: fx.Int32) -> fx.Int32:
            b0 = word & fx.Int32(255)
            b1 = word.shrui(fx.Int32(8)) & fx.Int32(255)
            b2 = word.shrui(fx.Int32(16)) & fx.Int32(255)
            b3 = word.shrui(fx.Int32(24)) & fx.Int32(255)
            got = (which == fx.Int32(1)).select(b1, b0)
            got = (which == fx.Int32(2)).select(b2, got)
            return (which == fx.Int32(3)).select(b3, got)

        def _load_one_byte(ptr: fx.Pointer, byte_index: fx.Int64) -> fx.Int32:
            # One dword. A 1-byte buffer load does not legalize.
            aligned = byte_index & fx.Int64(-4)
            which = fx.Int32(byte_index) & fx.Int32(3)
            word = Vec(_load_global_i8_vec_via_i32(ptr, aligned, 4, fx.Int32(0))).bitcast(fx.Int32)
            return _byte_from_dword(fx.Int32(word[0]), which)

        def _pack_bytes(raw: list) -> fx.Vector:
            words = []
            n_words = len(raw) // 4
            for w in range_constexpr(n_words):
                acc = fx.Int32(0)
                for t in range_constexpr(4):
                    acc = acc | (raw[w * 4 + t] << fx.Int32(8 * t))
                words.append(acc)
            return Vec.from_elements(words, fx.Int32).bitcast(fx.Int8)

        def load_k_global(row_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Vector:
            if const_expr(KV_VECTORIZED):
                return load_global_f8xN(k_elem_ptr, _vec_k_byte(row_idx, col), col)
            if const_expr(LOGICAL_D == HEAD_DIM):
                # Aligned: the buffer record is the real allocation. A short or
                # empty K returns zeros. No host pad, no tail branch.
                return load_global_f8xN(k_buf_ptr, k_idx(row_idx, col), col)
            # Odd D reloads offset 0 for an OOB row. An empty allocation has no
            # offset 0, so that specialization returns zeros instead of loading.
            if seq_len_kv_v > fx.Uint64(0):
                vec = load_global_f8xN(k_elem_ptr, global_idx_kv(row_idx, col), _col_in_alloc(row_idx, col))
            else:
                vec = _zero_i8(VEC_WIDTH)
            return vec

        def load_v_global(row_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int, width: int) -> fx.Vector:
            if const_expr(KV_VECTORIZED):
                raw = []
                for _i in range_constexpr(width):
                    raw.append(_load_one_byte(v_elem_ptr, _vec_v_byte(row_idx, fx.Int64(col) + fx.Int64(_i))))
                return _pack_bytes(raw)
            if const_expr(LOGICAL_D == HEAD_DIM):
                return _load_global_i8_vec_via_i32(v_buf_ptr, v_idx(row_idx, col), width, col)
            if seq_len_kv_v > fx.Uint64(0):
                vec = _load_global_i8_vec_via_i32(
                    v_elem_ptr, global_idx_kv(row_idx, col), width, _col_in_alloc(row_idx, col)
                )
            else:
                vec = _zero_i8(width)
            return vec

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

        def fp8_pack_v8(f32_vals: fx.Vector) -> fx.Vector:
            """Pack 8 f32 probs to E4M3FN/E5M2 bits via v_cvt_pk_{fp8,bf8}_f32 (as i8 x8).

            Uses build-scope ``_PK_CVT`` / ``_FP8_MAX`` (not kernel-local ternaries).
            """
            words = []
            _max = fx.Float32(_FP8_MAX)
            _nmax = fx.Float32(-_FP8_MAX)
            for i in range_constexpr(2):
                b = i * 4
                clipped = []
                for j in range_constexpr(4):
                    v = fx.Float32(f32_vals[b + j])
                    v = (v > _max).select(_max, (v < _nmax).select(_nmax, v))
                    clipped.append(v)
                lo = _PK_CVT(T.i32, clipped[0], clipped[1], fx.Int32(0), False)
                words.append(_PK_CVT(T.i32, clipped[2], clipped[3], lo, True))
            return Vec.from_elements(words, fx.Int32).bitcast(fx.Int8)

        def int8_pack_v8(f32_vals: fx.Vector) -> fx.Vector:
            """Quantize 8 f32 probs to signed int8 (scale 127) for iu8 PV WMMA."""
            elems = []
            scale = fx.Float32(127.0)
            half = fx.Float32(0.5)
            lo_i = fx.Int32(-128)
            hi_i = fx.Int32(127)
            for j in range_constexpr(8):
                x = fx.Float32(f32_vals[j]) * scale
                pos = x >= fx.Float32(0.0)
                adj = pos.select(x + half, x - half)
                qi = adj.to(fx.Int32)
                qi = fx.max(fx.min(qi, hi_i), lo_i)
                elems.append(qi)
            words = []
            for w in range_constexpr(2):
                b = w * 4
                packed = (
                    (elems[b + 0] & fx.Int32(255))
                    | ((elems[b + 1] & fx.Int32(255)) << fx.Int32(8))
                    | ((elems[b + 2] & fx.Int32(255)) << fx.Int32(16))
                    | ((elems[b + 3] & fx.Int32(255)) << fx.Int32(24))
                )
                words.append(packed)
            return Vec.from_elements(words, fx.Int32).bitcast(fx.Int8)

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
                lds_row = load_row_in_batch + row_offset
                row_idx = tile_start + lds_row
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = lds_row < fx.Int64(BLOCK_N)
                    if row_valid:
                        lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                        vec = load_k_global(row_idx, load_col_base)
                        lds_store_i8(lds_idx, Vec(vec))
                else:
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    vec = load_k_global(row_idx, load_col_base)
                    lds_store_i8(lds_idx, Vec(vec))

        def _v_store_row_major(
            v_base: fx.Int64, lds_row: fx.Int32 | fx.Int64 | int, col_extra: int, vec: fx.Vector
        ) -> None:
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base + col_extra
            lds_store_i8(lds_idx, Vec(vec))

        def coop_load_v_global(tile_start: fx.Int32 | fx.Int64 | int) -> list[fx.Vector]:
            tile_start = fx.Int64(tile_start)
            # Clamp surplus LDS rows; OOB global loads hit the bounds-checked
            # V descriptor (zero-fill) and stores are skipped in coop_store_v_lds.
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                lds_row = load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_cap = fx.Int64(BLOCK_N - 1)
                    lds_row = fx.Int64((lds_row < row_cap).select(lds_row, row_cap))
                row_idx = tile_start + lds_row
                for sv in range_constexpr(V_SUBVECS):
                    v_col = load_col_base + fx.Int64(sv * V_LOAD_WIDTH)
                    vecs.append(load_v_global(row_idx, v_col, V_LOAD_WIDTH))
            return vecs

        def coop_store_v_lds(vecs: list[fx.Vector], buf_id: fx.Int32 | int = 0) -> None:
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                lds_row = load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = lds_row < fx.Int64(BLOCK_N)
                    if row_valid:
                        for sv in range_constexpr(V_SUBVECS):
                            _v_store_row_major(
                                v_base,
                                lds_row,
                                sv * V_LOAD_WIDTH,
                                vecs[batch * V_SUBVECS + sv],
                            )
                else:
                    for sv in range_constexpr(V_SUBVECS):
                        _v_store_row_major(
                            v_base,
                            lds_row,
                            sv * V_LOAD_WIDTH,
                            vecs[batch * V_SUBVECS + sv],
                        )

        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)

        if const_expr(VARLEN):
            causal_len_q_i32 = sq_i32
            q_in_bounds = q_row < fx.Uint64(sq_i32)
        else:
            causal_len_q_i32 = fx.Int32(seq_len)
            q_in_bounds = q_row < seq_len_v
        q_row_safe = fx.Int64(q_in_bounds.select(q_row, fx.Int64(0)))

        # First KV column fully masked for this wave (bottom-right causal).
        wave_kv_limit_i32 = fx.Int32(q_start + wave_q_offset + fx.Int64(ROWS_PER_WAVE)) + (
            seq_len_kv_valid_i32 - causal_len_q_i32
        )
        # v1: no OOB select on fp8 vectors (arith.select f8<->i8 fails to
        # legalize with WMMA AB). q_row_safe zeros the address; output is
        # still gated by q_in_bounds. Revisit with i8-memory path later.
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Int64(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            q_b_packs.append(load_global_v8f8(q_elem_ptr, g_idx, q_col))

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
        causal_br_off_i32 = seq_len_kv_valid_i32 - causal_len_q_i32
        if const_expr(CAUSAL):
            _last_allow = (_q_end - fx.Int64(1)) + fx.Int64(causal_br_off_i32)
            _last_allow = fx.Int64((_last_allow < fx.Int64(0)).select(fx.Int64(0), _last_allow))
            _kv_end = _last_allow + fx.Int64(1)
            kv_upper = fx.Int64((_kv_end < seq_len_kv_v).select(_kv_end, seq_len_kv_v))
        else:
            kv_upper = seq_len_kv_v

        if const_expr(SPLIT):
            ns_i32 = fx.Int32(num_splits)
            sp_i32 = fx.Int32(split_idx)
            sk_for_split = seq_len_kv_valid_i32
            chunk_i32 = (sk_for_split + ns_i32 - fx.Int32(1)) // ns_i32
            kv_lo_i32 = sp_i32 * chunk_i32
            kv_hi_i32 = kv_lo_i32 + chunk_i32
            kv_hi_i32 = (kv_hi_i32 < sk_for_split).select(kv_hi_i32, sk_for_split)
            range_lo = fx.Int64(kv_lo_i32)
            _hi_cap = fx.Int32(kv_upper)
            range_hi = fx.Int64((_hi_cap < kv_hi_i32).select(_hi_cap, kv_hi_i32))
        else:
            range_lo = fx.Int64(0)
            range_hi = kv_upper

        # Non-causal carries prefetched V across iterations; causal avoids the
        # extra VGPR lifetime and loads V in the current iteration.
        PREFETCH_V_ACROSS_ITERS = not CAUSAL

        if const_expr(PREFETCH_V_ACROSS_ITERS):
            _v_vecs_init = coop_load_v_global(range_lo)

        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v8f32)
        if const_expr(PREFETCH_V_ACROSS_ITERS):
            for vi in range_constexpr(NUM_V_VECS):
                init_args.append(_v_vecs_init[vi])
        # 1 after the first tile that kept a key. Keeps fmax/exp2 off the
        # initial -inf when the window drops an entire leading KV tile.
        if const_expr(HAS_WINDOW):
            init_args.append(c_zero_f)

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(range_lo, range_hi, fx.Int64(BLOCK_N_OUT), init=init_args):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            if const_expr(PREFETCH_V_ACROSS_ITERS):
                _v_vecs_tile = [inner_iter_args[2 + D_CHUNKS + b] for b in range_constexpr(NUM_V_VECS)]
            row_live = c_zero_f
            if const_expr(HAS_WINDOW):
                _seen_slot = 2 + D_CHUNKS + (NUM_V_VECS if PREFETCH_V_ACROSS_ITERS else 0)
                seen = inner_iter_args[_seen_slot]

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
                    k_col = fx.Int64(ks * K_STEP_QK) + klane * WMMA_LANE_K

                    for st_idx in range_constexpr(N_SUB_TILES):
                        st_base_row = st_idx * K_SUB_N

                        k_row_a = lane16 + fx.Int64(st_base_row)
                        k_lds_a = k_base + k_row_a * K_STRIDE + k_col
                        k_pack_a = Vec(lds_load_i8(k_lds_a, 8))

                        k_row_b = lane16 + fx.Int64(st_base_row + 16)
                        k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                        k_pack_b = Vec(lds_load_i8(k_lds_b, 8))

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

                if tile_needs_mask:
                    s_raw = kill_score_columns(
                        s_raw,
                        kv_start_i32,
                        klane_i32 * fx.Int32(8),
                        ((">", q_row_i32 + causal_br_off_i32),),
                        c_neg_inf,
                    )
                if const_expr(HAS_WINDOW):
                    s_raw, row_live = apply_sliding_window(
                        s_raw,
                        kv_start_i32,
                        klane_i32,
                        q_row_i32,
                        swa_left=SWA_LEFT,
                        swa_right=SWA_RIGHT,
                        causal=True,
                        causal_br_off_i32=causal_br_off_i32,
                        seq_len_kv_valid_i32=seq_len_kv_valid_i32,
                        reduction_peer=reduction_peer,
                        fmax=_fmax,
                        c_neg_inf=c_neg_inf,
                        c_zero_f=c_zero_f,
                        c_one_f=c_one_f,
                    )

            # Non-causal: mask K/V columns past real (unpadded) seq.
            if const_expr(not CAUSAL):
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_i32 = fx.Int32(klane)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                tile_needs_pad_mask = max_kv_col_i32 >= seq_len_kv_valid_i32
                if tile_needs_pad_mask:
                    s_raw = kill_score_columns(
                        s_raw,
                        kv_start_i32,
                        klane_i32 * fx.Int32(8),
                        ((">=", seq_len_kv_valid_i32),),
                        c_neg_inf,
                    )
                if const_expr(HAS_WINDOW):
                    s_raw, row_live = apply_sliding_window(
                        s_raw,
                        kv_start_i32,
                        klane_i32,
                        q_row_i32,
                        swa_left=SWA_LEFT,
                        swa_right=SWA_RIGHT,
                        causal=False,
                        causal_br_off_i32=causal_br_off_i32,
                        seq_len_kv_valid_i32=seq_len_kv_valid_i32,
                        reduction_peer=reduction_peer,
                        fmax=_fmax,
                        c_neg_inf=c_neg_inf,
                        c_zero_f=c_zero_f,
                        c_one_f=c_one_f,
                    )

            if const_expr(SPLIT):
                # A split chunk can be narrower than the 32-wide tile. Drop the
                # columns that belong to another split so they are not counted twice.
                s_raw = kill_score_columns(
                    s_raw,
                    fx.Int32(kv_block_start),
                    fx.Int32(klane) * fx.Int32(8),
                    (("<", kv_lo_i32), (">=", kv_hi_i32)),
                    c_neg_inf,
                )

            if const_expr(HAS_ATTN_BIAS):
                if q_in_bounds:
                    kv_start_i32 = fx.Int32(kv_block_start)
                    klane_off_i32 = fx.Int32(klane) * fx.Int32(8)
                    # Bias is in dequantized logit units, added after sm_scale.
                    # Raw WMMA scores still include q_descale*k_descale and sm_scale.
                    _inv_qk = c_one_f / (fx.Float32(sm_scale) * q_descale * k_descale)
                    if const_expr(HAS_PER_HEAD_BIAS):
                        _bias_head_base = fx.Int64(head_idx) * fx.Int64(seq_len) * fx.Int64(seq_len_kv)
                        if const_expr(BIAS_BOTTOM_RIGHT):
                            _bias_row = fx.Int64(q_row_i32) + (fx.Int64(seq_len) - fx.Int64(sq_i32))
                            _col_off = fx.Int64(seq_len_kv) - fx.Int64(seq_len_kv_valid_i32)
                            _row_base = _bias_head_base + _bias_row * fx.Int64(seq_len_kv) + _col_off
                        else:
                            _row_base = _bias_head_base + fx.Int64(q_row_i32) * fx.Int64(seq_len_kv)
                    elif const_expr(VARLEN):
                        _row_base = (q_off_i64 + fx.Int64(q_row_i32)) * fx.Int64(seq_len_kv)
                    else:
                        _row_base = fx.Int64(q_row_i32) * fx.Int64(seq_len_kv)
                    s_raw = add_score_bias(
                        s_raw,
                        kv_start_i32,
                        klane_off_i32,
                        _row_base,
                        bias_buf,
                        scale=_inv_qk,
                        clamp=True,
                        seq_len_kv_valid_i32=seq_len_kv_valid_i32,
                        c_zero_f=c_zero_f,
                    )

            if const_expr(HAS_ALIBI):
                if q_in_bounds:
                    kv_start_i32 = fx.Int32(kv_block_start)
                    klane_off_i32 = fx.Int32(klane) * fx.Int32(8)
                    if const_expr(ALIBI_PER_HEAD):
                        slope_i = fx.Int32(head_idx)
                    else:
                        slope_i = fx.Int32(0)
                    slope_f = fx.Float32(fx.ptr_load(slopes_ptr + slope_i))
                    if const_expr(VARLEN):
                        alibi_sq = sq_i32
                    else:
                        alibi_sq = fx.Int32(seq_len)
                    _inv_qk = c_one_f / (fx.Float32(sm_scale) * q_descale * k_descale)
                    s_raw = add_alibi_scores(
                        s_raw,
                        kv_start_i32,
                        klane_off_i32,
                        q_row_i32,
                        alibi_sq,
                        seq_len_kv_valid_i32,
                        slope_f,
                        scale=_inv_qk,
                    )

            # Window and non-window corrections differ when the previous max is
            # -inf. Pass the window pair only on the window build.
            if const_expr(HAS_WINDOW):
                # Integer keep (row_live) decides the tile. A leading tile with
                # no kept key must not fmax/exp2 the initial -inf: under
                # no-nans that poisons l, and the epilogue then stores zeros.
                _window = (seen, row_live)
                _guard_prev_dead = False
            else:
                _window = None
                _guard_prev_dead = True
            p_vals, m_new_raw, l_new, o_accs, row_alive = online_softmax_tile(
                s_raw,
                m_running,
                l_running,
                o_accs,
                c_logit_scale,
                reduction_peer=reduction_peer,
                fmax=_fmax,
                fadd=_fadd,
                fmul=_fmul,
                fsub=_fsub,
                c_neg_inf=c_neg_inf,
                c_zero_f=c_zero_f,
                c_one_f=c_one_f,
                guard_prev_dead=_guard_prev_dead,
                window=_window,
            )

            coop_store_v_lds(_v_vecs_tile, 0)
            gpu.barrier()

            p_packs_all = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_packs_st = []
                for pks in range_constexpr(PV_K_STEPS):
                    acc_idx = st_idx * 2 + pks
                    p_base = acc_idx * 8
                    p_slice = [p_vals[p_base + j] for j in range(8)]
                    if const_expr(_IS_INT8):
                        p_packs_st.append(int8_pack_v8(p_slice))
                    else:
                        p_packs_st.append(fp8_pack_v8(p_slice))
                p_packs_all.append(p_packs_st)

            # O += V^T @ P, pipelined across V packs.
            v_base = v_buf_base(0)

            def _load_v_rowmajor(
                st_kv_base_val: fx.Int32 | fx.Int64 | int,
                pks_val: fx.Int32 | int,
                dc_val: fx.Int32 | int,
                v_base: fx.Int64 | int = v_base,
            ) -> fx.Vector:
                d_pos = fx.Int64(dc_val * D_CHUNK) + lane16
                v_elems = []
                for k_sub in range_constexpr(8):
                    kv_row = fx.Int64(st_kv_base_val + pks_val * PV_K_STEP) + klane * WMMA_LANE_K + fx.Int64(k_sub)
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                    v_elems.append(fx.ptr_load(lds_kv + fx.Int32(v_lds_idx)))
                return Vec.from_elements(v_elems, elem_dtype)

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

            m_running = row_alive.select(m_new_raw, m_running)
            l_running = row_alive.select(l_new, l_running)

            if const_expr(PREFETCH_V_ACROSS_ITERS):
                next_kv_start = fx.Int64(kv_block_start) + fx.Int64(BLOCK_N_OUT)
                _v_vecs_tile = coop_load_v_global(next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(PREFETCH_V_ACROSS_ITERS):
                for vi in range_constexpr(NUM_V_VECS):
                    _yield_args.append(_v_vecs_tile[vi])
            if const_expr(HAS_WINDOW):
                _yield_args.append(_fmax(seen, row_live))
            loop_results = yield _yield_args

        m_final = loop_results[0]
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]
        qk_scale = fx.Float32(sm_scale) * q_descale * k_descale
        c_log2e = fx.Float32(_LOG2E)
        if const_expr(SPLIT):
            # Partials are f32. v_descale (and the int8 1/127) fold in here so
            # the bf16 combine only merges online-softmax state.
            scale_o = v_descale
            if const_expr(_IS_INT8):
                scale_o = scale_o * fx.Float32(1.0 / 127.0)
            has_mass_pre = l_final > c_zero_f
            m_nat = has_mass_pre.select(_fmul(m_final, qk_scale), c_neg_inf)
            zero_vec = Vec.from_elements([c_zero_f], fx.Float32).broadcast_to(8)
            ws_m_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, WsM.address_space), WsM)
            ws_l_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, WsL.address_space), WsL)
            ws_o_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, WsO.address_space), WsO)
            if q_in_bounds:
                _ws_row = (
                    fx.Int64(split_idx) * fx.Int64(ws_rows)
                    + fx.Int64(batch_idx) * fx.Int64(NUM_HEADS) * fx.Int64(seq_len)
                    + fx.Int64(head_idx) * fx.Int64(seq_len)
                    + fx.Int64(q_row)
                )
                if klane == fx.Uint64(0):
                    _l_store = has_mass_pre.select(l_final, c_zero_f)
                    fx.ptr_store(m_nat, ws_m_ptr + fx.Int32(_ws_row))
                    fx.ptr_store(_l_store, ws_l_ptr + fx.Int32(_ws_row))
                for dc in range_constexpr(D_CHUNKS):
                    _ovec = has_mass_pre.select(o_finals[dc], zero_vec)
                    _dcol = fx.Int64(dc * D_CHUNK) + klane * fx.Uint64(8)
                    _obase = _ws_row * fx.Int64(HEAD_DIM) + _dcol
                    for _j in range_constexpr(8):
                        fx.ptr_store(
                            fx.Float32(Vec(_ovec)[_j]) * scale_o,
                            ws_o_ptr + fx.Int32(_obase) + fx.Int32(_j),
                        )
        if const_expr(HAS_SINK or RETURN_LSE):
            has_mass_pre = l_final > c_zero_f
            m_nat = has_mass_pre.select(_fmul(m_final, qk_scale), c_neg_inf)
            zero_vec = Vec.from_elements([c_zero_f], fx.Float32).broadcast_to(8)
            if const_expr(HAS_SINK):
                _sink_idx = fx.Int64(batch_idx) * fx.Int64(NUM_HEADS) + fx.Int64(head_idx)
                sink_logit = fx.Float32(fx.ptr_load(sink_elem_ptr + fx.Int32(_sink_idx)))
                o_finals, l_final, m_final_scaled = fold_attention_sink(
                    o_finals,
                    l_final,
                    m_nat,
                    sink_logit,
                    has_mass_pre,
                    fmax=_fmax,
                    fadd=_fadd,
                    fmul=_fmul,
                    fsub=_fsub,
                    c_zero_f=c_zero_f,
                    c_log2e=c_log2e,
                )
            else:
                m_final_scaled = m_nat
                o_finals = clear_o_if_empty(o_finals, has_mass_pre, zero_vec)
            if const_expr(RETURN_LSE):
                if q_in_bounds:
                    if klane == fx.Uint64(0):
                        lse_val = attention_lse(
                            m_final_scaled,
                            l_final,
                            has_mass_pre,
                            c_neg_inf,
                            has_sink=bool(HAS_SINK),
                        )
                        lse_idx = (
                            fx.Int64(batch_idx) * fx.Int64(NUM_HEADS) * fx.Int64(seq_len)
                            + fx.Int64(head_idx) * fx.Int64(seq_len)
                            + fx.Int64(q_row)
                        )
                        fx.ptr_store(lse_val, lse_elem_ptr + fx.Int32(lse_idx))
            has_mass = l_final > c_zero_f
            inv_l = attention_inv_l(l_final, has_mass, c_one_f, c_zero_f) * v_descale
            if const_expr(_IS_INT8):
                inv_l = inv_l * fx.Float32(1.0 / 127.0)
        else:
            # Default path (no sink/LSE): still guard empty/fully-masked rows.
            # Clear O before inv_l — 0*NaN stays NaN if a poisoned tile leaked.
            has_mass = l_final > c_zero_f
            zero_vec = Vec.from_elements([c_zero_f], fx.Float32).broadcast_to(8)
            o_finals = clear_o_if_empty(o_finals, has_mass, zero_vec)
            inv_l = attention_inv_l(l_final, has_mass, c_one_f, c_zero_f) * v_descale
            if const_expr(_IS_INT8):
                inv_l = inv_l * fx.Float32(1.0 / 127.0)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8)

        if const_expr(not SPLIT):
            if q_in_bounds:
                for dc in range_constexpr(D_CHUNKS):
                    o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                    o_trunc = Vec(o_norm_vec).to(out_dtype)
                    d_col = fx.Int64(dc * D_CHUNK) + klane * 8
                    o_global = global_idx(q_row, d_col)
                    if const_expr(LOGICAL_D == HEAD_DIM):
                        _store_global_half(o_elem_ptr, o_global, o_trunc)
                    else:
                        for ei in range_constexpr(8):
                            if d_col + fx.Int64(ei) < fx.Int64(LOGICAL_D):
                                _store_global_half(
                                    o_elem_ptr,
                                    o_global + fx.Int64(ei),
                                    Vec.from_elements([Vec(o_trunc)[ei]], out_dtype),
                                )

    _kind = "int8" if _IS_INT8 else ("e5m2" if _FP8_E5M2 else "e4m3")
    flash_attn_func_fp8_kernel.__name__ = (
        f"fa_quant_{_kind}_h{head_dim}_c{int(causal)}_kv{NUM_KV_HEADS}"
        f"_b{int(HAS_ATTN_BIAS)}{int(HAS_PER_HEAD_BIAS)}_l{int(RETURN_LSE)}_s{int(HAS_SINK)}"
        f"_sg{int(sign_a)}{int(sign_b)}_d{LOGICAL_D}"
        f"_m{int(VARLEN)}{int(PAGED)}{int(SPLIT)}_a{int(HAS_ALIBI)}{int(ALIBI_PER_HEAD)}"
        + (f"_w{SWA_LEFT}x{SWA_RIGHT}" if HAS_WINDOW else "")
    )

    @flyc.jit
    def launch_flash_attn_fp8_func(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        seq_len_kv_valid: fx.Int32,
        QDescale: fx.Pointer,
        KDescale: fx.Pointer,
        VDescale: fx.Pointer,
        Bias: fx.Pointer,
        Slopes: fx.Pointer,
        LSE: fx.Pointer,
        Sink: fx.Pointer,
        CuSeqlensQ: fx.Pointer,
        CuSeqlensKV: fx.Pointer,
        BlockTable: fx.Pointer,
        SeqlenK: fx.Pointer,
        block_table_stride: fx.Int32,
        num_splits: fx.Int32,
        WsM: fx.Pointer,
        WsL: fx.Pointer,
        WsO: fx.Pointer,
        ws_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ) -> None:
        ctx = CompilationContext.get_current()

        bs_idx = fx.Uint64(batch_size)
        sl_idx = fx.Uint64(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS
        if const_expr(SPLIT):
            grid_x = grid_x * fx.Uint64(num_splits)

        launcher = flash_attn_func_fp8_kernel(
            Q,
            K,
            V,
            O,  # noqa: E741
            seq_len,
            seq_len_kv,
            seq_len_kv_valid,
            QDescale,
            KDescale,
            VDescale,
            Bias,
            Slopes,
            LSE,
            Sink,
            CuSeqlensQ,
            CuSeqlensKV,
            BlockTable,
            SeqlenK,
            block_table_stride,
            num_splits,
            WsM,
            WsL,
            WsO,
            ws_rows,
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
            # Denormal/FTZ attrs follow daz; no-nans / unsafe-fp-math follow unsafe_fp_math.
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            if const_expr(unsafe_fp_math):
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
        for idx, a in enumerate(args):
            if hasattr(a, "data_ptr"):
                args[idx] = _ptr_fast(a)
        for name in (
            "Q",
            "K",
            "V",
            "O",
            "QDescale",
            "KDescale",
            "VDescale",
            "Bias",
            "Slopes",
            "LSE",
            "Sink",
        ):
            if name in kwargs and hasattr(kwargs[name], "data_ptr"):
                kwargs[name] = _ptr_fast(kwargs[name])
        return args, kwargs

    launch_flash_attn_fp8_func.compile_hints = dict(_fmha_compile_hints)
    _cached_cf = None  # closure-local CompiledFunction after first warm

    def _launch(*args, **kwargs) -> None:
        global _stack_opt_cf_hits, _stack_opt_run_compiled
        nonlocal _cached_cf
        args, kwargs = _wrap_qkvo_fast(args, kwargs)
        # Prefer torch current_stream — NEVER fx.Stream(None).
        stream = kwargs.pop("stream", None)
        if stream is None:
            stream = torch.cuda.current_stream()
        cf = _cached_cf or getattr(launch_flash_attn_fp8_func, "_cf", None)
        if cf is not None:  # noqa: E741
            _cached_cf = cf
            _stack_opt_cf_hits += 1
            cf(*args, stream)
            return
        _stack_opt_run_compiled += 1
        _run_compiled(launch_flash_attn_fp8_func, *args, stream)
        # Prefer CF attached to closed-over handle; else scan common stash.
        _cached_cf = getattr(launch_flash_attn_fp8_func, "_cf", None) or getattr(
            launch_flash_attn_fp8_func, "_last_compiled", None
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
        q_descale=None,
        k_descale=None,
        v_descale=None,
        Bias: torch.Tensor | None = None,
        Slopes: torch.Tensor | None = None,
        LSE: torch.Tensor | None = None,
        Sink: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> CompiledFunction | None:
        if stream is None:
            stream = torch.cuda.current_stream()
        if Bias is None:
            Bias = torch.zeros(1, device="cuda", dtype=torch.float32)
        if Slopes is None:
            Slopes = torch.zeros(1, device="cuda", dtype=torch.float32)
        if LSE is None:
            LSE = torch.zeros(1, device="cuda", dtype=torch.float32)
        if Sink is None:
            Sink = torch.zeros(1, device="cuda", dtype=torch.float32)
        if q_descale is None:
            q_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        elif isinstance(q_descale, (int, float)):
            q_descale = torch.tensor([float(q_descale)], device="cuda", dtype=torch.float32)
        if k_descale is None:
            k_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        elif isinstance(k_descale, (int, float)):
            k_descale = torch.tensor([float(k_descale)], device="cuda", dtype=torch.float32)
        if v_descale is None:
            v_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        elif isinstance(v_descale, (int, float)):
            v_descale = torch.tensor([float(v_descale)], device="cuda", dtype=torch.float32)
        if seq_len_kv is None:
            seq_len_kv = seq_len
        if seq_len_kv_valid is None:
            seq_len_kv_valid = seq_len_kv
        if int(seq_len) < 0:
            raise ValueError(f"fp8 gfx120x FA: seq_len must be >= 0, got {seq_len}")
        if int(seq_len_kv) < 0 or int(seq_len_kv_valid) < 0:
            raise ValueError(
                f"fp8 gfx120x FA: seq_len_kv/seq_len_kv_valid must be >= 0, " f"got {seq_len_kv}/{seq_len_kv_valid}"
            )
        return flyc.compile(
            launch_flash_attn_fp8_func,
            _ptr_arg(Q),
            _ptr_arg(K),
            _ptr_arg(V),
            _ptr_arg(O),
            batch_size,
            seq_len,
            seq_len_kv,
            seq_len_kv_valid,
            _ptr_arg(q_descale),
            _ptr_arg(k_descale),
            _ptr_arg(v_descale),
            _ptr_arg(Bias),
            _ptr_arg(Slopes),
            _ptr_arg(LSE),
            _ptr_arg(Sink),
            stream,
        )

    _launch.compile = _compile
    _launch.jit_function = launch_flash_attn_fp8_func
    return _launch


build_flash_attn_func_fp8_module = build_flash_attn_func_fp8_module_primary

# Back-compat alias (family name).
build_flash_attn_func_fp8_module_gfx120x = build_flash_attn_func_fp8_module_primary
