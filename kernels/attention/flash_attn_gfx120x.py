# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Flash Attention kernel for gfx120x (RDNA4; HW may report gfx1201).

Uses 16x16x16 wave32 WMMA, online softmax, pipelined V loads, and flattened
BSHD inputs. Requires ``head_dim >= 64``, ``head_dim % 32 == 0``, and
``head_dim <= 480`` (bf16 LDS fits at prefetch1 ≤64 KiB; multi-batch KV loader uses ceil coverage).

Supports dense non-causal cross-attention when ``seq_len`` (Q) and
``seq_len_kv`` (K/V) differ. Causal self and causal×cross use in-kernel bottom-right masking
(``j <= i + Skv_valid - Sq``); equal seqlens reduce to classic causal.

Optional dense extras (compile-time traits): fp32 LSE epilogue ``[B,H,Sq]``,
per-head additive bias ``[H,Sq,Sk]``, and attention-sink fold into online softmax.

Optional packed-varlen (``varlen=True``) and paged-KV (``paged=True``) may be
combined (packed Q, paged K/V). ``kv_cache_layout`` is ``linear``,
``linear3d`` (page_size 1, same bytes as linear), or ``vectorized``.
GQA uses ``kv_head = q_head // (H / Hkv)``. Split-K is a runtime ``num_splits``
with fp32 partials plus a device combine (not a host softmax loop). Dense
(``not varlen and not paged``) keeps the BSHD ABI; varlen/paged add trailing
CuSeqlens/BlockTable/SeqlenK pointers that dense launches ignore.
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
from flydsl.expr import const_expr, gpu, range_constexpr  # noqa: E402  # noqa: E741
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

KERNEL_NAME = "flash_attn_func_gfx120x_kernel"


def build_flash_attn_func_module_primary(
    num_heads: int,
    head_dim: int,
    causal: bool = True,
    dtype_str: str = "bf16",
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
    varlen: bool = False,
    paged: bool = False,
    page_size: int = 16,
    num_kv_heads: int | None = None,
    kv_cache_layout: str = "linear",
    logical_head_dim: int | None = None,
    kv_oob: bool = False,
    sliding_window: tuple[int, int] | None = None,
    bias_bottom_right: bool = False,
    has_alibi: bool = False,
    alibi_per_head: bool = False,
) -> Callable[..., None]:
    """Build the gfx120x Flash Attention kernel."""

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
    assert dtype_str in ("f16", "bf16")

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    # Aligned D keeps the historical stride (HEAD_DIM). A shorter logical D is a
    # separate specialization so the aligned kernel does not trace the tail.
    if logical_head_dim is None or int(logical_head_dim) == int(head_dim):
        LOGICAL_D = int(head_dim)
        ALIGNED_D = True
    else:
        LOGICAL_D = int(logical_head_dim)
        if LOGICAL_D < 1 or LOGICAL_D > int(head_dim):
            raise ValueError(f"gfx120x FA logical_head_dim={LOGICAL_D} is outside 1..{head_dim}")
        ALIGNED_D = False
    KV_OOB = bool(kv_oob)
    CAUSAL = causal
    HAS_ATTN_BIAS = bool(has_attn_bias) or bool(has_per_head_bias)
    HAS_PER_HEAD_BIAS = bool(has_per_head_bias)
    BIAS_BOTTOM_RIGHT = bool(bias_bottom_right)
    HAS_ALIBI = bool(has_alibi)
    ALIBI_PER_HEAD = bool(alibi_per_head)
    RETURN_LSE = bool(return_lse)
    HAS_SINK = bool(has_sink)
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
    VARLEN = bool(varlen)
    PAGED = bool(paged)
    PAGE_SIZE = int(page_size) if paged else 16
    if PAGED and PAGE_SIZE <= 0:
        raise ValueError(f"gfx120x FA: page_size must be > 0, got {PAGE_SIZE}")
    # Python-level bools for single const_expr(...) tests (avoid and/or of const_expr).
    CLAMP_KV_ROWS = VARLEN or PAGED
    NUM_KV_HEADS = int(num_heads if num_kv_heads is None else num_kv_heads)
    if NUM_KV_HEADS <= 0 or int(num_heads) % NUM_KV_HEADS != 0:
        raise ValueError(f"gfx120x FA: num_heads={int(num_heads)} must be divisible by num_kv_heads={NUM_KV_HEADS}")
    KV_GROUP = int(num_heads) // NUM_KV_HEADS
    if ALIGNED_D:
        Q_STRIDE = int(num_heads) * int(head_dim)
        KV_STRIDE = NUM_KV_HEADS * int(head_dim)
    else:
        Q_STRIDE = int(num_heads) * LOGICAL_D
        KV_STRIDE = NUM_KV_HEADS * LOGICAL_D
    STRIDE_TOKEN = Q_STRIDE
    _kv_layout = kv_cache_layout or "linear"
    if _kv_layout not in ("linear", "linear3d", "vectorized"):
        raise ValueError(f"gfx120x FA: unknown kv_cache_layout {_kv_layout!r}")
    KV_VECTORIZED = bool(paged) and _kv_layout == "vectorized"
    if bool(paged) and _kv_layout == "linear3d" and int(page_size) != 1:
        raise ValueError("gfx120x FA: linear3d paged KV requires page_size=1")
    # bf16/fp16 kVectorSize = 16/sizeof(elem) = 8. Matches flash_attn_interface.
    KV_VEC = 8
    if KV_VECTORIZED and (int(head_dim) % KV_VEC != 0 or PAGE_SIZE % KV_VEC != 0):
        raise ValueError(f"gfx120x FA: vectorized KV needs head_dim and page_size divisible by kVS={KV_VEC}")
    # LSE layout [B, H, Sq]. Compile-time head count is NUM_HEADS; seq_len is a runtime arg.

    # Padding reduces LDS bank conflicts.
    K_STRIDE = HEAD_DIM + 4
    V_STRIDE = HEAD_DIM + 4

    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    # Multi-batch KV loads: every BLOCK_N row must be covered. Floor division
    # silently dropped the tail when ROWS_PER_BATCH_LOAD did not divide BLOCK_N
    # (common for D>128 and some D=96 + small BLOCK_M combos) — wrong outputs.
    # Use ceil batches and guard surplus LDS rows whenever coverage is partial
    # or threads overshoot BLOCK_N.
    if ROWS_PER_BATCH_LOAD <= 0:
        raise ValueError(
            f"ROWS_PER_BATCH_LOAD must be > 0 (BLOCK_SIZE={BLOCK_SIZE}, "
            f"THREADS_PER_ROW_LOAD={THREADS_PER_ROW_LOAD}, head_dim={head_dim})"
        )
    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = (BLOCK_N + ROWS_PER_BATCH_LOAD - 1) // ROWS_PER_BATCH_LOAD
        # Always guard when multi-batch: last ceil batch may be partial.
        KV_NEEDS_GUARD = True

    # Buffer loads cap at dwordx4, so V rows are fetched in 8-element pieces.
    V_SUBVECS = VEC_WIDTH // 8
    NUM_V_VECS = NUM_BATCHES_KV * V_SUBVECS

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    _NUMERIC_MAP = {
        "f32": fx.Float32,
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
    }
    elem_numeric_cls = _NUMERIC_MAP[dtype_str]

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])  # noqa: E741
    def flash_attn_func_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        seq_len_kv_valid: fx.Int32,
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
        o_elem_ptr = _as_elem_ptr(O)
        ws_m_ptr = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, WsM.address_space),
            WsM,
        )
        ws_l_ptr = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, WsL.address_space),
            WsL,
        )
        ws_o_ptr = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, WsO.address_space),
            WsO,
        )

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
            if const_expr(VARLEN):
                # Packed [total_q, Sk_pad] — soft-large descriptor bound.
                bias_bytes = fx.Int64(0x7FFFFFFF)
            else:
                # Shared [Sq, Skv] or per-head [H, Sq, Skv] (Skv may be tile-padded).
                _bias_tiles = fx.Int64(seq_len) * fx.Int64(seq_len_kv)
                if const_expr(HAS_PER_HEAD_BIAS):
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
            # Full tensor bound: callers pass [B, H, Sq]; we index per batch below.
            # Use a large-enough per-launch bound via batch*H*Sq from grid math —
            # descriptor covers the whole LSE allocation (host passes exact size).
            # Bound is set per-row store via absolute pointer offset (no batch slice).

        if const_expr(HAS_SINK):
            sink_elem_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, Sink.address_space),
                Sink,
            )
            # Host expands sink to contiguous [B, H] fp32.

        wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype, fx.Float32))

        def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            a_frag = fx.make_rmem_tensor(8, elem_dtype)
            b_frag = fx.make_rmem_tensor(8, elem_dtype)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(a_v8))
            b_frag.store(Vec(b_v8))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)  # FlyDSL 0.3.4.1: a/b Sequence
            return Vec(c_frag.load())

        # seq_len = Q/O length (max for VARLEN grid; real for dense q_in_bounds).
        # seq_len_kv = K/V buffer length (may be tile-padded).
        # seq_len_kv_valid overridden after batch_idx for VARLEN/PAGED (sk).
        seq_len_q_v = fx.Uint64(seq_len)
        seq_len_kv_v = fx.Uint64(seq_len_kv)

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

        block_id_all = fx.Uint64(gpu.block_idx.x)
        ns_u = fx.Uint64(num_splits)
        split_idx = block_id_all % ns_u
        block_id = block_id_all // ns_u
        tid = fx.Uint64(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        wave_q_offset = wave_id * ROWS_PER_WAVE

        head_idx = block_id % NUM_HEADS
        kv_head_idx = head_idx // fx.Uint64(KV_GROUP)
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_q_v + BLOCK_M - 1) // BLOCK_M
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

        # --- per-batch lengths / packed offsets (VARLEN) or SeqlenK (PAGED) ---
        # Dense: seq_len_kv_valid from launch arg; q uses seq_len (max = real).
        # VARLEN: ignore launch seq_len_kv_valid; load cu_seqlens → q_off/sq, k_off/sk.
        # PAGED: load SeqlenK[batch] as sk; Q stays dense BSHD.
        if const_expr(VARLEN):
            _cu_q_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, CuSeqlensQ.address_space),
                CuSeqlensQ,
            )
            q_off_i32 = fx.Int32(fx.ptr_load(_cu_q_ptr + fx.Int32(batch_idx)))
            q_end_i32 = fx.Int32(fx.ptr_load(_cu_q_ptr + fx.Int32(batch_idx) + fx.Int32(1)))
            q_off_i64 = fx.Int64(q_off_i32)
            sq_i32 = q_end_i32 - q_off_i32
        else:
            q_off_i64 = fx.Int64(0)
            sq_i32 = fx.Int32(seq_len)

        if const_expr(PAGED):
            _sk_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, SeqlenK.address_space),
                SeqlenK,
            )
            sk_i32 = fx.Int32(fx.ptr_load(_sk_ptr + fx.Int32(batch_idx)))
            k_off_i64 = fx.Int64(0)
            seq_len_kv_valid_i32 = sk_i32
            _bt_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, BlockTable.address_space),
                BlockTable,
            )
            _bt_stride_i32 = fx.Int32(block_table_stride)
        elif const_expr(VARLEN):
            _cu_kv_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, CuSeqlensKV.address_space),
                CuSeqlensKV,
            )
            k_off_i32 = fx.Int32(fx.ptr_load(_cu_kv_ptr + fx.Int32(batch_idx)))
            k_end_i32 = fx.Int32(fx.ptr_load(_cu_kv_ptr + fx.Int32(batch_idx) + fx.Int32(1)))
            k_off_i64 = fx.Int64(k_off_i32)
            sk_i32 = k_end_i32 - k_off_i32
            seq_len_kv_valid_i32 = sk_i32
        else:
            sk_i32 = fx.Int32(seq_len_kv_valid)
            k_off_i64 = fx.Int64(0)
            seq_len_kv_valid_i32 = fx.Int32(seq_len_kv_valid)
        if const_expr(VARLEN):
            # Causal bottom-right uses local (sk - sq), not max_seqlen.
            causal_len_q_i32 = sq_i32
        else:
            causal_len_q_i32 = fx.Int32(seq_len)

        def global_idx_q(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            if const_expr(not ALIGNED_D):
                if const_expr(VARLEN):
                    token = q_off_i64 + fx.Int64(token_idx)
                else:
                    token = batch_idx * seq_len_q_v + token_idx
                return token * STRIDE_TOKEN + head_idx * LOGICAL_D + col
            if const_expr(VARLEN):
                token = q_off_i64 + fx.Int64(token_idx)
            else:
                token = batch_idx * seq_len_q_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def global_idx_kv(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            # PAGED wins over VARLEN so packed Q can attend a paged cache.
            if const_expr(PAGED):
                _tok = fx.Int32(token_idx)
                _page_idx = _tok // fx.Int32(PAGE_SIZE)
                _page_off = _tok % fx.Int32(PAGE_SIZE)
                _bt_idx = fx.Int32(batch_idx) * _bt_stride_i32 + _page_idx
                _pid = fx.Int32(fx.ptr_load(_bt_ptr + _bt_idx))
                # linear and linear3d (page_size=1) share this byte order.
                if const_expr(not ALIGNED_D):
                    return (
                        fx.Int64(_pid) * fx.Int64(PAGE_SIZE * KV_STRIDE)
                        + fx.Int64(_page_off) * fx.Int64(KV_STRIDE)
                        + kv_head_idx * LOGICAL_D
                        + col
                    )
                return (
                    fx.Int64(_pid) * fx.Int64(PAGE_SIZE * KV_STRIDE)
                    + fx.Int64(_page_off) * fx.Int64(KV_STRIDE)
                    + kv_head_idx * HEAD_DIM
                    + col
                )
            if const_expr(not ALIGNED_D):
                if const_expr(VARLEN):
                    token = k_off_i64 + fx.Int64(token_idx)
                    return token * KV_STRIDE + kv_head_idx * LOGICAL_D + col
                token = batch_idx * seq_len_kv_v + token_idx
                return token * KV_STRIDE + kv_head_idx * LOGICAL_D + col
            if const_expr(VARLEN):
                token = k_off_i64 + fx.Int64(token_idx)
                return token * KV_STRIDE + kv_head_idx * HEAD_DIM + col
            token = batch_idx * seq_len_kv_v + token_idx
            return token * KV_STRIDE + kv_head_idx * HEAD_DIM + col

        # Back-compat alias: historical call sites meant Q indexing.
        def global_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            return global_idx_q(token_idx, col)

        # Hardware OOB handling keeps the tail prefetch branch-free. A batch
        # slice must fit the descriptor's 32-bit num_records.
        ELEM_BYTES = (elem_numeric_cls.width + 7) // 8
        if const_expr(PAGED):
            # Whole-cache descriptor: absolute global_idx_kv addresses.
            # Soft-large bound; physical OOB is avoided by sk masking + host BT.
            v_buf_ptr = _bounds_checked_buf_ptr(
                v_elem_ptr,
                fx.Int64(0x7FFFFFFF),
            )

            def v_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                return global_idx_kv(token_idx, col)

        else:
            v_batch_elems = seq_len_kv_v * fx.Uint64(KV_STRIDE)
            if const_expr(VARLEN):
                # Packed KV: index from buffer base via global_idx_kv (k_off+tok).
                v_buf_ptr = _bounds_checked_buf_ptr(
                    v_elem_ptr,
                    fx.Int64(0x7FFFFFFF),
                )

                def v_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                    return global_idx_kv(token_idx, col)

            else:
                v_buf_ptr = _bounds_checked_buf_ptr(
                    fx.add_offset(v_elem_ptr, fx.Int64(batch_idx * v_batch_elems)),
                    fx.Int64(v_batch_elems) * fx.Int64(ELEM_BYTES),
                )

                def v_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                    if const_expr(not ALIGNED_D):
                        return token_idx * KV_STRIDE + kv_head_idx * LOGICAL_D + col
                    return token_idx * KV_STRIDE + kv_head_idx * HEAD_DIM + col

        def _load_global_half_vec(elem_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int, width: int) -> fx.Vector:
            view = fx.make_view(
                fx.add_offset(elem_ptr, fx.Int64(base_idx)),
                fx.make_layout(width, 1),
            )
            return Vec(view.load())

        def _store_global_half(elem_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int, val: fx.Vector) -> None:
            view = fx.make_view(
                fx.add_offset(elem_ptr, fx.Int64(base_idx)),
                fx.make_layout(val.numel, 1),
            )
            view.store(Vec(val))

        def load_global_f16xN(base_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int) -> fx.Vector:
            return _load_global_half_vec(base_ptr, base_idx, VEC_WIDTH)

        def load_global_v8f16(base_ptr: fx.Pointer, base_idx: fx.Int32 | fx.Int64 | int) -> fx.Vector:
            return _load_global_half_vec(base_ptr, base_idx, 8)

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
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype)

        def k_buf_base(buf_id: fx.Int32 | int) -> fx.Int64:
            if const_expr(isinstance(buf_id, int)):
                return fx.Int64(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Int64(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id: fx.Int32 | int) -> fx.Int64:
            return fx.Int64(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        def _clamp_kv_row(row_idx: fx.Int32 | fx.Int64 | int) -> fx.Int64:
            # VARLEN/PAGED: clamp past-sk loads to last valid token (softmax -inf masks
            # those cols). Avoids packed next-batch / invalid page reads.
            if const_expr(CLAMP_KV_ROWS):
                _sk_i64 = fx.Int64(seq_len_kv_valid_i32)
                _last = (_sk_i64 > fx.Int64(0)).select(_sk_i64 - fx.Int64(1), fx.Int64(0))
                return (fx.Int64(row_idx) < _sk_i64).select(fx.Int64(row_idx), _last)
            return fx.Int64(row_idx)

        def _paged_pid_off(token_idx: fx.Int32 | fx.Int64 | int) -> tuple[fx.Int32, fx.Int32]:
            _tok = fx.Int32(token_idx)
            _page_idx = _tok // fx.Int32(PAGE_SIZE)
            _page_off = _tok % fx.Int32(PAGE_SIZE)
            _bt_idx = fx.Int32(batch_idx) * _bt_stride_i32 + _page_idx
            _pid = fx.Int32(fx.ptr_load(_bt_ptr + _bt_idx))
            return _pid, _page_off

        def _load_tail(ptr: fx.Pointer, base_idx: fx.Int64, col: fx.Int32 | fx.Int64 | int, width: int) -> fx.Vector:
            """Scalar bf16/fp16. A wide load of a non-power-of-two stride is align-1.

            ``one`` is initialized before the dynamic if so scf.if can yield it.
            """
            elems = []
            for i in range_constexpr(width):
                one = elem_dtype(0)
                if fx.Int64(col) + fx.Int64(i) < fx.Int64(LOGICAL_D):
                    one = Vec(
                        fx.make_view(
                            fx.add_offset(ptr, base_idx + fx.Int64(i)),
                            fx.make_layout(1, 1),
                        ).load()
                    )[0]
                elems.append(one)
            return Vec.from_elements(elems, elem_dtype)

        if const_expr(KV_OOB):
            k_batch_elems = seq_len_kv_v * fx.Uint64(KV_STRIDE)
            k_buf_ptr = _bounds_checked_buf_ptr(
                fx.add_offset(k_elem_ptr, fx.Int64(batch_idx * k_batch_elems)),
                fx.Int64(k_batch_elems) * fx.Int64(ELEM_BYTES),
            )

            def k_local_idx(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Int64:
                return token_idx * KV_STRIDE + kv_head_idx * LOGICAL_D + col

        def _load_k_tile_vec(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Vector:
            if const_expr(not KV_VECTORIZED):
                if const_expr(not ALIGNED_D):
                    # Plain element pointer: a row past this batch's K is not in
                    # the allocation. Zero it. The aligned path uses the buffer record.
                    row_ok = fx.Uint64(token_idx) < fx.Uint64(seq_len_kv_valid_i32)
                    vec = Vec.from_elements(
                        [elem_dtype(0) for _ in range_constexpr(VEC_WIDTH)],
                        elem_dtype,
                    )
                    if row_ok:
                        vec = _load_tail(k_elem_ptr, global_idx_kv(token_idx, col), col, VEC_WIDTH)
                    return vec
                if const_expr(KV_OOB):
                    # v16bf16 through this descriptor is align-1 and does not
                    # legalize. Two v8 loads match the V buffer that does.
                    lo = _load_global_half_vec(k_buf_ptr, k_local_idx(token_idx, col), 8)
                    hi = _load_global_half_vec(k_buf_ptr, k_local_idx(token_idx, fx.Int64(col) + fx.Int64(8)), 8)
                    return Vec.from_elements(
                        [Vec(lo)[i] for i in range(8)] + [Vec(hi)[i] for i in range(8)], elem_dtype
                    )
                return load_global_f16xN(k_elem_ptr, global_idx_kv(token_idx, col))
            _pid, _poff = _paged_pid_off(token_idx)
            elems = []
            for _p in range_constexpr(VEC_WIDTH // KV_VEC):
                d0 = fx.Int64(col) + fx.Int64(_p * KV_VEC)
                group = d0 // fx.Int64(KV_VEC)
                base = (
                    (
                        (fx.Int64(_pid) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * fx.Int64(HEAD_DIM // KV_VEC)
                        + group
                    )
                    * fx.Int64(PAGE_SIZE)
                    + fx.Int64(_poff)
                ) * fx.Int64(KV_VEC)
                part = _load_global_half_vec(k_elem_ptr, base, KV_VEC)
                for _t in range_constexpr(KV_VEC):
                    elems.append(Vec(part)[_t])
            return Vec.from_elements(elems, elem_dtype)

        def _load_v8_vec(token_idx: fx.Int32 | fx.Int64 | int, col: fx.Int32 | fx.Int64 | int) -> fx.Vector:
            if const_expr(not KV_VECTORIZED):
                if const_expr(not ALIGNED_D):
                    row_ok = fx.Uint64(token_idx) < fx.Uint64(seq_len_kv_valid_i32)
                    vec = Vec.from_elements([elem_dtype(0) for _ in range_constexpr(8)], elem_dtype)
                    if row_ok:
                        vec = _load_tail(v_elem_ptr, global_idx_kv(token_idx, col), col, 8)
                    return vec
                return _load_global_half_vec(v_buf_ptr, v_idx(token_idx, col), 8)
            _pid, _poff = _paged_pid_off(token_idx)
            kg = fx.Int32(_poff) // fx.Int32(KV_VEC)
            kr = fx.Int32(_poff) % fx.Int32(KV_VEC)
            # v1bf16 buffer loads abort AMDGPU type legalization. The kVS group
            # is dword-aligned, so load the even bf16 pair and pick kr's lane.
            kr_pair = kr & fx.Int32(-2)
            elems = []
            for _i in range_constexpr(8):
                idx = (
                    (
                        (fx.Int64(_pid) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx))
                        * fx.Int64(PAGE_SIZE // KV_VEC)
                        + fx.Int64(kg)
                    )
                    * fx.Int64(HEAD_DIM)
                    + fx.Int64(col)
                    + fx.Int64(_i)
                ) * fx.Int64(KV_VEC) + fx.Int64(kr_pair)
                pair = _load_global_half_vec(v_buf_ptr, idx, 2)
                elems.append(Vec(pair)[kr & fx.Int32(1)])
            return Vec.from_elements(elems, elem_dtype)

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
                        vec = _load_k_tile_vec(_clamp_kv_row(row_idx), load_col_base)
                        lds_store(lds_idx, Vec(vec))
                else:
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    vec = _load_k_tile_vec(_clamp_kv_row(row_idx), load_col_base)
                    lds_store(lds_idx, Vec(vec))

        def _v_store_row_major(
            v_base: fx.Int64, lds_row: fx.Int32 | fx.Int64 | int, col_extra: int, vec: fx.Vector
        ) -> None:
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base + col_extra
            fx.ptr_store(Vec(vec), lds_kv + fx.Int32(lds_idx))

        def coop_load_v_global(tile_start: fx.Int32 | fx.Int64 | int) -> list[fx.Vector]:
            tile_start = fx.Int64(tile_start)
            # Clamp surplus LDS rows; OOB global loads hit the bounds-checked
            # V descriptor (zero-fill) and stores are skipped in coop_store_v_lds.
            # VARLEN/PAGED: clamp row via _clamp_kv_row (no runtime list-carry if).
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                lds_row = load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_cap = fx.Int64(BLOCK_N - 1)
                    lds_row = fx.Int64((lds_row < row_cap).select(lds_row, row_cap))
                row_idx = _clamp_kv_row(tile_start + lds_row)
                for sv in range_constexpr(V_SUBVECS):
                    vecs.append(_load_v8_vec(row_idx, load_col_base + fx.Int64(sv * 8)))
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
                            _v_store_row_major(v_base, lds_row, sv * 8, vecs[batch * V_SUBVECS + sv])
                else:
                    for sv in range_constexpr(V_SUBVECS):
                        _v_store_row_major(v_base, lds_row, sv * 8, vecs[batch * V_SUBVECS + sv])

        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)

        # Dense: token < max(=real) seq_len. VARLEN: local token < sq (idle tiles OK).
        if const_expr(VARLEN):
            q_in_bounds = q_row_i32 < sq_i32
        else:
            q_in_bounds = q_row < seq_len_q_v
        q_row_safe = fx.Int64(q_in_bounds.select(q_row, fx.Int64(0)))

        # First KV column fully masked for this wave (bottom-right causal).
        # VARLEN uses local (sk - sq); dense/paged use (valid - seq_len[/sq]).
        wave_kv_limit_i32 = fx.Int32(q_start + wave_q_offset + fx.Int64(ROWS_PER_WAVE)) + (
            seq_len_kv_valid_i32 - causal_len_q_i32
        )
        c_zero_v8f16 = Vec.filled(8, 0.0, elem_dtype)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Int64(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = global_idx_q(q_row_safe, q_col)
            if const_expr(not ALIGNED_D):
                raw = _load_tail(q_elem_ptr, g_idx, q_col, 8)
            else:
                raw = load_global_v8f16(q_elem_ptr, g_idx)
            q_b_packs.append(q_in_bounds.select(raw, c_zero_v8f16))

        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
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

        # Non-causal carries prefetched V across iterations; causal avoids the
        # extra VGPR lifetime and loads V in the current iteration.
        PREFETCH_V_ACROSS_ITERS = not CAUSAL

        ns_i32 = fx.Int32(num_splits)
        sp_i32 = fx.Int32(split_idx)
        sk_for_split = seq_len_kv_valid_i32
        chunk_i32 = (sk_for_split + ns_i32 - fx.Int32(1)) // ns_i32
        kv_lo_i32 = sp_i32 * chunk_i32
        kv_hi_i32 = kv_lo_i32 + chunk_i32
        kv_hi_i32 = (kv_hi_i32 < sk_for_split).select(kv_hi_i32, sk_for_split)
        is_split = ns_i32 > fx.Int32(1)
        range_lo = is_split.select(fx.Int64(kv_lo_i32), fx.Int64(0))
        _hi_cap = fx.Int32(kv_upper)
        _hi_i = (_hi_cap < kv_hi_i32).select(_hi_cap, kv_hi_i32)
        range_hi = is_split.select(fx.Int64(_hi_i), kv_upper)

        if const_expr(PREFETCH_V_ACROSS_ITERS):
            _v_vecs_init = coop_load_v_global(range_lo)

        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v8f32)
        if const_expr(PREFETCH_V_ACROSS_ITERS):
            for vi in range_constexpr(NUM_V_VECS):
                init_args.append(_v_vecs_init[vi])
        # 1 after the first tile that kept a key. Same f32 state the fp8 kernel carries.
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
                        k_pack_a = Vec(lds_load(k_lds_a, 8))

                        k_row_b = lane16 + fx.Int64(st_base_row + 16)
                        k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                        k_pack_b = Vec(lds_load(k_lds_b, 8))

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

            # Non-causal: mask K/V columns past the real (unpadded) seq so pad
            # tokens do not contribute exp(0)=1 into the softmax denominator.
            # Causal already excludes pad via future-mask when host pads equally.
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

            # Additive general attn-mask / bias: Bias[q_row, kv_col] or
            # Bias[head, q_row, kv_col] when HAS_PER_HEAD_BIAS (fp32).
            if const_expr(HAS_ATTN_BIAS):
                if q_in_bounds:
                    kv_start_i32 = fx.Int32(kv_block_start)
                    klane_off_i32 = fx.Int32(klane) * fx.Int32(8)
                    # Shared varlen bias is packed [total_q, max_kv]. Per-head bias is
                    # [H, max_sq, max_kv] and uses the local row, not q_off.
                    if const_expr(HAS_PER_HEAD_BIAS):
                        _bias_head_base = fx.Int64(head_idx) * fx.Int64(seq_len) * fx.Int64(seq_len_kv)
                        if const_expr(BIAS_BOTTOM_RIGHT):
                            # Shared [H, max_q, max_k] ALiBi. Local (i, j) is the
                            # bottom-right of that tensor, not its top-left.
                            _bias_row = fx.Int64(q_row_i32) + (fx.Int64(seq_len) - fx.Int64(sq_i32))
                            _col_off = fx.Int64(seq_len_kv) - fx.Int64(sk_i32)
                            _row_base = _bias_head_base + _bias_row * fx.Int64(seq_len_kv) + _col_off
                        else:
                            _row_base = _bias_head_base + fx.Int64(q_row_i32) * fx.Int64(seq_len_kv)
                    elif const_expr(VARLEN):
                        _row_base = (q_off_i64 + fx.Int64(q_row_i32)) * fx.Int64(seq_len_kv)
                    else:
                        _row_base = fx.Int64(q_row_i32) * fx.Int64(seq_len_kv)
                    _clamp_bias_col = bool(KV_OOB or VARLEN or PAGED)
                    # Scores are still unscaled. Divide the mask by sm_scale so the
                    # softmax sees softmax(qk * sm_scale + bias), matching SDPA.
                    s_raw = add_score_bias(
                        s_raw,
                        kv_start_i32,
                        klane_off_i32,
                        _row_base,
                        bias_buf,
                        scale=fx.Float32(1.0 / float(sm_scale)),
                        clamp=_clamp_bias_col,
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
                    s_raw = add_alibi_scores(
                        s_raw,
                        kv_start_i32,
                        klane_off_i32,
                        q_row_i32,
                        sq_i32,
                        seq_len_kv_valid_i32,
                        slope_f,
                        scale=fx.Float32(1.0 / float(sm_scale)),
                    )

            if is_split:
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_off_i32 = fx.Int32(klane) * fx.Int32(8)
                s_raw = kill_score_columns(
                    s_raw,
                    kv_start_i32,
                    klane_off_i32,
                    (("<", kv_lo_i32), (">=", kv_hi_i32)),
                    c_neg_inf,
                )

            # Fully masked so far: m_running and the tile max are both -inf.
            # guard_prev_dead stays false. The window build passes (seen, row_live)
            # so a leading empty tile does not fmax the initial -inf. Scores stay f32.
            if const_expr(HAS_WINDOW):
                _window = (seen, row_live)
            else:
                _window = None
            p_vals, m_new_raw, l_new, o_accs, row_alive = online_softmax_tile(
                s_raw,
                m_running,
                l_running,
                o_accs,
                c_sm_scale_log2e,
                reduction_peer=reduction_peer,
                fmax=_fmax,
                fadd=_fadd,
                fmul=_fmul,
                fsub=_fsub,
                c_neg_inf=c_neg_inf,
                c_zero_f=c_zero_f,
                c_one_f=c_one_f,
                guard_prev_dead=False,
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

                    if const_expr(dtype_str == "bf16"):
                        p_packs_st.append(bf16_trunc_pack_v8(p_slice))
                    else:
                        elem_list = []
                        for j in range_constexpr(8):
                            elem_list.append(fx.Float32(p_slice[j]).to(elem_dtype))
                        p_packs_st.append(Vec.from_elements(elem_list, elem_dtype))
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
            loop_results = yield _yield_args  # noqa: E741

        m_final = loop_results[0]
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        c_sm_scale = fx.Float32(sm_scale)
        c_log2e = fx.Float32(_LOG2E)

        # Empty row: all keys masked → l==0 (m may be -inf or NaN from (-inf)-(-inf)).
        # Sanitize before sink / LSE so we never propagate NaN.
        has_mass_pre = l_final > c_zero_f
        m_nat_raw = _fmul(m_final, c_sm_scale)
        m_nat = has_mass_pre.select(m_nat_raw, c_neg_inf)
        zero_vec = Vec.from_elements([c_zero_f], fx.Float32).broadcast_to(8)
        do_partial = fx.Int32(num_splits) > fx.Int32(1)
        if do_partial:
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
                            Vec(_ovec)[_j],
                            ws_o_ptr + fx.Int32(_obase) + fx.Int32(_j),
                        )
        else:
            if const_expr(HAS_SINK):
                # Contiguous [B, H] — one fp32 per (batch, head).
                # Empty KV: corr=0 → O stays 0; l becomes sink_w (=1 when m_nat=-inf).
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

            # LSE = m_scaled + ln(l). Empty+no-sink → -inf; empty+sink → sink (m=sink,l=1).
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

            # Guard empty denominator (no keys and no sink): store zeros, skip 1/0.
            has_mass = l_final > c_zero_f
            inv_l = attention_inv_l(l_final, has_mass, c_one_f, c_zero_f)
            inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8)

            if q_in_bounds:
                for dc in range_constexpr(D_CHUNKS):
                    o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                    o_trunc = Vec(o_norm_vec).to(elem_dtype)
                    d_col = fx.Int64(dc * D_CHUNK) + klane * 8
                    o_global = global_idx_q(q_row, d_col)
                    if const_expr(ALIGNED_D):
                        _store_global_half(o_elem_ptr, o_global, o_trunc)
                    else:
                        for ei in range_constexpr(8):
                            if d_col + fx.Int64(ei) < fx.Int64(LOGICAL_D):
                                _store_global_half(
                                    o_elem_ptr,
                                    o_global + fx.Int64(ei),
                                    Vec.from_elements([Vec(o_trunc)[ei]], elem_dtype),
                                )

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        seq_len_kv_valid: fx.Int32,
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
        grid_x = bs_idx * num_q_tiles * NUM_HEADS * fx.Uint64(num_splits)

        launcher = flash_attn_func_kernel(
            Q,
            K,
            V,
            O,
            seq_len,
            seq_len_kv,
            seq_len_kv_valid,
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
        for idx in range(min(4, len(args))):
            a = args[idx]
            if hasattr(a, "data_ptr"):
                args[idx] = _ptr_fast(a)
        # Bias/Slopes/LSE/Sink/CuQ/CuKV/BT/SeqlenK are positional 8..15.
        # block_table_stride is Int32 @ 16. Workspaces are 18..20.
        for _pi in (8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20):
            if len(args) > _pi and hasattr(args[_pi], "data_ptr"):
                args[_pi] = _ptr_fast(args[_pi])
        for name in (
            "Q",
            "K",
            "V",
            "O",
            "Bias",
            "Slopes",
            "LSE",
            "Sink",
            "CuSeqlensQ",
            "CuSeqlensKV",
            "BlockTable",
            "SeqlenK",
            "WsM",
            "WsL",
            "WsO",
        ):  # noqa: E741
            if name in kwargs and hasattr(kwargs[name], "data_ptr"):
                kwargs[name] = _ptr_fast(kwargs[name])
        return args, kwargs

    launch_flash_attn_func.compile_hints = dict(_fmha_compile_hints)
    _cached_cf = None  # closure-local CompiledFunction after first warm

    def _launch(*args, **kwargs) -> None:
        global _stack_opt_cf_hits, _stack_opt_run_compiled
        nonlocal _cached_cf
        args, kwargs = _wrap_qkvo_fast(args, kwargs)
        # Prefer torch current_stream — NEVER fx.Stream(None).
        stream = kwargs.pop("stream", None)
        if stream is None:
            stream = torch.cuda.current_stream()
        cf = _cached_cf or getattr(launch_flash_attn_func, "_cf", None)
        if cf is not None:
            _cached_cf = cf
            _stack_opt_cf_hits += 1
            cf(*args, stream)
            return
        _stack_opt_run_compiled += 1
        _run_compiled(launch_flash_attn_func, *args, stream)
        # Prefer CF attached to closed-over handle; else scan common stash.
        _cached_cf = getattr(launch_flash_attn_func, "_cf", None) or getattr(
            launch_flash_attn_func, "_last_compiled", None
        )

    def _compile(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,  # noqa: E741
        batch_size: int,
        seq_len: int,
        seq_len_kv: int | None = None,
        seq_len_kv_valid: int | None = None,
        Bias: torch.Tensor | None = None,
        Slopes: torch.Tensor | None = None,
        LSE: torch.Tensor | None = None,
        Sink: torch.Tensor | None = None,
        CuSeqlensQ: torch.Tensor | None = None,
        CuSeqlensKV: torch.Tensor | None = None,
        BlockTable: torch.Tensor | None = None,
        SeqlenK: torch.Tensor | None = None,
        block_table_stride: int = 0,
        num_splits: int = 1,
        WsM: torch.Tensor | None = None,
        WsL: torch.Tensor | None = None,
        WsO: torch.Tensor | None = None,
        ws_rows: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> CompiledFunction | None:  # noqa: E741
        if seq_len_kv is None:
            seq_len_kv = seq_len
        if seq_len_kv_valid is None:
            seq_len_kv_valid = seq_len_kv
        if Bias is None:
            # Dummy 1-elem fp32; never read when HAS_ATTN_BIAS is False.
            Bias = torch.zeros(1, device="cuda", dtype=torch.float32)
        if Slopes is None:
            Slopes = torch.zeros(1, device="cuda", dtype=torch.float32)
        if LSE is None:
            LSE = torch.zeros(1, device="cuda", dtype=torch.float32)
        if Sink is None:
            Sink = torch.zeros(1, device="cuda", dtype=torch.float32)
        if CuSeqlensQ is None:
            CuSeqlensQ = torch.zeros(1, device="cuda", dtype=torch.int32)
        if CuSeqlensKV is None:
            CuSeqlensKV = torch.zeros(1, device="cuda", dtype=torch.int32)
        if BlockTable is None:
            BlockTable = torch.zeros(1, device="cuda", dtype=torch.int32)
        if SeqlenK is None:
            SeqlenK = torch.zeros(1, device="cuda", dtype=torch.int32)
        if WsM is None:
            WsM = torch.zeros(1, device="cuda", dtype=torch.float32)
        if WsL is None:
            WsL = torch.zeros(1, device="cuda", dtype=torch.float32)
        if WsO is None:
            WsO = torch.zeros(1, device="cuda", dtype=torch.float32)
        if stream is None:
            stream = torch.cuda.current_stream()
        return flyc.compile(
            launch_flash_attn_func,
            _ptr_arg(Q),
            _ptr_arg(K),
            _ptr_arg(V),
            _ptr_arg(O),
            batch_size,
            seq_len,
            seq_len_kv,
            seq_len_kv_valid,
            _ptr_arg(Bias),
            _ptr_arg(Slopes),
            _ptr_arg(LSE),
            _ptr_arg(Sink),
            _ptr_arg(CuSeqlensQ),
            _ptr_arg(CuSeqlensKV),
            _ptr_arg(BlockTable),
            _ptr_arg(SeqlenK),
            int(block_table_stride),
            int(num_splits),
            _ptr_arg(WsM),
            _ptr_arg(WsL),
            _ptr_arg(WsO),
            int(ws_rows),
            stream,
        )

    _launch.compile = _compile
    # AOT / export_to_c: expose the underlying @flyc.jit for flyc.compile(...).export_to_c
    _launch.jit_function = launch_flash_attn_func
    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary

# Back-compat alias (family name).
build_flash_attn_func_module_gfx120x = build_flash_attn_func_module_primary
