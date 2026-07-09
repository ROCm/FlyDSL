# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""flash_attn_func kernel builder for FlyDSL.

- True MFMA32 remap: `mfma_f32_32x32x16bf16` / `mfma_f32_32x32x16f16` for both GEMM stages.
- Tile shape: BLOCK_M=128 or 256 (auto-selected), BLOCK_N=64.
- BLOCK_M=128: 4 waves (256 threads), BLOCK_M=256: 8 waves (512 threads).
- Per-wave Q rows: 32.
- GEMM1 uses `K @ Q^T` so S/P live in MFMA32 register layout.
- Online softmax over KV dimension is done in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS roundtrip.
- K and V use separate LDS regions with DMA-to-LDS prefetch and XOR swizzle.
- For H>=32, both M=128 and M=256 variants are built and dispatched at runtime.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) or (512,) depending on BLOCK_M.

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len % 128 == 0.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.common.kernels_common import dtype_to_elem_type

_LOG2E = host_math.log2(host_math.e)  # 1.4426950408889634
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _llvm_value(value):
    """Unwrap FlyDSL scalar/vector wrappers for LLVM pointer load ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    """Extract the aligned LLVM pointer from a FlyDSL tensor/memref."""
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def build_flash_attn_func_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    num_kv_heads=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cross_seqlen=False,
    varlen=False,
    paged=False,
    kv_cache_layout="linear",
):
    """Build the flash_attn_func launcher using the post-refactor FlyDSL API.

    For GQA/MQA pass ``num_kv_heads < num_heads``. ``num_heads`` is the Q head
    count, ``num_kv_heads`` is the KV head count, and we require
    ``num_heads % num_kv_heads == 0``. Default ``num_kv_heads = num_heads`` (MHA).
    Q/O still have ``num_heads`` heads; K/V have ``num_kv_heads`` heads, with
    every ``num_heads // num_kv_heads`` consecutive Q heads sharing one KV head.
    """
    gpu_arch = get_hip_arch()

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    if dtype_str == "fp8":
        raise ValueError("generic flash_attn_func supports f16/bf16 only; fp8 is routed by flash_attn_interface")

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64

    def _extract_seq_len(args, kwargs):
        """Return the launch-time seq_len as int, or None if not statically known."""
        S = args[5] if len(args) > 5 else kwargs.get("seq_len", None)
        try:
            return int(S)
        except (TypeError, ValueError):
            return None

    def _guard_seqlen(_dispatched):
        """Enforce the only correctness floor (seq_len >= 1). A symbolic/non-int
        seq_len is let through; dense routing is a perf policy, not a bound."""

        def _guarded(*args, **kwargs):
            S_int = _extract_seq_len(args, kwargs)
            if S_int is not None and S_int < 1:
                raise ValueError(f"flash_attn_func: seq_len must be >= 1, got {S_int}.")
            return _dispatched(*args, **kwargs)

        if hasattr(_dispatched, "compile"):
            _guarded.compile = _dispatched.compile
        return _guarded

    if block_m is not None:
        BLOCK_M = block_m
    else:
        BLOCK_M = 128

    if flat_work_group_size is None:
        if BLOCK_M <= 128:
            flat_work_group_size = 256
        else:
            flat_work_group_size = 512
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    assert ROWS_PER_WAVE == 32, f"BLOCK_M/NUM_WAVES must be 32, got {ROWS_PER_WAVE}"
    if path_tag.upper() in ("N32", "N128"):
        PATH_TAG = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        PATH_TAG = "N128"
    else:
        PATH_TAG = "N32"
    BLOCK_N_OUT = 128 if PATH_TAG == "N128" else BLOCK_N
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    ENABLE_PREFETCH_3BUF = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    # buffer_load_dwordx4_lds (16B DMA-to-LDS) requires gfx950+; gfx94x only has dword (4B).
    _has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    ENABLE_DMA = _has_lds_load_b128 and (
        PATH_TAG == "N128" or (os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1")
    )
    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    REDUCE_MODE = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if REDUCE_MODE not in ("xor", "ds_bpermute"):
        REDUCE_MODE = "xor"
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else (2 if ENABLE_DMA else 1)
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else 1
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)

    # gfx950+ has ds_read_tr16_b64 (HW transpose LDS read); gfx942 needs V^T stored in LDS.
    USE_HW_TR = gpu_arch.startswith("gfx950")

    # MFMA32 K-dimension: 16 on gfx950+ (CDNA4) for both GEMMs.
    USE_K16 = gpu_arch.startswith("gfx950")

    # 128-bit permlane-fused O-store needs gfx950 (permlane32_swap + cvt_pk_bf16_f32,
    # both CDNA4-only); gfx942 falls back to a per-lane dwordx2 store via .to(elem_dtype).
    USE_PERMLANE_OSTORE = gpu_arch.startswith("gfx950")
    K_STEP_QK = 16 if USE_K16 else 8
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16 if USE_K16 else 8
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2 steps per sub-tile (K=16) or 4 (K=8)

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert flat_work_group_size in (
        64,
        128,
        256,
        512,
    ), f"flat_work_group_size must be 64, 128, 256, or 512, got {flat_work_group_size}"
    assert dtype_str in ("f16", "bf16"), "flash_attn_func only supports f16 and bf16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT % BLOCK_N == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    HEAD_DIM = head_dim
    CAUSAL = causal
    # Varlen uses packed QKV via launch-time cu_seqlens.
    VARLEN = bool(varlen)
    CROSS_SEQLEN = bool(cross_seqlen)
    # Paged K/V read physical page BlockTable[b*stride + tile]; Q/O stay packed.
    PAGED = bool(paged)
    if PAGED and kv_cache_layout not in ("linear", "vectorized"):
        raise NotImplementedError(
            f"generic paged kernel supports linear/vectorized kv_cache_layout, got {kv_cache_layout!r}"
        )
    # Aiter vectorized 5D cache; kVS = 16B/elem = 8 for bf16/f16.
    KV_VECTORIZED = PAGED and (kv_cache_layout == "vectorized")
    KV_VEC_SIZE = 8
    PAGE_SIZE = BLOCK_N  # one KV tile maps to one physical page
    # vectorized K/V share page/head element strides: page = Hkv*D*PS, head = D*PS.
    PAGE_STRIDE_VEC = NUM_HEADS_KV * HEAD_DIM * PAGE_SIZE
    HEAD_STRIDE_VEC = HEAD_DIM * PAGE_SIZE
    if PAGED and ENABLE_DMA:
        raise NotImplementedError("generic paged kernel requires the non-DMA path (build with path_tag='N32')")
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    # Bank-conflict-free LDS strides.
    # K uses XOR swizzle (col ^ ((row & mask) << 4)) at 16-element granularity
    # instead of padding. This enables ds_read_b128 (stride is 256B-aligned).
    K_STRIDE = HEAD_DIM
    # Mask by the number of 16-element D blocks; D=64 must avoid swizzling past col 63.
    K_SWZ_ROWMASK = HEAD_DIM // 16 - 1
    if USE_HW_TR:
        V_STRIDE = HEAD_DIM if ENABLE_DMA else HEAD_DIM + 4
    else:
        VT_STRIDE = BLOCK_N + 2
        V_STRIDE = VT_STRIDE

    # Vectorized V uses no-major LDS so each P*V pack is one aligned v8 read.
    # elem(n,d) = (d//8)*544 + (n//8)*64 + (d%8)*8 + n%8.
    VEC_V_LINE = 544
    VEC_V_D128 = 64
    VEC_V_NGROUPS = BLOCK_N // 8  # n-groups of 8 (8 for BLOCK_N=64)
    if KV_VECTORIZED:
        # Split one v8 copy per (n_group, d) across the workgroup.
        TOTAL_V8 = VEC_V_NGROUPS * HEAD_DIM
        assert TOTAL_V8 % BLOCK_SIZE == 0
        NV8_PER_THREAD = TOTAL_V8 // BLOCK_SIZE
        assert not ENABLE_PREFETCH_3BUF, "KV_VECTORIZED no-major V unsupported with 3-buffer prefetch"
        # Direct GM->LDS DMA fills no-major V rows without VGPR round-trips.
        V_NOMAJOR_DMA = os.getenv("FLYDSL_FLASH_ATTN_FUNC_VEC_V_DMA", "1") == "1"

    # Vectorized cooperative load constants.
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # K/V circular buffers; defaults to 1/1, optional 3/3 with CK-like LDS sequence.
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    if KV_VECTORIZED:
        LDS_V_TILE_SIZE = (HEAD_DIM // 8) * VEC_V_LINE
    elif USE_HW_TR:
        LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    else:
        LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_func_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_generic_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        _load_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        _v1i32_type = Vec.make_type(1, fx.Int32)

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        # The unsafe_fp_math/fast_fp_math builder params control LLVM-level attributes only.
        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type if USE_K16 else v4f16_type
        MFMA_LANE_K = 8 if USE_K16 else 4

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v16f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def mfma_acc(a, b, c):
            if const_expr(dtype_str == "bf16"):
                if const_expr(USE_K16):
                    return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)
                a = Vec(a).bitcast(fx.Int16)
                b = Vec(b).bitcast(fx.Int16)
                return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
            if const_expr(USE_K16):
                return _mfma(rocdl.mfma_f32_32x32x16_f16, a, b, c)
            return _mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

        seq_len_v = fx.Index(seq_len)
        seq_len_kv_v = fx.Index(seq_len_kv)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        # ---- Thread / block indices ----
        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        # ---- Wave decomposition ----
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ---- ds_read_b64_tr_b16 lane decomposition ----
        # Hardware transposes 4x4 groups within 16 lanes; these indices select row/col groups.
        tr_k_group = (lane % 16) // 4  # 0..3: K-row offset within 4-row group
        tr_col_sub = lane % 4  # 0..3: 4-column sub-group
        tr_col_half = (lane % 32) // 16  # 0 or 1: first/second 16-column half

        # ---- ds_read_b64_tr_b16 helper ----

        def ds_read_tr_v4f16(lds_elem_idx):
            """Read v4f16 from LDS with hardware transpose.

            Within each block of 16 lanes, the hardware performs a 4×4
            transpose across 4 groups of 4 lanes.  After the transpose,
            result[lane, elem_e] = Input[source_lane, lane%4] where
            source_lane = e*4 + (lane%16)//4.  This naturally produces
            the MFMA A-operand layout when per-lane addresses point to
            the correct K-row and D-column sub-group.
            """
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = fx.Int64(byte_offset)
            ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- Wave offsets ----
        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id ----
        # Each block computes one Q head (per batch, per Q-tile).
        q_head_idx = block_id % NUM_HEADS_Q
        batch_q_tile_id = block_id // NUM_HEADS_Q
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M
        # GQA/MQA: every GQA_GROUP_SIZE consecutive Q heads share one KV head.
        # Use Python ternary (ast.IfExp) so FlyDSL's `if`-rewriter doesn't
        # turn this into a dynamic dispatch and lose `kv_head_idx`.
        kv_head_idx = q_head_idx if GQA_GROUP_SIZE == 1 else q_head_idx // GQA_GROUP_SIZE

        # Dense uses independent Q/KV lengths; varlen reads cu_seqlens[batch:batch+2].
        # num_q_tiles still uses max_seqlen_q so flat-grid decode matches launch.
        if const_expr(VARLEN):
            _cuq_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqQ), fx.make_layout(1, 1))
            _cuk_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqKv), fx.make_layout(1, 1))

            def _cu_load(div, idx):
                v = fly.copy_atom_call_ssa([_v1i32_type], _load_atom_32, fx.slice(div, (None, fx.Int32(idx))))
                return fx.Index(Vec(v, (1,), fx.Int32)[0])

            q_tok_base = _cu_load(_cuq_div, batch_idx)
            kv_tok_base = _cu_load(_cuk_div, batch_idx)
            seqlen_q_b = _cu_load(_cuq_div, batch_idx + fx.Index(1)) - q_tok_base
            seqlen_kv_b = _cu_load(_cuk_div, batch_idx + fx.Index(1)) - kv_tok_base
        else:
            q_tok_base = batch_idx * seq_len_v
            kv_tok_base = batch_idx * seq_len_kv_v
            seqlen_q_b = seq_len_v
            seqlen_kv_b = seq_len_kv_v

        # Dense/varlen fold batch into raw K/V pointers; paged adds page_id per tile.
        if const_expr(not PAGED):
            _kv_ptr_batch_off = kv_tok_base * fx.Index(STRIDE_TOKEN_KV)
            k_ptr = buffer_ops.get_element_ptr(k_ptr, _kv_ptr_batch_off, elem_type=elem_type)
            v_ptr = buffer_ops.get_element_ptr(v_ptr, _kv_ptr_batch_off, elem_type=elem_type)

        if const_expr(PAGED):
            _bt_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(BlockTable), fx.make_layout(1, 1))
            _bt_stride_v = fx.Index(block_table_stride)

            def _paged_page_id(tile_start):
                # page_id = BlockTable[batch_idx*block_table_stride + tile_start//PAGE_SIZE]
                _tile_idx = tile_start // fx.Index(PAGE_SIZE)
                _bt_off = batch_idx * _bt_stride_v + _tile_idx
                v = fly.copy_atom_call_ssa([_v1i32_type], _load_atom_32, fx.slice(_bt_div, (None, fx.Int32(_bt_off))))
                return fx.Index(Vec(v, (1,), fx.Int32)[0])

            if const_expr(KV_VECTORIZED):

                def _vk_load(page_id, kv_row, col):
                    # K[kv_row, col:col+VEC_WIDTH] from vectorized [.,Hkv,D/8,PS,8]:
                    # two contiguous vec8 blocks (dg=col//8 and dg+1), each
                    # cache[.,dg,kv_row,0:8] == K[kv_row, dg*8:dg*8+8].
                    _b = (
                        page_id * fx.Index(PAGE_STRIDE_VEC)
                        + kv_head_idx * fx.Index(HEAD_STRIDE_VEC)
                        + kv_row * fx.Index(KV_VEC_SIZE)
                    )
                    _dg = col // fx.Index(KV_VEC_SIZE)
                    _lo = _load_global_half_vec(k_ptr, _b + _dg * fx.Index(PAGE_SIZE * KV_VEC_SIZE), KV_VEC_SIZE)
                    _hi = _load_global_half_vec(
                        k_ptr, _b + (_dg + fx.Index(1)) * fx.Index(PAGE_SIZE * KV_VEC_SIZE), KV_VEC_SIZE
                    )
                    return Vec(_lo).shuffle(Vec(_hi), list(range(VEC_WIDTH))).ir_value()

                def _vv_load(page_id, kv_row, col):
                    # V[kv_row, col:col+VEC_WIDTH] from vectorized [.,Hkv,PS/8,D,8]:
                    # stride-8 gather, idx = kg*(D*8) + d*8 + (kv_row%8).
                    _kg = kv_row // fx.Index(KV_VEC_SIZE)
                    _kr = kv_row % fx.Index(KV_VEC_SIZE)
                    _b = (
                        page_id * fx.Index(PAGE_STRIDE_VEC)
                        + kv_head_idx * fx.Index(HEAD_STRIDE_VEC)
                        + _kg * fx.Index(HEAD_DIM * KV_VEC_SIZE)
                        + _kr
                    )
                    _elems = []
                    for _i in range_constexpr(VEC_WIDTH):
                        _v1 = _load_global_half_vec(v_ptr, _b + (col + fx.Index(_i)) * fx.Index(KV_VEC_SIZE), 1)
                        _elems.append(Vec(_v1)[0])
                    return Vec.from_elements(_elems, elem_dtype).ir_value()

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat indices ----
        # Token indices are 0-based because the descriptor or raw pointer folds batch base.
        def global_idx_q(token_idx, col):
            return token_idx * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col

        def global_idx_kv(token_idx, col):
            return token_idx * STRIDE_TOKEN_KV + kv_head_idx * HEAD_DIM + col

        def _kv_row_clamp(row_idx):
            # Raw non-DMA loads need an in-bounds row; score masks discard duplicated tails.
            last = seqlen_kv_b - fx.Index(1)
            return fx.Index(ArithValue(row_idx < seqlen_kv_b).select(row_idx, last))

        def _load_global_half_vec(ptr, base_idx, vec_elems: int):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            _pointer_store(val, gep)

        def load_global_f16x4(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, 4)

        def load_global_mfma_pack(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, MFMA_LANE_K)

        def load_global_f16xN(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, VEC_WIDTH)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _pack_bf16_pair(lo, hi, shift, mask):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack_v4(f32_vals):
            """Pack f32 values into bf16 by keeping the upper 16 bits."""
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            packed = [
                _pack_bf16_pair(f32_vals[0], f32_vals[1], _c16, _cmask),
                _pack_bf16_pair(f32_vals[2], f32_vals[3], _c16, _cmask),
            ]
            return Vec.from_elements(packed, fx.Int32).bitcast(elem_dtype).ir_value()

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits)."""
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            pairs = []
            for j in range_constexpr(4):
                pairs.append(_pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask))
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        # ---- K XOR swizzle: col ^ ((row & 7) << 4) at 16-element granularity ----
        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(K_SWZ_ROWMASK)) << fx.Index(4)
            return col_idx ^ mask

        # Sigma makes QK^T emit P in 8-consecutive-kv order for no-major V.
        def _sigma_kv(n):
            return (
                (n & fx.Index(3))
                | ((n & fx.Index(8)) >> fx.Index(1))
                | ((n & fx.Index(4)) << fx.Index(1))
                | (n & fx.Index(-16))
            )

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            if const_expr(PAGED):
                _pid_k = _paged_page_id(tile_start)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(PAGED):
                    row_idx = _pid_k * fx.Index(PAGE_SIZE) + load_row_in_batch + row_offset
                else:
                    row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        Vec(vec).store(lds_kv, [lds_idx])
                else:
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    if const_expr(KV_VECTORIZED):
                        vec = _vk_load(_pid_k, _sigma_kv(lds_row), load_col_base)
                    else:
                        vec = load_global_f16xN(k_ptr, global_idx_kv(row_idx, load_col_base))
                    Vec(vec).store(lds_kv, [lds_idx])

        # ---- Cooperative V load ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(VEC_WIDTH):
                elem = Vec(vec)[_e]
                vt_d = load_col_base + _e
                vt_idx = v_base + vt_d * VT_STRIDE + lds_row
                v1 = Vec.from_elements([elem], elem_dtype)
                v1.store(lds_kv, [vt_idx])

        _v_store_to_lds = _v_store_row_major if USE_HW_TR else _v_store_transposed

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            if const_expr(PAGED):
                _pid_v = _paged_page_id(tile_start)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(PAGED):
                    row_idx = _pid_v * fx.Index(PAGE_SIZE) + load_row_in_batch + row_offset
                else:
                    row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                else:
                    lds_row = load_row_in_batch + row_offset
                    if const_expr(KV_VECTORIZED):
                        vec = _vv_load(_pid_v, _sigma_kv(lds_row), load_col_base)
                    else:
                        vec = load_global_f16xN(v_ptr, global_idx_kv(row_idx, load_col_base))
                    _v_store_to_lds(v_base, lds_row, vec)

        def coop_load_v_global(tile_start):
            """Issue global V loads; vectorized mode returns no-major v8 rows."""
            if const_expr(KV_VECTORIZED):
                _pid_vg = _paged_page_id(tile_start)
                _base_v = _pid_vg * fx.Index(PAGE_STRIDE_VEC) + kv_head_idx * fx.Index(HEAD_STRIDE_VEC)
                vecs = []
                for j in range_constexpr(NV8_PER_THREAD):
                    _flat = tid + fx.Index(j * BLOCK_SIZE)
                    _d = _flat // fx.Index(VEC_V_NGROUPS)
                    _ng = _flat % fx.Index(VEC_V_NGROUPS)
                    _src = _base_v + _ng * fx.Index(HEAD_DIM * KV_VEC_SIZE) + _d * fx.Index(KV_VEC_SIZE)
                    vecs.append(_load_global_half_vec(v_ptr, _src, KV_VEC_SIZE))
                return vecs
            vecs = []
            if const_expr(PAGED):
                _pid_vg = _paged_page_id(tile_start)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(PAGED):
                    row_idx = _pid_vg * fx.Index(PAGE_SIZE) + load_row_in_batch + row_offset
                else:
                    row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                vecs.append(load_global_f16xN(v_ptr, global_idx_kv(row_idx, load_col_base)))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            """Write V vectors to LDS; vectorized mode uses no-major rows."""
            v_base = v_buf_base(buf_id)
            if const_expr(KV_VECTORIZED):
                for j in range_constexpr(NV8_PER_THREAD):
                    _flat = tid + fx.Index(j * BLOCK_SIZE)
                    _d = _flat // fx.Index(VEC_V_NGROUPS)
                    _ng = _flat % fx.Index(VEC_V_NGROUPS)
                    _dst = (
                        v_base
                        + (_d // fx.Index(8)) * fx.Index(VEC_V_LINE)
                        + _ng * fx.Index(VEC_V_D128)
                        + (_d % fx.Index(8)) * fx.Index(8)
                    )
                    Vec(vecs[j]).store(lds_kv, [_dst])
                return
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vecs[batch])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vecs[batch])

        # ---- KV_VECTORIZED V: no-major GM->LDS DMA ----
        if const_expr(KV_VECTORIZED and V_NOMAJOR_DMA):
            _v_dma_base_i64 = fx.Int64(buffer_ops.extract_base_index(V, address_space=1))
            _v_dma_page_bytes = fx.Int64(PAGE_STRIDE_VEC * 2)
            _v_dma_lds_base = buffer_ops.extract_base_index(lds_kv, address_space=3)
            _v_dma_sz = fx.Int32(16)
            _v_dma_z = fx.Int32(0)
            _v_dma_aux = fx.Int32(1)
            _v_dma_no = lane // fx.Index(8)
            _v_dma_dloc = lane % fx.Index(8)

            def coop_dma_v_nomajor(tile_start, buf_id=0):
                # Each lane DMAs one contiguous v8 into the no-major V line.
                _pid = _paged_page_id(tile_start)
                _paddr = _raw(_v_dma_base_i64 + fx.Int64(_pid) * _v_dma_page_bytes)
                _rsrc = buffer_ops.create_buffer_resource_from_addr(_paddr, num_records_bytes=_raw(_v_dma_page_bytes))
                if const_expr(isinstance(buf_id, int)):
                    _vb = _v_dma_lds_base + fx.Index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2)
                else:
                    _vb = _v_dma_lds_base + fx.Index(LDS_V_BASE * 2) + buf_id * fx.Index(LDS_V_TILE_SIZE * 2)
                for d in range_constexpr(NV8_PER_THREAD):
                    _row = wave_id * fx.Index(NV8_PER_THREAD) + fx.Index(d)
                    _lds_b = _vb + _row * fx.Index(VEC_V_LINE * 2)
                    _lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(_lds_b))
                    _lds_ptr = buffer_ops.create_llvm_ptr(_lds_lane0, address_space=3)
                    _dcol = _row * fx.Index(8) + _v_dma_dloc
                    _voff_e = (
                        kv_head_idx * fx.Index(HEAD_STRIDE_VEC)
                        + _v_dma_no * fx.Index(HEAD_DIM * 8)
                        + _dcol * fx.Index(8)
                    )
                    _voff = fx.Int32(_voff_e * fx.Index(2))
                    rocdl.raw_ptr_buffer_load_lds(_rsrc, _lds_ptr, _v_dma_sz, _voff, _v_dma_z, _v_dma_z, _v_dma_aux)

        # Per-batch descriptors keep global indices 0-based and bounded to one batch.
        # This keeps 32-bit offsets small while preserving arbitrary-seqlen OOB behavior.
        _kv_nrec_bytes = _raw(seqlen_kv_b * fx.Index(STRIDE_TOKEN_KV * 2))
        _q_nrec_bytes = _raw(seqlen_q_b * fx.Index(STRIDE_TOKEN_Q * 2))
        _q_batch_byte_off = _raw(q_tok_base * fx.Index(STRIDE_TOKEN_Q * 2))
        _kv_batch_byte_off = _raw(kv_tok_base * fx.Index(STRIDE_TOKEN_KV * 2))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )

        # ---- DMA loading for K (buffer_load_dwordx4 ... lds) ----
        if const_expr(ENABLE_DMA):
            k_rsrc = buffer_ops.create_buffer_resource(
                K, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            DMA_BYTES = 16  # buffer_load_dwordx4 = 16 bytes per lane
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
            NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def coop_dma_k(tile_start, buf_id=0):
                """Load K tile via DMA with XOR-swizzled global fetch."""
                if const_expr(isinstance(buf_id, int)):
                    k_lds_byte_base = lds_kv_base_idx + fx.Index(buf_id * LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * fx.Index(LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (
                        k_lds_byte_base + wave_id * fx.Index(WARP_SIZE * DMA_BYTES) + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_i64 = fx.Int64(lds_addr)
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                    row_in_tile = tid // LANES_PER_K_ROW + fx.Index(d * ROWS_PER_DMA_BATCH)
                    swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(0x7)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile  # 0-based: batch base folded into k/v_rsrc
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_KV * 2) + kv_head_idx * fx.Index(HEAD_DIM * 2) + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        # ---- V XOR swizzle: col ^ ((row & 3) << 4) at 16-element granularity ----
        def _v_swizzle(row_idx, col_idx):
            mask = (row_idx & fx.Index(0x3)) << fx.Index(4)
            return col_idx ^ mask

        # ---- DMA loading for V (buffer_load_dwordx4 ... lds) ----
        if const_expr(ENABLE_DMA):
            v_rsrc = buffer_ops.create_buffer_resource(
                V, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
            NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH_V = DMA_BATCH_BYTES // (HEAD_DIM * 2)

            def coop_dma_v(tile_start, buf_id=0):
                """Load V tile via DMA with XOR-swizzled global fetch."""
                v_lds_byte_base = lds_kv_base_idx + fx.Index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2)
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (
                        v_lds_byte_base + wave_id * fx.Index(WARP_SIZE * DMA_BYTES) + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_i64 = fx.Int64(lds_addr)
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                    row_in_tile = tid // LANES_PER_V_ROW + fx.Index(d * ROWS_PER_DMA_BATCH_V)
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(0x3)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile  # 0-based: batch base folded into k/v_rsrc
                    global_byte = (
                        global_row * fx.Index(STRIDE_TOKEN_KV * 2) + kv_head_idx * fx.Index(HEAD_DIM * 2) + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        # B operand: j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K. Q is
        # num_records-bounded (q_rsrc) so OOB rows read 0 -- no q_in_bounds select.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row, q_col)
            q_b_packs.append(buffer_ops.buffer_load(q_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_neg_floor = fx.Float32(-3.0e38)
        c_zero_f = fx.Float32(0.0)
        c_zero_i32 = fx.Int32(0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)
        c4_i32 = fx.Int32(4)
        lane_i32 = fx.Int32(lane)
        lane_xor_32_i32 = lane_i32 ^ shuf_32_i32
        lane_xor_32_byte = lane_xor_32_i32 * c4_i32

        def reduction_peer(v_f32):
            if const_expr(REDUCE_MODE == "ds_bpermute"):
                v_i32 = fx.Int32(ArithValue(v_f32).bitcast(fx.Int32.ir_type))
                peer_i32 = rocdl.ds_bpermute(fx.Int32.ir_type, lane_xor_32_byte, v_i32)
                return fx.Float32(ArithValue(peer_i32).bitcast(compute_type))
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop upper bound ----
        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            delta_i32 = fx.Int32(seqlen_kv_b) - fx.Int32(seqlen_q_b)
            causal_end_raw_i32 = fx.Int32(_q_end) + delta_i32
            causal_end_i32 = fx.Int32(
                ArithValue(causal_end_raw_i32 > fx.Int32(0)).select(causal_end_raw_i32, fx.Int32(0))
            )
            causal_end = fx.Index(causal_end_i32)
            kv_upper = fx.Index(ArithValue(causal_end < seqlen_kv_b).select(causal_end, seqlen_kv_b))
        else:
            kv_upper = seqlen_kv_b

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = ENABLE_DMA and not ENABLE_PREFETCH_3BUF
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if const_expr(_use_dma_dbuf):
            init_args.append(fx.Index(0))
            coop_dma_k(fx.Index(0), buf_id=0)

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(0, kv_upper, BLOCK_N_OUT, init=init_args):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _cur_buf_id = inner_iter_args[2 + D_CHUNKS] if _use_dma_dbuf else None
            preload_k_count = NUM_PREFETCH_K if NUM_PREFETCH_K < N_SUBTILES else N_SUBTILES

            if const_expr(ENABLE_PREFETCH_3BUF):
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = CK_LDS_SEQ[pre_k % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    pre_k_start = kv_block_start + pre_k * BLOCK_N
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        coop_load_k(pre_k_start, pre_k_slot)
                if const_expr(ENABLE_DMA):
                    rocdl.s_waitcnt(0)
                else:
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            for kv_sub in range_constexpr(N_SUBTILES):
                kv_start = kv_block_start + kv_sub * BLOCK_N

                if const_expr(ENABLE_PREFETCH_3BUF):
                    k_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                elif const_expr(_use_dma_dbuf):
                    if const_expr(kv_sub % 2 == 0):
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = fx.Index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = fx.Index(1) - _k_buf_id
                    if const_expr(kv_sub + 1 < N_SUBTILES):
                        coop_dma_k(
                            kv_block_start + (kv_sub + 1) * BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + fx.Index(BLOCK_N_OUT)
                        _has_next = _next_kv < kv_upper
                        if _has_next:
                            coop_dma_k(_next_kv, _next_k_buf_id)
                    rocdl.sched_barrier(0)
                    k_base = k_buf_base(_k_buf_id)
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if const_expr(not _use_dma_dbuf):
                    k_base = k_buf_base(k_slot)

                if const_expr(KV_VECTORIZED and V_NOMAJOR_DMA):
                    coop_dma_v_nomajor(kv_start, 0)
                elif const_expr(not USE_HW_TR or (not ENABLE_DMA and not ENABLE_PREFETCH_3BUF)):
                    _v_vecs_prefetch = coop_load_v_global(kv_start)

                # ==== GEMM1: bulk-read all K packs, then pipeline MFMAs ====
                k_hi_offset = K_SUB_N * K_STRIDE
                # XOR swizzle: col ^ ((row & 0x7) << 4) avoids LDS bank conflicts
                k_swz_mask = (lane_mod_32 & fx.Index(K_SWZ_ROWMASK)) << fx.Index(4)

                def _k_idx_lo(ks):
                    col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

                def _k_idx_hi(ks):
                    col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + k_hi_offset + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

                _QK_PREFETCH_DEPTH = 2
                k_packs_lo = [None] * K_STEPS_QK
                k_packs_hi = [None] * K_STEPS_QK
                for p in range_constexpr(_QK_PREFETCH_DEPTH):
                    k_packs_lo[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                    k_packs_hi[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_hi(p)])

                if const_expr(ENABLE_DMA and not ENABLE_PREFETCH_3BUF):
                    coop_dma_v(kv_start, 0)
                    rocdl.sched_barrier(0)

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                for ks in range_constexpr(K_STEPS_QK):
                    s_acc_lo = mfma_acc(k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                    s_acc_hi = mfma_acc(k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                    if const_expr(ks + _QK_PREFETCH_DEPTH < K_STEPS_QK):
                        k_packs_lo[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)]
                        )
                        k_packs_hi[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)]
                        )

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(Vec(s_acc_lo)[r])
                    s_raw_hi.append(Vec(s_acc_hi)[r])

                if const_expr(CAUSAL):
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_div_32_i32 = fx.Int32(lane_div_32)
                    q_start_i32 = fx.Int32(q_start) + delta_i32
                    q_mask_limit_i32 = q_row_i32 + delta_i32
                    max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                    tile_needs_mask = max_kv_col_i32 > q_start_i32
                    s_raw_lo_0 = s_raw_lo[0]
                    s_raw_lo_1 = s_raw_lo[1]
                    s_raw_lo_2 = s_raw_lo[2]
                    s_raw_lo_3 = s_raw_lo[3]
                    s_raw_lo_4 = s_raw_lo[4]
                    s_raw_lo_5 = s_raw_lo[5]
                    s_raw_lo_6 = s_raw_lo[6]
                    s_raw_lo_7 = s_raw_lo[7]
                    s_raw_lo_8 = s_raw_lo[8]
                    s_raw_lo_9 = s_raw_lo[9]
                    s_raw_lo_10 = s_raw_lo[10]
                    s_raw_lo_11 = s_raw_lo[11]
                    s_raw_lo_12 = s_raw_lo[12]
                    s_raw_lo_13 = s_raw_lo[13]
                    s_raw_lo_14 = s_raw_lo[14]
                    s_raw_lo_15 = s_raw_lo[15]
                    s_raw_hi_0 = s_raw_hi[0]
                    s_raw_hi_1 = s_raw_hi[1]
                    s_raw_hi_2 = s_raw_hi[2]
                    s_raw_hi_3 = s_raw_hi[3]
                    s_raw_hi_4 = s_raw_hi[4]
                    s_raw_hi_5 = s_raw_hi[5]
                    s_raw_hi_6 = s_raw_hi[6]
                    s_raw_hi_7 = s_raw_hi[7]
                    s_raw_hi_8 = s_raw_hi[8]
                    s_raw_hi_9 = s_raw_hi[9]
                    s_raw_hi_10 = s_raw_hi[10]
                    s_raw_hi_11 = s_raw_hi[11]
                    s_raw_hi_12 = s_raw_hi[12]
                    s_raw_hi_13 = s_raw_hi[13]
                    s_raw_hi_14 = s_raw_hi[14]
                    s_raw_hi_15 = s_raw_hi[15]

                    if tile_needs_mask:
                        # KV_VECTORIZED applies sigma(kv) in the K load, so the score at
                        # logical n_pos holds physical kv = kv_start + sigma(n_pos):
                        # lane_off bit2 -> bit3 (lane_div_32*8) and _off -> sigma(_off).
                        if const_expr(KV_VECTORIZED):
                            lane_off_i32 = lane_div_32_i32 * fx.Int32(8)
                            _MOFF = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)
                        else:
                            lane_off_i32 = lane_div_32_i32 * fx.Int32(4)
                            _MOFF = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)
                        kv_col_lo_0 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[0])
                        s_raw_lo_0 = ArithValue(kv_col_lo_0 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_0)
                        s_raw_hi_0 = ArithValue(kv_col_lo_0 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_0
                        )
                        kv_col_lo_1 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[1])
                        s_raw_lo_1 = ArithValue(kv_col_lo_1 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_1)
                        s_raw_hi_1 = ArithValue(kv_col_lo_1 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_1
                        )
                        kv_col_lo_2 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[2])
                        s_raw_lo_2 = ArithValue(kv_col_lo_2 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_2)
                        s_raw_hi_2 = ArithValue(kv_col_lo_2 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_2
                        )
                        kv_col_lo_3 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[3])
                        s_raw_lo_3 = ArithValue(kv_col_lo_3 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_3)
                        s_raw_hi_3 = ArithValue(kv_col_lo_3 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_3
                        )
                        kv_col_lo_4 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[4])
                        s_raw_lo_4 = ArithValue(kv_col_lo_4 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_4)
                        s_raw_hi_4 = ArithValue(kv_col_lo_4 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_4
                        )
                        kv_col_lo_5 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[5])
                        s_raw_lo_5 = ArithValue(kv_col_lo_5 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_5)
                        s_raw_hi_5 = ArithValue(kv_col_lo_5 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_5
                        )
                        kv_col_lo_6 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[6])
                        s_raw_lo_6 = ArithValue(kv_col_lo_6 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_6)
                        s_raw_hi_6 = ArithValue(kv_col_lo_6 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_6
                        )
                        kv_col_lo_7 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[7])
                        s_raw_lo_7 = ArithValue(kv_col_lo_7 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_7)
                        s_raw_hi_7 = ArithValue(kv_col_lo_7 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_7
                        )
                        kv_col_lo_8 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[8])
                        s_raw_lo_8 = ArithValue(kv_col_lo_8 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_8)
                        s_raw_hi_8 = ArithValue(kv_col_lo_8 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_8
                        )
                        kv_col_lo_9 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[9])
                        s_raw_lo_9 = ArithValue(kv_col_lo_9 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_9)
                        s_raw_hi_9 = ArithValue(kv_col_lo_9 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_9
                        )
                        kv_col_lo_10 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[10])
                        s_raw_lo_10 = ArithValue(kv_col_lo_10 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_10)
                        s_raw_hi_10 = ArithValue(kv_col_lo_10 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_10
                        )
                        kv_col_lo_11 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[11])
                        s_raw_lo_11 = ArithValue(kv_col_lo_11 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_11)
                        s_raw_hi_11 = ArithValue(kv_col_lo_11 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_11
                        )
                        kv_col_lo_12 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[12])
                        s_raw_lo_12 = ArithValue(kv_col_lo_12 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_12)
                        s_raw_hi_12 = ArithValue(kv_col_lo_12 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_12
                        )
                        kv_col_lo_13 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[13])
                        s_raw_lo_13 = ArithValue(kv_col_lo_13 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_13)
                        s_raw_hi_13 = ArithValue(kv_col_lo_13 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_13
                        )
                        kv_col_lo_14 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[14])
                        s_raw_lo_14 = ArithValue(kv_col_lo_14 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_14)
                        s_raw_hi_14 = ArithValue(kv_col_lo_14 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_14
                        )
                        kv_col_lo_15 = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[15])
                        s_raw_lo_15 = ArithValue(kv_col_lo_15 > q_mask_limit_i32).select(c_neg_inf, s_raw_lo_15)
                        s_raw_hi_15 = ArithValue(kv_col_lo_15 + fx.Int32(K_SUB_N) > q_mask_limit_i32).select(
                            c_neg_inf, s_raw_hi_15
                        )

                    s_raw_lo = [
                        s_raw_lo_0,
                        s_raw_lo_1,
                        s_raw_lo_2,
                        s_raw_lo_3,
                        s_raw_lo_4,
                        s_raw_lo_5,
                        s_raw_lo_6,
                        s_raw_lo_7,
                        s_raw_lo_8,
                        s_raw_lo_9,
                        s_raw_lo_10,
                        s_raw_lo_11,
                        s_raw_lo_12,
                        s_raw_lo_13,
                        s_raw_lo_14,
                        s_raw_lo_15,
                    ]
                    s_raw_hi = [
                        s_raw_hi_0,
                        s_raw_hi_1,
                        s_raw_hi_2,
                        s_raw_hi_3,
                        s_raw_hi_4,
                        s_raw_hi_5,
                        s_raw_hi_6,
                        s_raw_hi_7,
                        s_raw_hi_8,
                        s_raw_hi_9,
                        s_raw_hi_10,
                        s_raw_hi_11,
                        s_raw_hi_12,
                        s_raw_hi_13,
                        s_raw_hi_14,
                        s_raw_hi_15,
                    ]
                else:
                    # Mask physical KV columns outside seqlen so tail rows do not enter softmax.
                    kv_start_i32 = fx.Int32(kv_start)
                    seq_len_i32 = fx.Int32(seqlen_kv_b)
                    # KV_VECTORIZED: sigma(kv) in K load -> physical kv = kv_start +
                    # lane_div_32*8 + sigma(_off); hi adds K_SUB_N (bit5 unchanged).
                    if const_expr(KV_VECTORIZED):
                        lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(8)
                        _MOFF = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)
                    else:
                        lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
                        _MOFF = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)
                    for r in range_constexpr(16):
                        kv_col = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[r])
                        s_raw_lo[r] = ArithValue(kv_col >= seq_len_i32).select(c_neg_inf, s_raw_lo[r])
                        s_raw_hi[r] = ArithValue(kv_col + fx.Int32(K_SUB_N) >= seq_len_i32).select(
                            c_neg_inf, s_raw_hi[r]
                        )

                local_max = s_raw_lo[0]
                for r in range_constexpr(15):
                    local_max = _fmax(local_max, s_raw_lo[r + 1])
                for r in range_constexpr(16):
                    local_max = _fmax(local_max, s_raw_hi[r])
                peer_max = reduction_peer(local_max)
                row_max = _fmax(local_max, peer_max)
                m_new_raw = _fmax(m_running, row_max)
                if const_expr(CAUSAL and CROSS_SEQLEN):
                    m_new_raw = _fmax(m_new_raw, c_neg_floor)

                diff_m_raw = _fsub(m_running, m_new_raw)
                diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
                corr = ArithValue(diff_m_scaled).exp2(fastmath=fm_fast)

                scaled_max = _fmul(c_sm_scale_log2e, m_new_raw)
                neg_scaled_max = _fsub(c_zero_f, scaled_max)

                p_vals_lo = []
                p_vals_hi = []
                local_sum = c_zero_f
                for r in range_constexpr(16):
                    diff_lo = fmath.fma(s_raw_lo[r], c_sm_scale_log2e, neg_scaled_max, fastmath=fm_fast)
                    p_lo = ArithValue(diff_lo).exp2(fastmath=fm_fast)
                    p_vals_lo.append(p_lo)
                    local_sum = _fadd(local_sum, p_lo)
                for r in range_constexpr(16):
                    diff_hi = fmath.fma(s_raw_hi[r], c_sm_scale_log2e, neg_scaled_max, fastmath=fm_fast)
                    p_hi = ArithValue(diff_hi).exp2(fastmath=fm_fast)
                    p_vals_hi.append(p_hi)
                    local_sum = _fadd(local_sum, p_hi)

                peer_sum = reduction_peer(local_sum)
                tile_sum = _fadd(local_sum, peer_sum)
                l_corr = _fmul(corr, l_running)
                l_new = _fadd(l_corr, tile_sum)

                # ==== Rescale O accumulators ====
                corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
                if const_expr(not USE_HW_TR):
                    o_accs[0] = _fmul(Vec(o_accs[0]), corr_vec)
                else:
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = _fmul(Vec(o_accs[dc]), corr_vec)

                if const_expr(ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < N_SUBTILES):
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * BLOCK_N
                    next_k_slot = CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(next_k_start, next_k_slot)
                    else:
                        coop_load_k(next_k_start, next_k_slot)

                if const_expr(ENABLE_PREFETCH_3BUF):
                    v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                    v_base = v_buf_base(v_slot)
                    coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif const_expr(ENABLE_DMA):
                    v_base = v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                elif const_expr(KV_VECTORIZED and V_NOMAJOR_DMA):
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()

                # ==== Build P packs for lo and hi halves ====
                if const_expr(dtype_str == "bf16" and not USE_K16):
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 4
                        p_packs_lo.append(bf16_trunc_pack_v4(p_vals_lo[p_base : p_base + 4]))
                        p_packs_hi.append(bf16_trunc_pack_v4(p_vals_hi[p_base : p_base + 4]))
                elif const_expr(dtype_str == "bf16" and USE_K16):
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 8
                        p_packs_lo.append(bf16_trunc_pack_v8(p_vals_lo[p_base : p_base + 8]))
                        p_packs_hi.append(bf16_trunc_pack_v8(p_vals_hi[p_base : p_base + 8]))
                else:
                    p_f16_lo = []
                    p_f16_hi = []
                    for r in range_constexpr(16):
                        p_f16_lo.append(fx.Float32(p_vals_lo[r]).to(elem_dtype))
                        p_f16_hi.append(fx.Float32(p_vals_hi[r]).to(elem_dtype))

                    if const_expr(USE_K16):
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 8
                            p_packs_lo.append(
                                Vec.from_elements(
                                    [
                                        p_f16_lo[p_base + 0],
                                        p_f16_lo[p_base + 1],
                                        p_f16_lo[p_base + 2],
                                        p_f16_lo[p_base + 3],
                                        p_f16_lo[p_base + 4],
                                        p_f16_lo[p_base + 5],
                                        p_f16_lo[p_base + 6],
                                        p_f16_lo[p_base + 7],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )
                            p_packs_hi.append(
                                Vec.from_elements(
                                    [
                                        p_f16_hi[p_base + 0],
                                        p_f16_hi[p_base + 1],
                                        p_f16_hi[p_base + 2],
                                        p_f16_hi[p_base + 3],
                                        p_f16_hi[p_base + 4],
                                        p_f16_hi[p_base + 5],
                                        p_f16_hi[p_base + 6],
                                        p_f16_hi[p_base + 7],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )
                    else:
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 4
                            p_packs_lo.append(
                                Vec.from_elements(
                                    [
                                        p_f16_lo[p_base],
                                        p_f16_lo[p_base + 1],
                                        p_f16_lo[p_base + 2],
                                        p_f16_lo[p_base + 3],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )
                            p_packs_hi.append(
                                Vec.from_elements(
                                    [
                                        p_f16_hi[p_base],
                                        p_f16_hi[p_base + 1],
                                        p_f16_hi[p_base + 2],
                                        p_f16_hi[p_base + 3],
                                    ],
                                    elem_dtype,
                                ).ir_value()
                            )

                # Build flat (dc, pks) schedule for interleaved GEMM2.
                _steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
                TOTAL_PV = len(_steps)

                def _read_v_pack(step_idx):
                    dc, pks = _steps[step_idx]
                    if const_expr(KV_VECTORIZED):
                        # No-major V: one aligned v8 per (dc,pks) half. Lane l reads
                        # V[d=dc*32+l%32, n=pks*16+(l//32)*8+0..7] (lo) / +32 (hi).
                        _lm = lane_mod_32
                        v_lane_base = (
                            v_base
                            + (_lm // fx.Index(8)) * fx.Index(VEC_V_LINE)
                            + lane_div_32 * fx.Index(VEC_V_D128)
                            + (_lm % fx.Index(8)) * fx.Index(8)
                        )
                        _lo_off = dc * (D_CHUNK // 8) * VEC_V_LINE + pks * (PV_K_STEP // 8) * VEC_V_D128
                        _hi_off = _lo_off + (K_SUB_N // 8) * VEC_V_D128
                        vl = Vec.load(mfma_pack_type, lds_kv, [v_lane_base + fx.Index(_lo_off)])
                        vh = Vec.load(mfma_pack_type, lds_kv, [v_lane_base + fx.Index(_hi_off)])
                        return vl, vh
                    if const_expr(USE_HW_TR):
                        d_col = fx.Index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
                        k_row = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
                        _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                        lds_lo = v_base + k_row * V_STRIDE + _d_col_eff
                        lds_hi = lds_lo + fx.Index(K_SUB_N * V_STRIDE)
                        if const_expr(USE_K16):
                            vl_a = ds_read_tr_v4f16(lds_lo)
                            vl_b = ds_read_tr_v4f16(lds_lo + fx.Index(8 * V_STRIDE))
                            vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                            vh_a = ds_read_tr_v4f16(lds_hi)
                            vh_b = ds_read_tr_v4f16(lds_hi + fx.Index(8 * V_STRIDE))
                            vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        else:
                            vl = ds_read_tr_v4f16(lds_lo)
                            vh = ds_read_tr_v4f16(lds_hi)
                    else:
                        d_pos = fx.Index(dc * D_CHUNK) + lane_mod_32
                        k_base = fx.Index(pks * PV_K_STEP) + lane_div_32 * 4
                        v_lo_idx = v_base + d_pos * VT_STRIDE + k_base
                        v_hi_idx = v_lo_idx + fx.Index(K_SUB_N)
                        vl = Vec.load(v4f16_type, lds_kv, [v_lo_idx])
                        vh = Vec.load(v4f16_type, lds_kv, [v_hi_idx])
                    return vl, vh

                # Pre-read V for the first step.
                v_lo_cur, v_hi_cur = _read_v_pack(0)

                # ==== GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi ====
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_nxt, v_hi_nxt = _read_v_pack(si + 1)
                    o_accs[dc] = mfma_acc(v_lo_cur, p_packs_lo[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(v_hi_cur, p_packs_hi[pks], o_accs[dc])
                    if const_expr(not USE_HW_TR and dc == 0 and pks < D_CHUNKS - 1):
                        o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_cur = v_lo_nxt
                        v_hi_cur = v_hi_nxt

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(_use_dma_dbuf):
                if const_expr(N_SUBTILES % 2 == 1):
                    _yield_args.append(fx.Index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            loop_results = yield _yield_args

        def _zero_o_block():
            if const_expr(USE_PERMLANE_OSTORE):
                zero_pack = Vec.filled(4, 0, fx.Int32)
                for dc in range_constexpr(D_CHUNKS):
                    for g in range_constexpr(2):
                        d_col = fx.Index(dc * D_CHUNK) + (fx.Index(2 * g) + lane_div_32) * fx.Index(8)
                        o_global = global_idx_q(q_row, d_col)
                        buffer_ops.buffer_store(zero_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)
            else:
                zero_pack = Vec.filled(2, 0, fx.Int32)
                for dc in range_constexpr(D_CHUNKS):
                    for grp in range_constexpr(4):
                        d_col = fx.Index(dc * D_CHUNK) + lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                        o_global = global_idx_q(q_row, d_col)
                        buffer_ops.buffer_store(zero_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

        def _normalize_and_store_o():
            # gfx950 fuses half-wave partners into 8 cols/store; o_rsrc drops partial-tile OOB rows.
            l_final = loop_results[1]
            o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

            inv_l_rcp = rocdl.rcp(T.f32, l_final)
            if const_expr(CAUSAL and CROSS_SEQLEN):
                inv_l = ArithValue(fx.Float32(l_final) > c_zero_f).select(inv_l_rcp, c_zero_f)
            else:
                inv_l = inv_l_rcp
            inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
            v_o = [Vec(o_finals[dc]) * inv_l_vec for dc in range_constexpr(D_CHUNKS)]

            if const_expr(USE_PERMLANE_OSTORE):
                # gfx950: 128-bit permlane-fused store (cvt_pk_bf16_f32 + permlane32_swap).
                pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
                is_hi_half = ArithValue(lane_div_32 != fx.Index(0))

                def _o_pack_2dw(dc, store_group):
                    # 4 f32 outputs -> 2 packed-16bit dwords (lo = cols 0,1; hi = cols 2,3).
                    r_base = store_group * 4
                    if const_expr(dtype_str == "bf16"):
                        lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
                        hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
                        return lo, hi
                    o_f16 = [fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype) for i in range_constexpr(4)]
                    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
                    return _raw(pack[0]), _raw(pack[1])

                def _swap_halves(dw):
                    # permlane32_swap(a,b) -> (a.lo|b.lo, a.hi|b.hi); with a=b=dw the
                    # partner dword dw[lane^32] is result[1] on low lanes, [0] on high.
                    swapped = rocdl.permlane32_swap(pair_i32_ty, _raw(dw), _raw(dw), False, False)
                    lo_res = llvm.extractvalue(T.i32, swapped, [0])
                    hi_res = llvm.extractvalue(T.i32, swapped, [1])
                    return is_hi_half.select(lo_res, hi_res)

                for dc in range_constexpr(D_CHUNKS):
                    for g in range_constexpr(2):
                        d0_a, d1_a = _o_pack_2dw(dc, 2 * g)
                        d0_b, d1_b = _o_pack_2dw(dc, 2 * g + 1)
                        # low lanes: own group-2g cols 0-3 ++ partner's cols 4-7;
                        # high lanes: partner's group-(2g+1) cols 0-3 ++ own cols 4-7.
                        y0_a, y1_a = _swap_halves(d0_a), _swap_halves(d1_a)
                        y0_b, y1_b = _swap_halves(d0_b), _swap_halves(d1_b)
                        w0 = is_hi_half.select(y0_b, _raw(d0_a))
                        w1 = is_hi_half.select(y1_b, _raw(d1_a))
                        w2 = is_hi_half.select(_raw(d0_b), y0_a)
                        w3 = is_hi_half.select(_raw(d1_b), y1_a)
                        o_pack = Vec.from_elements([fx.Int32(w0), fx.Int32(w1), fx.Int32(w2), fx.Int32(w3)], fx.Int32)
                        d_col = fx.Index(dc * D_CHUNK) + (fx.Index(2 * g) + lane_div_32) * fx.Index(8)
                        o_global = global_idx_q(q_row, d_col)
                        buffer_ops.buffer_store(o_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)
            else:
                # gfx942 fallback (no permlane32_swap / cvt_pk_bf16_f32): each lane stores
                # its 16 cols as 4 dwordx2 groups via .to(elem_dtype); col map d_col =
                # dc*D_CHUNK + lane_div_32*4 + 8*grp + r. num_records bound drops OOB rows.
                for dc in range_constexpr(D_CHUNKS):
                    for grp in range_constexpr(4):
                        r0 = grp * 4
                        o_f16 = [fx.Float32(Vec(v_o[dc])[r0 + i]).to(elem_dtype) for i in range_constexpr(4)]
                        pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
                        o2 = Vec.from_elements([_raw(pack[0]), _raw(pack[1])], fx.Int32)
                        d_col = fx.Index(dc * D_CHUNK) + lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                        o_global = global_idx_q(q_row, d_col)
                        buffer_ops.buffer_store(o2, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

        @flyc.jit
        def _store_cross_o():
            if causal_end_raw_i32 <= c_zero_i32:
                _zero_o_block()
            else:
                _normalize_and_store_o()

        # ---- Normalize and store O (128-bit buffer_store_dwordx4) ----
        if const_expr(CAUSAL and CROSS_SEQLEN):
            _store_cross_o()
        else:
            _normalize_and_store_o()

    @flyc.jit
    def launch_flash_attn_generic(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS_Q

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_generic_kernel(
            Q,
            K,
            V,
            O,
            seq_len,
            seq_len_kv,
            CuSeqQ,
            CuSeqKv,
            BlockTable,
            block_table_stride,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{int(flat_work_group_size)},{int(flat_work_group_size)}"
                    if const_expr(flat_work_group_size is not None)
                    else None
                ),
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Best MI355X FMHA numbers were measured with ROCm/llvm-project `felix/tune_fmha`;
    # other LLVM revisions usually leave a few percent of peak throughput on the table.
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(
        Q,
        K,
        V,
        Out,
        batch_size,
        seq_len,
        *,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=0,
        seq_len_kv=None,
        stream=None,
    ):
        # Dense/non-paged pass the output tensor as a placeholder for the unused
        # cu_seqlens / block_table slots; the kernel only reads them under
        # const_expr(VARLEN) / const_expr(PAGED).
        cq = cu_seqlens_q if cu_seqlens_q is not None else Out
        ck = cu_seqlens_kv if cu_seqlens_kv is not None else Out
        bt = block_table if block_table is not None else Out
        skv = seq_len if seq_len_kv is None else seq_len_kv
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_generic(
                Q, K, V, Out, cq, ck, bt, block_table_stride, batch_size, seq_len, skv, fx.Stream(stream)
            )

    def _compile(Q, K, V, Out, batch_size, seq_len, seq_len_kv=None, stream=None):
        skv = seq_len if seq_len_kv is None else seq_len_kv
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flash_attn_generic, Q, K, V, Out, Out, Out, Out, 0, batch_size, seq_len, skv, fx.Stream(stream)
            )

    _launch.compile = _compile

    if block_m is None:
        return _guard_seqlen(_launch)
    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary
