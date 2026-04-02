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

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from kernels.kernels_common import dtype_to_elem_type
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref, scf, fly as _fly, llvm as _llvm, math as math_dialect

# ---- Module-level constants ----

KERNEL_NAME = "flash_attn_func_kernel"

_LOG2E = math.log2(math.e)  # 1.4426950408889634

_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel (0x80000000 as signed i32)

def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")

_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _waitcnt_lgkm0_vm_n(n):
    """Emit single s_waitcnt with lgkmcnt(0) + vmcnt(n) — HK pattern for clusters 1/3/5/7.
    Encodes both counters in one s_waitcnt instruction."""
    vm_lo = n & _VMCNT_LO_MASK
    vm_hi = ((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT
    lgkm_0 = 0 << 8  # lgkmcnt field at bits [12:8], 0 = wait for all
    expcnt_7 = 7 << 4  # expcnt field at bits [6:4], 7 = don't wait
    val = vm_lo | expcnt_7 | lgkm_0 | vm_hi
    rocdl.s_waitcnt(val)


# ---- HipKittens-style scheduling barrier helpers ----
# Match HipKittens' sched_barrier_pairs / sched_barrier_exp_pairs templates.
# These force the hardware scheduler to interleave MFMA instructions with
# VALU or EXP (transcendental) instructions to hide latency.
_MASK_VALU = 0x002
_MASK_MFMA = 0x008
_MASK_EXP = 0x400


def _sched_barrier_pairs(n_pairs, valu_cnt, group_id):
    """Emit n_pairs of (1 MFMA + valu_cnt VALU) scheduling barriers."""
    for _ in range_constexpr(n_pairs):
        rocdl.sched_group_barrier(_MASK_MFMA, 1, group_id)
        rocdl.sched_group_barrier(_MASK_VALU, valu_cnt, group_id)


def _sched_barrier_exp_pairs(n_pairs, exp_cnt, group_id):
    """Emit n_pairs of (1 MFMA + exp_cnt EXP) scheduling barriers."""
    for _ in range_constexpr(n_pairs):
        rocdl.sched_group_barrier(_MASK_MFMA, 1, group_id)
        rocdl.sched_group_barrier(_MASK_EXP, exp_cnt, group_id)


_RESCALE_THRESHOLD = 8.0


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
):
    """Build the flash_attn_func launcher using the post-refactor FlyDSL API."""
    gpu_arch = get_hip_arch()

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64

    # Auto tile selection: for H>=32, build both M=128 and M=256 variants
    # and dispatch at runtime based on B*S.
    if block_m is None and num_heads >= 32:
        _launcher_m128 = build_flash_attn_func_module_primary(
            num_heads, head_dim, causal, dtype_str, sm_scale, waves_per_eu,
            flat_work_group_size=256, block_m=128,
            unsafe_fp_math=unsafe_fp_math, fast_fp_math=fast_fp_math,
            daz=daz, path_tag=path_tag)
        _launcher_m256 = build_flash_attn_func_module_primary(
            num_heads, head_dim, causal, dtype_str, sm_scale, waves_per_eu,
            flat_work_group_size=512, block_m=256,
            unsafe_fp_math=unsafe_fp_math, fast_fp_math=fast_fp_math,
            daz=daz, path_tag=path_tag)
        _BS_THRESHOLD = 4096 * num_heads

        def _auto_launch(*args, **kwargs):
            B = args[4] if len(args) > 4 else kwargs.get('batch_size', 1)
            S = args[5] if len(args) > 5 else kwargs.get('seq_len', 128)
            bs = (B if isinstance(B, int) else 1) * (S if isinstance(S, int) else 128)
            if bs * num_heads >= _BS_THRESHOLD:
                return _launcher_m256(*args, **kwargs)
            return _launcher_m128(*args, **kwargs)

        return _auto_launch

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
    if path_tag.upper() in ("N32", "N128"):
        PATH_TAG = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        PATH_TAG = "N128"
    else:
        PATH_TAG = "N32"
    BLOCK_N_OUT = 128 if PATH_TAG == "N128" else BLOCK_N
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    ENABLE_PREFETCH_3BUF = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    )
    # buffer_load_dwordx4_lds (16B DMA-to-LDS) requires gfx950+; gfx94x only has dword (4B).
    _has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    ENABLE_DMA = _has_lds_load_b128 and (
        PATH_TAG == "N128" or (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1"
        )
    )
    ENABLE_LDS_VEC16 = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    )
    REDUCE_MODE = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if REDUCE_MODE not in ("xor", "ds_bpermute"):
        REDUCE_MODE = "xor"
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else (2 if ENABLE_DMA else 1)
    _USE_DMA_DBUF = ENABLE_DMA and not ENABLE_PREFETCH_3BUF
    _ENABLE_V_DBUF = _USE_DMA_DBUF
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else (2 if _ENABLE_V_DBUF else 1)
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)

    # gfx950+ has ds_read_tr16_b64 (HW transpose LDS read); gfx942 needs V^T stored in LDS.
    USE_HW_TR = gpu_arch.startswith("gfx950")

    # MFMA32 K-dimension: 16 on gfx950+ (CDNA4) for both GEMMs.
    USE_K16 = gpu_arch.startswith("gfx950")
    K_STEP_QK = 16 if USE_K16 else 8
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16 if USE_K16 else 8
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2 steps per sub-tile (K=16) or 4 (K=8)

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert flat_work_group_size in (128, 256, 512), (
        f"flat_work_group_size must be 128, 256, or 512, got {flat_work_group_size}"
    )
    assert dtype_str in ("f16", "bf16"), "flash_attn_func only supports f16 and bf16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT % BLOCK_N == 0

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # Bank-conflict-free LDS strides.
    # K uses XOR swizzle (col ^ ((row & 7) << 4)) at 16-element granularity
    # instead of padding. This enables ds_read_b128 (stride is 256B-aligned).
    K_STRIDE = HEAD_DIM
    if USE_HW_TR:
        V_STRIDE = HEAD_DIM if ENABLE_DMA else HEAD_DIM + 4
    else:
        VT_STRIDE = BLOCK_N + 2
        V_STRIDE = VT_STRIDE

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
    if USE_HW_TR:
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
    def flash_attn_func_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        seq_len: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        # The unsafe_fp_math/fast_fp_math builder params control LLVM-level attributes only.
        fm_fast = arith.FastMathFlags.fast
        v4f16_type = T.vec(4, elem_type)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        v16f32_type = T.vec(16, compute_type)
        mfma_pack_type = v8f16_type if USE_K16 else v4f16_type
        MFMA_LANE_K = 8 if USE_K16 else 4
        _mfma_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        def _mfma(ods_fn, a, b, c):
            return ods_fn(v16f32_type, a, b, c, _mfma_zero, _mfma_zero, _mfma_zero).result
        def mfma_acc(a, b, c):
            if dtype_str == "bf16":
                if USE_K16:
                    return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)
                a = vector.bitcast(T.i16x4, a)
                b = vector.bitcast(T.i16x4, b)
                return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
            if USE_K16:
                return _mfma(rocdl.mfma_f32_32x32x16_f16, a, b, c)
            return _mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

        seq_len_v = arith.index_cast(T.index, seq_len)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        # ---- Thread / block indices ----
        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        # ---- Wave decomposition ----
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ---- ds_read_b64_tr_b16 lane decomposition ----
        # Hardware does 4×4 transpose within blocks of 16 lanes.
        # tr_k_group selects which of 4 K-rows within the block,
        # tr_col_sub selects which 4-column sub-group within 16 columns.
        tr_k_group = (lane % 16) // 4   # 0..3: K-row offset within 4-row group
        tr_col_sub = lane % 4            # 0..3: 4-column sub-group
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
            byte_i64 = arith.index_cast(T.i64, byte_offset)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- Wave offsets ----
        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id ----
        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat index ----
        def global_idx(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def _gep_load(base_ptr, elem_idx, vec_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=elem_type,
                              noWrapFlags=0)
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store(val, base_ptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=elem_type,
                              noWrapFlags=0)
            _llvm.StoreOp(val, gep.result)

        def load_global_f16x4(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, v4f16_type)

        def load_global_mfma_pack(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, mfma_pack_type)

        def load_global_f16xN(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, vxf16_type)

        def bf16_trunc_pack_v4(f32_vals):
            """Pack 4 f32 values into v4bf16 via bitwise truncation (upper 16 bits).
            ~2 fewer instructions/element vs arith.TruncFOp round-to-nearest."""
            _v2i32 = T.vec(2, T.i32)
            _c16 = arith.constant(16, type=T.i32)
            _cmask = arith.constant(0xFFFF0000, type=T.i32)
            a0 = arith.ArithValue(f32_vals[0]).bitcast(T.i32)
            b0 = arith.ArithValue(f32_vals[1]).bitcast(T.i32)
            p0 = arith.OrIOp(arith.AndIOp(b0, _cmask).result,
                             arith.ShRUIOp(a0, _c16).result).result
            a1 = arith.ArithValue(f32_vals[2]).bitcast(T.i32)
            b1 = arith.ArithValue(f32_vals[3]).bitcast(T.i32)
            p1 = arith.OrIOp(arith.AndIOp(b1, _cmask).result,
                             arith.ShRUIOp(a1, _c16).result).result
            return vector.bitcast(v4f16_type, vector.from_elements(_v2i32, [p0, p1]))

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits)."""
            _v4i32 = T.vec(4, T.i32)
            _c16 = arith.constant(16, type=T.i32)
            _cmask = arith.constant(0xFFFF0000, type=T.i32)
            pairs = []
            for j in range_constexpr(4):
                a = arith.ArithValue(f32_vals[j * 2]).bitcast(T.i32)
                b = arith.ArithValue(f32_vals[j * 2 + 1]).bitcast(T.i32)
                p = arith.OrIOp(arith.AndIOp(b, _cmask).result,
                                arith.ShRUIOp(a, _c16).result).result
                pairs.append(p)
            return vector.bitcast(v8f16_type, vector.from_elements(_v4i32, pairs))

        def k_buf_base(buf_id):
            if isinstance(buf_id, int):
                return arith.index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * arith.index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            if isinstance(buf_id, int):
                return arith.index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)
            return arith.index(LDS_V_BASE) + buf_id * arith.index(LDS_V_TILE_SIZE)

        # ---- K XOR swizzle: col ^ ((row & 7) << 4) at 16-element granularity ----
        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(0x7)) << arith.index(4)
            return col_idx ^ mask

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_k = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_k.then_block):
                        g_idx = global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vec = load_global_f16xN(k_ptr, g_idx)
                    vector.store(vec, lds_kv, [lds_idx])

        # ---- Cooperative V load ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            vector.store(vec, lds_kv, [lds_idx])

        _v1_type = T.vec(1, elem_type) if not USE_HW_TR else None

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(VEC_WIDTH):
                elem = vector.extract(vec, static_position=[_e], dynamic_position=[])
                vt_d = load_col_base + _e
                vt_idx = v_base + vt_d * VT_STRIDE + lds_row
                v1 = vector.from_elements(_v1_type, [elem])
                vector.store(v1, lds_kv, [vt_idx])

        _v_store_to_lds = _v_store_row_major if USE_HW_TR else _v_store_transposed

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        g_idx = global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                        scf.YieldOp([])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(v_ptr, g_idx)
                    _v_store_to_lds(v_base, lds_row, vec)

        def coop_load_v_global(tile_start):
            """Issue global loads for V, return vectors (non-blocking)."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = global_idx(row_idx, load_col_base)
                vecs.append(load_global_f16xN(v_ptr, g_idx))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            """Write previously-loaded V vectors to LDS."""
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vecs[batch])
                        scf.YieldOp([])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vecs[batch])

        # ---- DMA loading for K (buffer_load_dwordx4 ... lds) ----
        if ENABLE_DMA:
            from flydsl._mlir.dialects import llvm
            k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
            _lds_ptr_ty = _llvm_lds_ptr_ty()
            DMA_BYTES = 16  # buffer_load_dwordx4 = 16 bytes per lane
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
            NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            lds_kv_base_idx = _memref.extract_aligned_pointer_as_index(lds_kv)
            _dma_size = arith.constant(DMA_BYTES, type=T.i32)
            _dma_soff = arith.constant(0, type=T.i32)
            _dma_off = arith.constant(0, type=T.i32)
            _dma_aux = arith.constant(1, type=T.i32)

            def coop_dma_k(tile_start, buf_id=0):
                """Load K tile via DMA with XOR-swizzled global fetch."""
                if isinstance(buf_id, int):
                    k_lds_byte_base = lds_kv_base_idx + arith.index(buf_id * LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * arith.index(LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (k_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_K_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH))
                    swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(0x7)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = (batch_idx * seq_len_v + tile_start
                                  + row_in_tile)
                    global_byte = (global_row * arith.index(STRIDE_TOKEN * 2)
                                   + head_idx * arith.index(HEAD_DIM * 2)
                                   + col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

        # ---- V XOR swizzle: col ^ ((row & 3) << 4) at 16-element granularity ----
        def _v_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(0x3)) << arith.index(4)
            return col_idx ^ mask

        # ---- DMA loading for V (buffer_load_dwordx4 ... lds) ----
        if ENABLE_DMA:
            v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
            V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
            NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH_V = DMA_BATCH_BYTES // (HEAD_DIM * 2)

            def coop_dma_v(tile_start, buf_id=0):
                """Load V tile via DMA with XOR-swizzled global fetch."""
                if isinstance(buf_id, int):
                    v_lds_byte_base = (lds_kv_base_idx
                                       + arith.index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2))
                else:
                    v_lds_byte_base = (lds_kv_base_idx
                                       + arith.index(LDS_V_BASE * 2)
                                       + buf_id * arith.index(LDS_V_TILE_SIZE * 2))
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (v_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_V_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH_V))
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(0x3)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = (batch_idx * seq_len_v + tile_start
                                  + row_in_tile)
                    global_byte = (global_row * arith.index(STRIDE_TOKEN * 2)
                                   + head_idx * arith.index(HEAD_DIM * 2)
                                   + col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = arith.index_cast(T.i32, q_row)
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row, seq_len_v)
        q_row_safe = arith.select(q_in_bounds, q_row, arith.index(0))
        c_zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_type)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            raw = load_global_mfma_pack(q_ptr, g_idx)
            q_b_packs.append(arith.select(q_in_bounds, raw, c_zero_mfma_pack))

        # ---- Constants ----
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_sm_scale_log2e = arith.constant(sm_scale * _LOG2E, type=compute_type)
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        shuf_32_i32 = arith.constant(32, type=T.i32)
        c4_i32 = arith.constant(4, type=T.i32)
        lane_i32 = arith.index_cast(T.i32, lane)
        lane_xor_32_i32 = arith.XOrIOp(lane_i32, shuf_32_i32).result
        lane_xor_32_byte = arith.MulIOp(lane_xor_32_i32, c4_i32).result

        def reduction_peer(v_f32):
            if REDUCE_MODE == "ds_bpermute":
                v_i32 = arith.ArithValue(v_f32).bitcast(T.i32)
                peer_i32 = rocdl.ds_bpermute(T.i32, lane_xor_32_byte, v_i32)
                return arith.ArithValue(peer_i32).bitcast(compute_type)
            return arith.ArithValue(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop upper bound ----
        _q_end = q_start + BLOCK_M
        if CAUSAL:
            kv_upper = arith.MinSIOp(_q_end, seq_len_v).result
        else:
            kv_upper = seq_len_v

        _use_dma_dbuf = ENABLE_DMA and not ENABLE_PREFETCH_3BUF

        # HK stagger flag (HK L194)
        _hk_enable_stagger = (NUM_WAVES >= 4 and _use_dma_dbuf)
        if _hk_enable_stagger:
            _stagger_flag = arith.cmpi(
                arith.CmpIPredicate.uge, wave_id, arith.index(NUM_WAVES // 2))

        # ---- K/V LDS index helpers (shared by all paths) ----
        k_hi_offset = K_SUB_N * K_STRIDE
        k_swz_mask = (lane_mod_32 & arith.index(0x7)) << arith.index(4)

        def _k_idx_lo_buf(ks, kb):
            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return kb + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

        def _k_idx_hi_buf(ks, kb):
            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return kb + k_hi_offset + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

        def _read_all_k_packs(kb):
            kp_lo = []; kp_hi = []
            for ks in range_constexpr(K_STEPS_QK):
                kp_lo.append(vector.load_op(mfma_pack_type, lds_kv, [_k_idx_lo_buf(ks, kb)]))
                kp_hi.append(vector.load_op(mfma_pack_type, lds_kv, [_k_idx_hi_buf(ks, kb)]))
            return kp_lo, kp_hi

        def _do_qk_gemm(kp_lo, kp_hi):
            acc_lo = c_zero_v16f32
            acc_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                acc_lo = mfma_acc(kp_lo[ks], q_b_packs[ks], acc_lo)
                acc_hi = mfma_acc(kp_hi[ks], q_b_packs[ks], acc_hi)
            return acc_lo, acc_hi

        def _extract_s_raw(s_acc_lo, s_acc_hi):
            slo = []; shi = []
            for r in range_constexpr(16):
                slo.append(vector.extract(s_acc_lo, static_position=[r], dynamic_position=[]))
                shi.append(vector.extract(s_acc_hi, static_position=[r], dynamic_position=[]))
            return slo, shi

        def _apply_causal_mask(s_raw_lo, s_raw_hi, kv_start):
            if not CAUSAL:
                return s_raw_lo, s_raw_hi
            kv_start_i32 = arith.index_cast(T.i32, kv_start)
            lane_div_32_i32 = arith.index_cast(T.i32, lane_div_32)
            q_start_i32 = arith.index_cast(T.i32, q_start)
            max_kv_col_i32 = arith.AddIOp(kv_start_i32, arith.constant(BLOCK_N - 1, type=T.i32)).result
            tile_needs_mask = arith.cmpi(arith.CmpIPredicate.ugt, max_kv_col_i32, q_start_i32)
            _mask_if = scf.IfOp(tile_needs_mask, [T.f32] * 32, has_else=True)
            with ir.InsertionPoint(_mask_if.then_block):
                _m_lo = []; _m_hi = []
                for r in range_constexpr(16):
                    r_off_i32 = arith.constant((r % 4) + (r // 4) * 8, type=T.i32)
                    lane_off_i32 = arith.MulIOp(lane_div_32_i32, arith.constant(4, type=T.i32)).result
                    kv_col_lo = arith.AddIOp(arith.AddIOp(kv_start_i32, lane_off_i32).result, r_off_i32).result
                    is_masked_lo = arith.cmpi(arith.CmpIPredicate.ugt, kv_col_lo, q_row_i32)
                    _m_lo.append(arith.select(is_masked_lo, c_neg_inf, s_raw_lo[r]))
                    kv_col_hi = arith.AddIOp(kv_col_lo, arith.constant(K_SUB_N, type=T.i32)).result
                    is_masked_hi = arith.cmpi(arith.CmpIPredicate.ugt, kv_col_hi, q_row_i32)
                    _m_hi.append(arith.select(is_masked_hi, c_neg_inf, s_raw_hi[r]))
                scf.YieldOp(_m_lo + _m_hi)
            with ir.InsertionPoint(_mask_if.else_block):
                scf.YieldOp(s_raw_lo + s_raw_hi)
            return ([_mask_if.results[i] for i in range(16)],
                    [_mask_if.results[16 + i] for i in range(16)])

        def _compute_max(s_raw_lo, s_raw_hi, m_old):
            _mfm = {"fastmath": fm_fast}
            loc = s_raw_lo[0]
            for r in range_constexpr(15):
                loc = arith.MaxNumFOp(loc, s_raw_lo[r + 1], **_mfm).result
            for r in range_constexpr(16):
                loc = arith.MaxNumFOp(loc, s_raw_hi[r], **_mfm).result
            peer = reduction_peer(loc)
            row = arith.MaxNumFOp(loc, peer, **_mfm).result
            return arith.MaxNumFOp(m_old, row, **_mfm).result

        def _compute_softmax_full(s_raw_lo, s_raw_hi, m_new, m_old, l_old, o_accs_in):
            corr_raw = arith.SubFOp(m_old, m_new, fastmath=fm_fast).result
            corr_scaled = arith.MulFOp(corr_raw, c_sm_scale_log2e, fastmath=fm_fast).result
            corr = arith.ArithValue(corr_scaled).exp2(fastmath=fm_fast)
            corr_vec = vector.broadcast(v16f32_type, corr)
            o_out = list(o_accs_in)
            for dc in range_constexpr(D_CHUNKS):
                o_out[dc] = arith.MulFOp(o_out[dc], corr_vec, fastmath=fm_fast).result
            sc_max = arith.MulFOp(c_sm_scale_log2e, m_new, fastmath=fm_fast).result
            neg_sc = arith.SubFOp(c_zero_f, sc_max, fastmath=fm_fast).result
            plo = []; phi = []; lsum = c_zero_f
            for r in range_constexpr(16):
                d = math_dialect.fma(s_raw_lo[r], c_sm_scale_log2e, neg_sc)
                p = arith.ArithValue(d).exp2(fastmath=fm_fast)
                plo.append(p)
                lsum = arith.AddFOp(lsum, p, fastmath=fm_fast).result
            for r in range_constexpr(16):
                d = math_dialect.fma(s_raw_hi[r], c_sm_scale_log2e, neg_sc)
                p = arith.ArithValue(d).exp2(fastmath=fm_fast)
                phi.append(p)
                lsum = arith.AddFOp(lsum, p, fastmath=fm_fast).result
            ps = reduction_peer(lsum)
            ts = arith.AddFOp(lsum, ps, fastmath=fm_fast).result
            lc = arith.MulFOp(corr, l_old, fastmath=fm_fast).result
            ln = arith.AddFOp(lc, ts, fastmath=fm_fast).result
            return plo, phi, m_new, ln, o_out

        def _pack_p_vals(p_vals_lo, p_vals_hi):
            if dtype_str == "bf16" and not USE_K16:
                plo = [bf16_trunc_pack_v4(p_vals_lo[pks*4:pks*4+4]) for pks in range(PV_K_STEPS)]
                phi = [bf16_trunc_pack_v4(p_vals_hi[pks*4:pks*4+4]) for pks in range(PV_K_STEPS)]
            elif dtype_str == "bf16" and USE_K16:
                plo = [bf16_trunc_pack_v8(p_vals_lo[pks*8:pks*8+8]) for pks in range(PV_K_STEPS)]
                phi = [bf16_trunc_pack_v8(p_vals_hi[pks*8:pks*8+8]) for pks in range(PV_K_STEPS)]
            else:
                f16lo = [arith.trunc_f(elem_type, v) for v in p_vals_lo]
                f16hi = [arith.trunc_f(elem_type, v) for v in p_vals_hi]
                if USE_K16:
                    plo = [vector.from_elements(v8f16_type, f16lo[pks*8:pks*8+8]) for pks in range(PV_K_STEPS)]
                    phi = [vector.from_elements(v8f16_type, f16hi[pks*8:pks*8+8]) for pks in range(PV_K_STEPS)]
                else:
                    plo = [vector.from_elements(v4f16_type, f16lo[pks*4:pks*4+4]) for pks in range(PV_K_STEPS)]
                    phi = [vector.from_elements(v4f16_type, f16hi[pks*4:pks*4+4]) for pks in range(PV_K_STEPS)]
            return plo, phi

        _pv_steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
        TOTAL_PV = len(_pv_steps)

        def _read_v_pack_from(v_base_arg, step_idx):
            dc, pks = _pv_steps[step_idx]
            if USE_HW_TR:
                d_col = arith.index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
                k_row = arith.index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
                _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                lds_lo = v_base_arg + k_row * V_STRIDE + _d_col_eff
                lds_hi = lds_lo + arith.index(K_SUB_N * V_STRIDE)
                if USE_K16:
                    vla = ds_read_tr_v4f16(lds_lo)
                    vlb = ds_read_tr_v4f16(lds_lo + arith.index(8 * V_STRIDE))
                    vl = vector.shuffle(vla, vlb, [0,1,2,3,4,5,6,7])
                    vha = ds_read_tr_v4f16(lds_hi)
                    vhb = ds_read_tr_v4f16(lds_hi + arith.index(8 * V_STRIDE))
                    vh = vector.shuffle(vha, vhb, [0,1,2,3,4,5,6,7])
                else:
                    vl = ds_read_tr_v4f16(lds_lo)
                    vh = ds_read_tr_v4f16(lds_hi)
            else:
                d_pos = arith.index(dc * D_CHUNK) + lane_mod_32
                k_base_pv = arith.index(pks * PV_K_STEP) + lane_div_32 * 4
                v_lo_idx = v_base_arg + d_pos * VT_STRIDE + k_base_pv
                v_hi_idx = v_lo_idx + arith.index(K_SUB_N)
                vl = vector.load(v4f16_type, lds_kv, [v_lo_idx])
                vh = vector.load(v4f16_type, lds_kv, [v_hi_idx])
            return vl, vh

        def _preload_v_packs(v_base_arg):
            """Pre-load all V packs from LDS into registers (HK: load(v_reg, v_smem))."""
            vp = []
            for si in range_constexpr(TOTAL_PV):
                vp.append(_read_v_pack_from(v_base_arg, si))
            return vp

        def _do_pv_gemm_from_regs(v_packs_all, pp_lo, pp_hi, o_accs_in):
            """PV GEMM using pre-loaded V packs (HK: mma_AtB(o, v_reg, att_bf16))."""
            o_out = list(o_accs_in)
            for si in range_constexpr(TOTAL_PV):
                dc, pks = _pv_steps[si]
                vl, vh = v_packs_all[si]
                o_out[dc] = mfma_acc(vl, pp_lo[pks], o_out[dc])
                o_out[dc] = mfma_acc(vh, pp_hi[pks], o_out[dc])
            return o_out

        def _do_pv_gemm(v_base_arg, pp_lo, pp_hi, o_accs_in):
            """PV GEMM reading V from LDS on-the-fly (fallback path)."""
            o_out = list(o_accs_in)
            rocdl.s_setprio(1)
            vl_c, vh_c = _read_v_pack_from(v_base_arg, 0)
            for si in range_constexpr(TOTAL_PV):
                dc, pks = _pv_steps[si]
                if si + 1 < TOTAL_PV:
                    vl_n, vh_n = _read_v_pack_from(v_base_arg, si + 1)
                o_out[dc] = mfma_acc(vl_c, pp_lo[pks], o_out[dc])
                o_out[dc] = mfma_acc(vh_c, pp_hi[pks], o_out[dc])
                if si + 1 < TOTAL_PV:
                    vl_c = vl_n; vh_c = vh_n
            rocdl.s_setprio(0)
            _sched_barrier_pairs(TOTAL_PV, 5, 0)
            rocdl.sched_barrier(0)
            return o_out

        # ================================================================
        # DMA path — HK-aligned 4-cluster pipeline, per-tile loop
        # Like gemm's range(start, stop, step, init=...) pattern.
        # Each iteration = 1 KV tile, step = BLOCK_N (not BLOCK_N_OUT).
        # Prologue processes tile 0 (HK L260-315).
        # Hot loop processes tiles 1..N-1 with 4 clusters per tile.
        # Epilogue finishes last deferred softmax + PV.
        #
        # Per-tile cluster layout (HK L322-470):
        #   Cluster 0: QK(carried k_packs) + finish_softmax(carried plo/dhi/corr)
        #   Cluster 1: K DMA(t+2 → buf_id) + V read(v_buf[1-buf_id]) → v_packs
        #   Cluster 2: PV(v_packs, p_packs) + partial_softmax(QK result)
        #   Cluster 3: V DMA(t+1 → v_buf[1-buf_id]) + K read(k_buf[1-buf_id]) → k_packs
        #
        # Buffer: buf_id = t%2. K DMA→k_buf[buf_id], K read→k_buf[1-buf_id].
        #         V read→v_buf[1-buf_id], V DMA→v_buf[1-buf_id].
        # ================================================================
        if _use_dma_dbuf:
            # ---- Prologue: tile 0 (HK L260-315) ----
            # DMA K[0]→k_buf[0]; wait; barrier  (HK L260-263)
            coop_dma_k(arith.index(0), buf_id=0)
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            gpu.barrier()

            # DMA K[1]→k_buf[1]; DMA V[0]→v_buf[0]  (HK L272-274)
            coop_dma_k(arith.index(BLOCK_N), buf_id=1)
            coop_dma_v(arith.index(0), buf_id=0)
            # K read from k_buf[0] = K[0]  (HK L275)
            _pro_kp_lo, _pro_kp_hi = _read_all_k_packs(k_buf_base(0))
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            gpu.barrier()

            # QK[0]  (HK L283-285)
            _pro_s_lo, _pro_s_hi = _do_qk_gemm(_pro_kp_lo, _pro_kp_hi)
            rocdl.sched_barrier(0)
            _pro_sraw_lo, _pro_sraw_hi = _extract_s_raw(_pro_s_lo, _pro_s_hi)
            _pro_sraw_lo, _pro_sraw_hi = _apply_causal_mask(
                _pro_sraw_lo, _pro_sraw_hi, arith.index(0))

            # Partial softmax[0]: col_max, sub_col, exp2 first half (HK L294-297)
            _pro_m = _compute_max(_pro_sraw_lo, _pro_sraw_hi, c_neg_inf)
            _pro_sc = arith.MulFOp(c_sm_scale_log2e, _pro_m, fastmath=fm_fast).result
            _pro_neg_sc = arith.SubFOp(c_zero_f, _pro_sc, fastmath=fm_fast).result
            _init_plo = []
            for _r in range_constexpr(16):
                _d = math_dialect.fma(_pro_sraw_lo[_r], c_sm_scale_log2e, _pro_neg_sc)
                _init_plo.append(arith.ArithValue(_d).exp2(fastmath=fm_fast))
            _init_dhi = []
            for _r in range_constexpr(16):
                _init_dhi.append(math_dialect.fma(
                    _pro_sraw_hi[_r], c_sm_scale_log2e, _pro_neg_sc))
            rocdl.sched_barrier(0)

            # Stagger  (HK L300-303)
            if _hk_enable_stagger:
                _stagger_if = scf.IfOp(_stagger_flag)
                with ir.InsertionPoint(_stagger_if.then_block):
                    rocdl.sched_barrier(0)
                    gpu.barrier()
                    scf.YieldOp([])

            # K read from k_buf[1] = K[1]  (HK L307)
            _init_kp_lo, _init_kp_hi = _read_all_k_packs(k_buf_base(1))
            # DMA K[2]→k_buf[0]; DMA V[1]→v_buf[1]  (HK L309-311)
            coop_dma_k(arith.index(2 * BLOCK_N), buf_id=0)
            coop_dma_v(arith.index(BLOCK_N), buf_id=1)
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            gpu.barrier()

            # ---- Build init_state for per-tile loop ----
            # Pack plo/dhi as v16f32 to reduce SSA count: 56 → 26 loop-carried values
            _init_plo_vec = vector.from_elements(v16f32_type, _init_plo)
            _init_dhi_vec = vector.from_elements(v16f32_type, _init_dhi)
            _init_corr = c_zero_f
            init_state = [_pro_m, c_zero_f]
            for _ in range_constexpr(D_CHUNKS):
                init_state.append(c_zero_v16f32)
            for _ks in range_constexpr(K_STEPS_QK):
                init_state.append(_init_kp_lo[_ks])
            for _ks in range_constexpr(K_STEPS_QK):
                init_state.append(_init_kp_hi[_ks])
            init_state.append(_init_plo_vec)
            init_state.append(_init_dhi_vec)
            init_state.append(_init_corr)
            init_state.append(arith.index(1))

            results = init_state

            # ---- Per-tile hot loop: tiles 1..N-1 (HK L320-471) ----
            _loop_start = arith.MinSIOp(arith.index(BLOCK_N), kv_upper).result

            for kv_tile, state in range(_loop_start, kv_upper, arith.index(BLOCK_N),
                                        init=init_state):
                _ci = 0
                m_running = state[_ci]; _ci += 1
                l_running = state[_ci]; _ci += 1
                o_accs = [state[_ci + i] for i in range_constexpr(D_CHUNKS)]; _ci += D_CHUNKS
                _kp_lo = [state[_ci + i] for i in range_constexpr(K_STEPS_QK)]; _ci += K_STEPS_QK
                _kp_hi = [state[_ci + i] for i in range_constexpr(K_STEPS_QK)]; _ci += K_STEPS_QK
                _plo_vec = state[_ci]; _ci += 1
                _dhi_vec = state[_ci]; _ci += 1
                _corr = state[_ci]; _ci += 1
                _buf = state[_ci]
                # Unpack v16f32 → individual f32 for element-wise ops
                _plo = [vector.extract(_plo_vec, static_position=[_r], dynamic_position=[]) for _r in range(16)]
                _dhi = [vector.extract(_dhi_vec, static_position=[_r], dynamic_position=[]) for _r in range(16)]

                _alt_buf = arith.index(1) - _buf

                # ==== Cluster 0: QK(k_packs) + finish_softmax(prev) (HK L322-339) ====
                s_acc_lo, s_acc_hi = _do_qk_gemm(_kp_lo, _kp_hi)
                # finish softmax: exp2 second half (HK L328/L400)
                _phi = []
                for _r in range_constexpr(16):
                    _phi.append(arith.ArithValue(_dhi[_r]).exp2(fastmath=fm_fast))
                # pending_scale: mul(norm_vec, scale_vec) (HK L329-331/L401-403)
                _lc = arith.MulFOp(_corr, l_running, fastmath=fm_fast).result
                # col_sum (HK L332/L404)
                _lsum = c_zero_f
                for _r in range_constexpr(16):
                    _lsum = arith.AddFOp(_lsum, _plo[_r], fastmath=fm_fast).result
                for _r in range_constexpr(16):
                    _lsum = arith.AddFOp(_lsum, _phi[_r], fastmath=fm_fast).result
                _ps = reduction_peer(_lsum)
                _ts = arith.AddFOp(_lsum, _ps, fastmath=fm_fast).result
                l_running = arith.AddFOp(_lc, _ts, fastmath=fm_fast).result
                # copy att to bf16 → p_packs (HK L333-334/L405-406)
                p_packs_lo, p_packs_hi = _pack_p_vals(_plo, _phi)
                _sched_barrier_exp_pairs(6, 3, 0)
                _sched_barrier_pairs(10, 5, 0)
                rocdl.sched_barrier(0)
                gpu.barrier()
                rocdl.sched_barrier(0)

                # ==== Cluster 1: K DMA(t+2→buf) + V read(alt→regs) (HK L341-350) ====
                _k_dma_tile = kv_tile + arith.index(2 * BLOCK_N)
                _has_k = arith.cmpi(arith.CmpIPredicate.slt, _k_dma_tile, kv_upper)
                _if_k = scf.IfOp(_has_k)
                with ir.InsertionPoint(_if_k.then_block):
                    coop_dma_k(_k_dma_tile, _buf)
                    scf.YieldOp([])
                # V read → regs (HK L345: load(v_reg, v_smem[0/1]))
                _v_packs = _preload_v_packs(v_buf_base(_alt_buf))
                # Extract QK result (causal mask applied in cluster 2 before softmax)
                s_raw_lo, s_raw_hi = _extract_s_raw(s_acc_lo, s_acc_hi)
                s_raw_lo, s_raw_hi = _apply_causal_mask(s_raw_lo, s_raw_hi, kv_tile)
                # HK L346-347: s_waitcnt lgkmcnt(0) vmcnt(4)
                rocdl.s_waitcnt(0)
                rocdl.sched_barrier(0)
                gpu.barrier()
                rocdl.sched_barrier(0)

                # ==== Cluster 2: PV + partial_softmax (HK L352-381) ====
                rocdl.s_setprio(1)
                o_accs = _do_pv_gemm_from_regs(_v_packs, p_packs_lo, p_packs_hi, o_accs)
                m_new = _compute_max(s_raw_lo, s_raw_hi, m_running)
                _cr_raw = arith.SubFOp(m_running, m_new, fastmath=fm_fast).result
                _cr_sc = arith.MulFOp(_cr_raw, c_sm_scale_log2e, fastmath=fm_fast).result
                _corr = arith.ArithValue(_cr_sc).exp2(fastmath=fm_fast)
                _corr_vec = vector.broadcast(v16f32_type, _corr)
                for _dc in range_constexpr(D_CHUNKS):
                    o_accs[_dc] = arith.MulFOp(o_accs[_dc], _corr_vec, fastmath=fm_fast).result
                m_running = m_new
                # sub_col + exp2 first half (HK L374-375)
                _sc = arith.MulFOp(c_sm_scale_log2e, m_new, fastmath=fm_fast).result
                _neg_sc = arith.SubFOp(c_zero_f, _sc, fastmath=fm_fast).result
                _plo = []
                for _r in range_constexpr(16):
                    _d = math_dialect.fma(s_raw_lo[_r], c_sm_scale_log2e, _neg_sc)
                    _plo.append(arith.ArithValue(_d).exp2(fastmath=fm_fast))
                _dhi = []
                for _r in range_constexpr(16):
                    _dhi.append(math_dialect.fma(s_raw_hi[_r], c_sm_scale_log2e, _neg_sc))
                _sched_barrier_pairs(6, 5, 0)
                _sched_barrier_exp_pairs(6, 3, 0)
                rocdl.s_setprio(0)
                rocdl.sched_barrier(0)
                gpu.barrier()
                rocdl.sched_barrier(0)

                # ==== Cluster 3: V DMA(t+1→alt) + K read(alt→k_packs) (HK L383-392) ====
                _v_dma_tile = kv_tile + arith.index(BLOCK_N)
                _has_v = arith.cmpi(arith.CmpIPredicate.slt, _v_dma_tile, kv_upper)
                _if_v = scf.IfOp(_has_v)
                with ir.InsertionPoint(_if_v.then_block):
                    coop_dma_v(_v_dma_tile, _alt_buf)
                    scf.YieldOp([])
                _kp_lo, _kp_hi = _read_all_k_packs(k_buf_base(_alt_buf))
                # HK L388-389: s_waitcnt lgkmcnt(0) vmcnt(4)
                rocdl.s_waitcnt(0)
                rocdl.sched_barrier(0)
                gpu.barrier()
                rocdl.sched_barrier(0)

                # ==== Yield all loop-carried state (packed) ====
                _plo_vec = vector.from_elements(v16f32_type, _plo)
                _dhi_vec = vector.from_elements(v16f32_type, _dhi)
                _yield = [m_running, l_running] + o_accs
                for _ks in range_constexpr(K_STEPS_QK):
                    _yield.append(_kp_lo[_ks])
                for _ks in range_constexpr(K_STEPS_QK):
                    _yield.append(_kp_hi[_ks])
                _yield.append(_plo_vec)
                _yield.append(_dhi_vec)
                _yield.append(_corr)
                _yield.append(_alt_buf)
                results = yield _yield

            # ---- Epilogue: finish last deferred softmax + PV (HK L634-691) ----
            _ci = 0
            _ci += 2  # skip m, l
            o_finals = [results[2 + dc] for dc in range_constexpr(D_CHUNKS)]
            _ci = 2 + D_CHUNKS + 2 * K_STEPS_QK
            _ep_plo_vec = results[_ci]; _ci += 1
            _ep_dhi_vec = results[_ci]; _ci += 1
            _ep_corr = results[_ci]; _ci += 1
            _ep_buf = results[_ci]
            _ep_plo = [vector.extract(_ep_plo_vec, static_position=[_r], dynamic_position=[]) for _r in range(16)]
            _ep_dhi = [vector.extract(_ep_dhi_vec, static_position=[_r], dynamic_position=[]) for _r in range(16)]
            _ep_alt = arith.index(1) - _ep_buf
            # finish softmax for last tile
            _ep_phi = []
            for _r in range_constexpr(16):
                _ep_phi.append(arith.ArithValue(_ep_dhi[_r]).exp2(fastmath=fm_fast))
            _ep_lsum = c_zero_f
            for _r in range_constexpr(16):
                _ep_lsum = arith.AddFOp(_ep_lsum, _ep_plo[_r], fastmath=fm_fast).result
            for _r in range_constexpr(16):
                _ep_lsum = arith.AddFOp(_ep_lsum, _ep_phi[_r], fastmath=fm_fast).result
            _ep_ps = reduction_peer(_ep_lsum)
            _ep_ts = arith.AddFOp(_ep_lsum, _ep_ps, fastmath=fm_fast).result
            _ep_lc = arith.MulFOp(_ep_corr, results[1], fastmath=fm_fast).result
            _final_l = arith.AddFOp(_ep_lc, _ep_ts, fastmath=fm_fast).result
            _ep_pp_lo, _ep_pp_hi = _pack_p_vals(_ep_plo, _ep_phi)
            # PV for last tile using V from v_buf[alt]
            rocdl.s_waitcnt(0)
            gpu.barrier()
            _ep_v_packs = _preload_v_packs(v_buf_base(_ep_alt))
            rocdl.s_setprio(1)
            o_finals = _do_pv_gemm_from_regs(_ep_v_packs, _ep_pp_lo, _ep_pp_hi, o_finals)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            gpu.barrier()

            # HK post-loop stagger (HK L678-680)
            if _hk_enable_stagger:
                _not_stagger = arith.cmpi(
                    arith.CmpIPredicate.ult, wave_id, arith.index(NUM_WAVES // 2))
                _stagger_if2 = scf.IfOp(_not_stagger)
                with ir.InsertionPoint(_stagger_if2.then_block):
                    gpu.barrier()
                    scf.YieldOp([])

        # ================================================================
        # Non-DMA-dbuf fallback paths (PREFETCH_3BUF or basic)
        # ================================================================
        else:
            init_args = [c_neg_inf, c_zero_f]
            for _ in range_constexpr(D_CHUNKS):
                init_args.append(c_zero_v16f32)

            for kv_block_start, inner_iter_args, loop_results in scf.for_(
                arith.index(0), kv_upper, arith.index(BLOCK_N_OUT),
                iter_args=init_args,
            ):
                m_running = inner_iter_args[0]
                l_running = inner_iter_args[1]
                o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]

                if ENABLE_PREFETCH_3BUF:
                    preload_k_count = min(NUM_PREFETCH_K, N_SUBTILES)
                    for pre_k in range_constexpr(preload_k_count):
                        pre_k_slot = CK_LDS_SEQ[pre_k % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                        pre_k_start = kv_block_start + pre_k * BLOCK_N
                        if ENABLE_DMA:
                            coop_dma_k(pre_k_start, pre_k_slot)
                        else:
                            coop_load_k(pre_k_start, pre_k_slot)
                    if ENABLE_DMA:
                        rocdl.s_waitcnt(0)
                    else:
                        rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                    gpu.barrier()

                for kv_sub in range_constexpr(N_SUBTILES):
                    kv_start = kv_block_start + kv_sub * BLOCK_N

                    if ENABLE_PREFETCH_3BUF:
                        k_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    else:
                        k_slot = 0
                        coop_load_k(kv_start, k_slot)
                        gpu.barrier()
                    k_base = k_buf_base(k_slot)

                    if not USE_HW_TR:
                        _v_vecs_prefetch = coop_load_v_global(kv_start)

                    s_acc_lo, s_acc_hi = _do_qk_gemm(*_read_all_k_packs(k_base))
                    rocdl.sched_barrier(0)

                    s_raw_lo, s_raw_hi = _extract_s_raw(s_acc_lo, s_acc_hi)
                    s_raw_lo, s_raw_hi = _apply_causal_mask(s_raw_lo, s_raw_hi, kv_start)

                    m_new = _compute_max(s_raw_lo, s_raw_hi, m_running)
                    p_vals_lo, p_vals_hi, m_running, l_running, o_accs = \
                        _compute_softmax_full(s_raw_lo, s_raw_hi, m_new, m_running, l_running, o_accs)
                    p_packs_lo, p_packs_hi = _pack_p_vals(p_vals_lo, p_vals_hi)

                    rocdl.sched_barrier(0)

                    if ENABLE_PREFETCH_3BUF:
                        v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                        v_base = v_buf_base(v_slot)
                        coop_load_v(kv_start, v_slot)
                        rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                        gpu.barrier()
                    elif ENABLE_DMA:
                        v_base = v_buf_base(0)
                        coop_dma_v(kv_start, 0)
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                    else:
                        v_base = v_buf_base(0)
                        _waitcnt_vm_n(0)
                        coop_store_v_lds(_v_vecs_prefetch, 0)
                        rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                        gpu.barrier()

                    o_accs = _do_pv_gemm(v_base, p_packs_lo, p_packs_hi, o_accs)

                    if ENABLE_PREFETCH_3BUF and (kv_sub + min(NUM_PREFETCH_K, N_SUBTILES)) < N_SUBTILES:
                        next_k_sub = kv_sub + min(NUM_PREFETCH_K, N_SUBTILES)
                        next_k_start = kv_block_start + next_k_sub * BLOCK_N
                        next_k_slot = CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                        if ENABLE_DMA:
                            coop_dma_k(next_k_start, next_k_slot)
                        else:
                            coop_load_k(next_k_start, next_k_slot)

                yield [m_running, l_running] + o_accs

            o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        # ---- Normalize and store O (skip OOB rows for partial Q tiles) ----
        if _use_dma_dbuf:
            l_final = _final_l
        else:
            l_final = loop_results[1]

        inv_l = arith.DivFOp(
            c_one_f,
            l_final,
            fastmath=fm_fast,
        ).result
        inv_l_vec = vector.broadcast(v16f32_type, inv_l)

        _o_guard = scf.IfOp(q_in_bounds, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = arith.MulFOp(
                    o_finals[dc],
                    inv_l_vec,
                    fastmath=fm_fast,
                ).result
                for r in range_constexpr(16):
                    o_val = vector.extract(
                        o_norm_vec,
                        static_position=[r],
                        dynamic_position=[],
                    )
                    o_f16 = arith.trunc_f(elem_type, o_val)

                    d_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                    d_col = arith.index(dc * D_CHUNK) + d_row_rel
                    o_global = global_idx(q_row, d_col)
                    _gep_store(o_f16, o_ptr, o_global)
            scf.YieldOp([])

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        launcher = flash_attn_func_kernel(Q, K, V, O, seq_len)

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32,
                            _wpe,
                        )
        if flat_work_group_size is not None:
            _fwgs = int(flat_work_group_size)
            if _fwgs >= 1:
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        if daz:
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"),
                ir.StringAttr.get("true"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"),
                ir.StringAttr.get("true"),
            ]))
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Best MI355X FMHA numbers so far were measured with ROCm/llvm-project
    # `felix/tune_fmha` at c8cf6da4367c010c7cbbb7789a9c4349e7407619.
    # Other LLVM revisions can compile/run this kernel, but usually leave a
    # few percent of peak throughput on the table.
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_func(*args, **kwargs)

    def _compile(Q, K, V, O, batch_size, seq_len, stream=None):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flash_attn_func, Q, K, V, O, batch_size, seq_len,
                fx.Stream(stream))

    _launch.compile = _compile

    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary
