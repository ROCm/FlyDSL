# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""flash_attn_func kernel builder for FlyDSL.

- True MFMA32 remap: `mfma_f32_32x32x16bf16` / `mfma_f32_32x32x16f16` for both GEMM stages.
- Tile shape: BLOCK_M=256, BLOCK_N=64, 8 waves (512 threads).
- Per-wave Q rows: 32.
- GEMM1 uses `K @ Q^T` so S/P live in MFMA32 register layout.
- Online softmax over KV dimension is done in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS roundtrip.
- K and V use separate LDS regions with DMA-to-LDS prefetch and XOR swizzle.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (512,) -- 8 waves of 64 on AMD (wave64).

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


def build_flash_attn_func_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
):
    """Build the flash_attn_func launcher using the post-refactor FlyDSL API."""
    gpu_arch = get_hip_arch()

    BLOCK_M = 256
    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64
    if flat_work_group_size is None:
        flat_work_group_size = 512  # only flat_work_group_size=512 is supported
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 32
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
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else 1
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
    assert flat_work_group_size == 512, (
        f"Only flat_work_group_size=512 is supported, got {flat_work_group_size}"
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
            return arith.index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

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
                v_lds_byte_base = (lds_kv_base_idx
                                   + arith.index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2))
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
        # B operand uses j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K.
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

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = ENABLE_DMA and not ENABLE_PREFETCH_3BUF
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if _use_dma_dbuf:
            init_args.append(arith.index(0))
            coop_dma_k(arith.index(0), buf_id=0)

        for kv_block_start, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            kv_upper,
            arith.index(BLOCK_N_OUT),
            iter_args=init_args,
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [
                inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)
            ]
            _cur_buf_id = inner_iter_args[2 + D_CHUNKS] if _use_dma_dbuf else None
            preload_k_count = (
                NUM_PREFETCH_K if NUM_PREFETCH_K < N_SUBTILES else N_SUBTILES
            )

            if ENABLE_PREFETCH_3BUF:
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
                elif _use_dma_dbuf:
                    if kv_sub % 2 == 0:
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = arith.index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = arith.index(1) - _k_buf_id
                    if kv_sub + 1 < N_SUBTILES:
                        coop_dma_k(
                            kv_block_start + (kv_sub + 1) * BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + arith.index(BLOCK_N_OUT)
                        _has_next = arith.cmpi(
                            arith.CmpIPredicate.slt, _next_kv, kv_upper)
                        _if_dma = scf.IfOp(_has_next)
                        with ir.InsertionPoint(_if_dma.then_block):
                            coop_dma_k(_next_kv, _next_k_buf_id)
                            scf.YieldOp([])
                    rocdl.sched_barrier(0)
                    k_base = k_buf_base(_k_buf_id)
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if not _use_dma_dbuf:
                    k_base = k_buf_base(k_slot)

                if not USE_HW_TR or (not ENABLE_DMA and not ENABLE_PREFETCH_3BUF):
                    _v_vecs_prefetch = coop_load_v_global(kv_start)

                # ==== GEMM1: bulk-read all K packs, then pipeline MFMAs ====
                k_hi_offset = K_SUB_N * K_STRIDE
                # XOR swizzle: col ^ ((row & 0x7) << 4) avoids LDS bank conflicts
                k_swz_mask = (lane_mod_32 & arith.index(0x7)) << arith.index(4)

                def _k_idx_lo(ks):
                    col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

                def _k_idx_hi(ks):
                    col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return (k_base + k_hi_offset
                            + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask))

                _QK_PREFETCH_DEPTH = 2
                k_packs_lo = [None] * K_STEPS_QK
                k_packs_hi = [None] * K_STEPS_QK
                for p in range_constexpr(_QK_PREFETCH_DEPTH):
                    k_packs_lo[p] = vector.load_op(
                        mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                    k_packs_hi[p] = vector.load_op(
                        mfma_pack_type, lds_kv, [_k_idx_hi(p)])

                if ENABLE_DMA and not ENABLE_PREFETCH_3BUF:
                    coop_dma_v(kv_start, 0)
                    rocdl.sched_barrier(0)

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                for ks in range_constexpr(K_STEPS_QK):
                    s_acc_lo = mfma_acc(
                        k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                    s_acc_hi = mfma_acc(
                        k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                    if ks + _QK_PREFETCH_DEPTH < K_STEPS_QK:
                        k_packs_lo[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                            mfma_pack_type, lds_kv,
                            [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)])
                        k_packs_hi[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                            mfma_pack_type, lds_kv,
                            [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)])

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(vector.extract(
                        s_acc_lo, static_position=[r], dynamic_position=[]))
                    s_raw_hi.append(vector.extract(
                        s_acc_hi, static_position=[r], dynamic_position=[]))

                if CAUSAL:
                    kv_start_i32 = arith.index_cast(T.i32, kv_start)
                    lane_div_32_i32 = arith.index_cast(T.i32, lane_div_32)
                    q_start_i32 = arith.index_cast(T.i32, q_start)
                    max_kv_col_i32 = arith.AddIOp(
                        kv_start_i32,
                        arith.constant(BLOCK_N - 1, type=T.i32)).result
                    tile_needs_mask = arith.cmpi(
                        arith.CmpIPredicate.ugt, max_kv_col_i32, q_start_i32)
                    _mask_if = scf.IfOp(
                        tile_needs_mask, [T.f32] * 32, has_else=True)
                    with ir.InsertionPoint(_mask_if.then_block):
                        _m_lo = []
                        _m_hi = []
                        for r in range_constexpr(16):
                            # MFMA 32x32 register remap: 16 elements -> (row, col)
                            r_off_i32 = arith.constant(
                                (r % 4) + (r // 4) * 8, type=T.i32)
                            lane_off_i32 = arith.MulIOp(
                                lane_div_32_i32,
                                arith.constant(4, type=T.i32)).result
                            kv_col_lo = arith.AddIOp(
                                arith.AddIOp(
                                    kv_start_i32, lane_off_i32).result,
                                r_off_i32).result
                            is_masked_lo = arith.cmpi(
                                arith.CmpIPredicate.ugt,
                                kv_col_lo, q_row_i32)
                            _m_lo.append(arith.select(
                                is_masked_lo, c_neg_inf, s_raw_lo[r]))
                            kv_col_hi = arith.AddIOp(
                                kv_col_lo,
                                arith.constant(K_SUB_N, type=T.i32)).result
                            is_masked_hi = arith.cmpi(
                                arith.CmpIPredicate.ugt,
                                kv_col_hi, q_row_i32)
                            _m_hi.append(arith.select(
                                is_masked_hi, c_neg_inf, s_raw_hi[r]))
                        scf.YieldOp(_m_lo + _m_hi)
                    with ir.InsertionPoint(_mask_if.else_block):
                        scf.YieldOp(s_raw_lo + s_raw_hi)
                    s_raw_lo = [_mask_if.results[i] for i in range(16)]
                    s_raw_hi = [_mask_if.results[16 + i] for i in range(16)]

                _max_fm = {"fastmath": fm_fast}
                local_max = s_raw_lo[0]
                for r in range_constexpr(15):
                    local_max = arith.MaxNumFOp(local_max, s_raw_lo[r + 1], **_max_fm).result
                for r in range_constexpr(16):
                    local_max = arith.MaxNumFOp(local_max, s_raw_hi[r], **_max_fm).result
                peer_max = reduction_peer(local_max)
                row_max = arith.MaxNumFOp(local_max, peer_max, **_max_fm).result
                m_new_raw = arith.MaxNumFOp(m_running, row_max, **_max_fm).result

                diff_m_raw = arith.SubFOp(m_running, m_new_raw, fastmath=fm_fast).result
                diff_m_scaled = arith.MulFOp(diff_m_raw, c_sm_scale_log2e, fastmath=fm_fast).result
                corr = arith.ArithValue(diff_m_scaled).exp2(fastmath=fm_fast)

                scaled_max = arith.MulFOp(c_sm_scale_log2e, m_new_raw, fastmath=fm_fast).result
                neg_scaled_max = arith.SubFOp(c_zero_f, scaled_max, fastmath=fm_fast).result

                p_vals_lo = []
                p_vals_hi = []
                local_sum = c_zero_f
                for r in range_constexpr(16):
                    diff_lo = math_dialect.fma(s_raw_lo[r], c_sm_scale_log2e, neg_scaled_max)
                    p_lo = arith.ArithValue(diff_lo).exp2(fastmath=fm_fast)
                    p_vals_lo.append(p_lo)
                    local_sum = arith.AddFOp(local_sum, p_lo, fastmath=fm_fast).result
                for r in range_constexpr(16):
                    diff_hi = math_dialect.fma(s_raw_hi[r], c_sm_scale_log2e, neg_scaled_max)
                    p_hi = arith.ArithValue(diff_hi).exp2(fastmath=fm_fast)
                    p_vals_hi.append(p_hi)
                    local_sum = arith.AddFOp(local_sum, p_hi, fastmath=fm_fast).result

                peer_sum = reduction_peer(local_sum)
                tile_sum = arith.AddFOp(local_sum, peer_sum, fastmath=fm_fast).result
                l_corr = arith.MulFOp(corr, l_running, fastmath=fm_fast).result
                l_new = arith.AddFOp(l_corr, tile_sum, fastmath=fm_fast).result

                # ==== Rescale O accumulators ====
                corr_vec = vector.broadcast(v16f32_type, corr)
                if not USE_HW_TR:
                    o_accs[0] = arith.MulFOp(o_accs[0], corr_vec, fastmath=fm_fast).result
                else:
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = arith.MulFOp(o_accs[dc], corr_vec, fastmath=fm_fast).result

                if ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < N_SUBTILES:
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * BLOCK_N
                    next_k_slot = (
                        CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    )
                    if ENABLE_DMA:
                        coop_dma_k(next_k_start, next_k_slot)
                    else:
                        coop_load_k(next_k_start, next_k_slot)

                if ENABLE_PREFETCH_3BUF:
                    v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                    v_base = v_buf_base(v_slot)
                    coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif ENABLE_DMA:
                    v_base = v_buf_base(0)
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
                if dtype_str == "bf16" and not USE_K16:
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 4
                        p_packs_lo.append(bf16_trunc_pack_v4(
                            p_vals_lo[p_base:p_base+4]))
                        p_packs_hi.append(bf16_trunc_pack_v4(
                            p_vals_hi[p_base:p_base+4]))
                elif dtype_str == "bf16" and USE_K16:
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 8
                        p_packs_lo.append(bf16_trunc_pack_v8(
                            p_vals_lo[p_base:p_base+8]))
                        p_packs_hi.append(bf16_trunc_pack_v8(
                            p_vals_hi[p_base:p_base+8]))
                else:
                    p_f16_lo = []
                    p_f16_hi = []
                    for r in range_constexpr(16):
                        p_f16_lo.append(arith.trunc_f(elem_type, p_vals_lo[r]))
                        p_f16_hi.append(arith.trunc_f(elem_type, p_vals_hi[r]))

                    if USE_K16:
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 8
                            p_packs_lo.append(vector.from_elements(v8f16_type, [
                                p_f16_lo[p_base+0], p_f16_lo[p_base+1],
                                p_f16_lo[p_base+2], p_f16_lo[p_base+3],
                                p_f16_lo[p_base+4], p_f16_lo[p_base+5],
                                p_f16_lo[p_base+6], p_f16_lo[p_base+7]]))
                            p_packs_hi.append(vector.from_elements(v8f16_type, [
                                p_f16_hi[p_base+0], p_f16_hi[p_base+1],
                                p_f16_hi[p_base+2], p_f16_hi[p_base+3],
                                p_f16_hi[p_base+4], p_f16_hi[p_base+5],
                                p_f16_hi[p_base+6], p_f16_hi[p_base+7]]))
                    else:
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 4
                            p_packs_lo.append(vector.from_elements(v4f16_type, [
                                p_f16_lo[p_base], p_f16_lo[p_base+1],
                                p_f16_lo[p_base+2], p_f16_lo[p_base+3]]))
                            p_packs_hi.append(vector.from_elements(v4f16_type, [
                                p_f16_hi[p_base], p_f16_hi[p_base+1],
                                p_f16_hi[p_base+2], p_f16_hi[p_base+3]]))

                # Build flat (dc, pks) schedule for interleaved GEMM2.
                _steps = [(dc, pks)
                          for dc in range(D_CHUNKS)
                          for pks in range(PV_K_STEPS)]
                TOTAL_PV = len(_steps)

                def _read_v_pack(step_idx):
                    dc, pks = _steps[step_idx]
                    if USE_HW_TR:
                        d_col = (arith.index(dc * D_CHUNK)
                                 + tr_col_half * 16 + tr_col_sub * 4)
                        k_row = (arith.index(pks * PV_K_STEP)
                                 + lane_div_32 * 4 + tr_k_group)
                        _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                        lds_lo = v_base + k_row * V_STRIDE + _d_col_eff
                        lds_hi = lds_lo + arith.index(K_SUB_N * V_STRIDE)
                        if USE_K16:
                            vl_a = ds_read_tr_v4f16(lds_lo)
                            vl_b = ds_read_tr_v4f16(
                                lds_lo + arith.index(8 * V_STRIDE))
                            vl = vector.shuffle(
                                vl_a, vl_b, [0, 1, 2, 3, 4, 5, 6, 7])
                            vh_a = ds_read_tr_v4f16(lds_hi)
                            vh_b = ds_read_tr_v4f16(
                                lds_hi + arith.index(8 * V_STRIDE))
                            vh = vector.shuffle(
                                vh_a, vh_b, [0, 1, 2, 3, 4, 5, 6, 7])
                        else:
                            vl = ds_read_tr_v4f16(lds_lo)
                            vh = ds_read_tr_v4f16(lds_hi)
                    else:
                        d_pos = arith.index(dc * D_CHUNK) + lane_mod_32
                        k_base = arith.index(pks * PV_K_STEP) + lane_div_32 * 4
                        v_lo_idx = v_base + d_pos * VT_STRIDE + k_base
                        v_hi_idx = v_lo_idx + arith.index(K_SUB_N)
                        vl = vector.load(v4f16_type, lds_kv, [v_lo_idx])
                        vh = vector.load(v4f16_type, lds_kv, [v_hi_idx])
                    return vl, vh

                # Pre-read V for the first step.
                v_lo_cur, v_hi_cur = _read_v_pack(0)

                # ==== GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi ====
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    if si + 1 < TOTAL_PV:
                        v_lo_nxt, v_hi_nxt = _read_v_pack(si + 1)
                    o_accs[dc] = mfma_acc(
                        v_lo_cur, p_packs_lo[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(
                        v_hi_cur, p_packs_hi[pks], o_accs[dc])
                    if not USE_HW_TR and dc == 0 and pks < D_CHUNKS - 1:
                        o_accs[pks + 1] = arith.MulFOp(
                            o_accs[pks + 1], corr_vec, fastmath=fm_fast,
                        ).result
                    if si + 1 < TOTAL_PV:
                        v_lo_cur = v_lo_nxt
                        v_hi_cur = v_hi_nxt

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if _use_dma_dbuf:
                if N_SUBTILES % 2 == 1:
                    _yield_args.append(arith.index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            yield _yield_args

        # ---- Normalize and store O (skip OOB rows for partial Q tiles) ----
        l_final = loop_results[1]
        o_finals = [
            loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)
        ]

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
