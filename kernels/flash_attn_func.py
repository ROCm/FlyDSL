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


def _asm(stmt):
    """Emit inline asm that survives MLIR→LLVM lowering (rocdl ops get stripped)."""
    from flydsl._mlir.dialects import llvm as _llvm_d
    _void = ir.Type.parse("!llvm.void")
    _llvm_d.InlineAsmOp(_void, [], stmt, "", has_side_effects=True, is_align_stack=False)


def _waitcnt_lgkm0_vm_n(n):
    """Emit single s_waitcnt with lgkmcnt(0) + vmcnt(n) — HK pattern for clusters 1/3/5/7.
    Encodes both counters in one s_waitcnt instruction."""
    vm_lo = n & _VMCNT_LO_MASK
    vm_hi = ((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT
    lgkm_0 = 0 << 8  # lgkmcnt field at bits [12:8], 0 = wait for all
    expcnt_7 = 7 << 4  # expcnt field at bits [6:4], 7 = don't wait
    val = vm_lo | expcnt_7 | lgkm_0 | vm_hi
    rocdl.s_waitcnt(val)


# ---- Scheduling barrier constants (matching HipKitten masks) ----
MASK_MFMA = 0x008
MASK_VALU = 0x002
MASK_EXP  = 0x400

# ---- Lazy rescaling threshold (from flash-attention v3 / Dao-AILab) ----
RESCALE_THRESHOLD = 8.0



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
    else:
        PATH_TAG = "N128"
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

        # ---- f32 fast-math shorthand (like gemm's operator style) ----
        def _sub(a, b): return arith.SubFOp(a, b, fastmath=fm_fast).result
        def _mul(a, b): return arith.MulFOp(a, b, fastmath=fm_fast).result
        def _add(a, b): return arith.AddFOp(a, b, fastmath=fm_fast).result
        def _fmax(a, b): return arith.MaxNumFOp(a, b, fastmath=fm_fast).result
        def _exp2(v): return rocdl.exp2(ir.F32Type.get(), v)
        def _fma(a, b, c): return math_dialect.fma(a, b, c)

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


        def _cvt_pk_bf16_f32(a_f32, b_f32):
            """Pack 2 f32 → 1 i32 (bf16x2) via v_cvt_pk_bf16_f32 inline asm.
            HK pattern: 1 instruction vs 3 (and+lshr+or) for bitwise truncation."""
            _i32_ty = T.i32
            result = _llvm.InlineAsmOp(
                _i32_ty, [a_f32, b_f32],
                "v_cvt_pk_bf16_f32 $0, $1, $2", "=v,v,v",
                has_side_effects=False, is_align_stack=False).result
            return result

        def bf16_trunc_pack_v4(f32_vals):
            """Pack 4 f32 → v4bf16 via v_cvt_pk_bf16_f32."""
            _v2i32 = T.vec(2, T.i32)
            p0 = _cvt_pk_bf16_f32(f32_vals[0], f32_vals[1])
            p1 = _cvt_pk_bf16_f32(f32_vals[2], f32_vals[3])
            return vector.bitcast(v4f16_type, vector.from_elements(_v2i32, [p0, p1]))

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 → v8bf16 via v_cvt_pk_bf16_f32."""
            _v4i32 = T.vec(4, T.i32)
            pairs = [_cvt_pk_bf16_f32(f32_vals[j*2], f32_vals[j*2+1])
                     for j in range_constexpr(4)]
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

        # (Non-DMA cooperative load functions removed — DMA path only)

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
        c_neg_inf = arith.constant(float("-inf"), type=T.f32)
        c_zero_f = arith.constant(0.0, type=T.f32)
        c_one_f = arith.constant(1.0, type=T.f32)
        c_sm_scale_log2e = arith.constant(sm_scale * _LOG2E, type=T.f32)
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)
        c_rescale_threshold = arith.constant(RESCALE_THRESHOLD, type=T.f32)
        c_all_ones_i64 = arith.constant(-1, type=T.i64)
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

        # ---- K/V LDS index helpers ----
        k_hi_offset = K_SUB_N * K_STRIDE
        k_swz_mask = (lane_mod_32 & arith.index(0x7)) << arith.index(4)

        def _k_idx_lo_buf(ks, kb):
            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return kb + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

        def _k_idx_hi_buf(ks, kb):
            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            return kb + k_hi_offset + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

        _QK_PREFETCH_DEPTH = 2

        def _do_qk_gemm_pipelined(kb):
            kp_lo = [None] * K_STEPS_QK
            kp_hi = [None] * K_STEPS_QK
            for p in range_constexpr(_QK_PREFETCH_DEPTH):
                kp_lo[p] = vector.load_op(mfma_pack_type, lds_kv, [_k_idx_lo_buf(p, kb)])
                kp_hi[p] = vector.load_op(mfma_pack_type, lds_kv, [_k_idx_hi_buf(p, kb)])
            acc_lo = c_zero_v16f32
            acc_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                acc_lo = mfma_acc(kp_lo[ks], q_b_packs[ks], acc_lo)
                acc_hi = mfma_acc(kp_hi[ks], q_b_packs[ks], acc_hi)
                if ks + _QK_PREFETCH_DEPTH < K_STEPS_QK:
                    kp_lo[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                        mfma_pack_type, lds_kv, [_k_idx_lo_buf(ks + _QK_PREFETCH_DEPTH, kb)])
                    kp_hi[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                        mfma_pack_type, lds_kv, [_k_idx_hi_buf(ks + _QK_PREFETCH_DEPTH, kb)])
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
            kv_i32 = arith.index_cast(T.i32, kv_start)
            ld32_i32 = arith.index_cast(T.i32, lane_div_32)
            tile_needs_mask = arith.cmpi(
                arith.CmpIPredicate.ugt,
                kv_i32 + fx.Int32(BLOCK_N - 1),
                arith.index_cast(T.i32, q_start))
            # Store defaults into pre-allocated slots
            for r in range_constexpr(16):
                _llvm.StoreOp(s_raw_lo[r], _cond_slot_slo[r])
                _llvm.StoreOp(s_raw_hi[r], _cond_slot_shi[r])
            _mask_if = scf.IfOp(tile_needs_mask, [], has_else=False)
            with ir.InsertionPoint(_mask_if.then_block):
                for r in range_constexpr(16):
                    kv_col = kv_i32 + ld32_i32 * fx.Int32(4) + fx.Int32((r % 4) + (r // 4) * 8)
                    is_lo = arith.cmpi(arith.CmpIPredicate.ugt, kv_col, q_row_i32)
                    _llvm.StoreOp(arith.select(is_lo, c_neg_inf, s_raw_lo[r]), _cond_slot_slo[r])
                    is_hi = arith.cmpi(arith.CmpIPredicate.ugt, kv_col + fx.Int32(K_SUB_N), q_row_i32)
                    _llvm.StoreOp(arith.select(is_hi, c_neg_inf, s_raw_hi[r]), _cond_slot_shi[r])
                scf.YieldOp([])
            return ([_load_f32(_cond_slot_slo[r]) for r in range_constexpr(16)],
                    [_load_f32(_cond_slot_shi[r]) for r in range_constexpr(16)])

        def _compute_max(s_raw_lo, s_raw_hi, m_old):
            loc = s_raw_lo[0]
            for r in range_constexpr(15):
                loc = _fmax(loc, s_raw_lo[r + 1])
            for r in range_constexpr(16):
                loc = _fmax(loc, s_raw_hi[r])
            return _fmax(m_old, _fmax(loc, reduction_peer(loc)))

        def _compute_softmax_lazy(s_raw_lo, s_raw_hi, m_new, m_old, l_old, o_accs_in):
            """Lazy rescaling using alloca (no PHI copies for unchanged values)."""
            diff = _sub(m_new, m_old)
            ok_i1 = arith.CmpFOp(arith.CmpFPredicate.OLE, diff, c_rescale_threshold).result
            ballot_mask = rocdl.ballot(T.i64, ok_i1)
            all_ok = arith.cmpi(arith.CmpIPredicate.eq, ballot_mask, c_all_ones_i64)

            # Store defaults (no-rescale path)
            _llvm.StoreOp(m_old, _cond_slot_m)
            _llvm.StoreOp(l_old, _cond_slot_l)
            for dc in range_constexpr(D_CHUNKS):
                _llvm.StoreOp(o_accs_in[dc], _cond_slot_o[dc])

            # Only else-branch modifies slots (HK pattern)
            _rif = scf.IfOp(all_ok, [], has_else=True)
            with ir.InsertionPoint(_rif.then_block):
                scf.YieldOp([])
            with ir.InsertionPoint(_rif.else_block):
                _corr = _exp2(_mul(_sub(m_old, m_new), c_sm_scale_log2e))
                _cv = vector.broadcast(v16f32_type, _corr)
                _llvm.StoreOp(m_new, _cond_slot_m)
                _llvm.StoreOp(_mul(l_old, _corr), _cond_slot_l)
                for dc in range_constexpr(D_CHUNKS):
                    _llvm.StoreOp(_mul(o_accs_in[dc], _cv), _cond_slot_o[dc])
                scf.YieldOp([])

            m_eff = _load_f32(_cond_slot_m)
            l_eff = _load_f32(_cond_slot_l)
            o_out = [_load_v16f32(_cond_slot_o[dc]) for dc in range_constexpr(D_CHUNKS)]

            neg_sc = _sub(c_zero_f, _mul(c_sm_scale_log2e, m_eff))
            plo = []; phi = []; lsum = c_zero_f
            for r in range_constexpr(16):
                p = _exp2(_fma(s_raw_lo[r], c_sm_scale_log2e, neg_sc))
                plo.append(p); lsum = _add(lsum, p)
            for r in range_constexpr(16):
                p = _exp2(_fma(s_raw_hi[r], c_sm_scale_log2e, neg_sc))
                phi.append(p); lsum = _add(lsum, p)
            l_new = _add(l_eff, _add(lsum, reduction_peer(lsum)))
            return plo, phi, m_eff, l_new, o_out

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

        def _do_pv_gemm(v_base_arg, pp_lo, pp_hi, o_accs_in):
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
            return o_out

        # ---- Scheduling barrier helpers (HK sched_barrier_pairs/exp_pairs) ----
        def _sched_barrier_pairs(n_pairs, valu_cnt, sgid):
            for _ in range_constexpr(n_pairs):
                rocdl.sched_group_barrier(MASK_MFMA, 1, sgid)
                rocdl.sched_group_barrier(MASK_VALU, valu_cnt, sgid)

        def _sched_barrier_exp_pairs(n_pairs, exp_cnt, sgid):
            for _ in range_constexpr(n_pairs):
                rocdl.sched_group_barrier(MASK_MFMA, 1, sgid)
                rocdl.sched_group_barrier(MASK_EXP, exp_cnt, sgid)

        c_zero_i32 = arith.constant(0, type=T.i32)
        c_one_i32 = arith.constant(1, type=T.i32)

        # ---- Pre-loop: DMA K[0..1] + V[0..1] into double buffers ----
        coop_dma_k(arith.index(0), buf_id=0)
        coop_dma_k(arith.index(BLOCK_N), buf_id=1)
        coop_dma_v(arith.index(0), buf_id=0)
        coop_dma_v(arith.index(BLOCK_N), buf_id=1)
        rocdl.s_waitcnt(0)
        gpu.barrier()

        _kv_last = arith.MaxSIOp(
            arith.SubIOp(kv_upper, arith.index(BLOCK_N)).result,
            arith.index(0)).result

        # ---- Alloca-based loop state (matches C++ pattern: LLVM mem2reg promotes to registers) ----
        _one_i64 = arith.constant(1, type=T.i64)
        _ptr_ty = _llvm_ptr_ty()
        _f32_attr = ir.TypeAttr.get(T.f32)
        _i32_attr = ir.TypeAttr.get(T.i32)
        _v16f32_attr = ir.TypeAttr.get(v16f32_type)

        def _alloca_f32(init_val):
            slot = _llvm.AllocaOp(_ptr_ty, _one_i64, _f32_attr).result
            _llvm.StoreOp(init_val, slot)
            return slot

        def _alloca_i32(init_val):
            slot = _llvm.AllocaOp(_ptr_ty, _one_i64, _i32_attr).result
            _llvm.StoreOp(init_val, slot)
            return slot

        def _alloca_v16f32(init_val):
            slot = _llvm.AllocaOp(_ptr_ty, _one_i64, _v16f32_attr).result
            _llvm.StoreOp(init_val, slot)
            return slot

        def _load_f32(slot):
            return _llvm.LoadOp(T.f32, slot).result

        def _load_i32(slot):
            return _llvm.LoadOp(T.i32, slot).result

        def _load_v16f32(slot):
            return _llvm.LoadOp(v16f32_type, slot).result

        # Conditional state slots for alloca-based IfOp (avoids PHI copies)
        _cond_slot_m = _alloca_f32(c_zero_f)
        _cond_slot_l = _alloca_f32(c_zero_f)
        _cond_slot_o = [_alloca_v16f32(c_zero_v16f32) for _ in range_constexpr(D_CHUNKS)]
        _cond_slot_slo = [_alloca_f32(c_zero_f) for _ in range_constexpr(16)]
        _cond_slot_shi = [_alloca_f32(c_zero_f) for _ in range_constexpr(16)]

        # Loop state slots
        slot_m = _alloca_f32(c_neg_inf)
        slot_l = _alloca_f32(c_zero_f)
        slot_o = [_alloca_v16f32(c_zero_v16f32) for _ in range_constexpr(D_CHUNKS)]

        for kv_block_start in scf.for_(arith.index(0), kv_upper, arith.index(BLOCK_N_OUT)):
            m_running = _load_f32(slot_m)
            l_running = _load_f32(slot_l)
            o_accs = [_load_v16f32(slot_o[dc]) for dc in range_constexpr(D_CHUNKS)]

            for kv_sub in range_constexpr(N_SUBTILES):
                kv_start = kv_block_start + kv_sub * BLOCK_N
                _sgid = kv_sub * 4 + 1

                # ==== Cluster 0: QK + softmax ====
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(0)
                gpu.barrier()
                rocdl.sched_barrier(0)

                s_acc_lo, s_acc_hi = _do_qk_gemm_pipelined(k_buf_base(kv_sub))
                s_raw_lo, s_raw_hi = _extract_s_raw(s_acc_lo, s_acc_hi)
                s_raw_lo, s_raw_hi = _apply_causal_mask(s_raw_lo, s_raw_hi, kv_start)
                m_new = _compute_max(s_raw_lo, s_raw_hi, m_running)
                p_vals_lo, p_vals_hi, m_running, l_running, o_accs = \
                    _compute_softmax_lazy(s_raw_lo, s_raw_hi, m_new, m_running, l_running, o_accs)
                p_packs_lo, p_packs_hi = _pack_p_vals(p_vals_lo, p_vals_hi)

                _sched_barrier_exp_pairs(6, 3, _sgid)
                _sched_barrier_pairs(10, 5, _sgid)
                rocdl.sched_barrier(0)
                gpu.barrier()

                # ==== Cluster 1: K DMA (clamped) ====
                rocdl.sched_barrier(0)
                coop_dma_k(arith.MinSIOp(kv_start + BLOCK_N, _kv_last).result, 1 - kv_sub)
                _asm("s_waitcnt lgkmcnt(0)")
                _waitcnt_vm_n(NUM_DMA_K * 2)
                rocdl.sched_barrier(0)
                gpu.barrier()

                # ==== Cluster 2: PV ====
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
                o_accs = _do_pv_gemm(v_buf_base(kv_sub), p_packs_lo, p_packs_hi, o_accs)
                rocdl.s_setprio(0)
                rocdl.sched_barrier(0)
                gpu.barrier()

                # ==== Cluster 3: V DMA (clamped) ====
                rocdl.sched_barrier(0)
                coop_dma_v(arith.MinSIOp(kv_start + 2 * BLOCK_N, _kv_last).result, kv_sub)
                _asm("s_waitcnt lgkmcnt(0)")
                _waitcnt_vm_n(NUM_DMA_K * 2)
                rocdl.sched_barrier(0)

            _llvm.StoreOp(m_running, slot_m)
            _llvm.StoreOp(l_running, slot_l)
            for dc in range_constexpr(D_CHUNKS):
                _llvm.StoreOp(o_accs[dc], slot_o[dc])

        o_finals = [_load_v16f32(slot_o[dc]) for dc in range_constexpr(D_CHUNKS)]
        l_final = _load_f32(slot_l)

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
            "enable-post-misched": True,
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
