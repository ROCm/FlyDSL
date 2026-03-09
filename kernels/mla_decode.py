"""MLA (Multi-Latent Attention) decode kernel for FlyDSL.

Optimized for short query sequences (seqlen_q <= 4) with GQA.
Implements absorbed MLA where the KV cache stores [c_kv || k_rope] per token.

Key design:
- MFMA 16x16x16f16 for both GEMM stages.
- 4 waves per workgroup, each wave handles one seqlen_q position.
  Each MFMA tile computes 16 Q heads simultaneously (lane_mod_16 = head).
- No if-else boundary checks for KV loads (assumes seqlen_kv % BLOCK_N == 0).
- Online softmax in registers; P kept in registers for direct GEMM2 feed.
- Three LDS buffers: two for K (double-buffered prefetch), one for V^T.
  V is NOT reloaded from global memory — it is read directly from the
  current K buffer in LDS (since c_kv serves as both K_nope and V),
  transposed in registers, and written to the V^T buffer.
- BLOCK_N=16 to fit all three buffers within 64 KB LDS.
- Paged KV cache via block_table: logical KV positions are mapped to
  physical blocks through a per-batch block table (i32 indices).

GEMM1: S = K @ Q^T   using HEAD_DIM_QK = kv_lora_rank + qk_rope_head_dim
GEMM2: O^T = V^T @ P  using HEAD_DIM_V  = kv_lora_rank

Layout (1D flattened):
  Q       : [batch, seqlen_q,  num_q_heads,  HEAD_DIM_QK]
  KV      : [num_physical_blocks, page_block_size, num_kv_heads, HEAD_DIM_QK]
  Mid_O   : [batch*seqlen_q, num_kv_splits, num_q_heads, HEAD_DIM_V]  (fp32)
  Mid_lse : [batch*seqlen_q, num_kv_splits, num_q_heads]              (fp32)
  block_table : [batch, max_num_blocks]   (i32 physical block ids)

KV split mode:
  The KV sequence is partitioned into num_kv_splits contiguous chunks.
  Each split produces normalised partial output (fp32) and log-sum-exp
  (lse = m + log(l)), compatible with the stage-2 combine kernel that
  merges splits via online log-sum-exp.

Grid:  (batch, num_head_groups, num_kv_splits)  -- num_head_groups = nhead/16
Block: (256,) -- 4 waves of 64, each wave handles one seqlen_q row

LDS layout (f16 element offsets):
  K_buf_0 : [0,                     K_BUF_ELEMS)
  K_buf_1 : [K_BUF_ELEMS,           2*K_BUF_ELEMS)
  V^T_buf : [2*K_BUF_ELEMS,         2*K_BUF_ELEMS + VT_BUF_ELEMS)

Requires: kv_lora_rank % 16 == 0, qk_rope_head_dim % 16 == 0,
          GQA group size >= 4, seqlen_kv % BLOCK_N == 0,
          page_block_size % BLOCK_N == 0, page_block_size <= 64.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects import math as _math
from flydsl._mlir.dialects import arith as _std_arith

from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

def _vt_swizzle(f16_idx):
    """XOR swizzle on VT f16 element index to avoid LDS bank conflicts.

    Maps byte-address bits [9:7] (row index) into bits [6:4] (bank group).
    Preserves 16-byte (8×f16) contiguity required by ds_read_b128.
    """
    raw = _raw(f16_idx) if not isinstance(f16_idx, ir.Value) else f16_idx
    i32_ty = ir.IntegerType.get_signless(32)
    idx32 = _std_arith.IndexCastOp(i32_ty, raw).result
    c3 = _std_arith.ConstantOp(i32_ty, ir.IntegerAttr.get(i32_ty, 3)).result
    c0x38 = _std_arith.ConstantOp(i32_ty, ir.IntegerAttr.get(i32_ty, 0x38)).result
    shifted = _std_arith.ShRUIOp(idx32, c3).result
    masked = _std_arith.AndIOp(shifted, c0x38).result
    swizzled = _std_arith.XOrIOp(idx32, masked).result
    return _std_arith.IndexCastOp(ir.IndexType.get(), swizzled).result


def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x).

    Bitfield: vmcnt[3:0]=bits[3:0], expcnt[2:0]=bits[6:4],
              lgkmcnt[5:0]=bits[13:8], vmcnt[5:4]=bits[15:14].
    """
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm.

    Bypasses LLVM SIInsertWaitcnts which would insert a conservative
    s_waitcnt vmcnt(0) lgkmcnt(0) before every S_BARRIER MI.
    """
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


_LDS_PTR_TYPE = None
def _inttoptr_lds(i64_val):
    """Convert i64 scalar to !llvm.ptr<3> (LDS pointer)."""
    global _LDS_PTR_TYPE
    if _LDS_PTR_TYPE is None:
        _LDS_PTR_TYPE = ir.Type.parse("!llvm.ptr<3>")
    return llvm.inttoptr(_LDS_PTR_TYPE, i64_val)


def _get_element_ptr(
    base_ptr,
    byte_offset=None,
    static_byte_offset=0,
    elem_type=None,
):
    """GEP-based pointer arithmetic that preserves provenance."""
    _GEP_DYN = -(2**31)

    if isinstance(base_ptr, ir.Value):
        raw_ptr = base_ptr
    else:
        raw_ptr = _raw(base_ptr)
    if elem_type is None:
        elem_type = ir.IntegerType.get_signless(8)

    if byte_offset is None:
        return llvm.GEPOp(
            raw_ptr.type, raw_ptr, [], [int(static_byte_offset)],
            elem_type, None,
        ).result
    elif isinstance(byte_offset, int):
        return llvm.GEPOp(
            raw_ptr.type, raw_ptr, [],
            [int(byte_offset) + int(static_byte_offset)],
            elem_type, None,
        ).result
    else:
        offset_val = _raw(byte_offset) if not isinstance(byte_offset, ir.Value) else byte_offset
        if isinstance(offset_val.type, ir.IndexType):
            i64_type = ir.IntegerType.get_signless(64)
            offset_val = _std_arith.IndexCastOp(i64_type, offset_val).result

        if static_byte_offset != 0:
            static_attr = ir.IntegerAttr.get(offset_val.type, int(static_byte_offset))
            static_const = _std_arith.ConstantOp(offset_val.type, static_attr).result
            offset_val = _std_arith.AddIOp(offset_val, static_const).result

        return llvm.GEPOp(
            raw_ptr.type, raw_ptr, [offset_val], [_GEP_DYN],
            elem_type, None,
        ).result


def _lds_load_prefer_agpr(byte_addr_index, vec_type, static_byte_offset=0):
    """LDS load with AGPR allocation hint via the nontemporal flag.

    The nontemporal attribute is a hardware no-op for DS (LDS) instructions
    (they have no cache-control bits).  Our custom LLVM backend pass
    (AMDGPUPreferAGPRForDSRead) detects the flag on DS loads and sets an
    AGPR register-allocation hint on the destination virtual register.
    """
    raw_addr = _raw(byte_addr_index) if not isinstance(byte_addr_index, ir.Value) else byte_addr_index
    i64_type = ir.IntegerType.get_signless(64)
    addr_i64 = _std_arith.IndexCastOp(i64_type, raw_addr).result
    lds_ptr = _inttoptr_lds(addr_i64)
    if static_byte_offset != 0:
        lds_ptr = _get_element_ptr(
            lds_ptr, static_byte_offset=static_byte_offset
        )
    return llvm.LoadOp(vec_type, lds_ptr, alignment=16, nontemporal=True).result


KERNEL_NAME = "mla_decode_kernel"


def _set_mfma_vgpr_form():
    """Force MFMA to use ACC_CD=0 (D/C in ArchVGPR) via LLVM cl::opt."""
    import ctypes
    import os
    lib_dir = os.path.dirname(
        __import__('flydsl._mlir._mlir_libs', fromlist=['_mlir_libs']).__file__
    )
    lib_name = 'libFlyPythonCAPI.so'
    lib_path = os.path.join(lib_dir, lib_name)
    if not os.path.exists(lib_path):
        lib_name = 'libFlirPythonCAPI.so'
        lib_path = os.path.join(lib_dir, lib_name)
    lib = ctypes.CDLL(lib_path)
    parse_fn = lib.LLVMParseCommandLineOptions
    parse_fn.restype = None
    parse_fn.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_char_p,
    ]
    argv = [
        b'mlir',
        b'-amdgpu-mfma-vgpr-form',
        b'--amdgpu-schedule-metric-bias=0',
        b'--enable-deferred-spilling',
    ]
    argv_arr = (ctypes.c_char_p * len(argv))(*argv)
    parse_fn(len(argv), argv_arr, None)


_set_mfma_vgpr_form()


def _index_cast_to_i32(value):
    """Cast index/ArithValue to i32."""
    raw = _raw(value) if not isinstance(value, ir.Value) else value
    i32_type = ir.IntegerType.get_signless(32)
    return _std_arith.IndexCastOp(i32_type, raw).result


def _math_exp2(val, fastmath=None):
    """Wrap math.Exp2Op, returning .result."""
    kw = {}
    if fastmath is not None:
        kw["fastmath"] = fastmath
    return _math.Exp2Op(_raw(val), **kw).result


def _math_log(val, fastmath=None):
    """Wrap math.LogOp, returning .result."""
    kw = {}
    if fastmath is not None:
        kw["fastmath"] = fastmath
    return _math.LogOp(_raw(val), **kw).result


def compile_mla_decode(
    num_q_heads=16,
    num_kv_heads=1,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    page_block_size=64,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
):
    gpu_arch = get_hip_arch()

    BLOCK_N = 16
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    HEADS_PER_WAVE = 16

    NUM_Q_HEADS = num_q_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM_QK = kv_lora_rank + qk_rope_head_dim
    HEAD_DIM_V = kv_lora_rank
    GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
    NUM_HEAD_GROUPS = NUM_Q_HEADS // HEADS_PER_WAVE
    CAUSAL = causal
    PAGE_BLOCK_SIZE = page_block_size

    assert NUM_Q_HEADS % HEADS_PER_WAVE == 0
    assert GQA_GROUP >= HEADS_PER_WAVE
    assert HEAD_DIM_V % 16 == 0
    assert qk_rope_head_dim % 16 == 0
    assert dtype_str == "f16"
    assert PAGE_BLOCK_SIZE % BLOCK_N == 0, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be a multiple of BLOCK_N ({BLOCK_N})"
    )
    assert PAGE_BLOCK_SIZE <= 64, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be <= 64"
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    K_STEPS = HEAD_DIM_QK // 32
    K_PREFETCH = 8
    K_REMAINING = K_STEPS - K_PREFETCH
    N_KV_SUBTILES = BLOCK_N // 16  # 1
    D_CHUNKS = HEAD_DIM_V // 16
    PV_K_STEPS = N_KV_SUBTILES     # 1

    RESHAPE_CPB = 4
    RESHAPE_BATCHES = D_CHUNKS // RESHAPE_CPB
    RESHAPE_ROW = RESHAPE_CPB * 16 + 4
    RESHAPE_WAVE = 16 * RESHAPE_ROW
    RESHAPE_TOTAL = NUM_WAVES * RESHAPE_WAVE

    STRIDE_TOKEN_Q = NUM_Q_HEADS * HEAD_DIM_QK
    STRIDE_TOKEN_KV = NUM_KV_HEADS * HEAD_DIM_QK
    STRIDE_PAGE = PAGE_BLOCK_SIZE * STRIDE_TOKEN_KV

    CKV_CHUNK_COLS = 64
    CKV_NUM_CHUNKS = HEAD_DIM_QK // CKV_CHUNK_COLS
    assert HEAD_DIM_QK % CKV_CHUNK_COLS == 0
    CKV_CHUNK_BYTES = CKV_CHUNK_COLS * 2 * 2
    CKV_HALF_BYTES = CKV_NUM_CHUNKS * CKV_CHUNK_BYTES
    CKV_PAD_BYTES = 32
    CKV_WAVE_BYTES = CKV_HALF_BYTES * 2 + CKV_PAD_BYTES
    CKV_HALF_F16 = CKV_HALF_BYTES // 2
    CKV_WAVE_F16 = CKV_WAVE_BYTES // 2
    CKV_CHUNK_F16 = CKV_CHUNK_BYTES // 2
    CKV_TOKEN_F16 = CKV_CHUNK_COLS
    K_SEQ_STRIDE = NUM_WAVES
    K_PAIRS_PER_WAVE = 2
    K_SEQS_PER_WAVE = K_PAIRS_PER_WAVE * 2
    COOP_K_VMEM_OPS = CKV_NUM_CHUNKS * K_PAIRS_PER_WAVE

    _GEMM1_HALF = K_STEPS - K_PREFETCH
    K_LO_CHUNKS = min(5, _GEMM1_HALF // K_PAIRS_PER_WAVE)
    K_HI0_CHUNKS = min(2, CKV_NUM_CHUNKS - K_LO_CHUNKS)
    K_HI1_CHUNKS = CKV_NUM_CHUNKS - K_LO_CHUNKS - K_HI0_CHUNKS
    K_LO_OPS = K_LO_CHUNKS * K_PAIRS_PER_WAVE
    K_HI0_OPS = K_HI0_CHUNKS * K_PAIRS_PER_WAVE
    K_HI1_OPS = K_HI1_CHUNKS * K_PAIRS_PER_WAVE

    K_STEP_OFFSETS_F16 = [
        (ks // 2) * CKV_CHUNK_F16 + (ks % 2) * 32
        for ks in range(K_STEPS)
    ]

    VT_STRIDE = (BLOCK_N // 4) * 8

    K_BUF_ELEMS = CKV_WAVE_BYTES * NUM_WAVES // 2
    _VT_DATA_ELEMS = (HEAD_DIM_V // 2) * VT_STRIDE
    _VT_WRITE_SPAN = 128 * 2 * VT_STRIDE
    VT_BUF_ELEMS = max(_VT_DATA_ELEMS, _VT_WRITE_SPAN)
    VT_BUF_OFFSET = 2 * K_BUF_ELEMS
    KV_LDS_SIZE = 2 * K_BUF_ELEMS + VT_BUF_ELEMS

    MAX_SEQLEN_Q = 4
    Q_TILE_DIM = 64
    Q_TILE_ROWS = MAX_SEQLEN_Q * HEADS_PER_WAVE
    Q_DIM_TILES = HEAD_DIM_QK // Q_TILE_DIM
    Q_TILE_ELEMS = Q_TILE_ROWS * Q_TILE_DIM
    assert HEAD_DIM_QK % Q_TILE_DIM == 0, (
        f"HEAD_DIM_QK ({HEAD_DIM_QK}) must be a multiple of Q_TILE_DIM ({Q_TILE_DIM})"
    )
    Q_CALLS_PER_WAVE = HEADS_PER_WAVE // 2
    Q_MAX_BATCH = KV_LDS_SIZE // Q_TILE_ELEMS

    LDS_SIZE = max(KV_LDS_SIZE, Q_TILE_ELEMS)

    assert HEAD_DIM_V % 16 == 0 and HEAD_DIM_V <= BLOCK_SIZE * 2

    _KLOAD_ASM = None
    _KLOAD_CONSTRAINTS = None

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    allocator_vt = SmemAllocator(None, arch=gpu_arch, global_sym_name="smemvt")
    allocator_red = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_red")

    # ── LDS sizing (pure Python, no MLIR ops) ──
    elem_bytes = 2  # f16
    lds_k_cur_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_k_cur_offset + K_BUF_ELEMS * elem_bytes

    lds_k_next_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_k_next_offset + K_BUF_ELEMS * elem_bytes

    # 64-byte pad shifts VT from bank 0 to bank 16, separating it from
    # K buffers (also bank-0 aligned) to reduce ds_write/ds_read cross-traffic.
    VT_BANK_PAD = 64
    allocator_vt.ptr += VT_BANK_PAD
    lds_vt_offset = allocator_vt._align(allocator_vt.ptr, 16)
    allocator_vt.ptr = lds_vt_offset + VT_BUF_ELEMS * elem_bytes

    lds_red_offset = allocator_red._align(allocator_red.ptr, 16)
    allocator_red.ptr = lds_red_offset + 256 * 4  # 256 x f32

    fm_fast = _std_arith.FastMathFlags.fast

    _Q_BATCHES = []
    _d = 0
    while _d < Q_DIM_TILES:
        _bs = min(Q_DIM_TILES - _d, Q_MAX_BATCH)
        _Q_BATCHES.append((_d, _bs))
        _d += _bs

    # ── Kernel function ──
    @flyc.kernel
    def mla_decode_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        Mid_O: fx.Tensor,
        Mid_lse: fx.Tensor,
        block_table: fx.Tensor,
        batch_size: fx.Int32,
        seqlen_q: fx.Int32,
        kv_indptr: fx.Tensor,
        max_num_blocks: fx.Int32,
        num_kv_splits: fx.Int32,
    ):
        compute_type = T.f32
        elem_type = T.f16

        v4f16_type = ir.VectorType.get([4], elem_type)
        v4f32_type = ir.VectorType.get([4], compute_type)
        v2f32_type = ir.VectorType.get([2], compute_type)
        v8f16_type = ir.VectorType.get([8], elem_type)
        i32_type = ir.IntegerType.get_signless(32)
        v2f16_type = ir.VectorType.get([2], elem_type)
        v1i32_type = ir.VectorType.get([1], i32_type)
        i64_type = ir.IntegerType.get_signless(64)

        batch_size_v = arith.index_cast(T.index, batch_size.ir_value())
        seqlen_q_v = arith.index_cast(T.index, seqlen_q.ir_value())
        max_num_blocks_v = arith.index_cast(T.index, max_num_blocks.ir_value())
        num_kv_splits_v_raw = arith.index_cast(T.index, num_kv_splits.ir_value())

        base_ptr = allocator_pong.get_base()
        base_ptr1 = allocator_ping.get_base()
        base_ptr_vt = allocator_vt.get_base()
        base_ptr_red = allocator_red.get_base()

        lds_k_cur_buf = SmemPtr(
            base_ptr, lds_k_cur_offset, T.f16, shape=(K_BUF_ELEMS,)
        ).get()
        lds_k_next_buf = SmemPtr(
            base_ptr1, lds_k_next_offset, T.f16, shape=(K_BUF_ELEMS,)
        ).get()
        lds_vt = SmemPtr(
            base_ptr_vt, lds_vt_offset, T.f16, shape=(VT_BUF_ELEMS,)
        ).get()
        lds_red = SmemPtr(
            base_ptr_red, lds_red_offset, T.f32, shape=(256,)
        ).get()
        lds_reshape = SmemPtr(
            base_ptr, lds_k_cur_offset, T.f32, shape=(RESHAPE_TOTAL,)
        ).get()

        # ---- Thread / block indices ----
        batch_idx = gpu.block_id("x")
        kv_indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr)
        batch_off_i32 = _index_cast_to_i32(batch_idx)
        batch_off_plus1_i32 = _std_arith.AddIOp(
            _raw(batch_off_i32),
            _raw(arith.constant(1, type=T.i32)),
        ).result
        kv_start_i32 = buffer_ops.buffer_load(
            kv_indptr_rsrc, batch_off_i32,
            vec_width=1, dtype=ir.IntegerType.get_signless(32),
        )
        kv_end_i32 = buffer_ops.buffer_load(
            kv_indptr_rsrc, batch_off_plus1_i32,
            vec_width=1, dtype=ir.IntegerType.get_signless(32),
        )
        kv_len_i32 = _std_arith.SubIOp(kv_end_i32, kv_start_i32).result
        seqlen_kv_v = arith.index_cast(T.index, kv_len_i32)
        head_group_idx = gpu.block_id("y")
        split_id = gpu.block_id("z")
        tid = gpu.thread_id("x")

        wave_id = tid / arith.index(WARP_SIZE)
        lane = tid % arith.index(WARP_SIZE)

        lane_mod_16 = lane % arith.index(16)
        lane_div_16 = lane / arith.index(16)
        lane_div_8 = lane / arith.index(8)
        lane_mod_8 = lane % arith.index(8)

        k_seq_wave = lane_mod_16 % arith.index(NUM_WAVES)
        k_seq_group = lane_mod_16 / arith.index(NUM_WAVES)
        k_group = k_seq_group / arith.index(2)
        k_token = k_seq_group % arith.index(2)
        k_read_base = (
            k_seq_wave * arith.index(CKV_WAVE_F16)
            + k_group * arith.index(CKV_HALF_F16)
            + k_token * arith.index(CKV_TOKEN_F16)
            + lane_div_16 * arith.index(8)
        )

        lane_mod_16_x2 = lane_mod_16 * arith.index(2)

        q_head_idx = head_group_idx * arith.index(HEADS_PER_WAVE) + lane_mod_16
        q_head_base = head_group_idx * arith.index(HEADS_PER_WAVE)
        kv_head_idx = q_head_base / arith.index(GQA_GROUP)

        kv_head_offset = kv_head_idx * arith.index(HEAD_DIM_QK)

        # ---- KV split range ----
        num_kv_splits_v = num_kv_splits_v_raw
        total_kv_tiles = (seqlen_kv_v + arith.index(BLOCK_N - 1)) / arith.index(BLOCK_N)
        tiles_nkv_m1 = num_kv_splits_v - arith.index(1)
        tiles_plus = total_kv_tiles + tiles_nkv_m1
        tiles_per_split = tiles_plus / num_kv_splits_v
        kv_per_split = tiles_per_split * arith.index(BLOCK_N)
        kv_split_start = split_id * kv_per_split
        _kv_split_end_raw = kv_split_start + kv_per_split
        _aligned_seqlen = total_kv_tiles * arith.index(BLOCK_N)
        kv_split_end = _std_arith.MinUIOp(_raw(_kv_split_end_raw), _raw(_aligned_seqlen)).result

        # ---- Mid_O / Mid_lse strides ----
        stride_mid_o_token = num_kv_splits_v * arith.index(NUM_Q_HEADS * HEAD_DIM_V)
        stride_mid_lse_token = num_kv_splits_v * arith.index(NUM_Q_HEADS)

        # ---- V^T transpose decomposition ----
        tid_mod_128 = tid % arith.index(128)
        tid_div_128 = tid / arith.index(128)
        vt_dim_quad = tid_mod_128 * arith.index(4)
        c_perm_lo = arith.constant(0x05040100, type=i32_type)
        c_perm_hi = arith.constant(0x07060302, type=i32_type)

        # ---- Global index helpers ----
        def q_global_idx(q_row, d_col):
            token = batch_idx * seqlen_q_v + q_row
            return (
                token * arith.index(STRIDE_TOKEN_Q)
                + q_head_idx * arith.index(HEAD_DIM_QK)
                + d_col
            )

        def mid_o_global_idx(q_row, d_col):
            total_q = batch_idx * seqlen_q_v + q_row
            return (
                total_q * stride_mid_o_token
                + split_id * arith.index(NUM_Q_HEADS * HEAD_DIM_V)
                + q_head_idx * arith.index(HEAD_DIM_V)
                + d_col
            )

        def mid_lse_global_idx(q_row):
            total_q = batch_idx * seqlen_q_v + q_row
            return (
                total_q * stride_mid_lse_token
                + split_id * arith.index(NUM_Q_HEADS)
                + q_head_idx
            )

        # ---- Block table lookup ----
        max_log_block = max_num_blocks_v - arith.index(1)

        bt_rsrc = buffer_ops.create_buffer_resource(block_table)

        def lookup_page(kv_abs_pos):
            log_block = kv_abs_pos / arith.index(PAGE_BLOCK_SIZE)
            log_block = _std_arith.MinUIOp(_raw(log_block), _raw(max_log_block)).result
            p_off = kv_abs_pos % arith.index(PAGE_BLOCK_SIZE)
            bt_idx = batch_idx * max_num_blocks_v + log_block
            bt_off_i32 = _index_cast_to_i32(bt_idx)
            phys_i32 = buffer_ops.buffer_load(
                bt_rsrc, bt_off_i32,
                vec_width=1, dtype=ir.IntegerType.get_signless(32),
            )
            phys = arith.index_cast(T.index, phys_i32)
            page_base = phys * arith.index(STRIDE_PAGE)
            return page_base, p_off
        _bt_aux = arith.constant(0, type=T.i32)
        _bt_base_i32 = _index_cast_to_i32(
            batch_idx * max_num_blocks_v
        )
        _c6_i32 = arith.constant(6, type=T.i32)
        _c2_i32 = arith.constant(2, type=T.i32)
        _bt_soff = _std_arith.ShLIOp(
            _raw(_bt_base_i32),
            _raw(_c2_i32),
        ).result
        _c_pbs_mask_i32 = arith.constant(
            PAGE_BLOCK_SIZE - 1, type=T.i32
        )

        def lookup_page_issue(kv_abs_pos, clamp=True,
                               pos_i32=None):
            if clamp:
                p_off = kv_abs_pos % arith.index(PAGE_BLOCK_SIZE)
                log_block = kv_abs_pos / arith.index(PAGE_BLOCK_SIZE)
                log_block = _std_arith.MinUIOp(
                    _raw(log_block), _raw(max_log_block)
                ).result
                bt_byte_off = _index_cast_to_i32(
                    log_block * arith.index(4)
                )
            else:
                if pos_i32 is None:
                    pos_i32 = _index_cast_to_i32(
                        kv_abs_pos)
                _and_res = _std_arith.AndIOp(
                    _raw(pos_i32),
                    _raw(_c_pbs_mask_i32),
                ).result
                p_off = arith.index_cast(T.index, _and_res)
                log_block_i32 = _std_arith.ShRUIOp(
                    _raw(pos_i32),
                    _raw(_c6_i32),
                ).result
                bt_byte_off = _std_arith.ShLIOp(
                    _raw(log_block_i32),
                    _raw(_c2_i32),
                ).result
            phys_i32 = rocdl.raw_ptr_buffer_load(
                i32_type, bt_rsrc, bt_byte_off,
                _raw(_bt_soff), _raw(_bt_aux),
            )
            return phys_i32, p_off

        def lookup_page_resolve(phys_i32):
            phys = arith.index_cast(T.index, phys_i32)
            return phys * arith.index(STRIDE_PAGE)

        c_soff_zero = arith.constant(0, type=T.i32)

        K_COL_OFFSETS = [
            arith.constant(j * 128, type=T.i32)
            for j in range(CKV_NUM_CHUNKS)
        ]

        lane_mod_32_for_k = lane % arith.index(32)
        lane_div_32_for_k = lane / arith.index(32)

        _wave_id_i32 = _index_cast_to_i32(wave_id)
        _wave_id_sgpr = rocdl.readfirstlane(i32_type, _wave_id_i32)
        _ckv_wave_bytes_c = arith.constant(CKV_WAVE_BYTES, type=T.i32)
        _wave_offset_sgpr = _std_arith.MulIOp(
            _raw(_wave_id_sgpr), _raw(_ckv_wave_bytes_c),
        ).result

        def _precompute_wave_base_i32(k_buf):
            """LDS wave base as i32 SGPR — no GEP, stays scalar."""
            lds_base = memref_dialect.extract_aligned_pointer_as_index(k_buf)
            lds_base_i32 = _index_cast_to_i32(lds_base)
            lds_base_sgpr = rocdl.readfirstlane(i32_type, lds_base_i32)
            return _std_arith.AddIOp(
                _raw(lds_base_sgpr), _raw(_wave_offset_sgpr),
            ).result

        lds_wptr_cur = _precompute_wave_base_i32(lds_k_cur_buf)
        lds_wptr_next = _precompute_wave_base_i32(lds_k_next_buf)

        def _i32_to_lds_ptr(i32_val):
            i64_val = _std_arith.ExtUIOp(i64_type, i32_val).result
            return _inttoptr_lds(i64_val)

        C_VOFF_G1_DELTA = 2 * K_SEQ_STRIDE * STRIDE_TOKEN_KV * 2
        _voff_g0_const = (
            (wave_id * arith.index(STRIDE_TOKEN_KV) + kv_head_offset)
            * arith.index(2)
            + lane_div_32_for_k
            * arith.index(K_SEQ_STRIDE * STRIDE_TOKEN_KV * 2)
            + lane_mod_32_for_k * arith.index(4)
        )
        _voff_g0_const_i32 = _index_cast_to_i32(_voff_g0_const)
        _voff_g1_const_i32 = _std_arith.AddIOp(
            _raw(_voff_g0_const_i32),
            _raw(arith.constant(C_VOFF_G1_DELTA, type=T.i32)),
        ).result

        def _coop_load_k_setup(kv_page_base, page_off, lds_wave_base_i32):
            page_byte_base = (
                kv_page_base + page_off * arith.index(STRIDE_TOKEN_KV)
            ) * arith.index(2)
            page_byte_base_i32 = _index_cast_to_i32(page_byte_base)
            voff_g0_i32 = _std_arith.AddIOp(
                _raw(page_byte_base_i32), _raw(_voff_g0_const_i32),
            ).result
            voff_g1_i32 = _std_arith.AddIOp(
                _raw(page_byte_base_i32), _raw(_voff_g1_const_i32),
            ).result
            return voff_g0_i32, voff_g1_i32, lds_wave_base_i32

        def _emit_k_chunk_pair(lds_wave_base_i32, voff_g0_i32, voff_g1_i32, j):
            _emit_k_single(lds_wave_base_i32, voff_g0_i32, j, 0)
            _emit_k_single(lds_wave_base_i32, voff_g1_i32, j, CKV_HALF_BYTES)

        def _emit_k_single(lds_wave_base_i32, voff_i32, j, half_offset):
            m0_val = _std_arith.AddIOp(
                lds_wave_base_i32,
                _raw(arith.constant(half_offset + j * 128, type=T.i32)),
            ).result
            lds_ptr = _i32_to_lds_ptr(m0_val)
            rocdl.raw_ptr_buffer_load_lds(
                kv_rsrc, lds_ptr, _raw(c_dword_sz),
                voff_i32, _raw(c_soff_zero),
                _raw(K_COL_OFFSETS[j]), _raw(aux))

        def coop_load_k(kv_page_base, page_off, lds_ptr_wave):
            voff_g0_i32, voff_g1_i32, lds_ptr_wave = \
                _coop_load_k_setup(kv_page_base, page_off, lds_ptr_wave)
            for j in range_constexpr(CKV_NUM_CHUNKS):
                _emit_k_chunk_pair(
                    lds_ptr_wave, voff_g0_i32, voff_g1_i32, j)

        def coop_load_k_lo(kv_page_base, page_off, lds_ptr_wave):
            voff_g0_i32, voff_g1_i32, lds_ptr_wave = \
                _coop_load_k_setup(kv_page_base, page_off, lds_ptr_wave)
            for j in range_constexpr(K_LO_CHUNKS):
                _emit_k_chunk_pair(
                    lds_ptr_wave, voff_g0_i32, voff_g1_i32, j)

        def coop_load_k_hi0(kv_page_base, page_off, lds_ptr_wave):
            voff_g0_i32, voff_g1_i32, lds_ptr_wave = \
                _coop_load_k_setup(kv_page_base, page_off, lds_ptr_wave)
            for j in range_constexpr(K_HI0_CHUNKS):
                _emit_k_chunk_pair(
                    lds_ptr_wave, voff_g0_i32, voff_g1_i32,
                    j + K_LO_CHUNKS)

        def coop_load_k_hi1(kv_page_base, page_off, lds_ptr_wave):
            voff_g0_i32, voff_g1_i32, lds_ptr_wave = \
                _coop_load_k_setup(kv_page_base, page_off, lds_ptr_wave)
            for j in range_constexpr(K_HI1_CHUNKS):
                _emit_k_chunk_pair(
                    lds_ptr_wave, voff_g0_i32, voff_g1_i32,
                    j + K_LO_CHUNKS + K_HI0_CHUNKS)

        def coop_load_k_asm(kv_page_base, page_off, lds_ptr_wave):
            coop_load_k(kv_page_base, page_off, lds_ptr_wave)

        # ---- V transpose: K_buf(LDS) → V^T_buf(LDS) ----
        v2i32_type = ir.VectorType.get([2], i32_type)

        def _vt_col_offset(vt_dim_quad_val):
            vt_chunk = vt_dim_quad_val / arith.index(CKV_CHUNK_COLS)
            vt_col_in_chunk = vt_dim_quad_val % arith.index(CKV_CHUNK_COLS)
            return vt_chunk * arith.index(CKV_CHUNK_F16) + vt_col_in_chunk

        def coop_load_v(k_buf):
            vt_col_off = _vt_col_offset(vt_dim_quad)
            vt_token_off = tid_div_128 * arith.index(CKV_TOKEN_F16)
            v_raw = []
            for g_pair in range_constexpr(BLOCK_N // 8):
                dwords_dp0 = []
                dwords_dp1 = []
                for s in range_constexpr(4):
                    lds_idx = (
                        arith.index(s * CKV_WAVE_F16
                                    + g_pair * CKV_HALF_F16)
                        + vt_token_off
                        + vt_col_off
                    )
                    quad = vector.load_op(v4f16_type, k_buf, [_raw(lds_idx)])
                    dw_pair = vector.bitcast(v2i32_type, quad)
                    dwords_dp0.append(
                        vector.extract(dw_pair, static_position=[0], dynamic_position=[])
                    )
                    dwords_dp1.append(
                        vector.extract(dw_pair, static_position=[1], dynamic_position=[])
                    )
                v_raw.append((dwords_dp0, dwords_dp1))
            return v_raw

        def coop_perm_store_v(v_raw):
            vt_write_base = (
                tid_mod_128 * arith.index(2 * VT_STRIDE)
                + tid_div_128 * arith.index(8)
            )
            for g_pair in range_constexpr(BLOCK_N // 8):
                dwords_dp0, dwords_dp1 = v_raw[g_pair]
                for dp_idx, dwords in enumerate([dwords_dp0, dwords_dp1]):
                    out = []
                    for src_pair, sel in [
                        ((1, 0), c_perm_lo), ((3, 2), c_perm_lo),
                        ((1, 0), c_perm_hi), ((3, 2), c_perm_hi),
                    ]:
                        dw = llvm.call_intrinsic(
                            i32_type, "llvm.amdgcn.perm",
                            [dwords[src_pair[0]], dwords[src_pair[1]], sel],
                            [], [],
                        )
                        v1 = vector.from_elements(v1i32_type, [dw])
                        out.append(vector.bitcast(v2f16_type, v1))
                    v4_lo = vector.shuffle(out[0], out[1], [0, 1, 2, 3])
                    v4_hi = vector.shuffle(out[2], out[3], [0, 1, 2, 3])
                    vec = vector.shuffle(
                        v4_lo, v4_hi, [0, 1, 2, 3, 4, 5, 6, 7],
                    )
                    vt_idx = (
                        vt_write_base
                        + arith.index(dp_idx * VT_STRIDE)
                        + arith.index(g_pair * 16)
                    )
                    vector.store(vec, lds_vt, [_raw(_vt_swizzle(vt_idx))])

        def coop_load_v_half(half, k_buf):
            vt_col_off = _vt_col_offset(vt_dim_quad)
            vt_token_off = tid_div_128 * arith.index(CKV_TOKEN_F16)
            dwords_dp0 = []
            dwords_dp1 = []
            for s in range_constexpr(4):
                lds_idx = (
                    arith.index(s * CKV_WAVE_F16
                                + half * CKV_HALF_F16)
                    + vt_token_off
                    + vt_col_off
                )
                quad = vector.load_op(v4f16_type, k_buf, [_raw(lds_idx)])
                dw_pair = vector.bitcast(v2i32_type, quad)
                dwords_dp0.append(
                    vector.extract(dw_pair, static_position=[0], dynamic_position=[])
                )
                dwords_dp1.append(
                    vector.extract(dw_pair, static_position=[1], dynamic_position=[])
                )
            return (dwords_dp0, dwords_dp1)

        def coop_perm_store_v_half(half, v_raw_half):
            vt_write_base = (
                tid_mod_128 * arith.index(2 * VT_STRIDE)
                + tid_div_128 * arith.index(8)
            )
            dwords_dp0, dwords_dp1 = v_raw_half
            for dp_idx, dwords in enumerate([dwords_dp0, dwords_dp1]):
                out = []
                for src_pair, sel in [
                    ((1, 0), c_perm_lo), ((3, 2), c_perm_lo),
                    ((1, 0), c_perm_hi), ((3, 2), c_perm_hi),
                ]:
                    dw = llvm.call_intrinsic(
                        i32_type, "llvm.amdgcn.perm",
                        [dwords[src_pair[0]], dwords[src_pair[1]], sel],
                        [], [],
                    )
                    v1 = vector.from_elements(v1i32_type, [dw])
                    out.append(vector.bitcast(v2f16_type, v1))
                v4_lo = vector.shuffle(out[0], out[1], [0, 1, 2, 3])
                v4_hi = vector.shuffle(out[2], out[3], [0, 1, 2, 3])
                vec = vector.shuffle(
                    v4_lo, v4_hi, [0, 1, 2, 3, 4, 5, 6, 7],
                )
                vt_idx = (
                    vt_write_base
                    + arith.index(dp_idx * VT_STRIDE)
                    + arith.index(half * 16)
                )
                vector.store(vec, lds_vt, [_raw(_vt_swizzle(vt_idx))])

        def _vh_flatten(v_raw_h1):
            dp0, dp1 = v_raw_h1
            return [_raw(v) for v in dp0] + [_raw(v) for v in dp1]

        def _vh_unflatten(ia, offset):
            dp0 = [ia[offset + i] for i in range(4)]
            dp1 = [ia[offset + 4 + i] for i in range(4)]
            return (dp0, dp1)

        def _vh_dummy():
            _di = arith.constant(0, type=T.i32)
            return ([_di] * 4, [_di] * 4)

        def coop_transpose_v_from_lds(k_buf):
            v_raw = coop_load_v(k_buf)
            coop_perm_store_v(v_raw)

        # ---- Preload Q via 64×64 tiled loop ----
        q_row = wave_id

        q_rsrc = buffer_ops.create_buffer_resource(Q)

        kv_total_blocks = batch_size_v * max_num_blocks_v
        kv_total_elems = kv_total_blocks * arith.index(STRIDE_PAGE)
        kv_size_bytes = kv_total_elems * arith.index(2)
        c_max_records = arith.index(0xEFFFFFFE)
        kv_size_capped = _std_arith.MinUIOp(_raw(kv_size_bytes), _raw(c_max_records)).result
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, num_records_bytes=kv_size_capped
        )
        mid_o_rsrc = buffer_ops.create_buffer_resource(Mid_O)
        mid_lse_rsrc = buffer_ops.create_buffer_resource(Mid_lse)
        lds_base_byte_idx = _memref.ExtractAlignedPointerAsIndexOp(lds_k_cur_buf).result
        c_dword_sz = arith.constant(4, type=T.i32)
        c_zero_i32 = arith.constant(0, type=T.i32)
        aux = arith.constant(0, type=T.i32)

        q_b_packs_lo = [None] * K_STEPS
        q_b_packs_hi = [None] * K_STEPS

        lane_div_32 = lane / arith.index(32)
        lane_mod_32 = lane % arith.index(32)

        q_voff_base_f16 = (
            batch_idx
            * seqlen_q_v
            * arith.index(STRIDE_TOKEN_Q)
            + wave_id * arith.index(STRIDE_TOKEN_Q)
            + head_group_idx
            * arith.index(HEADS_PER_WAVE * HEAD_DIM_QK)
            + lane_div_32 * arith.index(HEAD_DIM_QK)
            + lane_mod_32 * arith.index(2)
        )

        Q_LDS_PAIR_BYTES = 2 * Q_TILE_DIM * 2
        Q_GLOBAL_PAIR_BYTES = 2 * HEAD_DIM_QK * 2
        Q_CALL_SOFFSETS = [
            arith.constant(
                c * (Q_GLOBAL_PAIR_BYTES - Q_LDS_PAIR_BYTES),
                type=i32_type,
            )
            for c in range(Q_CALLS_PER_WAVE)
        ]
        Q_CALL_INST_OFFSETS = [
            arith.constant(c * Q_LDS_PAIR_BYTES, type=i32_type)
            for c in range(Q_CALLS_PER_WAVE)
        ]

        Q_TILE_BYTES = Q_TILE_ELEMS * 2

        pf0_phys_i32, pf0_p_off = lookup_page_issue(kv_split_start)
        tile1_start_early = kv_split_start + arith.index(BLOCK_N)
        pf1_phys_early, pf1_p_off_early = lookup_page_issue(
            tile1_start_early
        )

        for d_tile_start, batch_size_q in _Q_BATCHES:

            for t in range_constexpr(batch_size_q):
                d_tile = d_tile_start + t

                tile_lds_byte = (
                    lds_base_byte_idx
                    + arith.index(t * Q_TILE_BYTES)
                    + wave_id
                    * arith.index(HEADS_PER_WAVE * Q_TILE_DIM * 2)
                )
                tile_lds_i64 = arith.index_cast(T.i64, tile_lds_byte)
                tile_lds_scalar = rocdl.readfirstlane(
                    T.i64, tile_lds_i64
                )
                tile_lds_ptr = _inttoptr_lds(tile_lds_scalar)

                q_voff_byte = (
                    (q_voff_base_f16
                     + arith.index(d_tile * Q_TILE_DIM)) * arith.index(2)
                )
                q_voff_i32 = _index_cast_to_i32(
                    q_voff_byte
                )

                for c in range_constexpr(Q_CALLS_PER_WAVE):
                    rocdl.raw_ptr_buffer_load_lds(
                        q_rsrc, tile_lds_ptr,
                        _raw(c_dword_sz),
                        q_voff_i32,
                        _raw(Q_CALL_SOFFSETS[c]),
                        _raw(Q_CALL_INST_OFFSETS[c]),
                        _raw(aux),
                    )

            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            gpu.barrier()

            for t in range_constexpr(batch_size_q):
                d_tile = d_tile_start + t
                q_tile_byte_base = (
                    lds_base_byte_idx
                    + (arith.index(t * Q_TILE_ELEMS)
                       + (wave_id
                          * arith.index(HEADS_PER_WAVE)
                          + lane_mod_16)
                       * arith.index(Q_TILE_DIM)
                       + lane_div_16 * arith.index(8)) * arith.index(2)
                )
                for local_ks in range_constexpr(2):
                    global_ks = d_tile * 2 + local_ks
                    _base_off = local_ks * 64
                    q_wide = _lds_load_prefer_agpr(
                        q_tile_byte_base, v8f16_type,
                        static_byte_offset=_base_off,
                    )
                    q_b_packs_lo[global_ks] = vector.shuffle(
                        q_wide, q_wide, [0, 1, 2, 3])
                    q_b_packs_hi[global_ks] = vector.shuffle(
                        q_wide, q_wide, [4, 5, 6, 7])

            gpu.barrier()

        # ---- Constants ----
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_neg_large = arith.constant(-1e6, type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_sm_scale = arith.constant(sm_scale, type=compute_type)
        c_sm_scale_log2e = arith.constant(sm_scale * 1.4426950408889634, type=compute_type)
        c_zero_v4f32 = arith.constant_vector(0.0, v4f32_type)

        N_DP = D_CHUNKS // 2
        VT_PREFETCH = 8
        VT_REMAINING = N_DP - VT_PREFETCH

        def _vt_idx(dp):
            idx = (
                (arith.index(dp * 16) + lane_mod_16)
                * arith.index(VT_STRIDE)
                + lane_div_16 * arith.index(8)
            )
            return _vt_swizzle(idx)

        def vgpr_pin(val):
            return llvm.InlineAsmOp(
                res=_raw(val).type,
                operands_=[_raw(val)],
                asm_string="; vgpr_pin",
                constraints="=v,0",
                has_side_effects=True,
                is_align_stack=False,
            ).result

        def _softmax_reduce_issue(s_acc, kv_pos=None, do_causal=False):
            """Issue ds_write + ds_read for warp reduce; returns
            (s_vals, max_vec) where max_vec is NOT yet waited on."""
            s_vals = []
            for ii in range_constexpr(4):
                s_val = vector.extract(
                    s_acc, static_position=[ii], dynamic_position=[],
                )
                if CAUSAL and do_causal:
                    kv_abs = kv_pos + lane_div_16 * arith.index(4) + arith.index(ii)
                    q_abs_pos = seqlen_kv_v - seqlen_q_v + q_row
                    kv_abs_i64 = arith.index_cast(T.i64, kv_abs)
                    q_abs_i64 = arith.index_cast(T.i64, q_abs_pos)
                    is_masked = _std_arith.CmpIOp(
                        _std_arith.CmpIPredicate.ugt,
                        _raw(kv_abs_i64), _raw(q_abs_i64),
                    ).result
                    s_val = _std_arith.SelectOp(
                        is_masked, _raw(c_neg_large), _raw(s_val),
                    ).result
                s_vals.append(s_val)

            local_max = s_vals[0]
            for ii in range_constexpr(3):
                local_max = _std_arith.MaximumFOp(
                    _raw(local_max), _raw(s_vals[ii + 1]),
                    fastmath=fm_fast,
                ).result

            red_wr_idx = (
                wave_id * arith.index(64)
                + lane_mod_16 * arith.index(4)
                + lane_div_16
            )
            _memref.StoreOp(local_max, lds_red, [_raw(red_wr_idx)])

            red_rd_base = (
                wave_id * arith.index(64)
                + lane_mod_16 * arith.index(4)
            )
            max_vec = vector.load_op(v4f32_type, lds_red, [_raw(red_rd_base)])
            return s_vals, max_vec

        def _softmax_reduce_collect(max_vec):
            """Wait for max_vec ds_read and compute global max."""
            global_max = c_neg_large
            for g in range_constexpr(4):
                max_g = vector.extract(
                    max_vec, static_position=[g], dynamic_position=[],
                )
                global_max = _std_arith.MaximumFOp(
                    _raw(global_max), _raw(max_g),
                    fastmath=fm_fast,
                ).result
            return global_max

        def softmax_reduce(s_acc, kv_pos=None, do_causal=False):
            s_vals, max_vec = _softmax_reduce_issue(
                s_acc, kv_pos=kv_pos, do_causal=do_causal)
            global_max = _softmax_reduce_collect(max_vec)
            return global_max, s_vals

        def softmax_finalize(global_max, s_vals, m_old, l_old):
            m_new = _std_arith.MaximumFOp(
                _raw(m_old), _raw(global_max), fastmath=fm_fast,
            ).result

            scaled_m_new = _std_arith.MulFOp(
                _raw(c_sm_scale_log2e), _raw(m_new), fastmath=fm_fast,
            ).result

            rescale_arg = _std_arith.SubFOp(
                _std_arith.MulFOp(
                    _raw(m_old), _raw(c_sm_scale_log2e), fastmath=fm_fast,
                ).result,
                _raw(scaled_m_new),
                fastmath=fm_fast,
            ).result
            rescale = _math_exp2(rescale_arg, fastmath=fm_fast)

            p_vals = [None] * 4
            local_sum = c_zero_f
            for ii in range_constexpr(4):
                exp_arg = _std_arith.SubFOp(
                    _std_arith.MulFOp(
                        _raw(s_vals[ii]), _raw(c_sm_scale_log2e),
                        fastmath=fm_fast,
                    ).result,
                    _raw(scaled_m_new),
                    fastmath=fm_fast,
                ).result
                p = _math_exp2(exp_arg)
                p_vals[ii] = p
                local_sum = _std_arith.AddFOp(
                    _raw(local_sum), _raw(p), fastmath=fm_fast
                ).result

            l_scaled = _std_arith.MulFOp(
                _raw(l_old), _raw(rescale), fastmath=fm_fast
            ).result
            l_new = _std_arith.AddFOp(
                _raw(l_scaled), _raw(local_sum), fastmath=fm_fast
            ).result

            p_f16 = []
            for ii in range_constexpr(4):
                p_f16.append(
                    _std_arith.TruncFOp(elem_type, _raw(p_vals[ii])).result
                )
            p_pack = vector.from_elements(v4f16_type, p_f16)
            return m_new, l_new, p_pack, rescale

        def softmax_and_pack(s_acc, m_old, l_old,
                             kv_pos=None, do_causal=False):
            global_max, s_vals = softmax_reduce(
                s_acc, kv_pos=kv_pos, do_causal=do_causal)
            return softmax_finalize(global_max, s_vals, m_old, l_old)

        def _make_rescale_vec(rescale):
            # rescale_i32 = _std_arith.BitcastOp(T.i32, _raw(rescale)).result
            # rescale_sgpr = rocdl.readfirstlane(T.i32, rescale_i32)
            # rescale_dup = _std_arith.BitcastOp(compute_type, _raw(rescale_sgpr)).result
            # return vector.from_elements(
            #     v4f32_type,
            #     [rescale, rescale_dup,
            #      rescale, rescale_dup])
            return vector.broadcast(v4f32_type, rescale)

        def _rescale_acc(acc, rescale_vec):
            return _std_arith.MulFOp(
                _raw(acc), _raw(rescale_vec),
                fastmath=fm_fast,
            ).result

        def do_kv_tile(kv_pos, m_old, l_old, o_accs,
                       k_cur_buf, k_next_buf, k_buf,
                       wptr_cur=None, wptr_next=None,
                       v_raw_h1_in=None,
                       load_knn=True, do_vt=True,
                       do_causal=False):
            rocdl.sched_barrier(0)

            if load_knn:
                nn_start = kv_pos + arith.index(2 * BLOCK_N)
                nn_phys_i32, nn_p_off = lookup_page_issue(nn_start)

            for ks in range_constexpr(K_REMAINING):
                _ki = k_read_base + arith.index(K_STEP_OFFSETS_F16[ks + K_PREFETCH])
                k_buf[ks + K_PREFETCH] = vector.load_op(
                    v8f16_type, k_cur_buf, [_raw(_ki)],
                    nontemporal=True)

            # ---- GEMM1 Phase 1 ----
            s_accs = [c_zero_v4f32]
            DS_READ_INTER = 8
            for ks in range_constexpr(DS_READ_INTER):
                q_lo = q_b_packs_lo[ks]
                q_hi = q_b_packs_hi[ks]
                k_lo = vector.shuffle(
                    k_buf[ks], k_buf[ks], [0, 1, 2, 3])
                k_hi = vector.shuffle(
                    k_buf[ks], k_buf[ks], [4, 5, 6, 7])
                s_accs[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_lo, q_lo, s_accs[0], 0, 0, 0],
                )
                s_accs[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_hi, q_hi, s_accs[0], 0, 0, 0],
                )

            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3)

            coop_perm_store_v_half(1, v_raw_h1_in)

            if load_knn:
                nn_page_base = lookup_page_resolve(nn_phys_i32)
                voff_g0_t, voff_g1_t, wbase_t = \
                    _coop_load_k_setup(nn_page_base, nn_p_off, wptr_cur)

            if do_vt:
                v_raw_h0 = coop_load_v_half(0, k_next_buf)

            # ---- GEMM1 Phase 2 (with interleaved K loads) ----
            GEMM1_HALF = K_STEPS - DS_READ_INTER
            _N_LOADS = K_LO_CHUNKS * 2 if load_knn else 0
            for ks in range_constexpr(GEMM1_HALF):
                if load_knn and ks < _N_LOADS:
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(2)
                    _chunk = ks // 2
                    if ks % 2 == 0:
                        _emit_k_single(wbase_t, voff_g0_t, _chunk, 0)
                    else:
                        _emit_k_single(wbase_t, voff_g1_t, _chunk,
                                       CKV_HALF_BYTES)
                ks_inner = ks + DS_READ_INTER
                q_lo = q_b_packs_lo[ks_inner]
                q_hi = q_b_packs_hi[ks_inner]
                k_lo = vector.shuffle(
                    k_buf[ks_inner], k_buf[ks_inner], [0, 1, 2, 3])
                k_hi = vector.shuffle(
                    k_buf[ks_inner], k_buf[ks_inner], [4, 5, 6, 7])
                s_accs[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_lo, q_lo, s_accs[0], 0, 0, 0],
                )
                s_accs[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_hi, q_hi, s_accs[0], 0, 0, 0],
                )

            rocdl.sched_barrier(0)

            s_vals, max_vec = _softmax_reduce_issue(
                s_accs[0], kv_pos=kv_pos, do_causal=do_causal)

            if load_knn:
                coop_load_k_hi0(nn_page_base, nn_p_off, wptr_cur)
                coop_load_k_hi1(nn_page_base, nn_p_off, wptr_cur)

            global_max = _softmax_reduce_collect(max_vec)

            m_new, l_new, p_pack, rescale = softmax_finalize(
                global_max, s_vals, m_old, l_old)

            v_buf = [None] * N_DP
            for ks in range_constexpr(VT_PREFETCH):
                v_buf[ks] = vector.load_op(
                    v8f16_type, lds_vt, [_raw(_vt_idx(ks))],
                    nontemporal=True)

            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_barrier(0)

            rescale_vec = _make_rescale_vec(rescale)
            rocdl.sched_barrier(0)

            for dp in range_constexpr(N_DP):
                o_accs[dp * 2] = _rescale_acc(o_accs[dp * 2], rescale_vec)
                o_accs[dp * 2 + 1] = _rescale_acc(o_accs[dp * 2 + 1], rescale_vec)

            rocdl.sched_barrier(0)

            for vt in range_constexpr(VT_REMAINING):
                v_buf[vt + VT_PREFETCH] = vector.load_op(
                    v8f16_type, lds_vt, [_raw(_vt_idx(vt + VT_PREFETCH))],
                    nontemporal=True)

            k_buf_next = [None] * K_STEPS
            for dp in range_constexpr(N_DP):
                if dp < K_PREFETCH:
                    _kni = k_read_base + arith.index(K_STEP_OFFSETS_F16[dp])
                    k_buf_next[dp] = vector.load_op(
                        v8f16_type, k_next_buf, [_raw(_kni)],
                        nontemporal=True)

            N_DP_DS_WRITE_INTER = 1
            N_DP_LOOP = N_DP - N_DP_DS_WRITE_INTER
            for dp in range_constexpr(N_DP_LOOP):
                v_lo = vector.shuffle(
                    v_buf[dp], v_buf[dp], [0, 1, 2, 3])
                v_hi = vector.shuffle(
                    v_buf[dp], v_buf[dp], [4, 5, 6, 7])
                o_accs[dp * 2] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_lo, p_pack, o_accs[dp * 2], 0, 0, 0],
                )
                o_accs[dp * 2 + 1] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_hi, p_pack, o_accs[dp * 2 + 1], 0, 0, 0],
                )
            _k_dummy = arith.constant_vector(0.0, v8f16_type)
            for ks in range_constexpr(K_REMAINING):
                k_buf_next[ks + K_PREFETCH] = _k_dummy

            # vt load
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(4)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(4)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3)
            # v load
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(2)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(2)
            # k next load
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(2)
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(2)
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(2)
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(1)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=15)
            rocdl.sched_barrier(0)

            for dp in range_constexpr(N_DP_DS_WRITE_INTER):
                dp_inner = dp + N_DP_LOOP
                v_lo = vector.shuffle(
                    v_buf[dp_inner], v_buf[dp_inner], [0, 1, 2, 3])
                v_hi = vector.shuffle(
                    v_buf[dp_inner], v_buf[dp_inner], [4, 5, 6, 7])
                o_accs[dp_inner * 2] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_lo, p_pack, o_accs[dp_inner * 2], 0, 0, 0],
                )
                o_accs[dp_inner * 2 + 1] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_hi, p_pack, o_accs[dp_inner * 2 + 1], 0, 0, 0],
                )

            if do_vt:
                coop_perm_store_v_half(0, v_raw_h0)
                v_raw_h1_out = coop_load_v_half(1, k_next_buf)
            else:
                v_raw_h1_out = _vh_dummy()

            if do_vt:
                rocdl.sched_mfma(1)
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(1)
                rocdl.sched_dswr(1)
                rocdl.sched_dsrd(4)

            return m_new, l_new, o_accs, k_buf_next, v_raw_h1_out

        def do_kv_pair(kv_pos, m_old, l_old, o_accs,
                       k_cur_buf, k_next_buf, k_buf,
                       wptr_cur=None, wptr_next=None,
                       nn_a_phys_in=None,
                       v_raw_h1_in=None,
                       load_knn=True, do_vt_final=True):

            rocdl.sched_barrier(0)

            nn_a_phys_next = None
            if load_knn:
                nn_a_phys = nn_a_phys_in
                kv_pos_i32 = _index_cast_to_i32(kv_pos)
                c_2bn_i32 = arith.constant(2 * BLOCK_N, type=T.i32)
                c_3bn_i32 = arith.constant(3 * BLOCK_N, type=T.i32)
                c_4bn_i32 = arith.constant(4 * BLOCK_N, type=T.i32)
                nn_a_pos_i32 = _std_arith.AddIOp(
                    _raw(kv_pos_i32), _raw(c_2bn_i32),
                ).result
                _nn_a_and = _std_arith.AndIOp(
                    _raw(nn_a_pos_i32), _raw(_c_pbs_mask_i32),
                ).result
                nn_a_poff = arith.index_cast(T.index, _nn_a_and)
                nn_b_pos_i32 = _std_arith.AddIOp(
                    _raw(kv_pos_i32), _raw(c_3bn_i32),
                ).result
                nn_b_phys, nn_b_poff = \
                    lookup_page_issue(
                        None, clamp=False, pos_i32=nn_b_pos_i32)

            # ============ TILE A ============
            nn_a_base = lookup_page_resolve(nn_a_phys)
            # last half of v_raw_h1_in
            for ks in range_constexpr(K_REMAINING):
                _ki = k_read_base + arith.index(K_STEP_OFFSETS_F16[ks + K_PREFETCH])
                k_buf[ks + K_PREFETCH] = vector.load_op(
                    v8f16_type, k_cur_buf, [_raw(_ki)],
                    nontemporal=True)

            s = [c_zero_v4f32]
            DS_READ_INTER = 8
            for ks in range_constexpr(DS_READ_INTER):
                q_lo = q_b_packs_lo[ks]
                q_hi = q_b_packs_hi[ks]
                k_lo = vector.shuffle(
                    k_buf[ks], k_buf[ks], [0, 1, 2, 3])
                k_hi = vector.shuffle(
                    k_buf[ks], k_buf[ks], [4, 5, 6, 7])
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_lo, q_lo, s[0], 0, 0, 0],
                )
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_hi, q_hi, s[0], 0, 0, 0],
                )

            # asm style interleaved schedule
            rocdl.sched_mfma(1) # mfma 1
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 4
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 8
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 12
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 15
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 16

            coop_perm_store_v_half(1, v_raw_h1_in)
            voff_g0_a, voff_g1_a, wbase_a = \
                _coop_load_k_setup(nn_a_base, nn_a_poff, wptr_cur)

            GEMM1_HALF = K_STEPS - DS_READ_INTER
            _N_LOADS = K_LO_CHUNKS * 2
            for ks in range_constexpr(GEMM1_HALF):
                if ks < _N_LOADS:
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(2)
                    _chunk = ks // 2
                    if ks % 2 == 0:
                        _emit_k_single(wbase_a, voff_g0_a, _chunk, 0)
                    else:
                        _emit_k_single(wbase_a, voff_g1_a, _chunk,
                                       CKV_HALF_BYTES)
                ks_inner = ks + DS_READ_INTER
                q_lo = q_b_packs_lo[ks_inner]
                q_hi = q_b_packs_hi[ks_inner]
                k_lo = vector.shuffle(
                    k_buf[ks_inner], k_buf[ks_inner], [0, 1, 2, 3])
                k_hi = vector.shuffle(
                    k_buf[ks_inner], k_buf[ks_inner], [4, 5, 6, 7])
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_lo, q_lo, s[0], 0, 0, 0],
                )
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_hi, q_hi, s[0], 0, 0, 0],
                )

            rocdl.sched_barrier(0)

            s_vals_a, max_vec_a = _softmax_reduce_issue(s[0])

            coop_load_k_hi0(nn_a_base, nn_a_poff, wptr_cur)
            coop_load_k_hi1(nn_a_base, nn_a_poff, wptr_cur)

            if load_knn:
                _c4_i32 = arith.constant(4, type=T.i32)
                kv_block_i32 = _std_arith.ShRUIOp(
                    _raw(kv_pos_i32), _raw(_c6_i32),
                ).result
                kv_block_byte = _std_arith.ShLIOp(
                    _raw(kv_block_i32), _raw(_c2_i32),
                ).result
                nn_a_next_byte = _std_arith.AddIOp(
                    _raw(kv_block_byte), _raw(_c4_i32),
                ).result
                nn_a_phys_next = rocdl.raw_ptr_buffer_load(
                    i32_type, bt_rsrc, nn_a_next_byte,
                    _raw(_bt_soff), _raw(_bt_aux),
                )

            global_max_a = _softmax_reduce_collect(max_vec_a)

            m, l, p_pack_a, rescale_a = softmax_finalize(
                global_max_a, s_vals_a, m_old, l_old)

            
            v_buf = [None] * N_DP
            for ks in range_constexpr(VT_PREFETCH):
                v_buf[ks] = vector.load_op(
                    v8f16_type, lds_vt, [_raw(_vt_idx(ks))],
                    nontemporal=True)

            # _barrier(lgkmcnt=0)

            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_barrier(0)

            rescale_vec_a = _make_rescale_vec(rescale_a)
            rocdl.sched_barrier(0)

            for dp in range_constexpr(N_DP):
                o_accs[dp * 2] = _rescale_acc(o_accs[dp * 2], rescale_vec_a)
                o_accs[dp * 2 + 1] = _rescale_acc(o_accs[dp * 2 + 1], rescale_vec_a)

            rocdl.sched_barrier(0)

            for vt in range_constexpr(VT_REMAINING):
                v_buf[vt + VT_PREFETCH] = vector.load_op(
                    v8f16_type, lds_vt, [_raw(_vt_idx(vt + VT_PREFETCH))],
                    nontemporal=True)

            v_raw_b_h0 = coop_load_v_half(0, k_next_buf)

            for dp in range_constexpr(N_DP):
                if dp < K_PREFETCH:
                    _kbi = k_read_base + arith.index(K_STEP_OFFSETS_F16[dp])
                    k_buf[dp] = vector.load_op(
                        v8f16_type, k_next_buf, [_raw(_kbi)],
                        nontemporal=True)

            N_DP_DS_WRITE_INTER = 1
            N_DP_LOOP = N_DP - N_DP_DS_WRITE_INTER
            for dp in range_constexpr(N_DP_LOOP):
                v_lo = vector.shuffle(
                    v_buf[dp], v_buf[dp], [0, 1, 2, 3])
                v_hi = vector.shuffle(
                    v_buf[dp], v_buf[dp], [4, 5, 6, 7])
                o_accs[dp * 2] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_lo, p_pack_a, o_accs[dp * 2], 0, 0, 0],
                )
                o_accs[dp * 2 + 1] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_hi, p_pack_a, o_accs[dp * 2 + 1], 0, 0, 0],
                )
            _k_dummy = arith.constant_vector(0.0, v8f16_type)
            for ks in range_constexpr(K_REMAINING):
                k_buf[ks + K_PREFETCH] = _k_dummy

            # vt load
            rocdl.sched_mfma(1) # mfma 1
            rocdl.sched_dsrd(2) # ds 3
            rocdl.sched_mfma(4) # mfma 5
            rocdl.sched_dsrd(2) # ds 4
            rocdl.sched_mfma(4) # mfma 9
            rocdl.sched_dsrd(2) # ds 5
            rocdl.sched_mfma(3) # mfma 12
            rocdl.sched_dsrd(2) # ds 6
            rocdl.sched_mfma(3) # mfma 15

            # v laod
            rocdl.sched_dsrd(2) # ds 7
            rocdl.sched_mfma(2) # mfma 17
            rocdl.sched_dsrd(2) # ds 7
            rocdl.sched_mfma(2) # mfma 19
            
            # k next load
            rocdl.sched_dsrd(1) # ds 8
            rocdl.sched_mfma(1) # mfma 20
            rocdl.sched_dsrd(1) # ds 9
            rocdl.sched_mfma(2) # mfma 22
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(1) # mfma 23
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(2) # mfma 25
            rocdl.sched_dsrd(1) # ds 8
            rocdl.sched_mfma(1) # mfma 26
            rocdl.sched_dsrd(1) # ds 9
            rocdl.sched_mfma(2) # mfma 28
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(1) # mfma 29
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(1) # mfma 30

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=15)
            rocdl.sched_barrier(0)

            for dp in range_constexpr(N_DP_DS_WRITE_INTER):
                dp_inner = dp + N_DP_LOOP
                v_lo = vector.shuffle(
                    v_buf[dp_inner], v_buf[dp_inner], [0, 1, 2, 3])
                v_hi = vector.shuffle(
                    v_buf[dp_inner], v_buf[dp_inner], [4, 5, 6, 7])
                o_accs[dp_inner * 2] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_lo, p_pack_a, o_accs[dp_inner * 2], 0, 0, 0],
                )
                o_accs[dp_inner * 2 + 1] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_hi, p_pack_a, o_accs[dp_inner * 2 + 1], 0, 0, 0],
                )
            coop_perm_store_v_half(0, v_raw_b_h0)
            v_raw_b_h1 = coop_load_v_half(1, k_next_buf)

            rocdl.sched_dswr(1)
            rocdl.sched_mfma(1) # 30
            rocdl.sched_dswr(1)
            rocdl.sched_mfma(1) # 31
            rocdl.sched_dsrd(4)
            # rocdl.sched_barrier(0)
            # _barrier(lgkmcnt=15)
            # rocdl.sched_barrier(0)


            # ============ TILE B ============
            nn_b_base = lookup_page_resolve(nn_b_phys)

            for ks in range_constexpr(K_REMAINING):
                _ki = k_read_base + arith.index(K_STEP_OFFSETS_F16[ks + K_PREFETCH])
                k_buf[ks + K_PREFETCH] = vector.load_op(
                    v8f16_type, k_next_buf, [_raw(_ki)],
                    nontemporal=True)

            s[0] = c_zero_v4f32
            DS_READ_INTER = 8
            for ks in range_constexpr(DS_READ_INTER):
                q_lo = q_b_packs_lo[ks]
                q_hi = q_b_packs_hi[ks]
                k_lo = vector.shuffle(
                    k_buf[ks], k_buf[ks], [0, 1, 2, 3])
                k_hi = vector.shuffle(
                    k_buf[ks], k_buf[ks], [4, 5, 6, 7])
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_lo, q_lo, s[0], 0, 0, 0],
                )
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_hi, q_hi, s[0], 0, 0, 0],
                )
            # asm style interleaved schedule
            rocdl.sched_mfma(1) # mfma 1
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 4
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 8
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 12
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 15
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(3) # mfma 16
            # rocdl.sched_barrier(0)
            # _barrier(lgkmcnt=15)
            # rocdl.sched_barrier(0)

            coop_perm_store_v_half(1, v_raw_b_h1)
            voff_g0_b, voff_g1_b, wbase_b = \
                _coop_load_k_setup(nn_b_base, nn_b_poff, wptr_next)

            _N_LOADS = K_LO_CHUNKS * 2
            for ks in range_constexpr(GEMM1_HALF):
                if ks < _N_LOADS:
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(2)
                    _chunk = ks // 2
                    if ks % 2 == 0:
                        _emit_k_single(wbase_b, voff_g0_b, _chunk, 0)
                    else:
                        _emit_k_single(wbase_b, voff_g1_b, _chunk,
                                       CKV_HALF_BYTES)
                ks_inner = ks + DS_READ_INTER
                q_lo = q_b_packs_lo[ks_inner]
                q_hi = q_b_packs_hi[ks_inner]
                k_lo = vector.shuffle(
                    k_buf[ks_inner], k_buf[ks_inner], [0, 1, 2, 3])
                k_hi = vector.shuffle(
                    k_buf[ks_inner], k_buf[ks_inner], [4, 5, 6, 7])
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_lo, q_lo, s[0], 0, 0, 0],
                )
                s[0] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [k_hi, q_hi, s[0], 0, 0, 0],
                )

            rocdl.sched_barrier(0)

            s_vals_b, max_vec_b = _softmax_reduce_issue(s[0])

            coop_load_k_hi0(nn_b_base, nn_b_poff, wptr_next)
            coop_load_k_hi1(nn_b_base, nn_b_poff, wptr_next)

            global_max_b = _softmax_reduce_collect(max_vec_b)

            m, l, p_pack_b, rescale_b = softmax_finalize(
                global_max_b, s_vals_b, m, l)

            for ks in range_constexpr(VT_PREFETCH):
                v_buf[ks] = vector.load_op(
                    v8f16_type, lds_vt, [_raw(_vt_idx(ks))],
                    nontemporal=True)

            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_dsrd(1)
            rocdl.sched_valu(4)
            rocdl.sched_barrier(0)

            rescale_vec_b = _make_rescale_vec(rescale_b)
            rocdl.sched_barrier(0)

            for dp in range_constexpr(N_DP):
                o_accs[dp * 2] = _rescale_acc(o_accs[dp * 2], rescale_vec_b)
                o_accs[dp * 2 + 1] = _rescale_acc(o_accs[dp * 2 + 1], rescale_vec_b)

            rocdl.sched_barrier(0)
            # _barrier(lgkmcnt=15)
            
            for vt in range_constexpr(VT_REMAINING):
                v_buf[vt + VT_PREFETCH] = vector.load_op(
                    v8f16_type, lds_vt, [_raw(_vt_idx(vt + VT_PREFETCH))],
                    nontemporal=True)

            v_raw_next_h0 = coop_load_v_half(0, k_cur_buf)

            for dp in range_constexpr(N_DP):
                if dp < K_PREFETCH:
                    _kni = k_read_base + arith.index(K_STEP_OFFSETS_F16[dp])
                    k_buf[dp] = vector.load_op(
                        v8f16_type, k_cur_buf, [_raw(_kni)])

            N_DP_DS_WRITE_INTER = 1
            N_DP_LOOP = N_DP - N_DP_DS_WRITE_INTER
            for dp in range_constexpr(N_DP_LOOP):
                v_lo = vector.shuffle(
                    v_buf[dp], v_buf[dp], [0, 1, 2, 3])
                v_hi = vector.shuffle(
                    v_buf[dp], v_buf[dp], [4, 5, 6, 7])
                o_accs[dp * 2] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_lo, p_pack_b, o_accs[dp * 2], 0, 0, 0],
                )
                o_accs[dp * 2 + 1] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_hi, p_pack_b, o_accs[dp * 2 + 1], 0, 0, 0],
                )
            _k_dummy = arith.constant_vector(0.0, v8f16_type)
            for ks in range_constexpr(K_REMAINING):
                k_buf[ks + K_PREFETCH] = _k_dummy

            # vt load
            rocdl.sched_mfma(1) # mfma 1
            rocdl.sched_dsrd(2) # ds 3
            rocdl.sched_mfma(4) # mfma 5
            rocdl.sched_dsrd(2) # ds 4
            rocdl.sched_mfma(4) # mfma 9
            rocdl.sched_dsrd(2) # ds 5
            rocdl.sched_mfma(3) # mfma 12
            rocdl.sched_dsrd(2) # ds 6
            rocdl.sched_mfma(3) # mfma 15

            # v laod
            rocdl.sched_dsrd(2) # ds 7
            rocdl.sched_mfma(2) # mfma 17
            rocdl.sched_dsrd(2) # ds 7
            rocdl.sched_mfma(2) # mfma 19
            
            # k next load
            rocdl.sched_dsrd(1) # ds 8
            rocdl.sched_mfma(1) # mfma 20
            rocdl.sched_dsrd(1) # ds 9
            rocdl.sched_mfma(2) # mfma 22
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(1) # mfma 23
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(2) # mfma 25
            rocdl.sched_dsrd(1) # ds 8
            rocdl.sched_mfma(1) # mfma 26
            rocdl.sched_dsrd(1) # ds 9
            rocdl.sched_mfma(2) # mfma 28
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(1) # mfma 29
            rocdl.sched_dsrd(1) # ds 10
            rocdl.sched_mfma(1) # mfma 30

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=15)
            rocdl.sched_barrier(0)


            _barrier(vmcnt=6, lgkmcnt=15)
            rocdl.sched_barrier(0)
            
            for dp in range_constexpr(N_DP_DS_WRITE_INTER):
                dp_inner = dp + N_DP_LOOP
                v_lo = vector.shuffle(
                    v_buf[dp_inner], v_buf[dp_inner], [0, 1, 2, 3])
                v_hi = vector.shuffle(
                    v_buf[dp_inner], v_buf[dp_inner], [4, 5, 6, 7])
                o_accs[dp_inner * 2] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_lo, p_pack_b, o_accs[dp_inner * 2], 0, 0, 0],
                )
                o_accs[dp_inner * 2 + 1] = rocdl.mfma_f32_16x16x16f16(
                    v4f32_type,
                    [v_hi, p_pack_b, o_accs[dp_inner * 2 + 1], 0, 0, 0],
                )

            coop_perm_store_v_half(0, v_raw_next_h0)
            v_raw_h1_out = coop_load_v_half(1, k_cur_buf)

            rocdl.sched_mfma(1) # 30
            rocdl.sched_dswr(1)
            rocdl.sched_mfma(1) # 31
            rocdl.sched_dswr(1)
            rocdl.sched_dsrd(4)
            # rocdl.sched_barrier(0)

            return (m, l, o_accs, k_buf,
                    nn_a_phys_next, v_raw_h1_out)

        has_kv_work = _std_arith.CmpIOp(
            _std_arith.CmpIPredicate.ult,
            _raw(kv_split_start), _raw(kv_split_end),
        ).result
        page_base_0 = lookup_page_resolve(pf0_phys_i32)
        coop_load_k(page_base_0, pf0_p_off, lds_wptr_cur)

        page_base_1 = lookup_page_resolve(pf1_phys_early)
        coop_load_k(page_base_1, pf1_p_off_early, lds_wptr_next)
        # COOP_K_VMEM_OPS already defined in constants as CKV_NUM_CHUNKS * K_PAIRS_PER_WAVE
        _barrier(vmcnt=COOP_K_VMEM_OPS, lgkmcnt=0)

        v_raw_h0_init = coop_load_v_half(0, lds_k_cur_buf)
        coop_perm_store_v_half(0, v_raw_h0_init)
        v_raw_h1_init = coop_load_v_half(1, lds_k_cur_buf)
        _barrier(lgkmcnt=14)

        rocdl.sched_barrier(0)

        k_byte_base = (
            lds_base_byte_idx
            + k_read_base * arith.index(2)
        )
        k_buf_init = []
        for ks in range_constexpr(K_PREFETCH):
            _off = K_STEP_OFFSETS_F16[ks] * 2
            k_buf_init.append(_lds_load_prefer_agpr(
                k_byte_base, v8f16_type,
                static_byte_offset=_off,
            ))
        _k_dummy = arith.constant_vector(0.0, v8f16_type)
        for ks in range_constexpr(K_REMAINING):
            k_buf_init.append(_k_dummy)

        if has_kv_work:
            o_accs_init = [
                arith.constant_vector(0.0, v4f32_type)
                for _ in range_constexpr(D_CHUNKS)]
            m_init = c_neg_large
            l_init = c_zero_f

            three_tiles_end = kv_split_start + arith.index(3 * BLOCK_N)
            has_three = _std_arith.CmpIOp(
                _std_arith.CmpIPredicate.ule,
                _raw(three_tiles_end), _raw(kv_split_end),
            ).result
            body_end = _std_arith.SelectOp(
                has_three,
                _raw(kv_split_end - arith.index(2 * BLOCK_N)),
                _raw(kv_split_start),
            ).result

            four_tiles = kv_split_start + arith.index(4 * BLOCK_N)
            has_four = _std_arith.CmpIOp(
                _std_arith.CmpIPredicate.ule,
                _raw(four_tiles), _raw(kv_split_end),
            ).result
            pair_ub = _std_arith.SelectOp(
                has_four,
                _raw(kv_split_end - arith.index(3 * BLOCK_N)),
                _raw(kv_split_start),
            ).result

            rocdl.sched_barrier(0)

            nn_a_pf_start = kv_split_start + arith.index(2 * BLOCK_N)
            nn_a_pf_phys_v, _ = lookup_page_issue(nn_a_pf_start)
            nn_a_pf_phys = nn_a_pf_phys_v

            _PKB = 5 + D_CHUNKS
            _PNN = _PKB + K_STEPS
            _PVH = _PNN + 1
            _PWP = _PVH + 8
            pair_init = ([_raw(kv_split_start), _raw(m_init), _raw(l_init)]
                         + [_raw(v) for v in o_accs_init]
                         + [lds_k_cur_buf, lds_k_next_buf]
                         + [_raw(v) for v in k_buf_init]
                         + [_raw(nn_a_pf_phys)]
                         + _vh_flatten(v_raw_h1_init)
                         + [lds_wptr_cur, lds_wptr_next])
            for pair_pos, pair_ia, pair_results in _scf.for_(
                _raw(kv_split_start), _raw(pair_ub),
                _raw(arith.index(2 * BLOCK_N)),
                iter_args=pair_init
            ):
                m_pair = pair_ia[1]
                l_pair = pair_ia[2]
                o_pair = [
                    pair_ia[3 + dc]
                    for dc in range_constexpr(D_CHUNKS)
                ]
                k_pair_cur = pair_ia[3 + D_CHUNKS]
                k_pair_next = pair_ia[4 + D_CHUNKS]
                k_pair_buf = [
                    pair_ia[_PKB + ks]
                    for ks in range_constexpr(K_STEPS)
                ]
                nn_a_p_in = pair_ia[_PNN]
                v_raw_h1_pair = _vh_unflatten(pair_ia, _PVH)
                wptr_pair_cur = pair_ia[_PWP]
                wptr_pair_next = pair_ia[_PWP + 1]

                (m_pr, l_pr, o_pr, kbuf_pr,
                 nn_a_p_out, v_raw_h1_pair_out) = \
                    do_kv_pair(
                        pair_pos, m_pair, l_pair, o_pair,
                        k_pair_cur, k_pair_next,
                        k_pair_buf,
                        wptr_cur=wptr_pair_cur,
                        wptr_next=wptr_pair_next,
                        nn_a_phys_in=nn_a_p_in,
                        v_raw_h1_in=v_raw_h1_pair,
                    )
                next_pp = pair_pos + arith.index(2 * BLOCK_N)
                yield ([_raw(next_pp), _raw(m_pr), _raw(l_pr)]
                       + [_raw(v) for v in o_pr]
                       + [k_pair_cur, k_pair_next]
                       + [_raw(v) for v in kbuf_pr]
                       + [_raw(nn_a_p_out)]
                       + _vh_flatten(v_raw_h1_pair_out)
                       + [wptr_pair_cur, wptr_pair_next])

            pair_end = pair_results[0]
            m_from_pair = pair_results[1]
            l_from_pair = pair_results[2]
            o_from_pair = [
                pair_results[3 + dc]
                for dc in range_constexpr(D_CHUNKS)
            ]
            k_pc = pair_results[3 + D_CHUNKS]
            k_pn = pair_results[4 + D_CHUNKS]
            kbuf_fp = [
                pair_results[_PKB + ks]
                for ks in range_constexpr(K_STEPS)
            ]
            v_raw_h1_from_pair = _vh_unflatten(pair_results, _PVH)
            wptr_from_pair_cur = pair_results[_PWP]
            wptr_from_pair_next = pair_results[_PWP + 1]

            _KB = 4 + D_CHUNKS
            _VH = _KB + K_STEPS
            _BWP = _VH + 8
            body_init = ([m_from_pair, l_from_pair] + list(o_from_pair)
                         + [k_pc, k_pn] + list(kbuf_fp)
                         + _vh_flatten(v_raw_h1_from_pair)
                         + [wptr_from_pair_cur, wptr_from_pair_next])
            for kv_pos, body_ia, body_results in _scf.for_(
                pair_end, _raw(body_end), _raw(arith.index(BLOCK_N)),
                iter_args=body_init
            ):
                m_running = body_ia[0]
                l_running = body_ia[1]
                o_accs = [
                    body_ia[2 + dc]
                    for dc in range_constexpr(D_CHUNKS)
                ]
                k_cur_buf = body_ia[2 + D_CHUNKS]
                k_next_buf = body_ia[3 + D_CHUNKS]
                k_buf_loop = [
                    body_ia[_KB + ks]
                    for ks in range_constexpr(K_STEPS)
                ]
                v_raw_h1_loop = _vh_unflatten(body_ia, _VH)
                wptr_body_cur = body_ia[_BWP]
                wptr_body_next = body_ia[_BWP + 1]

                m_new, l_new, o_accs, k_buf_new, v_raw_h1_body_out = \
                    do_kv_tile(
                        kv_pos, m_running, l_running, o_accs,
                        k_cur_buf, k_next_buf, k_buf_loop,
                        wptr_cur=wptr_body_cur, wptr_next=wptr_body_next,
                        v_raw_h1_in=v_raw_h1_loop,
                    )
                yield ([_raw(m_new), _raw(l_new)]
                       + [_raw(v) for v in o_accs]
                       + [k_next_buf, k_cur_buf]
                       + [_raw(v) for v in k_buf_new]
                       + _vh_flatten(v_raw_h1_body_out)
                       + [wptr_body_next, wptr_body_cur])

            m_body = body_results[0]
            l_body = body_results[1]
            o_body = [
                body_results[2 + dc]
                for dc in range_constexpr(D_CHUNKS)
            ]
            k_body_cur = body_results[2 + D_CHUNKS]
            k_body_next = body_results[3 + D_CHUNKS]
            k_bufody = [
                body_results[_KB + ks]
                for ks in range_constexpr(K_STEPS)
            ]
            v_raw_h1_from_body = _vh_unflatten(body_results, _VH)
            wptr_from_body_cur = body_results[_BWP]
            wptr_from_body_next = body_results[_BWP + 1]

            two_tiles_end = kv_split_start + arith.index(2 * BLOCK_N)
            has_two = _std_arith.CmpIOp(
                _std_arith.CmpIPredicate.ule,
                _raw(two_tiles_end), _raw(kv_split_end),
            ).result
            penult_ub = _std_arith.SelectOp(
                has_two,
                _raw(kv_split_end - arith.index(BLOCK_N)),
                _raw(body_end),
            ).result
            _PLWP = _VH + 8
            penult_init = ([m_body, l_body] + list(o_body)
                           + [k_body_cur, k_body_next]
                           + list(k_bufody)
                           + _vh_flatten(v_raw_h1_from_body)
                           + [wptr_from_body_cur, wptr_from_body_next])
            for kv_pos_p, penult_ia, penult_results in _scf.for_(
                _raw(body_end), penult_ub, _raw(arith.index(BLOCK_N)),
                iter_args=penult_init
            ):
                m_p = penult_ia[0]
                l_p = penult_ia[1]
                o_p = [
                    penult_ia[2 + dc]
                    for dc in range_constexpr(D_CHUNKS)
                ]
                k_p_cur = penult_ia[2 + D_CHUNKS]
                k_p_next = penult_ia[3 + D_CHUNKS]
                k_buf_p = [
                    penult_ia[_KB + ks]
                    for ks in range_constexpr(K_STEPS)
                ]
                v_raw_h1_p = _vh_unflatten(penult_ia, _VH)
                wptr_p_cur = penult_ia[_PLWP]
                wptr_p_next = penult_ia[_PLWP + 1]

                m_pn, l_pn, o_pn, k_buf_pn, v_raw_h1_pn_out = do_kv_tile(
                    kv_pos_p, m_p, l_p, o_p,
                    k_p_cur, k_p_next, k_buf_p,
                    wptr_cur=wptr_p_cur, wptr_next=wptr_p_next,
                    v_raw_h1_in=v_raw_h1_p,
                    load_knn=False, do_vt=True,
                    do_causal=True,
                )
                yield ([_raw(m_pn), _raw(l_pn)]
                       + [_raw(v) for v in o_pn]
                       + [k_p_next, k_p_cur]
                       + [_raw(v) for v in k_buf_pn]
                       + _vh_flatten(v_raw_h1_pn_out)
                       + [wptr_p_next, wptr_p_cur])

            m_penult = penult_results[0]
            l_penult = penult_results[1]
            o_penult = [
                penult_results[2 + dc]
                for dc in range_constexpr(D_CHUNKS)
            ]
            k_last_cur = penult_results[2 + D_CHUNKS]
            k_last_next = penult_results[3 + D_CHUNKS]
            k_buf_last = [
                penult_results[_KB + ks]
                for ks in range_constexpr(K_STEPS)
            ]
            v_raw_h1_from_penult = _vh_unflatten(penult_results, _VH)
            wptr_from_penult_cur = penult_results[_PLWP]
            wptr_from_penult_next = penult_results[_PLWP + 1]

            _LLWP = _VH + 8
            last_init = ([m_penult, l_penult] + list(o_penult)
                         + [k_last_cur, k_last_next]
                         + list(k_buf_last)
                         + _vh_flatten(v_raw_h1_from_penult)
                         + [wptr_from_penult_cur, wptr_from_penult_next])
            for kv_pos_l, last_ia, last_results in _scf.for_(
                penult_ub, _raw(kv_split_end), _raw(arith.index(BLOCK_N)),
                iter_args=last_init
            ):
                m_l = last_ia[0]
                l_l = last_ia[1]
                o_l = [
                    last_ia[2 + dc]
                    for dc in range_constexpr(D_CHUNKS)
                ]
                k_l_cur = last_ia[2 + D_CHUNKS]
                k_l_next = last_ia[3 + D_CHUNKS]
                k_buf_l = [
                    last_ia[_KB + ks]
                    for ks in range_constexpr(K_STEPS)
                ]
                v_raw_h1_l = _vh_unflatten(last_ia, _VH)
                wptr_l_cur = last_ia[_LLWP]
                wptr_l_next = last_ia[_LLWP + 1]

                m_ln, l_ln, o_ln, k_buf_ln, v_raw_h1_ln_out = do_kv_tile(
                    kv_pos_l, m_l, l_l, o_l,
                    k_l_cur, k_l_next, k_buf_l,
                    wptr_cur=wptr_l_cur, wptr_next=wptr_l_next,
                    v_raw_h1_in=v_raw_h1_l,
                    load_knn=False, do_vt=False,
                    do_causal=True,
                )
                yield ([_raw(m_ln), _raw(l_ln)]
                       + [_raw(v) for v in o_ln]
                       + [k_l_next, k_l_cur]
                       + [_raw(v) for v in k_buf_ln]
                       + _vh_flatten(v_raw_h1_ln_out)
                       + [wptr_l_next, wptr_l_cur])

            m_final = last_results[0]
            l_partial = last_results[1]
            o_finals = [
                last_results[2 + dc]
                for dc in range_constexpr(D_CHUNKS)
            ]

            red_wr_idx_epi = (
                wave_id * arith.index(64)
                + lane_mod_16 * arith.index(4)
                + lane_div_16
            )
            _memref.StoreOp(l_partial, lds_red, [_raw(red_wr_idx_epi)])
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            red_rd_base_epi = (
                wave_id * arith.index(64)
                + lane_mod_16 * arith.index(4)
            )
            l_vec = vector.load_op(v4f32_type, lds_red,
                                    [_raw(red_rd_base_epi)])
            l_final = c_zero_f
            for g in range_constexpr(4):
                l_g = vector.extract(
                    l_vec, static_position=[g], dynamic_position=[],
                )
                l_final = _std_arith.AddFOp(
                    _raw(l_final), _raw(l_g), fastmath=fm_fast
                ).result
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            c_eps = arith.constant(1.0e-30, type=compute_type)
            l_safe = _std_arith.MaximumFOp(
                _raw(l_final), _raw(c_eps),
            ).result
            l_rcp = _std_arith.DivFOp(
                _raw(c_one_f), _raw(l_safe),
                fastmath=fm_fast,
            ).result
            inv_l_vec = vector.broadcast(v4f32_type, l_rcp)

            m_final_scaled = _std_arith.MulFOp(
                _raw(m_final), _raw(c_sm_scale), fastmath=fm_fast
            ).result
            log_l = _math_log(l_final, fastmath=fm_fast)
            lse_idx = mid_lse_global_idx(q_row)
            lse_off_i32 = _index_cast_to_i32(lse_idx)

            o_scaled = []
            for dc in range_constexpr(D_CHUNKS):
                o_scaled.append(_std_arith.MulFOp(
                    _raw(o_finals[dc]), _raw(inv_l_vec),
                    fastmath=fm_fast,
                ).result)

            _barrier(vmcnt=0, lgkmcnt=0)

            reshape_wb = wave_id * arith.index(RESHAPE_WAVE)
            total_q_epi = batch_idx * seqlen_q_v + q_row

            for rbatch in range_constexpr(RESHAPE_BATCHES):
                for pair_l in range_constexpr(RESHAPE_CPB // 2):
                    dp = rbatch * (RESHAPE_CPB // 2) + pair_l
                    dc_even = dp * 2
                    dc_odd = dp * 2 + 1
                    v4_lo = vector.shuffle(
                        o_scaled[dc_even], o_scaled[dc_odd],
                        [0, 4, 1, 5])
                    v4_hi = vector.shuffle(
                        o_scaled[dc_even], o_scaled[dc_odd],
                        [2, 6, 3, 7])
                    wr_base = (
                        reshape_wb
                        + lane_mod_16 * arith.index(RESHAPE_ROW)
                        + arith.index(pair_l * 32)
                        + lane_div_16 * arith.index(8)
                    )
                    vector.store(
                        v4_lo, lds_reshape, [_raw(wr_base)])
                    vector.store(
                        v4_hi, lds_reshape,
                        [_raw(wr_base + arith.index(4))])

                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

                for rp in range_constexpr(2):
                    for si in range_constexpr(2):
                        rd_idx = (
                            reshape_wb
                            + (arith.index(rp * 8) + lane_div_8)
                                * arith.index(RESHAPE_ROW)
                            + arith.index(si * 32)
                            + lane_mod_8 * arith.index(4)
                        )
                        rd_val = vector.load_op(
                            v4f32_type, lds_reshape,
                            [_raw(rd_idx)])
                        new_head = (
                            head_group_idx
                                * arith.index(HEADS_PER_WAVE)
                            + arith.index(rp * 8)
                            + lane_div_8
                        )
                        d_col = (
                            arith.index(
                                rbatch * RESHAPE_CPB * 16
                                + si * 32)
                            + lane_mod_8 * arith.index(4)
                        )
                        g_idx = (
                            total_q_epi * stride_mid_o_token
                            + split_id * arith.index(
                                NUM_Q_HEADS * HEAD_DIM_V)
                            + new_head * arith.index(HEAD_DIM_V)
                            + d_col
                        )
                        o_off_i32 = _index_cast_to_i32(g_idx)
                        buffer_ops.buffer_store(
                            rd_val, mid_o_rsrc, o_off_i32)

            lse_val = _std_arith.AddFOp(
                _raw(m_final_scaled), _raw(log_l), fastmath=fm_fast
            ).result
            buffer_ops.buffer_store(lse_val, mid_lse_rsrc, lse_off_i32)

    # ── Host launcher ──
    @flyc.jit
    def launch_mla_decode(
        Q: fx.Tensor,
        KV: fx.Tensor,
        Mid_O: fx.Tensor,
        Mid_lse: fx.Tensor,
        block_table: fx.Tensor,
        batch_size: fx.Int32,
        seqlen_q: fx.Int32,
        kv_indptr: fx.Tensor,
        max_num_blocks: fx.Int32,
        num_kv_splits: fx.Int32,
        stream: fx.Stream,
    ):
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        allocator_vt.finalized = False
        allocator_red.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()
            allocator_vt.finalize()
            allocator_red.finalize()

        bs_val = arith.index_cast(T.index, batch_size.ir_value())
        c_nhg = arith.index(NUM_HEAD_GROUPS)
        nkv_splits_val = arith.index_cast(T.index, num_kv_splits.ir_value())

        mla_decode_kernel(
            Q, KV, Mid_O, Mid_lse, block_table,
            batch_size, seqlen_q, kv_indptr, max_num_blocks, num_kv_splits,
        ).launch(
            grid=(bs_val, c_nhg, nkv_splits_val),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_mla_decode
