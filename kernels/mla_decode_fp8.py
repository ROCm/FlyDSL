"""MLA (Multi-Latent Attention) decode kernel — FP8 (e4m3fnuz on gfx942) variant.

Optimized for short query sequences (seqlen_q <= 4) with GQA.
Implements absorbed MLA where the KV cache stores [c_kv || k_rope] per token.

Key design:
- MFMA mfma_f32_16x16x32_fp8_fp8 for both GEMM stages.
- 4 waves per workgroup, each wave handles one seqlen_q position.
  Each MFMA tile computes 16 Q heads simultaneously (lane_mod_16 = head).
- No if-else boundary checks for KV loads (assumes seqlen_kv % BLOCK_N == 0).
- Online softmax in registers; P converted to fp8 via v_cvt_pk_fp8_f32.
- Three LDS buffers: two for K (double-buffered prefetch), one for V^T.
  V is NOT reloaded from global memory — it is read directly from the
  current K buffer in LDS (since c_kv serves as both K_nope and V),
  transposed in registers, and written to the V^T buffer.
- BLOCK_N=32 (32 KV tokens per tile), N_KV_SUBTILES=2 for GEMM1.
- Paged KV cache via block_table.

GEMM1: S[16×32] = K @ Q^T  (2 subtiles of 16×16, K_STEPS=18 per subtile)
GEMM2: O^T = V^T @ P       (mfma_f32_16x16x32_fp8_fp8, PV_K_STEPS=1)

Layout (1D flattened):
  Q       : [batch, seqlen_q,  num_q_heads,  HEAD_DIM_QK]  (fp8)
  KV      : [num_physical_blocks, page_block_size, num_kv_heads, HEAD_DIM_QK]  (fp8)
  Mid_O   : [batch*seqlen_q, num_kv_splits, num_q_heads, HEAD_DIM_V]  (fp32)
  Mid_lse : [batch*seqlen_q, num_kv_splits, num_q_heads]              (fp32)
  block_table : [batch, max_num_blocks]   (i32 physical block ids)

Grid:  (batch, num_head_groups, num_kv_splits)
Block: (256,) -- 4 waves of 64

LDS layout (byte offsets, i8 element):
  K_buf_0 : [0,                     K_BUF_BYTES)
  K_buf_1 : [K_BUF_BYTES,           2*K_BUF_BYTES)
  V^T_buf : [2*K_BUF_BYTES,         2*K_BUF_BYTES + VT_BUF_BYTES)

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
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects import math as _math
from flydsl._mlir.dialects import arith as _std_arith

from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

def _vt_swizzle(byte_idx):
    """XOR swizzle on VT byte index to avoid LDS bank conflicts.

    Mask 0x78 XORs bits [6:3] with bits [10:7] of the address.
    For ds_read_b128: read addresses are 16-byte aligned (bits 0-3 = 0),
    so (addr+8)>>4 == addr>>4 → swizzle(addr+8) = swizzle(addr)+8, which
    preserves stride-8 contiguity within b128 reads.
    Formula: swizzled = addr XOR ((addr >> 4) & 0x78).
    """
    raw = _raw(byte_idx) if not isinstance(byte_idx, ir.Value) else byte_idx
    i32_ty = ir.IntegerType.get_signless(32)
    idx32 = _std_arith.IndexCastOp(i32_ty, raw).result
    c4 = _std_arith.ConstantOp(i32_ty, ir.IntegerAttr.get(i32_ty, 4)).result
    c0x78 = _std_arith.ConstantOp(i32_ty, ir.IntegerAttr.get(i32_ty, 0x78)).result
    shifted = _std_arith.ShRUIOp(idx32, c4).result
    masked = _std_arith.AndIOp(shifted, c0x78).result
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


def _fast_exp2(val):
    """Bare v_exp_f32 — softmax inputs are bounded, range reduction unnecessary."""
    return llvm.InlineAsmOp(
        res=ir.F32Type.get(),
        operands_=[_raw(val)],
        asm_string="v_exp_f32 $0, $1",
        constraints="=v,v",
        has_side_effects=False,
        is_align_stack=False,
    ).result


def _math_log(val, fastmath=None):
    """Wrap math.LogOp, returning .result."""
    kw = {}
    if fastmath is not None:
        kw["fastmath"] = fastmath
    return _math.LogOp(_raw(val), **kw).result


def compile_mla_decode_fp8(
    num_q_heads=16,
    num_kv_heads=1,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    page_block_size=64,
    causal=True,
    sm_scale=None,
):
    gpu_arch = get_hip_arch()

    BLOCK_N = 32
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
    assert PAGE_BLOCK_SIZE % BLOCK_N == 0, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be a multiple of BLOCK_N ({BLOCK_N})"
    )
    assert PAGE_BLOCK_SIZE <= 64, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be <= 64"
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    K_STEPS = HEAD_DIM_QK // 32  # 18 (each fp8 MFMA has K=32)
    N_KV_SUBTILES = BLOCK_N // 16  # 2
    D_CHUNKS = HEAD_DIM_V // 16    # 32
    PV_K_STEPS = 1                 # mfma K=32 covers all 32 tokens at once

    RESHAPE_CPB = 4
    RESHAPE_BATCHES = D_CHUNKS // RESHAPE_CPB
    RESHAPE_ROW = RESHAPE_CPB * 16 + 4  # 68 f32 per head row (with 4-element bank-conflict padding)
    RESHAPE_WAVE = 16 * RESHAPE_ROW     # 1088 f32 per wave
    RESHAPE_TOTAL = NUM_WAVES * RESHAPE_WAVE  # 4352 f32

    STRIDE_TOKEN_Q = NUM_Q_HEADS * HEAD_DIM_QK
    STRIDE_TOKEN_KV = NUM_KV_HEADS * HEAD_DIM_QK
    STRIDE_PAGE = PAGE_BLOCK_SIZE * STRIDE_TOKEN_KV

    elem_bytes = 1  # fp8
    CKV_CHUNK_COLS = 64
    CKV_NUM_CHUNKS = HEAD_DIM_QK // CKV_CHUNK_COLS  # 9
    assert HEAD_DIM_QK % CKV_CHUNK_COLS == 0
    # Each chunk stores 4 fp8 tokens × 64 elements × 1 byte = 256 bytes
    CKV_TOKENS_PER_HALF = 4
    CKV_CHUNK_BYTES = CKV_CHUNK_COLS * CKV_TOKENS_PER_HALF * elem_bytes  # 256
    CKV_HALF_BYTES = CKV_NUM_CHUNKS * CKV_CHUNK_BYTES  # 2304
    CKV_PAD_BYTES = 32
    CKV_WAVE_BYTES = CKV_HALF_BYTES * 2 + CKV_PAD_BYTES  # 4640
    CKV_TOKEN_BYTES = CKV_CHUNK_COLS * elem_bytes  # 64 (stride between tokens in a chunk)
    K_SEQ_STRIDE = NUM_WAVES  # 4
    K_PAIRS_PER_WAVE = 2  # g0 and g1
    COOP_K_VMEM_OPS = CKV_NUM_CHUNKS * K_PAIRS_PER_WAVE  # 18

    K_CHUNKS = CKV_NUM_CHUNKS         # 9 (each chunk = 2 MFMA steps)
    K_PREFETCH_CHUNKS = 8             # prefetch 8 sub0 chunks in prev GEMM2
    K_REMAINING_CHUNKS = K_CHUNKS - K_PREFETCH_CHUNKS  # 1
    K_PREFETCH = K_PREFETCH_CHUNKS * 2  # 16 steps
    K_REMAINING = K_STEPS - K_PREFETCH  # 2

    _GEMM1_HALF = K_STEPS             # s_accs[1] phase has all steps for KV lo interleave
    K_LO_CHUNKS = min(5, _GEMM1_HALF // K_PAIRS_PER_WAVE)
    K_HI0_CHUNKS = min(2, CKV_NUM_CHUNKS - K_LO_CHUNKS)
    K_HI1_CHUNKS = CKV_NUM_CHUNKS - K_LO_CHUNKS - K_HI0_CHUNKS
    K_LO_OPS = K_LO_CHUNKS * K_PAIRS_PER_WAVE
    K_HI0_OPS = K_HI0_CHUNKS * K_PAIRS_PER_WAVE
    K_HI1_OPS = K_HI1_CHUNKS * K_PAIRS_PER_WAVE

    K_CHUNK_OFFSETS = [kc * CKV_CHUNK_BYTES for kc in range(K_CHUNKS)]

    # K_STEP byte offsets within the token's data across chunks
    # Each K_STEP: 8 bytes per lane; 2 K_STEPS per chunk (64 bytes per token per chunk)
    K_STEP_OFFSETS_BYTES = [
        (ks // 2) * CKV_CHUNK_BYTES + (ks % 2) * 32
        for ks in range(K_STEPS)
    ]

    # M0 stride per chunk for buffer_load_lds: offset param adds to both global and LDS
    # addresses, so M0 stride = CKV_CHUNK_BYTES - K_COL_OFFSET_STRIDE
    K_COL_OFFSET_STRIDE = CKV_CHUNK_COLS * elem_bytes  # 64 for fp8, 128 for f16
    K_LDS_M0_STRIDE = CKV_CHUNK_BYTES - K_COL_OFFSET_STRIDE  # 192 for fp8

    # VT buffer layout:
    # Write: addr = lane*16 + wave*VT_WAVE_STRIDE + dp*VT_DP_STRIDE + call*VT_CALL_STRIDE
    # Read:  addr = lane*16 + n*VT_READ_STRIDE  (no swizzle, no wave component)
    VT_WAVE_STRIDE = 2048   # bytes per wave in VT buffer
    VT_DP_STRIDE = 1024     # bytes between dim_pair 0 and dim_pair 1
    VT_CALL_STRIDE = VT_WAVE_STRIDE * NUM_WAVES  # 8192 = bytes between call 0 and call 1
    VT_READ_STRIDE = 1024   # bytes between consecutive ds_read_b128
    VT_BUF_BYTES = 2 * VT_CALL_STRIDE  # 16384 = 16KB

    K_BUF_BYTES = CKV_WAVE_BYTES * NUM_WAVES  # total K buffer in bytes
    K_BUF_ELEMS = K_BUF_BYTES  # 1 byte per fp8 element
    VT_BUF_ELEMS = VT_BUF_BYTES
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
    Q_CALLS_PER_WAVE = HEADS_PER_WAVE // 4
    Q_MAX_BATCH = KV_LDS_SIZE // Q_TILE_ELEMS

    LDS_SIZE = max(KV_LDS_SIZE, Q_TILE_ELEMS)

    assert HEAD_DIM_V % 16 == 0 and HEAD_DIM_V <= BLOCK_SIZE * 2

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    allocator_vt = SmemAllocator(None, arch=gpu_arch, global_sym_name="smemvt")
    allocator_red = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_red")

    # ── LDS sizing (pure Python, no MLIR ops) ──
    lds_k_cur_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_k_cur_offset + K_BUF_BYTES

    lds_k_next_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_k_next_offset + K_BUF_BYTES

    lds_vt_offset = allocator_vt._align(allocator_vt.ptr, 16)
    allocator_vt.ptr = lds_vt_offset + VT_BUF_BYTES

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
        lds_elem_type = ir.IntegerType.get_signless(8)  # LDS indexed in bytes

        v4f32_type = ir.VectorType.get([4], compute_type)
        v2f32_type = ir.VectorType.get([2], compute_type)
        i32_type = ir.IntegerType.get_signless(32)
        v4i32_type = ir.VectorType.get([4], i32_type)
        v2i32_type = ir.VectorType.get([2], i32_type)
        v1i32_type = ir.VectorType.get([1], i32_type)
        i64_type = ir.IntegerType.get_signless(64)
        v2i64_type = ir.VectorType.get([2], i64_type)

        def _mfma_fp8(result_type, operands, **kw):
            return rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, operands, **kw)

        batch_size_v = arith.index_cast(T.index, batch_size.ir_value())
        seqlen_q_v = arith.index_cast(T.index, seqlen_q.ir_value())
        max_num_blocks_v = arith.index_cast(T.index, max_num_blocks.ir_value())
        num_kv_splits_v_raw = arith.index_cast(T.index, num_kv_splits.ir_value())

        base_ptr = allocator_pong.get_base()
        base_ptr1 = allocator_ping.get_base()
        base_ptr_vt = allocator_vt.get_base()
        base_ptr_red = allocator_red.get_base()

        lds_k_cur_buf = SmemPtr(
            base_ptr, lds_k_cur_offset, lds_elem_type, shape=(K_BUF_ELEMS,)
        ).get()
        lds_k_next_buf = SmemPtr(
            base_ptr1, lds_k_next_offset, lds_elem_type, shape=(K_BUF_ELEMS,)
        ).get()
        lds_vt = SmemPtr(
            base_ptr_vt, lds_vt_offset, lds_elem_type, shape=(VT_BUF_ELEMS,)
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
        # For fp8: 4 tokens per half, k_seq_group(0-3) = token within half
        k_read_base_sub0 = (
            k_seq_wave * arith.index(CKV_WAVE_BYTES)
            + k_seq_group * arith.index(CKV_TOKEN_BYTES)
            + lane_div_16 * arith.index(16)
        )
        k_read_base_sub1 = k_read_base_sub0 + arith.index(CKV_HALF_BYTES)

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

        # ---- V^T transpose decomposition (fp8 byte-level perm) ----
        c_perm0 = arith.constant(0x05010400, type=i32_type)
        c_perm1 = arith.constant(0x07030602, type=i32_type)
        c_perm2 = arith.constant(0x05040100, type=i32_type)
        c_perm3 = arith.constant(0x07060302, type=i32_type)

        # ---- Global index helpers ----
        def q_global_idx(q_row, d_col):
            token = batch_idx * seqlen_q_v + q_row
            return (
                token * arith.index(STRIDE_TOKEN_Q)
                + q_head_idx * arith.index(HEAD_DIM_QK)
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
            arith.constant(j * (CKV_CHUNK_COLS * elem_bytes), type=T.i32)
            for j in range(CKV_NUM_CHUNKS)
        ]

        lane_mod_16_for_k = lane % arith.index(16)
        lane_div_16_for_k = lane / arith.index(16)

        _wave_id_i32 = _index_cast_to_i32(wave_id)
        _wave_id_sgpr = rocdl.readfirstlane(i32_type, _wave_id_i32)
        _ckv_wave_bytes_c = arith.constant(CKV_WAVE_BYTES, type=T.i32)
        _wave_offset_sgpr = _std_arith.MulIOp(
            _raw(_wave_id_sgpr), _raw(_ckv_wave_bytes_c),
        ).result

        def _precompute_wave_base_i32(k_buf):
            """LDS wave base as i32 SGPR — no GEP, stays scalar."""
            lds_base = _memref.ExtractAlignedPointerAsIndexOp(k_buf).result
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

        # fp8: 4 tokens per g0 (lane_div_16 = 0..3), 4 more per g1
        C_VOFF_G1_DELTA = CKV_TOKENS_PER_HALF * K_SEQ_STRIDE * STRIDE_TOKEN_KV * elem_bytes
        _voff_g0_const = (
            (wave_id * arith.index(STRIDE_TOKEN_KV) + kv_head_offset)
            * arith.index(elem_bytes)
            + lane_div_16_for_k
            * arith.index(K_SEQ_STRIDE * STRIDE_TOKEN_KV * elem_bytes)
            + lane_mod_16_for_k * arith.index(4)
        )
        _voff_g0_const_i32 = _index_cast_to_i32(_voff_g0_const)
        _voff_g1_const_i32 = _std_arith.AddIOp(
            _raw(_voff_g0_const_i32),
            _raw(arith.constant(C_VOFF_G1_DELTA, type=T.i32)),
        ).result

        def _coop_load_k_setup(kv_page_base, page_off, lds_wave_base_i32):
            page_byte_base = (
                kv_page_base + page_off * arith.index(STRIDE_TOKEN_KV)
            ) * arith.index(elem_bytes)
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
                _raw(arith.constant(half_offset + j * K_LDS_M0_STRIDE, type=T.i32)),
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

        # ---- V transpose: K_buf(LDS) → V^T_buf(LDS) (fp8 byte-level) ----
        v4i8_type = ir.VectorType.get([4], lds_elem_type)

        def _vt_col_offset_bytes(vt_dim_quad_val):
            """Dimension quad index → K buffer byte offset within a half."""
            vt_chunk = vt_dim_quad_val / arith.index(CKV_CHUNK_COLS)
            vt_col_in_chunk = vt_dim_quad_val % arith.index(CKV_CHUNK_COLS)
            return vt_chunk * arith.index(CKV_CHUNK_BYTES) + vt_col_in_chunk

        def _vt_perm(src_hi, src_lo, sel):
            return llvm.call_intrinsic(
                i32_type, "llvm.amdgcn.perm",
                [src_hi, src_lo, sel], [], [],
            )

        v16i8_type = ir.VectorType.get([16], lds_elem_type)

        def _vt_store_b128(data_list, byte_idx):
            """Store 4 x i32 as ds_write_b128 to VT buffer. No swizzle."""
            packed = vector.from_elements(v4i32_type, data_list)
            packed_i8 = vector.bitcast(v16i8_type, packed)
            vector.store(packed_i8, lds_vt, [_raw(byte_idx)])

        _half_bytes_const = arith.constant(CKV_HALF_BYTES, type=i32_type)
        _half_bytes_opaque = llvm.InlineAsmOp(
            res=i32_type,
            operands_=[_raw(_half_bytes_const)],
            asm_string="; half_bytes_opaque",
            constraints="=v,0",
            has_side_effects=False,
            is_align_stack=False,
        ).result
        _half_bytes_opaque_idx = _std_arith.IndexCastOp(
            ir.IndexType.get(), _half_bytes_opaque).result

        def _coop_load_v(vt_col_bytes, k_buf):
            """Load 8 dwords from K buffer for V transpose.

            token = lane_div_16 (implicit).
            src[0..3] = waves 0-3 at half 0 (same token),
            src[4..7] = waves 0-3 at half 1 (same token).

            h=1 addresses use opaque CKV_HALF_BYTES offset to prevent
            LoadStoreOptimizer from merging into ds_read2st64_b32.
            """
            src = []
            for h in range_constexpr(2):
                for s in range_constexpr(4):
                    lds_byte_idx_raw = _raw(
                        arith.index(s * CKV_WAVE_BYTES)
                        + lane_div_16 * arith.index(CKV_TOKEN_BYTES)
                        + vt_col_bytes
                    )
                    if h == 1:
                        lds_byte_idx_raw = _std_arith.AddIOp(
                            lds_byte_idx_raw,
                            _half_bytes_opaque_idx,
                        ).result
                    dw_bytes = vector.load_op(
                        v4i8_type, k_buf, [lds_byte_idx_raw])
                    dw = vector.extract(
                        vector.bitcast(v1i32_type, dw_bytes),
                        static_position=[0], dynamic_position=[],
                    )
                    src.append(dw)
            return src

        def _coop_perm_store(call_idx, src):
            """Apply V transpose perm and store via ds_write_b128.

            8 src dwords → 8 dst dwords via 2-level radix-2 perm.
            dst[0,1,2,3] = [dim_d_h0, dim_d_h1, dim_d+1_h0, dim_d+1_h1]
            dst[4,5,6,7] = [dim_d+2_h0, dim_d+2_h1, dim_d+3_h0, dim_d+3_h1]
            Write 1 (dp=0): ds_write_b128 [dst0..dst3]
            Write 2 (dp=1): ds_write_b128 [dst4..dst7]
            """
            tmp0 = _vt_perm(src[1], src[0], c_perm0)
            tmp1 = _vt_perm(src[1], src[0], c_perm1)
            tmp2 = _vt_perm(src[3], src[2], c_perm0)
            tmp3 = _vt_perm(src[3], src[2], c_perm1)

            dst0 = _vt_perm(tmp2, tmp0, c_perm2)
            dst2 = _vt_perm(tmp2, tmp0, c_perm3)
            dst4 = _vt_perm(tmp3, tmp1, c_perm2)
            dst6 = _vt_perm(tmp3, tmp1, c_perm3)

            tmp4 = _vt_perm(src[5], src[4], c_perm0)
            tmp5 = _vt_perm(src[5], src[4], c_perm1)
            tmp6 = _vt_perm(src[7], src[6], c_perm0)
            tmp7 = _vt_perm(src[7], src[6], c_perm1)

            dst1 = _vt_perm(tmp6, tmp4, c_perm2)
            dst3 = _vt_perm(tmp6, tmp4, c_perm3)
            dst5 = _vt_perm(tmp7, tmp5, c_perm2)
            dst7 = _vt_perm(tmp7, tmp5, c_perm3)

            vt_write_base = (
                lane * arith.index(16)
                + wave_id * arith.index(VT_WAVE_STRIDE)
                + arith.index(call_idx * VT_CALL_STRIDE)
            )
            _vt_store_b128([dst0, dst1, dst2, dst3], vt_write_base)
            _vt_store_b128(
                [dst4, dst5, dst6, dst7],
                vt_write_base + arith.index(VT_DP_STRIDE),
            )

        def coop_transpose_v_from_lds(k_buf):
            """Transpose V from K buffer to VT buffer.

            2 calls: each wave handles one chunk per call.
            call 0: chunk = wave_id (dims wave_id*64 .. wave_id*64+63)
            call 1: chunk = wave_id+4 (dims (wave_id+4)*64 .. +63)
            token = lane_div_16 (0..3).
            """
            for call_idx in range_constexpr(2):
                chunk = wave_id + arith.index(call_idx * 4)
                local_dim_quad = (
                    chunk * arith.index(CKV_CHUNK_COLS)
                    + lane_mod_16 * arith.index(4)
                )
                vt_col_bytes = _vt_col_offset_bytes(local_dim_quad)
                src = _coop_load_v(vt_col_bytes, k_buf)
                _coop_perm_store(call_idx, src)

        _vt_col_bytes_c0 = _vt_col_offset_bytes(
            wave_id * arith.index(CKV_CHUNK_COLS)
            + lane_mod_16 * arith.index(4)
        )
        _vt_col_bytes_c1 = _vt_col_offset_bytes(
            (wave_id + arith.index(4)) * arith.index(CKV_CHUNK_COLS)
            + lane_mod_16 * arith.index(4)
        )

        def coop_load_v_call(call_idx, k_buf):
            vt_cb = _vt_col_bytes_c0 if call_idx == 0 else _vt_col_bytes_c1
            return _coop_load_v(vt_cb, k_buf)

        def _coop_perm_only(call_idx, src):
            """Apply VT perm only, return (write_base, lo_packet, hi_packet).

            Splits perm from store so VT perm can overlap softmax
            while VT store is deferred to GEMM1 MFMAs.
            """
            tmp0 = _vt_perm(src[1], src[0], c_perm0)
            tmp1 = _vt_perm(src[1], src[0], c_perm1)
            tmp2 = _vt_perm(src[3], src[2], c_perm0)
            tmp3 = _vt_perm(src[3], src[2], c_perm1)

            dst0 = _vt_perm(tmp2, tmp0, c_perm2)
            dst2 = _vt_perm(tmp2, tmp0, c_perm3)
            dst4 = _vt_perm(tmp3, tmp1, c_perm2)
            dst6 = _vt_perm(tmp3, tmp1, c_perm3)

            tmp4 = _vt_perm(src[5], src[4], c_perm0)
            tmp5 = _vt_perm(src[5], src[4], c_perm1)
            tmp6 = _vt_perm(src[7], src[6], c_perm0)
            tmp7 = _vt_perm(src[7], src[6], c_perm1)

            dst1 = _vt_perm(tmp6, tmp4, c_perm2)
            dst3 = _vt_perm(tmp6, tmp4, c_perm3)
            dst5 = _vt_perm(tmp7, tmp5, c_perm2)
            dst7 = _vt_perm(tmp7, tmp5, c_perm3)

            vt_write_base = (
                lane * arith.index(16)
                + wave_id * arith.index(VT_WAVE_STRIDE)
                + arith.index(call_idx * VT_CALL_STRIDE)
            )
            return (vt_write_base,
                    [dst0, dst1, dst2, dst3],
                    [dst4, dst5, dst6, dst7])

        # ---- Preload Q via 64×64 tiled loop ----
        q_row = wave_id

        q_rsrc = buffer_ops.create_buffer_resource(Q)

        kv_total_blocks = batch_size_v * max_num_blocks_v
        kv_total_elems = kv_total_blocks * arith.index(STRIDE_PAGE)
        kv_size_bytes = kv_total_elems * arith.index(elem_bytes)
        c_max_records = arith.index(0xEFFFFFFE)
        kv_size_capped = _std_arith.MinUIOp(_raw(kv_size_bytes), _raw(c_max_records)).result
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, num_records_bytes=kv_size_capped
        )
        mid_o_rsrc = buffer_ops.create_buffer_resource(Mid_O)
        mid_lse_rsrc = buffer_ops.create_buffer_resource(Mid_lse)
        lds_base_byte_idx = _memref.ExtractAlignedPointerAsIndexOp(lds_k_cur_buf).result
        c_dword_sz = arith.constant(4, type=T.i32)
        aux = arith.constant(0, type=T.i32)

        q_packs = [None] * K_STEPS

        q_voff_base_bytes = (
            batch_idx
            * seqlen_q_v
            * arith.index(STRIDE_TOKEN_Q * elem_bytes)
            + wave_id * arith.index(STRIDE_TOKEN_Q * elem_bytes)
            + head_group_idx
            * arith.index(HEADS_PER_WAVE * HEAD_DIM_QK * elem_bytes)
            + lane_div_16 * arith.index(HEAD_DIM_QK * elem_bytes)
            + lane_mod_16 * arith.index(4)
        )

        Q_LDS_QUAD_BYTES = 4 * Q_TILE_DIM * elem_bytes
        Q_GLOBAL_QUAD_BYTES = 4 * HEAD_DIM_QK * elem_bytes
        Q_CALL_SOFFSETS = [
            arith.constant(
                c * (Q_GLOBAL_QUAD_BYTES - Q_LDS_QUAD_BYTES),
                type=i32_type,
            )
            for c in range(Q_CALLS_PER_WAVE)
        ]
        Q_CALL_INST_OFFSETS = [
            arith.constant(c * Q_LDS_QUAD_BYTES, type=i32_type)
            for c in range(Q_CALLS_PER_WAVE)
        ]

        Q_TILE_BYTES = Q_TILE_ELEMS * elem_bytes

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
                    * arith.index(HEADS_PER_WAVE * Q_TILE_DIM * elem_bytes)
                )
                tile_lds_i64 = arith.index_cast(T.i64, tile_lds_byte)
                tile_lds_scalar = rocdl.readfirstlane(
                    T.i64, tile_lds_i64
                )
                tile_lds_ptr = _inttoptr_lds(tile_lds_scalar)

                q_voff_byte = (
                    q_voff_base_bytes
                    + arith.index(d_tile * Q_TILE_DIM * elem_bytes)
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
                    + arith.index(t * Q_TILE_BYTES)
                    + (wave_id
                       * arith.index(HEADS_PER_WAVE)
                       + lane_mod_16)
                    * arith.index(Q_TILE_DIM * elem_bytes)
                    + lane_div_16 * arith.index(16)
                )
                q_b128 = _lds_load_prefer_agpr(
                    q_tile_byte_base, v2i64_type,
                )
                q_packs[d_tile * 2] = vector.extract(
                    q_b128, static_position=[0], dynamic_position=[],
                )
                q_packs[d_tile * 2 + 1] = vector.extract(
                    q_b128, static_position=[1], dynamic_position=[],
                )

            gpu.barrier()

        # ---- Constants ----
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_neg_large = arith.constant(-1e6, type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_sm_scale = arith.constant(sm_scale, type=compute_type)
        c_sm_scale_log2e = arith.constant(sm_scale * 1.4426950408889634, type=compute_type)
        c_zero_v4f32 = arith.constant_vector(0.0, v4f32_type)

        c_zero_i32 = arith.constant(0, type=i32_type)

        def _vt_load_b128(n):
            """Load VT data for 2 MFMA tiles via ds_read_b128.

            Uses vector.load_op on lds_vt memref so LLVM can prove
            non-aliasing with buffer_load_dword lds targets (k_cur_buf/k_next_buf).
            """
            vt_idx = (
                lane * arith.index(16)
                + arith.index(n * VT_READ_STRIDE)
            )
            raw = vector.load_op(
                v16i8_type, lds_vt, [_raw(vt_idx)],
                nontemporal=True)
            return vector.bitcast(v2i64_type, raw)

        def _softmax_reduce_write_fp8(s_acc0, s_acc1,
                                      kv_pos=None, do_causal=False):
            """Extract s_vals, compute local max, ds_write to LDS."""
            s_vals = []
            for sub, s_acc in enumerate([s_acc0, s_acc1]):
                for ii in range_constexpr(4):
                    s_val = vector.extract(
                        s_acc, static_position=[ii], dynamic_position=[],
                    )
                    if CAUSAL and do_causal:
                        kv_abs = (kv_pos
                                  + arith.index(sub * 16)
                                  + lane_div_16 * arith.index(4)
                                  + arith.index(ii))
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
            for ii in range_constexpr(7):
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
            return s_vals

        def _softmax_reduce_read():
            """Issue ds_read for cross-wave maxes."""
            red_rd_base = (
                wave_id * arith.index(64)
                + lane_mod_16 * arith.index(4)
            )
            max_vec = vector.load_op(v4f32_type, lds_red, [_raw(red_rd_base)])
            return max_vec

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

        def softmax_finalize_fp8(global_max, s_vals, m_old, l_old):
            """Finalize softmax for fp8: 8 values → i64 packed P."""
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
            rescale = _fast_exp2(rescale_arg)

            p_vals = [None] * 8
            local_sum = c_zero_f
            for ii in range_constexpr(8):
                exp_arg = _std_arith.SubFOp(
                    _std_arith.MulFOp(
                        _raw(s_vals[ii]), _raw(c_sm_scale_log2e),
                        fastmath=fm_fast,
                    ).result,
                    _raw(scaled_m_new),
                    fastmath=fm_fast,
                ).result
                p = _fast_exp2(exp_arg)
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

            w0 = rocdl.cvt_pk_fp8_f32(
                i32_type, _raw(p_vals[0]), _raw(p_vals[1]), c_zero_i32, 0)
            w0 = rocdl.cvt_pk_fp8_f32(
                i32_type, _raw(p_vals[2]), _raw(p_vals[3]), w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(
                i32_type, _raw(p_vals[4]), _raw(p_vals[5]), c_zero_i32, 0)
            w1 = rocdl.cvt_pk_fp8_f32(
                i32_type, _raw(p_vals[6]), _raw(p_vals[7]), w1, 1)
            w0_i64 = _std_arith.ExtUIOp(i64_type, w0).result
            w1_i64 = _std_arith.ExtUIOp(i64_type, w1).result
            c32_i64 = _std_arith.ConstantOp(
                i64_type, ir.IntegerAttr.get(i64_type, 32)).result
            w1_shifted = _std_arith.ShLIOp(w1_i64, c32_i64).result
            p_pack = _std_arith.OrIOp(w0_i64, w1_shifted).result
            return m_new, l_new, p_pack, rescale

        def _make_rescale_vec(rescale):
            return vector.broadcast(v4f32_type, rescale)

        def _rescale_acc(acc, rescale_vec):
            return _std_arith.MulFOp(
                _raw(acc), _raw(rescale_vec),
                fastmath=fm_fast,
            ).result

        def do_kv_tile(kv_pos, m_old, l_old, o_accs,
                       k_cur_buf, k_next_buf,
                       k_sub0, k_sub1,
                       nn_phys_i32, nn_p_off,
                       wptr_cur=None,
                       gk_voff_g0=None, gk_voff_g1=None,
                       load_knn=True, do_vt=True,
                       do_causal=False):

            _gk_voff_g0 = gk_voff_g0
            _gk_voff_g1 = gk_voff_g1
            _gk_lds_ptr = wptr_cur

            # ---- GEMM0: manually pipelined (target_fp8.s:624-726 pattern) ----
            rocdl.sched_barrier(0)
            v_buf = [None] * D_CHUNKS
            s_accs = [c_zero_v4f32, c_zero_v4f32]
            v_raw_c0 = [None] * 8
            v_raw_c1 = [None] * 8

            def _mfma_sub(sub_idx, k_sub, kc, half):
                k = vector.extract(
                    k_sub[kc], static_position=[half], dynamic_position=[])
                s_accs[sub_idx] = _mfma_fp8(
                    v4f32_type,
                    [k, q_packs[kc * 2 + half],
                     s_accs[sub_idx], 0, 0, 0])

            def _vt_read(n):
                vt_b128 = _vt_load_b128(n)
                v_buf[n * 2] = vector.extract(
                    vt_b128, static_position=[0], dynamic_position=[])
                v_buf[n * 2 + 1] = vector.extract(
                    vt_b128, static_position=[1], dynamic_position=[])

            def _do_v_read(vi):
                _V_MAP = [
                    (0,0,0), (1,0,0), (0,0,1), (1,0,1),
                    (0,0,2), (1,0,2), (0,0,3), (1,0,3),
                    (0,1,0), (1,1,0), (0,1,1), (1,1,1),
                    (0,1,2), (1,1,2), (0,1,3), (1,1,3),
                ]
                ci, h, s = _V_MAP[vi]
                vt_cb = _vt_col_bytes_c0 if ci == 0 else _vt_col_bytes_c1
                lds_byte_idx = _raw(
                    arith.index(s * CKV_WAVE_BYTES)
                    + lane_div_16 * arith.index(CKV_TOKEN_BYTES)
                    + vt_cb
                )
                if h == 1:
                    lds_byte_idx = _std_arith.AddIOp(
                        lds_byte_idx,
                        _raw(arith.index(CKV_HALF_BYTES)),
                    ).result
                dw_bytes = vector.load_op(
                    v4i8_type, k_next_buf, [lds_byte_idx],
                    nontemporal=True)
                dw = vector.extract(
                    vector.bitcast(v1i32_type, dw_bytes),
                    static_position=[0], dynamic_position=[],
                )
                if ci == 0:
                    v_raw_c0[h * 4 + s] = dw
                else:
                    v_raw_c1[h * 4 + s] = dw

            # ======== Pipeline schedule (target_fp8.s:624-726) ========
            # (sub, kc, half, k_load, v_read, vt_read)
            # k_load: (sub_key, chunk) or None
            # v_read/vt_read: flat index or None
            _SCHED = [
                (0,0,0, (0,0),  0, None),
                (0,0,1, (0,1),  1, None),
                (0,1,0, (0,2),  2, None),
                (0,1,1, (0,3),  3, None),
                (0,2,0, (0,4),  4, None),
                (0,2,1, None, None, 0),
                (0,3,0, (0,5),  5, None),
                (0,3,1, None, None, 1),
                (0,4,0, (0,6),  6, None),
                (0,4,1, None, None, 2),
                (0,5,0, (0,7),  7, None),
                (0,5,1, None, None, 3),
                (0,6,0, (0,8),  8, None),
                (0,6,1, None, None, 4),
                # lgkmcnt(14) before group 14
                (0,7,0, (1,0),  9, None),
                (0,7,1, None, None, None),
                (0,8,0, (1,1), 10, None),
                (0,8,1, None, None, 5),
                (1,0,0, (1,2), 11, None),
                (1,0,1, None, None, None),
                (1,1,0, (1,3), 12, None),
                (1,1,1, None, None, 6),
                (1,2,0, (1,4), 13, None),
                (1,2,1, None, None, None),
                (1,3,0, (1,5), 14, None),
                (1,3,1, None, None, 7),
                (1,4,0, (1,6), 15, None),
                (1,4,1, None, None, 8),
                (1,5,0, (1,7), None, None),
                (1,5,1, None, None, 9),
                (1,6,0, (1,8), None, None),
                (1,6,1, None, None, 10),
                (1,7,0, None, None, None),
                (1,7,1, None, None, 11),
                (1,8,0, None, None, None),
                (1,8,1, None, None, 12),
            ]

            for _gi, (_sub, _kc, _half, _kl, _vi, _vti) \
                    in enumerate(_SCHED):
                if _gi == 14:
                    rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=14))
                _ks = k_sub0 if _sub == 0 else k_sub1
                _mfma_sub(_sub, _ks, _kc, _half)
                if _kl is not None and load_knn:
                    _ksub, _kchunk = _kl
                    if _ksub == 0:
                        _emit_k_single(
                            _gk_lds_ptr, _gk_voff_g0, _kchunk, 0)
                    else:
                        _emit_k_single(
                            _gk_lds_ptr, _gk_voff_g1, _kchunk,
                            CKV_HALF_BYTES)
                if _vi is not None and do_vt:
                    _do_v_read(_vi)
                if _vti is not None:
                    _vt_read(_vti)
                rocdl.sched_barrier(0)

            for _vti in range_constexpr(3):
                _vt_read(13 + _vti)
                rocdl.sched_barrier(0)

            # Fill None v_raw entries for do_vt=False
            if not do_vt:
                _vz = arith.constant(0, type=i32_type)
                v_raw_c0 = [_vz] * 8
                v_raw_c1 = [_vz] * 8

            # ---- Issue page lookup for NEXT iteration's tile ----
            nnn_phys_i32 = None
            nnn_p_off = None
            if load_knn:
                nnn_start = kv_pos + arith.index(3 * BLOCK_N)
                nnn_phys_i32, nnn_p_off = lookup_page_issue(nnn_start)

            rocdl.sched_barrier(0)

            # ---- Softmax phase 1: local max + ds_write ----
            s_vals = _softmax_reduce_write_fp8(
                s_accs[0], s_accs[1], kv_pos=kv_pos, do_causal=do_causal)

            # ---- VT perm col0 (fills ds_write latency) ----
            vt_base_c0 = vt_lo_c0 = vt_hi_c0 = None
            if do_vt:
                vt_base_c0, vt_lo_c0, vt_hi_c0 = _coop_perm_only(
                    0, v_raw_c0)

            # ---- Softmax phase 2: ds_read ----
            max_vec = _softmax_reduce_read()

            # ---- VT perm col1 (fills ds_read latency) ----
            vt_base_c1 = vt_lo_c1 = vt_hi_c1 = None
            if do_vt:
                vt_base_c1, vt_lo_c1, vt_hi_c1 = _coop_perm_only(
                    1, v_raw_c1)

            # ---- Softmax phase 3: collect + finalize ----
            global_max = _softmax_reduce_collect(max_vec)
            m_new, l_new, p_pack, rescale = softmax_finalize_fp8(
                global_max, s_vals, m_old, l_old)

            rescale_vec = _make_rescale_vec(rescale)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=15)
            rocdl.sched_barrier(0)
            o_accs[0] = _rescale_acc(o_accs[0], rescale_vec)
            o_accs[1] = _rescale_acc(o_accs[1], rescale_vec)
            rocdl.sched_barrier(0)
            o_accs[0] = _mfma_fp8(
                v4f32_type,
                [v_buf[0], p_pack, o_accs[0], 0, 0, 0])

            rocdl.sched_barrier(0)
            if do_vt:
                _vt_store_b128(vt_lo_c0, vt_base_c0)
            rocdl.sched_barrier(0)

            o_accs[2] = _rescale_acc(o_accs[2], rescale_vec)
            o_accs[3] = _rescale_acc(o_accs[3], rescale_vec)
            rocdl.sched_barrier(0)
            o_accs[1] = _mfma_fp8(
                v4f32_type,
                [v_buf[1], p_pack, o_accs[1], 0, 0, 0])
            o_accs[2] = _mfma_fp8(
                v4f32_type,
                [v_buf[2], p_pack, o_accs[2], 0, 0, 0])
            rocdl.sched_barrier(0)
            if do_vt:
                _vt_store_b128(
                    vt_hi_c0,
                    vt_base_c0 + arith.index(VT_DP_STRIDE))
            rocdl.sched_barrier(0)

            o_accs[4] = _rescale_acc(o_accs[4], rescale_vec)
            o_accs[5] = _rescale_acc(o_accs[5], rescale_vec)
            rocdl.sched_barrier(0)
            o_accs[3] = _mfma_fp8(
                v4f32_type,
                [v_buf[3], p_pack, o_accs[3], 0, 0, 0])
            o_accs[4] = _mfma_fp8(
                v4f32_type,
                [v_buf[4], p_pack, o_accs[4], 0, 0, 0])
            rocdl.sched_barrier(0)
            if do_vt:
                _vt_store_b128(vt_lo_c1, vt_base_c1)
            rocdl.sched_barrier(0)

            o_accs[6] = _rescale_acc(o_accs[6], rescale_vec)
            o_accs[7] = _rescale_acc(o_accs[7], rescale_vec)
            rocdl.sched_barrier(0)

            o_accs[5] = _mfma_fp8(
                v4f32_type,
                [v_buf[5], p_pack, o_accs[5], 0, 0, 0])
            o_accs[6] = _mfma_fp8(
                v4f32_type,
                [v_buf[6], p_pack, o_accs[6], 0, 0, 0])

            rocdl.sched_barrier(0)
            if do_vt:
                _vt_store_b128(
                    vt_hi_c1,
                    vt_base_c1 + arith.index(VT_DP_STRIDE))
            rocdl.sched_barrier(0)

            # ---- Rescale batch 1: o_accs[8..19] ----
            for dc in range_constexpr(12):
                o_accs[dc + 8] = _rescale_acc(o_accs[dc + 8], rescale_vec)

            rocdl.sched_barrier(0)

            # ---- K read helpers ----
            k_sub0_next = [None] * K_CHUNKS
            k_sub1_next = [None] * K_CHUNKS

            def _k_read_pair(kc):
                _ki0 = k_read_base_sub0 + arith.index(
                    K_CHUNK_OFFSETS[kc])
                k_sub0_next[kc] = vector.bitcast(
                    v2i64_type,
                    vector.load_op(
                        v16i8_type, k_next_buf, [_raw(_ki0)],
                        nontemporal=True))
                _ki1 = k_read_base_sub1 + arith.index(
                    K_CHUNK_OFFSETS[kc])
                k_sub1_next[kc] = vector.bitcast(
                    v2i64_type,
                    vector.load_op(
                        v16i8_type, k_next_buf, [_raw(_ki1)],
                        nontemporal=True))

            def _k_read_single(kc, sub):
                if sub == 0:
                    _ki = k_read_base_sub0 + arith.index(
                        K_CHUNK_OFFSETS[kc])
                    k_sub0_next[kc] = vector.bitcast(
                        v2i64_type,
                        vector.load_op(
                            v16i8_type, k_next_buf,
                            [_raw(_ki)], nontemporal=True))
                else:
                    _ki = k_read_base_sub1 + arith.index(
                        K_CHUNK_OFFSETS[kc])
                    k_sub1_next[kc] = vector.bitcast(
                        v2i64_type,
                        vector.load_op(
                            v16i8_type, k_next_buf,
                            [_raw(_ki)], nontemporal=True))

            # ---- MFMA[8] + K reads chunk 0 ----
            rocdl.sched_barrier(0)
            o_accs[7] = _mfma_fp8(
                v4f32_type,
                [v_buf[7], p_pack, o_accs[7], 0, 0, 0])
            rocdl.sched_barrier(0)
            if do_vt:
                _k_read_pair(0)
            rocdl.sched_barrier(0)

            # ---- MFMA[9] + K reads chunk 1 ----
            o_accs[8] = _mfma_fp8(
                v4f32_type,
                [v_buf[8], p_pack, o_accs[8], 0, 0, 0])
            rocdl.sched_barrier(0)
            if do_vt:
                _k_read_pair(1)
            rocdl.sched_barrier(0)

            # ---- Rescale batch 2: o_accs[20..31] ----
            for dc in range_constexpr(12):
                o_accs[dc + 20] = _rescale_acc(
                    o_accs[dc + 20], rescale_vec)

            rocdl.sched_barrier(0)

            # ---- MFMA[10] + K reads chunk 2 ----
            o_accs[9] = _mfma_fp8(
                v4f32_type,
                [v_buf[9], p_pack, o_accs[9], 0, 0, 0])
            rocdl.sched_barrier(0)
            if do_vt:
                _k_read_pair(2)
            rocdl.sched_barrier(0)

            # ---- MFMA[11] + K reads chunk 3 ----
            o_accs[10] = _mfma_fp8(
                v4f32_type,
                [v_buf[10], p_pack, o_accs[10], 0, 0, 0])
            rocdl.sched_barrier(0)
            if do_vt:
                _k_read_pair(3)
            rocdl.sched_barrier(0)

            # ---- Page resolve for NEXT iteration (fills MFMA gap) ----
            gk_voff_g0_next = None
            gk_voff_g1_next = None
            if load_knn:
                _nn_page_base_next = lookup_page_resolve(nnn_phys_i32)
                gk_voff_g0_next, gk_voff_g1_next, _ = \
                    _coop_load_k_setup(
                        _nn_page_base_next, nnn_p_off, wptr_cur)
            rocdl.sched_barrier(0)

            # ---- GEMM1 tail: 10 groups of (2 MFMAs + 1 K read) ----
            # Matches target_fp8.s:909-942 pattern
            for grp in range_constexpr(10):
                _dc0 = grp * 2
                _dc1 = grp * 2 + 1
                o_accs[_dc0 + 11] = _mfma_fp8(
                    v4f32_type,
                    [v_buf[_dc0 + 11], p_pack,
                     o_accs[_dc0 + 11], 0, 0, 0])
                o_accs[_dc1 + 11] = _mfma_fp8(
                    v4f32_type,
                    [v_buf[_dc1 + 11], p_pack,
                     o_accs[_dc1 + 11], 0, 0, 0])
                _kc = grp // 2 + 4
                _ksub = grp % 2
                if do_vt:
                    _k_read_single(_kc, _ksub)
                rocdl.sched_barrier(0)

            if not do_vt:
                _k_dummy_0 = arith.constant(0, type=i64_type)
                _k_pair_dummy = vector.from_elements(
                    v2i64_type, [_k_dummy_0, _k_dummy_0])
                k_sub0_next = [_k_pair_dummy] * K_CHUNKS
                k_sub1_next = [_k_pair_dummy] * K_CHUNKS

            if nnn_phys_i32 is None:
                nnn_phys_i32 = arith.constant(0, type=T.i32)
                nnn_p_off = arith.index(0)
            if gk_voff_g0_next is None:
                gk_voff_g0_next = arith.constant(0, type=T.i32)
                gk_voff_g1_next = arith.constant(0, type=T.i32)

            return (m_new, l_new, o_accs, k_sub0_next, k_sub1_next,
                    nnn_phys_i32, nnn_p_off,
                    gk_voff_g0_next, gk_voff_g1_next)

        has_kv_work = _std_arith.CmpIOp(
            _std_arith.CmpIPredicate.ult,
            _raw(kv_split_start), _raw(kv_split_end),
        ).result
        page_base_0 = lookup_page_resolve(pf0_phys_i32)
        coop_load_k(page_base_0, pf0_p_off, lds_wptr_cur)

        page_base_1 = lookup_page_resolve(pf1_phys_early)
        coop_load_k(page_base_1, pf1_p_off_early, lds_wptr_next)
        _barrier(vmcnt=COOP_K_VMEM_OPS, lgkmcnt=0)
        coop_transpose_v_from_lds(lds_k_cur_buf)

        # Prefetch K for tile 0 from k_cur_buf
        k_sub0_init = [None] * K_CHUNKS
        k_sub1_init = [None] * K_CHUNKS
        for kc in range_constexpr(K_CHUNKS):
            _ki0 = k_read_base_sub0 + arith.index(K_CHUNK_OFFSETS[kc])
            _k0_raw = vector.load_op(
                v16i8_type, lds_k_cur_buf, [_raw(_ki0)],
                nontemporal=True)
            k_sub0_init[kc] = vector.bitcast(v2i64_type, _k0_raw)

            _ki1 = k_read_base_sub1 + arith.index(K_CHUNK_OFFSETS[kc])
            _k1_raw = vector.load_op(
                v16i8_type, lds_k_cur_buf, [_raw(_ki1)],
                nontemporal=True)
            k_sub1_init[kc] = vector.bitcast(v2i64_type, _k1_raw)

        # Prefetch page info for tile 2
        nn_start_init = kv_split_start + arith.index(2 * BLOCK_N)
        nn_phys_init, nn_p_off_init = lookup_page_issue(nn_start_init)

        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
        _nn_pb_init = lookup_page_resolve(nn_phys_init)
        gk_voff_g0_init, gk_voff_g1_init, _ = \
            _coop_load_k_setup(_nn_pb_init, nn_p_off_init,
                               lds_wptr_cur)

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

            rocdl.sched_barrier(0)

            _KS0 = 2 + D_CHUNKS
            _KS1 = _KS0 + K_CHUNKS
            _BUF = _KS1 + K_CHUNKS
            _WP = _BUF + 2
            _PG = _WP + 2
            _GK = _PG + 2

            # -- Pair loop indices (no BUF/WP in iter_args) --
            _PG_P = _KS1 + K_CHUNKS
            _GK_P = _PG_P + 2

            # pair_end: largest multiple of 2*BLOCK_N tiles from start
            _two_bn_raw = _raw(arith.index(2 * BLOCK_N))
            _pair_range = _std_arith.SubIOp(
                _raw(body_end), _raw(kv_split_start)).result
            _num_pair_tiles = _std_arith.DivUIOp(
                _pair_range, _two_bn_raw).result
            _pair_total = _std_arith.MulIOp(
                _num_pair_tiles, _two_bn_raw).result
            pair_end_raw = _std_arith.AddIOp(
                _raw(kv_split_start), _pair_total).result

            # ============ Main PAIR loop (stride 2*BLOCK_N) ============
            pair_init = ([_raw(m_init), _raw(l_init)]
                         + [_raw(v) for v in o_accs_init]
                         + [_raw(v) for v in k_sub0_init]
                         + [_raw(v) for v in k_sub1_init]
                         + [_raw(nn_phys_init), _raw(nn_p_off_init)]
                         + [gk_voff_g0_init, gk_voff_g1_init])
            for kv_pos, pair_ia, pair_results in _scf.for_(
                _raw(kv_split_start), pair_end_raw,
                _two_bn_raw,
                iter_args=pair_init
            ):
                m_running = pair_ia[0]
                l_running = pair_ia[1]
                o_accs = [pair_ia[2 + dc]
                          for dc in range_constexpr(D_CHUNKS)]
                ks0 = [pair_ia[_KS0 + i]
                       for i in range_constexpr(K_CHUNKS)]
                ks1 = [pair_ia[_KS1 + i]
                       for i in range_constexpr(K_CHUNKS)]
                pg_phys = pair_ia[_PG_P]
                pg_poff = pair_ia[_PG_P + 1]
                gk_g0 = pair_ia[_GK_P]
                gk_g1 = pair_ia[_GK_P + 1]

                # ---- Tile A: k_cur=lds_k_cur_buf, k_next=lds_k_next_buf
                (m_a, l_a, o_a, ks0_a, ks1_a,
                 pg_a, poff_a, gk0_a, gk1_a) = do_kv_tile(
                    kv_pos, m_running, l_running, o_accs,
                    lds_k_cur_buf, lds_k_next_buf, ks0, ks1,
                    pg_phys, pg_poff,
                    wptr_cur=lds_wptr_cur,
                    gk_voff_g0=gk_g0, gk_voff_g1=gk_g1,
                )

                # ---- Tile B: k_cur=lds_k_next_buf, k_next=lds_k_cur_buf
                kv_pos_b = _std_arith.AddIOp(
                    kv_pos, _raw(arith.index(BLOCK_N))).result
                (m_b, l_b, o_b, ks0_b, ks1_b,
                 pg_b, poff_b, gk0_b, gk1_b) = do_kv_tile(
                    kv_pos_b, m_a, l_a, o_a,
                    lds_k_next_buf, lds_k_cur_buf, ks0_a, ks1_a,
                    pg_a, poff_a,
                    wptr_cur=lds_wptr_next,
                    gk_voff_g0=gk0_a, gk_voff_g1=gk1_a,
                )

                yield ([_raw(m_b), _raw(l_b)]
                       + [_raw(v) for v in o_b]
                       + [_raw(v) for v in ks0_b]
                       + [_raw(v) for v in ks1_b]
                       + [_raw(pg_b), _raw(poff_b)]
                       + [gk0_b, gk1_b])

            # ============ Single-tile remainder (0 or 1 iters) ============
            # After pair loop, buffer state = lds_k_cur_buf is "cur".
            # Use CAPTURED (non-phi) memrefs so compiler keeps alias info.
            single_init = ([pair_results[0], pair_results[1]]
                           + [pair_results[2 + dc]
                              for dc in range_constexpr(D_CHUNKS)]
                           + [pair_results[_KS0 + i]
                              for i in range_constexpr(K_CHUNKS)]
                           + [pair_results[_KS1 + i]
                              for i in range_constexpr(K_CHUNKS)]
                           + [lds_k_cur_buf, lds_k_next_buf]
                           + [lds_wptr_cur, lds_wptr_next]
                           + [pair_results[_PG_P],
                              pair_results[_PG_P + 1]]
                           + [pair_results[_GK_P],
                              pair_results[_GK_P + 1]])
            for kv_pos_s, s_ia, s_results in _scf.for_(
                pair_end_raw, _raw(body_end),
                _raw(arith.index(BLOCK_N)),
                iter_args=single_init
            ):
                s_m = s_ia[0]
                s_l = s_ia[1]
                s_o = [s_ia[2 + dc]
                       for dc in range_constexpr(D_CHUNKS)]
                s_ks0 = [s_ia[_KS0 + i]
                         for i in range_constexpr(K_CHUNKS)]
                s_ks1 = [s_ia[_KS1 + i]
                         for i in range_constexpr(K_CHUNKS)]
                s_pg = s_ia[_PG]
                s_poff = s_ia[_PG + 1]
                s_gk0 = s_ia[_GK]
                s_gk1 = s_ia[_GK + 1]

                (s_mn, s_ln, s_on, s_ks0n, s_ks1n,
                 s_pgn, s_poffn, s_gk0n, s_gk1n) = do_kv_tile(
                    kv_pos_s, s_m, s_l, s_o,
                    lds_k_cur_buf, lds_k_next_buf, s_ks0, s_ks1,
                    s_pg, s_poff,
                    wptr_cur=lds_wptr_cur,
                    gk_voff_g0=s_gk0, gk_voff_g1=s_gk1,
                )
                yield ([_raw(s_mn), _raw(s_ln)]
                       + [_raw(v) for v in s_on]
                       + [_raw(v) for v in s_ks0n]
                       + [_raw(v) for v in s_ks1n]
                       + [lds_k_next_buf, lds_k_cur_buf]
                       + [lds_wptr_next, lds_wptr_cur]
                       + [_raw(s_pgn), _raw(s_poffn)]
                       + [s_gk0n, s_gk1n])

            # body_results aliases for downstream (penultimate/last)
            body_results = s_results

            m_body = body_results[0]
            l_body = body_results[1]
            o_body = [body_results[2 + dc]
                      for dc in range_constexpr(D_CHUNKS)]
            ks0_body = [body_results[_KS0 + i]
                        for i in range_constexpr(K_CHUNKS)]
            ks1_body = [body_results[_KS1 + i]
                        for i in range_constexpr(K_CHUNKS)]
            k_body_cur = body_results[_BUF]
            k_body_next = body_results[_BUF + 1]
            wptr_from_body_cur = body_results[_WP]
            wptr_from_body_next = body_results[_WP + 1]

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

            _pg_dummy_i32 = arith.constant(0, type=T.i32)
            _pg_dummy_idx = arith.index(0)

            penult_init = ([m_body, l_body] + list(o_body)
                           + list(ks0_body) + list(ks1_body)
                           + [k_body_cur, k_body_next]
                           + [wptr_from_body_cur, wptr_from_body_next]
                           + [_raw(_pg_dummy_i32), _raw(_pg_dummy_idx)])
            for kv_pos_p, penult_ia, penult_results in _scf.for_(
                _raw(body_end), penult_ub, _raw(arith.index(BLOCK_N)),
                iter_args=penult_init
            ):
                m_p = penult_ia[0]
                l_p = penult_ia[1]
                o_p = [penult_ia[2 + dc]
                       for dc in range_constexpr(D_CHUNKS)]
                ks0_p = [penult_ia[_KS0 + i]
                         for i in range_constexpr(K_CHUNKS)]
                ks1_p = [penult_ia[_KS1 + i]
                         for i in range_constexpr(K_CHUNKS)]
                k_p_cur = penult_ia[_BUF]
                k_p_next = penult_ia[_BUF + 1]

                (m_pn, l_pn, o_pn, ks0_pn, ks1_pn,
                 _, _, _, _) = do_kv_tile(
                    kv_pos_p, m_p, l_p, o_p,
                    k_p_cur, k_p_next, ks0_p, ks1_p,
                    penult_ia[_PG], penult_ia[_PG + 1],
                    load_knn=False, do_vt=True,
                    do_causal=True,
                )
                yield ([_raw(m_pn), _raw(l_pn)]
                       + [_raw(v) for v in o_pn]
                       + [_raw(v) for v in ks0_pn]
                       + [_raw(v) for v in ks1_pn]
                       + [k_p_next, k_p_cur]
                       + [penult_ia[_WP + 1], penult_ia[_WP]]
                       + [_raw(_pg_dummy_i32), _raw(_pg_dummy_idx)])

            m_penult = penult_results[0]
            l_penult = penult_results[1]
            o_penult = [penult_results[2 + dc]
                        for dc in range_constexpr(D_CHUNKS)]
            ks0_penult = [penult_results[_KS0 + i]
                          for i in range_constexpr(K_CHUNKS)]
            ks1_penult = [penult_results[_KS1 + i]
                          for i in range_constexpr(K_CHUNKS)]
            k_last_cur = penult_results[_BUF]
            k_last_next = penult_results[_BUF + 1]

            last_init = ([m_penult, l_penult] + list(o_penult)
                         + list(ks0_penult) + list(ks1_penult)
                         + [k_last_cur, k_last_next]
                         + [penult_results[_WP], penult_results[_WP + 1]]
                         + [_raw(_pg_dummy_i32), _raw(_pg_dummy_idx)])
            for kv_pos_l, last_ia, last_results in _scf.for_(
                penult_ub, _raw(kv_split_end), _raw(arith.index(BLOCK_N)),
                iter_args=last_init
            ):
                m_l = last_ia[0]
                l_l = last_ia[1]
                o_l = [last_ia[2 + dc]
                       for dc in range_constexpr(D_CHUNKS)]
                ks0_l = [last_ia[_KS0 + i]
                         for i in range_constexpr(K_CHUNKS)]
                ks1_l = [last_ia[_KS1 + i]
                         for i in range_constexpr(K_CHUNKS)]
                k_l_cur = last_ia[_BUF]
                k_l_next = last_ia[_BUF + 1]

                (m_ln, l_ln, o_ln, ks0_ln, ks1_ln,
                 _, _, _, _) = do_kv_tile(
                    kv_pos_l, m_l, l_l, o_l,
                    k_l_cur, k_l_next, ks0_l, ks1_l,
                    last_ia[_PG], last_ia[_PG + 1],
                    load_knn=False, do_vt=False,
                    do_causal=True,
                )
                yield ([_raw(m_ln), _raw(l_ln)]
                       + [_raw(v) for v in o_ln]
                       + [_raw(v) for v in ks0_ln]
                       + [_raw(v) for v in ks1_ln]
                       + [k_l_next, k_l_cur]
                       + [last_ia[_WP], last_ia[_WP + 1]]
                       + [_raw(_pg_dummy_i32), _raw(_pg_dummy_idx)])

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
                base_k = rbatch * RESHAPE_CPB
                for hs in range_constexpr(4):
                    vals = [
                        vector.extract(
                            o_scaled[base_k + dk],
                            static_position=[hs], dynamic_position=[],
                        )
                        for dk in range_constexpr(4)
                    ]
                    wr_data = vector.from_elements(v4f32_type, vals)
                    wr_idx = (
                        reshape_wb
                        + lane_mod_16 * arith.index(RESHAPE_ROW)
                        + lane_div_16 * arith.index(16)
                        + arith.index(hs * 4)
                    )
                    vector.store(
                        wr_data, lds_reshape, [_raw(wr_idx)])

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
