"""MLA (Multi-Latent Attention) decode kernel for FlyDSL — nhead16x1 variant.

- MFMA v_mfma_f32_16x16x32_fp8_fp8 for both GEMM stages.
- 4 waves per workgroup; seqlen_q == 1, q_row == 0 for all waves.
  Each wave loads a different 128-hdim slice from global memory.
  K is written to shared LDS (split lora 512d + rope 64d).
  V is transposed in registers (no VT LDS buffer).
- BLOCK_N=64: each tile processes 64 KV tokens via online softmax
  (16 tokens per GEMM0 call, 4 calls per tile with pi=0,1).
- K LDS: single buffer for lora dims (256 at a time, two parts),
  separate buffer for rope dims. No ping-pong double buffer.
- Paged KV cache via block_table.

GEMM1: S = K @ Q^T   using HEAD_DIM_QK = kv_lora_rank + qk_rope_head_dim
GEMM2: O^T = V^T @ P  using HEAD_DIM_V  = kv_lora_rank

Layout (1D flattened):
  Q       : [batch, 1, num_q_heads, HEAD_DIM_QK]
  KV      : [num_physical_blocks, page_block_size, num_kv_heads, HEAD_DIM_QK]
  Mid_O   : [batch, num_kv_splits*4, num_q_heads, HEAD_DIM_V]  (fp32)
  Mid_lse : [batch, num_kv_splits*4, num_q_heads]              (fp32)
  block_table : [batch, max_num_blocks]  (i32 physical block ids)

Grid:  (batch, num_head_groups, num_kv_splits)
Block: (256,) -- 4 waves of 64

Requires: kv_lora_rank % 16 == 0, qk_rope_head_dim % 16 == 0,
          GQA group size >= 16,
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
from flydsl._mlir.dialects import gpu as _gpu
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects import math as _math
from flydsl._mlir.dialects import arith as _std_arith

from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T, default_f8_type


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


def _lds_load(byte_addr_index, vec_type, static_byte_offset=0):
    """LDS load with nontemporal hint for register allocation quality.

    Q and K share the same LDS region (time-division multiplexed).
    nontemporal=True triggers the AMDGPUPreferAGPRForDSRead pass which
    improves register allocation for long-lived Q/K operands, preventing
    precision loss even though num_agpr=0.  alignment=16 ensures
    ds_read_b128 / ds_read2_b64 generation.
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
    dtype_str="fp8",
    sm_scale=None,
):
    gpu_arch = get_hip_arch()

    BLOCK_N = 64
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    HEADS_PER_WAVE = 16

    NUM_Q_HEADS = num_q_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM_QK = kv_lora_rank + qk_rope_head_dim
    HEAD_DIM_V = kv_lora_rank
    HEAD_DIM_LORA = kv_lora_rank       # 512
    HEAD_DIM_ROPE = qk_rope_head_dim   # 64
    GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
    NUM_HEAD_GROUPS = NUM_Q_HEADS // HEADS_PER_WAVE
    CAUSAL = causal
    PAGE_BLOCK_SIZE = page_block_size

    assert NUM_Q_HEADS % HEADS_PER_WAVE == 0
    assert GQA_GROUP >= HEADS_PER_WAVE
    assert HEAD_DIM_V % 16 == 0
    assert qk_rope_head_dim % 16 == 0
    assert dtype_str == "fp8"
    assert PAGE_BLOCK_SIZE % BLOCK_N == 0, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be a multiple of BLOCK_N ({BLOCK_N})"
    )
    assert PAGE_BLOCK_SIZE <= 64, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be <= 64"
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    K_STEPS = HEAD_DIM_QK // 32           # 18
    K_LORA_STEPS = HEAD_DIM_LORA // 32    # 16
    K_ROPE_STEPS = HEAD_DIM_ROPE // 32    # 2
    D_CHUNKS = HEAD_DIM_V // NUM_WAVES // 16  # 8 (128 V-dim per wave / 16)
    PV_K_STEPS = BLOCK_N // 32            # 2 (fp8 MFMA K=32)

    RESHAPE_CPB = 4
    RESHAPE_BATCHES = D_CHUNKS // RESHAPE_CPB  # 2
    RESHAPE_ROW = RESHAPE_CPB * 16 + 4         # 68 (keep padding for now)
    RESHAPE_WAVE = 16 * RESHAPE_ROW            # 1088
    RESHAPE_TOTAL = NUM_WAVES * RESHAPE_WAVE   # 4352 DW = 17408 bytes

    STRIDE_TOKEN_Q = NUM_Q_HEADS * HEAD_DIM_QK
    STRIDE_TOKEN_KV = NUM_KV_HEADS * HEAD_DIM_QK
    STRIDE_PAGE = PAGE_BLOCK_SIZE * STRIDE_TOKEN_KV

    elem_bytes = 1  # fp8

    # -- K lora LDS layout (per half-page, 32 tokens) --
    K_LORA_HALF_DIM = 256
    K_LORA_RID_STRIDE_DW = K_LORA_HALF_DIM * 8 // 2    # 1024 DW (no padding, use swizzle)
    K_LORA_GID_STRIDE_DW = K_LORA_HALF_DIM // 2         # 128 DW
    K_LORA_LDS_BYTES_HALF = K_LORA_RID_STRIDE_DW * 4 * 4  # 16384 bytes per half
    K_LORA_LDS_BYTES = 2 * K_LORA_LDS_BYTES_HALF          # 32768 bytes = 32KB (both halves)

    # K rope: bypass LDS, load directly into registers via buffer_load_dwordx2
    # in MFMA B-operand layout. No LDS allocation needed.

    # -- Round-2 / wave-distribution offsets (pure Python) --
    K_LORA_ROUND2_OFFSET = 4 * K_LORA_GID_STRIDE_DW   # 512 DW (sub-half offset)
    K_LORA_PI1_OFFSET_DW = K_LORA_LDS_BYTES_HALF // 4  # DW offset to pi=1 K lora

    # -- Softmax / S-reshape LDS (time-multiplexed with K lora after GEMM0) --
    DYN_MAX_LDS_BYTES = 16 * 16 * 4   # 1024 bytes
    S_RESP_STRIDE_DW = 16 + 2         # 18 DW (4 waves x 4 lane_div_16 = 16 DW + 2 pad)
    S_RESP_LDS_BYTES = S_RESP_STRIDE_DW * 16 * 4  # 1152 bytes (shared across waves)

    # -- CKV global load constants (per-wave 128-hdim) --
    CKV_HDIM_PER_WAVE = HEAD_DIM_LORA // NUM_WAVES  # 128
    CKV_MEM_LD_INSTS = BLOCK_N * HEAD_DIM_LORA * elem_bytes // 1024 // 2  # 16
    # -- K LDS write constants --
    K_LDS_WR_INSTS = 8   # per part, per wave

    # -- Total LDS sizing --
    # K lora = 32KB, softmax/reshape time-multiplexed within same region
    KV_LDS_SIZE = K_LORA_LDS_BYTES  # 32768 bytes = 32KB exactly

    MAX_SEQLEN_Q = 1
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

    # Single 32KB LDS allocator — all regions time-multiplexed
    allocator_lds = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")

    # ── LDS sizing (pure Python, no MLIR ops) ──
    # Phase A (main loop): K lora occupies full 32KB
    lds_k_lora_offset = allocator_lds._align(allocator_lds.ptr, 16)
    allocator_lds.ptr = lds_k_lora_offset + K_LORA_LDS_BYTES

    # Phase B (after GEMM0, K consumed): softmax/reshape reuse same 32KB
    # Offsets are relative to same base, placed at start of the 32KB region
    lds_dyn_max_offset = lds_k_lora_offset   # 0
    lds_red_offset = lds_k_lora_offset + DYN_MAX_LDS_BYTES  # 1024
    lds_s_resp_offset = lds_k_lora_offset + DYN_MAX_LDS_BYTES + 256 * 4  # 2048

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
        elem_type = default_f8_type()
        lds_elem_type = ir.IntegerType.get_signless(8)

        i32_type = ir.IntegerType.get_signless(32)
        i64_type = ir.IntegerType.get_signless(64)
        v4f32_type = ir.VectorType.get([4], compute_type)
        v2f32_type = ir.VectorType.get([2], compute_type)
        v1i32_type = ir.VectorType.get([1], i32_type)
        v2i64_type = ir.VectorType.get([2], i64_type)
        v4i32_type = ir.VectorType.get([4], i32_type)
        i8_type = ir.IntegerType.get_signless(8)
        v8i8_type = ir.VectorType.get([8], i8_type)
        v4f16_type = ir.VectorType.get([4], T.f16)
        v8f16_type = ir.VectorType.get([8], T.f16)

        batch_size_v = arith.index_cast(T.index, batch_size.ir_value())
        seqlen_q_v = arith.index_cast(T.index, seqlen_q.ir_value())
        max_num_blocks_v = arith.index_cast(T.index, max_num_blocks.ir_value())
        num_kv_splits_v_raw = arith.index_cast(T.index, num_kv_splits.ir_value())

        base_ptr_lds = allocator_lds.get_base()

        lds_k_lora_buf = SmemPtr(
            base_ptr_lds, lds_k_lora_offset, lds_elem_type,
            shape=(K_LORA_LDS_BYTES,)
        ).get()
        # Softmax reduction buffer (time-multiplexed: used after K lora consumed)
        lds_red = SmemPtr(
            base_ptr_lds, lds_red_offset, T.f32, shape=(256,)
        ).get()
        # O reshape buffer (time-multiplexed: same 32KB, used in epilogue)
        lds_reshape = SmemPtr(
            base_ptr_lds, lds_k_lora_offset, T.f32,
            shape=(RESHAPE_TOTAL,)
        ).get()
        # S→P reshape buffer (time-multiplexed: used after K lora consumed)
        lds_s_resp_buf = SmemPtr(
            base_ptr_lds, lds_s_resp_offset, lds_elem_type,
            shape=(S_RESP_LDS_BYTES,)
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
        lane_mod_8 = lane % arith.index(8)
        lane_div_8 = lane / arith.index(8)

        wave_half = wave_id / arith.index(2)   # 0 for waves 0,1; 1 for waves 2,3
        wave_sub  = wave_id % arith.index(2)   # 0 for waves 0,2; 1 for waves 1,3

        # K lora LDS read address:
        # wave_sub selects sub-half (16-token group within a half-page),
        # wave_half selects which half-page (pi=0 or pi=1).
        k_r_id = lane_mod_16 % arith.index(4)
        k_g_id = lane_mod_16 / arith.index(4)
        k_lora_wave_rd_dw = (
            wave_sub * arith.index(K_LORA_ROUND2_OFFSET)
            + wave_half * arith.index(K_LORA_PI1_OFFSET_DW)
        )
        k_lora_rd_full_dw = (
            k_r_id * arith.index(K_LORA_RID_STRIDE_DW)
            + k_g_id * arith.index(K_LORA_GID_STRIDE_DW)
            + lane_div_16 * arith.index(4)
            + k_lora_wave_rd_dw
        )
        k_lora_rd_base = k_lora_rd_full_dw * arith.index(4)

        # K rope: direct global load in MFMA B-operand layout (no LDS)
        # Lane t provides B[g*8:g*8+8, t%16] where g=t//16
        # voffset = lane_mod_16 * STRIDE_TOKEN_KV + lane_div_16 * 8 + HEAD_DIM_LORA
        _rope_lane_off = (
            lane_mod_16 * arith.index(STRIDE_TOKEN_KV * elem_bytes)
            + lane_div_16 * arith.index(8 * elem_bytes)
            + arith.index(HEAD_DIM_LORA * elem_bytes)
        )
        # soffset per wave: wave_token_base * STRIDE_TOKEN_KV
        # wave 0→0, wave 1→16*576, wave 2→32*576, wave 3→48*576
        _rope_wave_soff = rocdl.readfirstlane(
            i32_type,
            _index_cast_to_i32(
                wave_id * arith.index(16 * STRIDE_TOKEN_KV * elem_bytes)
            ),
        )

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
        # All 4 waves within one KV-split write different V-dim ranges
        # (wave_id * CKV_HDIM_PER_WAVE), so they share a single split entry.
        actual_num_splits = num_kv_splits_v
        wave_split_id = split_id
        stride_mid_o_token = actual_num_splits * arith.index(NUM_Q_HEADS * HEAD_DIM_V)
        stride_mid_lse_token = actual_num_splits * arith.index(NUM_Q_HEADS)

        # ---- V register transpose permutation constants ----
        # Stage 1 uses s_perm2, s_perm3; Stage 2 uses s_perm1, s_perm0
        s_perm0 = arith.constant(0x07060302, type=i32_type)
        s_perm1 = arith.constant(0x05040100, type=i32_type)
        s_perm2 = arith.constant(0x05010400, type=i32_type)
        s_perm3 = arith.constant(0x07030602, type=i32_type)

        # ---- S→P LDS address generation (4-wave shared layout) ----
        # After softmax, P values are redistributed via shared LDS so
        # each q_id lane gets P for 16 consecutive tokens across 4 wave groups.
        # Write: each wave writes 1 DW (4 fp8 P) at (q_id + wave_id*4 + head*STRIDE)
        # Read:  each lane reads i64 (8 fp8) from two offsets (tokens 0-31 and 32-63)
        lds_s_resp_base_idx = _memref.ExtractAlignedPointerAsIndexOp(
            lds_s_resp_buf).result
        lds_s_resp_base_i32 = rocdl.readfirstlane(
            i32_type, _index_cast_to_i32(lds_s_resp_base_idx))

        _s_resp_wr_off = (
            lane_div_16
            + wave_id * arith.index(4)
            + lane_mod_16 * arith.index(S_RESP_STRIDE_DW)
        ) * arith.index(4)
        _s_resp_wr_i32 = _std_arith.AddIOp(
            _raw(_index_cast_to_i32(_s_resp_wr_off)),
            _raw(lds_s_resp_base_i32)).result

        _s_resp_rd_off_0 = (
            lane_div_16 * arith.index(2)
            + lane_mod_16 * arith.index(S_RESP_STRIDE_DW)
        ) * arith.index(4)
        _s_resp_rd_i32_0 = _std_arith.AddIOp(
            _raw(_index_cast_to_i32(_s_resp_rd_off_0)),
            _raw(lds_s_resp_base_i32)).result

        _s_resp_rd_off_1 = _s_resp_rd_off_0 + arith.index(32)  # +8 DW = 32 bytes
        _s_resp_rd_i32_1 = _std_arith.AddIOp(
            _raw(_index_cast_to_i32(_s_resp_rd_off_1)),
            _raw(lds_s_resp_base_i32)).result

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
                + wave_split_id * arith.index(NUM_Q_HEADS * HEAD_DIM_V)
                + q_head_idx * arith.index(HEAD_DIM_V)
                + d_col
            )

        def mid_lse_global_idx(q_row):
            total_q = batch_idx * seqlen_q_v + q_row
            return (
                total_q * stride_mid_lse_token
                + wave_split_id * arith.index(NUM_Q_HEADS)
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

        _wave_id_i32 = _index_cast_to_i32(wave_id)
        _wave_id_sgpr = rocdl.readfirstlane(i32_type, _wave_id_i32)

        def _i32_to_lds_ptr(i32_val):
            i64_val = _std_arith.ExtUIOp(i64_type, i32_val).result
            return _inttoptr_lds(i64_val)

        # ---- K LDS base addresses ----
        lds_k_lora_base_idx = _memref.ExtractAlignedPointerAsIndexOp(
            lds_k_lora_buf).result
        lds_k_lora_base_i32 = rocdl.readfirstlane(
            i32_type, _index_cast_to_i32(lds_k_lora_base_idx))

        # ---- K LDS write address (per-thread) ----
        # Use wave-local q_id (lane/16, 0..3) so that all waves' data for the
        # same token sits in one contiguous 128-DW row within the r_id group.
        # Global q_id (tid/16, 0..15) would create a 256-DW stride between
        # q_ids, leaving most read addresses unwritten.
        _k_wr_q_id = lane_div_16
        _k_wr_p_id = tid % arith.index(16)
        # Remap p_id so each lane_div_16 group's 2 i64 are contiguous (ds_read_b128).
        # Groups: (0,1,8,9)→DW 0-3, (2,3,10,11)→DW 4-7, etc.
        _p_mod2 = _k_wr_p_id % arith.index(2)
        _p_div8 = _k_wr_p_id / arith.index(8)
        _p_mid  = (_k_wr_p_id % arith.index(8)) / arith.index(2)
        _k_wr_p_remapped = (
            _p_mid * arith.index(4)
            + _p_div8 * arith.index(2)
            + _p_mod2
        )
        _k_lora_wr_base_dw = (
            _k_wr_q_id * arith.index(256)
            + _k_wr_p_remapped
        )
        _k_lora_wr_wave_dw = wave_id * arith.index(32)
        _k_lora_wr_addr_dw = _k_lora_wr_base_dw + _k_lora_wr_wave_dw
        k_lora_wr_addr_bytes = _k_lora_wr_addr_dw * arith.index(4)
        k_lora_wr_addr_i32 = _index_cast_to_i32(k_lora_wr_addr_bytes)
        k_lora_wr_addr_i32_sgpr = _std_arith.AddIOp(
            _raw(k_lora_wr_addr_i32), _raw(lds_k_lora_base_i32)).result

        # K rope: no LDS write needed (direct register load in MFMA layout)

        # ---- CKV global load address generation ----
        # Each wave loads CKV_HDIM_PER_WAVE=128 dims × BLOCK_N=64 tokens
        # Thread layout: q_id=lane/16 (0..3), p_id=lane%16 (0..15)
        # Per-thread base: p_id * 4 bytes (fp8: 4 values) + wave_id * 128 bytes
        # Token mapping: q_id*8 + (inst//8)*32 + (inst%8)
        _ckv_p_id_bytes = lane_mod_16 * arith.index(4 * elem_bytes)
        _ckv_wave_bytes = wave_id * arith.index(CKV_HDIM_PER_WAVE * elem_bytes)
        _ckv_base_bytes = _ckv_p_id_bytes + _ckv_wave_bytes
        _ckv_q_id = lane_div_16
        _ckv_q_id_stride = _ckv_q_id * arith.index(8 * STRIDE_TOKEN_KV * elem_bytes)

        def _vt_perm(src_hi, src_lo, sel):
            return llvm.call_intrinsic(
                i32_type, "llvm.amdgcn.perm",
                [src_hi, src_lo, sel], [], [],
            )

        # ---- Preload Q via 64×64 tiled loop ----
        q_row = arith.index(0)

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
        lds_base_byte_idx = lds_k_lora_base_idx
        c_dword_sz = arith.constant(4, type=T.i32)
        c_zero_i32 = arith.constant(0, type=T.i32)
        aux = arith.constant(0, type=T.i32)

        q_packs = [None] * K_STEPS

        q_voff_base_bytes = (
            batch_idx
            * seqlen_q_v
            * arith.index(STRIDE_TOKEN_Q * elem_bytes)
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

        for d_tile_start, batch_size_q in _Q_BATCHES:

            for t in range_constexpr(batch_size_q):
                d_tile = d_tile_start + t

                tile_lds_byte = (
                    lds_base_byte_idx
                    + arith.index(t * Q_TILE_BYTES)
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
                    + lane_mod_16
                    * arith.index(Q_TILE_DIM * elem_bytes)
                    + lane_div_16 * arith.index(8)
                )
                for local_ks in range_constexpr(2):
                    global_ks = d_tile * 2 + local_ks
                    _base_off = local_ks * 32
                    q_packs[global_ks] = _lds_load(
                        q_tile_byte_base, i64_type,
                        static_byte_offset=_base_off,
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

        def _mfma_fp8(result_type, operands, **kw):
            return rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, operands, **kw)

        # ---- CKV global → VGPR loading ----
        # CKV_64x512_mem_ld_insts = 16 instruction groups total (both halves)
        # Per half (pi): 8 instruction groups, each loading 2 DWORDs (8 bytes)
        # Per half per thread: 8 * 8 = 64 bytes → 32 tokens * 128 dims / 64 threads ✓
        _CKV_INSTS_PER_HALF = CKV_MEM_LD_INSTS // 2  # 8

        def _ckv_global_load(kv_page_base, page_off, half_idx):
            """Load one 32-token half of CKV lora data: global → VGPRs.

            Token mapping (SP3 LTD layout):
              inst i, q_id q → token = q*8 + half_idx*32 + i
            Each inst loads 2 DWORDs: offset 0x0 and 0x40 (= +64 bytes).
            Per-inst offsets (inst*stride, ≤4032) fold into offset:imm.
            Half-page base (half_idx*32*stride) goes into soffset (SGPR).

            Returns list of 2*_CKV_INSTS_PER_HALF i32 VGPRs.
            """
            page_byte_base = kv_page_base * arith.index(elem_bytes)
            kv_head_byte = kv_head_offset * arith.index(elem_bytes)
            p_off_byte = page_off * arith.index(STRIDE_TOKEN_KV * elem_bytes)
            voff_base = (page_byte_base + kv_head_byte + p_off_byte
                         + _ckv_base_bytes + _ckv_q_id_stride)
            voff_base_i32 = _index_cast_to_i32(voff_base)
            half_soff = arith.constant(
                half_idx * 32 * STRIDE_TOKEN_KV * elem_bytes, type=T.i32)

            vgprs = []
            for inst in range_constexpr(_CKV_INSTS_PER_HALF):
                inst_off = inst * STRIDE_TOKEN_KV * elem_bytes
                if inst_off == 0:
                    voff_dw0 = voff_base_i32
                else:
                    voff_dw0 = _std_arith.AddIOp(
                        _raw(voff_base_i32),
                        _raw(arith.constant(inst_off, type=T.i32)),
                    ).result
                dw0 = rocdl.raw_ptr_buffer_load(
                    i32_type, kv_rsrc, voff_dw0,
                    _raw(half_soff), _raw(aux),
                )
                dw1_off = inst_off + 64 * elem_bytes
                voff_dw1 = _std_arith.AddIOp(
                    _raw(voff_base_i32),
                    _raw(arith.constant(dw1_off, type=T.i32)),
                ).result
                dw1 = rocdl.raw_ptr_buffer_load(
                    i32_type, kv_rsrc, voff_dw1,
                    _raw(half_soff), _raw(aux),
                )
                vgprs.append(dw0)
                vgprs.append(dw1)
            return vgprs

        # ---- K lora LDS write (SP3: K_lds_wr) ----
        def _k_lora_lds_write(ckv_vgprs, base_offset_bytes=0):
            """Write K lora data from CKV VGPRs to K lora LDS.

            SP3 K_lds_wr: 8 ds_write_b32 pairs per call.
            base_offset_bytes is folded into each GEP static offset so all
            stores share the same base pointer (avoids intermediate VGPR).
            """
            wr_ptr = _i32_to_lds_ptr(k_lora_wr_addr_i32_sgpr)
            for inst in range_constexpr(K_LDS_WR_INSTS):
                r_id = inst % 4
                g_id = inst // 4
                lds_off = base_offset_bytes + (
                    r_id * K_LORA_RID_STRIDE_DW
                    + g_id * K_LORA_GID_STRIDE_DW
                ) * 4
                lds_ptr_off = _get_element_ptr(
                    wr_ptr, static_byte_offset=lds_off)
                dw0 = ckv_vgprs[inst * 2 + 0]
                dw0_val = dw0 if isinstance(dw0, ir.Value) else _raw(dw0)
                llvm.StoreOp(dw0_val, lds_ptr_off)
                lds_ptr_off2 = _get_element_ptr(
                    wr_ptr, static_byte_offset=lds_off + 64)
                dw1 = ckv_vgprs[inst * 2 + 1]
                dw1_val = dw1 if isinstance(dw1, ir.Value) else _raw(dw1)
                llvm.StoreOp(dw1_val, lds_ptr_off2)

        # ---- K rope: direct global load in MFMA B-operand layout ----
        def _rope_direct_load(kv_page_base, page_off):
            """Load K rope directly into MFMA B-operand registers for this wave's 16 tokens.

            Each lane loads 8 fp8 values (i64) corresponding to its MFMA group:
              lane t → token t%16, K-dims [(t//16)*8 : (t//16)*8+8]
            Returns K_ROPE_STEPS (2) i64 values, directly usable as MFMA B operands.
            Uses buffer_load_dwordx2 (8 bytes = i64 = 8 fp8).
            """
            page_byte_base = kv_page_base * arith.index(elem_bytes)
            kv_head_byte = kv_head_offset * arith.index(elem_bytes)
            p_off_byte = page_off * arith.index(STRIDE_TOKEN_KV * elem_bytes)
            voff_base = page_byte_base + kv_head_byte + p_off_byte + _rope_lane_off
            voff_base_i32 = _index_cast_to_i32(voff_base)

            rope_ops = []
            for ks in range_constexpr(K_ROPE_STEPS):
                ks_off = ks * 32 * elem_bytes
                if ks_off == 0:
                    voff_i32 = voff_base_i32
                else:
                    voff_i32 = _std_arith.AddIOp(
                        _raw(voff_base_i32),
                        _raw(arith.constant(ks_off, type=T.i32)),
                    ).result
                dw_pair = rocdl.raw_ptr_buffer_load(
                    i64_type, kv_rsrc, voff_i32,
                    _raw(_rope_wave_soff), _raw(aux),
                )
                rope_ops.append(dw_pair)
            return rope_ops

        # ---- K LDS read (SP3: K_lds_rd) ----
        # 8 × ds_read_b128 at stride 64 bytes; each reads 16 contiguous
        # bytes = 2 i64 MFMA operands.  Total: 8 × 2 = 16 i64 = K_LORA_STEPS.
        K_LDS_RD_INSTS = K_LDS_WR_INSTS  # 8

        def _k_lora_lds_read(extra_offset_dw=0):
            """Read K lora from LDS → K_LORA_STEPS i64 operands via ds_read_b128."""
            k_ops = []
            for inst in range_constexpr(K_LDS_RD_INSTS):
                rd_addr = (
                    lds_k_lora_base_idx
                    + k_lora_rd_base
                    + arith.index((extra_offset_dw + inst * 16) * 4)
                )
                pair = _lds_load(rd_addr, v2i64_type)
                val0 = vector.extract(pair, static_position=[0],
                                      dynamic_position=[])
                val1 = vector.extract(pair, static_position=[1],
                                      dynamic_position=[])
                k_ops.append(val0)
                k_ops.append(val1)
            return k_ops

        # K rope: no LDS read needed (loaded directly into registers via _rope_direct_load)

        # ---- V register transpose (SP3: CKV_V_tr) ----
        def _v_register_transpose_half(half_vgprs):
            """Transpose V data from one 32-token half.

            Input: 16 i32 VGPRs (one ckv half).
            Output: D_CHUNKS = 8 i64 operands for GEMM1.

            Derived from the full 32-VGPR transpose by observing that
            sec=0,1 access ckv_h0 and sec=2,3 access ckv_h1.  Splitting
            gives sec=0..1 per half → 2 sec × 2 i = 4 groups → 16 acc
            values → 8 i64 v_ops.  These correspond exactly to the
            even-indexed (ks_pv=0) v_ops for h0, or odd-indexed (ks_pv=1)
            for h1, in the full transpose.
            """
            def _v(idx):
                v = half_vgprs[idx]
                return v if isinstance(v, ir.Value) else _raw(v)

            acc_V = [None] * 16
            for i in range_constexpr(2):
                for sec in range_constexpr(2):
                    s_base = sec * 8
                    s0 = _v(s_base + 0 + i)
                    s1 = _v(s_base + 2 + i)
                    s2 = _v(s_base + 4 + i)
                    s3 = _v(s_base + 6 + i)

                    tmp0 = _vt_perm(s1, s0, s_perm2)
                    tmp1 = _vt_perm(s1, s0, s_perm3)
                    tmp2 = _vt_perm(s3, s2, s_perm2)
                    tmp3 = _vt_perm(s3, s2, s_perm3)

                    d0 = _vt_perm(tmp2, tmp0, s_perm1)
                    d1 = _vt_perm(tmp2, tmp0, s_perm0)
                    d2 = _vt_perm(tmp3, tmp1, s_perm1)
                    d3 = _vt_perm(tmp3, tmp1, s_perm0)

                    acc_V[i * 8 + sec + 0] = d0
                    acc_V[i * 8 + sec + 2] = d1
                    acc_V[i * 8 + sec + 4] = d2
                    acc_V[i * 8 + sec + 6] = d3

            c32 = _std_arith.ConstantOp(
                i64_type, ir.IntegerAttr.get(i64_type, 32)).result
            v_ops = []
            for idx in range_constexpr(D_CHUNKS):
                a = acc_V[idx * 2]
                b = acc_V[idx * 2 + 1]
                a_i64 = _std_arith.ExtUIOp(i64_type, a).result
                b_i64 = _std_arith.ExtUIOp(i64_type, b).result
                b_sh = _std_arith.ShLIOp(b_i64, c32).result
                op = _std_arith.OrIOp(a_i64, b_sh).result
                v_ops.append(op)
            return v_ops

        # ---- Softmax for 16 tokens (wave-distributed) ----
        def _softmax_reduce_16(s_acc,
                               kv_pos=None, do_causal=False):
            """Extract 4 s_vals from one GEMM0 output, cross-wave max via LDS."""
            s_vals = []
            for ii in range_constexpr(4):
                s_val = vector.extract(
                    s_acc, static_position=[ii], dynamic_position=[],
                )
                if CAUSAL and do_causal:
                    kv_abs = (kv_pos
                              + wave_id * arith.index(16)
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
            _barrier(lgkmcnt=0)

            global_max = c_neg_large
            for w in range_constexpr(NUM_WAVES):
                red_rd_base = (
                    arith.index(w * 64)
                    + lane_mod_16 * arith.index(4)
                )
                max_vec = vector.load_op(
                    v4f32_type, lds_red, [_raw(red_rd_base)])
                for g in range_constexpr(4):
                    max_g = vector.extract(
                        max_vec, static_position=[g], dynamic_position=[],
                    )
                    global_max = _std_arith.MaximumFOp(
                        _raw(global_max), _raw(max_g),
                        fastmath=fm_fast,
                    ).result

            return global_max, s_vals

        def _softmax_finalize_64(global_max, s_vals, m_old, l_old):
            """Online softmax for 4 s_vals → 2 p_packs (each i64, 8 fp8).

            Each wave computes P for its own 16 tokens (4 per lane).
            P is written to shared S→P LDS with wave_id offset,
            then a cross-wave barrier syncs all writes before each
            lane reads two i64 values spanning all 4 wave groups (64 tokens).
            """
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

            w = rocdl.cvt_pk_fp8_f32(
                i32_type, _raw(p_vals[0]), _raw(p_vals[1]), c_zero_i32, 0)
            w = rocdl.cvt_pk_fp8_f32(
                i32_type, _raw(p_vals[2]), _raw(p_vals[3]), w, 1)

            llvm.InlineAsmOp(
                res=None,
                operands_=[_raw(_s_resp_wr_i32), _raw(w)],
                asm_string="ds_write_b32 $0, $1",
                constraints="v,v",
                has_side_effects=True,
                is_align_stack=False,
            )

            _barrier(lgkmcnt=0)

            p_pack_0 = llvm.InlineAsmOp(
                res=i64_type,
                operands_=[_raw(_s_resp_rd_i32_0)],
                asm_string=(
                    "ds_read_b64 $0, $1\n"
                    "s_waitcnt lgkmcnt(0)"
                ),
                constraints="=v,v",
                has_side_effects=True,
                is_align_stack=False,
            ).result

            p_pack_1 = llvm.InlineAsmOp(
                res=i64_type,
                operands_=[_raw(_s_resp_rd_i32_1)],
                asm_string=(
                    "ds_read_b64 $0, $1\n"
                    "s_waitcnt lgkmcnt(0)"
                ),
                constraints="=v,v",
                has_side_effects=True,
                is_align_stack=False,
            ).result

            return m_new, l_new, p_pack_0, p_pack_1, rescale

        def _make_rescale_vec(rescale):
            return vector.broadcast(v4f32_type, rescale)

        def _rescale_acc(acc, rescale_vec):
            return _std_arith.MulFOp(
                _raw(acc), _raw(rescale_vec),
                fastmath=fm_fast,
            ).result

        # ================================================================
        # _do_full — process one 64-token tile (pi=0 and pi=1 combined)
        # ================================================================
        # K+VT cross-iteration prefetch pipeline:
        # GEMM0 (k_ops from iter_args) → barrier → softmax →
        # vmcnt(0) → K_{N+1} ds_write → lgkmcnt(0)+barrier →
        # K_{N+1} ds_read || GEMM1 (vt from iter_args) || V_transpose
        # → return k_ops_next, vt_next_0, vt_next_1

        def _do_full(rope_ops, kv_pos,
                     m_cur, l_cur, o_accs, k_ops_pf,
                     vt_half0, vt_half1,
                     next_ckv_0, next_ckv_1, do_causal=False):

            s_acc = c_zero_v4f32
            for ks in range_constexpr(K_LORA_STEPS):
                s_acc = _mfma_fp8(
                    v4f32_type,
                    [k_ops_pf[ks], q_packs[ks], s_acc, 0, 0, 0],
                )
            for ks in range_constexpr(K_ROPE_STEPS):
                s_acc = _mfma_fp8(
                    v4f32_type,
                    [rope_ops[ks], q_packs[K_LORA_STEPS + ks],
                     s_acc, 0, 0, 0],
                )

            # Barrier: all waves finish previous K reads before softmax
            # reuses the LDS region for reduction buffers
            _barrier(lgkmcnt=0)

            global_max, s_vals = _softmax_reduce_16(
                s_acc,
                kv_pos=kv_pos if do_causal else None,
                do_causal=do_causal)
            m_cur, l_cur, p_pack_0, p_pack_1, rescale = _softmax_finalize_64(
                global_max, s_vals, m_cur, l_cur)

            rescale_vec = _make_rescale_vec(rescale)
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _rescale_acc(o_accs[dc], rescale_vec)

            # Wait for next tile's CKV global loads
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))

            # K_{N+1} write to LDS
            _k_lora_lds_write(next_ckv_0, base_offset_bytes=0)
            _k_lora_lds_write(next_ckv_1,
                              base_offset_bytes=K_LORA_LDS_BYTES_HALF)

            # Barrier: K writes done, all waves synced
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            gpu.barrier()

            rocdl.sched_barrier(0)
            # --- Three-way concurrent: K read || GEMM1 || V transpose ---
            # LLVM will interleave ds_read / v_mfma / v_perm freely
            k_ops_next = _k_lora_lds_read(extra_offset_dw=0)

            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _mfma_fp8(
                    v4f32_type,
                    [vt_half0[dc], p_pack_0, o_accs[dc], 0, 0, 0],
                )
                o_accs[dc] = _mfma_fp8(
                    v4f32_type,
                    [vt_half1[dc], p_pack_1, o_accs[dc], 0, 0, 0],
                )

            vt_next_0 = _v_register_transpose_half(next_ckv_0)
            vt_next_1 = _v_register_transpose_half(next_ckv_1)

            return m_cur, l_cur, o_accs, k_ops_next, vt_next_0, vt_next_1

        # ================================================================
        # Main loop — K+VT cross-iteration prefetch pipeline
        # ================================================================
        # Preamble: load CKV_0, K_0 write+read → k_ops_0,
        # V_transpose(CKV_0) → vt_0.  Both carried as iter_args.
        # CKV is NOT in iter_args — only its processed results are.
        has_kv_work = _std_arith.CmpIOp(
            _std_arith.CmpIPredicate.ult,
            _raw(kv_split_start), _raw(kv_split_end),
        ).result

        N_PREFIX = 2 + D_CHUNKS
        N_KOPS = K_LORA_STEPS   # 16 i64 prefetched K operands
        N_VT = D_CHUNKS         # 8 i64 per vt half
        N_ROPE = K_ROPE_STEPS   # 2 i64

        if has_kv_work:
            o_accs_init = [
                arith.constant_vector(0.0, v4f32_type)
                for _ in range_constexpr(D_CHUNKS)]
            m_init = c_neg_large
            l_init = c_zero_f

            rocdl.sched_barrier(0)

            # ---- Preamble ----
            pf_phys_0, pf_off_0 = lookup_page_issue(kv_split_start)
            pf_phys_1, pf_off_1 = lookup_page_issue(
                kv_split_start + arith.index(BLOCK_N))
            pf_base_0 = lookup_page_resolve(pf_phys_0)
            pf_ckv_0 = _ckv_global_load(pf_base_0, pf_off_0, 0)
            pf_ckv_1 = _ckv_global_load(pf_base_0, pf_off_0, 1)
            pf_rope = _rope_direct_load(pf_base_0, pf_off_0)

            # Wait for CKV_0 data
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            gpu.barrier()

            # K_0 write to LDS
            _k_lora_lds_write(pf_ckv_0, base_offset_bytes=0)
            _k_lora_lds_write(pf_ckv_1,
                              base_offset_bytes=K_LORA_LDS_BYTES_HALF)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            gpu.barrier()

            # K_0 read (prefetch) + V_transpose (overlaps K read latency)
            k_ops_init = _k_lora_lds_read(extra_offset_dw=0)
            vt_init_0 = _v_register_transpose_half(pf_ckv_0)
            vt_init_1 = _v_register_transpose_half(pf_ckv_1)

            body_init = (
                [_raw(m_init), _raw(l_init)]
                + [_raw(v) for v in o_accs_init]
                + [_raw(v) for v in k_ops_init]
                + [_raw(v) for v in vt_init_0]
                + [_raw(v) for v in vt_init_1]
                + [_raw(v) for v in pf_rope]
                + [_raw(pf_phys_1), _raw(pf_off_1)]
            )
            for kv_pos, body_ia, body_results in _scf.for_(
                _raw(kv_split_start), _raw(kv_split_end),
                _raw(arith.index(BLOCK_N)),
                iter_args=body_init
            ):
                m_running = body_ia[0]
                l_running = body_ia[1]
                o_accs = [
                    body_ia[2 + dc]
                    for dc in range_constexpr(D_CHUNKS)
                ]
                cur_k_ops = [
                    body_ia[N_PREFIX + i]
                    for i in range_constexpr(N_KOPS)
                ]
                cur_vt_0 = [
                    body_ia[N_PREFIX + N_KOPS + i]
                    for i in range_constexpr(N_VT)
                ]
                cur_vt_1 = [
                    body_ia[N_PREFIX + N_KOPS + N_VT + i]
                    for i in range_constexpr(N_VT)
                ]
                cur_rope = [
                    body_ia[N_PREFIX + N_KOPS + 2 * N_VT + i]
                    for i in range_constexpr(N_ROPE)
                ]
                next_phys = body_ia[N_PREFIX + N_KOPS + 2 * N_VT + N_ROPE]
                next_off = body_ia[
                    N_PREFIX + N_KOPS + 2 * N_VT + N_ROPE + 1]

                # Issue next tile's CKV + rope loads (overlap with GEMM0)
                next_base = lookup_page_resolve(next_phys)
                pf_ckv_next_0 = _ckv_global_load(next_base, next_off, 0)
                pf_ckv_next_1 = _ckv_global_load(next_base, next_off, 1)
                pf_rope_next = _rope_direct_load(next_base, next_off)
                nn_phys, nn_off = lookup_page_issue(
                    kv_pos + arith.index(2 * BLOCK_N))

                m_new, l_new, o_accs, k_ops_next, vt_next_0, vt_next_1 = \
                    _do_full(
                        cur_rope,
                        kv_pos, m_running, l_running, o_accs,
                        cur_k_ops, cur_vt_0, cur_vt_1,
                        pf_ckv_next_0, pf_ckv_next_1,
                        do_causal=CAUSAL)

                yield (
                    [_raw(m_new), _raw(l_new)]
                    + [_raw(v) for v in o_accs]
                    + [_raw(v) for v in k_ops_next]
                    + [_raw(v) for v in vt_next_0]
                    + [_raw(v) for v in vt_next_1]
                    + [_raw(v) for v in pf_rope_next]
                    + [_raw(nn_phys), _raw(nn_off)]
                )

            m_final = body_results[0]
            l_partial = body_results[1]
            o_finals = [
                body_results[2 + dc]
                for dc in range_constexpr(D_CHUNKS)
            ]

            red_wr_idx_epi = (
                wave_id * arith.index(64)
                + lane_mod_16 * arith.index(4)
                + lane_div_16
            )
            _memref.StoreOp(l_partial, lds_red, [_raw(red_wr_idx_epi)])
            _barrier(lgkmcnt=0)

            l_final = c_zero_f
            for w in range_constexpr(4):
                red_rd_base_epi = (
                    arith.index(w * 64)
                    + lane_mod_16 * arith.index(4)
                )
                l_vec = vector.load_op(v4f32_type, lds_red,
                                        [_raw(red_rd_base_epi)])
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

            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0, lgkmcnt=0))
            gpu.barrier()

            reshape_wb = wave_id * arith.index(RESHAPE_WAVE)
            total_q_epi = batch_idx * seqlen_q_v + q_row

            for rbatch in range_constexpr(RESHAPE_BATCHES):
                dc_base = rbatch * RESHAPE_CPB
                for j in range_constexpr(4):
                    v2_02 = vector.shuffle(
                        o_scaled[dc_base + 0], o_scaled[dc_base + 2],
                        [j, 4 + j])
                    v2_13 = vector.shuffle(
                        o_scaled[dc_base + 1], o_scaled[dc_base + 3],
                        [j, 4 + j])
                    v4_gather = vector.shuffle(
                        v2_02, v2_13, [0, 2, 1, 3])
                    wr_base = (
                        reshape_wb
                        + lane_mod_16 * arith.index(RESHAPE_ROW)
                        + lane_div_16 * arith.index(16)
                        + arith.index(j * 4)
                    )
                    vector.store(
                        v4_gather, lds_reshape, [_raw(wr_base)])

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
                            wave_id * arith.index(CKV_HDIM_PER_WAVE)
                            + arith.index(
                                rbatch * RESHAPE_CPB * 16
                                + si * 32)
                            + lane_mod_8 * arith.index(4)
                        )
                        g_idx = (
                            total_q_epi * stride_mid_o_token
                            + wave_split_id * arith.index(
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
        allocator_lds.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_lds.finalize()

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
