# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Dual-wave, software-pipelined flash-attention kernel for gfx950 (D=128, bf16/fp16).

This is the gfx950 fast path of FlyDSL flash attention. It computes the SAME
math as the generic ``flash_attn_generic.py`` BLOCK_M=256 path -- and reuses its
LDS / MFMA / Q-K-V-O addressing to inherit correctness -- but replaces the
compiler-driven schedule with a hand-built software pipeline plus a two
wave-group time-multiplexing scheme.

Dispatched (from ``flash_attn_generic.py``) only when:
    gpu_arch >= gfx950, head_dim == 128, dtype in (bf16, fp16),
    and (at runtime) seq_len % 256 == 0, seq_len >= 384.

Tile / occupancy:
    BLOCK_M = 256 (8 waves x 32 rows), BLOCK_N = 64, head_dim = 128,
    waves_per_eu = 2. MFMA is ``mfma_f32_32x32x16_{bf16,f16}`` (K=16, CDNA4).
    GEMM1 = K @ Q^T (scores land directly in MFMA32 register layout); the
    softmax is online over the KV dimension in registers; GEMM2 = V^T @ P.
    Supports causal / non-causal and MHA / GQA (num_kv_heads <= num_heads).

Execution model -- explicit 8-cluster software pipeline:
    Each main-loop iteration advances j by 2 (folds TWO KV tiles into the
    running (m_row, l_row, v_o) state) and is split into 8 clusters C0..C7 that
    strictly alternate a MEMORY stage (even) and a COMPUTE stage (odd); every
    cluster ends with ``rocdl.s_barrier()``. Instruction interleaving inside a
    cluster is pinned with ``rocdl.sched_barrier(0)`` fences and
    ``rocdl.sched_group_barrier`` (IGroupLP) MFMA/VALU/EXP group hints rather
    than left to the LLVM scheduler.
        even C0/C2/C4/C6 : global->LDS async DMA (double-buffered K/V)
                           + LDS->VGPR reads (+ causal mask)
        odd  C1/C5       : Q*K (mma0) + finish the previous tile's softmax
                           2nd-half exp2 + row-sum into l_row + cast P to bf16
        odd  C3/C7       : P*V (mma1, 4 step_k) + lazy rescale
                           + this tile's softmax 1st-half (row_max, sub_row,
                           exp2 elems 0..15)
    A large explicit prologue primes the pipeline (loads tile 0, first Q*K +
    softmax) and a fully-unrolled epilogue (Clusters 0..13) drains it for the
    final tiles that the loop leaves in flight.

Key gfx950 optimizations:
    * Two wave-groups (group A = waves 0-3, group B = waves 4-7,
      ``DUALWAVE_SWP_ENABLE_STAGGER``). One extra prologue ``s_barrier`` on group B
      offsets the groups by exactly ONE cluster (s_barriers match by ordinal,
      one per cluster), so group A runs one cluster AHEAD of group B: while one
      group COMPUTES (MFMA/VALU) the other LOADS (DMA + LDS reads + waitcnt
      stalls), hiding one group's memory latency behind the other's MFMA. The
      offset is closed by a matching extra barrier on group A in the epilogue.
    * ``rocdl.s_setprio(1)/(0)`` (``DUALWAVE_SWP_SETPRIO``) brackets the heavy compute
      clusters C3/C7 to hand the shared MFMA issue slots from the computing
      group to the group just entering its compute phase, keeping the
      compute/load alternation crisp.
    * Lazy rescaling (``DUALWAVE_SWP_LAZY_RESCALE``): a uniform ``ballot(below) == exec``
      scalar branch (``s_cbranch_scc``) skips the running O / l_row rescale
      (~32 ``v_pk_mul``) for a tile whenever every lane's row-max moved by
      <= ``RESCALE_THRESHOLD`` (8.0). The 1/sqrt(D) temperature scale is
      pre-applied to Q rather than folded into the exp.
    * Online-softmax ``exp2`` is split into a first half (elems 0..15) and a
      second half (16..31) placed in different clusters so the transcendental
      (TRANS) latency hides behind the MFMA chains.
    * Double-buffered K and V LDS (buf0/buf1) filled by async DMA-to-LDS
      (``buffer_load_dwordx4 ... lds``) and read with the gfx950 HW-transpose
      ``ds_read_b64_tr_b16``; inline-asm causal mask
      (``v_cmp_lt_i32 + v_cndmask_b32`` with immediate K-position thresholds).

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads,
head_dim). Grid = (num_heads, num_q_blocks, batch); Block = (512, 1, 1) = 8
waves. Requires head_dim == 128 (asserts gfx950+; no gfx942 fallback).
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.flash_attn_gfx950_atoms import FaMfma32x32x16
from kernels.kernels_common import dtype_to_elem_type

_LOG2E = host_math.log2(host_math.e)
# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.dualwave_swp.lds">'


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _ds_read_tr16_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b16 with DUALWAVE_SWP immediate byte offset."""
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [_llvm_value(addr_i32)],
        f"ds_read_b64_tr_b16 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return vector.BitCastOp(result_type, raw).result


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


def _lds_alias_scope_array(names):
    attrs = [f'#llvm.alias_scope<id = "{name}", domain = {_LDS_ALIAS_DOMAIN}>' for name in names]
    return ir.Attribute.parse(f"[{', '.join(attrs)}]")


def build_flash_attn_dualwave_swp_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
):
    """Build an DUALWAVE_SWP flash_attn launcher for D=128 bf16/f16 on gfx950.

    Launcher signature: ``launcher(Q, K, V, O, batch_size, seq_len, stride_kv_n=None, stride_q_n=None, head_dim_runtime=None, *, stream=None)``
    """
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(f"flash_attn_dualwave_swp requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}")
    if head_dim != 128:
        raise RuntimeError(f"flash_attn_dualwave_swp is D=128 only, got head_dim={head_dim}")
    if dtype_str not in ("bf16", "f16"):
        raise RuntimeError(f"flash_attn_dualwave_swp supports bf16/f16 only, got dtype={dtype_str}")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    # ──────────────────────────── Tile constants ────────────────────────────
    # Match existing flash_attn_generic BLOCK_M=256 path for layout compatibility.
    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_N_OUT = 64  # single sub-tile per outer iter (=BLOCK_N)
    BLOCK_N_OUT // BLOCK_N
    K_SUB_N = 32  # MFMA W_N
    WARP_SIZE = 64
    NUM_WAVES = 8  # BLOCK_M / 32
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 512
    ROWS_PER_WAVE = 32

    HEAD_DIM = head_dim
    K_STEP_QK = 16  # W_K
    K_STEPS_QK = HEAD_DIM // K_STEP_QK  # 8
    D_CHUNK = 32
    D_CHUNKS = HEAD_DIM // D_CHUNK  # 4
    PV_K_STEP = 16
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2
    MFMA_LANE_K = 8

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    CAUSAL = causal
    DEFAULT_STRIDE_Q_N = NUM_HEADS_Q * HEAD_DIM
    DEFAULT_STRIDE_KV_N = NUM_HEADS_KV * HEAD_DIM

    # ── DUALWAVE_SWP LDS trait constants (matches gqa_d128_kernel_template.hpp §4-5) ──
    # K/V LDS layout: interleaved double-buffer K0, V0, K1, V1.
    # Per-warp slab line stride: smem_linear_wave + smem_padding.
    #   K: 512 + 8 = 520 bf16 per line (smem_padding_16B = 16 B = 8 bf16)
    #   V: 512 + 32 = 544 bf16 per line (smem_padding_64B = 64 B = 32 bf16)
    # Per-buffer: smem_n_rpt * smem_d_rpt * line_stride = 8 * 2 * line_stride
    #   K: 8320 bf16 (16640 B), V: 8704 bf16 (17408 B)
    # Total LDS (2 K + 2 V): 68096 B
    BF16_BYTES = 2
    D_128B_SIZE = 64  # = 128 B / sizeof(bf16) = 64 bf16
    VEC_KV = 8  # bf16 per ds_read pack (also MFMA pack_a/pack_b)
    SMEM_LINEAR_WAVE = WARP_SIZE * 16 // BF16_BYTES  # 64 * 8 = 512 bf16 per wave per "line"
    SMEM_N_PER_WAVE = SMEM_LINEAR_WAVE // D_128B_SIZE  # 8 KV rows per wave per line
    SMEM_N_RPT = BLOCK_N // SMEM_N_PER_WAVE  # 64 / 8 = 8 lines along N
    SMEM_D_RPT = HEAD_DIM // D_128B_SIZE  # 128 / 64 = 2 lines along D
    SMEM_K_PAD = 16 // BF16_BYTES  # 8 bf16 (= 16 B padding)
    SMEM_V_PAD = 64 // BF16_BYTES  # 32 bf16 (= 64 B padding)
    SMEM_K_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_K_PAD  # 520 bf16
    SMEM_V_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_V_PAD  # 544 bf16
    SMEM_K_TILE_ELEMS = SMEM_N_RPT * SMEM_D_RPT * SMEM_K_LINE_STRIDE  # 8 * 2 * 520 = 8320
    SMEM_V_TILE_ELEMS = SMEM_N_RPT * SMEM_D_RPT * SMEM_V_LINE_STRIDE  # 8 * 2 * 544 = 8704
    NUM_PREFETCH_K = 2  # DUALWAVE_SWP double-buffer
    # DUALWAVE_SWP interleaved layout: [K0][V0][K1][V1]
    DUALWAVE_SWP_KV_PER_BUFFER = SMEM_K_TILE_ELEMS + SMEM_V_TILE_ELEMS  # 17024 bf16 per (K, V) pair
    LDS_KV_TOTAL_SIZE = NUM_PREFETCH_K * DUALWAVE_SWP_KV_PER_BUFFER  # 34048 bf16 = 68096 B
    # K and V buffer bases (bf16 element offsets within the unified LDS region).
    DUALWAVE_SWP_K_BUF_BASE = (0, DUALWAVE_SWP_KV_PER_BUFFER)  # K[0]=0, K[1]=17024
    DUALWAVE_SWP_V_BUF_BASE = (
        SMEM_K_TILE_ELEMS,  # V[0]=8320
        SMEM_K_TILE_ELEMS + DUALWAVE_SWP_KV_PER_BUFFER,
    )  # V[1]=25344
    # u_rk DUALWAVE_SWP strides (per derived element strides for the 8-axis u_rk layout).
    #   N-grp y-axis (axis 2)  : stride 256 bf16 (between v_s_lo and v_s_hi)
    #   K-step axis (axes 4, 5): inner stride 16 (i_5 step), outer 4160 (i_4 d_rpt)
    DUALWAVE_SWP_URK_N_STRIP_STRIDE = 256  # bf16 offset to add for v_s_hi (n_strip=1)
    DUALWAVE_SWP_URK_KSTEP_INNER = 16  # bf16 stride between consecutive K-steps within a d_rpt
    DUALWAVE_SWP_URK_KSTEP_OUTER = SMEM_N_RPT * SMEM_K_LINE_STRIDE  # 4160 bf16 between d_rpt=0/1 arrays
    # u_rv DUALWAVE_SWP per-lane base coefficients and step strides.
    #   base_per_lane(lane) = (lane/32)*DUALWAVE_SWP_URV_GRPK + ((lane%16)/4)*DUALWAVE_SWP_URV_LANE_HI
    #                       + ((lane/16)%2)*DUALWAVE_SWP_URV_GRP_N + (lane%4)*DUALWAVE_SWP_URV_LANE_LO
    DUALWAVE_SWP_URV_GRPK = 2176  # = 4 * 544 (grp_k stride, axes 2)
    DUALWAVE_SWP_URV_LANE_HI = SMEM_V_LINE_STRIDE  # 544 (lane_hi stride, axes 3)
    DUALWAVE_SWP_URV_GRP_N = 16  # 4 (lane_lo) * 4 (VEC_TR_V) = grp_n stride
    DUALWAVE_SWP_URV_LANE_LO = 4  # VEC_TR_V (lane_lo stride)
    DUALWAVE_SWP_URV_STEP_K_STRIDE = 128  # = 2 * 64 = lane_hi_y * D_128B_SIZE (axis 4 element stride)
    DUALWAVE_SWP_URV_DC_AXIS0 = SMEM_N_RPT * SMEM_V_LINE_STRIDE  # 4352 (d_rpt array, axis 0 element stride)
    DUALWAVE_SWP_URV_DC_AXIS1 = 32  # axis 1 element stride (within half-D sub-row)
    DUALWAVE_SWP_URV_I5_STRIDE = D_128B_SIZE  # 64 (axis 5 element stride within a step_k)

    # DMA load chunking
    PATH_TAG = "DUALWAVE_SWP"
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_dualwave_swp_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * BF16_BYTES  # 68096 B for DUALWAVE_SWP K0/V0/K1/V1

    # DUALWAVE_SWP lazy-rescale threshold (line 374)
    DUALWAVE_SWP_RESCALE_THRESHOLD = 8.0

    # Enable / disable individual DUALWAVE_SWP optimizations via builder parameters.
    DUALWAVE_SWP_LAZY_RESCALE = bool(dualwave_swp_lazy_rescale)
    DUALWAVE_SWP_SETPRIO = bool(dualwave_swp_setprio)
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS = bool(dualwave_swp_debug_lazy_counts)
    DUALWAVE_SWP_ENABLE_STAGGER = bool(dualwave_swp_enable_stagger)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_dualwave_swp_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        fm_fast = fx.arith.FastMathFlags.fast
        v4i32_type = Vec.make_type(4, fx.Int32)
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type

        _MFMA_MASK = 0x008
        _VALU_MASK = 0x002
        _EXP_MASK = 0x400

        seq_len_v = fx.Index(seq_len)
        stride_q_n_v = fx.Index(stride_q_n)
        stride_kv_n_v = fx.Index(stride_kv_n)
        stride_kv_n_bytes = stride_kv_n_v * fx.Index(BF16_BYTES)

        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()
        lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)
        lds_kv_base_ptr = buffer_ops.create_llvm_ptr(lds_kv_base_idx, address_space=3)

        lds_scope_names = ("lds_k0", "lds_k1", "lds_v0", "lds_v1")

        def _lds_scope(kind, buf_id):
            return f"lds_{kind}{buf_id}"

        def _lds_alias_scopes(name):
            return _lds_alias_scope_array([name])

        def _lds_noalias_scopes(name):
            return _lds_alias_scope_array([scope_name for scope_name in lds_scope_names if scope_name != name])

        h_idx = fx.Index(gpu.block_idx.x)
        q_block_idx = fx.Index(gpu.block_idx.y)
        batch_idx = fx.Index(gpu.block_idx.z)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32

        _tid_i32 = arith.index_cast(T.i32, _raw(tid))
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32,
            arith.divsi(_tid_i32, arith.constant(WARP_SIZE, type=T.i32)),
        )
        _stagger_i32 = arith.divsi(_wave_id_uni_i32, arith.constant(4, type=T.i32))
        wave_id_uni = fx.Index(arith.index_cast(T.index, _wave_id_uni_i32))
        stagger_is_one_i1 = arith.cmpi(arith.CmpIPredicate.ne, _stagger_i32, arith.constant(0, type=T.i32))
        arith.cmpi(arith.CmpIPredicate.eq, _stagger_i32, arith.constant(0, type=T.i32))

        (lane % 16) // 4
        lane % 4
        (lane % 32) // 16

        wave_q_offset = wave_id * ROWS_PER_WAVE
        q_block_size = BLOCK_M
        q_start = q_block_idx * q_block_size

        h_kv_idx = h_idx % NUM_HEADS_KV
        group_id = h_idx // NUM_HEADS_KV
        q_head_idx = h_kv_idx * GQA_GROUP_SIZE + group_id
        kv_head_idx = h_kv_idx

        q_gmem_byte_offset = (
            (batch_idx * seq_len_v + q_start) * stride_q_n_v + q_head_idx * fx.Index(HEAD_DIM)
        ) * fx.Index(BF16_BYTES)
        kv_gmem_byte_offset = (batch_idx * seq_len_v * stride_kv_n_v + kv_head_idx * fx.Index(HEAD_DIM)) * fx.Index(
            BF16_BYTES
        )
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True, base_byte_offset=q_gmem_byte_offset)
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True, base_byte_offset=kv_gmem_byte_offset)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True, base_byte_offset=kv_gmem_byte_offset)
        o_base_ptr = _extract_aligned_pointer(O)
        o_base_i64 = llvm.PtrToIntOp(T.i64, o_base_ptr).result
        o_base_lo = ArithValue(o_base_i64).trunci(T.i32)
        o_base_hi = ArithValue(ArithValue(o_base_i64).shrui(fx.Int64(32))).trunci(T.i32)
        o_rsrc = Vec.from_elements(
            [
                o_base_lo,
                o_base_hi,
                buffer_ops._create_i32_constant(0xFFFFFFFF),
                buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags()),
            ],
            fx.Int32,
        ).ir_value()

        DMA_BYTES = 16
        NUM_DMA_K = SMEM_D_RPT
        NUM_DMA_V = SMEM_D_RPT

        _dma_size = fx.Int32(DMA_BYTES)
        _dma_off = fx.Int32(0)
        _dma_aux = fx.Int32(0)

        lane_in_warp = tid % fx.Index(WARP_SIZE)
        n_in_warp = lane_in_warp // fx.Index(VEC_KV)
        d_bucket = lane_in_warp % fx.Index(VEC_KV)

        c_neg_inf = fx.Float32(float("-inf"))
        # c_neg_inf = fx.Float32(float(-1e30))
        c_zero_f = fx.Float32(0.0)
        head_dim_f32 = fx.Float32(fx.Int32(head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                _raw(fmath.rsqrt(head_dim_f32, fastmath=fm_fast)),
                _raw(c_log2e_f),
                fastmath=fm_fast,
            )
        )
        c_eight_f = fx.Float32(DUALWAVE_SWP_RESCALE_THRESHOLD)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        fx.Int32(WARP_SIZE)
        fx.Int32(32)
        fx.Int32(4)
        fx.Int32(lane)
        v64bf16_type = Vec.make_type(K_STEPS_QK * MFMA_LANE_K, elem_dtype)
        v64f32_type = Vec.make_type(K_STEPS_QK * MFMA_LANE_K, fx.Float32)
        v32bf16_type = Vec.make_type(PV_K_STEPS * 2 * 8, elem_dtype)
        v32f32_type = Vec.make_type(PV_K_STEPS * 2 * 8, fx.Float32)

        kv_tile_size = fx.Index(BLOCK_N)
        num_kv_tiles = (seq_len_v + kv_tile_size - fx.Index(1)) // kv_tile_size
        if const_expr(CAUSAL):
            q_block_end = q_start + fx.Index(BLOCK_M)
            causal_num_tiles = (q_block_end + kv_tile_size - fx.Index(1)) // kv_tile_size
            max_num_tiles = fx.Index(ArithValue(causal_num_tiles < num_kv_tiles).select(causal_num_tiles, num_kv_tiles))
        else:
            max_num_tiles = num_kv_tiles

        urk_base_per_lane = (
            (lane_mod_32 % fx.Index(8)) * fx.Index(SMEM_K_LINE_STRIDE)
            + (lane_mod_32 // fx.Index(8)) * fx.Index(D_128B_SIZE)
            + lane_div_32 * fx.Index(VEC_KV)
        )

        urv_base_per_lane = (
            lane_div_32 * fx.Index(DUALWAVE_SWP_URV_GRPK)
            + ((lane % fx.Index(16)) // fx.Index(4)) * fx.Index(DUALWAVE_SWP_URV_LANE_HI)
            + ((lane // fx.Index(16)) % fx.Index(2)) * fx.Index(DUALWAVE_SWP_URV_GRP_N)
            + (lane % fx.Index(4)) * fx.Index(DUALWAVE_SWP_URV_LANE_LO)
        )

        _NEG_INF_F32_BITS = 0xFF800000

        _LGKMCNT_0_ONLY = 0xC07F

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        _mfma_qkpv = FaMfma32x32x16(elem_dtype, v16f32_type)

        def _mfma_acc(a, b, c):
            return _mfma_qkpv.acc(a, b, c)

        def _sched_barrier_pairs(pairs, valu_cnt, group):
            """Emit `pairs` × {1 MFMA + valu_cnt VALU} sched_group_barrier groups.

            Matches gqa_d128_kernel_template.hpp's
            `sched_barrier_pairs<Pairs, VALU_CNT, Group>()` (lines 18-23).
            """
            for _ in range_constexpr(pairs):
                rocdl.sched_group_barrier(_MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(_VALU_MASK, valu_cnt, group)

        def _sched_barrier_exp_pairs(pairs, exp_cnt, group):
            """Emit `pairs` × {1 MFMA + exp_cnt EXP} sched_group_barrier groups.

            Matches gqa_d128_kernel_template.hpp's
            `sched_barrier_exp_pairs<Pairs, EXP_CNT, Group>()` (lines 25-30).
            """
            for _ in range_constexpr(pairs):
                rocdl.sched_group_barrier(_MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(_EXP_MASK, exp_cnt, group)

        def _ds_read_tr_v4f16_imm(lds_base_elem_idx, imm_bytes):
            byte_offset = lds_base_elem_idx * 2 + lds_kv_offset
            addr_i32 = fx.Int32(byte_offset)
            return _ds_read_tr16_b64_imm(v4f16_type, addr_i32, imm_bytes)

        def _global_idx_q(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * stride_q_n_v + q_head_idx * HEAD_DIM + col

        def _concat_vectors(lhs, rhs):
            lhs_vec = Vec(lhs)
            rhs_vec = Vec(rhs)
            return lhs_vec.shuffle(
                rhs_vec,
                list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
            )

        def _raw_buffer_load_bytes(result_type, rsrc, byte_offset_i32):
            zero = buffer_ops._create_i32_constant(0)
            aux = buffer_ops._create_i32_constant(0)
            return buffer_ops.rocdl.RawPtrBufferLoadOp(
                result_type,
                rsrc,
                _raw(byte_offset_i32),
                zero,
                aux,
            ).result

        def _load_q_all(q_row_in_block):
            q_raw_packs = []
            for ks in range_constexpr(K_STEPS_QK):
                q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                g_idx = q_row_in_block * stride_q_n_v + q_col
                q_byte_offset = fx.Int32(g_idx) << fx.Int32(1)
                q_i32_pack = _raw_buffer_load_bytes(v4i32_type, q_rsrc, q_byte_offset)
                q_raw_packs.append(Vec(q_i32_pack, (4,), fx.Int32).bitcast(elem_dtype).ir_value())
            q_16_packs = []
            for pair in range_constexpr(K_STEPS_QK // 2):
                q_16_packs.append(_concat_vectors(q_raw_packs[pair * 2], q_raw_packs[pair * 2 + 1]))

            q_32_packs = []
            for pair in range_constexpr(K_STEPS_QK // 4):
                q_32_packs.append(_concat_vectors(q_16_packs[pair * 2], q_16_packs[pair * 2 + 1]))

            q_all = _concat_vectors(q_32_packs[0], q_32_packs[1])
            return Vec(q_all, (K_STEPS_QK * MFMA_LANE_K,), elem_dtype)

        def _scale_q_all(q_all_bf16):
            fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
            q_all_f32_op = llvm.FPExtOp(v64f32_type, _raw(q_all_bf16))
            q_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            q_all_f32 = q_all_f32_op.result
            scale_vec = Vec.from_elements([c_sm_scale_log2e], fx.Float32).broadcast_to(K_STEPS_QK * MFMA_LANE_K)
            q_all_scaled_f32 = arith.mulf(
                _raw(scale_vec),
                _raw(q_all_f32),
                fastmath=fm_fast,
            )
            q_all_scaled_bf16_op = llvm.FPTruncOp(v64bf16_type, q_all_scaled_f32)
            q_all_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            q_all_scaled_bf16 = q_all_scaled_bf16_op.result
            return Vec(q_all_scaled_bf16, (K_STEPS_QK * MFMA_LANE_K,), elem_dtype)

        def _get_q_pack(q_all_scaled_bf16, ks):
            q_vec = Vec(q_all_scaled_bf16)
            base = ks * MFMA_LANE_K
            return q_vec.shuffle(q_vec, [base + i for i in range(MFMA_LANE_K)]).ir_value()

        def _make_raw_buffer_rsrc(tensor):
            base_ptr = _extract_aligned_pointer(tensor)
            base_i64 = llvm.PtrToIntOp(T.i64, base_ptr).result
            base_lo = ArithValue(base_i64).trunci(T.i32)
            base_hi = ArithValue(ArithValue(base_i64).shrui(fx.Int64(32))).trunci(T.i32)
            return Vec.from_elements(
                [
                    base_lo,
                    base_hi,
                    buffer_ops._create_i32_constant(0xFFFFFFFF),
                    buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags()),
                ],
                fx.Int32,
            ).ir_value()

        debug_counts_rsrc = _make_raw_buffer_rsrc(DebugCounts) if DUALWAVE_SWP_DEBUG_LAZY_COUNTS else None

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _bitcast_f32(value):
            return fx.Float32(ArithValue(value).bitcast(fx.Float32.ir_type))

        def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
            """DUALWAVE_SWP pair mask asm: 2 compares followed by 2 cndmasks."""
            asm_str = (
                f"v_cmp_lt_i32_e64 $0, $6, {int(thr_x)}\n\t"
                f"v_cmp_lt_i32_e64 $1, $6, {int(thr_y)}\n\t"
                "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
                "v_cndmask_b32_e64 $3, $5, $7, $1"
            )
            ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
            ret = llvm.inline_asm(
                ret_struct_ty,
                [
                    _llvm_value(x_ref_i32),
                    _llvm_value(y_ref_i32),
                    _llvm_value(rel_i32),
                    _llvm_value(neg_inf_i32),
                ],
                asm_str,
                "=s,=s,=v,=v,2,3,v,v,~{vcc}",
                has_side_effects=True,
            )
            return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])

        def _anchor_pair(v_s):
            lo, hi = v_s
            lo_ir = _llvm_value(lo)
            hi_ir = _llvm_value(hi)
            ret_ty = ir.Type.parse("!llvm.struct<(vector<16xf32>, vector<16xf32>)>")
            ret = llvm.inline_asm(
                ret_ty,
                [lo_ir, hi_ir],
                "",
                "=v,=v,0,1",
                has_side_effects=True,
            )
            return (
                llvm.extractvalue(lo_ir.type, ret, [0]),
                llvm.extractvalue(hi_ir.type, ret, [1]),
            )

        def _anchor_v_p(v_p):
            p_lo, p_hi = v_p
            p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
            p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
            p_all = _concat_vectors(p_lo_all, p_hi_all)
            p_all_ir = _llvm_value(p_all)
            p_all_anchored = llvm.inline_asm(
                p_all_ir.type,
                [p_all_ir],
                "",
                "=v,0",
                has_side_effects=True,
            )
            p_vec = Vec(p_all_anchored, (PV_K_STEPS * 2 * 8,), elem_dtype)
            anchored_lo = []
            anchored_hi = []
            for pks in range_constexpr(PV_K_STEPS):
                lo_base = pks * 8
                hi_base = PV_K_STEPS * 8 + pks * 8
                anchored_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
                anchored_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
            return anchored_lo, anchored_hi

        def _v_p_to_vec32(v_p):
            p_lo, p_hi = v_p
            p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
            p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
            return _concat_vectors(p_lo_all, p_hi_all).ir_value()

        def _v_vec32_to_p(v_p_all):
            p_vec = Vec(v_p_all, (PV_K_STEPS * 2 * 8,), elem_dtype)
            p_lo = []
            p_hi = []
            for pks in range_constexpr(PV_K_STEPS):
                lo_base = pks * 8
                hi_base = PV_K_STEPS * 8 + pks * 8
                p_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
                p_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
            return p_lo, p_hi

        def _scale_v_p(v_p, scale_scalar):
            fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
            p_all = _v_p_to_vec32(v_p)
            p_all_f32_op = llvm.FPExtOp(v32f32_type, _raw(p_all))
            p_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(PV_K_STEPS * 2 * 8)
            p_scaled_f32 = arith.mulf(
                _raw(scale_vec),
                _raw(p_all_f32_op.result),
                fastmath=fm_fast,
            )
            p_scaled_bf16_op = llvm.FPTruncOp(v32bf16_type, p_scaled_f32)
            p_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            return _v_vec32_to_p(p_scaled_bf16_op.result)

        def _stagger_extra_barrier_if_one():
            """Emit `sched_barrier(0); s_barrier;` only when stagger == 1.

            Matches C++ template gqa_d128_kernel_template.hpp lines 415-418.
            The body runs on warps 4-7 only, advancing their s_barrier ordinal
            by one relative to warps 0-3 → starts the dual-group phase shift.

            Emit real CFG + ROCDL barrier ops instead of opaque inline asm, so
            LLVM sees the `llvm.amdgcn.s.barrier` intrinsic and keeps the
            conditional barrier as a scheduling boundary.
            """
            if_op = scf.IfOp(stagger_is_one_i1, [], has_else=False, loc=ir.Location.unknown())
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                scf.YieldOp([])

        def _stagger_extra_barrier_if_zero():
            """Emit `s_barrier;` only when stagger == 0.

            Matches C++ template gqa_d128_kernel_template.hpp lines 748-750.
            The body runs on warps 0-3 only, letting them catch up by one
            s_barrier ordinal before the final global store → closes the
            dual-group phase shift opened in the prologue.
            """
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [_stagger_i32],
                ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
                "s",
                has_side_effects=True,
            )

        def _bf16_trunc_pack_v8(f32_vals):
            if const_expr(dtype_str == "bf16"):
                pairs = []
                for j in range_constexpr(4):
                    pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            # fp16: truncate each f32 -> f16 (RNE) and build the v8 pack directly.
            f16_vals = []
            for i in range_constexpr(8):
                f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
            return Vec.from_elements(f16_vals, elem_dtype).ir_value()

        def _k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(DUALWAVE_SWP_K_BUF_BASE[buf_id])
            # runtime buf_id (rare): K0=0, K1=DUALWAVE_SWP_KV_PER_BUFFER
            return buf_id * fx.Index(DUALWAVE_SWP_KV_PER_BUFFER)

        def _v_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(DUALWAVE_SWP_V_BUF_BASE[buf_id])
            # runtime buf_id (rare): V0=SMEM_K_TILE_ELEMS, V1=SMEM_K_TILE_ELEMS+DUALWAVE_SWP_KV_PER_BUFFER
            return fx.Index(SMEM_K_TILE_ELEMS) + buf_id * fx.Index(DUALWAVE_SWP_KV_PER_BUFFER)

        def _async_load_k(tile_start, buf_id):
            k_lds_byte_base = lds_kv_base_idx + _k_buf_base(buf_id) * fx.Index(BF16_BYTES)
            for d in range_constexpr(NUM_DMA_K):
                lds_addr = (
                    k_lds_byte_base
                    + wave_id_uni * fx.Index(SMEM_K_LINE_STRIDE * BF16_BYTES)
                    + fx.Index(d * SMEM_N_RPT * SMEM_K_LINE_STRIDE * BF16_BYTES)
                )
                lds_ptr = buffer_ops.create_llvm_ptr(lds_addr, address_space=3)

                n_in_tile = n_in_warp * fx.Index(NUM_WAVES) + wave_id
                global_d = d_bucket * fx.Index(VEC_KV) + fx.Index(d * D_128B_SIZE)
                uniform_byte = tile_start * stride_kv_n_bytes
                lane_byte = n_in_tile * stride_kv_n_bytes + global_d * fx.Index(BF16_BYTES)
                voffset = fx.Int32(lane_byte)
                soffset = fx.Int32(uniform_byte)
                scope_name = _lds_scope("k", buf_id)
                rocdl.raw_ptr_buffer_load_lds(
                    k_rsrc,
                    lds_ptr,
                    _dma_size,
                    voffset,
                    soffset,
                    _dma_off,
                    _dma_aux,
                    alias_scopes=_lds_alias_scopes(scope_name),
                    noalias_scopes=_lds_noalias_scopes(scope_name),
                )

        def _async_load_v(tile_start, buf_id):
            v_lds_byte_base = lds_kv_base_idx + _v_buf_base(buf_id) * fx.Index(BF16_BYTES)
            for d in range_constexpr(NUM_DMA_V):
                lds_addr = (
                    v_lds_byte_base
                    + wave_id_uni * fx.Index(SMEM_V_LINE_STRIDE * BF16_BYTES)
                    + fx.Index(d * SMEM_N_RPT * SMEM_V_LINE_STRIDE * BF16_BYTES)
                )
                lds_ptr = buffer_ops.create_llvm_ptr(lds_addr, address_space=3)

                n_in_tile = n_in_warp * fx.Index(NUM_WAVES) + wave_id
                global_d = d_bucket * fx.Index(VEC_KV) + fx.Index(d * D_128B_SIZE)
                uniform_byte = tile_start * stride_kv_n_bytes
                lane_byte = n_in_tile * stride_kv_n_bytes + global_d * fx.Index(BF16_BYTES)
                voffset = fx.Int32(lane_byte)
                soffset = fx.Int32(uniform_byte)
                scope_name = _lds_scope("v", buf_id)
                rocdl.raw_ptr_buffer_load_lds(
                    v_rsrc,
                    lds_ptr,
                    _dma_size,
                    voffset,
                    soffset,
                    _dma_off,
                    _dma_aux,
                    alias_scopes=_lds_alias_scopes(scope_name),
                    noalias_scopes=_lds_noalias_scopes(scope_name),
                )

        def _reduction_pair(v_f32):
            v_i32 = _raw(_bitcast_i32(v_f32))
            pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
            lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
            rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
            return _raw(_bitcast_f32(lhs_i32)), _raw(_bitcast_f32(rhs_i32))

        def _async_load_k_from_lds_to_vgpr(buf_id, urk_base):
            """Read all 16 K MFMA packs from LDS buffer `buf_id` (DUALWAVE_SWP u_rk)."""
            k_base = _k_buf_base(buf_id)
            k_lo = [None] * K_STEPS_QK
            k_hi = [None] * K_STEPS_QK

            def _load_k_pack_aligned(elem_idx):
                scope_name = _lds_scope("k", buf_id)
                byte_offset = elem_idx * fx.Index(BF16_BYTES)
                ptr = buffer_ops.get_element_ptr(lds_kv_base_ptr, byte_offset=byte_offset, elem_type=T.i8)
                return llvm.LoadOp(
                    mfma_pack_type,
                    ptr,
                    alignment=16,
                    alias_scopes=_lds_alias_scopes(scope_name),
                    noalias_scopes=_lds_noalias_scopes(scope_name),
                ).result

            for ks in range_constexpr(K_STEPS_QK):
                ks_offset = (ks // 4) * DUALWAVE_SWP_URK_KSTEP_OUTER + (ks % 4) * DUALWAVE_SWP_URK_KSTEP_INNER
                idx_lo = k_base + urk_base + fx.Index(ks_offset)
                idx_hi = idx_lo + fx.Index(DUALWAVE_SWP_URK_N_STRIP_STRIDE)
                k_lo[ks] = _load_k_pack_aligned(idx_lo)
                k_hi[ks] = _load_k_pack_aligned(idx_hi)
            return (k_lo, k_hi)

        def _read_v_packs_for_buf(buf_id, urv_base):
            """Read all V packs from LDS buffer `buf_id` in DUALWAVE_SWP issue order.

            Returns packs indexed as [k_substep][dc], but emits the ds_read_tr16_b64
            sequence as dc outer / k_substep inner to mirror DUALWAVE_SWP's tr_load layout
            issue order.
            """
            v_base = _v_buf_base(buf_id)
            lds_base = v_base + urv_base
            packs = [[None] * D_CHUNKS for _ in range(4)]
            for dc in range_constexpr(D_CHUNKS):
                i_0 = dc // 2  # axes 0 selection: 0 → D < 64, 1 → D >= 64 (d_rpt)
                i_1 = dc % 2  # axes 1 selection: half-D sub-row group
                dc_off = i_0 * DUALWAVE_SWP_URV_DC_AXIS0 + i_1 * DUALWAVE_SWP_URV_DC_AXIS1
                for k_substep in range_constexpr(4):
                    step_k_off = k_substep * DUALWAVE_SWP_URV_STEP_K_STRIDE
                    imm_lo = (step_k_off + dc_off) * BF16_BYTES
                    # axis 5 = 0 and axis 5 = 1 reads (in-register K stride 64 bf16)
                    a = _ds_read_tr_v4f16_imm(lds_base, imm_lo)
                    b = _ds_read_tr_v4f16_imm(
                        lds_base,
                        imm_lo + DUALWAVE_SWP_URV_I5_STRIDE * BF16_BYTES,
                    )
                    packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            return packs

        def _mma0(v_k):
            k_lo, k_hi = v_k
            v_s_lo = c_zero_v16f32
            v_s_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                q_pack = _get_q_pack(q_all_scaled_bf16, ks)
                v_s_lo = _mfma_acc(k_lo[ks], q_pack, v_s_lo)
                v_s_hi = _mfma_acc(k_hi[ks], q_pack, v_s_hi)
            return (v_s_lo, v_s_hi)

        def _causal_mask_inplace(v_s, tile_idx):
            """Apply causal mask using DUALWAVE_SWP inline-asm attn_mask_vec2_imm (DUALWAVE_SWP u_rk path).

            This mirrors C++ attn_mask_causal_tile:
              k_pos = kv_start + i_n * W_N + lane_group * 4
              thr(r) = (r//4) * 8 + (r%4)

            For MFMA-C output lane k (lane_group = lane_id / 32):
              v_s_lo[r] of lane k → N = lane_group*4 + (r//4)*8 + (r%4)
              v_s_hi[r] of lane k → N = 32 + lane_group*4 + (r//4)*8 + (r%4)

            Mask if M = q_row < kv_start + N → set to -inf.
            Rewrite as `rel < thr` with:
              rel_lo = q_row - kv_start - lane_group*4
              rel_hi = rel_lo - 32
              thr(r) = (r//4)*8 + (r%4)
            """
            s_lo, s_hi = v_s
            kv_tile_start = tile_idx * fx.Index(BLOCK_N)
            kv_start_i32 = fx.Int32(kv_tile_start)
            lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
            rel_lo_i32 = fx.Int32(q_row_i32 - kv_start_i32 - lane_off_i32)
            # v_s_hi: i_n=1, so N += W_N = 32
            rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
            neg_inf_i32 = fx.Int32(_NEG_INF_F32_BITS)

            # 8 (thr_x, thr_y) pairs matching C++ attn_mask_causal_tile.
            # For r=0..15: thr(r) = (r//4)*8 + (r%4):
            #   r= 0: 0  r= 1: 1   r= 2: 2  r= 3: 3
            #   r= 4: 8  r= 5: 9   r= 6:10  r= 7:11
            #   r= 8:16  r= 9:17   r=10:18  r=11:19
            #   r=12:24  r=13:25   r=14:26  r=15:27
            pair_thresholds = [
                (0, 1),
                (2, 3),  # r=0,1  r=2,3
                (8, 9),
                (10, 11),  # r=4,5  r=6,7
                (16, 17),
                (18, 19),  # r=8,9  r=10,11
                (24, 25),
                (26, 27),  # r=12,13 r=14,15
            ]
            for p in range_constexpr(len(pair_thresholds)):
                thr_x, thr_y = pair_thresholds[p]
                idx_x = p * 2
                idx_y = p * 2 + 1

                # s_lo pair (n_strip = 0)
                x_lo_bits = _raw(_bitcast_i32(s_lo[idx_x]))
                y_lo_bits = _raw(_bitcast_i32(s_lo[idx_y]))
                new_x_lo, new_y_lo = _attn_mask_vec2_imm(
                    rel_lo_i32,
                    neg_inf_i32,
                    thr_x,
                    thr_y,
                    x_lo_bits,
                    y_lo_bits,
                )
                s_lo[idx_x] = _raw(_bitcast_f32(new_x_lo))
                s_lo[idx_y] = _raw(_bitcast_f32(new_y_lo))

            for p in range_constexpr(len(pair_thresholds)):
                thr_x, thr_y = pair_thresholds[p]
                idx_x = p * 2
                idx_y = p * 2 + 1

                # s_hi pair (n_strip = 1, rel shifted by 4)
                x_hi_bits = _raw(_bitcast_i32(s_hi[idx_x]))
                y_hi_bits = _raw(_bitcast_i32(s_hi[idx_y]))
                new_x_hi, new_y_hi = _attn_mask_vec2_imm(
                    rel_hi_i32,
                    neg_inf_i32,
                    thr_x,
                    thr_y,
                    x_hi_bits,
                    y_hi_bits,
                )
                s_hi[idx_x] = _raw(_bitcast_f32(new_x_hi))
                s_hi[idx_y] = _raw(_bitcast_f32(new_y_hi))

        def _v_s_vec_to_lists(v_s):
            s_lo, s_hi = v_s
            return (
                [Vec(s_lo)[r] for r in range_constexpr(16)],
                [Vec(s_hi)[r] for r in range_constexpr(16)],
            )

        def _v_pair_to_vec32(v):
            return _concat_vectors(v[0], v[1]).ir_value()

        def _v_vec32_to_pair(v):
            v_vec = Vec(v, (32,), fx.Float32)
            v_lo = v_vec.shuffle(v_vec, [i for i in range(16)]).ir_value()
            v_hi = v_vec.shuffle(v_vec, [16 + i for i in range(16)]).ir_value()
            return v_lo, v_hi

        def _causal_mask_prologue_if_needed(v_s, tile_idx=fx.Index(0), kv_end_pos=fx.Index(BLOCK_N)):
            """Return masked score vectors when DUALWAVE_SWP's causal guard is active."""
            s_lo, s_hi = _v_s_vec_to_lists(v_s)
            acc_values = [_raw(v) for v in (s_lo + s_hi)]
            result_types = [v.type for v in acc_values]
            mask_needed = arith.cmpi(
                arith.CmpIPredicate.slt,
                q_start_pos_i32,
                _raw(fx.Int32(kv_end_pos)),
            )
            if_op = scf.IfOp(mask_needed, result_types, has_else=True, loc=ir.Location.unknown())

            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                then_lo = list(s_lo)
                then_hi = list(s_hi)
                _causal_mask_inplace((then_lo, then_hi), tile_idx)
                scf.YieldOp([_raw(v) for v in (then_lo + then_hi)])

            if len(if_op.regions[1].blocks) == 0:
                if_op.regions[1].blocks.append(*[])
            with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                scf.YieldOp(acc_values)

            results = list(if_op.results)
            return results[:16], results[16:]

        def _attn_row_max(v_s):
            s_lo, s_hi = v_s
            m = c_neg_inf
            for r in range_constexpr(16):
                m = _fmax(m, s_lo[r])
            for r in range_constexpr(16):
                m = _fmax(m, s_hi[r])
            lhs, rhs = _reduction_pair(m)
            return _fmax(lhs, rhs)

        def _mma1_step_k(step, v_p, v_v, v_o):
            v_p_lo, v_p_hi = v_p
            v_pk = v_v[step]
            if const_expr(step < 2):
                p_pk = v_p_lo[step]
            else:
                p_pk = v_p_hi[step - 2]
            for dc in range_constexpr(D_CHUNKS):
                v_o[dc] = _mfma_acc(v_pk[dc], p_pk, v_o[dc])
            return v_o

        def _mma1(v_p, v_v, v_o):
            for step in range_constexpr(4):
                v_o = _mma1_step_k(step, v_p, v_v, v_o)
            return v_o

        def _attn_sub_row(v_s, row_max):
            s_lo, s_hi = v_s
            lo_sub = []
            hi_sub = []
            for r in range_constexpr(16):
                lo_sub.append(_fsub(s_lo[r], row_max))
            for r in range_constexpr(16):
                hi_sub.append(_fsub(s_hi[r], row_max))
            lo_vec = Vec.from_elements(lo_sub, fx.Float32).ir_value()
            hi_vec = Vec.from_elements(hi_sub, fx.Float32).ir_value()
            return lo_vec, hi_vec

        def _attn_exp2_slice(v_s, start, length):
            if const_expr(start == 0):
                s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
                lo_partial = []
                for r in range_constexpr(16):
                    lo_partial.append(rocdl.exp2(T.f32, _raw(s_lo[r])))
                return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]

            lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            hi_full = []
            for r in range_constexpr(16):
                hi_full.append(rocdl.exp2(T.f32, _raw(Vec(v_s[1])[r])))
            return lo_partial, hi_full

        def _attn_sum(v_p):
            lo_partial_list, hi_full = v_p
            local_sum = c_zero_f
            for r in range_constexpr(16):
                local_sum = _fadd(local_sum, lo_partial_list[r])
            for r in range_constexpr(16):
                local_sum = _fadd(local_sum, hi_full[r])
            lhs_sum, rhs_sum = _reduction_pair(local_sum)
            return _fadd(lhs_sum, rhs_sum)

        def _cast_p(v_p):
            lo_partial_list, hi_full = v_p
            p_lo_packs = []
            p_hi_packs = []
            for pks in range_constexpr(PV_K_STEPS):
                p_base = pks * 8
                lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
                p_lo_packs.append(_bf16_trunc_pack_v8(lo_slice))
                hi_slice = hi_full[p_base : p_base + 8]
                p_hi_packs.append(_bf16_trunc_pack_v8(hi_slice))
            return p_lo_packs, p_hi_packs

        def _scale_o(v_o, scale_scalar):
            scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
            for dc in range_constexpr(D_CHUNKS):
                v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec)

        def _anchor_v_o(v_o):
            """Pin v_o accumulators at the current source position.

            Mirrors the C++ pattern:
              asm volatile("" : "+v"(v_o_pin[0]), ... ::);
            """
            acc_irs = [_llvm_value(v_o[dc]) for dc in range_constexpr(D_CHUNKS)]
            ret_ty = ir.Type.parse("!llvm.struct<(vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)>")
            ret = llvm.inline_asm(
                ret_ty,
                acc_irs,
                "",
                "=v,=v,=v,=v,0,1,2,3",
                has_side_effects=True,
            )
            return [llvm.extractvalue(acc_irs[dc].type, ret, [dc]) for dc in range_constexpr(D_CHUNKS)]

        def _debug_atomic_inc_lazy_count(byte_offset):
            rocdl.raw_buffer_atomic_fadd(
                _llvm_value(fx.Float32(1.0)),
                debug_counts_rsrc,
                _llvm_value(fx.Int32(byte_offset)),
                _llvm_value(fx.Int32(0)),
                _llvm_value(fx.Int32(0)),
            )

        def _debug_count_lazy_branch(all_below):
            if const_expr(not DUALWAVE_SWP_DEBUG_LAZY_COUNTS):
                return
            lane_i32 = arith.index_cast(T.i32, _raw(lane))
            lane_is_zero = arith.cmpi(
                arith.CmpIPredicate.eq,
                lane_i32,
                arith.constant(0, type=T.i32),
            )
            lane_if = scf.IfOp(
                lane_is_zero,
                [],
                has_else=False,
                loc=ir.Location.unknown(),
            )
            with ir.InsertionPoint(lane_if.regions[0].blocks[0]):
                branch_if = scf.IfOp(
                    all_below,
                    [],
                    has_else=True,
                    loc=ir.Location.unknown(),
                )
                with ir.InsertionPoint(branch_if.regions[0].blocks[0]):
                    _debug_atomic_inc_lazy_count(0)
                    scf.YieldOp([])
                if len(branch_if.regions[1].blocks) == 0:
                    branch_if.regions[1].blocks.append(*[])
                with ir.InsertionPoint(branch_if.regions[1].blocks[0]):
                    _debug_atomic_inc_lazy_count(4)
                    scf.YieldOp([])
                scf.YieldOp([])

        def _anchor_scalar_f32(x):
            """Pin a scalar f32 at the current source position (no-op asm).

            Used to define a value *inside* a branch region so LLVM cannot hoist
            it above the branch and fold a trivial 2-entry PHI into a `select`
            (which would force a VCC mask + v_cndmask on AMDGPU).
            """
            x_ir = _llvm_value(x)
            return llvm.inline_asm(
                x_ir.type,
                [x_ir],
                "",
                "=v,0",
                has_side_effects=True,
            )

        def _lazy_rescale_o(v_o, m_row, l_row, m_tile_max, v_p):
            """DUALWAVE_SWP lazy rescale before the remaining MMA1 steps.

            Cold path also scales v_p so the later step_k(1..3) contributions
            are accumulated in the new row-max basis.
            """
            m_diff = _fsub(m_tile_max, m_row)
            below = ArithValue(fx.Float32(m_diff) <= c_eight_f)
            ballot = rocdl.ballot(T.i64, _raw(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq,
                _raw(ballot),
                _read_exec_i64(),
            )
            _expect_true_i1 = arith.constant(1, type=ir.IntegerType.get_signless(1))
            all_below = llvm.intr_expect(all_below, _expect_true_i1)
            # all_below = fx.Boolean(False).ir_value()
            _debug_count_lazy_branch(all_below)

            current_values = [_raw(acc) for acc in v_o] + [_raw(m_row), _raw(l_row), _raw(_v_p_to_vec32(v_p))]
            result_types = [value.type for value in current_values]
            if_op = scf.IfOp(all_below, result_types, has_else=True, loc=ir.Location.unknown())

            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                scf.YieldOp(current_values)

            if len(if_op.regions[1].blocks) == 0:
                if_op.regions[1].blocks.append(*[])
            with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                corr = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_tile_max)))
                scaled_accs = list(v_o)
                _scale_o(scaled_accs, corr)
                scaled_v_p = _scale_v_p(v_p, corr)
                scaled_l_row = _fmul(l_row, corr)
                # Anchor m_tile_max *inside* the else so the merged m_row stays a
                # PHI (resolved as a copy in the cold path, like C++'s
                # `v_mov v219, v24`) instead of being folded into a select.
                m_tile_max_else = _anchor_scalar_f32(m_tile_max)
                scf.YieldOp(
                    [_raw(acc) for acc in scaled_accs]
                    + [m_tile_max_else, _raw(scaled_l_row), _raw(_v_p_to_vec32(scaled_v_p))]
                )

            results = list(if_op.results)
            return (
                results[:D_CHUNKS],
                results[D_CHUNKS],
                results[D_CHUNKS + 1],
                _v_vec32_to_p(results[D_CHUNKS + 2]),
            )

        # Prologue
        # Kick off the async (DMA) copy of the first K tile (kv_tile 0) from
        # global memory into LDS buffer 0, then wait for *all* outstanding
        # counters (s_waitcnt 0) so the LDS data is guaranteed visible. The
        # sched_barrier(0) is a scheduling fence (emits no instruction) that
        # stops LLVM from moving work across it, and s_barrier syncs the whole
        # workgroup so every wave sees the loaded K tile before MMA0.
        _async_load_k(fx.Index(0), 0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        # Load this wave's Q rows and pre-scale by the softmax temperature.
        # q_row_in_block: row index within the Q tile for this lane
        #   (wave base offset + lane's position inside the 32-lane group).
        # q_start_pos_i32 / q_row(_i32): absolute Q row position, used later for
        #   the causal mask comparison.
        # _load_q_all loads the full head_dim of Q for this lane (bf16);
        # _scale_q_all multiplies by 1/sqrt(D) temperature so MMA0 directly
        # produces scaled scores (mirrors the C++ v_q *= temperature_scale).
        q_row_in_block = wave_q_offset + lane_mod_32
        q_start_pos_i32 = fx.Int32(q_start + wave_id_uni * fx.Index(ROWS_PER_WAVE))
        q_row = q_start + q_row_in_block
        q_row_i32 = fx.Int32(q_row)
        q_all_bf16 = _load_q_all(q_row_in_block)
        q_all_scaled_bf16 = _scale_q_all(q_all_bf16)

        # Software-pipeline the next tiles while consuming the first one:
        #   - prefetch K tile 1 into LDS buffer 1, and V tile 0 into V LDS buf 0
        #     (these DMAs run in the background during the upcoming MMA0);
        #   - read the already-resident K tile 0 from LDS into VGPRs (v_k) for
        #     the prologue MMA0.
        # s_waitcnt(lgkmcnt==0) waits only for the LDS read (v_k) to land;
        # _waitcnt_vm_n(NUM_DMA_V) waits for just the V global loads, leaving
        # the K-tile-1 DMA still in flight (overlapped with compute).
        _async_load_k(fx.Index(BLOCK_N), 1)
        _async_load_v(fx.Index(0), 0)
        v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_V)

        # Wave-group stagger -- OPENS the phase shift. 8 waves split into group A
        # (waves 0-3) and group B (waves 4-7); _stagger_extra_barrier_if_one()
        # makes ONLY group B run one extra s_barrier here. As s_barriers match
        # by ordinal across the workgroup, every later group-B s_barrier then
        # aligns with group A's NEXT cluster's s_barrier -- i.e. group A runs one
        # cluster AHEAD of group B for the whole main loop/epilogue, so one group
        # computes while the other loads (see the diagram at "(4)" above the
        # loop). Closed in the epilogue. Disabled -> one plain barrier (lock-step).
        if const_expr(DUALWAVE_SWP_ENABLE_STAGGER):
            _stagger_extra_barrier_if_one()  # group B: +1 s_barrier -> open the shift
        else:
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

        # Prologue score computation + first softmax pass for KV tile 0:
        #   - _mma0: QK^T -> raw scores v_s_0 (scaled by the Q pre-scale above);
        #   - causal mask: for causal attention, mask out future keys in this
        #     tile (sets them to -inf); non-causal just reshapes to lists;
        #   - _attn_row_max: per-row running max m_row (softmax stability);
        #   - _attn_sub_row: scores - m_row;
        #   - _attn_exp2_slice(0,16): exp2 of the first half of the scores
        #     (the second half is computed later, in the main-loop Cluster 1,
        #     to overlap with the next MMA0).
        # The trailing sched_barrier/s_barrier pair fences this softmax work
        # and syncs the workgroup before the next K prefetch.
        v_s_0 = _mma0(v_k)
        rocdl.sched_barrier(0)
        if const_expr(CAUSAL):
            v_s_0 = _causal_mask_prologue_if_needed(v_s_0)
        else:
            v_s_0 = _v_s_vec_to_lists(v_s_0)
        m_row_pro = _attn_row_max(v_s_0)
        v_s_0 = _attn_sub_row(v_s_0, m_row_pro)
        v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Prefetch K tile 2 into LDS buffer 0 (reusing buffer 0 now that tile 0
        # has been consumed). This keeps the K double-buffer one step ahead of
        # the main loop, which starts at j=3.
        _async_load_k(fx.Index(2 * BLOCK_N), 0)

        # Build the loop-carried state for the main loop (scf.for init args):
        #   [0] m_row  : running row max (seeded from the prologue tile);
        #   [1] l_row  : running softmax denominator, starts at 0;
        #   [2..]      : D_CHUNKS output accumulator banks v_o, all zero;
        #   [last]     : packed v_p_0 (the prologue tile's exp'd probabilities)
        #                carried in as vec32 so it can be passed through scf.for.
        l_row_init = c_zero_f
        init_args = [m_row_pro, l_row_init]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        init_args.append(_v_pair_to_vec32(v_p_0))

        # ============================= Main loop =============================
        # Software-pipelined flash-attention inner loop. Each iteration advances
        # j by 2 and folds TWO KV tiles into the running (m_row, l_row, v_o)
        # state. The body is 8 "clusters" C0..C7 that strictly alternate a
        # MEMORY stage (even: C0/C2/C4/C6) and a COMPUTE stage (odd:
        # C1/C3/C5/C7); every cluster ends with rocdl.s_barrier(). K and V each
        # live in a 2-deep LDS double buffer (buf0/buf1).
        #
        # (1) Global memory -> LDS loads (background DMA, issued tiles ahead),
        #     alternating V/K across the memory stages and ping-ponging buffers:
        #       C0  _async_load_v(.., buf1)   V tile (j-2) -> V-LDS buf1
        #       C2  _async_load_k(.., buf1)   K tile (j)   -> K-LDS buf1
        #       C4  _async_load_v(.., buf0)   V tile (j-1) -> V-LDS buf0
        #       C6  _async_load_k(.., buf0)   K tile (j+1) -> K-LDS buf0
        #     The DMA is left in flight; each stage's s_waitcnt only drains what
        #     the next compute stage actually needs (the rest stays overlapped).
        #
        # (2) LDS -> VGPR reads (each feeds the immediately following compute):
        #       C0  K LDS buf1 -> v_k        (-> Q*K in C1)
        #       C2  V LDS buf0 -> v_v        (-> P*V in C3)
        #       C4  K LDS buf0 -> v_k        (-> Q*K in C5)
        #       C6  V LDS buf1 -> v_packs    (-> P*V in C7)
        #
        # (3) Compute order + softmax split (all in the odd/compute clusters):
        #       Q*K (mma0):  C1 -> v_s_1 (tile A) ; C5 -> v_s_0 (tile B)
        #       P*V (mma1):  C3 (4x step_k)       ; C7 (4x step_k)
        #     A tile's softmax is deliberately split across TWO compute clusters
        #     so the exp (TRANS) latency hides behind MFMA. For tile A (scores
        #     produced by C1):
        #       C3: _attn_row_max -> _attn_sub_row(S - m) -> _attn_exp2_slice
        #           FIRST half (elems 0..15), plus _lazy_rescale_o (rescale the
        #           running v_o / l_row to the new combined row max);
        #       C5: _attn_exp2_slice SECOND half (16..31) -> _attn_sum (row sum
        #           into l_row) -> _cast_p (probabilities -> bf16);
        #       C7: P*V consumes those bf16 probabilities.
        #     Tile B (scores from C5) runs the same chain shifted by 4 clusters:
        #     C7 (part 1) -> C1-of-next-iter (part 2) -> C3-of-next-iter (P*V).
        #     So every compute cluster runs P*V for an OLDER tile while doing
        #     softmax-part-1 for a NEWER tile, keeping MFMA and exp/VALU
        #     interleaved (shaped by the _sched_barrier_*_pairs hints).
        #
        # (4) Two wave-groups: 8 waves split into group A (waves 0-3) and group
        #     B (waves 4-7). The prologue's _stagger_extra_barrier_if_one()
        #     makes ONLY group B run one extra s_barrier. s_barriers are matched
        #     across the workgroup by ordinal position, and every cluster ends
        #     with exactly one s_barrier, so from that point on each group-B
        #     s_barrier aligns with group A's NEXT cluster's s_barrier.
        #     Equivalently: GROUP A RUNS ONE CLUSTER AHEAD OF GROUP B for the
        #     whole main loop and epilogue. Lining the two groups up by their
        #     matched s_barriers ('b', where the columns are forced to align):
        #
        #       group A:  [ P0+QK0 ]--b--[   C0   ]--b--[   C1   ]--b--[   C2   ]
        #       group B:  [   P0   ]--b--[  QK0   ]--b--[   C0   ]--b--[   C1   ]
        #         P0  = pre-loop: prefetch K tile1 + read K tile0 (LDS->VGPR)
        #         QK0 = prologue Q*K + first softmax pass (KV tile 0)
        #         C0  = main-loop Cluster 0 (also prefetches K tile2); C1, C2 ...
        #       (group B's first 'b' is the extra stagger barrier: it has only
        #        done P0 while group A has already done P0+QK0.)
        #
        #     Because of this one-cluster offset, whenever one group is in a
        #     COMPUTE cluster (MFMA/VALU) the other is in the adjacent MEMORY
        #     cluster (DMA issue + LDS reads + waitcnt stalls): one group
        #     computes while the other loads, and they swap at each s_barrier,
        #     so the memory latency of one group is hidden behind the MFMA of
        #     the other. The offset is closed at the end of the epilogue, where
        #     _stagger_extra_barrier_if_zero() gives group A its matching +1.
        #
        # (5) rocdl.s_setprio(1)/(0) time-multiplexes the shared SIMD issue/MFMA
        #     resources between the two groups. It wraps only the heaviest
        #     compute regions, C3 and C7 (P*V + lazy rescale + first-half exp2):
        #       - s_setprio(1) at the cluster start raises the computing group's
        #         priority so its MFMA chain issues without being preempted;
        #       - s_setprio(0) at the cluster end drops priority, ceding the
        #         units to the other group that is just entering its compute
        #         phase.
        #     This explicit hand-off keeps the "one computes / one loads"
        #     alternation crisp and prevents both groups from contending for the
        #     MFMA pipe at the same time.
        # =====================================================================
        loop_results = init_args
        for j, loop_args in range(
            fx.Index(3),
            max_num_tiles - fx.Index(1),
            fx.Index(2),
            init=init_args,
        ):
            m_row = loop_args[0]
            l_row = loop_args[1]
            v_o = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            v_p_0 = _v_vec32_to_pair(loop_args[2 + D_CHUNKS])
            j_idx = j

            # Main loop Cluster 0: memory stage for the current K tile.
            # The body processes two KV tiles per iteration; clusters alternate
            # memory stages (even) and compute stages (odd), double-buffered.
            # Here: kick off the next V-tile DMA (buffer 1) in the background,
            # read the already-resident K tile from LDS into VGPRs (v_k) for
            # MMA0, then wait only for the LDS read + needed DMAs and sync the
            # workgroup so all waves see the K data before MMA0.
            _async_load_v((j_idx - fx.Index(2)) * fx.Index(BLOCK_N), 1)
            v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Main loop Cluster 1: new scores + finish the previous tile's softmax.
            # _mma0 computes QK^T -> v_s_1 for this tile. In parallel, finish the
            # second-half exp2 of the previous tile's probabilities (v_p_0),
            # accumulate its row sum into the running denominator l_row, cast the
            # probabilities to bf16, and anchor them as inputs for the P*V MMA.
            # The sched_group_barrier hints keep exp2/MFMA interleaved.
            v_s_1 = _mma0(v_k)
            v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
            tile_sum_a = _attn_sum(v_p_0)
            l_row = _fadd(l_row, tile_sum_a)
            v_p_0 = _cast_p(v_p_0)
            v_p_0 = _anchor_v_p(v_p_0)
            _sched_barrier_exp_pairs(6, 3, 1)
            _sched_barrier_pairs(10, 5, 1)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Main loop Cluster 2: memory stage for V reads / next K prefetch.
            # Prefetch the next K tile (buffer 1) in the background, read this
            # tile's V from LDS into VGPR packs (v_v) for the P*V MMA, wait for
            # the LDS read + DMAs, and sync.
            _async_load_k(j_idx * fx.Index(BLOCK_N), 1)
            v_v = _read_v_packs_for_buf(0, urv_base_per_lane)
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Main loop Cluster 3: P*V accumulation + lazy rescale + next softmax.
            # Raise wave priority, do the first P*V MFMA step into the output
            # accumulators v_o, and reduce this tile's scores v_s_1 to per-row
            # max (m_tile_max_a). _lazy_rescale_o then conditionally rescales the
            # running output/denominator to the new combined row max (the branch
            # is skipped when all lanes are already within RESCALE_THRESHOLD).
            # Then the remaining 3 P*V steps, subtract the row max from the new
            # scores, and the first-half exp2. s_setprio(0) + fence + barrier
            # close the cluster (yielding the MFMA units to the other group).
            if const_expr(DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(1)
            v_o = _mma1_step_k(0, v_p_0, v_v, v_o)
            v_s_1 = _v_s_vec_to_lists(v_s_1)
            m_tile_max_a = _attn_row_max(v_s_1)

            # LLVM scheduling note:
            # `_sched_barrier_pairs(4, 5, 2)` emits four repetitions of:
            #   rocdl.sched.group.barrier(_MFMA_MASK=0x008, 1, groupId=2)
            #   rocdl.sched.group.barrier(_VALU_MASK=0x002, 5, groupId=2)
            # In MLIR this is `rocdl.sched.group.barrier mask, size, groupId`;
            # ROCDL lowers it to `llvm.amdgcn.sched.group.barrier`.
            # LLVM defines the intrinsic in `llvm/include/llvm/IR/IntrinsicsAMDGPU.td`:
            #   - mask selects the instruction class to synchronize,
            #   - size is how many matching instructions belong to this group,
            #   - groupId ties multiple group barriers into one synchronized pipeline.
            # The AMDGPU IGroupLP mutation (`AMDGPUIGroupLP.cpp`) turns these
            # pseudo ops into artificial scheduling edges. Its `SchedGroup`
            # classifier maps mask 0x008 to MFMA/WMMA and mask 0x002 to VALU.
            # Directionality: `sched.group.barrier` applies to candidate
            # instructions above it, not to future instructions below it.
            # `IGroupLPDAGMutation::initSchedGroupBarrierPipelineStage()` scans
            # from the barrier's reverse iterator to `SUnits.rend()`, i.e. from
            # this marker back toward the beginning of the scheduling region.
            # Multiple group barriers with the same groupId are then solved as
            # one pipeline, so they collectively order those previously emitted
            # MFMA/VALU candidates, but they do not directly capture operations
            # that are written after this marker.
            #
            # Here, the source is at the start of Cluster 3, after the first
            # P*V MFMA step and row-max reduction. The hint asks LLVM to keep a
            # local pattern of 4 MFMA groups interleaved with windows of 5 VALU
            # instructions in scheduling group 2. This is not a hardware
            # barrier and emits no real ISA instruction; it only constrains the
            # pre/post-RA scheduler so the following lazy-rescale/score-update
            # VALU does not drift into a worse position relative to the MFMA
            # chain.
            _sched_barrier_pairs(4, 6, 2)

            if const_expr(DUALWAVE_SWP_LAZY_RESCALE):
                v_o, m_row, l_row, v_p_0 = _lazy_rescale_o(v_o, m_row, l_row, m_tile_max_a, v_p_0)
            else:
                m_new_a = _fmax(m_row, m_tile_max_a)
                corr_a = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_a)))
                _scale_o(v_o, corr_a)
                v_o = _anchor_v_o(v_o)
                v_p_0 = _scale_v_p(v_p_0, corr_a)
                l_row = _fmul(l_row, corr_a)
                m_row = m_new_a
            v_o = _mma1_step_k(1, v_p_0, v_v, v_o)
            v_o = _mma1_step_k(2, v_p_0, v_v, v_o)
            v_o = _mma1_step_k(3, v_p_0, v_v, v_o)
            v_s_1 = _attn_sub_row(v_s_1, m_row)
            v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)

            _sched_barrier_pairs(6, 6, 2)
            # LLVM scheduling note:
            # `_sched_barrier_exp_pairs(6, 3, 2)` emits six repetitions of:
            #   rocdl.sched.group.barrier(_MFMA_MASK=0x008, 1, groupId=2)
            #   rocdl.sched.group.barrier(_EXP_MASK=0x400, 3, groupId=2)
            # The 0x400 mask is `SchedGroupMask::TRANS` in
            # `AMDGPUIGroupLP.cpp`; `SchedGroup::canAddMI()` classifies
            # transcendental operations through `TII->isTRANS(MI)`, which
            # covers the `v_exp_f32` generated by `_attn_exp2_slice`.
            # Like `_sched_barrier_pairs` above, this only gathers matching
            # instructions that have already appeared before the marker.
            # LLVM's implementation calls `SG.findCandidateSUnits(RIter,
            # SG.DAG->SUnits.rend(), ...)`, so the six MFMA/EXP groups are a
            # bottom-up description of the just-emitted MFMA/sub/exp work, not a
            # constraint on later Cluster 4 loads.
            #
            # Semantically, this point is after:
            #   - the remaining P*V MFMA steps,
            #   - subtraction of the selected `m_row`,
            #   - `attn_exp2_slice(v_s_1, 0, 16)`.
            # The hint therefore tells LLVM to schedule a second wave of the
            # group-2 pipeline as 6 MFMA groups, each paired with 3 EXP/TRANS
            # operations. This keeps the newly-created softmax exp work close to
            # the MFMA window where it was intended to sit, instead of letting
            # the scheduler bunch all `v_exp_f32` or all MFMA instructions
            # together.
            _sched_barrier_exp_pairs(6, 3, 2)
            if const_expr(DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(0)
            # LLVM scheduling note:
            # `rocdl.sched_barrier(0)` lowers to
            # `llvm.amdgcn.sched.barrier(i32 0)`, then to the AMDGPU
            # `SCHED_BARRIER` pseudo instruction. In `ROCDLOps.td` and
            # `IntrinsicsAMDGPU.td`, the mask is defined as the set of
            # instruction classes that may cross the barrier. Mask 0 means
            # "allow none": MFMA, VALU, SALU, TRANS, VMEM, and DS instructions
            # are all barred from being scheduled across this point.
            #
            # LLVM enforces this in `IGroupLPDAGMutation::addSchedBarrierEdges`
            # (`llvm/lib/Target/AMDGPU/AMDGPUIGroupLP.cpp`): it inverts the mask,
            # classifies all matching SUnits, and adds artificial DAG edges to
            # preserve their original order relative to the barrier. This is a
            # compiler scheduling fence only; the real cross-wave synchronization
            # is the following `rocdl.s_barrier()`.
            # `rocdl.s_barrier()` itself cannot be moved across this
            # `sched_barrier(0)`: the same file's
            # `SIInstrInfo::isSchedulingBoundary()` returns true for
            # `SCHED_BARRIER` with immediate 0, so the machine scheduler splits
            # the scheduling region at this point before it can reorder the
            # following real `S_BARRIER`.
            #
            # Placing the fence immediately after `s_setprio(0)` closes Cluster
            # 3 before the hardware `s_barrier`. It prevents independent MFMA,
            # exp, rescale, or memory instructions from being hoisted across the
            # priority drop and wave synchronization boundary, so Cluster 4's
            # next K/V prefetch starts from a clean phase boundary.
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Main loop Cluster 4: memory stage for the second tile (LDS buf 0).
            # Mirror of Cluster 0 for the other pipeline phase: prefetch V
            # (buffer 0), read K from LDS buffer 0 into v_k, wait + sync.
            _async_load_v((j_idx - fx.Index(1)) * fx.Index(BLOCK_N), 0)
            v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Main loop Cluster 5: scores for the second tile + finish its softmax.
            # Mirror of Cluster 1: _mma0 -> v_s_0; finish the second-half exp2 of
            # v_p_1 (the tile-B probabilities produced in Cluster 3), accumulate
            # its sum into l_row, cast to bf16 and anchor.
            v_s_0 = _mma0(v_k)
            v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
            tile_sum_b = _attn_sum(v_p_1)
            l_row = _fadd(l_row, tile_sum_b)
            v_p_1 = _cast_p(v_p_1)
            v_p_1 = _anchor_v_p(v_p_1)
            _sched_barrier_exp_pairs(6, 3, 3)
            _sched_barrier_pairs(10, 5, 3)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Main loop Cluster 6: memory stage + causal mask.
            # Prefetch the next K tile (buffer 0) and read this tile's V packs
            # (buffer 1). For causal attention, apply the causal mask to the
            # freshly computed v_s_0 scores (mask out future keys); non-causal
            # just reshapes to lists. Wait + sync.
            _async_load_k((j_idx + fx.Index(1)) * fx.Index(BLOCK_N), 0)
            v_packs_b = _read_v_packs_for_buf(1, urv_base_per_lane)
            if const_expr(CAUSAL):
                v_s_0 = _causal_mask_prologue_if_needed(
                    v_s_0,
                    j_idx - fx.Index(1),
                    j_idx * fx.Index(BLOCK_N),
                )
            else:
                v_s_0 = _v_s_vec_to_lists(v_s_0)
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Main loop Cluster 7: P*V accumulation + lazy rescale (second phase).
            # Mirror of Cluster 3 for v_p_1 / v_s_0: first P*V step, per-row max
            # of v_s_0, lazy rescale of the running output/denominator, remaining
            # 3 P*V steps, then subtract the row max and first-half exp2 of v_s_0.
            # Closes the iteration; yield_args carries the updated running state
            # (m_row, l_row, v_o accumulators, packed v_p_0) to the next iter.
            if const_expr(DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(1)
            v_v = v_packs_b
            v_o = _mma1_step_k(0, v_p_1, v_v, v_o)
            m_tile_max_b = _attn_row_max(v_s_0)
            _sched_barrier_pairs(4, 6, 4)

            if const_expr(DUALWAVE_SWP_LAZY_RESCALE):
                v_o, m_row, l_row, v_p_1 = _lazy_rescale_o(v_o, m_row, l_row, m_tile_max_b, v_p_1)
            else:
                m_new_b = _fmax(m_row, m_tile_max_b)
                corr_b = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_b)))
                _scale_o(v_o, corr_b)
                v_o = _anchor_v_o(v_o)
                v_p_1 = _scale_v_p(v_p_1, corr_b)
                l_row = _fmul(l_row, corr_b)
                m_row = m_new_b
            v_v = v_packs_b
            v_o = _mma1_step_k(1, v_p_1, v_v, v_o)
            v_o = _mma1_step_k(2, v_p_1, v_v, v_o)
            v_o = _mma1_step_k(3, v_p_1, v_v, v_o)
            v_s_0 = _attn_sub_row(v_s_0, m_row)
            v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
            _sched_barrier_pairs(6, 5, 4)
            _sched_barrier_exp_pairs(6, 3, 4)
            if const_expr(DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            yield_args = [m_row, l_row] + v_o + [_v_pair_to_vec32(v_p_0)]
            loop_results = yield yield_args

        # Epilogue: drain the software pipeline for the final KV tiles that the
        # main loop left in flight (the loop stops at max_num_tiles - 1 because
        # of its prefetch-ahead depth). First unpack the loop-carried state:
        #   running row max, running denominator l_row, output accumulators v_o,
        #   and the still-in-flight probabilities v_p_0. The epilogue clusters
        #   mirror the main-loop clusters but with no further prefetch-ahead and
        #   with unconditional (non-lazy) rescale, since this is the tail.
        m_row = loop_results[0]
        l_row = loop_results[1]
        v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]
        v_p_0 = _v_vec32_to_pair(loop_results[2 + D_CHUNKS])

        # Tile indices for the last three tiles handled by the epilogue.
        max_m3 = max_num_tiles - fx.Index(3)
        max_m2 = max_num_tiles - fx.Index(2)
        max_m1 = max_num_tiles - fx.Index(1)

        # Epilogue Cluster 0: memory stage (like main-loop Cluster 0).
        # Prefetch the V tile for max_m3 (buffer 1), read the resident K tile
        # from LDS buffer 1 into v_k, wait + sync.
        _async_load_v(max_m3 * fx.Index(BLOCK_N), 1)
        v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 1: scores + finish the carried tile's softmax.
        # _mma0 -> v_s_1; finish v_p_0's second-half exp2, add its sum into
        # l_row, cast to bf16 and anchor (like main-loop Cluster 1).
        v_s_1 = _mma0(v_k)
        v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
        tile_sum_e1 = _attn_sum(v_p_0)
        l_row = _fadd(l_row, tile_sum_e1)
        v_p_0 = _cast_p(v_p_0)
        v_p_0 = _anchor_v_p(v_p_0)
        _sched_barrier_exp_pairs(6, 3, 5)
        _sched_barrier_pairs(10, 5, 5)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 2: memory stage + causal mask.
        # Prefetch the K tile for max_m1, read V packs (buffer 0), and for
        # causal attention mask the v_s_1 scores. Wait + sync.
        _async_load_k(max_m1 * fx.Index(BLOCK_N), 1)
        v_packs_e3 = _read_v_packs_for_buf(0, urv_base_per_lane)
        if const_expr(CAUSAL):
            v_s_1 = _causal_mask_prologue_if_needed(
                v_s_1,
                max_m3,
                max_m2 * fx.Index(BLOCK_N),
            )
        else:
            v_s_1 = _v_s_vec_to_lists(v_s_1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 3: full P*V + unconditional rescale.
        # Unlike the main loop's lazy rescale, the epilogue always rescales:
        # full _mma1 (all 4 P*V steps) for v_p_0, compute the new combined row
        # max, rescale factor exp2(m_row - new_max), update m_row, subtract the
        # row max from v_s_1 and first-half exp2, then scale the output v_o by
        # the rescale factor and anchor it. s_setprio(0) + fence + barrier.
        if const_expr(DUALWAVE_SWP_SETPRIO):
            rocdl.s_setprio(1)
        v_o = _mma1(v_p_0, v_packs_e3, v_o)
        m_tile_max_e3 = _attn_row_max(v_s_1)
        row_max_e3 = _fmax(m_row, m_tile_max_e3)
        rescale_e3 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e3)))
        m_row = row_max_e3
        v_s_1 = _attn_sub_row(v_s_1, row_max_e3)
        v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
        _sched_barrier_pairs(10, 5, 6)
        _sched_barrier_exp_pairs(6, 3, 6)
        rocdl.sched_barrier(0)
        _scale_o(v_o, rescale_e3)
        v_o = _anchor_v_o(v_o)

        if const_expr(DUALWAVE_SWP_SETPRIO):
            rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 4: memory stage (buffer 0).
        # Prefetch the V tile for max_m2 (buffer 0), read K from LDS buffer 0
        # into v_k, wait + sync.
        _async_load_v(max_m2 * fx.Index(BLOCK_N), 0)
        v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 5: scores + finish previous softmax + apply rescale.
        # _mma0 -> v_s_0; also fold the Cluster-3 rescale into the running
        # denominator (l_row *= rescale_e3), finish v_p_1's second-half exp2,
        # add its sum into l_row, cast + anchor.
        v_s_0 = _mma0(v_k)
        l_row = _fmul(l_row, rescale_e3)
        v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
        tile_sum_e5 = _attn_sum(v_p_1)
        l_row = _fadd(l_row, tile_sum_e5)
        v_p_1 = _cast_p(v_p_1)
        v_p_1 = _anchor_v_p(v_p_1)
        _sched_barrier_exp_pairs(6, 3, 7)
        _sched_barrier_pairs(10, 5, 7)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 6: memory stage + causal mask.
        # Read V packs (buffer 1) for the next P*V, mask v_s_0 for causal, then
        # wait (only V DMAs remain outstanding now) + sync.
        v_packs_e7 = _read_v_packs_for_buf(1, urv_base_per_lane)
        if const_expr(CAUSAL):
            v_s_0 = _causal_mask_prologue_if_needed(
                v_s_0,
                max_m2,
                max_m1 * fx.Index(BLOCK_N),
            )
        else:
            v_s_0 = _v_s_vec_to_lists(v_s_0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_V)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 7: full P*V + unconditional rescale (mirror of 3).
        # Full _mma1 for v_p_1, new combined row max, rescale_e7, update m_row,
        # sub_row + first-half exp2 of v_s_0, scale v_o, anchor. Fence + barrier.
        if const_expr(DUALWAVE_SWP_SETPRIO):
            rocdl.s_setprio(1)
        v_o = _mma1(v_p_1, v_packs_e7, v_o)
        m_tile_max_e7 = _attn_row_max(v_s_0)
        row_max_e7 = _fmax(m_row, m_tile_max_e7)
        rescale_e7 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e7)))
        m_row = row_max_e7
        v_s_0 = _attn_sub_row(v_s_0, row_max_e7)
        v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
        _sched_barrier_pairs(10, 5, 8)
        _sched_barrier_exp_pairs(6, 3, 8)
        rocdl.sched_barrier(0)
        _scale_o(v_o, rescale_e7)
        v_o = _anchor_v_o(v_o)
        if const_expr(DUALWAVE_SWP_SETPRIO):
            rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 8: memory stage for the last tile (buffer 1).
        # Prefetch V for max_m1 (buffer 1), read K from LDS buffer 1 into v_k,
        # wait + sync.
        _async_load_v(max_m1 * fx.Index(BLOCK_N), 1)
        v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_V)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 9: scores for the last tile + finish previous softmax.
        # _mma0 -> v_s_1 (last tile); fold rescale_e7 into l_row, finish v_p_0's
        # second-half exp2, add its sum, cast + anchor.
        v_s_1 = _mma0(v_k)
        l_row = _fmul(l_row, rescale_e7)
        v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
        tile_sum_e9 = _attn_sum(v_p_0)
        l_row = _fadd(l_row, tile_sum_e9)
        v_p_0 = _cast_p(v_p_0)
        v_p_0 = _anchor_v_p(v_p_0)
        _sched_barrier_exp_pairs(6, 3, 9)
        _sched_barrier_pairs(10, 5, 9)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 10: memory stage + causal mask for the last tile.
        # Read the last V packs (buffer 0), mask v_s_1 for causal; all DMAs are
        # now drained (vmcnt 0). Wait + sync.
        v_packs_e11 = _read_v_packs_for_buf(0, urv_base_per_lane)
        if const_expr(CAUSAL):
            v_s_1 = _causal_mask_prologue_if_needed(
                v_s_1,
                max_m1,
                max_num_tiles * fx.Index(BLOCK_N),
            )
        else:
            v_s_1 = _v_s_vec_to_lists(v_s_1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 11: full P*V + rescale + complete the last softmax.
        # Full _mma1 for v_p_0, new combined row max, rescale_e11, update m_row,
        # sub_row + first-half exp2 of v_s_1. Then (since there is no further
        # main-loop pass) immediately complete v_p_1's second-half exp2, fold
        # rescale_e11 into l_row, add the last tile's sum, cast + anchor the
        # probabilities, and scale the output v_o by rescale_e11.
        v_o = _mma1(v_p_0, v_packs_e11, v_o)
        m_tile_max_e11 = _attn_row_max(v_s_1)
        row_max_e11 = _fmax(m_row, m_tile_max_e11)
        rescale_e11 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e11)))
        m_row = row_max_e11
        v_s_1 = _attn_sub_row(v_s_1, row_max_e11)
        v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
        _sched_barrier_pairs(9, 6, 10)
        _sched_barrier_exp_pairs(7, 3, 10)
        rocdl.sched_barrier(0)
        v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
        l_row = _fmul(l_row, rescale_e11)
        tile_sum_e11 = _attn_sum(v_p_1)
        l_row = _fadd(l_row, tile_sum_e11)
        v_p_1 = _cast_p(v_p_1)
        v_p_1 = _anchor_v_p(v_p_1)
        rocdl.sched_barrier(0)
        _scale_o(v_o, rescale_e11)
        v_o = _anchor_v_o(v_o)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 12: read the final V packs for the closing P*V.
        # Only the LDS read needs waiting on (lgkmcnt 0); then sync.
        v_packs_e13 = _read_v_packs_for_buf(1, urv_base_per_lane)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # Epilogue Cluster 13: final P*V accumulation.
        # The last full _mma1 (v_p_1 against the final V packs) completes the
        # output accumulator v_o. After this, v_o holds the unnormalized
        # attention output (sum of P*V over all KV tiles).
        v_o = _mma1(v_p_1, v_packs_e13, v_o)

        # Normalize O: divide the accumulated output by the softmax denominator.
        # inv_l = 1 / l_row (reciprocal), guarded so that a zero denominator
        # (e.g. a fully-masked row) yields 0 instead of inf/nan. _scale_o then
        # multiplies every output accumulator by inv_l to finish softmax(QK^T)*V.
        inv_l_rcp = rocdl.rcp(T.f32, _raw(l_row))
        inv_l = ArithValue(fx.Float32(l_row) > c_zero_f).select(inv_l_rcp, c_zero_f)
        _scale_o(v_o, inv_l)

        # Closing barrier -- CLOSES the phase shift. _stagger_extra_barrier_if_zero()
        # makes ONLY group A run one extra s_barrier here -- the exact complement
        # of the prologue's group-B extra barrier. It lets group A (which has run
        # one cluster AHEAD the whole time) wait for group B, so both realign
        # before the store. Disabled -> one plain barrier.
        if const_expr(DUALWAVE_SWP_ENABLE_STAGGER):
            _stagger_extra_barrier_if_zero()  # group A: +1 s_barrier -> close the shift
        else:
            rocdl.s_barrier()

        # Store O back to global memory (bounds-guarded for the ragged last
        # Q block). Only rows within seq_len are written. The output is laid
        # out as D_CHUNKS column banks; for each bank, 4 store groups each pack
        # 4 f32 accumulators into 2 16-bit dwords (bf16: v_cvt_pk_bf16_f32;
        # fp16: f32->f16 trunc), forming one 8-byte (dwordx2) write. d_row_rel/d_col map the lane's
        # MFMA output lane to its (row, head_dim column) destination, which
        # _global_idx_q converts to a linear element index, then to a byte
        # offset for the raw buffer store into O.
        q_in_bounds = q_row < seq_len_v
        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                for store_group in range_constexpr(4):
                    r_base = store_group * 4
                    # Pack 4 f32 outputs -> 2 packed-16bit dwords (lo, hi).
                    if const_expr(dtype_str == "bf16"):
                        lo = rocdl.cvt_pk_bf16_f32(
                            Vec(v_o[dc])[r_base],
                            Vec(v_o[dc])[r_base + 1],
                        )
                        hi = rocdl.cvt_pk_bf16_f32(
                            Vec(v_o[dc])[r_base + 2],
                            Vec(v_o[dc])[r_base + 3],
                        )
                        o_pack = Vec.from_elements([lo, hi], fx.Int32).ir_value()
                    else:
                        # fp16: trunc 4 f32 -> 4 f16 (RNE), view as 2 dwords.
                        o_f16 = []
                        for i in range_constexpr(4):
                            o_f16.append(fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype))
                        o_pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32).ir_value()
                    # Map this lane's MFMA output to (row, head_dim col).
                    d_row_rel = lane_div_32 * 4 + store_group * 8
                    d_col = fx.Index(dc * D_CHUNK) + d_row_rel
                    o_global = _global_idx_q(q_row, d_col)
                    o_byte_offset = fx.Int32(o_global * fx.Index(BF16_BYTES))
                    rocdl.raw_buffer_store(
                        o_pack,
                        o_rsrc,
                        _llvm_value(o_byte_offset),
                        _llvm_value(fx.Int32(0)),
                        _llvm_value(fx.Int32(0)),
                    )

    @flyc.jit
    def launch_flash_attn_dualwave_swp(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + BLOCK_M - 1) // BLOCK_M

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_dualwave_swp_gfx950_kernel(
            Q,
            K,
            V,
            O,
            DebugCounts,
            seq_len,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(NUM_HEADS_Q, num_q_blocks, bs_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _dualwave_swp_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,
        head_dim_runtime=None,
        debug_counts=None,
        *,
        stream=None,
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        if debug_counts is None:
            debug_counts = O
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            if stream is None:
                return launch_flash_attn_dualwave_swp(
                    Q, K, V, O, debug_counts, batch_size, seq_len, stride_q_n, stride_kv_n, head_dim_runtime
                )
            return launch_flash_attn_dualwave_swp(
                Q, K, V, O, debug_counts, batch_size, seq_len, stride_q_n, stride_kv_n, head_dim_runtime, stream=stream
            )

    def _compile(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,
        head_dim_runtime=None,
        debug_counts=None,
        *,
        stream=None,
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        if debug_counts is None:
            debug_counts = O
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return flyc.compile(
                launch_flash_attn_dualwave_swp,
                Q,
                K,
                V,
                O,
                debug_counts,
                batch_size,
                seq_len,
                stride_q_n,
                stride_kv_n,
                head_dim_runtime,
                fx.Stream(stream),
            )

    _launch.compile = _compile

    return _launch
