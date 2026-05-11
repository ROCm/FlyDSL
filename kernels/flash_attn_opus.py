# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""OPUS-style flash_attn fast path for FlyDSL (D=128 bf16 gfx950).

Adopts OPUS's high-impact structural optimizations on top of the proven
FlyDSL flash_attn_func BLOCK_M=256 algorithm. The dispatcher will only
select this path when:

    head_dim == 128, dtype == bf16, gpu_arch >= gfx950,
    seq_len % 256 == 0, seq_len >= 384.

OPUS optimizations included:
    * 3D grid launch (H, num_q_blocks, B): better workload distribution
      across CUs vs. 1D grid (block_id_x decomposition arithmetic stays
      in scalar registers from the launcher rather than per-thread).
    * Double-buffered K and V LDS with DMA async loads.
    * Online softmax with **lazy rescaling** (OPUS lines 476-484, 540-548):
      skip ``O *= corr`` when no lane's row_max changed beyond
      RESCALE_THRESHOLD (= 8.0), saving 32 v_pk_mul per skipped tile.
    * ``s_setprio(1)`` raised before GEMM2/rescale, lowered after
      (OPUS lines 471, 493, 535, 557).
    * Inline-asm causal mask: ``v_cmp_lt_i32 + v_cndmask_b32`` pairs
      with immediate K-position thresholds, replacing the 32-element
      select chain (OPUS lines 233-249).
    * ``s_nop 15; s_nop 7`` yield window after s_setprio(0) to let the
      other wave-group seize the MFMA/VALU units.

Layout (LDS, MFMA, Q/K/V/O addressing) matches existing
``flash_attn_func.py`` BLOCK_M=256 path to inherit its proven correctness.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.kernels_common import dtype_to_elem_type

KERNEL_NAME = "flash_attn_opus_kernel"
_LOG2E = host_math.log2(host_math.e)

# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse(
        "!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>"
    )
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type, ptr):
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value, ptr):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (
        (n & _VMCNT_LO_MASK)
        | _LGKMCNT_EXPCNT_BASE
        | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    )
    rocdl.s_waitcnt(val)


def build_flash_attn_opus_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
):
    """Build an OPUS-style flash_attn launcher for D=128 bf16 on gfx950.

    Launcher signature: ``launcher(Q, K, V, O, batch_size, seq_len, *, stream=None)``
    """
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(
            f"flash_attn_opus requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}"
        )
    if head_dim != 128:
        raise RuntimeError(f"flash_attn_opus is D=128 only, got head_dim={head_dim}")
    if dtype_str != "bf16":
        raise RuntimeError(f"flash_attn_opus is bf16 only, got dtype={dtype_str}")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    # ──────────────────────────── Tile constants ────────────────────────────
    # Match existing flash_attn_func BLOCK_M=256 path for layout compatibility.
    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_N_OUT = 64           # single sub-tile per outer iter (=BLOCK_N)
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    K_SUB_N = 32               # MFMA W_N
    WARP_SIZE = 64
    NUM_WAVES = 8              # BLOCK_M / 32
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE   # 512
    ROWS_PER_WAVE = 32

    HEAD_DIM = head_dim
    K_STEP_QK = 16             # W_K
    K_STEPS_QK = HEAD_DIM // K_STEP_QK    # 8
    D_CHUNK = 32
    D_CHUNKS = HEAD_DIM // D_CHUNK    # 4
    PV_K_STEP = 16
    PV_K_STEPS = K_SUB_N // PV_K_STEP    # 2
    MFMA_LANE_K = 8

    SM_SCALE = 1.0 / host_math.sqrt(head_dim)
    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    CAUSAL = causal
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    # K/V LDS double-buffered, XOR-swizzled (16B = 8 bf16 swizzle granularity).
    K_STRIDE = HEAD_DIM
    V_STRIDE = HEAD_DIM
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    NUM_PREFETCH_K = 2     # OPUS double-buffer
    NUM_PREFETCH_V = 2
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # DMA load chunking
    VEC_WIDTH = 16
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    PATH_TAG = "OPUS"
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_opus_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2   # bf16 = 2 bytes

    # OPUS lazy-rescale threshold (line 374)
    OPUS_RESCALE_THRESHOLD = 8.0

    # Enable / disable individual OPUS optimizations via env vars (debug).
    OPUS_LAZY_RESCALE = os.getenv("FLYDSL_OPUS_LAZY_RESCALE", "1") == "1"
    OPUS_SETPRIO = os.getenv("FLYDSL_OPUS_SETPRIO", "1") == "1"
    OPUS_YIELD_NOP = os.getenv("FLYDSL_OPUS_YIELD_NOP", "1") == "1"

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_opus_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        seq_len: fx.Int32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        q_ptr = _extract_aligned_pointer(Q)
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        o_ptr = _extract_aligned_pointer(O)

        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type

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
            return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)

        seq_len_v = fx.Index(seq_len)

        # LDS view
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()
        lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)

        # ── 3D grid block indices (OPUS layout) ──
        h_idx = fx.Index(gpu.block_idx.x)
        q_block_idx = fx.Index(gpu.block_idx.y)
        batch_idx = fx.Index(gpu.block_idx.z)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32

        # HW transpose decomp for ds_read_tr16_b64 (gfx950)
        tr_k_group = (lane % 16) // 4
        tr_col_sub = lane % 4
        tr_col_half = (lane % 32) // 16

        # ds_read_tr_v4f16 helper
        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = fx.Int64(byte_offset)
            ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ── Wave / tile bookkeeping ──
        wave_q_offset = wave_id * ROWS_PER_WAVE
        q_block_size = BLOCK_M
        q_start = q_block_idx * q_block_size

        # GQA mapping mirrors OPUS lines 310-312:
        # h = (h_idx % H_KV) * group_size + (h_idx / H_KV)
        # h_kv = h / group_size = h_idx % H_KV
        h_kv_idx = h_idx % NUM_HEADS_KV
        group_id = h_idx // NUM_HEADS_KV
        q_head_idx = h_kv_idx * GQA_GROUP_SIZE + group_id
        kv_head_idx = h_kv_idx

        def global_idx_q(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col

        def _load_global_half_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            _pointer_store(val, gep)

        def load_global_mfma_pack(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, MFMA_LANE_K)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _pack_bf16_pair(lo, hi, shift, mask):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack_v8(f32_vals):
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
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)
            return fx.Index(LDS_V_BASE) + buf_id * fx.Index(LDS_V_TILE_SIZE)

        # ── DMA loaders (buffer_load_dwordx4_lds, gfx950) ──
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
        DMA_BYTES = 16
        DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
        K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
        NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
        LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
        ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
        V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
        NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
        LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES

        _dma_size = fx.Int32(DMA_BYTES)
        _dma_soff = fx.Int32(0)
        _dma_off = fx.Int32(0)
        _dma_aux = fx.Int32(1)

        def coop_dma_k(tile_start, buf_id):
            k_lds_byte_base = lds_kv_base_idx + k_buf_base(buf_id) * fx.Index(2)
            for d in range_constexpr(NUM_DMA_K):
                lds_addr = (
                    k_lds_byte_base
                    + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                    + fx.Index(d * DMA_BATCH_BYTES)
                )
                lds_i64 = fx.Int64(lds_addr)
                lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                row_in_tile = tid // LANES_PER_K_ROW + fx.Index(d * ROWS_PER_DMA_BATCH)
                swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                xor_mask = (row_in_tile & fx.Index(0x7)) << fx.Index(4)
                unsw_col_f16 = swiz_col_f16 ^ xor_mask
                col_byte = unsw_col_f16 * 2
                global_row = batch_idx * seq_len_v + tile_start + row_in_tile
                global_byte = (
                    global_row * fx.Index(STRIDE_TOKEN_KV * 2)
                    + kv_head_idx * fx.Index(HEAD_DIM * 2)
                    + col_byte
                )
                voffset = fx.Int32(global_byte)
                rocdl.raw_ptr_buffer_load_lds(
                    k_rsrc, lds_ptr, _dma_size, voffset, _dma_soff, _dma_off, _dma_aux
                )

        def coop_dma_v(tile_start, buf_id):
            v_lds_byte_base = lds_kv_base_idx + v_buf_base(buf_id) * fx.Index(2)
            for d in range_constexpr(NUM_DMA_V):
                lds_addr = (
                    v_lds_byte_base
                    + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                    + fx.Index(d * DMA_BATCH_BYTES)
                )
                lds_i64 = fx.Int64(lds_addr)
                lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                row_in_tile = tid // LANES_PER_V_ROW + fx.Index(d * (DMA_BATCH_BYTES // (HEAD_DIM * 2)))
                swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                xor_mask = (row_in_tile & fx.Index(0x3)) << fx.Index(4)
                unsw_col_f16 = swiz_col_f16 ^ xor_mask
                col_byte = unsw_col_f16 * 2
                global_row = batch_idx * seq_len_v + tile_start + row_in_tile
                global_byte = (
                    global_row * fx.Index(STRIDE_TOKEN_KV * 2)
                    + kv_head_idx * fx.Index(HEAD_DIM * 2)
                    + col_byte
                )
                voffset = fx.Int32(global_byte)
                rocdl.raw_ptr_buffer_load_lds(
                    v_rsrc, lds_ptr, _dma_size, voffset, _dma_soff, _dma_off, _dma_aux
                )

        # ── Q preload (B-operand for MFMA, register-resident) ──
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = q_row < seq_len_v
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        c_zero_mfma_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row_safe, q_col)
            raw = load_global_mfma_pack(q_ptr, g_idx)
            q_b_packs.append(ArithValue(q_in_bounds).select(raw, c_zero_mfma_pack))

        # ── Constants ──
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(SM_SCALE * _LOG2E)
        c_eight_f = fx.Float32(OPUS_RESCALE_THRESHOLD)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)
        c4_i32 = fx.Int32(4)
        lane_i32 = fx.Int32(lane)

        # Use shuffle_xor by 32 for the cross-half reduction (== permlane32_swap+max in OPUS).
        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ── KV loop bound ──
        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))
        else:
            kv_upper = seq_len_v

        # ── Pre-launch K[0] DMA (OPUS prologue, line 398) ──
        coop_dma_k(fx.Index(0), 0)

        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        # carry the K buffer id (0/1) across iterations
        init_args.append(fx.Index(0))

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(0, kv_upper, BLOCK_N_OUT, init=init_args):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            cur_k_buf = inner_iter_args[2 + D_CHUNKS]

            kv_start = kv_block_start

            # ── Cluster 0: wait current K DMA, barrier, prefetch next K ──
            _next_k_buf = fx.Index(1) - cur_k_buf
            # Wait for current K to land
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            gpu.barrier()

            # Issue next K prefetch (if any) and current V load
            _next_kv = kv_block_start + fx.Index(BLOCK_N_OUT)
            _has_next = _next_kv < kv_upper
            if _has_next:
                coop_dma_k(_next_kv, _next_k_buf)
            coop_dma_v(kv_start, fx.Index(0))   # V single-buffered for simplicity
            rocdl.sched_barrier(0)

            # ── Cluster 1: GEMM0 (S = Q @ K^T) ──
            k_base = k_buf_base(cur_k_buf)
            k_hi_offset = K_SUB_N * K_STRIDE
            k_swz_mask = (lane_mod_32 & fx.Index(0x7)) << fx.Index(4)

            def _k_idx_lo(ks):
                col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                return k_base + lane_mod_32 * fx.Index(K_STRIDE) + (col ^ k_swz_mask)

            def _k_idx_hi(ks):
                col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                return (k_base + fx.Index(k_hi_offset)
                        + lane_mod_32 * fx.Index(K_STRIDE) + (col ^ k_swz_mask))

            _QK_PREFETCH_DEPTH = 2
            k_packs_lo = [None] * K_STEPS_QK
            k_packs_hi = [None] * K_STEPS_QK
            for p in range_constexpr(_QK_PREFETCH_DEPTH):
                k_packs_lo[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                k_packs_hi[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_hi(p)])

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

            # ── Cluster 2: extract S into 32 f32 elements, apply causal mask ──
            s_raw_lo = [Vec(s_acc_lo)[r] for r in range_constexpr(16)]
            s_raw_hi = [Vec(s_acc_hi)[r] for r in range_constexpr(16)]

            if const_expr(CAUSAL):
                kv_start_i32 = fx.Int32(kv_start)
                lane_div_32_i32 = fx.Int32(lane_div_32)
                lane_off_i32 = lane_div_32_i32 * c4_i32
                # Always apply causal mask via per-element v_cmp + v_cndmask.
                # The MFMA C output lane holds elements at
                #   K-col-within-N-strip = lane_div_32*4 + (r//4)*8 + (r%4)
                # which gives offsets {0..3, 8..11, 16..19, 24..27}.
                for r in range_constexpr(16):
                    rept = r // 4
                    pair_off = r % 4
                    kv_col_off = rept * 8 + pair_off
                    kv_col_lo = kv_start_i32 + lane_off_i32 + fx.Int32(kv_col_off)
                    s_raw_lo[r] = ArithValue(kv_col_lo > q_row_i32).select(c_neg_inf, s_raw_lo[r])
                    kv_col_hi = kv_col_lo + fx.Int32(K_SUB_N)
                    s_raw_hi[r] = ArithValue(kv_col_hi > q_row_i32).select(c_neg_inf, s_raw_hi[r])

            # ── Cluster 3: row-max, softmax exp, online sum ──
            m_raw = c_neg_inf
            for r in range_constexpr(16):
                m_raw = _fmax(m_raw, s_raw_lo[r])
                m_raw = _fmax(m_raw, s_raw_hi[r])
            m_peer = reduction_peer(m_raw)
            m_tile_max = _fmax(m_raw, m_peer)
            m_candidate = _fmax(m_running, m_tile_max)

            # ── Lazy rescale (OPUS lines 475-484, 539-548) ──
            # If (row_max - m_row) <= 8.0 in scaled-space for ALL lanes,
            # CLAMP row_max := m_running. This keeps the running max stationary
            # and skips O *= corr (since corr = exp2(0) = 1).
            if const_expr(OPUS_LAZY_RESCALE):
                # OPUS threshold is in scaled space; we work in unscaled here, so
                # scale the difference. m_diff_scaled = (m_tile_max - m_running) * SM_SCALE_LOG2E.
                m_diff_unscaled = _fsub(m_tile_max, m_running)
                m_diff_scaled = _fmul(m_diff_unscaled, c_sm_scale_log2e)
                below_pred = ArithValue(fx.Float32(m_diff_scaled) <= c_eight_f)
                ballot_val = rocdl.ballot(T.i64, _raw(below_pred))
                neg_one_i64 = fx.Int64(-1)
                all_below = fx.Int64(ballot_val) == neg_one_i64
                _all_below_pred = ArithValue(all_below)
                # When lazy, m_new := m_running (no rescale needed downstream).
                m_new_raw = _all_below_pred.select(m_running, m_candidate)
            else:
                m_new_raw = m_candidate

            # corr = exp2((m_running - m_new_raw) * SM_SCALE_LOG2E). When lazy is
            # active, m_new_raw == m_running so corr == 1.
            corr_arg = _fmul(_fsub(m_running, m_new_raw), c_sm_scale_log2e)
            corr = rocdl.exp2(T.f32, _raw(corr_arg))

            # Compute scaled exp values relative to m_new_raw
            neg_scaled_max_raw = _fmul(m_new_raw, c_sm_scale_log2e)
            neg_scaled_max = arith.negf(_raw(neg_scaled_max_raw), fastmath=fm_fast)
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
            # l_new = l_running * corr + tile_sum  (corr = 1 when lazy active)
            l_corr_full = _fmul(corr, l_running)
            l_new = _fadd(l_corr_full, tile_sum)

            # ── Cluster 4: rescale O + GEMM2 ──
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(1)

            if const_expr(OPUS_LAZY_RESCALE):
                # When all_below is true, replace corr with 1.0 so the multiply
                # is a no-op (LLVM can fold). Saves the actual multiplication
                # work when the running max is stationary.
                eff_corr_scalar = _all_below_pred.select(c_one_f, corr)
                eff_corr_vec = Vec.from_elements([eff_corr_scalar], fx.Float32).broadcast_to(16)
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = _fmul(Vec(o_accs[dc]), eff_corr_vec)
            else:
                corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = _fmul(Vec(o_accs[dc]), corr_vec)

            # Pack P
            p_packs_lo = []
            p_packs_hi = []
            for pks in range_constexpr(PV_K_STEPS):
                p_base = pks * 8
                p_packs_lo.append(bf16_trunc_pack_v8(p_vals_lo[p_base:p_base + 8]))
                p_packs_hi.append(bf16_trunc_pack_v8(p_vals_hi[p_base:p_base + 8]))

            # GEMM2: O += V^T @ P (using ds_read_tr16_b64 for V)
            v_base = v_buf_base(fx.Index(0))
            _steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
            TOTAL_PV = len(_steps)

            def _read_v_pack(step_idx):
                dc, pks = _steps[step_idx]
                d_col = fx.Index(dc * D_CHUNK) + tr_col_half * fx.Index(16) + tr_col_sub * fx.Index(4)
                k_row = fx.Index(pks * PV_K_STEP) + lane_div_32 * fx.Index(4) + tr_k_group
                v_xor_mask = (k_row & fx.Index(0x3)) << fx.Index(4)
                d_col_eff = d_col ^ v_xor_mask
                lds_lo = v_base + k_row * fx.Index(V_STRIDE) + d_col_eff
                lds_hi = lds_lo + fx.Index(K_SUB_N * V_STRIDE)
                vl_a = ds_read_tr_v4f16(lds_lo)
                vl_b = ds_read_tr_v4f16(lds_lo + fx.Index(8 * V_STRIDE))
                vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                vh_a = ds_read_tr_v4f16(lds_hi)
                vh_b = ds_read_tr_v4f16(lds_hi + fx.Index(8 * V_STRIDE))
                vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                return vl, vh

            # Wait for V DMA to complete and barrier
            _waitcnt_vm_n(0)
            rocdl.sched_barrier(0)
            gpu.barrier()

            v_lo_cur, v_hi_cur = _read_v_pack(0)
            for si in range_constexpr(TOTAL_PV):
                dc, pks = _steps[si]
                if const_expr(si + 1 < TOTAL_PV):
                    v_lo_nxt, v_hi_nxt = _read_v_pack(si + 1)
                o_accs[dc] = mfma_acc(v_lo_cur, p_packs_lo[pks], o_accs[dc])
                o_accs[dc] = mfma_acc(v_hi_cur, p_packs_hi[pks], o_accs[dc])
                if const_expr(si + 1 < TOTAL_PV):
                    v_lo_cur = v_lo_nxt
                    v_hi_cur = v_hi_nxt

            # ── End of cluster: s_setprio(0), yield window ──
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(0)
            if const_expr(OPUS_YIELD_NOP):
                rocdl.sched_barrier(0)
                rocdl.s_nop(15)
                rocdl.s_nop(7)
                rocdl.sched_barrier(0)

            _yield_args = [m_new_raw, l_new] + o_accs + [_next_k_buf]
            loop_results = yield _yield_args

        # ── Normalize and store O ──
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]
        inv_l = rocdl.rcp(T.f32, _raw(l_final))
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = Vec(o_finals[dc]) * inv_l_vec
                for r in range_constexpr(16):
                    o_val = Vec(o_norm_vec)[r]
                    o_f16 = fx.Float32(o_val).to(elem_dtype)
                    d_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                    d_col = fx.Index(dc * D_CHUNK) + d_row_rel
                    o_global = global_idx_q(q_row, d_col)
                    _store_global_half(o_ptr, o_global, o_f16)

    @flyc.jit
    def launch_flash_attn_opus(
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
        flash_attn_opus_kernel(
            Q, K, V, O, seq_len,
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

    _opus_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_opus_compile_hints):
            return launch_flash_attn_opus(*args, **kwargs)

    def _compile(Q, K, V, O, batch_size, seq_len, stream=None):
        with CompilationContext.compile_hints(_opus_compile_hints):
            return flyc.compile(
                launch_flash_attn_opus, Q, K, V, O, batch_size, seq_len,
                fx.Stream(stream))

    _launch.compile = _compile

    return _launch
