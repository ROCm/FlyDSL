"""Flash Attention V4.3 kernel builder for FlyDSL.

V4.3 optimization over V4.2:
- Q loaded directly from global memory to MFMA registers (no Q in LDS).
  LDS = KV(8.5KB) + P(4KB) = 12.5KB (was 29KB in V4.2).
  This enables 4 workgroups/CU -> 4 waves/SIMD (was 2 waves/SIMD).
- Eliminates 2 barriers (Q store + Q preload sync).

All other optimizations from V4.2:
- BLOCK_N=32 (vs 16): halves KV iterations and barriers
- Q@K^T produces [16,32] via two MFMA 16x16x16 in N dimension
- P@V uses K=32 via two MFMA 16x16x16 in K dimension
- Softmax over 32 positions per row (two 16-wide groups)
- V stored transposed in LDS with bank-conflict-free padding (from V4.1)
- Causal early-exit

Tile config: BLOCK_M=64, BLOCK_N=32, 4 waves (256 threads), mfma_f32_16x16x16f16.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) -- 4 waves of 64 on AMD (wave64).

Requires: head_dim % 16 == 0, seq_len % 64 == 0, head_dim >= 64.
"""

import math

from flydsl.dialects.ext import flir, arith, gpu, scf, rocdl
from flydsl.dialects.ext import vector as vec_ext
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext.scf import yield_ as scf_yield
from _mlir.dialects import memref as _memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "flash_attention_v4_3_kernel"


def build_flash_attention_v4_3_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
):
    """Build a FlyDSL Flash Attention V4.3 module (LDS overlay).

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension per head (must be divisible by 16, >= 64).
        causal: Whether to apply causal mask.
        dtype_str: "f16" (bf16 not yet supported).
        sm_scale: Softmax scale (default: 1/sqrt(head_dim)).

    Returns:
        MlirModule compilable via ``flydsl.compile(module)``.
    """
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    BLOCK_M = 64
    BLOCK_N = 32   # *** doubled from V4.1 ***
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 256
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 16
    K_STEPS = head_dim // 16
    # Number of 16-wide MFMA columns in Q@K^T N-dimension
    N_MFMA = BLOCK_N // 16  # 2

    assert head_dim % 16 == 0, f"head_dim ({head_dim}) must be divisible by 16"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert dtype_str == "f16", "V4.3 currently only supports f16"
    assert BLOCK_N % 16 == 0, f"BLOCK_N ({BLOCK_N}) must be divisible by 16"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # ---- Bank-conflict-free LDS strides ----
    K_STRIDE = HEAD_DIM + 2   # 130 for HD=128
    VT_STRIDE = BLOCK_N + 2   # 34 for BLOCK_N=32

    # ---- Vectorized cooperative load constants ----
    VEC_WIDTH = 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    assert BLOCK_M % ROWS_PER_BATCH_LOAD == 0
    NUM_BATCHES_Q = BLOCK_M // ROWS_PER_BATCH_LOAD

    # For KV tile (32 rows with 256 threads)
    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # LDS sizes (element counts, f16 = 2 bytes each)
    # No Q in LDS — loaded directly from global memory to MFMA registers
    LDS_KV_SIZE = max(BLOCK_N * K_STRIDE, HEAD_DIM * VT_STRIDE)  # 4352 elements = 8704 bytes
    LDS_P_SIZE = BLOCK_M * BLOCK_N           # 2048 elements = 4096 bytes

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _FlashAttentionV4_3(flir.MlirModule):
        GPU_MODULE_NAME = f"flash_attn_v4_3_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type = T.f16()
            _state["elem_type"] = elem_type
            _state["lds_kv"] = allocator.allocate_array(elem_type, LDS_KV_SIZE)
            _state["lds_p"] = allocator.allocate_array(elem_type, LDS_P_SIZE)
            allocator.finalize()

        @flir.kernel
        def flash_attention_v4_3_kernel(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, _state["elem_type"]),
            K: lambda: T.memref(DYN, _state["elem_type"]),
            V: lambda: T.memref(DYN, _state["elem_type"]),
            O: lambda: T.memref(DYN, _state["elem_type"]),
            seq_len: lambda: T.index(),
        ):
            compute_type = T.f32()
            elem_type = _state["elem_type"]
            fm_fast = flir.arith.FastMathFlags.fast

            v4f16_type = ir.VectorType.get([4], elem_type)
            v4f32_type = ir.VectorType.get([4], compute_type)
            v8f16_type = ir.VectorType.get([VEC_WIDTH], elem_type)

            seq_len_v = arith.as_value(seq_len)

            # ---- LDS views (KV + P only, no Q in LDS) ----
            base_ptr = allocator.get_base()
            lds_kv = _state["lds_kv"](base_ptr).get()
            lds_p = _state["lds_p"](base_ptr).get()

            # ---- Thread / block indices ----
            block_id = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            # ---- Wave decomposition ----
            c_ws = flir.const_index(WARP_SIZE)
            wave_id = arith.as_value(flir.arith.DivUIOp(tid, c_ws).result)
            lane = arith.as_value(flir.arith.RemUIOp(tid, c_ws).result)

            # ---- MFMA lane decomposition ----
            c16 = flir.const_index(16)
            lane_div_16 = arith.as_value(flir.arith.DivUIOp(lane, c16).result)
            lane_mod_16 = arith.as_value(flir.arith.RemUIOp(lane, c16).result)

            # ---- Wave offsets ----
            wave_q_offset = (arith.ArithValue(wave_id) * ROWS_PER_WAVE).value
            wave_p_offset = (arith.ArithValue(wave_id) * ROWS_PER_WAVE * BLOCK_N).value

            # ---- Decompose block_id ----
            c_nh = flir.const_index(NUM_HEADS)
            head_idx = arith.as_value(flir.arith.RemUIOp(block_id, c_nh).result)
            temp = arith.as_value(flir.arith.DivUIOp(block_id, c_nh).result)
            c_bm = flir.const_index(BLOCK_M)
            num_q_tiles = arith.as_value(flir.arith.DivUIOp(seq_len_v, c_bm).result)
            q_tile_idx = arith.as_value(flir.arith.RemUIOp(temp, num_q_tiles).result)
            batch_idx = arith.as_value(flir.arith.DivUIOp(temp, num_q_tiles).result)
            q_start = (arith.ArithValue(q_tile_idx) * BLOCK_M).value

            # ---- Load thread decomposition ----
            c_tpr = flir.const_index(THREADS_PER_ROW_LOAD)
            load_row_in_batch = arith.as_value(
                flir.arith.DivUIOp(tid, c_tpr).result
            )
            load_lane_in_row = arith.as_value(
                flir.arith.RemUIOp(tid, c_tpr).result
            )
            load_col_base = (
                arith.ArithValue(load_lane_in_row) * VEC_WIDTH
            ).value

            # ---- Helper: global flat index ----
            def global_idx(token_idx, col):
                token = (
                    arith.ArithValue(batch_idx) * arith.ArithValue(seq_len_v)
                    + arith.ArithValue(token_idx)
                )
                return (
                    token * STRIDE_TOKEN
                    + arith.ArithValue(head_idx) * HEAD_DIM
                    + arith.ArithValue(col)
                ).value

            # ---- Cooperative K load (row-major, padded stride) ----
            def coop_load_k(tile_start):
                for batch in range_constexpr(NUM_BATCHES_KV):
                    row_offset = batch * ROWS_PER_BATCH_LOAD
                    row_idx = (
                        arith.ArithValue(tile_start)
                        + arith.ArithValue(load_row_in_batch)
                        + row_offset
                    ).value
                    g_idx = global_idx(row_idx, load_col_base)
                    vec = arith.as_value(
                        vec_ext.load_op(v8f16_type, K, [g_idx])
                    )
                    lds_row = (
                        arith.ArithValue(load_row_in_batch) + row_offset
                    ).value
                    lds_idx = (
                        arith.ArithValue(lds_row) * K_STRIDE
                        + arith.ArithValue(load_col_base)
                    ).value
                    vec_ext.store(vec, lds_kv, [lds_idx])

            # ---- Cooperative V load (transposed, padded stride) ----
            def coop_load_v_transposed(tile_start):
                for batch in range_constexpr(NUM_BATCHES_KV):
                    row_offset = batch * ROWS_PER_BATCH_LOAD
                    row_idx = (
                        arith.ArithValue(tile_start)
                        + arith.ArithValue(load_row_in_batch)
                        + row_offset
                    ).value
                    g_idx = global_idx(row_idx, load_col_base)
                    vec = arith.as_value(
                        vec_ext.load_op(v8f16_type, V, [g_idx])
                    )
                    load_row = (
                        arith.ArithValue(load_row_in_batch) + row_offset
                    ).value
                    # Scatter-store transposed: V[row, col+e] -> lds_kv[(col+e)*VT_STRIDE + row]
                    for e in range_constexpr(VEC_WIDTH):
                        elem = arith.as_value(
                            vec_ext.extract(vec, static_position=[e], dynamic_position=[])
                        )
                        col_e = (arith.ArithValue(load_col_base) + e).value
                        lds_idx = (
                            arith.ArithValue(col_e) * VT_STRIDE
                            + arith.ArithValue(load_row)
                        ).value
                        _memref.StoreOp(elem, lds_kv, [lds_idx])

            # ---- Load Q directly from global memory to MFMA registers ----
            # Each MFMA lane (b=lane_div_16, n=lane_mod_16) loads v4f16 from
            # Q[q_start + wave_offset + n, ks*16 + b*4 : ks*16 + b*4 + 4].
            # No LDS needed for Q — eliminates overlay race condition.
            q_row = (
                arith.ArithValue(q_start)
                + arith.ArithValue(wave_q_offset)
                + arith.ArithValue(lane_mod_16)
            ).value
            q_a_packs = []
            for ks in range_constexpr(K_STEPS):
                q_col = flir.const_index(ks * 16 + 0)
                q_col = (arith.ArithValue(q_col) + arith.ArithValue(lane_div_16) * 4).value
                g_idx = global_idx(q_row, q_col)
                q_a_packs.append(arith.as_value(
                    vec_ext.load_op(v4f16_type, Q, [g_idx])
                ))

            # ---- Constants ----
            c_neg_inf = arith.constant(float("-inf"), type=compute_type)
            c_zero_f = arith.constant(0.0, type=compute_type)
            c_sm_scale = arith.constant(sm_scale, type=compute_type)
            c_log2e = arith.constant(1.4426950408889634, type=compute_type)
            c_zero_v4f32 = arith.as_value(
                arith.constant_vector(0.0, v4f32_type)
            )

            # ---- Init loop-carried state ----
            # m[4], l[4], o_accs[K_STEPS]
            init_args = []
            for _ in range_constexpr(4):
                init_args.append(arith.as_value(c_neg_inf))
            for _ in range_constexpr(4):
                init_args.append(arith.as_value(c_zero_f))
            for _ in range_constexpr(K_STEPS):
                init_args.append(c_zero_v4f32)

            # ---- KV loop upper bound ----
            # Causal early-exit: last Q row = q_start + BLOCK_M - 1,
            # so only need KV positions 0 .. q_start + BLOCK_M - 1.
            # q_start + BLOCK_M is always a multiple of BLOCK_N (64 % 32 == 0).
            if CAUSAL:
                kv_upper = (arith.ArithValue(q_start) + BLOCK_M).value
            else:
                kv_upper = seq_len_v

            # ---- KV loop (step BLOCK_N=32) ----
            with scf.for_(0, kv_upper, BLOCK_N, iter_args=init_args) as loop:
                kv_start = arith.as_value(loop.induction_variable)
                m_old = [arith.as_value(loop.inner_iter_args[i]) for i in range(4)]
                l_old = [arith.as_value(loop.inner_iter_args[4 + i]) for i in range(4)]
                o_accs = [arith.as_value(loop.inner_iter_args[8 + ds]) for ds in range(K_STEPS)]

                # ==== Cooperative K load -> LDS_KV (32 rows, padded stride) ====
                coop_load_k(kv_start)
                gpu.barrier()

                # ==== Q @ K^T via MFMA -> S[16, 32] ====
                # Two MFMA outputs: s_acc[0] for KV cols 0..15, s_acc[1] for KV cols 16..31
                s_accs = [c_zero_v4f32, c_zero_v4f32]
                for ks in range_constexpr(K_STEPS):
                    a_pack = q_a_packs[ks]
                    for nm in range_constexpr(N_MFMA):
                        # B operand (K^T): K row = nm*16 + lane_mod_16
                        k_row = nm * 16
                        k_lds_idx = (
                            (arith.ArithValue(lane_mod_16) + k_row) * K_STRIDE
                            + ks * 16
                            + arith.ArithValue(lane_div_16) * 4
                        ).value
                        b_pack = arith.as_value(
                            vec_ext.load_op(v4f16_type, lds_kv, [k_lds_idx])
                        )
                        s_accs[nm] = arith.as_value(
                            rocdl.mfma_f32_16x16x16f16(
                                v4f32_type, [a_pack, b_pack, s_accs[nm], 0, 0, 0]
                            )
                        )

                # ==== Online softmax over 32 positions ====
                # For each row ii (0..3): have values at lane_mod_16 in s_accs[0] and s_accs[1]
                # Need max and sum over all 32 positions
                s_vals_lo = []  # from s_accs[0], KV cols 0..15
                s_vals_hi = []  # from s_accs[1], KV cols 16..31
                for ii in range_constexpr(4):
                    s_lo = arith.as_value(
                        vec_ext.extract(s_accs[0], static_position=[ii], dynamic_position=[])
                    )
                    s_lo = arith.as_value(
                        flir.arith.MulFOp(s_lo, arith.as_value(c_sm_scale), fastmath=fm_fast).result
                    )
                    s_hi = arith.as_value(
                        vec_ext.extract(s_accs[1], static_position=[ii], dynamic_position=[])
                    )
                    s_hi = arith.as_value(
                        flir.arith.MulFOp(s_hi, arith.as_value(c_sm_scale), fastmath=fm_fast).result
                    )

                    if CAUSAL:
                        q_row = (
                            arith.ArithValue(q_start)
                            + arith.ArithValue(wave_q_offset)
                            + arith.ArithValue(lane_div_16) * 4
                            + ii
                        ).value
                        # Low half: KV col = kv_start + lane_mod_16
                        kv_col_lo = (arith.ArithValue(kv_start) + arith.ArithValue(lane_mod_16)).value
                        # High half: KV col = kv_start + 16 + lane_mod_16
                        kv_col_hi = (arith.ArithValue(kv_start) + 16 + arith.ArithValue(lane_mod_16)).value

                        q_row_i64 = arith.as_value(flir.arith.IndexCastOp(T.i64(), q_row).result)
                        kv_lo_i64 = arith.as_value(flir.arith.IndexCastOp(T.i64(), kv_col_lo).result)
                        kv_hi_i64 = arith.as_value(flir.arith.IndexCastOp(T.i64(), kv_col_hi).result)

                        is_masked_lo = arith.as_value(
                            flir.arith.CmpIOp(
                                flir.arith.CmpIPredicate.ugt, kv_lo_i64, q_row_i64,
                            ).result
                        )
                        is_masked_hi = arith.as_value(
                            flir.arith.CmpIOp(
                                flir.arith.CmpIPredicate.ugt, kv_hi_i64, q_row_i64,
                            ).result
                        )
                        s_lo = arith.as_value(
                            flir.arith.SelectOp(is_masked_lo, arith.as_value(c_neg_inf), s_lo).result
                        )
                        s_hi = arith.as_value(
                            flir.arith.SelectOp(is_masked_hi, arith.as_value(c_neg_inf), s_hi).result
                        )

                    s_vals_lo.append(s_lo)
                    s_vals_hi.append(s_hi)

                width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
                m_new = [None] * 4
                corr = [None] * 4
                p_vals_lo = [None] * 4
                p_vals_hi = [None] * 4
                l_new = [None] * 4

                for ii in range_constexpr(4):
                    # Max over 32 positions: max of lo-half and hi-half
                    row_max_lo = s_vals_lo[ii]
                    row_max_hi = s_vals_hi[ii]

                    # Reduce lo-half within 16 lanes
                    for sh in [8, 4, 2, 1]:
                        sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                        peer = arith.as_value(
                            gpu.ShuffleOp(row_max_lo, sh_i32, width_i32, mode="xor").shuffleResult
                        )
                        row_max_lo = arith.as_value(
                            flir.arith.MaximumFOp(row_max_lo, peer).result
                        )

                    # Reduce hi-half within 16 lanes
                    for sh in [8, 4, 2, 1]:
                        sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                        peer = arith.as_value(
                            gpu.ShuffleOp(row_max_hi, sh_i32, width_i32, mode="xor").shuffleResult
                        )
                        row_max_hi = arith.as_value(
                            flir.arith.MaximumFOp(row_max_hi, peer).result
                        )

                    # Combine lo and hi maxes
                    row_max = arith.as_value(
                        flir.arith.MaximumFOp(row_max_lo, row_max_hi).result
                    )

                    m_new[ii] = arith.as_value(
                        flir.arith.MaximumFOp(m_old[ii], row_max).result
                    )

                    diff_m = arith.as_value(
                        flir.arith.SubFOp(m_old[ii], m_new[ii], fastmath=fm_fast).result
                    )
                    diff_m_s = arith.as_value(
                        flir.arith.MulFOp(diff_m, arith.as_value(c_log2e), fastmath=fm_fast).result
                    )
                    corr[ii] = arith.as_value(flir.math.exp2(diff_m_s, fastmath=fm_fast))

                    # exp2 for both halves
                    diff_lo = arith.as_value(
                        flir.arith.SubFOp(s_vals_lo[ii], m_new[ii], fastmath=fm_fast).result
                    )
                    diff_lo_s = arith.as_value(
                        flir.arith.MulFOp(diff_lo, arith.as_value(c_log2e), fastmath=fm_fast).result
                    )
                    p_vals_lo[ii] = arith.as_value(flir.math.exp2(diff_lo_s, fastmath=fm_fast))

                    diff_hi = arith.as_value(
                        flir.arith.SubFOp(s_vals_hi[ii], m_new[ii], fastmath=fm_fast).result
                    )
                    diff_hi_s = arith.as_value(
                        flir.arith.MulFOp(diff_hi, arith.as_value(c_log2e), fastmath=fm_fast).result
                    )
                    p_vals_hi[ii] = arith.as_value(flir.math.exp2(diff_hi_s, fastmath=fm_fast))

                    # Sum over 32 positions
                    row_sum_lo = p_vals_lo[ii]
                    row_sum_hi = p_vals_hi[ii]

                    for sh in [8, 4, 2, 1]:
                        sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                        peer = arith.as_value(
                            gpu.ShuffleOp(row_sum_lo, sh_i32, width_i32, mode="xor").shuffleResult
                        )
                        row_sum_lo = arith.as_value(
                            flir.arith.AddFOp(row_sum_lo, peer, fastmath=fm_fast).result
                        )

                    for sh in [8, 4, 2, 1]:
                        sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                        peer = arith.as_value(
                            gpu.ShuffleOp(row_sum_hi, sh_i32, width_i32, mode="xor").shuffleResult
                        )
                        row_sum_hi = arith.as_value(
                            flir.arith.AddFOp(row_sum_hi, peer, fastmath=fm_fast).result
                        )

                    row_sum = arith.as_value(
                        flir.arith.AddFOp(row_sum_lo, row_sum_hi, fastmath=fm_fast).result
                    )

                    l_corr = arith.as_value(
                        flir.arith.MulFOp(corr[ii], l_old[ii], fastmath=fm_fast).result
                    )
                    l_new[ii] = arith.as_value(
                        flir.arith.AddFOp(l_corr, row_sum, fastmath=fm_fast).result
                    )

                # ==== Rescale O accumulators ====
                corr_vec = arith.as_value(
                    vec_ext.from_elements(v4f32_type, [corr[0], corr[1], corr[2], corr[3]])
                )
                for ds in range_constexpr(K_STEPS):
                    o_accs[ds] = arith.as_value(
                        flir.arith.MulFOp(o_accs[ds], corr_vec, fastmath=fm_fast).result
                    )

                # ==== P store to LDS_P ====
                # P is [16, 32] per wave. Two 16x16 blocks: lo (cols 0..15) and hi (cols 16..31)
                for ii in range_constexpr(4):
                    p_lo_f16 = arith.as_value(
                        flir.arith.TruncFOp(elem_type, p_vals_lo[ii]).result
                    )
                    p_hi_f16 = arith.as_value(
                        flir.arith.TruncFOp(elem_type, p_vals_hi[ii]).result
                    )
                    p_row = (arith.ArithValue(lane_div_16) * 4 + ii).value
                    # Lo: cols 0..15
                    p_lds_lo = (
                        arith.ArithValue(wave_p_offset)
                        + arith.ArithValue(p_row) * BLOCK_N
                        + arith.ArithValue(lane_mod_16)
                    ).value
                    _memref.StoreOp(p_lo_f16, lds_p, [p_lds_lo])
                    # Hi: cols 16..31
                    p_lds_hi = (
                        arith.ArithValue(wave_p_offset)
                        + arith.ArithValue(p_row) * BLOCK_N
                        + 16
                        + arith.ArithValue(lane_mod_16)
                    ).value
                    _memref.StoreOp(p_hi_f16, lds_p, [p_lds_hi])

                # ==== Barrier: ensure all waves done reading K ====
                gpu.barrier()

                # ==== Cooperative V load (transposed) ====
                coop_load_v_transposed(kv_start)
                gpu.barrier()

                # ==== P @ V via MFMA ====
                # P[16, 32] @ V[32, 16chunk] = O[16, 16chunk]
                # Split K=32 into two halves: P_lo[16,16] @ V_top[16,16] + P_hi[16,16] @ V_bot[16,16]

                # Load P A-operand packs: P_lo and P_hi
                p_a_lo_idx = (
                    arith.ArithValue(wave_p_offset)
                    + arith.ArithValue(lane_mod_16) * BLOCK_N
                    + arith.ArithValue(lane_div_16) * 4
                ).value
                p_pack_lo = arith.as_value(
                    vec_ext.load_op(v4f16_type, lds_p, [p_a_lo_idx])
                )

                p_a_hi_idx = (
                    arith.ArithValue(wave_p_offset)
                    + arith.ArithValue(lane_mod_16) * BLOCK_N
                    + 16
                    + arith.ArithValue(lane_div_16) * 4
                ).value
                p_pack_hi = arith.as_value(
                    vec_ext.load_op(v4f16_type, lds_p, [p_a_hi_idx])
                )

                for ds in range_constexpr(K_STEPS):
                    # V_top: V rows 0..15, B-operand from transposed LDS
                    v_top_idx = (
                        (ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE
                        + arith.ArithValue(lane_div_16) * 4
                    ).value
                    v_top = arith.as_value(
                        vec_ext.load_op(v4f16_type, lds_kv, [v_top_idx])
                    )
                    # Accumulate P_lo @ V_top
                    o_accs[ds] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type, [p_pack_lo, v_top, o_accs[ds], 0, 0, 0]
                        )
                    )

                    # V_bot: V rows 16..31, B-operand from transposed LDS
                    v_bot_idx = (
                        (ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE
                        + 16
                        + arith.ArithValue(lane_div_16) * 4
                    ).value
                    v_bot = arith.as_value(
                        vec_ext.load_op(v4f16_type, lds_kv, [v_bot_idx])
                    )
                    # Accumulate P_hi @ V_bot
                    o_accs[ds] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type, [p_pack_hi, v_bot, o_accs[ds], 0, 0, 0]
                        )
                    )

                # ==== Barrier: ensure all waves done reading V ====
                gpu.barrier()

                # ==== Yield ====
                yield_args = m_new + l_new + o_accs
                scf_yield(yield_args)

            # ---- Normalize and store O ----
            m_finals = [arith.as_value(loop.results[i]) for i in range(4)]
            l_finals = [arith.as_value(loop.results[4 + i]) for i in range(4)]
            o_finals = [arith.as_value(loop.results[8 + ds]) for ds in range(K_STEPS)]

            for ds in range_constexpr(K_STEPS):
                for ii in range_constexpr(4):
                    o_val = arith.as_value(
                        vec_ext.extract(o_finals[ds], static_position=[ii], dynamic_position=[])
                    )
                    o_norm = arith.as_value(
                        flir.arith.DivFOp(o_val, l_finals[ii], fastmath=fm_fast).result
                    )
                    o_f16 = arith.as_value(
                        flir.arith.TruncFOp(elem_type, o_norm).result
                    )
                    q_row = (
                        arith.ArithValue(q_start)
                        + arith.ArithValue(wave_q_offset)
                        + arith.ArithValue(lane_div_16) * 4
                        + ii
                    ).value
                    d_col = (flir.const_index(ds * 16) + arith.ArithValue(lane_mod_16)).value
                    o_global = global_idx(q_row, d_col)
                    _memref.StoreOp(o_f16, O, [o_global])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, _state["elem_type"]),
            K: lambda: T.memref(DYN, _state["elem_type"]),
            V: lambda: T.memref(DYN, _state["elem_type"]),
            O: lambda: T.memref(DYN, _state["elem_type"]),
            batch_size: lambda: T.index(),
            seq_len: lambda: T.index(),
        ):
            c1 = arith.as_value(flir.arith_ext.index(1))
            c_nh = arith.as_value(flir.arith_ext.index(NUM_HEADS))
            c_bm = arith.as_value(flir.arith_ext.index(BLOCK_M))
            bs_val = arith.as_value(batch_size)
            sl_val = arith.as_value(seq_len)
            num_q_tiles = arith.as_value(
                flir.arith.DivUIOp(sl_val, c_bm).result
            )
            bs_qt = arith.as_value(
                flir.arith.MulIOp(bs_val, num_q_tiles).result
            )
            grid_x = arith.as_value(
                flir.arith.MulIOp(bs_qt, c_nh).result
            )
            bx = arith.as_value(flir.arith_ext.index(BLOCK_SIZE))
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME],
                grid_size=(grid_x, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Q, K, V, O, seq_len],
            )

    return _FlashAttentionV4_3()
