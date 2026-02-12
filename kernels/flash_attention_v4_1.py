"""Flash Attention V4.1 kernel builder for FlyDSL.

V4.1 optimizations over V4.0:
- Q preloaded to registers (eliminates Q LDS reads from KV loop)
- V stored transposed in LDS (vectorized v4f16 B-operand loads)
- Bank-conflict-free LDS padding (K stride=HD+2, V transposed stride=BLOCK_N+2)

Tile config: BLOCK_M=64, BLOCK_N=16, 4 waves (256 threads), mfma_f32_16x16x16f16.

Expected improvements from V4.0:
- ~32% fewer LDS instructions (Q reads eliminated, V loads vectorized)
- Reduced LDS bank conflicts

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


KERNEL_NAME = "flash_attention_v4_1_kernel"


def build_flash_attention_v4_1_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
):
    """Build a FlyDSL Flash Attention V4.1 module.

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
    BLOCK_N = 16
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 256
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 16
    K_STEPS = head_dim // 16

    assert head_dim % 16 == 0, f"head_dim ({head_dim}) must be divisible by 16"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert dtype_str == "f16", "V4.1 currently only supports f16"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # ---- Bank-conflict-free LDS strides ----
    # K row-major: stride = HD + 2 (makes row stride odd in bank units)
    # V transposed: stride = BLOCK_N + 2 (same reasoning)
    K_STRIDE = HEAD_DIM + 2   # 130 for HD=128
    VT_STRIDE = BLOCK_N + 2   # 18 for BLOCK_N=16

    # ---- Vectorized cooperative load constants ----
    VEC_WIDTH = 8  # v8f16 = 16 bytes
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    # For Q tile (64 rows)
    assert BLOCK_M % ROWS_PER_BATCH_LOAD == 0
    NUM_BATCHES_Q = BLOCK_M // ROWS_PER_BATCH_LOAD

    # For KV tile (16 rows)
    assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0 or ROWS_PER_BATCH_LOAD >= BLOCK_N
    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # LDS sizes
    LDS_Q_SIZE = BLOCK_M * HEAD_DIM  # Q unpadded (only read once for register preload)
    LDS_KV_SIZE = max(BLOCK_N * K_STRIDE, HEAD_DIM * VT_STRIDE)  # max(K padded, Vt padded)
    LDS_P_SIZE = BLOCK_M * BLOCK_N

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _FlashAttentionV4_1(flir.MlirModule):
        GPU_MODULE_NAME = f"flash_attn_v4_1_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type = T.f16()
            _state["elem_type"] = elem_type
            _state["lds_q"] = allocator.allocate_array(elem_type, LDS_Q_SIZE)
            _state["lds_kv"] = allocator.allocate_array(elem_type, LDS_KV_SIZE)
            _state["lds_p"] = allocator.allocate_array(elem_type, LDS_P_SIZE)
            allocator.finalize()

        @flir.kernel
        def flash_attention_v4_1_kernel(
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

            # ---- LDS views ----
            base_ptr = allocator.get_base()
            lds_q = _state["lds_q"](base_ptr).get()
            lds_kv = _state["lds_kv"](base_ptr).get()
            lds_p = _state["lds_p"](base_ptr).get()

            # ---- Thread / block indices ----
            block_id = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            # ---- Wave decomposition ----
            c_ws = flir.const_index(WARP_SIZE)
            wave_id = arith.as_value(flir.arith.DivUIOp(tid, c_ws).result)
            lane = arith.as_value(flir.arith.RemUIOp(tid, c_ws).result)

            # ---- MFMA lane decomposition (within each wave) ----
            c16 = flir.const_index(16)
            lane_div_16 = arith.as_value(flir.arith.DivUIOp(lane, c16).result)
            lane_mod_16 = arith.as_value(flir.arith.RemUIOp(lane, c16).result)

            # ---- Wave's Q-row offset in the Q tile ----
            wave_q_offset = (arith.ArithValue(wave_id) * ROWS_PER_WAVE).value
            # Wave's P offset in lds_p
            wave_p_offset = (arith.ArithValue(wave_id) * ROWS_PER_WAVE * BLOCK_N).value

            # ---- Decompose block_id -> (batch_idx, q_tile_idx, head_idx) ----
            c_nh = flir.const_index(NUM_HEADS)
            head_idx = arith.as_value(flir.arith.RemUIOp(block_id, c_nh).result)
            temp = arith.as_value(flir.arith.DivUIOp(block_id, c_nh).result)
            c_bm = flir.const_index(BLOCK_M)
            num_q_tiles = arith.as_value(flir.arith.DivUIOp(seq_len_v, c_bm).result)
            q_tile_idx = arith.as_value(flir.arith.RemUIOp(temp, num_q_tiles).result)
            batch_idx = arith.as_value(flir.arith.DivUIOp(temp, num_q_tiles).result)
            q_start = (arith.ArithValue(q_tile_idx) * BLOCK_M).value

            # ---- Vectorized load thread decomposition ----
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

            # ---- Cooperative Q load (64 rows, all 256 threads, unpadded) ----
            def coop_load_q():
                for batch in range_constexpr(NUM_BATCHES_Q):
                    row_offset = batch * ROWS_PER_BATCH_LOAD
                    row_idx = (
                        arith.ArithValue(q_start)
                        + arith.ArithValue(load_row_in_batch)
                        + row_offset
                    ).value
                    g_idx = global_idx(row_idx, load_col_base)
                    vec = arith.as_value(
                        vec_ext.load_op(v8f16_type, Q, [g_idx])
                    )
                    lds_row = (
                        arith.ArithValue(load_row_in_batch) + row_offset
                    ).value
                    lds_idx = (
                        arith.ArithValue(lds_row) * HEAD_DIM
                        + arith.ArithValue(load_col_base)
                    ).value
                    vec_ext.store(vec, lds_q, [lds_idx])

            # ---- Cooperative K load (row-major with padded stride) ----
            def coop_load_k(tile_start):
                if KV_NEEDS_GUARD:
                    c_bn = flir.const_index(BLOCK_N)
                    row_ok = arith.as_value(
                        flir.arith.CmpIOp(
                            flir.arith.CmpIPredicate.ult,
                            load_row_in_batch, c_bn,
                        ).result
                    )
                    from flydsl.dialects.ext.scf import IfOp
                    if_op = IfOp(row_ok)
                    with if_op:
                        row_idx = (
                            arith.ArithValue(tile_start)
                            + arith.ArithValue(load_row_in_batch)
                        ).value
                        g_idx = global_idx(row_idx, load_col_base)
                        vec = arith.as_value(
                            vec_ext.load_op(v8f16_type, K, [g_idx])
                        )
                        lds_idx = (
                            arith.ArithValue(load_row_in_batch) * K_STRIDE
                            + arith.ArithValue(load_col_base)
                        ).value
                        vec_ext.store(vec, lds_kv, [lds_idx])
                else:
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

            # ---- Cooperative V load (transposed with padded stride) ----
            # Global V[row, col] -> LDS Vt[col, row] at lds_kv[col * VT_STRIDE + row]
            def coop_load_v_transposed(tile_start):
                if KV_NEEDS_GUARD:
                    c_bn = flir.const_index(BLOCK_N)
                    row_ok = arith.as_value(
                        flir.arith.CmpIOp(
                            flir.arith.CmpIPredicate.ult,
                            load_row_in_batch, c_bn,
                        ).result
                    )
                    from flydsl.dialects.ext.scf import IfOp
                    if_op = IfOp(row_ok)
                    with if_op:
                        row_idx = (
                            arith.ArithValue(tile_start)
                            + arith.ArithValue(load_row_in_batch)
                        ).value
                        g_idx = global_idx(row_idx, load_col_base)
                        vec = arith.as_value(
                            vec_ext.load_op(v8f16_type, V, [g_idx])
                        )
                        # Scatter-store transposed: V[row, col+e] -> lds[col_e * VT_STRIDE + row]
                        for e in range_constexpr(VEC_WIDTH):
                            elem = arith.as_value(
                                vec_ext.extract(vec, static_position=[e], dynamic_position=[])
                            )
                            col_e = (arith.ArithValue(load_col_base) + e).value
                            lds_idx = (
                                arith.ArithValue(col_e) * VT_STRIDE
                                + arith.ArithValue(load_row_in_batch)
                            ).value
                            _memref.StoreOp(elem, lds_kv, [lds_idx])
                else:
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
                        # Scatter-store transposed
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

            # ---- Load Q tile to LDS ----
            coop_load_q()
            gpu.barrier()

            # ---- Preload Q A-operand packs into registers ----
            # Each lane loads K_STEPS v4f16 packs from LDS_Q (one-time cost).
            # At step ks, thread (b,n) needs Q[wave_row + n, ks*16 + b*4 : ks*16+b*4+4]
            q_a_packs = []
            for ks in range_constexpr(K_STEPS):
                q_lds_idx = (
                    (arith.ArithValue(wave_q_offset)
                     + arith.ArithValue(lane_mod_16)) * HEAD_DIM
                    + ks * 16
                    + arith.ArithValue(lane_div_16) * 4
                ).value
                q_a_packs.append(arith.as_value(
                    vec_ext.load_op(v4f16_type, lds_q, [q_lds_idx])
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
            # [m_0..m_3, l_0..l_3, o_acc_0..o_acc_{K_STEPS-1}]
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
            if CAUSAL:
                kv_upper = (arith.ArithValue(q_start) + BLOCK_M).value
            else:
                kv_upper = seq_len_v

            # ---- KV loop ----
            with scf.for_(0, kv_upper, BLOCK_N, iter_args=init_args) as loop:
                kv_start = arith.as_value(loop.induction_variable)
                m_old = [arith.as_value(loop.inner_iter_args[i]) for i in range(4)]
                l_old = [arith.as_value(loop.inner_iter_args[4 + i]) for i in range(4)]
                o_accs = [arith.as_value(loop.inner_iter_args[8 + ds]) for ds in range(K_STEPS)]

                # ==== Cooperative K load -> LDS_KV (row-major, padded stride) ====
                coop_load_k(kv_start)
                gpu.barrier()

                # ==== Q @ K^T via MFMA (Q from registers, K from LDS) ====
                s_acc = c_zero_v4f32
                for ks in range_constexpr(K_STEPS):
                    # A operand (Q): from preloaded registers
                    a_pack = q_a_packs[ks]
                    # B operand (K^T): from LDS with padded stride
                    k_lds_idx = (
                        arith.ArithValue(lane_mod_16) * K_STRIDE
                        + ks * 16
                        + arith.ArithValue(lane_div_16) * 4
                    ).value
                    b_pack = arith.as_value(
                        vec_ext.load_op(v4f16_type, lds_kv, [k_lds_idx])
                    )
                    s_acc = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(v4f32_type, [a_pack, b_pack, s_acc, 0, 0, 0])
                    )

                # ==== Online softmax (per-wave, per-row) ====
                s_vals = []
                for ii in range_constexpr(4):
                    s_ii = arith.as_value(
                        vec_ext.extract(s_acc, static_position=[ii], dynamic_position=[])
                    )
                    s_ii = arith.as_value(
                        flir.arith.MulFOp(s_ii, arith.as_value(c_sm_scale), fastmath=fm_fast).result
                    )
                    if CAUSAL:
                        q_row = (
                            arith.ArithValue(q_start)
                            + arith.ArithValue(wave_q_offset)
                            + arith.ArithValue(lane_div_16) * 4
                            + ii
                        ).value
                        kv_col = (arith.ArithValue(kv_start) + arith.ArithValue(lane_mod_16)).value
                        q_row_i64 = arith.as_value(flir.arith.IndexCastOp(T.i64(), q_row).result)
                        kv_col_i64 = arith.as_value(flir.arith.IndexCastOp(T.i64(), kv_col).result)
                        is_masked = arith.as_value(
                            flir.arith.CmpIOp(
                                flir.arith.CmpIPredicate.ugt, kv_col_i64, q_row_i64,
                            ).result
                        )
                        s_ii = arith.as_value(
                            flir.arith.SelectOp(is_masked, arith.as_value(c_neg_inf), s_ii).result
                        )
                    s_vals.append(s_ii)

                width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
                m_new = [None] * 4
                corr = [None] * 4
                p_vals = [None] * 4
                l_new = [None] * 4

                for ii in range_constexpr(4):
                    row_max = s_vals[ii]
                    for sh in [8, 4, 2, 1]:
                        sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                        peer = arith.as_value(
                            gpu.ShuffleOp(row_max, sh_i32, width_i32, mode="xor").shuffleResult
                        )
                        row_max = arith.as_value(
                            flir.arith.MaximumFOp(row_max, peer).result
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

                    diff_s = arith.as_value(
                        flir.arith.SubFOp(s_vals[ii], m_new[ii], fastmath=fm_fast).result
                    )
                    diff_s_s = arith.as_value(
                        flir.arith.MulFOp(diff_s, arith.as_value(c_log2e), fastmath=fm_fast).result
                    )
                    p_vals[ii] = arith.as_value(flir.math.exp2(diff_s_s, fastmath=fm_fast))

                    row_sum = p_vals[ii]
                    for sh in [8, 4, 2, 1]:
                        sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                        peer = arith.as_value(
                            gpu.ShuffleOp(row_sum, sh_i32, width_i32, mode="xor").shuffleResult
                        )
                        row_sum = arith.as_value(
                            flir.arith.AddFOp(row_sum, peer, fastmath=fm_fast).result
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

                # ==== P store to LDS_P (each wave writes its 16x16 section) ====
                for ii in range_constexpr(4):
                    p_f16 = arith.as_value(
                        flir.arith.TruncFOp(elem_type, p_vals[ii]).result
                    )
                    p_row = (arith.ArithValue(lane_div_16) * 4 + ii).value
                    p_lds_idx = (
                        arith.ArithValue(wave_p_offset)
                        + arith.ArithValue(p_row) * BLOCK_N
                        + arith.ArithValue(lane_mod_16)
                    ).value
                    _memref.StoreOp(p_f16, lds_p, [p_lds_idx])

                # ==== Barrier: ensure all waves finished reading K from lds_kv ====
                gpu.barrier()

                # ==== Cooperative V load -> LDS_KV (transposed, padded stride) ====
                coop_load_v_transposed(kv_start)
                gpu.barrier()

                # ==== P load (A-operand, wave-local) ====
                p_a_idx = (
                    arith.ArithValue(wave_p_offset)
                    + arith.ArithValue(lane_mod_16) * BLOCK_N
                    + arith.ArithValue(lane_div_16) * 4
                ).value
                p_pack = arith.as_value(
                    vec_ext.load_op(v4f16_type, lds_p, [p_a_idx])
                )

                # ==== P @ V via MFMA (V from transposed LDS, vectorized v4f16 loads) ====
                # V transposed: V[row, col] at lds_kv[col * VT_STRIDE + row]
                # B-operand: V[b*4:b*4+4, ds*16+n] = lds_kv[(ds*16+n) * VT_STRIDE + b*4]
                # -> v4f16 at base (ds*16 + lane_mod_16) * VT_STRIDE + lane_div_16 * 4
                for ds in range_constexpr(K_STEPS):
                    v_lds_idx = (
                        (ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE
                        + arith.ArithValue(lane_div_16) * 4
                    ).value
                    v_pack = arith.as_value(
                        vec_ext.load_op(v4f16_type, lds_kv, [v_lds_idx])
                    )
                    o_accs[ds] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type, [p_pack, v_pack, o_accs[ds], 0, 0, 0]
                        )
                    )

                # ==== Barrier: ensure all waves finished P@V (reading lds_kv)
                #       before next iteration overwrites lds_kv with K ====
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

    return _FlashAttentionV4_1()
