"""Flash Attention V4.4 kernel builder for FlyDSL.

V4.4 design (CK-aligned direction, rewritten from V4.3):
- CK-aligned baseline tile family: BLOCK_M=64, BLOCK_N=32.
- Q loaded once from global memory into MFMA A-operand packs (register-resident).
- K/V streamed tile-by-tile through LDS.
- Online softmax in fp32 over 32 positions per iteration (2x 16-column groups).
- Causal early-exit keeps KV upper bound at q_start + BLOCK_M.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) -- 4 waves of 64 on AMD (wave64).

Requires: head_dim % 16 == 0, head_dim >= 64, seq_len % 64 == 0.
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


KERNEL_NAME = "flash_attention_v4_4_kernel"


def build_flash_attention_v4_4_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
):
    """Build a FlyDSL Flash Attention V4.4 module.

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

    # CK-oriented direction for the target (B=1, H=64, S=8192, D=128).
    BLOCK_M = 64
    BLOCK_N = 32
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 256
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 16
    K_STEPS = head_dim // 16
    N_MFMA = BLOCK_N // 16  # 2

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 16 == 0, f"head_dim ({head_dim}) must be divisible by 16"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert dtype_str == "f16", "V4.4 currently only supports f16"
    assert BLOCK_N % 16 == 0, f"BLOCK_N ({BLOCK_N}) must be divisible by 16"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # ---- Bank-conflict-friendly LDS strides ----
    K_STRIDE = HEAD_DIM + 2
    VT_STRIDE = BLOCK_N + 2

    # ---- Vectorized cooperative load constants ----
    VEC_WIDTH = 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    assert BLOCK_M % ROWS_PER_BATCH_LOAD == 0
    NUM_BATCHES_Q = BLOCK_M // ROWS_PER_BATCH_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # LDS sizes (element counts, f16 = 2 bytes each)
    # No Q in LDS: Q is read once from global memory to MFMA A packs.
    LDS_KV_SIZE = max(BLOCK_N * K_STRIDE, HEAD_DIM * VT_STRIDE)
    LDS_P_SIZE = BLOCK_M * BLOCK_N

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _FlashAttentionV4_4(flir.MlirModule):
        GPU_MODULE_NAME = f"flash_attn_v4_4_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type = T.f16()
            _state["elem_type"] = elem_type
            _state["lds_kv"] = allocator.allocate_array(elem_type, LDS_KV_SIZE)
            _state["lds_p"] = allocator.allocate_array(elem_type, LDS_P_SIZE)
            allocator.finalize()

        @flir.kernel
        def flash_attention_v4_4_kernel(
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

            # ---- Cooperative load decomposition ----
            c_tpr = flir.const_index(THREADS_PER_ROW_LOAD)
            load_row_in_batch = arith.as_value(flir.arith.DivUIOp(tid, c_tpr).result)
            load_lane_in_row = arith.as_value(flir.arith.RemUIOp(tid, c_tpr).result)
            load_col_base = (arith.ArithValue(load_lane_in_row) * VEC_WIDTH).value

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
                    if KV_NEEDS_GUARD:
                        c_bn = flir.const_index(BLOCK_N)
                        row_valid = arith.as_value(
                            flir.arith.CmpIOp(
                                flir.arith.CmpIPredicate.ult,
                                arith.ArithValue(load_row_in_batch).value,
                                c_bn,
                            ).result
                        )
                        with scf.if_(row_valid):
                            g_idx = global_idx(row_idx, load_col_base)
                            vec = arith.as_value(vec_ext.load_op(v8f16_type, K, [g_idx]))
                            lds_row = (
                                arith.ArithValue(load_row_in_batch) + row_offset
                            ).value
                            lds_idx = (
                                arith.ArithValue(lds_row) * K_STRIDE
                                + arith.ArithValue(load_col_base)
                            ).value
                            vec_ext.store(vec, lds_kv, [lds_idx])
                    else:
                        g_idx = global_idx(row_idx, load_col_base)
                        vec = arith.as_value(vec_ext.load_op(v8f16_type, K, [g_idx]))
                        lds_row = (arith.ArithValue(load_row_in_batch) + row_offset).value
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
                    if KV_NEEDS_GUARD:
                        c_bn = flir.const_index(BLOCK_N)
                        row_valid = arith.as_value(
                            flir.arith.CmpIOp(
                                flir.arith.CmpIPredicate.ult,
                                arith.ArithValue(load_row_in_batch).value,
                                c_bn,
                            ).result
                        )
                        with scf.if_(row_valid):
                            g_idx = global_idx(row_idx, load_col_base)
                            vec = arith.as_value(vec_ext.load_op(v8f16_type, V, [g_idx]))
                            load_row = (
                                arith.ArithValue(load_row_in_batch) + row_offset
                            ).value
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
                    else:
                        g_idx = global_idx(row_idx, load_col_base)
                        vec = arith.as_value(vec_ext.load_op(v8f16_type, V, [g_idx]))
                        load_row = (arith.ArithValue(load_row_in_batch) + row_offset).value
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

            # ---- Load Q once from global memory to MFMA A packs ----
            q_row = (
                arith.ArithValue(q_start)
                + arith.ArithValue(wave_q_offset)
                + arith.ArithValue(lane_mod_16)
            ).value
            q_a_packs = []
            for ks in range_constexpr(K_STEPS):
                q_col = flir.const_index(ks * 16)
                q_col = (arith.ArithValue(q_col) + arith.ArithValue(lane_div_16) * 4).value
                g_idx = global_idx(q_row, q_col)
                q_a_packs.append(arith.as_value(vec_ext.load_op(v4f16_type, Q, [g_idx])))

            # ---- Constants ----
            c_neg_inf = arith.constant(float("-inf"), type=compute_type)
            c_zero_f = arith.constant(0.0, type=compute_type)
            c_sm_scale = arith.constant(sm_scale, type=compute_type)
            c_log2e = arith.constant(1.4426950408889634, type=compute_type)
            c_zero_v4f32 = arith.as_value(arith.constant_vector(0.0, v4f32_type))

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
            if CAUSAL:
                kv_upper = (arith.ArithValue(q_start) + BLOCK_M).value
            else:
                kv_upper = seq_len_v

            # ---- KV loop (step BLOCK_N=64) ----
            with scf.for_(0, kv_upper, BLOCK_N, iter_args=init_args) as loop:
                kv_start = arith.as_value(loop.induction_variable)
                m_old = [arith.as_value(loop.inner_iter_args[i]) for i in range(4)]
                l_old = [arith.as_value(loop.inner_iter_args[4 + i]) for i in range(4)]
                o_accs = [arith.as_value(loop.inner_iter_args[8 + ds]) for ds in range(K_STEPS)]

                # ==== Cooperative K load -> LDS_KV ====
                coop_load_k(kv_start)
                gpu.barrier()

                # ==== Q @ K^T via MFMA -> S[16, BLOCK_N] ====
                s_accs = [c_zero_v4f32 for _ in range_constexpr(N_MFMA)]
                for ks in range_constexpr(K_STEPS):
                    a_pack = q_a_packs[ks]
                    for nm in range_constexpr(N_MFMA):
                        k_row = nm * 16
                        k_lds_idx = (
                            (arith.ArithValue(lane_mod_16) + k_row) * K_STRIDE
                            + ks * 16
                            + arith.ArithValue(lane_div_16) * 4
                        ).value
                        b_pack = arith.as_value(vec_ext.load_op(v4f16_type, lds_kv, [k_lds_idx]))
                        s_accs[nm] = arith.as_value(
                            rocdl.mfma_f32_16x16x16f16(
                                v4f32_type, [a_pack, b_pack, s_accs[nm], 0, 0, 0]
                            )
                        )

                # ==== Online softmax over BLOCK_N positions ====
                # s_vals[nm][ii] where nm in [0..3], ii in [0..3]
                s_vals = [[None for _ in range_constexpr(4)] for _ in range_constexpr(N_MFMA)]
                for ii in range_constexpr(4):
                    for nm in range_constexpr(N_MFMA):
                        s_val = arith.as_value(
                            vec_ext.extract(s_accs[nm], static_position=[ii], dynamic_position=[])
                        )
                        s_val = arith.as_value(
                            flir.arith.MulFOp(
                                s_val, arith.as_value(c_sm_scale), fastmath=fm_fast
                            ).result
                        )

                        if CAUSAL:
                            q_row_i = (
                                arith.ArithValue(q_start)
                                + arith.ArithValue(wave_q_offset)
                                + arith.ArithValue(lane_div_16) * 4
                                + ii
                            ).value
                            kv_col = (
                                arith.ArithValue(kv_start)
                                + nm * 16
                                + arith.ArithValue(lane_mod_16)
                            ).value
                            q_row_i64 = arith.as_value(
                                flir.arith.IndexCastOp(T.i64(), q_row_i).result
                            )
                            kv_col_i64 = arith.as_value(
                                flir.arith.IndexCastOp(T.i64(), kv_col).result
                            )
                            is_masked = arith.as_value(
                                flir.arith.CmpIOp(
                                    flir.arith.CmpIPredicate.ugt, kv_col_i64, q_row_i64
                                ).result
                            )
                            s_val = arith.as_value(
                                flir.arith.SelectOp(
                                    is_masked, arith.as_value(c_neg_inf), s_val
                                ).result
                            )
                        s_vals[nm][ii] = s_val

                width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
                m_new = [None] * 4
                corr = [None] * 4
                p_vals = [[None for _ in range_constexpr(4)] for _ in range_constexpr(N_MFMA)]
                l_new = [None] * 4

                for ii in range_constexpr(4):
                    row_maxes = []
                    for nm in range_constexpr(N_MFMA):
                        row_max_nm = s_vals[nm][ii]
                        for sh in [8, 4, 2, 1]:
                            sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                            peer = arith.as_value(
                                gpu.ShuffleOp(
                                    row_max_nm, sh_i32, width_i32, mode="xor"
                                ).shuffleResult
                            )
                            row_max_nm = arith.as_value(
                                flir.arith.MaximumFOp(row_max_nm, peer).result
                            )
                        row_maxes.append(row_max_nm)

                    combined_max = row_maxes[0]
                    for g in range_constexpr(N_MFMA - 1):
                        combined_max = arith.as_value(
                            flir.arith.MaximumFOp(combined_max, row_maxes[g + 1]).result
                        )

                    m_new[ii] = arith.as_value(
                        flir.arith.MaximumFOp(m_old[ii], combined_max).result
                    )

                    diff_m = arith.as_value(
                        flir.arith.SubFOp(m_old[ii], m_new[ii], fastmath=fm_fast).result
                    )
                    diff_m_s = arith.as_value(
                        flir.arith.MulFOp(
                            diff_m, arith.as_value(c_log2e), fastmath=fm_fast
                        ).result
                    )
                    corr[ii] = arith.as_value(flir.math.exp2(diff_m_s, fastmath=fm_fast))

                    row_sums = []
                    for nm in range_constexpr(N_MFMA):
                        diff = arith.as_value(
                            flir.arith.SubFOp(
                                s_vals[nm][ii], m_new[ii], fastmath=fm_fast
                            ).result
                        )
                        diff_s = arith.as_value(
                            flir.arith.MulFOp(
                                diff, arith.as_value(c_log2e), fastmath=fm_fast
                            ).result
                        )
                        p_vals[nm][ii] = arith.as_value(flir.math.exp2(diff_s, fastmath=fm_fast))

                        row_sum_nm = p_vals[nm][ii]
                        for sh in [8, 4, 2, 1]:
                            sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                            peer = arith.as_value(
                                gpu.ShuffleOp(
                                    row_sum_nm, sh_i32, width_i32, mode="xor"
                                ).shuffleResult
                            )
                            row_sum_nm = arith.as_value(
                                flir.arith.AddFOp(row_sum_nm, peer, fastmath=fm_fast).result
                            )
                        row_sums.append(row_sum_nm)

                    combined_sum = row_sums[0]
                    for g in range_constexpr(N_MFMA - 1):
                        combined_sum = arith.as_value(
                            flir.arith.AddFOp(combined_sum, row_sums[g + 1], fastmath=fm_fast).result
                        )

                    l_corr = arith.as_value(
                        flir.arith.MulFOp(corr[ii], l_old[ii], fastmath=fm_fast).result
                    )
                    l_new[ii] = arith.as_value(
                        flir.arith.AddFOp(l_corr, combined_sum, fastmath=fm_fast).result
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
                for ii in range_constexpr(4):
                    p_row = (arith.ArithValue(lane_div_16) * 4 + ii).value
                    for nm in range_constexpr(N_MFMA):
                        p_f16 = arith.as_value(
                            flir.arith.TruncFOp(elem_type, p_vals[nm][ii]).result
                        )
                        p_lds_idx = (
                            arith.ArithValue(wave_p_offset)
                            + arith.ArithValue(p_row) * BLOCK_N
                            + nm * 16
                            + arith.ArithValue(lane_mod_16)
                        ).value
                        _memref.StoreOp(p_f16, lds_p, [p_lds_idx])

                # ==== Barrier: ensure all waves done reading K ====
                gpu.barrier()

                # ==== Cooperative V load (transposed) ====
                coop_load_v_transposed(kv_start)
                gpu.barrier()

                # ==== P @ V via MFMA ====
                for ds in range_constexpr(K_STEPS):
                    for nm in range_constexpr(N_MFMA):
                        p_a_idx = (
                            arith.ArithValue(wave_p_offset)
                            + arith.ArithValue(lane_mod_16) * BLOCK_N
                            + nm * 16
                            + arith.ArithValue(lane_div_16) * 4
                        ).value
                        p_pack = arith.as_value(
                            vec_ext.load_op(v4f16_type, lds_p, [p_a_idx])
                        )

                        v_idx = (
                            (ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE
                            + nm * 16
                            + arith.ArithValue(lane_div_16) * 4
                        ).value
                        v_pack = arith.as_value(
                            vec_ext.load_op(v4f16_type, lds_kv, [v_idx])
                        )
                        o_accs[ds] = arith.as_value(
                            rocdl.mfma_f32_16x16x16f16(
                                v4f32_type, [p_pack, v_pack, o_accs[ds], 0, 0, 0]
                            )
                        )

                # ==== Barrier: ensure all waves done reading V ====
                gpu.barrier()

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
                    o_f16 = arith.as_value(flir.arith.TruncFOp(elem_type, o_norm).result)
                    q_row_o = (
                        arith.ArithValue(q_start)
                        + arith.ArithValue(wave_q_offset)
                        + arith.ArithValue(lane_div_16) * 4
                        + ii
                    ).value
                    d_col = (flir.const_index(ds * 16) + arith.ArithValue(lane_mod_16)).value
                    o_global = global_idx(q_row_o, d_col)
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
            num_q_tiles = arith.as_value(flir.arith.DivUIOp(sl_val, c_bm).result)
            bs_qt = arith.as_value(flir.arith.MulIOp(bs_val, num_q_tiles).result)
            grid_x = arith.as_value(flir.arith.MulIOp(bs_qt, c_nh).result)
            bx = arith.as_value(flir.arith_ext.index(BLOCK_SIZE))
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME],
                grid_size=(grid_x, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Q, K, V, O, seq_len],
            )

    return _FlashAttentionV4_4()
