"""Flash Attention V4.4 kernel builder for FlyDSL.

V4.4 optimization over V4.3 (CK-aligned design):
- BLOCK_N=64 (vs 32): halves KV loop iterations.
- K loaded in chunks of kK0=32 along head_dim (K0_LOOPS inner iterations).
  K_STRIDE = kK0 + 2 = 34  (was HEAD_DIM + 2 = 130).
- V loaded in chunks of kK1=32 (K1_LOOPS inner iterations).
  VT_STRIDE = kK1 + 2 = 34.
- K/V prefetching: overlaps global loads with MFMA computation.
  K[k0+1] is fetched while computing with K[k0].
  V[0] is fetched during last K computation.
  V[k1+1] is fetched while computing with V[k1].
- LDS reduced: max(64*34, 128*34) = 4352 elem = 8.5KB + 8KB(P) = 16.5KB.
- Softmax over 64 positions (four 16-wide groups).
- Causal early-exit retained.

Tile config: BLOCK_M=64, BLOCK_N=64, kK0=32, kK1=32,
            4 waves (256 threads), mfma_f32_16x16x16f16.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) -- 4 waves of 64 on AMD (wave64).

Requires: head_dim % 32 == 0, seq_len % 64 == 0, head_dim >= 64.
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
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    BLOCK_M = 64
    BLOCK_N = 64
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 256
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 16
    K_STEPS = head_dim // 16

    N_MFMA = BLOCK_N // 16  # 4

    kK0 = 32
    kK1 = 32
    K0_LOOPS = head_dim // kK0
    K1_LOOPS = BLOCK_N // kK1      # 2
    K_STEPS_PER_CHUNK = kK0 // 16  # 2

    assert head_dim % kK0 == 0
    assert BLOCK_N % kK1 == 0
    assert head_dim % 16 == 0
    assert head_dim >= 64
    assert dtype_str == "f16"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    K_STRIDE = kK0 + 2     # 34
    VT_STRIDE = kK1 + 2    # 34

    VEC_WIDTH = 8
    K_THREADS_PER_ROW = kK0 // VEC_WIDTH   # 4
    V_THREADS_PER_ROW = HEAD_DIM // VEC_WIDTH  # 16
    V_ROWS_PER_BATCH = BLOCK_SIZE // V_THREADS_PER_ROW  # 16
    NUM_BATCHES_V = kK1 // V_ROWS_PER_BATCH  # 2

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

            base_ptr = allocator.get_base()
            lds_kv = _state["lds_kv"](base_ptr).get()
            lds_p = _state["lds_p"](base_ptr).get()

            block_id = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            c_ws = flir.const_index(WARP_SIZE)
            wave_id = arith.as_value(flir.arith.DivUIOp(tid, c_ws).result)
            lane = arith.as_value(flir.arith.RemUIOp(tid, c_ws).result)

            c16 = flir.const_index(16)
            lane_div_16 = arith.as_value(flir.arith.DivUIOp(lane, c16).result)
            lane_mod_16 = arith.as_value(flir.arith.RemUIOp(lane, c16).result)

            wave_q_offset = (arith.ArithValue(wave_id) * ROWS_PER_WAVE).value
            wave_p_offset = (arith.ArithValue(wave_id) * ROWS_PER_WAVE * BLOCK_N).value

            c_nh = flir.const_index(NUM_HEADS)
            head_idx = arith.as_value(flir.arith.RemUIOp(block_id, c_nh).result)
            temp = arith.as_value(flir.arith.DivUIOp(block_id, c_nh).result)
            c_bm = flir.const_index(BLOCK_M)
            num_q_tiles = arith.as_value(flir.arith.DivUIOp(seq_len_v, c_bm).result)
            q_tile_idx = arith.as_value(flir.arith.RemUIOp(temp, num_q_tiles).result)
            batch_idx = arith.as_value(flir.arith.DivUIOp(temp, num_q_tiles).result)
            q_start = (arith.ArithValue(q_tile_idx) * BLOCK_M).value

            # K load decomposition (4 threads/row for kK0=32)
            c_ktpr = flir.const_index(K_THREADS_PER_ROW)
            k_load_row = arith.as_value(flir.arith.DivUIOp(tid, c_ktpr).result)
            k_load_col_lane = arith.as_value(flir.arith.RemUIOp(tid, c_ktpr).result)
            k_load_col_base = (arith.ArithValue(k_load_col_lane) * VEC_WIDTH).value

            # V load decomposition (16 threads/row for HEAD_DIM)
            c_vtpr = flir.const_index(V_THREADS_PER_ROW)
            v_load_row_in_batch = arith.as_value(flir.arith.DivUIOp(tid, c_vtpr).result)
            v_load_col_base = (
                arith.ArithValue(flir.arith.RemUIOp(tid, c_vtpr).result) * VEC_WIDTH
            ).value

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

            # ---- Prefetch helpers: separate load-to-regs from store-to-lds ----

            def load_k_to_regs(tile_start, k0_col_offset):
                """Issue global load for K chunk → returns v8f16 register."""
                row_idx = (arith.ArithValue(tile_start) + arith.ArithValue(k_load_row)).value
                col_idx = (flir.const_index(k0_col_offset) + arith.ArithValue(k_load_col_base)).value
                g_idx = global_idx(row_idx, col_idx)
                return arith.as_value(vec_ext.load_op(v8f16_type, K, [g_idx]))

            def store_k_regs_to_lds(k_reg):
                """Store K register data to LDS_KV."""
                lds_idx = (
                    arith.ArithValue(k_load_row) * K_STRIDE
                    + arith.ArithValue(k_load_col_base)
                ).value
                vec_ext.store(k_reg, lds_kv, [lds_idx])

            def load_v_to_regs(tile_start, k1_row_offset):
                """Issue global loads for V chunk → returns list of v8f16."""
                regs = []
                for batch in range_constexpr(NUM_BATCHES_V):
                    row_offset = batch * V_ROWS_PER_BATCH
                    row_idx = (
                        arith.ArithValue(tile_start) + k1_row_offset
                        + arith.ArithValue(v_load_row_in_batch) + row_offset
                    ).value
                    g_idx = global_idx(row_idx, v_load_col_base)
                    regs.append(arith.as_value(vec_ext.load_op(v8f16_type, V, [g_idx])))
                return regs

            def store_v_regs_to_lds(v_regs):
                """Scatter-store V registers transposed to LDS_KV."""
                for batch_idx in range_constexpr(NUM_BATCHES_V):
                    vec = v_regs[batch_idx]
                    load_row = (
                        arith.ArithValue(v_load_row_in_batch) + batch_idx * V_ROWS_PER_BATCH
                    ).value
                    for e in range_constexpr(VEC_WIDTH):
                        elem = arith.as_value(
                            vec_ext.extract(vec, static_position=[e], dynamic_position=[])
                        )
                        col_e = (arith.ArithValue(v_load_col_base) + e).value
                        lds_idx = (
                            arith.ArithValue(col_e) * VT_STRIDE
                            + arith.ArithValue(load_row)
                        ).value
                        _memref.StoreOp(elem, lds_kv, [lds_idx])

            # ---- Load Q to registers (once) ----
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
                q_a_packs.append(arith.as_value(
                    vec_ext.load_op(v4f16_type, Q, [g_idx])
                ))

            # ---- Constants ----
            c_neg_inf = arith.constant(float("-inf"), type=compute_type)
            c_zero_f = arith.constant(0.0, type=compute_type)
            c_sm_scale = arith.constant(sm_scale, type=compute_type)
            c_log2e = arith.constant(1.4426950408889634, type=compute_type)
            c_zero_v4f32 = arith.as_value(arith.constant_vector(0.0, v4f32_type))

            init_args = []
            for _ in range_constexpr(4):
                init_args.append(arith.as_value(c_neg_inf))
            for _ in range_constexpr(4):
                init_args.append(arith.as_value(c_zero_f))
            for _ in range_constexpr(K_STEPS):
                init_args.append(c_zero_v4f32)

            if CAUSAL:
                kv_upper = (arith.ArithValue(q_start) + BLOCK_M).value
            else:
                kv_upper = seq_len_v

            # ================================================================
            # KV loop (step BLOCK_N=64)
            # ================================================================
            with scf.for_(0, kv_upper, BLOCK_N, iter_args=init_args) as loop:
                kv_start = arith.as_value(loop.induction_variable)
                m_old = [arith.as_value(loop.inner_iter_args[i]) for i in range(4)]
                l_old = [arith.as_value(loop.inner_iter_args[4 + i]) for i in range(4)]
                o_accs = [arith.as_value(loop.inner_iter_args[8 + ds]) for ds in range(K_STEPS)]

                s_accs = [c_zero_v4f32] * N_MFMA

                # ========================================================
                # QK GEMM Phase with K prefetching
                # ========================================================
                # K[0]: load → store → prefetch K[1]
                k_reg = load_k_to_regs(kv_start, 0)
                store_k_regs_to_lds(k_reg)
                k_reg = load_k_to_regs(kv_start, kK0)  # Prefetch K[1]

                for k0 in range_constexpr(K0_LOOPS):
                    gpu.barrier()  # K[k0] in LDS

                    # Compute QK with K[k0]
                    for local_ks in range_constexpr(K_STEPS_PER_CHUNK):
                        global_ks = k0 * K_STEPS_PER_CHUNK + local_ks
                        a_pack = q_a_packs[global_ks]
                        for nm in range_constexpr(N_MFMA):
                            k_lds_idx = (
                                (arith.ArithValue(lane_mod_16) + nm * 16) * K_STRIDE
                                + local_ks * 16
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

                    if k0 < K0_LOOPS - 1:
                        gpu.barrier()  # All reads of K[k0] done

                        # Store prefetched K[k0+1] to LDS
                        store_k_regs_to_lds(k_reg)

                        # Prefetch K[k0+2] or V[0]
                        if k0 + 2 < K0_LOOPS:
                            k_reg = load_k_to_regs(kv_start, (k0 + 2) * kK0)
                        if k0 == K0_LOOPS - 2:
                            # Last K store: also prefetch V[0]
                            v_regs = load_v_to_regs(kv_start, 0)

                # After last QK compute: no barrier here yet
                # (softmax is register-only, no LDS_KV conflict)

                # ========================================================
                # Online Softmax over 64 positions
                # ========================================================
                s_vals = [[] for _ in range(N_MFMA)]
                for ii in range_constexpr(4):
                    for nm in range_constexpr(N_MFMA):
                        s_val = arith.as_value(
                            vec_ext.extract(s_accs[nm], static_position=[ii], dynamic_position=[])
                        )
                        s_val = arith.as_value(
                            flir.arith.MulFOp(s_val, arith.as_value(c_sm_scale), fastmath=fm_fast).result
                        )
                        if CAUSAL:
                            q_row_c = (
                                arith.ArithValue(q_start)
                                + arith.ArithValue(wave_q_offset)
                                + arith.ArithValue(lane_div_16) * 4
                                + ii
                            ).value
                            kv_col = (
                                arith.ArithValue(kv_start) + nm * 16
                                + arith.ArithValue(lane_mod_16)
                            ).value
                            q_row_i64 = arith.as_value(flir.arith.IndexCastOp(T.i64(), q_row_c).result)
                            kv_col_i64 = arith.as_value(flir.arith.IndexCastOp(T.i64(), kv_col).result)
                            is_masked = arith.as_value(
                                flir.arith.CmpIOp(
                                    flir.arith.CmpIPredicate.ugt, kv_col_i64, q_row_i64,
                                ).result
                            )
                            s_val = arith.as_value(
                                flir.arith.SelectOp(is_masked, arith.as_value(c_neg_inf), s_val).result
                            )
                        s_vals[nm].append(s_val)

                width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
                m_new = [None] * 4
                corr = [None] * 4
                p_vals = [[None] * 4 for _ in range(N_MFMA)]
                l_new = [None] * 4

                for ii in range_constexpr(4):
                    row_maxes = []
                    for nm in range_constexpr(N_MFMA):
                        row_max_nm = s_vals[nm][ii]
                        for sh in [8, 4, 2, 1]:
                            sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                            peer = arith.as_value(
                                gpu.ShuffleOp(row_max_nm, sh_i32, width_i32, mode="xor").shuffleResult
                            )
                            row_max_nm = arith.as_value(
                                flir.arith.MaximumFOp(row_max_nm, peer).result
                            )
                        row_maxes.append(row_max_nm)

                    combined_max = row_maxes[0]
                    for _g in range_constexpr(N_MFMA - 1):
                        combined_max = arith.as_value(
                            flir.arith.MaximumFOp(combined_max, row_maxes[_g + 1]).result
                        )

                    m_new[ii] = arith.as_value(
                        flir.arith.MaximumFOp(m_old[ii], combined_max).result
                    )

                    diff_m = arith.as_value(
                        flir.arith.SubFOp(m_old[ii], m_new[ii], fastmath=fm_fast).result
                    )
                    diff_m_s = arith.as_value(
                        flir.arith.MulFOp(diff_m, arith.as_value(c_log2e), fastmath=fm_fast).result
                    )
                    corr[ii] = arith.as_value(flir.math.exp2(diff_m_s, fastmath=fm_fast))

                    row_sums = []
                    for nm in range_constexpr(N_MFMA):
                        diff = arith.as_value(
                            flir.arith.SubFOp(s_vals[nm][ii], m_new[ii], fastmath=fm_fast).result
                        )
                        diff_s = arith.as_value(
                            flir.arith.MulFOp(diff, arith.as_value(c_log2e), fastmath=fm_fast).result
                        )
                        p_vals[nm][ii] = arith.as_value(flir.math.exp2(diff_s, fastmath=fm_fast))

                        row_sum_nm = p_vals[nm][ii]
                        for sh in [8, 4, 2, 1]:
                            sh_i32 = arith.as_value(arith.constant(sh, type=T.i32()))
                            peer = arith.as_value(
                                gpu.ShuffleOp(row_sum_nm, sh_i32, width_i32, mode="xor").shuffleResult
                            )
                            row_sum_nm = arith.as_value(
                                flir.arith.AddFOp(row_sum_nm, peer, fastmath=fm_fast).result
                            )
                        row_sums.append(row_sum_nm)

                    combined_sum = row_sums[0]
                    for _g in range_constexpr(N_MFMA - 1):
                        combined_sum = arith.as_value(
                            flir.arith.AddFOp(combined_sum, row_sums[_g + 1], fastmath=fm_fast).result
                        )

                    l_corr = arith.as_value(
                        flir.arith.MulFOp(corr[ii], l_old[ii], fastmath=fm_fast).result
                    )
                    l_new[ii] = arith.as_value(
                        flir.arith.AddFOp(l_corr, combined_sum, fastmath=fm_fast).result
                    )

                # Rescale O
                corr_vec = arith.as_value(
                    vec_ext.from_elements(v4f32_type, [corr[0], corr[1], corr[2], corr[3]])
                )
                for ds in range_constexpr(K_STEPS):
                    o_accs[ds] = arith.as_value(
                        flir.arith.MulFOp(o_accs[ds], corr_vec, fastmath=fm_fast).result
                    )

                # ========================================================
                # P store to LDS_P
                # ========================================================
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

                # ========================================================
                # PV GEMM Phase with V prefetching
                # ========================================================
                # Barrier: all K reads done + P stores visible
                gpu.barrier()

                # V[0] was prefetched during K phase → store to LDS
                store_v_regs_to_lds(v_regs)
                # Prefetch V[1]
                v_regs = load_v_to_regs(kv_start, kK1)

                for k1 in range_constexpr(K1_LOOPS):
                    gpu.barrier()  # V[k1] in LDS

                    # P A-operand
                    p_a_lo_idx = (
                        arith.ArithValue(wave_p_offset)
                        + arith.ArithValue(lane_mod_16) * BLOCK_N
                        + k1 * kK1
                        + arith.ArithValue(lane_div_16) * 4
                    ).value
                    p_pack_lo = arith.as_value(
                        vec_ext.load_op(v4f16_type, lds_p, [p_a_lo_idx])
                    )

                    p_a_hi_idx = (
                        arith.ArithValue(wave_p_offset)
                        + arith.ArithValue(lane_mod_16) * BLOCK_N
                        + k1 * kK1 + 16
                        + arith.ArithValue(lane_div_16) * 4
                    ).value
                    p_pack_hi = arith.as_value(
                        vec_ext.load_op(v4f16_type, lds_p, [p_a_hi_idx])
                    )

                    for ds in range_constexpr(K_STEPS):
                        v_top_idx = (
                            (ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE
                            + arith.ArithValue(lane_div_16) * 4
                        ).value
                        v_top = arith.as_value(
                            vec_ext.load_op(v4f16_type, lds_kv, [v_top_idx])
                        )
                        o_accs[ds] = arith.as_value(
                            rocdl.mfma_f32_16x16x16f16(
                                v4f32_type, [p_pack_lo, v_top, o_accs[ds], 0, 0, 0]
                            )
                        )

                        v_bot_idx = (
                            (ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE
                            + 16
                            + arith.ArithValue(lane_div_16) * 4
                        ).value
                        v_bot = arith.as_value(
                            vec_ext.load_op(v4f16_type, lds_kv, [v_bot_idx])
                        )
                        o_accs[ds] = arith.as_value(
                            rocdl.mfma_f32_16x16x16f16(
                                v4f32_type, [p_pack_hi, v_bot, o_accs[ds], 0, 0, 0]
                            )
                        )

                    if k1 < K1_LOOPS - 1:
                        gpu.barrier()  # All V[k1] reads done
                        store_v_regs_to_lds(v_regs)  # Store prefetched V[k1+1]

                # Final barrier
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
                    o_f16 = arith.as_value(
                        flir.arith.TruncFOp(elem_type, o_norm).result
                    )
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

    return _FlashAttentionV4_4()
