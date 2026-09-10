# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""gfx950 page-64 FP8 flash attention with independent value dimensions."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from kernels.attention.flash_attn_utils import (
    DualwaveFp8GemmHelper,
    DualwaveFp8KernelContext,
    DualwaveFp8KvGmemToLdsLoader,
    DualwaveFp8KvLdsToVgprLoader,
    DualwaveFp8SoftmaxHelper,
    DualwaveFp8StoreHelper,
    _make_paged_dualwave_swp_fp8_traits,
    _sched_barrier_exp_pairs,
    _sched_barrier_pairs,
    _stagger_extra_barrier_if_one,
    _stagger_extra_barrier_if_zero,
    _waitcnt_vm_n,
)
from kernels.common.kernels_common import dtype_to_elem_type
from kernels.common.tensor_shim import _run_compiled


def _query_bound_is_safe(ctx, upper_bound, maximum):
    # A proof must reject unordered comparisons and preserve overflow behavior.
    with fx.fastmath(None):
        safe = ((upper_bound - maximum) * ctx.c_logit_scale <= ctx.c_rescale_thr_f) & (ctx.c_logit_scale > 0)
    unsafe = safe == fx.Boolean(False)
    unsafe_lanes = fx.Int64(rocdl.ballot(fx.Int64.ir_type, unsafe.ir_value()))
    return unsafe_lanes == 0


def build_flash_attn_paged_fp8_module(
    num_heads,
    head_dim,
    value_head_dim=None,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    rescale_threshold=8.0,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    paged_bn128=False,
    paged_bn128_varlen=False,
    batch_interleave_group=1,
):
    """Build the gfx950 packed-varlen page-64 FP8 attention launcher."""
    gpu_arch = get_hip_arch()
    if value_head_dim is None:
        value_head_dim = head_dim

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(f"flash_attn_dualwave_swp requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}")
    if (
        not paged
        or dtype_str != "fp8"
        or not causal
        or not varlen
        or not cross_seqlen
        or (head_dim, value_head_dim) not in ((128, 128), (192, 128), (192, 192))
        or kv_cache_layout != "vectorized"
        or int(num_kv_splits) != 1
    ):
        raise RuntimeError(
            "paged FP8 flash_attn requires gfx950, causal packed-varlen cross-attention, "
            "page-64 vectorized KV, (head_dim,value_head_dim) in "
            "{(128,128),(192,128),(192,192)}, "
            "and num_kv_splits=1"
        )

    if num_kv_heads is None:
        num_kv_heads = num_heads
    batch_interleave_group = int(batch_interleave_group)
    if batch_interleave_group < 1:
        raise ValueError(f"batch_interleave_group must be positive, got {batch_interleave_group}")
    if batch_interleave_group > 1 and (
        (head_dim == 128 and not paged_bn128) or (paged_bn128 and not paged_bn128_varlen)
    ):
        raise ValueError("batch interleaving requires generic D192 or packed-varlen BN128")
    assert num_heads % num_kv_heads == 0
    traits = _make_paged_dualwave_swp_fp8_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        value_head_dim=value_head_dim,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
        rescale_threshold=rescale_threshold,
        dualwave_swp_setprio=dualwave_swp_setprio,
        dualwave_swp_debug_lazy_counts=dualwave_swp_debug_lazy_counts,
        dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        varlen=not paged_bn128 or paged_bn128_varlen,
        bn128=paged_bn128,
        batch_interleave_group=batch_interleave_group,
    )
    BLOCK_M = traits.BLOCK_M
    BLOCK_SIZE = traits.BLOCK_SIZE
    HEAD_DIM = traits.HEAD_DIM
    NUM_HEADS_Q = traits.NUM_HEADS_Q
    PAGED_BN128 = bool(paged_bn128)
    PAGED_BN128_VARLEN = bool(paged_bn128_varlen)
    BATCH_INTERLEAVE_GROUP = traits.BATCH_INTERLEAVE_GROUP
    DEFAULT_STRIDE_Q_N = traits.DEFAULT_STRIDE_Q_N
    DEFAULT_STRIDE_O_N = traits.NUM_HEADS_Q * traits.HEAD_DIM_V
    DEFAULT_STRIDE_KV_N = traits.DEFAULT_STRIDE_KV_N
    _dualwave_swp_fp8_cache_tag = traits.cache_tag
    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
        vt: fx.Array[fx.BFloat16, traits.VT_BF16_TOTAL, 16]

    # BN128: two BLOCK_N=64 KV tiles per iteration, one merged softmax correction.
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_paged_fp8_bn128_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
    ):
        ctx = DualwaveFp8KernelContext(
            traits,
            Q,
            K,
            V,
            O,
            DebugCounts,
            CuSeqQ,
            CuSeqKv,
            QDescale,
            KDescale,
            VDescale,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            stride_o_n=DEFAULT_STRIDE_O_N,
            BlockTable=BlockTable,
            block_table_stride=block_table_stride,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        if const_expr(PAGED_BN128_VARLEN):
            ctx.init_sequence_lengths()
            ctx.init_varlen_causal_lpt_order()
        else:
            ctx.init_causal_lpt_order()
            ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_descale()
        ctx.init_tile_bounds()

        gemm_helper = DualwaveFp8GemmHelper(ctx)
        softmax_helper = DualwaveFp8SoftmaxHelper(ctx)
        kv_gmem_to_lds = DualwaveFp8KvGmemToLdsLoader(ctx)
        kv_lds_to_regs = DualwaveFp8KvLdsToVgprLoader(ctx)
        output_store = DualwaveFp8StoreHelper(ctx)

        BN = traits.BLOCK_N
        D_CHUNKS = traits.D_CHUNKS
        NPF = traits.NUM_PREFETCH_K
        BOUNDED_MAX = (
            traits.HEAD_DIM == 128 and traits.DUALWAVE_SWP_LAZY_RESCALE and not traits.DUALWAVE_SWP_DEBUG_LAZY_COUNTS
        )
        t0 = ctx.split_t0
        t_end = ctx.split_t_end

        def _softmax_part(v_s, l_row, m_new):
            v_s = softmax_helper.sub_m(v_s, m_new)
            v_p = softmax_helper.exp2(v_s, 0, 16)
            v_p = softmax_helper.exp2(v_p, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = gemm_helper.cast_p_fp8_direct(v_p)
            return v_p, l_row

        def _subtile_tail(v_s, v_v, v_o, l_row, m_new):
            v_p, l_row = _softmax_part(v_s, l_row, m_new)
            # Keep the post-MFMA accumulators in SSA. Pinning them after each
            # subtile lengthens the paged schedule without reducing registers.
            v_o = gemm_helper.pv(v_p, v_v, v_o)
            return v_o, l_row

        def _correct_o(v_o, m_row, l_row, m_tile):
            if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                return softmax_helper.lazy_correct_o(v_o, m_row, l_row, m_tile)
            m_new, corr = softmax_helper.rescale_from_tile_max(m_row, m_tile)
            softmax_helper.scale_o(v_o, corr)
            return v_o, m_new, softmax_helper.apply_l_rescale(l_row, corr)

        def _merge_tile_max(v_s_a, v_s_b):
            m_tile = softmax_helper.reduce_max_pair(v_s_a, v_s_b)
            return softmax_helper.floor_masked_max(m_tile)

        page_t0, page_t1 = ctx.load_page_id_pair(t0 * BN)
        kv_gmem_to_lds.load_k(t0 * BN, t0 % NPF, page_id=page_t0)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        ctx.init_q_row()
        q_row = ctx.q_row

        q_wide = gemm_helper.load_q_wide()

        upper_bound = ctx.c_zero_f
        if const_expr(BOUNDED_MAX):
            # E4M3FN finite keys satisfy |K_i| <= 448. Bound every score by
            # 2 * 448 * sum(abs(Q_i)); factor two dominates FP32 summation error.
            max_key = fx.Vector.filled(8, 0x7E7E7E7E, fx.Int32)
            bound_tile = ctx.c_zero_v16f32
            for ws in range_constexpr(2):
                absolute_query = fx.Vector(q_wide[ws]) & fx.Int32(0x7F7F7F7F)
                bound_tile = gemm_helper._mfma_acc_fp8_wide(max_key, absolute_query, bound_tile)
            with fx.fastmath(None):
                upper_bound = fx.Float32(softmax_helper.anchor_scalar_f32(fx.Vector(bound_tile)[0] * fx.Float32(2.0)))

        page_t2, page_t3 = ctx.load_page_id_pair((t0 + 2) * BN)
        kv_gmem_to_lds.load_k((t0 + 1) * BN, (t0 + 1) % NPF, page_id=page_t1)
        kv_gmem_to_lds.load_v(t0 * BN, t0 % NPF, page_id=page_t0)
        kv_gmem_to_lds.load_v((t0 + 1) * BN, (t0 + 1) % NPF, page_id=page_t1)
        kv_gmem_to_lds.load_k((t0 + 2) * BN, (t0 + 2) % NPF, page_id=page_t2)
        kv_gmem_to_lds.load_k((t0 + 3) * BN, (t0 + 3) % NPF, page_id=page_t3)
        if const_expr(traits.FP8_PV_SEGMENTED):
            kv_gmem_to_lds.load_v((t0 + 2) * BN, (t0 + 2) % NPF, page_id=page_t2)
            kv_gmem_to_lds.load_v((t0 + 3) * BN, (t0 + 3) % NPF, page_id=page_t3)
        else:
            next_v_a = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
                (t0 + 2) * BN,
                page_id=page_t2,
            )
            next_v_b = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
                (t0 + 3) * BN,
                page_id=page_t3,
            )
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        m_row = ctx.c_neg_inf
        l_row = ctx.c_zero_f
        v_o = [ctx.c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]

        NPF_I = fx.Int64(NPF)

        def _ring_wrap(x):
            if const_expr(NPF == 8):
                return x & 7
            return (x >= NPF_I).select(x - NPF_I, x)

        init_args = [m_row, l_row] + v_o + [fx.Int64(t0) % NPF]
        if const_expr(not traits.FP8_PV_SEGMENTED):
            init_args += [next_v_a, next_v_b]
        loop_results = init_args
        next_v_arg_idx = 3 + D_CHUNKS

        def _iterate(j, loop_args, do_mask, skip_max=False, initialize=False):
            m_row = loop_args[0]
            l_row = loop_args[1]
            v_o = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]

            a_buf = fx.Int64(loop_args[2 + D_CHUNKS])
            b_buf = _ring_wrap(a_buf + 1)
            nn_a_buf = _ring_wrap(a_buf + 2)
            nn_b_buf = _ring_wrap(a_buf + 3)
            f_a_buf = _ring_wrap(a_buf + 4)
            f_b_buf = _ring_wrap(a_buf + 5)

            if const_expr(traits.FP8_PV_SEGMENTED):
                v_k_a = kv_lds_to_regs.load_k(a_buf)
                v_k_b = kv_lds_to_regs.load_k(b_buf)
                v_s_a = gemm_helper.qk(v_k_a, q_wide)
                v_s_b = gemm_helper.qk(v_k_b, q_wide)

                page_f_a, page_f_b = ctx.load_page_id_pair((j + 4) * BN)
                kv_gmem_to_lds.load_k((j + 4) * BN, f_a_buf, page_id=page_f_a)
                kv_gmem_to_lds.load_k((j + 5) * BN, f_b_buf, page_id=page_f_b)
                kv_gmem_to_lds.load_v((j + 4) * BN, f_a_buf, page_id=page_f_a)
                kv_gmem_to_lds.load_v((j + 5) * BN, f_b_buf, page_id=page_f_b)

                if const_expr(do_mask):
                    v_s_a, v_s_b = softmax_helper.causal_mask_pair_if_needed(v_s_a, v_s_b, j)
                m_tile = _merge_tile_max(v_s_a, v_s_b)
                v_o, m_new, l_row = _correct_o(v_o, m_row, l_row, m_tile)
                v_o = softmax_helper.anchor_v_o(v_o)

                v_p_a, l_row = _softmax_part(v_s_a, l_row, m_new)
                v_v_a = kv_lds_to_regs.load_v(a_buf)
                v_o = gemm_helper.pv(v_p_a, v_v_a, v_o)
                v_p_b, l_row = _softmax_part(v_s_b, l_row, m_new)
                v_v_b = kv_lds_to_regs.load_v(b_buf)
                v_o = gemm_helper.pv(v_p_b, v_v_b, v_o)
                m_row = m_new
                next_args = [m_row, l_row] + v_o + [nn_a_buf]
            else:
                next_v_a = loop_args[next_v_arg_idx]
                next_v_b = loop_args[next_v_arg_idx + 1]

                v_k_a = kv_lds_to_regs.load_k(a_buf)
                v_k_b = kv_lds_to_regs.load_k(b_buf)
                v_v_a = kv_lds_to_regs.load_v(a_buf)

                page_f_a, page_f_b = ctx.load_page_id_pair((j + 4) * BN)
                kv_gmem_to_lds.load_k((j + 4) * BN, f_a_buf, page_id=page_f_a)
                kv_gmem_to_lds.load_k((j + 5) * BN, f_b_buf, page_id=page_f_b)

                v_s_a = gemm_helper.qk(v_k_a, q_wide)
                kv_gmem_to_lds._store_v_fp8_vectorized_bankpad(next_v_a, nn_a_buf)
                v_f_a = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
                    (j + 4) * BN,
                    page_id=page_f_a,
                )
                v_f_b = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
                    (j + 5) * BN,
                    page_id=page_f_b,
                )
                v_s_b = gemm_helper.qk(v_k_b, q_wide)
                kv_gmem_to_lds._store_v_fp8_vectorized_bankpad(next_v_b, nn_b_buf)
                if const_expr(do_mask):
                    v_s_a, v_s_b = softmax_helper.causal_mask_pair_if_needed(v_s_a, v_s_b, j)
                m_new = m_row
                if const_expr(initialize):
                    # Zero O/l need no rescaling; avoid folding -inf through fast math.
                    with fx.fastmath(None):
                        m_new = _merge_tile_max(v_s_a, v_s_b)
                elif const_expr(not skip_max):
                    m_tile = _merge_tile_max(v_s_a, v_s_b)
                    v_o, m_new, l_row = _correct_o(v_o, m_row, l_row, m_tile)
                v_o = softmax_helper.anchor_v_o(v_o)

                v_o, l_row = _subtile_tail(v_s_a, v_v_a, v_o, l_row, m_new)
                v_v_b = kv_lds_to_regs.load_v(b_buf)
                v_o, l_row = _subtile_tail(v_s_b, v_v_b, v_o, l_row, m_new)
                m_row = m_new
                next_args = [m_row, l_row] + v_o + [nn_a_buf, v_f_a, v_f_b]

            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            return next_args

        if const_expr(BOUNDED_MAX):
            first_end = fx.min(fx.Int64(t_end), fx.Int64(t0) + 2)
            for j, loop_args in range(fx.Int64(t0), first_end, fx.Int64(2), init=init_args):
                next_args = _iterate(j, loop_args, True, initialize=True)
                loop_results = yield next_args
            sealed = _query_bound_is_safe(ctx, upper_bound, fx.Float32(loop_results[0]))
            fast_end = sealed.select(fx.Int64(t_end), first_end)
            first_state = loop_results
            # Waves can choose different loops, but pair order and barrier count agree.
            for j, loop_args in range(first_end, fast_end, fx.Int64(2), init=first_state):
                next_args = _iterate(j, loop_args, True, skip_max=True)
                loop_results = yield next_args
            slow_start = sealed.select(fx.Int64(t_end), first_end)
            slow_state = loop_results
            for j, loop_args in range(slow_start, fx.Int64(t_end), fx.Int64(2), init=slow_state):
                next_args = _iterate(j, loop_args, True)
                loop_results = yield next_args
        elif const_expr(traits.HEAD_DIM == 192 and traits.HEAD_DIM_V == 128):
            # The prefix boundary is wave-uniform, not CTA-uniform. Both loops
            # must keep the same pair order and one rendezvous per pair.
            prefix_end = fx.Int64(ctx.q_start_pos_i32 + ctx.delta_i32) // (2 * BN) * 2
            prefix_end = fx.min(fx.Int64(t_end), fx.max(fx.Int64(t0), prefix_end))
            for j, loop_args in range(fx.Int64(t0), prefix_end, fx.Int64(2), init=init_args):
                next_args = _iterate(j, loop_args, False)
                loop_results = yield next_args
            tail_init = loop_results
            for j, loop_args in range(prefix_end, fx.Int64(t_end), fx.Int64(2), init=tail_init):
                next_args = _iterate(j, loop_args, True)
                loop_results = yield next_args
        else:
            for j, loop_args in range(fx.Int64(t0), t_end, fx.Int64(2), init=init_args):
                next_args = _iterate(j, loop_args, True)
                loop_results = yield next_args
        m_row = loop_results[0]
        l_row = loop_results[1]
        v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]

        inv_l = softmax_helper.safe_l_inv(l_row)
        inv_l = inv_l * ctx.vd_fp8
        softmax_helper.scale_o(v_o, inv_l)
        rocdl.s_barrier()
        output_store.store_final_o(v_o, q_row)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_dualwave_swp_fp8_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_o_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
    ):
        ctx = DualwaveFp8KernelContext(
            traits,
            Q,
            K,
            V,
            O,
            DebugCounts,
            CuSeqQ,
            CuSeqKv,
            QDescale,
            KDescale,
            VDescale,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            stride_o_n=stride_o_n,
            BlockTable=BlockTable,
            block_table_stride=block_table_stride,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        if const_expr(traits.HEAD_DIM == 192 and traits.BATCH_INTERLEAVE_GROUP > 1):
            # Issue the longest active q-blocks first within each batch group.
            ctx.init_varlen_causal_lpt_order()
        ctx.init_descriptors()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_descale()
        ctx.init_tile_bounds()

        gemm_helper = DualwaveFp8GemmHelper(ctx)
        softmax_helper = DualwaveFp8SoftmaxHelper(ctx)
        kv_gmem_to_lds = DualwaveFp8KvGmemToLdsLoader(ctx)
        kv_lds_to_regs = DualwaveFp8KvLdsToVgprLoader(ctx)
        output_store = DualwaveFp8StoreHelper(ctx)

        # Skip packed-varlen q-blocks beyond this request's query length. The
        # condition is uniform across the workgroup, so barriers stay balanced.
        @flyc.jit
        def _run_q_block():
            kv_gmem_to_lds.load_k(ctx.split_t0 * traits.BLOCK_N, 0)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

            # Keep Q in registers; apply Q/K descales in softmax.
            ctx.init_q_row()
            q_row = ctx.q_row
            q_all_wide = gemm_helper.load_q_wide()

            kv_gmem_to_lds.load_k((ctx.split_t0 + 1) * traits.BLOCK_N, 1)
            kv_gmem_to_lds.load_v(ctx.split_t0 * traits.BLOCK_N, 0)
            v_k = kv_lds_to_regs.load_k(0)
            rocdl.sched_barrier(0)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _waitcnt_vm_n(ctx.NUM_DMA_V)

            # OPEN the wave-group phase shift: one extra s_barrier on group B
            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_one(ctx.stagger_i32)  # group B: +1 s_barrier -> open the shift
            else:
                rocdl.sched_barrier(0)
                rocdl.s_barrier()

            v_s_0 = gemm_helper.qk(v_k, q_all_wide)
            rocdl.sched_barrier(0)
            v_s_0 = softmax_helper.causal_mask_prologue_if_needed(v_s_0)
            m_row_pro = softmax_helper.reduce_max(v_s_0)
            # Floor fully-masked rows (-inf) to finite so exp2 yields 0, not NaN.
            m_row_pro = softmax_helper.floor_masked_max(m_row_pro)
            v_s_0 = softmax_helper.sub_m(v_s_0, m_row_pro)
            v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            kv_gmem_to_lds.load_k((ctx.split_t0 + 2) * traits.BLOCK_N, 0)

            l_row_init = ctx.c_zero_f
            init_args = [m_row_pro, l_row_init]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(ctx.c_zero_v16f32)
            init_args.append(ctx.v_pair_to_vec32(v_p_0))

            loop_lb = fx.Int64(3)
            loop_results = init_args
            for j, loop_args in range(
                loop_lb,
                ctx.split_t_end - 1,
                fx.Int64(2),
                init=init_args,
            ):
                m_row = loop_args[0]
                l_row = loop_args[1]
                v_o = [loop_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
                v_p_0 = ctx.v_vec32_to_pair(loop_args[2 + traits.D_CHUNKS])
                j_idx = j

                kv_gmem_to_lds.load_v((j_idx - 2) * traits.BLOCK_N, 1)
                v_k = kv_lds_to_regs.load_k(1)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                v_s_1 = gemm_helper.qk(v_k, q_all_wide)
                v_p_0 = softmax_helper.exp2(v_p_0, 16, 16)
                l_row = softmax_helper.reduce_sum(l_row, v_p_0)
                v_p_0 = softmax_helper.cast_p(v_p_0)
                v_p_0 = softmax_helper.anchor_v_p(v_p_0)
                _sched_barrier_exp_pairs(traits, 6, 3, 1)
                _sched_barrier_pairs(traits, 10, 5, 1)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                kv_gmem_to_lds.load_k(j_idx * traits.BLOCK_N, 1)
                v_v = kv_lds_to_regs.load_v(0)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    rocdl.s_setprio(1)
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_v, v_o)
                # Cross-length causal can put a diagonal tile in v_s_1; mask it here.
                v_s_1 = softmax_helper.causal_mask_prologue_if_needed(v_s_1, j_idx - 2, (j_idx - 1) * traits.BLOCK_N)
                m_tile_max_a = softmax_helper.reduce_max(v_s_1)

                _sched_barrier_pairs(traits, 4, 6, 2)

                if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                    v_o, m_row, l_row, v_p_0 = softmax_helper.lazy_rescale_o(v_o, m_row, l_row, m_tile_max_a, v_p_0)
                else:
                    v_o, m_row, l_row, v_p_0 = softmax_helper.rescale_o(v_o, m_row, l_row, m_tile_max_a, v_p_0)
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_v, v_o)
                if const_expr(traits.D_CHUNKS > 4):
                    v_s_1 = softmax_helper.sub_m(v_s_1, m_row)
                    v_p_1 = softmax_helper.exp2(v_s_1, 0, 16)
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    v_o = gemm_helper.pv_step_k(2, v_p_0, v_v, v_o)
                    v_o = gemm_helper.pv_step_k(3, v_p_0, v_v, v_o)
                else:
                    v_o = gemm_helper.pv_step_k(2, v_p_0, v_v, v_o)
                    v_o = gemm_helper.pv_step_k(3, v_p_0, v_v, v_o)
                    v_s_1 = softmax_helper.sub_m(v_s_1, m_row)
                    v_p_1 = softmax_helper.exp2(v_s_1, 0, 16)

                _sched_barrier_pairs(traits, 6, 6, 2)
                # Keep softmax EXP groups near their MFMA window.
                _sched_barrier_exp_pairs(traits, 6, 3, 2)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    rocdl.s_setprio(0)
                # Fence the closing priority/barrier pair at the cluster boundary.
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                kv_gmem_to_lds.load_v((j_idx - 1) * traits.BLOCK_N, 0)
                v_k = kv_lds_to_regs.load_k(0)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                v_s_0 = gemm_helper.qk(v_k, q_all_wide)
                v_p_1 = softmax_helper.exp2(v_p_1, 16, 16)
                l_row = softmax_helper.reduce_sum(l_row, v_p_1)
                v_p_1 = softmax_helper.cast_p(v_p_1)
                v_p_1 = softmax_helper.anchor_v_p(v_p_1)
                _sched_barrier_exp_pairs(traits, 6, 3, 3)
                _sched_barrier_pairs(traits, 10, 5, 3)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                kv_gmem_to_lds.load_k((j_idx + 1) * traits.BLOCK_N, 0)
                v_v = kv_lds_to_regs.load_v(1)
                v_s_0 = softmax_helper.causal_mask_prologue_if_needed(
                    v_s_0,
                    j_idx - 1,
                    j_idx * traits.BLOCK_N,
                )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    rocdl.s_setprio(1)
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_v, v_o)
                m_tile_max_b = softmax_helper.reduce_max(v_s_0)
                _sched_barrier_pairs(traits, 4, 6, 4)

                if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                    v_o, m_row, l_row, v_p_1 = softmax_helper.lazy_rescale_o(v_o, m_row, l_row, m_tile_max_b, v_p_1)
                else:
                    v_o, m_row, l_row, v_p_1 = softmax_helper.rescale_o(v_o, m_row, l_row, m_tile_max_b, v_p_1)
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_v, v_o)
                if const_expr(traits.D_CHUNKS > 4):
                    v_s_0 = softmax_helper.sub_m(v_s_0, m_row)
                    v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    v_o = gemm_helper.pv_step_k(2, v_p_1, v_v, v_o)
                    v_o = gemm_helper.pv_step_k(3, v_p_1, v_v, v_o)
                else:
                    v_o = gemm_helper.pv_step_k(2, v_p_1, v_v, v_o)
                    v_o = gemm_helper.pv_step_k(3, v_p_1, v_v, v_o)
                    v_s_0 = softmax_helper.sub_m(v_s_0, m_row)
                    v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 6, 5, 4)
                _sched_barrier_exp_pairs(traits, 6, 3, 4)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    rocdl.s_setprio(0)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                yield_args = [m_row, l_row] + v_o + [ctx.v_pair_to_vec32(v_p_0)]
                loop_results = yield yield_args

            # Drain the final three tiles without further prefetch-ahead.
            m_row = loop_results[0]
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_p_0 = ctx.v_vec32_to_pair(loop_results[2 + traits.D_CHUNKS])

            max_m3 = ctx.split_t_end - 3
            max_m2 = ctx.split_t_end - 2
            max_m1 = ctx.split_t_end - 1

            kv_gmem_to_lds.load_v(max_m3 * traits.BLOCK_N, 1)
            v_k = kv_lds_to_regs.load_k(1)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            v_s_1 = gemm_helper.qk(v_k, q_all_wide)
            v_p_0 = softmax_helper.exp2(v_p_0, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p_0)
            v_p_0 = softmax_helper.cast_p(v_p_0)
            v_p_0 = softmax_helper.anchor_v_p(v_p_0)
            _sched_barrier_exp_pairs(traits, 6, 3, 5)
            _sched_barrier_pairs(traits, 10, 5, 5)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            kv_gmem_to_lds.load_k(max_m1 * traits.BLOCK_N, 1)
            v_packs_e3 = kv_lds_to_regs.load_v(0)
            v_s_1 = softmax_helper.causal_mask_prologue_if_needed(
                v_s_1,
                max_m3,
                max_m2 * traits.BLOCK_N,
            )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(1)
            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_packs_e3, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_packs_e3, v_o)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                v_o = gemm_helper.pv_step_k(2, v_p_0, v_packs_e3, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_0, v_packs_e3, v_o)
            else:
                v_o = gemm_helper.pv(v_p_0, v_packs_e3, v_o)
            m_tile_max_e3 = softmax_helper.reduce_max(v_s_1)
            row_max_e3, rescale_e3 = softmax_helper.rescale_from_tile_max(m_row, m_tile_max_e3)
            m_row = row_max_e3
            v_s_1 = softmax_helper.sub_m(v_s_1, row_max_e3)
            v_p_1 = softmax_helper.exp2(v_s_1, 0, 16)
            _sched_barrier_pairs(traits, 10, 5, 6)
            _sched_barrier_exp_pairs(traits, 6, 3, 6)
            rocdl.sched_barrier(0)
            softmax_helper.scale_o(v_o, rescale_e3)
            v_o = softmax_helper.anchor_v_o(v_o)

            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            kv_gmem_to_lds.load_v(max_m2 * traits.BLOCK_N, 0)
            v_k = kv_lds_to_regs.load_k(0)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            v_s_0 = gemm_helper.qk(v_k, q_all_wide)
            l_row = softmax_helper.apply_l_rescale(l_row, rescale_e3)
            v_p_1 = softmax_helper.exp2(v_p_1, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p_1)
            v_p_1 = softmax_helper.cast_p(v_p_1)
            v_p_1 = softmax_helper.anchor_v_p(v_p_1)
            _sched_barrier_exp_pairs(traits, 6, 3, 7)
            _sched_barrier_pairs(traits, 10, 5, 7)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            v_packs_e7 = kv_lds_to_regs.load_v(1)
            v_s_0 = softmax_helper.causal_mask_prologue_if_needed(
                v_s_0,
                max_m2,
                max_m1 * traits.BLOCK_N,
            )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _waitcnt_vm_n(ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(1)
            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_packs_e7, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_packs_e7, v_o)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                v_o = gemm_helper.pv_step_k(2, v_p_1, v_packs_e7, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_1, v_packs_e7, v_o)
            else:
                v_o = gemm_helper.pv(v_p_1, v_packs_e7, v_o)
            m_tile_max_e7 = softmax_helper.reduce_max(v_s_0)
            row_max_e7, rescale_e7 = softmax_helper.rescale_from_tile_max(m_row, m_tile_max_e7)
            m_row = row_max_e7
            v_s_0 = softmax_helper.sub_m(v_s_0, row_max_e7)
            v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
            _sched_barrier_pairs(traits, 10, 5, 8)
            _sched_barrier_exp_pairs(traits, 6, 3, 8)
            rocdl.sched_barrier(0)
            softmax_helper.scale_o(v_o, rescale_e7)
            v_o = softmax_helper.anchor_v_o(v_o)
            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            kv_gmem_to_lds.load_v(max_m1 * traits.BLOCK_N, 1)
            v_k = kv_lds_to_regs.load_k(1)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _waitcnt_vm_n(ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            v_s_1 = gemm_helper.qk(v_k, q_all_wide)
            l_row = softmax_helper.apply_l_rescale(l_row, rescale_e7)
            v_p_0 = softmax_helper.exp2(v_p_0, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p_0)
            v_p_0 = softmax_helper.cast_p(v_p_0)
            v_p_0 = softmax_helper.anchor_v_p(v_p_0)
            _sched_barrier_exp_pairs(traits, 6, 3, 9)
            _sched_barrier_pairs(traits, 10, 5, 9)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            v_packs_e11 = kv_lds_to_regs.load_v(0)
            v_s_1 = softmax_helper.causal_mask_prologue_if_needed(
                v_s_1,
                max_m1,
                ctx.split_t_end * traits.BLOCK_N,
            )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _waitcnt_vm_n(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_packs_e11, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_packs_e11, v_o)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                v_o = gemm_helper.pv_step_k(2, v_p_0, v_packs_e11, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_0, v_packs_e11, v_o)
            else:
                v_o = gemm_helper.pv(v_p_0, v_packs_e11, v_o)
            m_tile_max_e11 = softmax_helper.reduce_max(v_s_1)
            row_max_e11, rescale_e11 = softmax_helper.rescale_from_tile_max(m_row, m_tile_max_e11)
            m_row = row_max_e11
            v_s_1 = softmax_helper.sub_m(v_s_1, row_max_e11)
            v_p_1 = softmax_helper.exp2(v_s_1, 0, 16)
            _sched_barrier_pairs(traits, 9, 6, 10)
            _sched_barrier_exp_pairs(traits, 7, 3, 10)
            rocdl.sched_barrier(0)
            v_p_1 = softmax_helper.exp2(v_p_1, 16, 16)
            l_row = softmax_helper.apply_l_rescale(l_row, rescale_e11)
            l_row = softmax_helper.reduce_sum(l_row, v_p_1)
            v_p_1 = softmax_helper.cast_p(v_p_1)
            v_p_1 = softmax_helper.anchor_v_p(v_p_1)
            rocdl.sched_barrier(0)
            softmax_helper.scale_o(v_o, rescale_e11)
            v_o = softmax_helper.anchor_v_o(v_o)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            v_packs_e13 = kv_lds_to_regs.load_v(1)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_packs_e13, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_packs_e13, v_o)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                v_o = gemm_helper.pv_step_k(2, v_p_1, v_packs_e13, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_1, v_packs_e13, v_o)
            else:
                v_o = gemm_helper.pv(v_p_1, v_packs_e13, v_o)

            # Apply V descale once after normalizing the native FP8 P*V result.
            inv_l = softmax_helper.safe_l_inv(l_row)
            inv_l = inv_l * ctx.vd_fp8
            softmax_helper.scale_o(v_o, inv_l)

            # Group A's extra barrier closes the prologue's phase shift before stores.
            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_zero(ctx.stagger_i32)  # group A: +1 s_barrier -> close the shift
            else:
                rocdl.s_barrier()

            # 128b stores fuse this lane and its half-wave partner, so each pair
            # covers 8 contiguous columns instead of two 64b stores.
            output_store.store_final_o_if_valid(v_o, q_row)

        if ctx.q_start < ctx.seqlen_q_v:
            _run_q_block()

    @flyc.jit
    def launch_flash_attn_dualwave_swp(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_o_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _dualwave_swp_fp8_cache_tag
        bs_idx = fx.Int64(batch_size)
        sl_idx = fx.Int64(seq_len)
        num_q_blocks = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_z = bs_idx

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        kernel_attrs = {
            "rocdl.waves_per_eu": waves_per_eu,
            "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
            "passthrough": passthrough_entries,
        }
        if const_expr(PAGED_BN128):
            flash_attn_paged_fp8_bn128_kernel(
                Q,
                K,
                V,
                O,
                DebugCounts,
                CuSeqQ,
                CuSeqKv,
                BlockTable,
                block_table_stride,
                QDescale,
                KDescale,
                VDescale,
                seq_len,
                seq_len_kv,
                stride_q_n,
                stride_kv_n,
                head_dim_runtime,
                value_attrs=kernel_attrs,
            ).launch(
                grid=(
                    NUM_HEADS_Q * BATCH_INTERLEAVE_GROUP,
                    num_q_blocks,
                    grid_z // BATCH_INTERLEAVE_GROUP,
                ),
                block=(BLOCK_SIZE, 1, 1),
                stream=stream,
            )
        else:
            flash_attn_dualwave_swp_fp8_gfx950_kernel(
                Q,
                K,
                V,
                O,
                DebugCounts,
                CuSeqQ,
                CuSeqKv,
                BlockTable,
                block_table_stride,
                QDescale,
                KDescale,
                VDescale,
                seq_len,
                seq_len_kv,
                stride_q_n,
                stride_o_n,
                stride_kv_n,
                head_dim_runtime,
                value_attrs=kernel_attrs,
            ).launch(
                grid=(
                    NUM_HEADS_Q * BATCH_INTERLEAVE_GROUP,
                    num_q_blocks,
                    grid_z // BATCH_INTERLEAVE_GROUP,
                ),
                block=(BLOCK_SIZE, 1, 1),
                stream=stream,
            )

    _dualwave_swp_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": True,
            "lsr-drop-solution": True,
            "disable-machine-sink": True,
        },
    }
    launch_flash_attn_dualwave_swp.compile_hints = dict(_dualwave_swp_compile_hints)

    def _validate_paged_bn128_launch(batch_size, seq_len_kv, block_table_stride):
        if not PAGED_BN128:
            return
        batch_size = int(batch_size)
        seq_len_kv = int(seq_len_kv)
        block_table_stride = int(block_table_stride)
        num_kv_pages = (seq_len_kv + traits.PAGE_SIZE - 1) // traits.PAGE_SIZE
        if (not PAGED_BN128_VARLEN and batch_size != 1) or num_kv_pages < 2 or num_kv_pages % 2 != 0:
            raise ValueError(
                "paged BN128 requires batch_size=1 unless compiled for packed varlen, "
                "and a positive even number "
                f"of KV pages; got batch_size={batch_size}, seq_len_kv={seq_len_kv}, "
                f"page_size={traits.PAGE_SIZE}"
            )
        if block_table_stride < num_kv_pages:
            raise ValueError(
                f"paged BN128 block table has too few entries: need {num_kv_pages}, got stride {block_table_stride}"
            )

    def _validate_batch_interleave_launch(batch_size):
        if int(batch_size) % BATCH_INTERLEAVE_GROUP != 0:
            raise ValueError(
                "paged FP8 batch size must be divisible by its interleave group: "
                f"batch_size={int(batch_size)}, group={BATCH_INTERLEAVE_GROUP}"
            )

    def _launch(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,
        stride_o_n=None,
        head_dim_runtime=None,
        debug_counts=None,
        *,
        seq_len_kv=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        stream=None,
        _compile_only=False,
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if stride_o_n is None:
            stride_o_n = DEFAULT_STRIDE_O_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        if seq_len_kv is None:
            seq_len_kv = seq_len
        if debug_counts is None:
            debug_counts = O
        # Non-varlen B=1 BN128 ignores the cu_seqlens slots; use O as a placeholder
        # for direct launcher calls that omit them.
        if cu_seqlens_q is None:
            cu_seqlens_q = O
        if cu_seqlens_kv is None:
            cu_seqlens_kv = O
        if block_table is None:
            block_table = O
        if block_table_stride is None:
            block_table_stride = 0
        if block_table is O and not _compile_only:
            raise ValueError("paged fp8 flash_attn requires block_table")
        _validate_paged_bn128_launch(batch_size, seq_len_kv, block_table_stride)
        _validate_batch_interleave_launch(batch_size)
        # Direct launcher calls must supply shape-[1] fp32 descales; O keeps the
        # compiled signature valid when a placeholder is needed.
        if q_descale is None:
            q_descale = O
        if k_descale is None:
            k_descale = O
        if v_descale is None:
            v_descale = O
        dispatch = flyc.compile if _compile_only else _run_compiled
        return dispatch(
            launch_flash_attn_dualwave_swp,
            Q,
            K,
            V,
            O,
            debug_counts,
            cu_seqlens_q,
            cu_seqlens_kv,
            block_table,
            block_table_stride,
            q_descale,
            k_descale,
            v_descale,
            batch_size,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_o_n,
            stride_kv_n,
            head_dim_runtime,
            fx.Stream(stream),
        )

    def _compile(*args, **kwargs):
        return _launch(*args, _compile_only=True, **kwargs)

    _launch.compile = _compile

    return _launch
