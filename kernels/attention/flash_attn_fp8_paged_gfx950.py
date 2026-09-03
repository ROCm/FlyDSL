# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""gfx950 page-64 FP8 flash attention with independent value dimensions."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from kernels.attention.flash_attn_utils import (
    DualwaveFp8GemmHelper,
    DualwaveFp8KernelContext,
    DualwaveFp8KvGmemToLdsLoader,
    DualwaveFp8KvLdsToVgprLoader,
    DualwaveFp8QLoader,
    DualwaveFp8SoftmaxHelper,
    DualwaveFp8StoreHelper,
    DualwaveSplitKCombineContext,
    DualwaveSplitKCombineHelper,
    _make_dualwave_swp_fp8_traits,
    _sched_barrier_exp_pairs,
    _sched_barrier_pairs,
    _stagger_extra_barrier_if_one,
    _stagger_extra_barrier_if_zero,
    _waitcnt_vm_n,
    dualwave_splitk_workspace_elems,  # noqa: F401
)
from kernels.common.kernels_common import dtype_to_elem_type
from kernels.common.tensor_shim import _run_compiled


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
    fp8_pv_segmented=False,
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
    if batch_interleave_group > 1 and (paged_bn128 or head_dim != 192):
        raise ValueError("batch interleaving is supported only by the generic paged D192 kernel")
    if paged_bn128 and (head_dim, value_head_dim) != (128, 128):
        raise RuntimeError("paged BN128 currently requires head_dim=value_head_dim=128")
    assert num_heads % num_kv_heads == 0
    NUM_KV_SPLITS = int(num_kv_splits)
    assert NUM_KV_SPLITS >= 1
    if varlen and num_kv_splits and int(num_kv_splits) > 1:
        raise ValueError("varlen is not supported together with num_kv_splits > 1")

    # All compile-time tile/layout constants live in the fp8 traits object.
    traits = _make_dualwave_swp_fp8_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        value_head_dim=value_head_dim,
        causal=causal,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
        rescale_threshold=rescale_threshold,
        dualwave_swp_setprio=dualwave_swp_setprio,
        dualwave_swp_debug_lazy_counts=dualwave_swp_debug_lazy_counts,
        dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        num_kv_splits=num_kv_splits,
        varlen=varlen if paged_bn128_varlen else False if paged_bn128 else varlen,
        cross_seqlen=cross_seqlen,
        paged=paged,
        kv_cache_layout=kv_cache_layout,
        fp8_pv_segmented=fp8_pv_segmented,
        force_bn128=paged_bn128,
        batch_interleave_group=batch_interleave_group,
    )
    # Builder-level aliases used by SharedStorage and the launch/compile wrappers.
    SPLITK = traits.SPLITK
    BLOCK_M = traits.BLOCK_M
    BLOCK_SIZE = traits.BLOCK_SIZE
    HEAD_DIM = traits.HEAD_DIM
    NUM_HEADS_Q = traits.NUM_HEADS_Q
    PAGED = traits.PAGED
    PAGED_BN128 = bool(paged_bn128)
    PAGED_BN128_VARLEN = bool(paged_bn128_varlen)
    BATCH_INTERLEAVE_GROUP = traits.BATCH_INTERLEAVE_GROUP
    DEFAULT_STRIDE_Q_N = traits.DEFAULT_STRIDE_Q_N
    DEFAULT_STRIDE_O_N = traits.NUM_HEADS_Q * traits.V_HEAD_DIM
    DEFAULT_STRIDE_KV_N = traits.DEFAULT_STRIDE_KV_N
    _dualwave_swp_fp8_cache_tag = traits.cache_tag
    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    if PAGED_BN128:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
            vt: fx.Array[fx.BFloat16, traits.VT_BF16_TOTAL, 16]
            q: fx.Array[_lds_elem_dtype, BLOCK_M * HEAD_DIM, 16]

    else:

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
            if const_expr(traits.CAUSAL):
                num_q_blocks = (ctx.seqlen_q_v + traits.BLOCK_M - 1) // traits.BLOCK_M
                active_q_block = ctx.q_block_idx < num_q_blocks
                reversed_q_block = num_q_blocks - fx.Index(1) - ctx.q_block_idx
                ctx.q_block_idx = active_q_block.select(reversed_q_block, ctx.q_block_idx)
                ctx.q_start = ctx.q_block_idx * traits.BLOCK_M
                ctx.q_gmem_elem_offset = (
                    ctx.q_tok_base + ctx.q_start
                ) * ctx.stride_q_n_v + ctx.q_head_idx * traits.HEAD_DIM
        else:
            if const_expr(traits.CAUSAL):
                ctx.init_causal_lpt_order()
            ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_descale()
        ctx.init_tile_bounds()
        ctx.init_workspace_io()

        q_loader = DualwaveFp8QLoader(ctx)
        gemm_helper = DualwaveFp8GemmHelper(ctx)
        softmax_helper = DualwaveFp8SoftmaxHelper(ctx)
        kv_gmem_to_lds = DualwaveFp8KvGmemToLdsLoader(ctx)
        kv_lds_to_regs = DualwaveFp8KvLdsToVgprLoader(ctx)
        output_store = DualwaveFp8StoreHelper(ctx)

        BN = traits.BLOCK_N
        D_CHUNKS = traits.D_CHUNKS
        NPF = const_expr(traits.NUM_PREFETCH_K)
        t0 = ctx.split_t0
        t_end = ctx.split_t_end

        def _softmax_part(v_s, l_row, m_new):
            v_s = softmax_helper.sub_m(v_s, m_new)
            v_p = softmax_helper.exp2(v_s, 0, 16)
            v_p = softmax_helper.exp2(v_p, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = gemm_helper.cast_p_fp8_direct(v_p)
            return v_p, l_row

        def _pv_part(v_p, v_v, v_o):
            # Keep the post-MFMA accumulators in SSA. Pinning them after each
            # subtile lengthens the paged schedule without reducing registers.
            return gemm_helper.pv(v_p, v_v, v_o)

        def _subtile_tail(v_s, v_v, v_o, l_row, m_new):
            v_p, l_row = _softmax_part(v_s, l_row, m_new)
            v_o = _pv_part(v_p, v_v, v_o)
            return v_o, l_row

        def _mask_pair(v_s_a, v_s_b, j):
            if const_expr(traits.CAUSAL):
                return softmax_helper.causal_mask_pair_if_needed(v_s_a, v_s_b, j)
            return v_s_a, v_s_b

        def _correct_o(v_o, m_row, l_row, m_tile):
            if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                return softmax_helper.lazy_correct_o(v_o, m_row, l_row, m_tile)
            m_new, corr = softmax_helper.rescale_from_tile_max(m_row, m_tile)
            softmax_helper.scale_o(v_o, corr)
            return v_o, m_new, softmax_helper.apply_l_rescale(l_row, corr)

        def _merge_tile_max(v_s_a, v_s_b):
            m_tile = softmax_helper.max2(softmax_helper.reduce_max(v_s_a), softmax_helper.reduce_max(v_s_b))
            if const_expr(traits.CAUSAL):
                m_tile = softmax_helper.floor_masked_max(m_tile)
            return m_tile

        page_t0, page_t1 = ctx.load_page_id_pair(t0 * BN)
        kv_gmem_to_lds.load_k(t0 * BN, t0 % fx.Index(NPF), page_id=page_t0)
        if const_expr(traits.QREG):
            q_loader.stage_q_to_lds()
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        ctx.init_q_row()
        q_row = ctx.q_row

        q_wide = gemm_helper.load_q_wide() if const_expr(traits.QREG) else None

        page_t2, page_t3 = ctx.load_page_id_pair((t0 + 2) * BN)
        kv_gmem_to_lds.load_k((t0 + 1) * BN, (t0 + 1) % fx.Index(NPF), page_id=page_t1)
        kv_gmem_to_lds.load_v(t0 * BN, t0 % fx.Index(NPF), page_id=page_t0)
        kv_gmem_to_lds.load_v((t0 + 1) * BN, (t0 + 1) % fx.Index(NPF), page_id=page_t1)
        kv_gmem_to_lds.load_k((t0 + 2) * BN, (t0 + 2) % fx.Index(NPF), page_id=page_t2)
        kv_gmem_to_lds.load_k((t0 + 3) * BN, (t0 + 3) % fx.Index(NPF), page_id=page_t3)
        next_v_a = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
            (t0 + 2) * BN,
            page_id=page_t2,
        )
        next_v_b = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
            (t0 + 3) * BN,
            page_id=page_t3,
        )
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        m_row = ctx.c_neg_inf
        l_row = ctx.c_zero_f
        v_o = [ctx.c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]

        NPF_I = const_expr(fx.Index(NPF))

        def _ring_wrap(x):
            return (x >= NPF_I).select(x - NPF_I, x)

        init_args = [m_row, l_row] + v_o + [t0 % fx.Index(NPF), next_v_a, next_v_b]
        loop_results = init_args
        next_v_arg_idx = 3 + D_CHUNKS
        for j, loop_args in range(fx.Index(t0), t_end, fx.Index(2), init=init_args):
            m_row = loop_args[0]
            l_row = loop_args[1]
            v_o = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]

            a_buf = loop_args[2 + D_CHUNKS]
            b_buf = _ring_wrap(a_buf + fx.Index(1))
            nn_a_buf = _ring_wrap(a_buf + fx.Index(2))
            nn_b_buf = _ring_wrap(a_buf + fx.Index(3))
            f_a_buf = _ring_wrap(a_buf + fx.Index(4))
            f_b_buf = _ring_wrap(a_buf + fx.Index(5))

            next_v_a = loop_args[next_v_arg_idx]
            next_v_b = loop_args[next_v_arg_idx + 1]

            v_k_a = kv_lds_to_regs.load_k(a_buf)
            v_k_b = kv_lds_to_regs.load_k(b_buf)
            v_v_a = kv_lds_to_regs.load_v(a_buf)

            page_f_a, page_f_b = ctx.load_page_id_pair((j + fx.Index(4)) * BN)
            kv_gmem_to_lds.load_k((j + fx.Index(4)) * BN, f_a_buf, page_id=page_f_a)
            kv_gmem_to_lds.load_k((j + fx.Index(5)) * BN, f_b_buf, page_id=page_f_b)

            v_s_a = gemm_helper.qk(v_k_a, q_wide)
            kv_gmem_to_lds._store_v_fp8_vectorized_bankpad(next_v_a, nn_a_buf)
            v_f_a = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
                (j + fx.Index(4)) * BN,
                page_id=page_f_a,
            )
            v_f_b = kv_gmem_to_lds._load_v_fp8_vectorized_bankpad_source(
                (j + fx.Index(5)) * BN,
                page_id=page_f_b,
            )
            v_s_b = gemm_helper.qk(v_k_b, q_wide)
            kv_gmem_to_lds._store_v_fp8_vectorized_bankpad(next_v_b, nn_b_buf)
            v_s_a, v_s_b = _mask_pair(v_s_a, v_s_b, j)
            m_tile = _merge_tile_max(v_s_a, v_s_b)
            v_o, m_new, l_row = _correct_o(v_o, m_row, l_row, m_tile)
            v_o = softmax_helper.anchor_v_o(v_o)

            v_o, l_row = _subtile_tail(v_s_a, v_v_a, v_o, l_row, m_new)
            v_v_b = kv_lds_to_regs.load_v(b_buf)
            v_o, l_row = _subtile_tail(v_s_b, v_v_b, v_o, l_row, m_new)
            m_row = m_new

            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            next_args = [m_row, l_row] + v_o + [nn_a_buf, v_f_a, v_f_b]
            loop_results = yield next_args
        m_row = loop_results[0]
        l_row = loop_results[1]
        v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]

        inv_l_rcp = rocdl.rcp(T.f32, _raw(l_row))
        inv_l = ArithValue(fx.Float32(l_row) > ctx.c_zero_f).select(inv_l_rcp, ctx.c_zero_f)
        if const_expr(traits.FP8_PV):
            inv_l = ArithValue(inv_l) * ctx.vd_fp8
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
        # Per-kernel setup lives in the fp8 context; the inline pipeline helpers below
        # bind the ctx fields to local names so the schedule reads unchanged.
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
        if const_expr(traits.HEAD_DIM == 192 and traits.CAUSAL and traits.BATCH_INTERLEAVE_GROUP > 1):
            # Issue the longest active q-blocks first within each batch group.
            num_q_blocks = (ctx.seqlen_q_v + traits.BLOCK_M - 1) // traits.BLOCK_M
            active_q_block = ctx.q_block_idx < num_q_blocks
            reversed_q_block = num_q_blocks - fx.Index(1) - ctx.q_block_idx
            ctx.q_block_idx = active_q_block.select(reversed_q_block, ctx.q_block_idx)
            ctx.q_start = ctx.q_block_idx * traits.BLOCK_M
            ctx.q_gmem_elem_offset = (
                ctx.q_tok_base + ctx.q_start
            ) * ctx.stride_q_n_v + ctx.q_head_idx * traits.HEAD_DIM
        ctx.init_descriptors()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_descale()
        ctx.init_tile_bounds()
        ctx.init_workspace_io()

        # fp8 pipeline helpers (logic lives in flash_attn_utils; the kernel drives the
        # software-pipeline schedule below and calls into these).
        q_loader = DualwaveFp8QLoader(ctx)
        gemm_helper = DualwaveFp8GemmHelper(ctx)
        softmax_helper = DualwaveFp8SoftmaxHelper(ctx)
        kv_gmem_to_lds = DualwaveFp8KvGmemToLdsLoader(ctx)
        kv_lds_to_regs = DualwaveFp8KvLdsToVgprLoader(ctx)
        output_store = DualwaveFp8StoreHelper(ctx)

        # Skip packed-varlen q-blocks beyond this request's query length. The
        # condition is uniform across the workgroup, so barriers stay balanced.
        @flyc.jit
        def _run_q_block():
            # Prologue: load K tile split_t0 -> LDS buf0, wait, and sync the workgroup.
            kv_gmem_to_lds.load_k(ctx.split_t0 * traits.BLOCK_N, 0)
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

            # Load this wave's raw fp8 Q rows for wide QK; q/k descale is applied to
            # the fp32 logits after the MFMA. init_q_row sets q_row/q_row_i32/
            # q_start_pos_i32 on ctx for the causal-mask helpers.
            ctx.init_q_row()
            q_row = ctx.q_row
            q_all_wide = q_loader.load_all_wide(ctx.q_row_in_block)

            # Pipeline ahead: prefetch K tile1 (buf1) + V tile0 (buf0) as background
            kv_gmem_to_lds.load_k((ctx.split_t0 + 1) * traits.BLOCK_N, 1)
            kv_gmem_to_lds.load_v(ctx.split_t0 * traits.BLOCK_N, 0)
            v_k = kv_lds_to_regs.load_k(0)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_V)

            # OPEN the wave-group phase shift: one extra s_barrier on group B
            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_one(ctx.stagger_i32)  # group B: +1 s_barrier -> open the shift
            else:
                rocdl.sched_barrier(0)
                rocdl.s_barrier()

            # Prologue scores + first softmax pass for KV tile 0
            v_s_0 = gemm_helper.qk(v_k, q_all_wide)
            rocdl.sched_barrier(0)
            if const_expr(traits.CAUSAL):
                if const_expr(SPLITK):
                    v_s_0 = softmax_helper.causal_mask_prologue_if_needed(
                        v_s_0, ctx.split_t0, (ctx.split_t0 + 1) * traits.BLOCK_N
                    )
                else:
                    v_s_0 = softmax_helper.causal_mask_prologue_if_needed(v_s_0)
            else:
                # Non-causal padding mask for the prologue tile too: for tiny seq_len
                # tile 0 is the only real tile, so its keys >= seq_len must be masked
                # here. Gated -> no-op once tile 0 is full (seq_len >= BLOCK_N).
                if const_expr(SPLITK):
                    v_s_0 = softmax_helper.seq_pad_mask_if_needed(v_s_0, ctx.split_t0)
                else:
                    v_s_0 = softmax_helper.seq_pad_mask_if_needed(v_s_0)
            m_row_pro = softmax_helper.reduce_max(v_s_0)
            if const_expr(traits.CAUSAL):
                # Floor fully-masked rows (-inf) to finite so exp2 yields 0, not NaN.
                m_row_pro = softmax_helper.floor_masked_max(m_row_pro)
            v_s_0 = softmax_helper.sub_m(v_s_0, m_row_pro)
            v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Prefetch K tile 2 into buf0, keeping the K double-buffer one step ahead
            kv_gmem_to_lds.load_k((ctx.split_t0 + 2) * traits.BLOCK_N, 0)

            # Loop-carried state (scf.for init args): m_row, l_row(=0), D_CHUNKS zero
            l_row_init = ctx.c_zero_f
            init_args = [m_row_pro, l_row_init]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(ctx.c_zero_v16f32)
            init_args.append(ctx.v_pair_to_vec32(v_p_0))

            # ============================= Main loop =============================
            # Software-pipelined inner loop
            if const_expr(SPLITK):
                loop_lb = ctx.split_t0 + 3
            else:
                loop_lb = fx.Index(3)
            loop_results = init_args
            for j, loop_args in range(
                loop_lb,
                ctx.split_t_end - fx.Index(1),
                fx.Index(2),
                init=init_args,
            ):
                m_row = loop_args[0]
                l_row = loop_args[1]
                v_o = [loop_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
                v_p_0 = ctx.v_vec32_to_pair(loop_args[2 + traits.D_CHUNKS])
                j_idx = j

                # Cluster 0 (memory): prefetch next V (buf1), read resident K from LDS
                # (v_k) for MMA0, wait + sync.
                kv_gmem_to_lds.load_v((j_idx - 2) * traits.BLOCK_N, 1)
                v_k = kv_lds_to_regs.load_k(1)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                # Cluster 1 (compute): MMA0 -> v_s_1; finish v_p_0's 2nd-half exp2,
                # sum into l_row, cast to bf16 for P*V.
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

                # Cluster 2 (memory): prefetch next K (buf1), read this tile's V from
                # LDS (v_v) for P*V, wait + sync.
                kv_gmem_to_lds.load_k(j_idx * traits.BLOCK_N, 1)
                if const_expr(traits.D_CHUNKS > 4):
                    v_v = kv_lds_to_regs.load_v_steps(0, 0, 2)
                else:
                    v_v = kv_lds_to_regs.load_v(0)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                # Cluster 3 (compute): first P*V step + row max of v_s_1, lazy
                # rescale, remaining 3 P*V steps, sub row + 1st-half exp2 of v_s_1.
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    rocdl.s_setprio(1)
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_v, v_o)
                # Cross-length causal can put a diagonal tile in v_s_1; mask it here.
                # Self-attention skips this to keep the existing schedule.
                if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                    v_s_1 = softmax_helper.causal_mask_prologue_if_needed(
                        v_s_1, j_idx - 2, (j_idx - 1) * traits.BLOCK_N
                    )
                else:
                    v_s_1 = softmax_helper.v_s_vec_to_lists(v_s_1)
                m_tile_max_a = softmax_helper.reduce_max(v_s_1)

                _sched_barrier_pairs(traits, 4, 6, 2)

                if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                    v_o, m_row, l_row, v_p_0 = softmax_helper.lazy_rescale_o(v_o, m_row, l_row, m_tile_max_a, v_p_0)
                else:
                    v_o, m_row, l_row, v_p_0 = softmax_helper.rescale_o(v_o, m_row, l_row, m_tile_max_a, v_p_0)
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_v, v_o)
                if const_expr(traits.D_CHUNKS > 4):
                    v_v_tail = kv_lds_to_regs.load_v_steps(0, 2, 2)
                    v_s_1 = softmax_helper.sub_m(v_s_1, m_row)
                    v_p_1 = softmax_helper.exp2(v_s_1, 0, 16)
                    rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
                    v_o = gemm_helper.pv_step_k(2, v_p_0, v_v_tail, v_o)
                    v_o = gemm_helper.pv_step_k(3, v_p_0, v_v_tail, v_o)
                else:
                    v_o = gemm_helper.pv_step_k(2, v_p_0, v_v, v_o)
                    v_o = gemm_helper.pv_step_k(3, v_p_0, v_v, v_o)
                    v_s_1 = softmax_helper.sub_m(v_s_1, m_row)
                    v_p_1 = softmax_helper.exp2(v_s_1, 0, 16)

                _sched_barrier_pairs(traits, 6, 6, 2)
                # IGroupLP hint (group 2): 6 MFMA each paired with 3 EXP/TRANS (mask
                # 0x400) so the new softmax exp2 stays near its MFMA window.
                _sched_barrier_exp_pairs(traits, 6, 3, 2)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    rocdl.s_setprio(0)
                # sched_barrier(0): compiler scheduling fence (mask 0 = nothing
                # crosses), pinning s_setprio(0) and the closing s_barrier at the
                # cluster boundary. Emits no ISA; the real sync is s_barrier().
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                # Cluster 4 (memory, mirror of C0): prefetch V (buf0), read K from
                # buf0 into v_k, wait + sync.
                kv_gmem_to_lds.load_v((j_idx - 1) * traits.BLOCK_N, 0)
                v_k = kv_lds_to_regs.load_k(0)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                # Cluster 5 (compute, mirror of C1): MMA0 -> v_s_0; finish v_p_1's
                # 2nd-half exp2, sum into l_row, cast to bf16.
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

                # Cluster 6 (memory): prefetch next K (buf0), read V packs (buf1),
                # apply causal mask to v_s_0 (if causal), wait + sync.
                kv_gmem_to_lds.load_k((j_idx + 1) * traits.BLOCK_N, 0)
                if const_expr(traits.D_CHUNKS > 4):
                    v_packs_b = kv_lds_to_regs.load_v_steps(1, 0, 2)
                else:
                    v_packs_b = kv_lds_to_regs.load_v(1)
                if const_expr(traits.CAUSAL):
                    v_s_0 = softmax_helper.causal_mask_prologue_if_needed(
                        v_s_0,
                        j_idx - 1,
                        j_idx * traits.BLOCK_N,
                    )
                else:
                    v_s_0 = softmax_helper.v_s_vec_to_lists(v_s_0)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                # Cluster 7 (compute, mirror of C3 for v_p_1/v_s_0): closes the iter,
                # yield_args carries (m_row, l_row, v_o, packed v_p_0) to the next.
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    rocdl.s_setprio(1)
                v_v = v_packs_b
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_v, v_o)
                m_tile_max_b = softmax_helper.reduce_max(v_s_0)
                _sched_barrier_pairs(traits, 4, 6, 4)

                if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                    v_o, m_row, l_row, v_p_1 = softmax_helper.lazy_rescale_o(v_o, m_row, l_row, m_tile_max_b, v_p_1)
                else:
                    v_o, m_row, l_row, v_p_1 = softmax_helper.rescale_o(v_o, m_row, l_row, m_tile_max_b, v_p_1)
                v_v = v_packs_b
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_v, v_o)
                if const_expr(traits.D_CHUNKS > 4):
                    v_v_tail = kv_lds_to_regs.load_v_steps(1, 2, 2)
                    v_s_0 = softmax_helper.sub_m(v_s_0, m_row)
                    v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
                    rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
                    v_o = gemm_helper.pv_step_k(2, v_p_1, v_v_tail, v_o)
                    v_o = gemm_helper.pv_step_k(3, v_p_1, v_v_tail, v_o)
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

            # Epilogue: drain the pipeline for the final tiles the loop left in
            # flight. Mirrors the main-loop clusters but with no further
            # prefetch-ahead. Unpack the loop-carried state:
            m_row = loop_results[0]
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_p_0 = ctx.v_vec32_to_pair(loop_results[2 + traits.D_CHUNKS])

            # Tile indices for the last three tiles handled by the epilogue.
            max_m3 = ctx.split_t_end - 3
            max_m2 = ctx.split_t_end - 2
            max_m1 = ctx.split_t_end - 1

            # Epilogue C0 (memory): prefetch V max_m3 (buf1), read K from buf1, sync.
            kv_gmem_to_lds.load_v(max_m3 * traits.BLOCK_N, 1)
            v_k = kv_lds_to_regs.load_k(1)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Epilogue C1 (compute): MMA0 -> v_s_1; finish v_p_0 softmax (like C1).
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

            # Epilogue C2 (memory): prefetch K max_m1, read V packs (buf0), causal mask v_s_1, sync.
            kv_gmem_to_lds.load_k(max_m1 * traits.BLOCK_N, 1)
            if const_expr(traits.D_CHUNKS > 4):
                v_packs_e3 = kv_lds_to_regs.load_v_steps(0, 0, 2)
            else:
                v_packs_e3 = kv_lds_to_regs.load_v(0)
            if const_expr(traits.CAUSAL):
                v_s_1 = softmax_helper.causal_mask_prologue_if_needed(
                    v_s_1,
                    max_m3,
                    max_m2 * traits.BLOCK_N,
                )
            else:
                v_s_1 = softmax_helper.seq_pad_mask_if_needed(v_s_1, max_m3)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Epilogue C3 (compute): full P*V + unconditional rescale
            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(1)
            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_packs_e3, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_packs_e3, v_o)
                v_packs_e3 = kv_lds_to_regs.load_v_steps(0, 2, 2)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
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

            # Epilogue C4 (memory): prefetch V max_m2 (buf0), read K from buf0, sync.
            kv_gmem_to_lds.load_v(max_m2 * traits.BLOCK_N, 0)
            v_k = kv_lds_to_regs.load_k(0)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Epilogue C5 (compute): MMA0 -> v_s_0; fold rescale_e3 into l_row, finish
            # v_p_1 softmax.
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

            # Epilogue C6 (memory): read V packs (buf1), causal mask v_s_0, sync.
            if const_expr(traits.D_CHUNKS > 4):
                v_packs_e7 = kv_lds_to_regs.load_v_steps(1, 0, 2)
            else:
                v_packs_e7 = kv_lds_to_regs.load_v(1)
            if const_expr(traits.CAUSAL):
                v_s_0 = softmax_helper.causal_mask_prologue_if_needed(
                    v_s_0,
                    max_m2,
                    max_m1 * traits.BLOCK_N,
                )
            else:
                v_s_0 = softmax_helper.seq_pad_mask_if_needed(v_s_0, max_m2)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Epilogue C7 (compute, mirror of C3): full P*V + unconditional rescale.
            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                rocdl.s_setprio(1)
            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_packs_e7, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_packs_e7, v_o)
                v_packs_e7 = kv_lds_to_regs.load_v_steps(1, 2, 2)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
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

            # Epilogue C8 (memory): prefetch V max_m1 (buf1), read K from buf1, sync.
            kv_gmem_to_lds.load_v(max_m1 * traits.BLOCK_N, 1)
            v_k = kv_lds_to_regs.load_k(1)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_V)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Epilogue C9 (compute): MMA0 -> v_s_1 (last tile); fold rescale_e7 into
            # l_row, finish v_p_0 softmax.
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

            # Epilogue C10 (memory): read last V packs (buf0), causal mask v_s_1,
            # drain all DMAs (vmcnt 0), sync.
            if const_expr(traits.D_CHUNKS > 4):
                v_packs_e11 = kv_lds_to_regs.load_v_steps(0, 0, 2)
            else:
                v_packs_e11 = kv_lds_to_regs.load_v(0)
            if const_expr(traits.CAUSAL):
                v_s_1 = softmax_helper.causal_mask_prologue_if_needed(
                    v_s_1,
                    max_m1,
                    ctx.split_t_end * traits.BLOCK_N,
                )
            else:
                v_s_1 = softmax_helper.seq_pad_mask_if_needed(v_s_1, max_m1)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Epilogue C11 (compute): full P*V + rescale for v_p_0, then complete the
            # last tile's softmax in-place (both exp2 halves, sum, cast) since no
            # further pass follows.
            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_packs_e11, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_packs_e11, v_o)
                v_packs_e11 = kv_lds_to_regs.load_v_steps(0, 2, 2)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
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

            # Epilogue C12 (memory): read the final V packs for the closing P*V.
            if const_expr(traits.D_CHUNKS > 4):
                v_packs_e13 = kv_lds_to_regs.load_v_steps(1, 0, 2)
            else:
                v_packs_e13 = kv_lds_to_regs.load_v(1)
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # Epilogue C13 (compute): final P*V -> v_o holds the unnormalized output.
            if const_expr(traits.D_CHUNKS > 4):
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_packs_e13, v_o)
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_packs_e13, v_o)
                v_packs_e13 = kv_lds_to_regs.load_v_steps(1, 2, 2)
                rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
                v_o = gemm_helper.pv_step_k(2, v_p_1, v_packs_e13, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_1, v_packs_e13, v_o)
            else:
                v_o = gemm_helper.pv(v_p_1, v_packs_e13, v_o)

            # Normalize by l_row; zero rows become zero instead of NaN.
            # Split-K normalizes before packing so O_partial keeps useful mantissa
            # range; the combine kernel later applies w_s*l_s.
            # HIPREC folds v_descale into the bf16 vt scratch. The direct FP8
            # D128 path keeps raw V and applies its descale once at the end.
            inv_l_rcp = rocdl.rcp(T.f32, _raw(l_row))
            inv_l = ArithValue(fx.Float32(l_row) > ctx.c_zero_f).select(inv_l_rcp, ctx.c_zero_f)
            if const_expr(traits.FP8_PV):
                inv_l = ArithValue(inv_l) * ctx.vd_fp8
            softmax_helper.scale_o(v_o, inv_l)

            # CLOSE the phase shift: one extra s_barrier on group A (complement of
            # the prologue's group-B barrier) realigns the two groups before the
            # store. Disabled -> one plain barrier.
            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_zero(ctx.stagger_i32)  # group A: +1 s_barrier -> close the shift
            else:
                rocdl.s_barrier()

            # 128b stores fuse this lane and its half-wave partner, so each pair
            # covers 8 contiguous columns instead of two 64b stores.
            if const_expr(not SPLITK):
                if const_expr(traits.VARLEN):
                    output_store.store_final_o_if_valid(v_o, q_row)
                else:
                    output_store.store_final_o(v_o, q_row)
            else:
                output_store.store_splitk_partial_o(v_o, m_row, l_row, q_row)

        if ctx.q_start < ctx.seqlen_q_v:
            _run_q_block()

        if const_expr(SPLITK):
            output_store.store_empty_split()

    # Combine kernel: out = sum_s w_s * O_s / sum_s w_s * l_s, w_s = exp2(m_s - m_max).
    # One wave row of 32 lanes covers a (b, h, s) row, 4 contiguous cols/lane.
    COMBINE_BLOCK = 256
    COMBINE_LANES_PER_ROW = traits.HEAD_DIM // 4
    COMBINE_ROWS_PER_BLOCK = COMBINE_BLOCK // COMBINE_LANES_PER_ROW

    @flyc.kernel(known_block_size=[COMBINE_BLOCK, 1, 1])
    def flash_attn_splitk_combine_kernel(
        O: fx.Tensor,  # noqa: E741
        WS: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
    ):
        ctx = DualwaveSplitKCombineContext(traits, O, WS, batch_size, seq_len, stride_q_n)
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_thread_mapping(COMBINE_ROWS_PER_BLOCK, COMBINE_LANES_PER_ROW)
        ctx.init_workspace()
        ctx.init_descriptors()

        combine = DualwaveSplitKCombineHelper(ctx)
        m_s, l_s = combine.load_ml_rows()
        m_max = combine.reduce_m_max(m_s)
        acc, den = combine.accumulate_splits(m_s, l_s, m_max)
        o_pack = combine.pack_output(acc, den)
        combine.store_output(o_pack)

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
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + BLOCK_M - 1) // BLOCK_M
        if const_expr(SPLITK):
            grid_z = bs_idx * NUM_KV_SPLITS
        else:
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
                grid=(NUM_HEADS_Q, num_q_blocks, grid_z),
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
                    (
                        NUM_HEADS_Q * traits.BATCH_INTERLEAVE_GROUP,
                        num_q_blocks,
                        grid_z // traits.BATCH_INTERLEAVE_GROUP,
                    )
                    if const_expr(traits.BATCH_INTERLEAVE_GROUP > 1)
                    else (NUM_HEADS_Q, num_q_blocks, grid_z)
                ),
                block=(BLOCK_SIZE, 1, 1),
                stream=stream,
            )
        if const_expr(SPLITK):
            combine_rows = bs_idx * NUM_HEADS_Q * sl_idx
            flash_attn_splitk_combine_kernel(O, DebugCounts, batch_size, seq_len, stride_o_n).launch(
                grid=(combine_rows // COMBINE_ROWS_PER_BLOCK, 1, 1),
                block=(COMBINE_BLOCK, 1, 1),
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
                "paged BN128 block table has too few entries: " f"need {num_kv_pages}, got stride {block_table_stride}"
            )

    def _validate_batch_interleave_launch(batch_size):
        if int(batch_size) % BATCH_INTERLEAVE_GROUP != 0:
            raise ValueError(
                "paged D192 batch size must be divisible by its interleave group: "
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
        workspace=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        stream=None,
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if stride_o_n is None:
            stride_o_n = DEFAULT_STRIDE_O_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        # seq_len_kv defaults to seq_len (self-attention / equal Q,KV lengths).
        if seq_len_kv is None:
            seq_len_kv = seq_len
        if SPLITK:
            if workspace is None:
                raise ValueError("num_kv_splits > 1 requires a fp32 workspace (see dualwave_splitk_workspace_elems)")
            debug_counts = workspace
        if debug_counts is None:
            debug_counts = O
        # Dense launches still pass valid tensors for the (unused) cu_seqlens slots;
        # the kernel only reads them under const_expr(VARLEN). Use O as a placeholder.
        if cu_seqlens_q is None:
            cu_seqlens_q = O
        if cu_seqlens_kv is None:
            cu_seqlens_kv = O
        if block_table is None:
            block_table = O
        if block_table_stride is None:
            block_table_stride = 0
        if PAGED and block_table is O:
            raise ValueError("paged fp8 flash_attn requires block_table")
        _validate_paged_bn128_launch(batch_size, seq_len_kv, block_table_stride)
        _validate_batch_interleave_launch(batch_size)
        # Per-tensor fp8 descales (shape-[1] fp32). The kernel only reads them on
        # the fp8 path; bf16/f16 launches pass O as an unused placeholder.
        if q_descale is None:
            q_descale = O
        if k_descale is None:
            k_descale = O
        if v_descale is None:
            v_descale = O
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return _run_compiled(
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

    def _compile(
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
        workspace=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        stream=None,
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
        if SPLITK:
            if workspace is None:
                raise ValueError("num_kv_splits > 1 requires a fp32 workspace (see dualwave_splitk_workspace_elems)")
            debug_counts = workspace
        if debug_counts is None:
            debug_counts = O
        if cu_seqlens_q is None:
            cu_seqlens_q = O
        if cu_seqlens_kv is None:
            cu_seqlens_kv = O
        if block_table is None:
            block_table = O
        if block_table_stride is None:
            block_table_stride = 0
        _validate_paged_bn128_launch(batch_size, seq_len_kv, block_table_stride)
        _validate_batch_interleave_launch(batch_size)
        if q_descale is None:
            q_descale = O
        if k_descale is None:
            k_descale = O
        if v_descale is None:
            v_descale = O
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return flyc.compile(
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

    _launch.compile = _compile

    return _launch
