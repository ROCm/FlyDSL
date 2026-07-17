# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Generic f16/bf16 flash-attention kernel builder for FlyDSL.

The kernel keeps Q, softmax probabilities, and O accumulators in MFMA32 register
layout, streams K/V through LDS, and performs the online softmax over KV blocks.
Q/K/V/O are flattened from BSHD, with GQA/MQA represented by a smaller KV-head
count.

Launches one block per Q tile and head:
``grid_x = batch * ceil(seq_len / BLOCK_M) * num_heads``. Tail Q/KV tiles are
handled by buffer-resource bounds and softmax masking, so dense seq_len only
needs to be positive rather than a multiple of 128.

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len >= 1.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator
from kernels.attention.flash_attn_utils import (
    GenericFlashAttnContext,
    GenericGemmHelper,
    GenericKvGmemToLdsLoader,
    GenericKvLdsToVgprLoader,
    GenericPageIdLoader,
    GenericQLoader,
    GenericSoftmaxHelper,
    GenericStoreHelper,
    _make_flash_attn_generic_traits,
    _waitcnt_vm_n,
    scf_if_dispatch,
)


def build_flash_attn_func_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    num_kv_heads=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cross_seqlen=False,
    varlen=False,
    paged=False,
    kv_cache_layout="linear",
):
    """Build a generic f16/bf16 flash-attention launcher.

    ``num_heads`` is the Q/O head count. For GQA/MQA, pass a smaller
    ``num_kv_heads``; each group of ``num_heads // num_kv_heads`` Q heads shares
    one K/V head. Dense launches accept any positive ``seq_len``.
    """
    gpu_arch = get_hip_arch()

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    if dtype_str == "fp8":
        raise ValueError("generic flash_attn_func supports f16/bf16 only; fp8 is routed by flash_attn_interface")

    _validate_block_m = 128 if block_m is None else block_m
    _validate_fwg = flat_work_group_size
    if _validate_fwg is None:
        _validate_fwg = 256 if _validate_block_m <= 128 else 512
    _validate_num_waves = _validate_fwg // 64
    _validate_rows_per_wave = _validate_block_m // _validate_num_waves
    if path_tag.upper() in ("N32", "N128"):
        _validate_path = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        _validate_path = "N128"
    else:
        _validate_path = "N32"
    _validate_block_n_out = 128 if _validate_path == "N128" else 64
    _validate_has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    _validate_enable_dma = _validate_has_lds_load_b128 and (
        _validate_path == "N128" or (os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1")
    )

    assert _validate_rows_per_wave == 32, f"BLOCK_M/NUM_WAVES must be 32, got {_validate_rows_per_wave}"
    assert _validate_block_m % _validate_num_waves == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert _validate_fwg in (
        64,
        128,
        256,
        512,
    ), f"flat_work_group_size must be 64, 128, 256, or 512, got {_validate_fwg}"
    assert dtype_str in ("f16", "bf16"), "flash_attn_func only supports f16 and bf16"
    assert _validate_block_n_out % 64 == 0

    paged = bool(paged)
    if paged and kv_cache_layout not in ("linear", "vectorized"):
        raise NotImplementedError(
            f"generic paged kernel supports linear/vectorized kv_cache_layout, got {kv_cache_layout!r}"
        )
    if paged and _validate_enable_dma:
        raise NotImplementedError("generic paged kernel requires the non-DMA path (build with path_tag='N32')")

    traits = _make_flash_attn_generic_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        gpu_arch,
        causal=causal,
        dtype_str=dtype_str,
        flat_work_group_size=flat_work_group_size,
        block_m=block_m,
        path_tag=path_tag,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        paged=paged,
        kv_cache_layout=kv_cache_layout,
        waves_per_eu=waves_per_eu,
        daz=daz,
        fast_fp_math=fast_fp_math,
        unsafe_fp_math=unsafe_fp_math,
        sm_scale=sm_scale,
    )
    _flash_attn_generic_cache_tag = traits.cache_tag

    def _extract_seq_len(args, kwargs):
        """Return the launch-time seq_len as int, or None if not statically known."""
        S = args[5] if len(args) > 5 else kwargs.get("seq_len", None)
        try:
            return int(S)
        except (TypeError, ValueError):
            return None

    def _guard_seqlen(_dispatched):
        """Enforce the only correctness floor (seq_len >= 1). A symbolic/non-int
        seq_len is let through; dense routing is a perf policy, not a bound."""

        def _guarded(*args, **kwargs):
            S_int = _extract_seq_len(args, kwargs)
            if S_int is not None and S_int < 1:
                raise ValueError(f"flash_attn_func: seq_len must be >= 1, got {S_int}.")
            return _dispatched(*args, **kwargs)

        if hasattr(_dispatched, "compile"):
            _guarded.compile = _dispatched.compile
        return _guarded

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_func_smem_{traits.PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + traits.LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def flash_attn_generic_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _flash_attn_generic_cache_tag
        ctx = GenericFlashAttnContext(traits, K, V, seq_len, seq_len_kv, allocator, lds_kv_offset)
        ctx.init_types_and_pointers()
        gemm_helper = GenericGemmHelper(ctx)
        softmax_helper = GenericSoftmaxHelper(ctx)
        kv_gmem_to_lds = GenericKvGmemToLdsLoader(ctx)
        kv_lds_to_vgpr = GenericKvLdsToVgprLoader(ctx)
        q_loader = GenericQLoader(ctx)
        store_helper = GenericStoreHelper(ctx)

        ctx.init_sequence_indices()
        ctx.init_lds_view()
        ctx.init_thread_mapping()
        ctx.init_block_mapping()
        ctx.init_sequence_lengths(CuSeqQ, CuSeqKv)
        ctx.init_load_mapping()

        # Dense/varlen fold batch into raw K/V pointers; paged adds page_id per tile.
        ctx.init_kv_batch_pointers()
        if const_expr(traits.PAGED):
            kv_gmem_to_lds.page_ids = GenericPageIdLoader(ctx, BlockTable, block_table_stride)
        if const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
            kv_gmem_to_lds.init_dma_nomajor()

        # Per-batch Q/O descriptors, then K/V DMA resources when DMA is enabled.
        ctx.init_descriptors(Q, O)
        if const_expr(traits.ENABLE_DMA):
            kv_gmem_to_lds.init_dma()

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        ctx.init_q_row()
        q_b_packs = q_loader.load_all(ctx.q_rsrc, ctx.q_row)

        ctx.init_constants(sm_scale)
        ctx.init_kv_bounds()
        kv_upper = ctx.kv_upper

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF
        init_args = [ctx.c_neg_inf, ctx.c_zero_f]
        for _ in range_constexpr(traits.D_CHUNKS):
            init_args.append(ctx.c_zero_v16f32)
        if const_expr(_use_dma_dbuf):
            init_args.append(fx.Index(0))
            kv_gmem_to_lds.coop_dma_k(fx.Index(0), buf_id=0)

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(0, kv_upper, traits.BLOCK_N_OUT, init=init_args):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            _cur_buf_id = inner_iter_args[2 + traits.D_CHUNKS] if _use_dma_dbuf else None
            preload_k_count = traits.NUM_PREFETCH_K if traits.NUM_PREFETCH_K < traits.N_SUBTILES else traits.N_SUBTILES

            if const_expr(traits.ENABLE_PREFETCH_3BUF):
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = traits.CK_LDS_SEQ[pre_k % len(traits.CK_LDS_SEQ)] % traits.NUM_PREFETCH_K
                    pre_k_start = kv_block_start + pre_k * traits.BLOCK_N
                    if const_expr(traits.ENABLE_DMA):
                        kv_gmem_to_lds.coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        kv_gmem_to_lds.coop_load_k(pre_k_start, pre_k_slot)
                if const_expr(traits.ENABLE_DMA):
                    rocdl.s_waitcnt(0)
                else:
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            for kv_sub in range_constexpr(traits.N_SUBTILES):
                kv_start = kv_block_start + kv_sub * traits.BLOCK_N

                if const_expr(traits.ENABLE_PREFETCH_3BUF):
                    k_slot = traits.CK_LDS_SEQ[kv_sub % len(traits.CK_LDS_SEQ)] % traits.NUM_PREFETCH_K
                elif const_expr(_use_dma_dbuf):
                    if const_expr(kv_sub % 2 == 0):
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = fx.Index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = fx.Index(1) - _k_buf_id
                    if const_expr(kv_sub + 1 < traits.N_SUBTILES):
                        kv_gmem_to_lds.coop_dma_k(
                            kv_block_start + (kv_sub + 1) * traits.BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + fx.Index(traits.BLOCK_N_OUT)
                        _has_next = _next_kv < kv_upper

                        def _prefetch_next_k():
                            kv_gmem_to_lds.coop_dma_k(_next_kv, _next_k_buf_id)

                        scf_if_dispatch(_has_next, _prefetch_next_k)
                    rocdl.sched_barrier(0)
                    k_base = kv_gmem_to_lds.k_buf_base(_k_buf_id)
                else:
                    k_slot = 0
                    kv_gmem_to_lds.coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if const_expr(not _use_dma_dbuf):
                    k_base = kv_gmem_to_lds.k_buf_base(k_slot)

                if const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
                    kv_gmem_to_lds.coop_dma_v_nomajor(kv_start, 0)
                elif const_expr(not traits.USE_HW_TR or (not traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF)):
                    _v_vecs_prefetch = kv_gmem_to_lds.coop_load_v_global(kv_start)

                # ==== GEMM1: S = K @ Q^T. Bulk-read K packs, DMA-prefetch V, pipeline MFMAs ====
                _k_lo, _k_hi = kv_lds_to_vgpr.load_k_packs(k_base)
                if const_expr(traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF):
                    kv_gmem_to_lds.coop_dma_v(kv_start, 0)
                    rocdl.sched_barrier(0)
                s_acc_lo, s_acc_hi = gemm_helper.gemm1_accumulate(kv_lds_to_vgpr, _k_lo, _k_hi, q_b_packs)

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo, s_raw_hi = softmax_helper.split_scores(s_acc_lo, s_acc_hi)
                s_raw_lo, s_raw_hi = softmax_helper.apply_kv_mask(s_raw_lo, s_raw_hi, kv_start)
                m_new_raw, l_new, corr, p_vals_lo, p_vals_hi = softmax_helper.online_softmax(
                    m_running, l_running, s_raw_lo, s_raw_hi
                )

                # ==== Rescale O accumulators ====
                o_accs, corr_vec = softmax_helper.rescale_o_accs(o_accs, corr)

                if const_expr(traits.ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < traits.N_SUBTILES):
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * traits.BLOCK_N
                    next_k_slot = traits.CK_LDS_SEQ[next_k_sub % len(traits.CK_LDS_SEQ)] % traits.NUM_PREFETCH_K
                    if const_expr(traits.ENABLE_DMA):
                        kv_gmem_to_lds.coop_dma_k(next_k_start, next_k_slot)
                    else:
                        kv_gmem_to_lds.coop_load_k(next_k_start, next_k_slot)

                if const_expr(traits.ENABLE_PREFETCH_3BUF):
                    v_slot = traits.CK_LDS_SEQ[kv_sub % len(traits.CK_LDS_SEQ)] % traits.NUM_PREFETCH_V
                    v_base = kv_gmem_to_lds.v_buf_base(v_slot)
                    kv_gmem_to_lds.coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif const_expr(traits.ENABLE_DMA):
                    v_base = kv_gmem_to_lds.v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                elif const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
                    v_slot = 0
                    v_base = kv_gmem_to_lds.v_buf_base(v_slot)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = kv_gmem_to_lds.v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    kv_gmem_to_lds.coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()

                # ==== Build P packs, then GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi ====
                p_packs_lo = softmax_helper.build_p_packs(p_vals_lo)
                p_packs_hi = softmax_helper.build_p_packs(p_vals_hi)
                o_accs = gemm_helper.gemm2_pv(kv_lds_to_vgpr, o_accs, p_packs_lo, p_packs_hi, v_base, corr_vec)

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(_use_dma_dbuf):
                if const_expr(traits.N_SUBTILES % 2 == 1):
                    _yield_args.append(fx.Index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            loop_results = yield _yield_args

        # ---- Normalize and store O (128-bit buffer_store_dwordx4) ----
        store_helper.finalize_o(loop_results)

    @flyc.jit
    def launch_flash_attn_generic(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
        grid_x = bs_idx * num_q_tiles * traits.NUM_HEADS_Q

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_generic_kernel(
            Q,
            K,
            V,
            O,
            seq_len,
            seq_len_kv,
            CuSeqQ,
            CuSeqKv,
            BlockTable,
            block_table_stride,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{int(flat_work_group_size)},{int(flat_work_group_size)}"
                    if const_expr(flat_work_group_size is not None)
                    else None
                ),
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Best MI355X FMHA numbers were measured with ROCm/llvm-project `felix/tune_fmha`;
    # other LLVM revisions usually leave a few percent of peak throughput on the table.
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(
        Q,
        K,
        V,
        Out,
        batch_size,
        seq_len,
        *,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=0,
        seq_len_kv=None,
        stream=None,
    ):
        # Dense/non-paged pass the output tensor as a placeholder for the unused
        # cu_seqlens / block_table slots; the kernel only reads them under
        # const_expr(VARLEN) / const_expr(PAGED).
        cq = cu_seqlens_q if cu_seqlens_q is not None else Out
        ck = cu_seqlens_kv if cu_seqlens_kv is not None else Out
        bt = block_table if block_table is not None else Out
        skv = seq_len if seq_len_kv is None else seq_len_kv
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_generic(
                Q, K, V, Out, cq, ck, bt, block_table_stride, batch_size, seq_len, skv, fx.Stream(stream)
            )

    def _compile(Q, K, V, Out, batch_size, seq_len, seq_len_kv=None, stream=None):
        skv = seq_len if seq_len_kv is None else seq_len_kv
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flash_attn_generic, Q, K, V, Out, Out, Out, Out, 0, batch_size, seq_len, skv, fx.Stream(stream)
            )

    _launch.compile = _compile

    if block_m is None:
        return _guard_seqlen(_launch)
    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary
