# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""flash_attn_func kernel builder for FlyDSL.

- True MFMA32 remap: `mfma_f32_32x32x16bf16` / `mfma_f32_32x32x16f16` for both GEMM stages.
- Tile shape: BLOCK_M=128 or 256 (auto-selected), BLOCK_N=64.
- BLOCK_M=128: 4 waves (256 threads), BLOCK_M=256: 8 waves (512 threads).
- Per-wave Q rows: 32.
- GEMM1 uses `K @ Q^T` so S/P live in MFMA32 register layout.
- Online softmax over KV dimension is done in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS roundtrip.
- K and V use separate LDS regions with DMA-to-LDS prefetch and XOR swizzle.
- For H>=32, both M=128 and M=256 variants are built and dispatched at runtime.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) or (512,) depending on BLOCK_M.

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len % 128 == 0.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
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
    """Build the flash_attn_func launcher using the post-refactor FlyDSL API.

    For GQA/MQA pass ``num_kv_heads < num_heads``. ``num_heads`` is the Q head
    count, ``num_kv_heads`` is the KV head count, and we require
    ``num_heads % num_kv_heads == 0``. Default ``num_kv_heads = num_heads`` (MHA).
    Q/O still have ``num_heads`` heads; K/V have ``num_kv_heads`` heads, with
    every ``num_heads // num_kv_heads`` consecutive Q heads sharing one KV head.
    """
    gpu_arch = get_rocm_arch()

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
    )

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
        ctx = GenericFlashAttnContext(traits, K, V, seq_len, seq_len_kv, allocator, lds_kv_offset)
        ctx.init_types_and_pointers()
        elem_dtype = ctx.elem_dtype
        elem_type = ctx.elem_type
        k_ptr = ctx.k_ptr
        v_ptr = ctx.v_ptr
        fm_fast = ctx.fm_fast
        v4f16_type = ctx.v4f16_type
        mfma_pack_type = ctx.mfma_pack_type
        MFMA_LANE_K = ctx.MFMA_LANE_K
        gemm_helper = GenericGemmHelper(ctx)
        softmax_helper = GenericSoftmaxHelper(ctx, gemm_helper)
        kv_gmem_to_lds = GenericKvGmemToLdsLoader(ctx)
        kv_lds_to_vgpr = GenericKvLdsToVgprLoader(ctx)
        q_loader = GenericQLoader(ctx)
        store_helper = GenericStoreHelper(ctx)
        _fadd = gemm_helper.fadd
        _fsub = gemm_helper.fsub
        _fmul = gemm_helper.fmul
        _fmax = gemm_helper.fmax
        mfma_acc = gemm_helper.mfma_acc

        ctx.init_sequence_indices()
        ctx.init_lds_view()
        ctx.init_thread_mapping()
        ctx.init_block_mapping()

        lds_kv = ctx.lds_kv
        tid = ctx.tid
        wave_id = ctx.wave_id
        lane = ctx.lane
        lane_mod_32 = ctx.lane_mod_32
        lane_div_32 = ctx.lane_div_32
        tr_k_group = ctx.tr_k_group
        tr_col_sub = ctx.tr_col_sub
        tr_col_half = ctx.tr_col_half
        ds_read_tr_v4f16 = kv_lds_to_vgpr.ds_read_tr_v4f16

        wave_q_offset = ctx.wave_q_offset
        q_start = ctx.q_start
        kv_head_idx = ctx.kv_head_idx

        ctx.init_sequence_lengths(CuSeqQ, CuSeqKv)
        ctx.init_load_mapping()
        q_tok_base = ctx.q_tok_base
        kv_tok_base = ctx.kv_tok_base
        seqlen_q_b = ctx.seqlen_q_b
        seqlen_kv_b = ctx.seqlen_kv_b

        # Dense/varlen fold batch into raw K/V pointers; paged adds page_id per tile.
        if const_expr(not traits.PAGED):
            _kv_ptr_batch_off = kv_tok_base * fx.Index(traits.STRIDE_TOKEN_KV)
            k_ptr = buffer_ops.get_element_ptr(k_ptr, _kv_ptr_batch_off, elem_type=elem_type)
            v_ptr = buffer_ops.get_element_ptr(v_ptr, _kv_ptr_batch_off, elem_type=elem_type)

        if const_expr(traits.PAGED):
            page_ids = GenericPageIdLoader(ctx, BlockTable, block_table_stride)

            def _paged_page_id(tile_start):
                return page_ids.page_id(tile_start)

            if const_expr(traits.KV_VECTORIZED):

                def _vk_load(page_id, kv_row, col):
                    return kv_gmem_to_lds.load_vectorized_k(k_ptr, page_id, kv_row, col)

                def _vv_load(page_id, kv_row, col):
                    return kv_gmem_to_lds.load_vectorized_v(v_ptr, page_id, kv_row, col)

        load_row_in_batch = ctx.load_row_in_batch
        load_col_base = ctx.load_col_base

        # ---- Helper: global flat indices ----
        # Token indices are 0-based because the descriptor or raw pointer folds batch base.
        global_idx_kv = kv_gmem_to_lds.global_idx
        _kv_row_clamp = kv_gmem_to_lds.row_clamp
        _load_global_half_vec = kv_gmem_to_lds.load_half_vec
        load_global_f16xN = kv_gmem_to_lds.load_f16xN
        k_buf_base = kv_gmem_to_lds.k_buf_base
        v_buf_base = kv_gmem_to_lds.v_buf_base

        # ---- K XOR swizzle: col ^ ((row & 7) << 4) at 16-element granularity ----
        _k_swizzle = kv_gmem_to_lds.k_swizzle
        _sigma_kv = kv_gmem_to_lds.sigma_kv

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            if const_expr(traits.PAGED):
                _pid_k = _paged_page_id(tile_start)
            for batch in range_constexpr(traits.NUM_BATCHES_KV):
                row_offset = batch * traits.ROWS_PER_BATCH_LOAD
                if const_expr(traits.PAGED):
                    row_idx = _pid_k * fx.Index(traits.PAGE_SIZE) + load_row_in_batch + row_offset
                else:
                    row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(traits.KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(traits.BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * traits.K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        Vec(vec).store(lds_kv, [lds_idx])
                else:
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * traits.K_STRIDE + swz_col
                    if const_expr(traits.KV_VECTORIZED):
                        vec = _vk_load(_pid_k, _sigma_kv(lds_row), load_col_base)
                    else:
                        vec = load_global_f16xN(k_ptr, global_idx_kv(row_idx, load_col_base))
                    Vec(vec).store(lds_kv, [lds_idx])

        # ---- Cooperative V load ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * traits.V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(traits.VEC_WIDTH):
                elem = Vec(vec)[_e]
                vt_d = load_col_base + _e
                vt_idx = v_base + vt_d * traits.VT_STRIDE + lds_row
                v1 = Vec.from_elements([elem], elem_dtype)
                v1.store(lds_kv, [vt_idx])

        _v_store_to_lds = _v_store_row_major if traits.USE_HW_TR else _v_store_transposed

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            if const_expr(traits.PAGED):
                _pid_v = _paged_page_id(tile_start)
            for batch in range_constexpr(traits.NUM_BATCHES_KV):
                row_offset = batch * traits.ROWS_PER_BATCH_LOAD
                if const_expr(traits.PAGED):
                    row_idx = _pid_v * fx.Index(traits.PAGE_SIZE) + load_row_in_batch + row_offset
                else:
                    row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                if const_expr(traits.KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(traits.BLOCK_N)
                    if row_valid:
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                else:
                    lds_row = load_row_in_batch + row_offset
                    if const_expr(traits.KV_VECTORIZED):
                        vec = _vv_load(_pid_v, _sigma_kv(lds_row), load_col_base)
                    else:
                        vec = load_global_f16xN(v_ptr, global_idx_kv(row_idx, load_col_base))
                    _v_store_to_lds(v_base, lds_row, vec)

        def coop_load_v_global(tile_start):
            """Issue global V loads; vectorized mode returns no-major v8 rows."""
            if const_expr(traits.KV_VECTORIZED):
                _pid_vg = _paged_page_id(tile_start)
                _base_v = _pid_vg * fx.Index(traits.PAGE_STRIDE_VEC) + kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
                vecs = []
                for j in range_constexpr(traits.NV8_PER_THREAD):
                    _flat = tid + fx.Index(j * traits.BLOCK_SIZE)
                    _d = _flat // fx.Index(traits.VEC_V_NGROUPS)
                    _ng = _flat % fx.Index(traits.VEC_V_NGROUPS)
                    _src = (
                        _base_v
                        + _ng * fx.Index(traits.HEAD_DIM * traits.KV_VEC_SIZE)
                        + _d * fx.Index(traits.KV_VEC_SIZE)
                    )
                    vecs.append(_load_global_half_vec(v_ptr, _src, traits.KV_VEC_SIZE))
                return vecs
            vecs = []
            if const_expr(traits.PAGED):
                _pid_vg = _paged_page_id(tile_start)
            for batch in range_constexpr(traits.NUM_BATCHES_KV):
                row_offset = batch * traits.ROWS_PER_BATCH_LOAD
                if const_expr(traits.PAGED):
                    row_idx = _pid_vg * fx.Index(traits.PAGE_SIZE) + load_row_in_batch + row_offset
                else:
                    row_idx = _kv_row_clamp(tile_start + load_row_in_batch + row_offset)
                vecs.append(load_global_f16xN(v_ptr, global_idx_kv(row_idx, load_col_base)))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            """Write V vectors to LDS; vectorized mode uses no-major rows."""
            v_base = v_buf_base(buf_id)
            if const_expr(traits.KV_VECTORIZED):
                for j in range_constexpr(traits.NV8_PER_THREAD):
                    _flat = tid + fx.Index(j * traits.BLOCK_SIZE)
                    _d = _flat // fx.Index(traits.VEC_V_NGROUPS)
                    _ng = _flat % fx.Index(traits.VEC_V_NGROUPS)
                    _dst = (
                        v_base
                        + (_d // fx.Index(8)) * fx.Index(traits.VEC_V_LINE)
                        + _ng * fx.Index(traits.VEC_V_D128)
                        + (_d % fx.Index(8)) * fx.Index(8)
                    )
                    Vec(vecs[j]).store(lds_kv, [_dst])
                return
            for batch in range_constexpr(traits.NUM_BATCHES_KV):
                row_offset = batch * traits.ROWS_PER_BATCH_LOAD
                if const_expr(traits.KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(traits.BLOCK_N)
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vecs[batch])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vecs[batch])

        # ---- KV_VECTORIZED V: no-major GM->LDS DMA ----
        if const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
            _v_dma_base_i64 = fx.Int64(buffer_ops.extract_base_index(V, address_space=1))
            _v_dma_page_bytes = fx.Int64(traits.PAGE_STRIDE_VEC * 2)
            _v_dma_lds_base = buffer_ops.extract_base_index(lds_kv, address_space=3)
            _v_dma_sz = fx.Int32(16)
            _v_dma_z = fx.Int32(0)
            _v_dma_aux = fx.Int32(1)
            _v_dma_no = lane // fx.Index(8)
            _v_dma_dloc = lane % fx.Index(8)

            def coop_dma_v_nomajor(tile_start, buf_id=0):
                # Each lane DMAs one contiguous v8 into the no-major V line.
                _pid = _paged_page_id(tile_start)
                _paddr = as_mlir_value(_v_dma_base_i64 + fx.Int64(_pid) * _v_dma_page_bytes)
                _rsrc = buffer_ops.create_buffer_resource_from_addr(
                    _paddr, num_records_bytes=as_mlir_value(_v_dma_page_bytes)
                )
                if const_expr(isinstance(buf_id, int)):
                    _vb = _v_dma_lds_base + fx.Index((traits.LDS_V_BASE + buf_id * traits.LDS_V_TILE_SIZE) * 2)
                else:
                    _vb = (
                        _v_dma_lds_base
                        + fx.Index(traits.LDS_V_BASE * 2)
                        + buf_id * fx.Index(traits.LDS_V_TILE_SIZE * 2)
                    )
                for d in range_constexpr(traits.NV8_PER_THREAD):
                    _row = wave_id * fx.Index(traits.NV8_PER_THREAD) + fx.Index(d)
                    _lds_b = _vb + _row * fx.Index(traits.VEC_V_LINE * 2)
                    _lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(_lds_b))
                    _lds_ptr = buffer_ops.create_llvm_ptr(_lds_lane0, address_space=3)
                    _dcol = _row * fx.Index(8) + _v_dma_dloc
                    _voff_e = (
                        kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
                        + _v_dma_no * fx.Index(traits.HEAD_DIM * 8)
                        + _dcol * fx.Index(8)
                    )
                    _voff = fx.Int32(_voff_e * fx.Index(2))
                    rocdl.raw_ptr_buffer_load_lds(_rsrc, _lds_ptr, _v_dma_sz, _voff, _v_dma_z, _v_dma_z, _v_dma_aux)

        # Per-batch descriptors keep global indices 0-based and bounded to one batch.
        # This keeps 32-bit offsets small while preserving arbitrary-seqlen OOB behavior.
        _kv_nrec_bytes = as_mlir_value(seqlen_kv_b * fx.Index(traits.STRIDE_TOKEN_KV * 2))
        _q_nrec_bytes = as_mlir_value(seqlen_q_b * fx.Index(traits.STRIDE_TOKEN_Q * 2))
        _q_batch_byte_off = as_mlir_value(q_tok_base * fx.Index(traits.STRIDE_TOKEN_Q * 2))
        _kv_batch_byte_off = as_mlir_value(kv_tok_base * fx.Index(traits.STRIDE_TOKEN_KV * 2))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )

        # ---- DMA loading for K (buffer_load_dwordx4 ... lds) ----
        if const_expr(traits.ENABLE_DMA):
            k_rsrc = buffer_ops.create_buffer_resource(
                K, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            DMA_BYTES = 16  # buffer_load_dwordx4 = 16 bytes per lane
            DMA_BATCH_BYTES = traits.BLOCK_SIZE * DMA_BYTES
            K_TILE_BYTES = traits.BLOCK_N * traits.K_STRIDE * 2
            NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_K_ROW = traits.HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (traits.HEAD_DIM * 2)
            lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)
            _dma_size = fx.Int32(DMA_BYTES)
            _dma_soff = fx.Int32(0)
            _dma_off = fx.Int32(0)
            _dma_aux = fx.Int32(1)

            def coop_dma_k(tile_start, buf_id=0):
                """Load K tile via DMA with XOR-swizzled global fetch."""
                if const_expr(isinstance(buf_id, int)):
                    k_lds_byte_base = lds_kv_base_idx + fx.Index(buf_id * traits.LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * fx.Index(traits.LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (
                        k_lds_byte_base
                        + wave_id * fx.Index(traits.WARP_SIZE * DMA_BYTES)
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
                    global_row = tile_start + row_in_tile  # 0-based: batch base folded into k/v_rsrc
                    global_byte = (
                        global_row * fx.Index(traits.STRIDE_TOKEN_KV * 2)
                        + kv_head_idx * fx.Index(traits.HEAD_DIM * 2)
                        + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        _v_swizzle = kv_gmem_to_lds.v_swizzle

        # ---- DMA loading for V (buffer_load_dwordx4 ... lds) ----
        if const_expr(traits.ENABLE_DMA):
            v_rsrc = buffer_ops.create_buffer_resource(
                V, max_size=False, num_records_bytes=_kv_nrec_bytes, base_byte_offset=_kv_batch_byte_off
            )
            V_TILE_BYTES = traits.BLOCK_N * traits.V_STRIDE * 2
            NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_V_ROW = traits.HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH_V = DMA_BATCH_BYTES // (traits.HEAD_DIM * 2)

            def coop_dma_v(tile_start, buf_id=0):
                """Load V tile via DMA with XOR-swizzled global fetch."""
                v_lds_byte_base = lds_kv_base_idx + fx.Index((traits.LDS_V_BASE + buf_id * traits.LDS_V_TILE_SIZE) * 2)
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (
                        v_lds_byte_base
                        + wave_id * fx.Index(traits.WARP_SIZE * DMA_BYTES)
                        + fx.Index(d * DMA_BATCH_BYTES)
                    )
                    lds_i64 = fx.Int64(lds_addr)
                    lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                    row_in_tile = tid // LANES_PER_V_ROW + fx.Index(d * ROWS_PER_DMA_BATCH_V)
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & fx.Index(0x3)) << fx.Index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = tile_start + row_in_tile  # 0-based: batch base folded into k/v_rsrc
                    global_byte = (
                        global_row * fx.Index(traits.STRIDE_TOKEN_KV * 2)
                        + kv_head_idx * fx.Index(traits.HEAD_DIM * 2)
                        + col_byte
                    )
                    voffset = fx.Int32(global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc,
                        lds_ptr,
                        _dma_size,
                        voffset,
                        _dma_soff,
                        _dma_off,
                        _dma_aux,
                    )

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        # B operand: j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K. Q is
        # num_records-bounded (q_rsrc) so OOB rows read 0 -- no q_in_bounds select.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        q_b_packs = q_loader.load_all(q_rsrc, q_row)

        # ---- Constants ----
        ctx.init_constants(sm_scale)
        c_neg_inf = ctx.c_neg_inf
        c_neg_floor = ctx.c_neg_floor
        c_zero_f = ctx.c_zero_f
        c_zero_i32 = ctx.c_zero_i32
        c_sm_scale_log2e = ctx.c_sm_scale_log2e
        c_zero_v16f32 = ctx.c_zero_v16f32
        reduction_peer = softmax_helper.reduction_peer

        # ---- KV loop upper bound ----
        _q_end = q_start + traits.BLOCK_M
        if const_expr(traits.CAUSAL):
            delta_i32 = fx.Int32(seqlen_kv_b) - fx.Int32(seqlen_q_b)
            causal_end_raw_i32 = fx.Int32(_q_end) + delta_i32
            causal_end_i32 = fx.Int32(
                ArithValue(causal_end_raw_i32 > fx.Int32(0)).select(causal_end_raw_i32, fx.Int32(0))
            )
            causal_end = fx.Index(causal_end_i32)
            kv_upper = fx.Index(ArithValue(causal_end < seqlen_kv_b).select(causal_end, seqlen_kv_b))
        else:
            kv_upper = seqlen_kv_b

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(traits.D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if const_expr(_use_dma_dbuf):
            init_args.append(fx.Index(0))
            coop_dma_k(fx.Index(0), buf_id=0)

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
                        coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        coop_load_k(pre_k_start, pre_k_slot)
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
                        coop_dma_k(
                            kv_block_start + (kv_sub + 1) * traits.BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + fx.Index(traits.BLOCK_N_OUT)
                        _has_next = _next_kv < kv_upper
                        if _has_next:
                            coop_dma_k(_next_kv, _next_k_buf_id)
                    rocdl.sched_barrier(0)
                    k_base = k_buf_base(_k_buf_id)
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if const_expr(not _use_dma_dbuf):
                    k_base = k_buf_base(k_slot)

                if const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
                    coop_dma_v_nomajor(kv_start, 0)
                elif const_expr(not traits.USE_HW_TR or (not traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF)):
                    _v_vecs_prefetch = coop_load_v_global(kv_start)

                # ==== GEMM1: bulk-read all K packs, then pipeline MFMAs ====
                k_hi_offset = traits.K_SUB_N * traits.K_STRIDE
                # XOR swizzle: col ^ ((row & 0x7) << 4) avoids LDS bank conflicts
                k_swz_mask = (lane_mod_32 & fx.Index(traits.K_SWZ_ROWMASK)) << fx.Index(4)

                def _k_idx_lo(ks):
                    col = fx.Index(ks * traits.K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + lane_mod_32 * traits.K_STRIDE + (col ^ k_swz_mask)

                def _k_idx_hi(ks):
                    col = fx.Index(ks * traits.K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    return k_base + k_hi_offset + lane_mod_32 * traits.K_STRIDE + (col ^ k_swz_mask)

                _QK_PREFETCH_DEPTH = 2
                k_packs_lo = [None] * traits.K_STEPS_QK
                k_packs_hi = [None] * traits.K_STEPS_QK
                for p in range_constexpr(_QK_PREFETCH_DEPTH):
                    k_packs_lo[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                    k_packs_hi[p] = Vec.load(mfma_pack_type, lds_kv, [_k_idx_hi(p)])

                if const_expr(traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF):
                    coop_dma_v(kv_start, 0)
                    rocdl.sched_barrier(0)

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                for ks in range_constexpr(traits.K_STEPS_QK):
                    s_acc_lo = mfma_acc(k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                    s_acc_hi = mfma_acc(k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                    if const_expr(ks + _QK_PREFETCH_DEPTH < traits.K_STEPS_QK):
                        k_packs_lo[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)]
                        )
                        k_packs_hi[ks + _QK_PREFETCH_DEPTH] = Vec.load(
                            mfma_pack_type, lds_kv, [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)]
                        )

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(Vec(s_acc_lo)[r])
                    s_raw_hi.append(Vec(s_acc_hi)[r])

                if const_expr(traits.CAUSAL):
                    # Keep the runtime tile_needs_mask guard (below-diagonal tiles skip the
                    # 32 selects) but fold the selects into a loop. A Python list cannot be
                    # carried across a dynamic `if`, so drive the scf.if directly with the 32
                    # scalar scores as explicit state -> byte-identical ISA to the unrolled form.
                    # KV_VECTORIZED applies sigma(kv) in the K load, so the score at logical
                    # n_pos holds physical kv = kv_start + sigma(n_pos).
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_div_32_i32 = fx.Int32(lane_div_32)
                    q_start_i32 = fx.Int32(q_start) + delta_i32
                    q_mask_limit_i32 = q_row_i32 + delta_i32
                    max_kv_col_i32 = kv_start_i32 + fx.Int32(traits.BLOCK_N - 1)
                    tile_needs_mask = max_kv_col_i32 > q_start_i32

                    # Everything the guard executed stays inside the branch (lane_off/_MOFF
                    # included) and the state/yield order is interleaved (lo_r, hi_r per r),
                    # so the emitted scf.if is structurally identical to the unrolled form
                    # and the ISA stays byte-identical.
                    def _apply_causal_mask(_names, *scores):
                        if const_expr(traits.KV_VECTORIZED):
                            lane_off_i32 = lane_div_32_i32 * fx.Int32(8)
                            _MOFF = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)
                        else:
                            lane_off_i32 = lane_div_32_i32 * fx.Int32(4)
                            _MOFF = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)
                        out = []
                        for r in range_constexpr(16):
                            kv_col = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[r])
                            out.append(ArithValue(kv_col > q_mask_limit_i32).select(c_neg_inf, scores[2 * r]))
                            out.append(
                                ArithValue(kv_col + fx.Int32(traits.K_SUB_N) > q_mask_limit_i32).select(
                                    c_neg_inf, scores[2 * r + 1]
                                )
                            )
                        return out

                    _mask_names = tuple("_sm%d" % i for i in range(32))
                    _interleaved = [v for r in range(16) for v in (s_raw_lo[r], s_raw_hi[r])]
                    _masked = scf_if_dispatch(
                        tile_needs_mask,
                        _apply_causal_mask,
                        state_names=_mask_names,
                        state_values=_interleaved,
                    )
                    s_raw_lo = [_masked[2 * r] for r in range(16)]
                    s_raw_hi = [_masked[2 * r + 1] for r in range(16)]
                else:
                    # Mask physical KV columns outside seqlen so tail rows do not enter softmax.
                    kv_start_i32 = fx.Int32(kv_start)
                    seq_len_i32 = fx.Int32(seqlen_kv_b)
                    # KV_VECTORIZED: sigma(kv) in K load -> physical kv = kv_start +
                    # lane_div_32*8 + sigma(_off); hi adds K_SUB_N (bit5 unchanged).
                    if const_expr(traits.KV_VECTORIZED):
                        lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(8)
                        _MOFF = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)
                    else:
                        lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
                        _MOFF = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)
                    for r in range_constexpr(16):
                        kv_col = kv_start_i32 + lane_off_i32 + fx.Int32(_MOFF[r])
                        s_raw_lo[r] = ArithValue(kv_col >= seq_len_i32).select(c_neg_inf, s_raw_lo[r])
                        s_raw_hi[r] = ArithValue(kv_col + fx.Int32(traits.K_SUB_N) >= seq_len_i32).select(
                            c_neg_inf, s_raw_hi[r]
                        )

                local_max = s_raw_lo[0]
                for r in range_constexpr(15):
                    local_max = _fmax(local_max, s_raw_lo[r + 1])
                for r in range_constexpr(16):
                    local_max = _fmax(local_max, s_raw_hi[r])
                peer_max = reduction_peer(local_max)
                row_max = _fmax(local_max, peer_max)
                m_new_raw = _fmax(m_running, row_max)
                if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                    m_new_raw = _fmax(m_new_raw, c_neg_floor)

                diff_m_raw = _fsub(m_running, m_new_raw)
                diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
                corr = ArithValue(diff_m_scaled).exp2(fastmath=fm_fast)

                scaled_max = _fmul(c_sm_scale_log2e, m_new_raw)
                neg_scaled_max = _fsub(c_zero_f, scaled_max)

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
                l_corr = _fmul(corr, l_running)
                l_new = _fadd(l_corr, tile_sum)

                # ==== Rescale O accumulators ====
                corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
                if const_expr(not traits.USE_HW_TR):
                    o_accs[0] = _fmul(Vec(o_accs[0]), corr_vec)
                else:
                    for dc in range_constexpr(traits.D_CHUNKS):
                        o_accs[dc] = _fmul(Vec(o_accs[dc]), corr_vec)

                if const_expr(traits.ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < traits.N_SUBTILES):
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * traits.BLOCK_N
                    next_k_slot = traits.CK_LDS_SEQ[next_k_sub % len(traits.CK_LDS_SEQ)] % traits.NUM_PREFETCH_K
                    if const_expr(traits.ENABLE_DMA):
                        coop_dma_k(next_k_start, next_k_slot)
                    else:
                        coop_load_k(next_k_start, next_k_slot)

                if const_expr(traits.ENABLE_PREFETCH_3BUF):
                    v_slot = traits.CK_LDS_SEQ[kv_sub % len(traits.CK_LDS_SEQ)] % traits.NUM_PREFETCH_V
                    v_base = v_buf_base(v_slot)
                    coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif const_expr(traits.ENABLE_DMA):
                    v_base = v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                elif const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()

                # ==== Build P packs for lo and hi halves ====
                # bf16 truncates (upper-16-bit pack); f16 rounds each element then packs.
                # Slice width is MFMA_LANE_K (8 for K16, 4 for K8) in every case.
                p_packs_lo = softmax_helper.build_p_packs(p_vals_lo)
                p_packs_hi = softmax_helper.build_p_packs(p_vals_hi)

                # Build flat (dc, pks) schedule for interleaved GEMM2.
                _steps = [(dc, pks) for dc in range(traits.D_CHUNKS) for pks in range(traits.PV_K_STEPS)]
                TOTAL_PV = len(_steps)

                def _read_v_pack(step_idx):
                    dc, pks = _steps[step_idx]
                    if const_expr(traits.KV_VECTORIZED):
                        # No-major V: one aligned v8 per (dc,pks) half. Lane l reads
                        # V[d=dc*32+l%32, n=pks*16+(l//32)*8+0..7] (lo) / +32 (hi).
                        _lm = lane_mod_32
                        v_lane_base = (
                            v_base
                            + (_lm // fx.Index(8)) * fx.Index(traits.VEC_V_LINE)
                            + lane_div_32 * fx.Index(traits.VEC_V_D128)
                            + (_lm % fx.Index(8)) * fx.Index(8)
                        )
                        _lo_off = (
                            dc * (traits.D_CHUNK // 8) * traits.VEC_V_LINE
                            + pks * (traits.PV_K_STEP // 8) * traits.VEC_V_D128
                        )
                        _hi_off = _lo_off + (traits.K_SUB_N // 8) * traits.VEC_V_D128
                        vl = Vec.load(mfma_pack_type, lds_kv, [v_lane_base + fx.Index(_lo_off)])
                        vh = Vec.load(mfma_pack_type, lds_kv, [v_lane_base + fx.Index(_hi_off)])
                        return vl, vh
                    if const_expr(traits.USE_HW_TR):
                        d_col = fx.Index(dc * traits.D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
                        k_row = fx.Index(pks * traits.PV_K_STEP) + lane_div_32 * 4 + tr_k_group
                        _d_col_eff = _v_swizzle(k_row, d_col) if traits.ENABLE_DMA else d_col
                        lds_lo = v_base + k_row * traits.V_STRIDE + _d_col_eff
                        lds_hi = lds_lo + fx.Index(traits.K_SUB_N * traits.V_STRIDE)
                        if const_expr(traits.USE_K16):
                            vl_a = ds_read_tr_v4f16(lds_lo)
                            vl_b = ds_read_tr_v4f16(lds_lo + fx.Index(8 * traits.V_STRIDE))
                            vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                            vh_a = ds_read_tr_v4f16(lds_hi)
                            vh_b = ds_read_tr_v4f16(lds_hi + fx.Index(8 * traits.V_STRIDE))
                            vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        else:
                            vl = ds_read_tr_v4f16(lds_lo)
                            vh = ds_read_tr_v4f16(lds_hi)
                    else:
                        d_pos = fx.Index(dc * traits.D_CHUNK) + lane_mod_32
                        k_base = fx.Index(pks * traits.PV_K_STEP) + lane_div_32 * 4
                        v_lo_idx = v_base + d_pos * traits.VT_STRIDE + k_base
                        v_hi_idx = v_lo_idx + fx.Index(traits.K_SUB_N)
                        vl = Vec.load(v4f16_type, lds_kv, [v_lo_idx])
                        vh = Vec.load(v4f16_type, lds_kv, [v_hi_idx])
                    return vl, vh

                # Pre-read V for the first step.
                v_lo_cur, v_hi_cur = _read_v_pack(0)

                # ==== GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi ====
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_nxt, v_hi_nxt = _read_v_pack(si + 1)
                    o_accs[dc] = mfma_acc(v_lo_cur, p_packs_lo[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(v_hi_cur, p_packs_hi[pks], o_accs[dc])
                    if const_expr(not traits.USE_HW_TR and dc == 0 and pks < traits.D_CHUNKS - 1):
                        o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
                    if const_expr(si + 1 < TOTAL_PV):
                        v_lo_cur = v_lo_nxt
                        v_hi_cur = v_hi_nxt

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(_use_dma_dbuf):
                if const_expr(traits.N_SUBTILES % 2 == 1):
                    _yield_args.append(fx.Index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            loop_results = yield _yield_args

        def _zero_o_block():
            store_helper.zero_o_block(o_rsrc, q_row)

        def _normalize_and_store_o():
            store_helper.normalize_and_store_o(loop_results, o_rsrc, q_row)

        @flyc.jit
        def _store_cross_o():
            if causal_end_raw_i32 <= c_zero_i32:
                _zero_o_block()
            else:
                _normalize_and_store_o()

        # ---- Normalize and store O (128-bit buffer_store_dwordx4) ----
        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            _store_cross_o()
        else:
            _normalize_and_store_o()

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
