# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM stage1/stage2 kernel implementations — **a8w4smooth-only**.

This is a dedicated, API-restricted variant of `kernels.moe_gemm_2stage`.
The compiler infrastructure is shared with the multi-dtype kernel, but the
public `compile_*` entry points refuse any `in_dtype` other than
`"a8w4smooth"` (or its alias `"uint4"`).

Use this module from production paths (e.g. aiter) where only the W4A8 +
smoothquant + per-K64 zero-point dequant path is exercised, so the import
graph is independent of the legacy multi-dtype kernel and its tests.

Public API (all keyword-only):
- `compile_moe_gemm1(..., in_dtype="a8w4smooth", ...)`
- `compile_moe_gemm2(..., in_dtype="a8w4smooth", ...)`
- `compile_moe_gemm2_ex(..., in_dtype="a8w4smooth", mode=MoeGemm2Mode.ATOMIC, ...)`
- `compile_moe_reduction(...)`           — dtype-agnostic, copied verbatim
- `MoeGemm2Mode`, `_MoeGemm2ReduceWrapper`
"""

import logging
import os
import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith
from flydsl.expr import gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr, const_expr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import (
        supports_bf16_global_atomics,
        bf16_global_atomics_arch_description,
    )
except ImportError:
    # Backward compatibility for runtime.device versions that only expose get_rocm_arch.
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

    def bf16_global_atomics_arch_description() -> str:
        return "gfx94+/gfx95+/gfx12+"

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr.typing import T



from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from kernels.mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog
from kernels.mfma_preshuffle_pipeline import crd2idx, _buffer_load_vec


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """Compat helper for SCF IfOp else-region across old/new Python APIs."""
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])



@functools.lru_cache(maxsize=1024)
def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    # NOTE: aiter swap passes these for API symmetry; stage1 uses dynamic memrefs so they are ignored.
    doweight_stage1: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    scale_is_bf16: bool = False,
    k_batch: int = 1,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype: must be "a8w4smooth" (or its "uint4" alias). Other dtypes are rejected.
    group_size: accepted for ABI compat with the legacy multi-dtype kernel; ignored.
    scale_is_bf16: accepted for ABI compat; ignored (a8w4smooth uses f32 scales).
    k_batch: Split-K factor. When >1, K is partitioned across k_batch CTAs that
      atomically accumulate gate/up partials. Caller must pre-zero output.
    """

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    # a8w4smooth-only entry point: refuse any other dtype at the API boundary.
    # ('uint4' is a backward-compatible alias for 'a8w4smooth'.)
    _valid_dtypes = ("a8w4smooth", "uint4")
    if in_dtype not in _valid_dtypes:
        raise ValueError(
            f"a8w4_moe_gemm_2stage only supports in_dtype in {_valid_dtypes}, got {in_dtype!r}. "
            f"For other dtypes, use kernels.moe_gemm_2stage."
        )
    # a8w4smooth/uint4: W4A8 + smoothquant + packed4 zero-point dequant.
    # Backward-compatible alias: 'uint4' behaves like 'a8w4smooth'.
    is_uint4 = in_dtype == "uint4"
    elem_bytes = 1
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")
    # NOTE: don't materialize MLIR types outside an active MLIR Context.
    out_mlir = lambda: (
        (lambda ty: ty() if callable(ty) else ty)(
            T.f16 if out_dtype == "f16" else T.bf16
        )
    )
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    # K64-byte micro-step: always 64 bytes per `ku`. For a8w4smooth this is 64 elements.
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )

    # A8W4SMOOTH dequant: PDF packed4 qparams, fixed k_quantize_block=64 (4 blocks per tile256).
    # qscale/qzero are passed as packed4 i32 words with physical layout:
    #   [expert, rows_per_expert//16, K//256, 16] (packed i32) == [expert, N//16, K//4, 16, 4] (u8)
    a8w4smooth_scale_block_k = 64
    # This kernel path assumes the a8w4smooth PDF packed4 layout + K64 interleave used by
    # `tests/kernels/test_moe_gemm.py:build_a8w4smooth_moe_weight`.
    _qparam_prefix = "FLIR_UINT4" if is_uint4 else "FLIR_A8W4SMOOTH"
    a8w4smooth_qparam_format = os.environ.get(f"{_qparam_prefix}_QPARAM_FORMAT", "packed4").strip().lower()
    if a8w4smooth_qparam_format != "packed4":
        raise ValueError(
            f"a8w4smooth only supports {_qparam_prefix}_QPARAM_FORMAT=packed4, got {a8w4smooth_qparam_format!r}"
        )
    a8w4smooth_interleave = (
        os.environ.get(f"{_qparam_prefix}_INTERLEAVE_K64", "1").strip().lower() not in ("0", "false", "no")
    )
    if not bool(a8w4smooth_interleave):
        raise ValueError(
            f"a8w4smooth kernel path requires {_qparam_prefix}_INTERLEAVE_K64=1 (K64 interleave 0,64,1,65,...)"
        )
    if (model_dim % 256) != 0:
        raise ValueError(f"a8w4smooth kernel path requires model_dim%256==0, got {model_dim!r}")
    if int(tile_k) not in (128, 256):
        raise ValueError(f"a8w4smooth kernel path requires tile_k in {{128,256}}, got {tile_k!r}")
    # Compile-time safety switch (override via FLIR_A8W4SMOOTH_OVERFLOW_GUARD env var).
    _overflow_prefix = "FLIR_UINT4" if (in_dtype == "uint4") else "FLIR_A8W4SMOOTH"
    overflow_guard = os.environ.get(f"{_overflow_prefix}_OVERFLOW_GUARD", "1") in ("1", "true", "True", "YES", "yes")

    # Split-K validation
    _is_splitk = k_batch > 1
    if _is_splitk:
        _k_per_batch = model_dim // k_batch
        assert model_dim % k_batch == 0, f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        assert _k_per_batch % tile_k == 0, f"K_per_batch={_k_per_batch} not divisible by tile_k={tile_k}"
        # The ping-pong K-loop requires an even number of K tiles (>=4).
        _k_tiles = _k_per_batch // tile_k
        assert _k_tiles >= 4 and _k_tiles % 2 == 0, (
            f"K_per_batch/tile_k={_k_tiles} must be even and >=4 for the ping-pong pipeline. "
            f"Try a different k_batch (model_dim={model_dim}, tile_k={tile_k})."
        )
    else:
        _k_per_batch = model_dim

    mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
        rocdl, "mfma_i32_16x16x32_i8", None
    )
    if mfma_i32_k32 is None:
        raise AttributeError(
            "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
            "(or `rocdl.mfma_i32_16x16x32_i8`)."
        )

    DYN = ir.ShapedType.get_dynamic_size()
    size_out = DYN
    size_x = DYN
    # W is packed int4 for a8w4smooth: 2 values per byte.
    size_w = (experts * (2 * inter_dim) * model_dim) // 2
    # A8W4SMOOTH qscale/qzero packed4 i32 words:
    # stage1 rows_per_expert = 2*inter_dim, rows_blk = rows_per_expert//16, num_k256 = model_dim//256.
    rows_blk_stage1 = (2 * inter_dim) // 16
    num_k256_stage1 = model_dim // 256
    size_qscale_w = experts * rows_blk_stage1 * num_k256_stage1 * 16
    size_qzero_w = size_qscale_w
    size_sorted = DYN
    size_expert_ids = DYN

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use direct buffer_load for smaller widths.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLYDSL_MOE_STAGE1_CSHUFFLE", "1") in ("1", "true", "True", "YES", "yes")
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    # Split-K uses f32 atomic CShuffle regardless of out_dtype, so skip this check.
    if out_dtype != "f16" and use_cshuffle_epilog and not _is_splitk:
        raise ValueError("stage1 cshuffle epilog currently supports only f16 output (out_dtype='f16')")

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Keep an explicit ABI tag so signature changes can't accidentally reuse an old binary.
    _split_k_tag = f"_splitk{k_batch}" if _is_splitk else ""
    module_name = (
        f"mfma_moe1_{in_dtype}_{out_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"{_split_k_tag}"
        f"_abi4"  # +a8w4smooth packed4 qparams + half128_sel plumbing
    ).replace("-", "_")

    # ── LDS sizing (pure Python; no MLIR Context needed) ─────────────────────
    # Reuse the same LDS bytes for both:
    # - ping-pong X tiles (2 * tile_m * lds_stride bytes)
    # - optional epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    # Split-K requires CShuffle epilogue (f32 atomic adds via store_pair callback)
    if _is_splitk:
        _use_cshuffle_epilog = True
    _cshuffle_elem_bytes = 4 if _is_splitk else 2  # f32 for split-K, f16 otherwise
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if True:
        @flyc.kernel
        def moe_gemm1(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_qscale_w: fx.Tensor,
            arg_qzero_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_max_token_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            inter_in = arith.index_cast(T.index, i32_inter_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(
                T.index, i32_size_expert_ids_in
            )
            # i32 versions for layout construction (fly.make_shape requires i32/i64)
            tokens_i32_v = i32_tokens_in
            k_i32_v = i32_k_in
            # a8w4smooth: X is int8, W is packed int4 (stored as i8) and unpacked in-kernel.
            x_elem = T.i8
            w_elem = T.i8
            vec16_elems = 16
            vec8_elems = 8
            vec4_elems = 4
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            def silu(x):
                # device fast path:
                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
                #   sig = rcp(1 + emu)                   -> v_rcp_f32
                #   y = x * sig
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                t = x * (-1.4426950408889634)  # -log2(e)
                emu = rocdl.exp2(T.f32, t)
                den = 1.0 + emu
                sig = rocdl.rcp(T.f32, den)
                return x * sig

            acc_init = arith.constant_vector(0, T.i32x4)

            # Layouts (use i32 values; fly.make_shape requires i32/i64, not index)
            layout_x = fx.make_layout((tokens_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: match GEMM test helper exactly.
            c_n_total = arith.index(experts * (2 * inter_dim))
            # a8w4smooth (packed4 + K64-interleave): B is stored as packed-u4 with kpack_bytes=16.
            kpack_bytes = 16
            w_elem_bytes = 1
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=w_elem_bytes,
                packed_4bit=True,
            )
            layout_b = b_layout.layout_b
            c_k0 = (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping (NSwizzle==false):
            # - blockIdx.x -> N dimension (tile along inter_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along inter_dim
            bx = gpu.block_id("y")  # tile along sorted M

            if _is_splitk:
                bz = gpu.block_id("z")  # K-batch id
                k_base_idx = bz * arith.index(_k_per_batch)
            else:
                k_base_idx = arith.index(0)

            # Block validity: compute as early as possible so invalid blocks skip all buffer-resource
            # setup, LDS pointer math, and gmem prefetch work.
            bx_m = bx * fx.Index(tile_m)
            maxids_rsrc = buffer_ops.create_buffer_resource(
                arg_max_token_ids,
                max_size=False,
                num_records_bytes=fx.Index(4),
            )
            max_token_id_i32 = buffer_ops.buffer_load(
                maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
            )
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32)
            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            # Everything below is gated by `blk_valid` to avoid doing buffer-resource setup and
            # gmem work for padding blocks.
            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                base_ptr = allocator.get_base()
                lds_x_ptr = SmemPtr(
                    base_ptr,
                    lds_alloc_offset,
                    T.i8,
                    shape=(lds_total_elems,),
                )
                lds_x = lds_x_ptr.get()
                # Alias LDS bytes for optional CShuffle epilogue.
                # Split-K uses f32 (4B) per element for atomic accumulation; normal uses f16 (2B).
                _lds_out_elem_type = T.f32 if _is_splitk else T.f16
                lds_out = (
                    SmemPtr(base_ptr, lds_x_ptr.byte_offset, _lds_out_elem_type, shape=(tile_m * tile_n,)).get()
                    if _use_cshuffle_epilog
                    else None
                )

                # Buffer resources: for dynamic memrefs, provide `num_records_bytes` explicitly so
                # hardware OOB behavior is stable (otherwise it falls back to a large max size).
                c_topk = fx.Index(topk)

                # X: [tokens, k] bytes = tokens*k*elem_bytes
                x_rows = tokens_in * c_topk
                x_nbytes_idx = x_rows * k_in * arith.index(int(elem_bytes))
                x_rsrc = buffer_ops.create_buffer_resource(
                    arg_x, max_size=False, num_records_bytes=x_nbytes_idx
                )

                w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

                # OUT: normal=[tokens, topk, inter] f16/bf16, split-K=[tokens*topk, 2*inter] f32
                out_elem_bytes = 4 if _is_splitk else 2
                if _is_splitk:
                    out_nbytes_idx = tokens_in * c_topk * inter_in * fx.Index(2 * out_elem_bytes)
                else:
                    out_nbytes_idx = tokens_in * c_topk * inter_in * fx.Index(out_elem_bytes)
                out_rsrc = buffer_ops.create_buffer_resource(
                    arg_out, max_size=False, num_records_bytes=out_nbytes_idx
                )

                # scale_x: a8w4smooth path uses per-(token,slot) scale.
                sx_rows = tokens_in * c_topk
                sx_nbytes_idx = sx_rows * fx.Index(4)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx
                )
                # scale_w: a8w4smooth needs dequant scale.
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

                # a8w4smooth qparams: packed4 [E, N//16, K//256, 16] of i32.
                qs_rsrc = buffer_ops.create_buffer_resource(arg_qscale_w, max_size=False)
                qz_rsrc = buffer_ops.create_buffer_resource(arg_qzero_w, max_size=False)

                sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
                sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)

                # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids,
                    max_size=False,
                    num_records_bytes=(size_expert_ids_in * fx.Index(4)),
                )

                # Expert id for this M tile (keep address math in `index`)
                expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
                expert_idx = arith.index_cast(T.index, expert_i32)
                inter2_idx = arith.index(2 * inter_dim)
                expert_off_idx = expert_idx * inter2_idx  # index
    
                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads.
                if bytes_per_thread_x % 16 == 0:
                    x_load_bytes = 16
                elif bytes_per_thread_x % 8 == 0:
                    x_load_bytes = 8
                elif bytes_per_thread_x % 4 == 0:
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                    )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)
    
                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                layout_x_div4 = fx.make_layout((tokens_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1))
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
                mask24 = fx.Int32(0xFFFFFF)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)
                topk_i32 = fx.Int32(topk)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                # decode token once (per thread's M-slice) and build a base row offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    # NOTE: rows beyond `num_valid_ids` can contain garbage (within the allocated
                    # buffer). That's OK as long as we never use an out-of-range token id to index X.
                    fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                    t_raw = fused_i & mask24
                    # NOTE: aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for X loads; explicitly mask to a safe row.
                    t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                    s_raw = fused_i >> 24
                    # X is indexed by token-slot in **slot-major** order:
                    #   row_ts = slot * tokens + token
                    # This matches CK's moe_smoothquant output layout.
                    row_ts_i32 = s_raw * tokens_i32 + t_raw
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    # Apply bounds check to token-slot index
                    row_ts_safe = t_valid_i32.select(row_ts_idx, fx.Index(0))
                    x_row_base_div4.append(row_ts_safe * c_k_div4)
    
                vec4_x = T.vec(4, x_elem)

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    idx_i32 is in dword units; convert to element index for _buffer_load_vec.
                    """
                    if x_load_bytes == 16:
                        idx_elem = idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                            elem_bytes=elem_bytes,
                        )
                    # For 8B/4B, load raw i32 dwords directly.
                    if x_load_bytes == 8:
                        return buffer_ops.buffer_load(x_rsrc, idx_i32, vec_width=2, dtype=T.i32)
                    return buffer_ops.buffer_load(x_rsrc, idx_i32, vec_width=1, dtype=T.i32)
    
                def load_x_tile(base_k):
                    """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                    base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(x_vec)
                        else:
                            parts.append(x_vec)
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)
    
                # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16 * a_kpack_elems.
                # A-side kpack is always 16 bytes (activation elements); B-side kpack_bytes
                # may differ (e.g. 8 for int4 weights), but that only affects B preshuffle.
                row_a_lds = lane_mod_16
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.index(int(elem_bytes)))
                )
    
                # Dynamic N tiling within block (same as existing kernels)
                by_n = by * fx.Index(tile_n)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_mod_4 = wave_id % fx.Index(4)
                n_tile_base = wave_mod_4 * c_n_per_wave
    
                # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
                n_intra_gate = []
                n_blk_gate = []
                n_intra_up = []
                n_blk_up = []
                col_g_list = []
                inter_idx = fx.Index(inter_dim)
                c_n0 = c_n_total // fx.Index(16)
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base
                    col_g = col_g + offset
                    col_g = col_g + lane_mod_16
                    col_g_list.append(col_g)
    
                    row_gate = expert_off_idx + col_g
                    row_up = row_gate + inter_idx
    
                    coord_gate = fx.idx2crd(row_gate, layout_n_blk_intra)
                    n_blk_gate.append(fx.get(coord_gate, 0))
                    n_intra_gate.append(fx.get(coord_gate, 1))
    
                    coord_up = fx.idx2crd(row_up, layout_n_blk_intra)
                    n_blk_up.append(fx.get(coord_up, 0))
                    n_intra_up.append(fx.get(coord_up, 1))
    
    
                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)

                # A8W4SMOOTH packed4 qparams: used by load_b_tile_a8w4smooth only.
                c_rows_blk_s1 = fx.Index((2 * inter_dim) // 16)
                c_num_k256_s1 = fx.Index(model_dim // 256)

                # --- A8W4SMOOTH helpers (port from FlyDSL ref) -----------------
                def load_b_tile_a8w4smooth(base_k, blk_list, intra_list):
                    """A8W4SMOOTH tile-k=256 loader (drop-in for load_b_tile)."""
                    if int(tile_k) != 256:
                        raise ValueError(f"a8w4smooth tile loader requires tile_k==256, got {tile_k!r}")
                    if int(kpack_bytes) != 16:
                        raise ValueError(f"a8w4smooth tile loader requires kpack_bytes==16, got {kpack_bytes!r}")

                    i32_ty = T.i32
                    vec1_i64_ty = T.vec(1, T.i64)
                    vec2_i32_ty = T.vec(2, T.i32)
                    vec4_i32_ty = T.i32x4

                    c16 = fx.Index(16)
                    c256 = fx.Index(256)
                    c2 = fx.Index(2)
                    c64 = fx.Index(64)
                    c4 = fx.Index(4)

                    # qparam K256 tile id
                    k256 = base_k // c256

                    # Pre-load packed4 qparams once per ni.
                    qs_word_list = []
                    qz_word_list = []
                    for ni in range_constexpr(num_acc_n):
                        n_blk_global = blk_list[ni]
                        n_lane = intra_list[ni]
                        expert_id = n_blk_global // c_rows_blk_s1
                        n_blk_local = n_blk_global - (expert_id * c_rows_blk_s1)
                        qs_idx = (
                            ((((expert_id * c_rows_blk_s1) + n_blk_local) * c_num_k256_s1) + k256)
                            * c16
                            + n_lane
                        )
                        qs_word_list.append(buffer_ops.buffer_load(qs_rsrc, qs_idx, vec_width=1, dtype=T.i32))
                        qz_word_list.append(buffer_ops.buffer_load(qz_rsrc, qs_idx, vec_width=1, dtype=T.i32))

                    # packed_4bit layout: K0 is in packed bytes (K/2), macro-step 64B => K0_base=base_k/128
                    base_k_packed_bytes = base_k // c2
                    k0_base = base_k_packed_bytes // c64

                    c_ff = arith.constant(0x000000FF, type=i32_ty)
                    c_sign_flip = arith.constant(0x80808080, type=i32_ty)
                    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=i32_ty)
                    c_zero_i32 = arith.constant(0, type=i32_ty)

                    def _dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte):
                        dwords = [d0, d1, d2, d3]
                        even_outs = []
                        odd_outs = []
                        for di in range_constexpr(4):
                            dw = dwords[di]
                            even = dw & c_0f0f0f0f
                            odd = (dw >> 4) & c_0f0f0f0f
                            if not bool(overflow_guard):
                                even_out = ((even * qs_lo) + qz_lo) ^ c_sign_flip
                                odd_out = ((odd * qs_hi) + qz_hi) ^ c_sign_flip
                            else:
                                c_255 = arith.constant(255, type=i32_ty)

                                def _clamp_u8(x):
                                    gt = arith.cmpi(arith.CmpIPredicate.ugt, x, c_255)
                                    return gt.select(c_255, x)

                                def _dequant_safe_single(v, qs, qz_byte):
                                    b0 = v & c_ff
                                    b1 = (v >> 8) & c_ff
                                    b2 = (v >> 16) & c_ff
                                    b3 = (v >> 24) & c_ff
                                    o0 = _clamp_u8((b0 * qs) + qz_byte)
                                    o1 = _clamp_u8((b1 * qs) + qz_byte)
                                    o2 = _clamp_u8((b2 * qs) + qz_byte)
                                    o3 = _clamp_u8((b3 * qs) + qz_byte)
                                    out = o0 | (o1 << 8) | (o2 << 16) | (o3 << 24)
                                    return out ^ c_sign_flip

                                even_out = _dequant_safe_single(even, qs_lo, qz_lo_byte)
                                odd_out = _dequant_safe_single(odd, qs_hi, qz_hi_byte)
                            even_outs.append(even_out)
                            odd_outs.append(odd_out)
                        return even_outs + odd_outs

                    def _pair_as_i64(a, b):
                        v2 = vector.from_elements(vec2_i32_ty, [a, b])
                        return vector.extract(vector.bitcast(vec1_i64_ty, v2), static_position=[0], dynamic_position=[])

                    b_tile = []
                    for pack_group in range_constexpr(k_unroll // 2):
                        k0 = k0_base + fx.Index(pack_group)

                        # Load 16B weight pack once per ni for this pack_group.
                        # idx_pack from crd2idx is in bytes; use offset_in_bytes=True so the
                        # primitive does the byte→dword conversion internally (avoids needing
                        # IntTuple // Index, which is unsupported).
                        w_i32x4_list = []
                        for ni in range_constexpr(num_acc_n):
                            coord_pack = (blk_list[ni], k0, lane_div_16, intra_list[ni], fx.Index(0))
                            idx_pack = crd2idx(coord_pack, layout_b)
                            b16 = _buffer_load_vec(
                                buffer_ops, vector, w_rsrc, idx_pack,
                                elem_type=w_elem, vec_elems=16, elem_bytes=1,
                                offset_in_bytes=True,
                            )
                            w_i32x4_list.append(vector.bitcast(vec4_i32_ty, b16))

                        if pack_group == 0:
                            perm_lo = arith.constant(0x00000000, type=i32_ty)
                            perm_hi = arith.constant(0x01010101, type=i32_ty)
                        else:
                            perm_lo = arith.constant(0x02020202, type=i32_ty)
                            perm_hi = arith.constant(0x03030303, type=i32_ty)

                        out8_list = []
                        for ni in range_constexpr(num_acc_n):
                            qs_word = qs_word_list[ni]
                            qz_word = qz_word_list[ni]
                            if pack_group == 0:
                                qs_lo = qs_word & c_ff
                                qs_hi = (qs_word >> 8) & c_ff
                                qz_lo_byte = qz_word & c_ff
                                qz_hi_byte = (qz_word >> 8) & c_ff
                            else:
                                qs_lo = (qs_word >> 16) & c_ff
                                qs_hi = (qs_word >> 24) & c_ff
                                qz_lo_byte = (qz_word >> 16) & c_ff
                                qz_hi_byte = (qz_word >> 24) & c_ff
                            qz_lo = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_lo], [], [])
                            qz_hi = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_hi], [], [])
                            w_i32x4 = w_i32x4_list[ni]
                            d0 = vector.extract(w_i32x4, static_position=[0], dynamic_position=[])
                            d1 = vector.extract(w_i32x4, static_position=[1], dynamic_position=[])
                            d2 = vector.extract(w_i32x4, static_position=[2], dynamic_position=[])
                            d3 = vector.extract(w_i32x4, static_position=[3], dynamic_position=[])
                            out8_list.append(
                                _dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte)
                            )

                        # Assemble b_tile: 2 entries per pack_group.
                        packs0_lo = []
                        packs1_lo = []
                        for ni in range_constexpr(num_acc_n):
                            out8 = out8_list[ni]
                            packs0_lo.append(_pair_as_i64(out8[0], out8[1]))
                            packs1_lo.append(_pair_as_i64(out8[2], out8[3]))
                        b_tile.append((packs0_lo, packs1_lo))

                        packs0_hi = []
                        packs1_hi = []
                        for ni in range_constexpr(num_acc_n):
                            out8 = out8_list[ni]
                            packs0_hi.append(_pair_as_i64(out8[4], out8[5]))
                            packs1_hi.append(_pair_as_i64(out8[6], out8[7]))
                        b_tile.append((packs0_hi, packs1_hi))

                    return b_tile

                def preload_qparams_tilek128(base_k_256, blk_list, intra_list):
                    """A8W4SMOOTH tile-k=128 qparams loader: load qparams once per K256 chunk."""
                    c16 = fx.Index(16)
                    c256 = fx.Index(256)
                    k256 = base_k_256 // c256
                    qs_word_list = []
                    qz_word_list = []
                    for ni in range_constexpr(num_acc_n):
                        n_blk_global = blk_list[ni]
                        n_lane = intra_list[ni]
                        expert_id = n_blk_global // c_rows_blk_s1
                        n_blk_local = n_blk_global - (expert_id * c_rows_blk_s1)
                        qs_idx = (
                            ((((expert_id * c_rows_blk_s1) + n_blk_local) * c_num_k256_s1) + k256)
                            * c16
                            + n_lane
                        )
                        qs_word_list.append(buffer_ops.buffer_load(qs_rsrc, qs_idx, vec_width=1, dtype=T.i32))
                        qz_word_list.append(buffer_ops.buffer_load(qz_rsrc, qs_idx, vec_width=1, dtype=T.i32))
                    return qs_word_list, qz_word_list

                def load_b_half_tilek128_raw(pack_group, qs_word_list, qz_word_list, base_k_256, blk_list, intra_list):
                    """A8W4SMOOTH tile-k=128 half loader: load one raw 128B packed half + qparams metadata."""
                    vec4_i32_ty = T.i32x4
                    c2 = fx.Index(2)
                    c64 = fx.Index(64)
                    c4 = fx.Index(4)
                    base_k_packed_bytes = base_k_256 // c2
                    k0_base = base_k_packed_bytes // c64

                    k0 = k0_base + fx.Index(pack_group)
                    w_i32x4_list = []
                    for ni in range_constexpr(num_acc_n):
                        coord_pack = (blk_list[ni], k0, lane_div_16, intra_list[ni], fx.Index(0))
                        idx_pack = crd2idx(coord_pack, layout_b)
                        b16 = _buffer_load_vec(
                            buffer_ops, vector, w_rsrc, idx_pack,
                            elem_type=w_elem, vec_elems=16, elem_bytes=1,
                            offset_in_bytes=True,
                        )
                        w_i32x4_list.append(vector.bitcast(vec4_i32_ty, b16))

                    return (qs_word_list, qz_word_list, w_i32x4_list)

                def dequant_b_half_tilek128(raw_half, pack_group):
                    """A8W4SMOOTH tile-k=128 helper: dequantize one raw 128B packed half before compute."""
                    qs_word_list, qz_word_list, w_i32x4_list = raw_half
                    i32_ty = T.i32
                    vec1_i64_ty = T.vec(1, T.i64)
                    vec2_i32_ty = T.vec(2, T.i32)

                    c_ff = arith.constant(0x000000FF, type=i32_ty)
                    c_sign_flip = arith.constant(0x80808080, type=i32_ty)
                    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=i32_ty)
                    c_zero_i32 = arith.constant(0, type=i32_ty)

                    if pack_group == 0:
                        perm_lo = arith.constant(0x00000000, type=i32_ty)
                        perm_hi = arith.constant(0x01010101, type=i32_ty)
                    else:
                        perm_lo = arith.constant(0x02020202, type=i32_ty)
                        perm_hi = arith.constant(0x03030303, type=i32_ty)

                    qparam_list = []
                    for ni in range_constexpr(num_acc_n):
                        qs_word = qs_word_list[ni]
                        qz_word = qz_word_list[ni]
                        if pack_group == 0:
                            qs_lo = qs_word & c_ff
                            qs_hi = (qs_word >> 8) & c_ff
                            qz_lo_byte = qz_word & c_ff
                            qz_hi_byte = (qz_word >> 8) & c_ff
                        else:
                            qs_lo = (qs_word >> 16) & c_ff
                            qs_hi = (qs_word >> 24) & c_ff
                            qz_lo_byte = (qz_word >> 16) & c_ff
                            qz_hi_byte = (qz_word >> 24) & c_ff
                        qz_lo = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_lo], [], [])
                        qz_hi = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_hi], [], [])
                        qparam_list.append((qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte))

                    def _dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte):
                        dwords = [d0, d1, d2, d3]
                        even_outs = []
                        odd_outs = []
                        for di in range_constexpr(4):
                            dw = dwords[di]
                            even = dw & c_0f0f0f0f
                            odd = (dw >> 4) & c_0f0f0f0f
                            if not bool(overflow_guard):
                                even_out = ((even * qs_lo) + qz_lo) ^ c_sign_flip
                                odd_out = ((odd * qs_hi) + qz_hi) ^ c_sign_flip
                            else:
                                c_255 = arith.constant(255, type=i32_ty)

                                def _clamp_u8(x):
                                    gt = arith.cmpi(arith.CmpIPredicate.ugt, x, c_255)
                                    return gt.select(c_255, x)

                                def _dequant_safe_single(v, qs, qz_byte):
                                    b0 = v & c_ff
                                    b1 = (v >> 8) & c_ff
                                    b2 = (v >> 16) & c_ff
                                    b3 = (v >> 24) & c_ff
                                    o0 = _clamp_u8((b0 * qs) + qz_byte)
                                    o1 = _clamp_u8((b1 * qs) + qz_byte)
                                    o2 = _clamp_u8((b2 * qs) + qz_byte)
                                    o3 = _clamp_u8((b3 * qs) + qz_byte)
                                    out = o0 | (o1 << 8) | (o2 << 16) | (o3 << 24)
                                    return out ^ c_sign_flip

                                even_out = _dequant_safe_single(even, qs_lo, qz_lo_byte)
                                odd_out = _dequant_safe_single(odd, qs_hi, qz_hi_byte)
                            even_outs.append(even_out)
                            odd_outs.append(odd_out)
                        return even_outs + odd_outs

                    def _pair_as_i64(a, b):
                        v2 = vector.from_elements(vec2_i32_ty, [a, b])
                        return vector.extract(vector.bitcast(vec1_i64_ty, v2), static_position=[0], dynamic_position=[])

                    out8_list = []
                    for ni in range_constexpr(num_acc_n):
                        qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte = qparam_list[ni]
                        w_i32x4 = w_i32x4_list[ni]
                        d0 = vector.extract(w_i32x4, static_position=[0], dynamic_position=[])
                        d1 = vector.extract(w_i32x4, static_position=[1], dynamic_position=[])
                        d2 = vector.extract(w_i32x4, static_position=[2], dynamic_position=[])
                        d3 = vector.extract(w_i32x4, static_position=[3], dynamic_position=[])
                        out8_list.append(_dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte))

                    packs0_lo = []
                    packs1_lo = []
                    for ni in range_constexpr(num_acc_n):
                        out8 = out8_list[ni]
                        packs0_lo.append(_pair_as_i64(out8[0], out8[1]))
                        packs1_lo.append(_pair_as_i64(out8[2], out8[3]))

                    packs0_hi = []
                    packs1_hi = []
                    for ni in range_constexpr(num_acc_n):
                        out8 = out8_list[ni]
                        packs0_hi.append(_pair_as_i64(out8[4], out8[5]))
                        packs1_hi.append(_pair_as_i64(out8[6], out8[7]))

                    b_half = []
                    b_half.append((packs0_lo, packs1_lo))
                    b_half.append((packs0_hi, packs1_hi))
                    return b_half

                # A8W4SMOOTH dispatch: load_b_tile is the packed-4bit dequant loader.
                load_b_tile = load_b_tile_a8w4smooth

                acc_gate = [acc_init] * (num_acc_n * m_repeat)
                acc_up = [acc_init] * (num_acc_n * m_repeat)

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                            )
                        else:
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                            )
    
                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                def compute_tile(
                    acc_gate_in,
                    acc_up_in,
                    b_gate_tile_in,
                    b_up_tile_in,
                    lds_base,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                ):
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in)
                    mfma_res_ty = T.i32x4
                    mfma_fn = mfma_i32_k32

                    # Optional: prefetch epilogue scales while we are about to run the last MFMA tile,
                    # matching the preshuffle GEMM pattern of overlapping scale loads with MFMA.
                    epilogue_pf = None
                    if prefetch_epilogue:
                        expert_off_pf = expert_off_idx
                        sw_gate_pf = []
                        sw_up_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_gate_idx = expert_off_pf + col_g
                            row_up_idx = row_gate_idx + inter_idx
                            sw_gate_pf.append(
                                buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32)
                            )
                            sw_up_pf.append(
                                buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32)
                            )
                        epilogue_pf = (sw_gate_pf, sw_up_pf)

                    def mfma_k64(acc_in, a0, a1, b0, b1):
                        acc_mid = mfma_fn(mfma_res_ty, [a0, b0, acc_in, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc_mid, 0, 0, 0])

                    for ku in range_constexpr(k_unroll):
                        b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]
                        b_up_packs0, b_up_packs1 = b_up_tile_in[ku]
                        ki64 = arith.index(ku * 64)
                        col_base = col_offset_base_bytes + ki64

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.index(mi * 16)
                            curr_row_a_lds = row_a_lds + mi_val

                            if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)

                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni
                                gate_list[acc_idx] = mfma_k64(
                                    gate_list[acc_idx],
                                    a0,
                                    a1,
                                    b_gate_packs0[ni],
                                    b_gate_packs1[ni],
                                )
                                up_list[acc_idx] = mfma_k64(
                                    up_list[acc_idx],
                                    a0,
                                    a1,
                                    b_up_packs0[ni],
                                    b_up_packs1[ni],
                                )
                    return gate_list, up_list, epilogue_pf

                def compute_tile_a8w4smooth_tilek128(
                    acc_gate_in, acc_up_in, b_gate_half, b_up_half, lds_base,
                    *, prefetch_epilogue: bool = False, a0_prefetch=None,
                ):
                    """Tile-k=128 compute step: uses a single half-tile (K128). Forwards to compute_tile."""
                    return compute_tile(
                        acc_gate_in, acc_up_in, b_gate_half, b_up_half, lds_base,
                        prefetch_epilogue=prefetch_epilogue, a0_prefetch=a0_prefetch,
                    )

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.index(tile_m * lds_stride)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems
    
                # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
                rocdl.sched_barrier(0)
    
                def hot_loop_scheduler():
                    rocdl.sched_barrier(0)

                if const_expr(int(tile_k) == 128):
                    # ─── A8W4SMOOTH tile_k=128 ping-pong ─────────────────────────────────
                    # Ported from FlyDSL kernels/moe_gemm_2stage.py:1232-1335.
                    # The K256-chunk main loop is Python-unrolled (range_constexpr) instead
                    # of scf.for so that qs/qz/raw ping-pong state can flow via Python rebind
                    # (FLIR's scf.for needs explicit init/yield for loop-carried values, and
                    # _k_per_batch is compile-time so unrolling is legal).
                    c128 = fx.Index(128)
                    c256 = fx.Index(256)

                    # Prologue: preload qparams + raw low(K128) of chunk0, then store X tile to LDS.
                    k0 = k_base_idx
                    x_regs0 = load_x_tile(k0)
                    qs_gate, qz_gate = preload_qparams_tilek128(k0, n_blk_gate, n_intra_gate)
                    qs_up, qz_up = preload_qparams_tilek128(k0, n_blk_up, n_intra_up)
                    raw_gate_pong = load_b_half_tilek128_raw(0, qs_gate, qz_gate, k0, n_blk_gate, n_intra_gate)
                    raw_up_pong = load_b_half_tilek128_raw(0, qs_up, qz_up, k0, n_blk_up, n_intra_up)
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                    gpu.barrier()

                    lds_base_pong = lds_base_cur  # current/compute
                    lds_base_ping = lds_base_nxt  # next/load+store

                    a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                    # Main loop: each iteration covers one K256 chunk via low128 + high128 ping-pong.
                    # Iterates over all-but-final K256 chunks; the last K256 chunk is the tail+epilogue.
                    num_main_iters_k256 = max((int(_k_per_batch) // 256) - 1, 0)
                    for iter_idx in range_constexpr(num_main_iters_k256):
                        k_iv = k_base_idx + arith.index(iter_idx * 256)

                        # ---- stage 0: Load High(N), Compute Low(N) ----
                        next_k1 = k_iv + c128
                        x_regs_ping = load_x_tile(next_k1)

                        raw_gate_ping = load_b_half_tilek128_raw(1, qs_gate, qz_gate, k_iv, n_blk_gate, n_intra_gate)
                        raw_up_ping = load_b_half_tilek128_raw(1, qs_up, qz_up, k_iv, n_blk_up, n_intra_up)

                        b_gate_pong = dequant_b_half_tilek128(raw_gate_pong, 0)
                        b_up_pong = dequant_b_half_tilek128(raw_up_pong, 0)

                        acc_gate, acc_up, _ = compute_tile_a8w4smooth_tilek128(
                            acc_gate, acc_up, b_gate_pong, b_up_pong, lds_base_pong, a0_prefetch=a0_prefetch_pong,
                        )
                        a0_prefetch_pong = None
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                        # ---- stage 1: Preload QParams(N+1), Load Low(N+1), Compute High(N) ----
                        next_k2 = k_iv + c256
                        x_regs_pong = load_x_tile(next_k2)

                        qs_gate, qz_gate = preload_qparams_tilek128(next_k2, n_blk_gate, n_intra_gate)
                        qs_up, qz_up = preload_qparams_tilek128(next_k2, n_blk_up, n_intra_up)

                        raw_gate_pong = load_b_half_tilek128_raw(0, qs_gate, qz_gate, next_k2, n_blk_gate, n_intra_gate)
                        raw_up_pong = load_b_half_tilek128_raw(0, qs_up, qz_up, next_k2, n_blk_up, n_intra_up)

                        b_gate_ping = dequant_b_half_tilek128(raw_gate_ping, 1)
                        b_up_ping = dequant_b_half_tilek128(raw_up_ping, 1)

                        acc_gate, acc_up, _ = compute_tile_a8w4smooth_tilek128(
                            acc_gate, acc_up, b_gate_ping, b_up_ping, lds_base_ping, a0_prefetch=a0_prefetch_ping,
                        )
                        a0_prefetch_ping = None
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                    # Tail: final K256 chunk = low128 (tail compute pong) then high128 (epilogue ping).
                    k_tail0 = k_base_idx + arith.index(int(_k_per_batch) - 256)
                    k_tail1 = k_base_idx + arith.index(int(_k_per_batch) - 128)
                    x_regs_ping = load_x_tile(k_tail1)

                    raw_gate_ping = load_b_half_tilek128_raw(1, qs_gate, qz_gate, k_tail0, n_blk_gate, n_intra_gate)
                    raw_up_ping = load_b_half_tilek128_raw(1, qs_up, qz_up, k_tail0, n_blk_up, n_intra_up)

                    b_gate_pong = dequant_b_half_tilek128(raw_gate_pong, 0)
                    b_up_pong = dequant_b_half_tilek128(raw_up_pong, 0)

                    acc_gate, acc_up, _ = compute_tile_a8w4smooth_tilek128(
                        acc_gate, acc_up, b_gate_pong, b_up_pong, lds_base_pong, a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()

                    a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                    # Epilogue: compute the final K128 high tile with prefetch_epilogue=True.
                    b_gate_ping = dequant_b_half_tilek128(raw_gate_ping, 1)
                    b_up_ping = dequant_b_half_tilek128(raw_up_ping, 1)
                    acc_gate, acc_up, epilogue_pf = compute_tile_a8w4smooth_tilek128(
                        acc_gate, acc_up, b_gate_ping, b_up_ping, lds_base_ping,
                        prefetch_epilogue=True, a0_prefetch=a0_prefetch_ping,
                    )
                else:
                    # Prologue: prefetch tile0, store to LDS(cur), sync.
                    k0 = k_base_idx
                    x_regs0 = load_x_tile(k0)
                    b_gate_cur = load_b_tile(k0, n_blk_gate, n_intra_gate)
                    b_up_cur = load_b_tile(k0, n_blk_up, n_intra_up)
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                    gpu.barrier()

                    # Loop-carried ping/pong state.
                    lds_base_pong = lds_base_cur  # current/compute
                    lds_base_ping = lds_base_nxt  # next/load+store

                    # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                    # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                    a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                    # Ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
                    # Uses scf.for with loop-carried accumulators, B-tile prefetch, and A0 LDS prefetch.
                    c2_tile_k = arith.index(tile_k * 2)
                    c_tile_k = arith.index(tile_k)
                    total_tiles = int(_k_per_batch) // int(tile_k)
                    pair_iters = max((total_tiles - 2) // 2, 0)

                    # B-tile data layout per k_unroll entry:
                    #   fp8/int8 (default — two register packs per ku):
                    #     (packs_even_list, packs_odd_list)
                    #     Two lists of num_acc_n regs for even/odd MFMA operands.
                    #     Flattened as: [even_0..N, odd_0..N]  -> 2 * num_acc_n values
                    _fields_per_ku = 2
                    _vals_per_b_tile = k_unroll * _fields_per_ku * num_acc_n

                    def _flatten_b_tile(b_tile):
                        """Flatten B tile to a 1-D list for scf.for loop-carried state."""
                        flat = []
                        for ku_entry in b_tile:
                            # (packs_even, packs_odd) -> [even_0..N, odd_0..N]
                            flat.extend(ku_entry[0])
                            flat.extend(ku_entry[1])
                        return flat

                    def _unflatten_b_tile(vals):
                        """Reconstruct B tile from flattened scf.for loop-carried state."""
                        b_tile, idx = [], 0
                        for _ in range_constexpr(k_unroll):
                            packs_even = list(vals[idx:idx + num_acc_n])
                            idx += num_acc_n
                            packs_odd = list(vals[idx:idx + num_acc_n])
                            idx += num_acc_n
                            b_tile.append((packs_even, packs_odd))
                        return b_tile

                    init_state = (
                        list(acc_gate) + list(acc_up)
                        + _flatten_b_tile(b_gate_cur) + _flatten_b_tile(b_up_cur)
                        + list(a0_prefetch_pong)
                    )

                    _n_acc = m_repeat * num_acc_n
                    _p_bg = 2 * _n_acc
                    _p_bu = _p_bg + _vals_per_b_tile
                    _p_a0 = _p_bu + _vals_per_b_tile

                    for pair_iv, state in range(0, pair_iters, 1, init=init_state):
                        _ag = list(state[:_n_acc])
                        _au = list(state[_n_acc:_p_bg])
                        _bg = _unflatten_b_tile(list(state[_p_bg:_p_bu]))
                        _bu = _unflatten_b_tile(list(state[_p_bu:_p_a0]))
                        _a0pf = (state[_p_a0], state[_p_a0 + 1])

                        k_iv = k_base_idx + pair_iv * (c_tile_k + c_tile_k)

                        # ---- stage 0: prefetch+store ping, compute pong ----
                        next_k1 = k_iv + c_tile_k
                        x_regs_ping = load_x_tile(next_k1)
                        _bg_ping = load_b_tile(next_k1, n_blk_gate, n_intra_gate)
                        _bu_ping = load_b_tile(next_k1, n_blk_up, n_intra_up)

                        _ag, _au, _ = compute_tile(_ag, _au, _bg, _bu, lds_base_pong, a0_prefetch=_a0pf)
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()

                        _a0pf_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                        # ---- stage 1: prefetch+store pong, compute ping ----
                        next_k2 = k_iv + c_tile_k + c_tile_k
                        x_regs_pong = load_x_tile(next_k2)
                        _bg_next = load_b_tile(next_k2, n_blk_gate, n_intra_gate)
                        _bu_next = load_b_tile(next_k2, n_blk_up, n_intra_up)

                        _ag, _au, _ = compute_tile(_ag, _au, _bg_ping, _bu_ping, lds_base_ping, a0_prefetch=_a0pf_ping)
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()

                        _a0pf_new = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                        loop_results = yield (
                            list(_ag) + list(_au)
                            + _flatten_b_tile(_bg_next) + _flatten_b_tile(_bu_next)
                            + list(_a0pf_new)
                        )

                    # After scf.for: extract final state from yielded results.
                    SmemPtr._view_cache = None
                    if pair_iters > 0:
                        acc_gate = list(loop_results[:_n_acc])
                        acc_up = list(loop_results[_n_acc:_p_bg])
                        b_gate_cur = _unflatten_b_tile(list(loop_results[_p_bg:_p_bu]))
                        b_up_cur = _unflatten_b_tile(list(loop_results[_p_bu:_p_a0]))
                        a0_prefetch_pong = (loop_results[_p_a0], loop_results[_p_a0 + 1])
                    k_tail1 = k_base_idx + arith.index(_k_per_batch - tile_k)
                    x_regs_ping = load_x_tile(k_tail1)
                    b_gate_ping = load_b_tile(k_tail1, n_blk_gate, n_intra_gate)
                    b_up_ping = load_b_tile(k_tail1, n_blk_up, n_intra_up)

                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        b_gate_cur,
                        b_up_cur,
                        lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()

                    # Cross-tile prefetch for the final ping tile.
                    a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                    # Epilogue: compute last tile with epilogue scale prefetch to overlap loads with MFMA.
                    acc_gate, acc_up, epilogue_pf = compute_tile(
                        acc_gate,
                        acc_up,
                        b_gate_ping,
                        b_up_ping,
                        lds_base_ping,
                        prefetch_epilogue=True,
                        a0_prefetch=a0_prefetch_ping,
                    )

                # Store epilogue to out[t, slot, inter]
                expert_off = expert_off_idx
                bx_m0 = bx_m
                tokens_i32_v = tokens_i32
                topk_i32_v = topk_i32
                inter_i32_v = fx.Int32(inter_dim)
                mask24_i32 = fx.Int32(0xFFFFFF)

                if epilogue_pf is not None:
                    sw_gate_vals, sw_up_vals = epilogue_pf
                else:
                    sw_gate_vals = []
                    sw_up_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_gate_idx = expert_off + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_vals.append(
                            buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32)
                        )
                        sw_up_vals.append(
                            buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32)
                        )

                # Epilogue hoists to keep IR + Python build time small:
                col_i32_list = []
                for ni in range_constexpr(num_acc_n):
                    col_i32_list.append(arith.index_cast(T.i32, col_g_list[ni]))
    
                lane_div_16_mul4 = lane_div_16 * fx.Index(4)
                inter_i32_local = inter_i32_v
    
                # Uses EVec=4 (buffer store "x4" of fp16 elements).
                use_cshuffle_epilog_flag = _use_cshuffle_epilog

                # ─── Split-K epilogue: two-pass gate/up with f32 atomic fadd ───
                if _is_splitk:
                    if lds_out is None:
                        raise RuntimeError("Split-K epilogue requires lds_out (CShuffle)")

                    out_base_idx = buffer_ops.extract_base_index(arg_out)
                    _split_k_out_row_stride = inter_dim * 2 * out_elem_bytes  # bytes per row
                    _split_k_e_vec = 2  # f32 vec2 for atomic fadd

                    # Mutable slot: 0 for gate pass, inter_dim for up pass
                    _split_k_n_offset = [0]

                    # Mutable slots for two-pass gate/up selection
                    _split_k_acc = [acc_gate]
                    _split_k_sw_vals = [sw_gate_vals]

                    def write_row_to_lds_splitk(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        """Write scaled f32 partial sums to LDS (no silu, no doweight)."""
                        _acc = _split_k_acc[0]
                        _sw = _split_k_sw_vals[0]
                        # Load per-row scale_x (sx) — same logic as normal epilogue.
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        s2 = fused2 >> 24
                        ts2 = s2 * tokens_i32_v + t2
                        sx = arith.select(
                            t_valid,
                            buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32),
                            fx.Float32(0.0),
                        )
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                _acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            v = arith.sitofp(T.f32, v)
                            v = v * sx * _sw[ni]
                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(T.vec(1, T.f32), [v])
                            vector.store(v1, lds_out, [lds_idx], alignment=4)

                    def precompute_row_splitk(*, row_local, row):
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        t_idx = arith.index_cast(T.index, t2)
                        s_idx = arith.index_cast(T.index, s2)
                        ts_idx = t_idx * arith.index(topk) + s_idx
                        row_byte_base = out_base_idx + ts_idx * arith.index(_split_k_out_row_stride)
                        return (row_byte_base, t_ok)

                    def store_pair_splitk(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        row_byte_base = row_ctx
                        col_idx = col_g0 + arith.index(_split_k_n_offset[0])
                        byte_off_col = col_idx * arith.index(out_elem_bytes)
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr = buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                        out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=_split_k_e_vec * out_elem_bytes,
                        )

                    _cshuffle_nlane_splitk = min(32, tile_n // _split_k_e_vec)
                    _splitk_frag_elem = ir.F32Type.get()

                    # Pass 1: gate (offset=0)
                    _split_k_acc[0] = acc_gate
                    _split_k_sw_vals[0] = sw_gate_vals
                    _split_k_n_offset[0] = 0
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_split_k_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_splitk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_splitk_frag_elem,
                        write_row_to_lds=write_row_to_lds_splitk,
                        precompute_row=precompute_row_splitk,
                        store_pair=store_pair_splitk,
                    )

                    gpu.barrier()

                    # Pass 2: up (offset=inter_dim)
                    _split_k_acc[0] = acc_up
                    _split_k_sw_vals[0] = sw_up_vals
                    _split_k_n_offset[0] = inter_dim
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_split_k_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_splitk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_splitk_frag_elem,
                        write_row_to_lds=write_row_to_lds_splitk,
                        precompute_row=precompute_row_splitk,
                        store_pair=store_pair_splitk,
                    )
                    return

                if use_cshuffle_epilog_flag:
                    if lds_out is None:
                        raise RuntimeError("CShuffle epilogue enabled but lds_out is not allocated/aliased.")
    
                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        # `row` is the sorted-row index (bx_m + row_in_tile).
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        # aiter moe_sorting uses sentinel token_id == tokens for padding.
                        # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        # slot-major: slot*tokens + token
                        ts2 = s2 * tokens_i32_v + t2
                        sx = arith.select(
                            t_valid,
                            buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32),
                            fx.Float32(0.0),
                        )

                        # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                        if doweight_stage1:
                            tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32)

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni]

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            vu = vector.extract(
                                acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                            )

                            vg = arith.sitofp(T.f32, vg)
                            vu = arith.sitofp(T.f32, vu)
                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up

                            y = silu(vg) * vu
                            if doweight_stage1:
                                y = y * tw
                            y16 = arith.trunc_f(T.f16, y)

                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(T.vec(1, T.f16), [y16])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        return (t2 * topk_i32_v + s2) * inter_i32_local

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        # Guard against sentinel token ids (t == tokens) produced by aiter moe_sorting padding.
                        # OOB buffer stores are not guaranteed to be safe on all paths, so predicate explicitly.
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        _if_valid = scf.IfOp(t_valid)
                        with _if_then(_if_valid):
                            idx0 = row_ctx
                            col_i32 = arith.index_cast(T.i32, col_g0)
                            idx_out = idx0 + col_i32
                            # Vectorized fp16 store (EVec=4).
                            buffer_ops.buffer_store(frag, out_rsrc, idx_out)
                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=4,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                    return
    
                def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                    # `row` is the sorted-row index (bx_m + row_in_tile).
                    # Block-level early-exit already guards `bx_m` range.
                    # Here we rely on buffer OOB semantics for any tail rows.
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                    t2_raw = fused2 & mask24_i32
                    s2_raw = fused2 >> 24
                    t2 = t2_raw
                    s2 = s2_raw
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)

                    # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                    # slot-major: slot*tokens + token
                    ts2 = s2 * tokens_i32_v + t2
                    sx = arith.select(
                        t_valid,
                        buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32),
                        fx.Float32(0.0),
                    )
                    zero_out = arith.constant(0.0, type=out_mlir())

                    # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                    idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                    # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32)

                    _if_valid = scf.IfOp(t_valid)
                    with _if_then(_if_valid):
                        for ni in range_constexpr(num_acc_n):
                            col_i32 = col_i32_list[ni]
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni]

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            vu = vector.extract(
                                acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                            )

                            vg = arith.sitofp(T.f32, vg)
                            vu = arith.sitofp(T.f32, vu)
                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up

                            y = silu(vg) * vu
                            if doweight_stage1:
                                y = y * tw
                            y = arith.trunc_f(out_mlir(), y)
                            idx_out0 = idx0 + col_i32
                            buffer_ops.buffer_store(y, out_rsrc, idx_out0)
    
                mfma_epilog(
    
                    use_cshuffle=False,
                    arith=arith,
                    range_constexpr=range_constexpr,
                    m_repeat=m_repeat,
                    lane_div_16=lane_div_16,
                    bx_m=bx_m,
                    body_row=_stage1_store_row,
                )

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    @flyc.jit
    def launch_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_qscale_w: fx.Tensor,
        arg_qzero_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(
            T.index, i32_size_expert_ids_in
        )
        gx = inter_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_qscale_w,
            arg_qzero_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, k_batch),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm1


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    accumulate: bool = True,
    scale_is_bf16: bool = False,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    in_dtype: must be "a8w4smooth" (or its "uint4" alias). Other dtypes are rejected.
    group_size: accepted for ABI compat; ignored.
    scale_is_bf16: accepted for ABI compat; ignored.

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    # a8w4smooth-only entry point: refuse any other dtype at the API boundary.
    # ('uint4' is a backward-compatible alias for 'a8w4smooth'.)
    _valid_dtypes = ("a8w4smooth", "uint4")
    if in_dtype not in _valid_dtypes:
        raise ValueError(
            f"a8w4_moe_gemm_2stage only supports in_dtype in {_valid_dtypes}, got {in_dtype!r}. "
            f"For other dtypes, use kernels.moe_gemm_2stage."
        )
    is_uint4 = in_dtype == "uint4"
    elem_bytes = 1
    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}")
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError("compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}")

    # a8w4smooth: env validation
    _qparam_prefix = "FLIR_UINT4" if is_uint4 else "FLIR_A8W4SMOOTH"
    _qp_format = os.environ.get(f"{_qparam_prefix}_QPARAM_FORMAT", "packed4")
    if _qp_format != "packed4":
        raise ValueError(
            f"a8w4smooth requires {_qparam_prefix}_QPARAM_FORMAT=packed4, got {_qp_format!r}"
        )
    _interleave = os.environ.get(f"{_qparam_prefix}_INTERLEAVE_K64", "1")
    if _interleave not in ("1", "true", "True", "YES", "yes"):
        raise ValueError(
            f"a8w4smooth requires {_qparam_prefix}_INTERLEAVE_K64=1, got {_interleave!r}"
        )
    if (inter_dim % 256) != 0:
        raise ValueError(
            f"a8w4smooth (stage2) requires inter_dim % 256 == 0, got inter_dim={inter_dim}"
        )
    if int(tile_k) not in (128, 256):
        raise ValueError(
            f"a8w4smooth requires tile_k in {{128, 256}}, got tile_k={tile_k}"
        )
    _overflow_prefix = "FLIR_UINT4" if (in_dtype == "uint4") else "FLIR_A8W4SMOOTH"
    overflow_guard = os.environ.get(f"{_overflow_prefix}_OVERFLOW_GUARD", "1") in ("1", "true", "True", "YES", "yes")

    mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
        rocdl, "mfma_i32_16x16x32_i8", None
    )
    if mfma_i32_k32 is None:
        raise AttributeError(
            "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
            "(or `rocdl.mfma_i32_16x16x32_i8`)."
        )

    DYN = ir.ShapedType.get_dynamic_size()
    size_out = DYN
    size_x = DYN
    size_sorted = DYN
    size_expert_ids_shape = DYN
    size_scale_x = DYN
    # W is packed int4 for W4A8/W4A16/W4A_FP8: 2 values per byte.
    size_w = (experts * model_dim * inter_dim) // 2

    # a8w4smooth qparams (stage2): packed4 [E, model_dim//16, inter_dim//256, 16] of i32.
    rows_blk_stage2 = model_dim // 16
    num_k256_stage2 = inter_dim // 256
    size_qscale_w = experts * rows_blk_stage2 * num_k256_stage2 * 16
    size_qzero_w = size_qscale_w

    total_threads = 256
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    # gfx950+ has buffer_atomic_pk_add_bf16 → bf16 can use buffer atomics (same as f16).
    # gfx942 only has global_atomic_pk_add_bf16 → must use global atomics with raw pointer.
    _has_buffer_atomic_bf16 = str(gpu_arch).startswith(("gfx95", "gfx12"))
    _needs_global_atomic_bf16 = out_is_bf16 and not _has_buffer_atomic_bf16
    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics ({bf16_global_atomics_arch_description()}), got arch={gpu_arch!r}"
            )

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        if _use_cshuffle_epilog:
            raise ValueError("out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False).")
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get("FLYDSL_MOE_STAGE2_CSHUFFLE", "1") in (
                "1",
                "true",
                "True",
                "YES",
                "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1)."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    def out_elem():
        ty = T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)
        return ty() if callable(ty) else ty
    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    module_name = (
        f"mfma_moe2_{in_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi3"  # bumped: a8w4smooth qparam args added to kernel signature
    ).replace("-", "_")

    # ── CShuffle epilogue e_vec (pure Python; must be computed before @flyc.kernel
    # because the AST rewriter intercepts `if` statements inside kernel bodies and
    # turns them into closure dispatches, which breaks variable reassignment) ────
    _cshuffle_nlane = 32
    if bool(accumulate):
        _e_vec = 2
    else:
        _e_vec = 8 if int(tile_n) % (_cshuffle_nlane * 8) == 0 else 2
        _cshuffle_stride = _cshuffle_nlane * _e_vec
        if int(tile_n) % _cshuffle_stride != 0:
            raise ValueError(
                f"tile_n={tile_n} must be divisible by {_cshuffle_stride} when accumulate=False"
            )

    # ── LDS sizing (pure Python; no MLIR Context needed) ─────────────────────
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0  # f16 bytes
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if True:
        @flyc.kernel
        def moe_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_qscale_w: fx.Tensor,
            arg_qzero_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            n_in = arith.index_cast(T.index, i32_n_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(
                T.index, i32_size_expert_ids_in
            )
            # i32 versions for layout construction (fly.make_shape requires i32/i64)
            tokens_i32_v = i32_tokens_in
            k_i32_v = i32_k_in
            # a8w4smooth: X is int8, W is packed int4 (stored as i8) and unpacked in-kernel.
            x_elem = T.i8
            w_elem = T.i8
            vec16_elems = 16
            vec8_elems = 8
            vec4_elems = 4
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            acc_init = arith.constant_vector(0, T.i32x4)

            # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
            topk_idx = fx.Index(topk)
            m_in = tokens_in * topk_idx
            m_i32_v = arith.index_cast(T.i32, m_in)
            layout_x = fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.index(experts * model_dim)
            # For a8w4smooth (packed4 + K64-interleave), B is stored as packed-u4 with kpack_bytes=16.
            kpack_bytes = 16
            w_elem_bytes = 1
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=w_elem_bytes,
                packed_4bit=True,
            )
            layout_b = b_layout.layout_b
            c_k0 = (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along model_dim
            bx = gpu.block_id("y")  # tile along sorted M

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
            layout_lin_rowcol = fx.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                T.i8,
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as f16/bf16 for epilogue shuffle.
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = fx.Index(topk)

            # X(A2): [tokens*topk, inter_dim] bytes = tokens*topk*k*elem_bytes
            x_nbytes_idx = (tokens_in * c_topk) * k_in * arith.index(int(elem_bytes))
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_idx
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * fx.Index(out_elem_bytes)
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in
                    * fx.Index(topk)
                    * n_in
                    * fx.Index(out_elem_bytes)
                )
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_idx
            )
            # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
            sx_nbytes_idx = (tokens_in * c_topk) * fx.Index(4)
            sx_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx
            )
            # scale_w: [experts*model_dim] f32 (static shape in practice)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            # a8w4smooth qparams (stage2): packed4 [E, model_dim//16, inter_dim//256, 16] of i32.
            qs_rsrc = buffer_ops.create_buffer_resource(arg_qscale_w, max_size=False)
            qz_rsrc = buffer_ops.create_buffer_resource(arg_qzero_w, max_size=False)

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = (
                size_expert_ids_in
                * fx.Index(tile_m)
                * fx.Index(4)
            )
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes_idx
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_idx
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * fx.Index(4)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_idx
            )
            bx_m = bx * fx.Index(tile_m)

            # Early-exit guard (as in 2ce65fb): some routing paths can produce extra/garbage
            # expert blocks beyond `num_valid_ids`. Skip those blocks entirely to avoid OOB.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=fx.Index(4),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
            )
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
                expert_idx = arith.index_cast(T.index, expert_i32)
                n_idx = fx.Index(model_dim)
                expert_off_idx = expert_idx * n_idx  # index
    
                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads.
                if bytes_per_thread_x % 16 == 0:
                    x_load_bytes = 16
                elif bytes_per_thread_x % 8 == 0:
                    x_load_bytes = 8
                elif bytes_per_thread_x % 4 == 0:
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                    )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                layout_x_div4 = fx.make_layout((m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1))
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
    
                topk_i32 = fx.Int32(topk)
                mask24 = fx.Int32(0xFFFFFF)
                # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                vec4_x = T.vec(4, x_elem)
    
                def load_x(idx_i32):
                    if x_load_bytes == 16:
                        idx_elem = idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                            elem_bytes=elem_bytes,
                        )
                    if x_load_bytes == 8:
                        return buffer_ops.buffer_load(x_rsrc, idx_i32, vec_width=2, dtype=T.i32)
                    return buffer_ops.buffer_load(x_rsrc, idx_i32, vec_width=1, dtype=T.i32)

                # decode routed token once (per thread's M-slice) and build a base offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)
    
                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                    t_i32 = fused_i & mask24
                    s_i32 = fused_i >> 24
                    # aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for A2/scale loads; explicitly mask.
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = t_valid & s_valid
                    t_safe = ts_valid.select(t_i32, fx.Int32(0))
                    s_safe = ts_valid.select(s_i32, fx.Int32(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)
    
                def load_x_tile(base_k):
                    base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(vector.bitcast(T.vec(2, T.i32), x_vec))
                        else:
                            parts.append(vector.bitcast(T.vec(1, T.i32), x_vec))
                    return parts
    
                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)
    
                row_a_lds = lane_mod_16
                # A-side kpack is always 16 bytes; kpack_bytes is B-side (may be 8 for int4).
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.index(int(elem_bytes)))
                )
    
                # Dynamic N tiling within block.
                by_n = by * fx.Index(tile_n)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_mod_4 = wave_id % fx.Index(4)
                n_tile_base = wave_mod_4 * c_n_per_wave
    
                # Precompute (n_blk, n_intra) for B, and col indices for output.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0 = c_n_total // fx.Index(16)
                c_n0_static = experts * model_dim // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base + offset + lane_mod_16
                    col_g_list.append(col_g)
    
                    row_w = expert_off_idx + col_g
                    coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(fx.get(coord_w, 0))
                    n_intra_list.append(fx.get(coord_w, 1))
    
                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)
    
                # A8W4SMOOTH packed4 qparams (stage2): used by load_b_tile_a8w4smooth only.
                c_rows_blk_s2 = fx.Index(model_dim // 16)
                c_num_k256_s2 = fx.Index(inter_dim // 256)

                # --- A8W4SMOOTH helpers (stage2 - capture n_blk_list/n_intra_list from outer scope) ---
                def load_b_tile_a8w4smooth(base_k):
                    """A8W4SMOOTH tile-k=256 loader (drop-in for load_b_tile)."""
                    if int(tile_k) != 256:
                        raise ValueError(f"a8w4smooth tile loader requires tile_k==256, got {tile_k!r}")
                    if int(kpack_bytes) != 16:
                        raise ValueError(f"a8w4smooth tile loader requires kpack_bytes==16, got {kpack_bytes!r}")

                    i32_ty = T.i32
                    vec1_i64_ty = T.vec(1, T.i64)
                    vec2_i32_ty = T.vec(2, T.i32)
                    vec4_i32_ty = T.i32x4

                    c16 = fx.Index(16)
                    c256 = fx.Index(256)
                    c2 = fx.Index(2)
                    c64 = fx.Index(64)
                    c4 = fx.Index(4)

                    k256 = base_k // c256

                    qs_word_list = []
                    qz_word_list = []
                    for ni in range_constexpr(num_acc_n):
                        n_blk_global = n_blk_list[ni]
                        n_lane = n_intra_list[ni]
                        expert_id = n_blk_global // c_rows_blk_s2
                        n_blk_local = n_blk_global - (expert_id * c_rows_blk_s2)
                        qs_idx = (
                            ((((expert_id * c_rows_blk_s2) + n_blk_local) * c_num_k256_s2) + k256)
                            * c16
                            + n_lane
                        )
                        qs_word_list.append(buffer_ops.buffer_load(qs_rsrc, qs_idx, vec_width=1, dtype=T.i32))
                        qz_word_list.append(buffer_ops.buffer_load(qz_rsrc, qs_idx, vec_width=1, dtype=T.i32))

                    base_k_packed_bytes = base_k // c2
                    k0_base = base_k_packed_bytes // c64

                    c_ff = arith.constant(0x000000FF, type=i32_ty)
                    c_sign_flip = arith.constant(0x80808080, type=i32_ty)
                    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=i32_ty)
                    c_zero_i32 = arith.constant(0, type=i32_ty)

                    def _dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte):
                        dwords = [d0, d1, d2, d3]
                        even_outs = []
                        odd_outs = []
                        for di in range_constexpr(4):
                            dw = dwords[di]
                            even = dw & c_0f0f0f0f
                            odd = (dw >> 4) & c_0f0f0f0f
                            if not bool(overflow_guard):
                                even_out = ((even * qs_lo) + qz_lo) ^ c_sign_flip
                                odd_out = ((odd * qs_hi) + qz_hi) ^ c_sign_flip
                            else:
                                c_255 = arith.constant(255, type=i32_ty)

                                def _clamp_u8(x):
                                    gt = arith.cmpi(arith.CmpIPredicate.ugt, x, c_255)
                                    return gt.select(c_255, x)

                                def _dequant_safe_single(v, qs, qz_byte):
                                    b0 = v & c_ff
                                    b1 = (v >> 8) & c_ff
                                    b2 = (v >> 16) & c_ff
                                    b3 = (v >> 24) & c_ff
                                    o0 = _clamp_u8((b0 * qs) + qz_byte)
                                    o1 = _clamp_u8((b1 * qs) + qz_byte)
                                    o2 = _clamp_u8((b2 * qs) + qz_byte)
                                    o3 = _clamp_u8((b3 * qs) + qz_byte)
                                    out = o0 | (o1 << 8) | (o2 << 16) | (o3 << 24)
                                    return out ^ c_sign_flip

                                even_out = _dequant_safe_single(even, qs_lo, qz_lo_byte)
                                odd_out = _dequant_safe_single(odd, qs_hi, qz_hi_byte)
                            even_outs.append(even_out)
                            odd_outs.append(odd_out)
                        return even_outs + odd_outs

                    def _pair_as_i64(a, b):
                        v2 = vector.from_elements(vec2_i32_ty, [a, b])
                        return vector.extract(vector.bitcast(vec1_i64_ty, v2), static_position=[0], dynamic_position=[])

                    b_tile = []
                    for pack_group in range_constexpr(k_unroll // 2):
                        k0 = k0_base + fx.Index(pack_group)

                        w_i32x4_list = []
                        for ni in range_constexpr(num_acc_n):
                            coord_pack = (n_blk_list[ni], k0, lane_div_16, n_intra_list[ni], fx.Index(0))
                            idx_pack = crd2idx(coord_pack, layout_b)
                            b16 = _buffer_load_vec(
                                buffer_ops, vector, w_rsrc, idx_pack,
                                elem_type=w_elem, vec_elems=16, elem_bytes=1,
                                offset_in_bytes=True,
                            )
                            w_i32x4_list.append(vector.bitcast(vec4_i32_ty, b16))

                        if pack_group == 0:
                            perm_lo = arith.constant(0x00000000, type=i32_ty)
                            perm_hi = arith.constant(0x01010101, type=i32_ty)
                        else:
                            perm_lo = arith.constant(0x02020202, type=i32_ty)
                            perm_hi = arith.constant(0x03030303, type=i32_ty)

                        out8_list = []
                        for ni in range_constexpr(num_acc_n):
                            qs_word = qs_word_list[ni]
                            qz_word = qz_word_list[ni]
                            if pack_group == 0:
                                qs_lo = qs_word & c_ff
                                qs_hi = (qs_word >> 8) & c_ff
                                qz_lo_byte = qz_word & c_ff
                                qz_hi_byte = (qz_word >> 8) & c_ff
                            else:
                                qs_lo = (qs_word >> 16) & c_ff
                                qs_hi = (qs_word >> 24) & c_ff
                                qz_lo_byte = (qz_word >> 16) & c_ff
                                qz_hi_byte = (qz_word >> 24) & c_ff
                            qz_lo = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_lo], [], [])
                            qz_hi = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_hi], [], [])
                            w_i32x4 = w_i32x4_list[ni]
                            d0 = vector.extract(w_i32x4, static_position=[0], dynamic_position=[])
                            d1 = vector.extract(w_i32x4, static_position=[1], dynamic_position=[])
                            d2 = vector.extract(w_i32x4, static_position=[2], dynamic_position=[])
                            d3 = vector.extract(w_i32x4, static_position=[3], dynamic_position=[])
                            out8_list.append(
                                _dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte)
                            )

                        packs0_lo = []
                        packs1_lo = []
                        for ni in range_constexpr(num_acc_n):
                            out8 = out8_list[ni]
                            packs0_lo.append(_pair_as_i64(out8[0], out8[1]))
                            packs1_lo.append(_pair_as_i64(out8[2], out8[3]))
                        b_tile.append((packs0_lo, packs1_lo))

                        packs0_hi = []
                        packs1_hi = []
                        for ni in range_constexpr(num_acc_n):
                            out8 = out8_list[ni]
                            packs0_hi.append(_pair_as_i64(out8[4], out8[5]))
                            packs1_hi.append(_pair_as_i64(out8[6], out8[7]))
                        b_tile.append((packs0_hi, packs1_hi))

                    return b_tile

                # --- A8W4SMOOTH tile_k=128 helpers (stage2) ---
                def preload_qparams_tilek128(base_k_256):
                    """Stage2 tile_k=128 qparams loader: load qparams once per K256 chunk."""
                    c16 = fx.Index(16)
                    c256 = fx.Index(256)
                    k256 = base_k_256 // c256
                    qs_word_list = []
                    qz_word_list = []
                    for ni in range_constexpr(num_acc_n):
                        n_blk_global = n_blk_list[ni]
                        n_lane = n_intra_list[ni]
                        expert_id = n_blk_global // c_rows_blk_s2
                        n_blk_local = n_blk_global - (expert_id * c_rows_blk_s2)
                        qs_idx = (
                            ((((expert_id * c_rows_blk_s2) + n_blk_local) * c_num_k256_s2) + k256)
                            * c16
                            + n_lane
                        )
                        qs_word_list.append(buffer_ops.buffer_load(qs_rsrc, qs_idx, vec_width=1, dtype=T.i32))
                        qz_word_list.append(buffer_ops.buffer_load(qz_rsrc, qs_idx, vec_width=1, dtype=T.i32))
                    return qs_word_list, qz_word_list

                def load_b_half_tilek128_raw(pack_group, qs_word_list, qz_word_list, base_k_256):
                    """Stage2 tile_k=128 half loader: load one raw 128B packed half + qparams metadata."""
                    vec4_i32_ty = T.i32x4
                    c2 = fx.Index(2)
                    c64 = fx.Index(64)
                    base_k_packed_bytes = base_k_256 // c2
                    k0_base = base_k_packed_bytes // c64

                    k0 = k0_base + fx.Index(pack_group)
                    w_i32x4_list = []
                    for ni in range_constexpr(num_acc_n):
                        coord_pack = (n_blk_list[ni], k0, lane_div_16, n_intra_list[ni], fx.Index(0))
                        idx_pack = crd2idx(coord_pack, layout_b)
                        b16 = _buffer_load_vec(
                            buffer_ops, vector, w_rsrc, idx_pack,
                            elem_type=w_elem, vec_elems=16, elem_bytes=1,
                            offset_in_bytes=True,
                        )
                        w_i32x4_list.append(vector.bitcast(vec4_i32_ty, b16))

                    return (qs_word_list, qz_word_list, w_i32x4_list)

                def dequant_b_half_tilek128(raw_half, pack_group):
                    """Stage2 tile_k=128 helper: dequantize one raw 128B packed half before compute."""
                    qs_word_list, qz_word_list, w_i32x4_list = raw_half
                    i32_ty = T.i32
                    vec1_i64_ty = T.vec(1, T.i64)
                    vec2_i32_ty = T.vec(2, T.i32)

                    c_ff = arith.constant(0x000000FF, type=i32_ty)
                    c_sign_flip = arith.constant(0x80808080, type=i32_ty)
                    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=i32_ty)
                    c_zero_i32 = arith.constant(0, type=i32_ty)

                    if pack_group == 0:
                        perm_lo = arith.constant(0x00000000, type=i32_ty)
                        perm_hi = arith.constant(0x01010101, type=i32_ty)
                    else:
                        perm_lo = arith.constant(0x02020202, type=i32_ty)
                        perm_hi = arith.constant(0x03030303, type=i32_ty)

                    qparam_list = []
                    for ni in range_constexpr(num_acc_n):
                        qs_word = qs_word_list[ni]
                        qz_word = qz_word_list[ni]
                        if pack_group == 0:
                            qs_lo = qs_word & c_ff
                            qs_hi = (qs_word >> 8) & c_ff
                            qz_lo_byte = qz_word & c_ff
                            qz_hi_byte = (qz_word >> 8) & c_ff
                        else:
                            qs_lo = (qs_word >> 16) & c_ff
                            qs_hi = (qs_word >> 24) & c_ff
                            qz_lo_byte = (qz_word >> 16) & c_ff
                            qz_hi_byte = (qz_word >> 24) & c_ff
                        qz_lo = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_lo], [], [])
                        qz_hi = llvm.call_intrinsic(i32_ty, "llvm.amdgcn.perm", [c_zero_i32, qz_word, perm_hi], [], [])
                        qparam_list.append((qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte))

                    def _dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte):
                        dwords = [d0, d1, d2, d3]
                        even_outs = []
                        odd_outs = []
                        for di in range_constexpr(4):
                            dw = dwords[di]
                            even = dw & c_0f0f0f0f
                            odd = (dw >> 4) & c_0f0f0f0f
                            if not bool(overflow_guard):
                                even_out = ((even * qs_lo) + qz_lo) ^ c_sign_flip
                                odd_out = ((odd * qs_hi) + qz_hi) ^ c_sign_flip
                            else:
                                c_255 = arith.constant(255, type=i32_ty)

                                def _clamp_u8(x):
                                    gt = arith.cmpi(arith.CmpIPredicate.ugt, x, c_255)
                                    return gt.select(c_255, x)

                                def _dequant_safe_single(v, qs, qz_byte):
                                    b0 = v & c_ff
                                    b1 = (v >> 8) & c_ff
                                    b2 = (v >> 16) & c_ff
                                    b3 = (v >> 24) & c_ff
                                    o0 = _clamp_u8((b0 * qs) + qz_byte)
                                    o1 = _clamp_u8((b1 * qs) + qz_byte)
                                    o2 = _clamp_u8((b2 * qs) + qz_byte)
                                    o3 = _clamp_u8((b3 * qs) + qz_byte)
                                    out = o0 | (o1 << 8) | (o2 << 16) | (o3 << 24)
                                    return out ^ c_sign_flip

                                even_out = _dequant_safe_single(even, qs_lo, qz_lo_byte)
                                odd_out = _dequant_safe_single(odd, qs_hi, qz_hi_byte)
                            even_outs.append(even_out)
                            odd_outs.append(odd_out)
                        return even_outs + odd_outs

                    def _pair_as_i64(a, b):
                        v2 = vector.from_elements(vec2_i32_ty, [a, b])
                        return vector.extract(vector.bitcast(vec1_i64_ty, v2), static_position=[0], dynamic_position=[])

                    out8_list = []
                    for ni in range_constexpr(num_acc_n):
                        qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte = qparam_list[ni]
                        w_i32x4 = w_i32x4_list[ni]
                        d0 = vector.extract(w_i32x4, static_position=[0], dynamic_position=[])
                        d1 = vector.extract(w_i32x4, static_position=[1], dynamic_position=[])
                        d2 = vector.extract(w_i32x4, static_position=[2], dynamic_position=[])
                        d3 = vector.extract(w_i32x4, static_position=[3], dynamic_position=[])
                        out8_list.append(_dequant_4pack(d0, d1, d2, d3, qs_lo, qs_hi, qz_lo, qz_hi, qz_lo_byte, qz_hi_byte))

                    packs0_lo = []
                    packs1_lo = []
                    for ni in range_constexpr(num_acc_n):
                        out8 = out8_list[ni]
                        packs0_lo.append(_pair_as_i64(out8[0], out8[1]))
                        packs1_lo.append(_pair_as_i64(out8[2], out8[3]))

                    packs0_hi = []
                    packs1_hi = []
                    for ni in range_constexpr(num_acc_n):
                        out8 = out8_list[ni]
                        packs0_hi.append(_pair_as_i64(out8[4], out8[5]))
                        packs1_hi.append(_pair_as_i64(out8[6], out8[7]))

                    b_half = []
                    b_half.append((packs0_lo, packs1_lo))
                    b_half.append((packs0_hi, packs1_hi))
                    return b_half

                # A8W4SMOOTH dispatch: load_b_tile is the packed-4bit dequant loader.
                load_b_tile = load_b_tile_a8w4smooth

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                            )
                        else:
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                            )
    
                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                def compute_tile(acc_in, b_tile_in, lds_base, *, prefetch_epilogue: bool = False, a0_prefetch=None):
                    acc_list = list(acc_in)
                    mfma_res_ty = T.i32x4
                    mfma_fn = mfma_i32_k32

                    epilogue_pf = None
                    if prefetch_epilogue:
                        expert_off_pf = expert_off_idx
                        sw_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_w_idx = expert_off_pf + col_g
                            sw_pf.append(
                                buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32)
                            )
                        # Also prefetch per-row routed/topk weights (sorted_weights) when enabled.
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * fx.Index(4)
                            ii_idx_list_pf = [fx.Index(ii) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.index(mi * 16)
                                for ii in range_constexpr(4):
                                    row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc, sorted_row_pf, vec_width=1, dtype=T.f32
                                        )
                                    )
                        epilogue_pf = (sw_pf, tw_pf)

                    def mfma_k64(acc0, a0, a1, b0, b1):
                        acc1 = mfma_fn(mfma_res_ty, [a0, b0, acc0, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc1, 0, 0, 0])

                    for ku in range_constexpr(k_unroll):
                        b_packs0, b_packs1 = b_tile_in[ku]
                        ki64 = arith.index(ku * 64)
                        col_base = col_offset_base_bytes + ki64

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.index(mi * 16)
                            curr_row_a_lds = row_a_lds + mi_val

                            if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)

                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni
                                acc_list[acc_idx] = mfma_k64(
                                    acc_list[acc_idx],
                                    a0,
                                    a1,
                                    b_packs0[ni],
                                    b_packs1[ni],
                                )
                    return acc_list, epilogue_pf
    
                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.index(tile_m * lds_stride)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems
    
                rocdl.sched_barrier(0)
    
                def hot_loop_scheduler():
                    rocdl.sched_barrier(0)

                def compute_tile_a8w4smooth_tilek128(
                    acc_in, b_half, lds_base, *, prefetch_epilogue: bool = False, a0_prefetch=None,
                ):
                    """Stage2 tile_k=128 compute step: forwards a single half-tile (K128) to compute_tile."""
                    return compute_tile(
                        acc_in, b_half, lds_base,
                        prefetch_epilogue=prefetch_epilogue, a0_prefetch=a0_prefetch,
                    )

                if const_expr(int(tile_k) == 128):
                    # ─── Stage2 A8W4SMOOTH tile_k=128 ping-pong ─────────────────────────
                    # Mirrors stage1's port: K256-chunk main loop is Python-unrolled, with
                    # K128 ping-pong inside each chunk. Single accumulator (no gate/up split).
                    c128 = fx.Index(128)
                    c256 = fx.Index(256)

                    # Prologue: preload qparams + raw low(K128) of chunk0, then store X tile.
                    k0 = fx.Index(0)
                    x_regs0 = load_x_tile(k0)
                    qs_v, qz_v = preload_qparams_tilek128(k0)
                    raw_pong = load_b_half_tilek128_raw(0, qs_v, qz_v, k0)
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                    gpu.barrier()

                    acc = [acc_init] * (num_acc_n * m_repeat)
                    lds_base_pong = lds_base_cur
                    lds_base_ping = lds_base_nxt

                    a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                    # Main K256 chunk loop, Python-unrolled.
                    num_main_iters_k256 = max((int(inter_dim) // 256) - 1, 0)
                    for iter_idx in range_constexpr(num_main_iters_k256):
                        k_iv = fx.Index(iter_idx * 256)

                        # ---- stage 0: Load High(N), Compute Low(N) ----
                        next_k1 = k_iv + c128
                        x_regs_ping = load_x_tile(next_k1)
                        raw_ping = load_b_half_tilek128_raw(1, qs_v, qz_v, k_iv)
                        b_pong = dequant_b_half_tilek128(raw_pong, 0)
                        acc, _ = compute_tile_a8w4smooth_tilek128(
                            acc, b_pong, lds_base_pong, a0_prefetch=a0_prefetch_pong,
                        )
                        a0_prefetch_pong = None
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                        # ---- stage 1: Preload QParams(N+1), Load Low(N+1), Compute High(N) ----
                        next_k2 = k_iv + c256
                        x_regs_pong = load_x_tile(next_k2)
                        qs_v, qz_v = preload_qparams_tilek128(next_k2)
                        raw_pong = load_b_half_tilek128_raw(0, qs_v, qz_v, next_k2)
                        b_ping = dequant_b_half_tilek128(raw_ping, 1)
                        acc, _ = compute_tile_a8w4smooth_tilek128(
                            acc, b_ping, lds_base_ping, a0_prefetch=a0_prefetch_ping,
                        )
                        a0_prefetch_ping = None
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                    # Tail: final K256 chunk = low128 (compute pong) + high128 (epilogue ping).
                    k_tail0 = fx.Index(int(inter_dim) - 256)
                    k_tail1 = fx.Index(int(inter_dim) - 128)
                    x_regs_ping = load_x_tile(k_tail1)
                    raw_ping = load_b_half_tilek128_raw(1, qs_v, qz_v, k_tail0)
                    b_pong = dequant_b_half_tilek128(raw_pong, 0)
                    acc, _ = compute_tile_a8w4smooth_tilek128(
                        acc, b_pong, lds_base_pong, a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()

                    a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)
                    b_ping = dequant_b_half_tilek128(raw_ping, 1)
                    acc, epilogue_pf = compute_tile_a8w4smooth_tilek128(
                        acc, b_ping, lds_base_ping,
                        prefetch_epilogue=True, a0_prefetch=a0_prefetch_ping,
                    )
                else:
                    # Prologue.
                    k0 = fx.Index(0)
                    x_regs0 = load_x_tile(k0)
                    b_cur = load_b_tile(k0)
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                    gpu.barrier()

                    acc = [acc_init] * (num_acc_n * m_repeat)
                    lds_base_pong = lds_base_cur
                    lds_base_ping = lds_base_nxt

                    # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                    # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                    a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                    # Main loop: process K tiles in 2-tile ping-pong steps.
                    #
                    # IMPORTANT: for odd number of K tiles, leave **1** tail tile; for even, leave **2**.
                    # Otherwise the 2-tile tail below would double-count the last tile when num_tiles is odd
                    # (e.g. inter_dim=192, tile_k=64 -> 3 tiles).
                    num_k_tiles_py = int(inter_dim) // int(tile_k)
                    odd_k_tiles = (num_k_tiles_py % 2) == 1
                    tail_tiles = 1 if odd_k_tiles else 2
                    k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                    if k_main2_py < 0:
                        k_main2_py = 0

                    c2_tile_k = arith.index(tile_k * 2)
                    c_tile_k_s2 = arith.index(tile_k)
                    pair_iters = k_main2_py // (int(tile_k) * 2)

                    # B-tile data layout per k_unroll entry:
                    #   fp8/int8 default — two register packs per ku, flattened as
                    #   [even_0..N, odd_0..N] -> 2 * num_acc_n values.
                    _fields_per_ku = 2
                    _vals_per_b_tile = k_unroll * _fields_per_ku * num_acc_n
                    _n_acc = m_repeat * num_acc_n
                    _p_b = _n_acc
                    _p_a0 = _p_b + _vals_per_b_tile

                    def _flatten_b_tile(b_tile):
                        """Flatten B tile to a 1-D list for scf.for loop-carried state."""
                        flat = []
                        for ku_entry in b_tile:
                            flat.extend(ku_entry[0])
                            flat.extend(ku_entry[1])
                        return flat

                    def _unflatten_b_tile(vals):
                        """Reconstruct B tile from flattened scf.for loop-carried state."""
                        b_tile, idx = [], 0
                        for _ in range_constexpr(k_unroll):
                            packs_even = list(vals[idx:idx + num_acc_n])
                            idx += num_acc_n
                            packs_odd = list(vals[idx:idx + num_acc_n])
                            idx += num_acc_n
                            b_tile.append((packs_even, packs_odd))
                        return b_tile

                    init_state = list(acc) + _flatten_b_tile(b_cur) + list(a0_prefetch_pong)

                    for pair_iv, state in range(0, pair_iters, 1, init=init_state):
                        _ac = list(state[:_n_acc])
                        _bc = _unflatten_b_tile(list(state[_p_b:_p_a0]))
                        _a0 = (state[_p_a0], state[_p_a0 + 1])

                        k_iv = pair_iv * (c_tile_k_s2 + c_tile_k_s2)

                        next_k1 = k_iv + c_tile_k_s2
                        x_regs_ping = load_x_tile(next_k1)
                        _bp = load_b_tile(next_k1)

                        _ac, _ = compute_tile(_ac, _bc, lds_base_pong, a0_prefetch=_a0)
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()

                        _a0p = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                        next_k2 = k_iv + c_tile_k_s2 + c_tile_k_s2
                        x_regs_pong = load_x_tile(next_k2)
                        _bn = load_b_tile(next_k2)

                        _ac, _ = compute_tile(_ac, _bp, lds_base_ping, a0_prefetch=_a0p)
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()

                        _a0n = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                        loop_results = yield list(_ac) + _flatten_b_tile(_bn) + list(_a0n)

                    SmemPtr._view_cache = None
                    if pair_iters > 0:
                        acc = list(loop_results[:_n_acc])
                        b_cur = _unflatten_b_tile(list(loop_results[_p_b:_p_a0]))
                        a0_prefetch_pong = (loop_results[_p_a0], loop_results[_p_a0 + 1])

                    if odd_k_tiles:
                        acc, epilogue_pf = compute_tile(
                            acc,
                            b_cur,
                            lds_base_pong,
                            prefetch_epilogue=True,
                            a0_prefetch=a0_prefetch_pong,
                        )
                    else:
                        k_tail1 = k_in - tile_k
                        x_regs_ping = load_x_tile(k_tail1)
                        b_ping = load_b_tile(k_tail1)

                        acc, _ = compute_tile(acc, b_cur, lds_base_pong, a0_prefetch=a0_prefetch_pong)
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)
                        acc, epilogue_pf = compute_tile(
                            acc, b_ping, lds_base_ping, prefetch_epilogue=True, a0_prefetch=a0_prefetch_ping
                        )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.
                expert_off = expert_off_idx
                mask24_i32 = fx.Int32(0xFFFFFF)
                model_i32 = fx.Int32(model_dim)
                topk_i32_v = topk_i32

                zero_i32 = fx.Int32(0)
                c2_i32 = fx.Int32(2)  # 2B element size for f16/bf16
                mask_even_i32 = fx.Int32(0xFFFFFFFE)  # align element index to even for half2 atomics

                e_vec = _e_vec

                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

                sw_pf = None
                tw_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf = epilogue_pf
    
                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if sw_pf is not None:
                    sw_vals = sw_pf
                else:
                    sw_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_w_idx = expert_off + col_g
                        sw_vals.append(
                            buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32)
                        )

                if out_is_f32:
                    # origin/dev_a16w4: f32 output uses scalar f32 atomics and skips CShuffle/LDS.
                    c4_i32 = fx.Int32(4)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32,
                            out_rsrc,
                            byte_off_i32,
                            zero_i32,
                            zero_i32,
                        )

                    def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24

                        # Mask sentinel (token_id==tokens, slot==topk) to avoid OOB scale_x loads.
                        # For invalid rows, force sx=0 so they contribute exactly 0 to output.
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_ok = t_ok & s_ok
                        t2_safe = ts_ok.select(t2, fx.Int32(0))
                        s2_safe = ts_ok.select(s2, fx.Int32(0))
                        ts2 = t2_safe * topk_i32_v + s2_safe
                        sx = arith.select(
                            ts_ok,
                            buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32),
                            fx.Float32(0.0),
                        )

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = ts_ok.select(tw_pf[tw_idx], fx.Float32(0.0))
                            else:
                                tw = arith.select(
                                    ts_ok,
                                    buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32),
                                    fx.Float32(0.0),
                                )

                        idx0 = t2_safe * model_i32  # i32 element index base (safe for sentinel rows)

                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            sw = sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                            v = arith.sitofp(T.f32, v)
                            v = v * sx * sw
                            if doweight_stage2:
                                v = v * tw
                            col_i32 = arith.index_cast(T.i32, col_g)
                            idx_elem = idx0 + col_i32
                            byte_off = idx_elem * c4_i32
                            atomic_add_f32(v, byte_off)

                    default_epilog(
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage2_row_atomic,
                    )
                else:
                    if lds_out is None:
                        raise RuntimeError(
                            "FLYDSL_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                        )

                    # For bf16 global atomics (gfx942 only), precompute the output base address.
                    # gfx950+ has buffer_atomic_pk_add_bf16, so bf16 uses buffer atomics there.
                    out_base_idx = None
                    if _needs_global_atomic_bf16:
                        out_base_idx = buffer_ops.extract_base_index(arg_out)

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        # Explicitly mask sentinel token/slot to avoid OOB scale_x loads.
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_ok = t_ok & s_ok
                        t2_safe = ts_ok.select(t2, fx.Int32(0))
                        s2_safe = ts_ok.select(s2, fx.Int32(0))
                        ts2 = t2_safe * topk_i32_v + s2_safe
                        sx = arith.select(
                            ts_ok,
                            buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32),
                            fx.Float32(0.0),
                        )

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                                )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw = sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                            v = arith.sitofp(T.f32, v)
                            v = v * sx * sw
                            if doweight_stage2:
                                v = v * tw
                            v_out = arith.trunc_f(out_elem(), v)

                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        # Precompute row context for cshuffle stores.
                        # Return (fused_i32, row_valid_i1) so the epilogue can skip the entire row
                        # for invalid tail rows (CK-style), avoiding per-store branching.
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = row_valid0 & t_ok & s_ok
                        return (fused2, row_valid)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused = row_ctx
                        t = fused & mask24_i32
                        s = fused >> 24
                        idx0 = t * model_i32
                        if not bool(accumulate):
                            ts = t * topk_i32_v + s
                            idx0 = ts * model_i32
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if _needs_global_atomic_bf16:
                            # gfx942: no buffer_atomic_pk_add_bf16, use global atomicrmw fadd
                            if bool(accumulate):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                                out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                                frag_v = frag._value if hasattr(frag, "_value") else frag
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    out_ptr_v,
                                    frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent",
                                    alignment=4,
                                )
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)
                        else:
                            # f16, or bf16 on gfx950+ (has buffer_atomic_pk_add_bf16)
                            byte_off = idx_elem_even * c2_i32
                            if bool(accumulate):
                                atomic_add_f16x2(frag, byte_off)
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)

                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=e_vec,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(T.bf16 if out_is_bf16 else T.f16),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
    
            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                _moe_gemm2_then_body()

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    @flyc.jit
    def launch_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_qscale_w: fx.Tensor,
        arg_qzero_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(
            T.index, i32_size_expert_ids_in
        )
        gx = n_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_qscale_w,
            arg_qzero_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm2


# MoE Reduction Kernel (reduce sum over topk dimension)
@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
            topk_ids [tokens, topk] i32 (optional, if use_mask=True)
            expert_mask [total_experts] i32 (optional, if use_mask=True)
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum(X[t, :, d]) for all t, d.
    When use_mask=True, only sums slots where expert_mask[topk_ids[t, k]] != 0.
    Used in conjunction with compile_moe_gemm2(accumulate=False) to avoid atomic contention.
    """
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8
    USE_NONTEMPORAL = True
    VEC_ALIGN = 16

    masked = "masked" if use_mask else ""

    if dtype_str == "f32":
        elem_type_tag = "f32"
    elif dtype_str == "f16":
        elem_type_tag = "f16"
    elif dtype_str == "bf16":
        elem_type_tag = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    compute_type = lambda: T.f32
    i32_type = lambda: T.i32
    i8_type = lambda: T.i8

    def elem_type():
        ty = T.f32 if elem_type_tag == "f32" else (T.f16 if elem_type_tag == "f16" else T.bf16)
        return ty() if callable(ty) else ty

    if True:
        @flyc.kernel
        def moe_reduction_kernel(
            X: fx.Tensor,
            Y: fx.Tensor,
            topk_ids: fx.Tensor,
            expert_mask: fx.Tensor,
            i32_m_tokens: fx.Int32,
        ):
            m_tokens = fx.Index(i32_m_tokens)
            c_topk = fx.Index(topk)
            c_model_dim = fx.Index(model_dim)
            topk_ids_nbytes_idx = m_tokens * c_topk * fx.Index(4)
            elem_bits = 32 if dtype_str == "f32" else 16
            copy_vec_width = 128 // elem_bits  # 8 for f16/bf16, 4 for f32
            n_sub = VEC_WIDTH // copy_vec_width  # 1 for f16/bf16, 2 for f32
            # Buffer-backed tensors via layout API (all dtypes)
            X_buf = fx.rocdl.make_buffer_tensor(X)
            Y_buf = fx.rocdl.make_buffer_tensor(Y)
            # Scalar buffer resources for tail path and mask
            x_rsrc = buffer_ops.create_buffer_resource(X, max_size=True)
            y_rsrc = buffer_ops.create_buffer_resource(Y, max_size=True)
            topk_ids_rsrc = buffer_ops.create_buffer_resource(
                topk_ids, max_size=False, num_records_bytes=topk_ids_nbytes_idx
            )
            expert_mask_rsrc = buffer_ops.create_buffer_resource(
                expert_mask, max_size=True
            )

            token_idx = gpu.block_id("x")
            tile_idx = gpu.block_id("y")
            tid = gpu.thread_id("x")

            # Guard: token in range (Index is unsigned → auto ult)
            tok_ok = token_idx < m_tokens
            _if_tok = scf.IfOp(tok_ok)
            with _if_then(_if_tok):
                tile_cols = BLOCK_SIZE * VEC_WIDTH
                c_tile_cols = fx.Index(tile_cols)
                c_vecw = fx.Index(VEC_WIDTH)

                col_base = tile_idx * c_tile_cols + tid * c_vecw

                # Guard: any work in bounds (Index < → ult)
                col_ok = col_base < c_model_dim
                _if_col = scf.IfOp(col_ok)
                with _if_then(_if_col):
                    # Fast path: full vector in-bounds (Index <= → ule)
                    end_ok = col_base + c_vecw <= c_model_dim
                    _if_full = scf.IfOp(end_ok, has_else=True)
                    with _if_then(_if_full):
                        # ── Vector path via layout API (all dtypes) ──
                        # fx.copy auto-iterates when atom width < VEC_WIDTH
                        # (e.g. f32: BufferCopy128b handles 4, fx.copy issues 2 calls for 8)
                        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
                        vec_type_c = T.vec(copy_vec_width, compute_type())
                        vec_type_e = T.vec(copy_vec_width, elem_type())

                        acc_vecs = [
                            vector.broadcast(vec_type_c, fx.Float32(0.0).ir_value())
                            for _ in range(n_sub)
                        ]
                        reg_ty = fx.MemRefType.get(
                            elem_type(), fx.LayoutType.get(copy_vec_width, 1), fx.AddressSpace.Register
                        )
                        reg_lay = fx.make_layout(copy_vec_width, 1)

                        tok_i32 = fx.Int32(token_idx)
                        tile_i32 = fx.Int32(tile_idx)
                        tid_i32 = fx.Int32(tid)

                        for k in range_constexpr(topk):
                            # X[token, k, :] → tile → thread's VEC_WIDTH slice
                            x_row = X_buf[tok_i32, fx.Int32(k), None]
                            x_tiled = fx.logical_divide(x_row, fx.make_layout(tile_cols, 1))
                            x_div = fx.logical_divide(x_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1))
                            x_thread = x_div[None, tid_i32]

                            if use_mask:
                                m_idx_i32 = fx.Int32(token_idx * c_topk + fx.Index(k))
                                expert_i32 = buffer_ops.buffer_load(
                                    topk_ids_rsrc, m_idx_i32, vec_width=1, dtype=i32_type()
                                )
                                mv = buffer_ops.buffer_load(
                                    expert_mask_rsrc, expert_i32, vec_width=1, dtype=i32_type()
                                )
                                mv_ok = mv != fx.Int32(0)

                            if n_sub > 1:
                                x_inner = fx.logical_divide(x_thread, fx.make_layout(copy_vec_width, 1))
                            for si in range_constexpr(n_sub):
                                src = x_inner[None, fx.Int32(si)] if n_sub > 1 else x_thread
                                r = fx.memref_alloca(reg_ty, reg_lay)
                                fx.copy_atom_call(copy_atom, src, r)
                                vec_e = fx.memref_load_vec(r)

                                if use_mask:
                                    zero_e = vector.broadcast(vec_type_e, arith.constant(0.0, type=elem_type()))
                                    vec_e = mv_ok.select(vec_e, zero_e)

                                if elem_bits < 32:
                                    vec_c = vec_e.extf(vec_type_c)
                                else:
                                    vec_c = vec_e
                                acc_vecs[si] = acc_vecs[si] + vec_c

                        # ── Store results ──
                        if n_sub > 1:
                            y_row = Y_buf[tok_i32, None]
                            y_tiled = fx.logical_divide(y_row, fx.make_layout(tile_cols, 1))
                            y_div = fx.logical_divide(y_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1))
                            y_inner = fx.logical_divide(y_div[None, tid_i32], fx.make_layout(copy_vec_width, 1))

                        for si in range_constexpr(n_sub):
                            out_vec = acc_vecs[si]
                            if elem_bits < 32:
                                out_vec = out_vec.truncf(vec_type_e)

                            if n_sub > 1:
                                dst = y_inner[None, fx.Int32(si)]
                            else:
                                y_row = Y_buf[tok_i32, None]
                                y_tiled = fx.logical_divide(y_row, fx.make_layout(tile_cols, 1))
                                y_div = fx.logical_divide(y_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1))
                                dst = y_div[None, tid_i32]

                            r_out = fx.memref_alloca(reg_ty, reg_lay)
                            fx.memref_store_vec(out_vec, r_out)
                            fx.copy_atom_call(copy_atom, r_out, dst)

                    with _if_else(_if_full):
                        # Tail path: scalar load/store per lane.
                        for lane in range_constexpr(VEC_WIDTH):
                            col = col_base + fx.Index(lane)
                            lane_ok = col < c_model_dim
                            _if_lane = scf.IfOp(lane_ok)
                            with _if_then(_if_lane):
                                a = arith.constant(0.0, type=compute_type())
                                token_base = token_idx * c_topk
                                for k in range_constexpr(topk):
                                    k_idx = fx.Index(k)
                                    x_idx_i32 = fx.Int32((token_base + k_idx) * c_model_dim + col)
                                    if use_mask:
                                        m_idx_i32 = fx.Int32(token_base + k_idx)
                                        expert_i32 = buffer_ops.buffer_load(
                                            topk_ids_rsrc, m_idx_i32, vec_width=1, dtype=i32_type()
                                        )
                                        mv = buffer_ops.buffer_load(
                                            expert_mask_rsrc, expert_i32, vec_width=1, dtype=i32_type()
                                        )
                                        v = (mv != fx.Int32(0)).select(
                                            buffer_ops.buffer_load(x_rsrc, x_idx_i32, vec_width=1, dtype=elem_type()),
                                            arith.constant(0.0, type=elem_type()),
                                        )
                                    else:
                                        v = buffer_ops.buffer_load(
                                            x_rsrc, x_idx_i32, vec_width=1, dtype=elem_type()
                                        )
                                    if dtype_str in ("f16", "bf16"):
                                        v = v.extf(compute_type())
                                    a = a + v

                                out = a
                                if dtype_str in ("f16", "bf16"):
                                    out = out.truncf(elem_type())
                                y_idx_i32 = fx.Int32(token_idx * c_model_dim + col)
                                buffer_ops.buffer_store(out, y_rsrc, y_idx_i32)

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction(
        X: fx.Tensor,
        Y: fx.Tensor,
        topk_ids: fx.Tensor,
        expert_mask: fx.Tensor,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Index(i32_m_tokens)
        moe_reduction_kernel(X, Y, topk_ids, expert_mask, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction


# MoE GEMM2 Execution Modes
class MoeGemm2Mode:
    """Execution mode for MoE GEMM2."""
    ATOMIC = "atomic"       # Use atomic accumulation (default)
    REDUCE = "reduce"       # Use non-atomic write + reduce kernel


class _MoeGemm2ReduceWrapper:
    """Wrapper combining GEMM2 (no atomics) with reduction kernel.

    This wrapper handles the intermediate buffer allocation and orchestrates
    the two-phase computation:
    1. GEMM2 outputs to [tokens*topk, model_dim] without atomics
    2. Reduce sums over topk to produce [tokens, model_dim]
    """

    def __init__(
        self,
        gemm2_exe,
        reduce_exe,
        topk: int,
        model_dim: int,
        out_dtype_str: str = "f16",
        use_mask: bool = False,
        zero_intermediate: bool = True,
    ):
        self._gemm2_exe = gemm2_exe
        self._reduce_exe = reduce_exe
        self._topk = topk
        self._model_dim = model_dim
        self._out_dtype_str = out_dtype_str
        self._use_mask = use_mask
        self._zero_intermediate = zero_intermediate
        
    def _get_torch_dtype(self):
        """Convert dtype string to torch dtype."""
        import torch
        dtype_map = {
            "f16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "f32": torch.float32,
        }
        return dtype_map.get(self._out_dtype_str, torch.float16)

    def __call__(
        self,
        arg_out,
        arg_x,
        arg_w,
        arg_scale_x,
        arg_scale_w,
        arg_qscale_w,
        arg_qzero_w,
        arg_sorted_token_ids,
        arg_expert_ids,
        arg_sorted_weights,
        arg_num_valid_ids,
        tokens_in,
        n_in,
        k_in,
        size_expert_ids_in,
        intermediate = None,
        valid_mask = None,
        topk_ids = None,
        expert_mask = None,
        stream_ptr = None,
    ):
        """Execute GEMM2 + reduce.

        Args match moe_gemm2 kernel signature (see compile_moe_gemm2).
        For masked reduction, pass `topk_ids` ([tokens, topk] int32) and
        `expert_mask` ([total_experts] int32); the kernel fuses the
        `expert_mask[topk_ids]` lookup inline.
        """
        import torch
        if stream_ptr is None:
            stream_ptr = torch.cuda.current_stream()
        if intermediate is None:
            intermediate = torch.empty(
                    tokens_in * self._topk, self._model_dim,
                    device=arg_out.device,
                    dtype=self._get_torch_dtype()
                )
        if self._zero_intermediate and not self._use_mask:
            intermediate.zero_()
        # Phase 1: GEMM2 (no atomics) -> [tokens*topk, model_dim]
        self._gemm2_exe(
            intermediate.view(-1),
            arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_qscale_w, arg_qzero_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, tokens_in, n_in, k_in, size_expert_ids_in,
            stream_ptr,
        )
        # Phase 2: Reduce over topk -> [tokens, model_dim]
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        Y = arg_out.view(tokens_in, self._model_dim)
        if self._use_mask:
            if topk_ids is None or expert_mask is None:
                if valid_mask is None:
                    raise ValueError(
                        "Masked reduction requires topk_ids and expert_mask"
                    )
                # Backward-compatible fallback for older callers that still provide
                # a precomputed valid_mask instead of topk_ids/expert_mask.
                Y.copy_(torch.sum(X * valid_mask.view(tokens_in, self._topk, 1).to(dtype=X.dtype), dim=1))
                return
            topk_ids = topk_ids.contiguous()
            expert_mask = expert_mask.contiguous()
        else:
            if valid_mask is not None:
                logging.warning("valid_mask provided but use_mask=False; ignoring valid_mask")
            topk_ids = torch.empty((0, self._topk), device=arg_out.device, dtype=torch.int32)
            expert_mask = torch.empty((0,), device=arg_out.device, dtype=torch.int32)
        self._reduce_exe(X, Y, topk_ids, expert_mask, tokens_in, stream_ptr)

    @property
    def mode(self) -> str:
        """Return the execution mode."""
        return MoeGemm2Mode.REDUCE


def compile_moe_gemm2_ex(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Extended parameters for mode control
    mode: str = MoeGemm2Mode.ATOMIC,
    valid_mask = None,
    zero_intermediate: bool = True,
    scale_is_bf16: bool = False,
):
    """Compile MoE GEMM2 kernel with optional reduction.

    This is the extended interface that supports explicit mode control.

    Args:
        mode: Execution mode selection:
            - "atomic": Use atomic accumulation (original behavior)
            - "reduce": Use non-atomic write + reduce kernel
        
        zero_intermediate: If all output slots are valid, 
            set False to increase performance

    Returns:
        Compiled executable (either wrapped or raw depending on mode).
    """
    # Compile based on mode
    if mode == MoeGemm2Mode.REDUCE:
        # Determine if we need masked reduction
        use_mask = valid_mask is not None
        
        # Compile GEMM2 with accumulate=False
        gemm2_exe = compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=False,
            scale_is_bf16=scale_is_bf16,
        )
        # Compile reduction kernel with masking support
        out_s = str(out_dtype).strip().lower()
        if out_s in ("f16", "fp16", "half"):
            dtype_str = "f16"
        elif out_s in ("bf16", "bfloat16"):
            dtype_str = "bf16"
        else:
            dtype_str = "f32"
        reduce_exe = compile_moe_reduction(
            topk=topk,
            model_dim=model_dim,
            dtype_str=dtype_str,
            use_mask=use_mask,
        )
        return _MoeGemm2ReduceWrapper(
            gemm2_exe=gemm2_exe,
            reduce_exe=reduce_exe,
            topk=topk,
            model_dim=model_dim,
            out_dtype_str=dtype_str,
            use_mask=use_mask,
            zero_intermediate=zero_intermediate,
        )
    else:
        # Compile GEMM2 with accumulate=True (atomic mode)
        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=True,
        )
