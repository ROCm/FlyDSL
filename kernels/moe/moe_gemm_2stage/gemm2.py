# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM stage2 (MFMA) kernel builder + reduce-mode dispatch."""

import functools
import logging
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import (
        bf16_global_atomics_arch_description,
        supports_bf16_global_atomics,
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
from kernels.common.kernels_common import _if_then
from kernels.common.mem_ops import buffer_atomic_add
from kernels.mma.mfma_epilogues import c_shuffle_epilog, default_epilog
from kernels.mma.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    extract_bf16_scale,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    load_b_pack_k32,
    load_b_raw_w4a16,
    load_b_raw_w4a16_groupwise,
    make_preshuffle_b_layout,
    preshuffle_crd2idx,
    swizzle_xor16,
    tile_chunk_coord_i32,
    unpack_b_w4a16,
)
from kernels.moe.moe_common import (
    i64_to_v4f16 as _i64_to_v4f16,
)
from kernels.moe.moe_common import (
    i64_to_v4i16 as _i64_to_v4i16,
)
from kernels.moe.moe_common import (
    i64x2_to_v8bf16 as _i64x2_to_v8bf16,
)
from kernels.moe.moe_common import (
    i64x2_to_v8f16 as _i64x2_to_v8f16,
)
from kernels.moe.moe_gemm_2stage.reduction import compile_moe_reduction


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

    in_dtype:
      - "fp8": A2/W are fp8
      - "fp16": A2/W are fp16
      - "bf16": A2/W are bf16
      - "int8": A2/W are int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 unpacked to int8 in-kernel
      - "int4_bf16": W4A16 path: A2 is bf16, W is packed int4 unpacked to bf16 in-kernel
    scale_is_bf16: When True, groupwise scales are bf16 (halves scale bandwidth).

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).
    """
    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    _valid_dtypes = ("fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16")
    if in_dtype not in _valid_dtypes:
        raise ValueError(f"in_dtype must be one of {_valid_dtypes}, got {in_dtype!r}")
    is_int4_bf16 = in_dtype == "int4_bf16"  # W4A16: bf16 activations, packed int4 weights
    is_f16 = in_dtype == "fp16"
    is_bf16 = is_int4_bf16 or in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    needs_scale_w = (not is_f16_or_bf16) or is_int4_bf16
    elem_bytes = 2 if is_f16_or_bf16 else 1
    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}")
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError("compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}")
    is_int4 = in_dtype == "int4"
    # w_is_int4: True for any variant where weights are packed int4.
    w_is_int4 = is_int4 or is_int4_bf16
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype in ("int8", "int8smooth")) or is_int4

    # Group-wise scale support for W4A16
    use_groupwise_scale = w_is_int4 and group_size > 0
    if use_groupwise_scale and group_size != 32:
        raise ValueError(
            f"FlyDSL groupwise scale only supports group_size=32, got {group_size}. "
            f"This is due to int4 preshuffle layout constraints. "
            f"Please use Triton kernel for other group sizes."
        )
    is_int4_bf16_groupwise = is_int4_bf16 and use_groupwise_scale
    # Stage2 K dimension is inter_dim (weight shape: [E, model_dim, inter_dim])
    num_groups = inter_dim // group_size if use_groupwise_scale else 1
    _scale_is_bf16 = scale_is_bf16 and use_groupwise_scale
    experts * model_dim * num_groups

    _is_gfx950 = "gfx95" in get_rocm_arch()
    _has_cvt_off_f32_i4 = hasattr(rocdl, "cvt_off_f32_i4")
    use_gfx950_cvt = is_int4_bf16 and _is_gfx950 and _has_cvt_off_f32_i4

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(rocdl, "mfma_i32_16x16x32_i8", None)
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` " "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    mfma_f32_bf16_k16 = None
    if is_bf16:
        mfma_f32_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
            rocdl, "mfma_f32_16x16x16_bf16_1k", None
        )
        if mfma_f32_bf16_k16 is None:
            raise AttributeError(
                "BF16 K16 MFMA op not found: expected `rocdl.mfma_f32_16x16x16bf16_1k` "
                "(or `rocdl.mfma_f32_16x16x16_bf16_1k`)."
            )

    # gfx950: use 16x16x32 MFMA for f16/bf16 (K=32 per MFMA, vs K=16 on gfx942).
    # Check if K=32 MFMA supports the (result_type, operands_list) calling convention.
    _has_k32_mfma_compat = False
    if _is_gfx950 and (is_f16 or is_bf16):
        import inspect

        _k32_fn = rocdl.mfma_f32_16x16x32_bf16 if is_bf16 else rocdl.mfma_f32_16x16x32_f16
        try:
            _k32_sig = inspect.signature(_k32_fn)
            _k32_params = list(_k32_sig.parameters.keys())
            # Compatible if second param is "operands" (list-based API)
            _has_k32_mfma_compat = len(_k32_params) >= 2 and _k32_params[1] == "operands"
        except (ValueError, TypeError):
            _has_k32_mfma_compat = False
    _use_mfma_k32 = _is_gfx950 and (is_f16 or is_bf16) and _has_k32_mfma_compat

    ir.ShapedType.get_dynamic_size()
    # W is packed int4 for W4A8/W4A16/W4A_FP8: 2 values per byte.
    ((experts * model_dim * inter_dim) // 2 if w_is_int4 else (experts * model_dim * inter_dim))

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

    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
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
            raise ValueError("stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1).")

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
    _gs_tag = f"_g{group_size}" if use_groupwise_scale else ""
    scale_tag = "_sbf16" if _scale_is_bf16 else ""
    (
        f"mfma_moe2_{in_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"{_gs_tag}{scale_tag}"
        f"_abi2"  # mask sentinel token ids on loads/stores to avoid illegal address faults
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
            raise ValueError(f"tile_n={tile_n} must be divisible by {_cshuffle_stride} when accumulate=False")

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
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
            # i32 versions for layout construction (fly.make_shape requires i32/i64)
            k_i32_v = i32_k_in
            x_elem = T.bf16 if is_bf16 else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
            # For int4/int4_bf16, weights are stored as packed bytes (i8) and unpacked in-kernel.
            w_elem = T.i8 if w_is_int4 else (T.bf16 if is_bf16 else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)))
            scale_dtype = T.bf16 if _scale_is_bf16 else T.f32
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            acc_init = arith.constant_vector(0, T.i32x4) if is_int8 else arith.constant_vector(0.0, T.f32x4)
            zero_f32_acc = arith.constant_vector(0.0, T.f32x4) if is_int4_bf16_groupwise else None

            # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
            topk_idx = fx.Index(topk)
            m_in = tokens_in * topk_idx
            m_i32_v = arith.index_cast(T.i32, m_in)
            fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.index(experts * model_dim)
            # For packed int4 (W4A8/W4A16/W4A_FP8), kpack_bytes=8.
            kpack_bytes = 8 if w_is_int4 else 16
            w_elem_bytes = 1 if w_is_int4 else elem_bytes
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=w_elem_bytes,
            )
            layout_b = b_layout.layout_b
            (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

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
            fx.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                (T.bf16 if is_bf16 else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))),
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
            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes_idx)

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * fx.Index(out_elem_bytes)
            if const_expr(not bool(accumulate)):
                out_nbytes_idx = tokens_in * fx.Index(topk) * n_in * fx.Index(out_elem_bytes)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False, num_records_bytes=out_nbytes_idx)
            # scale_x: fp16/bf16 path ignores (implicit scale=1.0); int4_bf16 also uses 1.0.
            if const_expr(is_f16_or_bf16):
                sx_rsrc = None
            else:
                # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                sx_nbytes_idx = (tokens_in * c_topk) * fx.Index(4)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx
                )
            # scale_w: fp16/bf16 (non-int4) path ignores; int4_bf16 needs dequant scale.
            if const_expr(not needs_scale_w):
                sw_rsrc = None
            else:
                # scale_w: [experts*model_dim] f32 (static shape in practice)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = size_expert_ids_in * fx.Index(tile_m) * fx.Index(4)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_idx,
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
            num_valid_i32 = buffer_ops.buffer_load(numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
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
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16/bf16 we require 16B.
                if const_expr(is_f16_or_bf16):
                    if const_expr(bytes_per_thread_x % 16 != 0):
                        raise ValueError(f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16")
                    x_load_bytes = 16
                else:
                    if const_expr(bytes_per_thread_x % 16 == 0):
                        x_load_bytes = 16
                    elif const_expr(bytes_per_thread_x % 8 == 0):
                        x_load_bytes = 8
                    elif const_expr(bytes_per_thread_x % 4 == 0):
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                fx.make_layout((m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1))
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
                    if const_expr(x_load_bytes == 16):
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
                    if const_expr(x_load_bytes == 8):
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
                        if const_expr(x_load_bytes == 16):
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif const_expr(x_load_bytes == 8):
                            parts.append(vector.bitcast(T.vec(2, T.i32), x_vec))
                        else:
                            parts.append(vector.bitcast(T.vec(1, T.i32), x_vec))
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(fx.Int32(tx), layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(fx.Int32(lane_id), layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                row_a_lds = lane_mod_16
                # A-side kpack is always 16 bytes; kpack_bytes is B-side (may be 8 for int4).
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base if elem_bytes == 1 else (col_offset_base * arith.index(int(elem_bytes)))
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
                c_n_total // fx.Index(16)
                c_n0_static = experts * model_dim // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base + offset + lane_mod_16
                    col_g_list.append(col_g)

                    row_w = expert_off_idx + col_g
                    coord_w = fx.idx2crd(fx.Int32(row_w), layout_n_blk_intra)
                    n_blk_list.append(fx.get(coord_w, 0))
                    n_intra_list.append(fx.get(coord_w, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)

                # --- B Load Logic (K64) ---
                def load_b_pack(base_k, ki_step, ni):
                    return load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ki_step,
                        n_blk=n_blk_list[ni],
                        n_intra=n_intra_list[ni],
                        lane_div_16=lane_div_16,  # 0..3
                        elem_type=w_elem,
                        kpack_bytes=kpack_bytes,
                        elem_bytes=w_elem_bytes,
                        unpack_int4=is_int4,
                    )

                def load_b_tile(base_k):
                    """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                    Returns a list of length `k_unroll`, where each entry is a tuple:
                      (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                    For groupwise variants, each entry also includes per-group scales:
                      (packs0[ni], packs1[ni], scales0[ni], scales1[ni])
                    """
                    if const_expr(is_int4_bf16_groupwise):
                        # W4A16 groupwise: load raw packed32 + scale; defer dequant to compute_tile.
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                packed32, scale_val = load_b_raw_w4a16_groupwise(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=n_blk_list[ni],
                                    n_intra=n_intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    scale_rsrc=sw_rsrc,
                                    expert_offset=expert_off_idx,
                                    num_groups=num_groups,
                                    group_size=group_size,
                                    n_per_expert=model_dim,
                                    kpack_bytes=kpack_bytes,
                                    scale_dtype=scale_dtype,
                                )
                                raw_ku.append((packed32, scale_val))
                            raw_data.append(raw_ku)
                        return raw_data
                    elif const_expr(is_int4_bf16):
                        # W4A16 per-row: load raw packed32; defer dequant to compute_tile.
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                raw = load_b_raw_w4a16(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=n_blk_list[ni],
                                    n_intra=n_intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    kpack_bytes=kpack_bytes,
                                )
                                raw_ku.append(raw)
                            raw_data.append(raw_ku)
                        return raw_data
                    else:
                        # fp8/int8/bf16/fp16: original code path
                        b_tile = []
                        for ku in range_constexpr(k_unroll):
                            packs0 = []
                            packs1 = []
                            for ni in range_constexpr(num_acc_n):
                                ki0 = (ku * 2) + 0
                                ki1 = (ku * 2) + 1
                                b0 = load_b_pack(base_k, ki0, ni)
                                b1 = load_b_pack(base_k, ki1, ni)
                                packs0.append(b0)
                                packs1.append(b1)
                            b_tile.append((packs0, packs1))
                        return b_tile

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if const_expr(x_load_bytes == 16):
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
                        elif const_expr(x_load_bytes == 8):
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
                        col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = preshuffle_crd2idx((fx.Int32(curr_row_a_lds), fx.Int32(col_base_swz)), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                    if const_expr(_use_mfma_k32):
                        mfma_fn = rocdl.mfma_f32_16x16x32_f16 if is_f16 else rocdl.mfma_f32_16x16x32_bf16
                    else:
                        mfma_fn = (
                            mfma_i32_k32
                            if is_int8
                            else (
                                mfma_f32_bf16_k16
                                if is_bf16
                                else (rocdl.mfma_f32_16x16x16f16 if is_f16 else rocdl.mfma_f32_16x16x32_fp8_fp8)
                            )
                        )

                    epilogue_pf = None
                    if const_expr(prefetch_epilogue and not use_groupwise_scale):
                        expert_off_pf = expert_off_idx
                        sw_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_w_idx = expert_off_pf + col_g
                            sw_pf.append(
                                fx.Float32(1.0)
                                if not needs_scale_w
                                else buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32)
                            )
                        # Also prefetch per-row routed/topk weights (sorted_weights) when enabled.
                        tw_pf = None
                        if const_expr(doweight_stage2):
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
                                            sorted_w_rsrc,
                                            sorted_row_pf,
                                            vec_width=1,
                                            dtype=T.f32,
                                        )
                                    )
                        epilogue_pf = (sw_pf, tw_pf)

                    def mfma_k64(acc0, a0, a1, b0, b1):
                        if const_expr(_use_mfma_k32):
                            # gfx950: single 16x16x32 MFMA consuming all 128 bits (K=32 f16/bf16)
                            if const_expr(is_f16):
                                av = _i64x2_to_v8f16(a0, a1)
                                bv = _i64x2_to_v8f16(b0, b1)
                            else:
                                av = _i64x2_to_v8bf16(a0, a1)
                                bv = _i64x2_to_v8bf16(b0, b1)
                            return mfma_fn(mfma_res_ty, [av, bv, acc0, 0, 0, 0])
                        if const_expr(is_f16):
                            a0v = _i64_to_v4f16(a0)
                            a1v = _i64_to_v4f16(a1)
                            b0v = _i64_to_v4f16(b0)
                            b1v = _i64_to_v4f16(b1)
                            acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                        if const_expr(is_bf16):
                            a0v = _i64_to_v4i16(a0)
                            a1v = _i64_to_v4i16(a1)
                            b0v = _i64_to_v4i16(b0)
                            b1v = _i64_to_v4i16(b1)
                            acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                        acc1 = mfma_fn(mfma_res_ty, [a0, b0, acc0, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc1, 0, 0, 0])

                    def _acc_scaled_f32(f32_acc_vec, f32_partial_vec, scale_val):
                        """MFMA f32 partial -> scale -> add to f32 accumulator via math.fma on vector."""
                        from flydsl._mlir.dialects._math_ops_gen import fma as _math_fma

                        _uw = arith._to_raw
                        scale_vec = _uw(vector.broadcast(T.f32x4, scale_val))
                        return arith.ArithValue(_math_fma(scale_vec, _uw(f32_partial_vec), _uw(f32_acc_vec)))

                    if const_expr(is_int4_bf16 or is_int4_bf16_groupwise):
                        # W4A16: deferred dequant -- unpack int4->bf16 right before MFMA
                        # to minimize VGPR lifetime of dequantized bf16 values.
                        _pending_acc = None
                        for ku in range_constexpr(k_unroll):
                            b_raw = b_tile_in[ku]
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                if const_expr((a0_prefetch is not None) and (ku == 0) and (mi == 0)):
                                    a0, a1 = a0_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    if const_expr(is_int4_bf16_groupwise):
                                        packed, sc = b_raw[ni]
                                        if const_expr(_scale_is_bf16):
                                            sc = extract_bf16_scale(arith, sc, ku)
                                    else:
                                        packed, sc = b_raw[ni], None
                                    if const_expr(is_int4_bf16_groupwise and use_gfx950_cvt):
                                        b0, b1 = unpack_b_w4a16(
                                            packed,
                                            arith,
                                            vector,
                                            scale_val=None,
                                            use_gfx950_cvt=True,
                                            defer_scale16=True,
                                        )
                                        tmp = mfma_k64(zero_f32_acc, a0, a1, b0, b1)
                                        if _pending_acc is not None:
                                            p_idx, p_tmp, p_sc = _pending_acc
                                            acc_list[p_idx] = _acc_scaled_f32(acc_list[p_idx], p_tmp, p_sc)
                                        _pending_acc = (acc_idx, tmp, sc)
                                    else:
                                        b0, b1 = unpack_b_w4a16(
                                            packed,
                                            arith,
                                            vector,
                                            scale_val=sc,
                                            use_gfx950_cvt=use_gfx950_cvt,
                                            defer_scale16=use_gfx950_cvt,
                                        )
                                        acc_list[acc_idx] = mfma_k64(acc_list[acc_idx], a0, a1, b0, b1)
                        # Drain last pending FMA.
                        if _pending_acc is not None:
                            p_idx, p_tmp, p_sc = _pending_acc
                            acc_list[p_idx] = _acc_scaled_f32(acc_list[p_idx], p_tmp, p_sc)
                    else:
                        for ku in range_constexpr(k_unroll):
                            b_packs0, b_packs1 = b_tile_in[ku]
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                if const_expr((a0_prefetch is not None) and (ku == 0) and (mi == 0)):
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

                # def hot_loop_scheduler():
                #     mfma_group = num_acc_n
                #     # K64 micro-step: 2x K32 MFMA per accumulator update.
                #     mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                #     mfma_per_iter = 2 * mfma_group
                #     sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                #     rocdl.sched_dsrd(2)
                #     rocdl.sched_mfma(1)
                #     rocdl.sched_mfma(1)
                #     if num_acc_n < 4:
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_vmem(1)

                #     dswr_tail = num_x_loads
                #     if dswr_tail > sche_iters:
                #         dswr_tail = sche_iters
                #     dswr_start = sche_iters - dswr_tail
                #     for sche_i in range_constexpr(sche_iters):
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(mfma_group)
                #         if sche_i >= dswr_start - 1:
                #             rocdl.sched_dswr(1)
                #     rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    rocdl.sched_barrier(0)
                    return
                    # - MFMA group size per "slot": num_acc_n
                    # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                    # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    if const_expr(num_acc_n < 4):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(1)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)

                    # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    if const_expr(dswr_tail > sche_iters):
                        dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail

                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if const_expr(sche_i >= dswr_start - 1):
                            rocdl.sched_dswr(1)

                    rocdl.sched_barrier(0)

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
                if const_expr(k_main2_py < 0):
                    k_main2_py = 0

                arith.index(tile_k * 2)
                c_tile_k_s2 = arith.index(tile_k)
                pair_iters = k_main2_py // (int(tile_k) * 2)

                # B-tile data layout per k_unroll entry (3 variants):
                #   See gemm1 _flatten_b_tile for full layout documentation.
                int4_bf16_single_field = is_int4_bf16 and not is_int4_bf16_groupwise
                _fields_per_ku = 1 if int4_bf16_single_field else 2
                _vals_per_b_tile = k_unroll * _fields_per_ku * num_acc_n
                _n_acc = m_repeat * num_acc_n
                _p_b = _n_acc
                _p_a0 = _p_b + _vals_per_b_tile

                def _flatten_b_tile(b_tile):
                    """Flatten B tile to a 1-D list for scf.for loop-carried state."""
                    flat = []
                    for ku_entry in b_tile:
                        if is_int4_bf16_groupwise:
                            flat.extend(t[0] for t in ku_entry)
                            flat.extend(t[1] for t in ku_entry)
                        elif int4_bf16_single_field:
                            flat.extend(ku_entry)
                        else:
                            flat.extend(ku_entry[0])
                            flat.extend(ku_entry[1])
                    return flat

                def _unflatten_b_tile(vals):
                    """Reconstruct B tile from flattened scf.for loop-carried state."""
                    b_tile, idx = [], 0
                    for _ in range_constexpr(k_unroll):
                        if is_int4_bf16_groupwise:
                            packed = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            scales = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            b_tile.append([(packed[ni], scales[ni]) for ni in range_constexpr(num_acc_n)])
                        elif int4_bf16_single_field:
                            b_tile.append(list(vals[idx : idx + num_acc_n]))
                            idx += num_acc_n
                        else:
                            packs_even = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            packs_odd = list(vals[idx : idx + num_acc_n])
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

                if const_expr(odd_k_tiles):
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
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
                        acc,
                        b_ping,
                        lds_base_ping,
                        prefetch_epilogue=True,
                        a0_prefetch=a0_prefetch_ping,
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
                    buffer_atomic_add(val_f16x2, out_rsrc, byte_off_i32, zero_i32, zero_i32)

                sw_pf = None
                tw_pf = None
                if const_expr(epilogue_pf is not None):
                    sw_pf, tw_pf = epilogue_pf

                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if const_expr(use_groupwise_scale):
                    # Groupwise: weight scale already applied per-group in K-loop.
                    sw_vals = [arith.constant(1.0, type=T.f32)] * num_acc_n
                elif const_expr(sw_pf is not None):
                    sw_vals = sw_pf
                else:
                    sw_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_w_idx = expert_off + col_g
                        sw_vals.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32)
                        )

                # When defer_scale16 was used, the x16 correction for v_cvt_off_f32_i4
                # was omitted from the hot loop.  Fold it into the epilogue scale.
                if const_expr(use_gfx950_cvt):
                    _c16 = fx.Float32(16.0)
                    sw_vals = [v * _c16 for v in sw_vals]

                if const_expr(out_is_f32):
                    # origin/dev_a16w4: f32 output uses scalar f32 atomics and skips CShuffle/LDS.
                    c4_i32 = fx.Int32(4)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        buffer_atomic_add(val_f32, out_rsrc, byte_off_i32, zero_i32, zero_i32)

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
                        sx = (
                            arith.select(ts_ok, fx.Float32(1.0), fx.Float32(0.0))
                            if is_f16_or_bf16
                            else arith.select(
                                ts_ok,
                                buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32),
                                fx.Float32(0.0),
                            )
                        )

                        if const_expr(doweight_stage2):
                            tw_idx = (mi * 4) + ii
                            if const_expr(tw_pf is not None):
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
                            if const_expr(is_int8):
                                v = arith.sitofp(T.f32, v)
                            v = v * sx * sw
                            if const_expr(doweight_stage2):
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
                    if const_expr(lds_out is None):
                        raise RuntimeError("FLYDSL_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased.")

                    # For bf16 global atomics (gfx942 only), precompute the output base address.
                    # gfx950+ has buffer_atomic_pk_add_bf16, so bf16 uses buffer atomics there.
                    out_base_idx = None
                    if const_expr(_needs_global_atomic_bf16):
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
                        sx = (
                            fx.Float32(1.0)
                            if is_f16_or_bf16
                            else arith.select(
                                ts_ok,
                                buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=T.f32),
                                fx.Float32(0.0),
                            )
                        )

                        if const_expr(doweight_stage2):
                            tw_idx = (mi * 4) + ii
                            if const_expr(tw_pf is not None):
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32)

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw = sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                            if const_expr(is_int8):
                                v = arith.sitofp(T.f32, v)
                            v = v * sx * sw
                            if const_expr(doweight_stage2):
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
                        if const_expr(not bool(accumulate)):
                            ts = t * topk_i32_v + s
                            idx0 = ts * model_i32
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if const_expr(_needs_global_atomic_bf16):
                            # gfx942: no buffer_atomic_pk_add_bf16, use global atomicrmw fadd
                            if const_expr(bool(accumulate)):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                                out_ptr_v = out_ptr._value if const_expr(hasattr(out_ptr, "_value")) else out_ptr
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
                            if const_expr(bool(accumulate)):
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
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = n_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
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


class MoeGemm2Mode:
    """Execution mode for MoE GEMM2."""

    ATOMIC = "atomic"  # Use atomic accumulation (default)
    REDUCE = "reduce"  # Use non-atomic write + reduce kernel


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
        arg_sorted_token_ids,
        arg_expert_ids,
        arg_sorted_weights,
        arg_num_valid_ids,
        tokens_in,
        n_in,
        k_in,
        size_expert_ids_in,
        valid_mask=None,
        stream=None,
    ):
        """Execute GEMM2 + reduce.

        Args match moe_gemm2 kernel signature (see compile_moe_gemm2).
        """
        import torch

        if stream is None:
            stream = torch.cuda.current_stream()
        intermediate = torch.empty(
            tokens_in * self._topk,
            self._model_dim,
            device=arg_out.device,
            dtype=self._get_torch_dtype(),
        )
        if self._zero_intermediate and not self._use_mask:
            intermediate.zero_()
        # Phase 1: GEMM2 (no atomics) -> [tokens*topk, model_dim]
        self._gemm2_exe(
            intermediate.view(-1),
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            tokens_in,
            n_in,
            k_in,
            size_expert_ids_in,
            stream,
        )
        # Phase 2: Reduce over topk -> [tokens, model_dim]
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        Y = arg_out.view(tokens_in, self._model_dim)
        if not self._use_mask:
            if valid_mask is not None:
                logging.warning("valid_mask provided but use_mask=False; ignoring valid_mask")
            valid_mask = torch.empty((0, self._topk), device=arg_out.device, dtype=torch.uint8)
        self._reduce_exe(X, Y, valid_mask, tokens_in, stream)

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
    valid_mask=None,
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
