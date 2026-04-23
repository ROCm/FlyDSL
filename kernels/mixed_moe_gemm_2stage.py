# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM stage1/stage2 kernel implementations (FlyDSL MFMA FP8/FP16/FP4).

"""

import functools
import os
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import supports_bf16_global_atomics
except ImportError:
    # Backward compatibility for runtime.device versions that only expose get_rocm_arch.
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, memref, llvm
from flydsl._mlir.dialects import fly as _fly

from kernels.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    crd2idx,
    lds_row_major_idx,
    lds_store_16b_xor16,
    lds_store_8b_xor16,
    lds_store_4b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    load_b_pack_k32,
    load_b_raw_mxfp4,
    load_b_raw_mxfp4_dwordx4,
    unpack_b_mxfp4_bf16,
    split_row_major_2d,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from kernels.mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog


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


def _w_elem_type(*, is_f4_b: bool, is_f16_b: bool):
    """Return the packed weight element type used by preshuffled B tiles."""
    if is_f4_b:
        return T.i8
    return T.f16 if is_f16_b else T.f8


@functools.lru_cache(maxsize=1024)
def compile_mixed_moe_gemm1(
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
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    act: str = "silu",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    a_dtype:
      - "fp8": X is fp8
      - "fp16": X is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": X is int8
      - "fp4": X is fp4

    b_dtype:
      - "fp8": W is fp8
      - "fp16": W is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": W is int8
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "fp4": W is fp4
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f16 = is_f16_a or is_f16_b

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    pack_M = 2
    pack_N = 2
    pack_K = 2

    elem_bytes = 1

    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    # K64-byte micro-step: always 64 bytes per `ku`. For fp16, this is 32 elements (2xK16 MFMA).
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
        )
    is_int4 = b_dtype == "int4"


    def _x_lds_elem_type():
        return T.f16 if is_f16_a else T.f8

    def _out_elem_type():
        return T.bf16 if out_dtype == "bf16" else T.f16

    def _out_lds_elem_type():
        return T.f32


    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    pad_k = 0
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLYDSL_MOE_STAGE1_CSHUFFLE", "0") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    use_cshuffle_epilog = bool(use_cshuffle_epilog)

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = (
        f"mfma_moe1_a{a_dtype}_w{b_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi35_ckstage1"
    ).replace("-", "_")
    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Reuse the same LDS bytes for both:
    # - ping-pong X tiles (2 * tile_m * lds_stride * elem_bytes bytes)
    # - optional CShuffle tile (stage1 uses 2xf16 vector store, sized in 4B pairs)
    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = 4 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    @flyc.kernel
    def moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        k_in = arith.index_cast(T.index, i32_k_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        tokens_i32_v = i32_tokens_in
        k_i32_v = i32_k_in
        x_elem = T.f16 if is_f16_a else T.f8
        vec4_f32 = T.vec(4, T.f32)
        vec4_i32 = T.vec(4, T.i32)
        vec1_f32 = T.vec(1, T.f32)
        vec16_elems = 16 if a_elem_bytes == 1 else 8
        vec16_x = T.vec(vec16_elems, x_elem)
        vec2_i64 = T.vec(2, T.i64)

        def silu(x):
            # Align with CK's device fast path:
            #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
            #   sig = rcp(1 + emu)                   -> v_rcp_f32
            #   y = x * sig
            t = x * (-1.4426950408889634)  # -log2(e)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        _arith_min = getattr(arith, "minimum", None) or getattr(arith, "minimumf")
        _arith_max = getattr(arith, "maximum", None) or getattr(arith, "maximumf")

        def swiglu(gate, up, alpha=1.702, limit=7.0):
            gate = _arith_min(gate, limit)
            up = _arith_min(up, limit)
            up = _arith_max(up, -limit)

            t = gate * alpha * (-1.4426950408889634)  # -log2(e)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return gate * sig * (up + 1.0)

        acc_init = arith.constant_vector(0.0, vec4_f32)

        # B preshuffle layout: match GEMM test helper exactly.
        c_n_total = fx.Index(experts * (2 * inter_dim))
        kpack_bytes = 8 if is_int4 else 16
        b_layout = make_preshuffle_b_layout(
            arith,
            c_n=c_n_total,
            c_k=k_in // pack_K,
            kpack_bytes=kpack_bytes,
            elem_bytes=b_elem_bytes,
        )
        layout_b = b_layout.layout_b

        m_repeat = tile_m // 16
        k_unroll = tile_k_bytes // 128  # K64-byte micro-step

        # A scale is sorted/padded by MoE routing, so its M dimension follows
        # the sorted row buffer (`blocks * tile_m`), not the raw token count.
        sorted_rows = size_expert_ids_in * fx.Index(tile_m)
        layout_a_scale = make_preshuffle_scale_layout(
            arith, c_mn=sorted_rows, c_k=k_in
        )
        layout_b_scale = make_preshuffle_scale_layout(
            arith, c_mn=c_n_total, c_k=k_in
        )

        shape_lds = fx.make_shape(tile_m, tile_k)
        stride_lds = fx.make_stride(lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        tx = gpu.thread_id("x")
        # Align with Aiter launch mapping (NSwizzle==false):
        # - blockIdx.x -> N dimension (tile along inter_dim)
        # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
        by = gpu.block_id("x")  # tile along inter_dim
        bx = gpu.block_id("y")  # tile along sorted M

        # Block validity: compute as early as possible so invalid blocks skip all buffer-resource
        # setup, LDS pointer math, and gmem prefetch work.
        bx_m = bx * fx.Index(tile_m)
        by_n = by * fx.Index(tile_n)

        maxids_rsrc = buffer_ops.create_buffer_resource(
            arg_max_token_ids, max_size=False, num_records_bytes=fx.Int32(4)
        )
        max_token_id_i32 = buffer_ops.buffer_load(
            maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
        )

        bias_rsrc = (
            buffer_ops.create_buffer_resource(arg_bias, max_size=False)
            if enable_bias
            else None
        )

        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32)
        # Common constants/atoms (hoisted): keep IR small like GEMM.
        # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
        k_blocks16 = fx.Index(tile_k_bytes // 16)
        _if_blk = scf.IfOp(blk_valid)
        with _if_then(_if_blk):
            x_lds_elem = _x_lds_elem_type()
            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                x_lds_elem,
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias LDS bytes as fp16 for optional CShuffle epilogue.
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)

            _lds_out_elems = tile_m * tile_n if _use_cshuffle_epilog else 0
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    _out_lds_elem_type(),
                    shape=(_lds_out_elems,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Use logical buffer sizes (descriptor num_records) so hardware OOB checking can be
            # used directly (CK-style). This allows us to avoid `select`-based masking for
            # invalid lanes and rely on the buffer instruction's built-in bounds behavior.
            x_nbytes = (
                tokens_in
                * (k_in // fx.Index(int(a_elem_vec_pack)))
                * fx.Index(int(elem_bytes))
            )
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes
            )
            _w_n = fx.Index(experts * (2 * inter_dim))
            _w_nbytes = (
                _w_n
                * (k_in // fx.Index(int(a_elem_vec_pack)))
                * fx.Index(int(elem_bytes))
            )
            _w_nbytes_i32 = arith.index_cast(T.i32, _w_nbytes)
            w_rsrc = buffer_ops.create_buffer_resource(
                arg_w, max_size=False, num_records_bytes=_w_nbytes_i32
            )

            # OUT: [tokens * topk * inter_dim] in f16/bf16 (2B each) or fp8 (1B each).
            _out_elem_bytes = 1 if out_dtype == "fp8" else 2
            out_nbytes = tokens_in * arith.constant(
                topk * inter_dim * _out_elem_bytes, index=True
            )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )

            if is_f16_a:
                sx_rsrc = None
            else:
                # A1 microscale: [sorted_rows, K/32] e8m0 bytes, packed as i32.
                _c32 = fx.Index(32)
                _kblk = k_in / _c32
                _sorted_rows = size_expert_ids_in * arith.constant(
                    tile_m, index=True
                )
                sx_nbytes = _sorted_rows * _kblk
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                )

            if is_f16_b:
                sw_rsrc = None
            else:
                # W1 microscale: [experts * 2 * inter_dim, K/32] e8m0 bytes.
                _c32_w = fx.Index(32)
                _kblk_w = k_in / _c32_w
                _mn_w = fx.Index(experts * (2 * inter_dim))
                sw_nbytes = _mn_w * _kblk_w
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes)
                sw_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32
                )

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes = size_expert_ids_in * arith.constant(
                tile_m * 4, index=True
            )
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i32,
            )
            sorted_w_rsrc = (
                buffer_ops.create_buffer_resource(
                    arg_sorted_weights,
                    max_size=False,
                    num_records_bytes=sorted_nbytes_i32,
                )
                if doweight_stage1
                else None
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_i32 = arith.index_cast(
                T.i32, size_expert_ids_in * fx.Index(4)
            )
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )

            # Expert id for this M tile (keep address math in `index`)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            exp_valid = arith.cmpi(
                arith.CmpIPredicate.ult, expert_i32, fx.Int32(experts)
            )  # todo fix
            _ifexpert_of = scf.IfOp(exp_valid)
            with _if_then(_ifexpert_of):
                expert_idx = arith.index_cast(T.index, expert_i32)
                inter2_idx = fx.Index(2 * inter_dim)
                expert_off_idx = expert_idx * inter2_idx  # index

                bx_m = bx * fx.Index(tile_m)

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Keep a fixed 16B gmem->reg schedule (dwordx4) to match preshuffle_gemm_flyc.py.
                if bytes_per_thread_x % 16 != 0:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                    )
                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

                # Work in dword units along K: K_dwords = (K_packed_bytes)/4.
                # For fp4, 2 elements per byte, so divide by a_elem_vec_pack.
                c_a_pack = fx.Index(int(a_elem_vec_pack))
                c_k_div4 = (
                    (k_in // c_a_pack) * fx.Index(int(elem_bytes))
                ) // arith.index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
                mask24 = fx.Int32(0xFFFFFF)
                # Keep i32 constants available for epilogue index math.
                topk_i32 = fx.Int32(topk)

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

                # CK-aligned: decode token once (per thread's M-slice) and build a base row offset.
                x_row_base_div4 = []
                x_row_valid = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    sorted_row_i32 = arith.index_cast(T.i32, sorted_row_i)
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, sorted_row_i32, max_token_id_i32
                    )
                    sorted_row_safe = arith.select(
                        row_valid, sorted_row_i, fx.Index(0)
                    )
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_safe, vec_width=1, dtype=T.i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = fused_i >> fx.Int32(24)
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = row_valid & (t_valid & s_valid)
                    x_row_valid.append(ts_valid)
                    t_safe = arith.select(ts_valid, t_i32, fx.Int32(0))
                    t_idx = arith.index_cast(T.index, t_safe)
                    x_row_base_div4.append(t_idx * c_k_div4)

                vec4_i32 = T.vec(4, T.i32)

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    """
                    idx_elem = (
                        idx_i32 if elem_bytes == 1 else (idx_i32 * arith.index(2))
                    )
                    return buffer_copy_gmem16_dwordx4(
                        buffer_ops,
                        vector,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                    )

                _zero_row_idx = fx.Index(0)

                def load_x_tile(base_k):
                    """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                    base_k_div4 = (
                        (base_k // c_a_pack)
                        * fx.Index(int(elem_bytes))
                    ) // arith.index(4)
                    zero_x_i32 = arith.constant_vector(0, vec4_i32)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        safe_base = arith.select(
                            x_row_valid[i], x_row_base_div4[i], _zero_row_idx
                        )
                        idx_i32 = safe_base + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        x_i32 = vector.bitcast(vec4_i32, x_vec)
                        x_i32 = arith.select(x_row_valid[i], x_i32, zero_x_i32)
                        parts.append(x_i32)
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                wave_id, lane_id = split_row_major_2d(tx, fx.Index(64))
                lane_div_16, lane_mod_16 = split_row_major_2d(
                    lane_id, fx.Index(16)
                )

                # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16*16B (KPackBytes=16).
                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * fx.Index(16)

                # Dynamic N tiling within block (same as existing kernels)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_mod_4 = wave_id % arith.index(4)
                n_tile_base = wave_mod_4 * c_n_per_wave

                # fp4 pack
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # Precompute gate/up B coordinates and output columns for CK-style stage1:
                # weights/scales are laid out as [gate rows][up rows], so each output column
                # pairs one gate row with the matching up row at +inter_dim.
                col_g_list = []
                inter_idx = fx.Index(inter_dim)
                out_block_base = by_n
                out_wave_base = n_tile_base
                gate_n_intra_list = []
                gate_n_blk_list = []
                up_n_intra_list = []
                up_n_blk_list = []
                for i in range_constexpr(num_acc_n):
                    offset = i * 16
                    c_offset = fx.Index(offset)
                    out_col = (
                        out_block_base + out_wave_base + c_offset + lane_mod_16
                    )
                    col_g_list.append(out_col)

                    gate_row_w = expert_off_idx + out_col
                    gate_n_blk, gate_n_intra = split_row_major_2d(
                        gate_row_w, fx.Index(16)
                    )
                    gate_n_blk_list.append(gate_n_blk)
                    gate_n_intra_list.append(gate_n_intra)

                    up_row_w = gate_row_w + inter_idx
                    up_n_blk, up_n_intra = split_row_major_2d(
                        up_row_w, fx.Index(16)
                    )
                    up_n_blk_list.append(up_n_blk)
                    up_n_intra_list.append(up_n_intra)

                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                    # K64 micro-step = 2x K32 MFMA steps. Reuse the shared helper.
                    b0 = load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ku * 2,
                        n_blk=n_blk,
                        n_intra=n_intra,
                        lane_div_16=lane_div_16,
                        elem_type=_w_elem_type(is_f4_b=is_f4_b, is_f16_b=is_f16),
                        kpack_bytes=kpack_bytes,
                        elem_bytes=b_elem_bytes,
                        unpack_int4=bool(is_int4),
                    )
                    b1 = load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ku * 2 + 1,
                        n_blk=n_blk,
                        n_intra=n_intra,
                        lane_div_16=lane_div_16,
                        elem_type=_w_elem_type(is_f4_b=is_f4_b, is_f16_b=is_f16),
                        kpack_bytes=kpack_bytes,
                        elem_bytes=b_elem_bytes,
                        unpack_int4=bool(is_int4),
                    )
                    return b0, b1

                def load_b_tile(base_k):
                    gate_b_tile = []
                    up_b_tile = []
                    for ku in range_constexpr(k_unroll):
                        gate_packs0 = []
                        gate_packs1 = []
                        up_packs0 = []
                        up_packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            gate_b0, gate_b1 = load_b_packs_k64(
                                base_k,
                                ku,
                                gate_n_blk_list[ni],
                                gate_n_intra_list[ni],
                            )
                            up_b0, up_b1 = load_b_packs_k64(
                                base_k,
                                ku,
                                up_n_blk_list[ni],
                                up_n_intra_list[ni],
                            )
                            gate_packs0.append(gate_b0)
                            gate_packs1.append(gate_b1)
                            up_packs0.append(up_b0)
                            up_packs1.append(up_b1)
                        gate_b_tile.append((gate_packs0, gate_packs1))
                        up_b_tile.append((up_packs0, up_packs1))
                    return gate_b_tile, up_b_tile

                def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    # Direct arith crd2idx: idx = mni*stride_n0 + ku*stride_k0 + k_lane*stride_klane + n_lane
                    idx_pack = (
                        mni * scale_info.stride_n0
                        + ku * scale_info.stride_k0
                        + k_lane * scale_info.stride_klane
                        + n_lane
                    )
                    s = buffer_ops.buffer_load(
                        rsrc, idx_pack, vec_width=1, dtype=T.i32
                    )
                    return vector.from_elements(T.vec(1, T.i32), [s])

                def load_scale_masked(arg_scale, rsrc, scale_info, ku, mni, valid):
                    safe_mni = arith.select(
                        valid, mni, fx.Index(0)
                    )
                    scale_i32 = vector.extract(
                        load_scale(arg_scale, rsrc, scale_info, ku, safe_mni),
                        static_position=[0],
                        dynamic_position=[],
                    )
                    scale_i32 = arith.select(valid, scale_i32, fx.Int32(0))
                    return vector.from_elements(T.vec(1, T.i32), [scale_i32])

                def load_b_scale_tile(base_k):
                    gate_b_scale_tile = []
                    up_b_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            col_offset = ni * 16 * pack_N
                            col_offset_idx = fx.Index(col_offset)
                            col_base = (
                                out_block_base + out_wave_base + col_offset_idx
                            )
                            col_valid = arith.cmpi(
                                arith.CmpIPredicate.ult, col_base, inter_idx
                            )
                            gate_mni = (
                                expert_off_idx + col_base
                            ) // fx.Index(32)
                            up_mni = (
                                expert_off_idx + inter_idx + col_base
                            ) // fx.Index(32)
                            gate_scale_i32 = load_scale_masked(
                                arg_scale_w,
                                sw_rsrc,
                                layout_b_scale,
                                ku + base_k,
                                gate_mni,
                                col_valid,
                            )
                            up_scale_i32 = load_scale_masked(
                                arg_scale_w,
                                sw_rsrc,
                                layout_b_scale,
                                ku + base_k,
                                up_mni,
                                col_valid,
                            )
                            gate_b_scale_tile.append(gate_scale_i32)
                            up_b_scale_tile.append(up_scale_i32)
                    return gate_b_scale_tile, up_b_scale_tile

                def load_a_scale_tile(base_k):
                    a_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            scale = load_scale(
                                arg_scale_x,
                                sx_rsrc,
                                layout_a_scale,
                                ku + base_k,
                                mi + bx_m // pack_M // 16,
                            )
                            a_scale_tile.append(scale)
                    return a_scale_tile

                def prefetch_ab_scale_tile(base_k):
                    gate_bs, up_bs = load_b_scale_tile(base_k)
                    return [load_a_scale_tile(base_k), gate_bs, up_bs]

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
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    # Swizzle in bytes, then convert to element offset for memref indexing.
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(2))
                    )
                    idx_a16 = lds_row_major_idx(
                        curr_row_a_lds,
                        col_base_swz,
                        fx.Index(lds_stride),
                        lds_base,
                    )
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_f8f6f4_tile(
                    acc_gate_in,
                    acc_up_in,
                    gate_b_tile_in,
                    up_b_tile_in,
                    lds_base,
                    *,
                    a0_prefetch=None,
                    a_scale=None,
                    gate_b_scale=None,
                    up_b_scale=None,
                    prefetch_epilogue: bool = False,
                ):
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in)

                    # Re-sync all threads before consuming the current LDS tile.
                    gpu.barrier()
                    rocdl.sched_barrier(0)

                    epilogue_pf = None
                    if enable_bias and prefetch_epilogue:
                        gate_bias = []
                        up_bias = []
                        for ni in range_constexpr(num_acc_n):
                            global_n = by_n + n_tile_base + ni * 16 + lane_mod_16
                            gate_offset = expert_off_idx + global_n
                            up_offset = expert_off_idx + global_n + inter_dim
                            gate_bias.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc, gate_offset, vec_width=1, dtype=T.f32
                                )
                            )
                            up_bias.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc, up_offset, vec_width=1, dtype=T.f32
                                )
                            )
                        epilogue_pf = (gate_bias, up_bias)

                    if (int(tile_k) % 128) != 0:
                        raise ValueError(
                            f"tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_k}"
                        )

                    mfma_res_ty = T.f32x4
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)
                    c0_i64 = arith.constant(0, type=T.i64)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    # Gate and Up MFMA interleaved in the same inner loop.
                    for ku128 in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                            a_scale_val = vector.extract(
                                a_scale_i32,
                                static_position=[0],
                                dynamic_position=[],
                            )
                            for ni in range_constexpr(num_acc_n_packed):
                                gate_bs_i32 = gate_b_scale[
                                    ku128 * num_acc_n_packed + ni
                                ]
                                gate_bs_val = vector.extract(
                                    gate_bs_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                up_bs_i32 = up_b_scale[
                                    ku128 * num_acc_n_packed + ni
                                ]
                                up_bs_val = vector.extract(
                                    up_bs_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl
                                    gate_bp0, gate_bp1 = gate_b_tile_in[k_idx]
                                    up_bp0, up_bp1 = up_b_tile_in[k_idx]
                                    col_base = (
                                        col_offset_base
                                        + (k_idx * 128) // a_elem_vec_pack
                                    )
                                    for imxdl in range_constexpr(pack_M):
                                        col_base0 = col_base
                                        mi_idx = mi * pack_M + imxdl
                                        mi_val = arith.constant(
                                            mi_idx * 16, index=True
                                        )
                                        curr_row_a_lds = row_a_lds + mi_val

                                        a0, a1 = lds_load_packs_k64(
                                            curr_row_a_lds, col_base0, lds_base
                                        )

                                        if is_f8_a:
                                            col_base1 = col_base + 64
                                            a2, a3 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base1, lds_base
                                            )
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, a2, a3
                                            )
                                        else:
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, c0_i64, c0_i64
                                            )

                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl
                                            acc_idx = mi_idx * num_acc_n + ni_idx

                                            gb0 = gate_bp0[ni_idx]
                                            gb1 = gate_bp1[ni_idx]
                                            gb128 = pack_i64x4_to_i32x8(
                                                gb0, gb1, c0_i64, c0_i64
                                            )

                                            ub0 = up_bp0[ni_idx]
                                            ub1 = up_bp1[ni_idx]
                                            ub128 = pack_i64x4_to_i32x8(
                                                ub0, ub1, c0_i64, c0_i64
                                            )

                                            rocdl.sched_barrier(0)
                                            gate_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        gb128,
                                                        gate_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        gate_bs_val,
                                                    ],
                                                )
                                            )
                                            rocdl.sched_barrier(0)
                                            up_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        ub128,
                                                        up_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        up_bs_val,
                                                    ],
                                                )
                                            )
                    return gate_list, up_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = fx.Index(tile_m * lds_stride)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems

                # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    mfma_group = num_acc_n * 2
                    # K64 micro-step: 2x K32 MFMA per gemm.
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

                    # DS-read preload (CK default is 2); clamp to non-negative.
                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(2)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)

                    # DS-write hints near the end: match total X LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    if dswr_tail > sche_iters:
                        dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail
                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if sche_i >= dswr_start - 1:
                            rocdl.sched_dswr(1)
                    rocdl.sched_barrier(0)

                # Prologue: prefetch tile0, store to LDS(cur), sync.
                k0 = arith.index(0)
                x_regs0 = load_x_tile(k0)
                gate_w0, up_w0 = load_b_tile(k0)

                a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(
                    k0 // 2
                )
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                gpu.barrier()

                # Loop-carried ping/pong state.
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt
                gate_w_pong = gate_w0
                up_w_pong = up_w0

                a0_prefetch_pong = None

                if os.environ.get("FLYDSL_STAGE1_EARLY_RETURN", "0") == "1":
                    return

                num_k_tiles_py = int(model_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if k_main2_py < 0:
                    k_main2_py = 0

                _skip_compute = (
                    os.environ.get("FLYDSL_STAGE1_SKIP_COMPUTE", "0")
                    == "1"
                )
                if k_main2_py > 0:
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        k_iv = k_iv_py
                        next_k1 = k_iv + tile_k
                        x_regs_ping = load_x_tile(next_k1)
                        gate_w_ping, up_w_ping = load_b_tile(next_k1 // 2)
                        a_scale_ping, gate_bs_ping, up_bs_ping = (
                            prefetch_ab_scale_tile(next_k1 // pack_K // 128)
                        )

                        if _skip_compute:
                            store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                            gpu.barrier()
                            a0_prefetch_ping = None
                            next_k2 = k_iv + (tile_k * 2)
                            x_regs_pong = load_x_tile(next_k2)
                            gate_w_pong, up_w_pong = load_b_tile(next_k2 // 2)
                            a_scale_pong, gate_bs_pong, up_bs_pong = (
                                prefetch_ab_scale_tile(next_k2 // pack_K // 128)
                            )
                            store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                            gpu.barrier()
                            a0_prefetch_pong = None
                            continue

                        acc_gate, acc_up, _ = compute_f8f6f4_tile(
                            acc_gate,
                            acc_up,
                            gate_w_pong,
                            up_w_pong,
                            lds_base_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a_scale=a_scale_pong,
                            gate_b_scale=gate_bs_pong,
                            up_b_scale=up_bs_pong,
                        )
                        a0_prefetch_pong = None
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        gpu.barrier()

                        a0_prefetch_ping = None

                        next_k2 = k_iv + (tile_k * 2)
                        x_regs_pong = load_x_tile(next_k2)
                        gate_w_pong, up_w_pong = load_b_tile(next_k2 // 2)
                        a_scale_pong, gate_bs_pong, up_bs_pong = (
                            prefetch_ab_scale_tile(next_k2 // pack_K // 128)
                        )

                        acc_gate, acc_up, _ = compute_f8f6f4_tile(
                            acc_gate,
                            acc_up,
                            gate_w_ping,
                            up_w_ping,
                            lds_base_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a_scale=a_scale_ping,
                            gate_b_scale=gate_bs_ping,
                            up_b_scale=up_bs_ping,
                        )
                        a0_prefetch_ping = None
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        gpu.barrier()

                        a0_prefetch_pong = None

                if odd_k_tiles:
                    acc_gate, acc_up, epilogue_pf = compute_f8f6f4_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a_scale=a_scale_pong,
                        gate_b_scale=gate_bs_pong,
                        up_b_scale=up_bs_pong,
                        prefetch_epilogue=True,
                    )
                else:
                    k_tail1 = k_in - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    gate_w_ping, up_w_ping = load_b_tile(k_tail1 // 2)
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                        k_tail1 // pack_K // 128
                    )

                    acc_gate, acc_up, _ = compute_f8f6f4_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a_scale=a_scale_pong,
                        gate_b_scale=gate_bs_pong,
                        up_b_scale=up_bs_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    gpu.barrier()

                    a0_prefetch_ping = None

                    acc_gate, acc_up, epilogue_pf = compute_f8f6f4_tile(
                        acc_gate,
                        acc_up,
                        gate_w_ping,
                        up_w_ping,
                        lds_base_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a_scale=a_scale_ping,
                        gate_b_scale=gate_bs_ping,
                        up_b_scale=up_bs_ping,
                        prefetch_epilogue=True,
                    )

                # Store epilogue to out[t, slot, inter]
                topk_i32_v = topk_i32
                inter_i32_v = fx.Int32(inter_dim)
                mask24_i32 = fx.Int32(0xFFFFFF)

                # Epilogue hoists to keep IR + Python build time small:
                col_i32_list = []
                for ni in range_constexpr(num_acc_n):
                    col_i32_list.append(arith.index_cast(T.i32, col_g_list[ni]))

                _lane_div_16_mul4 = lane_div_16 * arith.index(4)
                inter_i32_local = inter_i32_v

                # Optional: CK-style CShuffle epilogue for better global store coalescing.
                # Uses EVec=4 (buffer store "x4" of fp16 elements).
                _use_cshuffle_epilog = (out_dtype == "fp8") or bool(
                    use_cshuffle_epilog
                )

                _mask_even_i32 = fx.Int32(0xFFFFFFFE)

                if _use_cshuffle_epilog:
                    if lds_out is None:
                        raise RuntimeError(
                            "CShuffle epilogue enabled but lds_out is not allocated/aliased."
                        )

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
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, max_token_id_i32
                        )
                        row_safe = arith.select(
                            row_valid0, row, fx.Index(0)
                        )
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row_safe, vec_width=1, dtype=T.i32
                        )
                        _t2 = fused2 & mask24_i32

                        # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                        if doweight_stage1:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row_safe, vec_width=1, dtype=T.f32
                            )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            vu = vector.extract(
                                acc_up[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )

                            if enable_bias:
                                gate_bias_list, up_bias_list = epilogue_pf
                                vg = vg + gate_bias_list[ni]
                                vu = vu + up_bias_list[ni]

                            if act == "swiglu":
                                y = swiglu(vg, vu)
                            else:
                                y = silu(vg) * vu

                            if doweight_stage1:
                                y = y * tw

                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(vec1_f32, [y])
                            vector.store(v1, lds_out, [lds_idx], alignment=1)

                    def precompute_row(*, row_local, row):
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, max_token_id_i32
                        )
                        row_safe = arith.select(
                            row_valid0, row, fx.Index(0)
                        )
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row_safe, vec_width=1, dtype=T.i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_valid = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_valid = row_valid0 & (t_valid & s_valid)
                        t2_safe = arith.select(ts_valid, t2, fx.Int32(0))
                        s2_safe = arith.select(ts_valid, s2, fx.Int32(0))
                        idx0 = (t2_safe * topk_i32_v + s2_safe) * inter_i32_local
                        return idx0, ts_valid

                    def store_pair(
                        *, row_local, row, row_ctx, col_pair0, col_g0, frag
                    ):
                        idx0 = row_ctx
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_out = idx0 + col_i32
                        if out_dtype == "fp8":
                            frag = vector.bitcast(vec4_f32, frag)
                            frag0 = vector.extract(
                                frag, static_position=[0], dynamic_position=[]
                            )
                            frag1 = vector.extract(
                                frag, static_position=[1], dynamic_position=[]
                            )
                            frag2 = vector.extract(
                                frag, static_position=[2], dynamic_position=[]
                            )
                            frag3 = vector.extract(
                                frag, static_position=[3], dynamic_position=[]
                            )

                            out_fp8 = fx.Int32(0)
                            out_fp8 = rocdl.cvt_pk_fp8_f32(
                                src_a=arith.unwrap(frag0),
                                src_b=arith.unwrap(frag1),
                                old=arith.unwrap(out_fp8),
                                word_sel=0,
                                res=T.i32,
                            )
                            out_fp8 = rocdl.cvt_pk_fp8_f32(
                                src_a=arith.unwrap(frag2),
                                src_b=arith.unwrap(frag3),
                                old=arith.unwrap(out_fp8),
                                word_sel=1,
                                res=T.i32,
                            )
                            buffer_ops.buffer_store(out_fp8, out_rsrc, idx_out // 4)
                        else:
                            out_vec_ty = T.vec(4, _out_elem_type())
                            out_vals = []
                            for fi in range_constexpr(4):
                                frag_i = vector.extract(
                                    frag, static_position=[fi], dynamic_position=[]
                                )
                                out_vals.append(
                                    arith.trunc_f(_out_elem_type(), frag_i)
                                )
                            out_vec = vector.from_elements(out_vec_ty, out_vals)
                            buffer_ops.buffer_store(out_vec, out_rsrc, idx_out)

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
                        frag_elem_type=T.f32,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                    return

                def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                    # `row` is the sorted-row index (bx_m + row_in_tile).
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(
                        arith.CmpIPredicate.ult, row_i32, max_token_id_i32
                    )
                    row_safe = arith.select(
                        row_valid0, row, fx.Index(0)
                    )
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row_safe, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                    ts_valid = row_valid0 & (t_valid & s_valid)
                    t2_safe = arith.select(ts_valid, t2, fx.Int32(0))
                    s2_safe = arith.select(ts_valid, s2, fx.Int32(0))

                    # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                    idx0 = (t2_safe * topk_i32_v + s2_safe) * inter_i32_local

                    # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, row_safe, vec_width=1, dtype=T.f32
                        )

                    _if_valid = scf.IfOp(ts_valid)
                    with _if_then(_if_valid):
                        for ni in range_constexpr(num_acc_n):
                            col_i32 = col_i32_list[ni]
                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            vu = vector.extract(
                                acc_up[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            if enable_bias:
                                gate_bias_list, up_bias_list = epilogue_pf
                                vg = vg + gate_bias_list[ni]
                                vu = vu + up_bias_list[ni]

                            if act == "swiglu":
                                y = swiglu(vg, vu)
                            else:
                                y = silu(vg) * vu

                            if doweight_stage1:
                                y = y * tw

                            y = arith.trunc_f(_out_elem_type(), y)
                            idx_out = idx0 + col_i32
                            buffer_ops.buffer_store(y, out_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=False,
                    arith=arith,
                    range_constexpr=range_constexpr,
                    m_repeat=m_repeat,
                    lane_div_16=lane_div_16,
                    bx_m=bx_m,
                    body_row=_stage1_store_row,
                )

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage1,
        act,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
    )

    @flyc.jit
    def launch_mixed_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        inter_in = arith.index_cast(T.index, i32_inter_in)
        gx = inter_in // fx.Index(tile_n)
        gy = arith.index_cast(T.index, i32_size_expert_ids_in)

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mixed_moe_gemm1


@functools.lru_cache(maxsize=1024)
def compile_mixed_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Optional experiment: write per-(token,slot) output (no atomics) into an output shaped
    # [tokens*topk, model_dim] (or [tokens, topk, model_dim] flattened), then reduce over topk outside.
    # This can reduce atomic contention for small tokens at the cost of extra bandwidth / reduction.
    accumulate: bool = True,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 4,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    a_dtype:
      - "fp8": A2 is fp8
      - "fp16": A2 is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": A2 is int8
      - "fp4": A2 is fp4

    b_dtype:
      - "fp8": W is fp8
      - "fp16": W is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": W is int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "fp4": W is fp4

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    pack_M = 2
    pack_N = 2
    pack_K = 2

    elem_bytes = 1

    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    # K64-byte micro-step: always 64 bytes per `ku`. For fp16, this is 32 elements (2xK16 MFMA).
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
        )

    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    is_int4 = b_dtype == "int4"

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    pad_k = 0
    lds_stride = tile_k + pad_k
    # gfx950+ has buffer_atomic_pk_add_bf16; gfx942 uses global atomics via raw pointer.
    _has_buffer_atomic_bf16 = str(gpu_arch).startswith(("gfx95", "gfx12"))
    _needs_global_atomic_bf16 = out_is_bf16 and not _has_buffer_atomic_bf16
    if out_is_bf16 and not supports_bf16_global_atomics(gpu_arch):
        raise ValueError(f"out_dtype='bf16' requires bf16 global atomics, got arch={gpu_arch!r}")

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError(
                "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
            )
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
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    module_name = (
        f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_vscale_fix3"
    ).replace("-", "_")
    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Reuse a single allocation for both:
    # - ping-pong A2 tiles (2 * tile_m * lds_stride * elem_bytes bytes)
    # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )  # f16 bytes
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)

    def x_lds_elem():
        return T.f16 if is_f16_a else T.f8

    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
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
            arg_bias: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            n_in = arith.index_cast(T.index, i32_n_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
            x_elem = T.f16 if is_f16_a else T.f8
            vec4_f32 = T.vec(4, T.f32)
            vec4_i32 = T.vec(4, T.i32)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec8_elems = 8 if a_elem_bytes == 1 else 4
            vec4_elems = 4 if a_elem_bytes == 1 else 2
            vec16_x = T.vec(vec16_elems, x_elem)
            vec2_i64 = T.vec(2, T.i64)

            acc_init = arith.constant_vector(0.0, vec4_f32)

            # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
            topk_idx = fx.Index(topk)
            m_in = tokens_in * topk_idx
            # fly.make_shape requires i32/i64, not index
            m_i32_v = arith.index_cast(T.i32, m_in)
            k_i32_v = i32_k_in

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = fx.Index(experts * model_dim)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b

            # A&B's scale preshuffle layout
            # For fp4, k_in is already packed (inter_dim // a_elem_vec_pack), so we need original inter_dim
            c_k_orig = fx.Index(inter_dim)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=m_in, c_k=c_k_orig
            )
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=c_k_orig
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along model_dim
            bx_persist = gpu.block_id("y")  # tile along sorted M

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = fx.Index(tile_k_bytes // 16)
            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                x_lds_elem(),
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

            # lds_tid: alias LDS after max(x, out) for sorted_idx preload
            _lds_x_b = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
            _lds_out_b = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
            _lds_tid_off = max(_lds_x_b, _lds_out_b)
            lds_tid = SmemPtr(
                base_ptr, lds_x_ptr.byte_offset + _lds_tid_off, T.i32, shape=(tile_m,)
            ).get()

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = fx.Index(topk)

            # X(A2): buffer size in bytes, accounting for FP4 packing (2 elements per byte).
            # fp8/int8: 1 byte per element  -> bytes = tokens*topk * K
            # fp4:      2 elements per byte -> bytes = tokens*topk * K / 2
            c_a_pack = fx.Index(int(a_elem_vec_pack))
            c_elem_bytes = fx.Index(int(a_elem_bytes))
            x_nbytes_idx = ((tokens_in * c_topk) * k_in * c_elem_bytes) // c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = (
                tokens_in * n_in * fx.Index(out_elem_bytes)
            )
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in
                    * arith.index(topk)
                    * n_in
                    * fx.Index(out_elem_bytes)
                )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )

            # num_valid_ids (sorted padded MN) for scale sizing / guards.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=fx.Int32(4),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
            )
            num_valid_idx = arith.index_cast(T.index, num_valid_i32)

            # fp16 path ignores scales completely (implicit scale=1.0).
            if is_f16_a:
                sx_rsrc = None
            else:
                if is_f4_a:
                    # A2 microscale: packed i32 holding e8m0 bytes for [sorted_size, K/32].
                    c32 = fx.Index(32)
                    kblk = k_in // c32
                    # Total bytes = num_valid_ids * kblk.
                    sx_nbytes_idx = num_valid_idx * kblk
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                    )
                else:
                    # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                    sx_nbytes_idx = (tokens_in * c_topk) * fx.Index(4)
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                    )

            if is_f16_b:
                sw_rsrc = None
            else:
                # Weight microscale buffer (packed i32 holding e8m0 bytes).
                # Use an exact descriptor size so hardware OOB checking works.
                c32 = fx.Index(32)
                kblk_w = k_in // c32  # K/32
                mn_w = fx.Index(experts * model_dim)
                sw_nbytes_idx = mn_w * kblk_w  # bytes (e8m0)
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
                sw_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32
                )

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = (
                size_expert_ids_in
                * fx.Index(tile_m)
                * fx.Index(4)
            )
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i32,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * fx.Index(4)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )
            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )

            # ---- persist_m loop ----
            _PERSIST_M = persist_m
            _c0_p = arith.index(0)
            _c1_p = arith.index(1)
            _c_pm = arith.index(_PERSIST_M)
            _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p)
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()
            _mi_p = _for_persist.induction_variable
            bx = bx_persist * _c_pm + _mi_p
            bx_m = bx * fx.Index(tile_m)

            # Early-exit guard: skip garbage expert blocks beyond `num_valid_ids`.
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(T.index, expert_i32)
            exp_valid = arith.cmpi(
                arith.CmpIPredicate.ult, expert_i32, fx.Int32(experts)
            )

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                n_idx = fx.Index(model_dim)
                expert_off_idx = expert_idx * n_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16 we require 16B.
                if is_f16_a:
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                        )
                    x_load_bytes = 16
                else:
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
                vec4_i32 = T.vec(4, T.i32)
                vec2_i32 = T.vec(2, T.i32)
                vec1_i32 = T.vec(1, T.i32)

                c_k_div4 = (
                    (k_in // c_a_pack) * fx.Index(int(a_elem_bytes))
                ) // arith.index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                    4 * int(a_elem_vec_pack)
                )
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
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

                x_load_vec_elems = (
                    x_load_bytes if a_elem_bytes == 1 else x_load_bytes // a_elem_bytes
                )

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    """
                    if x_load_bytes == 16:
                        idx_elem = (
                            idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                        )
                    # 8B/4B: convert dword index to byte offset and use offset_in_bytes path.
                    idx_bytes = idx_i32 * arith.index(4)
                    return _buffer_load_vec(
                        buffer_ops,
                        vector,
                        x_rsrc,
                        idx_bytes,
                        elem_type=x_elem,
                        vec_elems=x_load_vec_elems,
                        elem_bytes=a_elem_bytes,
                        offset_in_bytes=True,
                    )

                # decode routed token once (per thread's M-slice) and build a base offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = fused_i >> fx.Int32(24)
                    # Keep `blk_valid` only; remove per-row token validity checks.

                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = t_valid & s_valid
                    t_safe = arith.select(ts_valid, t_i32, fx.Int32(0))
                    s_safe = arith.select(ts_valid, s_i32, fx.Int32(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)

                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        (base_k // c_a_pack)
                        * fx.Index(int(a_elem_bytes))
                    ) // arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)

                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(vector.bitcast(vec2_i32, x_vec))
                        else:
                            parts.append(vector.bitcast(vec1_i32, x_vec))
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                wave_id, lane_id = split_row_major_2d(tx, fx.Index(64))
                lane_div_16, lane_mod_16 = split_row_major_2d(
                    lane_id, fx.Index(16)
                )

                row_a_lds = lane_mod_16

                col_offset_base = lane_div_16 * fx.Index(16)

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
                for i in range_constexpr(num_acc_n):
                    offset = i * 16
                    c_offset = fx.Index(offset)
                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    row_w = expert_off_idx + global_n
                    n_blk, n_intra = split_row_major_2d(row_w, fx.Index(16))
                    n_blk_list.append(n_blk)
                    n_intra_list.append(n_intra)

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128  # K64-byte micro-step (2x MFMA)

                # fp4 pack
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                def load_b_packs_k64(base_k, ku: int, ni: int):
                    """Load one K64-byte B micro-step: single 16B load, split into 2x i64."""
                    c64 = fx.Index(64)
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
                    k0_base = base_k_bytes // c64
                    k0 = k0_base + fx.Index(ku)
                    k1 = lane_div_16
                    coord_pack = (
                        n_blk_list[ni],
                        k0,
                        k1,
                        n_intra_list[ni],
                        fx.Index(0),
                    )
                    idx_pack = crd2idx(coord_pack, layout_b)

                    vec_elems = kpack_bytes // int(b_elem_bytes)
                    b16 = _buffer_load_vec(
                        buffer_ops,
                        vector,
                        w_rsrc,
                        idx_pack,
                        elem_type=_w_elem_type(is_f4_b=is_f4_b, is_f16_b=is_f16_b),
                        vec_elems=vec_elems,
                        elem_bytes=b_elem_bytes,
                        offset_in_bytes=(b_elem_bytes == 1),
                    )
                    b_i64x2 = vector.bitcast(vec2_i64, b16)
                    b0 = vector.extract(
                        b_i64x2, static_position=[0], dynamic_position=[]
                    )
                    b1 = vector.extract(
                        b_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return b0, b1

                def load_b_tile(base_k):
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            b0, b1 = load_b_packs_k64(base_k, ku, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    # Direct arith crd2idx: idx = mni*stride_n0 + ku*stride_k0 + k_lane*stride_klane + n_lane
                    idx_pack = (
                        mni * scale_info.stride_n0
                        + ku * scale_info.stride_k0
                        + k_lane * scale_info.stride_klane
                        + n_lane
                    )
                    s = buffer_ops.buffer_load(rsrc, idx_pack, vec_width=1, dtype=T.i32)
                    return vector.from_elements(T.vec(1, T.i32), [s])

                def load_b_scale_tile(base_k):
                    b_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            scale = load_scale(
                                arg_scale_w,
                                sw_rsrc,
                                layout_b_scale,
                                ku + base_k,
                                ni
                                + (expert_off_idx + by_n + n_tile_base) // pack_N // 16,
                            )
                            b_scale_tile.append(scale)
                    return b_scale_tile

                def load_a_scale_tile(base_k):
                    a_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            scale = load_scale(
                                arg_scale_x,
                                sx_rsrc,
                                layout_a_scale,
                                ku + base_k,
                                mi + bx_m // pack_M // 16,
                            )
                            a_scale_tile.append(scale)
                    return a_scale_tile

                def prefetch_ab_scale_tile(base_k):
                    return [load_a_scale_tile(base_k), load_b_scale_tile(base_k)]

                vec8_x = T.vec(vec8_elems, x_elem)
                vec4_x_lds = T.vec(vec4_elems, x_elem)

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
                                tx_c4=arith.index(4),
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
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        else:  # x_load_bytes == 4
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x_lds,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    # Swizzle in bytes, then convert to element offset for memref indexing.
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(2))
                    )
                    idx_a16 = lds_row_major_idx(
                        curr_row_a_lds,
                        col_base_swz,
                        fx.Index(lds_stride),
                        lds_base,
                    )
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    a_scale=None,
                    b_scale=None,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = vec4_f32

                    epilogue_pf = None
                    bias = None
                    if prefetch_epilogue:
                        if enable_bias:
                            bias = []
                            for ni in range_constexpr(num_acc_n):
                                global_n = by_n + n_tile_base + ni * 16 + lane_mod_16
                                bias_offset = expert_off_idx + global_n
                                bias.append(
                                    buffer_ops.buffer_load(
                                        bias_rsrc, bias_offset, vec_width=1, dtype=T.f32
                                    )
                                )
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [
                                fx.Index(ii) for ii in range(4)
                            ]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = fx.Index(mi * 16)
                                for ii in range_constexpr(4):
                                    row_off_pf = (
                                        lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    )
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
                        epilogue_pf = (None, tw_pf, bias)

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    # fp4 path
                    for ku128 in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                            a_scale_val = vector.extract(
                                a_scale_i32, static_position=[0], dynamic_position=[]
                            )
                            for ni in range_constexpr(num_acc_n_packed):
                                b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                                b_scale_val = vector.extract(
                                    b_scale_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl

                                    b_packs0, b_packs1 = b_tile_in[k_idx]

                                    col_base = (
                                        col_offset_base
                                        + (k_idx * 128) // a_elem_vec_pack
                                    )

                                    for imxdl in range_constexpr(pack_M):
                                        col_base0 = col_base
                                        mi_idx = mi * pack_M + imxdl
                                        mi_val = fx.Index(mi_idx * 16)
                                        curr_row_a_lds = row_a_lds + mi_val

                                        if (
                                            (a0_prefetch is not None)
                                            and (k_idx == 0)
                                            and (mi_idx == 0)
                                        ):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base0, lds_base
                                            )

                                        if is_f8_a:
                                            col_base1 = col_base + 64
                                            a2, a3 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base1, lds_base
                                            )
                                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                        else:
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, c0_i64, c0_i64
                                            )

                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl

                                            b0 = b_packs0[ni_idx]
                                            b1 = b_packs1[ni_idx]
                                            b128 = pack_i64x4_to_i32x8(
                                                b0, b1, c0_i64, c0_i64
                                            )

                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            rocdl.sched_barrier(0)
                                            acc_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        b128,
                                                        acc_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        b_scale_val,
                                                    ],
                                                )
                                            )

                    return acc_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = fx.Index(tile_m * lds_stride)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    # - MFMA group size per "slot": num_acc_n
                    # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                    # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    if num_acc_n < 4:
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)

                    # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    if dswr_tail > sche_iters:
                        dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail

                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if sche_i >= dswr_start - 1:
                            rocdl.sched_dswr(1)

                    rocdl.sched_barrier(0)

                # Prologue.
                k0 = arith.index(0)
                x_regs0 = load_x_tile(k0)
                b_cur = load_b_tile(k0)
                a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(k0 // pack_K // 128)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                # Preload sorted_idx into lds_tid for epilogue precompute_row
                _c_tile_m_idx = fx.Index(tile_m)
                _tid_in_range = arith.cmpi(arith.CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with ir.InsertionPoint(_if_tid.then_block):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])
                    scf.YieldOp([])
                gpu.barrier()

                acc = [acc_init] * num_acc_n * m_repeat
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base, lds_base_pong
                )
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

                c2_tile_k = fx.Index(tile_k * 2)
                b_pong = b_cur
                # Only emit the scf.for when there are actually iterations to run.
                # When k_main2_py == 0 the loop body is empty; emitting an scf.for
                # would create a region whose internal SSA values cannot be used
                # by the post-loop tail code.
                if k_main2_py > 0:
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        k_iv = k_iv_py
                        next_k1 = k_iv + tile_k
                        x_regs_ping = load_x_tile(next_k1)
                        b_ping = load_b_tile(next_k1 // 2)
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                            next_k1 // pack_K // 128
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_pong,
                            lds_base_pong,
                            a_scale_pong,
                            b_scale_pong,
                            a0_prefetch=a0_prefetch_pong,
                        )
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the ping tile we are about to compute.
                        a0_prefetch_ping = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_ping
                        )

                        next_k2 = k_iv + c2_tile_k
                        x_regs_pong = load_x_tile(next_k2)
                        b_pong = load_b_tile(next_k2 // 2)
                        a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                            next_k2 // pack_K // 128
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_ping,
                            lds_base_ping,
                            a_scale_ping,
                            b_scale_ping,
                            a0_prefetch=a0_prefetch_ping,
                        )
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the next pong tile.
                        a0_prefetch_pong = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_pong
                        )

                if odd_k_tiles:
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        prefetch_epilogue=True,
                    )

                else:
                    # Tail: 2 remaining tiles.
                    k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    b_ping = load_b_tile(k_tail1 // 2)
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                        k_tail1 // pack_K // 128
                    )

                    acc, _ = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )

                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    # hot_loop_scheduler()
                    gpu.barrier()

                    # Epilogue tile with sw prefetch.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_ping
                    )
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_ping,
                        lds_base_ping,
                        a_scale_ping,
                        b_scale_ping,
                        a0_prefetch=a0_prefetch_ping,
                        prefetch_epilogue=True,
                    )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.

                model_i32 = fx.Int32(model_dim)
                zero_i32 = fx.Int32(0)
                c2_i32 = fx.Int32(2)
                mask_even_i32 = fx.Int32(0xFFFFFFFE)

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
                bias_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf, bias_pf = epilogue_pf

                mask24_i32 = fx.Int32(0xFFFFFF)
                topk_i32_v = topk_i32

                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if lds_out is None:
                    raise RuntimeError(
                        "FLYDSL_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                    )

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
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(
                            acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        if enable_bias:
                            v = v + bias_pf[ni]

                        if doweight_stage2:
                            v = v * tw
                        v_out = arith.trunc_f(out_elem(), v)

                        lds_idx = row_base_lds + col_local
                        vec1_out = T.vec(1, out_elem())
                        v1 = vector.from_elements(vec1_out, [v_out])

                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    # Use lds_tid (sorted_idx preloaded to LDS) instead of buffer_load
                    # to avoid extra VMEM round-trips in the epilogue.
                    fused2 = memref.load(lds_tid, [row_local])
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                    s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                    row_valid = row_valid0 & (t_ok & s_ok)

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
                    e_vec=2,
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
            with ir.InsertionPoint(_if_blk.then_block):
                _ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(_ifexpert_of.then_block):
                    _moe_gemm2_then_body()
                    scf.YieldOp([])
                scf.YieldOp([])

            gpu.barrier()
            scf.YieldOp([])
            _for_ip.__exit__(None, None, None)

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage2,
        accumulate,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
        persist_m,
    )

    @flyc.jit
    def launch_mixed_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_bias: fx.Tensor,
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
        gx = n_in // fx.Index(tile_n)
        _c_pm_l = fx.Index(persist_m)
        gy = (
            arith.index_cast(T.index, i32_size_expert_ids_in)
            + _c_pm_l
            - fx.Index(1)
        ) / _c_pm_l

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
            arg_bias,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mixed_moe_gemm2

# ---------------------------------------------------------------------------
# Stage 2: A16W4 MXFP4 kernel (BF16 activations x FP4 E2M1 weights)
# ---------------------------------------------------------------------------

def _decode_e8m0_byte_to_f32(byte_i8, arith_mod):
    """Convert a single E8M0 byte (i8) to f32 = 2^(e - 127)."""
    c23 = arith_mod.constant(23, type=T.i32)
    byte_u32 = arith_mod.extui(T.i32, byte_i8)
    scale_bits = arith_mod.shli(byte_u32, c23)
    return arith_mod.bitcast(T.f32, scale_bits)


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm.

    Bypasses LLVM SIInsertWaitcnts which would insert a conservative
    s_waitcnt vmcnt(0) lgkmcnt(0) before every S_BARRIER MI.
    """
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    parts.append("s_barrier")
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )

@functools.lru_cache(maxsize=1024)
def compile_a16w4_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    out_dtype: str = "bf16",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    accumulate: bool = True,
    waves_per_eu: int = 0,
    k_batch: int = 1,
):
    """Compile stage2 A16W4 MXFP4 kernel and return the compiled executable.

    A2 is bf16.  W is MXFP4 (FP4 E2M1) with E8M0 block scales, pre-shuffled
    by ``shuffle_weight_a16w4`` and ``shuffle_scale_a16w4``.

    enable_bias: add per-column f32 bias after GEMM accumulation.
    model_dim_pad / inter_dim_pad: padding semantics (see stage1).
    """
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    elem_bytes = 2  # bf16 activations
    kpack_bytes = 16  # MXFP4 preshuffle

    _mfma_k32_raw = getattr(rocdl, "mfma_f32_16x16x32_bf16_", None)
    if _mfma_k32_raw is None:
        raise AttributeError(
            "BF16 K32 MFMA op not found: expected `rocdl.mfma_f32_16x16x32_bf16_`"
        )
    _split_mfma = rocdl._split_mfma_operands

    def mfma_f32_bf16_k32(result_type, operands, *, loc=None, ip=None):
        a, b, c, cbsz, abid, blgp = _split_mfma(operands, loc=loc)
        return _mfma_k32_raw(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)

    tile_k_bytes = int(tile_k) * int(elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes}"
        )

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_a16w4_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )

    DYN = ir.ShapedType.get_dynamic_size()
    # FP4: 2 nibbles per byte → half the byte count.
    size_w = (experts * model_dim * inter_dim) // 2

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1", "true", "True", "YES", "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k

    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics, got arch={gpu_arch!r}"
            )

    if out_is_f32:
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError("out_dtype='f32' does not support CShuffle epilogue.")
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE2_CSHUFFLE", "1") in (
                "1", "true", "True", "YES", "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue."
            )

    def out_elem():
        ty = T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)
        return ty() if callable(ty) else ty

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

    _single_x_bytes = int(tile_m) * int(lds_stride) * int(elem_bytes)
    _single_x_elems = _single_x_bytes // int(elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )
    _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
    _ping_buffer_bytes = _single_x_bytes

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes

    _sorted_info_elems = int(tile_m)
    _sorted_info_bytes = _sorted_info_elems * 4
    lds_sorted_info_offset = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = lds_sorted_info_offset + _sorted_info_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    if waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    _k_batch = int(k_batch)
    if _k_batch > 1:
        if inter_dim % (_k_batch * tile_k) != 0:
            raise ValueError(
                f"inter_dim={inter_dim} must be divisible by k_batch*tile_k="
                f"{_k_batch * tile_k}"
            )
    _k_dim = inter_dim // _k_batch
    _total_tiles_check = _k_dim // tile_k
    if _total_tiles_check < 2 or _total_tiles_check % 2 != 0:
        raise ValueError(
            f"k_batch={_k_batch}: _k_dim/tile_k={_total_tiles_check} must be "
            f"even and >= 2 for the ping-pong pipeline"
        )

    _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu >= 1 else ""
    _kb_tag = f"_kb{_k_batch}" if _k_batch > 1 else ""
    _bias_tag = "_bias" if enable_bias else ""
    _pad_tag = f"_mp{model_dim_pad}_ip{inter_dim_pad}" if (model_dim_pad or inter_dim_pad) else ""
    module_name = (
        f"mfma_a16w4_moe2_mxfp4_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"{_wpe_tag}{_kb_tag}{_bias_tag}{_pad_tag}_abi1"
    ).replace("-", "_")

    if True:

        @flyc.kernel
        def moe_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in.ir_value())
            n_in = arith.ArithValue(arith.index_cast(T.index, i32_n_in.ir_value()))
            k_in = arith.index_cast(T.index, i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                T.index, i32_size_expert_ids_in.ir_value()
            )
            k_i32_v = i32_k_in.ir_value()

            x_elem = T.bf16
            w_elem = T.i8  # packed FP4 stored as bytes
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec1_i32 = T.vec(1, i32)
            vec4_i16 = T.vec(4, T.i16)
            vec8_bf16 = T.vec(8, x_elem)
            vec16_bf16 = T.vec(16, x_elem) if elem_bytes == 2 else T.vec(8, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            acc_init = arith.constant_vector(0.0, vec4_f32)

            # A2 layout
            topk_idx = arith.index(topk)
            m_in = tokens_in * topk_idx
            m_i32_v = arith.index_cast(i32, m_in)
            layout_x = fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: MXFP4 kpack=16, c_k = k_in // 2
            c_n_total = arith.index(experts * model_dim)
            c2 = arith.index(2)
            c_k_packed = k_in // c2  # FP4: 2 nibbles per byte
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=c_k_packed,
                kpack_bytes=kpack_bytes,
                elem_bytes=1,
            )
            layout_b = b_layout.layout_b

            # Scale layout
            layout_b_scale = make_preshuffle_scale_layout(
                arith,
                c_mn=c_n_total,
                c_k=k_in,
                mn_pack=2,
                k_pack=2,
                elem_bytes=4,
                scale_block_size=32,
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along model_dim
            bx = gpu.block_id("y")  # tile along sorted M

            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr_pong = allocator_pong.get_base()
            base_ptr_ping = allocator_ping.get_base()
            lds_x_pong = SmemPtr(
                base_ptr_pong, lds_pong_offset, T.bf16,
                shape=(_single_x_elems,),
            ).get()
            lds_x_ping = SmemPtr(
                base_ptr_ping, lds_ping_offset, T.bf16,
                shape=(_single_x_elems,),
            ).get()
            lds_out = (
                SmemPtr(
                    base_ptr_pong, lds_pong_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )
            lds_sorted_cache = SmemPtr(
                base_ptr_pong, lds_sorted_info_offset, T.i32,
                shape=(_sorted_info_elems,),
            ).get()

            c_topk = arith.index(topk)

            x_nbytes_idx = (tokens_in * c_topk) * k_in * arith.index(int(elem_bytes))
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_idx
            )
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * arith.index(out_elem_bytes)
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in * arith.index(topk) * n_in * arith.index(out_elem_bytes)
                )
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_idx
            )

            sorted_nbytes_idx = (
                size_expert_ids_in * arith.index(tile_m) * arith.index(4)
            )
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_idx,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_idx
            )

            eid_nbytes_idx = size_expert_ids_in * arith.index(4)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_idx
            )
            bx_m = bx * arith.index(tile_m)

            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids, max_size=False, num_records_bytes=arith.index(4),
            )
            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.index(0), vec_width=1, dtype=i32
            )
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            def _moe_gemm2_then_body():
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                n_idx = arith.index(model_dim)
                expert_off_idx = expert_idx * n_idx

                if bytes_per_thread_x >= 16 and bytes_per_thread_x % 16 == 0:
                    x_load_bytes = 16
                elif bytes_per_thread_x >= 8 and bytes_per_thread_x % 8 == 0:
                    x_load_bytes = 8
                elif bytes_per_thread_x >= 4 and bytes_per_thread_x % 4 == 0:
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be "
                        f"divisible by 4"
                    )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4
                x_vec_elems = x_load_bytes // elem_bytes
                x_vec_i32_ty = T.vec(chunk_i32, i32) if chunk_i32 > 1 else T.vec(1, i32)
                x_vec_x_ty = T.vec(x_vec_elems, x_elem)
                vec16_x = T.vec(8, x_elem)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // arith.index(4)
                c_k_div4_i32 = arith.index_cast(i32, c_k_div4)
                layout_x_div4 = fx.make_layout(
                    (m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                )
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.constant(topk, type=T.i32)
                mask24 = arith.constant(0xFFFFFF, type=T.i32)
                tokens_i32 = arith.index_cast(i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32
                    )
                    _fused_v1 = vector.from_elements(vec1_i32, [fused_i])
                    vector.store(_fused_v1, lds_sorted_cache, [row_local])
                    t_i32 = fused_i & mask24
                    s_i32 = arith.shrui(fused_i, arith.constant(24, type=T.i32))
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = t_valid & s_valid
                    t_safe = arith.select(
                        ts_valid, t_i32, arith.constant(0, type=T.i32)
                    )
                    s_safe = arith.select(
                        ts_valid, s_i32, arith.constant(0, type=T.i32)
                    )
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        idx_elem = idx_i32 * arith.index(2)
                        x_vec = _buffer_load_vec(
                            buffer_ops, vector, x_rsrc, idx_elem,
                            elem_type=x_elem,
                            vec_elems=x_vec_elems,
                            elem_bytes=elem_bytes,
                            offset_in_bytes=False,
                        )
                        parts.append(vector.bitcast(x_vec_i32_ty, x_vec))
                    return parts

                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                _dma_bytes = 16
                _wave_size = 64

                def dma_x_tile_to_lds(base_k, lds_buffer):
                    """Async DMA: global -> LDS via buffer_load_lds, no VGPR."""
                    c4_idx = arith.index(4)
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // arith.index(4)

                    lds_ptr_i64 = None
                    for i in range_constexpr(num_x_loads):
                        row_local_i = x_row_local[i]
                        col_local_i32_i = x_col_local_i32[i]
                        col_local_sw = swizzle_xor16(
                            row_local_i, col_local_i32_i * c4_idx, k_blocks16
                        )
                        row_k_dw = x_row_base_div4[i] + base_k_div4
                        global_byte_idx = row_k_dw * c4_idx + col_local_sw
                        global_offset = arith.index_cast(i32, global_byte_idx)

                        if i == 0:
                            lds_addr = memref.extract_aligned_pointer_as_index(
                                lds_buffer
                            ) + wave_id * arith.constant(
                                _wave_size * _dma_bytes, index=True
                            )
                            lds_ptr_i64 = rocdl.readfirstlane(
                                i64, arith.index_cast(i64, lds_addr)
                            )
                        else:
                            lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                total_threads * _dma_bytes, type=i64
                            )

                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            arith.constant(_dma_bytes, type=i32),
                            global_offset,
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                        )

                def prefetch_x_to_lds(base_k, lds_buffer):
                    dma_x_tile_to_lds(base_k, lds_buffer)

                row_a_lds = lane_mod_16
                _a_sublane_stride = 64   # 32 bf16 * 2 bytes
                _a_ku_stride_bytes = 16  # 8 bf16 * 2 bytes
                col_offset_base_bytes = lane_div_16 * arith.index(_a_sublane_stride)

                by_n = by * arith.index(tile_n)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.index(n_per_wave)
                wave_mod_4 = wave_id % arith.index(4)
                n_tile_base = wave_mod_4 * c_n_per_wave

                n_intra_list = []
                n_blk_list = []
                col_g_list = []
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
                k_unroll = tile_k_bytes // 64

                _pad_k_elems = (inter_dim_pad % tile_k) if (_k_batch == 1 and inter_dim_pad > 0) else 0
                _pad_ku_skip = _pad_k_elems // 32
                _tail_ku = k_unroll - _pad_ku_skip
                _tail_k0_count = (_tail_ku + 3) // 4 if _pad_ku_skip > 0 else None

                # ---- Scale index helpers ----
                # mni for each ni: (expert_off + by_n + n_tile_base + ni*16) // 32
                scale_mni_list = []
                scale_n_pack_list = []
                for ni in range_constexpr(num_acc_n):
                    n_global = expert_off_idx + by_n + n_tile_base + arith.index(ni * 16)
                    scale_mni_list.append(n_global // arith.index(32))
                    n_block_16 = n_global // arith.index(16)
                    scale_n_pack_list.append(n_block_16 % arith.index(2))

                def _load_scale_i32(scale_ku_idx, ni, scale_klane=None):
                    """Load one packed i32 from the scale buffer."""
                    _klane = scale_klane if scale_klane is not None else lane_div_16
                    idx = (scale_mni_list[ni] * layout_b_scale.stride_n0
                           + scale_ku_idx * layout_b_scale.stride_k0
                           + _klane * layout_b_scale.stride_klane
                           + lane_mod_16)
                    return buffer_ops.buffer_load(
                        sw_rsrc, idx, vec_width=1, dtype=i32
                    )

                def _extract_e8m0_f32_dynamic(packed_i32, byte_pos_idx):
                    """Extract E8M0 byte at runtime byte_pos and decode to f32."""
                    shift = arith.index_cast(i32, byte_pos_idx) * arith.constant(8, type=i32)
                    byte_i32 = arith.shrui(packed_i32, shift) & arith.constant(0xFF, type=i32)
                    scale_bits = arith.shli(byte_i32, arith.constant(23, type=i32))
                    return arith.bitcast(f32, scale_bits)

                # ---- B Load (dwordx4) + Scale for MXFP4 ----
                def _get_scale_f32(base_k, ku, ni, scale_cache):
                    """CK addressing for scale: adj_ku = base_k//32 + (ku//4)*4 + lane_div_16."""
                    _k0_blk = ku // 4
                    adj_ku = (base_k // arith.index(32)
                              + arith.index(_k0_blk * 4)
                              + lane_div_16)
                    scale_klane_rt = lane_div_16
                    k_pack_sub_rt = (adj_ku // arith.index(4)) % arith.index(2)
                    s_ku = adj_ku // arith.index(8)

                    cache_key = (_k0_blk, ni)
                    if cache_key not in scale_cache:
                        scale_cache[cache_key] = _load_scale_i32(
                            s_ku, ni, scale_klane=scale_klane_rt
                        )
                    packed = scale_cache[cache_key]
                    n_pack_sub_val = scale_n_pack_list[ni]
                    byte_pos_even = k_pack_sub_rt * arith.index(2)
                    byte_pos_odd = byte_pos_even + arith.index(1)
                    scale_even = _extract_e8m0_f32_dynamic(packed, byte_pos_even)
                    scale_odd = _extract_e8m0_f32_dynamic(packed, byte_pos_odd)
                    n_pack_is_zero = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.index_cast(i32, n_pack_sub_val),
                        arith.constant(0, type=i32),
                    )
                    return arith.select(n_pack_is_zero, scale_even, scale_odd)

                _k_per_dwordx4 = 128
                _k0_count = tile_k // _k_per_dwordx4

                def load_b_raw(base_k, k0_limit=_k0_count):
                    """Load raw FP4 data via dwordx4. Returns raw_v4[k0_idx][ni]."""
                    raw_all = []
                    for k0_idx in range_constexpr(k0_limit):
                        raw_k0 = []
                        k_off = base_k + arith.index(k0_idx * _k_per_dwordx4)
                        for ni in range_constexpr(num_acc_n):
                            v4 = load_b_raw_mxfp4_dwordx4(
                                buffer_ops, arith, vector,
                                arg_b=arg_w,
                                b_rsrc=w_rsrc,
                                layout_b=layout_b,
                                base_k=k_off,
                                n_blk=n_blk_list[ni],
                                n_intra=n_intra_list[ni],
                                lane_div_16=lane_div_16,
                                elem_type=w_elem,
                                kpack_bytes=kpack_bytes,
                                cache_modifier=2,
                            )
                            raw_k0.append(v4)
                        raw_all.append(raw_k0)
                    return raw_all

                def load_b_scale_raw(base_k, k0_limit=_k0_count):
                    """Issue scale buffer_loads only (no extraction).
                    Returns (packed_dict, kps_dict):
                      packed_dict: {(k0_blk, ni): packed_i32}
                      kps_dict: {k0_blk: k_pack_sub_rt}
                    """
                    packed_dict = {}
                    kps_dict = {}
                    for k0_blk in range_constexpr(k0_limit):
                        adj_ku = (base_k // arith.index(32)
                                  + arith.index(k0_blk * 4)
                                  + lane_div_16)
                        scale_klane_rt = lane_div_16
                        kps_dict[k0_blk] = (adj_ku // arith.index(4)) % arith.index(2)
                        s_ku = adj_ku // arith.index(8)
                        for ni in range_constexpr(num_acc_n):
                            packed_dict[(k0_blk, ni)] = _load_scale_i32(
                                s_ku, ni, scale_klane=scale_klane_rt
                            )
                    return packed_dict, kps_dict

                def extract_b_scales(packed_dict, kps_dict, ku_limit=k_unroll):
                    """Extract f32 scales from pre-loaded packed i32.
                    Returns scales[ku][ni] = f32.
                    """
                    scales = []
                    for ku in range_constexpr(ku_limit):
                        scales_ku = []
                        _k0_blk = ku // 4
                        k_pack_sub_rt = kps_dict[_k0_blk]
                        for ni in range_constexpr(num_acc_n):
                            packed = packed_dict[(_k0_blk, ni)]
                            n_pack_sub_val = scale_n_pack_list[ni]
                            byte_pos_even = k_pack_sub_rt * arith.index(2)
                            byte_pos_odd = byte_pos_even + arith.index(1)
                            scale_even = _extract_e8m0_f32_dynamic(packed, byte_pos_even)
                            scale_odd = _extract_e8m0_f32_dynamic(packed, byte_pos_odd)
                            n_pack_is_zero = arith.cmpi(
                                arith.CmpIPredicate.eq,
                                arith.index_cast(i32, n_pack_sub_val),
                                arith.constant(0, type=i32),
                            )
                            sf = arith.select(n_pack_is_zero, scale_even, scale_odd)
                            scales_ku.append(sf)
                        scales.append(scales_ku)
                    return scales

                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_buffer):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = col_base_swz_bytes // arith.index(int(elem_bytes))
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                def _a_col_bytes_for_ku(ku_val):
                    """CK-style A col address: L*64 + (ku%4)*16 + (ku//4)*256."""
                    _k0_blk = ku_val // 4
                    _ku_in = ku_val % 4
                    return col_offset_base_bytes + arith.index(
                        _ku_in * _a_ku_stride_bytes + _k0_blk * 256
                    )

                _total_a_slots = k_unroll * m_repeat

                def preload_a_from_lds(lds_buffer, ku_limit=k_unroll):
                    """Load all A tiles for ku_limit × m_repeat from LDS into VGPRs."""
                    a_tiles = [None] * (ku_limit * m_repeat)
                    for ku in range_constexpr(ku_limit):
                        for mi in range_constexpr(m_repeat):
                            col = _a_col_bytes_for_ku(ku)
                            row = row_a_lds + arith.index(mi * 16)
                            a_tiles[ku * m_repeat + mi] = lds_load_packs_k64(
                                row, col, lds_buffer
                            )
                    return a_tiles

                def _mfma_k32(acc_in, a0, a1, b0, b1):
                    a_v2 = vector.from_elements(vec2_i64, [a0, a1])
                    a_v8 = vector.bitcast(vec8_bf16, a_v2)
                    b_v2 = vector.from_elements(vec2_i64, [b0, b1])
                    b_v8 = vector.bitcast(vec8_bf16, b_v2)
                    return mfma_f32_bf16_k32(vec4_f32, [a_v8, b_v8, acc_in, 0, 0, 0])

                def compute_tile(
                    acc_in, b_v4, b_scales, a_tiles_cur,
                    *, ku_count=k_unroll, prefetch_epilogue: bool = False,
                ):
                    """Compute GEMM tile with preloaded A (pure compute, no ds_read).

                    Returns: (acc_list, epilogue_pf).
                    """
                    acc_list = list(acc_in)

                    epilogue_pf = None
                    if prefetch_epilogue:
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [arith.index(ii) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.index(mi * 16)
                                for ii in range_constexpr(4):
                                    row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc, sorted_row_pf,
                                            vec_width=1, dtype=f32,
                                        )
                                    )
                        epilogue_pf = (None, tw_pf)

                    for ni in range_constexpr(num_acc_n):
                        for ku in range_constexpr(ku_count):
                            _k0_idx = ku // 4
                            _ku_in_k0 = ku % 4

                            b_raw_ku = vector.extract(
                                b_v4[_k0_idx][ni],
                                static_position=[_ku_in_k0],
                                dynamic_position=[],
                            )
                            bb0, bb1 = unpack_b_mxfp4_bf16(
                                b_raw_ku, arith, vector,
                                scale_f32=b_scales[ku][ni],
                            )

                            for mi in range_constexpr(m_repeat):
                                _flat = ku * m_repeat + mi
                                a0, a1 = a_tiles_cur[_flat]

                                acc_idx = mi * num_acc_n + ni
                                acc_list[acc_idx] = _mfma_k32(
                                    acc_list[acc_idx], a0, a1, bb0, bb1,
                                )

                    return acc_list, epilogue_pf

                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    """CK-style scheduler: interleave MFMA, DS_READ, VMEM_READ."""
                    _dsread_per_wg = 1
                    _mfma_per_wg = 1
                    _NIterPerWarp = num_acc_n
                    _mfma_perM_perK = _NIterPerWarp * _mfma_per_wg

                    _HalfMIter = (m_repeat + 1) // 2

                    _Aload_num_perK = _dsread_per_wg * m_repeat
                    _Aload_rep = max((_Aload_num_perK + m_repeat - 1) // m_repeat, 1)
                    _Bload_num_perK = num_acc_n
                    _Bload_rep = max((_Bload_num_perK + _HalfMIter - 1) // _HalfMIter, 1)

                    for _ku in range_constexpr(k_unroll):
                        for _mi in range_constexpr(m_repeat):
                            _dsread_perM = _dsread_per_wg
                            _load_perM = 0

                            if _mi < _HalfMIter:
                                _load_perM = (
                                    (_Aload_rep if (_Aload_num_perK - (m_repeat - 1 - _mi) * _Aload_rep) > 0 else 0)
                                    + (_Bload_rep if (_Bload_num_perK - (_HalfMIter - 1 - _mi) * _Bload_rep) > 0 else 0)
                                )
                            else:
                                _load_perM = (
                                    _Aload_rep if (_Aload_num_perK - (m_repeat - 1 - _mi) * _Aload_rep) > 0 else 0
                                )

                            _sum_data = _dsread_perM + _load_perM
                            _round_data = max((_sum_data + _mfma_perM_perK - 1) // _mfma_perM_perK, 1)

                            _inst_order = []
                            _max_data = max(_load_perM, _dsread_perM)
                            for _j in range_constexpr(_max_data):
                                if _load_perM > _j:
                                    _inst_order.append(2)
                                if _dsread_perM > _j:
                                    _inst_order.append(3)
                            _pad_len = _mfma_perM_perK * _round_data - len(_inst_order)
                            _inst_order.extend([0] * _pad_len)

                            for _nj in range_constexpr(_mfma_perM_perK):
                                if _nj == 0:
                                    _inst_idx = 0
                                elif _nj == 1:
                                    _inst_idx = _mfma_perM_perK - 2 if _mfma_perM_perK > 2 else 1
                                elif _nj == 2:
                                    _inst_idx = _mfma_perM_perK - 1
                                else:
                                    _inst_idx = _mfma_perM_perK - _nj

                                rocdl.sched_mfma(1)

                                for _r in range_constexpr(_round_data):
                                    if _r % 2 == 0:
                                        _oi = _inst_idx + _r * _mfma_perM_perK
                                    else:
                                        _oi = (_r + 1) * _mfma_perM_perK - 1 - _inst_idx
                                    if _oi < len(_inst_order):
                                        if _inst_order[_oi] == 2:
                                            rocdl.sched_vmem(1)
                                        elif _inst_order[_oi] == 3:
                                            rocdl.sched_dsrd(1)

                    if _Aload_num_perK == 0:
                        rocdl.sched_vmem(1)
                    rocdl.sched_barrier(0)

                # ---- K-batch offset ----
                if _k_batch > 1:
                    bz = gpu.block_id("z")
                    k_base = bz * arith.index(_k_dim)
                else:
                    k_base = arith.index(0)

                # ---- CK-style pipeline: HEAD (scale prefetch) ----
                k0 = k_base
                prefetch_x_to_lds(k0, lds_x_pong)
                rocdl.sched_barrier(0)

                sc_raw_cur, kps_cur = load_b_scale_raw(k0)
                b_v4_cur = load_b_raw(k0)
                rocdl.sched_barrier(0)

                _k1 = k_base + arith.index(tile_k)
                prefetch_x_to_lds(_k1, lds_x_ping)
                rocdl.sched_barrier(0)

                acc = [acc_init] * (num_acc_n * m_repeat)

                rocdl.s_waitcnt(0)
                gpu.barrier()
                rocdl.sched_barrier(0)
                a_cur = preload_a_from_lds(lds_x_pong)
                b_sc_cur = extract_b_scales(sc_raw_cur, kps_cur)
                gpu.barrier()
                rocdl.sched_barrier(0)

                total_tiles = int(_k_dim) // int(tile_k)
                pair_iters = max((total_tiles - 2) // 2, 0)

                for pair_i in range_constexpr(pair_iters):
                    k_iv = k_base + arith.index(pair_i * (tile_k * 2))

                    # ---- Half 2i: scale prefetch -> B_raw -> compute -> extract -> barrier ----
                    rocdl.sched_barrier(0)
                    _k_a2 = k_iv + arith.index(tile_k * 2)
                    prefetch_x_to_lds(_k_a2, lds_x_pong)
                    rocdl.sched_barrier(0)
                    _k_b1 = k_iv + arith.index(tile_k)
                    sc_raw_nxt, kps_nxt = load_b_scale_raw(_k_b1)
                    rocdl.sched_barrier(0)

                    b_v4_nxt = load_b_raw(_k_b1)

                    rocdl.sched_barrier(0)
                    acc, _ = compute_tile(
                        acc, b_v4_cur, b_sc_cur, a_cur,
                    )
                    a_next = preload_a_from_lds(lds_x_ping)
                    rocdl.sched_barrier(0)
                    b_sc_nxt = extract_b_scales(sc_raw_nxt, kps_nxt)

                    rocdl.sched_barrier(0)
                    _barrier(lgkmcnt=2)
                    rocdl.sched_barrier(0)
                    a_cur = a_next

                    # ---- Half 2i+1: scale prefetch -> B_raw -> compute -> extract -> barrier ----
                    _k_a3 = k_iv + arith.index(tile_k * 3)
                    prefetch_x_to_lds(_k_a3, lds_x_ping)
                    rocdl.sched_barrier(0)

                    _k_b2 = k_iv + arith.index(tile_k * 2)
                    sc_raw_cur2, kps_cur2 = load_b_scale_raw(_k_b2)
                    b_v4_cur2 = load_b_raw(_k_b2)

                    rocdl.sched_barrier(0)
                    acc, _ = compute_tile(
                        acc, b_v4_nxt, b_sc_nxt, a_cur,
                    )
                    a_next = preload_a_from_lds(lds_x_pong)
                    b_sc_cur2 = extract_b_scales(sc_raw_cur2, kps_cur2)

                    rocdl.sched_barrier(0)
                    _barrier(lgkmcnt=2)
                    rocdl.sched_barrier(0)
                    b_v4_cur, b_sc_cur = b_v4_cur2, b_sc_cur2
                    a_cur = a_next

                # ---- TAIL: last 2 tiles (scale prefetch) ----
                k_tail1 = k_base + arith.index(_k_dim) - arith.index(tile_k)
                if _pad_ku_skip > 0:
                    sc_raw_tail, kps_tail = load_b_scale_raw(k_tail1, k0_limit=_tail_k0_count)
                    b_v4_tail = load_b_raw(k_tail1, k0_limit=_tail_k0_count)
                else:
                    sc_raw_tail, kps_tail = load_b_scale_raw(k_tail1)
                    b_v4_tail = load_b_raw(k_tail1)

                acc, _ = compute_tile(
                    acc, b_v4_cur, b_sc_cur, a_cur,
                )
                if _pad_ku_skip > 0:
                    a_next = preload_a_from_lds(lds_x_ping, ku_limit=_tail_ku)
                    b_sc_tail = extract_b_scales(sc_raw_tail, kps_tail, ku_limit=_tail_ku)
                else:
                    a_next = preload_a_from_lds(lds_x_ping)
                    b_sc_tail = extract_b_scales(sc_raw_tail, kps_tail)

                hot_loop_scheduler()
                rocdl.s_waitcnt(0)
                a_cur = a_next

                acc, epilogue_pf = compute_tile(
                    acc, b_v4_tail, b_sc_tail, a_cur,
                    ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                    prefetch_epilogue=True,
                )

                # ---- Bias: add to raw accumulators ----
                if enable_bias:
                    _bias_vals = []
                    for _ni in range_constexpr(num_acc_n):
                        _bn = by_n + n_tile_base + arith.index(_ni * 16) + lane_mod_16
                        _bias_vals.append(
                            buffer_ops.buffer_load(
                                bias_rsrc, expert_off_idx + _bn,
                                vec_width=1, dtype=f32
                            )
                        )
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = vector.splat(vec4_f32, _bias_vals[_ni])
                            acc[_aidx] = arith.addf(acc[_aidx], _bsplat)

                # ---- Epilogue ----
                expert_off = expert_off_idx
                mask24_i32 = arith.constant(0xFFFFFF, type=T.i32)
                model_i32 = arith.constant(model_dim, type=T.i32)
                topk_i32_v = topk_i32

                zero_i32 = arith.constant(0, type=T.i32)
                c2_i32 = arith.constant(2, type=T.i32)
                mask_even_i32 = arith.constant(0xFFFFFFFE, type=T.i32)
                e_vec = _e_vec

                sw_pf = None
                tw_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf = epilogue_pf

                # No per-channel weight scale for MXFP4 (scales already applied in dequant).
                sw_vals = [arith.constant(1.0, type=T.f32)] * num_acc_n

                if out_is_f32:
                    c4_i32 = arith.constant(4, type=T.i32)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32, out_rsrc, byte_off_i32, zero_i32, zero_i32,
                        )

                    def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                        _fv1 = vector.load_op(vec1_i32, lds_sorted_cache, [row_in_tile])
                        fused2 = vector.extract(_fv1, static_position=[0], dynamic_position=[])
                        t2 = fused2 & mask24_i32
                        s2 = arith.shrui(fused2, arith.constant(24, type=T.i32))
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_ok = t_ok & s_ok
                        t2_safe = arith.select(ts_ok, t2, arith.constant(0, type=T.i32))
                        s2_safe = arith.select(ts_ok, s2, arith.constant(0, type=T.i32))
                        sx = arith.select(
                            ts_ok,
                            arith.constant(1.0, type=T.f32),
                            arith.constant(0.0, type=T.f32),
                        )
                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = arith.select(
                                    ts_ok, tw_pf[tw_idx],
                                    arith.constant(0.0, type=T.f32),
                                )
                            else:
                                tw = arith.select(
                                    ts_ok,
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc, row, vec_width=1, dtype=f32
                                    ),
                                    arith.constant(0.0, type=T.f32),
                                )
                        idx0 = t2_safe * model_i32

                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            v = v * sx
                            if doweight_stage2:
                                v = v * tw
                            col_i32 = arith.index_cast(i32, col_g)
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
                        raise RuntimeError("CShuffle epilogue requires lds_out.")

                    out_base_idx = None
                    if out_is_bf16:
                        _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                        _out_base_ptr = _fly.extract_aligned_pointer_as_index(
                            _llvm_ptr_ty, arg_out
                        )
                        out_base_idx = arith.index_cast(
                            T.index, llvm.ptrtoint(T.i64, _out_base_ptr)
                        )

                    def write_row_to_lds(
                        *, mi: int, ii: int, row_in_tile, row,
                        row_base_lds, col_base_local, num_acc_n: int, lds_out,
                    ):
                        _fv1 = vector.load_op(vec1_i32, lds_sorted_cache, [row_in_tile])
                        fused2 = vector.extract(_fv1, static_position=[0], dynamic_position=[])
                        row_i32 = arith.index_cast(i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, num_valid_i32
                        )
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = row_valid0 & t_ok & s_ok
                        sx = arith.constant(1.0, type=T.f32)

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=f32
                                )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            v = v * sx
                            if doweight_stage2:
                                v = v * tw
                            v_out = arith.trunc_f(out_elem(), v)
                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        _fv1 = vector.load_op(vec1_i32, lds_sorted_cache, [row_local])
                        fused2 = vector.extract(_fv1, static_position=[0], dynamic_position=[])
                        row_i32 = arith.index_cast(i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, num_valid_i32
                        )
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = row_valid0 & t_ok & s_ok
                        return (fused2, row_valid)

                    def atomic_add_f16x2(val_f16x2, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f16x2, out_rsrc, byte_off_i32, zero_i32, zero_i32,
                        )

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused = row_ctx
                        t = fused & mask24_i32
                        s = fused >> 24
                        idx0 = t * model_i32
                        if not bool(accumulate):
                            ts = t * topk_i32_v + s
                            idx0 = ts * model_i32
                        col_i32 = arith.index_cast(i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if out_is_bf16:
                            if bool(accumulate):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(
                                    ptr_addr_idx, address_space=1
                                )
                                out_ptr_v = (
                                    out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                                )
                                frag_v = frag._value if hasattr(frag, "_value") else frag
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd, out_ptr_v, frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent", alignment=4,
                                )
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)
                        else:
                            byte_off = idx_elem_even * c2_i32
                            if bool(accumulate):
                                atomic_add_f16x2(frag, byte_off)
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)

                    c_shuffle_epilog(
                        arith=arith, vector=vector, gpu=gpu, scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m, tile_n=tile_n, e_vec=e_vec,
                        m_repeat=m_repeat, num_acc_n=num_acc_n,
                        tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16,
                        bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(T.bf16 if out_is_bf16 else T.f16),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                _moe_gemm2_then_body()

    _cache_tag = (
        module_name, out_dtype, tile_m, tile_n, tile_k,
        doweight_stage2, accumulate, use_cshuffle_epilog,
        enable_bias, model_dim_pad, inter_dim_pad,
        waves_per_eu, _k_batch,
    )

    @flyc.jit
    def launch_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        n_in = arith.ArithValue(arith.index_cast(T.index, i32_n_in.ir_value()))
        model_dim_pad_idx = arith.index(model_dim_pad)
        tile_n_index = arith.index(tile_n)
        size_expert_ids_in = arith.index_cast(
            T.index, i32_size_expert_ids_in.ir_value()
        )
        gx = (n_in - model_dim_pad_idx + tile_n_index - arith.index(1)) // tile_n_index
        gy = size_expert_ids_in

        moe_gemm2(
            arg_out, arg_x, arg_w, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, arg_bias,
            i32_tokens_in, i32_n_in, i32_k_in, i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, _k_batch),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm2
