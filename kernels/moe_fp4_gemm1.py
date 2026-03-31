# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE BF16×MXFP4 GEMM stage1 kernel with SwiGLU/SiLU activation (FlyDSL).

Implements the equivalent of the CK-tile MoeFlatmmKernel with:
  - A dtype: BF16 with per-token scale (FlatmmScalePointer<-1,0>)
  - B dtype: Packed MXFP4 E2M1 with per-1x32 scale (FlatmmScalePointer<1,32>)
  - Accumulator: FP32
  - Output: BF16 (or FP16)
  - Activation: SwiGLU or SiLU (gate × sigmoid(gate) × up)
  - Architecture: gfx950 only (requires mfma_scale_f32_16x16x128_f8f6f4)

Based on preshuffle_gemm.py (FP4 MFMA + scale loading + LDS ping-pong pipeline)
and moe_gemm_2stage.py (MoE routing, expert dispatch, gate+up split, epilogue).
"""

import functools
import os
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl
from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr.typing import T

from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    tile_chunk_coord_i32,
    swizzle_xor16,
    crd2idx,
)
from kernels.mfma_epilogues import mfma_epilog


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@functools.lru_cache(maxsize=1024)
def compile_moe_fp4_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 128,
    tile_k: int = 256,
    out_dtype: str = "bf16",
    activation: str = "swiglu",
    doweight_stage1: bool = False,
    waves_per_eu: int | None = None,
    use_cshuffle_epilog: bool | None = None,
):
    """Compile MoE BF16×MXFP4 gemm1 and return a JitFunction.

    Args:
        model_dim: Hidden dimension K (must be divisible by 256 for MXFP4).
        inter_dim: Intermediate dimension (gate and up each have inter_dim columns).
        experts: Number of experts.
        topk: Top-K routing factor.
        tile_m: M-dimension tile size (16, 32, or 64).
        tile_n: N-dimension tile size (128 or 256).
        tile_k: K-dimension tile size (128 or 256; must be divisible by 128 for MXFP4).
        out_dtype: Output dtype, "bf16" or "f16".
        activation: Gate activation, "swiglu" or "silu".
        doweight_stage1: Apply sorted routing weights in epilogue.
        waves_per_eu: Occupancy hint (None = default).
        use_cshuffle_epilog: Use CShuffle LDS epilogue (None = env var FLYDSL_MOE_STAGE1_CSHUFFLE).

    Returns:
        A @flyc.jit function that can be called as:
            launch_fn(out, x, w, scale_x, scale_w,
                      sorted_token_ids, expert_ids, sorted_weights, max_token_ids,
                      tokens, inter, k, size_expert_ids, stream)
    """
    # mfma_scale_f32_16x16x128_f8f6f4 (scaled MXFP4 MFMA) is available on CDNA4+.
    # Extend this tuple when future architectures gain the same instruction.
    _MXFP4_MFMA_ARCHS = ("gfx950",)
    gpu_arch = get_hip_arch()
    if not any(str(gpu_arch).startswith(a) for a in _MXFP4_MFMA_ARCHS):
        raise RuntimeError(
            f"MoE FP4 GEMM requires an architecture with scaled MXFP4 MFMA support "
            f"({', '.join(_MXFP4_MFMA_ARCHS)}), got {gpu_arch}"
        )
    if activation not in ("swiglu", "silu"):
        raise ValueError(f"activation must be 'swiglu' or 'silu', got {activation!r}")
    if out_dtype not in ("bf16", "f16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'f16', got {out_dtype!r}")
    if tile_k % 128 != 0:
        raise ValueError(f"tile_k must be divisible by 128 for MXFP4, got tile_k={tile_k}")
    if model_dim < tile_k * 2:
        raise ValueError(f"model_dim ({model_dim}) must be >= 2*tile_k ({tile_k * 2}) for ping-pong pipeline")

    # MXFP4: 2 elements per byte (E2M1 packed)
    fp4_pack = 2
    elem_bytes = 1  # storage bytes per packed pair = 1 byte = 2 fp4 values

    tile_k_bytes = tile_k * elem_bytes // fp4_pack  # bytes of A in LDS per tile (BF16: 2 bytes/elem)
    # A is BF16: 2 bytes per element, tile_k BF16 elements per row
    a_elem_bytes = 2  # BF16
    a_tile_k_bytes = tile_k * a_elem_bytes
    if a_tile_k_bytes % 64 != 0:
        raise ValueError(f"tile_k * a_elem_bytes ({a_tile_k_bytes}) must be divisible by 64")

    _out_is_bf16 = out_dtype == "bf16"

    # MXFP4 quantization parameters
    _quant_block_k = 32  # 1 scale per 32 elements in K (MXFP4 standard)
    _fp4_pack_M = 2
    _fp4_pack_N = 2
    _fp4_pack_K = 2
    _K1_outer = model_dim // (_quant_block_k * 4 * _fp4_pack_K)  # outer K loop count for scales
    _fp4_scale_chunk_k = _quant_block_k * 4 * _fp4_pack_K  # 256: elements per scale chunk
    _stride_k0_elems = 64  # elements per K0 block

    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLYDSL_MOE_STAGE1_CSHUFFLE", "1") in ("1", "true", "True", "YES", "yes")
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    if out_dtype != "f16" and use_cshuffle_epilog:
        raise ValueError("CShuffle epilogue currently supports only f16 output; set out_dtype='f16'")

    _use_cshuffle_epilog = use_cshuffle_epilog

    # BF16 A tile: 2 bytes/elem
    total_threads = 256
    bytes_a_per_tile = tile_m * tile_k * a_elem_bytes
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(f"tile_m * tile_k * 2 ({bytes_a_per_tile}) must be divisible by {total_threads}")
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}")
    num_a_loads = bytes_per_thread_a // a_load_bytes

    # LDS: A tiles in ping-pong (BF16, not packed). lds_stride = tile_k (XOR16 swizzle, no padding)
    lds_stride = tile_k  # BF16 elements
    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    if not _ck_lds128:
        lds_stride = tile_k + 8

    lds_a_bytes = 2 * tile_m * lds_stride * a_elem_bytes
    lds_out_bytes = 2 * tile_m * tile_n if _use_cshuffle_epilog else 0
    lds_total_bytes = max(lds_a_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes // a_elem_bytes  # in BF16 elements

    allocator = SmemAllocator(None, arch=gpu_arch)
    lds_alloc_bytes = lds_total_elems * a_elem_bytes
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    # Module name (compile cache key)
    _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu is not None else ""
    _epi_tag = "cshuffle" if _use_cshuffle_epilog else "direct"
    module_name = (
        f"moe_fp4_gemm1_{activation}_{out_dtype}_{_epi_tag}_t{tile_m}x{tile_n}x{tile_k}{_wpe_tag}_abi1"
    ).replace("-", "_")

    # ── Kernel function ────────────────────────────────────────────────────────
    @flyc.kernel(name=module_name)
    def moe_fp4_gemm1_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
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
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)

        # ── Activation function ────────────────────────────────────────────
        def _swish_gate(g):
            """swish(g) = g * sigmoid(g); identical to silu(g)."""
            t = g * (-1.4426950408889634)  # multiply by -log2(e)
            emu = rocdl.exp2(T.f32, t)
            sig = rocdl.rcp(T.f32, 1.0 + emu)
            return g * sig

        def gate_activation(g, u):
            """Apply gate activation and multiply by up: swiglu(g,u) = swish(g) * u."""
            return _swish_gate(g) * u

        # ── Output type ────────────────────────────────────────────────────
        out_elem = T.bf16 if _out_is_bf16 else T.f16
        acc_init = arith.constant_vector(0.0, T.f32x4)

        # ── B preshuffle layout (MXFP4, col-major B in CK-tile → preshuffle here) ──
        # B is [experts * 2 * inter_dim, model_dim] in preshuffle format.
        # FP4: b_elem_vec_pack=2, kpack_bytes=16 (same as fp4 in preshuffle_gemm)
        b_elem_vec_pack = 2
        kpack_bytes = 16
        w_elem_bytes = 1  # 1 byte = 2 fp4 values
        c_n_total = arith.index(experts * 2 * inter_dim)
        b_layout_obj = make_preshuffle_b_layout(
            arith, c_n=c_n_total, c_k=k_in, kpack_bytes=kpack_bytes, elem_bytes=w_elem_bytes
        )
        layout_b = b_layout_obj.layout_b

        # B load strides (pre-compute in index units for buffer_load address math)
        k_bytes_b = model_dim * w_elem_bytes // b_elem_vec_pack  # packed K bytes
        n0_val = experts * 2 * inter_dim // 16
        k0_val = k_bytes_b // 64
        _stride_nlane = kpack_bytes
        _stride_klane = 16 * _stride_nlane
        _stride_k0 = 4 * _stride_klane
        _stride_n0 = k0_val * _stride_k0
        _b_dword_stride_n0 = _stride_n0 // 4
        _b_dword_stride_k0 = _stride_k0 // 4
        _b_dword_stride_klane = _stride_klane // 4
        _b_dword_stride_nlane = _stride_nlane // 4

        # A layout (BF16, row-major)
        k_div4_factor = model_dim * a_elem_bytes // 4  # dwords per row
        layout_a_tile_div4 = fx.make_layout((tile_m, k_div4_factor), (k_div4_factor, 1))

        # LDS layout (BF16 elements)
        shape_lds = fx.make_shape(tile_m, tile_k)
        stride_lds = fx.make_stride(lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)
        k_blocks16_val = arith.index(a_tile_k_bytes // 16)

        # ── Thread/wave decomposition ──────────────────────────────────────
        tx = gpu.thread_id("x")
        # Block → N (x), M / expert-block (y): matches aiter MoE launch convention
        by = gpu.block_id("x")  # N tile index
        bx = gpu.block_id("y")  # M tile index (sorted expert block)

        bx_m = bx * fx.Index(tile_m)

        # Block validity guard (skip padding blocks beyond max_token_id)
        maxids_rsrc = buffer_ops.create_buffer_resource(
            arg_max_token_ids, max_size=False, num_records_bytes=fx.Index(4)
        )
        max_token_id_i32 = buffer_ops.buffer_load(maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32)

        _if_blk = scf.IfOp(blk_valid)
        with _if_then(_if_blk):
            # ── LDS allocation ─────────────────────────────────────────────
            base_ptr = allocator.get_base()
            lds_a_ptr = SmemPtr(base_ptr, lds_alloc_offset, T.bf16, shape=(lds_total_elems,))
            lds_a = lds_a_ptr.get()
            lds_out = (
                SmemPtr(base_ptr, lds_a_ptr.byte_offset, T.f16, shape=(tile_m * tile_n,)).get()
                if _use_cshuffle_epilog
                else None
            )

            # ── Buffer resources ────────────────────────────────────────────
            c_topk = fx.Index(topk)
            x_nbytes = tokens_in * k_in * fx.Index(a_elem_bytes)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=arith.index_cast(T.i64, x_nbytes)
            )
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            out_nbytes = tokens_in * c_topk * inter_in * fx.Index(2)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=arith.index_cast(T.i64, out_nbytes)
            )
            # scale_x: [num_k_blocks, tokens] f32, index = kb * tokens + token_id
            sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False)
            # scale_w: same layout as preshuffle_gemm fp4 scale_b
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids,
                max_size=False,
                num_records_bytes=arith.index_cast(T.i64, size_expert_ids_in * fx.Index(4)),
            )

            # ── Expert routing ──────────────────────────────────────────────
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            inter2_idx = arith.index(2 * inter_dim)
            expert_off_idx = expert_idx * inter2_idx

            # ── Thread/wave decomposition ────────────────────────────────────
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
            coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
            wave_id = fx.get(coord_wl, 0)
            lane_id = fx.get(coord_wl, 1)
            coord_l16 = fx.idx2crd(lane_id, layout_lane16)
            lane_div_16 = fx.get(coord_l16, 0)
            lane_mod_16 = fx.get(coord_l16, 1)

            # A LDS indexing: row = lane_mod_16, col_base = lane_div_16 * kpack_elems_bf16
            row_a_lds = lane_mod_16
            a_kpack_elems = 8  # 16 bytes / 2 (BF16) = 8 BF16 elements
            col_offset_base = lane_div_16 * arith.index(a_kpack_elems)
            col_offset_base_bytes = col_offset_base * arith.index(a_elem_bytes)

            # ── Token decode (pre-compute row base for A loads) ─────────────
            mask24 = fx.Int32(0xFFFFFF)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)
            topk_i32 = fx.Int32(topk)
            inter_i32_local = fx.Int32(inter_dim)

            # Compute k_div4 for load indexing
            c_k_div4 = (k_in * fx.Index(a_elem_bytes)) // fx.Index(4)
            tile_k_dwords = (tile_k * a_elem_bytes) // 4
            layout_a_tile_div4_inner = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
            tx_i32_base = tx * fx.Index(4)  # 4 dwords = 16 bytes per chunk

            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_a_loads):
                row_local, col_local_i32 = tile_chunk_coord_i32(
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_a_tile_div4_inner,
                    chunk_i32=4,
                )
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                t_raw = fused_i & mask24
                t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                t_idx = arith.index_cast(T.index, t_raw)
                t_safe = t_valid_i32.select(t_idx, fx.Index(0))
                x_row_base_div4.append(t_safe * c_k_div4)

            # ── N tile decomposition (gate and up) ───────────────────────────
            by_n = by * fx.Index(tile_n)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            n_tile_base = wave_id * fx.Index(n_per_wave)

            inter_idx = fx.Index(inter_dim)
            c_n0_static = experts * 2 * inter_dim // 16
            layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))

            n_intra_gate = []
            n_blk_gate = []
            n_intra_up = []
            n_blk_up = []
            col_g_list = []
            for ni in range_constexpr(num_acc_n):
                col_g = by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16
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
            # FP4 K unroll: tile_k / (32 * 2) packs of 128 bits each
            # mfma_scale_f32_16x16x128 consumes 128 K-elements of FP4 (64 bytes packed) per call.
            # fp4_pack_K=2 → k_unroll_packed = tile_k // (128 * fp4_pack_K)
            k_unroll_packed = tile_k // (128 * _fp4_pack_K)  # number of packed K128 steps
            if k_unroll_packed < 1:
                k_unroll_packed = 1

            # ── FP4 B load ───────────────────────────────────────────────────
            # Pre-compute N-dimension dword offsets for B (same as preshuffle_gemm).
            _b_n_gate_dword = []
            _b_n_up_dword = []
            for _ni in range_constexpr(num_acc_n):
                _n_gate_dw = (
                    n_blk_gate[_ni] * fx.Index(_b_dword_stride_n0)
                    + n_intra_gate[_ni] * fx.Index(_b_dword_stride_nlane)
                    + lane_div_16 * fx.Index(_b_dword_stride_klane)
                )
                _b_n_gate_dword.append(_n_gate_dw)
                _n_up_dw = (
                    n_blk_up[_ni] * fx.Index(_b_dword_stride_n0)
                    + n_intra_up[_ni] * fx.Index(_b_dword_stride_nlane)
                    + lane_div_16 * fx.Index(_b_dword_stride_klane)
                )
                _b_n_up_dword.append(_n_up_dw)

            _b_dword_stride_k0_c = fx.Index(_b_dword_stride_k0)

            def _load_b_fp4_single(k_dword_offset, ni, n_dword_list):
                """Load one 16B FP4 B vector at given K dword offset."""
                dword_idx = n_dword_list[ni] + k_dword_offset
                dword_idx_i32 = arith.index_cast(T.i32, dword_idx)
                b_vec4 = buffer_ops.buffer_load(w_rsrc, dword_idx_i32, vec_width=4, dtype=T.i32)
                # 16B = 32 FP4 elements packed into i8x16
                b_i8x16 = vector.bitcast(T.vec(16, T.i8), b_vec4)
                b_i64x2 = vector.bitcast(T.i64x2, b_i8x16)
                b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                return b0, b1

            def _pack_i64x4_to_i32x8(x0, x1, x2, x3):
                v4 = vector.from_elements(T.vec(4, T.i64), [x0, x1, x2, x3])
                return vector.bitcast(T.vec(8, T.i32), v4)

            c0_i64 = arith.constant(0, type=T.i64)

            def load_b_fp4_tile(base_k_packed, n_dword_list):
                """Load FP4 B tile for k_unroll_packed steps.

                Returns list of (i32x8, i32x8) for each packed KU step.
                Each i32x8 packs 2 consecutive K128 blocks (256 FP4 elems) per MFMA.
                """
                tile = []
                for ku in range_constexpr(k_unroll_packed):
                    packs = []
                    for ni in range_constexpr(num_acc_n):
                        # Two consecutive K64-dword blocks make one K128 MFMA operand
                        k0_dw = (base_k_packed * fx.Index(2) + fx.Index(ku * 2)) * _b_dword_stride_k0_c
                        k1_dw = k0_dw + _b_dword_stride_k0_c
                        b00, b01 = _load_b_fp4_single(k0_dw, ni, n_dword_list)
                        b10, b11 = _load_b_fp4_single(k1_dw, ni, n_dword_list)
                        b128 = _pack_i64x4_to_i32x8(b00, b01, b10, b11)
                        packs.append(b128)
                    tile.append(packs)
                return tile

            # ── FP4 scale loading ────────────────────────────────────────────
            # A scale (per-token, per-K-block): scale_x[kb, token_id] = scale_x[kb * tokens + token_id]
            # B scale (per-1x32 block): same layout as preshuffle_gemm fp4 scale_b
            #   scale_w[n_block * _K1_outer * 64 + ku_elem]

            _fp4_use_scheduler = tile_m >= 64

            # Scale layout constants
            _m_repeat_packed = m_repeat // _fp4_pack_M
            _n_packed = num_acc_n // _fp4_pack_N
            _scale_lane_elem_off = lane_div_16 * fx.Index(16) + lane_mod_16
            _scale_row_stride_elems = _K1_outer * 64

            # B gate and up scale base elements (pair of 16 N-cols)
            _scale_b_gate_base = []
            _scale_b_up_base = []
            for ni in range_constexpr(_n_packed):
                gate_n_base = by_n + n_tile_base + arith.index(ni * _fp4_pack_N * 16)
                mni_gate = (expert_off_idx + gate_n_base) // arith.index(_fp4_pack_N * 16)
                _scale_b_gate_base.append(mni_gate * arith.index(_scale_row_stride_elems) + _scale_lane_elem_off)
                up_n_base = gate_n_base + inter_idx
                mni_up = (expert_off_idx + up_n_base) // arith.index(_fp4_pack_N * 16)
                _scale_b_up_base.append(mni_up * arith.index(_scale_row_stride_elems) + _scale_lane_elem_off)

            def load_fp4_scales(base_k_scale_idx):
                """Load B (gate + up) FP4 scales for given K scale index.

                A (BF16) does not have per-1x32 scales; scale_x is per-token f32 loaded at epilogue.
                B scales use the preshuffle FP4 layout (one i32 per 32-element block pair).
                """
                b_gate_scales = []
                b_up_scales = []
                base_k_elem_off = base_k_scale_idx * fx.Index(_stride_k0_elems)
                for ku in range_constexpr(k_unroll_packed):
                    ku_elem_off = base_k_elem_off + fx.Index(ku * _stride_k0_elems)
                    for ni in range_constexpr(_n_packed):
                        b_gate_scales.append(
                            buffer_ops.buffer_load(
                                sw_rsrc, _scale_b_gate_base[ni] + ku_elem_off, vec_width=1, dtype=T.i32
                            )
                        )
                        b_up_scales.append(
                            buffer_ops.buffer_load(
                                sw_rsrc, _scale_b_up_base[ni] + ku_elem_off, vec_width=1, dtype=T.i32
                            )
                        )
                return b_gate_scales, b_up_scales

            def load_fp4_scale_chunk(base_k):
                return load_fp4_scales(base_k // fx.Index(_fp4_scale_chunk_k))

            # ── A load: BF16 global → regs ────────────────────────────────────
            vec8_bf16 = T.vec(8, T.bf16)  # 16 bytes

            def load_a_16(idx_elem):
                return buffer_copy_gmem16_dwordx4(
                    buffer_ops,
                    vector,
                    elem_type=T.bf16,
                    idx_i32=idx_elem,
                    rsrc=x_rsrc,
                    vec_elems=8,
                    elem_bytes=a_elem_bytes,
                )

            def load_a_tile(base_k_div4):
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local = x_row_local[i]
                    col_a_local_i32 = x_col_local_i32[i]
                    # base_k_div4 is in dword units for BF16 (2 bytes/elem → 2 dwords / 8 bytes)
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + col_a_local_i32
                    idx_elem = idx_i32 * fx.Index(2)  # dwords to BF16 element index
                    a_16b = load_a_16(idx_elem)
                    parts.append(vector.bitcast(T.i32x4, a_16b))
                return parts

            # ── A store: regs → LDS (BF16, XOR16 swizzle) ────────────────────
            k_blocks16_c = fx.Index(a_tile_k_bytes // 16)

            def store_a_tile_to_lds(vec_a_parts, lds_base_offset):
                for i in range_constexpr(num_a_loads):
                    row_a_local = x_row_local[i]
                    col_a_local_i32 = x_col_local_i32[i]
                    col_local_bytes = col_a_local_i32 * fx.Index(4)
                    lds_store_16b_xor16(
                        arith,
                        vector,
                        lds_memref=lds_a,
                        vec16_ty=vec8_bf16,
                        layout_lds=layout_lds,
                        row_local=row_a_local,
                        col_local_i32=col_a_local_i32,
                        tx_c4=fx.Index(4),
                        k_blocks16=k_blocks16_c,
                        lds_base=lds_base_offset,
                        vec_part_i32x4=vec_a_parts[i],
                        elem_bytes=a_elem_bytes,
                    )

            # ── A LDS → regs (for MFMA, 16B = 8 BF16 → 2×i64) ──────────────
            def lds_load_packs_k64(curr_row, col_base_bytes, lds_base_offset):
                col_base_swz_bytes = swizzle_xor16(curr_row, col_base_bytes, k_blocks16_c)
                col_base_swz = col_base_swz_bytes // arith.index(a_elem_bytes)
                idx_a16 = crd2idx((curr_row, col_base_swz), layout_lds) + lds_base_offset
                loaded = vector.load_op(vec8_bf16, lds_a, [idx_a16])
                a_i64x2 = vector.bitcast(T.i64x2, loaded)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0, a1

            # ── MFMA compute tile (FP4 scaled) ────────────────────────────────
            def compute_tile_fp4(
                acc_gate_in,
                acc_up_in,
                b_gate_tile,
                b_up_tile,
                lds_base_offset,
                fp4_scales,
                *,
                a0_prefetch=None,
            ):
                gate_list = list(acc_gate_in)
                up_list = list(acc_up_in)
                mfma_res_ty = T.f32x4
                b_gate_sc, b_up_sc = fp4_scales

                # FP4 MFMA: pack 4×i64 into i32x8 for a128/b128 operand
                # mfma_scale_f32_16x16x128_f8f6f4 with cbsz=4, blgp=4
                _fp4_cbsz = 4
                _fp4_blgp = 4
                # A is BF16 (unquantized): use E8M0=1.0 (0x7F7F7F7F) as fixed per-block A scale.
                # The per-token scale_x is applied post-MFMA in the epilogue.
                a_scale_fixed = arith.constant(0x7F7F7F7F, type=T.i32)

                for ku128 in range_constexpr(k_unroll_packed):
                    b_gate_scale_base = ku128 * _n_packed
                    b_up_scale_base = ku128 * _n_packed

                    for mi_p in range_constexpr(_m_repeat_packed):
                        for ni_p in range_constexpr(_n_packed):
                            b_gate_scale_val = b_gate_sc[b_gate_scale_base + ni_p]
                            b_up_scale_val = b_up_sc[b_up_scale_base + ni_p]
                            b_gate_128 = b_gate_tile[ku128][ni_p * _fp4_pack_N]
                            b_up_128 = b_up_tile[ku128][ni_p * _fp4_pack_N]

                            for imxdl in range_constexpr(_fp4_pack_M):
                                mi_idx = mi_p * _fp4_pack_M + imxdl
                                curr_row_a_lds = row_a_lds + arith.index(mi_idx * 16)
                                if (a0_prefetch is not None) and (ku128 == 0) and (mi_idx == 0):
                                    a0, a1 = a0_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_offset_base_bytes, lds_base_offset)
                                # Build a128: [a0, a1, 0, 0] as i32x8
                                a128 = _pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)

                                for inxdl in range_constexpr(_fp4_pack_N):
                                    ni_idx = ni_p * _fp4_pack_N + inxdl
                                    acc_idx = mi_idx * num_acc_n + ni_idx

                                    if not _fp4_use_scheduler:
                                        rocdl.sched_barrier(0)

                                    gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        mfma_res_ty,
                                        [
                                            a128,
                                            b_gate_128,
                                            gate_list[acc_idx],
                                            _fp4_cbsz,
                                            _fp4_blgp,
                                            imxdl,
                                            a_scale_fixed,
                                            inxdl,
                                            b_gate_scale_val,
                                        ],
                                    )
                                    up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        mfma_res_ty,
                                        [
                                            a128,
                                            b_up_128,
                                            up_list[acc_idx],
                                            _fp4_cbsz,
                                            _fp4_blgp,
                                            imxdl,
                                            a_scale_fixed,
                                            inxdl,
                                            b_up_scale_val,
                                        ],
                                    )

                return gate_list, up_list

            # ── Hot loop scheduler ────────────────────────────────────────────
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                if not _fp4_use_scheduler:
                    return
                mfma_group = _n_packed
                mfma_total = (
                    k_unroll_packed
                    * _m_repeat_packed
                    * _n_packed
                    * _fp4_pack_K
                    * _fp4_pack_M
                    * _fp4_pack_N
                    * 2  # gate + up
                )
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                rocdl.sched_mfma(1)
                dswr_tail = num_a_loads
                if dswr_tail > sche_iters:
                    dswr_tail = sche_iters
                dswr_start = max(sche_iters - dswr_tail, 0)
                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if sche_i >= dswr_start - 1:
                        rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)

            # ── Main 2-stage ping-pong loop ───────────────────────────────────
            # Indexing: each "chunk" = _fp4_scale_chunk_k = 256 K-elements.
            # model_dim must be >= 2*tile_k (enforced at compile time), so num_k_chunks >= 2.
            # chunk_i -> base BF16 dword = chunk_i * (_fp4_scale_chunk_k * a_elem_bytes) // 4
            #          = chunk_i * (256 * 2) // 4 = chunk_i * 128 dwords.
            num_k_chunks = model_dim // _fp4_scale_chunk_k
            _a_dwords_per_chunk = (_fp4_scale_chunk_k * a_elem_bytes) // 4  # 128 dwords

            def _a_base_dwords(chunk_i):
                return chunk_i * fx.Index(_a_dwords_per_chunk)

            lds_tile_elems = arith.index(tile_m * lds_stride)
            lds_base_cur = fx.Index(0)
            lds_base_nxt = lds_tile_elems

            # Prologue: load tile 0
            a_tile0 = load_a_tile(_a_base_dwords(fx.Index(0)))
            b_gate_cur = load_b_fp4_tile(fx.Index(0), _b_n_gate_dword)
            b_up_cur = load_b_fp4_tile(fx.Index(0), _b_n_up_dword)
            sc0 = load_fp4_scale_chunk(fx.Index(0))
            store_a_tile_to_lds(a_tile0, lds_base_cur)
            gpu.barrier()
            a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_cur)

            lds_base_pong = lds_base_cur
            lds_base_ping = lds_base_nxt

            acc_gate = [arith.constant_vector(0.0, T.f32x4)] * (num_acc_n * m_repeat)
            acc_up = [arith.constant_vector(0.0, T.f32x4)] * (num_acc_n * m_repeat)

            # Unrolled pair-iteration (2 K-chunks per iter), leaving 2 tail chunks.
            # pair_iters >= 0 always since num_k_chunks >= 2.
            pair_iters = max((num_k_chunks - 2) // 2, 0)

            for pair_i in range_constexpr(pair_iters):
                # pair_i covers chunks [pair_i*2, pair_i*2+1], tail covers last 2
                k_chunk1 = pair_i * 2 + 1
                k_chunk2 = pair_i * 2 + 2

                # ---- stage 0: prefetch ping (chunk k_chunk1), compute pong (chunk pair_i*2) ----
                a_tile_ping = load_a_tile(_a_base_dwords(fx.Index(k_chunk1)))
                b_gate_ping = load_b_fp4_tile(fx.Index(k_chunk1), _b_n_gate_dword)
                b_up_ping = load_b_fp4_tile(fx.Index(k_chunk1), _b_n_up_dword)
                sc_ping = load_fp4_scale_chunk(fx.Index(k_chunk1 * _fp4_scale_chunk_k))

                acc_gate, acc_up = compute_tile_fp4(
                    acc_gate,
                    acc_up,
                    b_gate_cur,
                    b_up_cur,
                    lds_base_pong,
                    sc0,
                    a0_prefetch=a0_prefetch_pong,
                )
                a0_prefetch_pong = None
                store_a_tile_to_lds(a_tile_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()
                a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                # ---- stage 1: prefetch pong (chunk k_chunk2), compute ping (chunk k_chunk1) ----
                a_tile_pong_new = load_a_tile(_a_base_dwords(fx.Index(k_chunk2)))
                b_gate_pong_new = load_b_fp4_tile(fx.Index(k_chunk2), _b_n_gate_dword)
                b_up_pong_new = load_b_fp4_tile(fx.Index(k_chunk2), _b_n_up_dword)
                sc_pong_new = load_fp4_scale_chunk(fx.Index(k_chunk2 * _fp4_scale_chunk_k))

                acc_gate, acc_up = compute_tile_fp4(
                    acc_gate,
                    acc_up,
                    b_gate_ping,
                    b_up_ping,
                    lds_base_ping,
                    sc_ping,
                    a0_prefetch=a0_prefetch_ping,
                )
                a0_prefetch_ping = None
                store_a_tile_to_lds(a_tile_pong_new, lds_base_pong)
                hot_loop_scheduler()
                gpu.barrier()
                a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

                b_gate_cur = b_gate_pong_new
                b_up_cur = b_up_pong_new
                sc0 = sc_pong_new

            # Tail: last 2 K chunks (chunk indices num_k_chunks-2 and num_k_chunks-1)
            _k_tail0 = num_k_chunks - 2
            _k_tail1 = num_k_chunks - 1
            # Reload tail0 B tiles (they may not match b_gate_cur when pair_iters==0)
            b_gate_cur = load_b_fp4_tile(fx.Index(_k_tail0), _b_n_gate_dword)
            b_up_cur = load_b_fp4_tile(fx.Index(_k_tail0), _b_n_up_dword)
            sc_tail0 = load_fp4_scale_chunk(fx.Index(_k_tail0 * _fp4_scale_chunk_k))
            a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

            a_tile_ping = load_a_tile(_a_base_dwords(fx.Index(_k_tail1)))
            b_gate_ping = load_b_fp4_tile(fx.Index(_k_tail1), _b_n_gate_dword)
            b_up_ping = load_b_fp4_tile(fx.Index(_k_tail1), _b_n_up_dword)
            sc_tail1 = load_fp4_scale_chunk(fx.Index(_k_tail1 * _fp4_scale_chunk_k))

            acc_gate, acc_up = compute_tile_fp4(
                acc_gate,
                acc_up,
                b_gate_cur,
                b_up_cur,
                lds_base_pong,
                sc_tail0,
                a0_prefetch=a0_prefetch_pong,
            )
            store_a_tile_to_lds(a_tile_ping, lds_base_ping)
            hot_loop_scheduler()
            gpu.barrier()
            a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

            acc_gate, acc_up = compute_tile_fp4(
                acc_gate,
                acc_up,
                b_gate_ping,
                b_up_ping,
                lds_base_ping,
                sc_tail1,
                a0_prefetch=a0_prefetch_ping,
            )

            # ── Epilogue ───────────────────────────────────────────────────────
            mask24_i32 = fx.Int32(0xFFFFFF)

            if _use_cshuffle_epilog:
                if lds_out is None:
                    raise RuntimeError("CShuffle epilogue enabled but lds_out is not allocated.")

                def write_row_to_lds(*, mi, ii, row_in_tile, row, row_base_lds, col_base_local, num_acc_n, lds_out):
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> fx.Int32(24)
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                    sx = arith.select(
                        t_valid,
                        buffer_ops.buffer_load(sx_rsrc, t2, vec_width=1, dtype=T.f32),
                        fx.Float32(0.0),
                    )
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32)

                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        vg = vector.extract(acc_gate[acc_idx], static_position=[ii], dynamic_position=[])
                        vu = vector.extract(acc_up[acc_idx], static_position=[ii], dynamic_position=[])
                        y = gate_activation(vg * sx, vu * sx)
                        if doweight_stage1:
                            y = y * tw
                        y16 = arith.trunc_f(T.f16, y)
                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(T.vec(1, T.f16), [y16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> fx.Int32(24)
                    inter_i32_local_v = fx.Int32(inter_dim)
                    return (t2 * topk_i32 + s2) * inter_i32_local_v

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                    t2 = fused2 & mask24_i32
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                    _if_valid = scf.IfOp(t_valid)
                    with _if_then(_if_valid):
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_out = row_ctx + col_i32
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
                    frag_elem_type=T.f16,
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )
            else:
                inter_i32_local_v = fx.Int32(inter_dim)
                col_i32_list = [arith.index_cast(T.i32, col_g_list[ni]) for ni in range_constexpr(num_acc_n)]

                def _store_row(*, mi, ii, row_in_tile, row):
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                    t2_raw = fused2 & mask24_i32
                    s2_raw = fused2 >> fx.Int32(24)
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2_raw, tokens_i32)
                    sx = arith.select(
                        t_valid,
                        buffer_ops.buffer_load(sx_rsrc, t2_raw, vec_width=1, dtype=T.f32),
                        fx.Float32(0.0),
                    )
                    idx0 = (t2_raw * topk_i32 + s2_raw) * inter_i32_local_v
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32)
                    _if_valid = scf.IfOp(t_valid)
                    with _if_then(_if_valid):
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(acc_gate[acc_idx], static_position=[ii], dynamic_position=[])
                            vu = vector.extract(acc_up[acc_idx], static_position=[ii], dynamic_position=[])
                            y = gate_activation(vg * sx, vu * sx)
                            if doweight_stage1:
                                y = y * tw
                            y_out = arith.trunc_f(out_elem, y)
                            idx_out = idx0 + col_i32_list[ni]
                            buffer_ops.buffer_store(y_out, out_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=False,
                    arith=arith,
                    range_constexpr=range_constexpr,
                    m_repeat=m_repeat,
                    lane_div_16=lane_div_16,
                    bx_m=bx_m,
                    body_row=_store_row,
                )

    # ── Host launcher ──────────────────────────────────────────────────────────
    @flyc.jit
    def launch_moe_fp4_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
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

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)

        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = inter_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        moe_fp4_gemm1_kernel(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_moe_fp4_gemm1


__all__ = ["compile_moe_fp4_gemm1"]
