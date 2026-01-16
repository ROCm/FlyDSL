"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import os

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator, SmemPtr

from _mlir import ir
import _mlir.extras.types as T
from flydsl.lang.ir.types import T as I

from flydsl.dialects.ext import arith, gpu, buffer_ops, llvm, vector, rocdl

from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
)
from kernels.mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog


def compile_moe_gemm1(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    sorted_size: int,
    size_expert_ids: int,
    doweight_stage1: bool,
    in_dtype: str = "fp8",
    use_cshuffle_epilog: bool | None = None,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype:
      - "fp8": X/W are fp8
      - "fp16": X/W are fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": X/W are int8
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if in_dtype not in ("fp8", "fp16", "int8", "int4"):
        raise ValueError(f"in_dtype must be one of ('fp8','fp16','int8','int4'), got {in_dtype!r}")
    is_f16 = in_dtype == "fp16"
    elem_bytes = 2 if is_f16 else 1
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    # K64-byte micro-step: always 64 bytes per `ku`. For fp16, this is 32 elements (2xK16 MFMA).
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: X is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    size_out = tokens * topk * inter_dim
    size_x = tokens * model_dim
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (experts * (2 * inter_dim) * model_dim) // 2 if is_int4 else (experts * (2 * inter_dim) * model_dim)
    size_sorted = int(sorted_size)
    size_expert_ids = int(size_expert_ids)

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use `flir.copy(load-only)` for buffer_load_dwordx4.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # CK-style LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE1_CSHUFFLE", "1") in ("1", "true", "True", "YES", "yes")
    use_cshuffle_epilog = bool(use_cshuffle_epilog)

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = f"mfma_moe1_{in_dtype}_{epilog_tag}".replace("-", "_")

    class _MOE1(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Optional epilogue CShuffle (LDS + vectorized buffer stores).
            # Reuse the same LDS bytes for both:
            # - ping-pong X tiles (2 * tile_m * lds_stride bytes; fp8/int8)
            # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
            lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
            lds_out_bytes = 2 * tile_m * tile_n if _use_cshuffle_epilog else 0
            lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
            lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)
            x_lds_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            _state["lds_x_decl"] = allocator.allocate_array(x_lds_elem, lds_total_elems)
            allocator.finalize()

        @flir.kernel
        def moe_gemm1(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(size_w, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            x_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            f16 = I.f16
            f32 = I.f32
            i32 = I.i32
            i64 = I.i64
            vec4_f32 = I.vec(4, f32)
            vec4_i32 = I.vec(4, i32)
            vec4_f16 = I.vec(4, f16)
            vec1_f16 = I.vec(1, f16)
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec4_elems = 4 if elem_bytes == 1 else 2
            vec8_x = I.vec(vec8_elems, x_elem)
            vec16_x = I.vec(vec16_elems, x_elem)
            vec1_i64 = I.vec(1, i64)
            vec2_i64 = I.vec(2, i64)

            def silu(x):
                # Align with CK's device fast path:
                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
                #   sig = rcp(1 + emu)                   -> v_rcp_f32
                #   y = x * sig
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                t = x * (-1.4426950408889634)  # -log2(e)
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                den = 1.0 + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return x * sig

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # Layouts
            layout_x = flir.make_layout((tokens_in, k_in), stride=(k_in, 1))

            # B preshuffle layout: match GEMM test helper exactly.
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                flir, arith, c_n=c_n_total, c_k=k_in, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes
            )
            layout_b = b_layout.layout_b
            # Only used by fp8/int8 path (16B gmem -> regs). Kept for backwards compat.
            atom_w_g2r16 = flir.make_copy_atom(w_elem, vector_size=16)

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")  # tile along sorted M
            by = gpu.block_id("y")  # tile along inter_dim

            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            atom_x_s16 = flir.make_copy_atom(x_elem, vector_size=16)
            atom_x_s8 = flir.make_copy_atom(x_elem, vector_size=8)
            atom_x_s4 = flir.make_copy_atom(x_elem, vector_size=4)
            atom_x_g2r16 = flir.make_copy_atom(x_elem, vector_size=vec16_elems)
            atom_x_g2r8 = flir.make_copy_atom(x_elem, vector_size=vec8_elems)
            atom_x_g2r4 = flir.make_copy_atom(x_elem, vector_size=vec4_elems)
            layout_tx_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = _state["lds_x_decl"](base_ptr)
            lds_x = lds_x_ptr.get()
            # Alias LDS bytes as fp16 for optional CShuffle epilogue.
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
            lds_out = (
                SmemPtr(base_ptr, lds_x_ptr.byte_offset, I.f16, shape=(tile_m * tile_n,)).get()
                if _use_cshuffle_epilog
                else None
            )

            # Use logical buffer sizes (descriptor num_records) so hardware OOB checking can be
            # used directly (CK-style). This allows us to avoid `select`-based masking for
            # invalid lanes and rely on the buffer instruction's built-in bounds behavior.
            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False)
            # fp16 path ignores scales completely (implicit scale=1.0).
            sx_rsrc = None if is_f16 else buffer_ops.create_buffer_resource(arg_scale_x, max_size=False)
            sw_rsrc = None if is_f16 else buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)

            # Expert id for this M tile (keep address math in `index`)
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            inter2_idx = arith.constant(2 * inter_dim, index=True)
            expert_off_idx = expert_idx * inter2_idx  # index

            bx_m = bx * arith.constant(tile_m, index=True)

            # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
            # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
            # 16, fall back to 8B (dwordx2) or 4B (dword) loads. This broadens supported tilings
            # (e.g. tile_m=16, tile_k=192 -> 12B/thread) at some performance cost.
            if is_f16:
                # fp16 path keeps the same fixed 16B gmem->reg schedule.
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

            # Work in dword units along K: K_dwords = (K_bytes)/4.
            c_k_div4 = (k_in * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
            layout_x_div4 = flir.make_layout((tokens_in, c_k_div4), stride=(c_k_div4, 1))
            tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
            layout_x_tile_div4 = flir.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            c_chunk_i32 = arith.constant(chunk_i32, index=True)
            tx_i32_base = tx * c_chunk_i32
            mask24 = arith.i32(0xFFFFFF)
            # Keep i32 constants available for epilogue index math.
            tokens_i32 = arith.i32(tokens)
            topk_i32 = arith.i32(topk)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                    chunk_i32=chunk_i32,
                )

            # CK-aligned: decode token once (per thread's M-slice) and build a base row offset.
            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32)
                t_i32 = arith.andi(fused_i, mask24)
                t_idx = arith.index_cast(ir.IndexType.get(), t_i32)
                x_row_base_div4.append(t_idx * c_k_div4)

            vec1_i32 = I.vec(1, i32)
            vec2_i32 = I.vec(2, i32)
            vec4_i32 = I.vec(4, i32)
            vec4_x = I.vec(4, x_elem)

            def load_x(idx_i32):
                """Load `x_load_bytes` bytes from X (gmem) into regs.

                For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                """
                if x_load_bytes == 16:
                    idx_elem = idx_i32 if elem_bytes == 1 else (idx_i32 * arith.index(2))
                    return buffer_copy_gmem16_dwordx4(
                        flir,
                        arg=arg_x,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        atom_g2r16=atom_x_g2r16,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                    )
                idx_bytes = idx_i32 * arith.index(4)
                atom = atom_x_g2r8 if x_load_bytes == 8 else atom_x_g2r4
                view = flir.TensorView(
                    arg_x,
                    (x_load_bytes,),
                    strides=(1,),
                    base_indices=(idx_bytes,),
                    element_type=x_elem,
                )
                return flir.copy(
                    atom,
                    view,
                    None,
                    alignment=x_load_bytes,
                    return_vector=True,
                    src_buffer_resource=x_rsrc,
                    src_buffer_offset_in_bytes=True,
                )

            def load_x_tile(base_k):
                """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                base_k_div4 = (base_k * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
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
            coord_wl = flir.idx2crd(tx, layout_tx_wave_lane)
            wave_id = flir.get(coord_wl, 0)
            lane_id = flir.get(coord_wl, 1)
            coord_l16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16*16B (KPackBytes=16).
            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * arith.constant(16, index=True)

            # Dynamic N tiling within block (same as existing kernels)
            by_n = by * arith.constant(tile_n, index=True)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = wave_id % arith.index(4)
            n_tile_base = wave_mod_4 * c_n_per_wave

            # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
            n_intra_gate = []
            n_blk_gate = []
            n_intra_up = []
            n_blk_up = []
            col_g_list = []
            valid_col_list = []
            inter_idx = arith.constant(inter_dim, index=True)
            # layout for (row -> (blk,intra)) where intra is 0..15
            c_n0 = c_n_total / arith.index(16)
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                offset = arith.constant(ni * 16, index=True)
                col_g = by_n + n_tile_base
                col_g = col_g + offset
                col_g = col_g + lane_mod_16
                col_g_list.append(col_g)

                row_gate = expert_off_idx + col_g
                row_up = row_gate + inter_idx

                coord_gate = flir.idx2crd(row_gate, layout_n_blk_intra)
                n_blk_gate.append(flir.get(coord_gate, 0))
                n_intra_gate.append(flir.get(coord_gate, 1))

                coord_up = flir.idx2crd(row_up, layout_n_blk_intra)
                n_blk_up.append(flir.get(coord_up, 0))
                n_intra_up.append(flir.get(coord_up, 1))

                valid_col_list.append(arith.ult(col_g, inter_idx))

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 64  # K64-byte micro-step

            # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
            def load_b_pack(base_k, ki_step, ni, blk_list, intra_list):
                return load_b_pack_k32(
                    buffer_ops,
                    flir,
                    arith,
                    vector,
                    arg_b=arg_w,
                    b_rsrc=w_rsrc,
                    layout_b=layout_b,
                    base_k=base_k,
                    ki_step=ki_step,
                    n_blk=blk_list[ni],
                    n_intra=intra_list[ni],
                    lane_div_16=lane_div_16,  # 0..3
                    elem_type=w_elem,
                    kpack_bytes=kpack_bytes,
                    elem_bytes=elem_bytes,
                    unpack_int4=is_int4,
                )

            def load_b_tile(base_k, blk_list, intra_list):
                """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                Returns a list of length `k_unroll`, where each entry is a tuple:
                  (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                """
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        if is_int4:
                            # Packed int4: reuse K32 loader twice (2x4B loads + unpack).
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack(base_k, ki0, ni, blk_list, intra_list)
                            b1 = load_b_pack(base_k, ki1, ni, blk_list, intra_list)
                        elif is_f16:
                            # fp16: reuse K32 loader twice (2x8B packs -> 2x MFMA16).
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack(base_k, ki0, ni, blk_list, intra_list)
                            b1 = load_b_pack(base_k, ki1, ni, blk_list, intra_list)
                        else:
                            # FP8/INT8: load 16 bytes (one full KPack) and split into 2x i64 halves.
                            k0_base = base_k / arith.index(64)
                            k0 = k0_base + arith.constant(ku, index=True)
                            k1 = lane_div_16
                            coord_pack = flir.make_coord(
                                blk_list[ni], k0, k1, intra_list[ni], arith.index(0)
                            )
                            idx_pack_bytes = flir.crd2idx(coord_pack, layout_b)
                            w_view = flir.TensorView(
                                arg_w,
                                (16,),
                                strides=(1,),
                                base_indices=(idx_pack_bytes,),
                                element_type=w_elem,
                            )
                            w16 = flir.copy(
                                atom_w_g2r16,
                                w_view,
                                None,
                                alignment=8,
                                return_vector=True,
                                src_buffer_resource=w_rsrc,
                                src_buffer_offset_in_bytes=True,
                            )
                            w_i64x2 = vector.bitcast(vec2_i64, w16)
                            b0 = vector.extract(w_i64x2, static_position=[0], dynamic_position=[])
                            b1 = vector.extract(w_i64x2, static_position=[1], dynamic_position=[])
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            acc_gate = [acc_init] * (num_acc_n * m_repeat)
            acc_up = [acc_init] * (num_acc_n * m_repeat)

            # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
            def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                for i in range_constexpr(num_x_loads):
                    row_local = x_row_local[i]
                    col_local_i32 = x_col_local_i32[i]
                    if x_load_bytes == 16:
                        lds_store_16b_xor16(
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec16_ty=vec16_x,
                            elem_type=x_elem,
                            atom_s16=atom_x_s16,
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
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec8_ty=vec8_x,
                            elem_type=x_elem,
                            atom_s8=atom_x_s8,
                            layout_lds=layout_lds,
                            row_local=row_local,
                            col_local_i32=col_local_i32,
                            tx_c4=arith.index(4),
                            k_blocks16=k_blocks16,
                            lds_base=lds_base,
                            vec_part_i32x2=vec_x_in_parts[i],
                            elem_bytes=elem_bytes,
                        )
                    else:
                        lds_store_4b_xor16(
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec4_ty=vec4_x,
                            elem_type=x_elem,
                            atom_s4=atom_x_s4,
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
                col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes / arith.index(2))
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                idx_a16 = idx_a16 + lds_base
                loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
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
                mfma_res_ty = vec4_i32 if is_int8 else vec4_f32
                mfma_fn = (
                    mfma_i32_k32
                    if is_int8
                    else (rocdl.mfma_f32_16x16x16f16 if is_f16 else rocdl.mfma_f32_16x16x32_fp8_fp8)
                )

                # Optional: prefetch epilogue scales while we are about to run the last MFMA tile,
                # matching the preshuffle GEMM pattern of overlapping scale loads with MFMA.
                epilogue_pf = None
                if prefetch_epilogue and (not is_f16):
                    expert_off_pf = expert_off_idx
                    sw_gate_pf = []
                    sw_up_pf = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        valid_col = valid_col_list[ni]
                        row_gate_idx = expert_off_pf + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_pf.append(buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=f32))
                        sw_up_pf.append(buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=f32))
                    epilogue_pf = (sw_gate_pf, sw_up_pf)

                def _i64_to_v4f16(x_i64):
                    v1 = vector.from_elements(vec1_i64, [x_i64])
                    return vector.bitcast(vec4_f16, v1)

                def mfma_k64(acc_in, a0, a1, b0, b1):
                    if is_f16:
                        a0v = _i64_to_v4f16(a0)
                        a1v = _i64_to_v4f16(a1)
                        b0v = _i64_to_v4f16(b0)
                        b1v = _i64_to_v4f16(b1)
                        acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc_in, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])
                    acc_mid = mfma_fn(mfma_res_ty, [a0, b0, acc_in, 0, 0, 0])
                    return mfma_fn(mfma_res_ty, [a1, b1, acc_mid, 0, 0, 0])

                for ku in range_constexpr(k_unroll):
                    b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]
                    b_up_packs0, b_up_packs1 = b_up_tile_in[ku]
                    ki64 = arith.constant(ku * 64, index=True)
                    col_base = col_offset_base + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
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

            # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
            lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
            lds_base_cur = arith.index(0)
            lds_base_nxt = lds_tile_elems

            # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                mfma_group = num_acc_n * 2
                # K64 micro-step: 2x K32 MFMA per gemm.
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

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
            b_gate_cur = load_b_tile(k0, n_blk_gate, n_intra_gate)
            b_up_cur = load_b_tile(k0, n_blk_up, n_intra_up)
            store_x_tile_to_lds(x_regs0, lds_base_cur)
            gpu.barrier()

            # Loop-carried ping/pong state.
            lds_base_pong = lds_base_cur  # current/compute
            lds_base_ping = lds_base_nxt  # next/load+store

            # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
            # tile we are about to compute from LDS, to overlap with upcoming VMEM.
            a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_pong)

            # Unrolled ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
            c2_tile_k = arith.constant(tile_k * 2, index=True)
            c_k_main2 = k_in - c2_tile_k

            for k_iv in range(arith.index(0), c_k_main2, c2_tile_k):
                # ---- stage 0: prefetch+store ping, compute pong ----
                next_k1 = k_iv + tile_k
                x_regs_ping = load_x_tile(next_k1)
                b_gate_ping = load_b_tile(next_k1, n_blk_gate, n_intra_gate)
                b_up_ping = load_b_tile(next_k1, n_blk_up, n_intra_up)

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

                # Cross-tile prefetch for the ping tile we are about to compute.
                a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_ping)

                # ---- stage 1: prefetch+store pong, compute ping ----
                next_k2 = k_iv + c2_tile_k
                x_regs_pong = load_x_tile(next_k2)
                b_gate_next = load_b_tile(next_k2, n_blk_gate, n_intra_gate)
                b_up_next = load_b_tile(next_k2, n_blk_up, n_intra_up)

                acc_gate, acc_up, _ = compute_tile(
                    acc_gate,
                    acc_up,
                    b_gate_ping,
                    b_up_ping,
                    lds_base_ping,
                    a0_prefetch=a0_prefetch_ping,
                )
                a0_prefetch_ping = None
                store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                hot_loop_scheduler()
                gpu.barrier()

                # Cross-tile prefetch for the next pong tile.
                a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_pong)

                # Advance pong state to next_k2 for next iteration.
                b_gate_cur = b_gate_next
                b_up_cur = b_up_next

            # Tail: 2 remaining tiles at (k_in - 2*tile_k) and (k_in - tile_k).
            k_tail1 = k_in - tile_k
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
            a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_ping)

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
            inter_i32_v = arith.i32(inter_dim)
            mask24_i32 = arith.i32(0xFFFFFF)

            if is_f16:
                # fp16: no scale fetch, no scale multiply in epilogue.
                sw_gate_vals = None
                sw_up_vals = None
            elif epilogue_pf is not None:
                sw_gate_vals, sw_up_vals = epilogue_pf
            else:
                sw_gate_vals = []
                sw_up_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = col_g_list[ni]
                    row_gate_idx = expert_off + col_g
                    row_up_idx = row_gate_idx + inter_idx
                    sw_gate_vals.append(buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=f32))
                    sw_up_vals.append(buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=f32))

            # Epilogue hoists to keep IR + Python build time small:
            col_i32_list = []
            for ni in range_constexpr(num_acc_n):
                col_i32_list.append(arith.index_cast(i32, col_g_list[ni]))

            lane_div_16_mul4 = lane_div_16 * arith.index(4)
            inter_i32_local = inter_i32_v

            # Optional: CK-style CShuffle epilogue for better global store coalescing.
            # Uses EVec=4 (buffer store "x4" of fp16 elements).
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)

            if _use_cshuffle_epilog:
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
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                    t2 = fused2 & mask24_i32
                    # No explicit mask: rely on buffer descriptor OOB to zero-fill when t2 is the
                    # sentinel (t2 == tokens) or otherwise out-of-range.
                    sx = None if is_f16 else buffer_ops.buffer_load(sx_rsrc, t2, vec_width=1, dtype=f32)

                    # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)

                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        sw_gate = sw_gate_vals[ni] if (not is_f16) else None
                        sw_up = sw_up_vals[ni] if (not is_f16) else None

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.extract(
                            acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        vu = vector.extract(
                            acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                        )

                        if is_int8:
                            vg = arith.sitofp(f32, vg)
                            vu = arith.sitofp(f32, vu)
                        if not is_f16:
                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up

                        y = silu(vg) * vu
                        if doweight_stage1:
                            y = y * tw
                        y16 = arith.trunc_f(T.f16(), y)

                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(vec1_f16, [y16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24
                    return (t2 * topk_i32_v + s2) * inter_i32_local

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    idx0 = row_ctx
                    col_i32 = arith.index_cast(i32, col_g0)
                    idx_out = idx0 + col_i32
                    # Vectorized fp16 store (EVec=4).
                    buffer_ops.buffer_store(frag, out_rsrc, idx_out)
                mfma_epilog(
                    use_cshuffle=True,
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
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
                fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                t2 = fused2 & mask24_i32
                s2 = fused2 >> 24
                # No explicit mask: rely on buffer descriptor OOB to zero-fill when t2 is the
                # sentinel (t2 == tokens) or otherwise out-of-range.
                sx = None if is_f16 else buffer_ops.buffer_load(sx_rsrc, t2, vec_width=1, dtype=f32)

                # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                if doweight_stage1:
                    tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)

                for ni in range_constexpr(num_acc_n):
                    col_i32 = col_i32_list[ni]
                    sw_gate = sw_gate_vals[ni] if (not is_f16) else None
                    sw_up = sw_up_vals[ni] if (not is_f16) else None

                    acc_idx = mi * num_acc_n + ni
                    vg = vector.extract(
                        acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                    )
                    vu = vector.extract(
                        acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                    )

                    if is_int8:
                        vg = arith.sitofp(f32, vg)
                        vu = arith.sitofp(f32, vu)
                    if not is_f16:
                        vg = vg * sx * sw_gate
                        vu = vu * sx * sw_up

                    y = silu(vg) * vu
                    if doweight_stage1:
                        y = y * tw
                    y = arith.trunc_f(T.f16(), y)
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

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(size_w, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            bdx = 256
            gx = size_expert_ids
            gy = inter_dim // tile_n
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm1"],
                grid_size=(gx, gy, 1),
                block_size=(bdx, 1, 1),
                kernel_operands=[
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    tokens_in,
                    inter_in,
                    k_in,
                ],
            )

    m = _MOE1()
    exe = flydsl.compile(m)
    return exe


def compile_moe_gemm2(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    sorted_size: int,
    size_expert_ids: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    out_dtype: str | None = None,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    in_dtype:
      - "fp8": A2/W are fp8
      - "fp16": A2/W are fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": A2/W are int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 unpacked to int8 in-kernel

    out_dtype:
      - "f16": output is f16, using the LDS CShuffle + half2 atomic path
      - "f32": output is f32, using the scalar f32 atomic path
      - None: fallback to env var `FLIR_MOE_STAGE2_CSHUFFLE` (backwards compatible)
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if in_dtype not in ("fp8", "fp16", "int8", "int4"):
        raise ValueError(f"in_dtype must be one of ('fp8','fp16','int8','int4'), got {in_dtype!r}")
    is_f16 = in_dtype == "fp16"
    elem_bytes = 2 if is_f16 else 1
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    size_out = tokens * model_dim
    size_x = tokens * topk * inter_dim
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (experts * model_dim * inter_dim) // 2 if is_int4 else (experts * model_dim * inter_dim)
    size_sorted = int(sorted_size)
    size_expert_ids = int(size_expert_ids)

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    # Stage2 output dtype / epilogue mode:
    # - f16: LDS CShuffle + half2 atomics
    # - f32: scalar f32 atomics (no CShuffle/LDS scratch)
    if out_dtype is None:
        _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE2_CSHUFFLE", "1") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    else:
        out_s = str(out_dtype).strip().lower()
        if out_s in ("f16", "fp16", "half"):
            _use_cshuffle_epilog = True
        elif out_s in ("f32", "fp32", "float"):
            _use_cshuffle_epilog = False
        else:
            raise ValueError(f"out_dtype must be 'f16' or 'f32', got {out_dtype!r}")

    # NOTE: Keep this as a callable (not `T.f16()` / `T.f32()`) so we don't require an
    # MLIR Context at Python-time. It will be invoked inside the module-building context.
    out_elem = T.f16 if _use_cshuffle_epilog else T.f32
    epilog_tag = "cshuffle" if _use_cshuffle_epilog else "f32"
    module_name = f"mfma_moe2_{in_dtype}_{epilog_tag}".replace("-", "_")

    class _MOE2(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # LDS scratch: reuse a single allocation for both
            # - ping-pong A2 tiles (2 * tile_m * lds_stride bytes; fp8/int8)
            # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
            #
            # This reduces LDS usage from sum(...) to max(...).
            lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
            lds_out_bytes = 2 * tile_m * tile_n if _use_cshuffle_epilog else 0  # f16 bytes
            lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
            lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)
            x_lds_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            _state["lds_x_decl"] = allocator.allocate_array(x_lds_elem, lds_total_elems)
            allocator.finalize()

        @flir.kernel
        def moe_gemm2(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, out_elem()),
            arg_x: lambda: T.memref(size_x, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(size_w, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(tokens * topk, T.f32()),
            arg_scale_w: lambda: T.memref(experts * model_dim, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            x_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            f16 = I.f16
            f32 = I.f32
            i32 = I.i32
            i64 = I.i64
            vec4_f32 = I.vec(4, f32)
            vec4_i32 = I.vec(4, i32)
            vec4_f16 = I.vec(4, f16)
            vec1_f16 = I.vec(1, f16)
            vec2_f16 = I.vec(2, f16)
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec4_elems = 4 if elem_bytes == 1 else 2
            vec8_x = I.vec(vec8_elems, x_elem)
            vec16_x = I.vec(vec16_elems, x_elem)
            vec1_i64 = I.vec(1, i64)
            vec2_i64 = I.vec(2, i64)

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # A2 layout (flatten token-slot -> M).
            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx
            layout_x = flir.make_layout((m_in, k_in), stride=(k_in, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.constant(experts * model_dim, index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                flir, arith, c_n=c_n_total, c_k=k_in, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes
            )
            layout_b = b_layout.layout_b
            # Only used by fp8/int8 path (16B gmem -> regs). Kept for backwards compat.
            atom_w_g2r16 = flir.make_copy_atom(w_elem, vector_size=16)

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")  # tile along sorted M
            by = gpu.block_id("y")  # tile along model_dim

            # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            atom_x_s16 = flir.make_copy_atom(x_elem, vector_size=16)
            atom_x_s8 = flir.make_copy_atom(x_elem, vector_size=8)
            atom_x_s4 = flir.make_copy_atom(x_elem, vector_size=4)
            atom_x_g2r16 = flir.make_copy_atom(x_elem, vector_size=vec16_elems)
            atom_x_g2r8 = flir.make_copy_atom(x_elem, vector_size=vec8_elems)
            atom_x_g2r4 = flir.make_copy_atom(x_elem, vector_size=vec4_elems)
            layout_tx_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            layout_lin_rowcol = flir.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = _state["lds_x_decl"](base_ptr)
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as fp16 for epilogue shuffle.
            lds_out = (
                SmemPtr(base_ptr, lds_x_ptr.byte_offset, I.f16, shape=(tile_m * tile_n,)).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources (logical sizes: allow hardware OOB checks).
            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False)
            # fp16 path ignores scales completely (implicit scale=1.0).
            sx_rsrc = None if is_f16 else buffer_ops.create_buffer_resource(arg_scale_x, max_size=False)
            sw_rsrc = None if is_f16 else buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)

            # Expert id for this M tile.
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            n_idx = arith.constant(model_dim, index=True)
            expert_off_idx = expert_idx * n_idx  # index

            bx_m = bx * arith.constant(tile_m, index=True)

            # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
            # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
            # 16, fall back to 8B (dwordx2) or 4B (dword) loads.
            if is_f16:
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
            vec4_i32 = I.vec(4, i32)

            c_k_div4 = (k_in * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
            layout_x_div4 = flir.make_layout((m_in, c_k_div4), stride=(c_k_div4, 1))
            tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
            layout_x_tile_div4 = flir.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            c_chunk_i32 = arith.constant(chunk_i32, index=True)
            tx_i32_base = tx * c_chunk_i32

            topk_i32 = arith.i32(topk)
            mask24 = arith.i32(0xFFFFFF)
            tokens_i32 = arith.i32(tokens)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                    chunk_i32=chunk_i32,
                )

            vec1_i32 = I.vec(1, i32)
            vec2_i32 = I.vec(2, i32)
            vec4_x = I.vec(4, x_elem)

            def load_x(idx_i32):
                if x_load_bytes == 16:
                    idx_elem = idx_i32 if elem_bytes == 1 else (idx_i32 * arith.index(2))
                    return buffer_copy_gmem16_dwordx4(
                        flir,
                        arg=arg_x,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        atom_g2r16=atom_x_g2r16,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                    )
                idx_bytes = idx_i32 * arith.index(4)
                atom = atom_x_g2r8 if x_load_bytes == 8 else atom_x_g2r4
                view = flir.TensorView(
                    arg_x,
                    (x_load_bytes,),
                    strides=(1,),
                    base_indices=(idx_bytes,),
                    element_type=x_elem,
                )
                return flir.copy(
                    atom,
                    view,
                    None,
                    alignment=x_load_bytes,
                    return_vector=True,
                    src_buffer_resource=x_rsrc,
                    src_buffer_offset_in_bytes=True,
                )

            # CK-aligned: decode routed token once (per thread's M-slice) and build a base offset.
            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32)
                t_i32 = arith.andi(fused_i, mask24)
                s_i32 = arith.shrui(fused_i, arith.i32(24))
                # Guard sentinel padded token id (t_i32 == tokens) / any OOB:
                # clamp row index so global loads stay in-bounds; masked stores/atomics
                # will zero out contributions for invalid rows.
                t_valid = arith.ult(t_i32, tokens_i32)
                t_i32_safe = arith.select(t_valid, t_i32, arith.i32(0))
                # A2 row index = t*topk + s (still i32 here).
                row_ts_i32 = t_i32_safe * topk_i32 + s_i32
                row_ts_idx = arith.index_cast(ir.IndexType.get(), row_ts_i32)
                # Base row offset in dword units: row_ts_idx * (k_in/4)
                x_row_base_div4.append(row_ts_idx * c_k_div4)

            def load_x_tile(base_k):
                base_k_div4 = (base_k * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
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
            coord_wl = flir.idx2crd(tx, layout_tx_wave_lane)
            wave_id = flir.get(coord_wl, 0)
            lane_id = flir.get(coord_wl, 1)
            coord_l16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * arith.constant(16, index=True)

            # Dynamic N tiling within block.
            by_n = by * arith.constant(tile_n, index=True)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = wave_id % arith.constant(4, index=True)
            n_tile_base = wave_mod_4 * c_n_per_wave

            # Precompute (n_blk, n_intra) for B, and col indices for output.
            n_intra_list = []
            n_blk_list = []
            col_g_list = []
            c_n0 = c_n_total / arith.index(16)
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                offset = arith.constant(ni * 16, index=True)
                col_g = by_n + n_tile_base + offset + lane_mod_16
                col_g_list.append(col_g)

                row_w = expert_off_idx + col_g
                coord_w = flir.idx2crd(row_w, layout_n_blk_intra)
                n_blk_list.append(flir.get(coord_w, 0))
                n_intra_list.append(flir.get(coord_w, 1))

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 64

            # --- B Load Logic (K64) ---
            def load_b_pack(base_k, ki_step, ni):
                return load_b_pack_k32(
                    buffer_ops,
                    flir,
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
                    elem_bytes=elem_bytes,
                    unpack_int4=is_int4,
                )

            def load_b_tile(base_k):
                """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                Returns a list of length `k_unroll`, where each entry is a tuple:
                  (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                """
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        if is_int4:
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack(base_k, ki0, ni)
                            b1 = load_b_pack(base_k, ki1, ni)
                        elif is_f16:
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack(base_k, ki0, ni)
                            b1 = load_b_pack(base_k, ki1, ni)
                        else:
                            k0_base = base_k / arith.index(64)
                            k0 = k0_base + arith.constant(ku, index=True)
                            k1 = lane_div_16
                            coord_pack = flir.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], arith.index(0))
                            idx_pack_bytes = flir.crd2idx(coord_pack, layout_b)
                            w_view = flir.TensorView(
                                arg_w,
                                (16,),
                                strides=(1,),
                                base_indices=(idx_pack_bytes,),
                                element_type=w_elem,
                            )
                            w16 = flir.copy(
                                atom_w_g2r16,
                                w_view,
                                None,
                                alignment=8,
                                return_vector=True,
                                src_buffer_resource=w_rsrc,
                                src_buffer_offset_in_bytes=True,
                            )
                            w_i64x2 = vector.bitcast(vec2_i64, w16)
                            b0 = vector.extract(w_i64x2, static_position=[0], dynamic_position=[])
                            b1 = vector.extract(w_i64x2, static_position=[1], dynamic_position=[])
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
            def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                for i in range_constexpr(num_x_loads):
                    row_local = x_row_local[i]
                    col_local_i32 = x_col_local_i32[i]
                    if x_load_bytes == 16:
                        lds_store_16b_xor16(
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec16_ty=vec16_x,
                            elem_type=x_elem,
                            atom_s16=atom_x_s16,
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
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec8_ty=vec8_x,
                            elem_type=x_elem,
                            atom_s8=atom_x_s8,
                            layout_lds=layout_lds,
                            row_local=row_local,
                            col_local_i32=col_local_i32,
                            tx_c4=arith.index(4),
                            k_blocks16=k_blocks16,
                            lds_base=lds_base,
                            vec_part_i32x2=vec_x_in_parts[i],
                            elem_bytes=elem_bytes,
                        )
                    else:
                        lds_store_4b_xor16(
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec4_ty=vec4_x,
                            elem_type=x_elem,
                            atom_s4=atom_x_s4,
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
                col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes / arith.index(2))
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                idx_a16 = idx_a16 + lds_base
                loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0, a1

            def compute_tile(acc_in, b_tile_in, lds_base, *, prefetch_epilogue: bool = False, a0_prefetch=None):
                acc_list = list(acc_in)
                mfma_res_ty = vec4_i32 if is_int8 else vec4_f32
                mfma_fn = (
                    mfma_i32_k32
                    if is_int8
                    else (rocdl.mfma_f32_16x16x16f16 if is_f16 else rocdl.mfma_f32_16x16x32_fp8_fp8)
                )

                epilogue_pf = None
                if prefetch_epilogue and (not is_f16):
                    expert_off_pf = expert_off_idx
                    sw_pf = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_w_idx = expert_off_pf + col_g
                        sw_pf.append(
                            buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=f32)
                        )
                    # Also prefetch per-row routed/topk weights (sorted_weights) when enabled.
                    tw_pf = None
                    if doweight_stage2:
                        tw_pf = []
                        lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                        ii_idx_list_pf = [arith.constant(ii, index=True) for ii in range(4)]
                        for mi in range_constexpr(m_repeat):
                            mi_base_pf = arith.constant(mi * 16, index=True)
                            for ii in range_constexpr(4):
                                row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                row_in_tile_pf = mi_base_pf + row_off_pf
                                sorted_row_pf = bx_m + row_in_tile_pf
                                tw_pf.append(
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc, sorted_row_pf, vec_width=1, dtype=f32
                                    )
                                )
                    epilogue_pf = (sw_pf, tw_pf)

                def mfma_k64(acc0, a0, a1, b0, b1):
                    if is_f16:
                        a0v = vector.bitcast(vec4_f16, vector.from_elements(vec1_i64, [a0]))
                        a1v = vector.bitcast(vec4_f16, vector.from_elements(vec1_i64, [a1]))
                        b0v = vector.bitcast(vec4_f16, vector.from_elements(vec1_i64, [b0]))
                        b1v = vector.bitcast(vec4_f16, vector.from_elements(vec1_i64, [b1]))
                        acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                    acc1 = mfma_fn(mfma_res_ty, [a0, b0, acc0, 0, 0, 0])
                    return mfma_fn(mfma_res_ty, [a1, b1, acc1, 0, 0, 0])

                for ku in range_constexpr(k_unroll):
                    b_packs0, b_packs1 = b_tile_in[ku]
                    ki64 = arith.constant(ku * 64, index=True)
                    col_base = col_offset_base + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
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
            lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
            lds_base_cur = arith.index(0)
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
                # - MFMA group size per "slot": num_acc_n
                # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                mfma_group = num_acc_n
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                # DS-read preload (CK default is 2).
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
            store_x_tile_to_lds(x_regs0, lds_base_cur)
            gpu.barrier()

            acc = [acc_init] * (num_acc_n * m_repeat)
            lds_base_pong = lds_base_cur
            lds_base_ping = lds_base_nxt

            # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
            # tile we are about to compute from LDS, to overlap with upcoming VMEM.
            a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_pong)

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

            c2_tile_k = arith.constant(tile_k * 2, index=True)
            c_k_main2 = arith.index(k_main2_py)
            for k_iv in range(arith.index(0), c_k_main2, arith.index(tile_k * 2)):
                next_k1 = k_iv + tile_k
                x_regs_ping = load_x_tile(next_k1)
                b_ping = load_b_tile(next_k1)

                acc, _ = compute_tile(acc, b_cur, lds_base_pong, a0_prefetch=a0_prefetch_pong)
                a0_prefetch_pong = None
                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()

                # Cross-tile prefetch for the ping tile we are about to compute.
                a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_ping)

                next_k2 = k_iv + c2_tile_k
                x_regs_pong = load_x_tile(next_k2)
                b_next = load_b_tile(next_k2)

                acc, _ = compute_tile(acc, b_ping, lds_base_ping, a0_prefetch=a0_prefetch_ping)
                a0_prefetch_ping = None
                store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                hot_loop_scheduler()
                gpu.barrier()

                # Cross-tile prefetch for the next pong tile.
                a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_pong)

                b_cur = b_next

            if odd_k_tiles:
                # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                acc, epilogue_pf = compute_tile(
                    acc,
                    b_cur,
                    lds_base_pong,
                    prefetch_epilogue=True,
                    a0_prefetch=a0_prefetch_pong,
                )
            else:
                # Tail: 2 remaining tiles.
                k_tail1 = k_in - tile_k
                x_regs_ping = load_x_tile(k_tail1)
                b_ping = load_b_tile(k_tail1)

                acc, _ = compute_tile(acc, b_cur, lds_base_pong, a0_prefetch=a0_prefetch_pong)
                a0_prefetch_pong = None
                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()

                # Epilogue tile with sw prefetch.
                a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base, lds_base_ping)
                acc, epilogue_pf = compute_tile(
                    acc, b_ping, lds_base_ping, prefetch_epilogue=True, a0_prefetch=a0_prefetch_ping
                )

            # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
            # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.
            expert_off = expert_off_idx
            mask24_i32 = arith.i32(0xFFFFFF)
            model_i32 = arith.i32(model_dim)
            topk_i32_v = topk_i32

            zero_i32 = arith.i32(0)
            c2_i32 = arith.i32(2)  # fp16 element size in bytes
            mask_even_i32 = arith.i32(0xFFFFFFFE)  # align element index to even for half2 atomics

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
            if is_f16:
                # fp16: no scale fetch, no scale multiply in epilogue.
                sw_vals = None
            elif sw_pf is not None:
                sw_vals = sw_pf
            else:
                sw_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = col_g_list[ni]
                    row_w_idx = expert_off + col_g
                    sw_vals.append(
                        buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=f32)
                    )

            if _use_cshuffle_epilog:
                if lds_out is None:
                    raise RuntimeError(
                        "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
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
                    # row is the sorted-row index (bx_m + row_in_tile).
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24

                    # a2_scale index = t*topk + s (i32). Hardware OOB handles sentinel padding.
                    ts2 = t2 * topk_i32_v + s2
                    sx = None if is_f16 else buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=f32)

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
                        sw = sw_vals[ni] if (not is_f16) else None
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                        if is_int8:
                            v = arith.sitofp(f32, v)
                        if not is_f16:
                            v = v * sx * sw
                        if doweight_stage2:
                            v = v * tw
                        v16 = arith.trunc_f(T.f16(), v)

                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(vec1_f16, [v16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    # Token id for this output row (same across all N pairs).
                    fused_s = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                    t_s = fused_s & mask24_i32
                    return t_s * model_i32  # i32 element index base

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    idx0 = row_ctx
                    col_i32 = arith.index_cast(i32, col_g0)
                    idx_elem = idx0 + col_i32
                    idx_elem_even = idx_elem & mask_even_i32
                    byte_off = idx_elem_even * c2_i32
                    atomic_add_f16x2(frag, byte_off)

                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
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
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )
            else:
                # Default (no CShuffle): output f32 and use scalar atomic fadd (baseline/compare).
                # NOTE: In this mode, `arg_out` is f32 and `out_rsrc` is a f32 buffer.
                c4_i32 = arith.i32(4)

                def atomic_add_f32(val_f32, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f32,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

                def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24

                    # a2_scale index = t*topk + s (i32). Hardware OOB handles sentinel padding.
                    ts2 = t2 * topk_i32_v + s2
                    sx = None if is_f16 else buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=f32)

                    # out element base = t*model_dim (i32)
                    idx0 = t2 * model_i32

                    if doweight_stage2:
                        tw_idx = (mi * 4) + ii
                        if tw_pf is not None:
                            tw = tw_pf[tw_idx]
                        else:
                            tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)

                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]  # index
                        sw = sw_vals[ni] if (not is_f16) else None
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                        if is_int8:
                            v = arith.sitofp(f32, v)
                        if not is_f16:
                            v = v * sx * sw
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

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, out_elem()),
            arg_x: lambda: T.memref(size_x, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(size_w, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(tokens * topk, T.f32()),
            arg_scale_w: lambda: T.memref(experts * model_dim, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            bdx = 256
            gx = size_expert_ids
            gy = model_dim // tile_n
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm2"],
                grid_size=(gx, gy, 1),
                block_size=(bdx, 1, 1),
                kernel_operands=[
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    tokens_in,
                    n_in,
                    k_in,
                ],
            )

    m = _MOE2()
    exe = flydsl.compile(m)
    return exe

