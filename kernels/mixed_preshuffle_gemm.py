"""Preshuffle GEMM kernel implementations (FLIR MFMA FP8/INT8).

This module intentionally contains the **kernel builder code** for the preshuffle GEMM,
extracted from `tests/kernels/test_preshuffle_gemm.py` in the same style as
`kernels/moe_gemm_2stage.py`:
- `kernels/` holds the implementation (compile functions)
- `tests/` holds correctness/perf harnesses

Pipelines:
- `pingpong`: tuned 2-stage pipeline with ping-pong LDS for A (2 LDS buffers)
- `ck_v1_single_lds`: CK-like Intrawave + bpreshuffle v1 spirit (single LDS buffer for A)
"""

import os

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator, SmemPtr

from _mlir import ir

from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl
from flydsl.lang.ir.types import T, memref

from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
)
from kernels.mfma_epilogues import mfma_epilog
from kernels.kernels_common import stream_ptr_to_async_token


def compile_mxfp4_preshuffle_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    lds_stage: int = 2,
    # Epilogue options
    use_cshuffle_epilog: bool = True,
    # Memory transfer options
    use_async_copy: bool = False,
):
    """Compile the preshuffle GEMM kernel and return the compiled executable.

    Args:
        M, N, K: GEMM sizes (A[M,K], B[N,K], C[M,N]).
        tile_m, tile_n, tile_k: block tile sizes.
        in_dtype:
          - "fp8": A/B are fp8 (1B/elem)
          - "int8": A/B are int8 (1B/elem)
          - "int4": W4A8 path: A is int8, B is packed int4 (2 values per byte) and unpacked to int8 in-kernel.
        lds_stage: 
          - 2: ping-pong LDS for A (2 LDS buffers), tuned schedule (original).
          - 1: single LDS buffer for A .
    """
    # if in_dtype not in ("fp8", "int8", "int4", "fp16", "bf16"):
    #     raise ValueError(
    #         "in_dtype must be one of ('fp8','int8','int4','fp16','bf16'), "
    #         f"got {in_dtype!r}"
    #     )

    is_fp4_a = a_dtype == "fp4"
    is_fp8_a = a_dtype == "fp8"
    is_fp4_b = b_dtype == "fp4"
    is_bf16_out = out_dtype == "bf16"

    a_elem_vec_pack = 2 if is_fp4_a else  1
    b_elem_vec_pack = 2

    elem_bytes = 1

    pack_M = 2
    pack_N = 2
    pack_K = 2

    quant_block_size_a = 32
    quant_block_size_b = 32



    cbsz = 0 if is_fp8_a else 4
    blgp = 4

    # Pipeline is byte-addressed along K (16B loads, XOR16 swizzle in bytes).
    # For fp16/bf16 (2B/elem), user passes tile_k halved so tile_k_bytes stays constant.
    tile_k_bytes = int(tile_k) * int(elem_bytes)

    # K64-byte micro-step wrapper uses 2x "half-pack" MFMA.
    # - fp8/i8: mfma K32, wrapper covers 64B (=64 elems)
    # - fp16/bf16: mfma K16, wrapper covers 64B (=32 elems)  -> effective K halves
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )

    # MXFP4 packing constraints:
    # - k_unroll_packed = (tile_k_bytes // 128) // pack_K must be >= 1 => tile_k >= 256
    # - num_acc_n_packed = (tile_n // 4 // 16) // pack_N must be >= 1 => tile_n >= 128
    num_waves = 4
    k_unroll_check = tile_k_bytes // 128
    k_unroll_packed_check = k_unroll_check // pack_K
    n_per_wave_check = int(tile_n) // num_waves
    num_acc_n_check = n_per_wave_check // 16
    num_acc_n_packed_check = num_acc_n_check // pack_N
    if k_unroll_packed_check < 1:
        raise ValueError(
            f"MXFP4 requires tile_k >= {128 * pack_K} (pack_K={pack_K}), got tile_k={tile_k} "
            f"(k_unroll={k_unroll_check}, k_unroll_packed={k_unroll_packed_check})"
        )
    if num_acc_n_packed_check < 1:
        raise ValueError(
            f"MXFP4 requires tile_n >= {num_waves * 16 * pack_N} (pack_N={pack_N}, num_waves={num_waves}), "
            f"got tile_n={tile_n} (n_per_wave={n_per_wave_check}, num_acc_n={num_acc_n_check}, "
            f"num_acc_n_packed={num_acc_n_packed_check})"
        )

    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    _state = {}

    # Default-on: cross-tile (tile_k) A0 LDS prefetch in the ping-pong pipeline (lds_stage=2).
    #
    # This issues the *first* A-pack LDS read for the next tile between barriers, to overlap
    # with the VMEM prefetch of the following tile.

    DYN = ir.ShapedType.get_dynamic_size()

    # Vector width calc (assume full tiles / no tail guards).
    total_threads = 256
    bytes_a_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes) // a_elem_vec_pack
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_a = bytes_a_per_tile // total_threads

    # Assume A loads are always 16B-aligned and use fixed dwordx4 (16B) buffer loads.
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )

    # Async DMA: gfx950 supports 16B direct GMEM→LDS, gfx942 supports 4B.
    a_async_load_bytes = 4 if gpu_arch == "gfx942" else 16
    a_async_load_dword = a_async_load_bytes // 4

    # CK-style LDS128: stride is in BYTES along K (for XOR16 swizzle).
    lds_stride_bytes = tile_k_bytes

    #TODO: use f4?
    def _a_elem_type():
        # return T.f8 if is_fp8_a else T.ui8
        return T.f8 if is_fp8_a else T.ui8

    def _b_elem_type():
        return T.ui8

    def _scale_elem_type():
        return T.i32

    #TODO: use f4 pack?
    def _a_vec16_type():
        return T.f8x16 if is_fp8_a else T.ui8x16

    def _b_vec16_type():
        return T.ui8x16

    def _mfma_pack_ty():
        # ROCDL MFMA intrinsics expect specific operand vector types:
        # - fp8/int8 paths use i64 packs (8 bytes)
        # - fp16 uses v4f16 (8 bytes)
        # - bf16 uses v4i16 (8 bytes) for *_bf16_1k variants
        if is_f16:
            return T.f16x4
        if is_bf16:
            return T.i16x4
        return T.i64

    def _out_elem_type():
        return T.bf16 if is_bf16_out else T.f16

    # GEMM epilogue toggle: optional LDS CShuffle + vectorized stores.
    # Default: off (current measured cases show no benefit).
    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = f"mfma_preshuffle_{lds_stage}stages_a{a_dtype}_b{b_dtype}_out{out_dtype}_{epilog_tag}".replace("-", "_")

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # LDS scratch:
            # - A tiles: separate ping/pong buffers for no-alias guarantee
            # - optional CShuffle output tile (fp16): reuses pong buffer
            lds_tile_bytes = int(tile_m) * int(lds_stride_bytes) // int(a_elem_vec_pack)
            lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0
            buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // lds_stage)
            buffer_size_elems = buffer_size_bytes if elem_bytes == 1 else (buffer_size_bytes // 2)
            _state["lds_a_pong"] = allocator_pong.allocate_array(_a_elem_type(), buffer_size_elems)
            _state["lds_a_ping"] = allocator_ping.allocate_array(_a_elem_type(), buffer_size_elems)
            allocator_pong.finalize()
            allocator_ping.finalize()

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, _a_elem_type()),
            arg_b: lambda: memref(DYN, _b_elem_type()),
            arg_scale_a: lambda: memref(DYN, _scale_elem_type()),
            arg_scale_b: lambda: memref(DYN, _scale_elem_type()),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            # ---- Types ----
            # NOTE: Some environments have multiple `flydsl` builds on PYTHONPATH.
            # Use explicit MLIR Values (not Python ints / wrapper objects) for ROCDL ops.
            acc_init = arith.unwrap(arith.constant_vector(0.0, T.f32x4))

            # Layouts
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            # A uses dword indexing (buffer-load dwordx4). Convert element index -> dword index:
            #   dword_index = (elem_index * elem_bytes) / 4
            if (int(elem_bytes) == 2):
                c_k_div4bytes = (c_k * 2) / 4
            else:
                c_k_div4bytes = c_k / 4 / a_elem_vec_pack
            layout_a_div4 = flir.make_layout((c_m, c_k_div4bytes), stride=(c_k_div4bytes, 1))

            c_k_b = c_k // b_elem_vec_pack
            # B preshuffle layout (shared with MoE kernels).
            kpack_bytes = 16
            layout_b = make_preshuffle_b_layout(
                flir, arith, c_n=c_n, c_k=c_k_b, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes
            ).layout_b

            layout_a_scale = make_preshuffle_scale_layout(
                flir, arith, c_mn=c_m, c_k=c_k,
            )
            layout_b_scale = make_preshuffle_scale_layout(
                flir, arith, c_mn=c_n, c_k=c_k,
            )

            # LDS layout is element-indexed, but XOR16 swizzle is byte-based.
            # Represent LDS as (tile_m, tile_k) in elements and scale swizzle math by elem_bytes.
            shape_lds = flir.make_shape(tile_m, tile_k // a_elem_vec_pack)
            stride_lds = flir.make_stride(tile_k // a_elem_vec_pack, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            # CK-style XOR16 swizzle parameter (const).
            # For FP4, LDS K dimension is tile_k_bytes // a_elem_vec_pack (128 bytes)
            # k_blocks16 must match actual LDS size to avoid swizzle address overflow
            lds_k_bytes = tile_k_bytes // a_elem_vec_pack
            k_blocks16 = arith.index(lds_k_bytes // 16)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr_pong = allocator_pong.get_base()
            base_ptr_ping = allocator_ping.get_base()
            lds_a_pong_ptr = _state["lds_a_pong"](base_ptr_pong)
            lds_a_ping_ptr = _state["lds_a_ping"](base_ptr_ping)
            lds_tile_shape = (tile_m * tile_k // a_elem_vec_pack,)
            lds_a_pong = SmemPtr(base_ptr_pong, lds_a_pong_ptr.byte_offset, _a_elem_type(), shape=lds_tile_shape).get()
            lds_a_ping = SmemPtr(base_ptr_ping, lds_a_ping_ptr.byte_offset, _a_elem_type(), shape=lds_tile_shape).get()
            lds_out = (
                SmemPtr(base_ptr_pong, lds_a_pong_ptr.byte_offset, T.f16, shape=(tile_m * tile_n,)).get()
                if use_cshuffle_epilog
                else None
            )

            # Note: We assume N is aligned (no N-tail support in this kernel).
            a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)

            b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

            bx_m = bx * tile_m
            by_n = by * tile_n

            # (thread_id.x) -> (wave_id, lane_id) via FLIR.
            layout_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            coord_wave_lane = flir.idx2crd(tx, layout_wave_lane)
            wave_id = flir.get(coord_wave_lane, 0)
            lane_id = flir.get(coord_wave_lane, 1)

            # lane_id -> (lane_div_16, lane_mod_16) via FLIR.
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            coord_lane16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_lane16, 0)
            lane_mod_16 = flir.get(coord_lane16, 1)

            row_a_lds = lane_mod_16
            # Per-`k1` (KLane) base offset along K inside a 64B K0 block.
            #
            # CK preshuffle uses KPackBytes=16 across dtypes, but KPackElems differs:
            # - fp8/int8: 16 elems (1B)
            # - fp16/bf16: 8 elems (2B)
            #
            # We express `col_offset_base` in *elements*.
            kpack_elems = 16 if elem_bytes == 1 else 8
            col_offset_base = lane_div_16 * arith.constant(int(kpack_elems), index=True)
            # `col_offset_base` is in element units (multiples of 16). We do LDS swizzle/math
            # in bytes, so scale by element size for fp16/bf16.
            col_offset_base_bytes = (
                col_offset_base
                if elem_bytes == 1
                else (col_offset_base * arith.constant(int(elem_bytes), index=True))
            )

            m_repeat = tile_m // 16
            # K stepping is byte-addressed: one "micro-tile step" is 64 bytes.
            k_unroll = tile_k_bytes // 128

            # --- Dynamic tiling along N (4 waves) ---
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            n_tile_base = wave_id * c_n_per_wave

            # fp4 pack
            k_unroll_packed  = k_unroll // pack_K
            m_repeat_packed  = m_repeat // pack_M
            num_acc_n_packed = num_acc_n // pack_N

            # Decompose global_n -> (n_blk, n_intra) once per ni.
            c_n0 = c_n / 16
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            n_intra_list = []
            n_blk_list = []
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)
                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                coord_n = flir.idx2crd(global_n, layout_n_blk_intra)
                n_blk_list.append(flir.get(coord_n, 0))
                n_intra_list.append(flir.get(coord_n, 1))

            # For FP8/INT8 we can load one 16B pack and extract both 8B halves (K64 bytes).
            # For INT4 (packed), reuse the existing K32 loader twice (2x4B loads + unpack).
            atom_b_g2r16 = flir.make_copy_atom(_b_elem_type(), vector_size=16)
            c64_b = 64
            c0_idx = 0

            def load_b_packs_k64(base_k, ku: int, ni: int):
                base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
                k0_base = base_k_bytes / c64_b
                k0 = k0_base + ku
                k1 = lane_div_16
                coord_pack = flir.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], c0_idx)
                idx_pack = flir.crd2idx(coord_pack, layout_b)
                vec_elems = 16 
                b_view = flir.TensorView(
                    arg_b,
                    (vec_elems,),
                    strides=(1,),
                    base_indices=(idx_pack,),
                    element_type=_b_elem_type(),
                )
                b16 = flir.copy(
                    flir.make_copy_atom(_b_elem_type(), vector_size=vec_elems),
                    b_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=(b_rsrc if elem_bytes == 1 else None),
                    src_buffer_offset_in_bytes=(elem_bytes == 1),
                )
                # Split 16B pack into two 8B halves.
                b_i64x2 = vector.bitcast(T.i64x2, b16)
                b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                return b0_i64, b1_i64

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

            def load_scale(arg_scale, rsrc, layout, ku, mni):
                k_lane = lane_div_16
                n_lane = lane_mod_16
                coord_pack = flir.make_coord(mni, ku, k_lane, n_lane)
                idx_pack = flir.crd2idx(coord_pack, layout)
                vec_elems = 1
                scale_view = flir.TensorView(
                    arg_scale,
                    (1,),
                    strides=(1,),
                    base_indices=(idx_pack,),
                    element_type=_scale_elem_type(),
                )
                scale = flir.copy(
                    flir.make_copy_atom(_scale_elem_type(), vector_size=1),
                    scale_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=rsrc,
                    src_buffer_offset_in_bytes=False,
                )
                # Split 16B pack into two 8B halves.
                return scale

            def load_b_scale_tile(base_k):
                b_scale_tile = []
                for ku in range_constexpr(k_unroll_packed):
                    for ni in range_constexpr(num_acc_n_packed):
                        scale = load_scale(
                            arg_scale_b,
                            scale_b_rsrc,
                            layout_b_scale,
                            ku + base_k,
                            ni + (by_n + n_tile_base) // pack_N // 16,
                        )
                        b_scale_tile.append(scale)
                return b_scale_tile

            def load_a_scale_tile(base_k):
                a_scale_tile = []
                for ku in range_constexpr(k_unroll_packed):
                    for mi in range_constexpr(m_repeat_packed):
                        scale = load_scale(
                            arg_scale_a,
                            scale_a_rsrc,
                            layout_a_scale,
                            ku + base_k,
                            mi + bx_m // pack_M // 16,
                        )
                        a_scale_tile.append(scale)
                return a_scale_tile

            def prefetch_ab_scale_tile(base_k):
                return [load_a_scale_tile(base_k), load_b_scale_tile(base_k)]

            def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
                # Swizzle in bytes, then convert to element offset for memref indexing.
                col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes / 2)
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                return vector.load_op(_a_vec16_type(), lds_buffer, [idx_a16])

            # --- A LDS load helper for K64-bytes (load 16B once, extract 2x i64 halves) ---
            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
                a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

                return a0_i64, a1_i64

            # --- A load/store (16B chunks), XOR16 swizzle ---
            num_a_loads = bytes_per_thread_a // a_load_bytes
            num_a_async_loads = bytes_per_thread_a // a_async_load_bytes
            wave_size = 64
            # A tile mapping in dwords along K:
            if elem_bytes == 2:
                tile_k_dwords = (tile_k * 2) // 4
            else:
                tile_k_dwords = tile_k // 4 // a_elem_vec_pack
            layout_a_tile_div4 = flir.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            # Also create byte-indexed layout for DMA path
            c_k_packed = c_k // a_elem_vec_pack
            layout_a_bytes = flir.make_layout((c_m, c_k_packed), stride=(c_k_packed, 1))
            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4
            tx_i32_async_base = tx * a_async_load_dword
            atom_a_g2r16 = flir.make_copy_atom(_a_elem_type(), vector_size=(16 if elem_bytes == 1 else 8))

            def load_a_16(idx_elem):
                return buffer_copy_gmem16_dwordx4(
                    flir,
                    arg=arg_a,
                    elem_type=_a_elem_type(),
                    idx_i32=idx_elem,
                    atom_g2r16=atom_a_g2r16,
                    rsrc=a_rsrc,
                    vec_elems=(16 if elem_bytes == 1 else 8),
                )

            def a_tile_chunk_coord_i32(i: int, tx_base=None, chunk_i32: int = 4):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_base if tx_base is not None else tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_a_tile_div4,
                    chunk_i32=chunk_i32,
                )

            def load_a_tile(base_k_div4):
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 + col_a_local_i32)
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    # `idx_i32` is a dword offset. For 2B element types (fp16/bf16),
                    # convert to element offset so the generic `vector.load` path reads
                    # the right address (FLIR only specializes buffer_load_dwordx4 for 1B types).
                    idx_elem = (
                        idx_i32
                        if elem_bytes == 1
                        else (idx_i32 * arith.constant(2, index=True))
                    )
                    a_16B = load_a_16(idx_elem)
                    
                    parts.append(vector.bitcast(T.i32x4, a_16B))
                return parts

            def store_a_tile_to_lds(vec_a_parts, lds_buffer):
                _lds_base_zero = arith.constant(0, index=True)
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    lds_store_16b_xor16(
                        flir,
                        arith,
                        vector,
                        lds_memref=lds_buffer,
                        vec16_ty=_a_vec16_type(),
                        elem_type=_a_elem_type(),
                        atom_s16=atom_a_g2r16,
                        layout_lds=layout_lds,
                        row_local=row_a_local,
                        col_local_i32=col_a_local_i32,
                        tx_c4=c4,
                        k_blocks16=k_blocks16,
                        lds_base=_lds_base_zero,
                        vec_part_i32x4=vec_a_parts[i],
                        elem_bytes=elem_bytes,
                    )

            # DMA async version: direct global-to-LDS transfer (no register intermediate)
            def dma_a_tile_to_lds(base_k_div4, lds_buffer):
                from _mlir.dialects import llvm, memref as memref_dialect

                dma_bytes = a_async_load_bytes
                bytes_per_dword = 4
                chunk_i32 = dma_bytes // bytes_per_dword

                for i in range_constexpr(num_a_async_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i, tx_base=tx_i32_async_base, chunk_i32=chunk_i32)
                    col_a_local_sw = flir.swizzle_xor16(row_a_local, col_a_local_i32 * c4, k_blocks16)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 * c4 + col_a_local_sw)
                    global_offset = arith.index_cast(T.i32, flir.crd2idx(coord_a_g, layout_a_bytes))

                    if i == 0:
                        lds_addr = memref_dialect.extract_aligned_pointer_as_index(lds_buffer) + wave_id * wave_size * dma_bytes
                        lds_ptr_i64_lane0 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                    else:
                        lds_ptr_i64_lane0 += total_threads * dma_bytes
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_ptr_i64_lane0, address_space=3)

                    size_i32 = arith.constant(dma_bytes, type=T.i32)
                    soffset = arith.constant(0, type=T.i32)
                    offset_imm = arith.constant(0, type=T.i32)
                    aux = arith.constant(1, type=T.i32)

                    rocdl.raw_ptr_buffer_load_lds(
                        a_rsrc,
                        lds_ptr,
                        arith.unwrap(size_i32),
                        arith.unwrap(global_offset),
                        arith.unwrap(soffset),
                        arith.unwrap(offset_imm),
                        arith.unwrap(aux),
                    )

            def prefetch_a_to_lds(base_k, lds_buffer):
                """Load A tile from global memory to LDS via DMA (async, no registers)."""
                base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
                base_k_div4 = base_k_bytes / 4
                dma_a_tile_to_lds(base_k_div4 // a_elem_vec_pack, lds_buffer)

            def prefetch_ab_tile(base_k):
                base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
                base_k_div4 = base_k_bytes / 4
                a_regs = load_a_tile(base_k_div4 // a_elem_vec_pack)
                b_regs = load_b_tile(base_k // 2)
                return a_regs, b_regs

            def compute_tile(
                    accs_in,
                    b_tile_in,
                    lds_buffer,
                    *,
                    a0_prefetch=None,
                    a_scale=None,
                    b_scale=None,
                    a_gmem_prefetch_fn=None,
                    lds_buffer_next=None,
                    # On-demand B loading: when b_base_k is set, load B per k-step
                    # inside the loop instead of using pre-loaded b_tile_in.
                    # This reduces peak VGPR count for large tiles (256x256).
                    b_base_k=None,
                ):
                current_accs_list = list(accs_in)

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

                # CK-aligned loop order: M-outer, K-middle, N-inner.
                # SwitchM: mid-compute A LDS buffer swap (CK pattern).
                # Only useful when m_repeat_packed > 2 (tile_m >= 128).
                _enable_switch_m = False  # TEMP: disable SwitchM for debugging
                switch_m = m_repeat_packed - 2 if _enable_switch_m else -1

                _active_lds = [lds_buffer]

                # For on-demand B: load B ONCE per tile (outside M loop) to minimize VMEM.
                # The compiler will keep B in VGPRs across M iterations.
                if b_base_k is not None:
                    _od_b_tile = []
                    for _ku in range_constexpr(k_unroll):
                        _packs0 = []
                        _packs1 = []
                        for _ni in range_constexpr(num_acc_n):
                            _b0, _b1 = load_b_packs_k64(b_base_k, _ku, _ni)
                            _packs0.append(_b0)
                            _packs1.append(_b1)
                        _od_b_tile.append((_packs0, _packs1))

                for mi in range_constexpr(m_repeat_packed):

                    # CK SwitchM: at this point, kick off next A GMEM->LDS
                    if _enable_switch_m and (mi == switch_m):
                        if a_gmem_prefetch_fn is not None:
                            a_gmem_prefetch_fn()
                        if lds_buffer_next is not None:
                            gpu.barrier()
                            _active_lds[0] = lds_buffer_next

                    for ku128 in range_constexpr(k_unroll_packed):

                        a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                        a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])

                        for ni in range_constexpr(num_acc_n_packed):
                            b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                            b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])

                            for ikxdl in range_constexpr(pack_K):
                                k_idx = ku128 * pack_K + ikxdl
                                if b_base_k is not None:
                                    b_packs0, b_packs1 = _od_b_tile[k_idx]
                                else:
                                    b_packs0, b_packs1 = b_tile_in[k_idx]

                                col_base = col_offset_base_bytes + (k_idx * 128) // a_elem_vec_pack

                                # Batch A ds_reads: pre-read ALL pack_M rows from LDS
                                # before the MFMA loop. This lets the compiler issue
                                # multiple ds_reads with a single lgkmcnt wait.
                                a128_batch = []
                                for imxdl in range_constexpr(pack_M):
                                    mi_idx = mi * pack_M + imxdl
                                    mi_val = arith.constant(mi_idx * 16, index=True)
                                    curr_row_a_lds = row_a_lds + mi_val

                                    if (a0_prefetch is not None) and (k_idx == 0) and (mi_idx == 0):
                                        a0, a1 = a0_prefetch
                                    else:
                                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, _active_lds[0])

                                    if is_fp8_a:
                                        col_base1 = col_base + 64
                                        a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, _active_lds[0])
                                        a128_batch.append(pack_i64x4_to_i32x8(a0, a1, a2, a3))
                                    else:
                                        a128_batch.append(pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64))

                                # MFMA loop: consume pre-read A from VGPRs
                                for imxdl in range_constexpr(pack_M):
                                    mi_idx = mi * pack_M + imxdl
                                    a128 = a128_batch[imxdl]

                                    for inxdl in range_constexpr(pack_N):
                                        ni_idx = ni * pack_N + inxdl
                                        b0 = b_packs0[ni_idx]
                                        b1 = b_packs1[ni_idx]
                                        b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)

                                        acc_idx = mi_idx * num_acc_n + ni_idx
                                        if not _use_scheduler:
                                            rocdl.sched_barrier(0)
                                        current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                            mfma_res_ty,
                                            [
                                                a128,
                                                b128,
                                                current_accs_list[acc_idx],
                                                cbsz,
                                                blgp,
                                                ikxdl * pack_M + imxdl,
                                                a_scale_val,
                                                ikxdl * pack_N + inxdl,
                                                b_scale_val,
                                            ],
                                        )

                return current_accs_list, None

            vec1_f16 = ir.VectorType.get([1], ir.F16Type.get())
            vec2_f16 = ir.VectorType.get([2], ir.F16Type.get())
            vec1_i16 = ir.VectorType.get([1], ir.IntegerType.get_signless(16))
            vec2_i16 = ir.VectorType.get([2], ir.IntegerType.get_signless(16))
            vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))
            vec4_i32 = ir.VectorType.get([4], ir.IntegerType.get_signless(32))

            def store_output(final_accs):
                if use_cshuffle_epilog:
                    if lds_out is None:
                        raise RuntimeError(
                            "use_cshuffle_epilog=True but lds_out is not allocated/aliased."
                        )
                    # We reuse the A LDS allocation as `lds_out` for the cshuffle epilogue.
                    # Add a block-wide barrier before starting to write into LDS to avoid
                    # racing with the tail of the mainloop's LDS reads (different waves can
                    # reach the epilogue at slightly different times).
                    gpu.barrier()

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
                        # Store packed half2 to LDS as i32:
                        # - Each lane computes one f16 (for its lane_mod_16 column)
                        # - Use ds_bpermute to grab the neighbor lane's f16 bits and pack (even, odd)
                        # - Store to the even column address / 2 in the i32 alias view
                        c0_i32 = arith.constant(0, type=T.i32)
                        c1_i32 = arith.constant(1, type=T.i32)
                        cFE_i32 = arith.constant(0xFFFFFFFE, type=T.i32)
                        c2_i32 = arith.constant(2, type=T.i32)

                        lane_id_i32 = arith.index_cast(T.i32, lane_id)
                        lane_lsb = arith.andi(lane_id_i32, c1_i32)
                        is_odd = lane_lsb != c0_i32
                        nbr_lane = arith.xori(lane_id_i32, c1_i32)
                        nbr_lane_bytes = arith.shli(nbr_lane, c2_i32)  # lane_id * 4 (bytes)

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            acc = final_accs[acc_idx]
                            val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                            v16 = arith.trunc_f(T.f16, val)

                            # v16 (f16) -> bits in i32 low16
                            v1_f16 = vector.from_elements(vec1_f16, [v16])
                            v1_i16 = vector.bitcast(vec1_i16, v1_f16)
                            v16_i16 = vector.extract(
                                v1_i16, static_position=[0], dynamic_position=[]
                            )
                            # Zero-extend i16 bits to i32:
                            # Build a 2xi16 vector (low16=v16 bits, high16=0) then bitcast to 1xi32.
                            z16 = arith.constant(0, type=T.i16)
                            v2_i16 = vector.from_elements(vec2_i16, [v16_i16, z16])
                            v16_i32 = vector.extract(
                                vector.bitcast(vec1_i32, v2_i16),
                                static_position=[0],
                                dynamic_position=[],
                            )

                            # Neighbor's bits (per-lane): ds_bpermute uses a byte index.
                            nbr_i32 = rocdl.ds_bpermute(
                                T.i32,
                                arith.unwrap(nbr_lane_bytes),
                                arith.unwrap(v16_i32),
                            )

                            # Convert neighbor bits back to f16 so we can store vec2<f16>.
                            nbr_v1_i32 = vector.from_elements(vec1_i32, [nbr_i32])
                            nbr_v2_i16 = vector.bitcast(vec2_i16, nbr_v1_i32)
                            nbr_i16 = vector.extract(
                                nbr_v2_i16, static_position=[0], dynamic_position=[]
                            )
                            nbr_v1_i16 = vector.from_elements(vec1_i16, [nbr_i16])
                            nbr_v1_f16 = vector.bitcast(vec1_f16, nbr_v1_i16)
                            nbr_f16 = vector.extract(
                                nbr_v1_f16, static_position=[0], dynamic_position=[]
                            )

                            even_f16 = arith.select(is_odd, nbr_f16, v16)
                            odd_f16 = arith.select(is_odd, v16, nbr_f16)

                            # Store [even, odd] as a single 32-bit LDS write (2xf16).
                            col_local_i32 = arith.index_cast(T.i32, col_local)
                            col_even_i32 = arith.andi(col_local_i32, cFE_i32)
                            col_even = arith.index_cast(T.index, col_even_i32)

                            lds_idx = row_base_lds + col_even
                            v2 = vector.from_elements(vec2_f16, [even_f16, odd_f16])
                            vector.store(v2, lds_out, [lds_idx], alignment=2)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        # Store vector<EVecxf16> to C at (row, col_g0).
                        #
                        # IMPORTANT:
                        # RawPtrBufferStoreOp offsets are in BYTES. `buffer_ops.buffer_store()`
                        # will scale by element bytes based on the *data type*. For f16 vectors,
                        # some backends/paths can be fragile. We explicitly bitcast to i32
                        # and pass a byte offset to keep the store well-defined.
                        idx_out = flir.crd2idx(flir.make_coord(row, col_g0), layout_c)  # f16 element offset
                        byte_off = idx_out * arith.constant(2, index=True)  # bytes

                        if e_vec == 4:
                            frag_i32x2 = vector.bitcast(T.vec(2, T.i32), frag)
                            buffer_ops.buffer_store(
                                frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True
                            )
                        else:
                            # e_vec == 2: pack 2xf16 -> 1xi32
                            frag_i32x1 = vector.bitcast(T.vec(1, T.i32), frag)
                            frag_i32 = vector.extract(
                                frag_i32x1, static_position=[0], dynamic_position=[]
                            )
                            buffer_ops.buffer_store(
                                frag_i32, c_rsrc, byte_off, offset_is_bytes=True
                            )

                    # Prefer 16B stores when possible:
                    # - EVec=4 => 4xf16 (8B) per store (and matches tile_n multiples of 128)
                    # - EVec=2 => 2xf16 (4B) per store (tile_n multiples of 64)
                    e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
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
                        write_row_to_lds=write_row_to_lds,
                        store_pair=store_pair,
                    )
                    return

                def body_row(*, mi: int, ii: int, row_in_tile, row):
                    col_base = by_n + n_tile_base + lane_mod_16
                    idx_base = flir.crd2idx(flir.make_coord(row, col_base), layout_c)
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                        # if is_int8:
                        #     val = arith.sitofp(T.f32, val)

                        # val_s = (val * s_a) * s_b_vals[ni]
                        if is_bf16_out:
                            val_out = arith.trunc_f(T.bf16, val)
                        else:
                            val_out = arith.trunc_f(T.f16, val)
                        idx_out = idx_base + arith.constant(ni * 16, index=True)
                        buffer_ops.buffer_store(val_out, c_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=False,
                    arith=arith,
                    range_constexpr=range_constexpr,
                    m_repeat=m_repeat,
                    lane_div_16=lane_div_16,
                    bx_m=bx_m,
                    body_row=body_row,
                )

            # ---------------- Scheduling hints (match CK-style) ----------------
            # These sched_group_barrier hints help the backend interleave VMEM/DS/MFMA
            # similarly to CK's tuned pipelines.
            rocdl.sched_barrier(0)

            # Scheduling is beneficial for compute-bound shapes (larger tiles with many MFMAs).
            # For small tile_m (bandwidth-bound), the compiler's default scheduling is near-optimal.
            _use_scheduler = (tile_m >= 64) and not os.environ.get('FLYDSL_NO_SCHEDULER')

            def hot_loop_scheduler():
                """CK-aligned two-stage intrawave scheduler.

                Directly mirrors CK's gemm_mxfp4_pipeline_ag_bg_cr_v3.hpp scheduling:

                Stage 1: Interleave VMEM loads (B on-demand + A DMA for next tile)
                         with MFMAs.  Buffer_loads have ~300 cycle latency; scheduling
                         them early behind MFMAs hides the latency.

                Stage 2: Interleave DS reads (A from LDS) with MFMAs.
                         ds_read_b128 has ~20 cycle latency; interleaving 1 ds_read
                         per MFMA (16 cycles) lets reads complete by next use.

                Total: all MFMAs, ds_reads, and vmem loads are accounted for
                in the scheduling groups, preventing the compiler from inserting
                conservative s_waitcnt lgkmcnt(0) / vmcnt(0).
                """
                if not _use_scheduler:
                    return

                num_mfma = (k_unroll_packed * m_repeat_packed * num_acc_n_packed
                            * pack_K * pack_M * pack_N)
                num_dsrd = (k_unroll_packed * m_repeat_packed * num_acc_n_packed
                            * pack_K * pack_M)
                # On-demand B loads per compute_tile
                num_vmem_b = (k_unroll * num_acc_n) if _ondemand_b else 0
                # DMA loads for next tile's A (in the scheduling region)
                num_vmem_dma = num_a_async_loads if _use_dma else 0
                num_vmem = num_vmem_b + num_vmem_dma

                if num_mfma == 0:
                    return

                # ds_read rate: 1 ds_read per MFMA (CK formula for 16x16 MFMA)
                ds_read_mfma_rate = 1

                # Stage 2 MFMA count: one MFMA per ds_read group
                num_dsread_groups = (num_dsrd + ds_read_mfma_rate - 1) // ds_read_mfma_rate if num_dsrd > 0 else 0
                mfma_stage2 = num_dsread_groups

                # Stage 1 MFMA count: remaining MFMAs
                mfma_stage1 = num_mfma - mfma_stage2
                if mfma_stage1 < 0:
                    mfma_stage1 = 0
                    mfma_stage2 = num_mfma

                # Stage 1: VMEM + MFMA interleaving
                if num_vmem > 0 and mfma_stage1 > 0:
                    mfma_per_vmem = max(1, mfma_stage1 // num_vmem)
                    for _vi in range_constexpr(num_vmem):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_per_vmem)
                    # Remaining stage1 MFMAs
                    leftover = mfma_stage1 - num_vmem * mfma_per_vmem
                    if leftover > 0:
                        rocdl.sched_mfma(leftover)
                elif mfma_stage1 > 0:
                    rocdl.sched_mfma(mfma_stage1)

                # Stage 2: DS read + MFMA interleaving
                remaining_dsrd = num_dsrd
                for _di in range_constexpr(num_dsread_groups):
                    rate = min(ds_read_mfma_rate, remaining_dsrd)
                    rocdl.sched_dsrd(rate)
                    rocdl.sched_mfma(1)
                    remaining_dsrd -= rate

                rocdl.sched_barrier(0)

            # ---------------- Pipeline ----------------
            # Separate ping/pong LDS buffers for no-alias optimization.
            if lds_stage == 2:
                # ---------------- Ping-pong pipeline (2 LDS buffers) ----------------
                # Cross-tile A0 LDS prefetch (default-on):
                # issue the first A-pack DS read for the next tile *between* barriers,
                # so it can overlap with the VMEM prefetch of the following tile.

                def prefetch_a0_pack(lds_buffer):
                    # (mi=0, ku=0): prefetch both K32 halves (K64) for the first A-pack.
                    return lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)

                # Helper: load A to LDS.
                # Use async DMA (buffer_load_lds) for large tiles to avoid VGPR intermediaries.
                # CK uses buffer_load_dwordx4 offen lds for A, saving ~32 VGPRs.
                _use_dma = use_async_copy or (tile_m >= 128)  # DMA for A: GMEM->LDS directly
                def load_a_to_lds(base_k, lds_buffer):
                    if _use_dma:
                        prefetch_a_to_lds(base_k, lds_buffer)
                    else:
                        base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
                        base_k_div4 = base_k_bytes / 4
                        a_regs = load_a_tile(base_k_div4 // a_elem_vec_pack)
                        store_a_tile_to_lds(a_regs, lds_buffer)

                # On-demand B loading: for large tiles (256x256) where preloading all B
                # causes VGPR spills, load B per-K-step inside compute_tile instead.
                # This trades VMEM re-issue for lower VGPR count (no spills).
                _ondemand_b = (m_repeat * num_acc_n > 32)  # on-demand B for large tiles

                # Prologue: tile-0
                k0 = arith.constant(0, index=True)
                if _ondemand_b:
                    b_tile_pong = []  # not used, B loaded on-demand
                else:
                    b_tile_pong = load_b_tile(k0 // 2)
                a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(k0 // 2)

                load_a_to_lds(k0, lds_a_pong)
                gpu.barrier()
                accs = [acc_init] * (num_acc_n * m_repeat)
                
                c_k_main = c_k - tile_k

                # Prefetch A0 for the first compute tile (overlap with the next VMEM prefetch).
                a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

                # For large tiles (m_repeat_packed > 2), A load is embedded at SwitchM.
                _embed_a_prefetch = False  # TEMP: disable for debugging

                num_tiles = K // tile_k

                # Helpers for B loading: on-demand or pre-loaded
                def _load_b(base_k):
                    return [] if _ondemand_b else load_b_tile(base_k)
                def _bk(base_k):
                    return base_k if _ondemand_b else None

                if (num_tiles % 2) == 1:
                    for k_iv in range(0, c_k_main, tile_k * 2):
                        next_k1 = k_iv + tile_k
                        b_tile_ping = _load_b(next_k1 // 2)
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(next_k1 // 256)
                        if not _embed_a_prefetch:
                            load_a_to_lds(next_k1, lds_a_ping)
                        _nk1, _lp = next_k1, lds_a_ping
                
                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_a_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a_scale=a_scale_pong, b_scale=b_scale_pong,
                            a_gmem_prefetch_fn=(lambda: load_a_to_lds(_nk1, _lp)) if _embed_a_prefetch else None,
                            lds_buffer_next=lds_a_ping if _embed_a_prefetch else None,
                            b_base_k=_bk(k_iv // 2),
                        )
                        a0_prefetch_pong = None
                
                        hot_loop_scheduler()
                        gpu.barrier()
                
                        a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)
                
                        next_k2 = k_iv + tile_k * 2
                        b_tile_pong = _load_b(next_k2 // 2)
                        a_scale_pong, b_scale_pong= prefetch_ab_scale_tile(next_k2 // 256)
                        if not _embed_a_prefetch:
                            load_a_to_lds(next_k2, lds_a_pong)
                        _nk2, _lq = next_k2, lds_a_pong
                
                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_a_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a_scale=a_scale_ping, b_scale=b_scale_ping,
                            a_gmem_prefetch_fn=(lambda: load_a_to_lds(_nk2, _lq)) if _embed_a_prefetch else None,
                            lds_buffer_next=lds_a_pong if _embed_a_prefetch else None,
                            b_base_k=_bk(next_k1 // 2),
                        )
                        a0_prefetch_ping = None
                
                        hot_loop_scheduler()
                        gpu.barrier()
                
                        a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)
                
                    # Final tile (no next prefetch)
                    _last_bk = _bk((K - tile_k) // 2) if (num_tiles % 2) == 1 else _bk(0)
                    final_accs, _ = compute_tile(
                        accs,
                        b_tile_pong,
                        lds_a_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a_scale=a_scale_pong,
                        b_scale=b_scale_pong,
                        b_base_k=_bk((K - tile_k) // 2),
                    )
                else:
                    c_k_stop = c_k - (tile_k * 3)
                    for k_iv in range(0, c_k_stop, tile_k * 2):
                        next_k1 = k_iv + tile_k
                        b_tile_ping = _load_b(next_k1 // 2)
                        a_scale_ping, b_scale_ping= prefetch_ab_scale_tile(next_k1 // 256)
                        if not _embed_a_prefetch:
                            load_a_to_lds(next_k1, lds_a_ping)
                        _nk1, _lp = next_k1, lds_a_ping
                
                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_a_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a_scale=a_scale_pong, b_scale=b_scale_pong,
                            a_gmem_prefetch_fn=(lambda: load_a_to_lds(_nk1, _lp)) if _embed_a_prefetch else None,
                            lds_buffer_next=lds_a_ping if _embed_a_prefetch else None,
                            b_base_k=_bk(k_iv // 2),
                        )
                        a0_prefetch_pong = None
                
                        hot_loop_scheduler()
                        gpu.barrier()
                
                        a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)
                
                        next_k2 = k_iv + tile_k * 2
                        b_tile_pong = _load_b(next_k2 // 2)
                        a_scale_pong, b_scale_pong= prefetch_ab_scale_tile(next_k2 // 256)
                        if not _embed_a_prefetch:
                            load_a_to_lds(next_k2, lds_a_pong)
                        _nk2, _lq = next_k2, lds_a_pong
                
                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_a_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a_scale=a_scale_ping, b_scale=b_scale_ping,
                            a_gmem_prefetch_fn=(lambda: load_a_to_lds(_nk2, _lq)) if _embed_a_prefetch else None,
                            lds_buffer_next=lds_a_pong if _embed_a_prefetch else None,
                            b_base_k=_bk(next_k1 // 2),
                        )
                        a0_prefetch_ping = None
                
                        hot_loop_scheduler()
                        gpu.barrier()
                
                        a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)
                
                    last_k = c_k - tile_k
                    b_tile_ping = _load_b(last_k // 2)
                    a_scale_ping, b_scale_ping= prefetch_ab_scale_tile(last_k // 256)
                    if not _embed_a_prefetch:
                        load_a_to_lds(last_k, lds_a_ping)
                    _lk, _lp2 = last_k, lds_a_ping
                
                    accs, _ = compute_tile(
                        accs, b_tile_pong, lds_a_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a_scale=a_scale_pong, b_scale=b_scale_pong,
                        a_gmem_prefetch_fn=(lambda: load_a_to_lds(_lk, _lp2)) if _embed_a_prefetch else None,
                        lds_buffer_next=lds_a_ping if _embed_a_prefetch else None,
                        b_base_k=_bk((K - 2*tile_k) // 2),
                    )

                    a0_prefetch_pong = None
                
                    hot_loop_scheduler()
                    gpu.barrier()
                
                    a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)
                
                    final_accs, _ = compute_tile(
                        accs,
                        b_tile_ping,
                        lds_a_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a_scale=a_scale_ping,
                        b_scale=b_scale_ping,
                        b_base_k=_bk((K - tile_k) // 2),
                    )

                store_output(final_accs)
            # else:
            #     # CK-like bpreshuffle v1 spirit:
            #     # - Intrawave schedule
            #     # - Global prefetch 2 (regs double-buffer)
            #     # - Local shared memory buffer 1 (single LDS tile for A)
            #     # Prologue: tile-0
            #     k0 = arith.constant(0, index=True)
            #     a_regs0, b_tile0 = prefetch_ab_tile(k0)
            #     store_a_tile_to_lds(a_regs0, lds_base0)
            #     gpu.barrier()
            #     accs = [acc_init] * (num_acc_n * m_repeat)
            #
            #     lds_base = lds_base0
            #     b_tile_cur = b_tile0
            #
            #     # For each tile except last: prefetch next tile, compute current, then overwrite LDS.
            #     for k_base in range(0, c_k - tile_k, tile_k):
            #         next_k = k_base + tile_k
            #         a_next, b_next = prefetch_ab_tile(next_k)
            #         accs, _ = compute_tile(accs, b_tile_cur, lds_base)
            #         # Single LDS buffer: ensure *all* waves are done reading A from LDS
            #         # before any wave overwrites it with the next tile.
            #         gpu.barrier()
            #         store_a_tile_to_lds(a_next, lds_base)
            #         # hot_loop_scheduler()
            #         gpu.barrier()
            #         b_tile_cur = b_next
            #
            #     final_accs, scales = compute_tile(
            #         accs, b_tile_cur, lds_base, is_last_tile=True
            #     )
            #     store_output(final_accs, scales)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, _a_elem_type()),
            arg_b: lambda: memref(DYN, _b_elem_type()),
            arg_scale_a: lambda: memref(DYN, _scale_elem_type()),
            arg_scale_b: lambda: memref(DYN, _scale_elem_type()),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
            stream_ptr: lambda: T.i64,  # PyTorch stream pointer
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(256, index=True)
            tm = arith.constant(tile_m, index=True)
            tn = arith.constant(tile_n, index=True)
            one = arith.constant(1, index=True)
            gx = (c_m + tm - one) / tm
            gy = c_n / tn

            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "kernel_gemm"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_c,
                    arg_a,
                    arg_b,
                    arg_scale_a,
                    arg_scale_b,
                    c_m,
                    c_n,
                    c_k,
                ],
                async_dependencies=[stream_token],
            )

    m = _GEMM()
    return flydsl.compile(
        m,
        use_bare_ptr_memref_call_conv=False,
        use_bare_pointers_for_host=False,
        use_bare_pointers_for_kernels=False,
    )


__all__ = ["compile_mxfp4_preshuffle_gemm"]

