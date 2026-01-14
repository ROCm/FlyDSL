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
    load_b_pack_k32,
    tile_chunk_coord_i32,
)
from kernels.mfma_epilogues import mfma_epilog


def compile_preshuffle_gemm_a8(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    lds_stage: int = 2,
    # Epilogue options
    use_cshuffle_epilog: bool = False,
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
    if in_dtype not in ("fp8", "int8", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', or 'int4', got {in_dtype!r}")
    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4

    # K64 micro-step wrapper uses 2x K32 MFMA. Require tile_k divisible by 64.
    if (int(tile_k) % 64) != 0:
        raise ValueError(f"tile_k must be divisible by 64 (for K64 unroll), got tile_k={tile_k}")

    # INT8 must use a K32 MFMA so the micro-step matches the FP8 path (strict alignment).
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

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    # Default-on: cross-tile (tile_k) A0 LDS prefetch in the ping-pong pipeline (lds_stage=2).
    #
    # This issues the *first* A-pack LDS read for the next tile between barriers, to overlap
    # with the VMEM prefetch of the following tile.

    size_c = int(M) * int(N)
    size_a = int(M) * int(K)
    # B is packed int4 for W4A8: 2 values per byte.
    size_b = (int(N) * int(K)) // 2 if is_int4 else (int(N) * int(K))

    # Vector width calc (assume full tiles / no tail guards).
    total_threads = 256
    elems_a_per_tile = tile_m * tile_k
    if elems_a_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k must be divisible by {total_threads}: tile_m={tile_m}, tile_k={tile_k}"
        )
    elems_per_thread_a = elems_a_per_tile // total_threads
    bytes_per_thread_a = elems_per_thread_a  # 1B elems

    # Assume A loads are always 16B-aligned and use fixed dwordx4 (16B) buffer loads.
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )

    # CK-style LDS128: stride == tile_k, XOR16 swizzle.
    lds_stride = tile_k

    def _elem_type():
        return T.i8 if is_int8 else T.f8

    def _vec16_type():
        return T.i8x16 if is_int8 else T.f8x16

    # GEMM epilogue toggle: optional LDS CShuffle + vectorized stores.
    # Default: off (current measured cases show no benefit).
    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = f"mfma_preshuffle_{lds_stage}stages_{in_dtype}_{epilog_tag}".replace("-", "_")

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # LDS scratch:
            # - A tiles (fp8/int8): lds_stage * tile_m * lds_stride bytes
            # - optional CShuffle output tile (fp16): tile_m * tile_n * 2 bytes
            #
            # When CShuffle is enabled, we reuse the same LDS allocation by aliasing it as fp16
            # in the epilogue (the A LDS is dead after the mainloop).
            lds_a_bytes = int(lds_stage) * int(tile_m) * int(lds_stride)  # 1B elems
            lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0
            lds_total_bytes = max(lds_a_bytes, lds_out_bytes)
            _state["lds_a_decl"] = allocator.allocate_array(_elem_type(), lds_total_bytes)
            allocator.finalize()

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: memref(size_c, T.f16),
            arg_a: lambda: memref(size_a, _elem_type()),
            arg_b: lambda: memref(size_b, _elem_type()),
            arg_scale_a: lambda: memref(M, T.f32),
            arg_scale_b: lambda: memref(N, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            # ---- Types ----
            # NOTE: Some environments have multiple `flydsl` builds on PYTHONPATH.
            # Use explicit MLIR Values (not Python ints / wrapper objects) for ROCDL ops.
            acc_init = arith.unwrap(
                arith.constant_vector(0, T.i32x4)
                if is_int8
                else arith.constant_vector(0.0, T.f32x4)
            )

            # Layouts
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            # A is 1B/elem but we use dwordx4 (16B) buffer loads, so build a /4 layout.
            c_k_div4 = c_k / 4
            layout_a_div4 = flir.make_layout((c_m, c_k_div4), stride=(c_k_div4, 1))

            # B preshuffle layout (shared with MoE kernels).
            kpack_bytes = 8 if is_int4 else 16
            layout_b = make_preshuffle_b_layout(
                flir, arith, c_n=c_n, c_k=c_k, kpack_bytes=kpack_bytes
            ).layout_b

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            # CK-style XOR16 swizzle parameter (const).
            k_blocks16 = arith.index(tile_k // 16)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr = allocator.get_base()
            lds_a_ptr = _state["lds_a_decl"](base_ptr)
            lds_a = lds_a_ptr.get()
            lds_out = (
                SmemPtr(base_ptr, lds_a_ptr.byte_offset, T.f16, shape=(tile_m * tile_n,)).get()
                if use_cshuffle_epilog
                else None
            )

            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)

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
            # lane_div_16 * 16 via FLIR crd2idx((lane_div_16,0), layout=(4,16):stride=(16,1))
            col_offset_base = flir.crd2idx(flir.make_coord(lane_div_16, 0), layout_lane16)

            m_repeat = tile_m // 16
            k_unroll = tile_k // 64  # K64 micro-step (2x K32 MFMA)

            # --- Dynamic tiling along N (4 waves) ---
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            n_tile_base = wave_id * c_n_per_wave

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

            # --- B load logic (K64 micro-step), return i64 packs (two halves) ---
            # Shared loader supports:
            # - FP8/INT8: explicit 16B load (one full KPack) + extract 8B for this micro-step
            # - INT4 (W4A8): 4B load + 7-op unpack to 8B (no v_perm)

            def load_b_pack(base_k, ki_step, ni):
                return load_b_pack_k32(
                    buffer_ops,
                    flir,
                    arith,
                    vector,
                    arg_b=arg_b,
                    b_rsrc=b_rsrc,
                    layout_b=layout_b,
                    base_k=base_k,
                    ki_step=ki_step,
                    n_blk=n_blk_list[ni],
                    n_intra=n_intra_list[ni],
                    lane_div_16=lane_div_16,
                    elem_type=_elem_type(),
                    kpack_bytes=kpack_bytes,
                    unpack_int4=is_int4,
                )

            # For FP8/INT8 we can load one 16B pack and extract both 8B halves (K64).
            # For INT4 (packed), reuse the existing K32 loader twice (2x4B loads + unpack).
            atom_b_g2r16 = flir.make_copy_atom(_elem_type(), vector_size=16)
            c64_b = 64
            c0_idx = 0

            def load_b_packs_k64(base_k, ku: int, ni: int):
                if is_int4:
                    ki0 = (ku * 2) + 0
                    ki1 = (ku * 2) + 1
                    return load_b_pack(base_k, ki0, ni), load_b_pack(base_k, ki1, ni)

                # FP8/INT8: load 16 bytes (one full KPack) and return both i64 halves.
                k0_base = base_k / c64_b
                k0 = k0_base + ku
                k1 = lane_div_16
                coord_pack = flir.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], c0_idx)
                idx_pack_bytes = flir.crd2idx(coord_pack, layout_b)
                b_view = flir.TensorView(
                    arg_b,
                    (16,),
                    strides=(1,),
                    base_indices=(idx_pack_bytes,),
                    element_type=_elem_type(),
                )
                b16 = flir.copy(
                    atom_b_g2r16,
                    b_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=b_rsrc,
                    src_buffer_offset_in_bytes=True,
                )
                b_i64x2 = vector.bitcast(T.i64x2, b16)
                b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                return b0, b1

            def load_b_tile(base_k):
                # b_tile[ku] = (packs_half0[ni], packs_half1[ni])
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

            # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                col_base_swz = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                idx_a16 = idx_a16 + lds_base
                loaded_a16 = vector.load_op(_vec16_type(), lds_a, [idx_a16])
                a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0, a1

            # --- A load/store (16B chunks), XOR16 swizzle ---
            num_a_loads = bytes_per_thread_a // a_load_bytes
            layout_a_tile_div4 = flir.make_layout(
                (tile_m, tile_k // 4), stride=(tile_k // 4, 1)
            )
            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4
            atom_a_g2r16 = flir.make_copy_atom(_elem_type(), vector_size=16)

            def load_a_16(idx_i32):
                return buffer_copy_gmem16_dwordx4(
                    flir,
                    arg=arg_a,
                    elem_type=_elem_type(),
                    idx_i32=idx_i32,
                    atom_g2r16=atom_a_g2r16,
                    rsrc=a_rsrc,
                )

            def a_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_a_tile_div4,
                )

            def load_a_tile(base_k_div4):
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 + col_a_local_i32)
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    a_16B = load_a_16(idx_i32)
                    parts.append(vector.bitcast(T.i32x4, a_16B))
                return parts

            def store_a_tile_to_lds(vec_a_parts, lds_base):
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    lds_store_16b_xor16(
                        flir,
                        arith,
                        vector,
                        lds_memref=lds_a,
                        vec16_ty=_vec16_type(),
                        elem_type=_elem_type(),
                        atom_s16=atom_a_g2r16,
                        layout_lds=layout_lds,
                        row_local=row_a_local,
                        col_local_i32=col_a_local_i32,
                        tx_c4=c4,
                        k_blocks16=k_blocks16,
                        lds_base=lds_base,
                        vec_part_i32x4=vec_a_parts[i],
                    )

            def prefetch_ab_tile(base_k):
                base_k_div4 = base_k / 4
                a_regs = load_a_tile(base_k_div4)
                b_regs = load_b_tile(base_k)
                return a_regs, b_regs

            def compute_tile(accs_in, b_tile_in, lds_base, *, is_last_tile=False, a0_prefetch=None):
                scales_pf = {}
                if is_last_tile:
                    # Prefetch scales (same as original kernel).
                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        s_b_vals.append(
                            buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=T.f32)
                        )
                    scales_pf["s_b_vals"] = s_b_vals
                    scales_pf["s_a_vecs"] = []
                    row_off_base = lane_div_16 * 4
                    for mi in range_constexpr(m_repeat):
                        row_base_m = bx_m + (mi * 16)
                        row_g_base = row_base_m + row_off_base
                        s_a_vec = buffer_ops.buffer_load(
                            scale_a_rsrc, row_g_base, vec_width=4, dtype=T.f32
                        )
                        scales_pf["s_a_vecs"].append(vector.bitcast(T.f32x4, s_a_vec))

                current_accs_list = list(accs_in)

                # ---------------- gfx95 fast path (K128 MFMA scale) ----------------
                # This is the key optimization from `zhimding/develop_0107` for FP8:
                # use mfma.scale 16x16x128 to reduce instruction count in the hot loop.
                #
                # Notes:
                # - Only valid for fp8 path (not int8/int4) and gfx95+
                # - Requires tile_k divisible by 128
                # - mfma.scale takes 9 operands: 3 vectors + 6 i32 flags/scales.
                use_mfma_scale_128 = (str(gpu_arch).startswith("gfx95") and (not is_int8) and (not is_int4))
                if use_mfma_scale_128:
                    if (int(tile_k) % 128) != 0:
                        raise ValueError(
                            f"tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_k}"
                        )

                    mfma_res_ty = T.f32x4
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    for ku128 in range_constexpr(k_unroll // 2):
                        ku0 = ku128 * 2
                        ku1 = ku0 + 1

                        b0_packs0, b0_packs1 = b_tile_in[ku0]
                        b1_packs0, b1_packs1 = b_tile_in[ku1]

                        col_base0 = col_offset_base + (ku0 * 64)
                        col_base1 = col_offset_base + (ku1 * 64)

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi * 16, index=True)
                            curr_row_a_lds = row_a_lds + mi_val

                            if (a0_prefetch is not None) and (ku0 == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_base)
                            a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_base)
                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                            for ni in range_constexpr(num_acc_n):
                                b0 = b0_packs0[ni]
                                b1 = b0_packs1[ni]
                                b2 = b1_packs0[ni]
                                b3 = b1_packs1[ni]
                                b128 = pack_i64x4_to_i32x8(b0, b1, b2, b3)

                                acc_idx = mi * num_acc_n + ni
                                current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    mfma_res_ty,
                                    [
                                        a128,
                                        b128,
                                        current_accs_list[acc_idx],
                                        # cbsz, abid, blgp: 0
                                        0,
                                        0,
                                        0,
                                        # op_sel_a + scale_a (1.0f as i32 bits)
                                        0x3F800000,
                                        # op_sel_b + scale_b (1.0f as i32 bits)
                                        0,
                                        0x3F800000,
                                    ],
                                )
                    return current_accs_list, scales_pf

                mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                mfma_fn = mfma_i32_k32 if is_int8 else rocdl.mfma_f32_16x16x32_fp8_fp8

                # MFMA K64 wrapper: two K32 MFMA back-to-back.
                def mfma_k64(acc_in, a0, a1, b0, b1):
                    acc_mid = mfma_fn(
                        mfma_res_ty,
                        [a0, b0, acc_in, 0, 0, 0],
                    )
                    return mfma_fn(
                        mfma_res_ty,
                        [a1, b1, acc_mid, 0, 0, 0],
                    )

                for ku in range_constexpr(k_unroll):
                    b_packs0, b_packs1 = b_tile_in[ku]
                    ki64 = ku * 64
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
                            current_accs_list[acc_idx] = mfma_k64(
                                current_accs_list[acc_idx],
                                a0,
                                a1,
                                b_packs0[ni],
                                b_packs1[ni],
                            )
                return current_accs_list, scales_pf

            vec1_f16 = ir.VectorType.get([1], ir.F16Type.get())
            vec2_f16 = ir.VectorType.get([2], ir.F16Type.get())
            vec1_i16 = ir.VectorType.get([1], ir.IntegerType.get_signless(16))
            vec2_i16 = ir.VectorType.get([2], ir.IntegerType.get_signless(16))
            vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))

            def store_output(final_accs, scales):
                s_b_vals = scales["s_b_vals"]
                s_a_vecs = scales["s_a_vecs"]

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

                        s_a_vec4 = s_a_vecs[mi]
                        s_a = vector.extract(s_a_vec4, static_position=[ii], dynamic_position=[])
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            acc = final_accs[acc_idx]
                            val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                            if is_int8:
                                val = arith.sitofp(T.f32, val)
                            val_s = (val * s_a) * s_b_vals[ni]
                            v16 = arith.trunc_f(T.f16, val_s)

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
                            vector.store(v2, lds_out, [lds_idx], alignment=4)

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
                    s_a_vec4 = s_a_vecs[mi]
                    s_a = vector.extract(s_a_vec4, static_position=[ii], dynamic_position=[])
                    col_base = by_n + n_tile_base + lane_mod_16
                    idx_base = flir.crd2idx(flir.make_coord(row, col_base), layout_c)
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                        if is_int8:
                            val = arith.sitofp(T.f32, val)
                        val_s = (val * s_a) * s_b_vals[ni]
                        val_f16 = arith.trunc_f(T.f16, val_s)
                        idx_out = idx_base + arith.constant(ni * 16, index=True)
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx_out)

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
                dswr_tail = num_a_loads
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

            # ---------------- Pipeline ----------------
            lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
            lds_base0 = arith.constant(0, index=True)
            lds_base1 = lds_tile_elems

            if lds_stage == 2:
                # ---------------- Ping-pong pipeline (2 LDS buffers) ----------------
                # Cross-tile A0 LDS prefetch (default-on):
                # issue the first A-pack DS read for the next tile *between* barriers,
                # so it can overlap with the VMEM prefetch of the following tile.

                def prefetch_a0_pack(lds_base):
                    # (mi=0, ku=0): prefetch both K32 halves (K64) for the first A-pack.
                    return lds_load_packs_k64(row_a_lds, col_offset_base, lds_base)

                # Prologue: tile-0
                k0 = arith.constant(0, index=True)
                a_regs0, b_tile0 = prefetch_ab_tile(k0)
                store_a_tile_to_lds(a_regs0, lds_base0)
                gpu.barrier()
                accs = [acc_init] * (num_acc_n * m_repeat)

                lds_base_pong = lds_base0
                lds_base_ping = lds_base1
                b_tile_pong = b_tile0
                c_k_main = c_k - tile_k

                # Prefetch A0 for the first compute tile (overlap with the next VMEM prefetch).
                a0_prefetch_pong = prefetch_a0_pack(lds_base_pong)

                num_tiles = K // tile_k
                if (num_tiles % 2) == 1:
                    for k_iv in range(0, c_k_main, tile_k * 2):
                        next_k1 = k_iv + tile_k
                        a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)

                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_base_pong, a0_prefetch=a0_prefetch_pong
                        )
                        a0_prefetch_pong = None

                        store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the ping tile we are about to compute.
                        a0_prefetch_ping = prefetch_a0_pack(lds_base_ping)

                        next_k2 = k_iv + tile_k * 2
                        a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)

                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_base_ping, a0_prefetch=a0_prefetch_ping
                        )
                        a0_prefetch_ping = None

                        store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the next pong tile.
                        a0_prefetch_pong = prefetch_a0_pack(lds_base_pong)

                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_pong,
                        lds_base_pong,
                        is_last_tile=True,
                        a0_prefetch=a0_prefetch_pong,
                    )
                else:
                    c_k_stop = c_k - (tile_k * 3)
                    for k_iv in range(0, c_k_stop, tile_k * 2):
                        next_k1 = k_iv + tile_k
                        a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)

                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_base_pong, a0_prefetch=a0_prefetch_pong
                        )
                        a0_prefetch_pong = None

                        store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_ping = prefetch_a0_pack(lds_base_ping)

                        next_k2 = k_iv + tile_k * 2
                        a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)

                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_base_ping, a0_prefetch=a0_prefetch_ping
                        )
                        a0_prefetch_ping = None

                        store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_pong = prefetch_a0_pack(lds_base_pong)

                    last_k = c_k - tile_k
                    a_regs_ping, b_tile_ping = prefetch_ab_tile(last_k)

                    accs, _ = compute_tile(
                        accs, b_tile_pong, lds_base_pong, a0_prefetch=a0_prefetch_pong
                    )
                    a0_prefetch_pong = None

                    store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()

                    a0_prefetch_ping = prefetch_a0_pack(lds_base_ping)

                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_ping,
                        lds_base_ping,
                        is_last_tile=True,
                        a0_prefetch=a0_prefetch_ping,
                    )

                store_output(final_accs, scales)
            else:
                # CK-like bpreshuffle v1 spirit:
                # - Intrawave schedule
                # - Global prefetch 2 (regs double-buffer)
                # - Local shared memory buffer 1 (single LDS tile for A)
                # Prologue: tile-0
                k0 = arith.constant(0, index=True)
                a_regs0, b_tile0 = prefetch_ab_tile(k0)
                store_a_tile_to_lds(a_regs0, lds_base0)
                gpu.barrier()
                accs = [acc_init] * (num_acc_n * m_repeat)

                lds_base = lds_base0
                b_tile_cur = b_tile0

                # For each tile except last: prefetch next tile, compute current, then overwrite LDS.
                for k_base in range(0, c_k - tile_k, tile_k):
                    next_k = k_base + tile_k
                    a_next, b_next = prefetch_ab_tile(next_k)
                    accs, _ = compute_tile(accs, b_tile_cur, lds_base)
                    # Single LDS buffer: ensure *all* waves are done reading A from LDS
                    # before any wave overwrites it with the next tile.
                    gpu.barrier()
                    store_a_tile_to_lds(a_next, lds_base)
                    hot_loop_scheduler()
                    gpu.barrier()
                    b_tile_cur = b_next

                final_accs, scales = compute_tile(
                    accs, b_tile_cur, lds_base, is_last_tile=True
                )
                store_output(final_accs, scales)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(size_c, T.f16),
            arg_a: lambda: memref(size_a, _elem_type()),
            arg_b: lambda: memref(size_b, _elem_type()),
            arg_scale_a: lambda: memref(M, T.f32),
            arg_scale_b: lambda: memref(N, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(256, index=True)
            gx = arith.constant(M // tile_m, index=True)
            gy = arith.constant(N // tile_n, index=True)
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
            )

    m = _GEMM()
    return flydsl.compile(m)


__all__ = ["compile_preshuffle_gemm_a8"]

