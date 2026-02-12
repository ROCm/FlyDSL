"""Blockscale Preshuffle GEMM kernel (FLIR MFMA FP8).

Block-scale quantization applies per-block scales to A and B during the GEMM.
This matches the CK ``gemm_a8w8_blockscale`` kernel semantics:

- ScaleBlockM = 1  (per-row for A)
- ScaleBlockN = 128
- ScaleBlockK = 128

Scale tensor layouts expected by this kernel:
- scale_a: [scale_k, M] flattened (K-major / transposed), where scale_k = K / scale_block_k.
  In Python: ``scale_a = scale_a_orig.T.contiguous().view(-1)``
  This allows vectorized loading of 4 consecutive rows' scales.
- scale_b: [scale_n, scale_k] row-major flattened, where scale_n = ceil(N / scale_block_k).

Math:
  C[m,n] = sum_kb( scale_a[m, kb] * scale_b[n//128, kb]
                    * sum_{k in block_kb}( A[m,k] * B[n,k] ) )

Pipeline:
- 2-stage ping-pong LDS (same as preshuffle_gemm).
- Within each tile, K-loop is split into scale-block-sized chunks (128 elems).
  After each chunk, block accumulators are multiplied by the per-block scales
  and accumulated into global accumulators.
"""

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator, SmemPtr

from _mlir import ir

from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl
from flydsl.dialects.ext.arith import ArithValue
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token
from flydsl.compiler.compiler import _apply_waves_per_eu_hint

from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
)
from kernels.mfma_epilogues import mfma_epilog


def compile_blockscale_preshuffle_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    scale_block_k: int = 128,
    out_dtype: str = "bf16",
    use_cshuffle_epilog: bool = False,
    waves_per_eu: int = None,
):
    """Compile the blockscale preshuffle GEMM kernel.

    Args:
        M, N, K: GEMM sizes (A[M,K] fp8, B[N,K] fp8 preshuffled, C[M,N]).
        tile_m, tile_n, tile_k: block tile sizes.
        scale_block_k: K-dimension block size for scales (default 128, matching CK).
        out_dtype: Output dtype, "fp16" or "bf16".
        use_cshuffle_epilog: Enable cross-shuffle epilogue.
        waves_per_eu: Occupancy hint (None = default).
    """
    if out_dtype not in ("fp16", "bf16"):
        raise ValueError(f"out_dtype must be 'fp16' or 'bf16', got {out_dtype!r}")
    if tile_k % scale_block_k != 0:
        raise ValueError(
            f"tile_k ({tile_k}) must be divisible by scale_block_k ({scale_block_k})"
        )
    if K % tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by tile_k ({tile_k})")
    if K % scale_block_k != 0:
        raise ValueError(
            f"K ({K}) must be divisible by scale_block_k ({scale_block_k})"
        )

    # Compile-time scale dimensions
    scale_k = K // scale_block_k
    scale_n = (N + scale_block_k - 1) // scale_block_k
    sb_per_tile = tile_k // scale_block_k  # scale blocks per tile
    # For fp8, 1 byte per element; each ku step = 64 bytes = 64 elements
    ku_per_sb = scale_block_k // 64

    elem_bytes = 1  # fp8
    tile_k_bytes = tile_k * elem_bytes

    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}"
        )

    is_bf16_out = out_dtype == "bf16"

    gpu_arch = get_hip_arch()

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    _state = {}

    DYN = ir.ShapedType.get_dynamic_size()

    total_threads = 256
    bytes_a_per_tile = tile_m * tile_k * elem_bytes
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k must be divisible by {total_threads}: "
            f"tile_m={tile_m}, tile_k={tile_k}"
        )
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )

    lds_stride_bytes = tile_k_bytes

    def _out_elem_type():
        return T.bf16 if is_bf16_out else T.f16

    def _out_vec1_type():
        return ir.VectorType.get([1], ir.BF16Type.get() if is_bf16_out else ir.F16Type.get())

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = f"mfma_blockscale_{out_dtype}_{epilog_tag}".replace("-", "_")

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            lds_tile_bytes = tile_m * lds_stride_bytes
            lds_out_bytes = 2 * tile_m * tile_n if use_cshuffle_epilog else 0

            assert lds_out_bytes % 2 == 0, "lds_out_bytes should be multiple of 2"
            buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // 2)
            buffer_size_elems = buffer_size_bytes  # fp8: 1 byte per elem
            _state["lds_a_pong"] = allocator_pong.allocate_array(T.f8, buffer_size_elems)
            _state["lds_a_ping"] = allocator_ping.allocate_array(T.f8, buffer_size_elems)

            allocator_pong.finalize()
            allocator_ping.finalize()

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, T.f8),
            arg_b: lambda: memref(DYN, T.f8),
            arg_scale_a: lambda: memref(DYN, T.f32),
            arg_scale_b: lambda: memref(DYN, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            # ---- Accumulator init ----
            acc_init = arith.unwrap(arith.constant_vector(0.0, T.f32x4))

            # ---- Layouts ----
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            c_k_div4bytes = c_k / 4
            c_k_bytes = c_k
            layout_a = flir.make_layout((c_m, c_k_bytes), stride=(c_k_bytes, 1))
            layout_a_div4 = flir.make_layout(
                (c_m, c_k_div4bytes), stride=(c_k_div4bytes, 1)
            )

            kpack_bytes = 16
            layout_b = make_preshuffle_b_layout(
                flir, arith, c_n=c_n, c_k=c_k, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes
            ).layout_b

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(tile_k, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            k_blocks16 = arith.index(tile_k_bytes // 16)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr, base_ptr1 = allocator_pong.get_base(), allocator_ping.get_base()

            lds_a_pong_ptr = _state["lds_a_pong"](base_ptr)
            lds_a_ping_ptr = _state["lds_a_ping"](base_ptr1)

            lds_a_pong = SmemPtr(
                base_ptr, lds_a_pong_ptr.byte_offset, T.f8, shape=(tile_m * tile_k,)
            ).get()
            lds_a_ping = SmemPtr(
                base_ptr1, lds_a_ping_ptr.byte_offset, T.f8, shape=(tile_m * tile_k,)
            ).get()

            if use_cshuffle_epilog:
                lds_out = SmemPtr(
                    base_ptr, lds_a_pong_ptr.byte_offset, _out_elem_type(),
                    shape=(tile_m * tile_n,)
                ).get()
            else:
                lds_out = None

            a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)

            b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

            bx_m = bx * tile_m
            by_n = by * tile_n

            # ---- Thread mapping ----
            wave_size = 64
            layout_wave_lane = flir.make_layout((4, wave_size), stride=(64, 1))
            coord_wave_lane = flir.idx2crd(tx, layout_wave_lane)
            wave_id = flir.get(coord_wave_lane, 0)
            lane_id = flir.get(coord_wave_lane, 1)

            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            coord_lane16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_lane16, 0)
            lane_mod_16 = flir.get(coord_lane16, 1)

            row_a_lds = lane_mod_16
            kpack_elems = 16
            col_offset_base = lane_div_16 * arith.constant(kpack_elems, index=True)
            col_offset_base_bytes = col_offset_base

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 64

            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            n_tile_base = wave_id * c_n_per_wave

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

            # ---- B load helpers ----
            def load_b_pack(base_k, ki_step, ni):
                return load_b_pack_k32(
                    buffer_ops, flir, arith, vector,
                    arg_b=arg_b, b_rsrc=b_rsrc, layout_b=layout_b,
                    base_k=base_k, ki_step=ki_step,
                    n_blk=n_blk_list[ni], n_intra=n_intra_list[ni],
                    lane_div_16=lane_div_16,
                    elem_type=T.f8, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes,
                )

            atom_b_g2r16 = flir.make_copy_atom(T.f8, vector_size=16)
            c64_b = 64
            c0_idx = 0

            def load_b_packs_k64(base_k, ku: int, ni: int):
                base_k_bytes = base_k
                k0_base = base_k_bytes / c64_b
                k0 = k0_base + ku
                k1 = lane_div_16
                coord_pack = flir.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], c0_idx)
                idx_pack = flir.crd2idx(coord_pack, layout_b)
                b_view = flir.TensorView(
                    arg_b, (16,), strides=(1,), base_indices=(idx_pack,), element_type=T.f8,
                )
                b16 = flir.copy(
                    flir.make_copy_atom(T.f8, vector_size=16),
                    b_view, None, alignment=8, return_vector=True,
                    src_buffer_resource=b_rsrc, src_buffer_offset_in_bytes=True,
                )
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

            # ---- A LDS helpers ----
            def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
                col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = col_base_swz_bytes
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                return vector.load_op(T.f8x16, lds_buffer, [idx_a16])

            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
                a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0_i64, a1_i64

            # ---- A global load / LDS store ----
            num_a_loads = bytes_per_thread_a // a_load_bytes
            tile_k_dwords = tile_k // 4
            layout_a_tile_div4 = flir.make_layout(
                (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
            )
            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4
            atom_a_g2r16 = flir.make_copy_atom(T.f8, vector_size=16)

            def load_a_16(idx_elem):
                return buffer_copy_gmem16_dwordx4(
                    flir, arg=arg_a, elem_type=T.f8, idx_i32=idx_elem,
                    atom_g2r16=atom_a_g2r16, rsrc=a_rsrc, vec_elems=16,
                )

            def a_tile_chunk_coord_i32(i: int, tx_i32_base_v):
                return tile_chunk_coord_i32(
                    flir, arith,
                    tx_i32_base=tx_i32_base_v, i=i, total_threads=total_threads,
                    layout_tile_div4=layout_a_tile_div4, chunk_i32=4,
                )

            def load_a_tile(base_k_div4):
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i, tx_i32_base)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 + col_a_local_i32)
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    a_16B = load_a_16(idx_i32)
                    parts.append(vector.bitcast(T.i32x4, a_16B))
                return parts

            def store_a_tile_to_lds(vec_a_parts, lds_buffer):
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i, tx_i32_base)
                    lds_store_16b_xor16(
                        flir, arith, vector,
                        lds_memref=lds_buffer, vec16_ty=T.f8x16, elem_type=T.f8,
                        atom_s16=atom_a_g2r16, layout_lds=layout_lds,
                        row_local=row_a_local, col_local_i32=col_a_local_i32,
                        tx_c4=c4, k_blocks16=k_blocks16,
                        lds_base=arith.constant(0, index=True),
                        vec_part_i32x4=vec_a_parts[i], elem_bytes=elem_bytes,
                    )

            def prefetch_a_tile(base_k):
                base_k_div4 = base_k / 4
                return load_a_tile(base_k_div4)

            def prefetch_b_tile(base_k):
                return load_b_tile(base_k)

            # ---- MFMA ops ----
            mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
            mfma_res_ty = T.f32x4

            def mfma_step(acc_in, a, b):
                return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

            def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                acc_mid = mfma_step(acc_in, a0, b0)
                return mfma_step(acc_mid, a1, b1)

            # ---- Scale constants ----
            c_scale_block_k = arith.constant(scale_block_k, index=True)
            c_scale_k = arith.constant(scale_k, index=True)
            c_128 = arith.constant(128, index=True)
            row_off_base = lane_div_16 * 4

            # ---- Blockscale compute tile (optimized) ----
            def compute_tile_blockscale(
                global_accs, b_tile_in, lds_buffer, k_base, *, a0_prefetch=None
            ):
                """Compute one tile's MFMA with per-block scale application.

                Optimizations vs naive element-wise approach:
                1. Scale loads issued BEFORE MFMA to overlap memory latency with compute.
                2. Combined scale vectors (sa * broadcast(sb)) pre-computed before accumulate.
                3. Vector-level ArithValue ops: 3 vector ops replace 16+ scalar extract/insert.
                """
                current_global = list(global_accs)

                for sb in range_constexpr(sb_per_tile):
                    # ---- STEP 1: Load scales FIRST (hide VMEM latency behind MFMA) ----
                    kb = k_base / c_scale_block_k + arith.constant(sb, index=True)

                    sa_base_offset = kb * c_m
                    s_a_vecs = []
                    for mi in range_constexpr(m_repeat):
                        row_base_m = bx_m + arith.constant(mi * 16, index=True)
                        row_g_base = row_base_m + row_off_base
                        sa_idx = sa_base_offset + row_g_base
                        s_a_vec = buffer_ops.buffer_load(
                            scale_a_rsrc, sa_idx, vec_width=4, dtype=T.f32
                        )
                        s_a_vecs.append(vector.bitcast(T.f32x4, s_a_vec))

                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_base_ni = by_n + n_tile_base + arith.constant(ni * 16, index=True)
                        n_block = col_base_ni / c_128
                        sb_idx = n_block * c_scale_k + kb
                        s_b_val = buffer_ops.buffer_load(
                            scale_b_rsrc, sb_idx, vec_width=1, dtype=T.f32
                        )
                        s_b_vals.append(s_b_val)

                    # ---- STEP 2: Pre-compute combined scale vectors ----
                    # combined[mi][ni] = sa_vec[mi] * broadcast(sb[ni])
                    # Broadcast sb scalar to vec4, then vec4 * vec4 multiply.
                    vec4_f32 = ir.VectorType.get([4], ir.F32Type.get())
                    s_b_vecs = []
                    for ni in range_constexpr(num_acc_n):
                        s_b_vecs.append(vector.broadcast(vec4_f32, s_b_vals[ni]))

                    combined_scales = []
                    for mi in range_constexpr(m_repeat):
                        mi_combined = []
                        for ni in range_constexpr(num_acc_n):
                            combined = ArithValue(s_a_vecs[mi]) * ArithValue(s_b_vecs[ni])
                            mi_combined.append(combined)
                        combined_scales.append(mi_combined)

                    # ---- STEP 3: MFMA for this scale block ----
                    block_accs = [acc_init] * (num_acc_n * m_repeat)

                    for ku_local in range_constexpr(ku_per_sb):
                        ku = sb * ku_per_sb + ku_local
                        b_packs0, b_packs1 = b_tile_in[ku]
                        ki64 = ku * 64
                        col_base = col_offset_base_bytes + ki64

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi * 16, index=True)
                            curr_row_a_lds = row_a_lds + mi_val

                            if (
                                a0_prefetch is not None
                                and sb == 0
                                and ku_local == 0
                                and mi == 0
                            ):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(
                                    curr_row_a_lds, col_base, lds_buffer
                                )

                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni
                                block_accs[acc_idx] = mfma_k64_bytes(
                                    block_accs[acc_idx],
                                    a0, a1,
                                    b_packs0[ni], b_packs1[ni],
                                )

                    # ---- STEP 4: Accumulate with vector ops ----
                    # global_acc += block_acc * combined_scale
                    # Uses ArithValue for vec4 multiply + add (3 vector ops per accumulator).
                    for mi in range_constexpr(m_repeat):
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            block_w = ArithValue(block_accs[acc_idx])
                            global_w = ArithValue(current_global[acc_idx])
                            combined = combined_scales[mi][ni]
                            # vec4 multiply + vec4 add
                            new_global = global_w + block_w * combined
                            current_global[acc_idx] = arith.unwrap(new_global)

                return current_global

            # ---- Epilogue: convert and store (no scale application) ----
            vec1_out = _out_vec1_type()

            def store_output(final_accs):
                if use_cshuffle_epilog:
                    if lds_out is None:
                        raise RuntimeError(
                            "use_cshuffle_epilog=True but lds_out is not allocated."
                        )
                    gpu.barrier()

                    def write_row_to_lds(
                        *, mi, ii, row_in_tile, row,
                        row_base_lds, col_base_local, num_acc_n, lds_out,
                    ):
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            acc = final_accs[acc_idx]
                            val = vector.extract(
                                acc, static_position=[ii], dynamic_position=[]
                            )
                            v_out = arith.trunc_f(_out_elem_type(), val)
                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        idx_out = flir.crd2idx(
                            flir.make_coord(row, col_g0), layout_c
                        )
                        byte_off = idx_out * arith.constant(2, index=True)
                        e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                        if e_vec == 4:
                            frag_i32x2 = vector.bitcast(T.vec(2, T.i32), frag)
                            buffer_ops.buffer_store(
                                frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True
                            )
                        else:
                            frag_i32x1 = vector.bitcast(T.vec(1, T.i32), frag)
                            frag_i32 = vector.extract(
                                frag_i32x1, static_position=[0], dynamic_position=[]
                            )
                            buffer_ops.buffer_store(
                                frag_i32, c_rsrc, byte_off, offset_is_bytes=True
                            )

                    e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                    frag_elem_type = ir.BF16Type.get() if is_bf16_out else ir.F16Type.get()
                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith, vector=vector, gpu=gpu,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m, tile_n=tile_n, e_vec=e_vec,
                        m_repeat=m_repeat, num_acc_n=num_acc_n,
                        tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16,
                        bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=frag_elem_type,
                        write_row_to_lds=write_row_to_lds,
                        store_pair=store_pair,
                    )
                    return

                # Direct epilogue: convert f32 -> f16/bf16 and store element-wise
                def body_row(*, mi, ii, row_in_tile, row):
                    col_base = by_n + n_tile_base + lane_mod_16
                    idx_base = flir.crd2idx(flir.make_coord(row, col_base), layout_c)
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(
                            acc, static_position=[ii], dynamic_position=[]
                        )
                        val_out = arith.trunc_f(_out_elem_type(), val)
                        idx_out = idx_base + arith.constant(ni * 16, index=True)
                        buffer_ops.buffer_store(val_out, c_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=False,
                    arith=arith, range_constexpr=range_constexpr,
                    m_repeat=m_repeat, lane_div_16=lane_div_16,
                    bx_m=bx_m, body_row=body_row,
                )

            # ---- Scheduling hints ----
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                mfma_group = num_acc_n
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

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

            # ---- 2-stage ping-pong pipeline ----
            def prefetch_a0_pack(lds_buffer):
                return lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)

            # Prologue: tile-0
            k0 = arith.constant(0, index=True)
            b_tile_pong = prefetch_b_tile(k0)
            store_a_tile_to_lds(prefetch_a_tile(k0), lds_a_pong)
            gpu.barrier()
            global_accs = [acc_init] * (num_acc_n * m_repeat)

            c_k_main = c_k - tile_k
            a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

            num_tiles = K // tile_k
            c_tile_k = arith.constant(tile_k, index=True)

            if (num_tiles % 2) == 1:
                for k_iv in range(0, c_k_main, tile_k * 2):
                    next_k1 = k_iv + tile_k
                    b_tile_ping = prefetch_b_tile(next_k1)
                    store_a_tile_to_lds(prefetch_a_tile(next_k1), lds_a_ping)

                    global_accs = compute_tile_blockscale(
                        global_accs, b_tile_pong, lds_a_pong, k_iv,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None

                    hot_loop_scheduler()
                    gpu.barrier()
                    a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

                    next_k2 = k_iv + tile_k * 2
                    b_tile_pong = prefetch_b_tile(next_k2)
                    store_a_tile_to_lds(prefetch_a_tile(next_k2), lds_a_pong)

                    global_accs = compute_tile_blockscale(
                        global_accs, b_tile_ping, lds_a_ping, next_k1,
                        a0_prefetch=a0_prefetch_ping,
                    )
                    a0_prefetch_ping = None

                    hot_loop_scheduler()
                    gpu.barrier()
                    a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

                # Last tile
                last_k = c_k - c_tile_k
                final_accs = compute_tile_blockscale(
                    global_accs, b_tile_pong, lds_a_pong, last_k,
                    a0_prefetch=a0_prefetch_pong,
                )
            else:
                c_k_stop = c_k - (tile_k * 3)
                for k_iv in range(0, c_k_stop, tile_k * 2):
                    next_k1 = k_iv + tile_k
                    b_tile_ping = prefetch_b_tile(next_k1)
                    store_a_tile_to_lds(prefetch_a_tile(next_k1), lds_a_ping)
                    global_accs = compute_tile_blockscale(
                        global_accs, b_tile_pong, lds_a_pong, k_iv,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    hot_loop_scheduler()
                    gpu.barrier()

                    a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

                    next_k2 = k_iv + tile_k * 2
                    b_tile_pong = prefetch_b_tile(next_k2)
                    store_a_tile_to_lds(prefetch_a_tile(next_k2), lds_a_pong)
                    global_accs = compute_tile_blockscale(
                        global_accs, b_tile_ping, lds_a_ping, next_k1,
                        a0_prefetch=a0_prefetch_ping,
                    )
                    a0_prefetch_ping = None

                    hot_loop_scheduler()
                    gpu.barrier()
                    a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

                # Second-to-last tile
                last_k = c_k - tile_k
                b_tile_ping = prefetch_b_tile(last_k)
                store_a_tile_to_lds(prefetch_a_tile(last_k), lds_a_ping)

                second_last_k = c_k - tile_k * 2
                global_accs = compute_tile_blockscale(
                    global_accs, b_tile_pong, lds_a_pong, second_last_k,
                    a0_prefetch=a0_prefetch_pong,
                )
                a0_prefetch_pong = None

                hot_loop_scheduler()
                gpu.barrier()
                a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

                # Last tile
                final_accs = compute_tile_blockscale(
                    global_accs, b_tile_ping, lds_a_ping, last_k,
                    a0_prefetch=a0_prefetch_ping,
                )

            store_output(final_accs)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, T.f8),
            arg_b: lambda: memref(DYN, T.f8),
            arg_scale_a: lambda: memref(DYN, T.f32),
            arg_scale_b: lambda: memref(DYN, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
            stream_ptr: lambda: T.i64,
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
                    arg_c, arg_a, arg_b,
                    arg_scale_a, arg_scale_b,
                    c_m, c_n, c_k,
                ],
                async_dependencies=[stream_token],
            )

    m = _GEMM()

    if waves_per_eu is not None:
        _apply_waves_per_eu_hint(m.module, waves_per_eu)

    return flydsl.compile(
        m,
        use_bare_ptr_memref_call_conv=False,
        use_bare_pointers_for_host=False,
        use_bare_pointers_for_kernels=False,
    )


def select_tile_config(M: int, N: int, K: int, scale_block_k: int = 128):
    """Select a good tile config (tile_m, tile_n, tile_k) for the given GEMM shape.

    Heuristic based on CK's tuned kernel selection patterns:
    - Small M: use smaller tile_m (16-32) with larger tile_k for latency hiding.
    - Medium M: balanced tile_m=32-64 for occupancy vs register trade-off.
    - Large M: tile_m=64 maximizes compute density.
    - tile_n=128 is the universal sweet-spot (4 waves * 32 cols each).
    """
    # All candidate tiles (matching CK's kernel instance set)
    candidates = [
        (16, 64, 256), (16, 128, 256),
        (32, 64, 128), (32, 64, 256), (32, 128, 128), (32, 128, 256),
        (64, 64, 128), (64, 64, 256), (64, 128, 128), (64, 128, 256), (64, 256, 128),
    ]

    def _valid(tm, tn, tk):
        if N % tn != 0 or K % tk != 0:
            return False
        if tk % scale_block_k != 0:
            return False
        bpt = tm * tk // 256
        if bpt < 16:
            return False
        return True

    valid = [(tm, tn, tk) for tm, tn, tk in candidates if _valid(tm, tn, tk)]
    if not valid:
        return (64, 128, 128)  # fallback

    # Score: based on actual benchmark results across shapes.
    # Key findings:
    #   M<=128: tile_m=32 best (m_repeat=2, good MFMA utilization)
    #   M=256-384: tile_m=32 (balance register/occupancy)
    #   M>=512: tile_m=64 (maximize compute density)
    #   tile_n=128 universally good; tile_k=128 usually beats 256
    def _score(tm, tn, tk):
        s = 0
        grid_m = (M + tm - 1) // tm
        grid_n = N // tn
        total_blocks = grid_m * grid_n
        # Need enough blocks to fill GPU (304 CUs on MI300X)
        if total_blocks >= 256:
            s += 15
        elif total_blocks >= 128:
            s += 10
        elif total_blocks >= 64:
            s += 5
        # tile_m selection by M size
        if M <= 48:
            s += 12 if tm == 16 else (8 if tm == 32 else 0)
        elif M <= 256:
            s += 12 if tm == 32 else (6 if tm == 64 else (3 if tm == 16 else 0))
        elif M <= 512:
            s += 12 if tm == 64 else (8 if tm == 32 else 0)
        else:
            s += 12 if tm == 64 else 0
        # tile_n: 128 is universal best; only prefer 64 when total_blocks is very low
        s += 8 if tn == 128 else (4 if tn == 64 else (2 if tn == 256 else 0))
        # tile_k: 128 generally better (less register pressure, more K-tiles for pipelining)
        # Exception: small M benefits from tile_k=256 (more compute per tile)
        if M <= 128:
            s += 4 if tk == 256 else 2
        else:
            s += 4 if tk == 128 else 2
        return s

    return max(valid, key=lambda t: _score(*t))


__all__ = ["compile_blockscale_preshuffle_gemm", "select_tile_config"]
