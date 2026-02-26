"""Preshuffle GEMM kernel using the @flyc.kernel API."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir

from flydsl.expr import arith, vector
from flydsl.expr import gpu
from flydsl.expr import buffer_ops, rocdl


from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from flydsl.expr import crd2idx, idx2crd, get as layout_get

from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
    swizzle_xor16,
    _buffer_load_vec,
)
from kernels.mfma_epilogues import mfma_epilog


def compile_preshuffle_gemm_a8(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
    waves_per_eu: int = None,
    use_async_copy: bool = False,
):
    """Compile the preshuffle GEMM kernel using the @flyc.kernel API.

    Returns a JitFunction that auto-compiles and executes when called.
    Signature:  launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, M, N, stream)

    Compile-time constants: K, tile_m/n/k, in_dtype (determine loop structure).
    Runtime parameters: M, N (passed as i32 kernel args).

    Args:
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
        use_async_copy: Use async DMA for A tile global-to-LDS transfer.
    """
    if in_dtype not in ("fp8", "int8", "int4", "fp16", "bf16"):
        raise ValueError(
            "in_dtype must be one of ('fp8','int8','int4','fp16','bf16'), "
            f"got {in_dtype!r}"
        )
    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    elem_bytes = 1 if (in_dtype in ("fp8", "int8", "int4")) else 2

    tile_k_bytes = int(tile_k) * int(elem_bytes)

    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )

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

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    total_threads = 256
    bytes_a_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_a = bytes_a_per_tile // total_threads

    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )
    a_async_load_bytes = 4 if gpu_arch == "gfx942" else 16
    a_async_load_dword = a_async_load_bytes // 4

    lds_stride_bytes = tile_k_bytes

    def _elem_type():
        if is_f16:
            return T.f16
        if is_bf16:
            return T.bf16
        return T.i8 if is_int8 else T.f8

    def _vec16_type():
        if is_f16:
            return T.f16x8
        if is_bf16:
            return T.bf16x8
        return T.i8x16 if is_int8 else T.f8x16

    def _mfma_pack_ty():
        if is_f16:
            return T.f16x4
        if is_bf16:
            return T.i16x4
        return T.i64

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = f"mfma_preshuffle_{lds_stage}stages_{in_dtype}_{epilog_tag}".replace("-", "_")

    # ── LDS sizing (pure Python, no MLIR ops) ────────────────────────────────
    lds_tile_bytes = int(tile_m) * int(lds_stride_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0

    if int(lds_stage) == 2:
        assert lds_out_bytes % 2 == 0, "lds_out_bytes should be multiple of 2"
        buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // lds_stage)
        buffer_size_elems = buffer_size_bytes if elem_bytes == 1 else (buffer_size_bytes // 2)

        lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_pong_offset + buffer_size_elems * elem_bytes

        lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = lds_ping_offset + buffer_size_elems * elem_bytes
    else:
        lds_total_bytes = max(lds_tile_bytes, lds_out_bytes)
        lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

        lds_alloc_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_alloc_offset + lds_total_elems * elem_bytes

    # ── Kernel function ────────────────────────────────────────────────────
    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        c_m = arith.index_cast(T.index, i32_m.ir_value())
        c_n = arith.index_cast(T.index, i32_n.ir_value())

        # ---- Types ----
        acc_init = (
            arith.constant_vector(0, T.i32x4)
            if is_int8
            else arith.constant_vector(0.0, T.f32x4)
        )

        # ---- Layouts ----
        _k_div4_factor = (K * elem_bytes) // 4

        kpack_bytes = 8 if is_int4 else 16
        kpack_elems = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
        k_bytes = K * elem_bytes
        n0_val = N // 16
        k0_val = k_bytes // 64
        _stride_nlane = kpack_elems
        _stride_klane = 16 * _stride_nlane
        _stride_k0 = 4 * _stride_klane
        _stride_n0 = k0_val * _stride_k0
        layout_b = fx.make_layout(
            (n0_val, k0_val, 4, 16, kpack_elems),
            (_stride_n0, _stride_k0, _stride_klane, _stride_nlane, 1),
        )

        shape_lds = fx.make_shape(tile_m, tile_k)
        stride_lds = fx.make_stride(tile_k, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        k_blocks16 = arith.index(tile_k_bytes // 16)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        # ---- LDS (separate ping/pong buffers for no-alias guarantee) ----
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()

        if lds_stage == 2:
            lds_a_pong = SmemPtr(
                base_ptr_pong, lds_pong_offset, _elem_type(), shape=(tile_m * tile_k,)
            ).get()
            lds_a_ping = SmemPtr(
                base_ptr_ping, lds_ping_offset, _elem_type(), shape=(tile_m * tile_k,)
            ).get()

            if use_cshuffle_epilog:
                lds_out = SmemPtr(
                    base_ptr_pong, lds_pong_offset, T.f16, shape=(tile_m * tile_n,)
                ).get()
            else:
                lds_out = None
        else:
            lds_a_ptr = SmemPtr(
                base_ptr_pong, lds_alloc_offset, _elem_type(), shape=(lds_total_elems,)
            )
            lds_a_pong = lds_a_ptr.get()
            lds_a_ping = lds_a_pong
            lds_out = (
                SmemPtr(base_ptr_pong, lds_alloc_offset, T.f16, shape=(tile_m * tile_n,)).get()
                if use_cshuffle_epilog
                else None
            )

        # ---- Buffer resources (runtime byte sizes for OOB protection) ----
        _a_nrec = arith.index_cast(T.i64, c_m * (K * elem_bytes))
        _c_nrec = arith.index_cast(T.i64, c_m * c_n * 2)
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False,
                                                   num_records_bytes=_a_nrec)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False,
                                                   num_records_bytes=_c_nrec)
        scale_a_rsrc = None if is_f16_or_bf16 else buffer_ops.create_buffer_resource(
            arg_scale_a, max_size=False,
            num_records_bytes=arith.index_cast(T.i64, c_m * 4))
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        scale_b_rsrc = None if is_f16_or_bf16 else buffer_ops.create_buffer_resource(
            arg_scale_b, max_size=True)

        bx_m = bx * tile_m
        by_n = by * tile_n

        # ---- Wave / lane decomposition ----
        wave_size = 64
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        coord_wave_lane = idx2crd(tx, layout_wave_lane)
        wave_id = layout_get(coord_wave_lane, 0)
        lane_id = layout_get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = idx2crd(lane_id, layout_lane16)
        lane_div_16 = layout_get(coord_lane16, 0)
        lane_mod_16 = layout_get(coord_lane16, 1)

        row_a_lds = lane_mod_16
        kpack_elems = 16 if elem_bytes == 1 else 8
        col_offset_base = lane_div_16 * kpack_elems
        col_offset_base_bytes = (
            col_offset_base if elem_bytes == 1 else col_offset_base * elem_bytes
        )

        m_repeat = tile_m // 16
        k_unroll = tile_k_bytes // 64

        num_waves = 4
        n_per_wave = tile_n // num_waves
        num_acc_n = n_per_wave // 16

        n_tile_base = wave_id * n_per_wave

        n_intra_list = []
        n_blk_list = []
        for i in range_constexpr(num_acc_n):
            global_n = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n / 16)
            n_intra_list.append(global_n % 16)

        # ── B load helpers ────────────────────────────────────────────────
        def load_b_pack(base_k, ki_step, ni):
            return load_b_pack_k32(
                buffer_ops, arith, vector,
                arg_b=arg_b, b_rsrc=b_rsrc, layout_b=layout_b,
                base_k=base_k, ki_step=ki_step,
                n_blk=n_blk_list[ni], n_intra=n_intra_list[ni],
                lane_div_16=lane_div_16,
                elem_type=_elem_type(), kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes, unpack_int4=is_int4,
            )

        c64_b = 64
        c0_idx = 0

        def load_b_packs_k64(base_k, ku: int, ni: int):
            if is_int4:
                ki0 = (ku * 2) + 0
                ki1 = (ku * 2) + 1
                return load_b_pack(base_k, ki0, ni), load_b_pack(base_k, ki1, ni)

            base_k_bytes = base_k * elem_bytes
            k0_base = base_k_bytes / c64_b
            k0 = k0_base + ku
            k1 = lane_div_16
            coord_pack = fx.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], c0_idx)
            idx_pack = crd2idx(coord_pack, layout_b)
            vec_elems = 16 if elem_bytes == 1 else 8
            b16 = _buffer_load_vec(
                buffer_ops, vector, b_rsrc, idx_pack,
                elem_type=_elem_type(), vec_elems=vec_elems,
                elem_bytes=elem_bytes, offset_in_bytes=(elem_bytes == 1),
            )
            b_i64x2 = vector.bitcast(T.i64x2, b16)
            b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
            b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])

            if not is_f16_or_bf16:
                return b0_i64, b1_i64

            b0_v1 = vector.from_elements(T.vec(1, T.i64), [b0_i64])
            b1_v1 = vector.from_elements(T.vec(1, T.i64), [b1_i64])
            if is_f16:
                return vector.bitcast(T.f16x4, b0_v1), vector.bitcast(T.f16x4, b1_v1)
            return vector.bitcast(T.i16x4, b0_v1), vector.bitcast(T.i16x4, b1_v1)

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

        # ── A LDS load/store helpers (now take lds_buffer memref directly) ──
        lds_base_zero = arith.index(0)

        def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
            col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            col_base_swz = col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes / 2)
            coord_a16 = fx.make_coord(curr_row_a_lds, col_base_swz)
            idx_a16 = crd2idx(coord_a16, layout_lds)
            return vector.load_op(_vec16_type(), lds_buffer, [idx_a16])

        def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
            loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
            a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

            if not is_f16_or_bf16:
                return a0_i64, a1_i64

            a0_v1 = vector.from_elements(T.vec(1, T.i64), [a0_i64])
            a1_v1 = vector.from_elements(T.vec(1, T.i64), [a1_i64])
            if is_f16:
                return vector.bitcast(T.f16x4, a0_v1), vector.bitcast(T.f16x4, a1_v1)
            return vector.bitcast(T.i16x4, a0_v1), vector.bitcast(T.i16x4, a1_v1)

        # ── A global→reg load ─────────────────────────────────────────────
        num_a_loads = bytes_per_thread_a // a_load_bytes
        tile_k_dwords = (tile_k * 2) // 4 if elem_bytes == 2 else tile_k // 4
        layout_a_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
        c4 = arith.index(4)
        tx_i32_base = tx * c4

        def load_a_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops, vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=a_rsrc, vec_elems=(16 if elem_bytes == 1 else 8),
                elem_bytes=elem_bytes,
            )

        def a_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                arith, tx_i32_base=tx_i32_base, i=i,
                total_threads=total_threads, layout_tile_div4=layout_a_tile_div4,
            )

        def load_a_tile(base_k_div4):
            parts = []
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                row_a_global = bx_m + row_a_local
                idx_i32 = row_a_global * _k_div4_factor + (base_k_div4 + col_a_local_i32)
                idx_elem = idx_i32 if elem_bytes == 1 else idx_i32 * 2
                a_16B = load_a_16(idx_elem)
                parts.append(vector.bitcast(T.i32x4, a_16B))
            return parts

        def store_a_tile_to_lds(vec_a_parts, lds_buffer):
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                lds_store_16b_xor16(
                    arith, vector,
                    lds_memref=lds_buffer, vec16_ty=_vec16_type(),
                    layout_lds=layout_lds, row_local=row_a_local,
                    col_local_i32=col_a_local_i32, tx_c4=c4,
                    k_blocks16=k_blocks16, lds_base=lds_base_zero,
                    vec_part_i32x4=vec_a_parts[i], elem_bytes=elem_bytes,
                )

        # ── A DMA async: direct global→LDS transfer ─────────────────────
        num_a_async_loads = bytes_per_thread_a // a_async_load_bytes
        tx_i32_async_base = tx * a_async_load_dword
        k_bytes_factor = K * elem_bytes
        layout_a_bytes = fx.make_layout((tile_m, k_bytes_factor), (k_bytes_factor, 1))

        def a_tile_chunk_coord_i32_async(i: int):
            return tile_chunk_coord_i32(
                arith, tx_i32_base=tx_i32_async_base, i=i,
                total_threads=total_threads, layout_tile_div4=layout_a_tile_div4,
                chunk_i32=a_async_load_dword,
            )

        def dma_a_tile_to_lds(base_k_div4, lds_buffer):
            from flydsl._mlir.dialects import llvm, memref as memref_dialect

            dma_bytes = a_async_load_bytes

            for i in range_constexpr(num_a_async_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32_async(i)
                col_a_local_sw = swizzle_xor16(row_a_local, col_a_local_i32 * c4, k_blocks16)
                row_a_global = bx_m + row_a_local
                global_byte_idx = row_a_global * k_bytes_factor + (base_k_div4 * c4 + col_a_local_sw)
                global_offset = arith.index_cast(T.i32, global_byte_idx)

                if i == 0:
                    lds_addr = memref_dialect.extract_aligned_pointer_as_index(lds_buffer) + wave_id * wave_size * dma_bytes
                    lds_ptr_i64_lane0 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                else:
                    lds_ptr_i64_lane0 += total_threads * dma_bytes
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64_lane0)

                size_i32 = arith.constant(dma_bytes, type=T.i32)
                soffset = arith.constant(0, type=T.i32)
                offset_imm = arith.constant(0, type=T.i32)
                aux = arith.constant(1, type=T.i32)

                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    lds_ptr,
                    _raw(size_i32),
                    _raw(global_offset),
                    _raw(soffset),
                    _raw(offset_imm),
                    _raw(aux),
                )

        def prefetch_a_to_lds(base_k, lds_buffer):
            base_k_div4 = base_k / 4
            dma_a_tile_to_lds(base_k_div4, lds_buffer)

        def prefetch_a_tile(base_k):
            base_k_bytes = base_k * elem_bytes
            base_k_div4 = base_k_bytes / 4
            return load_a_tile(base_k_div4)

        def prefetch_b_tile(base_k):
            return load_b_tile(base_k)

        def prefetch_ab_tile(base_k):
            a_regs = prefetch_a_tile(base_k)
            b_regs = prefetch_b_tile(base_k)
            return a_regs, b_regs

        # ── Compute tile (MFMA) ───────────────────────────────────────────
        def compute_tile(accs_in, b_tile_in, lds_buffer, *, is_last_tile=False, a0_prefetch=None):
            scales_pf = {}
            if is_last_tile and (not is_f16_or_bf16):
                s_b_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = by_n + n_tile_base + (ni * 16) + lane_mod_16
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

            use_mfma_scale_128 = (
                str(gpu_arch).startswith("gfx95")
                and (not is_int8) and (not is_int4) and (not is_f16_or_bf16)
            )
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
                    col_base0 = col_offset_base_bytes + (ku0 * 64)
                    col_base1 = col_offset_base_bytes + (ku1 * 64)

                    for mi in range_constexpr(m_repeat):
                        curr_row_a_lds = row_a_lds + (mi * 16)
                        if (a0_prefetch is not None) and (ku0 == 0) and (mi == 0):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_buffer)
                        a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_buffer)
                        a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                        for ni in range_constexpr(num_acc_n):
                            b128 = pack_i64x4_to_i32x8(
                                b0_packs0[ni], b0_packs1[ni],
                                b1_packs0[ni], b1_packs1[ni],
                            )
                            acc_idx = mi * num_acc_n + ni
                            current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                mfma_res_ty,
                                [a128, b128, current_accs_list[acc_idx],
                                 0, 0, 0, 0x3F800000, 0, 0x3F800000],
                            )
                return current_accs_list, scales_pf

            mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
            if is_int8:
                mfma_fn = mfma_i32_k32
            elif is_f16:
                mfma_fn = rocdl.mfma_f32_16x16x16f16
            elif is_bf16:
                mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
            else:
                mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

            def mfma_step(acc_in, a, b):
                return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

            def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                acc_mid = mfma_step(acc_in, a0, b0)
                return mfma_step(acc_mid, a1, b1)

            for ku in range_constexpr(k_unroll):
                b_packs0, b_packs1 = b_tile_in[ku]
                ki64 = ku * 64
                col_base = col_offset_base_bytes + ki64
                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + (mi * 16)
                    if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer)
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        current_accs_list[acc_idx] = mfma_k64_bytes(
                            current_accs_list[acc_idx], a0, a1,
                            b_packs0[ni], b_packs1[ni],
                        )
            return current_accs_list, scales_pf

        # ── Epilogue (store output) ───────────────────────────────────────
        vec1_f16 = T.vec(1, T.f16)
        vec2_f16 = T.f16x2
        vec1_i16 = T.vec(1, T.i16)
        vec2_i16 = T.i16x2
        vec1_i32 = T.vec(1, T.i32)

        def store_output(final_accs, scales):
            if is_f16_or_bf16:
                s_b_vals = None
                s_a_vecs = None
            else:
                s_b_vals = scales["s_b_vals"]
                s_a_vecs = scales["s_a_vecs"]

            if use_cshuffle_epilog:
                if lds_out is None:
                    raise RuntimeError("use_cshuffle_epilog=True but lds_out is not allocated.")
                gpu.barrier()

                def write_row_to_lds(*, mi, ii, row_in_tile, row, row_base_lds,
                                     col_base_local, num_acc_n, lds_out):
                    if not is_f16_or_bf16:
                        s_a_vec4 = s_a_vecs[mi]
                        s_a = vector.extract(s_a_vec4, static_position=[ii], dynamic_position=[])
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                        if is_int8:
                            val = arith.sitofp(T.f32, val)
                        if is_f16_or_bf16:
                            val_s = val
                        else:
                            val_s = (val * s_a) * s_b_vals[ni]
                        v16 = arith.trunc_f(T.f16, val_s)

                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(vec1_f16, [v16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    idx_out = row * c_n + col_g0
                    byte_off = idx_out * 2
                    e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                    if e_vec == 4:
                        frag_i32x2 = vector.bitcast(T.vec(2, T.i32), frag)
                        buffer_ops.buffer_store(frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True)
                    else:
                        frag_i32x1 = vector.bitcast(T.vec(1, T.i32), frag)
                        frag_i32 = vector.extract(frag_i32x1, static_position=[0], dynamic_position=[])
                        buffer_ops.buffer_store(frag_i32, c_rsrc, byte_off, offset_is_bytes=True)

                e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                mfma_epilog(
                    use_cshuffle=True, arith=arith, vector=vector, gpu=gpu,
                    range_constexpr=range_constexpr, tile_m=tile_m, tile_n=tile_n,
                    e_vec=e_vec, m_repeat=m_repeat, num_acc_n=num_acc_n, tx=tx,
                    lane_div_16=lane_div_16, lane_mod_16=lane_mod_16,
                    bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base, lds_out=lds_out,
                    write_row_to_lds=write_row_to_lds, store_pair=store_pair,
                )
                return

            def body_row(*, mi, ii, row_in_tile, row):
                if not is_f16_or_bf16:
                    s_a_vec4 = s_a_vecs[mi]
                    s_a = vector.extract(s_a_vec4, static_position=[ii], dynamic_position=[])
                col_base = by_n + n_tile_base + lane_mod_16
                idx_base = row * c_n + col_base
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                    if is_int8:
                        val = arith.sitofp(T.f32, val)
                    if is_f16_or_bf16:
                        val_s = val
                    else:
                        val_s = (val * s_a) * s_b_vals[ni]
                    val_f16 = arith.trunc_f(T.f16, val_s)
                    idx_out = idx_base + (ni * 16)
                    buffer_ops.buffer_store(val_f16, c_rsrc, idx_out)

            mfma_epilog(
                use_cshuffle=False, arith=arith, range_constexpr=range_constexpr,
                m_repeat=m_repeat, lane_div_16=lane_div_16, bx_m=bx_m, body_row=body_row,
            )

        # ── Scheduling hints ──────────────────────────────────────────────
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
                if (not use_async_copy) and (sche_i >= dswr_start - 1):
                    rocdl.sched_dswr(1)
            rocdl.sched_barrier(0)

        # ── Main pipeline ─────────────────────────────────────────────────
        def _flatten_b_tile(bt):
            flat = []
            for packs0, packs1 in bt:
                flat.extend(packs0)
                flat.extend(packs1)
            return flat

        def _unflatten_b_tile(flat):
            bt = []
            idx = 0
            for _ in range_constexpr(k_unroll):
                p0 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                p1 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                bt.append((p0, p1))
            return bt

        n_accs = num_acc_n * m_repeat
        n_btile = k_unroll * 2 * num_acc_n
        n_a0pf = 2

        def _pack_state(accs_l, bt_flat, a0pf):
            return list(accs_l) + list(bt_flat) + [a0pf[0], a0pf[1]]

        def _unpack_state(vals):
            accs_l = list(vals[:n_accs])
            bt_flat = list(vals[n_accs:n_accs + n_btile])
            a0pf = (vals[n_accs + n_btile], vals[n_accs + n_btile + 1])
            return accs_l, bt_flat, a0pf

        def _build_pingpong_body(k_iv, inner_state):
            accs_in, bt_flat_in, a0pf_in = _unpack_state(inner_state)
            b_tile_pong_in = _unflatten_b_tile(bt_flat_in)

            next_k1 = k_iv + tile_k
            b_tile_ping = prefetch_b_tile(next_k1)
            if use_async_copy:
                prefetch_a_to_lds(next_k1, lds_a_ping)
            else:
                a_regs_ping = prefetch_a_tile(next_k1)
            accs_in, _ = compute_tile(accs_in, b_tile_pong_in, lds_a_pong,
                                      a0_prefetch=a0pf_in)
            if not use_async_copy:
                store_a_tile_to_lds(a_regs_ping, lds_a_ping)
            hot_loop_scheduler()
            gpu.barrier()
            a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

            next_k2 = k_iv + (tile_k * 2)
            b_tile_pong_new = prefetch_b_tile(next_k2)
            if use_async_copy:
                prefetch_a_to_lds(next_k2, lds_a_pong)
            else:
                a_regs_pong = prefetch_a_tile(next_k2)
            accs_in, _ = compute_tile(accs_in, b_tile_ping, lds_a_ping,
                                      a0_prefetch=a0_prefetch_ping)
            if not use_async_copy:
                store_a_tile_to_lds(a_regs_pong, lds_a_pong)
            hot_loop_scheduler()
            gpu.barrier()
            a0_prefetch_pong_new = prefetch_a0_pack(lds_a_pong)

            return _pack_state(accs_in, _flatten_b_tile(b_tile_pong_new),
                               a0_prefetch_pong_new)

        if lds_stage == 2:
            def prefetch_a0_pack(lds_buffer):
                return lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)

            k0 = arith.index(0)
            b_tile0 = prefetch_b_tile(k0)
            if use_async_copy:
                prefetch_a_to_lds(k0, lds_a_pong)
            else:
                store_a_tile_to_lds(prefetch_a_tile(k0), lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

            num_tiles = K // tile_k
            if (num_tiles % 2) == 1:
                c_k_main = K - tile_k
                init_state = _pack_state(accs, _flatten_b_tile(b_tile0),
                                         a0_prefetch_pong)
                for iv, inner in range(0, c_k_main, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(iv, inner)
                accs, bt_flat, a0pf = _unpack_state(results)
                b_tile_pong_final = _unflatten_b_tile(bt_flat)
                final_accs, scales = compute_tile(
                    accs, b_tile_pong_final, lds_a_pong,
                    is_last_tile=True, a0_prefetch=a0pf,
                )
            else:
                c_k_stop = K - (tile_k * 3)
                init_state = _pack_state(accs, _flatten_b_tile(b_tile0),
                                         a0_prefetch_pong)
                for iv, inner in range(0, c_k_stop, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(iv, inner)
                accs, bt_flat, a0pf = _unpack_state(results)
                b_tile_pong_ep = _unflatten_b_tile(bt_flat)

                last_k = arith.index(K - tile_k)
                b_tile_ping = prefetch_b_tile(last_k)
                if use_async_copy:
                    prefetch_a_to_lds(last_k, lds_a_ping)
                else:
                    a_regs_ping = prefetch_a_tile(last_k)
                accs, _ = compute_tile(accs, b_tile_pong_ep, lds_a_pong,
                                       a0_prefetch=a0pf)
                if not use_async_copy:
                    store_a_tile_to_lds(a_regs_ping, lds_a_ping)
                hot_loop_scheduler()
                gpu.barrier()
                a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)
                final_accs, scales = compute_tile(
                    accs, b_tile_ping, lds_a_ping,
                    is_last_tile=True, a0_prefetch=a0_prefetch_ping,
                )
            store_output(final_accs, scales)
        else:
            a_regs0, b_tile0 = prefetch_ab_tile(arith.index(0))
            store_a_tile_to_lds(a_regs0, lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            bt_flat0 = _flatten_b_tile(b_tile0)

            init_state = list(accs) + list(bt_flat0)
            for iv, state in range(0, K - tile_k, tile_k, init=init_state):
                accs_in = list(state[:n_accs])
                bt_flat_in = list(state[n_accs:])
                b_tile_in = _unflatten_b_tile(bt_flat_in)

                next_k = iv + tile_k
                a_next, b_next = prefetch_ab_tile(next_k)
                accs_in, _ = compute_tile(accs_in, b_tile_in, lds_a_pong)
                gpu.barrier()
                store_a_tile_to_lds(a_next, lds_a_pong)
                hot_loop_scheduler()
                gpu.barrier()
                results = yield list(accs_in) + _flatten_b_tile(b_next)

            accs_final = list(results[:n_accs])
            bt_final = _unflatten_b_tile(list(results[n_accs:]))
            final_accs, scales = compute_tile(
                accs_final, bt_final, lds_a_pong, is_last_tile=True
            )
            store_output(final_accs, scales)

    # ── Host launcher ──────────────────────────────────────────────────────
    _cache_tag = (in_dtype, K, lds_stage, use_cshuffle_epilog)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + (tile_m - 1)) / tile_m)
        gy = _raw(idx_n / tile_n)

        kernel_gemm(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_gemm


__all__ = ["compile_preshuffle_gemm_a8"]
