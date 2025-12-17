#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using Rocir with B preshuffle (variable tiling) - Optimized Torch Version."""

import sys
import os
import logging
import functools

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)

import rocdsl
from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import rocdsl.dialects.ext.rocir as rocir
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco, pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest
import torch
import torch.nn.functional as F
import pytest
from _mlir import ir
from _mlir.dialects import vector, memref, builtin, llvm
from rocdsl.dialects.ext import arith, scf, gpu, buffer_ops
from _mlir.dialects import arith as _arith_mlir
import _mlir.dialects.rocdl as rocdl
import _mlir.extras.types as T
from hip import hip
import ctypes

# Aiter imports
try:
    import aiter
    from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight

    HAS_AITER = True
except ImportError:
    print("Warning: Aiter not found, skipping comparison")
    HAS_AITER = False


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    """
    Torch reference implementation (from aiter project).
    Dequantize FP8 inputs and compute FP32 matmul.
    """
    x = x.to(torch.float32) * x_scale
    weight = weight.to(torch.float32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias.dtype) + bias
    return out.to(dtype)


def unwrap(v):
    if isinstance(v, int):
        return arith.constant(v, index=True).value
    while hasattr(v, "value") or hasattr(v, "_value"):
        if hasattr(v, "_value"):
            v = v._value
        elif hasattr(v, "value"):
            v = v.value
    return v


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k", [(1024, 7168, 2048, 128, 128, 128)]
)
def test_mfma_fp8_rocir_preshuffle(M, N, K, tile_m, tile_n, tile_k):
    print("=" * 80)
    print(f"MFMA FP8 GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k}) [Torch Optimized]")
    print("=" * 80)
    gpu_arch = get_hip_arch()
    ctx = RAIIMLIRContextModule()
    try:
        import _rocirPassesExt

        _rocirPassesExt.register_dialect(ctx.module.context)
        print("✓ Registered Rocir dialect")
    except Exception as e:
        print(f"Warning: Could not register Rocir dialect: {e}")

    f8 = ir.Float8E4M3FNType.get()
    f32 = ir.F32Type.get()

    size_c = M * N
    size_a = M * K
    size_b = N * K

    # Vector width calc
    # We launch one wave per 16 columns of N (MFMA produces 16x16 per op).
    # For small tile_n (e.g. 32), using a fixed 4-wave block would make n_per_wave < 16,
    # leading to num_acc_n==0 and a no-op kernel. Make waves/thread-block size depend on tile_n.
    num_waves = max(1, min(4, tile_n // 16))
    assert (
        tile_n % num_waves == 0
    ), f"tile_n({tile_n}) must be divisible by num_waves({num_waves})"
    assert (tile_n // num_waves) % 16 == 0, "n_per_wave must be a multiple of 16"
    total_threads = num_waves * 64
    elems_a_per_tile = tile_m * tile_k
    elems_per_thread_a = elems_a_per_tile // total_threads
    # Force min 16 bytes to avoid small vector load issues?
    # NO: If tile is small (16x128), 16 bytes load means we load 4096 bytes (2x tile size).
    # Threads 128-255 would write OOB to LDS (allocated for 2048).
    # So we must use exact size.
    bytes_per_thread_a = elems_per_thread_a
    vec_width_a_i32 = bytes_per_thread_a // 4

    pad_k = 0  # Padding to avoid bank conflicts (stride 136 bytes -> bank inc 2)
    lds_stride = tile_k + pad_k

    allocator = SmemAllocator(ctx, arch=gpu_arch)
    # Double buffering: allocate 2x size
    lds_a_decl = allocator.allocate_array(f8, 2 * tile_m * lds_stride)

    @gpu.module(
        "mfma_mod",
        [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ],
    )
    def gpu_mod():
        allocator.finalize()

        @gpu.func(emit=True)
        def kernel_fixed(
            arg_c: T.memref(size_c, T.f16()),
            arg_a: T.memref(size_a, f8),
            arg_b: T.memref(size_b, f8),
            arg_scale_a: T.memref(M, T.f32()),
            arg_scale_b: T.memref(N, T.f32()),
            m_in: T.index(),
            n_in: T.index(),
            k_in: T.index(),
        ):
            c_m = m_in
            c_n = n_in
            c_k = k_in
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            c_tile_k = arith.constant(tile_k, index=True)

            i32_type = ir.IntegerType.get_signless(32)
            index_type = ir.IndexType.get()
            vec4_f32 = ir.VectorType.get([4], f32)
            vec8_f8 = ir.VectorType.get([8], f8)
            vec16_f8 = ir.VectorType.get([16], f8)
            vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
            vec2_i64 = ir.VectorType.get([2], ir.IntegerType.get_signless(64))
            vec2_i32 = ir.VectorType.get([2], i32_type)
            vec4_i32 = ir.VectorType.get([4], i32_type)

            vec_a_load_len = bytes_per_thread_a

            zero_attr = ir.DenseElementsAttr.get_splat(
                vec4_f32, ir.FloatAttr.get(f32, 0.0)
            )
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result

            layout_a = rocir.make_layout((c_m, c_k), stride=(c_k, 1))
            layout_c = rocir.make_layout((c_m, c_n), stride=(c_n, 1))

            c2 = arith.constant(2, index=True)
            c0_i32 = arith.i32(0)

            c32 = arith.constant(32, index=True)
            c16 = arith.constant(16, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c512 = arith.constant(512, index=True)
            c64 = arith.constant(64, index=True)
            c1024 = arith.constant(1024, index=True)

            def swizzle_idx(row_idx, col_idx):
                c16 = arith.constant(16, index=True)
                k_blocks16 = arith.constant(tile_k // 16, index=True)
                row_mod = _arith_mlir.RemUIOp(
                    unwrap(row_idx), unwrap(k_blocks16)
                ).result
                xor_mask = _arith_mlir.MulIOp(unwrap(row_mod), unwrap(c16)).result
                return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

            c_k0 = _arith_mlir.DivUIOp(unwrap(c_k), unwrap(c64)).result
            stride_n0 = _arith_mlir.MulIOp(unwrap(c_k0), unwrap(c1024)).result
            stride_b = (
                stride_n0,  # n0
                c1024,  # k0
                arith.constant(256, index=True),  # k1 (KLane)
                c16,  # n1
                arith.constant(1, index=True),  # k2
            )
            # Shape: (N0, K0, KLane, NLane, KPack)
            layout_b = rocir.make_layout(
                (
                    _arith_mlir.DivUIOp(unwrap(c_n), unwrap(c16)).result,
                    c_k0,
                    arith.constant(4, index=True),
                    c16,
                    c16,
                ),
                stride=stride_b,
            )

            shape_lds = rocir.make_shape(tile_m, tile_k)
            stride_lds = rocir.make_stride(lds_stride, 1)
            layout_lds = rocir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr = allocator.get_base()
            lds_a = lds_a_decl(base_ptr).get()

            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)

            tx_idx = _arith_mlir.IndexCastOp(index_type, unwrap(tx)).result
            vec_len_val = arith.constant(vec_a_load_len, index=True)
            linear_id = _arith_mlir.MulIOp(unwrap(tx_idx), unwrap(vec_len_val)).result

            c_tile_k_val = arith.constant(tile_k, index=True)
            row_a_local = _arith_mlir.DivUIOp(
                unwrap(linear_id), unwrap(c_tile_k_val)
            ).result
            col_a_local = _arith_mlir.RemUIOp(
                unwrap(linear_id), unwrap(c_tile_k_val)
            ).result

            bx_m = _arith_mlir.MulIOp(
                unwrap(bx), unwrap(arith.constant(tile_m, index=True))
            ).result
            row_a_global = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_a_local)).result
            by_n = _arith_mlir.MulIOp(
                unwrap(by), unwrap(arith.constant(tile_n, index=True))
            ).result

            coord_store = rocir.make_coord(unwrap(row_a_local), unwrap(col_a_local))
            lds_write_idx = rocir.crd2idx(coord_store, layout_lds)

            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16

            # MFMA operand mapping: match the known-good lane mapping (see `t1.py`)
            row_a_lds = lane_mod_16
            # CK mapping: lane_div_16 selects k1 (0..3), each k1 spans 16 fp8 within a 64-wide K block.
            col_offset_base = _arith_mlir.MulIOp(
                unwrap(lane_div_16), unwrap(c16)
            ).result

            row_b_lds = lane_mod_16

            c2 = arith.constant(2, index=True)
            c0_i32 = arith.i32(0)

            coord_a_base = rocir.make_coord(unwrap(row_a_global), unwrap(col_a_local))
            idx_a_base = rocir.crd2idx(coord_a_base, layout_a)
            idx_a_base_div4 = _arith_mlir.DivUIOp(unwrap(idx_a_base), unwrap(c4)).result

            m_repeat = tile_m // 16
            # K64 micro-step: two MFMA(x32) per step
            k_unroll = tile_k // 64

            lds_a_indices = []

            # --- Dynamic Tiling Logic ---
            # `num_waves` is chosen at Python time based on tile_n (see above).
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            c_num_waves = arith.constant(num_waves, index=True)
            wave_mod = _arith_mlir.RemUIOp(unwrap(wave_id), unwrap(c_num_waves)).result
            n_tile_base = _arith_mlir.MulIOp(
                unwrap(wave_mod), unwrap(c_n_per_wave)
            ).result

            # Global N calc loop
            n_intra_list = []
            n_blk_list = []

            for i in range(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)

                # global_n = by_n + n_tile_base + offset + row_b_lds
                tmp1 = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                global_n = _arith_mlir.AddIOp(unwrap(tmp2), unwrap(row_b_lds)).result

                n_intra = _arith_mlir.RemUIOp(unwrap(global_n), unwrap(c16)).result
                n_blk = _arith_mlir.DivUIOp(unwrap(global_n), unwrap(c16)).result

                n_intra_list.append(n_intra)
                n_blk_list.append(n_blk)

            for mi in range(m_repeat):
                mi_val = arith.constant(mi * 16, index=True)
                curr_row_a_lds = row_a_lds + mi_val

                for ki_step in range(k_unroll):
                    # Each MFMA step advances K by 32 for fp8 x32
                    ki = ki_step * 32
                    ki_val = arith.constant(ki, index=True)

                    col_lds = col_offset_base + ki_val
                    coord_a_lds = rocir.make_coord(
                        unwrap(curr_row_a_lds), unwrap(col_lds)
                    )
                    idx_a_mfma = rocir.crd2idx(coord_a_lds, layout_lds)
                    idx_a_idx = _arith_mlir.IndexCastOp(
                        index_type, unwrap(idx_a_mfma)
                    ).result
                    lds_a_indices.append(idx_a_idx)

            acc_inits = [acc_init] * (num_acc_n * m_repeat)

            # --- B Load Logic ---
            def load_b_tile(base_k):
                b_packs = []
                k0_base = _arith_mlir.DivUIOp(unwrap(base_k), unwrap(c64)).result
                k1 = lane_div_16  # 0..3
                k2_base = c0

                for i in range(num_acc_n):
                    n_intra = n_intra_list[i]
                    n_blk = n_blk_list[i]
                    for ki_step in range(k_unroll):
                        k0 = _arith_mlir.AddIOp(
                            unwrap(k0_base), unwrap(arith.constant(ki_step, index=True))
                        ).result
                        coord_b = rocir.make_coord(n_blk, k0, k1, n_intra, k2_base)
                        idx_bytes = rocir.crd2idx(coord_b, layout_b)
                        idx_i32 = _arith_mlir.DivUIOp(
                            unwrap(idx_bytes), unwrap(c4)
                        ).result

                        b16 = buffer_ops.buffer_load(
                            b_rsrc, idx_i32, vec_width=4, dtype=i32_type
                        )
                        b_vec128 = vector.BitCastOp(vec2_i64, b16).result
                        b0_pack = unwrap(
                            vector.ExtractOp(
                                b_vec128, static_position=[0], dynamic_position=[]
                            ).result
                        )
                        b1_pack = unwrap(
                            vector.ExtractOp(
                                b_vec128, static_position=[1], dynamic_position=[]
                            ).result
                        )
                        b_packs.append(b0_pack)
                        b_packs.append(b1_pack)
                return b_packs

            # Split A loads logic
            max_bytes_per_load = 16
            num_a_loads = (
                bytes_per_thread_a + max_bytes_per_load - 1
            ) // max_bytes_per_load

            vec_a_parts_types = []
            vec_a_parts_lens = []

            remaining_bytes = bytes_per_thread_a
            for i in range(num_a_loads):
                curr_bytes = min(remaining_bytes, max_bytes_per_load)
                vec_a_parts_lens.append(curr_bytes)
                # Force global dwordx4 loads: carry vector<4xi32> through the loop state.
                # For parts < 16B (e.g. 8B), we still load 16B but only store/use the first curr_bytes.
                vec_a_parts_types.append(vec4_i32)
                remaining_bytes -= curr_bytes

            # Helper to load A (split)
            def load_a_split(idx_div4):
                parts = []
                curr_off_i32 = 0
                for i in range(num_a_loads):
                    curr_bytes = vec_a_parts_lens[i]

                    curr_idx = idx_div4
                    if curr_off_i32 > 0:
                        curr_idx = _arith_mlir.AddIOp(
                            unwrap(idx_div4),
                            unwrap(arith.constant(curr_off_i32, index=True)),
                        ).result

                    # Force global dwordx4 load (16B).
                    val = buffer_ops.buffer_load(
                        a_rsrc, curr_idx, vec_width=4, dtype=i32_type
                    )
                    parts.append(val)
                    # Advance by the *logical* i32s covered by this part (can be < 4).
                    curr_off_i32 += curr_bytes // 4
                return parts

            def store_a_to_lds(vec_a_in_parts, buffer_idx):
                # buffer_idx: 0 or 1
                # LDS offset = buffer_idx * (tile_m * lds_stride)
                # Note: lds_stride is (tile_k + pad_k)
                # Actually, layout_lds has implicit stride.
                # crd2idx gives index within one tile buffer.
                # We need to add offset for the second buffer.

                buffer_offset_val = _arith_mlir.MulIOp(
                    unwrap(buffer_idx),
                    unwrap(arith.constant(tile_m * lds_stride, index=True)),
                ).result

                curr_store_off = 0
                for i in range(num_a_loads):
                    val_vec = vec_a_in_parts[i]
                    curr_bytes = vec_a_parts_lens[i]
                    curr_i32 = curr_bytes // 4

                    # Addr 0 (offset 0)
                    col_0 = _arith_mlir.AddIOp(
                        unwrap(col_a_local),
                        unwrap(arith.constant(curr_store_off, index=True)),
                    ).result

                    col_swizzled_0 = swizzle_idx(row_a_local, col_0)
                    coord_store_0 = rocir.make_coord(
                        unwrap(row_a_local), unwrap(col_swizzled_0)
                    )
                    idx_0 = rocir.crd2idx(coord_store_0, layout_lds)

                    # Add buffer offset
                    idx_0_final = _arith_mlir.AddIOp(
                        unwrap(idx_0), unwrap(buffer_offset_val)
                    ).result

                    if curr_i32 == 4:
                        val_16 = vector.BitCastOp(vec16_f8, val_vec).result
                        vector.StoreOp(val_16, lds_a, [unwrap(idx_0_final)])
                    elif curr_i32 == 2:
                        val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                        val_8 = vector.BitCastOp(vec8_f8, val_2_i32).result
                        vector.StoreOp(val_8, lds_a, [unwrap(idx_0_final)])
                    else:
                        vec_f8 = ir.VectorType.get([curr_bytes], f8)
                        if curr_bytes <= 4:
                            val_1_i32 = vector.ShuffleOp(val_vec, val_vec, [0]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_1_i32).result
                        else:
                            val_2_i32 = vector.ShuffleOp(
                                val_vec, val_vec, [0, 1]
                            ).result
                            val_f8 = vector.BitCastOp(vec_f8, val_2_i32).result
                        vector.StoreOp(val_f8, lds_a, [unwrap(idx_0_final)])

                    curr_store_off += curr_bytes

            # ----------------------------
            # Double-buffered LDS pipeline (OAI)
            # ----------------------------
            # Prolog: load/store tile 0 into LDS buffer 0, then barrier so all lanes can read.
            vec_a_cur = load_a_split(idx_a_base_div4)
            b_vals_cur = load_b_tile(c0)
            store_a_to_lds(vec_a_cur, c0)
            gpu.barrier()

            # Total tiles is known at Python time for this benchmark.
            assert K % tile_k == 0, f"K({K}) must be divisible by tile_k({tile_k})"
            num_tiles = K // tile_k
            assert num_tiles >= 1

            # Prefetch tile 1 (if exists) into registers for the pipeline.
            if num_tiles > 1:
                k1 = arith.constant(tile_k, index=True)
                k1_div4 = _arith_mlir.DivUIOp(unwrap(k1), unwrap(c4)).result
                idx_a_1_div4 = _arith_mlir.AddIOp(
                    unwrap(idx_a_base_div4), unwrap(k1_div4)
                ).result
                vec_a_next = load_a_split(idx_a_1_div4)
                b_vals_next = load_b_tile(k1)
            else:
                vec_a_next = vec_a_cur
                b_vals_next = b_vals_cur

            # Accumulators live in Python lists of SSA values.
            accs = list(acc_inits)
            buf_read = c0  # 0 or 1 (index type)

            def mfma_tile(accs_in, b_vals_in, buffer_read):
                """Compute one tile using A from LDS[buffer_read] and B from b_vals_in."""
                current_accs_list = list(accs_in)
                buffer_offset_val = _arith_mlir.MulIOp(
                    unwrap(buffer_read),
                    unwrap(arith.constant(tile_m * lds_stride, index=True)),
                ).result

                for ki_step in range(k_unroll):
                    b0_packs = []
                    b1_packs = []
                    for ni in range(num_acc_n):
                        base = ni * (2 * k_unroll)
                        b0_packs.append(b_vals_in[base + (2 * ki_step + 0)])
                        b1_packs.append(b_vals_in[base + (2 * ki_step + 1)])

                    ki = ki_step * 64
                    ki_val = arith.constant(ki, index=True)
                    col_lds0 = col_offset_base + ki_val

                    for mi in range(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val

                        col_lds0_swizzled = swizzle_idx(curr_row_a_lds, col_lds0)
                        coord_a0 = rocir.make_coord(
                            unwrap(curr_row_a_lds), unwrap(col_lds0_swizzled)
                        )
                        idx_a0 = rocir.crd2idx(coord_a0, layout_lds)
                        idx_a0 = _arith_mlir.AddIOp(
                            unwrap(idx_a0), unwrap(buffer_offset_val)
                        ).result
                        idx_a0_idx = _arith_mlir.IndexCastOp(
                            index_type, unwrap(idx_a0)
                        ).result

                        loaded_a16 = vector.LoadOp(
                            vec16_f8, lds_a, [unwrap(idx_a0_idx)]
                        ).result
                        a_vec128 = vector.BitCastOp(vec2_i64, loaded_a16).result
                        a0_pack = vector.ExtractOp(
                            a_vec128, static_position=[0], dynamic_position=[]
                        ).result
                        a1_pack = vector.ExtractOp(
                            a_vec128, static_position=[1], dynamic_position=[]
                        ).result

                        for ni in range(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            curr_acc = current_accs_list[acc_idx]
                            b0_pack = b0_packs[ni]
                            b1_pack = b1_packs[ni]

                            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a0_pack),
                                    unwrap(b0_pack),
                                    unwrap(curr_acc),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result
                            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a1_pack),
                                    unwrap(b1_pack),
                                    unwrap(acc0),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result
                            current_accs_list[acc_idx] = acc1

                return current_accs_list

            # Main tiles (except the last one): compute, then store next A to the other LDS buffer and barrier.
            for t in range(num_tiles - 1):
                accs = mfma_tile(accs, b_vals_cur, buf_read)

                buf_write = _arith_mlir.XOrIOp(unwrap(buf_read), unwrap(c1)).result
                store_a_to_lds(vec_a_next, buf_write)
                gpu.barrier()
                buf_read = buf_write

                # Shift B for next iteration.
                b_vals_cur = b_vals_next

                # Prefetch A/B for tile t+2 (if exists).
                if (t + 2) < num_tiles:
                    k2 = arith.constant((t + 2) * tile_k, index=True)
                    k2_div4 = _arith_mlir.DivUIOp(unwrap(k2), unwrap(c4)).result
                    idx_a_2_div4 = _arith_mlir.AddIOp(
                        unwrap(idx_a_base_div4), unwrap(k2_div4)
                    ).result
                    vec_a_next = load_a_split(idx_a_2_div4)
                    b_vals_next = load_b_tile(k2)

            # Last tile compute (no further store/barrier needed).
            accs = mfma_tile(accs, b_vals_cur, buf_read)

            # --- PREFETCH SCALES (epilogue) ---
            s_b_vals = []
            for ni in range(num_acc_n):
                offset = ni * 16
                c_offset = arith.constant(offset, index=True)
                tmp1 = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                col_g = _arith_mlir.AddIOp(unwrap(tmp2), unwrap(lane_mod_16)).result
                val = buffer_ops.buffer_load(
                    scale_b_rsrc, col_g, vec_width=1, dtype=f32
                )
                s_b_vals.append(val)

            s_a_vecs = []
            row_off_base = lane_div_16 * 4
            for mi in range(m_repeat):
                row_base_m = bx_m + (mi * 16)
                row_g_base = row_base_m + row_off_base
                s_a_vec = buffer_ops.buffer_load(
                    scale_a_rsrc, row_g_base, vec_width=4, dtype=f32
                )
                s_a_vec4 = vector.BitCastOp(vec4_f32, s_a_vec).result
                s_a_vecs.append(s_a_vec4)

            for mi in range(m_repeat):
                row_base_m = bx_m + (mi * 16)
                s_a_vec4 = s_a_vecs[mi]

                for i in range(4):
                    row_off = (lane_div_16 * 4) + i
                    row_g = row_base_m + row_off

                    s_a = vector.ExtractOp(
                        s_a_vec4, static_position=[i], dynamic_position=[]
                    ).result

                    for ni in range(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = accs[acc_idx]

                        val = vector.ExtractOp(acc, [], [i]).result

                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        tmp1 = _arith_mlir.AddIOp(
                            unwrap(by_n), unwrap(n_tile_base)
                        ).result
                        tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                        col_g = _arith_mlir.AddIOp(
                            unwrap(tmp2), unwrap(lane_mod_16)
                        ).result

                        s_b = s_b_vals[ni]

                        val_s = _arith_mlir.MulFOp(unwrap(val), unwrap(s_a)).result
                        val_s = _arith_mlir.MulFOp(unwrap(val_s), unwrap(s_b)).result
                        val_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(val_s)).result

                        idx = rocir.crd2idx(
                            rocir.make_coord(unwrap(row_g), unwrap(col_g)), layout_c
                        )
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx)

    # Request occupancy hint: waves-per-eu=2
    # Use a unique kernel_name so IR/asm dumps don't get overwritten by other tests.
    hsaco = compile_to_hsaco(
        ctx.module, kernel_name="mfma_fp8_preshuffle_oai", waves_per_eu=2
    )
    print(f"✓ Compiled ({len(hsaco)} bytes)")

    grid_x = M // tile_m
    grid_y = N // tile_n

    # --- Torch Data Gen & Baseline (AIter Style) ---
    device = torch.device("cuda")

    # 1. Source Data (FP32)
    torch.manual_seed(42)  # For reproducibility
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(
        N, K, device=device, dtype=torch.float32
    )  # (N, K) for weight

    # 2. Per-token Quantize to FP8 (E4M3)
    a_q_fp8, scale_a = pertoken_quant(
        a_fp32, quant_dtype=torch.float8_e4m3fnuz
    )  # (M, K)
    b_q_fp8, scale_b = pertoken_quant(
        b_fp32_t, quant_dtype=torch.float8_e4m3fnuz
    )  # (N, K)

    # When using forced global dwordx4 (16B) loads, some threads will over-read beyond the
    # logical (M*K) / (N*K) region. Pad the underlying storage so the over-read stays in-bounds.
    # This preserves correctness because we only *use* the required bytes.
    PAD_ELEMS = 64  # bytes for fp8; generous guard for safety
    if a_q_fp8.is_contiguous():
        a_flat = a_q_fp8.view(-1)
    else:
        a_flat = a_q_fp8.contiguous().view(-1)
    a_storage = torch.empty(
        a_flat.numel() + PAD_ELEMS, device=device, dtype=a_q_fp8.dtype
    )
    a_storage[: a_flat.numel()] = a_flat
    a_q_fp8 = a_storage[: a_flat.numel()].view(M, K)

    if b_q_fp8.is_contiguous():
        b_flat = b_q_fp8.view(-1)
    else:
        b_flat = b_q_fp8.contiguous().view(-1)
    b_storage = torch.empty(
        b_flat.numel() + PAD_ELEMS, device=device, dtype=b_q_fp8.dtype
    )
    b_storage[: b_flat.numel()] = b_flat
    b_q_fp8 = b_storage[: b_flat.numel()].view(N, K)

    # Keep the aiter-style preshuffle for optional aiter benchmark at the end.
    b_shuffled = shuffle_weight(b_q_fp8)  # (N, K) -> aiter shuffled layout

    # 4. Compute Baseline using AIter style (dequant + matmul)
    c_ref = run_torch(
        a_q_fp8, b_q_fp8, scale_a, scale_b, bias=None, dtype=torch.float32
    )
    # 5. Run Kernel (f16 output, in-kernel scaling)
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)

    arg_ptrs = [
        ctypes.c_void_p(c_out_raw.data_ptr()),
        ctypes.c_void_p(a_q_fp8.data_ptr()),
        ctypes.c_void_p(b_shuffled.data_ptr()),
        ctypes.c_void_p(scale_a.data_ptr()),
        ctypes.c_void_p(scale_b.data_ptr()),
        ctypes.c_long(M),
        ctypes.c_long(N),
        ctypes.c_long(K),
    ]

    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel_fixed"))

    def launch_kernel(c, a, b, sa, sb):
        arg_ptrs = [
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(sa.data_ptr()),
            ctypes.c_void_p(sb.data_ptr()),
            ctypes.c_long(M),
            ctypes.c_long(N),
            ctypes.c_long(K),
        ]
        current_args_array = (ctypes.c_void_p * 8)(
            *[ctypes.addressof(p) for p in arg_ptrs]
        )

        hip_check(
            hip.hipModuleLaunchKernel(
                kernel_func,
                grid_x,
                grid_y,
                1,
                total_threads,
                1,
                1,
                0,
                0,
                current_args_array,
                None,
            )
        )

    _, us = run_perftest(
        launch_kernel, c_out_raw, a_q_fp8, b_shuffled, scale_a, scale_b
    )
    c_out_scaled = c_out_raw.to(torch.float32)

    # 7. Verify
    # Keep output clean; enable these for debugging.
    # print(f"c_out_scaled: {c_out_scaled}")
    # print(f"c_ref: {c_ref}")
    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)
    # Benchmark
    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K

    tflops = flops / (us / 1e6) / 1e12
    bw = bytes_moved / 1e9 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {bw:.2f} GB/s")

    if HAS_AITER:
        print("-" * 40)
        print("Running Aiter Benchmark...")

        def launch_aiter(a, b, sa, sb):
            return aiter.gemm_a8w8_bpreshuffle(
                a, b, sa, sb, None, torch.float16  # bias
            )

        # Verify Aiter output first
        c_aiter, us1 = run_perftest(launch_aiter, a_q_fp8, b_shuffled, scale_a, scale_b)
        verify_output(c_aiter.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

        tflops_aiter = flops / (us1 / 1e6) / 1e12
        bw_aiter = bytes_moved / 1e9 / (us1 / 1e6)
        print(
            f"Aiter Throughput: {us1:.1f} us, {tflops_aiter:.2f} TFLOPS, BW: {bw_aiter:.2f} GB/s"
        )

        print(
            f"Speedup vs Aiter: {tflops / tflops_aiter:.2f}x, us {us1:.1f} vs {us:.1f}"
        )
        print("-" * 40)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    # Test cases
    print("Running Tiling Tests...")

    # test_mfma_fp8_rocir_preshuffle(1024, 7168, 2048, tile_m=128, tile_n=128, tile_k=128)
    # test_mfma_fp8_rocir_preshuffle(16, 7168, 2048, tile_m=16, tile_n=64, tile_k=256)
    test_mfma_fp8_rocir_preshuffle(16, 7168, 2048, tile_m=16, tile_n=32, tile_k=512)
    # test_mfma_fp8_rocir_preshuffle(16, 7168, 2048, tile_m=16, tile_n=256, tile_k=256)
    # test_mfma_fp8_rocir_preshuffle(1024, 7168, 2048, tile_m=128, tile_n=128, tile_k=128)
    # test_mfma_fp8_rocir_preshuffle(32, 7168, 2048, tile_m=32, tile_n=128, tile_k=256)
