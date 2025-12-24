#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using Rocir with B preshuffle (m1024, K32 micro-step + CK-v3-style IGLP scheduling)."""

import sys
import os
import logging
import functools

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)

import rocdsl
import rocdsl.dialects.ext.rocir as rocir
from rocdsl.dialects.ext.python_control_flow import range_constexpr
from rocdsl.runtime.hip_util import get_hip_arch
from rocdsl.utils import SmemAllocator
from tests.utils import pertoken_quant, shuffle_weight, compile_to_hsaco
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

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

# Aiter imports (optional)
# IMPORTANT: importing aiter can poison the HIP context on some setups and lead to
# `hipErrorNoBinaryForGpu` later. Only import it when explicitly requested.
RUN_AITER_BENCH = os.environ.get("ROCDSL_RUN_AITER_BENCH", "0") == "1"
if RUN_AITER_BENCH:
    try:
        import aiter
        from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight

        HAS_AITER = True
    except ImportError:
        print("Warning: Aiter not found, skipping comparison")
        HAS_AITER = False
else:
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
    "M, N, K, tile_m, tile_n, tile_k", [(1024, 7168, 2048, 128, 128, 64)]
)
def test_mfma_fp8_rocir_preshuffle(M, N, K, tile_m, tile_n, tile_k):
    print("=" * 80)
    print(f"MFMA FP8 GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k}) [Torch Optimized]")
    print("=" * 80)
    gpu_arch = get_hip_arch()

    def _f8():
        return ir.Float8E4M3FNType.get()

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    size_c = M * N
    size_a = M * K
    size_b = N * K

    # Vector width calc
    total_threads = 256
    elems_a_per_tile = tile_m * tile_k
    elems_per_thread_a = elems_a_per_tile // total_threads
    bytes_per_thread_a = elems_per_thread_a
    vec_width_a_i32 = bytes_per_thread_a // 4

    pad_k = 0  # Padding to avoid bank conflicts (stride 136 bytes -> bank inc 2)
    lds_stride = tile_k + pad_k

    class _MFMA(rocir.MlirModule):
        GPU_MODULE_NAME = "mfma_mod"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_a_decl"] = allocator.allocate_array(_f8(), tile_m * lds_stride)
            allocator.finalize()

        @rocir.kernel
        def kernel_fixed(
            self: rocir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            f8 = _f8()
            f32 = ir.F32Type.get()

            c_m = m_in
            c_n = n_in
            c_k = k_in
            c0 = arith.constant(0, index=True)
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

            vec_a_load_len = bytes_per_thread_a # fp8

            zero_attr = ir.DenseElementsAttr.get_splat(
                vec4_f32, ir.FloatAttr.get(f32, 0.0)
            )
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result

            layout_a = rocir.make_layout((c_m, c_k), stride=(c_k, 1))
            layout_c = rocir.make_layout((c_m, c_n), stride=(c_n, 1))

            c0_i32 = arith.i32(0)

            c1 = arith.constant(1, index=True)
            c4 = arith.constant(4, index=True)
            c16 = arith.constant(16, index=True)
            c256 = arith.constant(256, index=True)
            c1024 = arith.constant(1024, index=True)

            c32 = arith.constant(32, index=True)

            c_k0 = c_k / 64
            c_n0 = c_n / 16
            stride_n0 = c_k0 * 1024

            stride_b = (
                stride_n0,  # n0
                c1024,  # k0
                c256,  # k1 (KLane)
                c16,  # n1
                c1,  # k2
            )
            # Shape: (N0, K0, KLane, NLane, KPack)
            layout_b = rocir.make_layout(
                (
                    c_n0,  # N / 16
                    c_k0,  # K / 64
                    c4,
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
            lds_a = _state["lds_a_decl"](base_ptr).get()

            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)

            tx_idx = unwrap(tx)
            vec_len_val = arith.constant(vec_a_load_len, index=True)
            linear_id = tx_idx * vec_len_val

            c_tile_k_val = arith.constant(tile_k, index=True)
            row_a_local = linear_id / c_tile_k_val
            col_a_local = linear_id % c_tile_k_val

            bx_m = bx * tile_m
            row_a_global = bx_m + row_a_local
            by_n = by * tile_n

            coord_store = rocir.make_coord(unwrap(row_a_local), unwrap(col_a_local))
            lds_write_idx = rocir.crd2idx(coord_store, layout_lds)

            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16

            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * 16

            row_b_lds = lane_mod_16

            coord_a_base = rocir.make_coord(unwrap(row_a_global), unwrap(col_a_local))
            idx_a_base = rocir.crd2idx(coord_a_base, layout_a)
            idx_a_base_div4 = idx_a_base / 4

            m_repeat = tile_m // 16
            # K32 micro-step: one MFMA(x32) per step.
            k_unroll = tile_k // 32

            lds_a_indices = []

            # --- Dynamic Tiling Logic ---
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = wave_id % 4
            n_tile_base = wave_mod_4 * c_n_per_wave

            # Global N calc loop
            n_intra_list = []
            n_blk_list = []

            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)

                # global_n = by_n + n_tile_base + offset + row_b_lds
                global_n = by_n + n_tile_base + c_offset + row_b_lds

                n_intra = global_n % 16
                n_blk = global_n / 16

                n_intra_list.append(n_intra)
                n_blk_list.append(n_blk)

            for mi in range_constexpr(m_repeat):
                mi_val = arith.constant(mi * 16, index=True)
                curr_row_a_lds = row_a_lds + mi_val

                for ki_step in range_constexpr(k_unroll):
                    # Each MFMA step advances K by 32 for fp8 x32
                    ki = ki_step * 32
                    ki_val = arith.constant(ki, index=True)

                    col_lds = col_offset_base + ki_val
                    coord_a_lds = rocir.make_coord(
                        unwrap(curr_row_a_lds), unwrap(col_lds)
                    )
                    idx_a_mfma = rocir.crd2idx(coord_a_lds, layout_lds)
                    idx_a_idx = unwrap(idx_a_mfma)
                    lds_a_indices.append(idx_a_idx)

            acc_inits = [acc_init] * (num_acc_n * m_repeat)

            # --- B Load Logic (Global) ---
            def load_b_pack16_val(k_base, k_offset, ni):
                k0 = (k_base + k_offset) / 64
                k1 = lane_div_16
                k2_base = arith.constant(0, index=True)

                n_intra = n_intra_list[ni]
                n_blk = n_blk_list[ni]
                coord_b = rocir.make_coord(n_blk, k0, k1, n_intra, k2_base)
                idx_bytes = rocir.crd2idx(coord_b, layout_b)
                idx_i32 = idx_bytes / 4

                return buffer_ops.buffer_load(b_rsrc, idx_i32, vec_width=4, dtype=i32_type)

            def load_b_tile_flat(k_val):
                b_packs = []
                for ki in range_constexpr(k_unroll):
                    # For each K-step (32), we need 2 B-packs (each covers 16 N).
                    # Actually B loads are per k0/k1.
                    # Each ki_step is 32 K.
                    # k_unroll=4 -> 128 K.
                    # For each ki_step, we have half=0 (0-15) and half=1 (16-31).
                    # Wait, B loading logic in original code was:
                    # k0 = k0_base + (ki_step // 2)
                    # if half==0: load_b_pack16
                    # else: reuse
                    # So we only need to load for even ki_steps?
                    # No, load_b_pack16 loads 16 bytes.
                    # The layout_b has KPack=16.
                    # 16 bytes = 16 elements (fp8).
                    # So one load covers K=16.
                    # ki_step is 32. So we need 2 loads per ki_step?
                    # Original code:
                    # load_b_pack16 called for `ni`.
                    # It loads `vector<4xi32>` = 128 bits = 16 bytes.
                    # Shape B is (N0, K0, KLane, NLane, KPack).
                    # KPack=16.
                    # So one load gets 16 K values.
                    # ki_step covers 32 K.
                    # We need 2 loads to cover 32 K.
                    # But original code loaded once at `half==0` and stored `lo, hi`.
                    # `vec2_i64` is 16 bytes.
                    # So `lo` is 8B, `hi` is 8B.
                    # `lo` is K=0..7? No, FP8 is 1 byte.
                    # `lo` is 8 bytes = 8 elements.
                    # `hi` is 8 bytes = 8 elements.
                    # Total 16 elements (K=0..15).
                    # But ki_step is 32!
                    # If we only load 16 K, where do the other 16 come from?
                    # Ah, `ki_step` loop:
                    # ki_step=0 (K=0..31).
                    #   half=0 (K=0..15?): load.
                    #   half=1 (K=16..31?): reuse.
                    # This implies the load fetches 32 bytes?
                    # `load_b_pack16` loads `vec_width=4` (4xi32 = 16B).
                    # 16B = 16 elements.
                    # So it loads K=0..15.
                    # How does it handle K=16..31?
                    # Original code: `k0 = k0_base + (ki_step // 2)`.
                    # `ki_step` 0 -> k0=base.
                    # `ki_step` 1 -> k0=base.
                    # `ki_step` 2 -> k0=base+1.
                    # `ki_step` 3 -> k0=base+1.
                    # So `ki_step` 0 and 1 share the same `k0`.
                    # BUT `ki_step` 0 is `half=0`, `ki_step` 1 is `half=1`.
                    # The load in `half=0` (ki=0) loads 16B.
                    # It uses `lo` (8B) for `ki=0`? And `hi` (8B) for `ki=1`?
                    # If so, 8B = 8 elements. MFMA needs 8 elements?
                    # `mfma_f32_16x16x32_fp8_fp8`:
                    # A: 16x16x32.
                    # K=32.
                    # Input A: `i64` (8 bytes = 8 elems).
                    # Input B: `i64` (8 bytes = 8 elems).
                    # Wait, MFMA K=32 requires 32 elements?
                    # No, `mfma_f32_16x16x32` consumes 32 K per instruction?
                    # For FP8, `16x16x32`.
                    # 32 is the K-dimension.
                    # Each lane provides a subset?
                    # Wave64.
                    # It seems standard is 8 elems per thread?
                    # Yes.
                    # So we need 8 elems per step.
                    # One 16B load provides 16 elems.
                    # Enough for 2 steps.
                    # So: `load` at `ki=0` (half=0) provides data for `ki=0` and `ki=1`.
                    # `load` at `ki=2` (half=0) provides data for `ki=2` and `ki=3`.
                    # Correct.
                    
                    if (ki % 2) == 0:
                        ki_val = arith.constant(ki * 32, index=True)
                        for ni in range_constexpr(num_acc_n):
                            val = load_b_pack16_val(k_val, ki_val, ni)
                            b_packs.append(val)
                return b_packs

            # Split A loads logic
            max_bytes_per_load = 16
            num_a_loads = (
                bytes_per_thread_a + max_bytes_per_load - 1
            ) // max_bytes_per_load

            vec_a_parts_types = []
            vec_a_parts_lens = []

            remaining_bytes = bytes_per_thread_a
            for i in range_constexpr(num_a_loads):
                curr_bytes = min(remaining_bytes, max_bytes_per_load)
                vec_a_parts_lens.append(curr_bytes)
                vec_a_parts_types.append(vec4_i32)
                remaining_bytes -= curr_bytes

            def load_a_split(idx_div4):
                parts = []
                curr_off_i32 = 0
                for i in range_constexpr(num_a_loads):
                    curr_bytes = vec_a_parts_lens[i]
                    curr_idx = idx_div4
                    if curr_off_i32 > 0:
                        curr_idx = idx_div4 + curr_off_i32
                    val = buffer_ops.buffer_load(
                        a_rsrc, curr_idx, vec_width=4, dtype=i32_type
                    )
                    parts.append(val)
                    curr_off_i32 += curr_bytes // 4
                return parts

            # Initial Loads
            vec_a_inits = load_a_split(idx_a_base_div4)
            vec_b_inits = load_b_tile_flat(c0) # B for K=0

            accs = acc_inits
            vec_a_parts = vec_a_inits
            vec_b_parts = vec_b_inits

            def emit_tile(k_iv, accs_in, vec_a_in_parts, vec_b_in_parts, is_last_tile=False):
                # Store A to LDS
                curr_store_off = 0
                for i in range_constexpr(num_a_loads):
                    val_vec = vec_a_in_parts[i]
                    curr_bytes = vec_a_parts_lens[i]
                    curr_i32 = curr_bytes // 4
                    col_0 = col_a_local + curr_store_off

                    def swizzle_idx(row_idx, col_idx):
                        k_blocks16 = arith.constant(tile_k // 16, index=True)
                        row_mod = row_idx % k_blocks16
                        xor_mask = row_mod * 16
                        return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

                    col_swizzled_0 = swizzle_idx(row_a_local, col_0)
                    coord_store_0 = rocir.make_coord(
                        unwrap(row_a_local), unwrap(col_swizzled_0)
                    )
                    idx_0 = rocir.crd2idx(coord_store_0, layout_lds)

                    if curr_i32 == 4:
                        val_16 = vector.BitCastOp(vec16_f8, val_vec).result
                        vector.StoreOp(val_16, lds_a, [unwrap(idx_0)])
                    elif curr_i32 == 2:
                        val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                        val_8 = vector.BitCastOp(vec8_f8, val_2_i32).result
                        vector.StoreOp(val_8, lds_a, [unwrap(idx_0)])
                    else:
                        vec_f8 = ir.VectorType.get([curr_bytes], f8)
                        if curr_bytes <= 4:
                            val_1_i32 = vector.ShuffleOp(val_vec, val_vec, [0]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_1_i32).result
                        else:
                            val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_2_i32).result
                        vector.StoreOp(val_f8, lds_a, [unwrap(idx_0)])

                    curr_store_off += curr_bytes

                gpu.barrier()

                vec_a_next_parts = vec_a_in_parts
                vec_b_next_parts = []
                scales_pf = {}

                # Calculate indices for Next Tile Prefetch
                next_k = k_iv + c_tile_k
                
                # --- PREFETCH B SCHEDULING ---
                # Total B packs to load = k_unroll/2 * num_acc_n = 2 * 2 = 4 packs?
                # Wait, k_unroll=4. We load at ki=0, ki=2.
                # So 2 * num_acc_n = 4 loads.
                # Total pipeline slots = k_unroll * m_repeat = 4 * 8 = 32.
                # We can issue 1 load every 8 slots.
                b_load_queue = []
                if not is_last_tile:
                    for ki in range_constexpr(k_unroll):
                        if (ki % 2) == 0:
                            ki_val = arith.constant(ki * 32, index=True)
                            for ni in range_constexpr(num_acc_n):
                                # Lambda to delay generation
                                b_load_queue.append((next_k, ki_val, ni))
                
                b_loads_total = len(b_load_queue)
                # Distribute b_loads_total over 32 slots.
                # 4 loads over 32 slots -> 1 load every 8 slots.
                # Make VMEM cadence more CK-like by using an interval+phase schedule
                # (avoid always issuing at slot 0 of each interval).
                total_slots = k_unroll * m_repeat
                b_issue_interval = 1
                b_issue_phase = 0
                if b_loads_total > 0:
                    # Heuristic:
                    # - For very small numbers of loads (tile_k=64 -> b_loads_total=2),
                    #   favor issuing as early as possible to maximize lead time and
                    #   reduce `s_waitcnt vmcnt` at the next tile boundary.
                    # - For larger load counts, shift by half-interval to avoid
                    #   clustering and better match CK-like cadence.
                    b_issue_interval = max(1, total_slots // b_loads_total)
                    if b_loads_total <= 2:
                        b_issue_phase = 0
                    else:
                        b_issue_phase = b_issue_interval // 2
                
                current_accs_list = list(accs_in)
                
                # Helper for A-LDS Load
                # Load 16B (two 8B packs) so we can feed `half=0/1` without issuing
                # two separate LDS reads. This should lower ds_read density and help
                # the scheduler form "2 MFMA -> 1 VMEM" style cadence like CK.
                def load_a_pack16_for_mi(mi_i, col_base_i):
                    mi_val = arith.constant(mi_i * 16, index=True)
                    curr_row = row_a_lds + mi_val

                    def swizzle_idx_ld(row_idx, col_idx):
                        k_blocks16 = arith.constant(tile_k // 16, index=True)
                        row_mod = row_idx % k_blocks16
                        xor_mask = row_mod * 16
                        return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

                    col_base_sw = swizzle_idx_ld(curr_row, col_base_i)
                    coord_a = rocir.make_coord(unwrap(curr_row), unwrap(col_base_sw))
                    idx_a = rocir.crd2idx(coord_a, layout_lds)
                    loaded_a16 = vector.LoadOp(vec16_f8, lds_a, [unwrap(idx_a)]).result
                    return vector.BitCastOp(vec2_i64, loaded_a16).result

                # Unpack B from input regs
                # vec_b_in_parts is flat list of vector<4xi32>
                # Map to [ki/2][ni]
                b_packs_map = {} # Key: ki_idx (0,2), ni -> val
                b_in_idx = 0
                for ki in range_constexpr(k_unroll):
                    if (ki % 2) == 0:
                        for ni in range_constexpr(num_acc_n):
                            val_vec = vec_b_in_parts[b_in_idx]
                            
                            # Cast to i64x2
                            b_vec128 = vector.BitCastOp(vec2_i64, val_vec).result
                            b_lo = unwrap(vector.ExtractOp(b_vec128, static_position=[0], dynamic_position=[]).result)
                            b_hi = unwrap(vector.ExtractOp(b_vec128, static_position=[1], dynamic_position=[]).result)
                            
                            b_packs_map[(ki, ni, 0)] = b_lo
                            b_packs_map[(ki, ni, 1)] = b_hi # for ki+1
                            b_in_idx += 1

                b_next_loaded = []

                # We want LDS A reads to be 16B (ds_read_b128) and reused for 2 halves.
                # Keep the same global-load slot numbering by reconstructing `ki_step`.
                num_k_pairs = k_unroll // 2
                for ki_pair in range_constexpr(num_k_pairs):
                    ki64 = ki_pair * 64
                    ki64_val = arith.constant(ki64, index=True)
                    col_base = col_offset_base + ki64_val

                    # A LDS pipeline: keep one 16B pack "current", prefetch next-mi between halves.
                    rocdl.sched_group_barrier(0x100, 1, 0)
                    a_pack16_cur = load_a_pack16_for_mi(0, col_base)

                    for mi in range_constexpr(m_repeat):
                        a_pack16_next = None

                        # Two halves correspond to two K-steps within the 64B chunk
                        for half in range_constexpr(2):
                            ki_step = (ki_pair * 2) + half
                            a_pack_cur = vector.ExtractOp(
                                a_pack16_cur, static_position=[half], dynamic_position=[]
                            ).result

                            # --- MFMA ---
                            num_pairs = num_acc_n // 2
                            for pi in range_constexpr(num_pairs):
                                ni0 = pi * 2
                                ni1 = ni0 + 1

                                # B from registers
                                base_ki = ki_pair * 2
                                b_pack0 = b_packs_map[(base_ki, ni0, half)]
                                b_pack1 = b_packs_map[(base_ki, ni1, half)]

                                rocdl.sched_group_barrier(0x008, 2, 0)

                                acc_idx0 = mi * num_acc_n + ni0
                                curr_acc0 = current_accs_list[acc_idx0]
                                acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    vec4_f32,
                                    [
                                        unwrap(a_pack_cur),
                                        unwrap(b_pack0),
                                        unwrap(curr_acc0),
                                        unwrap(c0_i32),
                                        unwrap(c0_i32),
                                        unwrap(c0_i32),
                                    ],
                                ).result
                                current_accs_list[acc_idx0] = acc0

                                acc_idx1 = mi * num_acc_n + ni1
                                curr_acc1 = current_accs_list[acc_idx1]
                                acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    vec4_f32,
                                    [
                                        unwrap(a_pack_cur),
                                        unwrap(b_pack1),
                                        unwrap(curr_acc1),
                                        unwrap(c0_i32),
                                        unwrap(c0_i32),
                                        unwrap(c0_i32),
                                    ],
                                ).result
                                current_accs_list[acc_idx1] = acc1

                            # CK-like LDS_READ placement: prefetch A for next-mi *between* halves
                            # to overlap ds_read_b128 latency with the second half's MFMA.
                            if half == 0 and (mi + 1) < m_repeat:
                                rocdl.sched_group_barrier(0x100, 1, 0)
                                a_pack16_next = load_a_pack16_for_mi(mi + 1, col_base)

                            # --- Global Prefetch Injection (B) ---
                            # Place VMEM after the 2 MFMA ops to match CK's hotloop cadence.
                            # Slots: ki_step=0..(k_unroll-1), mi=0..7 -> 0..(k_unroll*m_repeat-1).
                            curr_slot = ki_step * m_repeat + mi
                            # Distribute b_loads_total over (k_unroll*m_repeat) slots.
                            # Using sched_group_barrier(0x020, 1, 0) immediately before load.
                            if b_loads_total > 0 and ((curr_slot - b_issue_phase) % b_issue_interval) == 0:
                                load_idx = (curr_slot - b_issue_phase) // b_issue_interval
                                if load_idx < b_loads_total:
                                    rocdl.sched_group_barrier(0x020, 1, 0)
                                    k_arg, ki_arg, ni_arg = b_load_queue[load_idx]
                                    val = load_b_pack16_val(k_arg, ki_arg, ni_arg)
                                    b_next_loaded.append(val)

                        if (mi + 1) < m_repeat:
                            # Carry prefetched A pack into next mi
                            if a_pack16_next is None:
                                rocdl.sched_group_barrier(0x100, 1, 0)
                                a_pack16_next = load_a_pack16_for_mi(mi + 1, col_base)
                            a_pack16_cur = a_pack16_next

                # End of Tile: Prefetch A for next tile
                if not is_last_tile:
                    next_k_div4 = next_k / 4
                    next_idx_a_div4 = idx_a_base_div4 + next_k_div4
                    vec_a_next_parts = load_a_split(next_idx_a_div4)
                    
                    # Also A-Scales/B-Scales prefetch for last tile...
                    # Original code handled scale prefetch only on last tile.
                else:
                    # Last Tile Logic (prefetch scales)
                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        val = buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=f32)
                        s_b_vals.append(val)
                    scales_pf["s_b_vals"] = s_b_vals

                    s_a_vecs = []
                    row_off_base = lane_div_16 * 4
                    for mi in range_constexpr(m_repeat):
                        row_base_m = bx_m + (mi * 16)
                        row_g_base = row_base_m + row_off_base
                        s_a_vec = buffer_ops.buffer_load(scale_a_rsrc, row_g_base, vec_width=4, dtype=f32)
                        s_a_vec4 = vector.BitCastOp(vec4_f32, s_a_vec).result
                        s_a_vecs.append(s_a_vec4)
                    scales_pf["s_a_vecs"] = s_a_vecs

                gpu.barrier()
                return current_accs_list, vec_a_next_parts, b_next_loaded, scales_pf

            # Main Loop
            c_k_main = c_k - c_tile_k
            for k_iv in range(c0, c_k_main, c_tile_k):
                accs, vec_a_parts, vec_b_parts, _ = emit_tile(
                    k_iv, accs, vec_a_parts, vec_b_parts, is_last_tile=False
                )

            # Epilogue
            final_accs, _, _, scales = emit_tile(
                c_k_main, accs, vec_a_parts, vec_b_parts, is_last_tile=True
            )

            s_b_vals = scales["s_b_vals"]
            s_a_vecs = scales["s_a_vecs"]

            for mi in range_constexpr(m_repeat):
                row_base_m = bx_m + (mi * 16)
                s_a_vec4 = s_a_vecs[mi]
                for i in range_constexpr(4):
                    row_off = (lane_div_16 * 4) + i
                    row_g = row_base_m + row_off
                    s_a = vector.ExtractOp(s_a_vec4, static_position=[i], dynamic_position=[]).result
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.ExtractOp(acc, [], [i]).result
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        s_b = s_b_vals[ni]
                        val_s = val * s_a
                        val_s = val_s * s_b
                        val_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(val_s)).result
                        idx = rocir.crd2idx(rocir.make_coord(unwrap(row_g), unwrap(col_g)), layout_c)
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx)

        @rocir.jit
        def __call__(
            self: rocir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            c1 = arith.constant(1, index=True).value
            bdx = arith.constant(256, index=True).value
            gx = arith.constant(M // tile_m, index=True).value
            gy = arith.constant(N // tile_n, index=True).value

            rocir.gpu_ext.LaunchFuncOp(
                ["mfma_mod", "kernel_fixed"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    unwrap(arg_c),
                    unwrap(arg_a),
                    unwrap(arg_b),
                    unwrap(arg_scale_a),
                    unwrap(arg_scale_b),
                    unwrap(m_in),
                    unwrap(n_in),
                    unwrap(k_in),
                ],
            )

    m = _MFMA()
    try:
        import _rocirPassesExt

        _rocirPassesExt.register_dialect(m.module.context)
        print("✓ Registered Rocir dialect")
    except Exception as e:
        print(f"Warning: Could not register Rocir dialect: {e}")

    if os.environ.get("ROCDSL_DUMP_ASM") == "1":
        print("Dumping assembly via compile_to_hsaco helper...")
        compile_to_hsaco(
            m.module,
            kernel_name=f"mfma_fp8_preshuffle_m1024_32_iglp_tk{tile_k}",
            waves_per_eu=2,
        )

    exe = rocdsl.compile(m)
    print("✓ Compiled")

    grid_x = M // tile_m
    grid_y = N // tile_n

    device = torch.device("cuda")
    torch.manual_seed(42)
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(
        N, K, device=device, dtype=torch.float32
    )

    a_q_fp8, scale_a = pertoken_quant(
        a_fp32, quant_dtype=torch.float8_e4m3fnuz
    )
    b_q_fp8, scale_b = pertoken_quant(
        b_fp32_t, quant_dtype=torch.float8_e4m3fnuz
    )

    PAD_ELEMS = 64
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

    b_shuffled = shuffle_weight(b_q_fp8)

    c_ref = run_torch(
        a_q_fp8, b_q_fp8, scale_a, scale_b, bias=None, dtype=torch.float32
    )
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)

    def launch_kernel(c, a, b, sa, sb):
        exe(c, a, b, sa, sb, M, N, K)

    _, us = run_perftest(
        launch_kernel, c_out_raw, a_q_fp8, b_shuffled, scale_a, scale_b
    )
    torch.cuda.synchronize()
    c_out_scaled = c_out_raw.to(torch.float32)

    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)
    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K

    tflops = flops / (us / 1e6) / 1e12
    bw = bytes_moved / 1e9 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {bw:.2f} GB/s")

    if HAS_AITER and RUN_AITER_BENCH:
        print("-" * 40)
        print("Running Aiter Benchmark...")

        def launch_aiter(a, b, sa, sb):
            return aiter.gemm_a8w8_bpreshuffle(
                a, b, sa, sb, None, torch.float16
            )

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
    elif HAS_AITER and not RUN_AITER_BENCH:
        print("-" * 40)
        print("Skipping Aiter benchmark (set ROCDSL_RUN_AITER_BENCH=1 to enable)")
        print("-" * 40)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    print("Running Tiling Tests...")

    # Keep this in sync with the pytest parameterization above.
    test_mfma_fp8_rocir_preshuffle(1024, 7168, 2048, tile_m=128, tile_n=128, tile_k=64)
    
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
