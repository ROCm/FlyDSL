#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using flir with B preshuffle (m1024, K32 micro-step variant)."""

import sys
import os
import logging
import functools

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)

import pyflir
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.runtime.device import get_rocm_arch as get_hip_arch
from pyflir.utils import SmemAllocator
from tests.utils import pertoken_quant, shuffle_weight, compile_to_hsaco
from tests.test_common import verify_output, run_perftest
import torch
import torch.nn.functional as F
import pytest
from _mlir import ir
from _mlir.dialects import vector, memref, builtin, llvm
from pyflir.dialects.ext import arith, scf, gpu, buffer_ops
from _mlir.dialects import arith as _arith_mlir
import pyflir.dialects.ext.rocdl as rocdl
import pyflir.lang.ir.types as mlir_types
from pyflir.lang.ir.types import T

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

# Aiter imports (optional)
try:
    import aiter
    HAS_AITER = True
except ImportError:
    print("Warning: Aiter not found, skipping comparison")
    HAS_AITER = False

RUN_AITER_BENCH = os.environ.get("COMPARE_AITER_CK", "1") == "1"


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


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k", [(1024, 7168, 2048, 128, 128, 128)]
)
def test_mfma_fp8_flir_preshuffle(M, N, K, tile_m, tile_n, tile_k):
    print("=" * 80)
    print(f"MFMA FP8 GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k}) [Torch Optimized]")
    print("=" * 80)
    gpu_arch = get_hip_arch()

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

    # Assume A loads are always 16B-aligned and use fixed dwordx4 (16B) buffer loads.
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by "
            f"a_load_bytes ({a_load_bytes})"
        )


    lds_stride = tile_k

    class _MFMA(flir.MlirModule):
        GPU_MODULE_NAME = "mfma_mod"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Ping-pong LDS for A (2-stage pipeline).
            _state["lds_a_decl"] = allocator.allocate_array(T.f8, 2 * tile_m * lds_stride)
            allocator.finalize()

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: mlir_types.memref(size_c, T.f16),
            arg_a: lambda: mlir_types.memref(size_a, T.f8),
            arg_b: lambda: mlir_types.memref(size_b, T.f8),
            arg_scale_a: lambda: mlir_types.memref(M, T.f32),
            arg_scale_b: lambda: mlir_types.memref(N, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            # ---- Types (centralized) ----

            zero_attr = ir.DenseElementsAttr.get_splat(
                T.f32x4, ir.FloatAttr.get(T.f32, 0.0)
            )
            acc_init = _arith_mlir.ConstantOp(T.f32x4, zero_attr).result

            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            c0_i32 = arith.i32(0).value

            # A is FP8 (1B/elem) but we use dwordx4 (16B) buffer loads, so build a /4 layout.
            c_k_div4 = c_k / 4
            layout_a_div4 = flir.make_layout((c_m, c_k_div4), stride=(c_k_div4, 1))

            c_k0 = c_k / 64
            c_n0 = c_n / 16
            stride_n0 = c_k0 * 1024

            stride_b = (
                stride_n0,  # n0
                1024,  # k0
                256,  # k1 (KLane)
                16,  # n1
                1,  # k2
            )
            # Shape: (N0, K0, KLane, NLane, KPack)
            layout_b = flir.make_layout(
                (
                    c_n0,  # N / 16
                    c_k0,  # K / 64
                    4,
                    16,
                    16,
                ),
                stride=stride_b,
            )

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k // 16, index=True)

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

            bx_m = bx * tile_m
            by_n = by * tile_n

            # (thread_id.x) -> (wave_id, lane_id) via FLIR (avoid explicit / and %).
            layout_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            coord_wave_lane = flir.idx2crd(tx, layout_wave_lane)
            wave_id = flir.get(coord_wave_lane, 0)
            lane_id = flir.get(coord_wave_lane, 1)

            # lane_id -> (lane_div_16, lane_mod_16) via FLIR (avoid explicit / and %).
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            coord_lane16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_lane16, 0)
            lane_mod_16 = flir.get(coord_lane16, 1)

            row_a_lds = lane_mod_16
            # lane_div_16 * 16 via FLIR crd2idx((lane_div_16,0), layout=(4,16):stride=(16,1))
            col_offset_base = flir.crd2idx(
                flir.make_coord(lane_div_16, 0), layout_lane16
            )

            row_b_lds = lane_mod_16

            m_repeat = tile_m // 16
            # K32 micro-step: one MFMA(x32) per step.
            k_unroll = tile_k // 32

            # --- Dynamic Tiling Logic ---
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            # wave_id is already in [0, 3] from FLIR (4,64) decomposition.
            n_tile_base = wave_id * c_n_per_wave

            # Global N calc loop
            n_intra_list = []
            n_blk_list = []

            # global_n -> (n_blk, n_intra) decomposition through FLIR (avoid %/ /).
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)

                # global_n = by_n + n_tile_base + offset + row_b_lds
                global_n = by_n + n_tile_base + c_offset + row_b_lds

                coord_n = flir.idx2crd(global_n, layout_n_blk_intra)
                n_blk = flir.get(coord_n, 0)
                n_intra = flir.get(coord_n, 1)

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
                    coord_a_lds = flir.make_coord(curr_row_a_lds, col_lds)
                    idx_a_mfma = flir.crd2idx(coord_a_lds, layout_lds)

            acc_inits = [acc_init] * (num_acc_n * m_repeat)

            # --- B Load Logic (K32) ---
            # CK intrawave_v3 interleaves B global loads with MFMA. Here we avoid carrying
            # the whole B tile through the loop state (which inflates VGPR/live ranges),
            # and instead load B packs per ki_step inside the MFMA loop.
            #
            # Helper layouts for index decomposition (avoid explicit div/mod).
            layout_k0_kpack64 = flir.make_layout((c_k0, 64), stride=(64, 1))
            layout_half8 = flir.make_layout((2, 8), stride=(8, 1))
            # B gmem->register via flir.copy (buffer-load backend).
            #
            # Keep the original 8B-per-step behavior: each MFMA(x32) step consumes one 8B pack
            # (half of the 16B KPack). This maps naturally to buffer_load_dwordx2.
            atom_b_g2r = flir.make_copy_atom(T.f8, vector_size=8)

            def load_b_pack(base_k, ki_step, ni):
                """
                Load one 8B (i64) B pack for a single MFMA(x32) step.
                We select the lower/upper half within the 16B KPack via k2_base = 0 or 8.
                """
                coord_k = flir.idx2crd(base_k, layout_k0_kpack64)
                k0_base = flir.get(coord_k, 0)
                k0 = k0_base + (ki_step // 2)
                k1 = lane_div_16  # 0..3
                half = ki_step % 2
                half_val = arith.constant(half, index=True)
                k2_base = flir.crd2idx(
                    flir.make_coord(half_val, 0), layout_half8
                )

                n_intra = n_intra_list[ni]
                n_blk = n_blk_list[ni]
                coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
                idx_bytes = flir.crd2idx(coord_b, layout_b)

                b_view = flir.TensorView(
                    arg_b,
                    (8,),
                    strides=(1,),
                    base_indices=(idx_bytes,),
                    element_type=T.f8,
                )
                b8_f8 = flir.copy(
                    atom_b_g2r,
                    b_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=b_rsrc,
                )
                b_vec64 = vector.BitCastOp(T.vec(1, T.i64), b8_f8).result
                return vector.ExtractOp(
                    b_vec64, static_position=[0], dynamic_position=[]
                ).result

            def load_b_tile(base_k):
                """Prefetch the entire per-thread B tile (gmem -> regs).

                Returns a python list-of-lists: b_tile[ki_step][ni] -> i64 pack.
                This is K-tile double-buffered (like A): compute uses current b_tile_in,
                while we prefetch the next tile into b_tile_next.
                """
                b_tile = []
                for ki_step in range_constexpr(k_unroll):
                    packs = []
                    for ni in range_constexpr(num_acc_n):
                        packs.append(load_b_pack(base_k, ki_step, ni))
                    b_tile.append(packs)
                return b_tile

            # A gmem->reg prefetch via flir.copy (buffer-load backend), keeping the original
            # pipelining structure: load next tile into regs (loop-carried), then store to LDS.
            # Fixed-width A loads (no tail handling): assume everything is 16B aligned.
            max_bytes_per_load = a_load_bytes  # 16
            num_a_loads = bytes_per_thread_a // max_bytes_per_load
            # Reindex A tile at dword granularity so lanes load contiguous 16B chunks:
            #   for load i: base = (tx + i*total_threads) * 16B
            # i32-element base = (tx*4 + i*(total_threads*4)).
            layout_a_tile_div4 = flir.make_layout((tile_m, tile_k // 4), stride=(tile_k // 4, 1))
            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4

            atom_a_g2r16 = flir.make_copy_atom(T.f8, vector_size=16)

            def load_a_16(idx_i32):
                """Load 16 fp8 bytes from A (gmem) into a vector via buffer_load backend.

                idx_i32 is in i32-elements (dword units) to avoid emitting per-load idx_bytes/4.
                """
                a_view = flir.TensorView(
                    arg_a,
                    (16,),
                    strides=(1,),
                    base_indices=(idx_i32,),
                    element_type=T.f8,
                )
                return flir.copy(
                    atom_a_g2r16,
                    a_view,
                    None,
                    alignment=16,
                    return_vector=True,
                    src_buffer_resource=a_rsrc,
                    src_buffer_offset_in_bytes=False,
                )

            def a_tile_chunk_coord_i32(i: int):
                """Map per-thread chunk `i` -> (row_a_local, col_a_local_i32) within the A tile.

                We want inter-thread contiguous 16B chunks:
                  chunk_linear = tx + i*total_threads
                  chunk_i32_base = chunk_linear * 4  (because 16B == 4 dwords)
                Then decompose into (row, col_i32) using a tile-local layout.
                """
                chunk_off_i32 = arith.constant(i * total_threads * 4, index=True)
                tile_idx_i32 = tx_i32_base + chunk_off_i32
                coord_a_local_i32 = flir.idx2crd(tile_idx_i32, layout_a_tile_div4)
                row_a_local = flir.get(coord_a_local_i32, 0)
                col_a_local_i32 = flir.get(coord_a_local_i32, 1)
                return row_a_local, col_a_local_i32

            def load_a_tile(k_base_div4):
                """Load the per-thread A tile portion (gmem -> regs) for a given K base (in /4 units)."""
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(
                        row_a_global, k_base_div4 + col_a_local_i32
                    )
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    a_f8 = load_a_16(idx_i32)
                    parts.append(vector.BitCastOp(T.i32x4, a_f8).result)
                return parts

            def store_a_tile_to_lds(vec_a_in_parts, lds_base):
                """Store the per-thread A tile portion (regs -> LDS), applying CK-style XOR16 swizzle."""
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    col_a_local_bytes = col_a_local_i32 * c4
                    col_swz = flir.swizzle_xor16(
                        row_a_local, col_a_local_bytes, k_blocks16
                    )
                    coord_store_0 = flir.make_coord(row_a_local, col_swz)
                    idx_0 = flir.crd2idx(coord_store_0, layout_lds)
                    idx_0 = (arith.ArithValue(idx_0) + lds_base).value

                    # Convert back to fp8 vector for LDS store.
                    a_vec = vector.BitCastOp(T.f8x16, vec_a_in_parts[i]).result
                    s_view = flir.TensorView(
                        lds_a,
                        (16,),
                        strides=(1,),
                        base_indices=(idx_0,),
                        element_type=T.f8,
                    )
                    flir.copy(atom_a_g2r16, a_vec, s_view, alignment=16)

            # Ping-pong LDS bases (index-typed).
            lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
            lds_base_cur = arith.constant(0, index=True)
            lds_base_nxt = lds_tile_elems

            def prefetch_ab_tile(base_k):
                """Prefetch full per-thread A+B tiles (gmem -> regs)."""
                base_k_div4 = base_k / 4
                a_regs = load_a_tile(base_k_div4)
                b_regs = load_b_tile(base_k)
                return a_regs, b_regs

            def compute_tile(accs_in, b_tile_in, lds_base, *, is_last_tile=False):
                """Compute one K-tile using A from ping-pong LDS and B from regs.

                For the last tile, we also prefetch scale vectors before MFMA so loads can
                overlap with MFMA.
                """
                scales_pf = {}
                if is_last_tile:
                    # --- PREFETCH SCALES (Last Iteration) ---
                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        val = buffer_ops.buffer_load(
                            scale_b_rsrc, col_g, vec_width=1, dtype=T.f32
                        )
                        s_b_vals.append(val)
                    scales_pf["s_b_vals"] = s_b_vals
                    scales_pf["s_a_vecs"] = []

                    row_off_base = lane_div_16 * 4
                    for mi in range_constexpr(m_repeat):
                        row_base_m = bx_m + (mi * 16)
                        row_g_base = row_base_m + row_off_base
                        s_a_vec = buffer_ops.buffer_load(
                            scale_a_rsrc, row_g_base, vec_width=4, dtype=T.f32
                        )
                        s_a_vec4 = vector.BitCastOp(T.f32x4, s_a_vec).result
                        scales_pf["s_a_vecs"].append(s_a_vec4)

                current_accs_list = list(accs_in)

                # MFMA: use current B tile resident in regs; use A from LDS (ping-pong base).
                for ki_step in range_constexpr(k_unroll):
                    b_packs = b_tile_in[ki_step]

                    ki64 = (ki_step // 2) * 64
                    half = ki_step % 2
                    col_base = col_offset_base + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val

                        col_base_swizzled = flir.swizzle_xor16(
                            curr_row_a_lds, col_base, k_blocks16
                        )
                        coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swizzled)
                        idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                        idx_a16 = (arith.ArithValue(idx_a16) + lds_base).value
                        loaded_a16 = vector.LoadOp(
                            T.f8x16, lds_a, [idx_a16]
                        ).result
                        a_vec128 = vector.BitCastOp(T.i64x2, loaded_a16).result
                        a_pack = vector.ExtractOp(
                            a_vec128, static_position=[half], dynamic_position=[]
                        ).result

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            curr_acc = current_accs_list[acc_idx]
                            b_pack = b_packs[ni]

                            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4,
                                [
                                    arith.ArithValue(a_pack).value,
                                    arith.ArithValue(b_pack).value,
                                    arith.ArithValue(curr_acc).value,
                                    c0_i32,
                                    c0_i32,
                                    c0_i32,
                                ],
                            ).result
                            current_accs_list[acc_idx] = acc0

                return current_accs_list, scales_pf

            # ---------------- Pipeline Prologue ----------------
            # Prefetch tile-0 into regs, then commit A to LDS(cur) and sync.
            k0 = arith.constant(0, index=True)
            a_regs0, b_tile_cur = prefetch_ab_tile(k0)
            store_a_tile_to_lds(a_regs0, lds_base_cur)
            gpu.barrier()

            # Loop-carried state
            accs = acc_inits

            # ---------------- Main Pipelined Loop ----------------
            # We execute tiles [0 .. K-tile_k) here. Each iteration:
            # - load ping buffer, compute pong, store ping
            # - barrier
            # - swap ping/pong roles
            #
            # Note: ping/pong *roles* are swapped each iteration (so compute always uses "pong",
            # while the physical buffer alternates). This matches the classic 2-stage pipeline.
            c_k_main = c_k - tile_k
            # Ping/pong LDS bases.
            lds_base_pong = lds_base_cur  # current/compute
            lds_base_ping = lds_base_nxt  # next/load+store
            # Ping/pong B reg tiles (pong is primed by the prologue).
            b_tile_pong = b_tile_cur  # current/compute
            rocdl.sched_barrier(0)
            def hot_loop_scheduler():
                # Derive CK-like schedule parameters from this tile:
                # - MFMA group size per "slot": num_acc_n
                # - Total MFMA per tile: k_unroll * m_repeat * num_acc_n
                # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                mfma_group = num_acc_n
                mfma_total = k_unroll * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                mfma_tail = mfma_total - sche_iters * mfma_per_iter

                # DS-read preload (CK default is 2); clamp to non-negative.
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
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
                    if sche_i >= dswr_start:
                        rocdl.sched_dswr(1)

                rocdl.sched_barrier(0)
            for k_iv in range(0, c_k_main, tile_k * 2):
                next_k1 = k_iv + tile_k
                # load ping, compute pong, store ping
                a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)
                accs, _ = compute_tile(accs, b_tile_pong, lds_base_pong)
                store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()
                next_k2 = k_iv + tile_k * 2
                # load ping, compute pong, store ping
                a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)
                accs, _ = compute_tile(accs, b_tile_ping, lds_base_ping)
                store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                hot_loop_scheduler()
                gpu.barrier()

            # ---------------- Epilogue (last tile) ----------------
            final_accs, scales = compute_tile(
                accs, b_tile_pong, lds_base_pong, is_last_tile=True
            )

            s_b_vals = scales["s_b_vals"]
            s_a_vecs = scales["s_a_vecs"]

            for mi in range_constexpr(m_repeat):
                row_base_m = bx_m + (mi * 16)
                s_a_vec4 = s_a_vecs[mi]

                for i in range_constexpr(4):
                    row_off = (lane_div_16 * 4) + i
                    row_g = row_base_m + row_off

                    s_a = vector.ExtractOp(
                        s_a_vec4, static_position=[i], dynamic_position=[]
                    ).result

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
                        val_f16 = _arith_mlir.TruncFOp(
                            T.f16, arith.ArithValue(val_s).value
                        ).result

                        idx = flir.crd2idx(
                            flir.make_coord(row_g, col_g), layout_c
                        )
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: mlir_types.memref(size_c, T.f16),
            arg_a: lambda: mlir_types.memref(size_a, T.f8),
            arg_b: lambda: mlir_types.memref(size_b, T.f8),
            arg_scale_a: lambda: mlir_types.memref(M, T.f32),
            arg_scale_b: lambda: mlir_types.memref(N, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            c1 = arith.constant(1, index=True).value
            bdx = arith.constant(256, index=True).value
            gx = arith.constant(M // tile_m, index=True).value
            gy = arith.constant(N // tile_n, index=True).value

            flir.gpu_ext.LaunchFuncOp(
                ["mfma_mod", "kernel_gemm"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arith.ArithValue(arg_c).value,
                    arith.ArithValue(arg_a).value,
                    arith.ArithValue(arg_b).value,
                    arith.ArithValue(arg_scale_a).value,
                    arith.ArithValue(arg_scale_b).value,
                    arith.ArithValue(c_m).value,
                    arith.ArithValue(c_n).value,
                    arith.ArithValue(c_k).value,
                ],
            )

    # Request occupancy hint: waves-per-eu=2
    # Use a unique kernel_name so IR/asm dumps don't get overwritten by other tests.
    m = _MFMA()

    exe = pyflir.compile(m)
    print("✓ Compiled")

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

    def launch_kernel(c, a, b, sa, sb):
        exe(c, a, b, sa, sb, M, N, K)

    # Perf note: GPU clocks can ramp; allow overriding warmup/iters for more stable peak numbers.
    bench_iters = int(os.environ.get("pyflir_BENCH_ITERS", "20"))
    bench_warmup = int(os.environ.get("pyflir_BENCH_WARMUP", "3"))
    _, us = run_perftest(
        launch_kernel,
        c_out_raw,
        a_q_fp8,
        b_shuffled,
        scale_a,
        scale_b,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
    )
    torch.cuda.synchronize()
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
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    if HAS_AITER and RUN_AITER_BENCH:
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
            f"Speedup vs Aiter: {tflops / tflops_aiter:.2f}x, Tflops {tflops:.1f} vs {tflops_aiter:.1f}"
        )
        print("-" * 40)
    elif HAS_AITER and not RUN_AITER_BENCH:
        print("-" * 40)
        print("Skipping Aiter benchmark (set pyflir_RUN_AITER_BENCH=1 to enable)")
        print("-" * 40)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    # Test cases
    print("Running Tiling Tests...")

    test_mfma_fp8_flir_preshuffle(5120, 5120, 8192+128, tile_m=64, tile_n=256, tile_k=128)
    