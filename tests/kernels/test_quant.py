#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Benchmark: Per-Token Quantization Kernel (flydsl API)

Per-token quant: for each row, find max(|x|), scale = max_abs / 127,
output[i] = fptosi(x[i] / scale).

Usage Examples:
    python tests/kernels/test_quant.py
    python tests/kernels/test_quant.py -m 2048 -n 4096
    python tests/kernels/test_quant.py --multi-test "2048x8192,4096x8192,8192x8192"
"""

import sys
import os
import argparse

import numpy as np
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_RUN_QUANT = os.environ.get("FLYDSL_RUN_QUANT", "").strip().lower() in ("1", "true", "yes", "on")
if not _RUN_QUANT:
    _reason = "Per-token quant benchmark temporarily disabled (set FLYDSL_RUN_QUANT=1 to enable)."
    if __name__ == "__main__":
        print(_reason)
        raise SystemExit(0)
    pytest.skip(_reason, allow_module_level=True)

try:
    import torch
except Exception:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU benchmarks.", allow_module_level=True)

try:
    import aiter
    from aiter.ops.quant import per_token_quant_hip
    HAS_AITER = True
except Exception:
    HAS_AITER = False

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir import ir
from flydsl.expr.arith import ArithValue
from flydsl.expr.vector import Vector, full, ReductionOp
from flydsl.expr.numeric import Float16, Float32, Int8
from tests.test_common import run_perftest

BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8


class KernelCompilationCache:

    def __init__(self):
        self.cache = {}

    def get_or_compile(self, N, compile_fn):
        if N in self.cache:
            return self.cache[N] + (True,)
        result = compile_fn()
        self.cache[N] = result
        return result + (False,)

    def clear(self):
        self.cache.clear()


def build_quant_module(N):
    """Build per-token quantization kernel for hidden dimension N.

    Returns (launch_fn, config) where launch_fn(Input, Output, Scales, M)
    runs the quantization kernel.
    """
    arch = get_rocm_arch()

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    num_tiles = (N + tile_cols - 1) // tile_cols
    RED_SLOTS = max(1, BLOCK_THREADS // WARP_SIZE)

    allocator = SmemAllocator(None, arch=arch)
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + RED_SLOTS * 4  # f32 scratch

    @flyc.kernel
    def quant_kernel(Input: fx.Tensor, Output: fx.Tensor, Scales: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()

        # ── Wave / block reduction (max) ─────────────────────────────────
        def wave_reduce_max(x):
            width_i32 = arith.constant(WARP_SIZE, type=T.i32)
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.constant(sh, type=T.i32)
                peer = w.shuffle_xor(off, width_i32)
                w = w.maximumf(peer)
            return w

        def block_reduce_max(val):
            if RED_SLOTS == 1:
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            c_zero_f = arith.constant(0.0, type=T.f32)

            w = wave_reduce_max(val)

            if arith.cmpi(arith.CmpIPredicate.eq, lane, Int32(0)):
                wave_idx = arith.index_cast(T.index, wave)
                SmemPtr.store(s_red, w, [wave_idx])
            gpu.barrier()

            if arith.cmpi(arith.CmpIPredicate.eq, wave, Int32(0)):
                in_range = lane < RED_SLOTS
                lane_safe = arith.select(in_range, lane, Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v = SmemPtr.load(s_red, [lane_safe_idx])
                ww = arith.select(in_range, v, c_zero_f)
                ww = wave_reduce_max(ww)

                if arith.cmpi(arith.CmpIPredicate.eq, lane, Int32(0)):
                    c0_idx = arith.constant(0, index=True)
                    SmemPtr.store(s_red, ww, [c0_idx])
            gpu.barrier()

            c0_idx = arith.constant(0, index=True)
            return SmemPtr.load(s_red, [c0_idx])

        # ── Layout API: buffer-backed tensors ────────────────────────────
        Input_buf = fx.rocdl.make_buffer_tensor(Input)
        Out_buf = fx.rocdl.make_buffer_tensor(Output)
        Scales_buf = fx.rocdl.make_buffer_tensor(Scales)

        # Slice at row 0; actual row offset via soffset (SGPR)
        bid_row_offset = ArithValue(bid) * fx.Int32(N)

        row_in = fx.slice(Input_buf, (0, None))
        in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))

        # Copy atom for f16 loads: 8 x f16 = 128b, with soffset for row
        copy_atom_in_base = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)
        copy_atom_in = copy_atom_in_base.set_value("soffset", bid_row_offset)

        vec_reg_ty_f16 = fx.MemRefType.get(
            T.f16, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
        )
        vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

        # Copy atom for i8 output: 8 x i8 = 64b, with soffset for row
        copy_atom_out_base = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), 8)
        copy_atom_out = copy_atom_out_base.set_value("soffset", bid_row_offset)
        out_row = fx.slice(Out_buf, (0, None))
        out_div = fx.logical_divide(out_row, fx.make_layout(VEC_WIDTH, 1))
        i8_vec_reg_ty = fx.MemRefType.get(
            T.i8, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
        )

        # Copy atom for f32 scales: scalar store with soffset for bid
        copy_atom_f32_base = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        copy_atom_f32 = copy_atom_f32_base.set_value("soffset", bid)
        scales_div = fx.logical_divide(Scales_buf, fx.make_layout(1, 1))
        scales_base = fx.slice(scales_div, (None, fx.Int32(0)))
        f32_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        f32_reg_lay = fx.make_layout(1, 1)

        def _load_vec_f16(div_tensor, idx):
            r = fx.memref_alloca(vec_reg_ty_f16, vec_reg_lay)
            fx.copy_atom_call(copy_atom_in, fx.slice(div_tensor, (None, idx)), r)
            return Vector(fx.memref_load_vec(r), VEC_WIDTH, Float16)

        abs_mask = full(VEC_WIDTH, Int32(0x7FFFFFFF), Int32)

        c_zero_f = arith.constant(0.0, type=T.f32)
        local_max = c_zero_f
        cached_vecs = []

        # ── Pass 1: load f16 → f32, compute row max|val|, cache ──────────
        for tile_i in range_constexpr(num_tiles):
            idx = tid + tile_i * BLOCK_THREADS
            col_end = ArithValue(tid) * VEC_WIDTH + (tile_i * tile_cols + VEC_WIDTH)
            is_valid = col_end <= N

            vec_f16 = _load_vec_f16(in_div, idx)
            vec_f32 = vec_f16.to(Float32)
            cached_vecs.append(vec_f32)

            vec_i32 = vec_f32.bitcast(Int32)
            vec_abs_i32 = vec_i32 & abs_mask
            vec_abs = vec_abs_i32.bitcast(Float32)

            chunk_max = vec_abs.reduce(ReductionOp.MAX)
            chunk_max_safe = arith.select(is_valid, chunk_max.ir_value(), c_zero_f)
            local_max = local_max.maximumf(chunk_max_safe)

        reduced_max = block_reduce_max(local_max)

        # ── Compute scale ────────────────────────────────────────────────
        c_127 = arith.constant(127.0, type=T.f32)
        c_1 = arith.constant(1.0, type=T.f32)
        scale = ArithValue(reduced_max) / c_127
        is_zero = scale == c_zero_f
        final_scale = arith.select(is_zero, c_1, scale)

        # thread 0 stores Scales[bid]
        if arith.cmpi(arith.CmpIPredicate.eq, tid, Int32(0)):
            r_sc = fx.memref_alloca(f32_reg_ty, f32_reg_lay)
            ts_sc = full(1, Float32(final_scale), Float32)
            fx.memref_store_vec(ts_sc, r_sc)
            fx.copy_atom_call(copy_atom_f32, r_sc, scales_base)

        # ── Pass 2: quantize f32 → i8, store ─────────────────────────────
        inv_scale = ArithValue(c_1) / ArithValue(final_scale)

        for tile_i in range_constexpr(num_tiles):
            vec_f32 = cached_vecs[tile_i]
            vec_scaled = vec_f32 * inv_scale
            vec_i8 = vec_scaled.to(Int8)
            idx_out = tid + tile_i * BLOCK_THREADS

            # Python-level check: only the last tile of non-aligned N needs a guard
            last_partial = (N % tile_cols != 0) and (tile_i == num_tiles - 1)
            if last_partial:
                col_end = ArithValue(tid) * VEC_WIDTH + (tile_i * tile_cols + VEC_WIDTH)
                is_valid = col_end <= N
                if is_valid:
                    r_out = fx.memref_alloca(i8_vec_reg_ty, vec_reg_lay)
                    fx.memref_store_vec(vec_i8, r_out)
                    fx.copy_atom_call(copy_atom_out, r_out, fx.slice(out_div, (None, idx_out)))
            else:
                r_out = fx.memref_alloca(i8_vec_reg_ty, vec_reg_lay)
                fx.memref_store_vec(vec_i8, r_out)
                fx.copy_atom_call(copy_atom_out, r_out, fx.slice(out_div, (None, idx_out)))

    @flyc.jit
    def launch_quant(
        Input: fx.Tensor,
        Output: fx.Tensor,
        Scales: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = arith.index_cast(T.index, m_in)
        launcher = quant_kernel(Input, Output, Scales)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    config = {
        'N': N,
        'BLOCK_THREADS': BLOCK_THREADS,
        'VEC_WIDTH': VEC_WIDTH,
        'ELEMS_PER_BLOCK_ITER': tile_cols,
        'ITERS': num_tiles,
    }

    return launch_quant, config


def compile_kernel_for_n(N, gpu_arch=None):
    """Compile kernel for hidden dim N, returning (launch_fn, config)."""
    print(f"Compiling kernel for N={N}...")
    launch_fn, config = build_quant_module(N)
    print("Compiled via flydsl")
    return launch_fn, config


def benchmark_per_token_quant(M=4096, N=8192, launch_fn=None, config=None):
    """Run Per-Token Quantization benchmark.

    Args:
        M: Number of tokens
        N: Hidden dimension
        launch_fn: Pre-compiled launcher (optional)
        config: Config dict from compile (optional)

    Returns:
        bool: True if correctness check passed
    """
    print("\n" + "=" * 80)
    print(f"Benchmark: Per-Token Quantization Performance (FlyDSL) [M={M}, N={N}]")
    print("=" * 80)

    if launch_fn is None or config is None:
        gpu_arch = get_rocm_arch()
        print(f"Detected ROCm Arch: {gpu_arch}")
        launch_fn, config = compile_kernel_for_n(N, gpu_arch)
    else:
        print("Using pre-compiled launcher")

    total_elements = M * N
    total_bytes_rw = (M * N * 2) + (M * N * 1) + (M * 4)

    print(f"Configuration:")
    print(f"  - Shape: [{M}, {N}]")
    print(f"  - Block Size: {config['BLOCK_THREADS']}")
    print(f"  - Total Elements: {total_elements/1e6:.2f}M")
    print(f"  - Loops per Block: {config['ITERS']}")
    print(f"  - Est. Memory Traffic: {total_bytes_rw/1e9:.2f} GB per call")

    np.random.seed(42)
    input_data_fp16 = np.random.uniform(-5.0, 5.0, size=(M, N)).astype(np.float16)
    input_data = input_data_fp16.astype(np.float32)
    dtypeMax = 127.0
    per_token_amax = np.max(np.abs(input_data), axis=1)
    per_token_scale = per_token_amax / dtypeMax
    per_token_scale[per_token_scale == 0] = 1.0
    scale_expanded = per_token_scale[:, np.newaxis]
    output_ref = (input_data / scale_expanded).astype(np.int8)

    input_torch = torch.from_numpy(input_data_fp16).to(device="cuda")
    output_torch = torch.empty((M, N), device="cuda", dtype=torch.int8)
    scales_torch = torch.empty((M,), device="cuda", dtype=torch.float32)

    def kernel_launch():
        launch_fn(input_torch, output_torch, scales_torch, M)
        torch.cuda.synchronize()

    print("Running benchmark...")
    _, avg_us = run_perftest(kernel_launch, num_iters=20, num_warmup=2)

    bandwidth_gbs = total_bytes_rw / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000

    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": total_elements,
        "total_bytes": total_bytes_rw,
    }

    output_host = output_torch.cpu().numpy()
    scales_host = scales_torch.cpu().numpy()

    scale_diff = np.max(np.abs(scales_host - per_token_scale))
    output_diff = np.max(
        np.abs(output_host.astype(np.float32) - output_ref.astype(np.float32))
    )

    print(f"\nFlyDSL Kernel Results:")
    print(f"  Max Scale Diff:  {scale_diff:.2e}")
    print(f"  Max Output Diff: {output_diff:.2e}")
    print(f"\nBandwidth: {bandwidth_gbs:.2f} GB/s")
    print(f"  {results}")

    if HAS_AITER:
        try:
            print("\n" + "=" * 80)
            print("Benchmarking Reference Implementation (aiter)")
            print("=" * 80)

            def launch_aiter():
                per_token_quant_hip(input_torch)
                torch.cuda.synchronize()

            _, aiter_avg_us = run_perftest(launch_aiter, num_iters=20, num_warmup=2)

            aiter_bandwidth_gbs = total_bytes_rw / (aiter_avg_us / 1e6) / 1e9
            aiter_avg_ms = aiter_avg_us / 1000

            aiter_results = {
                "avg_ms": aiter_avg_ms,
                "avg_us": aiter_avg_us,
                "bandwidth_gbs": aiter_bandwidth_gbs,
                "size": total_elements,
                "total_bytes": total_bytes_rw,
            }

            output_torch_ref, scale_torch_ref = per_token_quant_hip(input_torch)
            torch.cuda.synchronize()

            output_ref_torch = output_torch_ref.cpu().numpy()
            scale_ref_torch = scale_torch_ref.squeeze().cpu().numpy()

            scale_diff_ref = np.max(np.abs(scale_ref_torch - per_token_scale))
            output_diff_ref = np.max(
                np.abs(output_ref_torch.astype(np.float32) - output_ref.astype(np.float32))
            )

            print(f"\n  Reference Correctness Check:")
            print(f"  Max Scale Diff:  {scale_diff_ref:.2e}")
            print(f"  Max Output Diff: {output_diff_ref:.2e}")

            flydsl_time = results["avg_ms"]
            aiter_time = aiter_results["avg_ms"]
            speedup = aiter_time / flydsl_time

            print(f"\n" + "=" * 80)
            print(f"Performance Comparison:")
            print(
                f"  FlyDSL:   {flydsl_time:7.3f} ms  ({results['bandwidth_gbs']:8.2f} GB/s)"
            )
            print(
                f"  Reference:  {aiter_time:7.3f} ms  ({aiter_results['bandwidth_gbs']:8.2f} GB/s)"
            )
            print(f"  Speedup:    {speedup:7.2f}x")
            print("=" * 80)
        except Exception as e:
            print("\n" + "=" * 80)
            print("Benchmarking Reference Implementation (aiter)")
            print("=" * 80)
            print(f"SKIPPED: aiter reference backend is unavailable in this environment: {e}")

    return output_diff <= 1.0


def test_benchmark_per_token_quant():
    """Pytest wrapper for per-token quantization benchmark."""
    print("\n" + "=" * 80)
    print("ROCm GPU Benchmark - Per-Token Quantization")
    print(f"GPU: {get_rocm_arch()}")
    print("=" * 80)
    assert (
        benchmark_per_token_quant()
    ), "Per-token quantization benchmark failed correctness check"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Per-Token Quantization")
    parser.add_argument(
        "-m", "--tokens", type=int, default=4096, help="Number of tokens (M)"
    )
    parser.add_argument(
        "-n", "--hidden", type=int, default=8192, help="Hidden dimension (N)"
    )
    parser.add_argument(
        "--multi-test", type=str, default=None,
        help="Run multiple tests with different sizes. Format: 'MxN,MxN,...' (e.g., '2048x4096,4096x8192')"
    )
    args = parser.parse_args()

    if args.multi_test:
        test_configs = []
        for size_str in args.multi_test.split(','):
            m_str, n_str = size_str.strip().split('x')
            test_configs.append((int(m_str), int(n_str)))

        print(f"\n{'='*80}")
        print(f"Per-Token Quantization Multi-Size Benchmark")
        print(f"Test Configurations: {len(test_configs)}")
        for i, (m, n) in enumerate(test_configs, 1):
            print(f"  {i}. M={m}, N={n}")
        print(f"{'='*80}")

        from collections import defaultdict
        tests_by_n = defaultdict(list)
        for m, n in test_configs:
            tests_by_n[n].append((m, n))

        unique_ns = sorted(tests_by_n.keys())
        if len(unique_ns) == 1:
            print(f"\nAll tests use N={unique_ns[0]} - will compile once and reuse!")
        else:
            print(f"\n{len(unique_ns)} different N values detected: {unique_ns}")
            print(f"  Will compile once per N value (total {len(unique_ns)} compilations)")

        results = []
        cache = KernelCompilationCache()

        for n_val in unique_ns:
            n_tests = tests_by_n[n_val]

            print(f"\n{'='*80}")
            print(f"Processing N={n_val} group ({len(n_tests)} test(s))")
            print(f"{'='*80}")

            launch_fn, config, was_cached = cache.get_or_compile(
                n_val,
                lambda: compile_kernel_for_n(n_val)
            )

            if was_cached:
                print("Using cached launcher")
            else:
                print("Compiled")

            for M, N in n_tests:
                print(f"\n{'='*80}")
                print(f"Test {len(results)+1}/{len(test_configs)}: M={M}, N={N}")
                print(f"{'='*80}")

                success = benchmark_per_token_quant(M=M, N=N, launch_fn=launch_fn, config=config)

                results.append({
                    'M': M,
                    'N': N,
                    'success': success,
                })

                if not success:
                    print(f"Failed for M={M}, N={N}")

        print(f"\n{'='*80}")
        print(f"Summary of All Tests")
        print(f"{'='*80}")
        print(f"{'M':>6} {'N':>6} {'Status':>8}")
        print(f"{'-'*80}")
        for r in results:
            status = "Pass" if r['success'] else "Fail"
            print(f"{r['M']:6} {r['N']:6} {status:>8}")

        all_passed = all(r['success'] for r in results)
        print(f"\n{'='*80}")
        if all_passed:
            print("All tests passed!")
        else:
            print("Some tests failed")
        print(f"{'='*80}\n")

        sys.exit(0 if all_passed else 1)
    else:
        success = benchmark_per_token_quant(M=args.tokens, N=args.hidden)
        if not success:
            sys.exit(1)
