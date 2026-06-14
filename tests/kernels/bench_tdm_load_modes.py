#!/usr/bin/env python3
"""Micro-benchmark: wave-specialized vs cooperative TDM load.

Mode A (wave-specialized): 4 waves each load 1 of 4 tensors (num_warps=1 per descriptor).
Mode B (cooperative):      4 waves cooperate on each tensor (num_warps=4), 4 sequential loads.

Both modes load the same total bytes per step, consume via ds_load + accumulate,
and repeat for many rounds to get stable timing.
"""

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith, gpu, rocdl, tdm_ops, range_constexpr, const_expr
from flydsl.expr.typing import T
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx,
    get_lds_memref,
    lds_load_b128_raw,
    pipeline_fence,
    issue_tdm_loads,
)

WAVE_SIZE = 32
NUM_WARPS = 4
BLOCK_THREADS = NUM_WARPS * WAVE_SIZE

# Each "tensor" tile: TILE_ROWS x TILE_COLS bytes in LDS
TILE_ROWS = 16
TILE_COLS = 128
TILE_BYTES = TILE_ROWS * TILE_COLS  # 2048 bytes per tensor
NUM_TENSORS = 4
STAGE_BYTES = TILE_BYTES * NUM_TENSORS  # 8192 bytes per stage
NUM_ROUNDS = 64

# Global tensor shape: enough rows for all rounds of K-tiles
GLOBAL_ROWS = TILE_ROWS
GLOBAL_COLS = TILE_COLS * NUM_ROUNDS
GPU_ARCH = "gfx1250"


def _make_tdm_desc_simple(*, early_timeout=False, **kwargs):
    import inspect
    if "early_timeout" in inspect.signature(tdm_ops.make_tensor_descriptor_2d).parameters:
        kwargs["early_timeout"] = early_timeout
    return tdm_ops.make_tensor_descriptor_2d(**kwargs)


def compile_bench_kernel(mode: str):
    """Compile a TDM load benchmark kernel.

    mode="wave_spec": wave-specialized (1 tensor_load per wave, num_warps=1)
    mode="cooperative": cooperative (4 sequential tensor_loads, num_warps=4)
    """
    is_wave_spec = mode == "wave_spec"
    desc_num_warps = 1 if is_wave_spec else NUM_WARPS

    arena = SmemAllocator(None, arch=GPU_ARCH, global_sym_name=f"tdm_bench_{mode}")
    # 4 tensor regions in LDS
    tensor_lds_off = [i * TILE_BYTES for i in range(NUM_TENSORS)]
    arena.ptr = STAGE_BYTES

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel_tdm_bench(
        arg_t0: fx.Tensor,
        arg_t1: fx.Tensor,
        arg_t2: fx.Tensor,
        arg_t3: fx.Tensor,
        arg_out: fx.Tensor,
    ):
        tx = gpu.thread_id("x")

        # Extract lane info
        layout_thr = fx.make_layout(
            (NUM_WARPS, 2, 16),
            (WAVE_SIZE, 16, 1),
        )
        thr_coord = fx.idx2crd(tx, layout_thr)
        wave_idx = fx.get(thr_coord, 0)
        lane_kgrp = fx.get(thr_coord, 1)
        lane16 = fx.get(thr_coord, 2)

        args = [arg_t0, arg_t1, arg_t2, arg_t3]
        elem_ty_lds = T.f16
        lds_f16_count = TILE_BYTES // 2

        base_ptr = arena.get_base()
        lds_regions = [
            SmemPtr(base_ptr, tensor_lds_off[i], elem_ty_lds, shape=(lds_f16_count,))
            for i in range_constexpr(NUM_TENSORS)
        ]
        lds_mems = [lds_regions[i].get() for i in range_constexpr(NUM_TENSORS)]
        lds_idxs = [extract_lds_base_idx(lds_regions[i]) for i in range_constexpr(NUM_TENSORS)]

        # Build descriptors
        def make_desc(tensor_arg, lds_mem, k_off):
            return _make_tdm_desc_simple(
                global_ptr=tensor_arg,
                lds_memref=lds_mem,
                global_offset=(arith.index(0), k_off),
                tensor_shape=(TILE_ROWS, GLOBAL_COLS),
                strides=(GLOBAL_COLS, 1),
                tile_shape=(TILE_ROWS, TILE_COLS),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=desc_num_warps,
            )

        # Accumulator: reduce ds_load results to prevent dead-code elimination
        acc = arith.constant(T.i32, 0)

        # ds_load base for consumption: lane16 * row_stride + lane_kgrp * 16
        consume_base = lane16 * arith.index(TILE_COLS) + lane_kgrp * arith.index(16)

        if const_expr(is_wave_spec):
            # Wave-specialized: each wave loads one tensor
            tdm_wave_id = rocdl.wave_id()
            tdm_wave_is_0 = tdm_wave_id == fx.Int32(0)
            tdm_wave_is_1 = tdm_wave_id == fx.Int32(1)
            tdm_wave_is_2 = tdm_wave_id == fx.Int32(2)

            def _select4(v0, v1, v2, v3):
                r = arith.select(tdm_wave_is_2, v2, v3)
                r = arith.select(tdm_wave_is_1, v1, r)
                return arith.select(tdm_wave_is_0, v0, r)

            for rnd, state in range(0, NUM_ROUNDS, 1, init=[acc]):
                acc_in = state[0]
                k_off = rnd * arith.index(TILE_COLS)

                descs = [make_desc(args[i], lds_mems[i], k_off) for i in range_constexpr(NUM_TENSORS)]

                # Build per-wave descriptor via select
                dg0_vals = [descs[i].dgroup0 for i in range_constexpr(NUM_TENSORS)]
                dg1_vals = [descs[i].dgroup1 for i in range_constexpr(NUM_TENSORS)]
                active_dg0 = _select4(dg0_vals[0], dg0_vals[1], dg0_vals[2], dg0_vals[3])
                active_dg1 = _select4(dg1_vals[0], dg1_vals[1], dg1_vals[2], dg1_vals[3])
                desc = tdm_ops.TDMDescriptor2D(active_dg0, active_dg1)
                tdm_ops.tensor_load_2d(desc)

                pipeline_fence(outstanding=0)

                # Consume: each wave reads from all 4 LDS regions
                for t in range_constexpr(NUM_TENSORS):
                    v = fx.Vector(lds_load_b128_raw(lds_idxs[t], consume_base))
                    acc_in = acc_in + v[0]

                results = yield [acc_in]
            acc = results[0]

        else:
            # Cooperative: all 4 waves cooperate on each tensor_load
            for rnd, state in range(0, NUM_ROUNDS, 1, init=[acc]):
                acc_in = state[0]
                k_off = rnd * arith.index(TILE_COLS)

                for t in range_constexpr(NUM_TENSORS):
                    desc = make_desc(args[t], lds_mems[t], k_off)
                    tdm_ops.tensor_load_2d(desc)

                pipeline_fence(outstanding=0)

                for t in range_constexpr(NUM_TENSORS):
                    v = fx.Vector(lds_load_b128_raw(lds_idxs[t], consume_base))
                    acc_in = acc_in + v[0]

                results = yield [acc_in]
            acc = results[0]

        # Write acc to output to prevent DCE
        from flydsl.expr import buffer_ops
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False)
        out_off = arith.index_cast(T.i32, tx)
        buffer_ops.buffer_store(acc, out_rsrc, out_off)

    @flyc.jit
    def launch_tdm_bench(
        arg_t0: fx.Tensor,
        arg_t1: fx.Tensor,
        arg_t2: fx.Tensor,
        arg_t3: fx.Tensor,
        arg_out: fx.Tensor,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena.finalized = False
            arena.finalize()

        kernel_tdm_bench(
            arg_t0, arg_t1, arg_t2, arg_t3, arg_out,
        ).launch(
            grid=(1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_tdm_bench


def bench_mode(mode, warmup=10, iters=100):
    print(f"\nCompiling {mode}...")
    t0 = time.perf_counter()
    launch_fn = compile_bench_kernel(mode)

    # Allocate 4 global tensors
    tensors = [
        torch.randint(0, 256, (GLOBAL_ROWS, GLOBAL_COLS), dtype=torch.uint8, device="cuda")
        for _ in range(NUM_TENSORS)
    ]
    out = torch.zeros(BLOCK_THREADS, dtype=torch.int32, device="cuda")

    compiled = flyc.compile(
        launch_fn,
        tensors[0], tensors[1], tensors[2], tensors[3],
        out,
        torch.cuda.current_stream(),
    )
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"  Compile: {compile_ms:.0f} ms")

    def run():
        compiled(
            tensors[0], tensors[1], tensors[2], tensors[3],
            out,
            torch.cuda.current_stream(),
        )

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()

    us_per_launch = start.elapsed_time(end) * 1e3 / iters
    us_per_round = us_per_launch / NUM_ROUNDS
    total_bytes = TILE_BYTES * NUM_TENSORS * NUM_ROUNDS
    bw_gbs = total_bytes / (us_per_launch / 1e6) / 1e9

    print(f"  {mode}:")
    print(f"    Per launch:  {us_per_launch:.2f} us")
    print(f"    Per round:   {us_per_round:.3f} us")
    print(f"    Throughput:  {bw_gbs:.2f} GB/s")
    print(f"    Output sum:  {out.sum().item()} (sanity)")

    return us_per_launch


if __name__ == "__main__":
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"

    if not torch.cuda.is_available():
        print("No GPU available")
        sys.exit(1)

    from flydsl.runtime.device import get_rocm_arch
    arch = str(get_rocm_arch())
    print(f"GPU: {torch.cuda.get_device_name(0)}, arch={arch}")
    if arch != "gfx1250":
        print(f"Skipping: requires gfx1250, got {arch}")
        sys.exit(0)

    print(f"\nConfig: {NUM_TENSORS} tensors x {TILE_ROWS}x{TILE_COLS} bytes = "
          f"{TILE_BYTES * NUM_TENSORS} bytes/step, {NUM_ROUNDS} rounds")
    print("=" * 60)

    us_ws = bench_mode("wave_spec")
    us_coop = bench_mode("cooperative")

    print("\n" + "=" * 60)
    print(f"  wave_spec:   {us_ws:.2f} us")
    print(f"  cooperative: {us_coop:.2f} us")
    ratio = us_ws / us_coop if us_coop > 0 else float('inf')
    winner = "wave_spec" if us_ws < us_coop else "cooperative"
    print(f"  ratio:       {ratio:.3f}x  ({winner} wins)")
    print("=" * 60)
