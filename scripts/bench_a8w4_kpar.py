#!/usr/bin/env python3
"""Benchmark a8w4 GEMM: n_warp=4/k_warp=1 vs n_warp=1/k_warp=4.

Usage:
    python3 scripts/bench_a8w4_kpar.py --source unified -M 1
    python3 scripts/bench_a8w4_kpar.py --source original -M 1     # baseline (no k_warp)
    python3 scripts/bench_a8w4_kpar.py --source unified -M 1 64
"""
import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import flydsl.compiler as flyc
from tests.kernels.utils import fp4_utils

SCALE_BLOCK = 32
WARMUP = 8
ITERS = 30
N_PER_GRAPH = 20


def _align_up(v, a):
    return ((v + a - 1) // a) * a


def preshuffle_e8m0_scale(scale, warp_tile, scale_k_per_tile=4, WMMA_DIM=16):
    rows, K_scale = scale.shape
    align = warp_tile
    if rows % align != 0:
        pad = _align_up(rows, align) - rows
        scale = torch.cat(
            [scale, torch.full((pad, K_scale), 127, dtype=scale.dtype, device=scale.device)],
            dim=0,
        )
    SCALES_PER_WMMA = 4
    wmma_rep = warp_tile // WMMA_DIM
    k_groups = K_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    g = scale.view(-1, wmma_rep, WMMA_DIM, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def bench_kernel_us_cudagraph(run_fn, warmup=WARMUP, iters=ITERS, n_per_graph=N_PER_GRAPH):
    """Per-launch timer via hipGraph capture+replay."""
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            run_fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture_stream):
        for _ in range(n_per_graph):
            run_fn()
    torch.cuda.synchronize()

    # Sanity: compare direct vs graph
    ref_s = torch.cuda.Event(enable_timing=True)
    ref_e = torch.cuda.Event(enable_timing=True)
    ref_s.record()
    for _ in range(n_per_graph):
        run_fn()
    ref_e.record()
    torch.cuda.synchronize()
    ref_us = ref_s.elapsed_time(ref_e) * 1e3 / n_per_graph

    rep_s = torch.cuda.Event(enable_timing=True)
    rep_e = torch.cuda.Event(enable_timing=True)
    rep_s.record()
    g.replay()
    rep_e.record()
    torch.cuda.synchronize()
    first_us = rep_s.elapsed_time(rep_e) * 1e3 / n_per_graph

    if first_us < 1.0 and ref_us > 2.0:
        print(f"  WARNING: graph empty? ref={ref_us:.1f}us graph={first_us:.1f}us", flush=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (iters * n_per_graph)


def run_benchmark(compile_fn, *, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, k_warp, num_buffers, out_dtype, label):
    pack_b = 2  # a8w4 weight is fp4
    torch.manual_seed(42)

    a = torch.randint(0, 126, (M, K), dtype=torch.uint8)
    b = fp4_utils.random_fp4_packed(N, K)
    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)

    padded_m = _align_up(M, tile_m)
    if a.shape[0] < padded_m:
        a = torch.cat([a, torch.zeros(padded_m - M, K, dtype=a.dtype)], dim=0)
        a_scale = torch.cat(
            [a_scale, torch.full((padded_m - M, K // SCALE_BLOCK), 127, dtype=a_scale.dtype)],
            dim=0,
        )

    skt = tile_k // SCALE_BLOCK
    wt_m = tile_m // m_warp
    wt_n = tile_n // n_warp
    a_scale_p = preshuffle_e8m0_scale(a_scale, wt_m, scale_k_per_tile=skt)
    b_scale_p = preshuffle_e8m0_scale(b_scale, wt_n, scale_k_per_tile=skt)
    K_packed_b = K // pack_b
    b_p = fp4_utils.preshuffle_b_16x16(b, N, K_packed_b)

    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    torch_out_dtype = _dtype_map[out_dtype]
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4

    a_gpu = a.cuda().contiguous()
    b_gpu = b_p.cuda().contiguous()
    as_gpu = a_scale_p.cuda().contiguous()
    bs_gpu = b_scale_p.cuda().contiguous()
    c_gpu = torch.zeros(padded_m, N, dtype=torch_out_dtype, device="cuda")

    compile_kwargs = dict(
        data_format="a8w4",
        scale_mode="mxscale",
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        use_tdm_store=True,
        out_dtype=out_dtype,
        expert_sched_mode=True,
    )
    if k_warp != 1:
        compile_kwargs["k_warp"] = k_warp

    t0 = time.perf_counter()
    launch_fn = compile_fn(**compile_kwargs)
    compiled_exe = flyc.compile(
        launch_fn, c_gpu, a_gpu, b_gpu, as_gpu, bs_gpu,
        padded_m, N, K, N, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000

    def run_kernel():
        compiled_exe(c_gpu, a_gpu, b_gpu, as_gpu, bs_gpu, padded_m, N, K, N, torch.cuda.current_stream())

    us = bench_kernel_us_cudagraph(run_kernel)

    flops = 2.0 * M * N * K
    tflops = flops / (us * 1e-6) / 1e12

    bytes_a = padded_m * K  # fp8
    bytes_b = N * K // 2  # fp4
    bytes_scale = (padded_m + N) * (K // SCALE_BLOCK)
    bytes_d = padded_m * N * elem_bytes_d
    total_bytes = bytes_a + bytes_b + bytes_scale + bytes_d
    bw_gbs = total_bytes / (us * 1e-6) / 1e9

    print(f"  [{label}]")
    print(f"    Compile:    {compile_ms:.0f} ms")
    print(f"    Kernel:     {us:.1f} us")
    print(f"    TFLOPS:     {tflops:.2f}")
    print(f"    Bandwidth:  {bw_gbs:.1f} GB/s")
    print()
    return us, tflops, bw_gbs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="unified", choices=["unified", "original", "kpar"])
    parser.add_argument("-M", type=int, nargs="+", default=[1])
    parser.add_argument("-N", type=int, default=12288)
    parser.add_argument("-K", type=int, default=3072)
    parser.add_argument("--tile-m", type=int, default=16)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=512)
    parser.add_argument("--num-buffers", type=int, default=4)
    parser.add_argument("--out-dtype", default="bf16", choices=["f32", "bf16", "f16"])
    args = parser.parse_args()

    if args.source == "unified":
        from kernels.gemm_fp8fp4_gfx1250_unified import compile_fp8fp4_gemm
    elif args.source == "original":
        from kernels.gemm_fp8fp4_gfx1250 import compile_fp8fp4_gemm
    else:
        from kernels.gemm_fp8fp4_gfx1250_kpar import compile_fp8fp4_gemm

    configs = [
        {"m_warp": 1, "n_warp": 4, "k_warp": 1, "label": "n_warp=4, k_warp=1"},
    ]
    if args.source != "original":
        configs.append({"m_warp": 1, "n_warp": 1, "k_warp": 4, "label": "n_warp=1, k_warp=4"})

    for M in args.M:
        print("=" * 70)
        print(f"  A8W4 GEMM Benchmark — source={args.source}")
        print(f"  M={M}, N={args.N}, K={args.K}")
        print(f"  tile=({args.tile_m},{args.tile_n},{args.tile_k}), bufs={args.num_buffers}, out={args.out_dtype}")
        print(f"  warmup={WARMUP}, iters={ITERS}, hipGraph timing")
        print("=" * 70)

        results = {}
        for cfg in configs:
            try:
                us, tflops, bw = run_benchmark(
                    compile_fp8fp4_gemm,
                    M=M, N=args.N, K=args.K,
                    tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
                    num_buffers=args.num_buffers, out_dtype=args.out_dtype,
                    **cfg,
                )
                results[cfg["label"]] = (us, tflops, bw)
            except Exception as e:
                print(f"  [{cfg['label']}] FAILED: {e}\n")

        if len(results) >= 1:
            labels = list(results.keys())
            base_us = results[labels[0]][0]
            print("  --- Summary ---")
            for lbl, (us, tflops, bw) in results.items():
                ratio = base_us / us
                print(f"    {lbl:30s}  {us:7.1f} us  {tflops:8.2f} TFLOPS  {bw:7.1f} GB/s  {ratio:.3f}x")
            print()


if __name__ == "__main__":
    main()
