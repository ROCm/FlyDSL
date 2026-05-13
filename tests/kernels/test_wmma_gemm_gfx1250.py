#!/usr/bin/env python3
"""WMMA GEMM using TDM tests for gfx1250.

Kernel implementation lives in `kernels/wmma_gemm_gfx1250.py`.
This file is the correctness harness.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

# workaround for simulator
import flydsl  # noqa: E402,F401 -- preload system comgr before torch/HIP loads LLVM

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

import flydsl.compiler as flyc

from flydsl.runtime.device import get_rocm_arch
from kernels.wmma_gemm_gfx1250 import compile_wmma_gemm_tdm
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _validate_pipeline_depth(*, K, tile_k, num_buffers):
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        pytest.skip(
            f"{num_buffers}-buffer requires num_k_tiles >= {num_buffers}, got {num_k_tiles}"
        )


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _get_padded_problem_shape(M: int, N: int, K: int,
                              tile_m: int, tile_n: int, tile_k: int) -> tuple[int, int, int]:
    return (
        _align_up(M, tile_m),
        _align_up(N, tile_n),
        _align_up(K, tile_k),
    )


def _pad_2d_tensor(tensor: torch.Tensor, rows: int, cols: int, fill_value: float = 0.0) -> torch.Tensor:
    if tensor.shape == (rows, cols):
        return tensor
    padded = torch.full((rows, cols), fill_value, dtype=tensor.dtype, device=tensor.device)
    padded[:tensor.shape[0], :tensor.shape[1]] = tensor
    return padded


def _format_kernel_pad(M: int, N: int, K: int, mpad: int, npad: int, kpad: int) -> str:
    padded_dims = (mpad, npad, kpad)
    if padded_dims == (M, N, K):
        return ""
    return f", kernel_pad={padded_dims}"


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    [
        (128, 128, 64, 64, 128, 32),
        (256, 256, 256, 64, 256, 128),
        (512, 512, 512, 64, 256, 128),
        (300, 399, 256, 64, 256, 128),
        (1024, 1024, 1024, 256, 256, 128),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_wmma_gemm_tdm(in_dtype, M, N, K, tile_m, tile_n, tile_k,
                        num_buffers,
                        m_warp=2, n_warp=4, l2_prefetch_distance=2,
                        out_dtype=None, use_tdm_store=True,
                        cluster_m=1, cluster_n=1,
                        wave_specialized_tdm=False, inst_prefetch=False,
                        pipeline_mode="a_streaming"):
    """Non-cluster GEMM correctness test."""
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA requires gfx1250, got {arch}")

    mpad, npad, kpad = _get_padded_problem_shape(M, N, K, tile_m, tile_n, tile_k)
    _validate_pipeline_depth(K=kpad, tile_k=tile_k, num_buffers=num_buffers)

    lds_pad = 8
    elem_bytes = 2
    a_buf = tile_m * (tile_k + lds_pad) * elem_bytes
    b_buf = tile_k * (tile_n + lds_pad) * elem_bytes
    total_lds = (a_buf + b_buf) * num_buffers
    if total_lds > 327680:
        pytest.skip(f"LDS budget exceeded: {total_lds} > 327680")

    _eff_out = out_dtype or ("f16" if in_dtype == "fp16" else "bf16")
    _out_torch = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[_eff_out]
    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    torch.manual_seed(0)

    wg_m = mpad // tile_m
    wg_n = npad // tile_n

    if cluster_m < 1 or cluster_n < 1:
        pytest.skip(f"Invalid cluster dims: ({cluster_m}, {cluster_n}), both must be >= 1")
    if cluster_m > 1 or cluster_n > 1:
        if wg_m < cluster_m or wg_n < cluster_n:
            pytest.skip(
                "Cluster dims exceed launch grid: "
                f"wg_grid=({wg_m},{wg_n}), cluster=({cluster_m},{cluster_n})"
            )
        if (wg_m % cluster_m) != 0 or (wg_n % cluster_n) != 0:
            pytest.skip(
                "WG grid must be divisible by cluster dims: "
                f"wg_grid=({wg_m},{wg_n}), cluster=({cluster_m},{cluster_n})"
            )

    pad_str = _format_kernel_pad(M, N, K, mpad, npad, kpad)
    print(
        f"Running WMMA GEMM TDM: M={M}, N={N}, K={K}{pad_str}, ",
        end=""
    )
    print(
        f"dtype={in_dtype}, out={_eff_out}, bufs={num_buffers}, "
        f"tdm_store={use_tdm_store}, cluster=({cluster_m},{cluster_n}), "
        f"wave_spec_tdm={wave_specialized_tdm}, inst_prefetch={inst_prefetch}"
    )

    a = torch.randn((M, K), dtype=torch_dtype, device='cpu')
    b = torch.randn((K, N), dtype=torch_dtype, device='cpu')
    ref = torch.mm(a.to(torch.float32), b.to(torch.float32))

    a_pad = _pad_2d_tensor(a, mpad, kpad).cuda()
    b_pad = _pad_2d_tensor(b, kpad, npad).cuda()

    c_pad = torch.zeros((mpad, npad), dtype=_out_torch, device='cuda')

    launch_fn = compile_wmma_gemm_tdm(
        M=mpad, N=npad, K=kpad,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp, in_dtype=in_dtype,
        out_dtype=out_dtype,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        wave_specialized_tdm=wave_specialized_tdm,
        inst_prefetch=inst_prefetch,
        pipeline_mode=pipeline_mode,
    )

    # Pre-bind via flyc.compile so the launch goes through the
    # CompiledFunction ctypes fast path. The slow JitFunction path adds
    # ~17us of inspect.Signature.bind + _make_cache_key per call. Even on
    # a single-launch correctness test this keeps the pattern consistent
    # with bench paths and any production caller. flyc.compile() launches
    # the kernel once internally to trigger compilation, so no separate
    # eager call is needed.
    flyc.compile(
        launch_fn,
        c_pad.contiguous().view(-1),
        a_pad.contiguous().view(-1),
        b_pad.contiguous().view(-1),
        mpad, npad, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    rtol = 3e-2
    atol = 3e-2
    assert verify_output(c_pad[:M, :N].cpu().to(torch.float32), ref, rtol=rtol, atol=atol)
    print("PASSED")


@pytest.mark.parametrize("in_dtype", ["fp16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    [
        (1024, 1024, 1024, 128, 256, 128),
        (2048, 2048, 2048, 128, 256, 128),
    ],
)
@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 2), (4, 4)])
def test_wmma_gemm_tdm_mcast(in_dtype, M, N, K, tile_m, tile_n, tile_k,
                              cluster_m, cluster_n):
    """Cluster multicast GEMM correctness test (large shapes only)."""
    pytest.skip("Temporarily skip fp16 GEMM mcast tests.")
    test_wmma_gemm_tdm(
        in_dtype, M, N, K, tile_m, tile_n, tile_k,
        num_buffers=2, m_warp=2, n_warp=4,
        l2_prefetch_distance=2,
        cluster_m=cluster_m, cluster_n=cluster_n,
    )


@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        ("fp16", None),
        ("bf16", None),
        ("fp16", "f32"),
    ],
)
def test_wmma_gemm_tdm_buffer_store_variants(in_dtype, out_dtype):
    """Cover the delayed epilogue address-precompute path."""
    test_wmma_gemm_tdm(
        in_dtype,
        256, 256, 512,
        64, 256, 128,
        num_buffers=2,
        m_warp=2, n_warp=4,
        l2_prefetch_distance=2,
        out_dtype=out_dtype,
        use_tdm_store=False,
    )


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
def test_wmma_gemm_tdm_tdm_store_tail_regression(in_dtype):
    """Regression for no-extra tail with 3-buffer TDM-store epilogue."""
    test_wmma_gemm_tdm(
        in_dtype,
        512, 512, 1024,
        64, 256, 128,
        num_buffers=3,
        m_warp=2, n_warp=4,
        l2_prefetch_distance=2,
        use_tdm_store=True,
    )


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, use_tdm_store",
    [
        pytest.param(13, 512, 7168, 16, 128, 128, 1, 4, True, id="DS-TP-stage1"),
        pytest.param(6, 7168, 256, 16, 256, 128, 1, 4, False, id="DS-TP-stage2"),
        pytest.param(29, 3072, 5120, 32, 256, 128, 1, 4, True, id="DS-EP-stage1"),
        pytest.param(32, 5120, 1536, 32, 256, 128, 1, 4, False, id="DS-EP-stage2"),
        pytest.param(22, 5760, 2880, 32, 256, 128, 1, 4, True, id="GPTOSS-stage1"),
        pytest.param(23, 2880, 2880, 32, 256, 128, 1, 4, False, id="GPTOSS-stage2"),
    ],
)
def test_wmma_gemm_tdm_moe_shapes(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, use_tdm_store,
):
    test_wmma_gemm_tdm(
        "fp16",
        M, N, K,
        tile_m, tile_n, tile_k,
        num_buffers=2,
        m_warp=m_warp,
        n_warp=n_warp,
        l2_prefetch_distance=2,
        use_tdm_store=use_tdm_store,
    )


def test_wmma_gemm_tdm_mcast_tail():
    """Exercise cluster mode with an even number of K tiles (tail includes a load)."""
    pytest.skip("Temporarily skip fp16 GEMM mcast tests.")
    test_wmma_gemm_tdm(
        "fp16",
        512, 512, 512,
        128, 256, 128,
        num_buffers=2,
        m_warp=2, n_warp=4,
        l2_prefetch_distance=2,
        cluster_m=2,
        cluster_n=2,
    )


def _bench_kernel_us_cudagraph(run_fn, warmup=10, iters=100, prep_fn=None):
    """Per-launch timer that strips host launch overhead via hipGraph.

    Capture one launch, replay N times in a stream-ordered burst,
    per-launch time = total / N. No L2 flush (back-to-back replays are
    intentionally hot-cache; for cold-cache numbers use _bench_kernel_us
    with flush_l2=True).
    """
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            if prep_fn is not None:
                prep_fn()
            run_fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    if prep_fn is not None:
        prep_fn()
    with torch.cuda.graph(g, stream=capture_stream):
        run_fn()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        g.replay()
    end_ev.record()
    torch.cuda.synchronize()
    total_us = start_ev.elapsed_time(end_ev) * 1e3
    return total_us / iters


def _bench_kernel_us(run_fn, warmup=10, iters=50, flush_l2=True, prep_fn=None):
    """Per-iteration CUDA events timer with L2 flush, IQR outlier removal, median."""
    flush_buf = None
    if flush_l2:
        l2_bytes = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()),
            "L2_cache_size", 4 * 1024 * 1024)
        alloc_bytes = max(l2_bytes * 2, 8 * 1024 * 1024)
        flush_buf = torch.empty(alloc_bytes, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        run_fn()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        start_ev[i].record()
        run_fn()
        end_ev[i].record()

    torch.cuda.synchronize()

    latencies = sorted(
        start_ev[i].elapsed_time(end_ev[i]) * 1e3 for i in range(iters))

    n = len(latencies)
    if n >= 8:
        q1, q3 = latencies[n // 4], latencies[3 * n // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in latencies if lo <= x <= hi]
        if filtered:
            latencies = filtered

    del flush_buf
    return latencies[len(latencies) // 2]


def _run_benchmark(args):
    """Benchmark mode for the WMMA TDM GEMM kernel (fp16/bf16 inputs)."""
    import time

    in_dtype = args.dtype
    M, N, K = args.M, args.N, args.K
    tile_m, tile_n, tile_k = args.tile_m, args.tile_n, args.tile_k
    mpad, npad, kpad = _get_padded_problem_shape(M, N, K, tile_m, tile_n, tile_k)
    needs_pad = (mpad, npad, kpad) != (M, N, K)

    eff_out = args.out_dtype or ("f16" if in_dtype == "fp16" else "bf16")
    _out_torch = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[eff_out]
    elem_bytes_in = 2  # fp16 / bf16
    elem_bytes_d = 2 if eff_out in ("f16", "bf16") else 4
    torch_in_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16

    print("=" * 72)
    print(f"  WMMA TDM GEMM Benchmark on gfx1250 ({in_dtype} → {eff_out})")
    print(f"  PyTorch {torch.__version__}, Device: {torch.cuda.get_device_name(0)}")
    print(f"  Shape: M={M}, N={N}, K={K}")
    if needs_pad:
        print(f"  Kernel pad: M={mpad}, N={npad}, K={kpad}")
    print(f"  Tile: ({tile_m}, {tile_n}, {tile_k}), warps=({args.m_warp}x{args.n_warp})")
    print(f"  Buffers={args.num_buffers}, pipeline_mode={args.pipeline_mode}, "
          f"tdm_store={not args.no_tdm_store}, inst_prefetch={args.inst_prefetch}")
    print(f"  Warmup={args.warmup}, Iters={args.iters}, "
          f"L2 flush={'ON' if not args.no_flush_l2 else 'OFF'}")
    print("=" * 72)

    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch_in_dtype, device='cpu')
    b = torch.randn((K, N), dtype=torch_in_dtype, device='cpu')
    a_pad = _pad_2d_tensor(a, mpad, kpad).cuda()
    b_pad = _pad_2d_tensor(b, kpad, npad).cuda()
    c_pad = torch.zeros((mpad, npad), dtype=_out_torch, device='cuda')

    print(f"\n[1/3] Compiling kernel...")
    t0 = time.perf_counter()
    launch_fn = compile_wmma_gemm_tdm(
        M=mpad, N=npad, K=kpad,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=args.m_warp, n_warp=args.n_warp,
        in_dtype=in_dtype,
        out_dtype=args.out_dtype,
        num_buffers=args.num_buffers,
        waves_per_eu=args.waves_per_eu,
        l2_prefetch_distance=args.l2_prefetch_distance,
        use_tdm_store=not args.no_tdm_store,
        cluster_m=args.cluster_m, cluster_n=args.cluster_n,
        wave_specialized_tdm=args.wave_spec_tdm,
        inst_prefetch=args.inst_prefetch,
        pipeline_mode=args.pipeline_mode,
    )

    c_flat = c_pad.contiguous().view(-1)
    a_flat = a_pad.contiguous().view(-1)
    b_flat = b_pad.contiguous().view(-1)

    # Pre-bind via flyc.compile so the bench loop calls go through the
    # CompiledFunction ctypes fast path (~17us of inspect.Signature.bind +
    # _make_cache_key per call would otherwise dominate per-launch latency
    # for short kernels).
    compiled_exe = flyc.compile(
        launch_fn,
        c_flat, a_flat, b_flat,
        mpad, npad, torch.cuda.current_stream(),
    )

    def prep_kernel():
        c_pad.zero_()

    # Stream resolved lazily inside the closure so graph-bench captures on
    # the active capture stream rather than the stream bound before capture.
    def run_kernel():
        compiled_exe(
            c_flat, a_flat, b_flat,
            mpad, npad, torch.cuda.current_stream(),
        )

    prep_kernel()
    run_kernel()
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"      Compile + first launch: {compile_ms:.0f} ms")

    use_graph = getattr(args, "use_graph", False)
    if use_graph:
        print(f"[2/3] Warming up ({args.warmup} iters) + bench via hipGraph "
              f"({args.iters} replays)...")
        us = _bench_kernel_us_cudagraph(
            run_kernel, warmup=args.warmup, iters=args.iters)
    else:
        print(f"[2/3] Warming up ({args.warmup} iters) + benchmarking "
              f"({args.iters} iters)...")
        us = _bench_kernel_us(run_kernel, warmup=args.warmup, iters=args.iters,
                              flush_l2=not args.no_flush_l2, prep_fn=prep_kernel)

    logical_flops = 2.0 * M * N * K
    kernel_flops = 2.0 * mpad * npad * kpad
    time_s = us / 1e6
    logical_tflops = logical_flops / time_s / 1e12 if time_s > 0 else 0.0
    kernel_tflops = kernel_flops / time_s / 1e12 if time_s > 0 else 0.0

    bytes_a = mpad * kpad * elem_bytes_in
    bytes_b = kpad * npad * elem_bytes_in
    bytes_d = mpad * npad * elem_bytes_d
    read_bytes = bytes_a + bytes_b
    write_bytes = bytes_d
    bytes_moved = read_bytes + write_bytes
    bw_gbs = bytes_moved / 1e9 / time_s if time_s > 0 else 0.0
    read_bw_gbs = read_bytes / 1e9 / time_s if time_s > 0 else 0.0
    write_bw_gbs = write_bytes / 1e9 / time_s if time_s > 0 else 0.0

    WMMA_M_EFF, WMMA_N_EFF, WMMA_K_EFF = 16, 16, 32  # fp16/bf16 16x16x32
    warp_tile_m = tile_m // args.m_warp
    warp_tile_n = tile_n // args.n_warp
    wmma_m_rep = warp_tile_m // WMMA_M_EFF
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    k_wmma_steps = tile_k // WMMA_K_EFF
    wmma_per_tile = wmma_m_rep * wmma_n_rep * k_wmma_steps
    m_tiles = mpad // tile_m
    n_tiles = npad // tile_n
    k_tiles = kpad // tile_k
    seq_wmma = k_tiles * wmma_per_tile
    us_per_wmma = us / seq_wmma if seq_wmma > 0 else 0

    print(f"\n[3/3] Results:")
    print(f"      Kernel time:  {us:.1f} us ({us / 1e3:.4f} ms)")
    if not needs_pad:
        print(f"      TFLOPS:       {kernel_tflops:.4f}")
    else:
        print(f"      TFLOPS:       {logical_tflops:.4f} (logical), {kernel_tflops:.4f} (kernel)")
    print(f"      Bandwidth:    {bw_gbs:.1f} GB/s  "
          f"(read: {read_bw_gbs:.1f} + write: {write_bw_gbs:.1f})")
    print(f"      Bytes moved:  {bytes_moved / 1e6:.1f} MB  "
          f"(A={bytes_a / 1e6:.1f} B={bytes_b / 1e6:.1f} D={bytes_d / 1e6:.1f})")
    print(f"      ---")
    print(f"      WMMA/tile:    {wmma_per_tile} "
          f"({wmma_m_rep}m × {wmma_n_rep}n × {k_wmma_steps}k)")
    print(f"      Total tiles:  {m_tiles}×{n_tiles} spatial × {k_tiles} K-iters")
    print(f"      Seq WMMA/WG:  {seq_wmma}")
    print(f"      us/WMMA:      {us_per_wmma:.2f}")
    print("=" * 72)

    reported_tflops = kernel_tflops if not needs_pad else logical_tflops
    return us, reported_tflops, bw_gbs


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--out-dtype", type=str, default=None,
                        choices=["f16", "bf16", "f32"],
                        help="Output dtype (None = match input).")
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--l2-prefetch-distance", type=int, default=0)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    parser.add_argument("--wave-spec-tdm", action="store_true", default=False)
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="a_streaming",
        choices=["a_streaming", "simple"],
        help="a_streaming (default) or simple",
    )

    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Run benchmark mode (timing only, no correctness check)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-flush-l2", action="store_true", default=False)
    parser.add_argument("--use-graph", action="store_true", default=False,
                        help="Time via hipGraph capture+replay to strip host "
                             "launch overhead from per-launch latency. "
                             "Implicitly disables L2 flush (graph replays are "
                             "back-to-back, hot-cache).")
    return parser


def _run_cli_args(args, runner=None):
    if runner is None:
        runner = test_wmma_gemm_tdm

    runner(
        args.dtype, args.M, args.N, args.K,
        args.tile_m, args.tile_n, args.tile_k,
        num_buffers=args.num_buffers,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        l2_prefetch_distance=args.l2_prefetch_distance,
        use_tdm_store=not args.no_tdm_store,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        wave_specialized_tdm=args.wave_spec_tdm,
        inst_prefetch=args.inst_prefetch,
        pipeline_mode=args.pipeline_mode,
    )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.benchmark:
        _run_benchmark(args)
    else:
        _run_cli_args(args)
