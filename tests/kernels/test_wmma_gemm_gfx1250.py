#!/usr/bin/env python3
"""WMMA GEMM using TDM tests for gfx1250.

Kernel implementation lives in `kernels/wmma_gemm_gfx1250.py`.
This file is the correctness harness.
"""

import argparse
import os
import sys
import time

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

from flydsl.runtime.device import get_rocm_arch
from kernels.wmma_gemm_gfx1250 import compile_wmma_gemm_tdm
from tests.test_common import verify_output
from tests.kernels.benchmark_common import (
    bench_kernel_us,
    compute_gemm_tflops, compute_gemm_bandwidth,
    compute_wmma_stats, print_gemm_results,
)


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


def _make_wmma_case(
    in_dtype,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    *,
    num_buffers,
    m_warp=2,
    n_warp=4,
    l2_prefetch_distance=2,
    out_dtype=None,
    use_tdm_store=True,
    cluster_m=1,
    cluster_n=1,
    wave_specialized_tdm=False,
    inst_prefetch=False,
    waves_per_eu=None,
    expert_sched_mode=True,
    need_reference=True,
):
    """Prepare inputs, launch closure, and optional reference for a WMMA GEMM case."""
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
        f"warps=({m_warp},{n_warp}), waves_per_eu={waves_per_eu}, "
        f"tdm_store={use_tdm_store}, cluster=({cluster_m},{cluster_n}), "
        f"wave_spec_tdm={wave_specialized_tdm}, inst_prefetch={inst_prefetch}, "
        f"expert_sched_mode={expert_sched_mode}"
    )

    a = torch.randn((M, K), dtype=torch_dtype, device='cpu')
    b = torch.randn((K, N), dtype=torch_dtype, device='cpu')
    ref = torch.mm(a.to(torch.float32), b.to(torch.float32)) if need_reference else None

    a_pad = _pad_2d_tensor(a, mpad, kpad).cuda()
    b_pad = _pad_2d_tensor(b, kpad, npad).cuda()

    c_pad = torch.zeros((mpad, npad), dtype=_out_torch, device='cuda')

    launch_fn = compile_wmma_gemm_tdm(
        M=mpad, N=npad, K=kpad,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp, in_dtype=in_dtype,
        out_dtype=out_dtype,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        wave_specialized_tdm=wave_specialized_tdm,
        inst_prefetch=inst_prefetch,
        expert_sched_mode=expert_sched_mode,
    )
    return {
        "M": M,
        "N": N,
        "K": K,
        "mpad": mpad,
        "npad": npad,
        "kpad": kpad,
        "in_dtype": in_dtype,
        "out_dtype": _eff_out,
        "torch_out_dtype": _out_torch,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "m_warp": m_warp,
        "n_warp": n_warp,
        "num_buffers": num_buffers,
        "waves_per_eu": waves_per_eu,
        "l2_prefetch_distance": l2_prefetch_distance,
        "use_tdm_store": use_tdm_store,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "wave_specialized_tdm": wave_specialized_tdm,
        "inst_prefetch": inst_prefetch,
        "expert_sched_mode": expert_sched_mode,
        "launch_fn": launch_fn,
        "a_pad": a_pad,
        "b_pad": b_pad,
        "c_pad": c_pad,
        "a_flat": a_pad.contiguous().view(-1),
        "b_flat": b_pad.contiguous().view(-1),
        "c_flat": c_pad.contiguous().view(-1),
        "stream": torch.cuda.current_stream(),
        "ref": ref,
    }


def _launch_wmma_case(case):
    case["launch_fn"](
        case["c_flat"],
        case["a_flat"],
        case["b_flat"],
        case["mpad"],
        case["npad"],
        case["stream"],
    )


def _run_benchmark(args):
    """Benchmark mode: compile once, then time repeated launches."""
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
    out_dtype = None if args.out_dtype == "auto" else args.out_dtype
    effective_out = out_dtype or ("f16" if args.dtype == "fp16" else "bf16")
    elem_bytes_d = 2 if effective_out in ("bf16", "f16") else 4

    case = _make_wmma_case(
        args.dtype,
        args.M,
        args.N,
        args.K,
        args.tile_m,
        args.tile_n,
        args.tile_k,
        num_buffers=args.num_buffers,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        l2_prefetch_distance=args.l2_prefetch_distance,
        out_dtype=out_dtype,
        use_tdm_store=not args.no_tdm_store,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        wave_specialized_tdm=args.wave_spec_tdm,
        inst_prefetch=args.inst_prefetch,
        waves_per_eu=args.waves_per_eu,
        expert_sched_mode=args.expert_sched_mode,
        need_reference=False,
    )

    print("=" * 72)
    print(f"  WMMA GEMM Benchmark on gfx1250")
    print(f"  PyTorch {torch.__version__}, Device: {torch.cuda.get_device_name(0)}")
    needs_pad = (case["mpad"], case["npad"], case["kpad"]) != (args.M, args.N, args.K)
    print(f"  Shape: M={args.M}, N={args.N}, K={args.K}")
    if needs_pad:
        print(f"  Kernel pad: M={case['mpad']}, N={case['npad']}, K={case['kpad']}")
    print(
        f"  Tile: ({args.tile_m}, {args.tile_n}, {args.tile_k}), "
        f"warps=({args.m_warp}x{args.n_warp}), waves={args.m_warp * args.n_warp}"
    )
    print(
        f"  Buffers={args.num_buffers}, dtype={args.dtype}, out={effective_out}, "
        f"waves_per_eu={args.waves_per_eu}"
    )
    print(
        f"  tdm_store={not args.no_tdm_store}, wave_spec_tdm={args.wave_spec_tdm}, "
        f"inst_prefetch={args.inst_prefetch}, expert_sched_mode={args.expert_sched_mode}"
    )
    print(
        f"  Warmup={args.warmup}, Iters={args.iters}, "
        f"L2 flush={'ON' if not args.no_flush_l2 else 'OFF'}"
    )
    print("  Zero fill: ON (outside timing)")
    print("=" * 72)

    def prep_kernel():
        case["c_pad"].zero_()

    def run_kernel():
        _launch_wmma_case(case)

    print(f"\n[1/3] Compiling kernel...")
    t0 = time.perf_counter()
    prep_kernel()
    run_kernel()
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"      Compile + first launch: {compile_ms:.0f} ms")

    print(f"[2/3] Warming up ({args.warmup} iters) + benchmarking ({args.iters} iters)...")
    us = bench_kernel_us(
        run_kernel,
        warmup=args.warmup,
        iters=args.iters,
        flush_l2=not args.no_flush_l2,
        prep_fn=prep_kernel,
    )

    logical_tflops, kernel_tflops, time_s = compute_gemm_tflops(
        args.M, args.N, args.K,
        case["mpad"], case["npad"], case["kpad"], us)

    bytes_a = case["mpad"] * case["kpad"] * 2
    bytes_b = case["kpad"] * case["npad"] * 2
    bytes_d = case["mpad"] * case["npad"] * elem_bytes_d
    bw_gbs, read_bw_gbs, write_bw_gbs = compute_gemm_bandwidth(
        bytes_a + bytes_b, bytes_d, time_s)

    ws = compute_wmma_stats(
        args.tile_m, args.tile_n, args.tile_k,
        args.m_warp, args.n_warp,
        case["mpad"], case["npad"], case["kpad"], us,
        wmma_k=32, wmma_n_eff=16)

    return print_gemm_results(
        us, needs_pad, logical_tflops, kernel_tflops,
        bw_gbs, read_bw_gbs, write_bw_gbs,
        {"A": bytes_a, "B": bytes_b, "D": bytes_d}, ws)


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
                        waves_per_eu=None, expert_sched_mode=True):
    """Non-cluster GEMM correctness test."""
    case = _make_wmma_case(
        in_dtype,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        num_buffers=num_buffers,
        m_warp=m_warp,
        n_warp=n_warp,
        l2_prefetch_distance=l2_prefetch_distance,
        out_dtype=out_dtype,
        use_tdm_store=use_tdm_store,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        wave_specialized_tdm=wave_specialized_tdm,
        inst_prefetch=inst_prefetch,
        waves_per_eu=waves_per_eu,
        expert_sched_mode=expert_sched_mode,
    )
    _launch_wmma_case(case)
    torch.cuda.synchronize()

    rtol = 3e-2
    atol = 3e-2
    assert verify_output(case["c_pad"][:M, :N].cpu().to(torch.float32), case["ref"], rtol=rtol, atol=atol)
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
    parser.add_argument("--out-dtype", type=str, default="auto", choices=["auto", "f16", "bf16", "f32"])
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--l2-prefetch-distance", type=int, default=0)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    parser.add_argument("--wave-spec-tdm", action="store_true", default=False)
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--disable-expert-sched-mode", dest="expert_sched_mode",
                        action="store_false", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Run benchmark mode (timing only, no correctness check)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-flush-l2", action="store_true", default=False)
    return parser


def _run_cli_args(args, runner=None):
    if runner is None:
        runner = test_wmma_gemm_tdm

    out_dtype = None if args.out_dtype == "auto" else args.out_dtype

    runner(
        args.dtype, args.M, args.N, args.K,
        args.tile_m, args.tile_n, args.tile_k,
        num_buffers=args.num_buffers,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        l2_prefetch_distance=args.l2_prefetch_distance,
        out_dtype=out_dtype,
        use_tdm_store=not args.no_tdm_store,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        wave_specialized_tdm=args.wave_spec_tdm,
        inst_prefetch=args.inst_prefetch,
        waves_per_eu=args.waves_per_eu,
        expert_sched_mode=args.expert_sched_mode,
    )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.benchmark:
        _run_benchmark(args)
    else:
        _run_cli_args(args)
