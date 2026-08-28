#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Dense BF16/FP16 GEMM (C = A x B^T) tests for gfx1250."""

import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

import flydsl.compiler as flyc  # noqa: E402,I001
import flydsl.expr as fx  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.gemm_bf16_gfx1250 import launch_gemm_bf16  # noqa: E402

_DT = {"bf16": torch.bfloat16, "f16": torch.float16}


def _require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


def _bytes_moved(M, N, K, eb=2):
    return (M * K + N * K + M * N) * eb


def _tflops(M, N, K, us):
    return 2.0 * M * N * K / (us * 1e-6) / 1e12


def _err_stats(c, ref):
    """(max abs error, relative Frobenius error) against the f32 reference."""
    d = (c.float() - ref).abs()
    return float(d.max()), float(d.norm() / ref.norm().clamp_min(1e-12))


def _build_case(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    dtype="bf16",
    *,
    cluster_m=1,
    cluster_n=1,
    const_val=None,
    tdm_balance=0,
    wmma_b2b=0,
):
    """Inputs, a make_args(stream) thunk, the f32 reference, and tolerances."""
    if (N // tile_n) % cluster_n:
        raise ValueError(f"cluster_n={cluster_n} needs N/tile_n={N // tile_n} to be a multiple of it")
    dt = _DT[dtype]
    if const_val is None:
        a = torch.randn(M, K, dtype=torch.float32, device="cuda").to(dt)
        b = torch.randn(N, K, dtype=torch.float32, device="cuda").to(dt)
    else:
        a = torch.full((M, K), const_val, dtype=dt, device="cuda")
        b = torch.full((N, K), const_val, dtype=dt, device="cuda")
    c = torch.zeros(M, N, dtype=dt, device="cuda")
    ref = torch.matmul(a.float(), b.float().T)

    def make_args(stream):
        return (
            flyc.from_c_void_p(fx.Uint8, c.data_ptr()),
            flyc.from_c_void_p(fx.Uint8, a.data_ptr()),
            flyc.from_c_void_p(fx.Uint8, b.data_ptr()),
            M,
            stream,
            N,
            K,
            K,  # lda
            K,  # ldb
            N,  # ldc
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            num_buffers,
            1 if dtype == "f16" else 0,
            cluster_m,
            cluster_n,
            tdm_balance,
            wmma_b2b,
        )

    # bf16/f16 inputs with f32 accumulation: error grows with sqrt(K).
    tol = (2e-2, 2e-2 * math.sqrt(K))
    return c, make_args, ref, tol


def _bench_us(launch, *, warmup=10, iters=100):
    """Median per-launch latency (us) from saturated back-to-back throughput."""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    times.sort()
    return times[len(times) // 2]


def _run_case(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, dtype="bf16", **kw):
    _require_gfx1250()
    c, make_args, ref, (rtol, atol) = _build_case(
        M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, dtype, **kw
    )
    compiled = flyc.compile(launch_gemm_bf16, *make_args(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    max_err, rel_err = _err_stats(c, ref)
    print(
        f"  {M}x{N}x{K} {dtype} cluster={kw.get('cluster_m', 1)},{kw.get('cluster_n', 1)}: "
        f"max_err={max_err:.4g} rel_err={rel_err:.3g}"
    )
    torch.testing.assert_close(c.float(), ref, rtol=rtol, atol=atol)
    return c, make_args, compiled


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
_CASES = [
    (8, 256, 512, 16, 128, 128, 1, 2, 2),  # skinny M: exercises the per-tile M clamp
    (16, 256, 1024, 16, 128, 128, 1, 2, 4),  # MAB-shaped tile: 4 accs/wave, 4-deep ring
    (64, 64, 256, 64, 64, 64, 1, 1, 2),  # single wave issues both TDMs
    (128, 128, 512, 128, 128, 128, 2, 2, 2),
    (129, 512, 1024, 128, 128, 128, 2, 2, 2),  # ragged M
    (256, 256, 512, 128, 256, 128, 2, 4, 2),
]


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers", _CASES)
def test_gemm_bf16(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers):
    _run_case(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_gemm_dtypes(dtype):
    _run_case(128, 128, 512, 128, 128, 128, 2, 2, 2, dtype=dtype)


# A multicasts along N, B along M. Kept small on purpose: a single 16-wide
# cluster dim is known to wedge the queue on gfx1250.
@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 1), (1, 2), (2, 2)])
def test_gemm_cluster(cluster_m, cluster_n):
    # 4 M-tiles x 4 N-tiles so every cluster shape has real peers to span.
    _run_case(64, 512, 512, 16, 128, 128, 1, 2, 2, cluster_m=cluster_m, cluster_n=cluster_n)


def _parse_csv_ints(value, n, name):
    parts = [int(x) for x in value.split(",")]
    if len(parts) != n:
        raise SystemExit(f"-{name} needs {n} comma-separated ints, got {value!r}")
    return parts


def _parse_init_mode(value):
    """'random' -> None; 'const' or 'const,<float>' -> constant A/B fill (default 0.1)."""
    if value == "random":
        return None
    if value == "const":
        return 0.1
    kind, _, num = value.partition(",")
    if kind == "const" and num:
        return float(num)
    raise SystemExit(f"--init-mode expects 'random', 'const', or 'const,<float>', got {value!r}")


def _print_table(headers, rows):
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


def _main():
    import argparse
    import itertools

    parser = argparse.ArgumentParser(description="Manual correctness/perf run for the gfx1250 bf16 GEMM")
    parser.add_argument("-mnk", nargs="+", required=True, metavar="M,N,K", help="one or more shapes")
    parser.add_argument("-tiles", required=True, help="tile_m,tile_n,tile_k")
    parser.add_argument("-warps", required=True, help="m_warp,n_warp")
    parser.add_argument("-nb", type=int, required=True, help="num_buffers")
    parser.add_argument("-dtype", default="bf16", choices=["bf16", "f16"], nargs="+")
    parser.add_argument("-cluster", default="1,1", help="cluster_m,cluster_n (1,1 disables clustering)")
    parser.add_argument("-bench", action="store_true", help="also measure perf (warmup=10, iters=100)")
    parser.add_argument(
        "--init-mode",
        nargs="+",
        default=["random", "const"],
        metavar="MODE",
        help="A/B fill(s): 'random' and/or 'const' (0.1) or 'const,<float>' (default: both)",
    )
    parser.add_argument(
        "--tdm-balance",
        nargs="+",
        type=int,
        default=[0],
        choices=[0, 1],
        metavar="0|1",
        help="TDM staging: 0 = one fat descriptor per operand, 1 = each operand halved "
        "over two waves. Pass both to sweep (default: 0)",
    )
    parser.add_argument(
        "--wmma-b2b",
        nargs="+",
        type=int,
        default=[0],
        choices=[0, 1],
        metavar="0|1",
        help="1 sets SCHED_MODE.DISABLE_XDL_ARB_STALL so a wave can issue back-to-back "
        "WMMAs. Pass both to sweep (default: 0)",
    )
    args = parser.parse_args()

    shapes = [_parse_csv_ints(v, 3, "mnk") for v in args.mnk]
    tiles = _parse_csv_ints(args.tiles, 3, "tiles")
    warps = _parse_csv_ints(args.warps, 2, "warps")
    cluster = _parse_csv_ints(args.cluster, 2, "cluster")
    dtypes = args.dtype if isinstance(args.dtype, list) else [args.dtype]

    rows = []
    sweep = itertools.product(shapes, dtypes, args.init_mode, args.tdm_balance, args.wmma_b2b)
    for (M, N, K), dtype, init, tdm_bal, b2b in sweep:
        c, make_args, ref, (rtol, atol) = _build_case(
            M,
            N,
            K,
            *tiles,
            *warps,
            args.nb,
            dtype,
            cluster_m=cluster[0],
            cluster_n=cluster[1],
            const_val=_parse_init_mode(init),
            tdm_balance=tdm_bal,
            wmma_b2b=b2b,
        )
        compiled = flyc.compile(launch_gemm_bf16, *make_args(torch.cuda.current_stream()))
        torch.cuda.synchronize()
        max_err, rel_err = _err_stats(c, ref)
        ok = torch.allclose(c.float(), ref, rtol=rtol, atol=atol)
        perf = ["-", "-", "-"]
        if args.bench:
            us = _bench_us(lambda: compiled(*make_args(torch.cuda.current_stream())))
            moved = _bytes_moved(M, N, K)
            perf = [f"{us:.3f}", f"{_tflops(M, N, K, us):.2f}", f"{moved / (us * 1e-6) / 1e12:.3f}"]
        rows.append(
            [
                str(M),
                str(N),
                str(K),
                dtype,
                init,
                str(tdm_bal),
                str(b2b),
                *perf,
                f"{max_err:.4g}",
                f"{rel_err:.3g}",
                "PASS" if ok else "FAIL",
            ]
        )

    print(
        f"\ntiles={tiles[0]},{tiles[1]},{tiles[2]} warps={warps[0]},{warps[1]} "
        f"nb={args.nb} cluster={cluster[0]},{cluster[1]}"
    )
    _print_table(
        [
            "M",
            "N",
            "K",
            "dtype",
            "init_mode",
            "tdm_bal",
            "wmma_b2b",
            "latency us",
            "TFLOPS",
            "BW TB/s",
            "max_err",
            "rel_err",
            "result",
        ],
        rows,
    )
    if any(r[-1] == "FAIL" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
