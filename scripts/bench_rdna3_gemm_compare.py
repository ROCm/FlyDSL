#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare RDNA3 GEMM: FlyDSL TN / FlyDSL NN / Triton max-autotune.

Fair Triton path (default):
  * per-shape torch.compile (no dynamic-shape reuse across sizes)
  * no in-place copy_ (return torch.mm result so cudagraphs can arm)
  * mark tensor dims static before compile

FlyDSL RDNA3 kernel is TN-native (C = A[M,K] @ B_T[N,K].T).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)


def _bench_us(fn, *, warmup: int, iters: int, cudagraph_step: bool = False) -> float:
    import torch

    def _one():
        if cudagraph_step:
            torch.compiler.cudagraph_mark_step_begin()
        fn()

    for _ in range(warmup):
        _one()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _one()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def _tflops(m: int, n: int, k: int, us: float) -> float:
    return (2.0 * m * n * k) / (us * 1e-6) / 1e12


def _mark_static(*tensors) -> None:
    import torch

    for t in tensors:
        for dim in range(t.ndim):
            torch._dynamo.mark_static(t, dim)


def _compile_triton(layout: str, example_args, *, use_copy: bool):
    """Compile a fresh Triton max-autotune GEMM for one static shape."""
    import torch
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "TRITON"
    inductor_config.max_autotune_gemm_search_space = "DEFAULT"
    # max-autotune already prefers cudagraphs; be explicit.
    inductor_config.triton.cudagraphs = True
    torch._dynamo.reset()

    if layout == "nn":
        if use_copy:

            def gemm(a, b, c):
                c.copy_(torch.mm(a, b))
                return c

        else:

            def gemm(a, b):
                return torch.mm(a, b)

    elif layout == "tn":
        if use_copy:

            def gemm(a, bt, c):
                c.copy_(torch.mm(a, bt.t()))
                return c

        else:

            def gemm(a, bt):
                return torch.mm(a, bt.t())

    else:
        raise ValueError(layout)

    compiled = torch.compile(gemm, mode="max-autotune", fullgraph=True, dynamic=False)
    _mark_static(*example_args)
    t0 = time.time()
    out = compiled(*example_args)
    torch.cuda.synchronize()
    return compiled, out, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shapes",
        default="256,1024,2048,4096,8192",
        help="comma-separated square sizes (M=N=K)",
    )
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--nn-include-transpose",
        action="store_true",
        help="time B.T.contiguous() inside FlyDSL NN path",
    )
    p.add_argument(
        "--triton-copy",
        action="store_true",
        help="old unfair path: c.copy_(torch.mm(...)) disables cudagraphs",
    )
    p.add_argument("--skip-triton", action="store_true")
    p.add_argument("--skip-flydsl", action="store_true")
    args = p.parse_args()

    import torch

    from flydsl.runtime.device import get_rocm_arch

    arch = str(get_rocm_arch() or "")
    print(f"device={torch.cuda.get_device_name(0)} arch={arch}")
    print(f"triton_mode={'copy_(unfair)' if args.triton_copy else 'return+per-shape+static+cudagraph'}")
    if not arch.startswith("gfx11"):
        print(f"WARNING: expected gfx11*, got {arch}")

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]

    flydsl_launch = None
    if not args.skip_flydsl:
        from kernels.gemm.rdna3_f16_gemm_autotune import rdna3_gemm_autotuned

        flydsl_launch = rdna3_gemm_autotuned

    header = f"{'shape':>14s} {'impl':22s} {'us':>10s} {'TFLOPS':>8s} {'vs FlyDSL-TN':>12s}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for n in shapes:
        m = k = n
        # fewer iters for huge shapes
        warmup = args.warmup if n < 8192 else max(5, args.warmup // 2)
        iters = args.iters if n < 8192 else max(10, args.iters // 2)

        scale = 0.01
        A = torch.randn(m, k, dtype=torch_dtype, device="cuda") * scale
        B = torch.randn(k, n, dtype=torch_dtype, device="cuda") * scale
        B_T = B.t().contiguous()
        C = torch.empty(m, n, dtype=torch_dtype, device="cuda")

        ref_nn = torch.mm(A, B)
        ref_tn = torch.mm(A, B_T.t())
        results = {}
        rows = []

        if flydsl_launch is not None:
            C.zero_()
            flydsl_launch(C, A, B_T)
            torch.cuda.synchronize()
            maxdiff_tn = (C.float() - ref_tn.float()).abs().max().item()

            def run_tn():
                flydsl_launch(C, A, B_T)

            us_tn = _bench_us(run_tn, warmup=warmup, iters=iters)
            results["flydsl_tn"] = us_tn
            rows.append((f"{m}x{n}x{k}", "flydsl_tn", us_tn, maxdiff_tn))

            if args.nn_include_transpose:

                def run_nn():
                    flydsl_launch(C, A, B.t().contiguous())

            else:
                B_T2 = B.t().contiguous()

                def run_nn():
                    flydsl_launch(C, A, B_T2)

            flydsl_launch(C, A, B.t().contiguous())
            torch.cuda.synchronize()
            maxdiff_nn = (C.float() - ref_nn.float()).abs().max().item()
            us_nn = _bench_us(run_nn, warmup=warmup, iters=iters)
            results["flydsl_nn"] = us_nn
            tag = "flydsl_nn+T" if args.nn_include_transpose else "flydsl_nn"
            rows.append((f"{m}x{n}x{k}", tag, us_nn, maxdiff_nn))

        if not args.skip_triton:
            if args.triton_copy:
                triton_nn, _, t_nn = _compile_triton("nn", (A, B, C), use_copy=True)
                triton_tn, _, t_tn = _compile_triton("tn", (A, B_T, C), use_copy=True)
                print(f"[{m}] triton compile nn={t_nn:.1f}s tn={t_tn:.1f}s", flush=True)

                out = triton_nn(A, B, C)
                torch.cuda.synchronize()
                maxdiff = (out.float() - ref_nn.float()).abs().max().item()

                def run_triton_nn():
                    triton_nn(A, B, C)

                us = _bench_us(run_triton_nn, warmup=warmup, iters=iters)
                results["triton_auto_nn"] = us
                rows.append((f"{m}x{n}x{k}", "triton_auto_nn", us, maxdiff))

                out = triton_tn(A, B_T, C)
                torch.cuda.synchronize()
                maxdiff = (out.float() - ref_tn.float()).abs().max().item()

                def run_triton_tn():
                    triton_tn(A, B_T, C)

                us = _bench_us(run_triton_tn, warmup=warmup, iters=iters)
                results["triton_auto_tn"] = us
                rows.append((f"{m}x{n}x{k}", "triton_auto_tn", us, maxdiff))
            else:
                triton_nn, out_nn, t_nn = _compile_triton("nn", (A, B), use_copy=False)
                # Clone before next cudagraph run overwrites the pooled output.
                check_nn = out_nn.detach().clone()
                triton_tn, out_tn, t_tn = _compile_triton("tn", (A, B_T), use_copy=False)
                check_tn = out_tn.detach().clone()
                print(f"[{m}] triton compile nn={t_nn:.1f}s tn={t_tn:.1f}s", flush=True)

                torch.cuda.synchronize()
                maxdiff = (check_nn.float() - ref_nn.float()).abs().max().item()

                def run_triton_nn():
                    triton_nn(A, B)

                us = _bench_us(run_triton_nn, warmup=warmup, iters=iters, cudagraph_step=True)
                results["triton_auto_nn"] = us
                rows.append((f"{m}x{n}x{k}", "triton_auto_nn", us, maxdiff))

                torch.cuda.synchronize()
                maxdiff = (check_tn.float() - ref_tn.float()).abs().max().item()

                def run_triton_tn():
                    triton_tn(A, B_T)

                us = _bench_us(run_triton_tn, warmup=warmup, iters=iters, cudagraph_step=True)
                results["triton_auto_tn"] = us
                rows.append((f"{m}x{n}x{k}", "triton_auto_tn", us, maxdiff))

        def run_torch():
            torch.mm(A, B, out=C)

        us_torch = _bench_us(run_torch, warmup=warmup, iters=iters)
        results["torch_mm_nn"] = us_torch
        rows.append((f"{m}x{n}x{k}", "torch_mm_nn", us_torch, 0.0))

        base = results.get("flydsl_tn")
        for shape, impl, us, maxdiff in rows:
            tf = _tflops(m, n, k, us)
            rel = "-" if base is None else f"{base / us:.2f}x"
            print(
                f"{shape:>14s} {impl:22s} {us:10.1f} {tf:8.2f} {rel:>12s}  maxdiff={maxdiff:.3e}",
                flush=True,
            )

    print("=" * len(header))
    print(
        "Notes: FlyDSL RDNA3 kernel is TN-native. flydsl_nn reuses TN with B_T=B.T "
        "(prep outside timing unless --nn-include-transpose). "
        "Fair Triton: per-shape compile, static dims, return mm (cudagraph-friendly)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
