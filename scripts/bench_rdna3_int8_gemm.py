#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the gfx11 FlyDSL INT8 WMMA GEMM."""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_BUILD = os.path.join(_REPO, "build-fly", "python_packages")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import flydsl  # noqa: E402,F401 -- preload comgr before torch/HIP loads LLVM

_TARGET_ROUND_MS = 2.0
_MAX_GRAPH_LAUNCHES = 20


def _elapsed_us(run, count: int) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / count


def _bench_us(fn, *, warmup: int, iters: int, use_cudagraph: bool, rounds: int = 5) -> float:
    """Median microseconds per call.

    Each capture holds several launches, sized to roughly ``_TARGET_ROUND_MS``:
    replaying a graph that holds a single small kernel measures the replay
    overhead (~9 us on this machine) as much as the kernel itself. The median
    over rounds drops the occasional round that lands several percent high on a
    shared machine, which is enough to invert a comparison.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    run, per_run = fn, 1
    if use_cudagraph:
        per_call_us = max(_elapsed_us(fn, 5), 1e-1)
        per_run = max(1, min(_MAX_GRAPH_LAUNCHES, round(_TARGET_ROUND_MS * 1e3 / per_call_us)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(per_run):
                fn()
        run = graph.replay

        for _ in range(max(1, warmup // per_run)):
            run()
        torch.cuda.synchronize()

    per_round = max(1, iters // rounds // per_run)
    times = sorted(_elapsed_us(run, per_round) / per_run for _ in range(rounds))
    return times[len(times) // 2]


_ROCBLAS_NONE_OP = 111
_ROCBLAS_TRANS_OP = 112
_ROCBLAS_I8_R = 160
_ROCBLAS_I32_R = 162
_ROCBLAS_ALGO_STANDARD = 0


def _rocblas_gemm_ex():
    """Bind ``rocblas_gemm_ex`` for an INT8 GEMM with an INT32 accumulator.

    torch._int_mm always dispatches to hipBLASLt, and
    ``preferred_blas_library`` does not redirect it, so reaching rocBLAS at all
    means calling it directly.
    """
    import ctypes

    library = ctypes.CDLL("librocblas.so")
    handle = ctypes.c_void_p()
    if library.rocblas_create_handle(ctypes.byref(handle)):
        raise RuntimeError("rocblas_create_handle failed")
    library.rocblas_gemm_ex.restype = ctypes.c_int
    library.rocblas_gemm_ex.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]  # fmt: skip
    alpha = ctypes.c_int32(1)
    beta = ctypes.c_int32(0)

    def run(C, m, n, k, op_a, buf_a, lda, op_b, buf_b, ldb):
        import torch

        library.rocblas_set_stream(handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        status = library.rocblas_gemm_ex(
            handle, op_a, op_b,
            m, n, k, ctypes.byref(alpha),
            ctypes.c_void_p(buf_a.data_ptr()), _ROCBLAS_I8_R, lda,
            ctypes.c_void_p(buf_b.data_ptr()), _ROCBLAS_I8_R, ldb, ctypes.byref(beta),
            ctypes.c_void_p(C.data_ptr()), _ROCBLAS_I32_R, C.shape[1],
            ctypes.c_void_p(C.data_ptr()), _ROCBLAS_I32_R, C.shape[1],
            _ROCBLAS_I32_R, _ROCBLAS_ALGO_STANDARD, 0, 0,
        )  # fmt: skip
        if status:
            raise RuntimeError(f"rocblas_gemm_ex failed with status {status}")

    return run


def _rocblas_layouts(A, B_T):
    """The four transpose combinations that produce a row-major C.

    rocBLAS is column-major, so a row-major operand is already transposed
    there. Computing C^T = B_T . A^T lands the result in row-major C, and each
    operand can be fed either K-major or K-minor, which is what picks the NN /
    TN / NT / TT solution. TN needs no repacking at all: it takes exactly the
    buffers the FlyDSL kernel consumes.
    """
    m, k = A.shape
    n = B_T.shape[0]
    B = B_T.T.contiguous()
    A_T = A.T.contiguous()
    # rocBLAS sees the transposed problem, so its m/n are the shape's N/M.
    return {
        "NN": (n, m, k, _ROCBLAS_NONE_OP, B, n, _ROCBLAS_NONE_OP, A, k),
        "TN": (n, m, k, _ROCBLAS_TRANS_OP, B_T, k, _ROCBLAS_NONE_OP, A, k),
        "NT": (n, m, k, _ROCBLAS_NONE_OP, B, n, _ROCBLAS_TRANS_OP, A_T, m),
        "TT": (n, m, k, _ROCBLAS_TRANS_OP, B_T, k, _ROCBLAS_TRANS_OP, A_T, m),
    }


def _dequantise(accumulator, scale_a, scale_b, out_dtype):
    """Reference epilogue: the pass the fused kernel folds into its stores."""
    if scale_a is None:
        return accumulator if accumulator.dtype is out_dtype else accumulator.to(out_dtype)
    return (accumulator.float() * scale_a.unsqueeze(1) * scale_b.unsqueeze(0)).to(out_dtype)


def _tops(m: int, n: int, k: int, us: float) -> float:
    return (2.0 * m * n * k) / (us * 1e-6) / 1e12


def _parse_shape(text: str) -> tuple[int, int, int]:
    fields = text.lower().replace("*", "x").split("x")
    if len(fields) == 1:
        size = int(fields[0])
        return size, size, size
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(f"expected SIZE or MxNxK, got {text!r}")
    return tuple(int(field) for field in fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="1024,2048,4096",
        help="comma-separated SIZE or MxNxK shapes",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lds-layout", choices=("pad", "kblock"), default="pad")
    parser.add_argument("--out-dtype", choices=("i32", "f32", "bf16", "f16"), default="i32")
    parser.add_argument(
        "--scale-mode",
        choices=("none", "row_col"),
        default="none",
        help="row_col fuses per-row A and per-column B dequantisation into the epilogue",
    )
    parser.add_argument(
        "--autotuned",
        action="store_true",
        help="use the shape dispatcher (set FLYDSL_AUTOTUNE=1 to benchmark-search candidates)",
    )
    parser.add_argument("--no-sched-hint", action="store_true")
    parser.add_argument(
        "--no-cudagraph",
        action="store_true",
        help="measure direct launches (small kernels include HIP dispatch gaps)",
    )
    parser.add_argument("--stagger", type=int, default=1)
    parser.add_argument(
        "--split-k",
        type=int,
        default=1,
        help="K slices per output tile; each slice accumulates atomically into the i32 output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against torch._int_mm before timing",
    )
    parser.add_argument(
        "--torch-int-mm",
        action="store_true",
        help="also benchmark torch._int_mm (plus a dequant pass when --scale-mode=row_col)",
    )
    parser.add_argument(
        "--rocblas",
        action="store_true",
        help="also benchmark rocblas_gemm_ex directly, in all four transpose combinations "
        "(torch._int_mm always goes through hipBLASLt, and rocBLAS is layout-sensitive)",
    )
    args = parser.parse_args()
    if args.scale_mode == "row_col" and args.out_dtype == "i32":
        parser.error("--scale-mode=row_col dequantises to a float; pass --out-dtype f32/bf16/f16")

    import torch

    from flydsl.runtime.device import get_rocm_arch
    from kernels.gemm.rdna3_int8_gemm import create_wmma_int8_gemm_module
    from kernels.gemm.rdna3_int8_gemm_autotune import rdna3_int8_gemm_autotuned

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx11"):
        raise RuntimeError(f"RDNA3 INT8 WMMA requires gfx11*, got {arch!r}")

    out_torch_dtype = {
        "i32": torch.int32,
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
    }[args.out_dtype]
    scaled = args.scale_mode == "row_col"
    # Matches the tolerances the correctness tests use for the same epilogue.
    rtol, atol = {"i32": (0.0, 0.0), "f32": (0.0, 0.0), "bf16": (4e-3, 0.0), "f16": (1e-3, 2**-24)}[args.out_dtype]

    shapes = [_parse_shape(item.strip()) for item in args.shapes.split(",") if item.strip()]
    if args.autotuned:
        config = "config=autotuned"
    else:
        config = (
            f"layout={args.lds_layout} sched_hint={not args.no_sched_hint} "
            f"stagger={args.stagger} split_k={args.split_k}"
        )
    print(f"device={torch.cuda.get_device_name(0)} arch={arch}")
    print(
        f"{config} cudagraph={not args.no_cudagraph} "
        f"out_dtype={args.out_dtype} scale_mode={args.scale_mode} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(f"{'shape':>20s} {'implementation':>26s} {'time (us)':>12s} {'TOP/s':>10s}")

    for m, n, k in shapes:
        if args.autotuned:
            launch = None
            block_m = block_n = block_k = 0
        else:
            launch, block_m, block_n, block_k = create_wmma_int8_gemm_module(
                m,
                n,
                k,
                in_dtype="int8",
                out_dtype=args.out_dtype,
                scale_mode=args.scale_mode,
                lds_layout=args.lds_layout,
                sched_hint=not args.no_sched_hint,
                stagger=args.stagger,
                split_k=args.split_k,
            )

        torch.manual_seed(2026)
        A = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        B_T = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
        C = torch.empty((m, n), dtype=out_torch_dtype, device="cuda")
        scale_a = (torch.rand(m, device="cuda", dtype=torch.float32) * 0.01 + 0.001) if scaled else None
        scale_b = (torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001) if scaled else None

        def run_flydsl():
            # Resolve the stream at call time so CUDA Graph capture records the
            # launch on its capture stream rather than on the former default.
            if args.autotuned:
                rdna3_int8_gemm_autotuned(
                    C, A, B_T, scale_a, scale_b, out_dtype=args.out_dtype, scale_mode=args.scale_mode
                )
            else:
                launch(C, A, B_T, torch.cuda.current_stream(), scale_a, scale_b)

        run_flydsl()
        torch.cuda.synchronize()

        B = None
        accumulator = None
        reference = None
        if args.check or args.torch_int_mm or args.rocblas:
            B = B_T.T.contiguous()
            try:
                accumulator = torch._int_mm(A, B)
                reference = _dequantise(accumulator, scale_a, scale_b, out_torch_dtype)
                torch.cuda.synchronize()
            except (AttributeError, RuntimeError) as exc:
                if args.check:
                    raise RuntimeError("torch._int_mm is unavailable for correctness checking") from exc
                print(f"{m}x{n}x{k}: torch._int_mm unavailable: {exc}")

        if args.check:
            torch.testing.assert_close(C.float(), reference.float(), rtol=rtol, atol=atol)

        warmup = args.warmup if max(m, n, k) < 8192 else max(5, args.warmup // 2)
        iters = args.iters if max(m, n, k) < 8192 else max(10, args.iters // 2)
        us = _bench_us(run_flydsl, warmup=warmup, iters=iters, use_cudagraph=not args.no_cudagraph)
        tile = "auto" if args.autotuned else f"{block_m}x{block_n}x{block_k}"
        print(f"{m}x{n}x{k: <8d} {'FlyDSL (' + tile + ')':>26s} {us:12.2f} {_tops(m, n, k, us):10.2f}")

        suffix = " int_mm+deq" if scaled or out_torch_dtype is not torch.int32 else " int_mm"

        if args.torch_int_mm and reference is not None:
            assert B is not None

            def run_torch():
                _dequantise(torch._int_mm(A, B), scale_a, scale_b, out_torch_dtype)

            torch_us = _bench_us(run_torch, warmup=warmup, iters=iters, use_cudagraph=not args.no_cudagraph)
            label = "hipBLASLt" + suffix
            print(f"{m}x{n}x{k: <8d} {label:>26s} {torch_us:12.2f} {_tops(m, n, k, torch_us):10.2f}")

        if args.rocblas:
            rocblas_gemm = _rocblas_gemm_ex()
            rocblas_acc = torch.empty((m, n), dtype=torch.int32, device="cuda")
            timings = {}
            for name, gemm_args in _rocblas_layouts(A, B_T).items():

                def run_rocblas(gemm_args=gemm_args):
                    rocblas_gemm(rocblas_acc, *gemm_args)
                    _dequantise(rocblas_acc, scale_a, scale_b, out_torch_dtype)

                rocblas_acc.zero_()
                run_rocblas()
                torch.cuda.synchronize()
                if accumulator is not None:
                    torch.testing.assert_close(rocblas_acc, accumulator, atol=0, rtol=0)
                timings[name] = _bench_us(run_rocblas, warmup=warmup, iters=iters, use_cudagraph=not args.no_cudagraph)
            for name, rocblas_us in timings.items():
                best = " (best)" if rocblas_us == min(timings.values()) else ""
                label = f"rocBLAS {name}{suffix}{best}"
                print(f"{m}x{n}x{k: <8d} {label:>26s} {rocblas_us:12.2f} {_tops(m, n, k, rocblas_us):10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
