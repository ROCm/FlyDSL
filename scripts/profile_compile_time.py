#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Break down FlyDSL JIT compile time into phases.

The JIT ``__call__`` cost splits into a handful of distinct phases; this tool
times each one so regressions can be attributed to the right layer instead of
lumped into one opaque "compile time" number.

Phases reported (per kernel):

  fingerprint    toolchain cache-key fingerprint (``_flydsl_key``); a fixed
                 per-process cost, paid once (``@lru_cache``) on the first
                 compile of a fresh process.
  trace_build    Python AST tracing + MLIR module construction.
  mlir_pipeline  MLIR pass pipeline (lowering + canonicalize + ... + codegen).
  codegen        LLVM -> AMDGCN -> fatbin, a subset of mlir_pipeline (only
                 shown with ``--passes``, read from the MLIR pass timing report).
  origin_ir      line count of the origin MLIR module -- a proxy for IR size.
                 A kernel whose origin_ir grows with the problem size is fully
                 unrolling and will compile slower the larger it gets.

The ``fingerprint`` phase is measured in isolation (cache cleared) so it is
reported separately from the prewarmed per-kernel numbers.

Usage::

    python scripts/profile_compile_time.py --shape 16,2048 --dtype bf16
    python scripts/profile_compile_time.py --shape 16,128 --shape 16,2048 --passes
    python scripts/profile_compile_time.py --shape 16,128 --shape 16,2048 --compare

With ``--compare`` the same rmsnorm is also compiled with Triton and printed in
an aligned table, showing that Triton's vectorized kernel keeps a flat IR (and
compile time) as N grows while the fully unrolled FlyDSL kernel does not.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CompilePhases:
    """Wall-clock (ms) breakdown of a single JIT compile."""

    total: float = 0.0
    mlir_pipeline: float = 0.0
    trace_build: float = 0.0
    origin_ir_lines: int = -1
    per_pass: dict = field(default_factory=dict)


def _install_probes(enable_timing: bool):
    """Wrap ``MlirCompiler.compile`` to record its wall time + origin IR size.

    Returns ``(phases, restore)`` where ``phases`` is a mutable
    :class:`CompilePhases` populated on the next compile and ``restore`` undoes
    the monkeypatch.
    """
    from flydsl.compiler import jit_function as jf
    from flydsl.compiler.jit_function import MlirCompiler

    phases = CompilePhases()
    orig = MlirCompiler.compile.__func__

    def wrapped(cls, module, **kwargs):
        try:
            phases.origin_ir_lines = len(module.operation.get_asm().splitlines())
        except Exception:
            phases.origin_ir_lines = -1
        t0 = time.perf_counter()
        result = orig(cls, module, **kwargs)
        phases.mlir_pipeline = (time.perf_counter() - t0) * 1e3
        return result

    MlirCompiler.compile = classmethod(wrapped)

    restore_run_pipeline = None
    if enable_timing:
        # Swap in a pipeline runner that enables MLIR's per-pass timing report
        # (printed to stderr by the MLIR runtime at PassManager teardown).
        from flydsl._mlir.passmanager import PassManager

        orig_run = jf._run_pipeline

        def timed_run(module, fragments, *, verifier=False, print_after_all=False):
            pm = PassManager.parse(f"builtin.module({','.join(fragments)})")
            pm.enable_verifier(verifier)
            try:
                pm.enable_timing()
            except Exception:
                pass
            pm.run(module.operation)

        jf._run_pipeline = timed_run
        restore_run_pipeline = lambda: setattr(jf, "_run_pipeline", orig_run)  # noqa: E731

    def restore():
        MlirCompiler.compile = classmethod(orig)
        if restore_run_pipeline is not None:
            restore_run_pipeline()

    return phases, restore


def measure_fingerprint_ms() -> float:
    """Time the toolchain fingerprint (``_flydsl_key``) with a cold cache."""
    from flydsl.compiler import jit_function as jf

    jf._flydsl_key_cached.cache_clear()
    t0 = time.perf_counter()
    jf._flydsl_key()
    return (time.perf_counter() - t0) * 1e3


@contextmanager
def _compile_only_env():
    import os

    prev = {k: os.environ.get(k) for k in ("COMPILE_ONLY", "FLYDSL_RUNTIME_ENABLE_CACHE")}
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def profile_compile(invoke: Callable[[], None], *, passes: bool = False) -> CompilePhases:
    """Profile a single JIT compile.

    ``invoke`` is a zero-arg callable that triggers exactly one cold compile
    (e.g. ``lambda: launch(x, g, o, m, stream)``). The toolchain fingerprint is
    assumed already warmed (prewarm before calling this) so the returned numbers
    reflect the per-kernel cost only.
    """
    phases, restore = _install_probes(passes)
    try:
        t0 = time.perf_counter()
        invoke()
        phases.total = (time.perf_counter() - t0) * 1e3
    finally:
        restore()
    phases.trace_build = max(0.0, phases.total - phases.mlir_pipeline)
    return phases


# --------------------------------------------------------------------------- #
# Built-in rmsnorm example (also imported by the compile-time regression test).
# --------------------------------------------------------------------------- #


def build_rmsnorm_invoke(n: int, m: int, dtype: str):
    """Return ``(invoke, prewarm)`` closures for the rmsnorm launcher at (m, n)."""
    import torch

    from kernels.norm.rmsnorm_kernel import build_rmsnorm_module

    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[dtype]

    def make(nn, mm):
        launch = build_rmsnorm_module(nn, dtype)
        x = torch.randn(mm, nn, dtype=torch_dtype, device="cuda")
        g = torch.randn(nn, dtype=torch_dtype, device="cuda")
        o = torch.empty_like(x)
        stream = torch.cuda.current_stream()
        return lambda: launch(x, g, o, mm, stream)

    return make(n, m), make(64, 16)


def build_flydsl_loopvec_rmsnorm_invoke(n: int, m: int, dtype: str):
    """Return ``(invoke, prewarm)`` for a runtime-loop + vectorized rmsnorm.

    This is the Triton-aligned variant of the issue #862 kernel: it keeps the
    issue's vectorized layout-API loads (``load_vec`` over ``BufferCopy128b``)
    but replaces the ``range_constexpr`` (fully unrolled) column loop with a
    runtime ``range`` (``scf.for``). One program per row; a single wave does the
    reduction. IR size -- and therefore compile time -- stays flat as N grows.

    Restricted to 16-bit dtypes and power-of-two N (the issue's shapes), so the
    vectorization is exact with no tail masking.
    """
    import math

    import torch

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, range_constexpr
    from flydsl.expr import math as fmath
    from flydsl.expr.vector import ReductionOp
    from kernels.common.kernels_common import dtype_to_elem_type
    from kernels.norm.rmsnorm_common import (
        VEC_WIDTH,
        WARP_SIZE,
        load_vec,
        store_vec,
        to_elem_vec,
    )

    if dtype not in ("bf16", "f16"):
        raise ValueError("loopvec variant supports 16-bit dtypes only")
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    eps = 1e-5

    def build(nn):
        assert nn % VEC_WIDTH == 0 and (nn & (nn - 1)) == 0, "N must be power-of-two multiple of VEC_WIDTH"
        n_vecs = nn // VEC_WIDTH
        bt = min(WARP_SIZE, n_vecs)  # <= wave size -> single-wave reduce, no LDS
        assert n_vecs % bt == 0
        num_iters = n_vecs // bt
        log2_bt = int(math.log2(bt))
        elem_bits = 16
        elem_dtype = dtype_to_elem_type(dtype)

        @flyc.kernel(known_block_size=[bt, 1, 1])
        def rmsnorm_loopvec_kernel(Input: fx.Tensor, Gamma: fx.Tensor, Output: fx.Tensor):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            fm = arith.FastMathFlags.fast

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            thread_sumsq = fx.Float32(0.0)
            # Runtime loop (scf.for), NOT unrolled: op count independent of N.
            for it in range(fx.Int32(0), fx.Int32(num_iters), fx.Int32(1)):
                vi = fx.Int32(it) * fx.Int32(bt) + tid
                x = load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, vi).to(fx.Float32)
                thread_sumsq = thread_sumsq + (x * x).reduce(ReductionOp.ADD, fastmath=fm)

            w = thread_sumsq
            for _sh in range_constexpr(log2_bt):
                off = bt // (2 << _sh)
                w = w.addf(w.shuffle_xor(off, bt), fastmath=fm)
            rrms = fmath.rsqrt(w / float(nn) + eps, fastmath=fm)

            for it in range(fx.Int32(0), fx.Int32(num_iters), fx.Int32(1)):
                vi = fx.Int32(it) * fx.Int32(bt) + tid
                x = load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, vi).to(fx.Float32)
                g = load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, vi).to(fx.Float32)
                y_e = to_elem_vec(dtype, elem_dtype, False, (x * rrms) * g)
                store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, vi)

        @flyc.jit
        def launch(Input: fx.Tensor, Gamma: fx.Tensor, Output: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            rmsnorm_loopvec_kernel(Input, Gamma, Output).launch(grid=(m, 1, 1), block=(bt, 1, 1), stream=stream)

        return launch

    def make(nn, mm):
        launch = build(nn)
        x = torch.randn(mm, nn, dtype=torch_dtype, device="cuda")
        g = torch.randn(nn, dtype=torch_dtype, device="cuda")
        o = torch.empty_like(x)
        stream = torch.cuda.current_stream()
        return lambda: launch(x, g, o, stream), (x, g, o)

    invoke, tensors = make(n, m)
    prewarm, _ = make(64, m)
    return invoke, prewarm, tensors


# --------------------------------------------------------------------------- #
# Triton rmsnorm for an aligned FlyDSL-vs-Triton compile-time comparison.
#
# Triton is optional: these functions import it lazily and raise ImportError if
# it is absent, so the core tool/test stays usable without it.
# --------------------------------------------------------------------------- #


@dataclass
class TritonCompile:
    """Triton compile measurement for one shape."""

    jit_ms: float = 0.0
    ttgir_lines: int = -1
    amdgcn_lines: int = -1


def _triton_rmsnorm_kernel():
    """A minimal, vectorized (non-unrolled) Triton rmsnorm kernel.

    One program per row; the whole row is a single ``BLOCK_SIZE`` vector, so the
    op count -- and thus the IR -- is independent of N. This mirrors how Triton
    keeps compile time flat as N grows, in contrast to a fully unrolled kernel.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm(x_ptr, g_ptr, o_ptr, row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rrms = tl.rsqrt(var + eps)
        g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * g
        tl.store(o_ptr + row * row_stride + cols, y.to(o_ptr.dtype.element_ty), mask=mask)

    return _rmsnorm


def build_triton_rmsnorm_invoke(n: int, m: int, dtype: str):
    """Return ``(invoke, prewarm)`` closures for the Triton rmsnorm at (m, n)."""
    import torch
    import triton

    kernel = _triton_rmsnorm_kernel()
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[dtype]

    def make(nn, mm):
        x = torch.randn(mm, nn, dtype=torch_dtype, device="cuda")
        g = torch.randn(nn, dtype=torch_dtype, device="cuda")
        o = torch.empty_like(x)
        block = triton.next_power_of_2(nn)
        return lambda: kernel[(mm,)](x, g, o, x.stride(0), nn, 1e-5, block)

    return make(n, m), make(64, 16)


def profile_triton_compile(invoke: Callable[[], object]) -> TritonCompile:
    """Measure Triton first-compile time (first call minus steady state) + IR size.

    ``invoke`` must trigger a fresh specialization on its first call (a shape/
    constexpr not compiled before in this process).
    """
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled = invoke()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        invoke()
    torch.cuda.synchronize()
    steady_ms = (time.perf_counter() - t0) * 1e3 / 20

    asm = getattr(compiled, "asm", {}) or {}
    return TritonCompile(
        jit_ms=max(0.0, first_ms - steady_ms),
        ttgir_lines=len(asm["ttgir"].splitlines()) if "ttgir" in asm else -1,
        amdgcn_lines=len(asm["amdgcn"].splitlines()) if "amdgcn" in asm else -1,
    )


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shape",
        action="append",
        type=lambda s: tuple(int(v) for v in s.split(",")),
        help="M,N shape (repeatable). Default: 16,128 and 16,2048.",
    )
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "f16", "f32"))
    parser.add_argument("--passes", action="store_true", help="also dump the MLIR per-pass timing report")
    parser.add_argument(
        "--compare", action="store_true", help="also compile the same rmsnorm with Triton and print an aligned table"
    )
    args = parser.parse_args(argv)

    shapes = args.shape or [(16, 128), (16, 2048)]

    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA/ROCm device required for compile profiling.", file=sys.stderr)
            return 1
    except ImportError:
        print("torch is required.", file=sys.stderr)
        return 1

    with _compile_only_env():
        fp_ms = measure_fingerprint_ms()
        print(f"fingerprint (toolchain _flydsl_key, once per process): {fp_ms:7.1f} ms\n")

        # Prewarm so the fingerprint is excluded from per-kernel numbers below.
        _, prewarm = build_rmsnorm_invoke(shapes[0][1], shapes[0][0], args.dtype)
        prewarm()

        if not args.compare:
            print(f"{'shape':>12} {'total':>8} {'trace':>8} {'mlir':>8} {'origin_IR':>10}")
            for m, n in shapes:
                invoke, _ = build_rmsnorm_invoke(n, m, args.dtype)
                ph = profile_compile(invoke, passes=args.passes)
                print(
                    f"{f'{m}x{n}':>12} {ph.total:>7.1f}m {ph.trace_build:>7.1f}m "
                    f"{ph.mlir_pipeline:>7.1f}m {ph.origin_ir_lines:>10}"
                )
            return 0

        try:
            build_triton_rmsnorm_invoke(shapes[0][1], shapes[0][0], args.dtype)[1]()
        except ImportError:
            print("Triton not installed; --compare requires triton.", file=sys.stderr)
            return 1
        build_flydsl_loopvec_rmsnorm_invoke(shapes[0][1], shapes[0][0], args.dtype)[1]()

        hdr1 = f"{'FlyDSL unrolled':^20} | {'FlyDSL loop+vec':^20} | {'Triton':^18}"
        hdr2 = f"{'jit_ms':>8} {'origin_IR':>10} | {'jit_ms':>8} {'origin_IR':>10} | {'jit_ms':>8} {'ttgir':>8}"
        print(f"{'':>10} | {hdr1}")
        print(f"{'shape':>10} | {hdr2}")
        for m, n in shapes:
            fly, _ = build_rmsnorm_invoke(n, m, args.dtype)
            ph_u = profile_compile(fly)
            lv, _, _ = build_flydsl_loopvec_rmsnorm_invoke(n, m, args.dtype)
            ph_v = profile_compile(lv)
            tri, _ = build_triton_rmsnorm_invoke(n, m, args.dtype)
            tc = profile_triton_compile(tri)
            print(
                f"{f'{m}x{n}':>10} | {ph_u.total:>7.1f}m {ph_u.origin_ir_lines:>10} | "
                f"{ph_v.total:>7.1f}m {ph_v.origin_ir_lines:>10} | {tc.jit_ms:>7.1f}m {tc.ttgir_lines:>8}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
