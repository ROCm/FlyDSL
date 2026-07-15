#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""JIT compile-time characterization for the rmsnorm kernel (issue #862).

This does not test a fix; it *measures* where the JIT compile time goes for the
current kernel so the numbers are visible in CI logs and a catastrophic
regression fails the build. It uses ``scripts/profile_compile_time.py`` to split
each compile into its phases:

  * ``fingerprint``  -- toolchain cache-key hash, a fixed per-process cost.
  * ``trace_build``  -- Python AST tracing + MLIR module construction.
  * ``mlir_pipeline``-- MLIR pass pipeline (lowering + canonicalize + codegen).
  * ``origin_ir``    -- origin MLIR line count, a proxy for IR size.

The small-N rmsnorm kernel fully unrolls its per-element loops, so ``origin_ir``
(and hence compile time) grows with N -- that growth is what makes FlyDSL's JIT
compile slower than Triton's for these shapes. The test records that behavior
and guards a generous upper bound only.
"""

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower, pytest.mark.benchmark]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping compile-time tests.", allow_module_level=True)

# scripts/ lives at the repo root and is not an installed package; make it importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.profile_compile_time import (  # noqa: E402
    _compile_only_env,
    build_flydsl_loopvec_rmsnorm_invoke,
    build_rmsnorm_invoke,
    build_triton_rmsnorm_invoke,
    measure_fingerprint_ms,
    profile_compile,
    profile_triton_compile,
)


def _reference_rmsnorm(x, g, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * g


# Shapes exercise the small-N (fully unrolled) rmsnorm path across a 16x range in
# N, which is where compile time scales with the unrolled IR size.
_SHAPES = [(16, 128), (16, 512), (16, 2048)]

# Generous ceiling: this is a regression smoke guard, not a perf target. It must
# not be flaky on a loaded CI box, so it is set well above observed values
# (~0.15-0.9 s per shape on MI300X) rather than near them.
_MAX_COMPILE_MS = 10_000.0


def test_rmsnorm_compile_time_breakdown(capsys):
    """Measure and report the rmsnorm JIT compile-time breakdown vs N."""
    with _compile_only_env():
        fingerprint_ms = measure_fingerprint_ms()

        # Prewarm so the fingerprint (once-per-process) is excluded from the
        # per-kernel numbers below.
        _, prewarm = build_rmsnorm_invoke(_SHAPES[0][1], _SHAPES[0][0], "bf16")
        prewarm()

        rows = []
        for m, n in _SHAPES:
            invoke, _ = build_rmsnorm_invoke(n, m, "bf16")
            ph = profile_compile(invoke)
            rows.append((m, n, ph))

    # Human-readable table (visible with `pytest -s`).
    lines = [
        f"\nfingerprint (toolchain, once/process): {fingerprint_ms:.1f} ms",
        f"{'shape':>12} {'total':>9} {'trace':>9} {'mlir':>9} {'origin_IR':>10}",
    ]
    for m, n, ph in rows:
        lines.append(
            f"{f'{m}x{n}':>12} {ph.total:>7.1f}ms {ph.trace_build:>7.1f}ms "
            f"{ph.mlir_pipeline:>7.1f}ms {ph.origin_ir_lines:>10}"
        )
    report = "\n".join(lines)
    with capsys.disabled():
        print(report)

    # Sanity: every phase produced a real measurement.
    for m, n, ph in rows:
        assert ph.total > 0.0, f"{m}x{n}: no compile measured"
        assert ph.origin_ir_lines > 0, f"{m}x{n}: origin IR not captured"
        assert ph.mlir_pipeline <= ph.total + 1.0
        # Regression smoke guard (generous; catches only catastrophic blowups).
        assert ph.total < _MAX_COMPILE_MS, f"{m}x{n}: compile took {ph.total:.0f} ms (> {_MAX_COMPILE_MS:.0f} ms)"


def test_rmsnorm_origin_ir_grows_with_n():
    """Document the root cause: the unrolled kernel's origin IR scales with N.

    This is deterministic (IR line count is machine-independent), unlike wall
    time, so it is the stable signal. If the kernel is later changed to stop
    fully unrolling (e.g. runtime scf.for), this expectation flips -- update the
    test alongside that change.
    """
    with _compile_only_env():
        _, prewarm = build_rmsnorm_invoke(128, 16, "bf16")
        prewarm()

        small, _ = build_rmsnorm_invoke(128, 16, "bf16")
        large, _ = build_rmsnorm_invoke(2048, 16, "bf16")
        ph_small = profile_compile(small)
        ph_large = profile_compile(large)

    assert ph_small.origin_ir_lines > 0 and ph_large.origin_ir_lines > 0
    # 16x more columns -> materially larger IR under full unrolling.
    assert ph_large.origin_ir_lines > 2 * ph_small.origin_ir_lines, (
        f"expected unrolled IR to grow with N: "
        f"N=128 -> {ph_small.origin_ir_lines} lines, N=2048 -> {ph_large.origin_ir_lines} lines"
    )


def test_rmsnorm_compile_time_vs_triton(capsys):
    """Aligned FlyDSL-vs-Triton compile comparison for the same rmsnorm shapes.

    Reports both compile times side by side and asserts the deterministic root
    cause of the gap: FlyDSL's unrolled origin IR grows with N, while Triton's
    vectorized kernel keeps a flat op count (TTGIR line count independent of N).
    """
    triton = pytest.importorskip("triton")
    del triton

    with _compile_only_env():
        _, prewarm = build_rmsnorm_invoke(128, 16, "bf16")
        prewarm()
        build_triton_rmsnorm_invoke(128, 16, "bf16")[1]()

        rows = []
        for m, n in _SHAPES:
            fly, _ = build_rmsnorm_invoke(n, m, "bf16")
            ph = profile_compile(fly)
            tri, _ = build_triton_rmsnorm_invoke(n, m, "bf16")
            tc = profile_triton_compile(tri)
            rows.append((m, n, ph, tc))

    lines = [
        f"\n{'':>12} | {'FlyDSL (unrolled)':^22} | {'Triton (vectorized)':^24}",
        f"{'shape':>12} | {'jit_ms':>8} {'origin_IR':>10} | {'jit_ms':>8} {'ttgir':>7} {'amdgcn':>7}",
    ]
    for m, n, ph, tc in rows:
        lines.append(
            f"{f'{m}x{n}':>12} | {ph.total:>7.1f}m {ph.origin_ir_lines:>10} | "
            f"{tc.jit_ms:>7.1f}m {tc.ttgir_lines:>7} {tc.amdgcn_lines:>7}"
        )
    with capsys.disabled():
        print("\n".join(lines))

    small = rows[0]
    large = rows[-1]
    # FlyDSL: unrolled IR grows materially with N.
    assert large[2].origin_ir_lines > 2 * small[2].origin_ir_lines
    # Triton: vectorized op count stays flat -> TTGIR essentially constant in N.
    if small[3].ttgir_lines > 0 and large[3].ttgir_lines > 0:
        assert large[3].ttgir_lines <= small[3].ttgir_lines + 20, (
            f"expected flat Triton TTGIR across N, got "
            f"N={small[1]} -> {small[3].ttgir_lines}, N={large[1]} -> {large[3].ttgir_lines}"
        )


def test_rmsnorm_loopvec_is_correct_and_flat():
    """The runtime-loop + vectorized rmsnorm is correct and compiles flat in N.

    This is the Triton-aligned variant of the issue #862 kernel: vectorized
    loads (as in the issue's kernel) but a runtime ``scf.for`` instead of a fully
    unrolled loop. Correctness is checked against a torch reference; compile-time
    flatness is checked via origin IR line count (deterministic).
    """
    # Correctness (requires execution, so outside COMPILE_ONLY).
    for n in (128, 2048):
        invoke, _, (x, g, o) = build_flydsl_loopvec_rmsnorm_invoke(n, 16, "bf16")
        invoke()
        torch.cuda.synchronize()
        torch.testing.assert_close(o.float(), _reference_rmsnorm(x, g).float(), atol=2e-2, rtol=2e-2)

    # Flatness: origin IR must not scale with N (contrast the unrolled kernel,
    # which grows ~6x over this range -- see test_rmsnorm_origin_ir_grows_with_n).
    with _compile_only_env():
        build_flydsl_loopvec_rmsnorm_invoke(128, 16, "bf16")[1]()
        small = profile_compile(build_flydsl_loopvec_rmsnorm_invoke(128, 16, "bf16")[0])
        large = profile_compile(build_flydsl_loopvec_rmsnorm_invoke(2048, 16, "bf16")[0])

    assert small.origin_ir_lines > 0 and large.origin_ir_lines > 0
    assert large.origin_ir_lines <= small.origin_ir_lines + 20, (
        f"expected flat loop+vec IR across N, got "
        f"N=128 -> {small.origin_ir_lines}, N=2048 -> {large.origin_ir_lines}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
