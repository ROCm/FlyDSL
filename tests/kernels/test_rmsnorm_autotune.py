#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU integration test for direct-mode autotuned RMSNorm (#770).

Verifies the two-track direct-mode autotuner end to end:
  - zero-search default run produces correct output
  - forced-search (FLYDSL_AUTOTUNE=1) sweeps configs, picks one, stays correct
  - the tuned result is cached (a second call does not re-tune)
  - structural block sizes are JIT Constexpr specializations, including the
    ``known_block_size`` metadata required above AMDGPU's default limit
"""

import re

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

from kernels.norm.rmsnorm_autotune import rmsnorm_autotuned  # noqa: E402
from kernels.norm.rmsnorm_kernel import rmsnorm_direct  # noqa: E402

EPS = 1e-5


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Clear the tuner's in-memory cache so tuned results don't leak across
    tests (the disk cache is isolated per test via FLYDSL_AUTOTUNE_CACHE_DIR)."""
    rmsnorm_autotuned.tuner.cache.clear()
    yield
    rmsnorm_autotuned.tuner.cache.clear()


def _reference(x, g):
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + EPS)) * g.float()


def test_rmsnorm_autotuned_uses_direct_mode():
    """The adopter must rely on JIT Constexpr specialization, not a second
    builder/cache/routing path in the autotuner."""
    assert rmsnorm_autotuned.tuner.fn is rmsnorm_direct


@pytest.mark.parametrize(("block_threads", "red_slots"), [(512, 8), (1024, 16)])
def test_rmsnorm_direct_specializes_structural_ir(block_threads, red_slots):
    """A single direct JIT must materialize every structural consequence of
    BLOCK_THREADS independently for each specialization."""
    torch.manual_seed(0)
    x = torch.randn(1, 8192, device="cuda", dtype=torch.bfloat16)
    g = torch.rand(8192, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    rmsnorm_direct(
        x,
        g,
        out,
        1,
        8192,
        "bf16",
        block_threads,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    cache_key = rmsnorm_direct._last_call_cache_key.value
    artifact = rmsnorm_direct._mem_cache[cache_key]
    source_ir = artifact.source_ir
    compiled_ir = artifact.ir

    assert f"known_block_size = array<i32: {block_threads}, 1, 1>" in source_ir
    assert source_ir.count(f"allocBytes = {red_slots * 4} : i64") >= 2
    assert f"!fly.memref<f32, shared, {red_slots}:1" in source_ir
    assert re.search(rf"gpu\.launch_func .* threads in \(%c{block_threads}, %c1", source_ir)
    match = re.search(r"max_flat_workgroup_size\s*=\s*(\d+)", compiled_ir)
    assert match is not None
    assert int(match.group(1)) == block_threads

    err = (out.float() - _reference(x, g)).abs().max().item()
    assert err < 2e-2, f"block={block_threads} max_err={err}"


def test_rmsnorm_direct_preserves_small_n_route():
    """The direct adapter keeps the existing compile-time small-N kernel;
    its analytic geometry intentionally ignores the tuned block candidate."""
    torch.manual_seed(0)
    x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
    g = torch.rand(2048, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    rmsnorm_direct(x, g, out, 8, 2048, "bf16", 1024, torch.cuda.current_stream())
    torch.cuda.synchronize()

    cache_key = rmsnorm_direct._last_call_cache_key.value
    source_ir = rmsnorm_direct._mem_cache[cache_key].source_ir
    assert "gpu.func @rmsnorm_large_m_small_n_kernel" in source_ir
    assert "known_block_size = array<i32: 512, 1, 1>" in source_ir
    assert re.search(r"gpu\.launch_func .* threads in \(%c512, %c1", source_ir)

    err = (out.float() - _reference(x, g)).abs().max().item()
    assert err < 2e-2, f"small-N max_err={err}"


def _run(M, N, autotune_env, tmp_cache, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_cache))
    if autotune_env:
        monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    else:
        monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)

    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda").to(torch.bfloat16)
    g = torch.rand(N, device="cuda").to(torch.bfloat16)
    ref = _reference(x, g)
    s = torch.cuda.current_stream()

    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    rmsnorm_autotuned(x, g, out, M, dtype_str="bf16", stream=s)
    torch.cuda.synchronize()
    err = (out.float() - ref).abs().max().item()
    return err, (x, g, ref, s)


def test_rmsnorm_autotuned_default(tmp_path, monkeypatch):
    """Zero-search default run is correct."""
    err, _ = _run(4096, 8192, autotune_env=False, tmp_cache=tmp_path, monkeypatch=monkeypatch)
    assert err < 2e-2, f"default run max_err={err}"


def test_rmsnorm_autotuned_search_and_cache(tmp_path, monkeypatch):
    """Forced search is correct, and a subsequent normal call does NOT re-tune
    (it reuses the cached best) — proven by counting benchmark invocations."""
    err, (x, g, ref, s) = _run(
        4096,
        8192,
        autotune_env=True,
        tmp_cache=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert err < 2e-2, f"tuned run max_err={err}"

    # A tuned-config JSON must have been persisted.
    files = list(tmp_path.glob("*.json"))
    assert files, "no tuned-config cache file written"

    # Second call with tuning OFF: must serve from cache, no benchmarking.
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    n_bench = {"n": 0}
    orig = rmsnorm_autotuned.tuner._bench_one

    def counting_bench(*a, **k):
        n_bench["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(rmsnorm_autotuned.tuner, "_bench_one", counting_bench)

    out2 = torch.empty_like(ref, dtype=torch.bfloat16)
    rmsnorm_autotuned(x, g, out2, x.shape[0], dtype_str="bf16", stream=s)
    torch.cuda.synchronize()
    err2 = (out2.float() - ref).abs().max().item()
    assert err2 < 2e-2, f"cached run max_err={err2}"
    assert n_bench["n"] == 0, "second call re-tuned instead of using the cache"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
