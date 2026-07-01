#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU integration test for autotuned RMSNorm (first @autotune adopter, #770).

Verifies the two-track builder-mode autotuner end to end:
  - zero-search default run produces correct output
  - forced-search (FLYDSL_AUTOTUNE=1) sweeps configs, picks one, stays correct
  - the tuned result is cached (a second call does not re-tune)
"""

import os

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

from kernels.rmsnorm_autotune import rmsnorm_autotuned  # noqa: E402

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


def _run(M, N, autotune_env, tmp_cache):
    os.environ["FLYDSL_AUTOTUNE_CACHE_DIR"] = str(tmp_cache)
    if autotune_env:
        os.environ["FLYDSL_AUTOTUNE"] = "1"
    else:
        os.environ.pop("FLYDSL_AUTOTUNE", None)

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


def test_rmsnorm_autotuned_default(tmp_path):
    """Zero-search default run is correct."""
    err, _ = _run(4096, 8192, autotune_env=False, tmp_cache=tmp_path)
    assert err < 2e-2, f"default run max_err={err}"


def test_rmsnorm_autotuned_search_and_cache(tmp_path, monkeypatch):
    """Forced search is correct, and a subsequent normal call does NOT re-tune
    (it reuses the cached best) — proven by counting benchmark invocations."""
    err, (x, g, ref, s) = _run(4096, 8192, autotune_env=True, tmp_cache=tmp_path)
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
