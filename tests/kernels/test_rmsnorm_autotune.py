# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU contracts for the direct RMSNorm autotune adopter."""

import re

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
from kernels.norm.rmsnorm_autotune import _rmsnorm_tuner, rmsnorm_autotuned  # noqa: E402
from kernels.norm.rmsnorm_kernel import rmsnorm_direct  # noqa: E402

EPS = 1e-5


@pytest.fixture(autouse=True)
def _isolated_tuner(tmp_path, monkeypatch):
    _rmsnorm_tuner.cache.clear()
    monkeypatch.setattr(_rmsnorm_tuner, "_cache_file", tmp_path / "rmsnorm.json")
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    yield
    _rmsnorm_tuner.cache.clear()


def _reference(x, g):
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + EPS) * g.float()


def _inputs(M=32, N=8192):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    g = torch.rand(N, device="cuda", dtype=torch.bfloat16)
    return x, g, _reference(x, g), torch.cuda.current_stream()


def _assert_close(out, ref):
    assert (out.float() - ref).abs().max().item() < 2e-2


def test_rmsnorm_direct_specializes_known_block_size():
    x, g, ref, stream = _inputs(M=1)
    out = torch.empty_like(x)

    compiled = flyc.compile(rmsnorm_direct, x, g, out, 1, 8192, "bf16", 512, stream)
    torch.cuda.synchronize()
    artifact = compiled._keepalive

    assert "known_block_size = array<i32: 512, 1, 1>" in artifact.source_ir
    match = re.search(r"max_flat_workgroup_size\s*=\s*(\d+)", artifact.ir)
    assert match is not None and int(match.group(1)) == 512
    _assert_close(out, ref)


def test_rmsnorm_autotuned_default_skips_search(monkeypatch):
    monkeypatch.setattr(_rmsnorm_tuner, "_bench_one", lambda *args, **kwargs: pytest.fail("unexpected search"))
    x, g, ref, stream = _inputs()
    out = torch.empty_like(x)

    rmsnorm_autotuned(x, g, out, x.shape[0], stream=stream)
    torch.cuda.synchronize()
    _assert_close(out, ref)


def test_rmsnorm_autotuned_search_then_cache_hit(monkeypatch):
    calls = 0
    original = _rmsnorm_tuner._bench_one

    def counting_bench(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_rmsnorm_tuner, "_bench_one", counting_bench)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    x, g, ref, stream = _inputs()
    out = torch.empty_like(x)
    rmsnorm_autotuned(x, g, out, x.shape[0], stream=stream)
    torch.cuda.synchronize()
    _assert_close(out, ref)
    assert calls > 1

    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    searched = calls
    cached = torch.empty_like(x)
    rmsnorm_autotuned(x, g, cached, x.shape[0], stream=stream)
    torch.cuda.synchronize()
    _assert_close(cached, ref)
    assert calls == searched
