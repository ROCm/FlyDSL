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
from kernels.norm.rmsnorm_autotune import _SEARCH_CONFIGS, _rmsnorm_tuner, rmsnorm_autotuned  # noqa: E402
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
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16, generator=generator)
    g = torch.rand(N, device="cuda", dtype=torch.bfloat16, generator=generator)
    return x, g, _reference(x, g)


def _assert_close(out, ref):
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=2e-2)


def test_rmsnorm_direct_specializes_known_block_size():
    x, g, ref = _inputs(M=1)
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream()

    compiled = flyc.compile(rmsnorm_direct, x, g, out, x.shape[0], x.shape[1], "bf16", 512, stream)
    stream.synchronize()
    artifact = compiled._keepalive

    assert "known_block_size = array<i32: 512, 1, 1>" in artifact.source_ir
    match = re.search(r"max_flat_workgroup_size\s*=\s*(\d+)", artifact.ir)
    assert match is not None and int(match.group(1)) == 512
    _assert_close(out, ref)


def test_rmsnorm_autotuned_default_uses_current_stream_and_skips_search(monkeypatch):
    monkeypatch.setattr(_rmsnorm_tuner, "_bench_one", lambda *args, **kwargs: pytest.fail("unexpected search"))
    x, g, ref = _inputs()
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    observed_streams = []
    original_run_config = _rmsnorm_tuner._run_config

    def checked_run_config(config, args, kwargs):
        observed_streams.append(kwargs["stream"].cuda_stream)
        return original_run_config(config, args, kwargs)

    monkeypatch.setattr(_rmsnorm_tuner, "_run_config", checked_run_config)

    with torch.cuda.stream(stream):
        rmsnorm_autotuned(x, g, out, x.shape[0])
    stream.synchronize()

    assert observed_streams == [stream.cuda_stream]
    _assert_close(out, ref)


def test_rmsnorm_autotuned_search_then_cache_hit(monkeypatch):
    completed = 0
    x, g, ref = _inputs(M=8)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    raw_stream = stream.cuda_stream

    def bench_once(call, warmup, rep):
        nonlocal completed
        assert torch.cuda.current_stream().cuda_stream == stream.cuda_stream
        call()
        stream.synchronize()
        completed += 1
        return float(completed)

    monkeypatch.setattr(_rmsnorm_tuner, "_do_bench", bench_once)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    rmsnorm_autotuned(x, g, out, x.shape[0], stream=raw_stream)
    stream.synchronize()

    assert completed == len(_SEARCH_CONFIGS)
    _assert_close(out, ref)

    call_kwargs = {"N": x.shape[1], "dtype_str": "bf16", "stream": raw_stream}
    winner_key = _rmsnorm_tuner._make_key((x, g, out, x.shape[0]), call_kwargs)
    assert winner_key in _rmsnorm_tuner.cache
    other_m_key = _rmsnorm_tuner._make_key((x, g, out, x.shape[0] + 1), call_kwargs)
    assert other_m_key != winner_key

    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setattr(
        _rmsnorm_tuner,
        "default",
        lambda *args, **kwargs: pytest.fail("cached winner should take precedence over default"),
    )
    cached = torch.empty_like(x)
    rmsnorm_autotuned(x, g, cached, x.shape[0], stream=raw_stream)
    stream.synchronize()

    assert completed == len(_SEARCH_CONFIGS)
    _assert_close(cached, ref)
