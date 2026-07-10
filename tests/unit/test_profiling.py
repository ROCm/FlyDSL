# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GPU-free tests for the shared profiling helpers."""

import sys
from types import SimpleNamespace

import pytest

import flydsl.profiling as profiling
from flydsl.autotune import do_bench as autotune_do_bench

pytestmark = pytest.mark.l0_backend_agnostic


class FakeEvent:
    def __init__(self, device):
        self.device = device
        self.timestamp = None

    def record(self):
        self.timestamp = self.device.clock

    def elapsed_time(self, end):
        return end.timestamp - self.timestamp


class FakeDeviceInterface:
    def __init__(self):
        self.clock = 0.0
        self.event_count = 0
        self.synchronize_count = 0

    def Event(self, *, enable_timing):
        assert enable_timing
        self.event_count += 1
        return FakeEvent(self)

    def synchronize(self):
        self.synchronize_count += 1


def _install_fake(monkeypatch):
    device = FakeDeviceInterface()
    monkeypatch.setattr(profiling, "_get_torch_cuda", lambda: device)
    return device


def test_do_bench_warmup_repetitions_and_median(monkeypatch):
    device = _install_fake(monkeypatch)
    durations = iter([99.0, 3.0, 1.0, 5.0, 2.0, 4.0])
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        device.clock += next(durations)

    assert profiling.do_bench(fn, warmup=1, rep=5) == 3.0
    assert call_count == 6
    assert device.event_count == 10
    assert device.synchronize_count == 6


def test_do_bench_preserves_upper_middle_for_even_repetitions(monkeypatch):
    device = _install_fake(monkeypatch)
    durations = iter([4.0, 1.0, 3.0, 2.0])

    def fn():
        device.clock += next(durations)

    assert profiling.do_bench(fn, warmup=0, rep=4) == 3.0


def test_do_bench_quantiles(monkeypatch):
    device = _install_fake(monkeypatch)
    durations = iter([3.0, 1.0, 5.0, 2.0, 4.0])

    def fn():
        device.clock += next(durations)

    assert profiling.do_bench(fn, warmup=0, rep=5, quantiles=[0.0, 0.5, 0.9, 1.0]) == [1.0, 3.0, 5.0, 5.0]


def test_do_bench_runs_setup_before_each_iteration_and_outside_timing(monkeypatch):
    device = _install_fake(monkeypatch)
    durations = iter([99.0, 1.0, 2.0])
    order = []

    def setup():
        order.append("setup")
        device.clock += 100.0

    def fn():
        order.append("kernel")
        device.clock += next(durations)

    assert profiling.do_bench(fn, warmup=1, rep=2, setup=setup) == 2.0
    assert order == ["setup", "kernel", "setup", "kernel", "setup", "kernel"]


def test_do_bench_reports_missing_pytorch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="requires PyTorch"):
        profiling._get_torch_cuda()


def test_do_bench_reports_unavailable_device(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="available CUDA or HIP device"):
        profiling._get_torch_cuda()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup": -1, "rep": 1}, "warmup must be non-negative"),
        ({"warmup": 0, "rep": 0}, "rep must be positive"),
        ({"warmup": 0, "rep": 1, "quantiles": [-0.1]}, "quantiles must be between 0 and 1"),
        ({"warmup": 0, "rep": 1, "quantiles": [1.1]}, "quantiles must be between 0 and 1"),
    ],
)
def test_do_bench_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        profiling.do_bench(lambda: None, **kwargs)


def test_autotune_keeps_compatibility_alias():
    assert autotune_do_bench is profiling.do_bench
