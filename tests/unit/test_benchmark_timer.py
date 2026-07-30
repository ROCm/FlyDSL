# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU-free contract tests for the canonical CUDA/HIP event timer."""

import importlib
from contextlib import nullcontext

import pytest

from flydsl import BenchResult, do_bench

autotune = importlib.import_module("flydsl.autotune")


class _FakeEvent:
    def __init__(self, cuda):
        self._cuda = cuda
        self._timestamp = None
        self.recorded_stream = None

    def record(self, stream=None):
        self._timestamp = self._cuda.clock_ms
        self.recorded_stream = stream

    def synchronize(self):
        self._cuda.event_synchronize_count += 1

    def elapsed_time(self, other):
        return other._timestamp - self._timestamp


class _FakeBuffer:
    def __init__(self, device):
        self.device = device
        self.zero_count = 0

    def zero_(self):
        self.zero_count += 1


class _FakeCuda:
    def __init__(self):
        self.clock_ms = 0.0
        self.global_synchronize_count = 0
        self.event_synchronize_count = 0
        self.events = []

    def Event(self, enable_timing):
        assert enable_timing
        event = _FakeEvent(self)
        self.events.append(event)
        return event

    def synchronize(self):
        self.global_synchronize_count += 1

    def stream(self, stream):
        return nullcontext(stream)


class _FakeTorch:
    uint8 = object()

    def __init__(self):
        self.cuda = _FakeCuda()
        self.buffers = []

    def empty(self, size, *, dtype, device):
        assert size > 0
        assert dtype is self.uint8
        buffer = _FakeBuffer(device)
        self.buffers.append(buffer)
        return buffer


@pytest.fixture
def fake_torch(monkeypatch):
    fake = _FakeTorch()
    monkeypatch.setattr(autotune, "torch", fake)
    return fake


def test_legacy_do_bench_keeps_scalar_milliseconds(fake_torch):
    def fn():
        fake_torch.cuda.clock_ms += 0.002

    value = do_bench(fn, warmup=1, rep=3)
    quantiles = do_bench(fn, warmup=0, rep=3, quantiles=[0.5])

    assert value == pytest.approx(0.002)
    assert quantiles == pytest.approx([0.002])
    assert fake_torch.cuda.global_synchronize_count == 8
    assert fake_torch.cuda.event_synchronize_count == 0


def test_structured_result_uses_microseconds(fake_torch):
    def fn():
        fake_torch.cuda.clock_ms += 0.002

    result = do_bench(fn, warmup=1, rep=3, return_result=True)

    assert isinstance(result, BenchResult)
    assert result.value_us == pytest.approx(2.0)
    assert result.samples_us == pytest.approx((2.0, 2.0, 2.0))
    assert result.schedule == "isolated"
    assert result.statistic == "median"
    assert result.sample_count == 3
    assert result.cache_policy == "warm"


def test_pipelined_schedule_returns_average_per_call(fake_torch):
    def fn():
        fake_torch.cuda.clock_ms += 0.003

    result = do_bench(
        fn,
        warmup=0,
        rep=4,
        schedule="pipelined",
        statistic="mean",
        return_result=True,
    )

    assert result.value_us == pytest.approx(3.0)
    assert result.samples_us == pytest.approx((3.0,))


def test_per_iter_schedule_preserves_prep_flush_and_stream(fake_torch):
    prep_count = 0
    stream = object()

    def prep():
        nonlocal prep_count
        prep_count += 1

    def fn():
        fake_torch.cuda.clock_ms += 0.004

    result = do_bench(
        fn,
        warmup=2,
        rep=3,
        schedule="per_iter",
        prep_fn=prep,
        flush_bytes=4096,
        iqr=True,
        stream=stream,
        return_result=True,
    )

    assert result.value_us == pytest.approx(4.0)
    assert prep_count == 5
    assert fake_torch.buffers[0].zero_count == 5
    assert result.cache_policy == "flush:4096"
    assert all(event.recorded_stream is stream for event in fake_torch.cuda.events)


def test_pipelined_schedule_rejects_non_mean_statistic(fake_torch):
    with pytest.raises(ValueError, match="only supports"):
        do_bench(lambda: None, schedule="pipelined", statistic="median")
    with pytest.raises(ValueError, match="cannot include"):
        do_bench(
            lambda: None,
            schedule="pipelined",
            statistic="mean",
            prep_fn=lambda: None,
        )


def test_median_average_preserves_statistics_median_semantics(fake_torch):
    durations = iter([0.001, 0.003])

    def fn():
        fake_torch.cuda.clock_ms += next(durations)

    result = do_bench(
        fn,
        warmup=0,
        rep=2,
        schedule="per_iter",
        statistic="median_average",
        return_result=True,
    )

    assert result.value_us == pytest.approx(2.0)


def test_flush_buffer_uses_explicit_stream_device(fake_torch):
    class Stream:
        device = "cuda:7"

    do_bench(
        lambda: None,
        warmup=0,
        rep=1,
        schedule="per_iter",
        flush_bytes=1024,
        stream=Stream(),
    )

    assert fake_torch.buffers[0].device == "cuda:7"
