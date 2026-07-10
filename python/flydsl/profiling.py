# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GPU event timing shared across FlyDSL compile backends."""

from typing import Callable, List, Optional, Sequence, Union

__all__ = ["do_bench"]


def _get_torch_cuda():
    """Return PyTorch's CUDA-compatible namespace lazily.

    PyTorch exposes the same ``torch.cuda`` event API on CUDA and HIP builds,
    so callers do not need to branch on the FlyDSL compile backend.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU profiling requires PyTorch with CUDA or HIP support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires an available CUDA or HIP device")
    return torch.cuda


def do_bench(
    fn: Callable[[], object],
    warmup: int = 5,
    rep: int = 25,
    quantiles: Optional[Sequence[float]] = None,
    setup: Optional[Callable[[], object]] = None,
) -> Union[float, List[float]]:
    """Benchmark a GPU callable with CUDA/HIP events.

    ``warmup`` and ``rep`` are iteration counts. Timings are returned in
    milliseconds. By default the upper-middle sample is returned; when a
    non-empty ``quantiles`` sequence is provided, the corresponding sorted
    samples are returned. ``warmup`` must be non-negative, ``rep`` must be
    positive, and quantiles must be in the inclusive range ``[0, 1]``.

    ``fn`` must enqueue the measured work on PyTorch's current CUDA/HIP stream.
    Work submitted only to another stream is outside the event interval unless
    ``fn`` synchronizes that stream itself.

    When provided, ``setup`` runs before every warmup and measured iteration.
    For measured iterations it runs before the start event, so restore/reset
    work is not included in the reported kernel latency.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if rep <= 0:
        raise ValueError("rep must be positive")
    if quantiles is not None and any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError("quantiles must be between 0 and 1")

    device = _get_torch_cuda()

    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    device.synchronize()

    times = []
    for _ in range(rep):
        if setup is not None:
            setup()
        start = device.Event(enable_timing=True)
        end = device.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        device.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    if quantiles:
        return [times[min(int(q * len(times)), len(times) - 1)] for q in quantiles]
    return times[len(times) // 2]
