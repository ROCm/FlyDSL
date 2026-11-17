# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Roofline: how close a kernel is to what the device can do, and which limit.

A search needs a stopping rule and an agent needs a verdict ("memory-bound,
81% of achievable bandwidth: stop tuning tiles, look at the epilogue"). Both
come from three numbers: the kernel's flops, its minimum bytes moved, and the
device's peaks.

Two sources of peaks:

- :data:`NOMINAL` -- datasheet numbers for known products. An upper bound no
  kernel reaches; use it to see how far from the silicon limit a kernel is.
- :func:`measure_peaks` -- achievable bandwidth (a large device copy) and
  matrix throughput (a large torch matmul) measured on the actual device in
  this process. Use this as the ceiling that decides when to stop: a kernel
  at 95% of the measured copy bandwidth is done, whatever the datasheet says.

Peaks for architectures not in :data:`NOMINAL` are never guessed: pass a
:class:`Peaks` built from measured numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

__all__ = ["NOMINAL", "Peaks", "Roofline", "nominal_peaks", "measure_peaks", "gemm_roofline"]


@dataclass(frozen=True)
class Peaks:
    """Device ceilings. ``hbm_tbs`` is TB/s; ``tflops`` maps a dtype name
    (``"bf16"``, ``"fp16"``, ``"fp8"``, ``"fp32"``...) to dense matrix TFLOPS."""

    name: str
    hbm_tbs: float
    tflops: Mapping[str, float] = field(default_factory=dict)
    source: str = "nominal"

    def flops_peak(self, dtype: str) -> float:
        try:
            return self.tflops[dtype]
        except KeyError:
            raise KeyError(
                f"Peaks {self.name!r}: no {dtype!r} peak, has {sorted(self.tflops)}; "
                "pass a Peaks with that dtype (measure_peaks or by hand)"
            ) from None


# Vendor datasheet numbers, dense (no sparsity). Keyed by a substring of the
# device product name; the first match wins.
NOMINAL: dict[str, Peaks] = {
    "MI355X": Peaks("MI355X", 8.0, {"fp16": 2516.0, "bf16": 2516.0, "fp8": 5033.0}),
    "MI350X": Peaks("MI350X", 8.0, {"fp16": 2307.0, "bf16": 2307.0, "fp8": 4614.0}),
    "MI325X": Peaks("MI325X", 6.0, {"fp16": 1307.0, "bf16": 1307.0, "fp8": 2615.0}),
    "MI300X": Peaks("MI300X", 5.3, {"fp16": 1307.0, "bf16": 1307.0, "fp8": 2615.0}),
}


def nominal_peaks(device_name: str) -> Peaks | None:
    """Datasheet peaks for a device product name, or ``None`` if unknown."""
    for key, peaks in NOMINAL.items():
        if key.lower() in device_name.lower():
            return peaks
    return None


def measure_peaks(
    device=None, *, dtypes=("bf16", "fp16"), copy_mb: int = 2048, mm_n: int = 8192, iters: int = 20
) -> Peaks:
    """Achievable bandwidth and matrix throughput on this device, measured.

    Bandwidth: the better of a ``copy_mb`` device-to-device copy (read + write
    counted) and a read-only reduction over the same buffer.
    Throughput: a square ``mm_n`` matmul per dtype. Both use the best of
    ``iters`` timed repetitions after warmup. These are practical ceilings for
    a tuned kernel, not datasheet limits. The buffer is far larger than the
    last-level cache, so the bandwidth is HBM-bound: a kernel whose working set
    fits the cache and is timed hot can exceed it (fraction above 1). Time
    memory-bound kernels with ``Fidelity(cold=True)`` before reading the
    fraction.
    """
    import torch

    dev = torch.device("cuda") if device is None else torch.device(device)
    name = torch.cuda.get_device_name(dev)

    def best_ms(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize(dev)
        best = math.inf
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            best = min(best, s.elapsed_time(e))
        return best

    src = torch.empty(copy_mb * 1024 * 1024 // 2, dtype=torch.bfloat16, device=dev)
    dst = torch.empty_like(src)
    ms = best_ms(lambda dst=dst, src=src: dst.copy_(src))
    hbm_tbs = 2 * src.numel() * 2 / (ms * 1e-3) / 1e12
    ms = best_ms(lambda src=src: src.sum())
    hbm_tbs = max(hbm_tbs, src.numel() * 2 / (ms * 1e-3) / 1e12)
    del src, dst

    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tflops: dict[str, float] = {}
    for d in dtypes:
        a = torch.randn(mm_n, mm_n, device=dev, dtype=tdt[d])
        b = torch.randn(mm_n, mm_n, device=dev, dtype=tdt[d])
        ms = best_ms(lambda a=a, b=b: a @ b)
        tflops[d] = 2 * mm_n**3 / (ms * 1e-3) / 1e12
        del a, b
    return Peaks(name, hbm_tbs, tflops, source="measured")


@dataclass(frozen=True)
class Roofline:
    """One problem against one device's peaks. ``flops`` and ``bytes_moved``
    are the minimum the problem needs (inputs read once, outputs written
    once), not what a given kernel moves."""

    flops: float
    bytes_moved: float
    peaks: Peaks
    dtype: str

    @property
    def intensity(self) -> float:
        """Flops per byte."""
        return self.flops / self.bytes_moved

    @property
    def crossover(self) -> float:
        """Intensity above which the problem is compute-bound."""
        return self.peaks.flops_peak(self.dtype) * 1e12 / (self.peaks.hbm_tbs * 1e12)

    @property
    def bound(self) -> str:
        return "compute" if self.intensity > self.crossover else "memory"

    @property
    def floor_us(self) -> float:
        """The fastest time the peaks allow, microseconds."""
        t_mem = self.bytes_moved / (self.peaks.hbm_tbs * 1e12)
        t_cmp = self.flops / (self.peaks.flops_peak(self.dtype) * 1e12)
        return max(t_mem, t_cmp) * 1e6

    def fraction(self, time_us: float) -> float:
        """Fraction of the roofline a kernel running in ``time_us`` reaches
        (1.0 = on the roof)."""
        return self.floor_us / time_us

    def report(self, time_us: float) -> dict:
        return {
            "bound": self.bound,
            "intensity": self.intensity,
            "crossover": self.crossover,
            "floor_us": self.floor_us,
            "time_us": time_us,
            "fraction": self.fraction(time_us),
            "tflops": self.flops / (time_us * 1e-6) / 1e12,
            "tbs": self.bytes_moved / (time_us * 1e-6) / 1e12,
            "peaks": self.peaks.source,
        }


def gemm_roofline(
    m: int, n: int, k: int, peaks: Peaks, dtype: str, *, a_bytes: float, b_bytes: float, c_bytes: float
) -> Roofline:
    """Roofline of ``C[m,n] = A[m,k] B[k,n]`` with the given bytes per element
    of A, B and C (FP8: 1; BF16: 2; FP4: 0.5)."""
    return Roofline(
        flops=2.0 * m * n * k,
        bytes_moved=m * k * a_bytes + k * n * b_bytes + m * n * c_bytes,
        peaks=peaks,
        dtype=dtype,
    )
