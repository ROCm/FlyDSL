# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Native FlyDSL tuning cache, with optional bounded warmup calibration."""

import fcntl
import hashlib
import math
import os
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import torch

from flydsl.autotune import Autotuner, Config

from .dispatch import choose_config, tuning_candidates
from .dispatch_features import legal_config, normalize_config

_choices = {}
_lock = threading.RLock()
_KEY_FIELDS = ("m", "n", "k", "mode", "normal", "layout", "output_dtype", "device", "schema", "policy")


@lru_cache(maxsize=1)
def _schema():
    root = Path(__file__).resolve().parent
    files = [
        root / "dispatch.py",
        root / "dispatch_features.py",
        root / "dispatch_model.json",
        root / "tuning.py",
        root / "api.py",
    ]
    files += [root / (name + ".py") for name in ("register", "hybrid", "folding", "layouts")]
    return hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()


def cached_configurations(n, k, max_m, *, mode, normal_only, nibble_layout, dtype, device_id, cache_policy):
    """Read kernel-owned native tuning entries without launching or tuning."""
    name = "rdna4_mxfp4_choice_d" + str(device_id)
    directory = Path(os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", "~/.flydsl/autotune")).expanduser()
    if not (directory / (name + ".json")).is_file():
        return []

    def metadata_only():
        pass

    metadata_only.__name__ = name
    tuner = Autotuner(metadata_only, configs=(), key=(), warmup=0, rep=0)
    expected = dict(
        n=n,
        k=k,
        mode=mode,
        normal=normal_only,
        layout=nibble_layout,
        output_dtype=str(dtype),
        device=device_id,
        schema=_schema(),
        policy=cache_policy,
    )
    rows = []
    for key, value in tuner.cache.items():
        fields = dict(zip(_KEY_FIELDS, key))
        if not all(fields.get(field) == str(expected_value) for field, expected_value in expected.items()):
            continue
        m = int(fields["m"])
        if not 1 <= m <= max_m:
            continue
        config = normalize_config(value.kwargs["config"], m, k)
        if legal_config(config, m, n, k):
            rows.append((m, config))
    return rows


@contextmanager
def _write_lock(path, enabled):
    if not enabled:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


class _Context:
    def __init__(self, gemm, x, weights, scales, out, policy):
        self.gemm, self.x, self.weights, self.scales, self.out, self.policy = (
            gemm,
            x,
            weights,
            scales,
            out,
            policy,
        )
        self.rotation = None
        self.phase = "hot"
        self.flush = None

    def initialize(self):
        if self.rotation is not None:
            return
        m, k = self.x.shape
        n = self.weights.n
        copies = max(2, min(128, math.ceil(256 * 2**20 / self.weights.storage_bytes)))
        if copies * self.weights.storage_bytes < 128 * 2**20 or (m >= 128 and copies >= 8):
            copies = 1
        self.rotation = [self.weights] + [
            replace(
                self.weights,
                packed=self.weights.packed.clone(),
                scales=self.weights.scales.clone(),
                base_exponents=None if self.weights.base_exponents is None else self.weights.base_exponents.clone(),
            )
            for _ in range(copies - 1)
        ]
        self.flush = torch.ones(256 * 2**20, device=self.x.device, dtype=torch.uint8)
        self.rows = torch.linspace(0, m - 1, min(m, 8), device=self.x.device).long().unique()
        self.columns = torch.linspace(0, n - 1, min(n, 64), device=self.x.device).long().unique()
        columns = self.columns
        packed = self.weights.packed[columns // 16, :, :, columns % 16, :].reshape(len(columns), k // 2)
        if self.weights.nibble_layout == "split":
            v = packed.view(len(columns), k // 8, 4)
            packed = torch.stack(
                (
                    (v[..., 0] & 15) | ((v[..., 1] & 15) << 4),
                    (v[..., 2] & 15) | ((v[..., 3] & 15) << 4),
                    (v[..., 0] >> 4) | (v[..., 1] & 240),
                    (v[..., 2] >> 4) | (v[..., 3] & 240),
                ),
                -1,
            ).reshape(len(columns), k // 2)
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
        magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=self.x.device, dtype=torch.float64)
        decoded = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
        scales = self.weights.scales[columns // 16, :, columns % 16]
        factors = torch.exp2(scales.double() - 127).repeat_interleave(32, dim=1)
        decoded = decoded * factors
        decoded[torch.repeat_interleave(scales == 255, 32, dim=1)] = float("nan")
        sampled_x = self.x.view(torch.uint8)[self.rows].view(self.x.dtype).double()
        reference = (sampled_x @ decoded.T) * self.scales.view(-1)[self.rows, None].double()
        self.reference = reference.to(self.out.dtype).double()

    def launch(self, config):
        if self.phase == "rotating":
            for weights in self.rotation:
                self.gemm(self.x, weights, self.scales, out=self.out, config=config)
        else:
            self.gemm(self.x, self.weights, self.scales, out=self.out, config=config)

    @contextmanager
    def validate(self, _arguments):
        self.initialize()
        self.out.fill_(float("nan"))
        yield
        actual = self.out[self.rows[:, None], self.columns[None, :]].double()
        expected = self.reference
        if not torch.equal(torch.isnan(actual), torch.isnan(expected)):
            raise ValueError("calibration candidate produced unexpected NaNs")
        if not torch.equal(torch.isposinf(actual), torch.isposinf(expected)) or not torch.equal(
            torch.isneginf(actual), torch.isneginf(expected)
        ):
            raise ValueError("calibration candidate produced unexpected infinities")
        finite = torch.isfinite(expected)
        if finite.any():
            error = (
                (actual[finite] - expected[finite]).square().sum() / expected[finite].square().sum().clamp_min(1e-300)
            ).sqrt()
            if float(error) > 0.006:
                raise ValueError(f"calibration candidate relative RMS {float(error):.6f} exceeds 0.006")

    def measure(self, graph, calls, events, *, cold=False):
        # Captured events exclude CPU enqueue delay as well as the cache
        # flush. Synchronizing the stream ensures this replay's timestamps
        # are available even when an event was recorded by a prior replay.
        settle_until = time.monotonic() + 0.003
        while time.monotonic() < settle_until:
            graph.replay()
            torch.cuda.current_stream().synchronize()
        begin, end = events
        samples = []
        for _ in range(16 if cold else 8):
            graph.replay()
            torch.cuda.current_stream().synchronize()
            samples.append(begin.elapsed_time(end) / calls)
        return statistics.median(samples)

    def benchmark(self, fn, warmup=0, rep=0):
        values = {}
        phases = ("hot", "rotating") if self.policy == "auto" else (self.policy,)
        for phase in phases:
            self.phase = phase
            graph = torch.cuda.CUDAGraph()
            repeats = (8 if self.x.shape[0] < 128 else 2) if phase == "hot" else 1
            events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
            with torch.cuda.graph(graph):
                if phase == "rotating":
                    self.flush.sum(dtype=torch.int32)
                    self.x.view(torch.uint8).sum(dtype=torch.int32)
                    self.scales.sum()
                events[0].record()
                for _ in range(repeats):
                    fn()
                events[1].record()
            calls = repeats * (len(self.rotation) if phase == "rotating" else 1)
            elapsed = self.measure(graph, calls, events, cold=phase == "rotating")
            values[phase] = elapsed
        self.phase = "hot"
        return math.sqrt(values["hot"] * values["rotating"]) if self.policy == "auto" else values[self.policy]


def resolve_config(gemm, x, weights, scales, out, cache_policy, *, calibrate=False):
    """Resolve a measured native cache entry or the CPU heuristic.

    Explicit calibration benchmarks at most 64 legal configurations. Ordinary
    calls use cached measurements when available and otherwise select without
    GPU timing, unless FLYDSL_AUTOTUNE explicitly requests calibration.
    """
    m, k = x.shape
    n = weights.n
    scales = scales.view(m)
    choose_config(m, n, k, weights.mode, cache_policy=cache_policy)
    force = os.environ.get("FLYDSL_AUTOTUNE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    key = (
        x.device.index,
        m,
        n,
        k,
        weights.mode,
        weights.normal_only,
        weights.nibble_layout,
        str(out.dtype),
        cache_policy,
        os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", "~/.flydsl/autotune"),
    )
    previous = _choices.get(key)
    if previous is not None and (previous[1] or not (calibrate or force)):
        return dict(previous[0])
    if weights.mode == "grouped":
        return choose_config(m, n, k, weights.mode, cache_policy=cache_policy)
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            if calibrate or force:
                raise RuntimeError("calibrate MXFP4 before graph capture")
            return choose_config(m, n, k, weights.mode, cache_policy=cache_policy)
        with _lock:
            context = _Context(gemm, x, weights, scales, out, cache_policy)

            def launch(
                x,
                scales,
                out,
                context,
                m,
                n,
                k,
                mode,
                normal,
                layout,
                output_dtype,
                device,
                schema,
                policy,
                *,
                config,
            ):
                context.launch(config)

            launch.__name__ = "rdna4_mxfp4_choice_d" + str(x.device.index)
            lock_dir = Path(os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", "~/.flydsl/autotune")).expanduser()
            with _write_lock(lock_dir / (launch.__name__ + ".lock"), calibrate or force):
                tuner = Autotuner(
                    launch,
                    configs=lambda *args, **kwargs: [
                        Config(config=c) for c in tuning_candidates(m, n, k, cache_policy=cache_policy)
                    ],
                    key=_KEY_FIELDS,
                    warmup=0,
                    rep=5,
                    do_bench_fn=context.benchmark,
                    validate_hook=context.validate,
                    default=(
                        None
                        if calibrate
                        else lambda *args, **kwargs: Config(
                            config=choose_config(m, n, k, weights.mode, cache_policy=cache_policy)
                        )
                    ),
                )
                selected = tuner.resolve_config(
                    x,
                    scales,
                    out,
                    context,
                    m,
                    n,
                    k,
                    weights.mode,
                    weights.normal_only,
                    weights.nibble_layout,
                    str(out.dtype),
                    x.device.index,
                    _schema(),
                    cache_policy,
                )
                config = normalize_config(selected.kwargs["config"], m, k)
                measured = any(selected is entry for entry in tuner.cache.values())
                _choices[key] = (dict(config), measured)
                return dict(config)
