# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""The search engine's :class:`~.measure.Backend` for a ``@flyc.jit`` kernel.

One backend tunes one kernel on one problem: the launcher, its runtime
arguments, and an independent numeric oracle are fixed; the search config
supplies everything that changes the compiled binary.

Config keys
-----------
- ``waves_per_eu`` / ``num_warps`` -- the :class:`flydsl.autotune.Config`
  fields of the same name;
- ``llvm:<option>`` -- an LLVM option for this compile (``compile_hints
  ["llvm_options"]``), e.g. ``"llvm:amdgpu-kernarg-preload-count": 8``;
- any other key -- a ``Constexpr`` keyword argument of the launcher.

The three backend calls
-----------------------
- ``compile`` builds the binary without launching (the jit function's
  ``_compile_only``, which FlyDSL 0.5 and develop both provide) and reads its VGPR/SGPR/spill/LDS counts from the code
  object (:mod:`.resources`), so the evaluator rejects a spilling config
  before any GPU time is spent on it;
- ``check`` runs the config once with ``prepare()`` first (poison the outputs
  so a skipped store cannot pass) and then ``oracle()``;
- ``measure_batch`` times a round of configs interleaved (one sample per
  config per round, order shuffled each round) between GPU-side backlogs, so
  host launch gaps are not timed and slow drift hits every config alike.

Times are milliseconds.
"""

from __future__ import annotations

import inspect
import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .measure import CompileOutcome
from .resources import artifact_resources, occupancy_waves_per_simd
from .trial import Fidelity

__all__ = ["FlyBackend", "split_config"]

_BACKLOG_CYCLES = 20_000_000
_LLVM_PREFIX = "llvm:"


def split_config(config: Mapping[str, object]) -> tuple[dict, dict]:
    """``(launcher kwargs, compile hints)`` of a search config (see the
    module docstring for the key rules)."""
    from ..autotune import Config

    llvm = {k[len(_LLVM_PREFIX) :]: v for k, v in config.items() if k.startswith(_LLVM_PREFIX)}
    rest = {k: v for k, v in config.items() if not k.startswith(_LLVM_PREFIX)}
    cfg = Config(**rest)
    hints = dict(cfg.compiler_opts())
    if llvm:
        hints["llvm_options"] = llvm
    return cfg.all_kwargs(), hints


class _Built:
    """A compiled config: the launcher kwargs and compile hints that built it,
    and (after the checked first launch) the fast callable timing uses."""

    __slots__ = ("args", "kwargs", "hints", "run", "run_args")

    def __init__(self, args: tuple, kwargs: dict, hints: dict):
        self.args = args
        self.kwargs = kwargs
        self.hints = hints
        self.run = None
        self.run_args = ()


class FlyBackend:
    """Compile, check and time configs of one ``@flyc.jit`` launcher.

    ``launcher`` is the ``@flyc.jit`` function; ``args`` its runtime
    arguments (device tensors, scalars, stream) and ``kwargs`` the keyword
    arguments that stay fixed across the search. A launcher whose config
    reaches it as a structured argument (a ``param`` dataclass, a derived
    grid) passes ``configure(launcher_kwargs) -> (args, kwargs)`` instead: it
    receives the config's launcher-keyword part and returns the full call. ``oracle() -> (ok, why)``
    compares the launcher's outputs with an independent reference at a
    declared tolerance. ``prepare()`` runs before the checked launch (poison
    outputs; restore in-place inputs). ``before_launch()`` runs before every
    timed launch outside the timed interval (``reset_to_zero`` semantics).
    ``flush_mb`` is the L2/MALL flush used by cold fidelities.
    """

    def __init__(
        self,
        launcher,
        args: Sequence[Any],
        oracle: Callable[[], tuple[bool, str]],
        *,
        kwargs: Mapping[str, object] | None = None,
        configure: Callable[[dict], tuple[Sequence[Any], Mapping[str, object]]] | None = None,
        prepare: Callable[[], None] | None = None,
        before_launch: Callable[[], None] | None = None,
        flush_mb: int = 512,
        inner: int = 10,
        seed: int = 0,
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("FlyBackend times on a GPU; torch with ROCm support and a device are required")
        self.torch = torch
        self.launcher = launcher
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.configure = configure
        self.oracle = oracle
        self.prepare = prepare
        self.before_launch = before_launch
        self.inner = int(inner)
        self._rng = random.Random(seed)
        self._flush = torch.empty(flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        from flydsl.compiler.jit_function import _unwrap_jit_function

        self._jit, _ = _unwrap_jit_function(launcher)
        if self._jit is None:
            raise TypeError(f"FlyBackend launcher must be a @flyc.jit function, got {type(launcher).__name__}")
        fn = launcher.func if hasattr(launcher, "func") else launcher
        self._signature = inspect.signature(fn)

    # ------------------------------------------------------------- compile
    def _call_args(self, args: tuple, kw: Mapping[str, object]) -> tuple:
        bound = self._signature.bind(*args, **kw)
        bound.apply_defaults()
        return tuple(bound.arguments.values())

    def compile(self, config: Mapping[str, object], key: str) -> CompileOutcome:
        from flydsl.compiler.kernel_function import CompilationContext

        t0 = time.perf_counter()
        try:
            cfg_kwargs, hints = split_config(config)
            if self.configure is not None:
                args, kw = self.configure(cfg_kwargs)
                args, kw = tuple(args), dict(kw)
            else:
                args, kw = self.args, {**self.kwargs, **cfg_kwargs}
            with CompilationContext.compile_hints(hints):
                # Lowers and builds the binary; creates no engine and launches
                # nothing. Both FlyDSL 0.5 and develop record the artifact here.
                self._jit._compile_only(*args, **kw)
                artifact = self._jit._last_compiled[1]
            res = artifact_resources(artifact)
        except Exception as e:  # noqa: BLE001 - a compile error is a result, not a crash
            return CompileOutcome(False, detail=f"{type(e).__name__}: {e}"[:500], seconds=time.perf_counter() - t0)
        res.pop("kernels", None)
        res["waves_per_simd"] = occupancy_waves_per_simd(res["vgpr"])
        return CompileOutcome(
            True,
            artifact=_Built(args, kw, hints),
            seconds=time.perf_counter() - t0,
            resources={k: float(v) for k, v in res.items()},
        )

    # --------------------------------------------------------------- check
    def check(self, config: Mapping[str, object], key: str, artifact: "_Built") -> tuple[bool, str]:
        import flydsl.compiler as flyc
        from flydsl.compiler.kernel_function import CompilationContext

        if self.prepare is not None:
            self.prepare()
        # The first launch: ``flyc.compile`` hits the in-memory artifact the
        # compile step built, creates the engine, runs once and returns the
        # fast positional callable that timing uses.
        with CompilationContext.compile_hints(artifact.hints):
            artifact.run = flyc.compile(self.launcher, *artifact.args, **artifact.kwargs)
        artifact.run_args = self._call_args(artifact.args, artifact.kwargs)
        self.torch.cuda.synchronize()
        return self.oracle()

    # ------------------------------------------------------------- measure
    def _one_sample(self, artifact: Any, cold: bool) -> float:
        """One sample (ms per launch): GPU backlog, then events around a
        launch batch (hot) or a single flushed launch (cold)."""
        torch = self.torch
        compiled, run_args = artifact.run, artifact.run_args
        sleep = torch.cuda._sleep
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n = 1 if cold else self.inner
        sleep(_BACKLOG_CYCLES)
        if cold:
            self._flush.zero_()
        if self.before_launch is not None:
            self.before_launch()
        s.record()
        for _ in range(n):
            compiled(*run_args)
        e.record()
        e.synchronize()
        return s.elapsed_time(e) / n

    def measure_batch(self, items: Sequence[tuple], fidelity: Fidelity) -> list[list[float]]:
        """``items`` are ``(config, key, artifact)``; returns one sample list
        per item, ``fidelity.reps`` samples each, rounds interleaved."""
        arts = [a for _c, _k, a in items]
        for _ in range(fidelity.warmup):
            for a in arts:
                self._one_sample(a, fidelity.cold)
        out: list[list[float]] = [[] for _ in arts]
        order = list(range(len(arts)))
        for _ in range(fidelity.reps):
            self._rng.shuffle(order)
            for i in order:
                out[i].append(self._one_sample(arts[i], fidelity.cold))
        return out

    def measure(self, config: Mapping[str, object], key: str, artifact: Any, fidelity: Fidelity) -> Sequence[float]:
        return self.measure_batch([(config, key, artifact)], fidelity)[0]

    # ------------------------------------------------------------ environment
    def recover(self) -> None:
        self.torch.cuda.synchronize()

    def environment(self) -> dict:
        from .measure import _device_environment

        return _device_environment()
