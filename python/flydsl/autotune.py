# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL autotuner - benchmark multiple kernel configs, pick the fastest."""

import contextlib
import functools
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Dict, List

from .compile_hints import merge_compile_hint_layers, normalize_occupancy_hint
from .utils import env

try:
    import torch
except ImportError:
    torch = None


class _ConfigCompatibilityError(ValueError, TypeError):
    """A cached config is structurally incompatible with the current tuner."""


def _tuning_enabled() -> bool:
    """Whether to run the full search when a heuristic default exists.

    Off by default so normal runs (tests, serving) pay zero search cost and use
    the analytic default. Opt in with ``FLYDSL_AUTOTUNE=1`` to actually tune.
    Autotuners without a ``default`` always search (there is no fallback).
    """
    return env.autotune.enabled


def _env_fingerprint() -> tuple:
    """Sorted cache-invalidating env vars (reuses the JIT's canonical list)."""
    try:
        from .compiler.jit_function import _cache_invalidating_env_values

        return tuple(sorted(_cache_invalidating_env_values()))
    except Exception:
        return ()


def _toolchain_fingerprint() -> str:
    """Hash of the compiler toolchain, so a codegen change invalidates old
    configs. Reuses jit_function._flydsl_key(); falls back to the version."""
    try:
        from .compiler.jit_function import _flydsl_key

        return _flydsl_key()
    except Exception:
        try:
            import flydsl

            return str(getattr(flydsl, "__version__", ""))
        except Exception:
            return ""


def _device_fingerprint() -> str:
    """GPU arch string (e.g. 'gfx950'), or '' if unavailable."""
    try:
        from .runtime.device import get_rocm_arch

        return str(get_rocm_arch())
    except Exception:
        return ""


def _source_fingerprint(fns) -> str:
    """Short hash of the given callables' *source files*, so editing the kernel /
    config module -- including the module-level helpers and constants they use
    (VEC_WIDTH, _BLOCK_THREADS_CHOICES, ...), not just the function body --
    invalidates a stale cached tuned best. The toolchain fingerprint only covers
    flydsl core, not kernels/. Falls back to the function source, then repr."""
    h = hashlib.sha256()
    seen_files = set()
    for fn in fns:
        if fn is None:
            continue
        target = fn.func if hasattr(fn, "func") else fn
        path = None
        try:
            path = inspect.getsourcefile(target)
        except TypeError:
            pass
        if path and path not in seen_files:
            seen_files.add(path)
            try:
                with open(path, "rb") as f:
                    h.update(f.read())
                continue
            except OSError:
                pass
        try:
            h.update(inspect.getsource(target).encode())
        except (OSError, TypeError):
            h.update(repr(fn).encode())
    return h.hexdigest()[:16]


def _normalize_strides(t) -> tuple:
    """Bucket strides to {0, 1, other}: the layout *pattern* (broadcast /
    contiguous / strided) affects the best config, the exact numbers don't."""
    strides = getattr(t, "stride", None)
    if strides is None:
        return ()
    try:
        vals = strides() if callable(strides) else strides
    except Exception:
        return ()
    out = []
    for s in vals:
        if s == 0:
            out.append(0)
        elif s == 1:
            out.append(1)
        else:
            out.append("s")
    return tuple(out)


def _validate_json_compile_hint(value, path="compile_hints"):
    """Require a JSON round-trip to preserve every compile-hint value's type."""
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} floats must be finite for a stable JSON round-trip")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_compile_hint(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} mappings must have string keys, got {key!r}")
            _validate_json_compile_hint(item, f"{path}[{key!r}]")
        return
    if value_type is tuple:
        raise TypeError(f"{path} contains a tuple, which JSON would change to a list")
    raise TypeError(
        f"{path} must contain only JSON-roundtrip type-stable data "
        f"(dict[str, ...], list, or JSON scalars), got {value_type.__name__}"
    )


class Config:
    """A single tuning configuration.

    ``compile_hints`` is the generic compiler-option envelope. Known hints are
    canonicalized first; the resulting values must be JSON-roundtrip type-stable
    data: dictionaries with string keys, lists, and JSON scalars. Tuples and
    non-string mapping keys are rejected so the persisted autotune cache cannot
    silently change their types. Occupancy aliases remain as typed conveniences
    and override the same key in ``compile_hints`` when explicitly set.

    Occupancy options accept a scalar uniform override or a ``{kernel: value}``
    mapping. ``None`` inherits a lower-priority hint; ``0`` explicitly returns
    to source/compiler defaults after all hint layers have been resolved.
    """

    def __init__(
        self,
        *,
        num_warps=None,
        waves_per_eu=None,
        maxnreg=None,
        compile_hints: Mapping | None = None,
        pre_hook=None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.num_warps = num_warps
        aliases = {}
        if waves_per_eu is not None:
            aliases["waves_per_eu"] = normalize_occupancy_hint(waves_per_eu, "waves_per_eu")
        if maxnreg is not None:
            aliases["maxnreg"] = normalize_occupancy_hint(maxnreg, "maxnreg")
        self.compile_hints = merge_compile_hint_layers(compile_hints, aliases)
        _validate_json_compile_hint(self.compile_hints)
        self.pre_hook = pre_hook

    def _set_occupancy_alias(self, key, value):
        if value is None:
            self.compile_hints.pop(key, None)
        else:
            self.compile_hints = merge_compile_hint_layers(self.compile_hints, {key: value})

    @property
    def waves_per_eu(self):
        return self.compile_hints.get("waves_per_eu")

    @waves_per_eu.setter
    def waves_per_eu(self, value):
        self._set_occupancy_alias("waves_per_eu", value)

    @property
    def maxnreg(self):
        return self.compile_hints.get("maxnreg")

    @maxnreg.setter
    def maxnreg(self, value):
        self._set_occupancy_alias("maxnreg", value)

    def all_kwargs(self):
        """All kwargs to inject into @jit call."""
        d = dict(self.kwargs)
        if self.num_warps is not None:
            d["num_warps"] = self.num_warps
        return d

    def compiler_opts(self):
        """Compiler-level options (not user kwargs)."""
        compile_hints = merge_compile_hint_layers(self.compile_hints)
        _validate_json_compile_hint(compile_hints)
        return compile_hints

    def __repr__(self):
        def format_option(value):
            if isinstance(value, Mapping):
                return "{" + ", ".join(f"{key!r}: {value[key]!r}" for key in sorted(value)) + "}"
            return str(value)

        compile_hints = self.compiler_opts()
        parts = [f"{k}={v}" for k, v in self.kwargs.items()]
        if self.num_warps is not None:
            parts.append(f"num_warps={self.num_warps}")
        if self.waves_per_eu is not None:
            parts.append(f"waves_per_eu={format_option(self.waves_per_eu)}")
        if self.maxnreg is not None:
            parts.append(f"maxnreg={format_option(self.maxnreg)}")
        other_hints = {key: value for key, value in compile_hints.items() if key not in ("waves_per_eu", "maxnreg")}
        if other_hints:
            parts.append(f"compile_hints={format_option(other_hints)}")
        return f"Config({', '.join(parts)})"

    def to_dict(self):
        # Note: pre_hook is intentionally not serialized (it's a callable, not
        # JSON), so a pre_hook that affects correctness won't survive the disk
        # cache — keep pre_hook for timing side-effects only.
        d = dict(self.kwargs)
        compile_hints = self.compiler_opts()
        if self.num_warps is not None:
            d["num_warps"] = self.num_warps
        for k in ("waves_per_eu", "maxnreg"):
            v = compile_hints.get(k)
            if v is not None:
                d[k] = v
        other_hints = {key: value for key, value in compile_hints.items() if key not in ("waves_per_eu", "maxnreg")}
        if other_hints:
            d["compile_hints"] = other_hints
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(
            num_warps=d.pop("num_warps", None),
            waves_per_eu=d.pop("waves_per_eu", None),
            maxnreg=d.pop("maxnreg", None),
            compile_hints=d.pop("compile_hints", None),
            **d,
        )


def do_bench(fn, warmup=5, rep=25, quantiles=None, setup=None):
    """Benchmark a GPU kernel using CUDA/HIP events. Returns median ms. ``setup``,
    if given, runs before each (untimed) warmup and timed iteration -- used to
    restore/reset inputs without charging that copy to the measurement (it is
    enqueued before the start event, so it is not part of the timed span)."""
    for _ in range(warmup):
        if setup:
            setup()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        if setup:
            setup()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    if quantiles:
        return [times[min(int(q * len(times)), len(times) - 1)] for q in quantiles]
    return times[len(times) // 2]


class Autotuner:
    """Wraps a @jit function, benchmarks configs, caches best."""

    def __init__(
        self,
        fn,
        configs,
        key,
        warmup,
        rep,
        prune_configs_by=None,
        reset_to_zero=None,
        restore_value=None,
        pre_hook=None,
        post_hook=None,
        do_bench_fn=None,
        build_fn=None,
        default=None,
        name=None,
        key_fn=None,
        build_only=(),
        arg_names=None,
        positional_arg_names=None,
        structural=None,
        source_fingerprint=None,
    ):
        self.fn = fn  # JitFunction instance (None in builder mode)
        self.configs = configs  # list, or callable(*args, **kwargs) -> [Config]
        self.key = key or []
        self.warmup = warmup
        self.rep = rep
        self.prune_configs_by = prune_configs_by
        self.reset_to_zero = reset_to_zero or []
        self.restore_value = restore_value or []
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self._do_bench = do_bench_fn or do_bench
        self.cache: Dict[tuple, Config] = {}

        # Builder mode: build_fn rebuilds the module per config. `structural`
        # names the config kwargs build_fn consumes and keys the build cache.
        self.build_fn = build_fn
        self.default = default
        self.structural = tuple(structural) if structural is not None else None
        self._build_cache: Dict[tuple, object] = {}

        # Fingerprint of the adopter's build/config source, folded into the cache
        # key so editing the kernel or its config space invalidates a stale tuned
        # "best" (the toolchain fingerprint only covers flydsl core, not kernels/).
        self.source_fingerprint = source_fingerprint

        # key_fn(*args, **kwargs) -> ((name, value), ...): the specialization
        # axes. When set it replaces the self.key name lookup in _make_key, so
        # build-only scalars (dtype_str, causal, ...) enter the key too.
        self.key_fn = key_fn
        # Explicit caller-side names consumed by specialize/build rather than
        # the returned launch function. Unknown kwargs are never inferred as
        # build-only: they reach the launcher and fail normally.
        if isinstance(build_only, str):
            raise TypeError("build_only must be an iterable of parameter names")
        self.build_only = tuple(build_only)
        if any(not isinstance(param, str) for param in self.build_only):
            raise TypeError("build_only must be an iterable of parameter names")

        # Arg names for reset/restore/filter lookup: explicit > jit fn sig >
        # build_fn sig minus leading 'config'.
        derived_positional_arg_names = None
        if arg_names is not None:
            self.arg_names = list(arg_names)
        elif fn is not None:
            src = fn.func if hasattr(fn, "func") else fn
            params = list(inspect.signature(src).parameters.values())
            self.arg_names = [param.name for param in params]
            derived_positional_arg_names = [
                param.name
                for param in params
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        elif build_fn is not None:
            src = build_fn.func if hasattr(build_fn, "func") else build_fn
            params = list(inspect.signature(src).parameters.values())[1:]
            self.arg_names = [param.name for param in params]
            derived_positional_arg_names = [
                param.name
                for param in params
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        else:
            self.arg_names = []
        if positional_arg_names is None:
            self.positional_arg_names = (
                derived_positional_arg_names if derived_positional_arg_names is not None else self.arg_names
            )
        else:
            self.positional_arg_names = list(positional_arg_names)

        # Disk cache. Prefer an explicit name (required for builder mode, where
        # fn is None — otherwise every builder tuner would share unknown.json).
        cache_name = name
        if cache_name is None:
            cache_name = getattr(fn, "__name__", None) or getattr(fn, "func", None)
            if cache_name is not None and not isinstance(cache_name, str):
                cache_name = getattr(cache_name, "__name__", None)
        self.name = cache_name or "unknown"

        self._load_disk_cache()

    @property
    def _cache_file(self) -> Path:
        # Resolved per access so FLYDSL_AUTOTUNE_CACHE_DIR can change between
        # calls (a module-level tuner isn't pinned to the import-time dir).
        cache_dir = Path(env.autotune.cache_dir).expanduser()
        return cache_dir / f"{self.name}.json"

    def _make_key(self, args, kwargs):
        """Cache key over shape/dtype/stride + arch + toolchain + env. A config
        tuned under any of these axes must not be reused under another."""
        sig_args = dict(zip(self.arg_names, args))
        sig_args.update(kwargs)

        key_vals = []
        if self.key_fn is not None:
            # Explicit specialization axes (includes build-only scalars).
            key_vals.append(tuple(self.key_fn(*args, **kwargs)))
        else:
            for k in self.key:
                v = sig_args.get(k)
                if hasattr(v, "shape"):
                    key_vals.append(tuple(v.shape))
                elif hasattr(v, "dtype"):
                    key_vals.append(str(v.dtype))
                else:
                    key_vals.append(v)

        # Tensor dtypes + stride patterns, sorted so kwarg order doesn't change
        # the key (else identical calls would tune twice).
        dtype_parts = []
        stride_parts = []
        for name, val in sig_args.items():
            if hasattr(val, "dtype"):
                dtype_parts.append(f"{name}:{val.dtype}")
            if hasattr(val, "shape") and hasattr(val, "stride"):
                stride_parts.append(f"{name}:{_normalize_strides(val)}")
        key_vals.append(tuple(sorted(dtype_parts)))
        key_vals.append(tuple(sorted(stride_parts)))

        # Environment / toolchain / device specialization, all read live so a
        # mid-process change (arch override, compiler env) can't reuse a config
        # tuned under different conditions. _flydsl_key is lru_cached, so this is
        # cheap. (_toolchain/_device fingerprints are functions, not frozen at
        # construction — otherwise the device axis would go stale.)
        key_vals.append(("_env_", _env_fingerprint()))
        key_vals.append(("_toolchain_", _toolchain_fingerprint()))
        key_vals.append(("_device_", _device_fingerprint()))
        # Adopter source: the toolchain fingerprint only covers flydsl core, not
        # the kernel/config module. Fold in a source hash so editing build/config
        # invalidates a now-stale cached best.
        if self.source_fingerprint:
            key_vals.append(("_src_", self.source_fingerprint))

        return tuple(str(v) for v in key_vals)

    def _reset_tensors(self, args, kwargs):
        """Zero out reset_to_zero tensors before a run (each bench rep and the
        real post-tune / cache-hit call)."""
        if not self.reset_to_zero:
            return
        sig_args = dict(zip(self.arg_names, args))
        sig_args.update(kwargs)
        for name in self.reset_to_zero:
            t = sig_args.get(name)
            if t is not None and hasattr(t, "zero_"):
                t.zero_()

    def _snapshot_tensors(self, args, kwargs):
        """Clone restore_value tensors so each bench rep starts from pristine
        inputs. Without this, an in-place / accumulating kernel would mutate its
        own inputs across reps and the winning config would be chosen on
        corrupted data."""
        if not self.restore_value:
            return {}
        sig_args = dict(zip(self.arg_names, args))
        sig_args.update(kwargs)
        snapshot = {}
        for name in self.restore_value:
            t = sig_args.get(name)
            if t is not None and hasattr(t, "clone"):
                snapshot[name] = (t, t.clone())
        return snapshot

    @staticmethod
    def _restore_tensors(snapshot):
        """Copy each snapshotted tensor back into its original buffer."""
        for _name, (dst, src) in snapshot.items():
            dst.copy_(src)

    def _prune(self, configs, args, kwargs):
        if self.prune_configs_by is not None:
            sig_args = dict(zip(self.arg_names, args))
            sig_args.update(kwargs)
            return self.prune_configs_by(configs, sig_args)
        return configs

    def _reject_unroutable_config(self, config):
        """Raise for a config carrying a knob that can't be honored, so it fails
        loudly instead of being silently dropped or benchmarked as a no-op.

        Builder-mode structural choices must be kwargs consumed by build_fn.
        Compiler options remain valid: they are applied when the returned lazy
        JitFunction compiles. (Called up-front in the search loop, before the
        tolerant per-config except, and on the default/cache-hit path via
        _resolve_fn.)
        """
        if self.build_fn is None:
            return
        if config.num_warps is not None:
            raise _ConfigCompatibilityError(
                f"num_warps={config.num_warps} can't be honored in builder mode "
                "(block size is baked into build_fn). Make it a structural knob "
                "(config kwarg routed to build) or use direct @autotune instead."
            )
        if self.structural is not None:
            extra = [k for k in config.kwargs if k not in self.structural]
            if extra:
                raise _ConfigCompatibilityError(
                    f"config kwargs {extra} are not in structural={self.structural}; in "
                    "builder mode they route nowhere and would be silently dropped. Add "
                    "them to `structural` (routed to build) or remove them from the config."
                )

    def _resolve_fn(self, config, key, args, kwargs, *, compiler_opts=None):
        """Return the launch callable for a config.

        Direct mode: the wrapped jit fn. Builder mode: build the lazy launch
        function (cached) by specialization and the structural knobs consumed by
        build_fn. Compiler-option variants intentionally share this build; the
        JIT cache separates their compiled binaries.
        """
        if self.build_fn is None:
            return self.fn
        if compiler_opts is None:
            compiler_opts = config.compiler_opts()
        self._reject_unroutable_config(config)
        if self.structural is not None:
            knob_key = tuple((k, config.kwargs.get(k)) for k in self.structural)
        else:
            knob_key = repr(config)  # unknown knobs: fall back to full identity
        cache_key = (key, knob_key)
        built = self._build_cache.get(cache_key)
        if built is None:
            built = self.build_fn(config, *args, **kwargs)
            self._build_cache[cache_key] = built
        if compiler_opts:
            from .compiler.jit_function import JitFunction

            if not isinstance(built, JitFunction):
                raise _ConfigCompatibilityError(
                    f"{self.name}: compiler options {sorted(compiler_opts)} require build() "
                    "to return a lazy @flyc.jit JitFunction"
                )
        return built

    def _bench_one(self, config, key, args, kwargs):
        """Compile and benchmark one config. Returns time in ms."""
        compiler_opts = config.compiler_opts()
        fn = self._resolve_fn(config, key, args, kwargs, compiler_opts=compiler_opts)
        merged_kwargs = dict(self._filter_call_kwargs(kwargs))
        # In builder mode the config's structural kwargs (e.g. BLOCK_THREADS)
        # are consumed by build_fn, not passed to the launch call.
        if self.build_fn is None:
            merged_kwargs.update(config.all_kwargs())

        # Snapshot once before any rep runs, so restores are from pristine input.
        snapshot = self._snapshot_tensors(args, merged_kwargs)

        def setup():
            # Runs before each rep but OUTSIDE the timed region: a restore is a
            # full device copy that would swamp a small kernel and make configs
            # indistinguishable if timed. Order: restore/reset first, THEN the
            # pre_hooks, so a hook that sets up state isn't clobbered by the
            # restore. (Matches Triton: pre_hook runs on clean inputs.)
            self._restore_tensors(snapshot)
            self._reset_tensors(args, merged_kwargs)
            if config.pre_hook:
                config.pre_hook(merged_kwargs)
            if self.pre_hook:
                self.pre_hook(merged_kwargs)

        def kernel_call():
            self._run_with_hints(fn, compiler_opts, args, merged_kwargs)
            if self.post_hook:
                self.post_hook(merged_kwargs)

        try:
            return self._call_do_bench(kernel_call, setup)
        finally:
            # Leave the caller's tensors as a single clean run would.
            if snapshot:
                self._restore_tensors(snapshot)

    def _call_do_bench(self, kernel_call, setup):
        """Invoke the benchmarker, passing ``setup`` (untimed per-rep
        restore/reset/pre_hooks) when it supports the param; otherwise fold setup
        into the timed call so a custom do_bench_fn without ``setup`` still runs
        correctly (just times the setup too)."""
        try:
            params = inspect.signature(self._do_bench).parameters
        except (TypeError, ValueError):
            params = {}
        # Only pass `setup` when the benchmarker EXPLICITLY declares it. A
        # `**kwargs` catch-all that doesn't forward setup would silently drop it
        # (restore/reset would never run) -- for those, fold setup into the timed
        # call instead so it always runs (just times it too).
        if "setup" in params:
            return self._do_bench(kernel_call, warmup=self.warmup, rep=self.rep, setup=setup)

        def timed():
            setup()
            return kernel_call()

        return self._do_bench(timed, warmup=self.warmup, rep=self.rep)

    def _run_with_hints(self, fn, compiler_opts, args, kwargs):
        """Run a direct or builder-mode kernel with compiler-option overlays."""
        if compiler_opts:
            from .compiler.kernel_function import CompilationContext

            with CompilationContext.compile_hints(compiler_opts):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    def _filter_call_kwargs(self, kwargs):
        """Remove only explicitly declared build-only kwargs.

        Signature filtering used to discard every kwarg unknown to the launch
        function, which also hid caller typos. All other kwargs are left for the
        launcher to accept or reject.
        """
        if self.build_fn is None or not self.build_only:
            return kwargs
        return {key: value for key, value in kwargs.items() if key not in self.build_only}

    def _prepare_config_run(self, config, key, args, kwargs):
        """Resolve config-owned state before invoking the runtime launcher."""
        compiler_opts = config.compiler_opts()
        fn = self._resolve_fn(config, key, args, kwargs, compiler_opts=compiler_opts)
        merged = dict(self._filter_call_kwargs(kwargs))
        # Builder mode: structural kwargs (e.g. BLOCK_THREADS) go to build_fn,
        # not the launch call.
        if self.build_fn is None:
            merged.update(config.all_kwargs())
        return fn, compiler_opts, merged

    def _run_prepared_config(self, prepared, args):
        fn, compiler_opts, merged = prepared
        self._reset_tensors(args, merged)
        return self._run_with_hints(fn, compiler_opts, args, merged)

    def _run_config(self, config, key, args, kwargs):
        """Run one chosen config as a real (non-benchmark) call."""
        return self._run_prepared_config(self._prepare_config_run(config, key, args, kwargs), args)

    def _reject_positional_build_only(self, args):
        leaked = [param for param in self.positional_arg_names[: len(args)] if param in self.build_only]
        if leaked:
            raise TypeError(
                f"{self.name}: build-only arg(s) {leaked} must be passed by keyword, not positionally "
                f"(they route to build/specialize, not the launch call)"
            )

    def __call__(self, *args, **kwargs):
        self._reject_positional_build_only(args)
        self._load_disk_cache()  # pick up the current cache dir (may be set post-init)
        key = self._make_key(args, kwargs)

        # FLYDSL_AUTOTUNE=1 forces a fresh search: bypass the in-memory/disk
        # cache and the heuristic default so an explicit tune re-benchmarks and
        # re-emits, instead of short-circuiting on a stale cached best.
        force = _tuning_enabled()

        # 1. Cached best config from a prior tune (in-memory or disk).
        if not force and key in self.cache:
            try:
                prepared = self._prepare_config_run(self.cache[key], key, args, kwargs)
            except _ConfigCompatibilityError as e:
                # A stale / incompatible cached entry (for example, a structural
                # knob removed since tuning) must not hard-crash a
                # normal call: drop it (in-memory AND on disk) and fall through to
                # the default / a fresh search. Only errors raised by explicit
                # config-contract validation are caught here; builder, compiler,
                # and launcher failures propagate without invalidating the cache.
                from .utils import log

                log().warning("autotune[%s]: dropping stale cached config: %s", self.name, e)
                self.cache.pop(key, None)
                self._save_disk_cache()
            else:
                # Runtime/launcher failures are not evidence that the cached
                # config is stale. In particular, a misspelled caller kwarg must
                # propagate without deleting a valid tuning result.
                return self._run_prepared_config(prepared, args)

        # 2. Two-track heuristic: unless tuning is explicitly requested, take
        #    the analytic default and skip the search entirely (zero-search
        #    normal run). Mirrors Triton @heuristics / quack get_default.
        if not force and self.default is not None:
            cfg = self.default(*args, **kwargs)
            return self._run_config(cfg, key, args, kwargs)

        # 3. Full search: benchmark every config, pick fastest, cache. configs
        #    may be a callable(*args) -> [Config] to build the space per shape.
        configs = self.configs(*args, **kwargs) if callable(self.configs) else self.configs
        configs = self._prune(configs, args, kwargs)
        print(f"[autotune] tuning {len(configs)} configs...")
        results = []
        last_err = None
        for i, config in enumerate(configs):
            # Reject unroutable configs up front -- before the tolerant except
            # below -- so they fail loudly instead of being logged as "FAILED".
            self._reject_unroutable_config(config)
            try:
                t = self._bench_one(config, key, args, kwargs)
            except Exception as e:
                last_err = e
                print(f"  [{i+1}/{len(configs)}] {config} -> FAILED: {e}")
                continue
            results.append((config, t))
            print(f"  [{i+1}/{len(configs)}] {config} -> {t:.3f} ms")

        if not results:
            raise RuntimeError("All autotune configs failed") from last_err

        best_config, best_time = min(results, key=lambda x: x[1])
        print(f"[autotune] best: {best_config} ({best_time:.3f} ms)")

        self.cache[key] = best_config
        self._save_disk_cache()

        return self._run_config(best_config, key, args, kwargs)

    # --- Disk cache ---
    def _load_disk_cache(self):
        # Re-load when the resolved path changes (FLYDSL_AUTOTUNE_CACHE_DIR may be
        # set after a module-level tuner is constructed), so loads track the same
        # dir that saves write to — not just the import-time default. Clear the
        # in-memory cache too, or entries tuned under the old dir would be served
        # after switching to a new (possibly empty) dir.
        path = self._cache_file
        if getattr(self, "_loaded_cache_path", None) == path:
            return
        if getattr(self, "_loaded_cache_path", None) is not None:
            self.cache.clear()
        self._loaded_cache_path = path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            return  # unreadable / torn file -> start empty rather than crash
        if not isinstance(data, dict):
            return
        for key_str, cfg_dict in data.items():
            # Skip a single malformed entry instead of discarding the whole cache.
            try:
                self.cache[tuple(json.loads(key_str))] = Config.from_dict(cfg_dict)
            except Exception:
                continue

    def _save_disk_cache(self):
        # Best-effort persistence: the cache is an optimization, so a write
        # failure (read-only FS, full disk, permissions) must never crash an
        # otherwise-successful tune -- log and move on.
        path = self._cache_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {json.dumps(list(key)): config.to_dict() for key, config in self.cache.items()}
            # Atomic write (tmp + rename): a concurrent reader never sees a torn
            # or partial file (a bare write_text can be observed mid-write).
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    # Keep the on-disk cache deterministic (stable diffs, no churn).
                    json.dump(data, f, indent=2, sort_keys=True)
                os.replace(tmp, path)
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp)
                raise
        except Exception as e:
            from .utils import log

            log().warning("autotune[%s]: could not persist tuning cache: %s", self.name, e)


def autotune(
    configs: List[Config],
    key: List[str] = None,
    warmup: int = 5,
    rep: int = 25,
    prune_configs_by: Callable = None,
    reset_to_zero: List[str] = None,
    restore_value: List[str] = None,
    pre_hook: Callable = None,
    post_hook: Callable = None,
    do_bench: Callable = None,
    default: Callable = None,
):
    """Autotune decorator for @jit functions.

    Direct mode (structural knobs are jit Constexpr params)::

        @autotune(configs=[Config(BLOCK=128), Config(BLOCK=256)], key=['n'])
        @flyc.jit
        def myKernel(..., BLOCK: fx.Constexpr[int], ...):
            ...

    For kernels whose structural knobs are baked at module-build time, use
    ``autotune_builder`` instead.

    Args:
        default: optional heuristic ``default(*args, **kwargs) -> Config`` used
            without benchmarking unless ``FLYDSL_AUTOTUNE`` forces a search.
        restore_value: tensor args the kernel mutates in place (output overlaps
            input, or accumulation). Snapshotted and restored before each bench
            rep so every config is measured on identical inputs. Required when
            tuning any in-place kernel (e.g. fused-add rmsnorm).
        reset_to_zero: tensor args to zero before each rep (accumulate-into-zero
            kernels).
    """

    def decorator(fn):
        return Autotuner(
            fn,
            configs,
            key,
            warmup,
            rep,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
            restore_value=restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            do_bench_fn=do_bench,
            default=default,
            source_fingerprint=_source_fingerprint([fn, default, Config]),
        )

    return decorator


def autotune_builder(
    *,
    name,
    build,
    specialize,
    configs,
    default=None,
    structural=(),
    build_only=(),
    warmup=10,
    rep=50,
    restore_value=None,
    reset_to_zero=None,
    do_bench_fn=None,
):
    """One-call adoption for kernels whose knobs are baked at module-build time.

    You supply the kernel-specific pieces; this owns the Autotuner mechanics.

        rmsnorm_autotuned = autotune_builder(
            name="rmsnorm",
            build=build_rmsnorm_module,      # build(**spec, **structural) -> launch fn
            specialize=lambda inp, g, out, m, dtype_str="bf16", **kw: {
                "N": inp.shape[-1], "dtype_str": dtype_str},
            configs=get_all_configs,         # (**spec) -> [Config]
            default=get_default,             # (**spec) -> Config
            structural=("BLOCK_THREADS",),   # config kwargs routed into build()
            build_only=("dtype_str",),        # specialize/build only, not launcher
        )

    Args:
        name: artifact / disk-cache identity (required — builder tuners share a
            cache dir, so each needs a distinct name).
        build: build(**spec, **structural_knobs) -> lazy ``@flyc.jit`` launch
            function. A lazy JitFunction is required when configs carry compiler
            options, because those options are applied at JIT compile time.
        specialize: specialize(*args, **kwargs) -> dict of the build/lookup axes
            (shape + build-only scalars). Its items become the cache key, so a
            scalar like dtype_str can't be silently dropped from the key.
        configs / default: called as configs(**spec) / default(**spec).
        structural: config kwarg names passed to build(). Compiler options such
            as ``waves_per_eu`` bypass the builder and are applied when the
            returned JitFunction compiles.
        build_only: caller parameter names consumed by ``specialize``/``build``
            but not forwarded to the returned launch function. These parameters
            must be passed by keyword.
    """
    if not name or not isinstance(name, str):
        raise ValueError("autotune_builder requires a non-empty string name (the cache identity)")
    structural = tuple(structural)
    if isinstance(build_only, str):
        raise TypeError("build_only must be an iterable of parameter names")
    build_only = tuple(build_only)
    if any(not isinstance(param, str) for param in build_only):
        raise TypeError("build_only must be an iterable of parameter names")
    src = specialize.func if hasattr(specialize, "func") else specialize
    sig = inspect.signature(src)
    unknown_build_only = [param for param in build_only if param not in sig.parameters]
    if unknown_build_only:
        raise ValueError(f"{name}: build_only contains parameters absent from specialize(): {unknown_build_only}")
    source_fingerprint = _source_fingerprint([build, configs, default, specialize, Config])
    no_specialization = object()
    specialization = ContextVar(f"flydsl_autotune_{name}_specialization", default=no_specialization)

    def _compute_spec(args, kwargs):
        spec = specialize(*args, **kwargs)
        if not isinstance(spec, Mapping):
            raise TypeError(f"{name}: specialize() must return a mapping, got {type(spec).__name__}")
        return dict(spec)

    def _spec(args, kwargs):
        spec = specialization.get()
        return _compute_spec(args, kwargs) if spec is no_specialization else spec

    def _key_fn(*args, **kwargs):
        return tuple(sorted(_spec(args, kwargs).items()))

    def _build_fn(config, *args, **kwargs):
        spec = _spec(args, kwargs)
        knobs = {k: config.kwargs[k] for k in structural if k in config.kwargs}
        return build(**spec, **knobs)

    def _configs(*args, **kwargs):
        return configs(**_spec(args, kwargs))

    _default = None
    if default is not None:

        def _default(*args, **kwargs):
            return default(**_spec(args, kwargs))

    # specialize's signature names the full call, so restore_value/reset_to_zero
    # can look tensors up by name.
    launch_arg_names = list(sig.parameters.keys())
    positional_arg_names = [
        param.name
        for param in sig.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    tuner = Autotuner(
        fn=None,
        configs=_configs,
        key=None,
        warmup=warmup,
        rep=rep,
        restore_value=restore_value,
        reset_to_zero=reset_to_zero,
        build_fn=_build_fn,
        default=_default,
        name=name,
        key_fn=_key_fn,
        build_only=build_only,
        arg_names=launch_arg_names,
        positional_arg_names=positional_arg_names,
        structural=structural,
        source_fingerprint=source_fingerprint,
        do_bench_fn=do_bench_fn,
    )

    # A build-only scalar must be keyword-only at the public call site; otherwise
    # removing it would shift subsequent runtime launch arguments.
    @functools.wraps(src)
    def call(*args, **kwargs):
        tuner._reject_positional_build_only(args)
        spec = _compute_spec(args, kwargs)
        token = specialization.set(spec)
        try:
            return tuner(*args, **kwargs)
        finally:
            specialization.reset(token)

    call.tuner = tuner  # expose for tests / introspection
    return call
