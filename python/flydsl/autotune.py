# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL autotuner - benchmark multiple kernel configs, pick the fastest."""

import contextlib
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Dict, List

from .compile_hints import merge_compile_hint_layers, normalize_occupancy_hint, stable_hint_key
from .utils import env

try:
    import torch
except ImportError:
    torch = None


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
    """Wrap one JIT function, benchmark configs, and cache the winner."""

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
        default=None,
        source_fingerprint=None,
    ):
        self.fn = fn
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
        self.default = default
        self.source_fingerprint = source_fingerprint

        src = fn.func if hasattr(fn, "func") else fn
        self._signature = inspect.signature(src)
        accepts_extra_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in self._signature.parameters.values()
        )
        unknown_keys = [name for name in self.key if name not in self._signature.parameters]
        if unknown_keys and not accepts_extra_kwargs:
            raise ValueError(f"autotune key contains parameters absent from the JIT function: {unknown_keys}")

        cache_name = getattr(fn, "__name__", None) or getattr(src, "__name__", None)
        self.name = cache_name or "unknown"

        self._load_disk_cache()

    @property
    def _cache_file(self) -> Path:
        # Resolved per access so FLYDSL_AUTOTUNE_CACHE_DIR can change between
        # calls (a module-level tuner isn't pinned to the import-time dir).
        cache_dir = Path(env.autotune.cache_dir).expanduser()
        return cache_dir / f"{self.name}.json"

    def _bind_call(self, args, kwargs):
        """Bind one public call and materialize launcher defaults by name."""
        bound = self._signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        named = {}
        for name, value in bound.arguments.items():
            param = self._signature.parameters[name]
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                named.update(value)
            else:
                named[name] = value
        return named

    def _make_key(self, args, kwargs):
        """Cache key over shape/dtype/stride + arch + toolchain + env. A config
        tuned under any of these axes must not be reused under another."""
        sig_args = self._bind_call(args, kwargs)

        key_vals = []
        for name in self.key:
            if name not in sig_args:
                raise ValueError(f"autotune key parameter {name!r} is not bound and has no default")
            value = sig_args[name]
            if hasattr(value, "shape"):
                key_vals.append(tuple(value.shape))
            elif hasattr(value, "dtype"):
                key_vals.append(str(value.dtype))
            else:
                key_vals.append(value)

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
        effective_hints = getattr(self.fn, "_effective_compile_hints", None)
        if callable(effective_hints):
            key_vals.append(("_compile_hints_", effective_hints()))
        # Adopter source: the toolchain fingerprint only covers flydsl core, not
        # the kernel/config module. Fold in a source hash so editing build/config
        # invalidates a now-stale cached best.
        if self.source_fingerprint:
            key_vals.append(("_src_", self.source_fingerprint))

        return tuple(repr(stable_hint_key(value)) for value in key_vals)

    def _reset_tensors(self, args, kwargs):
        """Zero out reset_to_zero tensors before a run (each bench rep and the
        real post-tune / cache-hit call)."""
        if not self.reset_to_zero:
            return
        sig_args = self._bind_call(args, kwargs)
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
        sig_args = self._bind_call(args, kwargs)
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
            return self.prune_configs_by(configs, self._bind_call(args, kwargs))
        return configs

    def _bench_one(self, config, args, kwargs):
        """Compile and benchmark one config. Returns time in ms."""
        compiler_opts = config.compiler_opts()
        merged_kwargs = dict(kwargs)
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
            self._run_with_hints(compiler_opts, args, merged_kwargs)
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

    def _run_with_hints(self, compiler_opts, args, kwargs):
        """Run the JIT function with one candidate's compiler-hint overlay."""
        if compiler_opts:
            from .compiler.kernel_function import CompilationContext

            with CompilationContext.compile_hints(compiler_opts):
                return self.fn(*args, **kwargs)
        return self.fn(*args, **kwargs)

    def _run_config(self, config, args, kwargs):
        """Run one chosen config as a real (non-benchmark) call."""
        merged = dict(kwargs)
        merged.update(config.all_kwargs())
        self._reset_tensors(args, merged)
        return self._run_with_hints(config.compiler_opts(), args, merged)

    def __call__(self, *args, **kwargs):
        self._load_disk_cache()  # pick up the current cache dir (may be set post-init)
        key = self._make_key(args, kwargs)

        # FLYDSL_AUTOTUNE=1 forces a fresh search: bypass the in-memory/disk
        # cache and the heuristic default so an explicit tune re-benchmarks and
        # re-emits, instead of short-circuiting on a stale cached best.
        force = _tuning_enabled()

        # 1. Cached best config from a prior tune (in-memory or disk).
        if not force and key in self.cache:
            return self._run_config(self.cache[key], args, kwargs)

        # 2. Two-track heuristic: unless tuning is explicitly requested, take
        #    the analytic default and skip the search entirely (zero-search
        #    normal run). Mirrors Triton @heuristics / quack get_default.
        if not force and self.default is not None:
            cfg = self.default(*args, **kwargs)
            return self._run_config(cfg, args, kwargs)

        # 3. Full search: benchmark every config, pick fastest, cache. configs
        #    may be a callable(*args) -> [Config] to build the space per shape.
        configs = self.configs(*args, **kwargs) if callable(self.configs) else self.configs
        configs = self._prune(configs, args, kwargs)
        print(f"[autotune] tuning {len(configs)} configs...")
        results = []
        last_err = None
        for i, config in enumerate(configs):
            try:
                t = self._bench_one(config, args, kwargs)
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

        return self._run_config(best_config, args, kwargs)

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
    configs,
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

    Structural knobs are ordinary JIT ``Constexpr`` parameters::

        @autotune(configs=[Config(BLOCK=128), Config(BLOCK=256)], key=['n'])
        @flyc.jit
        def myKernel(..., BLOCK: fx.Constexpr[int], ...):
            ...

    Args:
        configs: sequence of :class:`Config`, or a callable returning one for
            the current arguments.
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
            source_fingerprint=_source_fingerprint(
                [fn, configs, default, prune_configs_by, pre_hook, post_hook, do_bench, Config]
            ),
        )

    return decorator
