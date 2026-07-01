# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL autotuner - benchmark multiple kernel configs, pick the fastest."""

import functools
import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List

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
    return os.environ.get("FLYDSL_AUTOTUNE", "0") not in ("0", "", "false", "False")


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


class Config:
    """A single tuning configuration."""

    def __init__(self, *, num_warps=None, waves_per_eu=None, maxnreg=None, pre_hook=None, **kwargs):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.waves_per_eu = waves_per_eu
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

    def all_kwargs(self):
        """All kwargs to inject into @jit call."""
        d = dict(self.kwargs)
        if self.num_warps is not None:
            d["num_warps"] = self.num_warps
        return d

    def compiler_opts(self):
        """Compiler-level options (not user kwargs)."""
        return {
            k: v
            for k, v in [
                ("waves_per_eu", self.waves_per_eu),
                ("maxnreg", self.maxnreg),
            ]
            if v is not None
        }

    def __repr__(self):
        parts = [f"{k}={v}" for k, v in self.kwargs.items()]
        if self.num_warps is not None:
            parts.append(f"num_warps={self.num_warps}")
        if self.waves_per_eu is not None:
            parts.append(f"waves_per_eu={self.waves_per_eu}")
        if self.maxnreg is not None:
            parts.append(f"maxnreg={self.maxnreg}")
        return f"Config({', '.join(parts)})"

    def to_dict(self):
        # Note: pre_hook is intentionally not serialized (it's a callable, not
        # JSON), so a pre_hook that affects correctness won't survive the disk
        # cache — keep pre_hook for timing side-effects only.
        d = dict(self.kwargs)
        for k in ("num_warps", "waves_per_eu", "maxnreg"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(
            num_warps=d.pop("num_warps", None),
            waves_per_eu=d.pop("waves_per_eu", None),
            maxnreg=d.pop("maxnreg", None),
            **d,
        )


def do_bench(fn, warmup=5, rep=25, quantiles=None):
    """Benchmark a GPU kernel using CUDA/HIP events. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
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
        arg_names=None,
        structural=None,
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
        # names the config kwargs build_fn consumes; the build cache keys only on
        # those, so hint-only variants (waves_per_eu) reuse one built module.
        self.build_fn = build_fn
        self.default = default
        self.structural = tuple(structural) if structural is not None else None
        self._build_cache: Dict[tuple, object] = {}

        # key_fn(*args, **kwargs) -> ((name, value), ...): the specialization
        # axes. When set it replaces the self.key name lookup in _make_key, so
        # build-only scalars (dtype_str, causal, ...) enter the key too.
        self.key_fn = key_fn

        # Arg names for reset/restore/filter lookup: explicit > jit fn sig >
        # build_fn sig minus leading 'config'.
        if arg_names is not None:
            self.arg_names = list(arg_names)
        elif fn is not None:
            src = fn.func if hasattr(fn, "func") else fn
            self.arg_names = list(inspect.signature(src).parameters.keys())
        elif build_fn is not None:
            src = build_fn.func if hasattr(build_fn, "func") else build_fn
            self.arg_names = list(inspect.signature(src).parameters.keys())[1:]
        else:
            self.arg_names = []

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
        cache_dir = Path(os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", os.path.expanduser("~/.flydsl/autotune")))
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

    def _resolve_fn(self, config, key, args, kwargs):
        """Return the launch callable for a config.

        Direct mode: the wrapped jit fn. Builder mode: build the module (cached),
        keyed on the spec + the structural knobs build_fn consumes — NOT the
        whole config, so configs differing only in compiler hints (waves_per_eu)
        share one build instead of recompiling per hint.
        """
        if self.build_fn is None:
            return self.fn
        if self.structural is not None:
            knob_key = tuple((k, config.kwargs.get(k)) for k in self.structural)
        else:
            knob_key = repr(config)  # unknown knobs: fall back to full identity
        cache_key = (key, knob_key)
        built = self._build_cache.get(cache_key)
        if built is None:
            built = self.build_fn(config, *args, **kwargs)
            self._build_cache[cache_key] = built
        return built

    def _bench_one(self, config, key, args, kwargs):
        """Compile and benchmark one config. Returns time in ms."""
        compiler_opts = config.compiler_opts()
        fn = self._resolve_fn(config, key, args, kwargs)
        merged_kwargs = dict(self._filter_call_kwargs(fn, kwargs))
        # In builder mode the config's structural kwargs (e.g. BLOCK_THREADS)
        # are consumed by build_fn, not passed to the launch call.
        if self.build_fn is None:
            merged_kwargs.update(config.all_kwargs())

        # Snapshot once before any rep runs, so restores are from pristine input.
        snapshot = self._snapshot_tensors(args, merged_kwargs)

        def kernel_call():
            # Order: restore/reset the inputs first, THEN run the pre_hooks, so a
            # hook that sets up state (incl. mutating a tensor) isn't clobbered
            # by the restore. (Matches Triton: pre_hook runs on clean inputs.)
            self._restore_tensors(snapshot)
            self._reset_tensors(args, merged_kwargs)
            if config.pre_hook:
                config.pre_hook(merged_kwargs)
            if self.pre_hook:
                self.pre_hook(merged_kwargs)
            self._run_with_hints(fn, compiler_opts, args, merged_kwargs)
            if self.post_hook:
                self.post_hook(merged_kwargs)

        try:
            return self._do_bench(kernel_call, warmup=self.warmup, rep=self.rep)
        finally:
            # Leave the caller's tensors as a single clean run would.
            if snapshot:
                self._restore_tensors(snapshot)

    def _run_with_hints(self, fn, compiler_opts, args, kwargs):
        """Run fn with optional compiler hints. Hints are set on fn.compile_hints
        (which enters the JIT cache key) and restored after, so each distinct
        waves_per_eu / maxnreg compiles a distinct binary instead of reusing a
        cached one. Import is deferred so the core stays importable unbuilt."""
        if not compiler_opts:
            return fn(*args, **kwargs)

        from .compiler.kernel_function import CompilationContext

        prev_hints = getattr(fn, "compile_hints", None)
        if prev_hints is not None:
            # JitFunction: fold hints into its cache key so each distinct
            # (waves_per_eu, maxnreg) compiles and caches a distinct binary.
            fn.compile_hints = {**prev_hints, **compiler_opts}
        try:
            with CompilationContext.compile_hints(compiler_opts):
                return fn(*args, **kwargs)
        finally:
            if prev_hints is not None:
                fn.compile_hints = prev_hints

    def _filter_call_kwargs(self, fn, kwargs):
        """Drop kwargs the launch fn doesn't accept. In builder mode the caller
        may pass build-only kwargs (e.g. dtype_str) that route to build_fn but
        aren't launch params; the built jit fn binds strictly."""
        if self.build_fn is None:
            return kwargs
        src = fn.func if hasattr(fn, "func") else fn
        try:
            params = inspect.signature(src).parameters
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    def _run_config(self, config, key, args, kwargs):
        """Run the chosen config as a real (non-benchmark) call. Resolves the
        launch fn (builder mode rebuilds per config), applies config kwargs +
        hints, and re-applies reset_to_zero so cache hits and the post-tune run
        behave like a single clean run (restore_value already handled)."""
        fn = self._resolve_fn(config, key, args, kwargs)
        merged = dict(self._filter_call_kwargs(fn, kwargs))
        # Builder mode: structural kwargs (e.g. BLOCK_THREADS) go to build_fn,
        # not the launch call.
        if self.build_fn is None:
            merged.update(config.all_kwargs())
        self._reset_tensors(args, merged)
        return self._run_with_hints(fn, config.compiler_opts(), args, merged)

    def __call__(self, *args, **kwargs):
        self._load_disk_cache()  # pick up the current cache dir (may be set post-init)
        key = self._make_key(args, kwargs)

        # FLYDSL_AUTOTUNE=1 forces a fresh search: bypass the in-memory/disk
        # cache and the heuristic default so an explicit tune re-benchmarks and
        # re-emits, instead of short-circuiting on a stale cached best.
        force = _tuning_enabled()

        # 1. Cached best config from a prior tune (in-memory or disk).
        if not force and key in self.cache:
            return self._run_config(self.cache[key], key, args, kwargs)

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
        for i, config in enumerate(configs):
            try:
                t = self._bench_one(config, key, args, kwargs)
                results.append((config, t))
                print(f"  [{i+1}/{len(configs)}] {config} -> {t:.3f} ms")
            except Exception as e:
                print(f"  [{i+1}/{len(configs)}] {config} -> FAILED: {e}")

        if not results:
            raise RuntimeError("All autotune configs failed")

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
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for key_str, cfg_dict in data.items():
                    key = tuple(json.loads(key_str))
                    self.cache[key] = Config.from_dict(cfg_dict)
            except Exception:
                pass

    def _save_disk_cache(self):
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, config in self.cache.items():
            data[json.dumps(list(key))] = config.to_dict()
        self._cache_file.write_text(json.dumps(data, indent=2))


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
    build_fn: Callable = None,
    default: Callable = None,
):
    """Autotune decorator for @jit functions.

    Direct mode (structural knobs are jit Constexpr params)::

        @autotune(configs=[Config(BLOCK=128), Config(BLOCK=256)], key=['n'])
        @flyc.jit
        def myKernel(..., BLOCK: fx.Constexpr[int], ...):
            ...

    For kernels whose structural knobs are baked at module-build time (as every
    current FlyDSL kernel is), prefer ``autotune_builder`` — it wires build_fn /
    default / configs / structural for you. The low-level ``build_fn`` / ``default``
    args below are the primitives it builds on.

    Args:
        build_fn: build_fn(config, *args, **kwargs) -> launch_callable. Enables
            builder mode; built modules are cached per (key, config).
        default: two-track heuristic default(*args, **kwargs) -> Config. When
            set, normal runs use it and skip the search (zero-search); set
            FLYDSL_AUTOTUNE=1 to force the full search. Without a default, every
            uncached run searches.
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
            build_fn=build_fn,
            default=default,
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
        )

    Args:
        name: artifact / disk-cache identity (required — builder tuners share a
            cache dir, so each needs a distinct name).
        build: build(**spec, **structural_knobs) -> launch callable.
        specialize: specialize(*args, **kwargs) -> dict of the build/lookup axes
            (shape + build-only scalars). Its items become the cache key, so a
            scalar like dtype_str can't be silently dropped from the key.
        configs / default: called as configs(**spec) / default(**spec).
        structural: config kwarg names passed to build() (vs compiler hints,
            which flow through Config.compiler_opts()).
    """
    if not name or not isinstance(name, str):
        raise ValueError("autotune_builder requires a non-empty string name (the cache identity)")
    structural = tuple(structural)

    def _spec(args, kwargs):
        return specialize(*args, **kwargs)

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
    src = specialize.func if hasattr(specialize, "func") else specialize
    sig = inspect.signature(src)
    launch_arg_names = list(sig.parameters.keys())

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
        arg_names=launch_arg_names,
        structural=structural,
        do_bench_fn=do_bench_fn,
    )

    # A build-only scalar (a param that is also a spec key, e.g. dtype_str) must
    # be keyword — positionally it would shift the launch args to the wrong slot.
    @functools.wraps(src)
    def call(*args, **kwargs):
        leaked = [n for n in list(sig.parameters)[: len(args)] if n in _spec(args, kwargs)]
        if leaked:
            raise TypeError(
                f"{name}: build-only arg(s) {leaked} must be passed by keyword, not positionally "
                f"(they route to build/specialize, not the launch call)"
            )
        return tuner(*args, **kwargs)

    call.tuner = tuner  # expose for tests / introspection
    return call
