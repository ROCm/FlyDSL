# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL autotuner - benchmark multiple kernel configs, pick the fastest."""

import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List

try:
    import torch
except ImportError:
    torch = None


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
    ):
        self.fn = fn  # JitFunction instance
        self.configs = configs
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

        # Infer arg names from the underlying function
        if hasattr(fn, "func"):
            self.arg_names = list(inspect.signature(fn.func).parameters.keys())
        else:
            self.arg_names = list(inspect.signature(fn).parameters.keys())

        # Disk cache
        fn_name = getattr(fn, "__name__", None) or getattr(fn, "func", None)
        if fn_name is not None and not isinstance(fn_name, str):
            fn_name = getattr(fn_name, "__name__", "unknown")
        fn_name = fn_name or "unknown"
        cache_dir = Path(os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", os.path.expanduser("~/.flydsl/autotune")))
        self._cache_file = cache_dir / f"{fn_name}.json"

        self._load_disk_cache()

    def _make_key(self, args, kwargs):
        """Cache key over shape/dtype/stride + arch + toolchain + env. A config
        tuned under any of these axes must not be reused under another."""
        sig_args = dict(zip(self.arg_names, args))
        sig_args.update(kwargs)

        key_vals = []
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

    def _bench_one(self, config, args, kwargs):
        """Compile and benchmark one config. Returns time in ms."""
        merged_kwargs = dict(kwargs)
        merged_kwargs.update(config.all_kwargs())
        compiler_opts = config.compiler_opts()

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
            self._run_with_hints(compiler_opts, args, merged_kwargs)
            if self.post_hook:
                self.post_hook(merged_kwargs)

        try:
            return self._do_bench(kernel_call, warmup=self.warmup, rep=self.rep)
        finally:
            # Leave the caller's tensors as a single clean run would.
            if snapshot:
                self._restore_tensors(snapshot)

    def _run_with_hints(self, compiler_opts, args, kwargs):
        """Run the kernel with optional compiler hints. Import is deferred so
        the core stays importable without the compiled bindings when unused."""
        if compiler_opts:
            from .compiler.kernel_function import CompilationContext

            with CompilationContext.compile_hints(compiler_opts):
                self.fn(*args, **kwargs)
        else:
            self.fn(*args, **kwargs)

    def _run_config(self, config, args, kwargs):
        """Run the chosen config as a real (non-benchmark) call. Re-applies
        reset_to_zero so cache hits and the post-tune run behave like a single
        clean run (restore_value tensors are already restored by _bench_one)."""
        merged = dict(kwargs)
        merged.update(config.all_kwargs())
        self._reset_tensors(args, merged)
        return self._run_with_hints(config.compiler_opts(), args, merged)

    def __call__(self, *args, **kwargs):
        key = self._make_key(args, kwargs)
        if key in self.cache:
            return self._run_config(self.cache[key], args, kwargs)

        # Benchmark all configs
        configs = self._prune(self.configs, args, kwargs)
        print(f"[autotune] tuning {len(configs)} configs...")
        results = []
        for i, config in enumerate(configs):
            try:
                t = self._bench_one(config, args, kwargs)
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

        return self._run_config(best_config, args, kwargs)

    # --- Disk cache ---
    def _load_disk_cache(self):
        if self._cache_file.exists():
            try:
                data = json.loads(self._cache_file.read_text())
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
):
    """Autotune decorator for @jit functions.

    Usage:
        @autotune(configs=[Config(BLOCK=128), Config(BLOCK=256)], key=['n'])
        @flyc.jit
        def myKernel(..., BLOCK: fx.Constexpr[int], ...):
            ...

    Args:
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
        )

    return decorator
