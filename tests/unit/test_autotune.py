# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU-free unit tests for flydsl.autotune.

Covers the parts that must be correct before any real kernel adopts @autotune:
  - Config serialization / kwargs / compiler-opts split
  - Cache-key axes: shape, dtype, stride pattern, device, toolchain, env
  - restore_value snapshot/restore (in-place correctness)
  - reset_to_zero
  - config pruning
  - disk-cache round-trip

These use fake tensor and fake JIT-function stand-ins so they run anywhere,
with no GPU, no torch, and no compiled bindings.
"""

import json

import pytest

from flydsl.autotune import Autotuner, Config, _normalize_strides, autotune


@pytest.fixture(autouse=True)
def _isolate_disk_cache(tmp_path, monkeypatch):
    """Every test gets a private autotune disk cache so results don't leak
    across tests (a cached best-config would skip the benchmark loop)."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "autotune_cache"))


# ── Fakes ────────────────────────────────────────────────────────────────
class FakeTensor:
    """Minimal tensor stand-in with the attributes _make_key / restore_value use."""

    def __init__(self, shape, dtype="float32", strides=None, fill=0.0):
        self.shape = tuple(shape)
        self.dtype = dtype
        if strides is None:
            # row-major contiguous strides
            strides = []
            acc = 1
            for s in reversed(self.shape):
                strides.append(acc)
                acc *= s
            strides = tuple(reversed(strides))
        self._strides = tuple(strides)
        n = 1
        for s in self.shape:
            n *= s
        self._data = [fill] * n

    def stride(self):
        return self._strides

    def zero_(self):
        self._data = [0.0] * len(self._data)

    def clone(self):
        t = FakeTensor(self.shape, self.dtype, self._strides)
        t._data = list(self._data)
        return t

    def copy_(self, other):
        self._data = list(other._data)


def _make_tuner(fn=None, **kw):
    """Build an Autotuner with a no-op fake JIT function."""

    def default_fn(a, out):
        pass

    return Autotuner(
        fn=fn or default_fn,
        configs=kw.pop("configs", [Config(BLOCK=128), Config(BLOCK=256)]),
        key=kw.pop("key", ["a"]),
        warmup=kw.pop("warmup", 1),
        rep=kw.pop("rep", 2),
        **kw,
    )


# ── Config ───────────────────────────────────────────────────────────────
def test_config_roundtrip():
    c = Config(BLOCK=128, num_warps=4, waves_per_eu=2, maxnreg=128)
    d = c.to_dict()
    c2 = Config.from_dict(d)
    assert c2.to_dict() == d
    assert c2.kwargs == {"BLOCK": 128}
    assert c2.num_warps == 4


def test_config_kwargs_vs_compiler_opts():
    c = Config(BLOCK=128, num_warps=4, waves_per_eu=2, maxnreg=96)
    # num_warps is a jit kwarg; waves_per_eu/maxnreg are compiler opts.
    assert c.all_kwargs() == {"BLOCK": 128, "num_warps": 4}
    assert c.compiler_opts() == {"waves_per_eu": 2, "maxnreg": 96}


def test_config_no_compiler_opts_when_unset():
    c = Config(BLOCK=64)
    assert c.compiler_opts() == {}
    assert c.all_kwargs() == {"BLOCK": 64}


def test_config_preserves_per_kernel_occupancy_interface():
    config = Config(
        BLOCK=128,
        waves_per_eu={"kernel_b": 4, "kernel_a": 2},
        maxnreg={"kernel_a": 128},
    )

    assert config.compiler_opts() == {
        "waves_per_eu": {"kernel_b": 4, "kernel_a": 2},
        "maxnreg": {"kernel_a": 128},
    }
    assert Config.from_dict(config.to_dict()).compiler_opts() == config.compiler_opts()
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()
    assert repr(Config(waves_per_eu={"b": 1, "a": 2})) == repr(Config(waves_per_eu={"a": 2, "b": 1}))


def test_config_generic_compile_hints_roundtrip():
    config = Config(
        BLOCK=128,
        compile_hints={
            "future_hint": {"mode": "aggressive"},
            "llvm_options": {"enable-post-misched": False},
            "future_schedule": [1, "two", False, None, {"nested": 3.5}],
        },
    )

    serialized = config.to_dict()
    assert json.loads(json.dumps(serialized)) == serialized
    assert Config.from_dict(serialized).compiler_opts() == config.compiler_opts()
    assert config.compiler_opts() == {
        "future_hint": {"mode": "aggressive"},
        "llvm_options": {"enable-post-misched": False},
        "future_schedule": [1, "two", False, None, {"nested": 3.5}],
    }
    assert "compile_hints=" in repr(config)


def test_config_canonicalizes_fastmath_before_json_persistence():
    config = Config(compile_hints={"fastmath": {"reassoc", "contract"}})

    assert config.compiler_opts() == {"fastmath": "contract,reassoc"}
    assert Config.from_dict(config.to_dict()).compiler_opts() == config.compiler_opts()


@pytest.mark.parametrize(
    ("compile_hints", "match"),
    [
        ({"future_hint": (1, 2)}, "tuple"),
        ({"future_hint": [{"nested": (1,)}]}, "tuple"),
        ({"future_hint": {1: "one"}}, "string keys"),
        ({"future_hint": object()}, "scalar values"),
        ({"future_hint": float("nan")}, "finite"),
    ],
)
def test_config_rejects_compile_hints_that_are_not_json_type_stable(compile_hints, match):
    with pytest.raises((TypeError, ValueError), match=match):
        Config(compile_hints=compile_hints)


def test_config_revalidates_mutated_compile_hints_before_use():
    config = Config(compile_hints={"future_hint": [1, 2]})
    config.compile_hints["future_hint"] = (1, 2)

    with pytest.raises(TypeError, match="tuple"):
        config.compiler_opts()


def test_config_typed_aliases_override_generic_compile_hints():
    config = Config(
        compile_hints={"waves_per_eu": 1, "maxnreg": 64, "future_hint": True},
        waves_per_eu=2,
        maxnreg=0,
    )

    assert config.compiler_opts() == {"waves_per_eu": 2, "maxnreg": 0, "future_hint": True}
    assert config.to_dict() == {
        "waves_per_eu": 2,
        "maxnreg": 0,
        "compile_hints": {"future_hint": True},
    }
    assert Config(compile_hints={"waves_per_eu": 3}, waves_per_eu=None).compiler_opts()["waves_per_eu"] == 3


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (-1, ValueError),
        ("2", TypeError),
        ({2: 2}, TypeError),
        ({"kernel": True}, TypeError),
        ({"kernel": -1}, ValueError),
    ],
)
def test_config_rejects_invalid_waves_per_eu(value, error):
    with pytest.raises(error, match="waves_per_eu"):
        Config(waves_per_eu=value)


def test_config_zero_waves_per_eu_is_an_explicit_reset():
    # Zero must survive the candidate layer so it can override an outer or
    # persistent WPE.  The final compile-hint resolver removes it only after
    # precedence has been resolved, returning codegen to the source baseline.
    assert Config(waves_per_eu=0).compiler_opts() == {"waves_per_eu": 0}
    assert Config(waves_per_eu={"a": 0, "b": 2}).compiler_opts() == {"waves_per_eu": {"a": 0, "b": 2}}


# ── stride normalization ─────────────────────────────────────────────────
def test_normalize_strides_buckets():
    assert _normalize_strides(FakeTensor((4, 8))) == ("s", 1)  # contiguous: inner=1, outer=other
    assert _normalize_strides(FakeTensor((4, 8), strides=(0, 1))) == (0, 1)  # broadcast
    assert _normalize_strides(FakeTensor((4, 8), strides=(16, 2))) == ("s", "s")


# ── cache key ────────────────────────────────────────────────────────────
def test_key_stable_for_same_inputs():
    t = _make_tuner()
    a = FakeTensor((32, 512))
    out = FakeTensor((32, 512))
    assert t._make_key((a, out), {}) == t._make_key((a, out), {})


def test_key_varies_with_shape():
    t = _make_tuner()
    k1 = t._make_key((FakeTensor((32, 512)), FakeTensor((32, 512))), {})
    k2 = t._make_key((FakeTensor((32, 256)), FakeTensor((32, 256))), {})
    assert k1 != k2


def test_key_varies_with_dtype():
    t = _make_tuner()
    k1 = t._make_key((FakeTensor((8, 8), "float32"), FakeTensor((8, 8), "float32")), {})
    k2 = t._make_key((FakeTensor((8, 8), "float16"), FakeTensor((8, 8), "float16")), {})
    assert k1 != k2


def test_key_varies_with_stride_pattern():
    t = _make_tuner()
    contig = FakeTensor((8, 8))
    broadcast = FakeTensor((8, 8), strides=(0, 1))
    k1 = t._make_key((contig, contig), {})
    k2 = t._make_key((broadcast, contig), {})
    assert k1 != k2


def test_key_contains_device_toolchain_env_axes():
    t = _make_tuner()
    key = t._make_key((FakeTensor((8, 8)), FakeTensor((8, 8))), {})
    joined = "".join(key)
    assert "_env_" in joined
    assert "_toolchain_" in joined
    assert "_device_" in joined


def test_key_varies_with_toolchain_fingerprint(monkeypatch):
    import importlib

    at = importlib.import_module("flydsl.autotune")
    t = _make_tuner()
    a = FakeTensor((8, 8))
    k1 = t._make_key((a, a), {})
    # read live per key, so a toolchain change mid-process invalidates the key
    monkeypatch.setattr(at, "_toolchain_fingerprint", lambda: "a-different-fingerprint")
    k2 = t._make_key((a, a), {})
    assert k1 != k2


def test_key_varies_with_device_fingerprint(monkeypatch):
    import importlib

    at = importlib.import_module("flydsl.autotune")
    t = _make_tuner()
    a = FakeTensor((8, 8))
    k1 = t._make_key((a, a), {})
    monkeypatch.setattr(at, "_device_fingerprint", lambda: "gfx_other")
    k2 = t._make_key((a, a), {})
    assert k1 != k2  # arch is a real key axis, read live (not frozen at construction)


def test_key_varies_with_env_fingerprint(monkeypatch):
    """The env axis actually changes the key when the fingerprint changes.

    _env_fingerprint() may degrade to () without the compiled bindings, so we
    patch it at the module level to prove _make_key folds it in (rather than
    only asserting the marker string is present)."""
    import importlib

    at = importlib.import_module("flydsl.autotune")  # module, not the shadowing fn

    t = _make_tuner()
    a = FakeTensor((8, 8))
    monkeypatch.setattr(at, "_env_fingerprint", lambda: (("FLYDSL_COMPILE_OPT_LEVEL", "0"),))
    k1 = t._make_key((a, a), {})
    monkeypatch.setattr(at, "_env_fingerprint", lambda: (("FLYDSL_COMPILE_OPT_LEVEL", "3"),))
    k2 = t._make_key((a, a), {})
    assert k1 != k2


def test_key_insensitive_to_kwarg_order():
    """Semantically identical calls with tensor kwargs in different order must
    produce the same key (no duplicate tuning / cache files)."""
    t = _make_tuner(key=["a"])
    a = FakeTensor((8, 8))
    out = FakeTensor((8, 8), "float16")
    k1 = t._make_key((), {"a": a, "out": out})
    k2 = t._make_key((), {"out": out, "a": a})
    assert k1 == k2


def test_key_normalizes_omitted_and_explicit_function_defaults():
    def fn(a, out, dtype_str="bf16", BLOCK=128):
        pass

    tuner = _make_tuner(fn=fn, key=["dtype_str"])
    args = (FakeTensor((8, 8)), FakeTensor((8, 8)))

    omitted = tuner._make_key(args, {})
    keyword = tuner._make_key(args, {"dtype_str": "bf16"})
    positional = tuner._make_key((*args, "bf16"), {})

    assert omitted == keyword == positional


def test_key_rejects_unknown_launcher_parameter_name():
    with pytest.raises(ValueError, match="dytpe_str"):
        _make_tuner(key=["dytpe_str"])


def test_scalar_key_preserves_value_type():
    def fn(a, out, mode=None):
        pass

    tuner = _make_tuner(fn=fn, key=["mode"])
    args = (FakeTensor((8, 8)), FakeTensor((8, 8)))

    assert tuner._make_key(args, {"mode": 1}) != tuner._make_key(args, {"mode": "1"})


def test_key_varies_with_baseline_compile_hints():
    baseline = {"waves_per_eu": 1}

    def fn(a, out):
        pass

    fn._effective_compile_hints = lambda: dict(baseline)
    tuner = _make_tuner(fn=fn)
    args = (FakeTensor((8, 8)), FakeTensor((8, 8)))
    first = tuner._make_key(args, {})
    baseline["waves_per_eu"] = 2
    second = tuner._make_key(args, {})

    assert first != second


# ── restore_value (in-place correctness) ────────────────────────────────
def test_restore_value_restores_between_reps():
    """A kernel that mutates its input in place must see pristine inputs on
    every rep. We record the input's first element at kernel entry across reps;
    without restore they'd diverge, with restore they stay identical."""
    seen = []

    def in_place_fn(a, out, **kw):
        seen.append(a._data[0])
        a._data[0] += 100.0  # corrupt the input, as an in-place kernel would

    t = _make_tuner(
        fn=in_place_fn,
        configs=[Config(BLOCK=128)],
        restore_value=["a"],
        do_bench_fn=lambda call, warmup, rep: ([call() for _ in range(warmup + rep)], 1.0)[1],
    )
    a = FakeTensor((4,), fill=7.0)
    out = FakeTensor((4,))
    t(a, out)
    # Every observed entry value must be the pristine 7.0.
    assert seen, "kernel never ran"
    assert all(v == 7.0 for v in seen), f"input corrupted across reps: {seen}"


def test_restore_value_no_op_without_list():
    """Without restore_value, an in-place kernel corrupts across reps (baseline
    that proves the mechanism is what fixes it)."""
    seen = []

    def in_place_fn(a, out, **kw):
        seen.append(a._data[0])
        a._data[0] += 100.0

    t = _make_tuner(
        fn=in_place_fn,
        configs=[Config(BLOCK=128)],
        do_bench_fn=lambda call, warmup, rep: ([call() for _ in range(warmup + rep)], 1.0)[1],
    )
    t(FakeTensor((4,), fill=7.0), FakeTensor((4,)))
    assert seen[0] == 7.0 and seen[-1] != 7.0  # corrupted without restore


def test_reset_to_zero():
    seen = []

    def acc_fn(a, out, **kw):
        seen.append(out._data[0])
        out._data[0] += 1.0

    t = _make_tuner(
        fn=acc_fn,
        configs=[Config(BLOCK=128)],
        reset_to_zero=["out"],
        do_bench_fn=lambda call, warmup, rep: ([call() for _ in range(warmup + rep)], 1.0)[1],
    )
    out = FakeTensor((4,), fill=5.0)
    t(FakeTensor((4,)), out)
    # Every benchmark rep AND the final real run must see a freshly-zeroed out.
    assert all(v == 0.0 for v in seen), seen
    # And the user-visible result must equal a single clean run (accumulate once
    # from zero -> 1.0), not carry benchmark-rep state.
    assert out._data[0] == 1.0, out._data[0]


def test_reset_to_zero_on_cache_hit():
    """A cached best-config call must also reset (not just the tuning run)."""

    def acc_fn(a, out, **kw):
        acc_fn.entry = out._data[0]
        out._data[0] += 1.0

    t = _make_tuner(
        fn=acc_fn,
        configs=[Config(BLOCK=128)],
        reset_to_zero=["out"],
        do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1],
    )
    a, out = FakeTensor((4,)), FakeTensor((4,))
    t(a, out)  # tune + populate cache
    out2 = FakeTensor((4,), fill=99.0)
    t(a, out2)  # cache hit
    assert acc_fn.entry == 0.0  # reset happened on the cache-hit path
    assert out2._data[0] == 1.0


# ── pruning ──────────────────────────────────────────────────────────────
def test_prune_configs_by():
    def only_small(configs, sig_args):
        return [c for c in configs if c.kwargs.get("BLOCK", 0) <= 128]

    def bench(call, warmup, rep):
        call()
        # cheaper config (smaller block) should still be the only survivor
        return 1.0

    t = _make_tuner(
        fn=lambda a, out, **kw: None,
        configs=[Config(BLOCK=64), Config(BLOCK=128), Config(BLOCK=512)],
        prune_configs_by=only_small,
        do_bench_fn=bench,
    )
    pruned = t._prune(t.configs, (FakeTensor((4,)), FakeTensor((4,))), {})
    assert [c.kwargs["BLOCK"] for c in pruned] == [64, 128]


# ── disk cache ───────────────────────────────────────────────────────────
def test_disk_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    calls = {"n": 0}

    def bench(call, warmup, rep):
        calls["n"] += 1
        call()
        return float(calls["n"])  # first config slower than second-run cache hit

    t1 = _make_tuner(fn=lambda a, out, **kw: None, configs=[Config(BLOCK=128)], do_bench_fn=bench)
    a = FakeTensor((16, 64))
    out = FakeTensor((16, 64))
    t1(a, out)
    n_after_tune = calls["n"]
    assert n_after_tune >= 1

    # A fresh tuner should load the persisted best config and skip benchmarking.
    t2 = _make_tuner(fn=lambda a, out, **kw: None, configs=[Config(BLOCK=128)], do_bench_fn=bench)
    key = t2._make_key((a, out), {})
    assert key in t2.cache, "best config was not persisted/reloaded"

    # The persisted file is valid JSON keyed by the serialized cache key.
    files = list(tmp_path.glob("*.json"))
    assert files, "no disk cache file written"
    data = json.loads(files[0].read_text())
    assert data, "empty disk cache"


# ── decorator ────────────────────────────────────────────────────────────
def test_autotune_decorator_wraps_into_autotuner():
    """@autotune returns an Autotuner that forwards restore_value/reset_to_zero."""

    def fake_jit(a, out, **kw):
        pass

    default = lambda a, out: Config(BLOCK=64)
    tuned = autotune(
        configs=[Config(BLOCK=128)],
        key=["a"],
        restore_value=["a"],
        reset_to_zero=["out"],
        default=default,
    )(fake_jit)

    assert isinstance(tuned, Autotuner)
    assert tuned.restore_value == ["a"]
    assert tuned.reset_to_zero == ["out"]
    assert tuned.default is default
    assert [c.kwargs["BLOCK"] for c in tuned.configs] == [128]


def test_autotune_source_fingerprint_includes_selection_callables(monkeypatch):
    import importlib

    at = importlib.import_module("flydsl.autotune")
    captured = []

    def configs(a, out):
        return [Config(BLOCK=128)]

    def default(a, out):
        return Config(BLOCK=128)

    def prune(configs, named_args):
        return configs

    def bench(call, warmup, rep):
        return 1.0

    monkeypatch.setattr(at, "_source_fingerprint", lambda callables: captured.extend(callables) or "fingerprint")

    @autotune(configs=configs, key=["a"], default=default, prune_configs_by=prune, do_bench=bench)
    def fn(a, out, BLOCK=128):
        pass

    assert fn.source_fingerprint == "fingerprint"
    assert configs in captured
    assert default in captured
    assert prune in captured
    assert bench in captured


# ── two-track default ────────────────────────────────────────────
def _bench_run_all(call, warmup, rep):
    # deterministic fake do_bench: run once, return a constant time
    call()
    return 1.0


def test_default_skips_search(monkeypatch):
    """With a default heuristic and FLYDSL_AUTOTUNE off, no benchmarking runs."""
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    benched = {"n": 0}

    def bench(call, warmup, rep):
        benched["n"] += 1
        call()
        return 1.0

    def fn(a, out, BLOCK):
        out._data[0] = float(BLOCK)

    def default(a, out, **kw):
        return Config(BLOCK=999)

    t = Autotuner(
        fn=fn,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        key=["a"],
        warmup=1,
        rep=1,
        default=default,
        do_bench_fn=bench,
    )
    out = FakeTensor((1,))
    t(FakeTensor((8,)), out)
    assert benched["n"] == 0  # no search
    assert out._data[0] == 999.0  # heuristic default was used


def test_default_forced_search_with_env(monkeypatch):
    """FLYDSL_AUTOTUNE=1 forces the full search even when a default exists."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    benched = {"n": 0}

    def bench(call, warmup, rep):
        benched["n"] += 1
        call()
        return 1.0

    def fn(a, out, BLOCK):
        out._data[0] = float(BLOCK)

    t = Autotuner(
        fn=fn,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        key=["a"],
        warmup=1,
        rep=1,
        default=lambda a, out, **kw: Config(BLOCK=64),
        do_bench_fn=bench,
    )
    t(FakeTensor((8,)), FakeTensor((1,)))
    assert benched["n"] == 2  # both configs searched


def test_cache_hit_runtime_error_does_not_delete_valid_tuning_result():
    def fn(a, out, BLOCK):
        raise TypeError("launcher bug")

    tuned = Autotuner(
        fn=fn,
        configs=[Config(BLOCK=64)],
        key=["a"],
        warmup=1,
        rep=1,
    )
    args = (FakeTensor((8,)), FakeTensor((1,)))
    key = tuned._make_key(args, {})
    tuned.cache[key] = Config(BLOCK=64)

    with pytest.raises(TypeError, match="launcher bug"):
        tuned(*args)

    assert tuned.cache[key].kwargs == {"BLOCK": 64}


def test_tuning_enabled_env(monkeypatch):
    from flydsl.autotune import _tuning_enabled

    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    assert _tuning_enabled() is False
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    assert _tuning_enabled() is False
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    assert _tuning_enabled() is True
    monkeypatch.setenv("FLYDSL_AUTOTUNE", " FALSE ")
    assert _tuning_enabled() is False


def test_run_with_hints_uses_thread_local_not_shared_attr():
    """A candidate hint is a thread-local overlay, never a mutation of the
    shared cached JitFunction."""
    pytest.importorskip("flydsl._mlir._mlir_libs._mlirDialectsFly")
    from flydsl.compiler.kernel_function import CompilationContext

    class FakeJit:
        def __init__(self):
            self.compile_hints = {"baseline": 1}
            self.seen = None

        def __call__(self, *a, **k):
            self.seen = dict(CompilationContext.get_compile_hints())

    fn = FakeJit()
    t = _make_tuner(fn=fn, key=[], configs=[Config(BLOCK=1)])
    with CompilationContext.compile_hints({"outer": 7, "waves_per_eu": 1}):
        t._run_with_hints({"waves_per_eu": 2}, (), {})
    assert fn.seen == {"outer": 7, "waves_per_eu": 2}
    assert fn.compile_hints == {"baseline": 1}
    assert CompilationContext.get_compile_hints() == {}


def test_candidate_zero_resets_outer_and_persistent_occupancy():
    """A candidate zero is a real overlay, not an absent candidate option."""
    pytest.importorskip("flydsl._mlir._mlir_libs._mlirDialectsFly")
    import flydsl.compiler as flyc
    from flydsl.compiler.kernel_function import CompilationContext

    @flyc.jit
    def hint_owner():
        pass

    hint_owner.compile_hints = {"persistent": 1, "waves_per_eu": 4}

    class Observer:
        def __call__(self):
            self.seen = hint_owner._effective_compile_hints()

    observer = Observer()
    tuner = _make_tuner(fn=observer, key=[])
    with CompilationContext.compile_hints({"outer": 2, "waves_per_eu": 3}):
        tuner._run_with_hints(Config(waves_per_eu=0).compiler_opts(), (), {})

    assert observer.seen == {"persistent": 1, "outer": 2}


def test_cache_hit_applies_compile_hints_to_direct_function():
    pytest.importorskip("flydsl._mlir._mlir_libs._mlirDialectsFly")
    from flydsl.compiler.kernel_function import CompilationContext

    class Observer:
        def __init__(self):
            self.compile_hints = {"baseline": 1}
            self.seen = None

        def __call__(self, a, out, BLOCK):
            self.seen = (BLOCK, dict(CompilationContext.get_compile_hints()))

    fn = Observer()
    tuner = _make_tuner(fn=fn, configs=[Config(BLOCK=64, waves_per_eu=2)])
    args = (FakeTensor((8,)), FakeTensor((1,)))
    key = tuner._make_key(args, {})
    tuner.cache[key] = Config(BLOCK=64, waves_per_eu=2)

    with CompilationContext.compile_hints({"outer": 7, "waves_per_eu": 1}):
        tuner(*args)

    assert fn.seen == (64, {"outer": 7, "waves_per_eu": 2})
    assert fn.compile_hints == {"baseline": 1}


def test_search_loop_chains_last_error_when_all_fail():
    """If every config fails to benchmark, the RuntimeError must chain the last
    underlying error (not discard it) so the real cause is recoverable."""

    def boom(a, out, **kw):
        raise RuntimeError("kernel boom")

    t = _make_tuner(fn=boom, configs=[Config(BLOCK=1)], do_bench_fn=_bench_run_all)
    with pytest.raises(RuntimeError, match="All autotune configs failed") as ei:
        t(FakeTensor((8,)), FakeTensor((1,)))
    assert isinstance(ei.value.__cause__, RuntimeError) and "boom" in str(ei.value.__cause__)


def test_source_fingerprint_folds_into_key():
    """A change in the adopter's kernel/config source (fingerprint) must change
    the cache key, so a stale tuned best isn't served after a kernel edit."""
    a = FakeTensor((8, 8))
    t1 = _make_tuner()
    t1.source_fingerprint = "aaaa"
    t2 = _make_tuner()
    t2.source_fingerprint = "bbbb"
    assert t1._make_key((a, a), {}) != t2._make_key((a, a), {})


def test_disk_cache_skips_malformed_entry(tmp_path, monkeypatch):
    """One malformed disk-cache entry must be skipped, not discard the whole
    cache (previously a single bad entry dropped everything)."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    t = _make_tuner(fn=lambda a, out, **kw: None, configs=[Config(BLOCK=128)], do_bench_fn=_bench_run_all)
    a, out = FakeTensor((16, 64)), FakeTensor((16, 64))
    t(a, out)  # tune -> writes one valid entry to disk
    cache_file = next(tmp_path.glob("*.json"))
    data = json.loads(cache_file.read_text())
    assert len(data) == 1
    data["not-a-json-key"] = {"BLOCK": 1}  # malformed key_str -> parse fails on load
    cache_file.write_text(json.dumps(data))

    t2 = _make_tuner(fn=lambda a, out, **kw: None, configs=[Config(BLOCK=128)], do_bench_fn=_bench_run_all)
    assert t2._make_key((a, out), {}) in t2.cache  # good entry survived
    assert len(t2.cache) == 1  # malformed one skipped


def test_call_do_bench_passes_setup_when_supported():
    """When the benchmarker accepts `setup`, it's passed through (so restore/reset
    runs untimed) -- setup then kernel, in that order."""
    order = []

    def bench_with_setup(fn, warmup, rep, setup=None):
        setup()
        fn()
        return 1.0

    t = _make_tuner(do_bench_fn=bench_with_setup)
    t._call_do_bench(lambda: order.append("kernel"), lambda: order.append("setup"))
    assert order == ["setup", "kernel"]


def test_call_do_bench_folds_setup_when_unsupported():
    """A custom do_bench_fn without a `setup` param still runs setup (folded into
    the timed call) -- so restore/reset isn't skipped."""
    order = []

    def bench_no_setup(fn, warmup, rep):
        fn()
        return 1.0

    t = _make_tuner(do_bench_fn=bench_no_setup)
    t._call_do_bench(lambda: order.append("kernel"), lambda: order.append("setup"))
    assert order == ["setup", "kernel"]


def test_call_do_bench_folds_setup_for_kwargs_only_benchmarker():
    """A do_bench_fn with **kwargs but no explicit `setup` must still run setup
    (folded), not have it passed-and-silently-dropped into **kwargs."""
    order = []

    def bench_kwargs(fn, warmup, rep, **kwargs):  # does NOT forward setup
        fn()
        return 1.0

    t = _make_tuner(do_bench_fn=bench_kwargs)
    t._call_do_bench(lambda: order.append("kernel"), lambda: order.append("setup"))
    assert order == ["setup", "kernel"]  # setup ran (folded), not dropped


@pytest.mark.parametrize(
    ("arch", "extra_pairs"),
    [
        ("gfx950", set()),
        ("gfx1201", {(512, 1), (1024, 1), (1024, 2)}),
    ],
)
def test_rmsnorm_configs_route_wpe_as_compile_option(arch, extra_pairs):
    """RMSNorm keeps JIT config kwargs separate from backend options."""
    pytest.importorskip("flydsl._mlir._mlir_libs._mlirDialectsFly")
    from kernels.norm.rmsnorm_autotune import rmsnorm_autotuned
    from kernels.norm.rmsnorm_config import _BLOCK_THREADS_CHOICES, get_all_configs
    from kernels.norm.rmsnorm_kernel import rmsnorm_direct

    cfgs = get_all_configs(8192, "f32", arch=arch)
    blocks = sorted({c.kwargs["BLOCK_THREADS"] for c in cfgs})
    assert blocks == sorted(_BLOCK_THREADS_CHOICES)  # every block present (no tile filter for f32)
    assert all("WAVES_PER_EU" not in c.kwargs for c in cfgs)

    def effective_wpe(config):
        return config.compiler_opts().get("waves_per_eu", 0)

    assert {effective_wpe(c) for c in cfgs} == {0, 1, 2}
    expected_pairs = {
        (128, 0),
        (128, 1),
        (128, 2),
        (256, 0),
        (256, 1),
        (256, 2),
        (512, 0),
        (512, 2),
        (1024, 0),
    }
    assert {(c.kwargs["BLOCK_THREADS"], effective_wpe(c)) for c in cfgs} == expected_pairs | extra_pairs
    assert rmsnorm_autotuned.tuner.fn is rmsnorm_direct


def test_get_default_bf16_hits_vectorized_tile():
    """get_default must pick a BLOCK_THREADS whose tile divides N for bf16/f16,
    so the zero-search default hits the vectorized fast path (regression: N=5120
    used to pick 256 -> tile 2048 -> scalar). f32 is unaffected (scalar path)."""
    pytest.importorskip("flydsl._mlir._mlir_libs._mlirDialectsFly")
    from kernels.norm.rmsnorm_config import _BLOCK_THREADS_CHOICES, _elem_bits, get_default

    vec_width = 128 // _elem_bits("bf16")
    for N in (4096, 5120, 7168, 8192):
        block = get_default(N, "bf16").kwargs["BLOCK_THREADS"]
        assert N % (block * vec_width) == 0, f"bf16 N={N}: block {block} misses the vectorized tile"
    # N=5120 specifically resolves to 128 (256's tile 2048 does not divide 5120)
    assert get_default(5120, "bf16").kwargs["BLOCK_THREADS"] == 128
    # f32 uses the scalar path, so the divisibility filter does not apply
    assert get_default(5120, "f32").kwargs["BLOCK_THREADS"] in _BLOCK_THREADS_CHOICES


def test_cache_dir_change_does_not_serve_stale_config(tmp_path, monkeypatch):
    """Switching FLYDSL_AUTOTUNE_CACHE_DIR must drop the in-memory config tuned
    under the old dir. The fake tune picks BLOCK=64; the default is BLOCK=7.
    After switching to an empty dir with tuning OFF, the call must fall to the
    default (7) — proving the stale dir-A best (64) was cleared, not served."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "A"))

    def fn(a, out, BLOCK):
        out._data[0] = float(BLOCK)

    tuner = Autotuner(
        fn=fn,
        configs=[Config(BLOCK=64)],
        key=["a"],
        warmup=1,
        rep=1,
        default=lambda a, out: Config(BLOCK=7),
        do_bench_fn=_bench_run_all,
    )

    # 1. Force a tune into dir A -> in-memory best BLOCK=64.
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    out = FakeTensor((1,))
    tuner(FakeTensor((16, 512)), out)
    assert out._data[0] == 64.0  # tuned config in memory

    # 2. Switch to empty dir B, tuning OFF: stale in-memory best must be dropped,
    #    so this serves the heuristic default (7), not dir-A's 64.
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "B"))
    out2 = FakeTensor((1,))
    tuner(FakeTensor((16, 512)), out2)
    assert out2._data[0] == 7.0, "served a stale in-memory config from the old cache dir"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
