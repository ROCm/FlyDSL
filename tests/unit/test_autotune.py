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

from flydsl.autotune import Autotuner, Config, _normalize_strides, autotune, autotune_builder


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
    """Build an Autotuner with named args (a, out) and a no-op fake jit fn."""

    def default_fn(a, out):  # signature drives arg_names
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

    tuned = autotune(
        configs=[Config(BLOCK=128)],
        key=["a"],
        restore_value=["a"],
        reset_to_zero=["out"],
    )(fake_jit)

    assert isinstance(tuned, Autotuner)
    assert tuned.restore_value == ["a"]
    assert tuned.reset_to_zero == ["out"]
    assert [c.kwargs["BLOCK"] for c in tuned.configs] == [128]


# ── builder mode + two-track default ─────────────────────────────────────
def _bench_run_all(call, warmup, rep):
    # deterministic fake do_bench: run once, return a constant time
    call()
    return 1.0


def test_builder_mode_rebuilds_per_config():
    """build_fn is called once per config; the returned fn is what runs."""
    built = []

    def build_fn(config, a, out, **kw):
        block = config.kwargs["BLOCK"]
        built.append(block)

        def launch(a, out, **kw):
            out._data[0] = float(block)  # record which build ran

        return launch

    t = Autotuner(
        fn=None,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        key=["a"],
        warmup=1,
        rep=1,
        build_fn=build_fn,
        do_bench_fn=_bench_run_all,
    )
    a = FakeTensor((8,))
    out = FakeTensor((1,))
    t(a, out)
    # both configs built + benchmarked exactly once
    assert sorted(built) == [64, 128]
    # arg_names inferred from build_fn with leading 'config' stripped
    assert t.arg_names[:2] == ["a", "out"]


def test_builder_mode_caches_built_modules():
    """Re-running the same (key, config) does not rebuild."""
    n_builds = {"n": 0}

    def build_fn(config, a, out, **kw):
        n_builds["n"] += 1
        return lambda a, out, **kw: None

    t = Autotuner(
        fn=None,
        configs=[Config(BLOCK=64)],
        key=["a"],
        warmup=1,
        rep=1,
        build_fn=build_fn,
        do_bench_fn=_bench_run_all,
    )
    a, out = FakeTensor((8,)), FakeTensor((1,))
    t(a, out)  # tune: builds once
    t(a, out)  # cached best: reuses build
    assert n_builds["n"] == 1


def test_default_skips_search(monkeypatch):
    """With a default heuristic and FLYDSL_AUTOTUNE off, no benchmarking runs."""
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    benched = {"n": 0}

    def bench(call, warmup, rep):
        benched["n"] += 1
        call()
        return 1.0

    ran = {"block": None}

    def build_fn(config, a, out, **kw):
        return lambda a, out, **kw: ran.__setitem__("block", config.kwargs["BLOCK"])

    def default(a, out, **kw):
        return Config(BLOCK=999)

    t = Autotuner(
        fn=None,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        key=["a"],
        warmup=1,
        rep=1,
        build_fn=build_fn,
        default=default,
        do_bench_fn=bench,
    )
    t(FakeTensor((8,)), FakeTensor((1,)))
    assert benched["n"] == 0  # no search
    assert ran["block"] == 999  # heuristic default was used


def test_default_forced_search_with_env(monkeypatch):
    """FLYDSL_AUTOTUNE=1 forces the full search even when a default exists."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    benched = {"n": 0}

    def bench(call, warmup, rep):
        benched["n"] += 1
        call()
        return 1.0

    t = Autotuner(
        fn=None,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        key=["a"],
        warmup=1,
        rep=1,
        build_fn=build_fn_noop,
        default=lambda a, out, **kw: Config(BLOCK=64),
        do_bench_fn=bench,
    )
    t(FakeTensor((8,)), FakeTensor((1,)))
    assert benched["n"] == 2  # both configs searched


def build_fn_noop(config, a, out, **kw):
    return lambda a, out, **kw: None


def test_filter_call_kwargs_drops_build_only_kwargs():
    """Builder-only kwargs (e.g. dtype_str) must not reach the launch fn."""

    def build_fn(config, a, out, dtype_str="bf16", **kw):
        # launch fn only accepts a, out — not dtype_str
        def launch(a, out):
            pass

        return launch

    t = Autotuner(
        fn=None,
        configs=[Config(BLOCK=64)],
        key=["a"],
        warmup=1,
        rep=1,
        build_fn=build_fn,
        default=lambda a, out, **kw: Config(BLOCK=64),
        do_bench_fn=_bench_run_all,
    )
    # dtype_str would raise TypeError if not filtered out before the launch call
    t(FakeTensor((8,)), FakeTensor((1,)), dtype_str="f16")


def test_tuning_enabled_env(monkeypatch):
    from flydsl.autotune import _tuning_enabled

    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    assert _tuning_enabled() is False
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    assert _tuning_enabled() is False
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    assert _tuning_enabled() is True


# ── autotune_builder (one-call adoption) ─────────────────────────────────
def _make_builder(**over):
    """A fake-kernel autotune_builder: build() records which config ran into the
    output tensor's [0] slot; specialize keys on N + dtype_str."""
    built = over.pop("_built_log", [])

    def build(N, dtype_str, BLOCK=0):
        built.append((N, dtype_str, BLOCK))

        def launch(a, out, dtype_str="bf16", stream=None):
            out._data[0] = float(BLOCK)

        return launch

    def specialize(a, out, dtype_str="bf16", stream=None):
        return {"N": a.shape[-1], "dtype_str": dtype_str}

    kw = dict(
        name="fakeop",
        build=build,
        specialize=specialize,
        configs=lambda N, dtype_str: [Config(BLOCK=64), Config(BLOCK=128)],
        default=lambda N, dtype_str: Config(BLOCK=7),
        structural=("BLOCK",),
        warmup=1,
        rep=1,
    )
    kw.update(over)
    t = autotune_builder(**kw)
    return t, built


def test_builder_default_runs_without_search(monkeypatch):
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    t, built = _make_builder()
    a, out = FakeTensor((16, 512)), FakeTensor((1,))
    t(a, out, dtype_str="bf16")
    assert out._data[0] == 7.0  # heuristic default's BLOCK
    assert built == [(512, "bf16", 7)]  # built once, from the default


def test_builder_search_sweeps_space(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    t, built = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    a, out = FakeTensor((16, 512)), FakeTensor((1,))
    t(a, out, dtype_str="bf16")
    swept = sorted(b for _, _, b in built)
    assert swept == [64, 128]  # both configs from get_all_configs were built


def test_builder_scalar_enters_key():
    """dtype_str is build-only but must change the cache key (bug: scalars that
    change codegen without changing shape were dropped from the key)."""
    t, _ = _make_builder()
    a = FakeTensor((16, 512))
    k_bf16 = t.tuner._make_key((), {"a": a, "out": FakeTensor((1,)), "dtype_str": "bf16"})
    k_f16 = t.tuner._make_key((), {"a": a, "out": FakeTensor((1,)), "dtype_str": "f16"})
    assert k_bf16 != k_f16


def test_builder_requires_name():
    """Builder mode must carry a distinct cache name (else tuners collide on
    unknown.json)."""
    t, _ = _make_builder(name="softmax")
    assert t.tuner.name == "softmax"
    assert t.tuner._cache_file.name == "softmax.json"
    # empty / missing name must be rejected (else tuners collide on unknown.json)
    with pytest.raises(ValueError):
        _make_builder(name="")


def test_builder_positional_scalar_rejected(monkeypatch):
    """A build-only scalar (dtype_str) passed positionally is rejected up front
    (codex#1: silently binding it to the wrong launch slot is worse)."""
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)

    def build(N, dtype_str, BLOCK=0):
        return lambda a, out, stream=None: None

    def specialize(a, out, dtype_str="bf16", stream=None):
        return {"N": a.shape[-1], "dtype_str": dtype_str}

    t = autotune_builder(
        name="op",
        build=build,
        specialize=specialize,
        configs=lambda N, dtype_str: [Config(BLOCK=1)],
        default=lambda N, dtype_str: Config(BLOCK=1),
        structural=("BLOCK",),
        warmup=1,
        rep=1,
    )
    # dtype_str keyword: fine.
    t(FakeTensor((16, 512)), FakeTensor((1,)), dtype_str="f16")
    # dtype_str positional (5th arg): rejected with a clear error.
    with pytest.raises(TypeError, match="by keyword"):
        t(FakeTensor((16, 512)), FakeTensor((1,)), "f16")


def test_builder_build_cache_ignores_compiler_hints(monkeypatch, tmp_path):
    """configs differing only in waves_per_eu must build the module once, not
    once per hint (C1: build cache was over-keyed on repr(config))."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    n_build = {"n": 0}

    def build(N, dtype_str, BLOCK=0):
        n_build["n"] += 1
        return lambda a, out, stream=None: None

    def specialize(a, out, dtype_str="bf16", stream=None):
        return {"N": a.shape[-1], "dtype_str": dtype_str}

    # 3 configs, same BLOCK, differing only in waves_per_eu.
    space = [Config(BLOCK=64, waves_per_eu=w) for w in (None, 1, 2)]
    t = autotune_builder(
        name="op",
        build=build,
        specialize=specialize,
        configs=lambda N, dtype_str: space,
        structural=("BLOCK",),
        warmup=1,
        rep=1,
        do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1],
    )
    t(FakeTensor((16, 512)), FakeTensor((1,)), dtype_str="bf16")
    assert n_build["n"] == 1, f"rebuilt {n_build['n']}x for hint-only variants (should be 1)"


def test_run_with_hints_sets_and_restores_compile_hints():
    """_run_with_hints must set fn.compile_hints during the call (so the hint
    enters the JIT cache key) and restore it after (no cross-config leak)."""

    class FakeJit:
        def __init__(self):
            self.compile_hints = {}
            self.seen = None

        def __call__(self, *a, **k):
            self.seen = dict(self.compile_hints)

    # No compiler bindings needed: with hints present, _run_with_hints imports
    # CompilationContext, so skip when the compiled bindings are absent.
    pytest.importorskip("flydsl._mlir._mlir_libs._mlirDialectsFly")
    fn = FakeJit()
    t = _make_tuner(fn=lambda a, out, **kw: None, configs=[Config(BLOCK=1)])
    t._run_with_hints(fn, Config(BLOCK=1, waves_per_eu=2).compiler_opts(), (), {})
    assert fn.seen == {"waves_per_eu": 2}  # in the cache-key dict during compile
    assert fn.compile_hints == {}  # restored afterward


def test_cache_dir_change_does_not_serve_stale_config(tmp_path, monkeypatch):
    """Switching FLYDSL_AUTOTUNE_CACHE_DIR must drop the in-memory config tuned
    under the old dir. The fake tune picks BLOCK=64; the default is BLOCK=7.
    After switching to an empty dir with tuning OFF, the call must fall to the
    default (7) — proving the stale dir-A best (64) was cleared, not served."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "A"))

    # 1. Force a tune into dir A -> in-memory best BLOCK=64.
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    t, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    out = FakeTensor((1,))
    t(FakeTensor((16, 512)), out, dtype_str="bf16")
    assert out._data[0] == 64.0  # tuned config in memory

    # 2. Switch to empty dir B, tuning OFF: stale in-memory best must be dropped,
    #    so this serves the heuristic default (7), not dir-A's 64.
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "B"))
    out2 = FakeTensor((1,))
    t(FakeTensor((16, 512)), out2, dtype_str="bf16")
    assert out2._data[0] == 7.0, "served a stale in-memory config from the old cache dir"


# ── offline artifacts (aiter/SGLang model) ───────────────────────────────
def test_offline_filename_is_key_and_sanitized():
    from flydsl.autotune import offline_config_filename

    name = offline_config_filename("op", [("N", 8192), ("dtype_str", "bf16")], device_name="AMD MI300X")
    assert name == "op,N=8192,dtype_str=bf16,device_name=AMD_MI300X.json"
    # injection is neutralized: no raw delimiters / path separators survive
    evil = offline_config_filename("../x", [("N", "1,dtype_str=y")], device_name="a/b")
    assert "/" not in evil and evil.count(".json") == 1


def _write_offline(cfg_dir, spec_items, config, name="fakeop", device=None, raw=None):
    """Write an offline artifact; returns its path. `raw` overrides the payload."""
    from flydsl.autotune import _device_name, offline_config_filename

    cfg_dir.mkdir(parents=True, exist_ok=True)
    device = device or _device_name()
    path = cfg_dir / offline_config_filename(name, spec_items, device_name=device)
    payload = (
        raw
        if raw is not None
        else {
            "name": name,
            "spec": dict(spec_items),
            "device_name": device,
            "config": config,
        }
    )
    path.write_text(json.dumps(payload))
    return path


def test_offline_emit_then_served_is_the_offline_config(tmp_path, monkeypatch):
    """Prove the OFFLINE config is what serves — not the heuristic default.

    The fake build writes the chosen BLOCK into out[0]. Forced tune picks BLOCK=64
    (first config, equal timings); the default is BLOCK=7. A fresh tuner with
    tuning OFF and a *separate* scratch cache dir must produce 64, so we know the
    value came from the committed artifact, not the default and not disk cache."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "cfg"))

    # 1. Force a tune -> emits the artifact (BLOCK=64).
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "scratch1"))
    t, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    t(FakeTensor((16, 512)), FakeTensor((1,)), dtype_str="bf16")
    payload = json.loads(next((tmp_path / "cfg").glob("*.json")).read_text())
    assert payload["spec"] == {"N": 512, "dtype_str": "bf16"}
    assert payload["config"]["BLOCK"] == 64

    # 2. Fresh tuner, tuning OFF, SEPARATE scratch dir (so a disk-cache hit can't
    #    masquerade as an offline hit), and no benchmarking allowed.
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "scratch2"))
    t2, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    n = {"bench": 0}
    orig = t2.tuner._bench_one
    t2.tuner._bench_one = lambda *a, **k: (n.__setitem__("bench", n["bench"] + 1), orig(*a, **k))[1]
    out = FakeTensor((1,))
    t2(FakeTensor((16, 512)), out, dtype_str="bf16")
    assert n["bench"] == 0  # no search
    assert out._data[0] == 64.0  # served the OFFLINE config (64), not default (7)


def test_offline_stale_artifact_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    # filename matches N=512, but content claims N=4096 (stale / hand-edited)
    _write_offline(
        tmp_path / "cfg",
        [("N", 512), ("dtype_str", "bf16")],
        {"BLOCK": 1},
        raw={
            "name": "fakeop",
            "spec": {"N": 4096, "dtype_str": "bf16"},
            "device_name": __import__("flydsl.autotune", fromlist=["_device_name"])._device_name(),
            "config": {"BLOCK": 1},
        },
    )
    t, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    out = FakeTensor((1,))
    t(FakeTensor((16, 512)), out, dtype_str="bf16")
    assert "ignoring" in capsys.readouterr().out  # rejected
    assert out._data[0] == 7.0  # fell back to default (BLOCK=7), not the stale config


def test_offline_malformed_spec_does_not_crash(tmp_path, monkeypatch, capsys):
    """spec=null (present but not a dict) must warn + fall back, never raise."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    _write_offline(
        tmp_path / "cfg",
        [("N", 512), ("dtype_str", "bf16")],
        None,
        raw={
            "name": "fakeop",
            "spec": None,
            "device_name": __import__("flydsl.autotune", fromlist=["_device_name"])._device_name(),
            "config": {"BLOCK": 1},
        },
    )
    t, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    out = FakeTensor((1,))
    t(FakeTensor((16, 512)), out, dtype_str="bf16")  # must not raise
    assert "ignoring" in capsys.readouterr().out
    assert out._data[0] == 7.0  # default fallback


def test_offline_empty_config_rejected(tmp_path, monkeypatch, capsys):
    """A matching artifact with an empty/partial config must be rejected, not
    served (else builder mode silently runs build_fn's default structural knob).
    The fake build's structural knob is BLOCK; an empty config lacks it."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    _write_offline(
        tmp_path / "cfg",
        [("N", 512), ("dtype_str", "bf16")],
        {},  # empty config body
        raw={
            "name": "fakeop",
            "spec": {"N": 512, "dtype_str": "bf16"},
            "device_name": __import__("flydsl.autotune", fromlist=["_device_name"])._device_name(),
            "config": {},
        },
    )
    t, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    out = FakeTensor((1,))
    t(FakeTensor((16, 512)), out, dtype_str="bf16")
    assert "ignoring" in capsys.readouterr().out
    assert out._data[0] == 7.0  # fell back to default (BLOCK=7), not the empty artifact


def test_offline_corrupt_and_missing_fields(tmp_path, monkeypatch, capsys):
    from flydsl.autotune import _device_name, offline_config_filename

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(cfg))
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    fname = offline_config_filename("fakeop", [("N", 512), ("dtype_str", "bf16")], device_name=_device_name())
    # unparseable JSON
    (cfg / fname).write_text("{ not json")
    t, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    t(FakeTensor((16, 512)), FakeTensor((1,)), dtype_str="bf16")
    assert "unreadable" in capsys.readouterr().out
    # present file, missing required fields
    (cfg / fname).write_text(json.dumps({"config": {"BLOCK": 1}}))
    t2, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    t2(FakeTensor((16, 512)), FakeTensor((1,)), dtype_str="bf16")
    assert "missing" in capsys.readouterr().out


def test_offline_tuple_spec_round_trips(tmp_path, monkeypatch):
    """A spec value that JSON turns into a list (tuple) must still self-match on
    lookup — the artifact must not reject its own emitted config."""

    def build(N, tile, BLOCK=0):
        return lambda a, out, stream=None: out._data.__setitem__(0, float(BLOCK))

    def specialize(a, out, stream=None):
        return {"N": a.shape[-1], "tile": (16, 32)}

    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "s1"))
    mk = dict(
        name="op",
        build=build,
        specialize=specialize,
        configs=lambda N, tile: [Config(BLOCK=5)],
        default=lambda N, tile: Config(BLOCK=9),
        structural=("BLOCK",),
        warmup=1,
        rep=1,
        do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1],
    )
    autotune_builder(**mk)(FakeTensor((8, 512)), FakeTensor((1,)))
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "s2"))
    out = FakeTensor((1,))
    autotune_builder(**mk)(FakeTensor((8, 512)), out)
    assert out._data[0] == 5.0  # offline (5) served, not default (9)


def test_offline_force_bypasses_and_reemits(tmp_path, monkeypatch):
    """FLYDSL_AUTOTUNE=1 must ignore an existing offline artifact, re-benchmark,
    and re-emit — not short-circuit on the committed config."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "s1"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    t, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    t(FakeTensor((16, 512)), FakeTensor((1,)), dtype_str="bf16")  # emits artifact

    # Force again with a fresh scratch dir: must still benchmark (not serve offline).
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "s2"))
    t2, _ = _make_builder(do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1])
    n = {"bench": 0}
    orig = t2.tuner._bench_one
    t2.tuner._bench_one = lambda *a, **k: (n.__setitem__("bench", n["bench"] + 1), orig(*a, **k))[1]
    t2(FakeTensor((16, 512)), FakeTensor((1,)), dtype_str="bf16")
    assert n["bench"] > 0  # force bypassed the offline artifact


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
