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
from contextlib import contextmanager

import pytest

from flydsl.autotune import (
    Autotuner,
    Config,
    _bench_batch_sizes,
    _normalize_strides,
    autotune,
    do_bench,
)


@pytest.fixture(autouse=True)
def _isolate_disk_cache(tmp_path, monkeypatch):
    """Every test gets a private autotune disk cache so results don't leak
    across tests (a cached best-config would skip the benchmark loop)."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "autotune_cache"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)


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


class _FakeBenchEvent:
    def __init__(self, cuda, enable_timing):
        assert enable_timing
        self.cuda = cuda
        self.timestamp = None

    def record(self):
        self.timestamp = self.cuda.clock
        self.cuda.operations.append("event")

    def synchronize(self):
        self.cuda.event_synchronizes += 1

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class _FakeBenchCuda:
    def __init__(self):
        self.clock = 0.0
        self.operations = []
        self.device_synchronizes = 0
        self.event_synchronizes = 0

    def is_available(self):
        return True

    def synchronize(self):
        self.device_synchronizes += 1

    def _sleep(self, cycles):
        assert cycles > 0
        self.operations.append("backlog")
        self.clock += 100.0

    def Event(self, enable_timing):
        return _FakeBenchEvent(self, enable_timing)


class _FakeBenchTorch:
    def __init__(self):
        self.cuda = _FakeBenchCuda()


# ── device benchmark timing ─────────────────────────────────────────────
def test_bench_batch_sizes_preserve_the_requested_call_count():
    assert _bench_batch_sizes(1) == [1]
    assert _bench_batch_sizes(7) == [2, 2, 1, 1, 1]
    assert sum(_bench_batch_sizes(25)) == 25
    with pytest.raises(ValueError, match="positive integer"):
        _bench_batch_sizes(0)


def test_do_bench_uses_a_backlogged_batched_event_window(monkeypatch):
    """The timer must enqueue a GPU backlog before each event window and must
    synchronize per batch, not per kernel launch."""
    import importlib

    at = importlib.import_module("flydsl.autotune")
    fake_torch = _FakeBenchTorch()
    monkeypatch.setattr(at, "torch", fake_torch)
    calls = 0

    def fn():
        nonlocal calls
        calls += 1
        fake_torch.cuda.operations.append("kernel")
        fake_torch.cuda.clock += 2.0

    assert do_bench(fn, warmup=3, rep=7) == 2.0
    assert calls == 10
    assert fake_torch.cuda.device_synchronizes == 1
    assert fake_torch.cuda.event_synchronizes == 5

    timed = fake_torch.cuda.operations[3:]
    assert timed == [
        "backlog",
        "event",
        "kernel",
        "kernel",
        "event",
        "backlog",
        "event",
        "kernel",
        "kernel",
        "event",
        "backlog",
        "event",
        "kernel",
        "event",
        "backlog",
        "event",
        "kernel",
        "event",
        "backlog",
        "event",
        "kernel",
        "event",
    ]


def test_do_bench_quantiles_summarize_batch_averages(monkeypatch):
    import importlib

    at = importlib.import_module("flydsl.autotune")
    fake_torch = _FakeBenchTorch()
    monkeypatch.setattr(at, "torch", fake_torch)

    def fn():
        fake_torch.cuda.clock += 3.0

    assert do_bench(fn, warmup=0, rep=5, quantiles=[0.0, 0.5, 0.9]) == [3.0, 3.0, 3.0]


def test_do_bench_fails_closed_without_a_gpu_backlog(monkeypatch):
    import importlib

    at = importlib.import_module("flydsl.autotune")
    fake_torch = _FakeBenchTorch()
    fake_torch.cuda._sleep = None
    monkeypatch.setattr(at, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="GPU-side backlog"):
        do_bench(lambda: None)


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
    assert _normalize_strides(FakeTensor((4, 8))) == (
        "s",
        1,
    )  # contiguous: inner=1, outer=other
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


def test_key_varies_with_effective_compile_hints():
    hints = {"waves_per_eu": 1}

    def fn(a, out, **kw):
        pass

    fn._effective_compile_hints = lambda: dict(hints)
    tuner = _make_tuner(fn=fn)
    args = (FakeTensor((8, 8)), FakeTensor((8, 8)))
    first = tuner._make_key(args, {})
    hints["waves_per_eu"] = 2

    assert tuner._make_key(args, {}) != first


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
        do_bench_fn=lambda call, warmup, rep: (
            [call() for _ in range(warmup + rep)],
            1.0,
        )[1],
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
        do_bench_fn=lambda call, warmup, rep: (
            [call() for _ in range(warmup + rep)],
            1.0,
        )[1],
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
        do_bench_fn=lambda call, warmup, rep: (
            [call() for _ in range(warmup + rep)],
            1.0,
        )[1],
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
    """@autotune returns an Autotuner and forwards its options."""

    def fake_jit(a, out, **kw):
        pass

    def configs(a, out):
        return [Config(BLOCK=128)]

    def default(a, out):
        return Config(BLOCK=64)

    tuned = autotune(
        configs=configs,
        key=["a"],
        default=default,
        artifact_name="fake-kernel",
        restore_value=["a"],
        reset_to_zero=["out"],
    )(fake_jit)

    assert isinstance(tuned, Autotuner)
    assert tuned.restore_value == ["a"]
    assert tuned.reset_to_zero == ["out"]
    assert tuned.configs is configs
    assert tuned.default is default
    assert tuned.artifact_name == "fake-kernel"


def test_resolve_config_selects_without_launching(monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    launches = []
    tuner = _make_tuner(
        fn=lambda a, out, BLOCK: launches.append(BLOCK),
        configs=[Config(BLOCK=128)],
        default=lambda *_args, **_kwargs: Config(BLOCK=64),
        do_bench_fn=lambda *_args, **_kwargs: pytest.fail("default resolution benchmarked"),
    )
    args = (FakeTensor((8,)), FakeTensor((8,)))

    config = tuner.resolve_config(*args)

    assert config.kwargs["BLOCK"] == 64
    assert launches == []


def test_call_runs_exactly_the_config_returned_by_resolve(monkeypatch):
    tuner = _make_tuner(fn=lambda a, out, BLOCK: None)
    selected = Config(BLOCK=777)
    observed = []
    monkeypatch.setattr(tuner, "resolve_config", lambda *args, **kwargs: selected)
    monkeypatch.setattr(
        tuner,
        "_run_config",
        lambda config, args, kwargs: observed.append((config, args, kwargs)),
    )
    args = (FakeTensor((8,)), FakeTensor((8,)))

    tuner(*args)

    assert observed == [(selected, args, {})]


def test_select_config_policy_must_return_a_measured_pair(monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    first = Config(BLOCK=64)
    second = Config(BLOCK=128)
    times = iter([2.0, 1.0])
    tuner = _make_tuner(
        configs=[first, second],
        do_bench_fn=lambda call, warmup, rep: next(times),
        select_config=lambda results: results[0],
    )
    args = (FakeTensor((8,)), FakeTensor((8,)))

    assert tuner.resolve_config(*args) is first

    invalid = _make_tuner(
        configs=[first],
        do_bench_fn=lambda call, warmup, rep: 1.0,
        select_config=lambda results: (Config(BLOCK=999), 1.0),
    )
    with pytest.raises(ValueError, match="measured"):
        invalid.resolve_config(*args)


# ── two-track default/search ─────────────────────────────────────────────
def test_cache_hit_precedes_default_and_search(monkeypatch):
    # The broad test runner uses the explicit off value; search stays opt-in.
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    default_calls = 0

    def fn(a, out, BLOCK):
        out._data[0] = float(BLOCK)

    def default(a, out):
        nonlocal default_calls
        default_calls += 1
        return Config(BLOCK=999)

    def fail_bench(call, warmup, rep):
        pytest.fail("normal path benchmarked configs")

    tuner = _make_tuner(
        fn=fn,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        default=default,
        do_bench_fn=fail_bench,
    )
    a = FakeTensor((8,))
    out = FakeTensor((1,))
    args = (a, out)
    tuner.cache[tuner._make_key(args, {})] = Config(BLOCK=128)

    tuner(*args)

    assert out._data[0] == 128.0
    assert default_calls == 0

    tuner.cache.clear()
    tuner(*args)

    assert out._data[0] == 999.0
    assert default_calls == 1


def test_force_search_bypasses_cache_and_default(monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    calls = {"configs": 0, "default": 0, "bench": 0}

    def fn(a, out, BLOCK):
        out._data[0] = float(BLOCK)

    def configs(a, out):
        calls["configs"] += 1
        return [Config(BLOCK=64), Config(BLOCK=128)]

    def default(a, out):
        calls["default"] += 1
        return Config(BLOCK=7)

    def bench(call, warmup, rep):
        calls["bench"] += 1
        call()
        return float(calls["bench"])

    tuner = _make_tuner(fn=fn, configs=configs, default=default, do_bench_fn=bench)
    args = (FakeTensor((8,)), FakeTensor((1,)))
    tuner.cache[tuner._make_key(args, {})] = Config(BLOCK=999)

    tuner(*args)

    assert calls == {"configs": 1, "default": 0, "bench": 2}
    assert args[1]._data[0] == 64.0


# ── offline config artifacts ────────────────────────────────────────────
class FakeConstexprInt:
    @classmethod
    def __coerce__(cls, value):
        if type(value) is not int:
            raise TypeError(f"expects int, got {type(value).__name__}")
        return value


def _artifact_kernel(a, out, m_in, N, dtype_str, BLOCK_THREADS: FakeConstexprInt, stream: int = 0):
    out._data[0] = float(BLOCK_THREADS)


def _artifact_default(a, out, m_in, N, dtype_str):
    return Config(BLOCK_THREADS=7)


def _make_artifact_tuner(**overrides):
    options = {
        "fn": _artifact_kernel,
        "configs": [Config(BLOCK_THREADS=64)],
        "key": ["m_in", "N", "dtype_str"],
        "default": _artifact_default,
        "artifact_name": "rmsnorm",
        "do_bench_fn": lambda call, warmup, rep: (call(), 1.0)[1],
    }
    options.update(overrides)
    return _make_tuner(**options)


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    import importlib

    at = importlib.import_module("flydsl.autotune")

    path = tmp_path / "artifacts"
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CONFIG_DIR", str(path))
    monkeypatch.setattr(
        at,
        "_device_descriptor",
        lambda device=None: {
            "name": "Test GPU",
            "arch": "gfx-test",
            "compute_units": 1,
        },
        raising=False,
    )
    return path


def _emit_artifact(monkeypatch, artifact_dir, *, args=None, config=None):
    args = args or (FakeTensor((16, 512)), FakeTensor((1,)), 16, 512, "bf16")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    tuner = _make_artifact_tuner(configs=[config or Config(BLOCK_THREADS=64)])
    tuner(*args)
    files = list(artifact_dir.glob("*.json"))
    assert len(files) == 1
    return files[0], args


def test_artifact_lifecycle(monkeypatch, tmp_path, artifact_dir):
    path, args = _emit_artifact(monkeypatch, artifact_dir)
    payload = json.loads(path.read_text())

    assert payload["identity"]["key"] == {"m_in": 16, "N": 512, "dtype_str": "bf16"}
    assert payload["config"] == {"BLOCK_THREADS": 64}

    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "fresh-cache"))

    def fail_configs(*_args, **_kwargs):
        pytest.fail("artifact lookup evaluated the search space")

    def fail_default(*_args, **_kwargs):
        pytest.fail("artifact lookup evaluated the heuristic default")

    loaded = _make_artifact_tuner(
        configs=fail_configs,
        default=fail_default,
        do_bench_fn=lambda *args, **kwargs: pytest.fail("artifact lookup benchmarked"),
    )
    out = FakeTensor((1,))
    loaded(args[0], out, *args[2:])

    assert out._data[0] == 64.0
    assert loaded.cache == {}, "artifact decisions must not enter the searched-winner cache"

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    replacement = _make_artifact_tuner(configs=[Config(BLOCK_THREADS=128)])
    replacement(*args)
    assert json.loads(path.read_text())["config"] == {"BLOCK_THREADS": 128}


def test_artifact_schema_partitions_searched_winner_cache(monkeypatch, artifact_dir):
    args = (FakeTensor((16, 512)), FakeTensor((1,)), 16, 512, "bf16")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    old_schema = _make_artifact_tuner(
        artifact_name="rmsnorm-v1",
        configs=[Config(BLOCK_THREADS=64)],
    )
    old_schema(*args)

    new_schema = _make_artifact_tuner(artifact_name="rmsnorm-v2")
    ref = new_schema._artifact_ref(args, {}, required=True)
    new_schema._emit_artifact(Config(BLOCK_THREADS=128), ref, args, {})

    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    loaded = _make_artifact_tuner(artifact_name="rmsnorm-v2")
    out = FakeTensor((1,))
    loaded(args[0], out, *args[2:])

    assert out._data[0] == 128.0


def test_artifact_identity_uses_declared_key_and_device(artifact_dir, monkeypatch):
    import importlib

    at = importlib.import_module("flydsl.autotune")

    def kernel(a, out, N, BLOCK_THREADS: int, dtype_str: str = "bf16"):
        pass

    tuner = _make_tuner(fn=kernel, key=["N", "dtype_str"], artifact_name="defaults")
    args = (FakeTensor((8, 512)), FakeTensor((1,)), 512)
    first = tuner._artifact_ref(args, {}, required=True)
    scratch_key = tuner._make_key(args, {})

    assert first == tuner._artifact_ref(args, {"dtype_str": "bf16"}, required=True)
    assert first != tuner._artifact_ref((args[0], args[1], 1024), {}, required=True)

    monkeypatch.setattr(
        at,
        "_device_descriptor",
        lambda device=None: {
            "name": "Other GPU",
            "arch": "gfx-test",
            "compute_units": 2,
        },
    )
    assert first != tuner._artifact_ref(args, {}, required=True)
    assert scratch_key != tuner._make_key(args, {})


def test_cached_artifact_is_revalidated_for_each_call(monkeypatch, tmp_path, artifact_dir):
    _, args = _emit_artifact(monkeypatch, artifact_dir)
    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "fresh-cache"))
    tuner = _make_artifact_tuner()
    ref = tuner._artifact_ref(args, {}, required=True)

    assert tuner._load_artifact(ref, args, {}).to_dict() == {"BLOCK_THREADS": 64}
    assert tuner._load_artifact(ref, args, {"BLOCK_THREADS": 32}) is None
    assert tuner._load_artifact(ref, args, {}).to_dict() == {"BLOCK_THREADS": 64}


@pytest.mark.parametrize("case", ["corrupt", "version", "identity", "config", "override", "type"])
def test_invalid_artifact_falls_back(monkeypatch, tmp_path, artifact_dir, case):
    path, args = _emit_artifact(monkeypatch, artifact_dir)
    payload = json.loads(path.read_text())
    if case == "corrupt":
        path.write_text("{")
    else:
        if case == "version":
            payload["version"] = 2
        elif case == "identity":
            payload["identity"]["key"]["N"] = 1024
        elif case == "config":
            payload["config"] = {"UNKNOWN": 64}
        elif case == "type":
            payload["config"] = {"BLOCK_THREADS": "64"}
        else:
            payload["config"] = {"BLOCK_THREADS": 64, "N": 1024}
        path.write_text(json.dumps(payload))
    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "fresh-cache"))

    tuner = _make_artifact_tuner(
        configs=lambda *_args, **_kwargs: pytest.fail("invalid artifact triggered search"),
        do_bench_fn=lambda *args, **kwargs: pytest.fail("invalid artifact benchmarked"),
    )
    out = FakeTensor((1,))
    tuner(args[0], out, *args[2:])

    assert out._data[0] == 7.0


def test_artifact_runtime_failure_is_not_masked_by_default(monkeypatch, tmp_path, artifact_dir):
    _, args = _emit_artifact(monkeypatch, artifact_dir)
    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "fresh-cache"))
    seen = []

    def failing_kernel(a, out, m_in, N, dtype_str, BLOCK_THREADS: int):
        seen.append(BLOCK_THREADS)
        if BLOCK_THREADS == 64:
            raise RuntimeError("kernel failed")
        out._data[0] = float(BLOCK_THREADS)

    tuner = _make_artifact_tuner(fn=failing_kernel)

    with pytest.raises(RuntimeError, match="kernel failed"):
        tuner(*args)

    assert seen == [64]


@pytest.mark.parametrize(
    "config, message",
    [
        pytest.param(
            Config(BLOCK_THREADS=64, pre_hook=lambda kwargs: None),
            "pre_hook",
            id="pre-hook",
        ),
        pytest.param(Config(BLOCK_THREADS=(64,)), "preserve their types", id="json-type-change"),
    ],
)
def test_unpersistable_artifact_config_blocks_generation(monkeypatch, artifact_dir, config, message):
    def kernel(a, out, m_in, N, dtype_str, BLOCK_THREADS):
        pass

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    tuner = _make_artifact_tuner(fn=kernel, configs=[config])
    args = (FakeTensor((16, 512)), FakeTensor((1,)), 16, 512, "bf16")

    with pytest.raises(ValueError, match=message):
        tuner(*args)

    assert tuner.cache == {}
    assert not tuner._cache_file.exists()
    assert not list(artifact_dir.glob("*.json"))


def test_unavailable_device_identity_falls_back_but_blocks_generation(monkeypatch, artifact_dir):
    import importlib

    at = importlib.import_module("flydsl.autotune")
    monkeypatch.setattr(at, "_device_descriptor", lambda device=None: None)
    args = (FakeTensor((16, 512)), FakeTensor((1,)), 16, 512, "bf16")
    tuner = _make_artifact_tuner()

    tuner(*args)
    assert args[1]._data[0] == 7.0

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    with pytest.raises(ValueError, match="cannot generate offline config identity"):
        tuner(*args)


@pytest.mark.parametrize("name", ["../rmsnorm", 123])
def test_artifact_name_must_be_safe(name):
    with pytest.raises((TypeError, ValueError), match="artifact_name"):
        _make_artifact_tuner(artifact_name=name)


def test_artifact_key_must_name_a_kernel_parameter():
    with pytest.raises(ValueError, match="artifact keys"):
        _make_artifact_tuner(key=["mni"])


# ── return-value propagation ─────────────────────────────────────────────
def test_call_returns_tuned_fn_value(monkeypatch):
    """The tuned function's return value must propagate through __call__ on both
    the search path and the cache-hit path."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")  # force the search path first

    def fn(a, out, BLOCK):
        return ("result", BLOCK)

    tuner = _make_tuner(
        fn=fn,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1],
    )
    args = (FakeTensor((8,)), FakeTensor((1,)))

    searched = tuner(*args)  # search path -> runs the winning config
    assert searched == ("result", 64)

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    hit = tuner(*args)  # cache-hit path
    assert hit == ("result", 64)


def test_call_returns_none_when_fn_returns_none(monkeypatch):
    """In-place kernels (write into `out`, return nothing) still return None --
    the fix only forwards whatever the tuned fn returns, it doesn't fabricate."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")

    def fn(a, out, BLOCK):
        out._data[0] = float(BLOCK)  # writes in place, returns None

    tuner = _make_tuner(
        fn=fn,
        configs=[Config(BLOCK=64)],
        do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1],
    )
    args = (FakeTensor((8,)), FakeTensor((1,)))
    assert tuner(*args) is None
    assert args[1]._data[0] == 64.0


# ── validate_hook (untimed candidate correctness gate) ───────────────────
def test_validate_hook_runs_once_per_candidate_outside_the_timed_reps(monkeypatch):
    """Validation setup and checking must run once per candidate, but neither may
    be folded into the timed repetitions."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    runs = []

    def fn(a, out, BLOCK):
        runs.append(("run", BLOCK))

    @contextmanager
    def validate(sig_args):
        runs.append(("prepare", sig_args["BLOCK"]))
        yield
        runs.append(("validate", sig_args["BLOCK"]))
        # The hook sees positional args by name, which pre_hook/post_hook cannot.
        assert sig_args["a"].shape == (8,)

    def bench(call, warmup, rep):
        runs.append(("bench", warmup, rep))
        for _ in range(warmup + rep):
            call()
        return 1.0

    tuner = _make_tuner(
        fn=fn,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        validate_hook=validate,
        do_bench_fn=bench,
        warmup=1,
        rep=1,
    )
    tuner(FakeTensor((8,)), FakeTensor((8,)))

    # Per candidate: untimed setup/run/check, then the timed repetitions.
    assert runs[:6] == [
        ("prepare", 64),
        ("run", 64),
        ("validate", 64),
        ("bench", 1, 1),
        ("run", 64),
        ("run", 64),
    ]
    assert runs[6:12] == [
        ("prepare", 128),
        ("run", 128),
        ("validate", 128),
        ("bench", 1, 1),
        ("run", 128),
        ("run", 128),
    ]


def test_validate_hook_receives_defaulted_and_variadic_arguments(monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    observed = []

    def fn(a, out, *extra, BLOCK, scale=2.0, stream=None, **metadata):
        pass

    @contextmanager
    def validate(sig_args):
        observed.append(sig_args)
        yield

    tuner = _make_tuner(
        fn=fn,
        configs=[Config(BLOCK=64)],
        validate_hook=validate,
        do_bench_fn=lambda call, warmup, rep: 1.0,
    )
    a = FakeTensor((8,))
    out = FakeTensor((8,))
    marker = object()
    tuner(a, out, marker, stream="stream-0", tag="softmax")

    assert observed == [
        {
            "a": a,
            "out": out,
            "extra": (marker,),
            "BLOCK": 64,
            "scale": 2.0,
            "stream": "stream-0",
            "tag": "softmax",
        }
    ]


def test_validate_hook_rejects_a_candidate_that_does_not_write_output(monkeypatch):
    """A no-op candidate must not inherit a previous candidate's valid output."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    bench_calls = 0

    def fn(a, out, BLOCK):
        if BLOCK == 64:
            out._data = [1.0] * len(out._data)

    @contextmanager
    def validate(sig_args):
        sig_args["out"]._data = [float("nan")] * len(sig_args["out"]._data)
        yield
        if any(value != value for value in sig_args["out"]._data):
            raise ValueError("candidate left output elements unwritten")

    def bench(call, warmup, rep):
        nonlocal bench_calls
        bench_calls += 1
        call()
        return 1.0

    tuner = _make_tuner(
        fn=fn,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        validate_hook=validate,
        do_bench_fn=bench,
    )
    args = (FakeTensor((8,)), FakeTensor((8,), fill=1.0))
    tuner(*args)

    key = tuner._make_key(args, {})
    assert tuner.cache[key].kwargs["BLOCK"] == 64
    assert bench_calls == 1
    assert args[1]._data == [1.0] * 8


def test_validate_hook_rejection_drops_only_that_candidate(monkeypatch):
    """A candidate that launches but computes the wrong answer must not win."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")

    @contextmanager
    def validate(sig_args):
        yield
        if sig_args["BLOCK"] == 64:
            raise ValueError("wrong numerics")

    seen = []

    def fn(a, out, BLOCK):
        seen.append(BLOCK)

    # The rejected candidate is also the fastest, so it would win without the gate.
    def bench(call, warmup, rep):
        call()
        return {64: 0.1, 128: 1.0}[seen[-1]]

    tuner = _make_tuner(
        fn=fn,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        validate_hook=validate,
        do_bench_fn=bench,
    )
    args = (FakeTensor((8,)), FakeTensor((8,)))
    tuner(*args)
    key = tuner._make_key(args, {})
    assert tuner.cache[key].kwargs["BLOCK"] == 128


def test_all_candidates_failing_chains_the_underlying_error(monkeypatch):
    """Without the cause, a validation rejection is indistinguishable from a
    compile failure once the loop has swallowed every exception."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")

    @contextmanager
    def validate(sig_args):
        yield
        raise ValueError(f"bad numerics for BLOCK={sig_args['BLOCK']}")

    tuner = _make_tuner(
        fn=lambda a, out, BLOCK: None,
        configs=[Config(BLOCK=64), Config(BLOCK=128)],
        validate_hook=validate,
        do_bench_fn=lambda call, warmup, rep: 1.0,
    )
    with pytest.raises(RuntimeError, match="All autotune configs failed") as excinfo:
        tuner(FakeTensor((8,)), FakeTensor((8,)))
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "BLOCK=128" in str(excinfo.value.__cause__)


def test_no_validate_hook_leaves_the_timed_path_untouched(monkeypatch):
    """Adopters that do not pass a gate must see exactly the previous behavior."""
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    runs = []

    tuner = _make_tuner(
        fn=lambda a, out, BLOCK: runs.append(BLOCK),
        configs=[Config(BLOCK=64)],
        do_bench_fn=lambda call, warmup, rep: (call(), 1.0)[1],
    )
    tuner(FakeTensor((8,)), FakeTensor((8,)))
    assert runs == [64, 64]  # one bench call, one final _run_config


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
