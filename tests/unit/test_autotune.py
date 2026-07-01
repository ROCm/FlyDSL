# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU-free unit tests for flydsl.autotune.

Covers the parts that must be correct before any real kernel adopts @autotune:
  - Config serialization / kwargs / compiler-opts split
  - Cache-key axes: shape, dtype, stride pattern, device, toolchain, env
  - restore_value snapshot/restore (the in-place correctness soul)
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


def test_key_varies_with_toolchain_fingerprint():
    t = _make_tuner()
    a = FakeTensor((8, 8))
    k1 = t._make_key((a, a), {})
    t._toolchain_fp = "a-different-fingerprint"
    k2 = t._make_key((a, a), {})
    assert k1 != k2


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


# ── restore_value (correctness soul) ─────────────────────────────────────
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
