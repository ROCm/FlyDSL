# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Coverage for the cache-key completeness work: env-var drift, nested jit/kernel
args, globals-drift detection, and the ir.Type fallback in ``_arg_cache_sig``."""

from __future__ import annotations

import inspect
import textwrap

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function
from flydsl.compiler.protocol import cache_signature


def _key(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._resolve_and_make_cache_key(bound.arguments)


def test_env_var_drift_changes_cache_key(monkeypatch):
    @flyc.jit
    def k(A: fx.Tensor):
        pass

    A = torch.zeros(8, dtype=torch.float32)

    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "2")
    k1 = _key(k, A)

    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "0")
    k2 = _key(k, A)

    assert k1 != k2, "cache key must change when whitelisted env var changes"


def test_globals_drift_default_raises(tmp_path, monkeypatch):
    """mutating a referenced global must raise."""
    src = tmp_path / "drift_default.py"
    src.write_text(
        "import flydsl.compiler as flyc\n"
        "import flydsl.expr as fx\n"
        "FOO = 1\n"
        "@flyc.jit\n"
        "def k(A: fx.Tensor):\n"
        "    _ = FOO\n"
        "    return A\n"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("drift_default", src)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("COMPILE_ONLY", "1")  # auto-restored, no leak across tests
    spec.loader.exec_module(mod)
    A = torch.zeros(8, dtype=torch.float32)
    mod.k(A)
    mod.FOO = 2
    with pytest.raises(RuntimeError, match="FOO"):
        mod.k(A)


def test_env_var_not_cached_within_process(monkeypatch):
    """Regression: env var lookup must re-read os.environ on every call.

    A previous implementation wrapped _cache_invalidating_env_values in
    lru_cache(maxsize=1), which froze the first observed values into every
    subsequent cache key — flipping FLYDSL_COMPILE_OPT_LEVEL mid-process
    produced a silent stale-hit.
    """

    @flyc.jit
    def k(A: fx.Tensor):
        pass

    A = torch.zeros(8, dtype=torch.float32)

    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "2")
    k1 = _key(k, A)
    monkeypatch.setenv("FLYDSL_COMPILE_OPT_LEVEL", "0")
    k2 = _key(k, A)
    assert k1 != k2


def test_cache_key_is_device_independent():
    """The cache key's target is arch-only, with no device id component.

    The compiled kernel binary is device-independent, so a single in-process
    artifact / func_exe is shared across same-arch GPUs (as on main). Folding a
    device id into the key would needlessly split the cache per device.
    """
    from flydsl.compiler.backends import GPUTarget, get_backend

    @flyc.jit
    def k(A: fx.Tensor):
        pass

    A = torch.zeros(8, dtype=torch.float32)
    k._ensure_sig()
    key1 = _key(k, A)
    target_entry = next(v for n, v in key1 if n == "_target_")
    assert isinstance(target_entry, GPUTarget)
    assert target_entry == get_backend().target


def test_globals_snapshot_folded_into_cache_key_default_mode():
    """Regression: the stable globals snapshot must be folded into the cache key
    so two processes sharing a disk cache but observing different global values
    cannot collide.

    Each process is a fresh JitFunction instance that snapshots the global value
    it observes on its first call (the snapshot is then memoized — within one
    process a later change raises via drift detection rather than re-keying). So
    cross-process divergence is modelled here by two independent module loads
    with different ``FOO`` values, not by mutating one instance in place.
    """
    import importlib.util
    import tempfile
    import textwrap

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(textwrap.dedent("""
                import flydsl.compiler as flyc
                import flydsl.expr as fx

                FOO = 7

                @flyc.jit
                def k(A: fx.Tensor):
                    # reference FOO so it lands in co_names and the snapshot
                    _ = FOO
                """))
        path = f.name

    def _full_key(jit_fn, *args):
        jit_fn._ensure_sig()
        bound = jit_fn._sig.bind(*args)
        bound.apply_defaults()
        return jit_fn._build_full_cache_key(bound.arguments)

    def _load_with_foo(name, foo):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.FOO = foo  # set before first call, as a second process would observe it
        return mod

    A = torch.zeros(8, dtype=torch.float32)
    # Two independent "processes" observing FOO=7 vs FOO=99 on their first call.
    k1 = _full_key(_load_with_foo("_gkmod_a", 7).k, A)
    k2 = _full_key(_load_with_foo("_gkmod_b", 99).k, A)
    assert k1 != k2, "stable globals snapshot must be in cache key in default mode"


def test_cache_signature_requires_protocol_method():
    """Types lacking __cache_signature__ must raise — no silent type-only collapse.

    With __cache_signature__ promoted to a required JitArgument protocol method,
    cache_signature() no longer falls back to str(ir.Type). Unknown leaf objects
    bottom out at the Constexpr encoder, which only accepts the supported scalar
    shapes, so arbitrary instances raise instead of colliding under a shared key.
    """

    class _NoSig:
        pass

    with pytest.raises(TypeError, match="__cache_signature__"):
        cache_signature(_NoSig())


def _load_mod(tmp_path, name, body):
    import importlib.util

    src = tmp_path / f"{name}.py"
    src.write_text(textwrap.dedent(body))
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reset_manager_key(jit_fn):
    jit_fn.manager_key = None
    jit_fn._manager_owner_cls = None
    jit_fn.cache_manager = None


def _manager_key(jit_fn, *, reset=False):
    if reset:
        _reset_manager_key(jit_fn)
    jit_fn._ensure_cache_manager()
    return jit_fn.manager_key


def test_container_global_folded_by_value_not_collapsed(tmp_path):
    """A tuple/list/dict global must be folded into the key by *content*, not
    collapsed to ('obj', <type>). Two independent loads (≈ two processes)
    observing CFG=(128, 64) vs CFG=(128, 32) must produce different keys."""
    body = (
        "import flydsl.compiler as flyc\n"
        "import flydsl.expr as fx\n"
        "CFG = (128, 64)\n"
        "@flyc.jit\n"
        "def k(A: fx.Tensor):\n"
        "    _ = CFG\n"
        "    return A\n"
    )

    def _full_key(mod):
        mod.k._ensure_sig()
        bound = mod.k._sig.bind(torch.zeros(8, dtype=torch.float32))
        bound.apply_defaults()
        return mod.k._build_full_cache_key(bound.arguments)

    k1 = _full_key(_load_mod(tmp_path, "cfg_a", body))
    k2 = _full_key(_load_mod(tmp_path, "cfg_b", body.replace("(128, 64)", "(128, 32)")))
    assert k1 != k2, "container global contents must participate in the cache key"


def test_dict_global_inplace_mutation_raises(tmp_path, monkeypatch):
    """In-place mutation of a referenced dict/list global must be detected by
    drift (value-based snapshot), not silently reuse the old artifact."""
    mod = _load_mod(
        tmp_path,
        "cfg_mut",
        "import flydsl.compiler as flyc\n"
        "import flydsl.expr as fx\n"
        "CFG = {'tile': 64}\n"
        "@flyc.jit\n"
        "def k(A: fx.Tensor):\n"
        "    _ = CFG\n"
        "    return A\n",
    )
    monkeypatch.setenv("COMPILE_ONLY", "1")
    A = torch.zeros(8, dtype=torch.float32)
    mod.k(A)
    mod.CFG["tile"] = 128  # in-place mutation (same object id)
    with pytest.raises(RuntimeError, match="CFG"):
        mod.k(A)


def test_drift_baseline_is_per_owner_cls():
    """A JIT method reused across owner classes must drift-check each class
    against its own baseline, not the first owner's. Otherwise a global seen only
    under the second owner is skipped (not in the first owner's baseline) and a
    later mutation silently reuses the memoized key segment instead of raising."""
    from flydsl.compiler.jit_function import _snapshot_refs

    @flyc.jit
    def launch(A: fx.Tensor):
        return A

    # Two owner classes whose discovered refs read different module globals.
    g = {"__name__": "m", "FOO": 1, "BAR": 1}

    class A:
        pass

    class B:
        pass

    launch._global_refs_cache[A] = [("FOO", "m", g)]
    launch._global_refs_cache[B] = [("BAR", "m", g)]
    launch._used_global_vals[A] = _snapshot_refs(launch._global_refs_cache[A], stable=False)
    launch._used_global_vals[B] = _snapshot_refs(launch._global_refs_cache[B], stable=False)

    g["BAR"] = 2  # mutate a global seen only under owner B

    launch._check_globals_drift(A)  # A's baseline (FOO) unchanged → no raise
    with pytest.raises(RuntimeError, match="BAR"):
        launch._check_globals_drift(B)  # B's own baseline catches the BAR drift


def test_top_level_launch_named_jit_does_not_recurse_forever(tmp_path, monkeypatch):
    """A top-level ``@flyc.jit`` named ``launch`` whose body calls
    ``kernel(...).launch(...)`` must not blow the stack while building its key.
    """
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")
    mod = _load_mod(
        tmp_path,
        "launch_attr_collision",
        """
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        @flyc.kernel
        def my_kernel(C: fx.Tensor):
            pass

        @flyc.jit
        def launch(C: fx.Tensor):
            my_kernel(C).launch(grid=(1, 1, 1), block=[1, 1, 1])
        """,
    )

    key1 = _manager_key(mod.launch, reset=True)  # must not raise RecursionError
    key2 = _manager_key(mod.launch)
    dep_sources = jit_function._collect_dependency_sources(mod.launch.func, inspect.getfile(mod.launch.func))

    assert key1 == key2
    assert any("my_kernel" in source for source in dep_sources)


def test_jit_self_reference_does_not_recurse_forever(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")
    mod = _load_mod(
        tmp_path,
        "jit_self_ref",
        """
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        @flyc.jit
        def solo(A: fx.Tensor):
            _ = solo
            return A
        """,
    )

    key1 = _manager_key(mod.solo, reset=True)
    key2 = _manager_key(mod.solo)

    assert key1 == key2


def test_mutually_referential_jits_do_not_recurse_forever(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")
    mod = _load_mod(
        tmp_path,
        "jit_mutual_ref",
        """
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        @flyc.jit
        def first(A: fx.Tensor):
            _ = second
            return A

        @flyc.jit
        def second(A: fx.Tensor):
            _ = first
            return A
        """,
    )

    _reset_manager_key(mod.first)
    _reset_manager_key(mod.second)
    first_key = _manager_key(mod.first)
    second_key = _manager_key(mod.second)

    assert first_key == _manager_key(mod.first)
    assert second_key == _manager_key(mod.second)


def test_in_progress_stack_is_thread_local():
    """Each thread must get its own in-progress set."""
    import threading

    main_stack = jit_function._keys_in_progress()
    assert jit_function._keys_in_progress() is main_stack

    other = {}

    def worker():
        other["stack"] = jit_function._keys_in_progress()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert other["stack"] is not main_stack


def test_key_computation_leaves_no_in_progress_leak(tmp_path, monkeypatch):
    """The in-progress stack must be empty again after keying."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")
    mod = _load_mod(
        tmp_path,
        "jit_no_leak",
        """
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        @flyc.jit
        def solo(A: fx.Tensor):
            _ = solo
            return A
        """,
    )

    _manager_key(mod.solo, reset=True)
    assert jit_function._keys_in_progress() == set()
