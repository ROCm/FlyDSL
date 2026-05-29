# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Coverage for the cache-key completeness work: env-var drift, nested jit/kernel
args, globals-drift detection, and the ir.Type fallback in ``_arg_cache_sig``."""

from __future__ import annotations

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
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


def test_device_id_in_cache_key_is_live():
    """Regression: cache key's device_id slot must reflect the current device,
    not a value frozen at first call.

    A previous implementation cached (target, device_id) inside _ensure_sig(),
    so switching device with torch.cuda.set_device() after the first launch
    would silently hit the other device's cached artifact.

    device_id is now resolved through the active DeviceRuntime (per-backend),
    queried live on every call.
    """
    from flydsl.runtime.device_runtime import get_device_runtime

    @flyc.jit
    def k(A: fx.Tensor):
        pass

    A = torch.zeros(8, dtype=torch.float32)
    k._ensure_sig()
    # Sanity check: target tuple's second slot is rebuilt per call, not cached.
    key1 = _key(k, A)
    target_entry = next(v for n, v in key1 if n == "_target_")
    assert target_entry[1] == get_device_runtime().current_device_id()


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
    src.write_text(body)
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
