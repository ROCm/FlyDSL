# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Coverage for the cache-key completeness work: env-var drift, nested jit/kernel
args, globals-drift detection, and the ir.Type fallback in ``_arg_cache_sig``."""

from __future__ import annotations

import os

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


def test_globals_drift_default_raises(tmp_path):
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
    os.environ["COMPILE_ONLY"] = "1"
    try:
        spec.loader.exec_module(mod)
        A = torch.zeros(8, dtype=torch.float32)
        mod.k(A)
        mod.FOO = 2
        with pytest.raises(RuntimeError, match="FOO"):
            mod.k(A)
    finally:
        os.environ.pop("COMPILE_ONLY", None)


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
        f.write(
            textwrap.dedent(
                """
                import flydsl.compiler as flyc
                import flydsl.expr as fx

                FOO = 7

                @flyc.jit
                def k(A: fx.Tensor):
                    # reference FOO so it lands in co_names and the snapshot
                    _ = FOO
                """
            )
        )
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
