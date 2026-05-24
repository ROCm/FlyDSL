from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def _future_runtime_launch(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    pass


def _cache_key(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._make_cache_key(bound.arguments)


def test_future_annotations_runtime_int32_ignores_value_in_cache_key():
    key1 = _cache_key(_future_runtime_launch, 1)
    key2 = _cache_key(_future_runtime_launch, 2)

    assert key1 == key2
    assert ("n", int) in key1
    assert ("n", (int, 1)) not in key1
    assert ("n", (int, 2)) not in key1
