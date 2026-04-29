# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import JitArgumentRegistry


class _FakeCudaStream:
    cuda_stream = 1234


JitArgumentRegistry.register(_FakeCudaStream)(fx.Stream)


@flyc.jit
def _stream_launch(stream: fx.Stream = fx.Stream(None)):
    pass


@flyc.jit
def _constexpr_launch(value: fx.Constexpr[int]):
    pass


def _cache_key(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._make_cache_key(bound.arguments)


def test_stream_cache_key_ignores_runtime_representation():
    """CPU AOT can use raw 0 while GPU runtime passes a stream object."""
    keys = [
        _cache_key(_stream_launch),
        _cache_key(_stream_launch, 0),
        _cache_key(_stream_launch, fx.Stream(0)),
        _cache_key(_stream_launch, _FakeCudaStream()),
    ]

    assert keys[0] == keys[1] == keys[2] == keys[3]
    assert ("stream", fx.Stream) in keys[0]


def test_constexpr_values_still_participate_in_cache_key():
    assert _cache_key(_constexpr_launch, 1) != _cache_key(_constexpr_launch, 2)
