# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import JitArgumentRegistry
from flydsl.compiler.kernel_function import CompilationContext


class _FakeCudaStream:
    cuda_stream = 1234


JitArgumentRegistry.register(_FakeCudaStream)(fx.Stream)


@flyc.jit
def _stream_launch(stream: fx.Stream = fx.Stream(None)):
    pass


@flyc.jit
def _constexpr_launch(value: fx.Constexpr[int]):
    pass


@flyc.jit
def _runtime_int32_launch(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    pass


def _cache_key(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._resolve_and_make_cache_key(bound.arguments)


def test_stream_cache_key_ignores_runtime_representation():
    """CPU AOT can use raw 0 while GPU runtime passes a stream object."""
    keys = [
        _cache_key(_stream_launch),
        _cache_key(_stream_launch, 0),
        _cache_key(_stream_launch, fx.Stream(0)),
        _cache_key(_stream_launch, _FakeCudaStream()),
    ]

    assert keys[0] == keys[1] == keys[2] == keys[3]
    assert ("stream", (fx.Stream,)) in keys[0]


def test_constexpr_values_still_participate_in_cache_key():
    assert _cache_key(_constexpr_launch, 1) != _cache_key(_constexpr_launch, 2)


def test_future_annotations_runtime_int32_ignores_value_in_cache_key():
    """`from __future__ import annotations` stringifies fx.Int32; resolve_signature must eval it back so the value stays out of the cache key."""
    key1 = _cache_key(_runtime_int32_launch, 1)
    key2 = _cache_key(_runtime_int32_launch, 2)

    assert key1 == key2
    assert ("n", (fx.Int32,)) in key1
    assert ("n", (int, 1)) not in key1


def test_thread_local_compile_options_enter_cache_key_before_build():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    # The same backend-option path is public outside autotune.
    flyc.compile[{"fast_fp_math": True, "waves_per_eu": 4}](launch)
    baseline = _cache_key(launch)
    with CompilationContext.compile_hints({"waves_per_eu": 1}):
        wpe1 = _cache_key(launch)
    with CompilationContext.compile_hints({"waves_per_eu": 2}):
        wpe2 = _cache_key(launch)
    with CompilationContext.compile_hints({"waves_per_eu": "2"}):
        invalid_string = _cache_key(launch)

    assert len({baseline, wpe1, wpe2, invalid_string}) == 4
    hints = dict(next(value for name, value in wpe2 if name == "_hints_"))
    assert hints["fast_fp_math"] == (bool, "True")
    assert hints["waves_per_eu"] == (int, "2")  # thread-local candidate wins


def test_mapping_compile_options_have_canonical_cache_keys():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    with CompilationContext.compile_hints(
        {"waves_per_eu": {"kernel_a": 2, "kernel_b": 4}, "llvm_options": {"x": 1, "y": 2}}
    ):
        ordered = _cache_key(launch)
    with CompilationContext.compile_hints(
        {"llvm_options": {"y": 2, "x": 1}, "waves_per_eu": {"kernel_b": 4, "kernel_a": 2}}
    ):
        reversed_order = _cache_key(launch)

    assert ordered == reversed_order


def test_compile_hint_snapshot_couples_cache_key_to_compilation_options():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    launch.compile_hints = {"waves_per_eu": {"kernel": 1}}
    snapshot = launch._effective_compile_hints()
    launch.compile_hints["waves_per_eu"]["kernel"] = 2

    launch._ensure_sig()
    bound = launch._sig.bind()
    bound.apply_defaults()
    snapshot_key = launch._resolve_and_make_cache_key(bound.arguments, effective_hints=snapshot)

    launch.compile_hints = {"waves_per_eu": {"kernel": 1}}
    expected_key = _cache_key(launch)
    launch.compile_hints = {"waves_per_eu": {"kernel": 2}}
    mutated_key = _cache_key(launch)

    assert snapshot == {"waves_per_eu": {"kernel": 1}}
    assert snapshot_key == expected_key
    assert snapshot_key != mutated_key
