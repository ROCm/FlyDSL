# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

from enum import IntEnum

import pytest

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
    with pytest.raises(TypeError, match="waves_per_eu"):
        with CompilationContext.compile_hints({"waves_per_eu": "2"}):
            _cache_key(launch)

    assert len({baseline, wpe1, wpe2}) == 3
    hints = dict(next(value for name, value in wpe2 if name == "_hints_"))
    assert hints["fast_fp_math"] == (bool, "True")
    assert hints["waves_per_eu"] == (int, "2")  # thread-local candidate wins


def test_compile_hint_zero_resets_persistent_occupancy_after_layer_merge():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    launch.compile_hints = {"fast_fp_math": True, "waves_per_eu": 4}

    with CompilationContext.compile_hints({"waves_per_eu": None}):
        inherited = launch._effective_compile_hints()
    with CompilationContext.compile_hints({"waves_per_eu": 0}):
        reset = launch._effective_compile_hints()
        reset_key = _cache_key(launch)

    launch.compile_hints = {"fast_fp_math": True}
    baseline_key = _cache_key(launch)

    assert inherited == {"fast_fp_math": True, "waves_per_eu": 4}
    assert reset == {"fast_fp_math": True}
    assert reset_key == baseline_key


def test_compile_hint_mapping_overlay_replaces_layer_before_zero_is_canonicalized():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    launch.compile_hints = {"waves_per_eu": {"kernel_a": 1, "kernel_b": 2}}
    with CompilationContext.compile_hints({"waves_per_eu": {"kernel_a": 0, "kernel_c": 3}}):
        effective = launch._effective_compile_hints()

    assert effective == {"waves_per_eu": {"kernel_c": 3}}


def test_nested_compile_hint_contexts_shallow_overlay_and_restore():
    with CompilationContext.compile_hints({"waves_per_eu": 2, "future_hint": {"outer": 1}}):
        with CompilationContext.compile_hints({"waves_per_eu": None, "maxnreg": 64}):
            assert CompilationContext.get_compile_hints() == {
                "waves_per_eu": 2,
                "maxnreg": 64,
                "future_hint": {"outer": 1},
            }
        with CompilationContext.compile_hints({"waves_per_eu": 0, "future_hint": {"inner": 2}}):
            assert CompilationContext.get_compile_hints() == {
                "waves_per_eu": 0,
                "future_hint": {"inner": 2},
            }
        assert CompilationContext.get_compile_hints() == {
            "waves_per_eu": 2,
            "future_hint": {"outer": 1},
        }

    assert CompilationContext.get_compile_hints() == {}


def test_fastmath_compile_hint_normalizes_supported_flag_containers():
    flags = {"reassoc", "contract"}
    with CompilationContext.compile_hints({"fastmath": flags}):
        assert CompilationContext.get_compile_hints() == {"fastmath": "contract,reassoc"}
        flags.add("nnan")
        assert CompilationContext.get_compile_hints() == {"fastmath": "contract,reassoc"}

    assert CompilationContext.get_compile_hints() == {}


def test_occupancy_hint_normalizes_int_enum_to_plain_int():
    class Waves(IntEnum):
        TWO = 2

    with CompilationContext.compile_hints({"waves_per_eu": Waves.TWO}):
        assert CompilationContext.get_compile_hints() == {"waves_per_eu": 2}
        assert type(CompilationContext.get_compile_hints()["waves_per_eu"]) is int


def test_compile_callable_uses_the_same_layer_semantics():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    launch.compile_hints = {"waves_per_eu": 4, "future_hint": {"outer": 1}}
    flyc.compile[{"waves_per_eu": None, "maxnreg": 64}](launch)
    assert launch.compile_hints == {
        "waves_per_eu": 4,
        "maxnreg": 64,
        "future_hint": {"outer": 1},
    }

    flyc.compile[{"waves_per_eu": 0}](launch)
    assert launch.compile_hints["waves_per_eu"] == 0
    assert launch._effective_compile_hints() == {
        "maxnreg": 64,
        "future_hint": {"outer": 1},
    }


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


def test_generic_compile_hint_snapshot_detaches_nested_mutable_values():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    source = {"schedule": [{"stage": 1}]}
    launch.compile_hints = {"future_hint": source}
    snapshot = launch._effective_compile_hints()
    source["schedule"][0]["stage"] = 2

    assert snapshot == {"future_hint": {"schedule": [{"stage": 1}]}}
    assert launch._effective_compile_hints() == {"future_hint": {"schedule": [{"stage": 2}]}}


@pytest.mark.parametrize(
    "hints",
    [
        {"future_hint": {"nested": {1, 2}}},
        {"future_hint": bytearray(b"mutable")},
        {"future_hint": object()},
        {"future_hint": float("nan")},
        {1: "non-string top-level key"},
        {"future_hint": {1: "non-string nested key"}},
    ],
)
def test_generic_compile_hints_reject_values_without_stable_snapshot_identity(hints):
    with pytest.raises((TypeError, ValueError), match="compile_hint|compile hint"):
        with CompilationContext.compile_hints(hints):
            pass
