# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_function import JitFunction


def test_raw_arg_cache_signature_uses_annotation_type_for_scalar_runtime_args():
    assert JitFunction._raw_arg_cache_signature(7, fx.Int32) == (fx.Int32,)


def test_direct_call_state_reconstructs_kwargs_after_warmup():
    @flyc.jit
    def launch(a: fx.Int32, b: fx.Int32 = 1):
        pass

    launch._ensure_sig()
    launch._set_direct_call_state((1, 2), object())
    direct = launch._direct_call_state

    assert launch._direct_call_args_tuple((3, 4), {}, direct) == (3, 4)
    assert launch._direct_call_args_tuple((3,), {"b": 4}, direct) == (3, 4)
    assert launch._direct_call_args_tuple((), {"a": 3, "b": 4}, direct) == (3, 4)
    assert launch._direct_call_args_tuple((3,), {"a": 4}, direct) is None


def test_direct_call_state_skips_constexpr_launchers():
    @flyc.jit
    def launch(a: fx.Int32, n: fx.Constexpr[int]):
        pass

    launch._ensure_sig()

    assert not launch._can_direct_call_state((1, 2))
