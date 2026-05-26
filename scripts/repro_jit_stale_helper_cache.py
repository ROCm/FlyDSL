#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Reproduce stale in-process JIT cache when a helper dependency changes."""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops


def _helper(value):
    return value


@flyc.kernel
def _write_helper_result(out: fx.Tensor):
    value = _helper(fx.Int32(7))
    rsrc = buffer_ops.create_buffer_resource(out)
    buffer_ops.buffer_store(value, rsrc, fx.Int32(0))


@flyc.jit
def _launch(out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _write_helper_result(out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def _sentinel(_value):
    raise RuntimeError("sentinel helper observed during retrace")


def _make_out():
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    return out, t_out


def _run_once(label):
    out, t_out = _make_out()
    _launch(t_out)
    torch.cuda.synchronize()
    print(f"[{label}] out={out.item()}")
    return out.item()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required")

    print("Step 1: compile and cache the baseline launch.")
    assert _run_once("baseline") == 7

    print("Step 2: replace _helper with a sentinel that must fail if retraced.")
    globals()["_helper"] = _sentinel

    print("Step 3: rerun the same shape without clearing JIT caches.")
    stale_reuse = False
    try:
        _run_once("cached_after_helper_change")
    except RuntimeError as exc:
        print(f"[cached_after_helper_change] observed sentinel: {exc}")
        print("[fixed] helper change invalidated the in-process cache.")
    else:
        stale_reuse = True
        print("[cached_after_helper_change] sentinel was not observed; stale in-process cache was reused.")

    print("Step 4: clear FlyDSL in-process caches and rerun.")
    _launch._clear_inprocess_caches()
    try:
        _run_once("after_cache_clear")
    except RuntimeError as exc:
        print(f"[after_cache_clear] observed sentinel: {exc}")
    else:
        raise AssertionError("expected sentinel after manual cache clear")

    if stale_reuse:
        raise AssertionError("stale in-process cache reused after helper dependency changed")


if __name__ == "__main__":
    main()
