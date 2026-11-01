# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile the heuristic's bounded specialization space before serving."""

import os
import threading

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .api import _arguments, _m_alignment, specialization_key
from .hybrid import build_hybrid
from .register import build_register

_compile_lock = threading.RLock()


def get_warmup_keys(
    n,
    k,
    max_m,
    *,
    mode="e4m3",
    normal_only=False,
    nibble_layout="split",
    dtype=torch.bfloat16,
    device_id=0,
    cache_policy="auto"
):
    """Enumerate heuristic and cached measured keys, without GPU work."""
    from .tuning import cached_configurations

    if max_m < 1:
        raise ValueError("max_m must be positive")
    keys = dict.fromkeys(
        specialization_key(
            m,
            n,
            k,
            mode=mode,
            normal_only=normal_only,
            nibble_layout=nibble_layout,
            dtype=dtype,
            device_id=device_id,
            cache_policy=cache_policy,
        )
        for m in range(1, max_m + 1)
    )
    for m, config in cached_configurations(
        n,
        k,
        max_m,
        mode=mode,
        normal_only=normal_only,
        nibble_layout=nibble_layout,
        dtype=dtype,
        device_id=device_id,
        cache_policy=cache_policy,
    ):
        keys[
            specialization_key(
                m,
                n,
                k,
                mode=mode,
                normal_only=normal_only,
                nibble_layout=nibble_layout,
                dtype=dtype,
                device_id=device_id,
                config=config,
                cache_policy=cache_policy,
            )
        ] = None
    return list(keys)


def compile_gemm(key):
    """Compile one key using meta tensors and FlyDSL's compile-only mode.

    Call during single-threaded startup, before inference or graph capture.
    No real tensors or dummy GPU launches are required. Runtime materializes
    the already compiled artifact on its first use.
    """
    engine, kernel_m, n, k, parameters = key
    options = dict(parameters)
    alignment = _m_alignment(engine, options)
    if kernel_m:
        m = kernel_m
    elif options.get("aligned_m", False):
        m = (
            options.get("tile_m", 256)
            if engine == "hybrid"
            else 16 * options.get("waves_m", 1) * options.get("repeat_m", 1)
        )
    else:
        m = 1
    npad = (n + 15) // 16 * 16
    mode = options["fold_mode"]
    dtype = torch.bfloat16 if options["output_dtype"] == "bf16" else torch.float16
    with torch.device("meta"):
        x = torch.empty(m, k, dtype=torch.float8_e4m3fn)
        packed = torch.empty(npad // 16, k // 16, 2, 16, 4, dtype=torch.uint8)
        scales = torch.empty(npad // 16, k // 32, 16, dtype=torch.uint8)
        x_scale = torch.empty(m, dtype=torch.float32)
        out = torch.empty(m, n, dtype=dtype)
        refs = None if mode == "grouped" else torch.empty(npad, dtype=torch.int32)
        table = None if mode == "grouped" else torch.empty(29 if mode == "e5m2" else 15, 2, dtype=torch.int32)
    args = _arguments(x, packed, scales, x_scale, out, refs, table, kernel_m, alignment)
    factory = build_hybrid if engine == "hybrid" else build_register
    launcher = factory(kernel_m, n, k, **options)
    with _compile_lock:
        previous = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            flyc.compile(launcher, *args, fx.Stream(None))
        finally:
            if previous is None:
                os.environ.pop("COMPILE_ONLY", None)
            else:
                os.environ["COMPILE_ONLY"] = previous
