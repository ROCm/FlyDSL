#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Guards the JIT cache-key fast path.

``TensorAdaptor.lean_cache_signature`` and ``JitFunction._fast_cache_key`` must
produce byte-identical keys to the full path (``__cache_signature__`` /
``_build_full_cache_key``); otherwise the fast probe would miss or, worse,
dispatch to the wrong compiled variant.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

from flydsl.compiler.jit_argument import TensorAdaptor  # noqa: E402

pytestmark = [pytest.mark.l2_device]


def _tensors():
    dev = "cuda"
    return [
        torch.empty((8192, 8192), device=dev, dtype=torch.bfloat16),
        torch.empty((64, 256), device=dev, dtype=torch.float32),
        torch.empty((1024,), device=dev, dtype=torch.float16),
        torch.empty((4, 128, 256), device=dev, dtype=torch.bfloat16),
        torch.empty((128, 256), device=dev, dtype=torch.float32).t(),  # transposed
        torch.empty((128, 128), device=dev, dtype=torch.float8_e4m3fnuz),  # fp8
        torch.empty((1, 512), device=dev, dtype=torch.float32),  # unit-size dim
        torch.empty((33, 1025), device=dev, dtype=torch.bfloat16),  # non-pow2
    ]


def test_lean_cache_signature_matches_adaptor():
    for t in _tensors():
        ref = TensorAdaptor(t).__cache_signature__()
        got = TensorAdaptor.lean_cache_signature(t)
        assert got == ref, f"shape={tuple(t.shape)} stride={tuple(t.stride())}: {got!r} != {ref!r}"


def test_fast_cache_key_matches_full_key():
    from kernels.softmax_kernel import build_softmax_module

    jf = build_softmax_module(256, 512, "bf16")
    a = torch.empty((256, 512), device="cuda", dtype=torch.bfloat16)
    c = torch.empty_like(a)
    s = torch.cuda.current_stream()
    jf(a, c, 256, stream=s)  # warm: build sig + caches
    torch.cuda.synchronize()

    sig = jf._sig
    # Fresh bound for the lean key (must NOT be mutated by _fast_cache_key).
    bound = sig.bind(a, c, 256, stream=s)
    bound.apply_defaults()
    fast = jf._fast_cache_key(bound.arguments, owner_cls=None, bound_self=None)
    assert all(not hasattr(v, "_tensor_keepalive") for v in bound.arguments.values()), "fast key must not mutate args"

    # Full key from a separate bound (it mutates in place).
    bound2 = sig.bind(a, c, 256, stream=s)
    bound2.apply_defaults()
    full = jf._build_full_cache_key(bound2.arguments, owner_cls=None, bound_self=None)
    assert fast == full, f"\nfast={fast}\nfull={full}"
