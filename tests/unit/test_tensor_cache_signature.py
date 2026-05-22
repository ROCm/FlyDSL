#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch

import flydsl.compiler as flyc
from flydsl.compiler.jit_argument import TensorAdaptor


def test_raw_tensor_cache_signature_defaults_to_dynamic_layout():
    a = torch.empty((2, 4), dtype=torch.float32, device="cpu")
    same_dynamic_layout = torch.empty((5, 8), dtype=torch.float32, device="cpu")
    different_leading_dim = torch.empty((2, 4), dtype=torch.float32, device="cpu").t()
    different_address_space = a.to("cuda") if torch.cuda.is_available() else None

    assert TensorAdaptor.raw_cache_signature(a) == TensorAdaptor.raw_cache_signature(same_dynamic_layout)
    assert TensorAdaptor.raw_cache_signature(a) != TensorAdaptor.raw_cache_signature(different_leading_dim)
    if different_address_space is not None:
        assert TensorAdaptor.raw_cache_signature(a) != TensorAdaptor.raw_cache_signature(different_address_space)


def test_from_dlpack_cache_signature_defaults_to_dynamic_layout():
    a = flyc.from_dlpack(torch.empty((2, 4), dtype=torch.float32))
    different_shape = flyc.from_dlpack(torch.empty((5, 8), dtype=torch.float32))

    assert a.__cache_signature__() == different_shape.__cache_signature__()


def test_dynamic_layout_cache_signature_excludes_dynamic_values():
    a = flyc.from_dlpack(torch.empty((2, 4), dtype=torch.float32)).mark_layout_dynamic(
        leading_dim=1,
        divisibility=4,
    )
    same_dynamic_layout = flyc.from_dlpack(torch.empty((5, 8), dtype=torch.float32)).mark_layout_dynamic(
        leading_dim=1,
        divisibility=4,
    )
    different_divisibility = flyc.from_dlpack(torch.empty((5, 8), dtype=torch.float32)).mark_layout_dynamic(
        leading_dim=1,
        divisibility=8,
    )

    assert a.__cache_signature__() == same_dynamic_layout.__cache_signature__()
    assert a.__cache_signature__() != different_divisibility.__cache_signature__()


def test_dynamic_layout_requires_inferable_leading_dim():
    non_compact = torch.empty_strided((2, 3), (5, 2), dtype=torch.float32)

    try:
        TensorAdaptor.raw_cache_signature(non_compact)
    except RuntimeError as exc:
        assert "Cannot determine leading dimension" in str(exc)
    else:
        raise AssertionError("expected non-compact tensor to reject default dynamic layout")
