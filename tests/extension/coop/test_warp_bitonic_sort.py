# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Warp bitonic sort contracts, shapes, types and algorithms."""

import coop_warp_utils as checks
import pytest
import torch
from coop_common import DTYPES, WARP_WIDTHS, dtype_id
from coop_test_utils import ARCHES
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import warp_default_device as warp_default_device


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("primitive", ["warp_bitonic_sort"])
@pytest.mark.parametrize("block,count", [(1, 1), (8, 3), (64, 4), (128, 1)])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.usefixtures("warp_default_device")
def test_sort(primitive, block, count, descending):
    checks.check_sort(primitive, block, count, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("primitive,count", [("warp_bitonic_sort", 4)])
@pytest.mark.parametrize("width", [1, 2, 32, None])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.usefixtures("coop_default_device")
def test_sort_pairs(primitive, count, width, descending):
    checks.check_sort_pairs(primitive, count, width, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("count", [1, 3, 5])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.usefixtures("coop_default_device")
def test_bitonic_striped_pairs(count, partial):
    checks.check_bitonic_striped_pairs(count, partial)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["bitonic"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.usefixtures("warp_default_device")
def test_record_keys_and_record_values(case, count):
    checks.check_record_keys_and_record_values(case, count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["bitonic"])
@pytest.mark.parametrize("dtype,torch_dtype", checks._FLOATS, ids=["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("universal", [False, True], ids=["public", "universal"])
@pytest.mark.usefixtures("warp_default_device")
def test_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal):
    "Stable entry points preserve duplicate and signed-zero payload order."
    checks.check_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["bitonic"])
@pytest.mark.parametrize("projected", [False, True], ids=["lexicographic", "equivalent_fields"])
@pytest.mark.parametrize("universal", [False, True], ids=["public", "universal"])
@pytest.mark.usefixtures("coop_default_device")
def test_nested_float_record_keys(case, projected, universal):
    "Nested float fields and untouched fields follow comparator/decomposer order."
    checks.check_nested_float_record_keys(case, projected, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", (1, *WARP_WIDTHS))
@pytest.mark.parametrize("count", [1, 9])
@pytest.mark.parametrize("primitive", ["warp_bitonic_sort"])
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_warp_dtypes_and_widths(entry, width, count, primitive):
    checks.check_warp_dtypes_and_widths(entry, width, count, primitive)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_bitonic_sort"])
def test_compile_family(monkeypatch, arch, case):
    checks.check_compile_family(monkeypatch, arch, case)
