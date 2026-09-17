# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Warp merge sort contracts, shapes, types and algorithms."""

import coop_warp_primitive_utils as primitive_checks
import coop_warp_sorting_utils
import pytest
import torch
from coop_common import DTYPES, WARP_WIDTHS, dtype_id
from coop_test_utils import ARCHES
from coop_test_utils import warp_default_device as warp_default_device
from coop_warp_sorting_utils import _FLOATS


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("primitive", ["warp_merge_sort"])
@pytest.mark.parametrize("block,count", [(1, 1), (8, 3), (64, 4), (128, 1)])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.usefixtures("warp_default_device")
def test_sort(primitive, block, count, descending):
    coop_warp_sorting_utils.check_sort(primitive, block, count, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("primitive,count", [("warp_merge_sort", 3)])
@pytest.mark.parametrize("width", [1, 2, 32, None])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_pairs(primitive, count, width, descending):
    coop_warp_sorting_utils.check_sort_pairs(primitive, count, width, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("scope,block", [("warp", 16)])
@pytest.mark.parametrize("descending", [False, True])
def test_merge_comparator_pairs_partial(scope, block, descending):
    coop_warp_sorting_utils.check_merge_comparator_pairs_partial(scope, block, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["warp_merge"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.usefixtures("warp_default_device")
def test_record_keys_and_record_values(case, count):
    coop_warp_sorting_utils.check_record_keys_and_record_values(case, count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["warp_merge"])
@pytest.mark.parametrize("dtype,torch_dtype", _FLOATS, ids=["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("universal", [False, True], ids=["public", "universal"])
@pytest.mark.usefixtures("warp_default_device")
def test_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal):
    "Stable entry points preserve duplicate and signed-zero payload order."
    coop_warp_sorting_utils.check_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["warp_merge"])
@pytest.mark.parametrize("projected", [False, True], ids=["lexicographic", "equivalent_fields"])
@pytest.mark.parametrize("universal", [False, True], ids=["public", "universal"])
def test_nested_float_record_keys(case, projected, universal):
    "Nested float fields and untouched fields follow comparator/decomposer order."
    coop_warp_sorting_utils.check_nested_float_record_keys(case, projected, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", (1, *WARP_WIDTHS))
@pytest.mark.parametrize("count", [1, 9])
@pytest.mark.parametrize("primitive", ["warp_merge_sort"])
def test_matrix_warp_dtypes_and_widths(entry, width, count, primitive):
    primitive_checks.check_warp_dtypes_and_widths(entry, width, count, primitive)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_merge_sort"])
def test_compile_family(monkeypatch, arch, case):
    primitive_checks.check_compile_family(monkeypatch, arch, case)
