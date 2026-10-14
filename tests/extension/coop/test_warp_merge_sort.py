# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Warp merge sort contracts, shapes, types and algorithms."""

import coop_warp_utils as checks
import pytest
import torch
from coop_common import DTYPES, WARP_WIDTHS
from coop_test_utils import ARCHES, matrix_cases
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import warp_default_device as warp_default_device


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "primitive,block,count,descending",
    [
        (primitive, block, count, descending)
        for primitive, (block, count), descending in matrix_cases(
            ["warp_merge_sort"],
            [(1, 1), (8, 3), (64, 4), (128, 1)],
            [False, True],
        )
    ],
)
@pytest.mark.usefixtures("warp_default_device")
def test_sort(primitive, block, count, descending):
    checks.check_sort(primitive, block, count, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "primitive,count,width,descending",
    [
        (*primitive, width, descending)
        for primitive, width, descending in matrix_cases(
            [("warp_merge_sort", 3)],
            [1, 2, 32, None],
            [False, True],
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_sort_pairs(primitive, count, width, descending):
    checks.check_sort_pairs(primitive, count, width, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("scope,block", [("warp", 16)])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.usefixtures("coop_default_device")
def test_merge_comparator_pairs_partial(scope, block, descending):
    checks.check_merge_comparator_pairs_partial(scope, block, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["warp_merge"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.usefixtures("warp_default_device")
def test_record_keys_and_record_values(case, count):
    checks.check_record_keys_and_record_values(case, count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case,dtype,torch_dtype,descending,universal",
    [
        (case, *entry, descending, universal)
        for case, entry, descending, universal in matrix_cases(
            ["warp_merge"],
            checks._FLOATS,
            [False, True],
            [False, True],
        )
    ],
)
@pytest.mark.usefixtures("warp_default_device")
def test_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal):
    "Stable entry points preserve duplicate and signed-zero payload order."
    checks.check_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case,projected,universal",
    matrix_cases(["warp_merge"], [False, True], [False, True]),
)
@pytest.mark.usefixtures("coop_default_device")
def test_nested_float_record_keys(case, projected, universal):
    "Nested float fields and untouched fields follow comparator/decomposer order."
    checks.check_nested_float_record_keys(case, projected, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,width,count,primitive",
    matrix_cases(DTYPES, (1, *WARP_WIDTHS), [1, 9], ["warp_merge_sort"]),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_warp_dtypes_and_widths(entry, width, count, primitive):
    checks.check_warp_dtypes_and_widths(entry, width, count, primitive)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch,case", matrix_cases(ARCHES, ["warp_merge_sort"]))
def test_compile_family(monkeypatch, arch, case):
    checks.check_compile_family(monkeypatch, arch, case)
