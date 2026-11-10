# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block merge sort contracts, shapes, types and algorithms."""

import coop_block_utils as checks
import pytest
import torch
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import full_matrix_enabled, matrix_cases

import flydsl.expr as fx


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "primitive,block,count,descending",
    [
        (primitive, block, count, descending)
        for primitive, (block, count), descending in matrix_cases(
            ["BlockMergeSort"], [(64, 1), (64, 3), (64, 4), (128, 1)], [False, True]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_sort(primitive, block, count, descending):
    checks.check_sort(primitive, block, count, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "scope,block,descending",
    [
        (scope, block, descending)
        for (scope, block), descending in matrix_cases([("block", 64), ("block", 128)], [False, True])
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_merge_comparator_pairs_partial(scope, block, descending):
    checks.check_merge_comparator_pairs_partial(scope, block, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("name", ["merge"])
@pytest.mark.usefixtures("coop_default_device")
def test_large_tile_low_shared_memory(name):
    checks.check_large_tile_low_shared_memory(name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case,count", matrix_cases(["block_merge"], [1, 3]))
@pytest.mark.usefixtures("coop_default_device")
def test_record_keys_and_record_values(case, count):
    checks.check_record_keys_and_record_values(case, count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.usefixtures("coop_default_device")
def test_compact_record_merge():
    checks.check_compact_record_merge()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["merge"])
@pytest.mark.usefixtures("coop_default_device")
def test_int128_register_keys(case):
    checks.check_int128_register_keys(case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case,dtype,torch_dtype,descending,universal",
    [
        (case, dtype, torch_dtype, descending, universal)
        for case, (dtype, torch_dtype), descending, universal in matrix_cases(
            ["block_merge"], checks._FLOATS, [False, True], [False, True]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal):
    "Stable entry points preserve duplicate and signed-zero payload order."
    checks.check_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case,projected,universal", matrix_cases(["block_merge"], [False, True], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_nested_float_record_keys(case, projected, universal):
    "Nested float fields and untouched fields follow comparator/decomposer order."
    checks.check_nested_float_record_keys(case, projected, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,count,case", matrix_cases(DTYPES, [1, 9], [("BlockMergeSort", "sort")], independent=True)
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_dtypes(entry, count, case):
    block = 128 if full_matrix_enabled() or count == 1 else 2
    checks.check_block_case(entry, block, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,case", matrix_cases(BLOCK_THREADS, [1, 9], [("BlockMergeSort", "sort")], independent=True)
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_shapes(block, count, case):
    checks.check_block_case((fx.Int32, "torch.int32"), block, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", [("BlockMergeSort", "sort")], ids=lambda c: c[1])
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_three_dimensional_block(case):
    checks.check_block_case((fx.Float64, "torch.float64"), (8, 4, 4), 3, case)
