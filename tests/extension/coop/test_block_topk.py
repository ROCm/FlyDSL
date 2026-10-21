# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block topk contracts, shapes, types and algorithms."""

import coop_block_utils as checks
import pytest
import torch
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import matrix_cases

import flydsl.expr as fx
from flydsl.extension.coop.block.topk import BlockTopKAlgorithm


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("k,largest", matrix_cases([0, 1, 17, 192], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_topk(k, largest):
    checks.check_topk(k, largest)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm,largest,k", matrix_cases(list(BlockTopKAlgorithm), [False, True], [0, 7, 41]))
@pytest.mark.usefixtures("coop_default_device")
def test_topk_runtime_pairs_partial(algorithm, largest, k):
    checks.check_topk_runtime_pairs_partial(algorithm, largest, k)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("name", ["topk", "air"])
@pytest.mark.usefixtures("coop_default_device")
def test_large_tile_low_shared_memory(name):
    checks.check_large_tile_low_shared_memory(name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "dtype,name,case",
    [
        (dtype, name, case)
        for (dtype, name), case in matrix_cases(
            [(fx.Uint8, "uint8"), (fx.Int64, "int64"), (fx.Float64, "float64")], ["air"]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_wide_numeric_policies(dtype, name, case):
    checks.check_wide_numeric_policies(dtype, name, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case,count", matrix_cases(["topk_merge"], [1, 3]))
@pytest.mark.usefixtures("coop_default_device")
def test_record_keys_and_record_values(case, count):
    checks.check_record_keys_and_record_values(case, count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.usefixtures("coop_default_device")
def test_air_record_payload_and_partial_record_sort():
    checks.check_air_record_payload_and_partial_record_sort()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["air"])
@pytest.mark.usefixtures("coop_default_device")
def test_int128_register_keys(case):
    checks.check_int128_register_keys(case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,universal,count,case",
    [
        (*shape, count, case)
        for shape, count, case in matrix_cases(
            matrix_cases(BLOCK_THREADS, [False, True]), [1, 9], ["BlockTopK"], independent=True
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_block_shapes(block, universal, count, case):
    checks.check_every_block_width(block, count, case, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry,count", matrix_cases(DTYPES, [1, 9], independent=True))
@pytest.mark.usefixtures("coop_default_device")
def test_topk_dtypes(entry, count):
    checks.check_topk_dtypes(entry, count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case,dtype,torch_dtype,descending,universal",
    [
        (case, dtype, torch_dtype, descending, universal)
        for case, (dtype, torch_dtype), descending, universal in matrix_cases(
            ["topk_merge", "air"], checks._FLOATS, [False, True], [False, True]
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
@pytest.mark.parametrize("case,projected,universal", matrix_cases(["topk_merge"], [False, True], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_nested_float_record_keys(case, projected, universal):
    "Nested float fields and untouched fields follow comparator/decomposer order."
    checks.check_nested_float_record_keys(case, projected, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "dtype,unsigned,mantissa,algorithm,descending,universal",
    [
        (dtype, unsigned, mantissa, algorithm, descending, universal)
        for (dtype, unsigned, mantissa), algorithm, descending, universal in matrix_cases(
            [
                (fx.Float16, fx.Uint16, 10),
                (fx.BFloat16, fx.Uint16, 7),
                (fx.Float32, fx.Uint32, 23),
                (fx.Float64, fx.Uint64, 52),
            ],
            ["air"],
            [False, True],
            [False, True],
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_float_radix_exact_bits(dtype, unsigned, mantissa, algorithm, descending, universal):
    checks.check_float_radix_exact_bits(dtype, unsigned, mantissa, algorithm, descending, universal)
