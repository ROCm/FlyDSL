# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block radix rank contracts, shapes, types and algorithms."""

import coop_block_utils as checks
import pytest
import torch
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import ARCHES, matrix_cases
from coop_test_utils import coop_default_device as coop_default_device

import flydsl.expr as fx
from flydsl.extension.coop.block.radix_rank import BlockRadixRankAlgorithm

ALGORITHMS = list(fx.coop.BlockRadixRankAlgorithm)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "begin,end,descending,universal",
    [
        (begin, end, descending, universal)
        for (begin, end), descending, universal in matrix_cases([(0, 0), (0, 4), (2, 6)], [False, True], [False, True])
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_radix_rank(begin, end, descending, universal):
    checks.check_radix_rank(begin, end, descending, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm,block", matrix_cases(list(BlockRadixRankAlgorithm), [64, 128]))
@pytest.mark.usefixtures("coop_default_device")
def test_rank_algorithms_layout(algorithm, block):
    checks.check_rank_algorithms_layout(algorithm, block)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "algorithm,descending",
    matrix_cases(
        [
            BlockRadixRankAlgorithm.MATCH,
            BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ANY,
            BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ATOMIC_OR,
        ],
        [False, True],
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_rank_counts_callback(algorithm, descending):
    checks.check_rank_counts_callback(algorithm, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm", list(BlockRadixRankAlgorithm))
@pytest.mark.usefixtures("coop_default_device")
def test_rank_nine_bit_configuration(algorithm):
    checks.check_rank_nine_bit_configuration(algorithm)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("name", ["rank"])
@pytest.mark.usefixtures("coop_default_device")
def test_large_tile_low_shared_memory(name):
    checks.check_large_tile_low_shared_memory(name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["rank"])
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
            matrix_cases(BLOCK_THREADS, [False, True]), [1, 9], ["BlockRadixRank"], independent=True
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_block_shapes(block, universal, count, case):
    checks.check_every_block_width(block, count, case, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry,count", matrix_cases([e for e in DTYPES if not e[0].is_float], [1, 9]))
@pytest.mark.usefixtures("coop_default_device")
def test_radix_rank_dtypes(entry, count):
    checks.check_radix_rank_dtypes(entry, count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "kind,algorithm,block,count,form,descending",
    [
        (kind, algorithm, block, count, form, descending)
        for kind, algorithm, (block, count, form), descending in matrix_cases(
            ["float16", "bfloat16", "float32", "float64", "record"],
            ALGORITHMS,
            [(64, 1, "scalar"), (128, 3, "vector"), (64, 9, "vector")],
            [False, True],
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_noninteger_rank(kind, algorithm, block, count, form, descending):
    checks.check_noninteger_rank(kind, algorithm, block, count, form, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("form,algorithm", matrix_cases(["list", "tuple"], ALGORITHMS))
@pytest.mark.usefixtures("coop_default_device")
def test_record_rank_range_input(form, algorithm):
    checks.check_record_rank_range_input(form, algorithm)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("kind,algorithm,descending", matrix_cases(["float64", "record"], ALGORITHMS, [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_noninteger_counts_and_prefixes(kind, algorithm, descending):
    checks.check_noninteger_counts_and_prefixes(kind, algorithm, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.usefixtures("coop_default_device")
def test_extracted_digit_wider_than_key(algorithm):
    checks.check_extracted_digit_wider_than_key(algorithm)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch,kind,algorithm", matrix_cases(ARCHES, ["float32", "record"], ALGORITHMS))
def test_compile_noninteger_rank(monkeypatch, arch, kind, algorithm):
    checks.check_compile_noninteger_rank(monkeypatch, arch, kind, algorithm)


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize(
    "ns,record,case",
    matrix_cases([fx.coop, fx.coop.universal], [False, True], ["missing_extractor", "missing_bits", "multiple_passes"]),
)
def test_noninteger_rank_configuration(ns, record, case):
    checks.check_noninteger_rank_configuration(ns, record, case)


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("case", ["float", "record", "negative", "overflow"])
def test_bad_extracted_digit(monkeypatch, case):
    checks.check_bad_extracted_digit(monkeypatch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("algorithm,descending", matrix_cases(ALGORITHMS, [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_unsigned_digits_across_passes(algorithm, descending):
    checks.check_rank_unsigned_digits(algorithm, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("coop_default_device")
def test_unsigned_shared_values():
    checks.check_unsigned_shared_values()


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize(
    "algorithm,universal", matrix_cases([BlockRadixRankAlgorithm.BASIC, BlockRadixRankAlgorithm.MEMOIZE], [False, True])
)
def test_count_callbacks_require_match(insert_point, monkeypatch, algorithm, universal):
    monkeypatch.setenv("ARCH", "gfx942")
    ns = fx.coop.universal if universal else fx.coop
    p = ns.BlockRadixRank[fx.Int32, 64, 1, algorithm]
    with pytest.raises(ValueError, match="callbacks require a MATCH"):
        p(fx.Int32(0), storage=None, end_bit=2, count_callback=lambda counts: None)


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("algorithm", [BlockRadixRankAlgorithm.BASIC, BlockRadixRankAlgorithm.MATCH])
def test_prefix_requires_one_pass(insert_point, monkeypatch, algorithm):
    monkeypatch.setenv("ARCH", "gfx942")
    p = fx.coop.BlockRadixRank[fx.Int32, 64, 1, algorithm]
    with pytest.raises(ValueError, match="with_prefix requires one digit pass"):
        p(fx.Int32(0), storage=None, end_bit=8, radix_bits=4, with_prefix=True)
