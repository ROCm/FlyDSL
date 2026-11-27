# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cross-target smoke matrices shared by cooperative primitive families.

Every primitive and target is visited by default. Exhaustive mode restores
all primitive/target pairs, including those formerly split across files.
"""

import coop_block_utils as block_checks
import coop_warp_utils as warp_checks
import pytest
import torch
from coop_test_utils import ARCHES, matrix_cases


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case",
    matrix_cases(ARCHES, ["difference", "discontinuity", "histogram", "decode_lengths", "decode_offsets", "shuffle"]),
)
def test_block_sequence_contracts_compile(monkeypatch, arch, case):
    block_checks.check_sequence_contracts_compile(monkeypatch, arch, case)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case",
    matrix_cases(
        ARCHES,
        [
            "BlockAdjacentDifference",
            "BlockDiscontinuity",
            "BlockExchange",
            "BlockHistogram",
            "BlockLoad",
            "BlockStore",
            "BlockMergeSort",
            "BlockRadixRank",
            "BlockRadixSort",
            "BlockRunLengthDecode",
            "BlockShuffle",
            "BlockTopK",
        ],
    ),
)
def test_block_compile_family(monkeypatch, arch, case):
    block_checks.check_compile_family(monkeypatch, arch, case)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case",
    matrix_cases(
        ARCHES,
        [
            "record_merge",
            "rank_memoize",
            "rank_match",
            "rank_early_any",
            "rank_early_atomic_or",
            "radix_digit",
            "record_radix",
            "air",
        ],
    ),
)
def test_block_compile_sorting_policies(monkeypatch, arch, case):
    block_checks.check_compile_sorting_policies(monkeypatch, arch, case)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case",
    matrix_cases(
        ARCHES,
        [
            "warp_bitonic_sort",
            "warp_blocked_to_striped",
            "warp_striped_to_blocked",
            "warp_load",
            "warp_store",
            "warp_merge_sort",
            "warp_reduce_batched",
        ],
    ),
)
def test_warp_compile_family(monkeypatch, arch, case):
    warp_checks.check_compile_family(monkeypatch, arch, case)


# The cases whose choice depends on the target, or on what the gfx9 merge can rank, run on one gfx9 and one
# other target in the default matrix.
_SORT_BY_TARGET = ("64-bit keys", "16-bit keys", "64-bit payload", "pointer payload", "pointer keys")


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case,universal",
    sorted(
        {
            *((arch, case, False) for arch, case in matrix_cases(ARCHES, list(warp_checks.SORT_CHOICES))),
            *((arch, case, False) for arch in ("gfx950", "gfx1201") for case in _SORT_BY_TARGET),
            ("gfx942", "64-bit keys", False),
            *matrix_cases(["gfx942", "gfx1100"], ["payload", "64-bit keys"], [True]),
        }
    ),
)
def test_warp_sort_choice(monkeypatch, arch, case, universal, tmp_path):
    warp_checks.check_sort_choice(monkeypatch, arch, case, universal, tmp_path)
