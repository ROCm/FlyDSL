# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Batched warp reductions, result ownership and synchronization scopes."""

import coop_warp_primitive_utils as primitive_checks
import coop_warp_reduce_record_utils as record_checks
import coop_warp_reduce_utils as reduce_checks
import pytest
import torch
from coop_common import DTYPES, WARP_WIDTHS, dtype_id
from coop_test_utils import ARCHES
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.expr as fx


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("batches", [0, 1, 3, 8, 19])
@pytest.mark.parametrize("layout", ["blocked", "striped"])
@pytest.mark.usefixtures("warp_default_device")
def test_batched_result_ownership(batches, layout):
    reduce_checks.check_batched_result_ownership(batches, layout)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("universal", [False, True])
def test_batched_scalar_and_warp_broadcast(universal):
    reduce_checks.check_batched_scalar_and_warp_broadcast(universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("sync_physical_warp", [False, True])
@pytest.mark.parametrize("layout", ["blocked", "striped"])
def test_batched_wspro_synchronization_scope(sync_physical_warp, layout):
    reduce_checks.check_batched_wspro_synchronization_scope(sync_physical_warp, layout)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_batched_wspro"])
def test_synchronization_policies_compile(monkeypatch, arch, case):
    reduce_checks.check_reduce_scan_policies_compile(monkeypatch, arch, case)


@pytest.mark.rocm_lower
@pytest.mark.l2_device
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["batched"])
@pytest.mark.parametrize("universal", [False, True])
def test_nested_record_reduce(case, universal):
    record_checks.check_nested_record_warp_collectives(case, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", [1, 8, None])
@pytest.mark.parametrize("universal", [False, True])
def test_batched_reduce(width, universal):
    reduce_checks.check_batched_reduce(width, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", (1, *WARP_WIDTHS))
@pytest.mark.parametrize("op", [fx.ReductionOp.ADD, fx.ReductionOp.MUL, fx.ReductionOp.MIN, fx.ReductionOp.MAX])
@pytest.mark.usefixtures("warp_default_device")
def test_matrix_batched_reduction_dtypes_and_operations(entry, width, op):
    primitive_checks.check_batched_reduction_dtypes_and_operations(entry, width, op)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_reduce_batched"])
def test_compile_family(monkeypatch, arch, case):
    primitive_checks.check_compile_family(monkeypatch, arch, case)
