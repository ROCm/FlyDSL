# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for cooperative-test parameter selection."""

import itertools

import pytest
from coop_test_utils import matrix_cases


@pytest.mark.l0_backend_agnostic
def test_matrix_cases_covers_every_axis_value(monkeypatch):
    monkeypatch.delenv("FLYDSL_COOP_TESTS_FULL", raising=False)
    axes = (("a", "b"), (1, 2, 3), (False, True))

    cases = matrix_cases(*axes)

    assert len(cases) == max(map(len, axes))
    for index, axis in enumerate(axes):
        assert {case[index] for case in cases} == set(axis)


@pytest.mark.l0_backend_agnostic
def test_matrix_cases_restores_the_exact_full_product(monkeypatch):
    monkeypatch.setenv("FLYDSL_COOP_TESTS_FULL", "1")
    axes = (("a", "b"), (1, 2, 3), (False, True))

    assert matrix_cases(*axes) == tuple(itertools.product(*axes))


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("axes", [(), ((1,), ())])
def test_matrix_cases_rejects_missing_axes(axes):
    with pytest.raises(ValueError, match="non-empty parameter axes"):
        matrix_cases(*axes)
