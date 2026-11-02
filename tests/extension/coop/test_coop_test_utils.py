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
@pytest.mark.parametrize("independent", [False, True])
def test_matrix_cases_restores_the_exact_full_product(monkeypatch, independent):
    monkeypatch.setenv("FLYDSL_COOP_TESTS_FULL", "1")
    axes = (("a", "b"), (1, 2, 3), (False, True))

    assert matrix_cases(*axes, independent=independent) == tuple(itertools.product(*axes))


@pytest.mark.l0_backend_agnostic
def test_independent_matrix_keeps_expensive_axes_separate(monkeypatch):
    monkeypatch.delenv("FLYDSL_COOP_TESTS_FULL", raising=False)
    axes = ((1, 64, 1024), (1, 9), ("int32", "float64"))
    cases = matrix_cases(*axes, independent=True)
    assert len(cases) == 1 + sum(len(axis) - 1 for axis in axes)
    assert len(set(cases)) == len(cases)
    for index, axis in enumerate(axes):
        assert {case[index] for case in cases} == set(axis)
    assert all(sum(value != axis[0] for value, axis in zip(case, axes)) <= 1 for case in cases)


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("axes", [(), ((1,), ())])
def test_matrix_cases_rejects_missing_axes(axes):
    with pytest.raises(ValueError, match="non-empty parameter axes"):
        matrix_cases(*axes)
