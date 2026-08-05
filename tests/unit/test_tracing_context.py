# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for the ambient key/value options carried by ``tracing_context``."""

import pytest

from flydsl.expr.meta import tracing_context, tracing_option

pytestmark = pytest.mark.l0_backend_agnostic


def _traced():
    pass


def test_option_is_visible_inside_the_scope_only():
    assert tracing_option("fastmath") is None
    with tracing_context(_traced, fastmath="fast"):
        assert tracing_option("fastmath") == "fast"
    assert tracing_option("fastmath") is None


def test_inner_scope_shadows_and_restores():
    with tracing_context(fastmath="fast", unroll=4):
        with tracing_context(fastmath="contract"):
            assert tracing_option("fastmath") == "contract"
            # Keys the inner frame does not set stay visible.
            assert tracing_option("unroll") == 4
        assert tracing_option("fastmath") == "fast"


def test_none_option_does_not_shadow_an_outer_value():
    with tracing_context(fastmath="fast"):
        with tracing_context(_traced, fastmath=None):
            assert tracing_option("fastmath") == "fast"


def test_default_is_returned_for_unset_keys():
    with tracing_context(_traced):
        assert tracing_option("missing", "fallback") == "fallback"
