# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp exports, portable dispatch fallbacks and invalid shapes."""

import pytest

import flydsl.expr as fx
from flydsl.extension.coop import warp


def test_all_public_names_have_a_universal_implementation():
    assert set(fx.coop.__all__) == set(fx.coop.universal.__all__)
    for name in fx.coop.__all__:
        assert hasattr(fx.coop.universal, name), name


def test_new_warp_functions_retain_portable_fallbacks():
    from flydsl.extension.coop.warp import rocdl

    for name in warp.__all__:
        portable = getattr(fx.coop.universal, name)
        expected = getattr(rocdl, name) if name in rocdl.__all__ else portable
        assert warp._dispatch._resolve("rocm", name, portable) is expected
        assert warp._dispatch._resolve("unknown-backend", name, portable) is portable


@pytest.mark.parametrize("width", [0, 3, -1, 128])
@pytest.mark.parametrize(
    "name",
    ["warp_blocked_to_striped", "warp_striped_to_blocked", "warp_bitonic_sort", "warp_merge_sort"],
)
def test_bad_warp_widths(name, width, insert_point):
    value = fx.Vector.from_elements([fx.Int32(1)])
    kwargs = {"compare_op": lambda a, b: a < b} if name.endswith("sort") else {}
    with pytest.raises(ValueError, match="power of two|must not exceed"):
        getattr(fx.coop, name)(value, width=width, **kwargs)


def test_tile_input_extent_is_checked(insert_point):
    value = fx.Vector.from_elements([fx.Int32(1), fx.Int32(2)])
    with pytest.raises(ValueError, match="same shape"):
        fx.coop.warp_merge_sort(value, fx.Int32(1), compare_op=lambda a, b: a < b)
