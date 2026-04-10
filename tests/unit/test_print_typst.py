#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import importlib

typst_mod = importlib.import_module("flydsl.expr.utils.print_typst")


def _pt_value(size: str) -> float:
    assert size.endswith("pt")
    return float(size[:-2])


def test_typst_grid_uses_single_cell_size_within_grid():
    cells = {
        (0, 0): ("#ffffff", "T9 #linebreak() V0"),
        (0, 1): ("#ffffff", "T10 #linebreak() V0"),
    }

    width, height = typst_mod._typst_grid_cell_size(1, 12, cells, row_labels=True, col_labels=True)
    block = typst_mod._typst_grid_block(1, 12, cells)

    assert _pt_value(height) > _pt_value(width)
    assert f"#box(width: {width}, height: {height}, inset: 1pt)" in block
    assert "inset: 0pt," in block


def test_typst_grid_block_accepts_explicit_cell_size():
    cells = {
        (0, 0): ("#ffffff", "1"),
    }

    block = typst_mod._typst_grid_block(1, 1, cells, cell_width="30pt", cell_height="40pt")

    assert "#box(width: 30pt, height: 40pt, inset: 1pt)" in block


def test_typst_grid_cell_size_is_local_to_each_grid():
    small_cells = {
        (0, 0): ("#ffffff", "3"),
    }
    large_cells = {
        (0, 0): ("#ffffff", "T100 #linebreak() V12"),
    }

    small_width, small_height = typst_mod._typst_grid_cell_size(1, 1, small_cells, row_labels=True, col_labels=True)
    large_width, large_height = typst_mod._typst_grid_cell_size(1, 1, large_cells, row_labels=True, col_labels=True)

    assert _pt_value(large_width) > _pt_value(small_width)
    assert _pt_value(large_height) > _pt_value(small_height)


def test_typst_copy_grid_cell_sizes_share_height():
    cells_src = {
        (0, 0): ("#ffffff", "1"),
    }
    cells_dst = {
        (0, 0): ("#ffffff", "T0 #linebreak() V0 #linebreak() X"),
    }

    src_size, dst_size = typst_mod._typst_copy_grid_cell_sizes(1, 1, cells_src, cells_dst)

    assert src_size[1] == dst_size[1]


def test_typst_mma_grid_cell_sizes_share_ac_height_and_bc_width():
    cells_A = {
        (0, 0): ("#ffffff", "T0 #linebreak() V0 #linebreak() X"),
    }
    cells_B = {
        (0, 0): ("#ffffff", "T1000 #linebreak() V0"),
    }
    cells_C = {
        (0, 0): ("#ffffff", "T0 #linebreak() V0"),
    }

    sizes = typst_mod._typst_mma_grid_cell_sizes(1, 1, 1, cells_A, cells_B, cells_C)

    assert sizes["A"][1] == sizes["C"][1]
    assert sizes["B"][0] == sizes["C"][0]
