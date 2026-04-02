# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
import sys
from typing import Callable

# ---------------------------------------------------------------------------
# Color palettes (mirrors CuTe TikzColor_*)
# Make users feel like they are in CuTe :D
# ---------------------------------------------------------------------------


def _color_bw(idx: int) -> str:
    BW_COLORS_8 = [
        "#ffffff",  # black!00
        "#999999",  # black!40
        "#cccccc",  # black!20
        "#666666",  # black!60
        "#e5e5e5",  # black!10
        "#808080",  # black!50
        "#b2b2b2",  # black!30
        "#4c4c4c",  # black!70
    ]
    return BW_COLORS_8[idx % 8]


def _color_tv(tid: int, vid: int) -> str:
    TV_COLORS_8 = [
        "#afafff",  # rgb(175,175,255)
        "#afffaf",  # rgb(175,255,175)
        "#ffffaf",  # rgb(255,255,175)
        "#ffafaf",  # rgb(255,175,175)
        "#d2d2ff",  # rgb(210,210,255)
        "#d2ffd2",  # rgb(210,255,210)
        "#ffffd2",  # rgb(255,255,210)
        "#ffd2d2",  # rgb(255,210,210)
    ]
    return TV_COLORS_8[tid % 8]


def _typst_header() -> str:
    return '#set page(width: auto, height: auto, margin: 8pt)\n#set text(font: ("Ubuntu Mono", "New Computer Modern"), size: 9pt)\n'


def _typst_grid_block(
    M: int,
    N: int,
    cells: dict[tuple[int, int], tuple[str, str]],
    *,
    row_labels: bool = True,
    col_labels: bool = True,
    title: str = "",
) -> str:
    lines: list[str] = []
    if title:
        lines.append(f'#align(center, text(font: "New Computer Modern", weight: "bold", size: 11pt)[{title}])')
        lines.append("")

    col_count = N + (1 if row_labels else 0)
    col_spec = ", ".join(["auto"] * col_count)

    lines.append("#grid(")
    lines.append(f"  columns: ({col_spec}),")
    lines.append("  gutter: 0pt,")
    lines.append("  stroke: 0.5pt + black,")
    lines.append("  align: center + horizon,")
    lines.append("  inset: 4pt,")

    if col_labels:
        if row_labels:
            lines.append("  grid.cell(stroke: none)[], ")
        for n in range(N):
            lines.append(
                f'  grid.cell(stroke: none, inset: 2pt)[#text(font: "New Computer Modern", weight: "bold")[{n}]],'
            )

    for m in range(M):
        if row_labels:
            lines.append(
                f'  grid.cell(stroke: none, inset: 2pt)[#text(font: "New Computer Modern", weight: "bold")[{m}]],'
            )
        for n in range(N):
            fill, content = cells.get((m, n), ("white", ""))
            lines.append(f'  grid.cell(fill: rgb("{fill}"))[#text(font: "New Computer Modern")[{content}]],')

    lines.append(")")
    return "\n".join(lines)


def _escape_typst_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("<", "\\<").replace(">", "\\>")


def _typst_text_panel(lines: list[str]) -> str:
    panel = [
        "#box(",
        "  inset: 6pt,",
        "  stroke: 0.5pt + black,",
        "  radius: 2pt,",
        ")[",
    ]
    for i, line in enumerate(lines):
        if i:
            panel.append("  #linebreak()")
        panel.append(
            f'  #text(font: ("Ubuntu Mono", "New Computer Modern"))[{_escape_typst_text(line).replace(" ", "~")}]'
        )
    panel.append("]")
    return "\n".join(panel)


def _tiled_mma_text_lines(mma) -> list[str]:
    return [
        "Tiled MMA",
        f"  Thr Layout VMNK: {mma.thr_layout_vmnk}",
        f"  Permutation MNK: {mma.permutation_mnk}",
        "MMA Atom",
        f"  Atom Layout:     {mma.atom_layout}",
        f"  Tile Size MNK:   {mma.tile_size_mnk}",
        f"  TV Layout A:     {mma.tv_layout_A_tiled}",
        f"  TV Layout B:     {mma.tv_layout_B_tiled}",
        f"  TV Layout C:     {mma.tv_layout_C_tiled}",
    ]


def _tiled_copy_text_lines(copy) -> list[str]:
    return [
        "Tiled Copy",
        f"  TiledCopy:       {copy}",
        f"  Tile MN:         {copy.tile_mn}",
        f"  TV Layout tiled: {copy.layout_tv_tiled}",
        f"  Src Layout:      {copy.layout_src_tv_tiled}",
        f"  Dst Layout:      {copy.layout_dst_tv_tiled}",
    ]


def _get_shape_dims(layout) -> list[int]:
    from .. import primitive as prim

    shape = prim.get_shape(layout)
    flat = prim.int_tuple_product_each(shape)
    return flat.to_py_value()


def _tv_cells(
    layout,
    rows: int,
    cols: int,
    color: Callable[[int, int], str],
):
    from .. import primitive as prim

    dims = _get_shape_dims(layout)
    T_size, V_size = dims[0], dims[1]
    cells: dict[tuple[int, int], tuple[str, str]] = {}
    filled = [[False] * cols for _ in range(rows)]
    linear = 0
    for tid in range(T_size):
        for vid in range(V_size):
            linear = tid + vid * T_size
            idx = prim.crd2idx(linear, layout).get_static_leaf_int
            r, c = idx % rows, idx // rows
            if not filled[r][c]:
                filled[r][c] = True
                cells[(r, c)] = (color(tid, vid), f"T{tid} #linebreak() V{vid}")
    return cells


def _tv_cells_B_top(
    layout,
    N: int,
    K: int,
    color: Callable[[int, int], str],
):
    from .. import primitive as prim

    dims = _get_shape_dims(layout)
    T_size, V_size = dims[0], dims[1]
    cells: dict[tuple[int, int], tuple[str, str]] = {}
    filled = [[False] * N for _ in range(K)]
    for tid in range(T_size):
        for vid in range(V_size):
            linear = tid + vid * T_size
            idx = prim.crd2idx(linear, layout).get_static_leaf_int
            n, k = idx % N, idx // N
            if not filled[k][n]:
                filled[k][n] = True
                cells[(k, n)] = (color(tid, vid), f"T{tid} #linebreak() V{vid}")
    return cells


def _typst_layout(
    layout,
    color: Callable[[int], str],
) -> str:
    from .. import primitive as prim

    if not layout.is_static:
        raise ValueError(f"print_typst requires a fully static layout, got {layout}")

    rank = layout.rank
    if rank > 2:
        raise ValueError(f"print_typst only supports rank <= 2, got {rank}")

    cells: dict[tuple[int, int], tuple[str, str]] = {}
    if rank <= 1:
        dims = _get_shape_dims(layout)
        M, N = 1, dims[0]
        for n in range(N):
            idx = prim.crd2idx(n, layout).get_static_leaf_int
            cells[(0, n)] = (color(idx), str(idx))
    else:
        dims = _get_shape_dims(layout)
        M, N = dims[0], dims[1]
        linear = 0
        for n in range(N):
            for m in range(M):
                idx = prim.crd2idx(linear, layout).get_static_leaf_int
                cells[(m, n)] = (color(idx), str(idx))
                linear += 1

    layout_str = str(layout)
    return _typst_text_panel([f"{layout_str}"]) + "\n\n" + _typst_grid_block(M, N, cells, title="Layout") + "\n"


def _typst_mma(
    mma,
    color: Callable[[int, int], str],
) -> str:
    tile_mnk = mma.tile_size_mnk
    M, N, K = tile_mnk.to_py_value()

    layout_C = mma.tv_layout_C_tiled
    layout_A = mma.tv_layout_A_tiled
    layout_B = mma.tv_layout_B_tiled

    cells_C = _tv_cells(layout_C, M, N, color)
    cells_A = _tv_cells(layout_A, M, K, color)
    cells_B = _tv_cells_B_top(layout_B, N, K, color)

    doc = _typst_text_panel(_tiled_mma_text_lines(mma))
    doc += "\n\n#grid(\n  columns: (auto, auto, auto),\n  rows: (auto, auto),\n  gutter: 12pt,\n  align: center + horizon,\n"
    doc += "  [],\n  [\n"
    doc += _typst_grid_block(K, N, cells_B, title="B (K x N)")
    doc += "\n  ],\n  [],\n  [\n"
    doc += _typst_grid_block(M, K, cells_A, title="A (M x K)")
    doc += "\n  ],\n  [\n"
    doc += _typst_grid_block(M, N, cells_C, title="C (M x N)")
    doc += "\n  ],\n  [],\n)\n"
    return doc


def _typst_copy(
    copy,
    color: Callable[[int, int], str],
) -> str:
    tile_mn = str(copy.tile_mn)
    # hacky way to extract the tile dimensions for now
    inner = tile_mn[tile_mn.index("[") + 1 : tile_mn.index("]")]
    m_str, n_str = inner.split("|")
    M = int(m_str.split(":")[0].strip())
    N = int(n_str.split(":")[0].strip())

    layout_src = copy.layout_src_tv_tiled
    layout_dst = copy.layout_dst_tv_tiled

    cells_S = _tv_cells(layout_src, M, N, color)
    cells_D = _tv_cells(layout_dst, M, N, color)

    doc = _typst_text_panel(_tiled_copy_text_lines(copy))
    doc += "\n\n#grid(\n  columns: (auto, 20pt, auto),\n  gutter: 0pt,\n"
    doc += "  [\n"
    doc += _typst_grid_block(M, N, cells_S, title="Src (M x N)")
    doc += "\n  ],\n  [],\n  [\n"
    doc += _typst_grid_block(M, N, cells_D, title="Dst (M x N)")
    doc += "\n  ],\n)\n"
    return doc


def print_typst(
    arg,
    *,
    color: Callable | None = None,
    file: str | None = None,
) -> None:
    """Print a layout visualization as Typst markup.

    Dispatches based on the type of *arg*:

    * **Layout / ComposedLayout** -- index grid coloured by linear index.
    * **TiledMma** -- LayoutABC text plus a B-over-C / A-left-of-C view.
    * **TiledCopy** -- side-by-side Src, Dst thread-value grids.

    Multiple calls to the same *file* are concatenated with ``#pagebreak()``
    separators; the Typst header is emitted only once.

    Args:
        arg: A static Layout, ComposedLayout, TiledMma, or TiledCopy.
        color: Optional colour function.  For Layout the signature is
            ``(idx) -> hex``; for TiledMma/TiledCopy it is
            ``(tid, vid) -> hex``.  Defaults to grayscale for Layout
            and pastel-by-thread for MMA/Copy.
        file: Output filename.  ``None`` -> ``sys.stdout``.
    """
    from ..typing import ComposedLayout, Layout, TiledCopy, TiledMma

    if isinstance(arg, (Layout, ComposedLayout)):
        body = _typst_layout(arg, color or _color_bw)
    elif isinstance(arg, TiledMma):
        body = _typst_mma(arg, color or _color_tv)
    elif isinstance(arg, TiledCopy):
        body = _typst_copy(arg, color or _color_tv)
    else:
        raise ValueError(
            f"print_typst expects Layout, ComposedLayout, TiledMma, or TiledCopy, got {type(arg).__name__}"
        )

    if file is None:
        print(_typst_header() + body, file=sys.stdout)
        return

    exists = os.path.isfile(file) and os.path.getsize(file) > 0
    with open(file, "a") as f:
        if not exists:
            f.write(_typst_header() + "\n")
        else:
            f.write("\n#pagebreak()\n\n")
        f.write(body)
