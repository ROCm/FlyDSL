# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Generate Typst visualizations for Layout/ComposedLayout, TiledMma, and TiledCopy.
After the Typst source is written, run:

    typst compile layout.typ

to produce the corresponding PDF document.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

OUTPUT_TYPST = "layout.typ"


@flyc.jit
def visualize():
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16)),
        fx.make_layout((1, 2, 1), (0, 1, 0)),
        fx.make_tile(16, 32, fx.make_layout((4, 4, 2), (1, 8, 4))),
    )

    swizzle_layout = fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 0, 3)),
        fx.make_ordered_layout((8, 8), (1, 0)),
    )

    tiled_copy = fx.make_tiled_copy_A(
        fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float16),
        tiled_mma,
    )

    fx.utils.print_typst(tiled_mma, file=OUTPUT_TYPST)
    fx.utils.print_typst(tiled_copy, file=OUTPUT_TYPST)
    fx.utils.print_typst(swizzle_layout, file=OUTPUT_TYPST)


visualize()

print(f"Wrote to {OUTPUT_TYPST}")
