# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import itertools
import math

from ..._mlir.dialects.fly import GemmTraversalOrder


def _gemm_tile_coords(bounds, traversal_order, traversal_layout):
    """Match ExpandGemmOpLowering's static traversal, including serpentine parity."""
    from ..primitive import cosize, crd2idx, idx2crd, make_layout

    if traversal_layout is not None:
        total = math.prod(bounds)
        if cosize(traversal_layout).get_static_leaf_int != total:
            raise ValueError("traversal_layout cosize must equal the number of GEMM tiles")
        natural = make_layout(bounds, tuple(math.prod(bounds[:i]) for i in range(len(bounds))))
        for i in range(total):
            coord = idx2crd(crd2idx(i, traversal_layout), natural)
            yield tuple(coord[j].get_static_leaf_int for j in range(len(bounds)))
        return

    name = "NMK" if traversal_order is None else GemmTraversalOrder(traversal_order).name
    serpentine = name.endswith("_Serpentine")
    # Enum letters go from innermost to outermost; rank-2 operands omit K.
    order = ["MNK".index(dim) for dim in reversed(name.split("_")[0]) if "MNK".index(dim) < len(bounds)]
    for indices in itertools.product(*(range(bounds[dim]) for dim in order)):
        coord = [0] * len(bounds)
        outer_iteration = 0
        for dim, index in zip(order, indices):
            coord[dim] = bounds[dim] - 1 - index if serpentine and outer_iteration % 2 else index
            outer_iteration = outer_iteration * bounds[dim] + index
        yield tuple(coord)
