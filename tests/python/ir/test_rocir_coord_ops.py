#!/usr/bin/env python3
"""Test the new flir coordinate operations (make_coord, crd2idx, idx2crd)"""

from pyflir.dialects.ext import arith, flir


class _CoordOps(flir.MlirModule):
    @flir.jit
    def coord_ops(self: flir.T.i64):
        i = arith.index(4)
        j = arith.index(7)
        coord_2d = flir.make_coord(arith.unwrap(i), arith.unwrap(j))

        m = arith.index(32)
        n = arith.index(64)
        one = arith.index(1)
        shape = flir.make_shape(arith.unwrap(m), arith.unwrap(n))
        stride = flir.make_stride(arith.unwrap(n), arith.unwrap(one))  # row-major: stride=(64,1)
        layout = flir.make_layout(shape, stride)

        linear_idx = flir.crd2idx(coord_2d, layout)
        idx_test = arith.index(263)
        coord_back = flir.idx2crd(arith.unwrap(idx_test), layout)

        k = arith.index(42)
        coord_1d = flir.make_coord(arith.unwrap(k))
        size_1d = arith.index(1024)
        stride_1d = arith.index(1)
        layout_1d = flir.make_layout(
            flir.make_shape(arith.unwrap(size_1d)),
            flir.make_stride(arith.unwrap(stride_1d)),
        )
        idx_1d = flir.crd2idx(coord_1d, layout_1d)

        # Keep values alive in IR.
        return [
            arith.unwrap(linear_idx),
            arith.unwrap(flir.crd2idx(coord_back, layout)),
            arith.unwrap(idx_1d),
        ]


def test_coord_operations():
    m = _CoordOps()
    s = str(m.module)
    assert "flir.make_coord" in s
    assert "flir.crd2idx" in s
    assert "flir.idx2crd" in s
