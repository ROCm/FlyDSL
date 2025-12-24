#!/usr/bin/env python3
"""Test the new flir coordinate operations (make_coord, crd2idx, idx2crd)"""

from pyflir.dialects.ext import arith, flir


class _CoordOps(flir.MlirModule):
    @flir.jit
    def coord_ops(self: flir.T.i64):
        i = arith.index(4)
        j = arith.index(7)
        coord_2d = flir.make_coord(i.value, j.value)

        m = arith.index(32)
        n = arith.index(64)
        one = arith.index(1)
        shape = flir.make_shape(m.value, n.value)
        stride = flir.make_stride(n.value, one.value)  # row-major: stride=(64,1)
        layout = flir.make_layout(shape, stride)

        linear_idx = flir.crd2idx(coord_2d, layout)
        idx_test = arith.index(263)
        coord_back = flir.idx2crd(idx_test.value, layout)

        k = arith.index(42)
        coord_1d = flir.make_coord(k.value)
        size_1d = arith.index(1024)
        stride_1d = arith.index(1)
        layout_1d = flir.make_layout(flir.make_shape(size_1d.value), flir.make_stride(stride_1d.value))
        idx_1d = flir.crd2idx(coord_1d, layout_1d)

        # Keep values alive in IR.
        return [
            linear_idx.value,
            flir.crd2idx(coord_back, layout).value,
            idx_1d.value,
        ]


def test_coord_operations():
    m = _CoordOps()
    s = str(m.module)
    assert "flir.make_coord" in s
    assert "flir.crd2idx" in s
    assert "flir.idx2crd" in s
