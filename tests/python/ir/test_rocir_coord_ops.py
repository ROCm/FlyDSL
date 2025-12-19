#!/usr/bin/env python3
"""Test rocir coordinate operations (make_coord, crd2idx, idx2crd) (Flyx style)."""

from rocdsl.dialects.ext import arith, rocir


class _CoordOps(rocir.MlirModule):
    @rocir.jit
    def coord_ops(self: rocir.T.i64):
        i = arith.index(4)
        j = arith.index(7)
        coord_2d = rocir.make_coord(i.value, j.value)

        m = arith.index(32)
        n = arith.index(64)
        one = arith.index(1)
        shape = rocir.make_shape(m.value, n.value)
        stride = rocir.make_stride(n.value, one.value)  # row-major: stride=(64,1)
        layout = rocir.make_layout(shape, stride)

        linear_idx = rocir.crd2idx(coord_2d, layout)
        idx_test = arith.index(263)
        coord_back = rocir.idx2crd(idx_test.value, layout)

        k = arith.index(42)
        coord_1d = rocir.make_coord(k.value)
        size_1d = arith.index(1024)
        stride_1d = arith.index(1)
        layout_1d = rocir.make_layout(rocir.make_shape(size_1d.value), rocir.make_stride(stride_1d.value))
        idx_1d = rocir.crd2idx(coord_1d, layout_1d)

        # Keep values alive in IR.
        return [
            linear_idx.value,
            rocir.crd2idx(coord_back, layout).value,
            idx_1d.value,
        ]


def test_coord_operations():
    m = _CoordOps()
    s = str(m.module)
    assert "rocir.make_coord" in s
    assert "rocir.crd2idx" in s
    assert "rocir.idx2crd" in s
