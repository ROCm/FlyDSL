"""Test basic Rocir operations: make_shape, make_stride, make_layout, size, rank, etc."""

import pytest
from rocdsl.dialects.ext import arith, rocir


class _BasicOps(rocir.MlirModule):
    @rocir.jit
    def shape_rank2(self: rocir.T.i64):
        c8 = 8
        c16 = 16
        shape = rocir.make_shape(c8, c16)
        rank = rocir.rank(shape)
        return rank.value

    @rocir.jit
    def layout_creation(self: rocir.T.i64):
        c8 = 8
        c16 = 16
        c1 = 1
        shape = rocir.make_shape(c8, c16)
        stride = rocir.make_stride(c1, c8)
        layout = rocir.make_layout(shape, stride)
        return [layout]

    @rocir.jit
    def size(self: rocir.T.i64):
        c4 = 4
        c8 = 8
        shape = rocir.make_shape(c4, c8)
        total_size = rocir.size(shape)
        return total_size.value

    @rocir.jit
    def extract(self: rocir.T.i64):
        c8 = 8
        c16 = 16
        c1 = 1
        shape = rocir.make_shape(c8, c16)
        stride = rocir.make_stride(c1, c8)
        layout = rocir.make_layout(shape, stride)
        extracted_shape = rocir.get_shape(layout)
        extracted_stride = rocir.get_stride(layout)
        size_val = rocir.size(extracted_shape)
        return size_val.value

    @rocir.jit
    def rank(self: rocir.T.i64):
        c2 = 2
        c3 = 3
        c4 = 4
        shape = rocir.make_shape(c2, c3, c4)
        rank_val = rocir.rank(shape)
        return rank_val.value

    @rocir.jit
    def cosize(self: rocir.T.i64):
        c8 = 8
        c16 = 16
        c1 = 1
        shape = rocir.make_shape(c8, c16)
        stride = rocir.make_stride(c1, c8)
        layout = rocir.make_layout(shape, stride)
        cosize_val = rocir.cosize(layout)
        return cosize_val.value

    @rocir.jit
    def compose(self: rocir.T.i64):
        c8 = 8
        c16 = 16
        c4 = 4
        c2 = 2
        c1 = 1
        shape_a = rocir.make_shape(c8, c16)
        stride_a = rocir.make_stride(c1, c8)
        layout_a = rocir.make_layout(shape_a, stride_a)
        shape_b = rocir.make_shape(c4, c2)
        stride_b = rocir.make_stride(c2, c1)
        layout_b = rocir.make_layout(shape_b, stride_b)
        composed = rocir.composition(layout_a, layout_b)
        return [composed]


def test_make_shape():
    """Test creating shapes with different ranks."""
    m = _BasicOps()
    assert "rocir.make_shape" in str(m.module)


def test_make_layout():
    """Test creating layouts from shape and stride."""
    m = _BasicOps()
    assert "rocir.make_layout" in str(m.module)


def test_size_operation():
    """Test size computation for shapes and layouts."""
    m = _BasicOps()
    assert "rocir.size" in str(m.module)


def test_get_shape_stride():
    """Test extracting shape and stride from layout."""
    m = _BasicOps()
    s = str(m.module)
    assert "rocir.get_shape" in s
    assert "rocir.get_stride" in s


def test_rank_operation():
    """Test rank operation on shapes and layouts."""
    m = _BasicOps()
    assert "rocir.rank" in str(m.module)

    
    


def test_cosize_operation():
    """Test cosize (stride extent) computation."""
    m = _BasicOps()
    assert "rocir.cosize" in str(m.module)

    
    


def test_composition():
    """Test layout composition with Pythonic operators."""
    m = _BasicOps()
    assert "rocir.composition" in str(m.module)

    
    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
