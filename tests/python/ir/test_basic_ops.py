"""Test basic Flir operations: make_shape, make_stride, make_layout, size, rank, etc."""

import pytest
from pyflir.dialects.ext import arith, flir


class _BasicOps(flir.MlirModule):
    @flir.jit
    def shape_rank2(self: flir.T.i64):
        c8 = 8
        c16 = 16
        shape = flir.make_shape(c8, c16)
        rank = flir.rank(shape)
        return rank.value

    @flir.jit
    def layout_creation(self: flir.T.i64):
        c8 = 8
        c16 = 16
        c1 = 1
        shape = flir.make_shape(c8, c16)
        stride = flir.make_stride(c1, c8)
        layout = flir.make_layout(shape, stride)
        return [layout]

    @flir.jit
    def size(self: flir.T.i64):
        c4 = 4
        c8 = 8
        shape = flir.make_shape(c4, c8)
        total_size = flir.size(shape)
        return total_size.value

    @flir.jit
    def extract(self: flir.T.i64):
        c8 = 8
        c16 = 16
        c1 = 1
        shape = flir.make_shape(c8, c16)
        stride = flir.make_stride(c1, c8)
        layout = flir.make_layout(shape, stride)
        extracted_shape = flir.get_shape(layout)
        extracted_stride = flir.get_stride(layout)
        size_val = flir.size(extracted_shape)
        return size_val.value

    @flir.jit
    def rank(self: flir.T.i64):
        c2 = 2
        c3 = 3
        c4 = 4
        shape = flir.make_shape(c2, c3, c4)
        rank_val = flir.rank(shape)
        return rank_val.value

    @flir.jit
    def cosize(self: flir.T.i64):
        c8 = 8
        c16 = 16
        c1 = 1
        shape = flir.make_shape(c8, c16)
        stride = flir.make_stride(c1, c8)
        layout = flir.make_layout(shape, stride)
        cosize_val = flir.cosize(layout)
        return cosize_val.value

    @flir.jit
    def compose(self: flir.T.i64):
        c8 = 8
        c16 = 16
        c4 = 4
        c2 = 2
        c1 = 1
        shape_a = flir.make_shape(c8, c16)
        stride_a = flir.make_stride(c1, c8)
        layout_a = flir.make_layout(shape_a, stride_a)
        shape_b = flir.make_shape(c4, c2)
        stride_b = flir.make_stride(c2, c1)
        layout_b = flir.make_layout(shape_b, stride_b)
        composed = flir.composition(layout_a, layout_b)
        return [composed]


def test_make_shape():
    """Test creating shapes with different ranks."""
    m = _BasicOps()
    assert "flir.make_shape" in str(m.module)


def test_make_layout():
    """Test creating layouts from shape and stride."""
    m = _BasicOps()
    assert "flir.make_layout" in str(m.module)


def test_size_operation():
    """Test size computation for shapes and layouts."""
    m = _BasicOps()
    assert "flir.size" in str(m.module)


def test_get_shape_stride():
    """Test extracting shape and stride from layout."""
    m = _BasicOps()
    s = str(m.module)
    assert "flir.get_shape" in s
    assert "flir.get_stride" in s


def test_rank_operation():
    """Test rank operation on shapes and layouts."""
    m = _BasicOps()
    assert "flir.rank" in str(m.module)

    
    


def test_cosize_operation():
    """Test cosize (stride extent) computation."""
    m = _BasicOps()
    assert "flir.cosize" in str(m.module)

    
    


def test_composition():
    """Test layout composition with Pythonic operators."""
    m = _BasicOps()
    assert "flir.composition" in str(m.module)

    
    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
