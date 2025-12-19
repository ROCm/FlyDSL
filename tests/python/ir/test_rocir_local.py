"""Tests for Rocir local operations (Flyx style)."""

from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index


def test_local_partition():
    class _M(rocir.MlirModule):
        @rocir.jit
        def local_partition(self: rocir.T.i64):
            global_layout = rocir.make_layout(
                rocir.make_shape(Index(128), Index(256)),
                rocir.make_stride(Index(1), Index(128)),
            )
            tile = rocir.make_layout(
                rocir.make_shape(Index(8), Index(16)),
                rocir.make_stride(Index(1), Index(8)),
            )
            thread_data = rocir.local_partition(global_layout, tile, Index(0))
            return [rocir.size(thread_data).value]

    s = str(_M().module)
    assert "rocir.local_partition" in s


def test_local_tile():
    class _M(rocir.MlirModule):
        @rocir.jit
        def local_tile(self: rocir.T.i64):
            global_layout = rocir.make_layout(
                rocir.make_shape(Index(128), Index(256)),
                rocir.make_stride(Index(1), Index(128)),
            )
            cta_shape = rocir.make_shape(Index(32), Index(64))
            cta_coord = rocir.make_shape(Index(0), Index(0))
            cta_tile = rocir.local_tile(global_layout, cta_shape, cta_coord)
            return [rocir.size(cta_tile).value]

    s = str(_M().module)
    assert "rocir.local_tile" in s


def test_composition():
    class _M(rocir.MlirModule):
        @rocir.jit
        def composition(self: rocir.T.i64):
            layout_a = rocir.make_layout(
                rocir.make_shape(Index(8), Index(16)),
                rocir.make_stride(Index(1), Index(8)),
            )
            layout_b = rocir.make_layout(
                rocir.make_shape(Index(4), Index(2)),
                rocir.make_stride(Index(2), Index(1)),
            )
            composed = rocir.composition(layout_a, layout_b)
            return [rocir.size(composed).value]

    s = str(_M().module)
    assert "rocir.composition" in s


def test_thread_block_hierarchy():
    class _M(rocir.MlirModule):
        @rocir.jit
        def hierarchy(self: rocir.T.i64):
            global_layout = rocir.make_layout(
                rocir.make_shape(Index(256), Index(512)),
                rocir.make_stride(Index(1), Index(256)),
            )
            block_layout = rocir.make_layout(
                rocir.make_shape(Index(16), Index(32)),
                rocir.make_stride(Index(1), Index(16)),
            )
            partitioned = rocir.local_partition(global_layout, block_layout, Index(0))
            tile_layout = rocir.make_layout(
                rocir.make_shape(Index(4), Index(8)),
                rocir.make_stride(Index(1), Index(4)),
            )
            tiled = rocir.local_tile(partitioned, tile_layout, Index(0))
            return [rocir.size(tiled).value]

    s = str(_M().module)
    assert "rocir.local_partition" in s
    assert "rocir.local_tile" in s


