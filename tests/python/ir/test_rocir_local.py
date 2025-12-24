from pyflir.dialects.ext import flir
from pyflir.dialects.ext.arith import Index


def test_local_partition():
    class _M(flir.MlirModule):
        @flir.jit
        def local_partition(self: flir.T.i64):
            global_layout = flir.make_layout(
                flir.make_shape(Index(128), Index(256)),
                flir.make_stride(Index(1), Index(128)),
            )
            tile = flir.make_layout(
                flir.make_shape(Index(8), Index(16)),
                flir.make_stride(Index(1), Index(8)),
            )
            thread_data = flir.local_partition(global_layout, tile, Index(0))
            return [flir.size(thread_data).value]

    s = str(_M().module)
    assert "flir.local_partition" in s


def test_local_tile():
    class _M(flir.MlirModule):
        @flir.jit
        def local_tile(self: flir.T.i64):
            global_layout = flir.make_layout(
                flir.make_shape(Index(128), Index(256)),
                flir.make_stride(Index(1), Index(128)),
            )
            cta_shape = flir.make_shape(Index(32), Index(64))
            cta_coord = flir.make_shape(Index(0), Index(0))
            cta_tile = flir.local_tile(global_layout, cta_shape, cta_coord)
            return [flir.size(cta_tile).value]

    s = str(_M().module)
    assert "flir.local_tile" in s


def test_composition():
    class _M(flir.MlirModule):
        @flir.jit
        def composition(self: flir.T.i64):
            layout_a = flir.make_layout(
                flir.make_shape(Index(8), Index(16)),
                flir.make_stride(Index(1), Index(8)),
            )
            layout_b = flir.make_layout(
                flir.make_shape(Index(4), Index(2)),
                flir.make_stride(Index(2), Index(1)),
            )
            composed = flir.composition(layout_a, layout_b)
            return [flir.size(composed).value]

    s = str(_M().module)
    assert "flir.composition" in s


def test_thread_block_hierarchy():
    class _M(flir.MlirModule):
        @flir.jit
        def hierarchy(self: flir.T.i64):
            global_layout = flir.make_layout(
                flir.make_shape(Index(256), Index(512)),
                flir.make_stride(Index(1), Index(256)),
            )
            block_layout = flir.make_layout(
                flir.make_shape(Index(16), Index(32)),
                flir.make_stride(Index(1), Index(16)),
            )
            partitioned = flir.local_partition(global_layout, block_layout, Index(0))
            tile_layout = flir.make_layout(
                flir.make_shape(Index(4), Index(8)),
                flir.make_stride(Index(1), Index(4)),
            )
            tiled = flir.local_tile(partitioned, tile_layout, Index(0))
            return [flir.size(tiled).value]

    s = str(_M().module)
    assert "flir.local_partition" in s
    assert "flir.local_tile" in s




