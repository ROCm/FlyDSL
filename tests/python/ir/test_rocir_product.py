from pyflir.dialects.ext import flir
from pyflir.dialects.ext.arith import Index


def test_logical_product():
    class _M(flir.MlirModule):
        @flir.jit
        def logical_product(self: flir.T.i64):
            base = flir.make_layout(
                flir.make_shape(Index(16), Index(32)),
                flir.make_stride(Index(1), Index(16)),
            )
            tiler = flir.make_layout(
                flir.make_shape(Index(4), Index(8)),
                flir.make_stride(Index(1), Index(4)),
            )
            tiled = flir.logical_product(base, tiler)
            return [flir.size(tiled).value]

    s = str(_M().module)
    assert "flir.logical_product" in s


def test_zipped_product():
    class _M(flir.MlirModule):
        @flir.jit
        def zipped_product(self: flir.T.i64):
            base = flir.make_layout(
                flir.make_shape(Index(8), Index(16)),
                flir.make_stride(Index(1), Index(8)),
            )
            tiler = flir.make_layout(
                flir.make_shape(Index(2), Index(4)),
                flir.make_stride(Index(1), Index(2)),
            )
            zipped = flir.zipped_product(base, tiler)
            return [flir.size(zipped).value]

    s = str(_M().module)
    assert "flir.zipped_product" in s


def test_tiled_product():
    class _M(flir.MlirModule):
        @flir.jit
        def tiled_product(self: flir.T.i64):
            base = flir.make_layout(
                flir.make_shape(Index(32), Index(64)),
                flir.make_stride(Index(1), Index(32)),
            )
            tiler = flir.make_layout(
                flir.make_shape(Index(8), Index(16)),
                flir.make_stride(Index(1), Index(8)),
            )
            tiled = flir.tiled_product(base, tiler)
            return [flir.size(tiled).value]

    s = str(_M().module)
    assert "flir.tiled_product" in s


def test_flat_product():
    class _M(flir.MlirModule):
        @flir.jit
        def flat_product(self: flir.T.i64):
            base = flir.make_layout(
                flir.make_shape(Index(16), Index(8)),
                flir.make_stride(Index(1), Index(16)),
            )
            tiler = flir.make_layout(
                flir.make_shape(Index(4), Index(2)),
                flir.make_stride(Index(1), Index(4)),
            )
            flat = flir.flat_product(base, tiler)
            return [flir.size(flat).value]

    s = str(_M().module)
    assert "flir.flat_product" in s


def test_raked_product():
    class _M(flir.MlirModule):
        @flir.jit
        def raked_product(self: flir.T.i64):
            base = flir.make_layout(
                flir.make_shape(Index(32), Index(32)),
                flir.make_stride(Index(1), Index(32)),
            )
            raker = flir.make_layout(
                flir.make_shape(Index(4), Index(8)),
                flir.make_stride(Index(1), Index(4)),
            )
            raked = flir.raked_product(base, raker)
            return [flir.size(raked).value]

    s = str(_M().module)
    assert "flir.raked_product" in s


def test_blocked_product():
    class _M(flir.MlirModule):
        @flir.jit
        def blocked_product(self: flir.T.i64):
            base = flir.make_layout(
                flir.make_shape(Index(64), Index(128)),
                flir.make_stride(Index(1), Index(64)),
            )
            blocker = flir.make_layout(
                flir.make_shape(Index(16), Index(16)),
                flir.make_stride(Index(1), Index(16)),
            )
            blocked = flir.blocked_product(base, blocker)
            return [flir.size(blocked).value]

    s = str(_M().module)
    assert "flir.blocked_product" in s




