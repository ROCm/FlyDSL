from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index


def test_logical_product():
    class _M(rocir.MlirModule):
        @rocir.jit
        def logical_product(self: rocir.T.i64):
            base = rocir.make_layout(
                rocir.make_shape(Index(16), Index(32)),
                rocir.make_stride(Index(1), Index(16)),
            )
            tiler = rocir.make_layout(
                rocir.make_shape(Index(4), Index(8)),
                rocir.make_stride(Index(1), Index(4)),
            )
            tiled = rocir.logical_product(base, tiler)
            return [rocir.size(tiled).value]

    s = str(_M().module)
    assert "rocir.logical_product" in s


def test_zipped_product():
    class _M(rocir.MlirModule):
        @rocir.jit
        def zipped_product(self: rocir.T.i64):
            base = rocir.make_layout(
                rocir.make_shape(Index(8), Index(16)),
                rocir.make_stride(Index(1), Index(8)),
            )
            tiler = rocir.make_layout(
                rocir.make_shape(Index(2), Index(4)),
                rocir.make_stride(Index(1), Index(2)),
            )
            zipped = rocir.zipped_product(base, tiler)
            return [rocir.size(zipped).value]

    s = str(_M().module)
    assert "rocir.zipped_product" in s


def test_tiled_product():
    class _M(rocir.MlirModule):
        @rocir.jit
        def tiled_product(self: rocir.T.i64):
            base = rocir.make_layout(
                rocir.make_shape(Index(32), Index(64)),
                rocir.make_stride(Index(1), Index(32)),
            )
            tiler = rocir.make_layout(
                rocir.make_shape(Index(8), Index(16)),
                rocir.make_stride(Index(1), Index(8)),
            )
            tiled = rocir.tiled_product(base, tiler)
            return [rocir.size(tiled).value]

    s = str(_M().module)
    assert "rocir.tiled_product" in s


def test_flat_product():
    class _M(rocir.MlirModule):
        @rocir.jit
        def flat_product(self: rocir.T.i64):
            base = rocir.make_layout(
                rocir.make_shape(Index(16), Index(8)),
                rocir.make_stride(Index(1), Index(16)),
            )
            tiler = rocir.make_layout(
                rocir.make_shape(Index(4), Index(2)),
                rocir.make_stride(Index(1), Index(4)),
            )
            flat = rocir.flat_product(base, tiler)
            return [rocir.size(flat).value]

    s = str(_M().module)
    assert "rocir.flat_product" in s


def test_raked_product():
    class _M(rocir.MlirModule):
        @rocir.jit
        def raked_product(self: rocir.T.i64):
            base = rocir.make_layout(
                rocir.make_shape(Index(32), Index(32)),
                rocir.make_stride(Index(1), Index(32)),
            )
            raker = rocir.make_layout(
                rocir.make_shape(Index(4), Index(8)),
                rocir.make_stride(Index(1), Index(4)),
            )
            raked = rocir.raked_product(base, raker)
            return [rocir.size(raked).value]

    s = str(_M().module)
    assert "rocir.raked_product" in s


def test_blocked_product():
    class _M(rocir.MlirModule):
        @rocir.jit
        def blocked_product(self: rocir.T.i64):
            base = rocir.make_layout(
                rocir.make_shape(Index(64), Index(128)),
                rocir.make_stride(Index(1), Index(64)),
            )
            blocker = rocir.make_layout(
                rocir.make_shape(Index(16), Index(16)),
                rocir.make_stride(Index(1), Index(16)),
            )
            blocked = rocir.blocked_product(base, blocker)
            return [rocir.size(blocked).value]

    s = str(_M().module)
    assert "rocir.blocked_product" in s




