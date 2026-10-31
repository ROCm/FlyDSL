import sys
sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def test_for_index_stop(n: fx.Int32):
    idx_n = fx.Index(n)
    count = fx.Int32(0)
    for iv in range(fx.Int32(0), idx_n):
        count = count + fx.Int32(1)
    return count

test_for_index_stop(10)
