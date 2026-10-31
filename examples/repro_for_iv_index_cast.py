import sys
sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")

import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.jit
def test_for_iv(start: fx.Int32, stop: fx.Int32):
    count = fx.Int32(0)
    for iv in range(start, stop):
        count = iv
    return count

test_for_iv(0, 10)
