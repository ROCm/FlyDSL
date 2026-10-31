import sys
sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")

import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.jit
def test_ifexp(x: fx.Int32):
    a = 1 if x == 0 else 2
    fx.printf("a = {}\n", a)
    return a

test_ifexp(2)
