import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def logicalProductStaticKernel():
    inp = fx.make_layout((4, 2), (1, 4))
    tiler = fx.make_layout((2, 2), (1, 2))
    out = fx.logical_product(inp, tiler)

    fx.printf("=== logical_product static demo ===")
    fx.printf("input layout   : {}", inp)
    fx.printf("tiler layout   : {}", tiler)
    fx.printf("product layout : {}", out)


@fmc.jit
def runLogicalProductStatic(stream: fx.Stream = fx.Stream(None)):
    logicalProductStaticKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


if __name__ == "__main__":
    runLogicalProductStatic()
