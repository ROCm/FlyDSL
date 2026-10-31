import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def logicalProductDynInShapeKernel(v: fx.Int32):
    inp = fx.make_layout((v, 2), (1, 4))
    tiler = fx.make_layout((2, 2), (1, 2))
    fx.printf("logical_product/dyn(in.shape)     : {}", fx.logical_product(inp, tiler))


@fmc.jit
def runLogicalProductDynInShape(v: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    logicalProductDynInShapeKernel(v).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


if __name__ == "__main__":
    runLogicalProductDynInShape(4)
