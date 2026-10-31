import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def algebraStaticKernel():
    # Base layout and helper tilers.
    base = fx.make_layout((4, 8), (1, 4))
    div_tiler = fx.make_layout((2, 4), (1, 2))
    prod_tiler = fx.make_layout((2, 2), (1, 2))

    # Core algebra ops.
    composed = fx.composition(base, div_tiler)
    logical_prod = fx.logical_product(base, prod_tiler)
    logical_div = fx.logical_divide(base, div_tiler)
    right_inv = fx.right_inverse(base)

    # Also show divide variants to compare layout forms.
    zipped_div = fx.zipped_divide(base, div_tiler)
    tiled_div = fx.tiled_divide(base, div_tiler)
    flat_div = fx.flat_divide(base, div_tiler)

    fx.printf("=== Static Algebra Demo ===")
    fx.printf("base               : {}", base)
    fx.printf("div_tiler          : {}", div_tiler)
    fx.printf("prod_tiler         : {}", prod_tiler)
    fx.printf("composition        : {}", composed)
    fx.printf("logical_product    : {}", logical_prod)
    fx.printf("logical_divide     : {}", logical_div)
    fx.printf("zipped_divide      : {}", zipped_div)
    fx.printf("tiled_divide       : {}", tiled_div)
    fx.printf("flat_divide        : {}", flat_div)
    fx.printf("right_inverse      : {}", right_inv)


@fmc.kernel
def algebraDynamicKernel(dyn_stride: fx.Int32):
    # Dynamic stride in the last mode: (2,3,4):(1,2,dyn_stride)
    dyn_layout = fx.make_layout((2, 3, 4), (1, 2, dyn_stride))
    div_tiler = fx.make_layout((2, 2, 2), (1, 2, 4))
    # prod_tiler = fx.make_layout((2,), (1,))

    # complement_mid = fx.complement(dyn_layout, div_tiler)
    # fx.printf("complement(mid)    : {}", complement_mid)
    logical_div = fx.logical_divide(dyn_layout, div_tiler)
    # logical_prod = fx.logical_product(dyn_layout, prod_tiler)
    # right_inv = fx.right_inverse(dyn_layout)

    fx.printf("=== Dynamic Algebra Demo ===")
    fx.printf("dyn_stride         : {}", dyn_stride)
    fx.printf("dyn_layout         : {}", dyn_layout)
    fx.printf("div_tiler          : {}", div_tiler)
    # fx.printf("logical_divide     : {}", logical_div)
    # fx.printf("logical_product    : {}", logical_prod)
    # fx.printf("right_inverse      : {}", right_inv)


@fmc.jit
def runAlgebraStatic(stream: fx.Stream = fx.Stream(None)):
    algebraStaticKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


@fmc.jit
def runAlgebraDynamic(dyn_stride: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    algebraDynamicKernel(dyn_stride).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


if __name__ == "__main__":
    # runAlgebraStatic()
    runAlgebraDynamic(6)
