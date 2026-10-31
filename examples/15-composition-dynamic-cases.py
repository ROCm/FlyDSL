import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def cp_dyn_lhs_stride_kernel(v: fx.Int32):
    lhs = fx.make_layout(16, v)
    rhs = fx.make_layout((2, 8), (8, 1))
    fx.printf("composition/dyn(lhs.stride)       : {}", fx.composition(lhs, rhs))


# @fmc.kernel
# def cp_dyn_rhs_shape_kernel(v: fx.Int32):
#     lhs = fx.make_layout(16, 1)
#     rhs = fx.make_layout((2, v), (8, 1))
#     fx.printf("composition/dyn(rhs.shape)        : {}", fx.composition(lhs, rhs))


@fmc.jit
def run_cp_dyn_lhs_stride(v: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    cp_dyn_lhs_stride_kernel(v).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


# @fmc.jit
# def run_cp_dyn_rhs_shape(v: fx.Int32, stream: fx.Stream = fx.Stream(None)):
#     cp_dyn_rhs_shape_kernel(v).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


if __name__ == "__main__":
    run_cp_dyn_lhs_stride(1)
    # run_cp_dyn_rhs_shape(8)
