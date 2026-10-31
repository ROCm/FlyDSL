import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def recastUpcastDynamicKernel(dyn_stride: fx.Int32):
    # Case requested: (2, 3, 4):(1, 2, ?) with upcast factor N=2 (f16 -> f32).
    layout = fx.make_layout((2, 3, 4), (1, 2, dyn_stride))
    recast = fx.recast_layout(layout, 16, 32)
    fx.printf("input dynamic stride: {}", dyn_stride)
    fx.printf("before recast_layout: {}", layout)
    fx.printf("after recast_layout: {}", recast)


@fmc.jit
def runRecastUpcastDynamic(dyn_stride: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    recastUpcastDynamicKernel(dyn_stride).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


runRecastUpcastDynamic(7)
