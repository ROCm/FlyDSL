import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def recastDowncastStaticKernel():
    # Static downcast case: (2, 3, 4):(1, 2, 4), recast f32 -> f16 (N=2).
    layout = fx.make_layout((2, 3, 4), (1, 2, 4))
    recast = fx.recast_layout(layout, 32, 16)
    fx.printf("before recast_layout: {}", layout)
    fx.printf("after recast_layout: {}", recast)


@fmc.jit
def runRecastDowncastStatic(stream: fx.Stream = fx.Stream(None)):
    recastDowncastStaticKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


runRecastDowncastStatic()
