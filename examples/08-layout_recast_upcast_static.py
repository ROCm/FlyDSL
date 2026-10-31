import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def recastUpcastStaticKernel():
    # Static case: (2, 3, 4):(1, 2, 4), recast f16 -> f32 (upcast N=2).
    layout = fx.make_layout((2, 3, 4), (1, 2, 4))
    recast = fx.recast_layout(layout, 16, 32)
    fx.printf("before recast_layout: {}", layout)
    fx.printf("after recast_layout: {}", recast)


@fmc.jit
def runRecastUpcastStatic(stream: fx.Stream = fx.Stream(None)):
    recastUpcastStaticKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


runRecastUpcastStatic()
