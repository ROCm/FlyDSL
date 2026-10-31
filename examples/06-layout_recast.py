import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def recastLayoutKernel():
    layout = fx.make_layout((4, 2, 3), (3, 12, 1))
    recast = fx.recast_layout(layout, 32, 16)
    fx.printf("before recast_layout: {}", layout)
    fx.printf("after recast_layout: {}", recast)


@fmc.jit
def runRecastLayout(stream: fx.Stream = fx.Stream(None)):
    recastLayoutKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


runRecastLayout()

