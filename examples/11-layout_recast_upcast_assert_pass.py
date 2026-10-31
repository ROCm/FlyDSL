import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def recastUpcastAssertPassKernel():
    # Upcast factor N=2 (f16 -> f32), stride=4 satisfies divisibility condition.
    layout = fx.make_layout((2,), (4,))
    recast = fx.recast_layout(layout, 16, 32)
    fx.printf("before recast_layout: {}", layout)
    fx.printf("after recast_layout: {}", recast)


@fmc.jit
def runRecastUpcastAssertPass(stream: fx.Stream = fx.Stream(None)):
    recastUpcastAssertPassKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


runRecastUpcastAssertPass()
