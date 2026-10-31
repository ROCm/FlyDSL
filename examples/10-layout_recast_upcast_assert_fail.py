import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def recastUpcastAssertFailKernel():
    # Upcast factor N=2 (f16 -> f32), but stride=3 violates:
    #   absStride % N == 0 || N % absStride == 0
    # so this should trigger the assert in layoutUpcastImpl.
    layout = fx.make_layout((2,), (3,))
    recast = fx.recast_layout(layout, 16, 32)
    fx.printf("before recast_layout: {}", layout)
    fx.printf("after recast_layout: {}", recast)


@fmc.jit
def runRecastUpcastAssertFail(stream: fx.Stream = fx.Stream(None)):
    recastUpcastAssertFailKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


runRecastUpcastAssertFail()
