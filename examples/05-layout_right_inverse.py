import flydsl.compiler as fmc
import flydsl.expr as fx


@fmc.kernel
def rightInverseKernel():
    layout = fx.make_layout((4, 2, 3), (3, 12, 1))
    inv = fx.right_inverse(layout)
    fx.printf("before right_inverse: {}", layout)
    fx.printf("after right_inverse: {}", inv)


@fmc.jit
def runRightInverse(stream: fx.Stream = fx.Stream(None)):
    rightInverseKernel().launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


runRightInverse()
