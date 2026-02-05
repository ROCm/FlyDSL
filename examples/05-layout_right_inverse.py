import flydsl
from flydsl import lang as fx


class RightInverseDemo(fx.MlirModule):
    def __init__(self):
        super().__init__()

    @fx.kernel
    def kernel(self: fx.T.i64()):
        layout = fx.make_layout((4, 2, 3), (3, 12, 1))
        inv = fx.right_inverse(layout)
        fx.print_op("right_inverse: {}", inv)

    @fx.jit
    def __call__(self: fx.T.i64()):
        x = fx.arith.constant(fx.T.i64(), 1)
        c1 = fx.arith.constant(fx.T.index(), 1)
        kernel_sym = fx.ir.SymbolRefAttr.get(["kernels", "kernel"])
        fx.LaunchFuncOp(
            kernel_sym,
            grid_size=[c1, c1, c1],
            block_size=[c1, c1, c1],
            kernel_operands=[x],
        )


RightInverseDemo_Module = RightInverseDemo()
print(RightInverseDemo_Module)

# Run pass pipeline and print IR after each pass.
RightInverseDemo_Executor = flydsl.compile(RightInverseDemo_Module, print_after_all=False)
