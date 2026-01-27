import flydsl
from flydsl import lang as fx

M = 16
N = 32
memrefTy = fx.ir.Type.parse(f"!fly.memref<f32, global, (16,32):(1,16)>")


class VecCopy(fx.MlirModule):
    def __init__(self, thr_dim, val_dim):
        super().__init__()

    @fx.kernel
    def kernel(
        self: fx.T.i64(),
        A: memrefTy,
        B: memrefTy,
    ):
        tid = fx.arith.index_cast(fx.T.i32(), fx.thread_idx.x)
        bid = fx.arith.index_cast(fx.T.i32(), fx.block_idx.x)

        print(type(tid), tid)

        l16 = fx.make_layout(16, 1)
        tile = fx.make_tile([l16, l16])

        tA = fx.logical_divide(A, tile)
        tB = fx.logical_divide(B, tile)

        tA = fx.zipped_divide(A, tile)
        tB = fx.zipped_divide(B, tile)

        tA = fx.slice(tA, ((None, None), bid))
        tB = fx.slice(tB, ((None, None), bid))

        vec = fx.memref_load(tA, tid)
        fx.memref_store(vec, tB, tid)

    @fx.jit
    def __call__(
        self: fx.T.i64(),
        A: memrefTy,
        B: memrefTy,
    ):
        x = fx.arith.constant(fx.T.i64(), 16)
        c1 = fx.arith.constant(fx.T.index(), 1)
        c256 = fx.arith.constant(fx.T.index(), 256)
        gN = fx.arith.constant(fx.T.index(), N // 16)

        kernel_sym = fx.ir.SymbolRefAttr.get(["kernels", "kernel"])
        fx.LaunchFuncOp(
            kernel_sym,
            grid_size=[gN, c1, c1],
            block_size=[c256, c1, c1],
            kernel_operands=[x, A, B],
        )


ThrPerBlock = 256
ValPerThr = 8

VecCopy_Module = VecCopy(thr_dim=ThrPerBlock, val_dim=ValPerThr)
print(VecCopy_Module)

VecCopy_Executor = flydsl.compile(VecCopy_Module, print_after_all=False)
