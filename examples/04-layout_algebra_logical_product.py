import flydsl
from flydsl import lang as fx

# Logical product example:
# Per-thread layout A = (3,6):(6,1), block=(4,5) threads.
# A * tiler(4,5):(1,4) => layout B = ((3,6),(4,5)):((6,1),(18,72)).
# Each thread reads its (3,6) tile and stores into a contiguous output.

M = 3
N = 6
BlockX = 4
BlockY = 5
Stride1 = 6
Stride2 = 1

# Layout A (per-thread view) and layout B (after logical_product).
layoutA = fx.LayoutType.get((M, N), (Stride1, Stride2))
layoutB = fx.LayoutType.get(
    ((M, N), (BlockX, BlockY)),
    ((Stride1, Stride2), (M * N, M * N * 4)),
)

memrefTyB = fx.MemRefType.get(
    fx.T.f32(), layoutB, fx.AddressSpace.Global
)
memrefTyOut = fx.MemRefType.get(
    fx.T.f32(),
    fx.LayoutType.get(M * N * BlockX * BlockY, 1),
    fx.AddressSpace.Global,
)


class LogicalProductReorder(fx.MlirModule):
    def __init__(self):
        super().__init__()

    @fx.kernel
    def kernel(
        self: fx.T.i64(),
        X: memrefTyB,
        X_out: memrefTyOut,
    ):
        tid_x = fx.arith.index_cast(fx.T.i32(), fx.thread_idx.x)
        tid_y = fx.arith.index_cast(fx.T.i32(), fx.thread_idx.y)

        # Build layout A and tiler, then take logical_product to get layout B.
        layoutA_val = fx.make_layout((M, N), (Stride1, Stride2))
        tiler = fx.make_layout((BlockX, BlockY), (1, 4))
        layoutB_val = fx.raked_product(layoutA_val, tiler)

        # View original memory X with layout B and select this thread's slice.
        x_view = fx.make_view(fx.get_iter(X), layoutB_val)
        # logical_product keeps a nested structure: ((M, N), (BlockX, BlockY))
        # so slice coordinates must mirror that shape.
        x_tile = fx.slice(x_view, ((None, None), (tid_x, tid_y)))

        tmp = fx.arith.muli(tid_y, fx.arith.constant(fx.T.i32(), BlockX))
        tid = fx.arith.addi(tmp, tid_x)
        base = fx.arith.muli(tid, fx.arith.constant(fx.T.i32(), M * N))

        # For demo: store only one element from the tile.
        coord = fx.make_coord(0, 0)
        idx_in = fx.crd2idx(coord, fx.get_layout(x_tile))
        v = fx.memref_load(x_tile, idx_in)
        idx = fx.arith.addi(base, fx.arith.constant(fx.T.i32(), 0))
        fx.memref_store(v, X_out, idx)

    @fx.jit
    def __call__(
        self: fx.T.i64(),
        X: memrefTyB,
        X_out: memrefTyOut,
    ):
        x = fx.arith.constant(fx.T.i64(), 1)
        c1 = fx.arith.constant(fx.T.index(), 1)
        c4 = fx.arith.constant(fx.T.index(), BlockX)
        c5 = fx.arith.constant(fx.T.index(), BlockY)

        kernel_sym = fx.ir.SymbolRefAttr.get(["kernels", "kernel"])
        fx.LaunchFuncOp(
            kernel_sym,
            grid_size=[c1, c1, c1],
            block_size=[c4, c5, c1],
            kernel_operands=[x, X, X_out],
        )


LogicalProductReorder_Module = LogicalProductReorder()
print(LogicalProductReorder_Module)

LogicalProductReorder_Executor = flydsl.compile(
    LogicalProductReorder_Module, print_after_all=False
)
