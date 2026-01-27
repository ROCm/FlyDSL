import flydsl
from flydsl import lang as fx

N = 64
memrefTy = fx.MemRefType.get(
    fx.T.f32(), fx.LayoutType.get(64, 1), fx.AddressSpace.Global
)


class VecAdd(fx.MlirModule):
    def __init__(self):
        super().__init__()

    @fx.kernel
    def kernel(
        self: fx.T.i64(),
        A: memrefTy,
        B: memrefTy,
        C: memrefTy,
    ):
        tid = fx.arith.index_cast(fx.T.i32(), fx.thread_idx.x)
        bid = fx.arith.index_cast(fx.T.i32(), fx.block_idx.x)

        tA = fx.logical_divide(A, fx.make_layout(16, 1))
        tB = fx.logical_divide(B, fx.make_layout(16, 1))
        tC = fx.logical_divide(C, fx.make_layout(16, 1))

        tA = fx.slice(tA, (None, bid))
        tB = fx.slice(tB, (None, bid))
        tC = fx.slice(tC, (None, bid))
        tA = fx.logical_divide(tA, fx.make_layout(1, 1))
        tB = fx.logical_divide(tB, fx.make_layout(1, 1))
        tC = fx.logical_divide(tC, fx.make_layout(1, 1))

        RABMemRefTy = fx.MemRefType.get(
            fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        copyAtom = fx.make_atom(fx.CopyAtomUniversalCopyType.get(32))
        rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
        rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
        rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))

        fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
        fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

        vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
        fx.memref_store_vec(vC, rC)

        fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

    @fx.jit
    def __call__(
        self: fx.T.i64(),
        A: memrefTy,
        B: memrefTy,
        C: memrefTy,
    ):
        size = fx.size(A)

        size = fx.get_scalar(size)

        x = fx.arith.constant(fx.T.i64(), 16)
        c1 = fx.arith.constant(fx.T.index(), 1)
        c16 = fx.arith.constant(fx.T.index(), 16)

        gN = fx.arith.ceildivsi(size, fx.arith.constant(fx.T.i32(), 16))
        gN = fx.arith.IndexCastOp(fx.T.index(), gN)

        kernel_sym = fx.ir.SymbolRefAttr.get(["kernels", "kernel"])
        fx.LaunchFuncOp(
            kernel_sym,
            grid_size=[gN, c1, c1],
            block_size=[c16, c1, c1],
            kernel_operands=[x, A, B, C],
        )


VecAdd_Module = VecAdd()
print(VecAdd_Module)


VecAdd_Executor = flydsl.compile(VecAdd_Module, print_after_all=False)
# VecAdd_Asm = flydsl.compile(VecAdd_Module, output_format="assembly")
# print(VecAdd_Asm)

import torch

tA = torch.randint(0, 10, (N,), dtype=torch.float32, device="cuda")

tB = torch.randint(0, 10, (N,), dtype=torch.float32, device="cuda")
tC = torch.randint(0, 10, (N,), dtype=torch.float32, device="cuda")

tAmk = torch.randint(0, 10, (N, N), dtype=torch.float32, device="cuda")

VecAdd_Executor(tA, tB, tC)
is_closed = torch.allclose(tC, tA + tB)
print("Result correct:", is_closed)


if not is_closed:
    print("tA:", tA[:32])
    print("tB:", tB[:32])
    print("tC:", tC[:32])


print("Hello, Fly!")
