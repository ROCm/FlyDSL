import flydsl
from flydsl import lang as fx

MN = 16
K = 4
ABMemRefTy = fx.MemRefType.get(
    fx.T.f32(), fx.LayoutType.get((MN, K), (K, 1)), fx.AddressSpace.Global
)
CMemRefTy = fx.MemRefType.get(
    fx.T.f32(), fx.LayoutType.get((MN, MN), (1, MN)), fx.AddressSpace.Global
)


class MmaAtom(fx.MlirModule):
    def __init__(self):
        super().__init__()

    @fx.kernel
    def kernel(
        self: fx.T.i64(),
        A: ABMemRefTy,
        B: ABMemRefTy,
        C: CMemRefTy,
    ):
        tid = fx.arith.index_cast(fx.T.i32(), fx.thread_idx.x)

        rA = fx.memref_alloca(
            fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1)),
            fx.make_layout(1, 1),
        )
        rB = fx.memref_alloca(
            fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1)),
            fx.make_layout(1, 1),
        )

        copyAtom = fx.make_atom(fx.CopyAtomUniversalCopyType.get(32))
        mmaAtom = fx.make_atom(
            fx.MmaAtomCDNA3_MFMAType.get(16, 16, 16, fx.T.f32(), fx.T.f32(), fx.T.f32())
        )

        tA = fx.logical_divide(A, fx.make_layout(1, 1))
        tB = fx.logical_divide(B, fx.make_layout(1, 1))
        fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
        fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

        rAcc = fx.memref_alloca(
            fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(4, 1)), fx.make_layout(4, 1)
        )
        f0 = fx.arith.constant(fx.T.f32(), 0.0)
        fx.memref_store(f0, rAcc, 0)
        fx.memref_store(f0, rAcc, 1)
        fx.memref_store(f0, rAcc, 2)
        fx.memref_store(f0, rAcc, 3)
        fx.mma_atom_call(mmaAtom, rAcc, rA, rB, rAcc)

        tC = fx.zipped_divide(
            C, fx.make_tile([fx.make_layout(4, 1), fx.make_layout(1, 1)])
        )
        permutation_tile = fx.make_tile([fx.make_layout(1, 1), fx.make_layout(16, 4)])
        tC = fx.logical_divide(tC, permutation_tile)

        fx.copy_atom_call(copyAtom, rAcc, fx.slice(tC, (None, tid)))

    @fx.jit
    def __call__(
        self: fx.T.i64(),
        A: ABMemRefTy,
        B: ABMemRefTy,
        C: CMemRefTy,
    ):
        c1 = fx.arith.constant(fx.T.index(), 1)
        c64 = fx.arith.constant(fx.T.index(), 64)

        kernel_sym = fx.ir.SymbolRefAttr.get(["kernels", "kernel"])
        fx.LaunchFuncOp(
            kernel_sym,
            grid_size=[c1, c1, c1],
            block_size=[c64, c1, c1],
            kernel_operands=[A, B, C],  # kernel's self is skipped
        )


MmaAtom_Module = MmaAtom()
print(MmaAtom_Module)

MmaAtom_Executor = flydsl.compile(MmaAtom_Module, print_after_all=False)
# MmaAtom_Asm = flydsl.compile(MmaAtom_Module, output_format="assembly")
# print(MmaAtom_Asm)

import torch

tA = torch.randint(
    0,
    10,
    (MN, K),
    dtype=torch.float32,
    device="cuda",
)
tB = torch.randint(
    0,
    10,
    (MN, K),
    dtype=torch.float32,
    device="cuda",
)
tC = torch.empty(
    (MN, MN),
    dtype=torch.float32,
    device="cuda",
)
tC_ref = tA @ tB.T

MmaAtom_Executor(tA, tB, tC)
is_closed = torch.allclose(tC.T, tC_ref)
print("Result correct:", is_closed)

if not is_closed:
    print("tA:", tA)
    print("tB:", tB)
    print("tC:", tC.T)
    print("tC:", tC_ref)

print("Hello, Fly!")
