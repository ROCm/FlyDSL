import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))

    RABMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    copyAtom = fx.make_copy_atom(fx.CopyOpUniversalCopyType.get(32), fx.Float32)
    rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vectorAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,  # omitted for auto induction
    n: fx.Int32,  # dynamic int32
    const_n: fx.Constexpr[int],  # static int32, it has an effect on function cache-key
    stream: fx.Stream = fx.Stream(None),
):
    print("> Compile: n={} const_n={}", n, const_n)
    fx.printf("> Runtime: n={} const_n={}", n, const_n)

    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim

    vectorAddKernel(A, B, C, block_dim).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)


n = 128
A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
C = torch.zeros(n, dtype=torch.float32).cuda()

tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
vectorAdd(tA, B, C, n, n + 1, stream=torch.cuda.Stream())

torch.cuda.synchronize()

is_closed = torch.allclose(C, A + B)
print("Result correct:", is_closed)
if not is_closed:
    print("tA:", A[:32])
    print("tB:", B[:32])
    print("tC:", C[:32])
print("Hello, Fly!")
