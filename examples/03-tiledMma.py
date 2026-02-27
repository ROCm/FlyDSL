import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

block_m = 8
block_n = 24
block_k = 4


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tileA = fx.make_tile([fx.make_layout(block_m, 1), fx.make_layout(block_k, 1)])
    tileB = fx.make_tile([fx.make_layout(block_n, 1), fx.make_layout(block_k, 1)])
    tileC = fx.make_tile([fx.make_layout(block_m, 1), fx.make_layout(block_n, 1)])

    bA = fx.zipped_divide(A, tileA)
    bB = fx.zipped_divide(B, tileB)
    bC = fx.zipped_divide(C, tileC)

    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))
    bC = fx.slice(bC, (None, bid))

    mma_atom = fx.make_mma_atom(fx.UniversalFMA(fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((4, 8, 1), (8, 1, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    partition_A = thr_mma.partition_A(bA)
    partition_B = thr_mma.partition_B(bB)
    partition_C = thr_mma.partition_C(bC)
    fx.gemm(mma_atom, partition_C, partition_A, partition_B, partition_C)


@flyc.jit
def tiledMma(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


M, N, K = block_m, block_n, block_k
A = torch.randn(M, K, dtype=torch.float32).cuda()
B = torch.randn(N, K, dtype=torch.float32).cuda()
C = torch.zeros(M, N, dtype=torch.float32).cuda()

tiledMma(A, B, C, stream=torch.cuda.Stream())

torch.cuda.synchronize()

expected = A @ B.T
is_correct = torch.allclose(C, expected, atol=1e-5, rtol=1e-5)
print("Result correct:", is_correct)
if not is_correct:
    print("Max diff:", (C - expected).abs().max().item())
    print("Expected:", expected)
    print("Got:", C)
