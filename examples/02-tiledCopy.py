import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    block_m = 8
    block_n = 24
    tile = fx.make_tile([fx.make_layout(block_m, 1), fx.make_layout(block_n, 1)])

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)

    bA = fx.zipped_divide(A, tile)
    bB = fx.zipped_divide(B, tile)
    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))

    thr_layout = fx.make_layout((4, 1), (1, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferLDST128b(), fx.Float32)
    layout_thr_val = fx.logical_product(thr_layout, val_layout)
    layout_thr_val = fx.raked_product(thr_layout, val_layout)
    tile_mn = fx.make_tile([fx.make_layout(4, 1), fx.make_layout(8, 1)])

    tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    partition_src = thr_copy.partition_S(bA)
    partition_dst = thr_copy.partition_D(bB)
    fx.copy(copy_atom, partition_src, partition_dst, pred=None)


@flyc.jit
def tiledCopy(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    copy_kernel(A, B).launch(grid=(15, 1, 1), block=(4, 1, 1), stream=stream)


M, N = 8 * 3, 24 * 5
A = torch.arange(M * N, dtype=torch.float32).reshape(M, N).cuda()
B = torch.zeros(M, N, dtype=torch.float32).cuda()


tiledCopy(A, B, stream=torch.cuda.Stream())

torch.cuda.synchronize()

is_correct = torch.allclose(A, B)
print("Result correct:", is_correct)
if not is_correct:
    print("A:", A)
    print("B:", B)
