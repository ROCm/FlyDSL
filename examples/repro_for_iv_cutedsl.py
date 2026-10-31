import cutlass
import cutlass.cute as cute


@cute.kernel
def for_iv_assign_mismatch_kernel(start: cutlass.Int32, stop: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()

    count = cutlass.Int32(0)
    for iv in range(start, stop):
        count = iv

    if tidx == 0:
        cute.printf("count={}", count)


for_iv_assign_mismatch_kernel(0, 10, grid=(1, 1, 1), block=(1, 1, 1))
