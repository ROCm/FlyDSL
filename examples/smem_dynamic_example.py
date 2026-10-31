"""FlyDSL 动态 shared memory (LDS) 用例  ——  SharedAllocator(static=False)。

动态 smem：所有子缓冲共享一个 `get_dyn_shared` 基址 + 编译期偏移，
类似 C/C++ 的 `extern __shared__ float[];`。所需字节由 kernel function
自动追踪（allocated_bytes），launch 时自动作为 dynamic LDS 大小传给 driver
——使用者不用手写 smem 大小、也不用手动 finalize。

程序员写法与静态版几乎一样，唯一区别是 `SharedAllocator(static=False)`。
做同样的事（block 内跨线程反转），结果一致：O == reverse(A)。

运行：
    python /data/users/xudong/test/FLYIR2/FlyDSL/examples/smem_dynamic_example.py
（或在仓库根目录 `python examples/smem_dynamic_example.py`；路径已自动定位，任意 cwd 均可）
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "build-fly", "python_packages"))
sys.path.insert(1, _ROOT)

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

N = 64


@fx.struct
class SharedStorage:
    buf: fx.Array[fx.Float32, N]


@flyc.kernel(known_block_size=[N, 1, 1])
def reverse_kernel(A: fx.Tensor, O: fx.Tensor):  # noqa: E741
    tid = fx.thread_idx.x

    # 唯一区别：static=False → 单个 get_dyn_shared 基址 + 偏移（extern __shared__ 语义）
    lds = fx.SharedAllocator(static=False).allocate(SharedStorage).peek()

    s = lds.buf.view(fx.make_layout(N, 1))

    fx.memref_store(fx.memref_load(A, tid), s, tid)
    fx.gpu.barrier()
    fx.memref_store(fx.memref_load(s, N - 1 - tid), O, tid)


@flyc.jit
def run(A: fx.Tensor, O: fx.Tensor):  # noqa: E741
    # 动态 smem：launch 不用手写 smem 大小 —— 编译期由 SharedAllocator 自动追踪并传入。
    # （也可显式 .launch(..., smem=<字节>)，但不得小于自动追踪值。）
    reverse_kernel(A, O).launch(grid=(1, 1, 1), block=(N, 1, 1))


def main():
    A = torch.arange(N, device="cuda", dtype=torch.float32)
    O = torch.zeros(N, device="cuda", dtype=torch.float32)
    run(A, O)
    torch.cuda.synchronize()
    expected = torch.flip(A, dims=[0])
    ok = torch.allclose(O, expected)
    print("[dynamic smem] O == reverse(A):", ok)
    print("  A[:5] =", A[:5].tolist())
    print("  O[:5] =", O[:5].tolist())
    assert ok, "dynamic smem result mismatch"
    print("PASSED")


if __name__ == "__main__":
    main()
