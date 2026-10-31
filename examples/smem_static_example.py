"""FlyDSL 静态 shared memory (LDS) 用例  ——  SharedAllocator(static=True)。

静态 smem：每个字段编译期各自一个 `@__shared_alloc_<id>` LDS global，
类似 C/C++ 的 `__shared__ float buf[N];`。大小/偏移编译期固定，
launch 不需要传动态 smem 大小。

做的事（smem 的典型用途：block 内跨线程通信）：
    每线程把 A[tid] 写进 smem[tid] → barrier → 读 smem[N-1-tid] 写回 O[tid]
    结果 O == reverse(A)。

运行：
    python /data/users/xudong/test/FLYIR2/FlyDSL/examples/smem_static_example.py
（或在仓库根目录 `python examples/smem_static_example.py`；路径已自动定位，任意 cwd 均可）
"""
import os
import sys

# 自动定位 FlyDSL 仓库根，指向已编译的 build-fly 产物（editable 源码缺 _mlir）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "build-fly", "python_packages"))
sys.path.insert(1, _ROOT)

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

N = 64  # block size = smem 元素个数


# 1) 用 @fx.struct 声明 smem 布局：大小/偏移/对齐由类型推导
@fx.struct
class SharedStorage:
    buf: fx.Array[fx.Float32, N]     # N 个 f32 的一维 LDS 缓冲


@flyc.kernel(known_block_size=[N, 1, 1])
def reverse_kernel(A: fx.Tensor, O: fx.Tensor):  # noqa: E741
    tid = fx.thread_idx.x

    # 2) 分配 + peek 拿类型化结构（static=True 是默认，这里显式写出以作对比）
    lds = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()

    # 3) 给字段绑一个 layout，得到可按坐标读写的 view
    s = lds.buf.view(fx.make_layout(N, 1))

    # 4) 每线程把 A[tid] 写进 smem[tid]
    fx.memref_store(fx.memref_load(A, tid), s, tid)

    # 5) barrier：所有线程写完再跨线程读
    fx.gpu.barrier()

    # 6) 读 smem[N-1-tid]（反转）写回 O[tid]
    fx.memref_store(fx.memref_load(s, N - 1 - tid), O, tid)


@flyc.jit
def run(A: fx.Tensor, O: fx.Tensor):  # noqa: E741
    # 静态 smem：launch 不需要传 smem 大小
    reverse_kernel(A, O).launch(grid=(1, 1, 1), block=(N, 1, 1))


def main():
    A = torch.arange(N, device="cuda", dtype=torch.float32)
    O = torch.zeros(N, device="cuda", dtype=torch.float32)
    run(A, O)
    torch.cuda.synchronize()
    expected = torch.flip(A, dims=[0])
    ok = torch.allclose(O, expected)
    print("[static smem]  O == reverse(A):", ok)
    print("  A[:5] =", A[:5].tolist())
    print("  O[:5] =", O[:5].tolist())
    assert ok, "static smem result mismatch"
    print("PASSED")


if __name__ == "__main__":
    main()
