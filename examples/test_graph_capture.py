"""Test CUDA Graph Capture with vectorAdd kernel.

The fly-gpu-stream-inject pass wires the user-supplied stream directly as
asyncObject in gpu.launch_func, so graph capture works without any TLS hack.
"""
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
    copyAtom = fx.make_copy_atom(fx.CopyAtomUniversalCopyType.get(32))
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
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    vectorAddKernel(A, B, C, block_dim).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)


def run_eager():
    """Normal execution (no graph capture) - should always work."""
    n = 128
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    vectorAdd(tA, B, C, n, n + 1, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    ok = torch.allclose(C, A + B)
    print(f"[Eager] Result correct: {ok}")
    return ok


def run_graph_capture():
    """CUDA Graph Capture via fly-gpu-stream-inject (no TLS hack needed)."""
    n = 128
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)

    # Warmup (triggers JIT compilation)
    vectorAdd(tA, B, C, n, n + 1, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    C.zero_()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            vectorAdd(tA, B, C, n, n + 1, stream=capture_stream)

    C.zero_()
    graph.replay()
    torch.cuda.synchronize()

    ok = torch.allclose(C, A + B)
    print(f"[Graph Capture] Result correct: {ok}")
    if not ok:
        print(f"  Expected: {(A + B)[:16]}")
        print(f"  Got:      {C[:16]}")
    return ok


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: Eager execution")
    print("=" * 50)
    ok1 = run_eager()

    print()
    print("=" * 50)
    print("Test 2: CUDA Graph Capture")
    print("=" * 50)
    try:
        ok2 = run_graph_capture()
    except Exception as e:
        print(f"[Graph Capture] FAILED with exception: {e}")
        ok2 = False

    print()
    print(f"All passed: {ok1 and ok2}")
