#!/usr/bin/env python3
"""CPU dispatch overhead benchmark: FlyDSL vs Triton vs PyTorch.

Measures pure CPU-side dispatch time (no GPU sync) for a small vec_add kernel.
This isolates the Python -> JIT call overhead from actual GPU execution time.

Layers measured:
  L0: Full @flyc.jit call (sig.bind + cache + dispatch)
  L1: CompiledArtifact.__call__ (DLPack path)
  L2: CallState.__call__ (current fast path)
  L3: TVMFFIDispatcher.__call__ (new TVM FFI path)
  L_raw: Raw ctypes call (absolute baseline)
  Triton: For comparison
  PyTorch: For comparison

Usage:
    FLYDSL_RUNTIME_ENABLE_TVM_FFI=1 PYTHONPATH=./ python tests/pyir/test_launch_overhead.py
"""

import time
import torch

# ─────────────────────────────────────────────────────────────────
# 1. FlyDSL vec_add
# ─────────────────────────────────────────────────────────────────
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vecAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width
    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))
    copy_bits = vec_width * 32
    RABMemRefTy = fx.MemRefType.get(
        fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register
    )
    copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)
    rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)
    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vecAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    vecAddKernel(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
    )


# ─────────────────────────────────────────────────────────────────
# 2. Triton vec_add
# ─────────────────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl

    @triton.jit
    def triton_vec_add_kernel(a_ptr, b_ptr, c_ptr, n,
                              BLOCK: tl.constexpr = 1024):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(c_ptr + offsets, a + b, mask=mask)

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def bench_wallclock(fn, n_warmup=20, n_iters=1000):
    """Measure wall-clock time per call (no GPU sync between calls).

    This measures CPU dispatch overhead: the time from Python calling the
    function to the function returning control to Python (kernel queued
    on GPU stream but not necessarily finished).
    """
    # warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # timed loop -- no sync between iterations
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_iters * 1e6  # us


def main():
    SIZE = 1024 * 256  # 256K elements -- small enough that GPU time is trivial
    BLOCK = 256
    VEC = 4
    N_ITERS = 2000

    print("=" * 70)
    print("CPU Dispatch Overhead Benchmark (vec_add, no GPU sync per iter)")
    print(f"  SIZE={SIZE}  BLOCK={BLOCK}  VEC={VEC}  N_ITERS={N_ITERS}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    a = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    stream = torch.cuda.current_stream()

    # -- FlyDSL warmup (triggers JIT compilation on first call) --
    vecAdd(a, b, c, SIZE, SIZE, BLOCK, VEC, stream=stream)
    torch.cuda.synchronize()

    # verify correctness
    ref = a + b
    err = (c - ref).abs().max().item()
    assert err < 1e-5, f"FlyDSL correctness failed: max_err={err}"

    # -- Check TVM FFI status --
    from flydsl.utils import env
    jf = vecAdd
    tvm_ffi_enabled = env.runtime.enable_tvm_ffi
    has_tvm_ffi = jf._tvm_ffi_dispatch is not None

    # -- Bench FlyDSL (full path) --
    flydsl_us = bench_wallclock(
        lambda: vecAdd(a, b, c, SIZE, SIZE, BLOCK, VEC, stream=stream),
        n_iters=N_ITERS,
    )

    # -- Bench FlyDSL CallState (disable TVM FFI temporarily) --
    saved_dispatch = jf._tvm_ffi_dispatch
    jf._tvm_ffi_dispatch = None
    callstate_us = bench_wallclock(
        lambda: vecAdd(a, b, c, SIZE, SIZE, BLOCK, VEC, stream=stream),
        n_iters=N_ITERS,
    )
    jf._tvm_ffi_dispatch = saved_dispatch

    # -- Bench PyTorch --
    torch_us = bench_wallclock(
        lambda: torch.add(a, b, out=c),
        n_iters=N_ITERS,
    )

    # -- Bench Triton --
    triton_us = None
    if HAS_TRITON:
        grid = ((SIZE + 1023) // 1024,)
        # warmup triton
        triton_vec_add_kernel[grid](a, b, c, SIZE)
        torch.cuda.synchronize()

        triton_us = bench_wallclock(
            lambda: triton_vec_add_kernel[grid](a, b, c, SIZE),
            n_iters=N_ITERS,
        )

    # -- Results --
    print()
    print(f"  {'Framework':<28s} {'Dispatch (us)':>14s} {'vs PyTorch':>12s}")
    print("  " + "-" * 56)
    print(f"  {'PyTorch (torch.add)':<28s} {torch_us:>14.2f} {'1.00x':>12s}")

    if has_tvm_ffi:
        print(f"  {'FlyDSL (TVM FFI)':<28s} {flydsl_us:>14.2f} {flydsl_us / torch_us:>11.2f}x")
    else:
        print(f"  {'FlyDSL (CallState)':<28s} {flydsl_us:>14.2f} {flydsl_us / torch_us:>11.2f}x")

    if has_tvm_ffi:
        print(f"  {'FlyDSL (CallState)':<28s} {callstate_us:>14.2f} {callstate_us / torch_us:>11.2f}x")

    if triton_us is not None:
        print(f"  {'Triton (@triton.jit)':<28s} {triton_us:>14.2f} {triton_us / torch_us:>11.2f}x")
    print()

    if has_tvm_ffi:
        print(f"  TVM FFI vs CallState: {callstate_us/flydsl_us:.2f}x faster")
    print()

    # -- Layered breakdown for FlyDSL --
    print("-" * 70)
    print("FlyDSL layered breakdown:")

    # L0: full @flyc.jit call (same as above)
    print(f"  L0 @flyc.jit call:              {flydsl_us:>8.2f} us"
          f"  ({'TVM FFI' if has_tvm_ffi else 'CallState'})")

    if has_tvm_ffi:
        print(f"  L0 @flyc.jit (CallState):       {callstate_us:>8.2f} us")

    # L1: CompiledArtifact.__call__ (bypass JitFunction)
    import inspect
    sig = inspect.signature(jf.func)
    bound = sig.bind(a, b, c, SIZE, SIZE, BLOCK, VEC, stream=stream)
    bound.apply_defaults()
    jf._ensure_sig()
    cache_key = jf._make_cache_key(bound.arguments)
    artifact = jf._mem_cache.get(cache_key)

    if artifact is not None:
        from flydsl.compiler.jit_argument import convert_to_jit_arguments
        from flydsl.compiler.jit_function import _ensure_stream_arg
        from flydsl._mlir import ir
        from flydsl.compiler.protocol import fly_pointers

        # Build jit_args once for L1 benchmark
        ctx = ir.Context()
        ctx.load_all_available_dialects()
        ctx.__enter__()
        _, jit_args, _, _ = convert_to_jit_arguments(sig, bound)
        _ensure_stream_arg(jit_args)

        l1_us = bench_wallclock(
            lambda: artifact(*jit_args),
            n_iters=N_ITERS,
        )
        print(f"  L1 CompiledArtifact.__call__:    {l1_us:>8.2f} us")

        # L2: CallState.__call__
        call_state = jf._call_state_cache.get(cache_key)
        if call_state is not None:
            args_tuple = tuple(bound.arguments.values())
            l2_us = bench_wallclock(
                lambda: call_state(args_tuple),
                n_iters=N_ITERS,
            )
            print(f"  L2 CallState.__call__:           {l2_us:>8.2f} us")

        # L3: TVMFFIDispatcher.__call__
        if has_tvm_ffi:
            args_tuple = tuple(bound.arguments.values())
            dispatcher = jf._tvm_ffi_dispatch
            l3_us = bench_wallclock(
                lambda: dispatcher(args_tuple),
                n_iters=N_ITERS,
            )
            print(f"  L3 TVMFFIDispatcher.__call__:    {l3_us:>8.2f} us")

        # L_raw: raw ctypes func_exe
        all_c_ptrs = []
        for arg in jit_args:
            all_c_ptrs.extend(fly_pointers(arg))

        func_exe = artifact._get_func_exe()
        packed = artifact._packer.pack(all_c_ptrs)

        l_raw_us = bench_wallclock(
            lambda: func_exe(packed),
            n_iters=N_ITERS,
        )
        print(f"  L_raw ctypes call:              {l_raw_us:>8.2f} us")

        ctx.__exit__(None, None, None)

        print(f"  {'':>36s}──────────")
        overhead_total = flydsl_us - l_raw_us
        print(f"  Total Python overhead (L0-raw):  {overhead_total:>8.2f} us")
        if has_tvm_ffi:
            print(f"  TVM FFI overhead (L3-raw):       {l3_us - l_raw_us:>8.2f} us")
        if call_state is not None:
            print(f"  CallState overhead (L2-raw):     {l2_us - l_raw_us:>8.2f} us")
        print(f"  Artifact overhead (L1-raw):      {l1_us - l_raw_us:>8.2f} us")

    print()


if __name__ == "__main__":
    main()
