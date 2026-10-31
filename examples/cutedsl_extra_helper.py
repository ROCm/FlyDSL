#!/usr/bin/env python3

"""
CuTeDSL demo:
@cute.kernel calls a helper function, and the helper contains `if tid == 0:`.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


def _cu_ok(ret):
    if ret[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {ret[0]}")
    if len(ret) == 1:
        return None
    if len(ret) == 2:
        return ret[1]
    return ret[1:]


def helper_with_if(tid: cutlass.Int32):
    out = cutlass.Int32(-1)
    if tid == 0:
        out = cutlass.Int32(100)
    else:
        out = cutlass.Int32(200)
    return out


@cute.jit
def helper_with_if_jit(tid: cutlass.Int32):
    out = cutlass.Int32(-1)
    if tid == 0:
        out = cutlass.Int32(100)
    else:
        out = cutlass.Int32(200)
    return out


@cute.kernel
def kernel_calls_helper_if():
    tid, _, _ = cute.arch.thread_idx()
    v = helper_with_if(tid)

    if tid == 0:
        cute.printf("tid={} v={}", tid, v)


@cute.kernel
def kernel_calls_jit_helper_if():
    tid, _, _ = cute.arch.thread_idx()
    v = helper_with_if_jit(tid)

    if tid == 0:
        cute.printf("jit-helper tid={} v={}", tid, v)


@cute.jit
def run_case(stream: cuda.CUstream):
    kernel_calls_helper_if().launch(
        grid=[1, 1, 1],
        block=[2, 1, 1],
        stream=stream,
    )


@cute.jit
def run_jit_helper_case(stream: cuda.CUstream):
    kernel_calls_jit_helper_if().launch(
        grid=[1, 1, 1],
        block=[2, 1, 1],
        stream=stream,
    )


def main():
    _cu_ok(cuda.cuInit(0))
    dev = _cu_ok(cuda.cuDeviceGet(0))
    primary_ctx = _cu_ok(cuda.cuDevicePrimaryCtxRetain(dev))
    _cu_ok(cuda.cuCtxSetCurrent(primary_ctx))

    stream = cuda.CUstream(0)
    try:
        print("=== plain helper case ===")
        run_case(stream)
        _cu_ok(cuda.cuCtxSynchronize())
    except Exception as e:
        print(f"plain helper case failed: {e}")

    print("=== @cute.jit helper case ===")
    run_jit_helper_case(stream)
    _cu_ok(cuda.cuCtxSynchronize())
    print("PASS")


if __name__ == "__main__":
    main()
