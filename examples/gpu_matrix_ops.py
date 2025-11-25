#!/usr/bin/env python3
"""
RocDSL GPU Example: Matrix Operations with Layout-based Indexing

This example demonstrates:
1. Compiling GPU kernels with HIP runtime compilation
2. Using layout-based indexing functions in GPU kernels
3. Matrix transpose and element-wise operations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from hip import hip, hiprtc
import ctypes
from rocdsl.runtime.hip_util import hip_check, get_hip_arch

def compile_kernel(kernel_src: str, kernel_name: str) -> tuple:
    """Compile a HIP kernel and return (module, kernel_func)"""
    arch = get_hip_arch()
    prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_src.encode(), kernel_name.encode(), 0, [], []))
    opts = [b"--gpu-architecture=" + arch.encode()]
    result, = hiprtc.hiprtcCompileProgram(prog, len(opts), opts)
    
    if result != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hiprtc.hiprtcGetProgramLog(prog, log)
        raise RuntimeError(f"Kernel compilation failed:\n{log.decode()}")
    
    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code_bin = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code_bin))
    module = hip_check(hip.hipModuleLoadData(code_bin))
    kernel = hip_check(hip.hipModuleGetFunction(module, kernel_name.encode()))
    
    return module, kernel

def main():
    print("RocDSL GPU Matrix Operations Example")
    print("=" * 80)
    print(f"GPU Architecture: {get_hip_arch()}")
    print()

    # Example 1: Matrix Transpose
    print("Example 1: Matrix Transpose (64x128 -> 128x64)")
    print("-" * 80)

    M, N = 64, 128
    
    transpose_kernel = r"""
__device__ int crd2idx_rowmajor(int row, int col, int stride) {
    return row * stride + col;
}

extern "C" __global__ void transpose(float* input, float* output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        output[crd2idx_rowmajor(col, row, M)] = input[crd2idx_rowmajor(row, col, N)];
    }
}
"""

    # Compile
    module1, kernel1 = compile_kernel(transpose_kernel, "transpose")
    print("✓ Kernel compiled")

    # Prepare data
    input_mat = np.arange(M * N, dtype=np.float32).reshape(M, N)
    output_mat = np.zeros((N, M), dtype=np.float32)

    # GPU memory
    d_input = hip_check(hip.hipMalloc(M * N * 4))
    d_output = hip_check(hip.hipMalloc(M * N * 4))
    hip_check(hip.hipMemcpy(d_input, input_mat.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    # Launch
    block = (16, 16, 1)
    grid = ((N + 15) // 16, (M + 15) // 16, 1)
    M_val, N_val = ctypes.c_int(M), ctypes.c_int(N)
    args = (ctypes.c_void_p * 4)(d_input.createRef().as_c_void_p(), d_output.createRef().as_c_void_p(), ctypes.addressof(M_val), ctypes.addressof(N_val))
    hip_check(hip.hipModuleLaunchKernel(kernel1, *grid, *block, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    print(f"✓ Kernel executed: grid={grid}, block={block}")

    # Verify
    hip_check(hip.hipMemcpy(output_mat.ctypes.data, d_output, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    error = np.max(np.abs(output_mat - input_mat.T))
    print(f"✓ Verification: max error = {error}")
    assert error < 1e-5
    print()

    # Example 2: Element-wise GELU Activation
    print("Example 2: Element-wise GELU Activation")
    print("-" * 80)

    size = 1024 * 1024
    gelu_kernel = r"""
extern "C" __global__ void gelu(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
"""

    module2, kernel2 = compile_kernel(gelu_kernel, "gelu")
    print("✓ GELU kernel compiled")

    # Data
    x = np.random.randn(size).astype(np.float32)
    y = np.zeros(size, dtype=np.float32)

    d_x = hip_check(hip.hipMalloc(size * 4))
    d_y = hip_check(hip.hipMalloc(size * 4))
    hip_check(hip.hipMemcpy(d_x, x.ctypes.data, size * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    # Launch
    threads = 256
    blocks = (size + threads - 1) // threads
    size_val = ctypes.c_int(size)
    args2 = (ctypes.c_void_p * 3)(d_x.createRef().as_c_void_p(), d_y.createRef().as_c_void_p(), ctypes.addressof(size_val))
    
    # Timing
    start_event = hip_check(hip.hipEventCreate())
    stop_event = hip_check(hip.hipEventCreate())
    hip_check(hip.hipEventRecord(start_event, 0))
    hip_check(hip.hipModuleLaunchKernel(kernel2, blocks, 1, 1, threads, 1, 1, 0, 0, args2, None))
    hip_check(hip.hipEventRecord(stop_event, 0))
    hip_check(hip.hipEventSynchronize(stop_event))
    elapsed = hip_check(hip.hipEventElapsedTime(start_event, stop_event))
    print(f"✓ Kernel executed: {blocks} blocks x {threads} threads")
    print(f"✓ Execution time: {elapsed:.3f} ms")
    print(f"✓ Throughput: {size * 4 / (elapsed * 1e-3) / 1e9:.2f} GB/s")

    # Verify (CPU reference)
    hip_check(hip.hipMemcpy(y.ctypes.data, d_y, size * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    expected = 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    error = np.max(np.abs(y - expected))
    print(f"✓ Verification: max error = {error}")
    assert error < 1e-4

    # Cleanup
    hip_check(hip.hipFree(d_input))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipFree(d_x))
    hip_check(hip.hipFree(d_y))
    hip_check(hip.hipModuleUnload(module1))
    hip_check(hip.hipModuleUnload(module2))
    hip_check(hip.hipEventDestroy(start_event))
    hip_check(hip.hipEventDestroy(stop_event))

    print()
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
