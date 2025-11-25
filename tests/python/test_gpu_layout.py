#!/usr/bin/env python3
"""GPU kernel test with layout-based indexing"""

import sys
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/python")
import numpy as np
from hip import hip, hiprtc
import ctypes
from rocdsl.runtime.hip_util import hip_check, get_hip_arch

# Test 1: Matrix transpose using layout-based indexing
print("="*80)
print("Test 1: Matrix Transpose with Layout Indexing")
print("="*80)

M, N = 32, 64
kernel_src = r"""
// Simple 2D layout: row-major indexing
__device__ int crd2idx_rowmajor(int row, int col, int stride) {
    return row * stride + col;
}

extern "C" __global__ void transpose(
    float* input, float* output, int M, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // Use layout function for indexing
        int in_idx = crd2idx_rowmajor(row, col, N);
        int out_idx = crd2idx_rowmajor(col, row, M);
        output[out_idx] = input[in_idx];
    }
}
"""

# Host data
input_data = np.arange(M * N, dtype=np.float32).reshape(M, N)
output_data = np.zeros((N, M), dtype=np.float32)

# Compile kernel
arch = get_hip_arch()
print(f"GPU: {arch}")
prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_src.encode(), b"transpose.cu", 0, [], []))
opts = [b"--gpu-architecture=" + arch.encode()]
result, = hiprtc.hiprtcCompileProgram(prog, len(opts), opts)
if result != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
    log = bytearray(log_size)
    hiprtc.hiprtcGetProgramLog(prog, log)
    raise RuntimeError(f"Compile failed: {log.decode()}")

code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
code_bin = bytearray(code_size)
hip_check(hiprtc.hiprtcGetCode(prog, code_bin))
module = hip_check(hip.hipModuleLoadData(code_bin))
kernel = hip_check(hip.hipModuleGetFunction(module, b"transpose"))
print("Kernel compiled and loaded")

# Device memory
d_input = hip_check(hip.hipMalloc(M * N * 4))
d_output = hip_check(hip.hipMalloc(M * N * 4))
hip_check(hip.hipMemcpy(d_input, input_data.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# Launch configuration
block_x, block_y = 16, 16
grid_x = (N + block_x - 1) // block_x
grid_y = (M + block_y - 1) // block_y

# Prepare arguments
M_val = ctypes.c_int(M)
N_val = ctypes.c_int(N)
args = (ctypes.c_void_p * 4)(d_input.createRef().as_c_void_p(), d_output.createRef().as_c_void_p(), ctypes.addressof(M_val), ctypes.addressof(N_val))

print(f"Launching: grid=({grid_x}, {grid_y}), block=({block_x}, {block_y})")
hip_check(hip.hipModuleLaunchKernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, args, None))
hip_check(hip.hipDeviceSynchronize())

# Copy result back
hip_check(hip.hipMemcpy(output_data.ctypes.data, d_output, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# Verify transpose
expected = input_data.T
error = np.max(np.abs(output_data - expected))
print(f"Max error: {error}")
print(f"Input shape: {input_data.shape}, Output shape: {output_data.shape}")
print(f"Sample input[0,:5]: {input_data[0,:5]}")
print(f"Sample output[:,0][:5]: {output_data[:,0][:5]}")

hip_check(hip.hipFree(d_input))
hip_check(hip.hipFree(d_output))
hip_check(hip.hipModuleUnload(module))

assert error < 1e-5, f"Test failed: error={error}"
print("✓ Test 1 PASSED")

print("="*80)
print("Test 2: Strided Layout Access")
print("="*80)

kernel_src2 = r"""
// Layout with custom stride
__device__ int crd2idx_strided(int row, int col, int row_stride, int col_stride) {
    return row * row_stride + col * col_stride;
}

extern "C" __global__ void copy_strided(
    float* input, float* output, int M, int N, int in_stride, int out_stride
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int in_idx = crd2idx_strided(row, col, in_stride, 1);
        int out_idx = crd2idx_strided(row, col, out_stride, 1);
        output[out_idx] = input[in_idx] * 2.0f;
    }
}
"""

# Host data with padding
M2, N2 = 16, 32
in_stride = N2 + 8
out_stride = N2 + 4
input_padded = np.random.rand(M2 * in_stride).astype(np.float32)
output_padded = np.zeros(M2 * out_stride, dtype=np.float32)

# Fill input
for i in range(M2):
    for j in range(N2):
        input_padded[i * in_stride + j] = i * N2 + j

# Compile
prog2 = hip_check(hiprtc.hiprtcCreateProgram(kernel_src2.encode(), b"strided.cu", 0, [], []))
result, = hiprtc.hiprtcCompileProgram(prog2, len(opts), opts)
if result != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog2))
    log = bytearray(log_size)
    hiprtc.hiprtcGetProgramLog(prog2, log)
    raise RuntimeError(f"Compile failed: {log.decode()}")

code_size2 = hip_check(hiprtc.hiprtcGetCodeSize(prog2))
code_bin2 = bytearray(code_size2)
hip_check(hiprtc.hiprtcGetCode(prog2, code_bin2))
module2 = hip_check(hip.hipModuleLoadData(code_bin2))
kernel2 = hip_check(hip.hipModuleGetFunction(module2, b"copy_strided"))
print("Strided kernel compiled and loaded")

# Device memory
d_in2 = hip_check(hip.hipMalloc(M2 * in_stride * 4))
d_out2 = hip_check(hip.hipMalloc(M2 * out_stride * 4))
hip_check(hip.hipMemcpy(d_in2, input_padded.ctypes.data, M2 * in_stride * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# Launch
grid_x2 = (N2 + 15) // 16
grid_y2 = (M2 + 15) // 16
M2_val = ctypes.c_int(M2)
N2_val = ctypes.c_int(N2)
in_stride_val = ctypes.c_int(in_stride)
out_stride_val = ctypes.c_int(out_stride)
args2 = (ctypes.c_void_p * 6)(d_in2.createRef().as_c_void_p(), d_out2.createRef().as_c_void_p(), ctypes.addressof(M2_val), ctypes.addressof(N2_val), ctypes.addressof(in_stride_val), ctypes.addressof(out_stride_val))

print(f"Launching strided kernel: grid=({grid_x2}, {grid_y2}), block=(16, 16)")
hip_check(hip.hipModuleLaunchKernel(kernel2, grid_x2, grid_y2, 1, 16, 16, 1, 0, 0, args2, None))
hip_check(hip.hipDeviceSynchronize())

# Verify
hip_check(hip.hipMemcpy(output_padded.ctypes.data, d_out2, M2 * out_stride * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# Check values
errors = 0
for i in range(M2):
    for j in range(N2):
        expected_val = (i * N2 + j) * 2.0
        actual_val = output_padded[i * out_stride + j]
        if abs(actual_val - expected_val) > 1e-5:
            errors += 1
            if errors <= 5:
                print(f"Error at ({i},{j}): expected {expected_val}, got {actual_val}")

print(f"Total errors: {errors}/{M2*N2}")
if errors == 0:
    print(f"Sample output[0,:5]: {output_padded[0:5]}")
    print(f"Sample output[1,:5]: {output_padded[out_stride:out_stride+5]}")

hip_check(hip.hipFree(d_in2))
hip_check(hip.hipFree(d_out2))
hip_check(hip.hipModuleUnload(module2))

assert errors == 0, f"Test failed: {errors} errors"
print("✓ Test 2 PASSED")

print("="*80)
print("All GPU layout tests PASSED!")
print("="*80)
