#!/usr/bin/env python3
import sys
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/python")
import numpy as np
from hip import hip, hiprtc
import ctypes
from rocdsl.runtime.hip_util import hip_check, get_hip_arch

n = 1024
kernel_src = r"""
extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"""

a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.zeros(n, dtype=np.float32)

arch = get_hip_arch()
print("GPU:", arch)
prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_src.encode(), b"k.cu", 0, [], []))
opts = [b"--gpu-architecture=" + arch.encode()]
result, = hiprtc.hiprtcCompileProgram(prog, len(opts), opts)
if result != hiprtc.hiprtcResult.HIPRTC_SUCCESS: raise RuntimeError("Compile failed")
code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
code_bin = bytearray(code_size)
hip_check(hiprtc.hiprtcGetCode(prog, code_bin))
module = hip_check(hip.hipModuleLoadData(code_bin))
kernel = hip_check(hip.hipModuleGetFunction(module, b"vector_add"))
print("Loaded")

d_a = hip_check(hip.hipMalloc(n * 4))
d_b = hip_check(hip.hipMalloc(n * 4))
d_c = hip_check(hip.hipMalloc(n * 4))
hip_check(hip.hipMemcpy(d_a, a.ctypes.data, n * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(d_b, b.ctypes.data, n * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

threads = 256
blocks = (n + threads - 1) // threads
n_val = ctypes.c_int(n)
args = (ctypes.c_void_p * 4)(d_a.createRef().as_c_void_p(), d_b.createRef().as_c_void_p(), d_c.createRef().as_c_void_p(), ctypes.addressof(n_val))
print("Launch:", blocks, "x", threads)
hip_check(hip.hipModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, 0, args, None))
hip_check(hip.hipDeviceSynchronize())

hip_check(hip.hipMemcpy(c.ctypes.data, d_c, n * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
expected = a + b
error = np.max(np.abs(c - expected))
print("Error:", error)
for i in range(5): print(" ", a[i], "+", b[i], "=", c[i])
hip_check(hip.hipFree(d_a))
hip_check(hip.hipFree(d_b))
hip_check(hip.hipFree(d_c))
hip_check(hip.hipModuleUnload(module))
assert error < 1e-5
print("PASS")
