//===- FlirRocmRuntimeWrappers.cpp - Thin ROCm runtime wrappers ------------===//
//
// This file is derived from LLVM Project:
//   mlir/lib/ExecutionEngine/RocmRuntimeWrappers.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the ROCm HIP runtime for easy linking in ORC JIT.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <vector>

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "hip/hip_runtime.h"

// Symbol export macro: ensure mgpu* runtime functions are visible when linking
// with hidden visibility preset. Required for MLIR JIT ExecutionEngine to resolve symbols.
#ifdef __GNUC__
#define FLIR_EXPORT __attribute__((visibility("default")))
#else
#define FLIR_EXPORT
#endif

#define HIP_REPORT_IF_ERROR(expr)                                              \
  [](hipError_t result) {                                                      \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

thread_local static int32_t defaultDevice = 0;

extern "C" FLIR_EXPORT hipModule_t mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" hipModule_t mgpuModuleLoadJIT(void *data, int optLevel) {
  (void)data;
  (void)optLevel;
  assert(false && "This function is not available in HIP.");
  return nullptr;
}

extern "C" FLIR_EXPORT void mgpuModuleUnload(hipModule_t module) {
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" FLIR_EXPORT hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of ROCm's unsigned int to match MLIR's index
// type, avoiding casts in generated code.
extern "C" FLIR_EXPORT void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem,
                                 hipStream_t stream, void **params,
                                 void **extra, size_t /*paramsCount*/) {
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                            blockY, blockZ, smem, stream, params,
                                            extra));
}

extern "C" FLIR_EXPORT hipStream_t mgpuStreamCreate() {
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  return stream;
}

extern "C" FLIR_EXPORT void mgpuStreamDestroy(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" FLIR_EXPORT void mgpuStreamSynchronize(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" FLIR_EXPORT void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" FLIR_EXPORT hipEvent_t mgpuEventCreate() {
  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  return event;
}

extern "C" FLIR_EXPORT void mgpuEventDestroy(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" FLIR_EXPORT void mgpuEventSynchronize(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" FLIR_EXPORT void mgpuEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

extern "C" FLIR_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/,
                              bool /*isHostShared*/) {
  void *ptr = nullptr;
  HIP_REPORT_IF_ERROR(hipMalloc(&ptr, sizeBytes));
  return ptr;
}

extern "C" FLIR_EXPORT void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
  HIP_REPORT_IF_ERROR(hipFree(ptr));
}

extern "C" FLIR_EXPORT void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                           hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" FLIR_EXPORT void mgpuMemset32(void *dst, int value, size_t count,
                             hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

extern "C" FLIR_EXPORT void mgpuMemset16(void *dst, int shortValue, size_t count,
                             hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), shortValue, count,
                        stream));
}

// Helper functions for writing mlir example code.
extern "C" FLIR_EXPORT void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

extern "C" FLIR_EXPORT void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {
  // `descriptor` is passed as an opaque pointer, but its layout matches the
  // strided memref descriptor:
  //   {T *allocated, T *aligned, i64 offset, i64 sizes[rank], i64 strides[rank]}
  //
  // Because we accept `StridedMemRefType<char, 1>*`, we must compute the stride
  // pointer via `sizes + rank` (we cannot use `descriptor->strides` directly for
  // rank > 1).
  int64_t *sizes = descriptor->sizes;
  int64_t *strides = sizes + rank;

  // Compute the total number of elements (product of sizes) and check for dense
  // row-major packing.
  std::vector<int64_t> denseStrides(static_cast<size_t>(rank));
  if (rank > 0) {
    denseStrides[static_cast<size_t>(rank - 1)] = sizes[rank - 1];
    for (int64_t i = rank - 2; i >= 0; --i)
      denseStrides[static_cast<size_t>(i)] =
          sizes[i] * denseStrides[static_cast<size_t>(i + 1)];
  }
  auto sizeBytes = (rank > 0 ? denseStrides[0] : 1) * elementSizeBytes;

  // Only densely packed tensors are currently supported.
  // Expected dense strides: [sizes[1]*...*sizes[rank-1], ..., 1]
  for (int64_t i = 0; i < rank - 1; ++i)
    denseStrides[static_cast<size_t>(i)] =
        denseStrides[static_cast<size_t>(i + 1)];
  if (rank > 0)
    denseStrides[static_cast<size_t>(rank - 1)] = 1;

  for (int64_t i = 0; i < rank; ++i)
    assert(strides[i] == denseStrides[static_cast<size_t>(i)]);

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

extern "C" FLIR_EXPORT void mgpuMemHostUnregister(void *ptr) {
  HIP_REPORT_IF_ERROR(hipHostUnregister(ptr));
}

extern "C" FLIR_EXPORT void
mgpuMemHostUnregisterMemRef(int64_t /*rank*/,
                            StridedMemRefType<char, 1> *descriptor,
                            int64_t elementSizeBytes) {
  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

template <typename T>
static void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  HIP_REPORT_IF_ERROR(hipSetDevice(defaultDevice));
  HIP_REPORT_IF_ERROR(
      hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0));
}

extern "C" FLIR_EXPORT StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float * /*allocated*/, float *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" FLIR_EXPORT StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t * /*allocated*/, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" FLIR_EXPORT void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  HIP_REPORT_IF_ERROR(hipSetDevice(device));
}

