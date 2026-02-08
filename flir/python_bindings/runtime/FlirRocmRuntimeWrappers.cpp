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

extern "C" FLIR_EXPORT hipModule_t mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" FLIR_EXPORT hipModule_t mgpuModuleLoadJIT(void *data, int optLevel) {
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
