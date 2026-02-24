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

thread_local static int32_t defaultDevice = 0;
// When set (from the Python host), force all runtime ops to use this stream.
// This allows integrating with frameworks like PyTorch/vLLM that require all GPU
// work to be enqueued onto the framework's "current" stream (including during
// HIP graph capture).
thread_local static bool externalStreamEnabled = false;
thread_local static hipStream_t externalStream = nullptr;

static inline hipStream_t mgpuResolveStream(hipStream_t stream) {
  return externalStreamEnabled ? externalStream : stream;
}

extern "C" FLIR_EXPORT void mgpuSetExternalStream(intptr_t stream) {
  externalStream = reinterpret_cast<hipStream_t>(stream);
  externalStreamEnabled = true;
}

extern "C" FLIR_EXPORT void mgpuClearExternalStream() {
  externalStreamEnabled = false;
  externalStream = nullptr;
}

extern "C" FLIR_EXPORT intptr_t mgpuGetExternalStream() {
  return reinterpret_cast<intptr_t>(externalStream);
}

static inline bool mgpuStreamIsCapturing(hipStream_t stream) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  hipError_t err = hipStreamIsCapturing(stream, &status);
  if (err != hipSuccess)
    return false;
  return status != hipStreamCaptureStatusNone;
}

extern "C" FLIR_EXPORT hipModule_t mgpuModuleLoad(void *data,
                                                  size_t /*gpuBlobSize*/) {
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
  stream = mgpuResolveStream(stream);
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                            blockY, blockZ, smem, stream, params,
                                            extra));
}

extern "C" FLIR_EXPORT hipStream_t mgpuStreamCreate() {
  // If the host provided an external stream, always use it.
  if (externalStreamEnabled)
    return externalStream;

  // In HIP graph capture, stream creation is not permitted. Use per-thread
  // default stream when the current thread is capturing.
  if (mgpuStreamIsCapturing(hipStreamPerThread))
    return hipStreamPerThread;

  hipStream_t stream = nullptr;
  hipError_t result = hipStreamCreate(&stream);
  if (result != hipSuccess) {
    // If failed due to capture, return per-thread stream as fallback.
    if (result == hipErrorStreamCaptureUnsupported ||
        result == hipErrorStreamCaptureInvalidated) {
      return hipStreamPerThread;
    }
    const char *name = hipGetErrorName(result);
    if (!name) name = "<unknown>";
    fprintf(stderr, "'hipStreamCreate' failed with '%s'\n", name);
    return hipStreamPerThread;  // Fallback to per-thread stream.
  }
  return stream;
}

extern "C" FLIR_EXPORT void mgpuStreamDestroy(hipStream_t stream) {
  // Never destroy implicit streams.
  if (stream == nullptr || stream == hipStreamPerThread || stream == hipStreamLegacy)
    return;
  // Never destroy externally owned stream.
  if (externalStreamEnabled && stream == externalStream)
    return;
  // Don't destroy streams while capturing.
  if (mgpuStreamIsCapturing(stream))
    return;
  hipError_t result = hipStreamDestroy(stream);
  if (result != hipSuccess &&
      result != hipErrorStreamCaptureUnsupported &&
      result != hipErrorStreamCaptureInvalidated) {
    const char *name = hipGetErrorName(result);
    if (!name) name = "<unknown>";
    fprintf(stderr, "'hipStreamDestroy' failed with '%s'\n", name);
  }
}

extern "C" FLIR_EXPORT void mgpuStreamSynchronize(hipStream_t stream) {
  stream = mgpuResolveStream(stream);
  // Stream synchronization is not permitted during HIP graph capture.
  // Try to detect capture state first, but also handle errors gracefully.
  if (mgpuStreamIsCapturing(hipStreamPerThread) || mgpuStreamIsCapturing(stream))
    return;
  
  // Even if capture detection fails (e.g., capture stream not checked),
  // try the sync and silently ignore capture-related errors.
  hipError_t result = hipStreamSynchronize(stream);
  if (result != hipSuccess && 
      result != hipErrorStreamCaptureUnsupported &&
      result != hipErrorStreamCaptureInvalidated) {
    const char *name = hipGetErrorName(result);
    if (!name) name = "<unknown>";
    fprintf(stderr, "'hipStreamSynchronize(stream)' failed with '%s'\n", name);
  }
}

extern "C" FLIR_EXPORT void mgpuStreamWaitEvent(hipStream_t stream,
                                                hipEvent_t event) {
  stream = mgpuResolveStream(stream);
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" FLIR_EXPORT hipEvent_t mgpuEventCreate() {
  // Event creation is not permitted during HIP graph capture.
  if (mgpuStreamIsCapturing(hipStreamPerThread))
    return nullptr;

  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  return event;
}

extern "C" FLIR_EXPORT void mgpuEventDestroy(hipEvent_t event) {
  // Event destruction is not permitted during HIP graph capture.
  if (event == nullptr || mgpuStreamIsCapturing(hipStreamPerThread))
    return;
  HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" FLIR_EXPORT void mgpuEventSynchronize(hipEvent_t event) {
  // Skip if event is null (e.g., created during capture) or if capturing.
  if (event == nullptr || mgpuStreamIsCapturing(hipStreamPerThread))
    return;
  
  // Try the sync and silently ignore capture-related errors.
  hipError_t result = hipEventSynchronize(event);
  if (result != hipSuccess && 
      result != hipErrorStreamCaptureUnsupported &&
      result != hipErrorStreamCaptureInvalidated) {
    const char *name = hipGetErrorName(result);
    if (!name) name = "<unknown>";
    fprintf(stderr, "'hipEventSynchronize(event)' failed with '%s'\n", name);
  }
}

extern "C" FLIR_EXPORT void mgpuEventRecord(hipEvent_t event,
                                            hipStream_t stream) {
  // Skip if event is null (e.g., created during capture).
  if (event == nullptr)
    return;
  stream = mgpuResolveStream(stream);
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

extern "C" FLIR_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/,
                              bool /*isHostShared*/) {
  // hipMalloc is not permitted during HIP graph capture.
  // Skip allocation entirely if capturing to avoid invalidating the graph.
  if (mgpuStreamIsCapturing(hipStreamPerThread))
    return nullptr;

  void *ptr = nullptr;
  hipError_t result = hipMalloc(&ptr, sizeBytes);
  if (result != hipSuccess) {
    if (result != hipErrorStreamCaptureUnsupported &&
        result != hipErrorStreamCaptureInvalidated) {
      const char *name = hipGetErrorName(result);
      if (!name) name = "<unknown>";
      fprintf(stderr, "'hipMalloc' failed with '%s'\n", name);
    }
    return nullptr;
  }
  return ptr;
}

extern "C" FLIR_EXPORT void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
  // hipFree is not permitted during HIP graph capture.
  // Skip free entirely if capturing to avoid invalidating the graph.
  if (mgpuStreamIsCapturing(hipStreamPerThread))
    return;

  hipError_t result = hipFree(ptr);
  if (result != hipSuccess &&
      result != hipErrorStreamCaptureUnsupported &&
      result != hipErrorStreamCaptureInvalidated) {
    const char *name = hipGetErrorName(result);
    if (!name) name = "<unknown>";
    fprintf(stderr, "'hipFree' failed with '%s'\n", name);
  }
}

extern "C" FLIR_EXPORT void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                           hipStream_t stream) {
  stream = mgpuResolveStream(stream);
  HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" FLIR_EXPORT void mgpuMemset32(void *dst, int value, size_t count,
                             hipStream_t stream) {
  stream = mgpuResolveStream(stream);
  HIP_REPORT_IF_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

extern "C" FLIR_EXPORT void mgpuMemset16(void *dst, int shortValue, size_t count,
                             hipStream_t stream) {
  stream = mgpuResolveStream(stream);
  HIP_REPORT_IF_ERROR(
      hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), shortValue, count,
                        stream));
}
