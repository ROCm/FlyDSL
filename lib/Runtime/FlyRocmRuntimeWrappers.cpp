//===- FlyRocmRuntimeWrappers.cpp - ROCm runtime with module caching ------===//
//
// Derived from LLVM Project: mlir/lib/ExecutionEngine/RocmRuntimeWrappers.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin ROCm runtime wrappers for MLIR ExecutionEngine JIT.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <vector>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "hip/hip_runtime.h"

#ifdef _WIN32
#define FLY_RUNTIME_EXPORT __declspec(dllexport)
#else
#define FLY_RUNTIME_EXPORT __attribute__((visibility("default")))
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

extern "C" FLY_RUNTIME_EXPORT hipModule_t mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" FLY_RUNTIME_EXPORT hipModule_t mgpuModuleLoadJIT(void *data, int optLevel) {
  (void)data;
  (void)optLevel;
  assert(false && "This function is not available in HIP.");
  return nullptr;
}

extern "C" FLY_RUNTIME_EXPORT void mgpuModuleUnload(hipModule_t module) {
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" FLY_RUNTIME_EXPORT hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

extern "C" FLY_RUNTIME_EXPORT void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem,
                                 hipStream_t stream, void **params,
                                 void **extra, size_t /*paramsCount*/) {
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuLaunchClusterKernel(hipFunction_t function,
                                        intptr_t clusterX, intptr_t clusterY,
                                        intptr_t clusterZ,
                                        intptr_t gridX, intptr_t gridY,
                                        intptr_t gridZ,
                                        intptr_t blockX, intptr_t blockY,
                                        intptr_t blockZ, int32_t smem,
                                        hipStream_t stream, void **params,
                                        void **extra, size_t /*paramsCount*/) {
#ifdef hipLaunchAttributeClusterDimension
  hipLaunchAttribute attrs[1];
  attrs[0].id = hipLaunchAttributeClusterDimension;
  attrs[0].value.clusterDim.x = static_cast<unsigned>(clusterX);
  attrs[0].value.clusterDim.y = static_cast<unsigned>(clusterY);
  attrs[0].value.clusterDim.z = static_cast<unsigned>(clusterZ);

  HIP_LAUNCH_CONFIG config{};
  config.gridDimX = static_cast<unsigned>(gridX);
  config.gridDimY = static_cast<unsigned>(gridY);
  config.gridDimZ = static_cast<unsigned>(gridZ);
  config.blockDimX = static_cast<unsigned>(blockX);
  config.blockDimY = static_cast<unsigned>(blockY);
  config.blockDimZ = static_cast<unsigned>(blockZ);
  config.sharedMemBytes = static_cast<unsigned>(smem);
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = 1;

  hipError_t err = hipDrvLaunchKernelEx(&config, function, params, extra);
  if (err == hipSuccess)
    return;

  const bool requestedRealCluster =
      (clusterX > 1) || (clusterY > 1) || (clusterZ > 1);
  if (requestedRealCluster) {
    fprintf(stderr,
            "[mgpuLaunchClusterKernel] hipDrvLaunchKernelEx failed (err=%d) "
            "for requested cluster=(%ld,%ld,%ld); not falling back to "
            "hipModuleLaunchKernel.\n",
            static_cast<int>(err), static_cast<long>(clusterX),
            static_cast<long>(clusterY), static_cast<long>(clusterZ));
    HIP_REPORT_IF_ERROR(err);
    return;
  }

  fprintf(stderr,
          "[mgpuLaunchClusterKernel] hipDrvLaunchKernelEx failed (err=%d) "
          "for cluster=(1,1,1); falling back to hipModuleLaunchKernel.\n",
          static_cast<int>(err));
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
#else
  // Cluster launch not supported by this HIP version; ignore cluster dims
  // and fall back to regular kernel launch.
  if ((clusterX > 1) || (clusterY > 1) || (clusterZ > 1)) {
    fprintf(stderr,
            "[mgpuLaunchClusterKernel] cluster=(%ld,%ld,%ld) requested but "
            "hipLaunchAttributeClusterDimension is not available in this HIP "
            "version; falling back to hipModuleLaunchKernel.\n",
            static_cast<long>(clusterX), static_cast<long>(clusterY),
            static_cast<long>(clusterZ));
  }
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
#endif
}

extern "C" FLY_RUNTIME_EXPORT hipStream_t mgpuStreamCreate() {
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  return stream;
}

extern "C" FLY_RUNTIME_EXPORT void mgpuStreamDestroy(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuStreamSynchronize(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" FLY_RUNTIME_EXPORT hipEvent_t mgpuEventCreate() {
  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  return event;
}

extern "C" FLY_RUNTIME_EXPORT void mgpuEventDestroy(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuEventSynchronize(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

extern "C" FLY_RUNTIME_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/,
                              bool /*isHostShared*/) {
  void *ptr = nullptr;
  HIP_REPORT_IF_ERROR(hipMalloc(&ptr, sizeBytes));
  return ptr;
}

extern "C" FLY_RUNTIME_EXPORT void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
  HIP_REPORT_IF_ERROR(hipFree(ptr));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                           hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuMemset32(void *dst, int value, size_t count,
                             hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuMemset16(void *dst, int shortValue, size_t count,
                             hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        shortValue, count, stream));
}

extern "C" FLY_RUNTIME_EXPORT void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

extern "C" FLY_RUNTIME_EXPORT void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {
  int64_t *sizes = descriptor->sizes;
  int64_t *strides = sizes + rank;

  std::vector<int64_t> denseStrides(static_cast<size_t>(rank));
  if (rank > 0) {
    denseStrides[static_cast<size_t>(rank - 1)] = sizes[rank - 1];
    for (int64_t i = rank - 2; i >= 0; --i)
      denseStrides[static_cast<size_t>(i)] =
          sizes[i] * denseStrides[static_cast<size_t>(i + 1)];
  }
  auto sizeBytes = (rank > 0 ? denseStrides[0] : 1) * elementSizeBytes;

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

extern "C" FLY_RUNTIME_EXPORT void mgpuMemHostUnregister(void *ptr) {
  HIP_REPORT_IF_ERROR(hipHostUnregister(ptr));
}

extern "C" FLY_RUNTIME_EXPORT void
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

extern "C" FLY_RUNTIME_EXPORT StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float * /*allocated*/, float *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" FLY_RUNTIME_EXPORT StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t * /*allocated*/, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" FLY_RUNTIME_EXPORT void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  HIP_REPORT_IF_ERROR(hipSetDevice(device));
}
