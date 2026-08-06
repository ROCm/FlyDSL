//===- FlyCudaRuntimeWrappers.cpp - CUDA runtime for MLIR JIT -------------===//
//
// Derived from LLVM Project: mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin CUDA driver-API runtime wrappers for the MLIR ExecutionEngine JIT.
// Exposes the same vendor-neutral `mgpu*` symbol set as the ROCm fork
// (FlyRocmRuntimeWrappers.cpp) so the host-side launch / explicit-module
// offloading IR (which emits calls to mgpuModuleLoad / mgpuLaunchKernel /
// mgpuModuleUnload) links unchanged against either backend.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>

#include "cuda.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#define CUDA_REPORT_IF_ERROR(expr)                                                                 \
  [](CUresult result) {                                                                            \
    if (!result)                                                                                   \
      return;                                                                                      \
    const char *name = nullptr;                                                                    \
    cuGetErrorName(result, &name);                                                                 \
    if (!name)                                                                                     \
      name = "<unknown>";                                                                          \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                                       \
  }(expr)

thread_local static int32_t defaultDevice = 0;

// Ensure a CUDA context is current for the duration of the instance.
//
// Unlike upstream's CudaRuntimeWrappers.cpp we do NOT unconditionally push the
// primary context of `defaultDevice`. FlyDSL is embedded in a host framework
// (PyTorch) that owns device selection and hands us its streams, so forcing
// device 0's context makes every launch on `cuda:N` (N > 0) fail with
// CUDA_ERROR_INVALID_HANDLE -- the stream belongs to a different context. The
// ROCm fork has no such problem because HIP simply inherits the calling
// thread's current device; this mirrors that behaviour.
//
// When a context is already current we use it as-is. Only in the standalone
// case (no host framework, nothing bound) do we retain and push the primary
// context of `defaultDevice`, which is also what makes mgpuSetDefaultDevice()
// meaningful. Retained contexts are cached and never released -- the process
// keeps its device contexts alive for its whole lifetime, as upstream does.
namespace {
class ScopedContext {
public:
  ScopedContext() {
    // Raw call: CUDA_ERROR_NOT_INITIALIZED here just means "no host framework
    // has set anything up", which is the fallback path below, not an error.
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr)
      return;
    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(getPrimaryContext(defaultDevice)));
    pushed = true;
  }
  ~ScopedContext() {
    if (pushed)
      CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr));
  }

  // Retain (once per device) and return the primary context of `ordinal`.
  static CUcontext getPrimaryContext(int32_t ordinal) {
    static std::once_flag initFlag;
    std::call_once(initFlag, [] { CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0)); });

    static std::mutex mutex;
    static std::map<int32_t, CUcontext> contexts;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = contexts.find(ordinal);
    if (it != contexts.end())
      return it->second;

    CUdevice device;
    CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/ordinal));
    CUcontext ctx = nullptr;
    CUDA_REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&ctx, device));
    contexts[ordinal] = ctx;
    return ctx;
  }

private:
  bool pushed = false;
};
} // namespace

// Opt the kernel into `smem` bytes of dynamic shared memory. Anything above
// 48KB needs this explicit opt-in; warn up front when the request exceeds what
// the device allows, since the driver would otherwise only report a bare
// CUDA_ERROR_INVALID_VALUE. cuKernelSetAttribute takes the device explicitly.
static void setDynamicSharedMemory(CUkernel kernel, int32_t smem) {
  if (smem <= 0)
    return;
  int32_t maxShmem = 0;
  CUdevice device;
  CUDA_REPORT_IF_ERROR(cuCtxGetDevice(&device));
  CUDA_REPORT_IF_ERROR(cuDeviceGetAttribute(
      &maxShmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
  if (maxShmem < smem)
    fprintf(stderr,
            "Requested shared memory (%dB) is larger than the maximum allowed "
            "shared memory (%dB) for this device\n",
            smem, maxShmem);
  CUDA_REPORT_IF_ERROR(
      cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem, kernel, device));
}

// MLIR's offloading handler loads each GPU binary exactly once per process (a
// module constructor calls mgpuModuleLoad), but a CUmodule/CUfunction pair is
// bound to the CUDA context it was created in -- reusing a device-0 CUfunction
// on device 1 fails with CUDA_ERROR_INVALID_HANDLE and silently wrong results.
//
// CUDA 12's library management API exists for exactly this: a CUlibrary is
// context-independent and the driver instantiates it per context on demand. So
// we hand MLIR a CUlibrary/CUkernel behind its CUmodule/CUfunction handles and
// resolve the context-bound CUfunction at launch. Requires a CUDA 12 / r525+ driver.

extern "C" CUmodule mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  ScopedContext scopedContext;
  CUlibrary library = nullptr;
  CUDA_REPORT_IF_ERROR(cuLibraryLoadData(&library, data, nullptr, nullptr, 0, nullptr, nullptr, 0));
  return reinterpret_cast<CUmodule>(library);
}

extern "C" CUmodule mgpuModuleLoadJIT(void *data, int optLevel) {
  ScopedContext scopedContext;
  CUlibrary library = nullptr;
  char jitErrorBuffer[4096] = {0};
  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                               CU_JIT_OPTIMIZATION_LEVEL};
  void *jitOptionsVals[] = {jitErrorBuffer, reinterpret_cast<void *>(sizeof(jitErrorBuffer)),
                            reinterpret_cast<void *>(optLevel)};
  CUresult result =
      cuLibraryLoadData(&library, data, jitOptions, jitOptionsVals, 3, nullptr, nullptr, 0);
  if (result) {
    fprintf(stderr, "JIT compilation failed with: '%s'\n", jitErrorBuffer);
    CUDA_REPORT_IF_ERROR(result);
  }
  return reinterpret_cast<CUmodule>(library);
}

extern "C" void mgpuModuleUnload(CUmodule module) {
  CUDA_REPORT_IF_ERROR(cuLibraryUnload(reinterpret_cast<CUlibrary>(module)));
}

extern "C" CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUkernel kernel = nullptr;
  CUDA_REPORT_IF_ERROR(cuLibraryGetKernel(&kernel, reinterpret_cast<CUlibrary>(module), name));
  return reinterpret_cast<CUfunction>(kernel);
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match MLIR's
// index type, avoiding casts in the generated code.
extern "C" void mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                                 intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                                 int32_t smem, CUstream stream, void **params, void **extra,
                                 size_t /*paramsCount*/) {
  ScopedContext scopedContext;
  auto kernel = reinterpret_cast<CUkernel>(function);
  CUfunction fn = nullptr;
  CUDA_REPORT_IF_ERROR(cuKernelGetFunction(&fn, kernel));
  if (!fn)
    return;
  setDynamicSharedMemory(kernel, smem);
  CUDA_REPORT_IF_ERROR(
      cuLaunchKernel(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra));
}

// Stage-one CUDA backend does not support thread-block clusters (Hopper+).
// The cluster launch path (cuLaunchKernelEx / CUlaunchConfig) is gated behind
// CUDA-version macros and may be unavailable at compile time, so we degrade to
// a plain launch and only warn if a non-trivial cluster was actually requested.
// Real cluster support belongs to a later stage.
extern "C" void mgpuLaunchClusterKernel(CUfunction function, intptr_t clusterX, intptr_t clusterY,
                                        intptr_t clusterZ, intptr_t gridX, intptr_t gridY,
                                        intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                                        intptr_t blockZ, int32_t smem, CUstream stream,
                                        void **params, void **extra, size_t /*paramsCount*/) {
  ScopedContext scopedContext;
  if ((clusterX > 1) || (clusterY > 1) || (clusterZ > 1)) {
    fprintf(stderr,
            "[mgpuLaunchClusterKernel] cluster=(%ld,%ld,%ld) requested but the "
            "stage-one CUDA backend has no cluster support; falling back to a "
            "plain launch.\n",
            static_cast<long>(clusterX), static_cast<long>(clusterY), static_cast<long>(clusterZ));
  }
  auto kernel = reinterpret_cast<CUkernel>(function);
  CUfunction fn = nullptr;
  CUDA_REPORT_IF_ERROR(cuKernelGetFunction(&fn, kernel));
  if (!fn)
    return;
  setDynamicSharedMemory(kernel, smem);
  CUDA_REPORT_IF_ERROR(
      cuLaunchKernel(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra));
}

extern "C" CUstream mgpuStreamCreate() {
  ScopedContext scopedContext;
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

extern "C" void mgpuStreamDestroy(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(CUstream stream, CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" CUevent mgpuEventCreate() {
  ScopedContext scopedContext;
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" void mgpuEventDestroy(CUevent event) { CUDA_REPORT_IF_ERROR(cuEventDestroy(event)); }

extern "C" void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern "C" void mgpuEventRecord(CUevent event, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, CUstream /*stream*/, bool /*isHostShared*/) {
  ScopedContext scopedContext;
  CUdeviceptr ptr = 0;
  if (sizeBytes != 0)
    CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" void mgpuMemFree(void *ptr, CUstream /*stream*/) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src), sizeBytes, stream));
}

extern "C" void mgpuMemset32(void *dst, int value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst), value, count, stream));
}

extern "C" void mgpuMemset16(void *dst, int shortValue, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(
      cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst), shortValue, count, stream));
}

extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostRegister(ptr, sizeBytes, /*flags=*/0));
}

extern "C" void mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                                          int64_t elementSizeBytes) {
  int64_t *sizes = descriptor->sizes;
  int64_t *strides = sizes + rank;

  int64_t denseStride = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    (void)strides;
    denseStride *= sizes[i];
  }
  auto sizeBytes = denseStride * elementSizeBytes;
  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

extern "C" void mgpuMemHostUnregister(void *ptr) { CUDA_REPORT_IF_ERROR(cuMemHostUnregister(ptr)); }

extern "C" void mgpuMemHostUnregisterMemRef(int64_t /*rank*/,
                                            StridedMemRefType<char, 1> *descriptor,
                                            int64_t elementSizeBytes) {
  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

extern "C" void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  // Retain the new device's primary context eagerly so a bad ordinal is
  // reported here rather than at the next launch (mirrors the ROCm fork's
  // hipSetDevice).
  (void)ScopedContext::getPrimaryContext(device);
}
