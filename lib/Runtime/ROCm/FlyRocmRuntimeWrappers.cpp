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
#include <dlfcn.h>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "hip/hip_runtime.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

// First runtime error observed on this thread since the last
// flydslRuntimeTakeError(); 0 when none.  Positive values are hipError_t,
// negative values are the FLYDSL_AOT_ERR_* codes below.
thread_local static int32_t lastError = 0;

static void recordError(int32_t error) {
  if (error && !lastError)
    lastError = error;
}

#define HIP_REPORT_IF_ERROR(expr)                                                                  \
  [](hipError_t result) {                                                                          \
    if (!result)                                                                                   \
      return;                                                                                      \
    recordError(static_cast<int32_t>(result));                                                     \
    const char *name = hipGetErrorName(result);                                                    \
    if (!name)                                                                                     \
      name = "<unknown>";                                                                          \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                                       \
  }(expr)

thread_local static int32_t defaultDevice = 0;

extern "C" hipModule_t mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
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

extern "C" void mgpuModuleUnload(hipModule_t module) {
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module, const char *name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

extern "C" void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX, intptr_t gridY,
                                 intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                                 int32_t smem, hipStream_t stream, void **params, void **extra,
                                 size_t /*paramsCount*/) {
  // A null function means resolving it already failed and recorded an error.
  if (!function)
    return;
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                            smem, stream, params, extra));
}

extern "C" void mgpuLaunchClusterKernel(hipFunction_t function, intptr_t clusterX,
                                        intptr_t clusterY, intptr_t clusterZ, intptr_t gridX,
                                        intptr_t gridY, intptr_t gridZ, intptr_t blockX,
                                        intptr_t blockY, intptr_t blockZ, int32_t smem,
                                        hipStream_t stream, void **params, void **extra,
                                        size_t /*paramsCount*/) {
  if (!function)
    return;
  // Resolve hipDrvLaunchKernelEx at runtime via dlsym so that the same
  // shared library works across HIP versions (required for wheel builds).
  // Mirrors Triton's approach: triton/third_party/amd/backend/driver.c.
  using LaunchKernelExFn =
      hipError_t (*)(const HIP_LAUNCH_CONFIG *, hipFunction_t, void **, void **);
  static auto launchKernelEx =
      reinterpret_cast<LaunchKernelExFn>(dlsym(RTLD_DEFAULT, "hipDrvLaunchKernelEx"));

  if (launchKernelEx) {
    hipLaunchAttribute attrs[1];
    // hipLaunchAttributeClusterDimension == 4, hardcoded to avoid a
    // compile-time dependency on HIP headers that define the enum value.
    attrs[0].id = static_cast<hipLaunchAttributeID>(4);
    auto *clusterDims = reinterpret_cast<unsigned *>(attrs[0].value.pad);
    clusterDims[0] = static_cast<unsigned>(clusterX);
    clusterDims[1] = static_cast<unsigned>(clusterY);
    clusterDims[2] = static_cast<unsigned>(clusterZ);

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

    HIP_REPORT_IF_ERROR(launchKernelEx(&config, function, params, extra));
  } else {
    if ((clusterX > 1) || (clusterY > 1) || (clusterZ > 1)) {
      fprintf(stderr,
              "[mgpuLaunchClusterKernel] cluster=(%ld,%ld,%ld) requested but "
              "hipDrvLaunchKernelEx is unavailable; "
              "falling back to hipModuleLaunchKernel.\n",
              static_cast<long>(clusterX), static_cast<long>(clusterY),
              static_cast<long>(clusterZ));
    }
    HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                              smem, stream, params, extra));
  }
}

extern "C" hipStream_t mgpuStreamCreate() {
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  return stream;
}

extern "C" void mgpuStreamDestroy(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" hipEvent_t mgpuEventCreate() {
  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  return event;
}

extern "C" void mgpuEventDestroy(hipEvent_t event) { HIP_REPORT_IF_ERROR(hipEventDestroy(event)); }

extern "C" void mgpuEventSynchronize(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" void mgpuEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/, bool /*isHostShared*/) {
  void *ptr = nullptr;
  HIP_REPORT_IF_ERROR(hipMalloc(&ptr, sizeBytes));
  return ptr;
}

extern "C" void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
  HIP_REPORT_IF_ERROR(hipFree(ptr));
}

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" void mgpuMemset32(void *dst, int value, size_t count, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst), value, count, stream));
}

extern "C" void mgpuMemset16(void *dst, int shortValue, size_t count, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), shortValue, count, stream));
}

extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

extern "C" void mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                                          int64_t elementSizeBytes) {
  int64_t *sizes = descriptor->sizes;
  int64_t *strides = sizes + rank;

  std::vector<int64_t> denseStrides(static_cast<size_t>(rank));
  if (rank > 0) {
    denseStrides[static_cast<size_t>(rank - 1)] = sizes[rank - 1];
    for (int64_t i = rank - 2; i >= 0; --i)
      denseStrides[static_cast<size_t>(i)] = sizes[i] * denseStrides[static_cast<size_t>(i + 1)];
  }
  auto sizeBytes = (rank > 0 ? denseStrides[0] : 1) * elementSizeBytes;

  for (int64_t i = 0; i < rank - 1; ++i)
    denseStrides[static_cast<size_t>(i)] = denseStrides[static_cast<size_t>(i + 1)];
  if (rank > 0)
    denseStrides[static_cast<size_t>(rank - 1)] = 1;

  for (int64_t i = 0; i < rank; ++i)
    assert(strides[i] == denseStrides[static_cast<size_t>(i)]);

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

extern "C" void mgpuMemHostUnregister(void *ptr) { HIP_REPORT_IF_ERROR(hipHostUnregister(ptr)); }

extern "C" void mgpuMemHostUnregisterMemRef(int64_t /*rank*/,
                                            StridedMemRefType<char, 1> *descriptor,
                                            int64_t elementSizeBytes) {
  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

template <typename T> static void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  HIP_REPORT_IF_ERROR(hipSetDevice(defaultDevice));
  HIP_REPORT_IF_ERROR(hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0));
}

extern "C" StridedMemRefType<float, 1> mgpuMemGetDeviceMemRef1dFloat(float * /*allocated*/,
                                                                     float *aligned, int64_t offset,
                                                                     int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" StridedMemRefType<int32_t, 1> mgpuMemGetDeviceMemRef1dInt32(int32_t * /*allocated*/,
                                                                       int32_t *aligned,
                                                                       int64_t offset, int64_t size,
                                                                       int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  HIP_REPORT_IF_ERROR(hipSetDevice(device));
}

//===----------------------------------------------------------------------===//
// Exported (AOT) GPU modules
//===----------------------------------------------------------------------===//
//
// An object exported by flyc.compile_aot(...).export_to_c(...) owns one
// pointer-sized state slot per embedded GPU binary.  The functions below turn
// that slot into a table of per-device module handles so one exported artifact
// can be loaded on several devices, and make init/load/unload idempotent.
//
// Every entry point takes the slot, never the state it points to: init and
// unload replace the state under an exclusive lock, while load and launch hold
// a shared lock for as long as they use it.  A launch racing with unload
// therefore either completes its host submission before unload proceeds, or
// fails with FLYDSL_AOT_ERR_NOT_INITIALIZED without touching freed memory.

enum : int32_t {
  FLYDSL_AOT_ERR_NOT_INITIALIZED = -1,
  FLYDSL_AOT_ERR_NOT_LOADED = -2,
  FLYDSL_AOT_ERR_INVALID_DEVICE = -3,
};

namespace {

struct AotDeviceModule {
  hipModule_t module = nullptr;
  // Keyed by the kernel-name constant of the exported object, which lives as
  // long as the object's state slot, so launches never copy the name.
  std::unordered_map<const char *, hipFunction_t> functions;
};

struct AotModule {
  explicit AotModule(const void *binary) : binary(binary) {}

  const void *binary;
  // Guards `devices`.
  std::mutex mutex;
  std::unordered_map<int, AotDeviceModule> devices;
};

// Guards every state slot: exclusive to create or destroy an AotModule,
// shared to use one.
std::shared_mutex aotStateMutex;

// Resolve `name` in `aot` on the calling thread's current device.  The caller
// must hold a shared or exclusive `aotStateMutex` lock so `aot` and its device
// modules stay alive through any subsequent use of the returned handle.
hipFunction_t getAotModuleFunction(AotModule *aot, const char *name) {
  int device = 0;
  if (hipError_t err = hipGetDevice(&device)) {
    recordError(err);
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(aot->mutex);
  auto it = aot->devices.find(device);
  if (it == aot->devices.end()) {
    recordError(FLYDSL_AOT_ERR_NOT_LOADED);
    return nullptr;
  }
  auto &functions = it->second.functions;
  auto cached = functions.find(name);
  if (cached != functions.end())
    return cached->second;

  hipFunction_t function = nullptr;
  if (hipError_t err = hipModuleGetFunction(&function, it->second.module, name)) {
    recordError(err);
    return nullptr;
  }
  functions.emplace(name, function);
  return function;
}

} // namespace

extern "C" int32_t flydslRuntimeTakeError() {
  int32_t error = lastError;
  lastError = 0;
  return error;
}

extern "C" int32_t flydslAotModuleInit(AotModule **slot, const void *binary) {
  std::unique_lock<std::shared_mutex> lock(aotStateMutex);
  if (!*slot)
    *slot = new AotModule(binary);
  return 0;
}

// Load the module on `device`, or on the current device when `device` < 0.
extern "C" int32_t flydslAotModuleLoad(AotModule **slot, int32_t device) {
  std::shared_lock<std::shared_mutex> stateLock(aotStateMutex);
  AotModule *aot = *slot;
  if (!aot)
    return FLYDSL_AOT_ERR_NOT_INITIALIZED;

  int current = 0;
  if (hipError_t err = hipGetDevice(&current))
    return err;
  int count = 0;
  if (hipError_t err = hipGetDeviceCount(&count))
    return err;
  if (device < 0)
    device = current;
  if (device >= count)
    return FLYDSL_AOT_ERR_INVALID_DEVICE;

  std::lock_guard<std::mutex> lock(aot->mutex);
  if (aot->devices.count(device))
    return 0;
  if (device != current) {
    if (hipError_t err = hipSetDevice(device))
      return err;
  }
  hipModule_t module = nullptr;
  hipError_t err = hipModuleLoadData(&module, aot->binary);
  if (device != current)
    (void)hipSetDevice(current);
  if (err)
    return err;
  aot->devices[device].module = module;
  return 0;
}

// Unload the module from every device it was loaded on and release the state.
extern "C" int32_t flydslAotModuleUnload(AotModule **slot) {
  std::unique_lock<std::shared_mutex> lock(aotStateMutex);
  AotModule *aot = *slot;
  *slot = nullptr;
  if (!aot)
    return 0;

  int32_t status = 0;
  int current = 0;
  bool restore = hipGetDevice(&current) == hipSuccess;
  for (auto &[device, entry] : aot->devices) {
    hipError_t err = hipSetDevice(device);
    if (!err)
      err = hipModuleUnload(entry.module);
    if (err && !status)
      status = err;
  }
  if (restore)
    (void)hipSetDevice(current);
  delete aot;
  return status;
}

// Resolve `name` in the module loaded on the calling thread's current device.
// Failures are recorded for flydslRuntimeTakeError() and return null, which
// mgpuLaunchKernel treats as "skip this launch".
extern "C" hipFunction_t flydslAotModuleGetFunction(AotModule **slot, const char *name) {
  std::shared_lock<std::shared_mutex> stateLock(aotStateMutex);
  AotModule *aot = *slot;
  if (!aot) {
    recordError(FLYDSL_AOT_ERR_NOT_INITIALIZED);
    return nullptr;
  }
  return getAotModuleFunction(aot, name);
}

extern "C" void flydslAotModuleLaunchKernel(AotModule **slot, const char *name, intptr_t gridX,
                                            intptr_t gridY, intptr_t gridZ, intptr_t blockX,
                                            intptr_t blockY, intptr_t blockZ, int32_t smem,
                                            hipStream_t stream, void **params, void **extra,
                                            size_t paramsCount) {
  std::shared_lock<std::shared_mutex> stateLock(aotStateMutex);
  AotModule *aot = *slot;
  if (!aot) {
    recordError(FLYDSL_AOT_ERR_NOT_INITIALIZED);
    return;
  }
  hipFunction_t function = getAotModuleFunction(aot, name);
  mgpuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params,
                   extra, paramsCount);
}

extern "C" void flydslAotModuleLaunchClusterKernel(AotModule **slot, const char *name,
                                                   intptr_t clusterX, intptr_t clusterY,
                                                   intptr_t clusterZ, intptr_t gridX,
                                                   intptr_t gridY, intptr_t gridZ, intptr_t blockX,
                                                   intptr_t blockY, intptr_t blockZ, int32_t smem,
                                                   hipStream_t stream, void **params, void **extra,
                                                   size_t paramsCount) {
  std::shared_lock<std::shared_mutex> stateLock(aotStateMutex);
  AotModule *aot = *slot;
  if (!aot) {
    recordError(FLYDSL_AOT_ERR_NOT_INITIALIZED);
    return;
  }
  hipFunction_t function = getAotModuleFunction(aot, name);
  mgpuLaunchClusterKernel(function, clusterX, clusterY, clusterZ, gridX, gridY, gridZ, blockX,
                          blockY, blockZ, smem, stream, params, extra, paramsCount);
}
