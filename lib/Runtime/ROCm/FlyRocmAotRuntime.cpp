//===- FlyRocmAotRuntime.cpp - Self-contained ROCm AOT runtime ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HIP module lifecycle and launch support embedded into exported AOT objects.
// This intentionally uses only HIP, libc, libdl and pthread APIs so a final
// application or shared library has no FlyDSL runtime dependency.
//
//===----------------------------------------------------------------------===//

#include "FlyRocmRuntimeError.h"

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <pthread.h>

enum : int32_t {
  FLYDSL_AOT_ERR_NOT_INITIALIZED = -1,
  FLYDSL_AOT_ERR_NOT_LOADED = -2,
};

namespace {

struct AotFunction {
  const char *name;
  hipFunction_t function;
  AotFunction *next;
};

struct AotDeviceModule {
  int device;
  hipModule_t module;
  AotFunction *functions;
  AotDeviceModule *next;
};

struct AotModule {
  const void *binary;
  pthread_mutex_t mutex;
  AotDeviceModule *devices;
};

// Guards every state slot: exclusive to create or destroy an AotModule,
// shared to use one. Embedded runtime symbols are localized into each export,
// so independently exported objects do not share this lock.
pthread_rwlock_t aotStateLock = PTHREAD_RWLOCK_INITIALIZER;

AotDeviceModule *findDevice(AotModule *aot, int device) {
  for (AotDeviceModule *entry = aot->devices; entry; entry = entry->next)
    if (entry->device == device)
      return entry;
  return nullptr;
}

hipFunction_t getFunction(AotModule *aot, const char *name) {
  int device = 0;
  if (hipError_t err = hipGetDevice(&device)) {
    flydslRecordRuntimeError(err);
    return nullptr;
  }

  pthread_mutex_lock(&aot->mutex);
  AotDeviceModule *entry = findDevice(aot, device);
  if (!entry) {
    pthread_mutex_unlock(&aot->mutex);
    flydslRecordRuntimeError(FLYDSL_AOT_ERR_NOT_LOADED);
    return nullptr;
  }
  for (AotFunction *cached = entry->functions; cached; cached = cached->next) {
    // Kernel names are internal constants in the exported object, so pointer
    // identity is stable and avoids copying or comparing their contents.
    if (cached->name == name) {
      hipFunction_t function = cached->function;
      pthread_mutex_unlock(&aot->mutex);
      return function;
    }
  }

  hipFunction_t function = nullptr;
  if (hipError_t err = hipModuleGetFunction(&function, entry->module, name)) {
    pthread_mutex_unlock(&aot->mutex);
    flydslRecordRuntimeError(err);
    return nullptr;
  }
  if (auto *cached = static_cast<AotFunction *>(malloc(sizeof(AotFunction)))) {
    cached->name = name;
    cached->function = function;
    cached->next = entry->functions;
    entry->functions = cached;
  }
  pthread_mutex_unlock(&aot->mutex);
  return function;
}

void launchKernel(hipFunction_t function, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
                  intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem,
                  hipStream_t stream, void **params, void **extra) {
  if (!function)
    return;
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                            smem, stream, params, extra));
}

void launchClusterKernel(hipFunction_t function, intptr_t clusterX, intptr_t clusterY,
                         intptr_t clusterZ, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
                         intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem,
                         hipStream_t stream, void **params, void **extra) {
  if (!function)
    return;
  using LaunchKernelExFn =
      hipError_t (*)(const HIP_LAUNCH_CONFIG *, hipFunction_t, void **, void **);
  auto launchKernelEx =
      reinterpret_cast<LaunchKernelExFn>(dlsym(RTLD_DEFAULT, "hipDrvLaunchKernelEx"));

  if (launchKernelEx) {
    hipLaunchAttribute attrs[1];
    // hipLaunchAttributeClusterDimension == 4. Keep the numeric value so the
    // object remains linkable against HIP versions that predate the enum.
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
    return;
  }

  if ((clusterX > 1) || (clusterY > 1) || (clusterZ > 1)) {
    fprintf(stderr,
            "[FlyDSL AOT] cluster=(%ld,%ld,%ld) requested but hipDrvLaunchKernelEx is "
            "unavailable; falling back to hipModuleLaunchKernel.\n",
            static_cast<long>(clusterX), static_cast<long>(clusterY), static_cast<long>(clusterZ));
  }
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                            smem, stream, params, extra));
}

void freeFunctions(AotFunction *function) {
  while (function) {
    AotFunction *next = function->next;
    free(function);
    function = next;
  }
}

} // namespace

extern "C" FLYDSL_RUNTIME_API int32_t flydslAotModuleInit(AotModule **slot, const void *binary) {
  pthread_rwlock_wrlock(&aotStateLock);
  if (!*slot) {
    auto *aot = static_cast<AotModule *>(calloc(1, sizeof(AotModule)));
    if (!aot) {
      pthread_rwlock_unlock(&aotStateLock);
      return hipErrorOutOfMemory;
    }
    if (pthread_mutex_init(&aot->mutex, nullptr)) {
      free(aot);
      pthread_rwlock_unlock(&aotStateLock);
      return hipErrorUnknown;
    }
    aot->binary = binary;
    *slot = aot;
  }
  pthread_rwlock_unlock(&aotStateLock);
  return 0;
}

// Load the module on `device`, or on the current device when `device` < 0.
extern "C" FLYDSL_RUNTIME_API int32_t flydslAotModuleLoad(AotModule **slot, int32_t device) {
  pthread_rwlock_rdlock(&aotStateLock);
  AotModule *aot = *slot;
  if (!aot) {
    pthread_rwlock_unlock(&aotStateLock);
    return FLYDSL_AOT_ERR_NOT_INITIALIZED;
  }

  int current = 0;
  if (hipError_t err = hipGetDevice(&current)) {
    pthread_rwlock_unlock(&aotStateLock);
    return err;
  }
  if (device < 0)
    device = current;

  pthread_mutex_lock(&aot->mutex);
  if (findDevice(aot, device)) {
    pthread_mutex_unlock(&aot->mutex);
    pthread_rwlock_unlock(&aotStateLock);
    return 0;
  }
  auto *entry = static_cast<AotDeviceModule *>(calloc(1, sizeof(AotDeviceModule)));
  if (!entry) {
    pthread_mutex_unlock(&aot->mutex);
    pthread_rwlock_unlock(&aotStateLock);
    return hipErrorOutOfMemory;
  }
  if (device != current) {
    if (hipError_t err = hipSetDevice(device)) {
      // Some HIP implementations leave the thread's device selection in an
      // unusable state after a failed set. Restore the known-good device.
      (void)hipSetDevice(current);
      (void)hipGetLastError();
      free(entry);
      pthread_mutex_unlock(&aot->mutex);
      pthread_rwlock_unlock(&aotStateLock);
      return err;
    }
  }

  hipModule_t module = nullptr;
  hipError_t err = hipModuleLoadData(&module, aot->binary);
  if (device != current)
    (void)hipSetDevice(current);
  if (err) {
    free(entry);
    pthread_mutex_unlock(&aot->mutex);
    pthread_rwlock_unlock(&aotStateLock);
    return err;
  }

  entry->device = device;
  entry->module = module;
  entry->next = aot->devices;
  aot->devices = entry;
  pthread_mutex_unlock(&aot->mutex);
  pthread_rwlock_unlock(&aotStateLock);
  return 0;
}

// Unload the module from every device it was loaded on and release the state.
extern "C" FLYDSL_RUNTIME_API int32_t flydslAotModuleUnload(AotModule **slot) {
  pthread_rwlock_wrlock(&aotStateLock);
  AotModule *aot = *slot;
  *slot = nullptr;
  if (!aot) {
    pthread_rwlock_unlock(&aotStateLock);
    return 0;
  }

  int32_t status = 0;
  int current = 0;
  bool restore = hipGetDevice(&current) == hipSuccess;
  AotDeviceModule *entry = aot->devices;
  while (entry) {
    hipError_t err = hipSetDevice(entry->device);
    if (!err) {
      // The state lock prevents new host submissions, but previously submitted
      // kernels may still be executing asynchronously.
      err = hipDeviceSynchronize();
      hipError_t unloadErr = hipModuleUnload(entry->module);
      if (!err)
        err = unloadErr;
    }
    if (err && !status)
      status = err;
    freeFunctions(entry->functions);
    AotDeviceModule *next = entry->next;
    free(entry);
    entry = next;
  }
  if (restore)
    (void)hipSetDevice(current);
  pthread_mutex_destroy(&aot->mutex);
  free(aot);
  pthread_rwlock_unlock(&aotStateLock);
  return status;
}

extern "C" FLYDSL_RUNTIME_API void
flydslAotModuleLaunchKernel(AotModule **slot, const char *name, intptr_t gridX, intptr_t gridY,
                            intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                            int32_t smem, hipStream_t stream, void **params, void **extra,
                            size_t /*paramsCount*/) {
  pthread_rwlock_rdlock(&aotStateLock);
  AotModule *aot = *slot;
  if (!aot) {
    flydslRecordRuntimeError(FLYDSL_AOT_ERR_NOT_INITIALIZED);
    pthread_rwlock_unlock(&aotStateLock);
    return;
  }
  hipFunction_t function = getFunction(aot, name);
  launchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra);
  pthread_rwlock_unlock(&aotStateLock);
}

extern "C" FLYDSL_RUNTIME_API void
flydslAotModuleLaunchClusterKernel(AotModule **slot, const char *name, intptr_t clusterX,
                                   intptr_t clusterY, intptr_t clusterZ, intptr_t gridX,
                                   intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                                   intptr_t blockZ, int32_t smem, hipStream_t stream, void **params,
                                   void **extra, size_t /*paramsCount*/) {
  pthread_rwlock_rdlock(&aotStateLock);
  AotModule *aot = *slot;
  if (!aot) {
    flydslRecordRuntimeError(FLYDSL_AOT_ERR_NOT_INITIALIZED);
    pthread_rwlock_unlock(&aotStateLock);
    return;
  }
  hipFunction_t function = getFunction(aot, name);
  launchClusterKernel(function, clusterX, clusterY, clusterZ, gridX, gridY, gridZ, blockX, blockY,
                      blockZ, smem, stream, params, extra);
  pthread_rwlock_unlock(&aotStateLock);
}
