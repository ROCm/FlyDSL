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
#include <mutex>
#include <unordered_map>

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

// ---------------------------------------------------------------------------
// Caching layer: avoids calling hipModuleLoadData / hipModuleGetFunction /
// hipStreamCreate on every kernel launch.  The MLIR gpu-to-llvm lowering
// emits calls to mgpuModuleLoad + mgpuModuleGetFunction + mgpuStreamCreate
// on *every* invocation of the host wrapper, which adds ~0.8 ms of overhead
// per call.
//
// Cache key: we use a content hash of the blob data.  The JIT may reuse
// the same address for different blobs (after freeing a previous module),
// so pointer-based keys are unsafe.  We compute a fast hash of the blob
// content and use (hash, size) as the cache key.  For repeated calls from
// the same JIT'd function, the blob content is identical → cache hit.
// ---------------------------------------------------------------------------

static std::mutex g_cache_mutex;

// Simple FNV-1a hash of blob content
static uint64_t hash_blob(const void *data, size_t size) {
  const uint8_t *p = static_cast<const uint8_t *>(data);
  uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
  for (size_t i = 0; i < size; i++) {
    hash ^= p[i];
    hash *= 1099511628211ULL;  // FNV prime
  }
  return hash;
}

struct BlobKey {
  uint64_t content_hash;
  size_t size;
  bool operator==(const BlobKey &o) const {
    return content_hash == o.content_hash && size == o.size;
  }
};

struct BlobKeyHash {
  size_t operator()(const BlobKey &k) const {
    size_t h = std::hash<uint64_t>()(k.content_hash);
    h ^= std::hash<size_t>()(k.size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

static std::unordered_map<BlobKey, hipModule_t, BlobKeyHash> g_module_cache;
static std::unordered_map<hipModule_t, std::unordered_map<std::string, hipFunction_t>> g_func_cache;
static hipStream_t g_cached_stream = nullptr;
static bool g_stream_initialized = false;

extern "C" FLIR_EXPORT hipModule_t mgpuModuleLoad(void *data, size_t gpuBlobSize) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  uint64_t h = hash_blob(data, gpuBlobSize);
  BlobKey key{h, gpuBlobSize};
  auto it = g_module_cache.find(key);
  if (it != g_module_cache.end()) {
    return it->second;
  }
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  g_module_cache[key] = module;
  return module;
}

extern "C" FLIR_EXPORT hipModule_t mgpuModuleLoadJIT(void *data, int optLevel) {
  (void)data;
  (void)optLevel;
  assert(false && "This function is not available in HIP.");
  return nullptr;
}

extern "C" FLIR_EXPORT void mgpuModuleUnload(hipModule_t module) {
  // Don't unload cached modules — they're reused across calls.
  (void)module;
}

extern "C" FLIR_EXPORT hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                                const char *name) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  auto &func_map = g_func_cache[module];
  std::string key(name);
  auto it = func_map.find(key);
  if (it != func_map.end()) {
    return it->second;
  }
  hipFunction_t function = nullptr;
  hipError_t err = hipModuleGetFunction(&function, module, name);
  if (err != hipSuccess) {
    fprintf(stderr, "mgpuModuleGetFunction: failed for name='%s' module=%p err=%d\n",
            name, (void*)module, (int)err);
  }
  func_map[key] = function;
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
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  if (g_stream_initialized) {
    return g_cached_stream;
  }
  g_stream_initialized = true;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&g_cached_stream));
  return g_cached_stream;
}

// Allow Python to set the stream we use for kernel launches.
// Call this with torch.cuda.current_stream().cuda_stream to share PyTorch's stream.
extern "C" FLIR_EXPORT void mgpuSetStream(hipStream_t stream) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  g_cached_stream = stream;
  g_stream_initialized = true;
}

extern "C" FLIR_EXPORT void mgpuStreamDestroy(hipStream_t stream) {
  // Don't destroy the cached stream — it's reused across calls.
  // The MLIR lowering emits a destroy after every launch_func.
  (void)stream;
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
