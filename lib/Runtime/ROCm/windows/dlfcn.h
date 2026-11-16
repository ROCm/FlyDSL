// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// Windows stand-in for the part of <dlfcn.h> that FlyRocmAotRuntime.cpp uses:
// dlsym(RTLD_DEFAULT, name) returns the first export of that name among the
// modules loaded in the process, so a HIP that lacks the symbol still links.

#ifndef FLYDSL_RUNTIME_ROCM_WINDOWS_DLFCN_H
#define FLYDSL_RUNTIME_ROCM_WINDOWS_DLFCN_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// Windows 7 and later resolve EnumProcessModules to kernel32 (PSAPI_VERSION 2).
#include <psapi.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

#define RTLD_DEFAULT nullptr

struct FlydslDlsymHit {
  void *symbol;
  char name[1];
};

static inline void *dlsym(void *, const char *name) {
  // The runtime looks the same symbol up on every cluster launch, so the first
  // symbol found is kept for the life of the process. A miss is never kept:
  // HIP may be loaded later.
  static std::atomic<FlydslDlsymHit *> hit{nullptr};
  if (FlydslDlsymHit *cached = hit.load(std::memory_order_acquire))
    if (std::strcmp(cached->name, name) == 0)
      return cached->symbol;

  HMODULE modules[1024];
  DWORD bytes = 0;
  if (!EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &bytes))
    return nullptr;
  void *symbol = nullptr;
  DWORD count = bytes / sizeof(HMODULE);
  for (DWORD i = 0; i < count && i < 1024 && !symbol; ++i)
    symbol = reinterpret_cast<void *>(GetProcAddress(modules[i], name));
  if (!symbol)
    return nullptr;

  size_t size = std::strlen(name) + 1;
  if (auto *entry = static_cast<FlydslDlsymHit *>(std::malloc(sizeof(FlydslDlsymHit) + size))) {
    entry->symbol = symbol;
    std::memcpy(entry->name, name, size);
    FlydslDlsymHit *expected = nullptr;
    // Entries never change once published; a thread that loses the race drops its own.
    if (!hit.compare_exchange_strong(expected, entry, std::memory_order_acq_rel))
      std::free(entry);
  }
  return symbol;
}

#endif // FLYDSL_RUNTIME_ROCM_WINDOWS_DLFCN_H
