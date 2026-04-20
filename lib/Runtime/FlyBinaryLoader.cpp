// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Thin C API wrapper around LLVM LLJIT for loading relocatable .o files
// in-process without requiring gcc at load time.
//
// Exposes three functions:
//   flyBinaryModuleCreate(data, size, shared_libs, num_libs) -> handle
//   flyBinaryModuleLookup(handle, name) -> void*
//   flyBinaryModuleDestroy(handle)

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/DynamicLibrary.h"
#include <cstdio>
#include <memory>
#include <mutex>

using namespace llvm;
using namespace llvm::orc;

namespace {

struct FlyBinaryModule {
  std::unique_ptr<LLJIT> jit;
};

std::once_flag g_initFlag;

void ensureInitialized() {
  std::call_once(g_initFlag, [] {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
  });
}

} // namespace

#define FLY_EXPORT __attribute__((visibility("default")))

extern "C" {

/// Create a binary module from a raw .o file in memory.
/// shared_libs: array of paths to shared libraries to resolve symbols from.
/// Returns opaque handle, or nullptr on failure.
FLY_EXPORT
void *flyBinaryModuleCreate(const char *objData, size_t objSize,
                            const char **sharedLibs, size_t numLibs) {
  ensureInitialized();

  // Load shared libraries so their symbols are available for linking.
  for (size_t i = 0; i < numLibs; i++) {
    std::string errMsg;
    if (sys::DynamicLibrary::LoadLibraryPermanently(sharedLibs[i], &errMsg)) {
      fprintf(stderr, "FlyBinaryLoader: failed to load %s: %s\n",
              sharedLibs[i], errMsg.c_str());
      return nullptr;
    }
  }

  // Create LLJIT instance.
  auto jitExpected = LLJITBuilder().create();
  if (!jitExpected) {
    fprintf(stderr, "FlyBinaryLoader: failed to create LLJIT: %s\n",
            toString(jitExpected.takeError()).c_str());
    return nullptr;
  }
  auto jit = std::move(*jitExpected);

  // Register process symbols (so mgpuModuleLoad etc. from loaded libs resolve).
  auto &mainDylib = jit->getMainJITDylib();
  auto gen = DynamicLibrarySearchGenerator::GetForCurrentProcess(
      jit->getDataLayout().getGlobalPrefix());
  if (!gen) {
    fprintf(stderr, "FlyBinaryLoader: failed to create symbol generator: %s\n",
            toString(gen.takeError()).c_str());
    return nullptr;
  }
  mainDylib.addGenerator(std::move(*gen));

  // Add the object file.
  auto buf = MemoryBuffer::getMemBufferCopy(StringRef(objData, objSize));
  if (auto err = jit->addObjectFile(std::move(buf))) {
    fprintf(stderr, "FlyBinaryLoader: failed to add object file: %s\n",
            toString(std::move(err)).c_str());
    return nullptr;
  }

  // Run initializers (ELF .init_array — loads GPU modules via mgpuModuleLoad).
  if (auto err = jit->initialize(mainDylib)) {
    fprintf(stderr, "FlyBinaryLoader: failed to initialize: %s\n",
            toString(std::move(err)).c_str());
    return nullptr;
  }

  auto mod = new FlyBinaryModule{std::move(jit)};
  return static_cast<void *>(mod);
}

/// Look up a symbol by name. Returns function pointer or nullptr.
FLY_EXPORT
void *flyBinaryModuleLookup(void *handle, const char *name) {
  auto *mod = static_cast<FlyBinaryModule *>(handle);
  auto sym = mod->jit->lookup(name);
  if (!sym) {
    // Consume error silently — caller checks null.
    consumeError(sym.takeError());
    return nullptr;
  }
  return reinterpret_cast<void *>(sym->getValue());
}

/// Destroy the binary module.
FLY_EXPORT
void flyBinaryModuleDestroy(void *handle) {
  delete static_cast<FlyBinaryModule *>(handle);
}

} // extern "C"
