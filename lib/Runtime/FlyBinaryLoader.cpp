// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// LLJIT-based .o loader: load relocatable objects in-process without gcc.

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

FLY_EXPORT
void *flyBinaryModuleCreate(const char *objData, size_t objSize,
                            const char **sharedLibs, size_t numLibs) {
  ensureInitialized();

  for (size_t i = 0; i < numLibs; i++) {
    std::string errMsg;
    if (sys::DynamicLibrary::LoadLibraryPermanently(sharedLibs[i], &errMsg)) {
      fprintf(stderr, "FlyBinaryLoader: failed to load %s: %s\n",
              sharedLibs[i], errMsg.c_str());
      return nullptr;
    }
  }

  auto jitExpected = LLJITBuilder().create();
  if (!jitExpected) {
    fprintf(stderr, "FlyBinaryLoader: failed to create LLJIT: %s\n",
            toString(jitExpected.takeError()).c_str());
    return nullptr;
  }
  auto jit = std::move(*jitExpected);

  auto &mainDylib = jit->getMainJITDylib();
  auto gen = DynamicLibrarySearchGenerator::GetForCurrentProcess(
      jit->getDataLayout().getGlobalPrefix());
  if (!gen) {
    fprintf(stderr, "FlyBinaryLoader: failed to create symbol generator: %s\n",
            toString(gen.takeError()).c_str());
    return nullptr;
  }
  mainDylib.addGenerator(std::move(*gen));

  auto buf = MemoryBuffer::getMemBufferCopy(StringRef(objData, objSize));
  if (auto err = jit->addObjectFile(std::move(buf))) {
    fprintf(stderr, "FlyBinaryLoader: failed to add object file: %s\n",
            toString(std::move(err)).c_str());
    return nullptr;
  }

  if (auto err = jit->initialize(mainDylib)) {
    fprintf(stderr, "FlyBinaryLoader: failed to initialize: %s\n",
            toString(std::move(err)).c_str());
    return nullptr;
  }

  auto mod = new FlyBinaryModule{std::move(jit)};
  return static_cast<void *>(mod);
}

FLY_EXPORT
void *flyBinaryModuleLookup(void *handle, const char *name) {
  auto *mod = static_cast<FlyBinaryModule *>(handle);
  auto sym = mod->jit->lookup(name);
  if (!sym) {
    consumeError(sym.takeError());
    return nullptr;
  }
  return reinterpret_cast<void *>(sym->getValue());
}

FLY_EXPORT
void flyBinaryModuleDestroy(void *handle) {
  delete static_cast<FlyBinaryModule *>(handle);
}

} // extern "C"
