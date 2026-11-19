// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// In-process loader for objects written by export_to_c: JIT-links the
// relocatable object into this process with ORC JITLink, so no system linker
// or shared library on disk is involved.
//
// Nothing here plays the part of the platform runtime: static constructors
// are not run and thread-local storage is not serviced. Exported objects use
// neither, and objects that do are rejected before linking.

#include "HostObject/host_object.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>
#include <vector>

namespace {

using namespace llvm;
using namespace llvm::orc;

// Reject sections that only a platform runtime would service.
Error checkSupported(StringRef object) {
  auto file = object::ELF64LEFile::create(object);
  if (!file)
    return file.takeError();
  auto sections = file->sections();
  if (!sections)
    return sections.takeError();
  for (const object::ELF64LE::Shdr &section : *sections) {
    auto name = file->getSectionName(section);
    if (!name)
      return name.takeError();
    const char *kind = nullptr;
    if (section.sh_flags & ELF::SHF_TLS)
      kind = "thread-local storage";
    else if (section.sh_type == ELF::SHT_INIT_ARRAY || section.sh_type == ELF::SHT_PREINIT_ARRAY ||
             name->starts_with(".ctors"))
      kind = "static constructors";
    else if (section.sh_type == ELF::SHT_FINI_ARRAY || name->starts_with(".dtors"))
      kind = "static destructors";
    if (kind)
      return make_error<StringError>("section " + *name + " holds " + kind +
                                         ", which in-process loading does not support",
                                     inconvertibleErrorCode());
    if (section.sh_type != ELF::SHT_SYMTAB)
      continue;
    // Also catches thread-local data defined outside the object.
    auto symbols = file->symbols(&section);
    if (!symbols)
      return symbols.takeError();
    for (const object::ELF64LE::Sym &symbol : *symbols)
      if (symbol.getType() == ELF::STT_TLS)
        return make_error<StringError>(
            "the object uses thread-local storage, which in-process loading does not support",
            inconvertibleErrorCode());
  }
  return Error::success();
}

struct LoadedObject {
  std::unique_ptr<ExecutionSession> session;
  std::unique_ptr<ObjectLinkingLayer> layer;
  JITDylib *dylib = nullptr;

  ~LoadedObject() {
    if (session)
      consumeError(session->endSession());
  }
};

Expected<std::unique_ptr<LoadedObject>> loadObject(StringRef object, ArrayRef<std::string> libs) {
  if (Error err = checkSupported(object))
    return std::move(err);
  for (const std::string &lib : libs) {
    std::string error;
    // Global and permanent, so the code may also find them via dlsym(RTLD_DEFAULT).
    if (sys::DynamicLibrary::LoadLibraryPermanently(lib.c_str(), &error))
      return make_error<StringError>("cannot load " + lib + ": " + error, inconvertibleErrorCode());
  }

  auto control = SelfExecutorProcessControl::Create();
  if (!control)
    return control.takeError();
  auto loaded = std::make_unique<LoadedObject>();
  loaded->session = std::make_unique<ExecutionSession>(std::move(*control));
  ExecutionSession &session = *loaded->session;
  auto memory = session.getExecutorProcessControl().createDefaultMemoryManager();
  if (!memory)
    return memory.takeError();
  loaded->layer = std::make_unique<ObjectLinkingLayer>(session, std::move(*memory));

  auto ehFrames = EHFrameRegistrationPlugin::Create(session);
  if (!ehFrames)
    return ehFrames.takeError();
  loaded->layer->addPlugin(std::move(*ehFrames));

  JITDylib &dylib = session.createBareJITDylib("main");
  loaded->dylib = &dylib;
  auto process = DynamicLibrarySearchGenerator::GetForCurrentProcess(/*GlobalPrefix=*/'\0');
  if (!process)
    return process.takeError();
  dylib.addGenerator(std::move(*process));

  if (Error err = loaded->layer->add(
          dylib, MemoryBuffer::getMemBufferCopy(object, "flydsl-exported-object")))
    return std::move(err);
  return loaded;
}

void reportError(MlirStringCallback onError, void *userData, Error err) {
  std::string message = toString(std::move(err));
  onError(mlirStringRefCreate(message.data(), message.size()), userData);
}

} // namespace

extern "C" {

__attribute__((visibility("default"))) void *
flydslLoadHostObject(MlirStringRef object, const MlirStringRef *libs, intptr_t numLibs,
                     MlirStringCallback onError, void *userData) {
  std::vector<std::string> libPaths;
  for (intptr_t i = 0; i < numLibs; ++i)
    libPaths.emplace_back(libs[i].data, libs[i].length);
  auto loaded = loadObject(StringRef(object.data, object.length), libPaths);
  if (!loaded) {
    reportError(onError, userData, loaded.takeError());
    return nullptr;
  }
  return loaded->release();
}

__attribute__((visibility("default"))) uint64_t flydslHostObjectLookup(void *handle,
                                                                       MlirStringRef name,
                                                                       MlirStringCallback onError,
                                                                       void *userData) {
  auto *loaded = static_cast<LoadedObject *>(handle);
  auto symbol = loaded->session->lookup({loaded->dylib}, StringRef(name.data, name.length));
  if (!symbol) {
    reportError(onError, userData, symbol.takeError());
    return 0;
  }
  return symbol->getAddress().getValue();
}

__attribute__((visibility("default"))) void flydslDestroyHostObject(void *handle) {
  delete static_cast<LoadedObject *>(handle);
}

} // extern "C"
