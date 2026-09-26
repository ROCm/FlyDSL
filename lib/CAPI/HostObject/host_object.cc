// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Host object emission for exported (AOT) launchers.  Linked into
// libFlyPythonCAPI.so so translation sees the dialect and offloading
// translation interfaces registered on the Python-side contexts.

#include "HostObject/host_object.h"

#include "mlir/CAPI/IR.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <string>

namespace {

llvm::Triple hostTriple() { return llvm::Triple(llvm::sys::getProcessTriple()); }

void reportString(MlirStringCallback callback, void *userData, llvm::StringRef str) {
  callback(mlirStringRefCreate(str.data(), str.size()), userData);
}

// Builds a PIC target machine for the generic CPU of the host architecture so
// the object runs on any machine of that architecture, not only this one.
llvm::Expected<std::unique_ptr<llvm::TargetMachine>> createHostTargetMachine(int optLevel) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::Triple triple = hostTriple();
  std::string error;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return llvm::createStringError(error);

  auto codeGenLevel = llvm::CodeGenOpt::getLevel(optLevel);
  if (!codeGenLevel)
    return llvm::createStringError("invalid optimization level %d", optLevel);
  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(triple, "generic", "", llvm::TargetOptions(), llvm::Reloc::PIC_,
                                  std::nullopt, *codeGenLevel));
  if (!machine)
    return llvm::createStringError("cannot create a target machine for " + triple.str());
  return machine;
}

llvm::Expected<llvm::SmallVector<char, 0>> emitHostObject(mlir::Operation *module, int optLevel) {
  auto machine = createHostTargetMachine(optLevel);
  if (!machine)
    return machine.takeError();

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return llvm::createStringError("failed to translate the module to LLVM IR");
  llvmModule->setTargetTriple((*machine)->getTargetTriple());
  llvmModule->setDataLayout((*machine)->createDataLayout());
  llvmModule->setPICLevel(llvm::PICLevel::BigPIC);

  if (llvm::Error err = mlir::makeOptimizingTransformer(optLevel, /*sizeLevel=*/0,
                                                        machine->get())(llvmModule.get()))
    return std::move(err);

  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  llvm::legacy::PassManager passes;
  if ((*machine)->addPassesToEmitFile(passes, stream, nullptr, llvm::CodeGenFileType::ObjectFile))
    return llvm::createStringError("the host target cannot emit object files");
  passes.run(*llvmModule);
  return buffer;
}

} // namespace

extern "C" {

__attribute__((visibility("default"))) void flydslHostTargetTriple(MlirStringCallback callback,
                                                                   void *userData) {
  reportString(callback, userData, hostTriple().str());
}

__attribute__((visibility("default"))) MlirLogicalResult
flydslEmitHostObject(MlirOperation module, int optLevel, MlirStringCallback onObject,
                     MlirStringCallback onError, void *userData) {
  llvm::Expected<llvm::SmallVector<char, 0>> object = emitHostObject(unwrap(module), optLevel);
  if (!object) {
    reportString(onError, userData, llvm::toString(object.takeError()));
    return mlirLogicalResultFailure();
  }
  reportString(onObject, userData, llvm::StringRef(object->data(), object->size()));
  return mlirLogicalResultSuccess();
}

} // extern "C"
