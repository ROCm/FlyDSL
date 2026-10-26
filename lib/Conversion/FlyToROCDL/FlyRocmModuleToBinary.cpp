// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_FLYROCMMODULETOBINARYPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

FailureOr<SmallVector<char, 0>> linkObject(ArrayRef<char> object, StringRef linker,
                                           function_ref<InFlightDiagnostic()> reportError) {
  int fd;
  SmallString<128> objectPath, binaryPath, errorPath;
  if (auto error = llvm::sys::fs::createTemporaryFile("flydsl-rocm", "o", fd, objectPath))
    return reportError() << "cannot create GPU object file: " << error.message();
  llvm::FileRemover removeObject(objectPath);
  {
    llvm::raw_fd_ostream stream(fd, /*shouldClose=*/true);
    stream.write(object.data(), object.size());
    stream.close();
    if (stream.has_error()) {
      stream.clear_error();
      return reportError() << "cannot write GPU object file: " << objectPath;
    }
  }
  if (auto error = llvm::sys::fs::createTemporaryFile("flydsl-rocm", "hsaco", binaryPath))
    return reportError() << "cannot create GPU binary file: " << error.message();
  llvm::FileRemover removeBinary(binaryPath);
  if (auto error = llvm::sys::fs::createTemporaryFile("flydsl-rocm", "log", errorPath))
    return reportError() << "cannot create linker diagnostic file: " << error.message();
  llvm::FileRemover removeError(errorPath);

  // ROCm SDK launchers locate their resources through argv[0]. Keep the real
  // executable path here, including the ld.lld basename that selects ELF mode.
  std::string executionError;
  int status = llvm::sys::ExecuteAndWait(
      linker, {linker, "-shared", objectPath, "-o", binaryPath}, std::nullopt,
      {std::nullopt, std::nullopt, StringRef(errorPath)}, /*SecondsToWait=*/60,
      /*MemoryLimit=*/0, &executionError);
  if (status != 0) {
    auto diagnostic = reportError();
    diagnostic << "GPU linker '" << linker << "' failed (exit " << status << ")";
    if (!executionError.empty())
      diagnostic << ": " << executionError;
    if (auto errors = llvm::MemoryBuffer::getFile(errorPath))
      diagnostic << "\n" << (*errors)->getBuffer();
    return failure();
  }
  auto binary = llvm::MemoryBuffer::getFile(binaryPath, /*IsText=*/false);
  if (!binary)
    return reportError() << "cannot read GPU binary: " << binary.getError().message();
  StringRef bytes = (*binary)->getBuffer();
  return SmallVector<char, 0>(bytes.begin(), bytes.end());
}

class FlyRocmSerializer : public ROCDL::SerializeGPUModuleBase {
public:
  FlyRocmSerializer(Operation &op, ROCDL::ROCDLTargetAttr target, const gpu::TargetOptions &options,
                    StringRef linker)
      : SerializeGPUModuleBase(op, target, options), options(options), linker(linker) {}

protected:
  FailureOr<SmallVector<char, 0>> moduleToObject(llvm::Module &module) override {
    return moduleToObjectImpl(options, module);
  }

  FailureOr<SmallVector<char, 0>> compileToBinary(StringRef isa) override {
    auto reportError = [&]() { return getOperation().emitError(); };
    auto object = ROCDL::assembleIsa(isa, target.getTriple(), target.getChip(),
                                     target.getFeatures(), reportError);
    if (failed(object))
      return failure();
    return linkObject(*object, linker, reportError);
  }

private:
  const gpu::TargetOptions &options;
  StringRef linker;
};

class FlyRocmModuleToBinaryPass
    : public impl::FlyRocmModuleToBinaryPassBase<FlyRocmModuleToBinaryPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto format = llvm::StringSwitch<std::optional<gpu::CompilationTarget>>(compilationTarget)
                      .Cases({"offloading", "llvm"}, gpu::CompilationTarget::Offload)
                      .Cases({"assembly", "isa"}, gpu::CompilationTarget::Assembly)
                      .Cases({"binary", "bin"}, gpu::CompilationTarget::Binary)
                      .Cases({"fatbinary", "fatbin"}, gpu::CompilationTarget::Fatbin)
                      .Default(std::nullopt);
    if (!format) {
      getOperation()->emitError("invalid GPU binary format: ") << compilationTarget;
      return signalPassFailure();
    }
    if (!objectFiles.empty() && *format < gpu::CompilationTarget::Binary) {
      getOperation()->emitError("object-files requires format=bin or format=fatbin");
      return signalPassFailure();
    }

    std::string toolkit = toolkitPath.empty() ? ROCDL::getROCMPath().str() : toolkitPath;
    SmallString<256> linker(linkerPath.getValue());
    if (linker.empty()) {
      linker = toolkit;
      llvm::sys::path::append(linker, "llvm", "bin", "ld.lld");
      if (!llvm::sys::fs::exists(linker)) {
        linker = toolkit;
        llvm::sys::path::append(linker, "bin", "ld.lld");
      }
    }
    if (auto error = llvm::sys::fs::make_absolute(linker)) {
      getOperation()->emitError("cannot resolve GPU linker: ") << error.message();
      return signalPassFailure();
    }

    std::optional<SymbolTable> parentTable;
    auto getSymbols = [&]() -> SymbolTable * {
      if (!parentTable) {
        Operation *parent = SymbolTable::getNearestSymbolTable(getOperation());
        if (!parent)
          return nullptr;
        parentTable.emplace(parent);
      }
      return &*parentTable;
    };
    SmallVector<Attribute> libraries;
    for (const auto &path : linkFiles)
      libraries.push_back(StringAttr::get(&getContext(), path));
    gpu::TargetOptions options(toolkit, libraries, cmdOptions, elfSection, *format, getSymbols);

    unsigned objectIndex = 0;
    for (Region &region : getOperation()->getRegions()) {
      for (Block &block : region) {
        for (auto module : llvm::make_early_inc_range(block.getOps<gpu::GPUModuleOp>())) {
          if (failed(serializeModule(module, options, linker, objectIndex)))
            return signalPassFailure();
        }
      }
    }
    if (objectIndex != objectFiles.size()) {
      getOperation()->emitError("object-files count does not match GPU target count");
      signalPassFailure();
    }
  }

private:
  LogicalResult serializeModule(gpu::GPUModuleOp module, const gpu::TargetOptions &options,
                                StringRef linker, unsigned &objectIndex) {
    if (!module.getTargetsAttr() || module.getTargetsAttr().empty())
      return module.emitError("the module has no target attributes");
    SmallVector<Attribute> objects;
    for (Attribute attr : module.getTargetsAttr()) {
      auto targetInterface = dyn_cast<gpu::TargetAttrInterface>(attr);
      if (!targetInterface)
        return module.emitError("GPU target does not implement TargetAttrInterface");
      std::optional<gpu::SerializedObject> serialized;
      if (auto target = dyn_cast<ROCDL::ROCDLTargetAttr>(attr)) {
        if (!objectFiles.empty()) {
          if (objectIndex >= objectFiles.size())
            return module.emitError("object-files count does not match GPU target count");
          auto input = llvm::MemoryBuffer::getFile(objectFiles[objectIndex++], /*IsText=*/false);
          if (!input)
            return module.emitError("cannot read external GPU object: ")
                   << input.getError().message();
          StringRef bytes = (*input)->getBuffer();
          auto binary = linkObject(ArrayRef<char>(bytes.data(), bytes.size()), linker,
                                   [&]() { return module.emitError(); });
          if (failed(binary))
            return failure();
          serialized.emplace(std::move(*binary));
        } else {
          FlyRocmSerializer serializer(*module, target, options, linker);
          serializer.init();
          auto binary = serializer.run();
          if (!binary)
            return failure();
          serialized.emplace(std::move(*binary));
        }
      } else {
        serialized = targetInterface.serializeToObject(module, options);
        if (!serialized)
          return failure();
      }
      // Retain upstream format selection and ELF kernel metadata, using the
      // original module while its kernel attributes are still available.
      Attribute object = targetInterface.createObject(module, *serialized, options);
      if (!object)
        return module.emitError("failed to create GPU object");
      objects.push_back(object);
    }
    auto handler = dyn_cast_or_null<gpu::OffloadingLLVMTranslationAttrInterface>(
        module.getOffloadingHandlerAttr());
    OpBuilder builder(module);
    builder.setInsertionPointAfter(module);
    gpu::BinaryOp::create(builder, module.getLoc(), module.getName(), handler,
                          builder.getArrayAttr(objects));
    module.erase();
    return success();
  }
};

} // namespace
