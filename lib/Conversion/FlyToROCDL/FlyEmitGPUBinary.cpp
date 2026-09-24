// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#ifdef FLYDSL_HAS_LLD_LIBRARY
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf)
#endif // FLYDSL_HAS_LLD_LIBRARY

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

namespace mlir {
#define GEN_PASS_DEF_FLYEMITGPUBINARYPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

#ifdef FLYDSL_HAS_LLD_LIBRARY

// Run the LLD ELF driver linked into this library.  Returns the linker
// diagnostics on failure, std::nullopt on success.
//
// `--threads=1` is not a performance choice: LLD links through LLVM's global
// thread pool, and in a forked child process (autotune workers) the inherited
// pool has no live workers, so `~TaskGroup()` blocks forever.  A single kernel
// object links in microseconds either way.
//
// `canRunAgain` reports whether LLD's global state survived the call.  A JIT
// links thousands of times per process, so a false here must abort rather than
// let the next link run on corrupted state.
std::optional<std::string> runLLD(StringRef objectPath, StringRef hsacoPath) {
  std::string objectPathStr = objectPath.str();
  std::string hsacoPathStr = hsacoPath.str();
  std::array<const char *, 6> args{"ld.lld", "--threads=1",       "-shared", objectPathStr.c_str(),
                                   "-o",     hsacoPathStr.c_str()};

  std::string errString;
  llvm::raw_string_ostream errStream(errString);
  lld::Result result = lld::lldMain(args, llvm::outs(), errStream, {{lld::Gnu, &lld::elf::link}});
  if (result.retCode != 0 || !result.canRunAgain) {
    errStream.flush();
    return errString.empty() ? std::string("unknown lld failure") : errString;
  }
  return std::nullopt;
}

// Link a relocatable AMDGPU ELF into an HSA code object.  The LLD driver API
// only accepts file paths, so the object round-trips through temporary files
// exactly as the upstream implementation does; what is saved is the fork/exec
// and the toolkit path lookup.
FailureOr<SmallVector<char, 0>> linkObjectCode(ArrayRef<char> objectCode,
                                               function_ref<InFlightDiagnostic()> emitError) {
  int objectFd = -1;
  SmallString<128> objectPath;
  if (llvm::sys::fs::createTemporaryFile("flydsl-kernel%%", "o", objectFd, objectPath))
    return emitError() << "failed to create a temporary file for the ISA binary";
  llvm::FileRemover objectRemover(objectPath);
  {
    llvm::raw_fd_ostream objectOs(objectFd, /*shouldClose=*/true);
    objectOs << StringRef(objectCode.data(), objectCode.size());
    objectOs.flush();
  }

  SmallString<128> hsacoPath;
  if (llvm::sys::fs::createTemporaryFile("flydsl-kernel%%", "hsaco", hsacoPath))
    return emitError() << "failed to create a temporary file for the HSA code object";
  llvm::FileRemover hsacoRemover(hsacoPath);

  if (std::optional<std::string> error = runLLD(objectPath, hsacoPath))
    return emitError() << "in-process lld failed to link the HSA code object: " << *error;

  auto hsacoFile = llvm::MemoryBuffer::getFile(hsacoPath, /*IsText=*/false);
  if (!hsacoFile)
    return emitError() << "failed to read the HSA code object from " << hsacoPath;

  StringRef buffer = (*hsacoFile)->getBuffer();
  return SmallVector<char, 0>(buffer.begin(), buffer.end());
}

// Replace every ROCDL assembly object of `binary` with the linked fatbin.  The
// AMDGPU MC target is already registered here: `gpu-module-to-binary` ran
// `SerializeGPUModuleBase::init()` while producing the ISA we consume.
LogicalResult compileAssemblyObjects(gpu::BinaryOp binary) {
  ArrayRef<Attribute> objects = binary.getObjectsAttr().getValue();
  SmallVector<Attribute> compiled;
  compiled.reserve(objects.size());
  bool changed = false;

  for (Attribute attr : objects) {
    auto object = dyn_cast<gpu::ObjectAttr>(attr);
    auto target =
        object ? dyn_cast<ROCDL::ROCDLTargetAttr>(object.getTarget()) : ROCDL::ROCDLTargetAttr();
    if (!object || !target || object.getFormat() != gpu::CompilationTarget::Assembly) {
      compiled.push_back(attr);
      continue;
    }

    auto emitError = [&]() { return binary.emitError(); };
    FailureOr<SmallVector<char, 0>> objectCode =
        ROCDL::assembleIsa(object.getObject().getValue(), target.getTriple(), target.getChip(),
                           target.getFeatures(), emitError);
    if (failed(objectCode))
      return failure();

    FailureOr<SmallVector<char, 0>> hsaco = linkObjectCode(*objectCode, emitError);
    if (failed(hsaco))
      return failure();

    compiled.push_back(gpu::ObjectAttr::get(
        object.getTarget(), gpu::CompilationTarget::Fatbin,
        StringAttr::get(binary.getContext(), StringRef(hsaco->data(), hsaco->size())),
        object.getProperties(), object.getKernels()));
    changed = true;
  }

  if (changed)
    binary.setObjectsAttr(ArrayAttr::get(binary.getContext(), compiled));
  return success();
}

#endif // FLYDSL_HAS_LLD_LIBRARY

class FlyEmitGPUBinaryPass : public mlir::impl::FlyEmitGPUBinaryPassBase<FlyEmitGPUBinaryPass> {
public:
  using mlir::impl::FlyEmitGPUBinaryPassBase<FlyEmitGPUBinaryPass>::FlyEmitGPUBinaryPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    GpuModuleToBinaryPassOptions binaryOptions;
    binaryOptions.toolkitPath = toolkitPath;
    binaryOptions.linkFiles.assign(linkFiles.begin(), linkFiles.end());
    binaryOptions.cmdOptions = cmdOptions;
    binaryOptions.elfSection = elfSection;
#ifdef FLYDSL_HAS_LLD_LIBRARY
    // Stop before the link so upstream never looks for `ld.lld`.
    binaryOptions.compilationTarget = "isa";
#else
    binaryOptions.compilationTarget = "fatbin";
#endif

    OpPassManager pm(op->getName());
    pm.addPass(createGpuModuleToBinaryPass(binaryOptions));
    if (failed(runPipeline(pm, op))) {
      signalPassFailure();
      return;
    }

#ifdef FLYDSL_HAS_LLD_LIBRARY
    WalkResult walked = op->walk([&](gpu::BinaryOp binary) {
      return failed(compileAssemblyObjects(binary)) ? WalkResult::interrupt()
                                                    : WalkResult::advance();
    });
    if (walked.wasInterrupted())
      signalPassFailure();
#endif
  }
};

} // namespace
