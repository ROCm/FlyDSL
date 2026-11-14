// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#ifdef FLYDSL_HAS_LLD_LIBRARY
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf)

#include <mutex>
#endif // FLYDSL_HAS_LLD_LIBRARY

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

namespace mlir {
#define GEN_PASS_DEF_FLYEMITGPUBINARYPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#ifdef FLYDSL_HAS_LLD_LIBRARY
// lldMain mutates process-wide driver state; serialize concurrent JIT links
// (autotune / multi-threaded host compilers in the same process).
static std::mutex lldMutex;

// Run the LLD ELF driver linked into this library.  Returns the linker
// diagnostics on failure, std::nullopt on success.
//
// `--threads=1` is not a performance choice: LLD links through LLVM's global
// thread pool, and in a forked child process (autotune workers) the inherited
// pool has no live workers, so `~TaskGroup()` blocks forever.  A single kernel
// object links in microseconds either way.
//
// An LLD fatal error ends in exitLld(), which `_exit`s the whole process unless
// lldMain runs under an enabled CrashRecoveryContext; with one, it unwinds back
// here and reports `canRunAgain = false`.  LLD's global state is then not safe
// to reuse, so the error is returned and every later link is refused rather than
// run on corrupted state.
std::optional<std::string> mlir::fly::linkHsacoInProcess(StringRef objectPath,
                                                         StringRef hsacoPath) {
  std::lock_guard<std::mutex> lock(lldMutex);

  static std::optional<std::string> unusableReason;
  if (unusableReason)
    return "in-process lld is unusable after an earlier fatal error: " + *unusableReason;
  llvm::CrashRecoveryContext::Enable();

  std::string objectPathStr = objectPath.str();
  std::string hsacoPathStr = hsacoPath.str();
  std::array<const char *, 6> args{"ld.lld", "--threads=1",       "-shared", objectPathStr.c_str(),
                                   "-o",     hsacoPathStr.c_str()};

  std::string errString;
  llvm::raw_string_ostream errStream(errString);
  lld::Result result = lld::lldMain(args, llvm::outs(), errStream, {{lld::Gnu, &lld::elf::link}});
  errStream.flush();
  std::string diag = errString.empty() ? std::string("unknown lld failure") : errString;
  if (!result.canRunAgain) {
    unusableReason = diag;
    return diag;
  }
  if (result.retCode != 0)
    return diag;
  return std::nullopt;
}

#endif // FLYDSL_HAS_LLD_LIBRARY

namespace {

#ifdef FLYDSL_HAS_LLD_LIBRARY

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

  if (std::optional<std::string> error = fly::linkHsacoInProcess(objectPath, hsacoPath))
    return emitError() << "in-process lld failed to link the HSA code object: " << *error;

  auto hsacoFile = llvm::MemoryBuffer::getFile(hsacoPath, /*IsText=*/false);
  if (!hsacoFile)
    return emitError() << "failed to read the HSA code object from " << hsacoPath;

  StringRef buffer = (*hsacoFile)->getBuffer();
  return SmallVector<char, 0>(buffer.begin(), buffer.end());
}

// Kernel entries of each gpu.module, keyed by module name, recorded before
// `gpu-module-to-binary` erases the modules.  Upstream builds the
// `#gpu.kernel_table` only for format=bin/fatbin; at format=isa there is no
// table, so it is rebuilt here from these entries and the linked ELF.
using KernelEntries = llvm::StringMap<SmallVector<gpu::KernelMetadataAttr>>;

KernelEntries collectKernelEntries(Operation *op) {
  KernelEntries entries;
  op->walk([&](gpu::GPUModuleOp module) {
    auto &kernels = entries[module.getName()];
    for (auto funcOp : module.getBody()->getOps<LLVM::LLVMFuncOp>())
      if (funcOp->getDiscardableAttr("rocdl.kernel"))
        kernels.push_back(gpu::KernelMetadataAttr::get(funcOp));
  });
  return entries;
}

// Same table ROCDL::getKernelMetadata() attaches for format=fatbin, built from
// the recorded entries because the gpu.module no longer exists.
gpu::KernelTableAttr getKernelTable(MLIRContext *context, ArrayRef<gpu::KernelMetadataAttr> entries,
                                    ArrayRef<char> elfData) {
  Builder builder(context);
  std::optional<DenseMap<StringAttr, NamedAttrList>> mdMapOrNull =
      ROCDL::getAMDHSAKernelsELFMetadata(builder, elfData);
  SmallVector<gpu::KernelMetadataAttr> kernels;
  for (gpu::KernelMetadataAttr kernel : entries)
    kernels.push_back(kernel.appendMetadata(mdMapOrNull
                                                ? mdMapOrNull->lookup(kernel.getName()).getAttrs()
                                                : ArrayRef<NamedAttribute>()));
  return gpu::KernelTableAttr::get(context, kernels);
}

// Replace every ROCDL assembly object of `binary` with the linked fatbin.
//
// Called both after an in-process `gpu-module-to-binary{format=isa}` and after
// an external mlir-opt that produced Assembly objects: the latter never ran
// SerializeGPUModuleBase::init() in this process, so initialize AMDGPU MC here.
// `kernelEntries` is empty in the external case, which leaves no kernel table.
LogicalResult compileAssemblyObjects(gpu::BinaryOp binary, const KernelEntries &kernelEntries) {
  ROCDL::SerializeGPUModuleBase::init();

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

    gpu::KernelTableAttr kernels = object.getKernels();
    auto entries = kernelEntries.find(binary.getName());
    if (entries != kernelEntries.end())
      kernels = getKernelTable(binary.getContext(), entries->second, *hsaco);

    compiled.push_back(gpu::ObjectAttr::get(
        object.getTarget(), gpu::CompilationTarget::Fatbin,
        StringAttr::get(binary.getContext(), StringRef(hsaco->data(), hsaco->size())),
        object.getProperties(), kernels));
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

#ifdef FLYDSL_HAS_LLD_LIBRARY
    KernelEntries kernelEntries = collectKernelEntries(op);
#endif

    OpPassManager pm(op->getName());
    pm.addPass(createGpuModuleToBinaryPass(binaryOptions));
    if (failed(runPipeline(pm, op))) {
      signalPassFailure();
      return;
    }

#ifdef FLYDSL_HAS_LLD_LIBRARY
    WalkResult walked = op->walk([&](gpu::BinaryOp binary) {
      return failed(compileAssemblyObjects(binary, kernelEntries)) ? WalkResult::interrupt()
                                                                   : WalkResult::advance();
    });
    if (walked.wasInterrupted())
      signalPassFailure();
#endif
  }
};

} // namespace
