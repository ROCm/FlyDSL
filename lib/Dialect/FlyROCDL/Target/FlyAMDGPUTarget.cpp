//===-- FlyAMDGPUTarget.cpp -------------------------------------------------===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// FlyDSL-owned `gpu::TargetAttrInterface` impl on `#rocdl.target`.
//
// Subclasses `mlir::ROCDL::SerializeGPUModuleBase` to reuse the upstream
// LLVM-IR-side bitcode loading / device-libs linking logic, but overrides
// `moduleToObject` so we own the LLVM-IR -> object code path and can call
// `TargetPassConfig::insertPass(...)` to splice our two MachineFunctionPasses
// into the codegen pipeline at PreRegAlloc — without patching LLVM source.
//
//===----------------------------------------------------------------------===//

#include "flydsl/Dialect/FlyROCDL/Target/FlyAMDGPUTarget.h"

#include "flydsl/Codegen/AMDGPU/MFMATieVDSTToSrc2.h"
#include "flydsl/Codegen/AMDGPU/PassRegistration.h"
#include "flydsl/Codegen/AMDGPU/PreferAGPRForDSRead.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"

#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;
using namespace mlir::ROCDL;

namespace llvm {
// Anchor pass we want to insert right after.  Declared here as a forward
// extern so we don't have to drag llvm/lib/Target/AMDGPU/AMDGPU.h (which is
// not in the LLVM install) into the FlyDSL build.  The symbol resolves at
// link time against the AMDGPUCodeGen library that's already linked in via
// MLIRROCDLTarget.
extern char &AMDGPUPrepareAGPRAllocLegacyID;
} // namespace llvm

namespace {

/// Per-module switches read from the `fly.amdgpu_codegen_passes` attribute on
/// the gpu.module being serialized.  Both default to false: when *all* are
/// false, FlySerializeAMDGPUModule does no insertPass and lets upstream
/// codegen run unmodified.
struct FlyAMDGPUCodegenPassFlags {
  bool preferAGPRForDSRead = false;
  bool mfmaTieVDSTToSrc2 = false;

  bool any() const { return preferAGPRForDSRead || mfmaTieVDSTToSrc2; }
};

/// FlyDSL's serializer subclass.  Same flow as
/// mlir::ROCDL::SerializeGPUModuleBase's default, plus our pass insertion.
class FlySerializeAMDGPUModule : public SerializeGPUModuleBase {
public:
  FlySerializeAMDGPUModule(Operation &module, ROCDLTargetAttr target,
                           const gpu::TargetOptions &targetOptions,
                           FlyAMDGPUCodegenPassFlags flags)
      : SerializeGPUModuleBase(module, target, targetOptions),
        flyTargetOptions(targetOptions), flyFlags(flags) {}

protected:
  /// Override: produce object bytes from the (already linked & optimized)
  /// LLVM module by running our own legacy PM that calls insertPass(...)
  /// before driving the codegen pipeline.  Falls back to the parent
  /// implementation if anything goes wrong before we get to codegen — that
  /// way a future LLVM API change degrades to "no FlyDSL passes" rather
  /// than "compilation broken".
  FailureOr<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

private:
  /// Builds an AMDGPU TargetMachine matching the parent class's setup, runs
  /// codegen with our two passes injected, and returns the object bytes.
  /// Returns failure() on any LLVM-side error.
  FailureOr<SmallVector<char, 0>>
  emitWithFlyPasses(llvm::Module &llvmModule, llvm::TargetMachine &TM);

  /// SerializeGPUModuleBase doesn't expose its targetOptions; we keep our
  /// own copy so moduleToObject can branch on getCompilationTarget()
  /// (Assembly vs Binary vs Offload), matching upstream's behavior.
  gpu::TargetOptions flyTargetOptions;
  FlyAMDGPUCodegenPassFlags flyFlags;
};

FailureOr<SmallVector<char, 0>>
FlySerializeAMDGPUModule::emitWithFlyPasses(llvm::Module &llvmModule,
                                            llvm::TargetMachine &TM) {
  // SmallVector<char, 0> matches what upstream returns to the
  // gpu::TargetAttrInterface caller.
  SmallString<0> isaBuf;
  {
    llvm::raw_svector_ostream isaOs(isaBuf);

    // We mirror the body of LLVMTargetMachine::addPassesToGenerateCode +
    // addPassesToEmitFile manually so we can call PC->insertPass between
    // createPassConfig and addISelPasses.  The order matters: TPC must be
    // added to the PM first, then MMI (every MachineFunctionPass requires
    // it), then our insertPass calls take effect when addMachinePasses
    // schedules the AMDGPU PreRegAlloc anchor.
    llvm::CodeGenTargetMachineImpl &LTM =
        static_cast<llvm::CodeGenTargetMachineImpl &>(TM);

    llvm::legacy::PassManager codegenPM;
    auto *PC = LTM.createPassConfig(codegenPM);
    PC->setDisableVerify(false);
    codegenPM.add(PC);
    // Own the MMI so we can hand its MCContext to addAsmPrinter below
    // (CodeGenTargetMachineImpl no longer exposes a getMCContext()).
    auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(&LTM);
    codegenPM.add(MMIWP);

    // Inject FlyDSL passes immediately after AMDGPU's PrepareAGPRAlloc pass,
    // the last pass scheduled in GCNPassConfig::addPreRegAlloc().  insertPass
    // records the pair into the per-TPC InsertedPasses list; the splice fires
    // when addMachinePasses() reaches the anchor's addPass() call.
    //
    // Each pass is opt-in via `fly.amdgpu_codegen_passes` on the gpu.module;
    // emitWithFlyPasses isn't reached at all when no flag is set (see
    // moduleToObject below), so reaching this point implies at least one
    // flag is true.
    if (flyFlags.preferAGPRForDSRead)
      PC->insertPass(&llvm::AMDGPUPrepareAGPRAllocLegacyID,
                     &llvm::FlyAMDGPUPreferAGPRForDSReadLegacyID);
    if (flyFlags.mfmaTieVDSTToSrc2)
      PC->insertPass(&llvm::AMDGPUPrepareAGPRAllocLegacyID,
                     &llvm::FlyAMDGPUMFMATieVDSTToSrc2LegacyID);

    if (PC->addISelPasses())
      return failure();
    PC->addMachinePasses();
    PC->setInitialized();

    if (LTM.addAsmPrinter(codegenPM, isaOs, /*DwoOut=*/nullptr,
                          llvm::CodeGenFileType::AssemblyFile,
                          MMIWP->getMMI().getContext())) {
      return failure();
    }

    codegenPM.run(llvmModule);
  }

  // Notify the user-supplied isaCallback (parent class wires this up via
  // ModuleToObject's fields).
  if (isaCallback)
    isaCallback(isaBuf.str());

  // Honor gpu::TargetOptions::CompilationTarget so format=isa (used by
  // FLYDSL_DUMP_IR's _dump_isa) returns the textual ASM, while the default
  // Binary/Fatbin path goes through compileToBinary like upstream.
  if (flyTargetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
    SmallVector<char, 0> result;
    result.reserve(isaBuf.size());
    result.append(isaBuf.begin(), isaBuf.end());
    return result;
  }

  // Hand the textual ISA over to compileToBinary which assembles + links to
  // an ELF blob, exactly like the parent class does.
  return compileToBinary(isaBuf.str());
}

FailureOr<SmallVector<char, 0>>
FlySerializeAMDGPUModule::moduleToObject(llvm::Module &llvmModule) {
  // Opt-in: with no FlyDSL pass requested for this gpu.module, run the
  // stock upstream ROCDL codegen path (moduleToObjectImpl is protected on
  // SerializeGPUModuleBase and is exactly what AMDGPUSerializer calls).
  // We never touch the codegen pipeline unless the user opts in.
  if (!flyFlags.any())
    return moduleToObjectImpl(flyTargetOptions, llvmModule);

  // Offload target = LLVM IR; no codegen to run.  Defer to upstream which
  // returns the bitcode bytes directly without building a TargetMachine.
  if (flyTargetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Offload)
    return moduleToObjectImpl(flyTargetOptions, llvmModule);

  auto TMOrErr = getOrCreateTargetMachine();
  if (failed(TMOrErr) || !*TMOrErr) {
    // Fall back to upstream codegen.  Parent does its own diagnostics.
    return moduleToObjectImpl(flyTargetOptions, llvmModule);
  }

  auto Result = emitWithFlyPasses(llvmModule, **TMOrErr);
  if (succeeded(Result))
    return Result;

  // If our path failed for any reason, defer to upstream so the user still
  // gets a build (without our optimizations).  This keeps the failure mode
  // visible (no perf gain) rather than fatal.
  return moduleToObjectImpl(flyTargetOptions, llvmModule);
}

// ---------------------------------------------------------------------------
// External model on #rocdl.target
// ---------------------------------------------------------------------------

class FlyROCDLTargetAttrImpl
    : public gpu::TargetAttrInterface::ExternalModel<FlyROCDLTargetAttrImpl,
                                                     ROCDLTargetAttr> {
public:
  std::optional<gpu::SerializedObject>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const gpu::SerializedObject &object,
                         const gpu::TargetOptions &options) const;
};

std::optional<gpu::SerializedObject>
FlyROCDLTargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  if (!module)
    return std::nullopt;
  auto target = cast<ROCDLTargetAttr>(attribute);
  // Make sure the AMDGPU backend is initialized; SerializeGPUModuleBase::init
  // is idempotent.
  SerializeGPUModuleBase::init();
  // Make sure FlyDSL passes are known to the global PassRegistry; this is
  // also idempotent and cheap.  Doing it unconditionally keeps the symbols
  // discoverable for fly-opt and tests, even when the per-module opt-in is
  // off.
  flydsl::registerFlyAMDGPUCodegenPasses();

  // Read the per-gpu.module opt-in attribute.  See
  // `fly-rocdl-tag-amdgpu-codegen-passes` in lib/Conversion/FlyToROCDL.
  FlyAMDGPUCodegenPassFlags flags;
  if (auto dict =
          module->getAttrOfType<DictionaryAttr>("fly.amdgpu_codegen_passes")) {
    flags.preferAGPRForDSRead =
        static_cast<bool>(dict.get("prefer_agpr_for_ds_read"));
    flags.mfmaTieVDSTToSrc2 =
        static_cast<bool>(dict.get("mfma_tie_vdst_to_src2"));
  }

  FlySerializeAMDGPUModule serializer(*module, target, options, flags);
  std::optional<SmallVector<char, 0>> bytes = serializer.run();
  if (!bytes)
    return std::nullopt;
  return gpu::SerializedObject(std::move(*bytes));
}

Attribute FlyROCDLTargetAttrImpl::createObject(
    Attribute attribute, Operation *module,
    const gpu::SerializedObject &object,
    const gpu::TargetOptions &options) const {
  // Same semantics as upstream's createObject: wrap the raw bytes into a
  // gpu::ObjectAttr keyed by the (now serialized) target.  We stay close to
  // upstream's choice of attributes/format here so that downstream consumers
  // (`gpu-to-llvm`, the host-side binary embedder) don't need to special-case
  // FlyDSL.
  auto target = cast<ROCDLTargetAttr>(attribute);
  gpu::CompilationTarget format = options.getCompilationTarget();
  const SmallVector<char, 0> &bytes = object.getObject();
  return gpu::ObjectAttr::get(
      attribute.getContext(), target, format,
      StringAttr::get(attribute.getContext(),
                      StringRef(bytes.data(), bytes.size())),
      /*properties=*/object.getMetadata(),
      /*kernels=*/nullptr);
}

} // namespace

namespace flydsl {

void registerFlyAMDGPUTargetInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, ROCDL::ROCDLDialect *dialect) {
        ROCDLTargetAttr::attachInterface<FlyROCDLTargetAttrImpl>(*ctx);
      });
}

void registerFlyAMDGPUTargetInterfaceExternalModels(MLIRContext &context) {
  DialectRegistry registry;
  registerFlyAMDGPUTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

} // namespace flydsl
