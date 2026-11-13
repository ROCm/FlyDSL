// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// GPUTargetInfoAttrInterface for #rocdl.target. Resource limits come from the LLVM AMDGPU
// subtarget of the target's chip, so they are the limits the backend enforces.

#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Triple.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

using namespace mlir;

namespace {

/// The size AMDGPU::IsaInfo::getAddressableLocalMemorySize reads from the subtarget, which
/// the AMDGPU backend checks each kernel's LDS size against. Returns std::nullopt unless
/// LLVM knows the chip and exactly one known size feature is set; the backend's 32 KiB
/// fallback is not a fact about the chip.
std::optional<int64_t> addressableLocalMemorySize(ROCDL::ROCDLTargetAttr target) {
  ROCDL::SerializeGPUModuleBase::init();
  llvm::Triple triple(target.getTriple());
  std::string error;
  const llvm::Target *llvmTarget = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!llvmTarget)
    return std::nullopt;

  // Validate the chip first: creating a subtarget for an unknown CPU warns on stderr.
  std::unique_ptr<llvm::MCSubtargetInfo> generic(llvmTarget->createMCSubtargetInfo(triple, "", ""));
  if (!generic || !generic->isCPUStringValid(target.getChip()))
    return std::nullopt;

  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      llvmTarget->createMCSubtargetInfo(triple, target.getChip(), target.getFeatures()));
  std::optional<int64_t> size;
  for (int64_t bytes : {32768, 65536, 163840, 327680}) {
    if (!sti->checkFeatures(("+addressablelocalmemorysize" + llvm::Twine(bytes)).str()))
      continue;
    if (size)
      return std::nullopt;
    size = bytes;
  }
  return size;
}

struct ROCDLTargetInfo
    : public fly::GPUTargetInfoAttrInterface::ExternalModel<ROCDLTargetInfo,
                                                            ROCDL::ROCDLTargetAttr> {
  std::optional<int64_t> getResourceLimit(Attribute attr, Attribute resource) const {
    auto generic = dyn_cast<fly::KernelResourceAttr>(resource);
    if (generic && generic.getValue() == fly::KernelResource::SharedMemory)
      return addressableLocalMemorySize(cast<ROCDL::ROCDLTargetAttr>(attr));
    return std::nullopt;
  }
};

} // namespace

void mlir::fly_rocdl::registerTargetInfoExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *) {
    ROCDL::ROCDLTargetAttr::attachInterface<ROCDLTargetInfo>(*ctx);
  });
}
