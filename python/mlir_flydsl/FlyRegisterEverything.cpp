// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "flydsl-c/FlyDialect.h"
#include "flydsl-c/FlyROCDLDialect.h"

NB_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "MLIR All Upstream Dialects, Translations and Passes Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    // Attach FlyDSL's gpu::TargetAttrInterface external model on
    // #rocdl.target *before* mlirRegisterAllDialects.  MLIR's InterfaceMap
    // ignores repeated attachInterface for the same interface id (see
    // detail::InterfaceMap::insert in mlir/lib/Support/InterfaceSupport.cpp),
    // so the FIRST registration wins.  If we registered after upstream,
    // upstream's ROCDLTargetAttrImpl would silently win and our serializer
    // (which injects our two AMDGPU MachineFunctionPasses at PreRegAlloc)
    // would never run.  See docs/codegen_pass_plugin.md.
    mlirRegisterFlyAMDGPUTargetInterfaceExternalModels(registry);

    mlirRegisterAllDialects(registry);

    MlirDialectHandle flyHandle = mlirGetDialectHandle__fly__();
    mlirDialectHandleInsertDialect(flyHandle, registry);
    MlirDialectHandle flyROCDLHandle = mlirGetDialectHandle__fly_rocdl__();
    mlirDialectHandleInsertDialect(flyROCDLHandle, registry);
  });
  m.def("register_llvm_translations",
        [](MlirContext context) { mlirRegisterAllLLVMTranslations(context); });

  mlirRegisterAllPasses();
  mlirRegisterFlyPasses();
  mlirRegisterFlyToROCDLConversionPass();
  mlirRegisterFlyROCDLClusterAttrPass();
  mlirRegisterFlyAMDGPUCodegenPasses();
}
