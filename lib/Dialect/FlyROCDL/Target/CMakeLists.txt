# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyDSL's gpu::TargetAttrInterface external model on #rocdl.target.
# Depends on the FlyDSL AMDGPU codegen passes and on the upstream
# MLIR ROCDL target serialization base classes.
add_mlir_dialect_library(MLIRFlyROCDLTarget
  FlyAMDGPUTarget.cpp

  LINK_LIBS PUBLIC
  FlyDSLAMDGPUCodegen
  MLIRGPUDialect
  MLIRIR
  MLIRROCDLDialect
  MLIRROCDLTarget
  MLIRTargetLLVM
  MLIRSupport
)
