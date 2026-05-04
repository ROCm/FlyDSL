# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyDSL-owned AMDGPU MachineFunctionPasses.  Linked into the same
# shared object as the rest of FlyDSL so we share the AMDGPU codegen
# PassRegistry / cl::opt registry / OpName tables.
add_llvm_library(FlyDSLAMDGPUCodegen
  PreferAGPRForDSRead.cpp
  MFMATieVDSTToSrc2.cpp
  PassRegistration.cpp

  LINK_COMPONENTS
  AMDGPUCodeGen
  CodeGen
  Core
  Support
)
