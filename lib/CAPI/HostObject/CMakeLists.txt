# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
add_mlir_public_c_api_library(MLIRCPIHostObject
  host_object.cc
  object_loader.cc
  LINK_COMPONENTS
  Core
  JITLink
  MC
  Object
  OrcJIT
  Support
  Target
  TargetParser
  nativecodegen
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRExecutionEngineUtils
  MLIRTargetLLVMIRExport
)
