# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
add_mlir_public_c_api_library(MLIRCPILLVMOption
  llvm.cc
  LINK_LIBS PUBLIC
  LLVMSupport
)
