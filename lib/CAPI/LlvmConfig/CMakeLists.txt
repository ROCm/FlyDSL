# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
llvm_map_components_to_libnames(register_layout_libs
  MC TargetParser AllTargetsInfos AllTargetsDescs)
add_mlir_public_c_api_library(MLIRCPILLVMOption
  llvm.cc
  LINK_LIBS PUBLIC
  LLVMSupport
  ${register_layout_libs}
)
