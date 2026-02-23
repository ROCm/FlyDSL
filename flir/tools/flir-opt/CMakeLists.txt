# SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set(LIBS
  FlirDialect
  FlirTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRControlFlowDialect
  MLIRMlirOptMain
)

add_llvm_tool(flir-opt
  flir-opt.cpp
)

target_link_libraries(flir-opt PRIVATE ${LIBS})
llvm_map_components_to_libnames(llvm_libs Support)
target_link_libraries(flir-opt PRIVATE ${llvm_libs})
