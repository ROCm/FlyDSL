# SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(FlirDialect
  FlirDialect.cpp
  FlirOps.cpp
  FlirLayoutAlgebra.cpp
  
  DEPENDS
  FlirDialectIncGen
  FlirAttrDefsIncGen
  FlirTypeDefsIncGen
  FlirOpsIncGen
  FlirAttrsIncGen
  FlirTypesIncGen
  
  LINK_LIBS
  MLIRIR
)
