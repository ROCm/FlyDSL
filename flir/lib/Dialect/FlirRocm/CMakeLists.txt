# SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(FlirRocmDialect
  FlirRocmDialect.cpp
  FlirRocmOps.cpp
  
  DEPENDS
  FlirRocmDialectIncGen
  FlirRocmOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
