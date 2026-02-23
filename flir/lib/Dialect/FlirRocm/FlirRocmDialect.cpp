// SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "flir/FlirRocmDialect.h"
#include "flir/FlirRocmDialect.cpp.inc"

void mlir::flir::rocm::FlirRocmDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "flir/FlirRocmOps.cpp.inc"
  >();
}
