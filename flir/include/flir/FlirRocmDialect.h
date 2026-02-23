/*
 * SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FLIR_ROCM_DIALECT_H
#define FLIR_ROCM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "flir/FlirRocmDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flir/FlirRocmTypes.h.inc"

#endif // FLIR_ROCM_DIALECT_H

