/*
 * SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FLIR_ROCM_OPS_H
#define FLIR_ROCM_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "flir/FlirRocmDialect.h"

#define GET_OP_CLASSES
#include "flir/FlirRocmOps.h.inc"

#endif // FLIR_ROCM_OPS_H

