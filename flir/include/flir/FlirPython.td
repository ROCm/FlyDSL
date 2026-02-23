// SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

//===- FlirPython.td - Python Bindings Definition ---------*- tablegen -*-===//
//
// Combined definition file for generating Python bindings
// Includes both core Flir ops and FlirRocm ops
//
//===----------------------------------------------------------------------===//

#ifndef FLIR_PYTHON
#define FLIR_PYTHON

include "flir/FlirOps.td"
include "flir/FlirRocmOps.td"

#endif // FLIR_PYTHON
