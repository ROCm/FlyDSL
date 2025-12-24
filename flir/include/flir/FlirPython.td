//===- FlirPython.td - Python Bindings Definition ---------*- tablegen -*-===//
//
// Combined definition file for generating Python bindings
// Includes both core Flir ops and FlirRocm ops
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_PYTHON
#define ROCIR_PYTHON

include "flir/FlirOps.td"
include "flir/FlirRocmOps.td"

#endif // ROCIR_PYTHON
