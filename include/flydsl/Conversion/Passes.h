
#ifndef FLY_CONVERSION_PASSES_H
#define FLY_CONVERSION_PASSES_H

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/Passes.h.inc"

} // namespace mlir

#endif // FLY_CONVERSION_PASSES_H
