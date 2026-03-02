#ifndef CONVERSION_FLYGPUTOLLVM_FLYGPUTOLLVM_H
#define CONVERSION_FLYGPUTOLLVM_FLYGPUTOLLVM_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYGPUTOLLVMPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYGPUTOLLVM_FLYGPUTOLLVM_H
