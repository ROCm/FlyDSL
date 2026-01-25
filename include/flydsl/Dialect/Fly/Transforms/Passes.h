#ifndef FLY_TRANSFORM_H
#define FLY_TRANSFORM_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace fly {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"

} // namespace fly
} // namespace mlir

#endif // FLY_TRANSFORM_H
