#ifndef ROCIR_PASSES_H
#define ROCIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rocir {

// Lowering passes
std::unique_ptr<Pass> createRocirToStandardPass();
std::unique_ptr<Pass> createRocirCoordLoweringPass();
std::unique_ptr<Pass> createRocirLayoutCanonicalizePass();
std::unique_ptr<Pass> createRocirToRocmPass();
std::unique_ptr<Pass> createRocirRocmToGPUPass();

} // namespace rocir
} // namespace mlir

#endif // ROCIR_PASSES_H
