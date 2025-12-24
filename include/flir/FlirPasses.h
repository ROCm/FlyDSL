#ifndef FLIR_PASSES_H
#define FLIR_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace flir {

// Generate pass declarations (including option structs + base classes decls).
#define GEN_PASS_DECL
#include "flir/FlirPasses.h.inc"

// Convenience constructors used throughout the codebase.
std::unique_ptr<Pass> createFlirToStandardPass();
std::unique_ptr<Pass> createFlirTrivialDCEPass();

} // namespace flir
} // namespace mlir

#endif // FLIR_PASSES_H
