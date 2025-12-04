//===- RocirToRocm.cpp - Lower Rocir IR to ROCm Dialect ------------------===//
//
// This pass converts rocdsl operations to the ROCm dialect for GFX942
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "rocir/RocirRocmDialect.h"
#include "rocir/RocirRocmOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rocir {

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

namespace {

/// Type converter for Rocir to ROCm lowering
class RocirToRocmTypeConverter : public TypeConverter {
public:
  RocirToRocmTypeConverter() {
    // Keep MLIR builtin types (index, integers, floats) as-is
    addConversion([](Type type) { return type; });
    
    // TODO: Add conversions for Rocir types to ROCm types
    // For now, most types remain unchanged as they're already in the correct dialect
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

// NOTE: product_each and make_layout_tv patterns removed
// These are now implemented at Python level using existing operations

} // namespace

//===----------------------------------------------------------------------===//
// RocirToRocm Pass
//===----------------------------------------------------------------------===//

namespace {

/// Pass to lower Rocir IR to the ROCm dialect for AMD GFX942.
struct RocirToRocmPass : public PassWrapper<RocirToRocmPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocirToRocmPass)
  
  StringRef getArgument() const final { return "rocir-to-rocm"; }
  StringRef getDescription() const final {
    return "Lower Rocir IR to ROCm dialect for AMD GFX942";
  }
  
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    
    // Mark as targeting AMD GFX942
    module->setAttr("rocir.target_arch", 
                    StringAttr::get(context, "gfx942"));
    module->setAttr("rocir.target_vendor",
                    StringAttr::get(context, "amd"));
    
    // Set up type converter
    RocirToRocmTypeConverter typeConverter;
    
    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, 
                           func::FuncDialect,
                           gpu::GPUDialect,
                           memref::MemRefDialect,
                           rocir_rocm::RocirRocmDialect>();
    
    // Most Rocir ops remain legal (they're lowered in later passes)
    target.addLegalDialect<RocirDialect>();
    
    // Populate patterns (currently empty - patterns will be added as needed)
    RewritePatternSet patterns(context);
    
    // For now, this pass just marks the module with target attributes
    // Actual lowering patterns will be added in future iterations
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createRocirToRocmPass() {
  return std::make_unique<RocirToRocmPass>();
}

} // namespace rocir
} // namespace mlir
