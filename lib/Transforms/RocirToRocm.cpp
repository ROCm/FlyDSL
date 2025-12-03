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
// RocirToRocm Pass
//===----------------------------------------------------------------------===//

namespace {

/// Pass to lower Rocir IR to the ROCm dialect for AMD GFX942.
struct RocirToRocmPass : public PassWrapper<RocirToRocmPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocirToRocmPass)
  
  StringRef getArgument() const final { return "cute-to-rocm"; }
  StringRef getDescription() const final { 
    return "Lower Rocir IR to ROCm dialect for AMD GFX942";
  }
  
  void runOnOperation() override {
    auto module = getOperation();
    
    // Mark as targeting AMD GFX942
    module->setAttr("rocir.target_arch", 
                    StringAttr::get(&getContext(), "gfx942"));
    module->setAttr("rocir.target_vendor",
                    StringAttr::get(&getContext(), "amd"));
    
    // TODO: add dialect conversion once ROCm lowering patterns are available.
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createCuteToRocmPass() {
  return std::make_unique<RocirToRocmPass>();
}

} // namespace rocir
} // namespace mlir
