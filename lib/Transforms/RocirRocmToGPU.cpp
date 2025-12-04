//===- RocirRocmToGPU.cpp - Lower Rocir ROCm to GPU/ROCDL ---------------===//
//
// Lower Rocir ROCm operations to GPU and ROCDL dialects
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rocir {

namespace {

/// Pass to lower Rocir ROCm IR to GPU/ROCDL
struct RocirRocmToGPUPass
    : public PassWrapper<RocirRocmToGPUPass, OperationPass<gpu::GPUModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocirRocmToGPUPass)
  
  StringRef getArgument() const final { return "rocir-rocm-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower Rocir ROCm IR to MLIR GPU/ROCDL dialect";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, vector::VectorDialect>();
  }
  
  void runOnOperation() override {
    auto gpuModule = getOperation();
    
    // Mark as targeting AMD GFX942
    gpuModule->setAttr("rocdl.target_arch", 
                       StringAttr::get(&getContext(), targetArch));
    
    // TODO: Add conversion patterns for ROCm operations
    // For now, just mark the target architecture
  }
  
  Option<std::string> targetArch{
    *this, "target-arch", 
    llvm::cl::desc("Target AMD GPU architecture"),
    llvm::cl::init("gfx942")
  };
};

} // namespace

std::unique_ptr<Pass> createRocirRocmToGPUPass() {
  return std::make_unique<RocirRocmToGPUPass>();
}

} // namespace rocir
} // namespace mlir

