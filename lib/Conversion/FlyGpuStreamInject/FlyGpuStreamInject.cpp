#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Conversion/FlyGpuStreamInject/FlyGpuStreamInject.h"

namespace mlir {
#define GEN_PASS_DEF_FLYGPUSTREAMINJECTPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Try to find an existing block argument of the given type that has zero
/// uses — this is the original stream parameter whose asyncObject link was
/// dropped by gpu-to-llvm.  Search from the back because the stream arg is
/// conventionally the last parameter.
static Value findOrphanedArg(Block &entry, Type targetTy) {
  for (BlockArgument arg : llvm::reverse(entry.getArguments())) {
    if (arg.getType() == targetTy && arg.use_empty())
      return arg;
  }
  return nullptr;
}

static void injectStreamIntoFunction(Operation *funcLikeOp, MLIRContext *ctx) {
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);

  SmallVector<gpu::LaunchFuncOp> launches;
  funcLikeOp->walk([&](gpu::LaunchFuncOp op) { launches.push_back(op); });
  if (launches.empty())
    return;

  Value streamArg;
  if (auto funcOp = dyn_cast<func::FuncOp>(funcLikeOp)) {
    Block &entry = funcOp.getBody().front();
    streamArg = findOrphanedArg(entry, ptrTy);
    if (!streamArg) {
      unsigned idx = funcOp.getNumArguments();
      (void)funcOp.insertArgument(idx, ptrTy, {}, funcOp.getLoc());
      streamArg = funcOp.getArgument(idx);
    }
  } else if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(funcLikeOp)) {
    Block &entry = llvmFunc.getBody().front();
    streamArg = findOrphanedArg(entry, ptrTy);
    if (!streamArg) {
      auto oldTy = llvmFunc.getFunctionType();
      SmallVector<Type> newInputs(oldTy.getParams());
      newInputs.push_back(ptrTy);
      auto newTy = LLVM::LLVMFunctionType::get(
          oldTy.getReturnType(), newInputs, oldTy.isVarArg());
      llvmFunc.setFunctionType(newTy);
      streamArg = entry.addArgument(ptrTy, llvmFunc.getLoc());
    }
  } else {
    return;
  }

  // Replace every gpu.launch_func with a copy carrying asyncObject = stream.
  OpBuilder builder(ctx);
  for (gpu::LaunchFuncOp launch : launches) {
    builder.setInsertionPoint(launch);

    gpu::KernelDim3 grid{launch.getGridSizeX(), launch.getGridSizeY(),
                         launch.getGridSizeZ()};
    gpu::KernelDim3 block{launch.getBlockSizeX(), launch.getBlockSizeY(),
                          launch.getBlockSizeZ()};

    std::optional<gpu::KernelDim3> cluster;
    if (launch.hasClusterSize())
      cluster = gpu::KernelDim3{launch.getClusterSizeX(),
                                launch.getClusterSizeY(),
                                launch.getClusterSizeZ()};

    gpu::LaunchFuncOp::create(
        builder, launch.getLoc(), launch.getKernelAttr(), grid, block,
        launch.getDynamicSharedMemorySize(), launch.getKernelOperands(),
        /*asyncObject=*/streamArg, cluster);

    launch.erase();
  }
}

class FlyGpuStreamInjectPass
    : public mlir::impl::FlyGpuStreamInjectPassBase<FlyGpuStreamInjectPass> {
public:
  using FlyGpuStreamInjectPassBase::FlyGpuStreamInjectPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Handle both func.func and llvm.func since gpu-to-llvm may have already
    // converted the function op.
    SmallVector<Operation *> targets;
    moduleOp.walk([&](func::FuncOp op) { targets.push_back(op); });
    moduleOp.walk([&](LLVM::LLVMFuncOp op) {
      if (!op.isExternal())
        targets.push_back(op);
    });

    for (Operation *op : targets)
      injectStreamIntoFunction(op, ctx);
  }
};

} // namespace
