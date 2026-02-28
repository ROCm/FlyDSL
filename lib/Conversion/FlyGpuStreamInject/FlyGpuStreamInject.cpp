#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Conversion/FlyGpuStreamInject/FlyGpuStreamInject.h"

namespace mlir {
#define GEN_PASS_DEF_FLYGPUSTREAMMARKPASS
#define GEN_PASS_DEF_FLYGPUSTREAMINJECTPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static constexpr StringLiteral kStreamArgIndexAttr("fly.stream_arg_index");

namespace {

// ===----------------------------------------------------------------------===//
// Phase 1: fly-gpu-stream-mark  (runs BEFORE gpu-to-llvm)
//
// While gpu.launch_func still carries asyncObject and the function is still
// func.func with !gpu.async.token, record which arg index is the stream.
// ===----------------------------------------------------------------------===//

class FlyGpuStreamMarkPass
    : public mlir::impl::FlyGpuStreamMarkPassBase<FlyGpuStreamMarkPass> {
public:
  using FlyGpuStreamMarkPassBase::FlyGpuStreamMarkPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    moduleOp.walk([&](func::FuncOp funcOp) {
      int64_t streamArgIdx = -1;
      bool consistent = true;

      funcOp.walk([&](gpu::LaunchFuncOp launch) {
        Value asyncObj = launch.getAsyncObject();
        if (!asyncObj)
          return;
        auto blockArg = dyn_cast<BlockArgument>(asyncObj);
        if (!blockArg) {
          consistent = false;
          return;
        }
        int64_t idx = blockArg.getArgNumber();
        if (streamArgIdx < 0)
          streamArgIdx = idx;
        else if (streamArgIdx != idx)
          consistent = false;
      });

      if (consistent && streamArgIdx >= 0) {
        funcOp->setAttr(kStreamArgIndexAttr,
                        IntegerAttr::get(IndexType::get(ctx), streamArgIdx));
      }
    });
  }
};

// ===----------------------------------------------------------------------===//
// Phase 2: fly-gpu-stream-inject  (runs AFTER gpu-to-llvm)
//
// Re-wire the stream argument as asyncObject on each gpu.launch_func.
// ===----------------------------------------------------------------------===//

static Value addNewStreamArg(Operation *funcLikeOp, MLIRContext *ctx) {
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);

  if (auto funcOp = dyn_cast<func::FuncOp>(funcLikeOp)) {
    unsigned idx = funcOp.getNumArguments();
    (void)funcOp.insertArgument(idx, ptrTy, {}, funcOp.getLoc());
    return funcOp.getArgument(idx);
  }
  if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(funcLikeOp)) {
    auto oldTy = llvmFunc.getFunctionType();
    SmallVector<Type> newInputs(oldTy.getParams());
    newInputs.push_back(ptrTy);
    auto newTy = LLVM::LLVMFunctionType::get(oldTy.getReturnType(), newInputs,
                                              oldTy.isVarArg());
    llvmFunc.setFunctionType(newTy);
    Block &entry = llvmFunc.getBody().front();
    return entry.addArgument(ptrTy, llvmFunc.getLoc());
  }
  return nullptr;
}

static Block *getEntryBlock(Operation *funcLikeOp) {
  if (auto f = dyn_cast<func::FuncOp>(funcLikeOp))
    return &f.getBody().front();
  if (auto f = dyn_cast<LLVM::LLVMFuncOp>(funcLikeOp))
    return &f.getBody().front();
  return nullptr;
}

static void injectStreamIntoFunction(Operation *funcLikeOp, MLIRContext *ctx) {
  SmallVector<gpu::LaunchFuncOp> launches;
  funcLikeOp->walk([&](gpu::LaunchFuncOp op) {
    if (!op.getAsyncObject())
      launches.push_back(op);
  });
  if (launches.empty())
    return;

  Value streamArg;

  // Try the attribute left by fly-gpu-stream-mark.
  if (auto attr =
          funcLikeOp->getAttrOfType<IntegerAttr>(kStreamArgIndexAttr)) {
    unsigned argIdx = attr.getValue().getZExtValue();
    Block *entry = getEntryBlock(funcLikeOp);
    if (entry && argIdx < entry->getNumArguments())
      streamArg = entry->getArgument(argIdx);
    funcLikeOp->removeAttr(kStreamArgIndexAttr);
  }

  // Fallback: no attribute → add a new trailing stream arg.
  if (!streamArg)
    streamArg = addNewStreamArg(funcLikeOp, ctx);

  if (!streamArg)
    return;

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
