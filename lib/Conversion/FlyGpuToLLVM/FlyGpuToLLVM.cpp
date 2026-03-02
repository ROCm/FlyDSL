#include "flydsl/Conversion/FlyGpuToLLVM/FlyGpuToLLVM.h"

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FLYGPUTOLLVMPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Higher-priority pattern for gpu.launch_func that preserves asyncObject.
///
/// The upstream LegalizeLaunchFuncOpPattern ignores the asyncObject operand
/// and always creates/destroys its own stream via mgpuStreamCreate.  This
/// pattern adds a single check: if the original op carries an asyncObject,
/// use it as the stream instead.
class FlyLaunchFuncOpPattern
    : public ConvertOpToLLVMPattern<gpu::LaunchFuncOp> {
public:
  FlyLaunchFuncOpPattern(const LLVMTypeConverter &converter,
                         bool kernelBarePtrCallConv,
                         bool kernelIntersperseSizeCallConv)
      : ConvertOpToLLVMPattern<gpu::LaunchFuncOp>(converter,
                                                  /*benefit=*/2),
        kernelBarePtrCallConv(kernelBarePtrCallConv),
        kernelIntersperseSizeCallConv(kernelIntersperseSizeCallConv),
        streamCreateCallBuilder(
            "mgpuStreamCreate",
            LLVM::LLVMPointerType::get(&converter.getContext()), {}) {}

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    for (Value operand : adaptor.getOperands())
      if (!LLVM::isCompatibleType(operand.getType()))
        return rewriter.notifyMatchFailure(launchOp,
                                           "operand is not an LLVM type");

    if (launchOp.getAsyncDependencies().size() > 1)
      return rewriter.notifyMatchFailure(
          launchOp, "Cannot convert with more than one async dependency.");

    if (!launchOp.getAsyncToken() && !launchOp.getAsyncDependencies().empty())
      return rewriter.notifyMatchFailure(
          launchOp, "Cannot convert non-async op with async dependencies.");

    Location loc = launchOp.getLoc();

    // --- Stream selection (the fix over upstream) ---
    Value stream = Value();
    if (!adaptor.getAsyncDependencies().empty())
      stream = adaptor.getAsyncDependencies().front();
    else if (adaptor.getAsyncObject())
      stream = adaptor.getAsyncObject();
    else if (launchOp.getAsyncToken())
      stream = streamCreateCallBuilder.create(loc, rewriter, {}).getResult();

    // Lower kernel operands to LLVM types.
    OperandRange origArguments = launchOp.getKernelOperands();
    SmallVector<Value, 8> llvmArguments = getTypeConverter()->promoteOperands(
        loc, origArguments, adaptor.getKernelOperands(), rewriter,
        /*useBarePtrCallConv=*/kernelBarePtrCallConv);

    SmallVector<Value, 8> llvmArgumentsWithSizes;
    if (kernelIntersperseSizeCallConv) {
      if (origArguments.size() != llvmArguments.size())
        return rewriter.notifyMatchFailure(
            launchOp,
            "Cannot add sizes to arguments with one-to-many expansion.");

      llvmArgumentsWithSizes.reserve(llvmArguments.size() * 2);
      for (auto [llvmArg, origArg] :
           llvm::zip_equal(llvmArguments, origArguments)) {
        auto memrefTy = dyn_cast<MemRefType>(origArg.getType());
        if (!memrefTy)
          return rewriter.notifyMatchFailure(
              launchOp, "Operand to launch op is not a memref.");
        if (!memrefTy.hasStaticShape() ||
            !memrefTy.getElementType().isIntOrFloat())
          return rewriter.notifyMatchFailure(
              launchOp, "Operand is not a static-shape int/float memref.");
        unsigned bitwidth = memrefTy.getElementTypeBitWidth();
        if (bitwidth % 8 != 0)
          return rewriter.notifyMatchFailure(
              launchOp, "Operand element type is not byte-aligned.");
        uint64_t staticSize = static_cast<uint64_t>(bitwidth / 8) *
                              static_cast<uint64_t>(memrefTy.getNumElements());
        Value sizeArg = LLVM::ConstantOp::create(
            rewriter, loc, getIndexType(), rewriter.getIndexAttr(staticSize));
        llvmArgumentsWithSizes.push_back(llvmArg);
        llvmArgumentsWithSizes.push_back(sizeArg);
      }
    }

    std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
    if (launchOp.hasClusterSize())
      clusterSize =
          gpu::KernelDim3{adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
                          adaptor.getClusterSizeZ()};

    gpu::LaunchFuncOp::create(
        rewriter, launchOp.getLoc(), launchOp.getKernelAttr(),
        gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                        adaptor.getGridSizeZ()},
        gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
                        adaptor.getBlockSizeZ()},
        adaptor.getDynamicSharedMemorySize(),
        llvmArgumentsWithSizes.empty() ? llvmArguments : llvmArgumentsWithSizes,
        stream, clusterSize);

    if (launchOp.getAsyncToken())
      rewriter.replaceOp(launchOp, {stream});
    else
      rewriter.eraseOp(launchOp);
    return success();
  }

private:
  bool kernelBarePtrCallConv;
  bool kernelIntersperseSizeCallConv;
  FunctionCallBuilder streamCreateCallBuilder;
};

// ===----------------------------------------------------------------------===//
// fly-gpu-to-llvm pass: drop-in replacement for gpu-to-llvm
// ===----------------------------------------------------------------------===//

class FlyGpuToLLVMPass
    : public mlir::impl::FlyGpuToLLVMPassBase<FlyGpuToLLVMPass> {
public:
  using FlyGpuToLLVMPassBase::FlyGpuToLLVMPassBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    FlyGpuToLLVMPassBase::getDependentDialects(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // Phase 1: Progressive vector lowering (same as upstream).
    {
      RewritePatternSet patterns(context);
      vector::populateVectorTransferLoweringPatterns(patterns,
                                                     /*maxTransferRank=*/1);
      vector::populateVectorFromElementsUnrollPatterns(patterns);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
    }

    // Phase 2: GPU-to-LLVM conversion with our custom pattern.
    LowerToLLVMOptions options(context);
    options.useBarePtrCallConv = hostBarePtrCallConv;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    LLVMTypeConverter converter(context, options);

    for (Dialect *dialect : context->getLoadedDialects()) {
      auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;
      iface->populateConvertToLLVMConversionPatterns(target, converter,
                                                     patterns);
    }

    target.addLegalOp<gpu::GPUModuleOp, gpu::BinaryOp>();
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
        [&](gpu::LaunchFuncOp op) -> bool { return converter.isLegal(op); });

    populateVectorToLLVMConversionPatterns(converter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                      target);
    // Upstream patterns (benefit=1 for LaunchFuncOp).
    populateGpuToLLVMConversionPatterns(converter, patterns,
                                        kernelBarePtrCallConv,
                                        kernelIntersperseSizeCallConv);
    // Our pattern (benefit=2) — takes priority for gpu.launch_func.
    patterns.add<FlyLaunchFuncOpPattern>(converter, kernelBarePtrCallConv,
                                         kernelIntersperseSizeCallConv);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
