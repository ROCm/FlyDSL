#include "cute/CuteDialect.h"
#include "cute/CutePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::cute;

namespace {

// Lower crd2idx to arithmetic computation: sum(coord[i] * stride[i])
struct Crd2IdxOpLowering : public RewritePattern {
  Crd2IdxOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.crd2idx", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    if (op->getNumOperands() != 2)
      return failure();
      
    Value coord = op->getOperand(0);
    Value layout = op->getOperand(1);
    
    auto *coordOp = coord.getDefiningOp();
    auto *layoutOp = layout.getDefiningOp();
    
    if (!coordOp || coordOp->getName().getStringRef() != "cute.make_coord")
      return failure();
    if (!layoutOp || layoutOp->getName().getStringRef() != "cute.make_layout")
      return failure();
    
    if (layoutOp->getNumOperands() < 2)
      return failure();
      
    auto *strideOp = layoutOp->getOperand(1).getDefiningOp();
    if (!strideOp || strideOp->getName().getStringRef() != "cute.make_stride")
      return failure();
    
    auto coordValues = coordOp->getOperands();
    auto strideValues = strideOp->getOperands();
    
    if (coordValues.size() != strideValues.size())
      return failure();
    
    if (coordValues.empty()) {
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.replaceOp(op, zero.getResult());
      return success();
    }
    
    // Compute sum(coord[i] * stride[i])
    Value result = nullptr;
    for (size_t i = 0; i < coordValues.size(); ++i) {
      auto product = rewriter.create<arith::MulIOp>(loc, 
        coordValues[i], strideValues[i]);
      
      if (result) {
        result = rewriter.create<arith::AddIOp>(loc, result, product.getResult());
      } else {
        result = product.getResult();
      }
    }
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower make_* operations by erasing them (their values are used directly)
struct MakeOpLowering : public RewritePattern {
  MakeOpLowering(StringRef opName, MLIRContext *ctx)
      : RewritePattern(opName, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

#define GEN_PASS_DEF_CUTETOSTANDARDPASS
#include "cute/CutePasses.h.inc"

struct CuteToStandardPass
    : public impl::CuteToStandardPassBase<CuteToStandardPass> {
  
  using impl::CuteToStandardPassBase<CuteToStandardPass>::CuteToStandardPassBase;
  
  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();
    
    ConversionTarget target(*ctx);
    
    target.addLegalDialect<arith::ArithDialect,
                          memref::MemRefDialect,
                          func::FuncDialect,
                          scf::SCFDialect>();
    
    // Mark cute dialect as illegal to trigger lowering
    target.addIllegalDialect<CuteDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<Crd2IdxOpLowering>(ctx);
    patterns.add<MakeOpLowering>("cute.make_coord", ctx);
    patterns.add<MakeOpLowering>("cute.make_stride", ctx);
    patterns.add<MakeOpLowering>("cute.make_layout", ctx);
    patterns.add<MakeOpLowering>("cute.make_shape", ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}

namespace mlir {
namespace cute {

std::unique_ptr<Pass> createCuteToStandardPass() {
  return std::make_unique<CuteToStandardPass>();
}

}
}
