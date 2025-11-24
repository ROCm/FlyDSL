#include "cute/CuteDialect.h"
#include "cute/CutePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::cute;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Get rank from ranked type
static int getRankFromType(Type type) {
  if (auto shapeType = llvm::dyn_cast<ShapeType>(type))
    return shapeType.getRank();
  if (auto strideType = llvm::dyn_cast<StrideType>(type))
    return strideType.getRank();
  if (auto coordType = llvm::dyn_cast<CoordType>(type))
    return coordType.getRank();
  if (auto layoutType = llvm::dyn_cast<LayoutType>(type))
    return layoutType.getRank();
  return -1;
}

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

// Lower cute.size to product of shape elements
struct SizeOpLowering : public RewritePattern {
  SizeOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.size", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value input = op->getOperand(0);
    
    // Get the defining operation
    auto *defOp = input.getDefiningOp();
    if (!defOp)
      return failure();
    
    // Handle make_shape
    if (defOp->getName().getStringRef() == "cute.make_shape") {
      auto values = defOp->getOperands();
      if (values.empty()) {
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        rewriter.replaceOp(op, one.getResult());
        return success();
      }
      
      // Compute product of all shape dimensions
      Value result = values[0];
      for (size_t i = 1; i < values.size(); ++i) {
        result = rewriter.create<arith::MulIOp>(loc, result, values[i]);
      }
      rewriter.replaceOp(op, result);
      return success();
    }
    
    // Handle make_layout - get shape then compute size
    if (defOp->getName().getStringRef() == "cute.make_layout") {
      auto shapeOp = defOp->getOperand(0).getDefiningOp();
      if (!shapeOp || shapeOp->getName().getStringRef() != "cute.make_shape")
        return failure();
      
      auto values = shapeOp->getOperands();
      if (values.empty()) {
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        rewriter.replaceOp(op, one.getResult());
        return success();
      }
      
      Value result = values[0];
      for (size_t i = 1; i < values.size(); ++i) {
        result = rewriter.create<arith::MulIOp>(loc, result, values[i]);
      }
      rewriter.replaceOp(op, result);
      return success();
    }
    
    return failure();
  }
};

// Lower cute.cosize to max(coord[i] * stride[i]) + 1
struct CosizeOpLowering : public RewritePattern {
  CosizeOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.cosize", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value layout = op->getOperand(0);
    
    auto *layoutOp = layout.getDefiningOp();
    if (!layoutOp || layoutOp->getName().getStringRef() != "cute.make_layout")
      return failure();
    
    auto *shapeOp = layoutOp->getOperand(0).getDefiningOp();
    auto *strideOp = layoutOp->getOperand(1).getDefiningOp();
    
    if (!shapeOp || !strideOp)
      return failure();
    if (shapeOp->getName().getStringRef() != "cute.make_shape" ||
        strideOp->getName().getStringRef() != "cute.make_stride")
      return failure();
    
    auto shapes = shapeOp->getOperands();
    auto strides = strideOp->getOperands();
    
    if (shapes.size() != strides.size() || shapes.empty())
      return failure();
    
    // Compute max((shape[i]-1) * stride[i]) + 1
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value maxSpan = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    
    for (size_t i = 0; i < shapes.size(); ++i) {
      auto shapeMinus1 = rewriter.create<arith::SubIOp>(loc, shapes[i], one);
      auto span = rewriter.create<arith::MulIOp>(loc, shapeMinus1, strides[i]);
      
      // max = (span > max) ? span : max
      auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, 
                                                 span, maxSpan);
      maxSpan = rewriter.create<arith::SelectOp>(loc, cmp, span, maxSpan);
    }
    
    auto result = rewriter.create<arith::AddIOp>(loc, maxSpan, one);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

// Lower cute.get to extract element at index
struct GetOpLowering : public RewritePattern {
  GetOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.get", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2)
      return failure();
    
    Value input = op->getOperand(0);
    Value idx = op->getOperand(1);
    
    auto *defOp = input.getDefiningOp();
    if (!defOp)
      return failure();
    
    auto opName = defOp->getName().getStringRef();
    if (opName != "cute.make_shape" && opName != "cute.make_stride" && 
        opName != "cute.make_coord")
      return failure();
    
    auto values = defOp->getOperands();
    
    // If idx is constant, extract directly
    if (auto constOp = idx.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t idxVal = constOp.value();
      if (idxVal >= 0 && idxVal < (int64_t)values.size()) {
        rewriter.replaceOp(op, values[idxVal]);
        return success();
      }
    }
    
    // For dynamic index, would need to use scf.switch or similar
    // For now, just fail if not constant
    return failure();
  }
};

// Lower cute.rank to constant
struct RankOpLowering : public RewritePattern {
  RankOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.rank", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value input = op->getOperand(0);
    
    int rank = getRankFromType(input.getType());
    if (rank < 0)
      return failure();
    
    auto rankConst = rewriter.create<arith::ConstantIndexOp>(loc, rank);
    rewriter.replaceOp(op, rankConst.getResult());
    return success();
  }
};

// Lower cute.crd2idx to arithmetic: sum(coord[i] * stride[i])
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

// Lower cute.idx2crd to compute multi-dim coordinate from linear index
struct Idx2CrdOpLowering : public RewritePattern {
  Idx2CrdOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.idx2crd", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value idx = op->getOperand(0);
    Value layout = op->getOperand(1);
    
    auto *layoutOp = layout.getDefiningOp();
    if (!layoutOp || layoutOp->getName().getStringRef() != "cute.make_layout")
      return failure();
    
    auto *shapeOp = layoutOp->getOperand(0).getDefiningOp();
    if (!shapeOp || shapeOp->getName().getStringRef() != "cute.make_shape")
      return failure();
    
    auto shapes = shapeOp->getOperands();
    if (shapes.empty())
      return failure();
    
    // Compute coordinate: for row-major (stride-1 last dim):
    // coord[n-1] = idx % shape[n-1]
    // coord[n-2] = (idx / shape[n-1]) % shape[n-2]
    // coord[0] = idx / (shape[1] * ... * shape[n-1])
    
    SmallVector<Value> coords;
    Value remaining = idx;
    
    // For each dimension from last to first
    for (int i = shapes.size() - 1; i >= 0; --i) {
      if (i == shapes.size() - 1) {
        // Last dimension: coord = idx % shape
        auto coord = rewriter.create<arith::RemSIOp>(loc, remaining, shapes[i]);
        coords.insert(coords.begin(), coord.getResult());
        remaining = rewriter.create<arith::DivSIOp>(loc, remaining, shapes[i]);
      } else if (i == 0) {
        // First dimension: coord = remaining
        coords.insert(coords.begin(), remaining);
      } else {
        // Middle dimensions: coord = remaining % shape, then div
        auto coord = rewriter.create<arith::RemSIOp>(loc, remaining, shapes[i]);
        coords.insert(coords.begin(), coord.getResult());
        remaining = rewriter.create<arith::DivSIOp>(loc, remaining, shapes[i]);
      }
    }
    
    // Create make_coord with computed coordinates
    auto coordType = op->getResult(0).getType();
    auto makeCoord = rewriter.create<MakeCoordOp>(loc, coordType, coords);
    rewriter.replaceOp(op, makeCoord.getResult());
    return success();
  }
};

// Lower cute.get_shape to extract shape from layout
struct GetShapeOpLowering : public RewritePattern {
  GetShapeOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.get_shape", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value layout = op->getOperand(0);
    auto *layoutOp = layout.getDefiningOp();
    
    if (!layoutOp || layoutOp->getName().getStringRef() != "cute.make_layout")
      return failure();
    
    // Simply forward the shape operand
    rewriter.replaceOp(op, layoutOp->getOperand(0));
    return success();
  }
};

// Lower cute.get_stride to extract stride from layout  
struct GetStrideOpLowering : public RewritePattern {
  GetStrideOpLowering(MLIRContext *ctx)
      : RewritePattern("cute.get_stride", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value layout = op->getOperand(0);
    auto *layoutOp = layout.getDefiningOp();
    
    if (!layoutOp || layoutOp->getName().getStringRef() != "cute.make_layout")
      return failure();
    
    // Simply forward the stride operand
    rewriter.replaceOp(op, layoutOp->getOperand(1));
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
    
    RewritePatternSet patterns(&getContext());
    
    // Add all lowering patterns
    patterns.add<SizeOpLowering>(&getContext());
    patterns.add<CosizeOpLowering>(&getContext());
    patterns.add<GetOpLowering>(&getContext());
    patterns.add<RankOpLowering>(&getContext());
    patterns.add<Crd2IdxOpLowering>(&getContext());
    patterns.add<Idx2CrdOpLowering>(&getContext());
    patterns.add<GetShapeOpLowering>(&getContext());
    patterns.add<GetStrideOpLowering>(&getContext());
    

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
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
