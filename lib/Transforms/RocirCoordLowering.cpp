//===- RocirCoordLowering.cpp - Lower Rocir Coord Ops to Arith -----------===//
//
// Implements lowering patterns for Rocir coordinate operations:
// - rocir.make_coord → Extract individual indices (identity)
// - rocir.crd2idx → Arithmetic: sum(coord[i] * stride[i])
// - rocir.idx2crd → Arithmetic: division/modulo chain
// - rocir.rank → Constant folding
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "rocir/RocirPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::rocir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Get rank from type
static int getRankFromType(Type type) {
  if (auto coordType = llvm::dyn_cast<CoordType>(type))
    return coordType.getRank();
  if (auto layoutType = llvm::dyn_cast<LayoutType>(type))
    return layoutType.getRank();
  if (auto shapeType = llvm::dyn_cast<ShapeType>(type))
    return shapeType.getRank();
  if (auto strideType = llvm::dyn_cast<StrideType>(type))
    return strideType.getRank();
  return -1;
}

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower rocir.crd2idx to arithmetic operations
/// Computes: sum(coord[i] * stride[i]) for all dimensions
///
/// Example for 2D:
///   %idx = rocir.crd2idx %coord, %layout
/// becomes:
///   %c0 = %coord_value_0
///   %c1 = %coord_value_1
///   %s0 = %stride_value_0
///   %s1 = %stride_value_1
///   %t0 = arith.muli %c0, %s0
///   %t1 = arith.muli %c1, %s1
///   %idx = arith.addi %t0, %t1
struct Crd2IdxOpLowering : public OpRewritePattern<Crd2IdxOp> {
  using OpRewritePattern<Crd2IdxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Crd2IdxOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value coord = op.getCoord();
    Value layout = op.getLayout();

    // Get the coordinate and stride values
    auto coordDef = coord.getDefiningOp<MakeCoordOp>();
    if (!coordDef)
      return failure(); // Coord must be from make_coord

    auto layoutDef = layout.getDefiningOp<MakeLayoutOp>();
    if (!layoutDef)
      return failure(); // Layout must be from make_layout

    Value stride = layoutDef.getStride();
    auto strideDef = stride.getDefiningOp<MakeStrideOp>();
    if (!strideDef)
      return failure(); // Stride must be from make_stride

    auto coordValues = coordDef.getValues();
    auto strideValues = strideDef.getValues();

    if (coordValues.size() != strideValues.size())
      return failure(); // Rank mismatch

    if (coordValues.empty()) {
      // 0D case: return constant 0
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.replaceOp(op, zero.getResult());
      return success();
    }

    // Compute sum(coord[i] * stride[i])
    Value result = nullptr;
    for (size_t i = 0; i < coordValues.size(); ++i) {
      // coord[i] * stride[i]
      Value term = rewriter.create<arith::MulIOp>(loc, coordValues[i], strideValues[i]);
      
      if (result == nullptr) {
        result = term;
      } else {
        // Accumulate: result += term
        result = rewriter.create<arith::AddIOp>(loc, result, term);
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower rocir.idx2crd to arithmetic operations
/// Converts linear index to multi-dimensional coordinate
/// Uses division and modulo for each dimension
///
/// Example for 2D (row-major, shape=[M, N], stride=[N, 1]):
///   %coord = rocir.idx2crd %idx, %layout
/// becomes:
///   %N = %shape_dim_1
///   %row = arith.divui %idx, %N        // idx / N
///   %col = arith.remui %idx, %N        // idx % N
///   %coord = rocir.make_coord %row, %col
struct Idx2CrdOpLowering : public OpRewritePattern<Idx2CrdOp> {
  using OpRewritePattern<Idx2CrdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Idx2CrdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value idx = op.getIdx();
    Value layout = op.getLayout();

    auto layoutDef = layout.getDefiningOp<MakeLayoutOp>();
    if (!layoutDef)
      return failure();

    Value shape = layoutDef.getShape();

    auto shapeDef = shape.getDefiningOp<MakeShapeOp>();
    if (!shapeDef)
      return failure();

    auto shapeValues = shapeDef.getValues();

    if (shapeValues.empty()) {
      // 0D case: empty coordinate
      auto emptyCoord = rewriter.create<MakeCoordOp>(
          loc, CoordType::get(rewriter.getContext(), 0), ValueRange{});
      rewriter.replaceOp(op, emptyCoord.getResult());
      return success();
    }

    // For simplicity, implement for contiguous layouts only
    // Full implementation would need to sort strides and handle general layouts
    
    // For row-major 2D: coord[0] = idx / stride[0], coord[1] = (idx % stride[0]) / stride[1]
    // For general case, we use modulo/division chain
    
    SmallVector<Value> coordValues;
    Value remaining = idx;

    // Process dimensions from highest to lowest stride
    // For row-major (N, 1): first dim = idx/N, second = idx%N
    int rank = shapeValues.size();
    
    if (rank == 1) {
      // 1D case: coord = idx (assuming stride = 1)
      coordValues.push_back(idx);
    } else if (rank == 2) {
      // 2D row-major case: [stride_row, stride_col] = [N, 1]
      // row = idx / N
      // col = idx % N
      Value dimSize = shapeValues[1]; // N (assuming row-major)
      Value row = rewriter.create<arith::DivUIOp>(loc, idx, dimSize);
      Value col = rewriter.create<arith::RemUIOp>(loc, idx, dimSize);
      coordValues.push_back(row);
      coordValues.push_back(col);
    } else {
      // General case: iterative division
      // This is a simplified version - full implementation needs stride sorting
      for (int i = 0; i < rank - 1; ++i) {
        Value divider = shapeValues[i + 1];
        for (int j = i + 2; j < rank; ++j) {
          divider = rewriter.create<arith::MulIOp>(loc, divider, shapeValues[j]);
        }
        Value coord_i = rewriter.create<arith::DivUIOp>(loc, remaining, divider);
        coordValues.push_back(coord_i);
        remaining = rewriter.create<arith::RemUIOp>(loc, remaining, divider);
      }
      // Last dimension
      coordValues.push_back(remaining);
    }

    auto coordType = CoordType::get(rewriter.getContext(), rank);
    auto newCoord = rewriter.create<MakeCoordOp>(loc, coordType, coordValues);
    rewriter.replaceOp(op, newCoord.getResult());
    return success();
  }
};

/// Lower rocir.rank to constant
/// Simply returns the rank from the type
struct RankOpLowering : public OpRewritePattern<RankOp> {
  using OpRewritePattern<RankOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RankOp op,
                                PatternRewriter &rewriter) const override {
    int rank = getRankFromType(op.getInput().getType());
    if (rank < 0)
      return failure();

    auto rankConst = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), rank);
    rewriter.replaceOp(op, rankConst.getResult());
    return success();
  }
};

// Pattern to remove unused make_* operations (dead code elimination)
struct RemoveUnusedMakeShapeOp : public OpRewritePattern<MakeShapeOp> {
  using OpRewritePattern<MakeShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MakeShapeOp op, PatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct RemoveUnusedMakeStrideOp : public OpRewritePattern<MakeStrideOp> {
  using OpRewritePattern<MakeStrideOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MakeStrideOp op, PatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct RemoveUnusedMakeLayoutOp : public OpRewritePattern<MakeLayoutOp> {
  using OpRewritePattern<MakeLayoutOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MakeLayoutOp op, PatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct RemoveUnusedMakeCoordOp : public OpRewritePattern<MakeCoordOp> {
  using OpRewritePattern<MakeCoordOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MakeCoordOp op, PatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};


//===----------------------------------------------------------------------===//
// Coordinate Lowering Pass
//===----------------------------------------------------------------------===//

struct RocirCoordLoweringPass
    : public PassWrapper<RocirCoordLoweringPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocirCoordLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const final { return "rocir-coord-lowering"; }
  StringRef getDescription() const final {
    return "Lower Rocir coordinate operations to arithmetic";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add lowering patterns
    patterns.add<Crd2IdxOpLowering>(context);
    patterns.add<Idx2CrdOpLowering>(context);
    patterns.add<RankOpLowering>(context);
    patterns.add<RemoveUnusedMakeShapeOp>(context);
    patterns.add<RemoveUnusedMakeStrideOp>(context);
    patterns.add<RemoveUnusedMakeLayoutOp>(context);
    patterns.add<RemoveUnusedMakeCoordOp>(context);
    
    // Add dead code elimination for unused rocir constructors
    patterns.add<RemoveUnusedMakeShapeOp>(context);
    patterns.add<RemoveUnusedMakeStrideOp>(context);
    patterns.add<RemoveUnusedMakeLayoutOp>(context);
    patterns.add<RemoveUnusedMakeCoordOp>(context);

    // Apply patterns
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace rocir {

std::unique_ptr<Pass> createRocirCoordLoweringPass() {
  return std::make_unique<RocirCoordLoweringPass>();
}

} // namespace rocir
} // namespace mlir

//===----------------------------------------------------------------------===//
// Dead Code Elimination for Rocir Constructor Operations
//===----------------------------------------------------------------------===//

// Pattern to remove unused make_shape operations
struct RemoveUnusedMakeShapeOp : public OpRewritePattern<MakeShapeOp> {
  using OpRewritePattern<MakeShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeShapeOp op,
                                PatternRewriter &rewriter) const override {
    // If the result is not used, erase the operation
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Pattern to remove unused make_stride operations
struct RemoveUnusedMakeStrideOp : public OpRewritePattern<MakeStrideOp> {
  using OpRewritePattern<MakeStrideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeStrideOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Pattern to remove unused make_layout operations
struct RemoveUnusedMakeLayoutOp : public OpRewritePattern<MakeLayoutOp> {
  using OpRewritePattern<MakeLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Pattern to remove unused make_coord operations  
struct RemoveUnusedMakeCoordOp : public OpRewritePattern<MakeCoordOp> {
  using OpRewritePattern<MakeCoordOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeCoordOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};
