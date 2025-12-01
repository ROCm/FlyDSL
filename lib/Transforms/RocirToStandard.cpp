#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h.inc"
#include "rocir/RocirPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::rocir;

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
      : RewritePattern("rocir.size", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value input = op->getOperand(0);
    
    // Get the defining operation
    auto *defOp = input.getDefiningOp();
    if (!defOp)
      return failure();
    
    // Handle make_shape
    if (defOp->getName().getStringRef() == "rocir.make_shape") {
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
    if (defOp->getName().getStringRef() == "rocir.make_layout") {
      auto shapeOp = defOp->getOperand(0).getDefiningOp();
      if (!shapeOp || shapeOp->getName().getStringRef() != "rocir.make_shape")
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
      : RewritePattern("rocir.cosize", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value layout = op->getOperand(0);
    
    auto *layoutOp = layout.getDefiningOp();
    if (!layoutOp || layoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    auto *shapeOp = layoutOp->getOperand(0).getDefiningOp();
    auto *strideOp = layoutOp->getOperand(1).getDefiningOp();
    
    if (!shapeOp || !strideOp)
      return failure();
    if (shapeOp->getName().getStringRef() != "rocir.make_shape" ||
        strideOp->getName().getStringRef() != "rocir.make_stride")
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
      : RewritePattern("rocir.get", 1, ctx) {}

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
    if (opName != "rocir.make_shape" && opName != "rocir.make_stride" && 
        opName != "rocir.make_coord")
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
      : RewritePattern("rocir.rank", 1, ctx) {}

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
      : RewritePattern("rocir.crd2idx", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    if (op->getNumOperands() != 2)
      return failure();
      
    Value coord = op->getOperand(0);
    Value layout = op->getOperand(1);
    
    auto *coordOp = coord.getDefiningOp();
    auto *layoutOp = layout.getDefiningOp();
    
    if (!coordOp || coordOp->getName().getStringRef() != "rocir.make_coord")
      return failure();
    if (!layoutOp || layoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    if (layoutOp->getNumOperands() < 2)
      return failure();
      
    auto *strideOp = layoutOp->getOperand(1).getDefiningOp();
    if (!strideOp || strideOp->getName().getStringRef() != "rocir.make_stride")
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
      : RewritePattern("rocir.idx2crd", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value idx = op->getOperand(0);
    Value layout = op->getOperand(1);
    
    auto *layoutOp = layout.getDefiningOp();
    if (!layoutOp || layoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    auto *shapeOp = layoutOp->getOperand(0).getDefiningOp();
    if (!shapeOp || shapeOp->getName().getStringRef() != "rocir.make_shape")
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
      if (i == static_cast<int>(shapes.size()) - 1) {
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
      : RewritePattern("rocir.get_shape", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value layout = op->getOperand(0);
    auto *layoutOp = layout.getDefiningOp();
    
    if (!layoutOp || layoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    // Simply forward the shape operand
    rewriter.replaceOp(op, layoutOp->getOperand(0));
    return success();
  }
};

// Lower cute.get_stride to extract stride from layout  
struct GetStrideOpLowering : public RewritePattern {
  GetStrideOpLowering(MLIRContext *ctx)
      : RewritePattern("rocir.get_stride", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value layout = op->getOperand(0);
    auto *layoutOp = layout.getDefiningOp();
    
    if (!layoutOp || layoutOp->getName().getStringRef() != "rocir.make_layout")
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

// Lower cute.composition to create composed layout
struct CompositionOpLowering : public OpRewritePattern<CompositionOp> {
  using OpRewritePattern<CompositionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompositionOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value layoutA = op.getOperand(0);
    Value layoutB = op.getOperand(1);
    
    // Get defining operations
    auto *layoutAOp = layoutA.getDefiningOp();
    auto *layoutBOp = layoutB.getDefiningOp();
    
    if (!layoutAOp || !layoutBOp)
      return failure();
    if (layoutAOp->getName().getStringRef() != "rocir.make_layout" ||
        layoutBOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    // Extract shape and stride from both layouts
    auto *shapeAOp = layoutAOp->getOperand(0).getDefiningOp();
    auto *strideAOp = layoutAOp->getOperand(1).getDefiningOp();
    auto *shapeBOp = layoutBOp->getOperand(0).getDefiningOp();
    auto *strideBOp = layoutBOp->getOperand(1).getDefiningOp();
    
    if (!shapeAOp || !strideAOp || !shapeBOp || !strideBOp)
      return failure();
    
    // auto shapeAVals = shapeAOp->getOperands();
    auto strideAVals = strideAOp->getOperands();
    auto shapeBVals = shapeBOp->getOperands();
    auto strideBVals = strideBOp->getOperands();
    
    // Composition: result.shape = shapeB, result.stride[i] = strideB[i] * strideA[i]
    if (strideAVals.size() != strideBVals.size())
      return failure();
    
    // Compute composed strides: strideB[i] * strideA[i]
    SmallVector<Value> composedStrides;
    for (size_t i = 0; i < strideBVals.size(); ++i) {
      auto mul = rewriter.create<arith::MulIOp>(loc, strideBVals[i], strideAVals[i]);
      composedStrides.push_back(mul.getResult());
    }
    
    // Create new shape and stride
    auto resultType = op.getResult().getType();
    auto shapeType = layoutBOp->getOperand(0).getType();
    auto strideType = layoutBOp->getOperand(1).getType();
    
    auto newShape = rewriter.create<MakeShapeOp>(loc, shapeType, shapeBVals);
    auto newStride = rewriter.create<MakeStrideOp>(loc, strideType, composedStrides);
    auto newLayout = rewriter.create<MakeLayoutOp>(loc, resultType, 
                                                    newShape.getResult(), 
                                                    newStride.getResult());
    
    rewriter.replaceOp(op, newLayout.getResult());
    return success();
  }
};

// Forwarding patterns for product operations - these preserve the operations
// but allow get_shape/get_stride to work on them
struct LogicalProductOpLowering : public OpRewritePattern<LogicalProductOp> {
  using OpRewritePattern<LogicalProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LogicalProductOp op,
                                PatternRewriter &rewriter) const override {
    // logical_product(block, tiler) creates a tiled layout  
    // Use composition as approximation
    auto loc = op.getLoc();
    Value inputLayout = op.getOperand(0);  // block
    Value tilerLayout = op.getOperand(1);  // tiler
    
    auto composed = rewriter.create<CompositionOp>(
        loc, op.getResult().getType(), inputLayout, tilerLayout);
    
    rewriter.replaceOp(op, composed.getResult());
    return success();
  }
};

struct ZippedProductOpLowering : public OpRewritePattern<ZippedProductOp> {
  using OpRewritePattern<ZippedProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ZippedProductOp op,
                                PatternRewriter &rewriter) const override {
    // zipped_product = tile_unzip(logical_product(block, tiler), tiler)
    // Simplified: just use logical_product
    auto loc = op.getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalProd.getResult());
    return success();
  }
};

struct LogicalDivideOpLowering : public OpRewritePattern<LogicalDivideOp> {
  using OpRewritePattern<LogicalDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LogicalDivideOp op,
                                PatternRewriter &rewriter) const override {
    // logical_divide(layout, tiler) partitions the layout by tiler
    // Implementation: composition(layout, make_layout(tiler, complement(...)))
    auto loc = op.getLoc();
    Value inputLayout = op.getOperand(0);  // target
    Value tilerLayout = op.getOperand(1);  // tiler
    
    // Simplified: use composition in reverse
    auto composed = rewriter.create<CompositionOp>(
        loc, op.getResult().getType(), tilerLayout, inputLayout);
    
    rewriter.replaceOp(op, composed.getResult());
    return success();
  }
};

struct TiledDivideOpLowering : public OpRewritePattern<TiledDivideOp> {
  using OpRewritePattern<TiledDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledDivideOp op,
                                PatternRewriter &rewriter) const override {
    // tiled_divide is similar to logical_divide but with different packing
    // Simplified: use logical_divide
    auto loc = op.getLoc();
    auto logicalDiv = rewriter.create<LogicalDivideOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalDiv.getResult());
    return success();
  }
};

struct TiledProductOpLowering : public OpRewritePattern<TiledProductOp> {
  using OpRewritePattern<TiledProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledProductOp op,
                                PatternRewriter &rewriter) const override {
    // tiled_product = tile_to_shape(logical_product(block, tiler), block, tiler)
    // Simplified: use logical_product
    auto loc = op.getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalProd.getResult());
    return success();
  }
};

struct FlatProductOpLowering : public OpRewritePattern<FlatProductOp> {
  using OpRewritePattern<FlatProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FlatProductOp op,
                                PatternRewriter &rewriter) const override {
    // flat_product flattens the result of logical_product
    // Simplified: use logical_product (flattening happens later)
    auto loc = op.getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalProd.getResult());
    return success();
  }
};

struct RakedProductOpLowering : public OpRewritePattern<RakedProductOp> {
  using OpRewritePattern<RakedProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RakedProductOp op,
                                PatternRewriter &rewriter) const override {
    // raked_product creates a raked (interleaved) layout
    // Simplified: use logical_product
    auto loc = op.getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalProd.getResult());
    return success();
  }
};

struct BlockedProductOpLowering : public OpRewritePattern<BlockedProductOp> {
  using OpRewritePattern<BlockedProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockedProductOp op,
                                PatternRewriter &rewriter) const override {
    // blocked_product creates a blocked layout
    // Simplified: use logical_product
    auto loc = op.getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalProd.getResult());
    return success();
  }
};

struct ZippedDivideOpLowering : public OpRewritePattern<ZippedDivideOp> {
  using OpRewritePattern<ZippedDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ZippedDivideOp op,
                                PatternRewriter &rewriter) const override {
    // zipped_divide is similar to logical_divide with zipping
    // Simplified: use logical_divide
    auto loc = op.getLoc();
    auto logicalDiv = rewriter.create<LogicalDivideOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalDiv.getResult());
    return success();
  }
};

struct FlatDivideOpLowering : public OpRewritePattern<FlatDivideOp> {
  using OpRewritePattern<FlatDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FlatDivideOp op,
                                PatternRewriter &rewriter) const override {
    // flat_divide flattens the result of logical_divide
    // Simplified: use logical_divide
    auto loc = op.getLoc();
    auto logicalDiv = rewriter.create<LogicalDivideOp>(
        loc, op.getResult().getType(), op.getOperand(0), op.getOperand(1));
    
    rewriter.replaceOp(op, logicalDiv.getResult());
    return success();
  }
};


struct LocalPartitionOpLowering : public OpRewritePattern<LocalPartitionOp> {
  using OpRewritePattern<LocalPartitionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalPartitionOp op,
                                PatternRewriter &rewriter) const override {
    // local_partition(tensor, tile, index) partitions the tensor for thread index
    // Semantics: outer_partition(tensor, tile, coord_to_index(tile, index))
    // Simplified: use logical_divide as approximation
    auto loc = op.getLoc();
    auto inputLayout = op.getOperand(0);  // tensor
    auto tilerLayout = op.getOperand(1);  // tile
    // Note: index is used to select which partition, but we simplify to just divide
    
    auto logicalDiv = rewriter.create<LogicalDivideOp>(
        loc, op.getResult().getType(), inputLayout, tilerLayout);
    
    rewriter.replaceOp(op, logicalDiv.getResult());
    return success();
  }
};

struct LocalTileOpLowering : public OpRewritePattern<LocalTileOp> {
  using OpRewritePattern<LocalTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalTileOp op,
                                PatternRewriter &rewriter) const override {
    // local_tile(tensor, tiler, coord) extracts a tile at specific coordinates
    // Semantics: zipped_divide(tensor, tiler)[coord]
    // Simplified: use logical_divide (coordinate selection happens at runtime)
    auto loc = op.getLoc();
    auto inputLayout = op.getOperand(0);  // tensor layout
    auto tilerShape = op.getOperand(1);   // tiler shape
    // coord is used to select which tile, but we simplify to just tiling
    
    // Create a layout from the tiler shape (with default strides)
    auto tilerRank = llvm::cast<ShapeType>(tilerShape.getType()).getRank();
    SmallVector<Value> ones;
    for (int i = 0; i < tilerRank; ++i) {
      ones.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }
    
    auto strideType = StrideType::get(rewriter.getContext(), tilerRank);
    auto makeStrideOp = rewriter.create<MakeStrideOp>(loc, strideType, ones);
    
    auto layoutType = LayoutType::get(rewriter.getContext(), tilerRank);
    auto tilerLayout = rewriter.create<MakeLayoutOp>(
        loc, layoutType, tilerShape, makeStrideOp.getResult());
    
    // Use logical_divide to partition the tensor
    auto logicalDiv = rewriter.create<LogicalDivideOp>(
        loc, inputLayout.getType(), inputLayout, tilerLayout.getResult());
    
    rewriter.replaceOp(op, logicalDiv.getResult());
    return success();
  }
};


// DCE patterns for make_shape, make_stride, make_layout
struct MakeShapeOpLowering : public OpRewritePattern<MakeShapeOp> {
  using OpRewritePattern<MakeShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeShapeOp op,
                                PatternRewriter &rewriter) const override {
    // If the result is unused, erase the op
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct MakeStrideOpLowering : public OpRewritePattern<MakeStrideOp> {
  using OpRewritePattern<MakeStrideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeStrideOp op,
                                PatternRewriter &rewriter) const override {
    // If the result is unused, erase the op
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct MakeLayoutOpLowering : public OpRewritePattern<MakeLayoutOp> {
  using OpRewritePattern<MakeLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeLayoutOp op,
                                PatternRewriter &rewriter) const override {
    // If the result is unused, erase the op
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

#define GEN_PASS_DEF_ROCIRTOSTANDARDPASS
#include "rocir/RocirPasses.h.inc"

struct RocirToStandardPass
    : public impl::RocirToStandardPassBase<RocirToStandardPass> {
  
  using impl::RocirToStandardPassBase<RocirToStandardPass>::RocirToStandardPassBase;
  
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
    patterns.add<CompositionOpLowering>(&getContext());
    patterns.add<LogicalProductOpLowering>(&getContext());
    patterns.add<ZippedProductOpLowering>(&getContext());
    patterns.add<LogicalDivideOpLowering>(&getContext());
    patterns.add<TiledProductOpLowering>(&getContext());
    patterns.add<FlatProductOpLowering>(&getContext());
    patterns.add<RakedProductOpLowering>(&getContext());
    patterns.add<BlockedProductOpLowering>(&getContext());
    patterns.add<ZippedDivideOpLowering>(&getContext());
    patterns.add<FlatDivideOpLowering>(&getContext());
    patterns.add<LocalPartitionOpLowering>(&getContext());
    patterns.add<LocalTileOpLowering>(&getContext());
    patterns.add<MakeShapeOpLowering>(&getContext());
    patterns.add<MakeStrideOpLowering>(&getContext());
    patterns.add<MakeLayoutOpLowering>(&getContext());
    patterns.add<TiledDivideOpLowering>(&getContext());
    

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}

namespace mlir {
namespace rocir {

std::unique_ptr<Pass> createRocirToStandardPass() {
  return std::make_unique<RocirToStandardPass>();
}

}
}




