#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "rocir/RocirPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include <vector>
#include <functional>
#include <optional>
#include <utility>
#include <numeric>
#include <algorithm>

using namespace mlir;
using namespace mlir::rocir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions and LayoutNode
//===----------------------------------------------------------------------===//

// Structure to represent the tuple tree of a Layout (Shape and Stride)
struct LayoutNode {
  bool isLeaf;
  Value value; // Only for leaf
  std::vector<LayoutNode> children; // Only for non-leaf
  
  // Constructors
  LayoutNode(Value v) : isLeaf(true), value(v) {}
  LayoutNode(std::vector<LayoutNode> c) : isLeaf(false), children(std::move(c)) {}
  
  // Helpers
  bool isTuple() const { return !isLeaf; }
  size_t rank() const { return isLeaf ? 1 : children.size(); }
  
  // Flatten back to values and structure encoding
  void flatten(SmallVectorImpl<Value>& values, SmallVectorImpl<int32_t>& structure) const {
    if (isLeaf) {
      values.push_back(value);
      structure.push_back(-1);
    } else {
      structure.push_back(static_cast<int32_t>(children.size()));
      for (const auto& child : children) {
        child.flatten(values, structure);
      }
    }
  }

  // Helper to check if fully static (if Value is ConstantOp)
  std::optional<int64_t> getConstantValue() const {
    if (!isLeaf) return std::nullopt;
    if (!value) return std::nullopt;
    if (auto constOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
      return constOp.value();
    }
    return std::nullopt;
  }
};

// Helper to deserialize from MakeShape/MakeStride ops
static LayoutNode deserializeLayoutNode(Operation* op) {
    SmallVector<Value> values;
    ArrayRef<int32_t> structure;
    
    if (isa<MakeShapeOp>(op)) {
        auto makeShape = cast<MakeShapeOp>(op);
        values = makeShape.getValues();
        if (auto type = dyn_cast<ShapeType>(makeShape.getResult().getType())) {
            structure = type.getStructure();
        }
    } else if (isa<MakeStrideOp>(op)) {
        auto makeStride = cast<MakeStrideOp>(op);
        values = makeStride.getValues();
        if (auto type = dyn_cast<StrideType>(makeStride.getResult().getType())) {
            structure = type.getStructure();
        }
    } else {
        // Not a make op, assume leaf or fail?
        // If it's a value coming from elsewhere (e.g. function arg), we might assume flat structure or treat as leaf.
        // For now, treat as single leaf if it produces a shape/stride type with rank 0/1, or fail.
        if (!op) return LayoutNode(Value()); 
        return LayoutNode(op->getResult(0)); 
    }
    
    if (structure.empty()) {
        // Assume flat structure - but values might be nested shapes/strides!
        // Need to recursively deserialize nested values
        std::vector<LayoutNode> children;
        for (auto v : values) {
            // Check if this value is a nested shape/stride
            if (auto *defOp = v.getDefiningOp()) {
                auto opName = defOp->getName().getStringRef();
                if (opName == "rocir.make_shape" || opName == "rocir.make_stride") {
                    // Recursively deserialize nested structure
                    children.push_back(deserializeLayoutNode(defOp));
                    continue;
                }
            }
            // Leaf value (index)
            children.push_back(LayoutNode(v));
        }
        if (children.empty()) {
             // Empty tuple
             return LayoutNode(std::vector<LayoutNode>{});
        }
        return LayoutNode(children);
    }
    
    // Parse structure
    int valueIdx = 0;
    int structIdx = 0;
    
    std::function<LayoutNode()> parse = [&]() -> LayoutNode {
        if (structIdx >= static_cast<int>(structure.size())) return LayoutNode(Value());
        
        int32_t code = structure[structIdx++];
        if (code == -1) {
            // Leaf - get next value
            if (valueIdx >= static_cast<int>(values.size())) return LayoutNode(Value());
            Value v = values[valueIdx++];
            
            // Check if this value is actually a nested shape/stride
            if (auto *defOp = v.getDefiningOp()) {
                auto opName = defOp->getName().getStringRef();
                if (opName == "rocir.make_shape" || opName == "rocir.make_stride") {
                    // Recursively deserialize nested structure
                    return deserializeLayoutNode(defOp);
                }
            }
            
            return LayoutNode(v);
        } else {
            std::vector<LayoutNode> children;
            for (int i = 0; i < code; ++i) {
                children.push_back(parse());
            }
            return LayoutNode(children);
        }
    };
    
    return parse();
}

// Forward declaration
static void flatten_to_leaves(const LayoutNode& node, std::vector<LayoutNode>& leaves);

// Serialize LayoutNode back to nested MakeShapeOp/MakeStrideOp operations
// This preserves the nested structure instead of flattening
static Value serializeLayoutNodeToShape(const LayoutNode& node, Location loc, 
                                        PatternRewriter& rewriter, MLIRContext* ctx) {
    if (node.isLeaf) {
        // ERROR: should not call this on a leaf! 
        // Leaves should be part of their parent tuple
        llvm::errs() << "ERROR: serializeLayoutNodeToShape called on leaf!\n";
        auto shapeType = ShapeType::get(ctx, 1);
        return rewriter.create<MakeShapeOp>(loc, shapeType, ValueRange{node.value}).getResult();
    }
    
    // Tuple - check if children are all leaves or if they're sub-tuples
    bool allLeavesChildren = true;
    for (const auto& child : node.children) {
        if (!child.isLeaf) {
            allLeavesChildren = false;
            break;
        }
    }
    
    if (allLeavesChildren) {
        // Children are all leaves - create flat shape with their values
        SmallVector<Value> leafValues;
        for (const auto& child : node.children) {
            leafValues.push_back(child.value);
        }
        auto shapeType = ShapeType::get(ctx, leafValues.size());
        return rewriter.create<MakeShapeOp>(loc, shapeType, leafValues).getResult();
    } else {
        // Children are sub-tuples - recursively serialize and create nested shape
        SmallVector<Value> childShapes;
        for (const auto& child : node.children) {
            childShapes.push_back(serializeLayoutNodeToShape(child, loc, rewriter, ctx));
        }
        auto shapeType = ShapeType::get(ctx, childShapes.size());
        return rewriter.create<MakeShapeOp>(loc, shapeType, childShapes).getResult();
    }
}

static Value serializeLayoutNodeToStride(const LayoutNode& node, Location loc, 
                                         PatternRewriter& rewriter, MLIRContext* ctx) {
    if (node.isLeaf) {
        llvm::errs() << "ERROR: serializeLayoutNodeToStride called on leaf!\n";
        auto strideType = StrideType::get(ctx, 1);
        return rewriter.create<MakeStrideOp>(loc, strideType, ValueRange{node.value}).getResult();
    }
    
    // Check if children are all leaves
    bool allLeavesChildren = true;
    for (const auto& child : node.children) {
        if (!child.isLeaf) {
            allLeavesChildren = false;
            break;
        }
    }
    
    if (allLeavesChildren) {
        // Children are all leaves - create flat stride with their values
        SmallVector<Value> leafValues;
        for (const auto& child : node.children) {
            leafValues.push_back(child.value);
        }
        auto strideType = StrideType::get(ctx, leafValues.size());
        return rewriter.create<MakeStrideOp>(loc, strideType, leafValues).getResult();
    } else {
        // Children are sub-tuples - recursively serialize and create nested stride
        SmallVector<Value> childStrides;
        for (const auto& child : node.children) {
            childStrides.push_back(serializeLayoutNodeToStride(child, loc, rewriter, ctx));
        }
        auto strideType = StrideType::get(ctx, childStrides.size());
        return rewriter.create<MakeStrideOp>(loc, strideType, childStrides).getResult();
    }
}

// Helper to aggressively fold binary arithmetic operations on constants
static Value foldBinaryOp(Location loc, Value lhs, Value rhs, 
                          std::function<int64_t(int64_t, int64_t)> op,
                          std::function<Value(Location, Value, Value, PatternRewriter&)> createOp,
                          PatternRewriter& rewriter) {
    auto lhsConst = dyn_cast_or_null<arith::ConstantOp>(lhs.getDefiningOp());
    auto rhsConst = dyn_cast_or_null<arith::ConstantOp>(rhs.getDefiningOp());
    
    if (lhsConst && rhsConst) {
        auto lhsAttr = dyn_cast<IntegerAttr>(lhsConst.getValue());
        auto rhsAttr = dyn_cast<IntegerAttr>(rhsConst.getValue());
        if (lhsAttr && rhsAttr) {
            int64_t result = op(lhsAttr.getInt(), rhsAttr.getInt());
            return rewriter.create<arith::ConstantIndexOp>(loc, result);
        }
    }
    
    return createOp(loc, lhs, rhs, rewriter);
}

// Helper to compute complement inline (without creating ComplementOp)
// Returns (complementShape, complementStride) as Values
static std::pair<Value, Value> computeComplementInline(
    const LayoutNode& tilerShapeNode, const LayoutNode& tilerStrideNode,
    Value targetSize, Location loc, PatternRewriter& rewriter) {
    
    // Flatten to leaves
    std::vector<LayoutNode> shapeLeaves;
    std::vector<LayoutNode> strideLeaves;
    flatten_to_leaves(tilerShapeNode, shapeLeaves);
    flatten_to_leaves(tilerStrideNode, strideLeaves);
    
    SmallVector<Value> shapes;
    SmallVector<Value> strides;
    for (const auto& leaf : shapeLeaves) {
      shapes.push_back(leaf.value);
    }
    for (const auto& leaf : strideLeaves) {
      strides.push_back(leaf.value);
    }
    
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    if (shapes.empty()) {
      // Empty tiler means complement covers entire target
      auto resultShapeType = ShapeType::get(rewriter.getContext(), 1);
      auto resultStrideType = StrideType::get(rewriter.getContext(), 1);
      auto complementShape = rewriter.create<MakeShapeOp>(loc, resultShapeType, ValueRange{targetSize});
      auto complementStride = rewriter.create<MakeStrideOp>(loc, resultStrideType, ValueRange{one});
      return {complementShape.getResult(), complementStride.getResult()};
    }
    
    // Rank-1 complement: split the target into the initial gap and remainder.
    // For rank-1: result_shape = (min_stride / last_result_stride, ceil_div(target, min_stride * shape))
    //            result_stride = (last_result_stride, min_stride * shape)
    if (shapes.size() == 1) {
      auto minStride = strides[0];  // Only one mode
      auto modeShape = shapes[0];
      auto lastResultStride = one;  // Initial result stride is 1
      
      // new_shape = min_stride / last_result_stride
      auto newShape = foldBinaryOp(loc, minStride, lastResultStride,
          [](int64_t a, int64_t b) { return b != 0 ? a / b : a; },
          [](Location l, Value a, Value b, PatternRewriter& r) {
              return r.create<arith::DivUIOp>(l, a, b).getResult();
          },
          rewriter);
      
      // new_stride = min_stride * modeShape
      auto newStride = foldBinaryOp(loc, minStride, modeShape,
          [](int64_t a, int64_t b) { return a * b; },
          [](Location l, Value a, Value b, PatternRewriter& r) {
              return r.create<arith::MulIOp>(l, a, b).getResult();
          },
          rewriter);
      
      // rest_shape = ceil_div(target_size, new_stride)
      auto restShape = foldBinaryOp(loc, targetSize, newStride,
          [](int64_t a, int64_t b) { return b != 0 ? (a + b - 1) / b : a; },
          [](Location l, Value a, Value b, PatternRewriter& r) {
              return r.create<arith::CeilDivUIOp>(l, a, b).getResult();
          },
          rewriter);
      
      SmallVector<Value> resultShapes;
      SmallVector<Value> resultStrides;
      
      // Filter new_shape
      bool newIsOne = false;
      if (auto constOp = newShape.getDefiningOp<arith::ConstantIndexOp>()) {
          newIsOne = (constOp.value() == 1);
      } else if (auto constOp = newShape.getDefiningOp<arith::ConstantOp>()) {
          if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              newIsOne = (attr.getInt() == 1);
          }
      }
      
      if (!newIsOne) {
          resultShapes.push_back(newShape);
          resultStrides.push_back(lastResultStride);
      }
      
      // Filter rest_shape
      bool restIsOne = false;
      if (auto constOp = restShape.getDefiningOp<arith::ConstantIndexOp>()) {
          restIsOne = (constOp.value() == 1);
      } else if (auto constOp = restShape.getDefiningOp<arith::ConstantOp>()) {
          if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              restIsOne = (attr.getInt() == 1);
          }
      }
      
      if (!restIsOne) {
          resultShapes.push_back(restShape);
          resultStrides.push_back(newStride);
      }
      
      // Result: (new_shape, rest_shape):(last_result_stride, new_stride)
      auto resultShapeType = ShapeType::get(rewriter.getContext(), resultShapes.size());
      auto resultStrideType = StrideType::get(rewriter.getContext(), resultStrides.size());
      auto complementShape = rewriter.create<MakeShapeOp>(loc, resultShapeType, resultShapes);
      auto complementStride = rewriter.create<MakeStrideOp>(loc, resultStrideType, resultStrides);
      return {complementShape.getResult(), complementStride.getResult()};
    }
    
    // Rank-2+ complement: sort by stride and fold to compute gap modes.
    struct Mode {
        Value shape;
        Value stride;
        int64_t constStride;
    };
    std::vector<Mode> modes;
    bool allStatic = true;
    
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto *strideDefOp = strides[i].getDefiningOp();
        if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(strideDefOp)) {
            modes.push_back({shapes[i], strides[i], constOp.value()});
        } else {
            allStatic = false;
            modes.push_back({shapes[i], strides[i], 0});
        }
    }
    
    // Sort by stride (only if all static)
    if (allStatic) {
        std::sort(modes.begin(), modes.end(), [](const Mode& a, const Mode& b) {
            return a.constStride < b.constStride;
        });
    }
    
    // Fold: build complement modes from stride-sorted entries.
    SmallVector<Value> compShapeVals;
    SmallVector<Value> compStrideVals;
    Value currStride = one;  // result_stride starts with 1
    
    for (size_t i = 0; i + 1 < modes.size(); ++i) {  // R-1 iterations
        Value minStride = modes[i].stride;
        Value modeShape = modes[i].shape;
        
        // new_shape = min_stride / last_result_stride
        auto newShape = foldBinaryOp(loc, minStride, currStride,
            [](int64_t a, int64_t b) { return b != 0 ? std::max(int64_t(1), a / b) : a; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::DivUIOp>(l, a, b).getResult();
            },
            rewriter);
        
        // Skip size-1 gap modes.
        bool isOne = false;
        if (auto constOp = newShape.getDefiningOp<arith::ConstantIndexOp>()) {
            isOne = (constOp.value() == 1);
        } else if (auto constOp = newShape.getDefiningOp<arith::ConstantOp>()) {
            if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue())) {
                isOne = (attr.getInt() == 1);
            }
        }
        
        if (!isOne) {
            compShapeVals.push_back(newShape);
            compStrideVals.push_back(currStride);
        }
        
        // new_stride = min_stride * modeShape (for next iteration)
        currStride = foldBinaryOp(loc, minStride, modeShape,
            [](int64_t a, int64_t b) { return a * b; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::MulIOp>(l, a, b).getResult();
            },
            rewriter);
    }
    
    // After fold: append last mode.
    Value lastMinStride = modes.back().stride;
    Value lastModeShape = modes.back().shape;
    auto lastNewShape = foldBinaryOp(loc, lastMinStride, currStride,
        [](int64_t a, int64_t b) { return b != 0 ? std::max(int64_t(1), a / b) : a; },
        [](Location l, Value a, Value b, PatternRewriter& r) {
            return r.create<arith::DivUIOp>(l, a, b).getResult();
        },
        rewriter);
    
    // Filter last size-1 mode
    bool lastIsOne = false;
    if (auto constOp = lastNewShape.getDefiningOp<arith::ConstantIndexOp>()) {
        lastIsOne = (constOp.value() == 1);
    } else if (auto constOp = lastNewShape.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            lastIsOne = (attr.getInt() == 1);
        }
    }
    
    if (!lastIsOne) {
        compShapeVals.push_back(lastNewShape);
        compStrideVals.push_back(currStride);
    }
    
    // Update currStride for rest computation
    currStride = foldBinaryOp(loc, lastMinStride, lastModeShape,
        [](int64_t a, int64_t b) { return a * b; },
        [](Location l, Value a, Value b, PatternRewriter& r) {
            return r.create<arith::MulIOp>(l, a, b).getResult();
        },
        rewriter);
    
    // Append rest mode that extends to targetSize.
    auto restShape = foldBinaryOp(loc, targetSize, currStride,
        [](int64_t a, int64_t b) { return b != 0 ? (a + b - 1) / b : a; },
        [](Location l, Value a, Value b, PatternRewriter& r) {
            return r.create<arith::CeilDivUIOp>(l, a, b).getResult();
        },
        rewriter);
        
    // Filter rest size-1 mode
    bool restIsOne = false;
    if (auto constOp = restShape.getDefiningOp<arith::ConstantIndexOp>()) {
        restIsOne = (constOp.value() == 1);
    } else if (auto constOp = restShape.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            restIsOne = (attr.getInt() == 1);
        }
    }
    
    if (!restIsOne) {
        compShapeVals.push_back(restShape);
        compStrideVals.push_back(currStride);
    }
    
    // Create result
    auto resultShapeType = ShapeType::get(rewriter.getContext(), compShapeVals.size());
    auto resultStrideType = StrideType::get(rewriter.getContext(), compStrideVals.size());
    auto complementShape = rewriter.create<MakeShapeOp>(loc, resultShapeType, compShapeVals);
    auto complementStride = rewriter.create<MakeStrideOp>(loc, resultStrideType, compStrideVals);
    return {complementShape.getResult(), complementStride.getResult()};
}

// Helper to coalesce a LayoutNode (combines consecutive modes where stride[i]*shape[i] == stride[i+1])
static std::pair<LayoutNode, LayoutNode> coalesceLayoutNode(
    const LayoutNode& shapeNode, const LayoutNode& strideNode,
    Location loc, PatternRewriter& rewriter) {
  
  // Flatten to leaves
  std::vector<LayoutNode> shapeLeaves;
  std::vector<LayoutNode> strideLeaves;
  flatten_to_leaves(shapeNode, shapeLeaves);
  flatten_to_leaves(strideNode, strideLeaves);
  
  if (shapeLeaves.size() != strideLeaves.size() || shapeLeaves.empty()) {
    // Can't coalesce - return as-is
    return {shapeNode, strideNode};
  }
  
  // Coalesce logic: accumulate modes that are contiguous
  std::vector<LayoutNode> coalescedShape;
  std::vector<LayoutNode> coalescedStride;
  
  Value accumShape = shapeLeaves[0].value;
  Value accumStride = strideLeaves[0].value;
  
  for (size_t i = 1; i < shapeLeaves.size(); ++i) {
    // Compute accumStride * accumShape
    auto accumSizeMul = rewriter.create<arith::MulIOp>(loc, accumStride, accumShape);
    Value accumSize = accumSizeMul.getResult();
    
    // Check if contiguous: accumSize == strideLeaves[i].value
    auto accumSizeConstOp = dyn_cast_or_null<arith::ConstantOp>(accumSize.getDefiningOp());
    auto currStrideConstOp = dyn_cast_or_null<arith::ConstantOp>(strideLeaves[i].value.getDefiningOp());
    
    bool canCombine = false;
    if (accumSizeConstOp && currStrideConstOp) {
      auto accumSizeAttr = dyn_cast<IntegerAttr>(accumSizeConstOp.getValue());
      auto currStrideAttr = dyn_cast<IntegerAttr>(currStrideConstOp.getValue());
      if (accumSizeAttr && currStrideAttr) {
        canCombine = (accumSizeAttr.getInt() == currStrideAttr.getInt());
      }
    }
    
    if (canCombine) {
      // Combine modes
      accumShape = rewriter.create<arith::MulIOp>(loc, accumShape, shapeLeaves[i].value);
    } else {
      // Emit accumulated mode
      coalescedShape.push_back(LayoutNode(accumShape));
      coalescedStride.push_back(LayoutNode(accumStride));
      accumShape = shapeLeaves[i].value;
      accumStride = strideLeaves[i].value;
    }
  }
  
  // Emit final accumulated mode
  coalescedShape.push_back(LayoutNode(accumShape));
  coalescedStride.push_back(LayoutNode(accumStride));
  
  // If we ended up with a single mode, return as leaf
  if (coalescedShape.size() == 1) {
    return {coalescedShape[0], coalescedStride[0]};
  }
  
  // Return as tuple
  return {LayoutNode(coalescedShape), LayoutNode(coalescedStride)};
}

// Helper to compute GCD
[[maybe_unused]] static Value computeGCD(Location loc, Value a, Value b, PatternRewriter &rewriter) {
  // Try constant folding
  auto *aOp = a.getDefiningOp();
  auto *bOp = b.getDefiningOp();
  if (aOp && bOp) {
    auto aConst = dyn_cast<arith::ConstantIndexOp>(aOp);
    auto bConst = dyn_cast<arith::ConstantIndexOp>(bOp);
    if (aConst && bConst) {
       int64_t valA = aConst.value();
       int64_t valB = bConst.value();
       while (valB != 0) {
         int64_t t = valB;
         valB = valA % valB;
         valA = t;
       }
       return rewriter.create<arith::ConstantIndexOp>(loc, valA);
    }
  }
  // Runtime GCD implementation (Euclidean algorithm)
  // while (b != 0) { t = b; b = a % b; a = t; }
  
  // We need to use SCF for loop
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  
  // Initial values
  Value initialA = a;
  Value initialB = b;
  
  auto whileOp = rewriter.create<scf::WhileOp>(
      loc, TypeRange{rewriter.getIndexType(), rewriter.getIndexType()}, 
      ValueRange{initialA, initialB},
      [&](OpBuilder &b, Location loc, ValueRange args) {
          // Condition: b != 0
          Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, args[1], zero);
          b.create<scf::ConditionOp>(loc, cond, args);
      },
      [&](OpBuilder &b, Location loc, ValueRange args) {
          // Body: t = b; b = a % b; a = t;
          Value currA = args[0];
          Value currB = args[1];
          Value rem = b.create<arith::RemUIOp>(loc, currA, currB);
          b.create<scf::YieldOp>(loc, ValueRange{currB, rem});
      });
      
  return whileOp.getResult(0);
}

// Helper for ceil_div: (a + b - 1) / b
static Value ceilDiv(Location loc, Value a, Value b, PatternRewriter &rewriter) {
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto aMinusOne = rewriter.create<arith::SubIOp>(loc, a, one);
    auto sum = rewriter.create<arith::AddIOp>(loc, aMinusOne, b);
    return rewriter.create<arith::DivUIOp>(loc, sum, b);
}

// Helper to create a constant LayoutNode
static LayoutNode makeConstNode(int64_t val, Location loc, PatternRewriter& rewriter) {
    return LayoutNode(rewriter.create<arith::ConstantIndexOp>(loc, val).getResult());
}

// ============================================================================
// Layout Algorithm Helpers
// ============================================================================

// composition(LHS, RHS)
// Forward decl
static std::pair<LayoutNode, LayoutNode> composition_impl(
    const LayoutNode& lhsShape, const LayoutNode& lhsStride,
    const LayoutNode& rhsShape, const LayoutNode& rhsStride,
    Location loc, PatternRewriter& rewriter);

// flatten_to_tuple: recursively flattens a LayoutNode into a tuple of leaves
// Used to convert nested structure into a flat list of leaves for processing
static void flatten_to_leaves(const LayoutNode& node, std::vector<LayoutNode>& leaves) {
    if (node.isLeaf) {
        leaves.push_back(node);
    } else {
        for (const auto& child : node.children) {
            flatten_to_leaves(child, leaves);
        }
    }
}

// composition_impl
// lhs: (shape, stride), rhs: (shape, stride)
static std::pair<LayoutNode, LayoutNode> composition_impl(
    const LayoutNode& lhsShape, const LayoutNode& lhsStride,
    const LayoutNode& rhsShape, const LayoutNode& rhsStride,
    Location loc, PatternRewriter& rewriter) {
    
    // Case 1: RHS is a tuple -> Distribute LHS over RHS (transform_layout)
    if (rhsShape.isTuple()) {
        std::vector<LayoutNode> resShapeChildren;
        std::vector<LayoutNode> resStrideChildren;
        
        // Handle rhsStride that may not mirror rhsShape; assume structural match for now.
        
        size_t nChildren = rhsShape.children.size();
        for (size_t i = 0; i < nChildren; ++i) {
            // Access corresponding stride child. If stride is leaf 0, use 0.
            LayoutNode subStride = (rhsStride.isTuple() && i < rhsStride.children.size()) 
                                   ? rhsStride.children[i] 
                                   : rhsStride; // Failover/Broadcast?
            
            auto [subResShape, subResStride] = composition_impl(
                lhsShape, lhsStride, 
                rhsShape.children[i], subStride, 
                loc, rewriter);
            resShapeChildren.push_back(subResShape);
            resStrideChildren.push_back(subResStride);
        }
        return {LayoutNode(resShapeChildren), LayoutNode(resStrideChildren)};
    }
    
    // Case 2: RHS is a leaf (Integral) -> Fold LHS over RHS
    // Implements the fold logic from composition_impl.
    
    if (!rhsShape.isLeaf || !rhsStride.isLeaf) {
        // Error: RHS should be leaf at this point
        return {rhsShape, rhsStride}; // Fallback: return RHS as-is
    }
    
    Value restShape = rhsShape.value;
    Value restStride = rhsStride.value;
    
    // Verify these are index types, not layout types
    if (!restShape || !restStride) {
        return {rhsShape, rhsStride}; // Fallback
    }
    
    if (!restShape.getType().isIndex() || !restStride.getType().isIndex()) {
        // Type mismatch - cannot compose with non-index values
        return {rhsShape, rhsStride}; // Fallback
    }
    
    std::vector<LayoutNode> resultShapeNodes;
    std::vector<LayoutNode> resultStrideNodes;
    
    // Flatten LHS to leaves for the fold.
    std::vector<LayoutNode> lhsLeavesShape;
    std::vector<LayoutNode> lhsLeavesStride;
    flatten_to_leaves(lhsShape, lhsLeavesShape);
    flatten_to_leaves(lhsStride, lhsLeavesStride);
    
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    for (size_t i = 0; i < lhsLeavesShape.size(); ++i) {
        Value currShape = lhsLeavesShape[i].value;
        Value currStride = lhsLeavesStride[i].value;
        
        // Fold logic derived from composition semantics.
        // next_shape = ceil_div(curr_shape, abs(rest_stride))
        Value nextShape = foldBinaryOp(loc, currShape, restStride,
            [](int64_t a, int64_t b) { return b != 0 ? (a + b - 1) / b : a; },  // ceil_div
            [](Location l, Value a, Value b, PatternRewriter& r) { 
                return r.create<arith::CeilDivUIOp>(l, a, b).getResult(); 
            },
            rewriter);
        
        // Update restStride before checking for early exit.
        // Important: the fold returns next_stride even when breaking early.
        Value nextStride = foldBinaryOp(loc, restStride, currShape,
            [](int64_t a, int64_t b) { return b != 0 ? (a + b - 1) / b : a; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::CeilDivUIOp>(l, a, b).getResult();
            },
            rewriter);
        
        // Check for early exit
        auto nextShapeConst = dyn_cast_or_null<arith::ConstantOp>(nextShape.getDefiningOp());
        auto restShapeConst = dyn_cast_or_null<arith::ConstantOp>(restShape.getDefiningOp());
        
        bool nextIsOne = false, restIsOne = false;
        if (nextShapeConst) {
            if (auto attr = dyn_cast<IntegerAttr>(nextShapeConst.getValue())) {
                nextIsOne = (attr.getInt() == 1);
            }
        }
        if (restShapeConst) {
            if (auto attr = dyn_cast<IntegerAttr>(restShapeConst.getValue())) {
                restIsOne = (attr.getInt() == 1);
            }
        }
        
        // If rest_shape is 1, we are done (nothing more to distribute)
        if (restIsOne) {
             // Update before breaking because next_stride captures the state we need.
             restStride = nextStride;
             break;
        }

        // If next_shape is 1, this LHS atom is skipped by RHS.
        // We MUST NOT add it to the result, but we MUST continue folding
        // so that restStride is properly updated against the remaining LHS atoms.
        if (nextIsOne) {
             restStride = nextStride;
             // Update restShape?
             // restShape = restShape / nextShape = restShape / 1 = restShape.
             // No change to restShape.
             continue;
        }
        
        // new_shape = min(next_shape, rest_shape)
        Value newShape = foldBinaryOp(loc, nextShape, restShape,
            [](int64_t a, int64_t b) { return std::min(a, b); },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::MinUIOp>(l, a, b).getResult();
            },
            rewriter);
        
        // new_stride = currStride * restStride
        Value newStride = foldBinaryOp(loc, currStride, restStride,
            [](int64_t a, int64_t b) { return a * b; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::MulIOp>(l, a, b).getResult();
            },
            rewriter);
        
        // Append to result
        resultShapeNodes.push_back(LayoutNode(newShape));
        resultStrideNodes.push_back(LayoutNode(newStride));
        
        // Update restShape = restShape / newShape with aggressive folding.
        restShape = foldBinaryOp(loc, restShape, newShape,
            [](int64_t a, int64_t b) { return b != 0 ? a / b : a; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::DivUIOp>(l, a, b).getResult();
            },
            rewriter);
        
        // Update restStride (already computed as nextStride above)
        restStride = nextStride;
    }
    
    // Handle remainder of RHS after the fold.
    // Three cases:
    // 1. If result is empty: return (rest_shape, rest_stride * last_lhs_stride)
    // 2. If rest_shape == 1: don't append (early return)
    // 3. Otherwise: append (rest_shape, rest_stride * last_lhs_stride)
    
    if (resultShapeNodes.empty()) {
        // Case 1: No modes were added - return rest as the only mode (as leaf)
        Value lastLhsStride = lhsLeavesStride.empty() ? one : lhsLeavesStride.back().value;
        Value tailStride = foldBinaryOp(loc, restStride, lastLhsStride,
            [](int64_t a, int64_t b) { return a * b; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::MulIOp>(l, a, b).getResult();
            },
            rewriter);
        return {LayoutNode(restShape), LayoutNode(tailStride)};
    }
    
    // Case 2: Check if rest_shape == 1 (try constant folding)
    auto restShapeConst = dyn_cast_or_null<arith::ConstantOp>(restShape.getDefiningOp());
    if (restShapeConst) {
        if (auto attr = dyn_cast<IntegerAttr>(restShapeConst.getValue())) {
            if (attr.getInt() == 1) {
                // rest_shape is 1, don't append
                return {LayoutNode(resultShapeNodes), LayoutNode(resultStrideNodes)};
            }
        }
    }
    
    // Case 3: Append rest to result
    resultShapeNodes.push_back(LayoutNode(restShape));
    Value lastLhsStride = lhsLeavesStride.empty() ? one : lhsLeavesStride.back().value;
    Value tailStride = foldBinaryOp(loc, restStride, lastLhsStride,
        [](int64_t a, int64_t b) { return a * b; },
        [](Location l, Value a, Value b, PatternRewriter& r) {
            return r.create<arith::MulIOp>(l, a, b).getResult();
        },
        rewriter);
    resultStrideNodes.push_back(LayoutNode(tailStride));
    
    return {LayoutNode(resultShapeNodes), LayoutNode(resultStrideNodes)};
}

// complement(shape, stride, cosize_hi)
// Returns a layout that complements the input within `cosize_hi`.
static std::pair<LayoutNode, LayoutNode> complement_impl(
    const LayoutNode& shape, const LayoutNode& stride, Value cosizeHi,
    Location loc, PatternRewriter& rewriter) {
    
    // 1. Work on flattened leaves so filtering can happen deterministically.
    
    std::vector<LayoutNode> leavesShape;
    std::vector<LayoutNode> leavesStride;
    flatten_to_leaves(shape, leavesShape);
    flatten_to_leaves(stride, leavesStride);
    
    // 2. Sort by stride when all strides are static; otherwise rely on the provided order.
    
    // Pack into struct for sorting
    struct Mode {
        Value shape;
        Value stride;
        int64_t constStride;
    };
    std::vector<Mode> modes;
    bool allStatic = true;
    
    for (size_t i = 0; i < leavesShape.size(); ++i) {
        auto s = leavesStride[i].getConstantValue();
        if (s.has_value()) {
            modes.push_back({leavesShape[i].value, leavesStride[i].value, s.value()});
        } else {
            allStatic = false;
            modes.push_back({leavesShape[i].value, leavesStride[i].value, 0}); // Unsortable
        }
    }
    
    if (allStatic) {
        std::sort(modes.begin(), modes.end(), [](const Mode& a, const Mode& b) {
            return a.constStride < b.constStride;
        });
    }
    // If not all static, we proceed in given order (risky but best effort).
    
    // 3. Fold logic: accumulate gaps before each stride milestone.
    std::vector<Value> resShape;
    std::vector<Value> resStride;
    Value currStride = rewriter.create<arith::ConstantIndexOp>(loc, 1); // starts at 1
    resStride.push_back(currStride);
    
    // Iterate through sorted modes
    for (const auto& mode : modes) {
        Value minStride = mode.stride;
        Value modeShape = mode.shape;
        
        // Gap before the current mode.
        Value newShape = rewriter.create<arith::DivUIOp>(loc, minStride, currStride);
        
        resShape.push_back(newShape);
        Value nextStride = rewriter.create<arith::MulIOp>(loc, minStride, modeShape);
        
        // Advance coverage to include this mode.
        currStride = nextStride;
    }
    
    // 4. Append the final mode that stretches coverage to cosizeHi.
    Value restShape = ceilDiv(loc, cosizeHi, currStride, rewriter);
    resShape.push_back(restShape);
    // Stride is currStride
    // resStride already has it? No, we need to record the stride for *this* shape.
    
    // Let's rebuild the vectors properly.
    // We need parallel vectors for shape and stride of the COMPLEMENT.
    std::vector<LayoutNode> compShapeNodes;
    std::vector<LayoutNode> compStrideNodes;
    
    // Reset currStride for generation
    currStride = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    for (const auto& mode : modes) {
        Value minStride = mode.stride;
        Value modeShape = mode.shape;
        
        // Gap
        Value gap = rewriter.create<arith::DivUIOp>(loc, minStride, currStride);
        
        // Record this gap mode even if it is size 1; later passes can coalesce it.
        compShapeNodes.push_back(LayoutNode(gap));
        compStrideNodes.push_back(LayoutNode(currStride));
        
        // Advance
        currStride = rewriter.create<arith::MulIOp>(loc, minStride, modeShape);
    }
    
    // Final rest mode
    Value finalRest = ceilDiv(loc, cosizeHi, currStride, rewriter);
    compShapeNodes.push_back(LayoutNode(finalRest));
    compStrideNodes.push_back(LayoutNode(currStride));
    
    // Coalesce the result (optional but good)
    // We return as a tuple of these modes.
    return {LayoutNode(compShapeNodes), LayoutNode(compStrideNodes)};
}

// logical_divide(layout, tiler)
static std::pair<LayoutNode, LayoutNode> logical_divide_impl(
    const LayoutNode& layoutShape, const LayoutNode& layoutStride,
    const LayoutNode& tilerShape, const LayoutNode& tilerStride,
    Location loc, PatternRewriter& rewriter) {
    
    // Compute logical_divide(layout, tiler) by composing layout with
    // make_layout(tiler, complement(tiler, size(layout))).
    //
    // 1. Flatten layout to determine its total size.
    
    // Calculate size of input layout
    std::vector<LayoutNode> layoutLeaves;
    flatten_to_leaves(layoutShape, layoutLeaves);
    Value inputSize = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (auto& leaf : layoutLeaves) {
        inputSize = rewriter.create<arith::MulIOp>(loc, inputSize, leaf.value);
    }
    
    // 2. Compute complement of tiler relative to input size
    auto [compShape, compStride] = complement_impl(
        tilerShape, tilerStride, inputSize, loc, rewriter);
        
    // 3. Construct RHS layout: (tiler, complement)
    // RHS shape = (tilerShape, compShape)
    // RHS stride = (tilerStride, compStride)
    // This creates a tuple of 2 elements.
    std::vector<LayoutNode> rhsShapeChildren = {tilerShape, compShape};
    std::vector<LayoutNode> rhsStrideChildren = {tilerStride, compStride};
    LayoutNode rhsShape(rhsShapeChildren);
    LayoutNode rhsStride(rhsStrideChildren);
    
    // 4. Composition: layout ◦ rhs
    return composition_impl(layoutShape, layoutStride, rhsShape, rhsStride, loc, rewriter);
}


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

// Helper to compute size of a Value (index or shape/tuple) recursively
static Value computeRecursiveSize(Location loc, Value val, PatternRewriter &rewriter) {
  if (val.getType().isIndex())
    return val;
    
  if (auto *defOp = val.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "rocir.make_shape") {
       auto values = defOp->getOperands();
       if (values.empty())
         return rewriter.create<arith::ConstantIndexOp>(loc, 1);
         
       Value product = computeRecursiveSize(loc, values[0], rewriter);
       for (size_t i = 1; i < values.size(); ++i) {
         Value next = computeRecursiveSize(loc, values[i], rewriter);
         product = rewriter.create<arith::MulIOp>(loc, product, next);
       }
       return product;
    }
  }
  // Create SizeOp for other cases (e.g. block arguments, other ops)
  return rewriter.create<SizeOp>(loc, rewriter.getIndexType(), val).getResult();
}

// Lower size op to the product of shape elements.
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
    
    auto opName = defOp->getName().getStringRef();
    
    // Handle make_shape
    if (opName == "rocir.make_shape") {
      // Use recursive helper to handle nested shapes
      rewriter.replaceOp(op, computeRecursiveSize(loc, input, rewriter));
      return success();
    }
    
    // Handle make_layout - get shape then compute size
    if (opName == "rocir.make_layout") {
      auto shape = defOp->getOperand(0);
      // Use recursive helper on the shape
      rewriter.replaceOp(op, computeRecursiveSize(loc, shape, rewriter));
      return success();
    }
    
    // Handle product operations: size(product(A, B)) = size(A) * size(B)
    if (opName == "rocir.logical_product" || opName == "rocir.zipped_product" ||
        opName == "rocir.tiled_product" || opName == "rocir.flat_product" ||
        opName == "rocir.raked_product" || opName == "rocir.blocked_product") {
      
      Value layoutA = defOp->getOperand(0);
      Value layoutB = defOp->getOperand(1);
      
      // Compute size(A) * size(B)
      auto sizeA = rewriter.create<SizeOp>(loc, rewriter.getIndexType(), layoutA);
      auto sizeB = rewriter.create<SizeOp>(loc, rewriter.getIndexType(), layoutB);
      auto product = rewriter.create<arith::MulIOp>(loc, sizeA, sizeB);
      
      rewriter.replaceOp(op, product.getResult());
      return success();
    }
    
    return failure();
  }
};

// Lower cosize to max(coord[i] * stride[i]) + 1.
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
    
    // Flatten shapes and strides to leaves to handle nested structures
    LayoutNode shapeNode = deserializeLayoutNode(shapeOp);
    LayoutNode strideNode = deserializeLayoutNode(strideOp);
    
    std::vector<LayoutNode> shapeLeaves;
    std::vector<LayoutNode> strideLeaves;
    flatten_to_leaves(shapeNode, shapeLeaves);
    flatten_to_leaves(strideNode, strideLeaves);
    
    if (shapeLeaves.size() != strideLeaves.size() || shapeLeaves.empty())
      return failure();
    
    // Compute max((shape[i]-1) * stride[i]) + 1
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value maxSpan = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    
    for (size_t i = 0; i < shapeLeaves.size(); ++i) {
      Value shape = shapeLeaves[i].value;
      Value stride = strideLeaves[i].value;
      auto shapeMinus1 = rewriter.create<arith::SubIOp>(loc, shape, one);
      auto span = rewriter.create<arith::MulIOp>(loc, shapeMinus1, stride);
      
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

// Lower get to extract an element at the requested index.
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
    
    // Flatten nested structures to leaves for indexing
    // This allows get(shape, 2) to work on nested ((2,2),(2,3)) by flattening to [2,2,2,3]
    SmallVector<Value> flatValues;
    std::function<void(Value)> flattenValue = [&](Value v) {
      // Check if this value is a shape/stride (nested)
      if (auto *vDefOp = v.getDefiningOp()) {
        auto vOpName = vDefOp->getName().getStringRef();
        if (vOpName == "rocir.make_shape" || vOpName == "rocir.make_stride") {
          // Recursively flatten
          for (auto operand : vDefOp->getOperands()) {
            flattenValue(operand);
          }
          return;
        }
      }
      // Leaf value (index)
      flatValues.push_back(v);
    };
    
    for (auto v : values) {
      flattenValue(v);
    }
    
    // If idx is constant, extract from flattened values
    if (auto constOp = idx.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t idxVal = constOp.value();
      if (idxVal >= 0 && idxVal < (int64_t)flatValues.size()) {
        rewriter.replaceOp(op, flatValues[idxVal]);
        return success();
      }
    }
    
    // For dynamic index, would need to use scf.switch or similar
    // For now, just fail if not constant
    return failure();
  }
};

// Lower rank to a constant.
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

// Lower crd2idx to arithmetic: sum(coord[i] * stride[i]).
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

// Lower idx2crd to compute the multi-dim coordinate from a linear index.
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

// Lower get_shape to extract the shape from a layout.
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

// Lower get_stride to extract the stride from a layout.
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


struct CoalesceOpLowering : public OpRewritePattern<CoalesceOp> {
  using OpRewritePattern<CoalesceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoalesceOp op,
                                PatternRewriter &rewriter) const override {
    // Coalesce combines consecutive modes when stride[i] * shape[i] == stride[i+1]
    // Example: (2,1,6):(1,6,2) => can't combine 2:1 and 1:6 (1*2≠6), but 1:6 and 6:2 => 6:2
    //          Actually: (2,1,6):(1,6,2) => (2,6):(1,2) => 12:1
    auto loc = op->getLoc();
    Value layout = op.getLayout();
    
    // Get the layout definition
    auto *layoutOp = layout.getDefiningOp();
    if (!layoutOp || layoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    auto shape = layoutOp->getOperand(0);
    auto stride = layoutOp->getOperand(1);
    
    auto *shapeOp = shape.getDefiningOp();
    auto *strideOp = stride.getDefiningOp();
    
    if (!shapeOp || !strideOp ||
        shapeOp->getName().getStringRef() != "rocir.make_shape" ||
        strideOp->getName().getStringRef() != "rocir.make_stride")
      return failure();
    
    auto shapeDims = shapeOp->getOperands();
    auto strideDims = strideOp->getOperands();
    
    if (shapeDims.size() != strideDims.size() || shapeDims.empty())
      return failure();
    
    // Coalesce: combine consecutive modes where stride[i]*shape[i] == stride[i+1]
    SmallVector<Value> coalescedShape;
    SmallVector<Value> coalescedStride;
    
    // Start with first mode
    Value accumShape = shapeDims[0];
    Value accumStride = strideDims[0];
    
    for (size_t i = 1; i < shapeDims.size(); ++i) {
      // Compute accum_stride * accum_shape
      auto accumSizeMul = rewriter.create<arith::MulIOp>(loc, accumStride, accumShape);
      Value accumSize = accumSizeMul.getResult();
      
      // Check if we can combine: accumSize == strideDims[i]
      // For dynamic values, we need to actually check. For now, always try to combine.
      // A proper implementation would use scf.if to conditionally combine.
      
      // Try to get constant values to decide
      auto accumSizeConstOp = dyn_cast_or_null<arith::ConstantOp>(accumSize.getDefiningOp());
      auto currStrideConstOp = dyn_cast_or_null<arith::ConstantOp>(strideDims[i].getDefiningOp());
      
      bool canCombine = false;
      if (accumSizeConstOp && currStrideConstOp) {
        // Both are constants - check if equal
        auto accumSizeAttr = dyn_cast<IntegerAttr>(accumSizeConstOp.getValue());
        auto currStrideAttr = dyn_cast<IntegerAttr>(currStrideConstOp.getValue());
        if (accumSizeAttr && currStrideAttr) {
          canCombine = (accumSizeAttr.getInt() == currStrideAttr.getInt());
        }
      }
      
      if (canCombine) {
        // Combine: multiply shapes, keep stride
        accumShape = rewriter.create<arith::MulIOp>(loc, accumShape, shapeDims[i]);
        // accumStride stays the same
      } else {
        // Cannot combine - emit accumulated mode and start new one
        coalescedShape.push_back(accumShape);
        coalescedStride.push_back(accumStride);
        accumShape = shapeDims[i];
        accumStride = strideDims[i];
      }
    }
    
    // Don't forget the last accumulated mode
    coalescedShape.push_back(accumShape);
    coalescedStride.push_back(accumStride);
    
    // Create coalesced layout
    auto *ctx = rewriter.getContext();
    size_t coalescedRank = coalescedShape.size();
    auto coalescedShapeType = ShapeType::get(ctx, coalescedRank);
    auto coalescedStrideType = StrideType::get(ctx, coalescedRank);
    
    auto newShape = rewriter.create<MakeShapeOp>(loc, coalescedShapeType, coalescedShape);
    auto newStride = rewriter.create<MakeStrideOp>(loc, coalescedStrideType, coalescedStride);
    
    auto coalescedLayoutType = LayoutType::get(ctx, coalescedRank);
    auto newLayout = rewriter.create<MakeLayoutOp>(
        loc, coalescedLayoutType, newShape.getResult(), newStride.getResult());
    
    rewriter.replaceOp(op, newLayout.getResult());
    return success();
  }
};

struct CompositionOpLowering : public OpRewritePattern<CompositionOp> {
  using OpRewritePattern<CompositionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompositionOp op,
                                PatternRewriter &rewriter) const override {
    // Composition: R = A ◦ B means R(c) = A(B(c)).
    // Coalesce A before invoking composition_impl so the fold sees flat modes.
    
    auto loc = op->getLoc();
    Value layoutA = op.getLayoutA();
    Value layoutB = op.getLayoutB();
    
    // First, coalesce layout A (LHS)
    auto layoutAType = layoutA.getType();
    Value coalescedA = rewriter.create<CoalesceOp>(loc, layoutAType, layoutA);
    
    auto *coalescedAOp = coalescedA.getDefiningOp();
    auto *layoutBOp = layoutB.getDefiningOp();
    
    if (!coalescedAOp || !layoutBOp)
      return failure();
    
    // We need to inline the coalesce operation to get the actual values
    // For now, assume it will be lowered in a subsequent iteration
    // Actually, we can directly work with the original layout if coalesce isn't lowered yet
    
    // Let's work with original A for now (coalesce will run separately)
    auto *layoutAOp = layoutA.getDefiningOp();
    
    if (!layoutAOp || !layoutBOp)
      return failure();
    if (layoutAOp->getName().getStringRef() != "rocir.make_layout" ||
        layoutBOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    // Extract shape and stride operands
    auto shapeAOp = layoutAOp->getOperand(0).getDefiningOp();
    auto strideAOp = layoutAOp->getOperand(1).getDefiningOp();
    auto shapeBOp = layoutBOp->getOperand(0).getDefiningOp();
    auto strideBOp = layoutBOp->getOperand(1).getDefiningOp();
    
    // Deserialize to LayoutNodes
    LayoutNode shapeA = deserializeLayoutNode(shapeAOp);
    LayoutNode strideA = deserializeLayoutNode(strideAOp);
    LayoutNode shapeB = deserializeLayoutNode(shapeBOp);
    LayoutNode strideB = deserializeLayoutNode(strideBOp);
    
    // Coalesce the LHS before composition when it is safe; tuple RHS would lose structure.
    // TODO: implement proper coprofile-based coalescing.
    LayoutNode lhsShapeToUse = shapeA;
    LayoutNode lhsStrideToUse = strideA;
    
    if (!shapeB.isTuple()) {
        // Only coalesce for non-tuple RHS
        auto [coalescedShapeA, coalescedStrideA] = coalesceLayoutNode(shapeA, strideA, loc, rewriter);
        lhsShapeToUse = coalescedShapeA;
        lhsStrideToUse = coalescedStrideA;
    }
    
    // Compute composition recursively
    auto [resShapeNode, resStrideNode] = composition_impl(
        lhsShapeToUse, lhsStrideToUse, shapeB, strideB, loc, rewriter);
    
    // Serialize back to nested MakeShape/MakeStride ops
    // This preserves the nested structure from composition_impl
    auto ctx = rewriter.getContext();
    Value makeShape = serializeLayoutNodeToShape(resShapeNode, loc, rewriter, ctx);
    Value makeStride = serializeLayoutNodeToStride(resStrideNode, loc, rewriter, ctx);
    
    // Determine rank - if nested, use number of top-level children
    int rank = resShapeNode.isLeaf ? 1 : resShapeNode.children.size();
    auto layoutType = LayoutType::get(ctx, rank);
    auto makeLayout = rewriter.create<MakeLayoutOp>(
        loc, layoutType, makeShape, makeStride);
    
    rewriter.replaceOp(op, makeLayout.getResult());
    return success();
  }
};

// Forwarding patterns for product operations - these preserve the operations
// but allow get_shape/get_stride to work on them
struct LogicalProductOpLowering : public OpRewritePattern<LogicalProductOp> {
  using OpRewritePattern<LogicalProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LogicalProductOp op,
                                PatternRewriter &rewriter) const override {
    // logical_product(block, tiler) creates a layout with size = size(block) * size(tiler)
    // For now, create a layout that preserves the correct size semantics
    auto loc = op->getLoc();
    Value blockLayout = op.getInput();
    Value tilerLayout = op.getTiler();
    
    // Get shapes from both layouts
    auto *blockLayoutOp = blockLayout.getDefiningOp();
    auto *tilerLayoutOp = tilerLayout.getDefiningOp();
    
    if (!blockLayoutOp || !tilerLayoutOp ||
        blockLayoutOp->getName().getStringRef() != "rocir.make_layout" ||
        tilerLayoutOp->getName().getStringRef() != "rocir.make_layout") {
      // Fallback to composition
      auto composed = rewriter.create<CompositionOp>(
          loc, op.getResult().getType(), blockLayout, tilerLayout);
      rewriter.replaceOp(op, composed.getResult());
      return success();
    }
    
    auto blockShape = blockLayoutOp->getOperand(0);
    auto blockStride = blockLayoutOp->getOperand(1);
    auto tilerShape = tilerLayoutOp->getOperand(0);
    auto tilerStride = tilerLayoutOp->getOperand(1);
    
    auto *blockShapeOp = blockShape.getDefiningOp();
    auto *tilerShapeOp = tilerShape.getDefiningOp();
    
    if (!blockShapeOp || !tilerShapeOp ||
        blockShapeOp->getName().getStringRef() != "rocir.make_shape" ||
        tilerShapeOp->getName().getStringRef() != "rocir.make_shape") {
      // Fallback
      auto composed = rewriter.create<CompositionOp>(
          loc, op.getResult().getType(), blockLayout, tilerLayout);
      rewriter.replaceOp(op, composed.getResult());
      return success();
    }
    
    // Concatenate block and tiler shape dimensions to create product shape
    // product shape = (block_dims..., tiler_dims...)
    SmallVector<Value> productShapeDims;
    for (auto dim : blockShapeOp->getOperands())
      productShapeDims.push_back(dim);
    for (auto dim : tilerShapeOp->getOperands())
      productShapeDims.push_back(dim);
    
    // Determine the rank of the product
    size_t productRank = productShapeDims.size();
    
    // Create shape and stride types with the correct rank
    auto *ctx = rewriter.getContext();
    auto productShapeType = ShapeType::get(ctx, productRank);
    auto productStrideType = StrideType::get(ctx, productRank);
    
    // Create new shape with combined dimensions
    auto productShape = rewriter.create<MakeShapeOp>(
        loc, productShapeType, productShapeDims);
    
    // For stride, we need to compute appropriate strides
    // For simplicity, create strides that maintain the product structure
    SmallVector<Value> productStrideDims;
    
    // Get block strides
    auto *blockStrideOp = blockStride.getDefiningOp();
    if (blockStrideOp && blockStrideOp->getName().getStringRef() == "rocir.make_stride") {
      for (auto stride : blockStrideOp->getOperands())
        productStrideDims.push_back(stride);
    }
    
    // Get tiler strides and scale them by block size
    auto *tilerStrideOp = tilerStride.getDefiningOp();
    if (tilerStrideOp && tilerStrideOp->getName().getStringRef() == "rocir.make_stride") {
      // Compute block size
      Value blockSize = blockShapeOp->getOperand(0);
      for (size_t i = 1; i < blockShapeOp->getNumOperands(); ++i) {
        blockSize = rewriter.create<arith::MulIOp>(loc, blockSize, 
                                                    blockShapeOp->getOperand(i));
      }
      
      // Scale tiler strides by block size
      for (auto stride : tilerStrideOp->getOperands()) {
        auto scaledStride = rewriter.create<arith::MulIOp>(loc, stride, blockSize);
        productStrideDims.push_back(scaledStride);
      }
    }
    
    auto productStride = rewriter.create<MakeStrideOp>(
        loc, productStrideType, productStrideDims);
    
    // Create the product layout with the correct type
    auto productLayoutType = LayoutType::get(ctx, productRank);
    auto productLayout = rewriter.create<MakeLayoutOp>(
        loc, productLayoutType, productShape.getResult(), 
        productStride.getResult());
    
    rewriter.replaceOp(op, productLayout.getResult());
    return success();
  }
};

struct ZippedProductOpLowering : public OpRewritePattern<ZippedProductOp> {
  using OpRewritePattern<ZippedProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ZippedProductOp op,
                                PatternRewriter &rewriter) const override {
    // zipped_product = tile_unzip(logical_product(block, tiler), tiler)
    // Simplified: just use logical_product
    auto loc = op->getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getInput(), op.getTiler());
    
    rewriter.replaceOp(op, logicalProd.getResult());
    return success();
  }
};

// Helper to compute size product from LayoutNode
static Value computeSizeFromNode(const LayoutNode& node, Location loc, PatternRewriter& rewriter) {
    std::vector<LayoutNode> leaves;
    flatten_to_leaves(node, leaves);
    if (leaves.empty()) return rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    Value size = leaves[0].value;
    for (size_t i = 1; i < leaves.size(); ++i) {
        size = rewriter.create<arith::MulIOp>(loc, size, leaves[i].value);
    }
    return size;
}

// Helper to determine if a LayoutNode represents a "Tuple of Layouts" or a "Layout of Tuples" (Atomic Layout).
// Mirrors the is_tuple<T> trait logic for Layout vs Tuple.
static bool isTuple(const LayoutNode& node) {
    if (node.isLeaf) return false;
    // In Rocir, "Tuple of Layouts" is a Node whose children are Nodes (Shapes).
    // "Atomic Layout" is a Node whose children are Leaves (Indices).
    for (const auto& child : node.children) {
        if (!child.isLeaf) return true;
    }
    return false;
}

static Value lowerLogicalDivide(
    const LayoutNode& inputShape, const LayoutNode& inputStride,
    const LayoutNode& tilerShape, const LayoutNode& tilerStride,
    Location loc, PatternRewriter &rewriter, MLIRContext* ctx,
    bool zipResults) 
{
    // Logical divide behavior:
    // if tiler is a tuple: distribute (transform_layout)
    // else: atomic (composition(..., complement(...))).
    
    bool tilerIsTuple = isTuple(tilerShape);
    bool inputIsTuple = isTuple(inputShape); // For safety, ensure input matches structure

    // Check for distribution
    bool shouldDistribute = tilerIsTuple && inputIsTuple && 
                            (tilerShape.children.size() <= inputShape.children.size());

    if (shouldDistribute) {
        
        SmallVector<Value> resShapes;
        SmallVector<Value> resStrides;
        
        std::vector<Value> subResults;

        // 1. Distribute over matching children
        for (size_t i = 0; i < tilerShape.children.size(); ++i) {
            Value subResult = lowerLogicalDivide(
                inputShape.children[i], inputStride.children[i],
                tilerShape.children[i], tilerStride.children[i],
                loc, rewriter, ctx, zipResults);
            
            if (!subResult) return nullptr;
            subResults.push_back(subResult);
        }

        if (zipResults) {
            // Perform Zipping: Transpose from ((T0, C0), (T1, C1)) to ((T0, T1), (C0, C1))
            // Note: This assumes each subResult is a Tuple of 2: (T, C).
            // If subResult is not (T, C) (e.g. atomic base case returns ((T), (C))), we flatten and regroup.
            
            SmallVector<Value> tilerParts;
            SmallVector<Value> complementParts;
            SmallVector<Value> tilerStrides;
            SmallVector<Value> complementStrides;

            for (auto res : subResults) {
                auto *resOp = res.getDefiningOp(); // MakeLayoutOp
                auto shapeOp = resOp->getOperand(0).getDefiningOp(); // MakeShapeOp
                auto strideOp = resOp->getOperand(1).getDefiningOp(); // MakeStrideOp
                
                // Assuming res is (T, C) tuple structure
                if (shapeOp->getNumOperands() >= 2) {
                    tilerParts.push_back(shapeOp->getOperand(0));
                    complementParts.push_back(shapeOp->getOperand(1));
                    tilerStrides.push_back(strideOp->getOperand(0));
                    complementStrides.push_back(strideOp->getOperand(1));
                } else {
                    // Fallback: treat entire result as Tiler? Or error?
                    // Base case logical_divide returns ((T), (C)).
                    // So it has 2 operands.
                     tilerParts.push_back(shapeOp->getOperand(0));
                     // If only 1 operand, complement is empty?
                     if (shapeOp->getNumOperands() > 1)
                        complementParts.push_back(shapeOp->getOperand(1));
                }
            }
            
            // Construct T_combined and C_combined
            auto tShape = rewriter.create<MakeShapeOp>(loc, ShapeType::get(ctx, tilerParts.size()), tilerParts);
            auto cShape = rewriter.create<MakeShapeOp>(loc, ShapeType::get(ctx, complementParts.size()), complementParts);
            auto tStride = rewriter.create<MakeStrideOp>(loc, StrideType::get(ctx, tilerStrides.size()), tilerStrides);
            auto cStride = rewriter.create<MakeStrideOp>(loc, StrideType::get(ctx, complementStrides.size()), complementStrides);

            resShapes = {tShape.getResult(), cShape.getResult()};
            resStrides = {tStride.getResult(), cStride.getResult()};

        } else {
            // Standard Concatenation (No Zip)
            for (auto res : subResults) {
                auto *subResultOp = res.getDefiningOp();
                resShapes.push_back(subResultOp->getOperand(0));
                resStrides.push_back(subResultOp->getOperand(1));
            }
            
            // 2. Append remaining input children (rest of broadcast)
            for (size_t i = tilerShape.children.size(); i < inputShape.children.size(); ++i) {
                resShapes.push_back(serializeLayoutNodeToShape(inputShape.children[i], loc, rewriter, ctx));
                resStrides.push_back(serializeLayoutNodeToStride(inputStride.children[i], loc, rewriter, ctx));
            }
        }
        
        auto shapeType = ShapeType::get(ctx, resShapes.size());
        auto strideType = StrideType::get(ctx, resStrides.size());
        auto makeShape = rewriter.create<MakeShapeOp>(loc, shapeType, resShapes);
        auto makeStride = rewriter.create<MakeStrideOp>(loc, strideType, resStrides);
        
        auto layoutType = LayoutType::get(ctx, resShapes.size());
        return rewriter.create<MakeLayoutOp>(loc, layoutType, makeShape, makeStride).getResult();
    }
    
    // Base Case: Global Divide
    Value inputSize = computeSizeFromNode(inputShape, loc, rewriter);
    
    auto [complementShape, complementStride] = computeComplementInline(
        tilerShape, tilerStride, inputSize, loc, rewriter);
        
    if (!complementShape || !complementStride)
        return nullptr;
    
    Value tilerShapeVal = serializeLayoutNodeToShape(tilerShape, loc, rewriter, ctx);
    Value tilerStrideVal = serializeLayoutNodeToStride(tilerStride, loc, rewriter, ctx);
    
    SmallVector<Value> combinedShapeVals;
    SmallVector<Value> combinedStrideVals;
    
    if (tilerShape.isLeaf) {
        combinedShapeVals = {tilerShape.value, complementShape};
        combinedStrideVals = {tilerStride.value, complementStride};
    } else {
        combinedShapeVals = {tilerShapeVal, complementShape};
        combinedStrideVals = {tilerStrideVal, complementStride};
    }
    
    int combinedRank = combinedShapeVals.size();
    auto combinedShapeType = ShapeType::get(ctx, combinedRank);
    auto combinedStrideType = StrideType::get(ctx, combinedRank);
    
    auto makeCombinedShape = rewriter.create<MakeShapeOp>(loc, combinedShapeType, combinedShapeVals);
    auto makeCombinedStride = rewriter.create<MakeStrideOp>(loc, combinedStrideType, combinedStrideVals);
    
    LayoutNode combinedShapeNode = deserializeLayoutNode(makeCombinedShape.getOperation());
    LayoutNode combinedStrideNode = deserializeLayoutNode(makeCombinedStride.getOperation());
    
    auto [coalescedInputShape, coalescedInputStride] = coalesceLayoutNode(inputShape, inputStride, loc, rewriter);
    
    auto [resShapeNode, resStrideNode] = composition_impl(
        coalescedInputShape, coalescedInputStride, combinedShapeNode, combinedStrideNode, loc, rewriter);
        
    Value resultShape = serializeLayoutNodeToShape(resShapeNode, loc, rewriter, ctx);
    Value resultStride = serializeLayoutNodeToStride(resStrideNode, loc, rewriter, ctx);
    
    int resultRank = resShapeNode.isLeaf ? 1 : resShapeNode.children.size();
    auto resultLayoutType = LayoutType::get(ctx, resultRank);
    return rewriter.create<MakeLayoutOp>(loc, resultLayoutType, resultShape, resultStride).getResult();
}

struct LogicalDivideOpLowering : public OpRewritePattern<LogicalDivideOp> {
  using OpRewritePattern<LogicalDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LogicalDivideOp op,
                                PatternRewriter &rewriter) const override {
    // logical_divide(layout, tiler)
    // Support both global divide (Layout, Layout) and per-mode divide (Tuple, Tuple)
    // via recursive helper lowerLogicalDivide
    
    auto loc = op->getLoc();
    Value inputLayout = op.getInput();
    Value tilerLayout = op.getTiler();
    
    auto *inputLayoutOp = inputLayout.getDefiningOp();
    auto *tilerLayoutOp = tilerLayout.getDefiningOp();
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "rocir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp());
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp());
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp());
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp());
    
    auto ctx = rewriter.getContext();
    // logical_divide does not zip results of distribution
    Value result = lowerLogicalDivide(
        inputShapeNode, inputStrideNode, 
        tilerShapeNode, tilerStrideNode, 
        loc, rewriter, ctx, 
        /*zipResults=*/false);
    
    if (!result) return failure();
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TiledDivideOpLowering : public OpRewritePattern<TiledDivideOp> {
  using OpRewritePattern<TiledDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledDivideOp op,
                                PatternRewriter &rewriter) const override {
    // tiled_divide(layout, tiler) = tile_unzip(logical_divide(layout, tiler), tiler)
    // Implemented by calling lowerLogicalDivide with zipResults=true
    auto loc = op->getLoc();
    Value inputLayout = op.getInput();
    Value tilerLayout = op.getTiler();
    
    auto *inputLayoutOp = inputLayout.getDefiningOp();
    auto *tilerLayoutOp = tilerLayout.getDefiningOp();
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "rocir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp());
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp());
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp());
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp());
    
    auto ctx = rewriter.getContext();
    // tiled_divide zips results of distribution
    Value result = lowerLogicalDivide(
        inputShapeNode, inputStrideNode, 
        tilerShapeNode, tilerStrideNode, 
        loc, rewriter, ctx, 
        /*zipResults=*/true);
    
    if (!result) return failure();
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TiledProductOpLowering : public OpRewritePattern<TiledProductOp> {
  using OpRewritePattern<TiledProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledProductOp op,
                                PatternRewriter &rewriter) const override {
    // tiled_product = tile_to_shape(logical_product(block, tiler), block, tiler)
    // Simplified: use logical_product
    auto loc = op->getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getInput(), op.getTiler());
    
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
    auto loc = op->getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getInput(), op.getTiler());
    
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
    auto loc = op->getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getInput(), op.getTiler());
    
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
    auto loc = op->getLoc();
    auto logicalProd = rewriter.create<LogicalProductOp>(
        loc, op.getResult().getType(), op.getInput(), op.getTiler());
    
    rewriter.replaceOp(op, logicalProd.getResult());
    return success();
  }
};

struct ZippedDivideOpLowering : public OpRewritePattern<ZippedDivideOp> {
  using OpRewritePattern<ZippedDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ZippedDivideOp op,
                                PatternRewriter &rewriter) const override {
    // zipped_divide(layout, tiler) = tile_unzip(logical_divide(layout, tiler), tiler)
    // Implemented by calling lowerLogicalDivide with zipResults=true
    auto loc = op->getLoc();
    Value inputLayout = op.getInput();
    Value tilerLayout = op.getTiler();
    
    auto *inputLayoutOp = inputLayout.getDefiningOp();
    auto *tilerLayoutOp = tilerLayout.getDefiningOp();
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "rocir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp());
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp());
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp());
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp());
    
    auto ctx = rewriter.getContext();
    // zipped_divide zips results of distribution
    Value result = lowerLogicalDivide(
        inputShapeNode, inputStrideNode, 
        tilerShapeNode, tilerStrideNode, 
        loc, rewriter, ctx, 
        /*zipResults=*/true);
    
    if (!result) return failure();
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct FlatDivideOpLowering : public OpRewritePattern<FlatDivideOp> {
  using OpRewritePattern<FlatDivideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FlatDivideOp op,
                                PatternRewriter &rewriter) const override {
    // flat_divide(layout, tiler) = flatten(zipped_divide(layout, tiler))
    // Implemented by calling lowerLogicalDivide with zipResults=true
    // FlatDivideOpLowering in Rocir likely expects a flattened result?
    // But current lowering just returned Layout.
    // If we return zipped structure ((T0, T1), (C0, C1)), it is nested.
    // We should probably flatten it. But without FlattenOp support, we leave it nested.
    // The user can use flatten() on result.
    
    auto loc = op->getLoc();
    Value inputLayout = op.getInput();
    Value tilerLayout = op.getTiler();
    
    auto *inputLayoutOp = inputLayout.getDefiningOp();
    auto *tilerLayoutOp = tilerLayout.getDefiningOp();
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "rocir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp());
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp());
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp());
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp());
    
    auto ctx = rewriter.getContext();
    // flat_divide uses zipped structure internally before flattening
    Value result = lowerLogicalDivide(
        inputShapeNode, inputStrideNode, 
        tilerShapeNode, tilerStrideNode, 
        loc, rewriter, ctx, 
        /*zipResults=*/true);
    
    if (!result) return failure();
    
    rewriter.replaceOp(op, result);
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
struct ComplementOpLowering : public OpRewritePattern<ComplementOp> {
  using OpRewritePattern<ComplementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComplementOp op,
                                PatternRewriter &rewriter) const override {
    // complement(tiler, target_size) computes the "rest" modes not covered by tiler.
    // Algorithm outline:
    // 1. Filter tiler (remove stride-0, size-1 modes) 
    // 2. Sort by stride
    // 3. Fold: at each step, remove min-stride mode, compute new_shape = min_stride / last_stride
    // 4. Compute rest_shape = ceil_div(target_size, final_stride)
    // 5. Return coalesced layout
    
    auto loc = op->getLoc();
    Value tilerLayout = op.getTiler();
    Value targetSize = op.getTargetSize();
    
    auto *tilerLayoutOp = tilerLayout.getDefiningOp();
    if (!tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "rocir.make_layout")
      return failure();
    
    auto tilerShape = tilerLayoutOp->getOperand(0);
    auto tilerStride = tilerLayoutOp->getOperand(1);
    
    auto *shapeOp = tilerShape.getDefiningOp();
    auto *strideOp = tilerStride.getDefiningOp();
    
    if (!shapeOp || !strideOp ||
        shapeOp->getName().getStringRef() != "rocir.make_shape" ||
        strideOp->getName().getStringRef() != "rocir.make_stride")
      return failure();
    
    SmallVector<Value> shapes(shapeOp->getOperands());
    SmallVector<Value> strides(strideOp->getOperands());
    
    if (shapes.empty()) {
      // Empty tiler means complement covers entire target
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto resultShapeType = ShapeType::get(rewriter.getContext(), 1);
      auto resultStrideType = StrideType::get(rewriter.getContext(), 1);
      auto resultLayoutType = LayoutType::get(rewriter.getContext(), 1);
      
      auto complementShape = rewriter.create<MakeShapeOp>(loc, resultShapeType, ValueRange{targetSize});
      auto complementStride = rewriter.create<MakeStrideOp>(loc, resultStrideType, ValueRange{one});
      auto complementLayout = rewriter.create<MakeLayoutOp>(loc, resultLayoutType, 
                                                             complementShape.getResult(), 
                                                             complementStride.getResult());
      rewriter.replaceOp(op, complementLayout.getResult());
      return success();
    }
    
    // Step 1: Filter out stride-0 and size-1 modes
    SmallVector<Value> filteredShapes;
    SmallVector<Value> filteredStrides;
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    for (size_t i = 0; i < shapes.size(); ++i) {
      auto isZeroStride = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, strides[i], zero);
      auto isOneShape = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, shapes[i], one);
      // Skip if stride is 0 OR shape is 1
      // For simplicity, we'll keep all for now (dynamic filtering is complex)
      // In practice this filtering is often resolved statically.
      filteredShapes.push_back(shapes[i]);
      filteredStrides.push_back(strides[i]);
    }
    
    if (filteredShapes.empty()) {
      // All filtered out, return layout covering target
      auto resultShapeType = ShapeType::get(rewriter.getContext(), 1);
      auto resultStrideType = StrideType::get(rewriter.getContext(), 1);
      auto resultLayoutType = LayoutType::get(rewriter.getContext(), 1);
      
      auto complementShape = rewriter.create<MakeShapeOp>(loc, resultShapeType, ValueRange{targetSize});
      auto complementStride = rewriter.create<MakeStrideOp>(loc, resultStrideType, ValueRange{one});
      auto complementLayout = rewriter.create<MakeLayoutOp>(loc, resultLayoutType,
                                                             complementShape.getResult(),
                                                             complementStride.getResult());
      rewriter.replaceOp(op, complementLayout.getResult());
      return success();
    }
    
    // Step 2-3: Fold algorithm (simplified for rank-1 and rank-2)
    // For now, implement simplified version for common cases
    
    if (filteredShapes.size() == 1) {
      // Rank-1 case: complement is just ceil_div(target_size, stride * shape)
      auto totalStride = rewriter.create<arith::MulIOp>(loc, filteredStrides[0], filteredShapes[0]);
      auto complementShapeVal = rewriter.create<arith::CeilDivUIOp>(loc, targetSize, totalStride);
      
      auto resultShapeType = ShapeType::get(rewriter.getContext(), 1);
      auto resultStrideType = StrideType::get(rewriter.getContext(), 1);
      auto resultLayoutType = LayoutType::get(rewriter.getContext(), 1);
      
      auto complementShape = rewriter.create<MakeShapeOp>(loc, resultShapeType, ValueRange{complementShapeVal});
      auto complementStride = rewriter.create<MakeStrideOp>(loc, resultStrideType, ValueRange{totalStride});
      auto complementLayout = rewriter.create<MakeLayoutOp>(loc, resultLayoutType,
                                                             complementShape.getResult(),
                                                             complementStride.getResult());
      rewriter.replaceOp(op, complementLayout.getResult());
      return success();
    }
    
    // Rank-2+ case: implement fold algorithm
    // result_shape starts empty, result_stride starts with [1]
    SmallVector<Value> resultShapes;
    SmallVector<Value> resultStrides;
    resultStrides.push_back(one);
    
    SmallVector<Value> remainingShapes = filteredShapes;
    SmallVector<Value> remainingStrides = filteredStrides;
    
    // Fold: at each iteration, find min stride, remove it, compute new mode
    for (size_t iter = 0; iter < filteredStrides.size() - 1; ++iter) {
      // Find minimum stride (simplified: assume sorted or use comparison chain)
      // For now, just process in order (assumes already sorted)
      size_t minIdx = 0;
      Value minStride = remainingStrides[0];
      
      // new_shape = min_stride / result_stride[iter]
      auto newShape = rewriter.create<arith::DivUIOp>(loc, minStride, resultStrides[iter]);
      // new_stride = min_stride * remaining_shapes[minIdx]
      auto newStride = rewriter.create<arith::MulIOp>(loc, minStride, remainingShapes[minIdx]);
      
      resultShapes.push_back(newShape);
      resultStrides.push_back(newStride);
      
      // Remove minIdx from remaining
      remainingShapes.erase(remainingShapes.begin() + minIdx);
      remainingStrides.erase(remainingStrides.begin() + minIdx);
    }
    
    // Append last mode
    auto lastNewShape = rewriter.create<arith::DivUIOp>(loc, remainingStrides[0], 
                                                          resultStrides[resultStrides.size() - 1]);
    auto lastNewStride = rewriter.create<arith::MulIOp>(loc, remainingStrides[0], remainingShapes[0]);
    resultShapes.push_back(lastNewShape);
    
    // Compute rest_shape = ceil_div(target_size, lastNewStride)
    auto restShape = rewriter.create<arith::CeilDivUIOp>(loc, targetSize, lastNewStride);
    resultShapes.push_back(restShape);
    resultStrides.push_back(lastNewStride);
    
    // Create result layout
    auto resultShapeType = ShapeType::get(rewriter.getContext(), resultShapes.size());
    auto resultStrideType = StrideType::get(rewriter.getContext(), resultStrides.size());
    auto resultLayoutType = LayoutType::get(rewriter.getContext(), resultShapes.size());
    
    auto complementShape = rewriter.create<MakeShapeOp>(loc, resultShapeType, resultShapes);
    auto complementStride = rewriter.create<MakeStrideOp>(loc, resultStrideType, resultStrides);
    auto complementLayout = rewriter.create<MakeLayoutOp>(loc, resultLayoutType,
                                                           complementShape.getResult(),
                                                           complementStride.getResult());
    
    // Apply coalesce to the result
    auto coalescedLayout = rewriter.create<CoalesceOp>(loc, op.getResult().getType(), 
                                                        complementLayout.getResult());
    
    rewriter.replaceOp(op, coalescedLayout.getResult());
    return success();
  }
};

struct MakeShapeOpLowering : public OpRewritePattern<MakeShapeOp> {
  using OpRewritePattern<MakeShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeShapeOp op,
                                PatternRewriter &rewriter) const override {
    // If the result is unused, erase the op
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op.getOperation());
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
      rewriter.eraseOp(op.getOperation());
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
      rewriter.eraseOp(op.getOperation());
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
    patterns.add<CoalesceOpLowering>(&getContext());
    patterns.add<ComplementOpLowering>(&getContext());
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
    

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
