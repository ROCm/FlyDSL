#include "flir/FlirDialect.h"
#include "flir/FlirOps.h"
#include "flir/FlirPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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
using namespace mlir::flir;

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

// Try to extract an index constant from a Value.
static std::optional<int64_t> tryGetConstIndex(Value v) {
  if (!v)
    return std::nullopt;
  if (auto constIdx = v.getDefiningOp<arith::ConstantIndexOp>())
    return constIdx.value();
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (llvm::isa<IndexType>(cst.getType())) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        return intAttr.getInt();
    }
  }
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return tryGetConstIndex(cast.getIn());
  return std::nullopt;
}

// Helper to deserialize from MakeShape/MakeStride ops
static LayoutNode deserializeLayoutNode(Operation* op, PatternRewriter& rewriter, Location loc) {
    SmallVector<Value> values;

    if (isa<MakeShapeOp>(op)) {
        auto makeShape = cast<MakeShapeOp>(op);
        values = makeShape.getValues();
    } else if (isa<MakeStrideOp>(op)) {
        auto makeStride = cast<MakeStrideOp>(op);
        values = makeStride.getValues();
    } else {
        if (!op) return LayoutNode(Value());
        return LayoutNode(op->getResult(0));
    }

    // Type-mode:
    // - Type carries a pattern Attribute tree.
    // - Dynamic leaves are DyncI32/DyncI64 with dyncElemIdx indexing into operands.
    auto tryTypeMode = [&]() -> std::optional<LayoutNode> {
      if (!op || op->getNumResults() == 0)
        return std::nullopt;

      Type ty = op->getResult(0).getType();
      Attribute pat;
      if (auto st = dyn_cast<ShapeType>(ty)) {
        pat = st.getPattern();
      } else if (auto st = dyn_cast<StrideType>(ty)) {
        pat = st.getPattern();
      } else {
        return std::nullopt;
      }
      if (!pat)
        return std::nullopt;

      std::function<LayoutNode(Attribute)> parse = [&](Attribute a) -> LayoutNode {
        if (auto arr = dyn_cast<ArrayAttr>(a)) {
          std::vector<LayoutNode> kids;
          kids.reserve(arr.size());
          for (auto e : arr.getValue())
            kids.push_back(parse(e));
          return LayoutNode(kids);
        }
        if (auto i = dyn_cast<IntegerAttr>(a)) {
          Value c = rewriter.create<arith::ConstantIndexOp>(loc, i.getInt()).getResult();
          return LayoutNode(c);
        }
        if (auto d64 = dyn_cast<DyncI64Attr>(a)) {
          int32_t idx = d64.getDyncElemIdx();
          if (idx < 0 || idx >= static_cast<int32_t>(values.size()))
            return LayoutNode(Value());
          return LayoutNode(values[idx]);
        }
        if (auto d32 = dyn_cast<DyncI32Attr>(a)) {
          int32_t idx = d32.getDyncElemIdx();
          if (idx < 0 || idx >= static_cast<int32_t>(values.size()))
            return LayoutNode(Value());
          return LayoutNode(values[idx]);
        }
        // Underscore or unknown => reject (not expected in Shape/Stride here).
        return LayoutNode(Value());
      };

      return parse(pat);
    };

    if (auto n = tryTypeMode())
      return *n;

    // Not type-mode => reject.
    return LayoutNode(Value());
}

// Forward declaration
static void flatten_to_leaves(const LayoutNode& node, std::vector<LayoutNode>& leaves);

// Build type-mode MakeShapeOp from a LayoutNode:
// - Type carries a pattern Attribute tree (IntegerAttr / ArrayAttr / DyncI64Attr).
// - Operands contain only dynamic leaves, indexed by dyncElemIdx.
static Value serializeLayoutNodeToShape(const LayoutNode& node, Location loc, 
                                        PatternRewriter& rewriter, MLIRContext* ctx) {
    SmallVector<Value> dynOperands;
    int32_t dyncIdx = 0;
    Builder b(ctx);

    std::function<Attribute(const LayoutNode&)> buildPattern = [&](const LayoutNode &n) -> Attribute {
      if (n.isLeaf) {
        if (auto c = tryGetConstIndex(n.value)) {
          return IntegerAttr::get(b.getIntegerType(64),
                                  APInt(64, static_cast<uint64_t>(*c), true));
        }
        int32_t idx = dyncIdx++;
        dynOperands.push_back(n.value);
        return DyncI64Attr::get(ctx, idx, /*divisibility=*/1);
      }
      SmallVector<Attribute, 8> elems;
      elems.reserve(n.children.size());
      for (auto &c : n.children)
        elems.push_back(buildPattern(c));
      return ArrayAttr::get(ctx, elems);
    };

    Attribute pattern = buildPattern(node);
    auto shapeType = ShapeType::get(ctx, pattern);
    return rewriter.create<MakeShapeOp>(loc, shapeType, dynOperands).getResult();
}

static Value serializeLayoutNodeToStride(const LayoutNode& node, Location loc, 
                                         PatternRewriter& rewriter, MLIRContext* ctx) {
    SmallVector<Value> dynOperands;
    int32_t dyncIdx = 0;
    Builder b(ctx);

    std::function<Attribute(const LayoutNode&)> buildPattern = [&](const LayoutNode &n) -> Attribute {
      if (n.isLeaf) {
        if (auto c = tryGetConstIndex(n.value)) {
          return IntegerAttr::get(b.getIntegerType(64),
                                  APInt(64, static_cast<uint64_t>(*c), true));
        }
        int32_t idx = dyncIdx++;
        dynOperands.push_back(n.value);
        return DyncI64Attr::get(ctx, idx, /*divisibility=*/1);
      }
      SmallVector<Attribute, 8> elems;
      elems.reserve(n.children.size());
      for (auto &c : n.children)
        elems.push_back(buildPattern(c));
      return ArrayAttr::get(ctx, elems);
    };

    Attribute pattern = buildPattern(node);
    auto strideType = StrideType::get(ctx, pattern);
    return rewriter.create<MakeStrideOp>(loc, strideType, dynOperands).getResult();

    // unreachable
}

// Build a flat (rank-N) type-mode shape/stride from leaf values.
static Value makeFlatShapeTypeMode(ValueRange leafValues, Location loc,
                                   PatternRewriter &rewriter, MLIRContext *ctx) {
  Builder b(ctx);
  SmallVector<Value> dynOperands;
  int32_t dyncIdx = 0;

  SmallVector<Attribute, 8> elems;
  elems.reserve(leafValues.size());
  for (auto v : leafValues) {
    if (auto c = tryGetConstIndex(v)) {
      elems.push_back(IntegerAttr::get(b.getIntegerType(64),
                                       APInt(64, static_cast<uint64_t>(*c), true)));
    } else {
      int32_t idx = dyncIdx++;
      dynOperands.push_back(v);
      elems.push_back(DyncI64Attr::get(ctx, idx, /*divisibility=*/1));
    }
  }

  Attribute pat;
  if (elems.size() == 1)
    pat = elems.front();
  else
    pat = ArrayAttr::get(ctx, elems);

  auto shapeType = ShapeType::get(ctx, pat);
  return rewriter.create<MakeShapeOp>(loc, shapeType, dynOperands).getResult();
}

static Value makeFlatStrideTypeMode(ValueRange leafValues, Location loc,
                                    PatternRewriter &rewriter, MLIRContext *ctx) {
  Builder b(ctx);
  SmallVector<Value> dynOperands;
  int32_t dyncIdx = 0;

  SmallVector<Attribute, 8> elems;
  elems.reserve(leafValues.size());
  for (auto v : leafValues) {
    if (auto c = tryGetConstIndex(v)) {
      elems.push_back(IntegerAttr::get(b.getIntegerType(64),
                                       APInt(64, static_cast<uint64_t>(*c), true)));
    } else {
      int32_t idx = dyncIdx++;
      dynOperands.push_back(v);
      elems.push_back(DyncI64Attr::get(ctx, idx, /*divisibility=*/1));
    }
  }

  Attribute pat;
  if (elems.size() == 1)
    pat = elems.front();
  else
    pat = ArrayAttr::get(ctx, elems);

  auto strideType = StrideType::get(ctx, pat);
  return rewriter.create<MakeStrideOp>(loc, strideType, dynOperands).getResult();
}

/// Compute a compact LayoutLeft (col-major) stride from a type-mode ShapeType,
/// entirely at compile time.
///
/// @pre All shape leaves must be compile-time constants (no dynamic `?` leaves).
///      This is required so the lowering does not introduce runtime arithmetic
///      for stride construction.
/// @note We mirror the common compact-stride convention where a size-1 mode
///       yields a stride-0 leaf (enables filtering/coalescing behavior).
static Value makeLayoutLeftStrideFromStaticShapeTypeMode(Value shapeVal,
                                                         Location loc,
                                                         PatternRewriter &rewriter,
                                                         MLIRContext *ctx) {
  auto shapeTy = llvm::dyn_cast<ShapeType>(shapeVal.getType());
  if (!shapeTy) {
    emitError(loc, "Expected flir.shape type for tiler shape.");
    return Value();
  }

  Attribute shapePat = shapeTy.getPattern();
  if (!shapePat) {
    emitError(loc, "Expected flir.shape type with pattern.");
    return Value();
  }

  SmallVector<Value> dynLeafOperands;
  if (auto makeShape = shapeVal.getDefiningOp<MakeShapeOp>()) {
    dynLeafOperands.append(makeShape.getValues().begin(), makeShape.getValues().end());
  }

  // Flatten leaf sizes (must become compile-time constants).
  SmallVector<int64_t> resolvedShapeDims;
  auto fail = [&](llvm::StringRef msg) -> Value {
    emitError(loc, msg);
    return Value();
  };

  std::function<void(Attribute)> collect = [&](Attribute a) {
    if (auto arr = dyn_cast<ArrayAttr>(a)) {
      for (auto e : arr.getValue())
        collect(e);
      return;
    }
    if (auto i = dyn_cast<IntegerAttr>(a)) {
      resolvedShapeDims.push_back(i.getInt());
      return;
    }
    if (auto d64 = dyn_cast<DyncI64Attr>(a)) {
      int32_t idx = d64.getDyncElemIdx();
      if (idx < 0 || idx >= static_cast<int32_t>(dynLeafOperands.size())) {
        (void)fail("Malformed flir.make_shape: operand count mismatch.");
        return;
      }
      auto c = tryGetConstIndex(dynLeafOperands[idx]);
      if (!c.has_value()) {
        (void)fail("local_tile requires a tiler shape whose leaves fold to compile-time constants.");
        return;
      }
      resolvedShapeDims.push_back(*c);
      return;
    }
    if (auto d32 = dyn_cast<DyncI32Attr>(a)) {
      int32_t idx = d32.getDyncElemIdx();
      if (idx < 0 || idx >= static_cast<int32_t>(dynLeafOperands.size())) {
        (void)fail("Malformed flir.make_shape: operand count mismatch.");
        return;
      }
      auto c = tryGetConstIndex(dynLeafOperands[idx]);
      if (!c.has_value()) {
        (void)fail("local_tile requires a tiler shape whose leaves fold to compile-time constants.");
        return;
      }
      resolvedShapeDims.push_back(*c);
      return;
    }
    (void)fail("local_tile requires a tiler shape whose leaves are compile-time constants.");
  };
  collect(shapePat);
  if (resolvedShapeDims.empty()) {
    emitError(loc, "Malformed flir.shape pattern for local_tile.");
    return Value();
  }

  SmallVector<int64_t> strideDims;
  strideDims.reserve(resolvedShapeDims.size());

  int64_t current = 1;
  for (int64_t s : resolvedShapeDims) {
    if (s == 1) {
      strideDims.push_back(0);
      // current unchanged
    } else {
      strideDims.push_back(current);
      current *= s;
    }
  }

  // Rebuild stride pattern preserving the tuple structure of shapePat.
  Builder b(ctx);
  int64_t leafIdx = 0;
  std::function<Attribute(Attribute)> rebuild = [&](Attribute a) -> Attribute {
    if (auto arr = dyn_cast<ArrayAttr>(a)) {
      SmallVector<Attribute, 8> elems;
      elems.reserve(arr.size());
      for (auto e : arr.getValue())
        elems.push_back(rebuild(e));
      return ArrayAttr::get(ctx, elems);
    }
    // leaf
    int64_t v = strideDims[leafIdx++];
    return IntegerAttr::get(b.getIntegerType(64), APInt(64, static_cast<uint64_t>(v), true));
  };
  Attribute stridePat = rebuild(shapePat);
  auto strideTy = StrideType::get(ctx, stridePat);
  return rewriter.create<MakeStrideOp>(loc, strideTy, ValueRange{}).getResult();
}

// Helper to aggressively fold binary arithmetic operations on constants
static Value foldBinaryOp(Location loc, Value lhs, Value rhs, 
                          std::function<int64_t(int64_t, int64_t)> op,
                          std::function<Value(Location, Value, Value, PatternRewriter&)> createOp,
                          PatternRewriter& rewriter) {
    // Prefer the shared constant extractor so we fold both arith.constant and
    // arith.constant_index (and simple casts) uniformly.
    auto lhsC = tryGetConstIndex(lhs);
    auto rhsC = tryGetConstIndex(rhs);
    if (lhsC && rhsC) {
      int64_t result = op(*lhsC, *rhsC);
      return rewriter.create<arith::ConstantIndexOp>(loc, result).getResult();
    }

    return createOp(loc, lhs, rhs, rewriter);
}

// Ensure a denominator is non-zero before creating/folding divisions.
// This prevents compile-time constant-fold UB (division by zero) and inserts a
// runtime assertion for dynamic values.
static bool ensureNonZero(Location loc, Value denom, PatternRewriter &rewriter,
                          llvm::StringRef message) {
  auto c = tryGetConstIndex(denom);
  if (c.has_value() && *c == 0) {
    emitError(loc, message);
    return false;
  }
  if (!c.has_value()) {
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
    Value ok = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, denom, zero).getResult();
    std::string runtimeMsg = (llvm::Twine(message) + " (runtime).").str();
    rewriter.create<cf::AssertOp>(loc, ok, rewriter.getStringAttr(runtimeMsg));
  }
  return true;
}

static std::pair<Value, Value> emitFatalComplementError(Location loc, PatternRewriter &rewriter,
                                                        llvm::StringRef message) {
  // Compile-time failure (original behavior): emit an error diagnostic and
  // return null Values. The pass has a ScopedDiagnosticHandler and will
  // signalPassFailure() on any error diagnostic.
  emitError(loc, message);
  return {Value(), Value()};
}

// Helper to compute complement inline (without creating ComplementOp)
// Returns (complementShape, complementStride) as Values
//
// @pre The input tiler has been conceptually normalized for the complement
//      algorithm:
//      - We operate on a flattened (leaf) view of (shape, stride).
//      - For rank > 1 layouts, all stride leaves must be compile-time constants
//        (we reject dynamic-stride complements for rank > 1).
//      - Rank-1 stride-0 is handled as a special-case.
//      - Divisions require non-zero denominators; dynamic cases are guarded via
//        runtime asserts.
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
      auto *ctx = rewriter.getContext();
      Value complementShape = makeFlatShapeTypeMode(ValueRange{targetSize}, loc, rewriter, ctx);
      Value complementStride = makeFlatStrideTypeMode(ValueRange{one}, loc, rewriter, ctx);
      return {complementShape, complementStride};
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
          [](int64_t a, int64_t b) { return a / b; },
          [](Location l, Value a, Value b, PatternRewriter& r) {
              return r.create<arith::DivUIOp>(l, a, b).getResult();
          },
          rewriter);

      // Non-injective check for rank-1:
      // If stride==0 and shape>1, the layout overlaps (all indices map to 0).
      // In our algorithm, this manifests as gap(new_shape) == 0.
      if (auto c = tryGetConstIndex(newShape)) {
        if (*c == 0) {
          return emitFatalComplementError(loc, rewriter,
                                          "Non-injective Layout detected in complement.");
        }
      } else {
        // Runtime check for dynamic stride: gap must be non-zero.
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
        Value ok = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, newShape, zero).getResult();
        rewriter.create<cf::AssertOp>(
            loc, ok,
            rewriter.getStringAttr("Non-injective Layout detected in complement (runtime)."));
      }
      
      // new_stride = min_stride * modeShape
      auto newStride = foldBinaryOp(loc, minStride, modeShape,
          [](int64_t a, int64_t b) { return a * b; },
          [](Location l, Value a, Value b, PatternRewriter& r) {
              return r.create<arith::MulIOp>(l, a, b).getResult();
          },
          rewriter);

      // Denominator safety for ceil_div: newStride must be non-zero.
      // We assert here so we never constant-fold a division-by-zero.
      if (auto ns = tryGetConstIndex(newStride); ns.has_value() && *ns == 0) {
        return emitFatalComplementError(loc, rewriter, "Zero stride encountered in complement.");
      }
      if (!tryGetConstIndex(newStride).has_value()) {
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
        Value ok = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, newStride, zero).getResult();
        rewriter.create<cf::AssertOp>(
            loc, ok,
            rewriter.getStringAttr("Zero stride encountered in complement (runtime)."));
      }
      
      // rest_shape = ceil_div(target_size, new_stride)
      auto restShape = foldBinaryOp(loc, targetSize, newStride,
          [](int64_t a, int64_t b) { return (a + b - 1) / b; },
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
      auto *ctx = rewriter.getContext();
      Value complementShape = makeFlatShapeTypeMode(resultShapes, loc, rewriter, ctx);
      Value complementStride = makeFlatStrideTypeMode(resultStrides, loc, rewriter, ctx);
      return {complementShape, complementStride};
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

    // Complement requires a deterministic stride order for rank > 1.
    // If any stride leaf is dynamic, sorting/folding by stride becomes unsound.
    // In FLIR lowering, if rank > 1 and any stride is not a compile-time constant,
    // we cannot soundly sort/fold by stride, so we reject early.
    if (!allStatic && shapes.size() > 1) {
      return emitFatalComplementError(loc, rewriter,
                                      "Dynamic-stride complement only for rank-1 layouts.");
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
        
        if (!ensureNonZero(loc, currStride, rewriter,
                           "Zero stride encountered in complement.")) {
          return {Value(), Value()};
        }

        // new_shape = min_stride / last_result_stride
        auto newShape = foldBinaryOp(loc, minStride, currStride,
            [](int64_t a, int64_t b) { return a / b; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::DivUIOp>(l, a, b).getResult();
            },
            rewriter);
            
        // Check for Non-injective Layout (gap size == 0)
        if (auto constNewShape = tryGetConstIndex(newShape)) {
             if (constNewShape.value() == 0) {
                 return emitFatalComplementError(loc, rewriter,
                                                 "Non-injective Layout detected in complement.");
             }
        } else {
             auto minStrideC = tryGetConstIndex(minStride);
             auto currStrideC = tryGetConstIndex(currStride);
             if (minStrideC && currStrideC && *minStrideC < *currStrideC) {
                 return emitFatalComplementError(loc, rewriter,
                                                 "Non-injective Layout detected in complement.");
             }
             // Runtime check for dynamic cases: gap must be non-zero.
             // This makes the failure deterministic at runtime instead of silent UB.
             if (!(minStrideC && currStrideC)) {
                 Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
                 Value ok = rewriter.create<arith::CmpIOp>(
                     loc, arith::CmpIPredicate::ne, newShape, zero).getResult();
                 rewriter.create<cf::AssertOp>(
                     loc, ok,
                     rewriter.getStringAttr("Non-injective Layout detected in complement (runtime)."));
             }
        }

        
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
    if (!ensureNonZero(loc, currStride, rewriter,
                       "Zero stride encountered in complement.")) {
      return emitFatalComplementError(loc, rewriter, "Zero stride encountered in complement.");
    }
    auto lastNewShape = foldBinaryOp(loc, lastMinStride, currStride,
        [](int64_t a, int64_t b) { return a / b; },
        [](Location l, Value a, Value b, PatternRewriter& r) {
            return r.create<arith::DivUIOp>(l, a, b).getResult();
        },
        rewriter);

    // Check for Non-injective Layout (gap size == 0)
    if (auto constNewShape = tryGetConstIndex(lastNewShape)) {
            if (constNewShape.value() == 0) {
                return emitFatalComplementError(loc, rewriter,
                                                "Non-injective Layout detected in complement.");
            }
    } else {
            auto minStrideC = tryGetConstIndex(lastMinStride);
            auto currStrideC = tryGetConstIndex(currStride);
            if (minStrideC && currStrideC && *minStrideC < *currStrideC) {
                return emitFatalComplementError(loc, rewriter,
                                                "Non-injective Layout detected in complement.");
            }
            // Runtime check for dynamic cases: last gap must be non-zero.
            if (!(minStrideC && currStrideC)) {
                Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
                Value ok = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::ne, lastNewShape, zero).getResult();
                rewriter.create<cf::AssertOp>(
                    loc, ok,
                    rewriter.getStringAttr("Non-injective Layout detected in complement (runtime)."));
            }
    }
    
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
    if (!ensureNonZero(loc, currStride, rewriter,
                       "Zero stride encountered in complement.")) {
      return emitFatalComplementError(loc, rewriter, "Zero stride encountered in complement.");
    }
    auto restShape = foldBinaryOp(loc, targetSize, currStride,
        [](int64_t a, int64_t b) { return (a + b - 1) / b; },
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
    auto *ctx = rewriter.getContext();
    Value complementShape = makeFlatShapeTypeMode(compShapeVals, loc, rewriter, ctx);
    Value complementStride = makeFlatStrideTypeMode(compStrideVals, loc, rewriter, ctx);
    return {complementShape, complementStride};
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
[[maybe_unused]] static LayoutNode makeConstNode(int64_t val, Location loc, PatternRewriter& rewriter) {
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
        if (!ensureNonZero(loc, restStride, rewriter,
                           "Division by zero encountered in composition.")) {
          return {rhsShape, rhsStride};
        }
        Value nextShape = foldBinaryOp(loc, currShape, restStride,
            [](int64_t a, int64_t b) { return (a + b - 1) / b; },  // ceil_div
            [](Location l, Value a, Value b, PatternRewriter& r) { 
                return r.create<arith::CeilDivUIOp>(l, a, b).getResult(); 
            },
            rewriter);
        
        // Update restStride before checking for early exit.
        // Important: the fold returns next_stride even when breaking early.
        if (!ensureNonZero(loc, currShape, rewriter,
                           "Division by zero encountered in composition.")) {
          return {rhsShape, rhsStride};
        }
        Value nextStride = foldBinaryOp(loc, restStride, currShape,
            [](int64_t a, int64_t b) { return (a + b - 1) / b; },
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
        if (!ensureNonZero(loc, newShape, rewriter,
                           "Division by zero encountered in composition.")) {
          return {rhsShape, rhsStride};
        }
        restShape = foldBinaryOp(loc, restShape, newShape,
            [](int64_t a, int64_t b) { return a / b; },
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
//
// @pre The input layout has been conceptually normalized for the complement
//      algorithm:
//      - We operate on a flattened (leaf) view of (shape, stride).
//      - For rank > 1 layouts, all stride leaves must be compile-time constants
//        (we reject dynamic-stride complements for rank > 1).
//      - Divisions require non-zero denominators; dynamic cases are guarded via
//        runtime asserts.
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

    // Complement requires a deterministic stride order for rank > 1.
    // If any stride leaf is dynamic, sorting/folding by stride becomes unsound.
    // Reject rank>1 dynamic-stride complements in FLIR lowering.
    if (!allStatic && leavesShape.size() > 1) {
      emitError(loc, "Dynamic-stride complement only for rank-1 layouts.");
      return {LayoutNode(Value()), LayoutNode(Value())};
    }
    
    if (allStatic) {
        std::sort(modes.begin(), modes.end(), [](const Mode& a, const Mode& b) {
            return a.constStride < b.constStride;
        });
    }
    // If not all static, rank must be 1 (checked above), so sorting is unnecessary.
    
    // 3. Fold logic: accumulate gaps before each stride milestone.
    std::vector<Value> resShape;
    std::vector<Value> resStride;
    Value currStride = rewriter.create<arith::ConstantIndexOp>(loc, 1); // starts at 1
    std::optional<int64_t> currStrideC = 1; // Compile-time tracking

    resStride.push_back(currStride);
    
    // Iterate through sorted modes
    for (const auto& mode : modes) {
        Value minStride = mode.stride;
        Value modeShape = mode.shape;
        auto minStrideC = tryGetConstIndex(minStride);
        auto modeShapeC = tryGetConstIndex(modeShape);
        
        // Gap before the current mode.
        if (!ensureNonZero(loc, currStride, rewriter,
                           "Zero stride encountered in complement.")) {
          return {LayoutNode(Value()), LayoutNode(Value())};
        }
        Value newShape = foldBinaryOp(loc, minStride, currStride,
            [](int64_t a, int64_t b) { return a / b; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::DivUIOp>(l, a, b).getResult();
            },
            rewriter);

        // Check for non-injective layouts.
        // If the gap size is 0, it means minStride < currStride, implying overlap.
        // We use our explicit compile-time tracking (currStrideC) for robust checking.

        // Compile-time check (best effort).
        if (auto constNewShape = tryGetConstIndex(newShape); constNewShape.has_value()) {
            if (constNewShape.value() == 0) {
                emitError(loc, "Non-injective Layout detected in complement.");
                return {LayoutNode(Value()), LayoutNode(Value())};
            }
        }

        if (minStrideC.has_value() && currStrideC.has_value()) {
            if (*minStrideC < *currStrideC) {
                 emitError(loc, "Non-injective Layout detected in complement.");
                 return {LayoutNode(Value()), LayoutNode(Value())};
            }
        } else {
            // Runtime check for dynamic cases: gap must be non-zero.
            Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
            Value ok = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ne, newShape, zero).getResult();
            rewriter.create<cf::AssertOp>(
                loc, ok,
                rewriter.getStringAttr("Non-injective Layout detected in complement (runtime)."));
        }
        
        resShape.push_back(newShape);
        Value nextStride = foldBinaryOp(loc, minStride, modeShape,
            [](int64_t a, int64_t b) { return a * b; },
            [](Location l, Value a, Value b, PatternRewriter& r) {
                return r.create<arith::MulIOp>(l, a, b).getResult();
            },
            rewriter);
        
        // Advance coverage to include this mode.
        currStride = nextStride;
        
        // Update compile-time tracking
        if (minStrideC.has_value() && modeShapeC.has_value()) {
            currStrideC = (*minStrideC) * (*modeShapeC);
        } else {
            currStrideC = std::nullopt;
        }
    }
    
    // 4. Append the final mode that stretches coverage to cosizeHi.
    if (!ensureNonZero(loc, currStride, rewriter,
                       "Zero stride encountered in complement.")) {
      return {LayoutNode(Value()), LayoutNode(Value())};
    }
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
    if (!ensureNonZero(loc, currStride, rewriter,
                       "Zero stride encountered in complement.")) {
      return {LayoutNode(Value()), LayoutNode(Value())};
    }
    Value finalRest = ceilDiv(loc, cosizeHi, currStride, rewriter);
    compShapeNodes.push_back(LayoutNode(finalRest));
    compStrideNodes.push_back(LayoutNode(currStride));
    
    // Coalesce the result (optional but good)
    // We return as a tuple of these modes.
    return {LayoutNode(compShapeNodes), LayoutNode(compStrideNodes)};
}

// logical_divide(layout, tiler)
[[maybe_unused]] static std::pair<LayoutNode, LayoutNode> logical_divide_impl(
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
  llvm_unreachable("flir.rank input must be a flir ranked type");
}

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

// Helper to compute size of a Value (index or shape/tuple) recursively
static Value computeRecursiveSize(Location loc, Value val, PatternRewriter &rewriter) {
  if (val.getType().isIndex())
    return val;
    
  if (auto *defOp = val.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "flir.make_shape") {
       // Type-mode: constants are encoded in the shape type, operands are dynamic leaves only.
       LayoutNode shapeNode = deserializeLayoutNode(defOp, rewriter, loc);
       std::vector<LayoutNode> leaves;
       flatten_to_leaves(shapeNode, leaves);
       Value product = rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
       for (auto &leaf : leaves) {
         if (!leaf.value || !leaf.value.getType().isIndex())
           return rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
         product = rewriter.create<arith::MulIOp>(loc, product, leaf.value).getResult();
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
      : RewritePattern("flir.size", 1, ctx) {}

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
    if (opName == "flir.make_shape") {
      // Use recursive helper to handle nested shapes
      rewriter.replaceOp(op, computeRecursiveSize(loc, input, rewriter));
      return success();
    }
    
    // Handle make_layout - get shape then compute size
    if (opName == "flir.make_layout") {
      auto shape = defOp->getOperand(0);
      // Use recursive helper on the shape
      rewriter.replaceOp(op, computeRecursiveSize(loc, shape, rewriter));
      return success();
    }
    
    // Handle product operations: size(product(A, B)) = size(A) * size(B)
    if (opName == "flir.logical_product" || opName == "flir.zipped_product" ||
        opName == "flir.tiled_product" || opName == "flir.flat_product" ||
        opName == "flir.raked_product" || opName == "flir.blocked_product") {
      
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
      : RewritePattern("flir.cosize", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value layout = op->getOperand(0);
    
    auto *layoutOp = layout.getDefiningOp();
    if (!layoutOp || layoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
    
    auto *shapeOp = layoutOp->getOperand(0).getDefiningOp();
    auto *strideOp = layoutOp->getOperand(1).getDefiningOp();
    
    if (!shapeOp || !strideOp)
      return failure();
    if (shapeOp->getName().getStringRef() != "flir.make_shape" ||
        strideOp->getName().getStringRef() != "flir.make_stride")
      return failure();
    
    // Flatten shapes and strides to leaves to handle nested structures
    LayoutNode shapeNode = deserializeLayoutNode(shapeOp, rewriter, loc);
    LayoutNode strideNode = deserializeLayoutNode(strideOp, rewriter, loc);
    
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
      : RewritePattern("flir.get", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    if (op->getNumOperands() != 2)
      return failure();
    
    Value input = op->getOperand(0);
    Value idx = op->getOperand(1);
    
    auto *defOp = input.getDefiningOp();
    if (!defOp)
      return failure();
    
    auto opName = defOp->getName().getStringRef();
    if (opName != "flir.make_shape" && opName != "flir.make_stride" && 
        opName != "flir.make_coord")
      return failure();

    // Type-mode-aware flattening:
    // - For shape/stride: constants may live in the result type (structure+dims).
    // - For coord: coordinates are represented as a flat list of index operands.
    SmallVector<Value> flatValues;
    if (opName == "flir.make_shape" || opName == "flir.make_stride") {
      LayoutNode node = deserializeLayoutNode(defOp, rewriter, loc);
      std::vector<LayoutNode> leaves;
      flatten_to_leaves(node, leaves);
      flatValues.reserve(leaves.size());
      for (auto &leaf : leaves)
        flatValues.push_back(leaf.value);
    } else {
      // make_coord operands are the coordinate leaf values.
      for (auto v : defOp->getOperands())
        flatValues.push_back(v);
    }
    
    // If idx is constant, extract from flattened values.
    // Accept common constant forms:
    // - arith.constant : index
    // - arith.constant_index (C++ wrapper)
    // - arith.index_cast of a constant integer.
    std::function<std::optional<int64_t>(Value)> getConstIndex =
        [&](Value v) -> std::optional<int64_t> {
      return tryGetConstIndex(v);
    };

    if (auto idxValOpt = getConstIndex(idx)) {
      int64_t idxVal = *idxValOpt;
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

// Traverse a LayoutNode and apply a callback to each leaf in left-to-right order.
template <typename F>
static LogicalResult forEachLeafInOrder(const LayoutNode &n, F &&f) {
  if (n.isLeaf) {
    return f(n.value);
  }
  for (auto &c : n.children) {
    if (failed(forEachLeafInOrder(c, f)))
      return failure();
  }
  return success();
}

// Traverse a LayoutNode in reverse leaf order (right-to-left) and apply a callback.
template <typename F>
static LogicalResult forEachLeafInReverseOrder(const LayoutNode &n, F &&f) {
  if (n.isLeaf) {
    return f(n.value);
  }
  for (auto it = n.children.rbegin(); it != n.children.rend(); ++it) {
    if (failed(forEachLeafInReverseOrder(*it, f)))
      return failure();
  }
  return success();
}

static int64_t countLeaves(const LayoutNode &n) {
  if (n.isLeaf)
    return 1;
  int64_t c = 0;
  for (auto &x : n.children)
    c += countLeaves(x);
  return c;
}

// Lower rank to a constant.
struct RankOpLowering : public RewritePattern {
  RankOpLowering(MLIRContext *ctx)
      : RewritePattern("flir.rank", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value input = op->getOperand(0);
    
    int rank = getRankFromType(input.getType());
    
    auto rankConst = rewriter.create<arith::ConstantIndexOp>(loc, rank);
    rewriter.replaceOp(op, rankConst.getResult());
    return success();
  }
};

// Lower crd2idx to arithmetic: sum(coord[i] * stride[i]).
struct Crd2IdxOpLowering : public RewritePattern {
  Crd2IdxOpLowering(MLIRContext *ctx)
      : RewritePattern("flir.crd2idx", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    if (op->getNumOperands() != 2)
      return failure();
      
    Value coord = op->getOperand(0);
    Value layout = op->getOperand(1);
    
    auto *coordOp = coord.getDefiningOp();
    auto *layoutOp = layout.getDefiningOp();
    
    if (!coordOp || coordOp->getName().getStringRef() != "flir.make_coord")
      return failure();
    if (!layoutOp || layoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
    
    if (layoutOp->getNumOperands() < 2)
      return failure();
      
    auto *strideOp = layoutOp->getOperand(1).getDefiningOp();
    if (!strideOp || strideOp->getName().getStringRef() != "flir.make_stride")
      return failure();
    
    // Deserialize stride into a nested tree and traverse leaves without flattening into a list.
    LayoutNode strideNode = deserializeLayoutNode(strideOp, rewriter, loc);
    int64_t strideLeaves = countLeaves(strideNode);
    // Coords are represented as a flat list of index operands. No nested coord SSA trees.
    auto coordVals = coordOp->getOperands();
    if ((int64_t)coordVals.size() != strideLeaves)
      return failure();

    // Compute sum(coord_leaf[i] * stride_leaf[i]) in leaf order.
    Value acc = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
    int64_t coordIdx = 0;
    auto walkResult = forEachLeafInOrder(strideNode, [&](Value strideLeaf) -> LogicalResult {
      if (!strideLeaf || !strideLeaf.getType().isIndex())
        return failure();
      Value coordLeaf = coordVals[coordIdx++];
      if (!coordLeaf || !coordLeaf.getType().isIndex())
        return failure();
      auto prod = rewriter.create<arith::MulIOp>(loc, coordLeaf, strideLeaf).getResult();
      acc = rewriter.create<arith::AddIOp>(loc, acc, prod).getResult();
      return success();
    });
    if (failed(walkResult))
      return failure();

    rewriter.replaceOp(op, acc);
    return success();
  }
};

// Lower idx2crd to compute the multi-dim coordinate from a linear index.
struct Idx2CrdOpLowering : public RewritePattern {
  Idx2CrdOpLowering(MLIRContext *ctx)
      : RewritePattern("flir.idx2crd", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value idx = op->getOperand(0);
    Value layout = op->getOperand(1);
    
    auto *layoutOp = layout.getDefiningOp();
    if (!layoutOp || layoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
    
    auto *shapeOp = layoutOp->getOperand(0).getDefiningOp();
    if (!shapeOp || shapeOp->getName().getStringRef() != "flir.make_shape")
      return failure();
    
    // Deserialize shape into a nested tree and traverse leaves in reverse order.
    LayoutNode shapeNode = deserializeLayoutNode(shapeOp, rewriter, loc);
    int64_t totalLeaves = countLeaves(shapeNode);
    if (totalLeaves <= 0)
      return failure();

    // Ensure result coord type rank matches leaf count when available.
    if (auto ct = dyn_cast<CoordType>(op->getResult(0).getType())) {
      if (ct.getRank() >= 0 && ct.getRank() != (int)totalLeaves)
        return failure();
    }

    // Compute row-major coordinates over the leaf dimensions in left-to-right order,
    // without flattening the shape tree into a list.
    //
    // Process leaves in reverse order:
    //   for last..1: coord = remaining % dim; remaining = remaining / dim
    //   for first:   coord = remaining
    SmallVector<Value> coordsReversed;
    coordsReversed.reserve(totalLeaves);
    Value remaining = idx;
    int64_t processed = 0;

    auto walkResult = forEachLeafInReverseOrder(shapeNode, [&](Value dim) -> LogicalResult {
      if (!dim || !dim.getType().isIndex())
        return failure();

      if (processed == totalLeaves - 1) {
        // First dimension (in forward order): coord = remaining.
        coordsReversed.push_back(remaining);
      } else {
        auto coord = rewriter.create<arith::RemSIOp>(loc, remaining, dim).getResult();
        coordsReversed.push_back(coord);
        remaining = rewriter.create<arith::DivSIOp>(loc, remaining, dim).getResult();
      }
      ++processed;
      return success();
    });
    if (failed(walkResult))
      return failure();
    if (processed != totalLeaves)
      return failure();

    SmallVector<Value> coords;
    coords.reserve(totalLeaves);
    for (auto it = coordsReversed.rbegin(); it != coordsReversed.rend(); ++it)
      coords.push_back(*it);
    // Emit a single make_coord with flat index operands.
    Type coordType = op->getResult(0).getType();
    Value coordVal = rewriter.create<MakeCoordOp>(loc, coordType, coords).getResult();
    rewriter.replaceOp(op, coordVal);
    return success();
  }
};

// Lower swizzle_xor16 to arithmetic:
//   row_mod = row % kBlocks16
//   mask    = row_mod * 16
//   out     = col xor mask
struct SwizzleXor16OpLowering : public RewritePattern {
  SwizzleXor16OpLowering(MLIRContext *ctx)
      : RewritePattern("flir.swizzle_xor16", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    if (op->getNumOperands() != 3)
      return failure();
    Value row = op->getOperand(0);
    Value col = op->getOperand(1);
    Value kBlocks16 = op->getOperand(2);
    if (!row.getType().isIndex() || !col.getType().isIndex() ||
        !kBlocks16.getType().isIndex())
      return failure();

    // CK-style fast path for power-of-two kBlocks16:
    //   row_mod = row & (kBlocks16-1)
    //   mask    = row_mod << 4   // 16 = 2^4
    //   out     = col xor mask
    //
    // Fallback:
    //   row_mod = row remui kBlocks16
    //   mask    = row_mod * 16
    //   out     = col xor mask
    Value rowMod;
    if (auto kConstOpt = tryGetConstIndex(kBlocks16)) {
      int64_t k = *kConstOpt;
      if (k > 0 && (k & (k - 1)) == 0) {
        auto km1 = rewriter.create<arith::ConstantIndexOp>(loc, k - 1).getResult();
        rowMod = rewriter.create<arith::AndIOp>(loc, row, km1).getResult();
      }
    }
    if (!rowMod)
      rowMod = rewriter.create<arith::RemUIOp>(loc, row, kBlocks16).getResult();

    Value mask;
    // Prefer shift when kBlocks16 was constant power-of-two (or if we just want bit semantics).
    // Shift amount is always 4 for the 16B granularity swizzle (fp8 element indexing).
    if (rowMod.getDefiningOp() && rowMod.getDefiningOp()->getName().getStringRef() == "arith.andi") {
      auto sh4 = rewriter.create<arith::ConstantIndexOp>(loc, 4).getResult();
      mask = rewriter.create<arith::ShLIOp>(loc, rowMod, sh4).getResult();
    } else {
      auto c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16).getResult();
      mask = rewriter.create<arith::MulIOp>(loc, rowMod, c16).getResult();
    }

    auto out = rewriter.create<arith::XOrIOp>(loc, col, mask).getResult();
    rewriter.replaceOp(op, out);
    return success();
  }
};

// Lower get_shape to extract the shape from a layout.
struct GetShapeOpLowering : public RewritePattern {
  GetShapeOpLowering(MLIRContext *ctx)
      : RewritePattern("flir.get_shape", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value layout = op->getOperand(0);
    auto *layoutOp = layout.getDefiningOp();
    
    if (!layoutOp || layoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
    
    // Simply forward the shape operand
    rewriter.replaceOp(op, layoutOp->getOperand(0));
    return success();
  }
};

// Lower get_stride to extract the stride from a layout.
struct GetStrideOpLowering : public RewritePattern {
  GetStrideOpLowering(MLIRContext *ctx)
      : RewritePattern("flir.get_stride", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value layout = op->getOperand(0);
    auto *layoutOp = layout.getDefiningOp();
    
    if (!layoutOp || layoutOp->getName().getStringRef() != "flir.make_layout")
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
    if (!layoutOp || layoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
    
    auto shape = layoutOp->getOperand(0);
    auto stride = layoutOp->getOperand(1);
    
    auto *shapeOp = shape.getDefiningOp();
    auto *strideOp = stride.getDefiningOp();
    
    if (!shapeOp || !strideOp ||
        shapeOp->getName().getStringRef() != "flir.make_shape" ||
        strideOp->getName().getStringRef() != "flir.make_stride")
      return failure();
    
    // Collect leaf index values without flattening into a list first:
    // We deserialize to a nested tree and then iterate leaves in order.
    LayoutNode shapeNode = deserializeLayoutNode(shapeOp, rewriter, loc);
    LayoutNode strideNode = deserializeLayoutNode(strideOp, rewriter, loc);

    SmallVector<Value> shapeDims;
    SmallVector<Value> strideDims;
    shapeDims.reserve(countLeaves(shapeNode));
    strideDims.reserve(countLeaves(strideNode));

    if (failed(forEachLeafInOrder(shapeNode, [&](Value v) -> LogicalResult {
          if (!v || !v.getType().isIndex())
            return failure();
          shapeDims.push_back(v);
          return success();
        })))
      return failure();

    if (failed(forEachLeafInOrder(strideNode, [&](Value v) -> LogicalResult {
          if (!v || !v.getType().isIndex())
            return failure();
          strideDims.push_back(v);
          return success();
        })))
      return failure();
    
    if (shapeDims.size() != strideDims.size() || shapeDims.empty())
      return failure();

    // (types already checked during leaf collection)
    
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
    Value newShape = makeFlatShapeTypeMode(coalescedShape, loc, rewriter, ctx);
    Value newStride = makeFlatStrideTypeMode(coalescedStride, loc, rewriter, ctx);
    
    auto newLayout = rewriter.create<MakeLayoutOp>(
        loc, op.getResult().getType(), newShape, newStride);
    
    rewriter.replaceOp(op, newLayout.getResult());
    return success();
  }
};

struct CompositionOpLowering : public OpRewritePattern<CompositionOp> {
  using OpRewritePattern<CompositionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompositionOp op,
                                PatternRewriter &rewriter) const override {
    // Composition: R = A ◦ B means R(c) = A(B(c)).
    // NOTE: Keep this pattern side-effect-free unless we are sure we will rewrite,
    // otherwise the greedy driver may report failure if we create ops and then
    // return failure().
    
    auto loc = op->getLoc();
    Value layoutA = op.getLayoutA();
    Value layoutB = op.getLayoutB();
    
    // Require both operands to be direct make_layout for now.
    // (This matches the existing lowering strategy for other ops.)
    auto *layoutAOp = layoutA.getDefiningOp();
    auto *layoutBOp = layoutB.getDefiningOp();
    
    if (!layoutAOp || !layoutBOp)
      return failure();
    if (layoutAOp->getName().getStringRef() != "flir.make_layout" ||
        layoutBOp->getName().getStringRef() != "flir.make_layout")
      return failure();
    
    // Extract shape and stride operands
    auto shapeAOp = layoutAOp->getOperand(0).getDefiningOp();
    auto strideAOp = layoutAOp->getOperand(1).getDefiningOp();
    auto shapeBOp = layoutBOp->getOperand(0).getDefiningOp();
    auto strideBOp = layoutBOp->getOperand(1).getDefiningOp();
    
    // Deserialize to LayoutNodes
    LayoutNode shapeA = deserializeLayoutNode(shapeAOp, rewriter, loc);
    LayoutNode strideA = deserializeLayoutNode(strideAOp, rewriter, loc);
    LayoutNode shapeB = deserializeLayoutNode(shapeBOp, rewriter, loc);
    LayoutNode strideB = deserializeLayoutNode(strideBOp, rewriter, loc);
    
    // Compute composition recursively.
    auto [resShapeNode, resStrideNode] = composition_impl(shapeA, strideA, shapeB, strideB, loc, rewriter);
    
    // Serialize back to nested MakeShape/MakeStride ops
    // This preserves the nested structure from composition_impl
    auto ctx = rewriter.getContext();
    Value makeShape = serializeLayoutNodeToShape(resShapeNode, loc, rewriter, ctx);
    Value makeStride = serializeLayoutNodeToStride(resStrideNode, loc, rewriter, ctx);
    
    // Preserve the result type requested by the op (typically rank-only layout).
    auto makeLayout = rewriter.create<MakeLayoutOp>(
        loc, op.getResult().getType(), makeShape, makeStride);
    
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
        blockLayoutOp->getName().getStringRef() != "flir.make_layout" ||
        tilerLayoutOp->getName().getStringRef() != "flir.make_layout") {
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
    
    // Build product shape/stride from LayoutNodes so we support:
    // - legacy mode (shape dims in operands)
    LayoutNode blockShapeNode = deserializeLayoutNode(blockShape.getDefiningOp(), rewriter, loc);
    LayoutNode blockStrideNode = deserializeLayoutNode(blockStride.getDefiningOp(), rewriter, loc);
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp(), rewriter, loc);
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp(), rewriter, loc);

    std::vector<LayoutNode> blockShapeLeaves;
    std::vector<LayoutNode> blockStrideLeaves;
    std::vector<LayoutNode> tilerShapeLeaves;
    std::vector<LayoutNode> tilerStrideLeaves;
    flatten_to_leaves(blockShapeNode, blockShapeLeaves);
    flatten_to_leaves(blockStrideNode, blockStrideLeaves);
    flatten_to_leaves(tilerShapeNode, tilerShapeLeaves);
    flatten_to_leaves(tilerStrideNode, tilerStrideLeaves);

    SmallVector<Value> productShapeDims;
    productShapeDims.reserve(blockShapeLeaves.size() + tilerShapeLeaves.size());
    for (auto &x : blockShapeLeaves)
      productShapeDims.push_back(x.value);
    for (auto &x : tilerShapeLeaves)
      productShapeDims.push_back(x.value);
    
    // Determine the rank of the product
    size_t productRank = productShapeDims.size();
    
    auto *ctx = rewriter.getContext();
    // Create type-mode shape with combined dimensions
    Value productShape = makeFlatShapeTypeMode(productShapeDims, loc, rewriter, ctx);
    
    // For stride, we need to compute appropriate strides
    // For simplicity, create strides that maintain the product structure
    SmallVector<Value> productStrideDims;
    
    // Block strides: keep as-is (leaf order).
    productStrideDims.reserve(blockStrideLeaves.size() + tilerStrideLeaves.size());
    for (auto &x : blockStrideLeaves)
      productStrideDims.push_back(x.value);

    // Tiler strides: scale by block size.
    Value blockSize = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (auto &x : blockShapeLeaves) {
      blockSize = rewriter.create<arith::MulIOp>(loc, blockSize, x.value);
    }
    for (auto &x : tilerStrideLeaves) {
      auto scaledStride = rewriter.create<arith::MulIOp>(loc, x.value, blockSize);
      productStrideDims.push_back(scaledStride);
    }
    
    Value productStride = makeFlatStrideTypeMode(productStrideDims, loc, rewriter, ctx);
    
    // Create the product layout with the correct type
    auto productLayoutType = LayoutType::get(ctx, productRank);
    auto productLayout = rewriter.create<MakeLayoutOp>(
        loc, productLayoutType, productShape, 
        productStride);
    
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
    // In Flir, "Tuple of Layouts" is a Node whose children are Nodes (Shapes).
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
        
        std::vector<LayoutNode> resShapeNodes;
        std::vector<LayoutNode> resStrideNodes;
        
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
            
            std::vector<LayoutNode> tilerPartShapes;
            std::vector<LayoutNode> complementPartShapes;
            std::vector<LayoutNode> tilerPartStrides;
            std::vector<LayoutNode> complementPartStrides;

            for (auto res : subResults) {
                auto *resOp = res.getDefiningOp(); // MakeLayoutOp
                if (!resOp || resOp->getName().getStringRef() != "flir.make_layout")
                  return nullptr;
                auto *shapeOp = resOp->getOperand(0).getDefiningOp();
                auto *strideOp = resOp->getOperand(1).getDefiningOp();
                if (!shapeOp || !strideOp)
                  return nullptr;

                LayoutNode shapeNode = deserializeLayoutNode(shapeOp, rewriter, loc);
                LayoutNode strideNode = deserializeLayoutNode(strideOp, rewriter, loc);
                if (shapeNode.isLeaf || strideNode.isLeaf)
                  return nullptr;
                if (shapeNode.children.size() != 2 || strideNode.children.size() != 2)
                  return nullptr;

                tilerPartShapes.push_back(shapeNode.children[0]);
                complementPartShapes.push_back(shapeNode.children[1]);
                tilerPartStrides.push_back(strideNode.children[0]);
                complementPartStrides.push_back(strideNode.children[1]);
            }
            
            // Construct zipped result nodes: ((T0,T1,...),(C0,C1,...))
            LayoutNode tShapeNode(tilerPartShapes);
            LayoutNode cShapeNode(complementPartShapes);
            LayoutNode tStrideNode(tilerPartStrides);
            LayoutNode cStrideNode(complementPartStrides);

            resShapeNodes = {tShapeNode, cShapeNode};
            resStrideNodes = {tStrideNode, cStrideNode};

        } else {
            // Standard Concatenation (No Zip)
            for (auto res : subResults) {
                auto *subResultOp = res.getDefiningOp();
                if (!subResultOp || subResultOp->getName().getStringRef() != "flir.make_layout")
                  return nullptr;
                auto *shapeOp = subResultOp->getOperand(0).getDefiningOp();
                auto *strideOp = subResultOp->getOperand(1).getDefiningOp();
                if (!shapeOp || !strideOp)
                  return nullptr;
                resShapeNodes.push_back(deserializeLayoutNode(shapeOp, rewriter, loc));
                resStrideNodes.push_back(deserializeLayoutNode(strideOp, rewriter, loc));
            }
            
            // 2. Append remaining input children (rest of broadcast)
            for (size_t i = tilerShape.children.size(); i < inputShape.children.size(); ++i) {
                resShapeNodes.push_back(inputShape.children[i]);
                resStrideNodes.push_back(inputStride.children[i]);
            }
        }

        // Build output layout from nodes (type-mode, no nested SSA trees).
        LayoutNode outShapeNode(resShapeNodes);
        LayoutNode outStrideNode(resStrideNodes);
        Value makeShape = serializeLayoutNodeToShape(outShapeNode, loc, rewriter, ctx);
        Value makeStride = serializeLayoutNodeToStride(outStrideNode, loc, rewriter, ctx);
        auto outShapeTy = dyn_cast<ShapeType>(makeShape.getType());
        auto outStrideTy = dyn_cast<StrideType>(makeStride.getType());
        if (!outShapeTy || !outStrideTy)
          return nullptr;
        auto layoutType = LayoutType::get(ctx, outShapeTy, outStrideTy);
        return rewriter.create<MakeLayoutOp>(loc, layoutType, makeShape, makeStride).getResult();
    }
    
    // Base Case: Global Divide
    Value inputSize = computeSizeFromNode(inputShape, loc, rewriter);
    
    auto [complementShape, complementStride] = computeComplementInline(
        tilerShape, tilerStride, inputSize, loc, rewriter);
        
    if (!complementShape || !complementStride)
        return nullptr;
    
    // Build combined RHS (tiler, complement) purely as nodes to avoid nested SSA trees.
    auto *compShapeOp = complementShape.getDefiningOp();
    auto *compStrideOp = complementStride.getDefiningOp();
    if (!compShapeOp || !compStrideOp)
      return nullptr;
    LayoutNode compShapeNode = deserializeLayoutNode(compShapeOp, rewriter, loc);
    LayoutNode compStrideNode = deserializeLayoutNode(compStrideOp, rewriter, loc);
    LayoutNode combinedShapeNode(std::vector<LayoutNode>{tilerShape, compShapeNode});
    LayoutNode combinedStrideNode(std::vector<LayoutNode>{tilerStride, compStrideNode});
    
    auto [coalescedInputShape, coalescedInputStride] = coalesceLayoutNode(inputShape, inputStride, loc, rewriter);
    
    auto [resShapeNode, resStrideNode] = composition_impl(
        coalescedInputShape, coalescedInputStride, combinedShapeNode, combinedStrideNode, loc, rewriter);
        
    Value resultShape = serializeLayoutNodeToShape(resShapeNode, loc, rewriter, ctx);
    Value resultStride = serializeLayoutNodeToStride(resStrideNode, loc, rewriter, ctx);
    
    auto outShapeTy = dyn_cast<ShapeType>(resultShape.getType());
    auto outStrideTy = dyn_cast<StrideType>(resultStride.getType());
    if (!outShapeTy || !outStrideTy)
      return nullptr;
    auto resultLayoutType = LayoutType::get(ctx, outShapeTy, outStrideTy);
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
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "flir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp(), rewriter, loc);
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp(), rewriter, loc);
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp(), rewriter, loc);
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp(), rewriter, loc);
    
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
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "flir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp(), rewriter, loc);
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp(), rewriter, loc);
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp(), rewriter, loc);
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp(), rewriter, loc);
    
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
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "flir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp(), rewriter, loc);
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp(), rewriter, loc);
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp(), rewriter, loc);
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp(), rewriter, loc);
    
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
    // FlatDivideOpLowering in Flir likely expects a flattened result?
    // But current lowering just returned Layout.
    // If we return zipped structure ((T0, T1), (C0, C1)), it is nested.
    // We should probably flatten it. But without FlattenOp support, we leave it nested.
    // The user can use flatten() on result.
    
    auto loc = op->getLoc();
    Value inputLayout = op.getInput();
    Value tilerLayout = op.getTiler();
    
    auto *inputLayoutOp = inputLayout.getDefiningOp();
    auto *tilerLayoutOp = tilerLayout.getDefiningOp();
    
    if (!inputLayoutOp || inputLayoutOp->getName().getStringRef() != "flir.make_layout" ||
        !tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
      
    Value inputShape = inputLayoutOp->getOperand(0);
    Value inputStride = inputLayoutOp->getOperand(1);
    Value tilerShape = tilerLayoutOp->getOperand(0);
    Value tilerStride = tilerLayoutOp->getOperand(1);
    
    LayoutNode inputShapeNode = deserializeLayoutNode(inputShape.getDefiningOp(), rewriter, loc);
    LayoutNode inputStrideNode = deserializeLayoutNode(inputStride.getDefiningOp(), rewriter, loc);
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShape.getDefiningOp(), rewriter, loc);
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStride.getDefiningOp(), rewriter, loc);
    
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
    
    auto *ctx = rewriter.getContext();
    // Create a layout from the tiler shape using compact LayoutLeft strides.
    // This must be computed at compile time; dynamic tiler shapes are rejected.
    Value makeStrideVal =
        makeLayoutLeftStrideFromStaticShapeTypeMode(tilerShape, loc, rewriter, ctx);
    if (!makeStrideVal)
      return failure();
    auto layoutType = LayoutType::get(ctx,
                                      llvm::cast<ShapeType>(tilerShape.getType()),
                                      llvm::cast<StrideType>(makeStrideVal.getType()));
    auto tilerLayout = rewriter.create<MakeLayoutOp>(
        loc, layoutType, tilerShape, makeStrideVal);
    
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
    // @pre We work on a flattened (leaf) view of (shape, stride). For rank > 1
    //      tilers, all stride leaves must be compile-time constants (dynamic-stride
    //      complements are rejected).
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
    if (!tilerLayoutOp || tilerLayoutOp->getName().getStringRef() != "flir.make_layout")
      return failure();
    
    auto tilerShapeVal = tilerLayoutOp->getOperand(0);
    auto tilerStrideVal = tilerLayoutOp->getOperand(1);

    // Use LayoutNodes so we support type-mode constants.
    LayoutNode tilerShapeNode = deserializeLayoutNode(tilerShapeVal.getDefiningOp(), rewriter, loc);
    LayoutNode tilerStrideNode = deserializeLayoutNode(tilerStrideVal.getDefiningOp(), rewriter, loc);

    auto [compShape, compStride] = computeComplementInline(
        tilerShapeNode, tilerStrideNode, targetSize, loc, rewriter);

    if (!compShape || !compStride)
      return failure();

    int rank = 1;
    if (auto st = dyn_cast<ShapeType>(compShape.getType()))
      rank = st.getRank();
    else if (auto st = dyn_cast<StrideType>(compStride.getType()))
      rank = st.getRank();

    auto resultLayoutType = LayoutType::get(rewriter.getContext(), rank);
    auto complementLayout = rewriter.create<MakeLayoutOp>(loc, resultLayoutType, compShape, compStride);

    // Keep behavior consistent with previous lowering: coalesce the result.
    auto coalescedLayout = rewriter.create<CoalesceOp>(loc, op.getResult().getType(), complementLayout.getResult());
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

#define GEN_PASS_DEF_FLIRTOSTANDARDPASS
#include "flir/FlirPasses.h.inc"

struct FlirToStandardPass
    : public impl::FlirToStandardPassBase<FlirToStandardPass> {
  
  using impl::FlirToStandardPassBase<FlirToStandardPass>::FlirToStandardPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // The lowering may materialize `cf.assert` for runtime checks.
    registry.insert<mlir::cf::ControlFlowDialect>();
  }
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // Add all lowering patterns
    patterns.add<SizeOpLowering>(&getContext());
    patterns.add<CosizeOpLowering>(&getContext());
    patterns.add<GetOpLowering>(&getContext());
    patterns.add<RankOpLowering>(&getContext());
    patterns.add<Crd2IdxOpLowering>(&getContext());
    patterns.add<Idx2CrdOpLowering>(&getContext());
    patterns.add<SwizzleXor16OpLowering>(&getContext());
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
    

    GreedyRewriteConfig config;
    // The flir-to-standard pipeline intentionally creates new flir ops (e.g. SizeOp)
    // that are then rewritten in subsequent greedy iterations. Some programs need
    // more than the MLIR default iteration limit to reach a fixpoint.
    config.enableFolding(true);
    config.setMaxIterations(256);

    FrozenRewritePatternSet frozen(std::move(patterns));
    // If we emitted any error diagnostics (e.g. non-injective layout checks),
    // we must fail the pass so the Python bindings convert it into an exception
    // instead of aborting on "unhandled captured errors".
    bool hadError = false;
    int errorCount = 0;
    ScopedDiagnosticHandler diagHandler(&getContext(), [&](Diagnostic &d) {
      if (d.getSeverity() == DiagnosticSeverity::Error) {
        hadError = true;
        ++errorCount;
        // Greedy rewriting may retry and re-emit the same error many times.
        // Keep the first error, suppress subsequent ones to avoid log spam.
        if (errorCount > 1)
          return success();
      }
      // Do not suppress non-error diagnostics.
      return failure();
    });

    auto res = applyPatternsGreedily(getOperation(), frozen, config);
    if (hadError || failed(res))
      signalPassFailure();
  }
};

}

namespace mlir {
namespace flir {

std::unique_ptr<Pass> createFlirToStandardPass() {
  return std::make_unique<FlirToStandardPass>();
}

}
}
