#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"

#include <functional>

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYREWRITEFUNCSIGNATUREPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

void collectDynamicLeaves(IntTupleAttr attr, SmallVectorImpl<IntAttr> &leaves) {
  if (attr.isLeaf()) {
    auto intAttr = attr.extractIntFromLeaf();
    if (!intAttr.isStatic())
      leaves.push_back(intAttr);
  } else {
    for (int i = 0; i < attr.rank(); ++i)
      collectDynamicLeaves(attr.at(i), leaves);
  }
}

bool isStaticNarrowLayout(Attribute attr) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return layout.isStatic();
  if (auto composed = dyn_cast<ComposedLayoutAttr>(attr))
    return composed.isStatic();
  return true;
}

//===----------------------------------------------------------------------===//
// DSL type -> LLVM packed struct type
//
// Only dynamic sub-components generate struct fields.
// Fully static sub-components are omitted from the struct.
//===----------------------------------------------------------------------===//

LLVM::LLVMStructType getIntTupleStructType(MLIRContext *ctx, IntTupleAttr attr) {
  SmallVector<IntAttr> leaves;
  collectDynamicLeaves(attr, leaves);
  SmallVector<Type> fields;
  fields.reserve(leaves.size());
  for (auto leaf : leaves)
    fields.push_back(IntegerType::get(ctx, leaf.getWidth()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

LLVM::LLVMStructType getLayoutStructType(MLIRContext *ctx, LayoutAttr attr) {
  SmallVector<Type> fields;
  if (!attr.getShape().isStatic())
    fields.push_back(getIntTupleStructType(ctx, attr.getShape()));
  if (!attr.getStride().isStatic())
    fields.push_back(getIntTupleStructType(ctx, attr.getStride()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

LLVM::LLVMStructType getComposedInnerStructType(MLIRContext *ctx, Attribute attr);

LLVM::LLVMStructType getComposedLayoutStructType(MLIRContext *ctx, ComposedLayoutAttr attr) {
  SmallVector<Type> fields;
  if (!attr.isStaticOuter())
    fields.push_back(getLayoutStructType(ctx, attr.getOuter()));
  if (!attr.isStaticOffset())
    fields.push_back(getIntTupleStructType(ctx, attr.getOffset()));
  if (!attr.isStaticInner())
    fields.push_back(getComposedInnerStructType(ctx, attr.getInner()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

LLVM::LLVMStructType getComposedInnerStructType(MLIRContext *ctx, Attribute attr) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return getLayoutStructType(ctx, layoutAttr);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return getComposedLayoutStructType(ctx, composedAttr);
  // swizzle should be handled by the caller
  llvm_unreachable("unexpected inner attribute type");
}

LLVM::LLVMStructType getNarrowLayoutStructType(MLIRContext *ctx, Attribute attr) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return getLayoutStructType(ctx, layout);
  if (auto composed = dyn_cast<ComposedLayoutAttr>(attr))
    return getComposedLayoutStructType(ctx, composed);
  llvm_unreachable("unexpected layout attribute type");
}

LLVM::LLVMStructType getCoordTensorStructType(MLIRContext *ctx, CoordTensorType ty) {
  SmallVector<Type> fields;
  if (!ty.getBase().isStatic())
    fields.push_back(getIntTupleStructType(ctx, ty.getBase()));
  if (!isStaticNarrowLayout(ty.getLayout()))
    fields.push_back(getNarrowLayoutStructType(ctx, ty.getLayout()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

bool memrefHasDynamicLayout(fly::MemRefType ty) { return !isStaticNarrowLayout(ty.getLayout()); }

//===----------------------------------------------------------------------===//
// Arg expansion: DSL type -> single struct type (or passthrough)
//
// MemRef is expanded to separate arguments: fly.ptr + layout-struct (if dynamic)
//===----------------------------------------------------------------------===//

enum class ArgKind {
  PassThrough,
  IntTuple,
  Layout,
  ComposedLayout,
  CoordTensor,
  MemRefStatic,
  MemRefDynamic,
};

struct ExpandedArg {
  ArgKind kind;
  SmallVector<Type, 2> types;
};

ExpandedArg expandArgType(Type ty) {
  auto *ctx = ty.getContext();
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty))
    return {ArgKind::IntTuple, {getIntTupleStructType(ctx, intTupleTy.getAttr())}};
  if (auto layoutTy = dyn_cast<LayoutType>(ty))
    return {ArgKind::Layout, {getLayoutStructType(ctx, layoutTy.getAttr())}};
  if (auto composedTy = dyn_cast<ComposedLayoutType>(ty))
    return {ArgKind::ComposedLayout, {getComposedLayoutStructType(ctx, composedTy.getAttr())}};
  if (auto coordTy = dyn_cast<CoordTensorType>(ty))
    return {ArgKind::CoordTensor, {getCoordTensorStructType(ctx, coordTy)}};
  if (auto memrefTy = dyn_cast<fly::MemRefType>(ty)) {
    auto ptrTy = memrefTy.getPointerType();
    if (memrefHasDynamicLayout(memrefTy)) {
      auto layoutStructTy = getNarrowLayoutStructType(ctx, memrefTy.getLayout());
      return {ArgKind::MemRefDynamic, {ptrTy, layoutStructTy}};
    }
    return {ArgKind::MemRefStatic, {ptrTy}};
  }
  return {ArgKind::PassThrough, {ty}};
}

//===----------------------------------------------------------------------===//
// Pack: DSL value -> LLVM struct value (for launch_func call site)
//===----------------------------------------------------------------------===//

Value packIntTupleToStruct(OpBuilder &builder, Location loc, Value intTuple, IntTupleAttr attr,
                           LLVM::LLVMStructType structTy) {
  Value result = LLVM::UndefOp::create(builder, loc, structTy);

  int32_t structIdx = 0;
  // TODO: use get_leaves op instead of recursion
  std::function<void(Value, IntTupleAttr)> extract = [&](Value cur, IntTupleAttr curAttr) {
    if (curAttr.isLeaf()) {
      if (!curAttr.isStatic()) {
        Value scalar = GetScalarOp::create(builder, loc, cur);
        result = LLVM::InsertValueOp::create(builder, loc, structTy, result, scalar,
                                             ArrayRef<int64_t>{static_cast<int64_t>(structIdx)});
        structIdx++;
      }
      return;
    }
    for (int32_t i = 0; i < curAttr.rank(); ++i) {
      IntTupleType childTy = IntTupleType::get(curAttr.at(i));
      if (!childTy.isStatic()) {
        Value child = GetOp::create(builder, loc, childTy, cur,
                                    DenseI32ArrayAttr::get(builder.getContext(), {i}));
        extract(child, curAttr.at(i));
      }
    }
  };

  extract(intTuple, attr);
  return result;
}

Value packLayoutToStruct(OpBuilder &builder, Location loc, Value layout, LayoutAttr attr,
                         LLVM::LLVMStructType structTy) {
  Value result = LLVM::UndefOp::create(builder, loc, structTy);
  int64_t idx = 0;
  if (!attr.getShape().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value shapeValue = GetShapeOp::create(builder, loc, layout);
    Value shapeStruct = packIntTupleToStruct(builder, loc, shapeValue, attr.getShape(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, shapeStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!attr.getStride().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value strideValue = GetStrideOp::create(builder, loc, layout);
    Value strideStruct = packIntTupleToStruct(builder, loc, strideValue, attr.getStride(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, strideStruct,
                                         ArrayRef<int64_t>{idx});
  }
  return result;
}

Value packComposedInnerToStruct(OpBuilder &builder, Location loc, Value inner, Attribute attr,
                                LLVM::LLVMStructType innerStructTy);

Value packComposedLayoutToStruct(OpBuilder &builder, Location loc, Value composed,
                                 ComposedLayoutAttr attr, LLVM::LLVMStructType structTy) {
  Value result = LLVM::UndefOp::create(builder, loc, structTy);
  int64_t idx = 0;
  if (!attr.getOuter().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value outerVal = ComposedGetOuterOp::create(builder, loc, composed);
    Value outerStruct = packLayoutToStruct(builder, loc, outerVal, attr.getOuter(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, outerStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!attr.getOffset().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value offsetVal = ComposedGetOffsetOp::create(builder, loc, composed);
    Value offsetStruct = packIntTupleToStruct(builder, loc, offsetVal, attr.getOffset(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, offsetStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!attr.isStaticInner()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value innerVal = ComposedGetInnerOp::create(builder, loc, composed);
    Value innerStruct = packComposedInnerToStruct(builder, loc, innerVal, attr.getInner(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, innerStruct,
                                         ArrayRef<int64_t>{idx});
  }
  return result;
}

Value packComposedInnerToStruct(OpBuilder &builder, Location loc, Value inner, Attribute attr,
                                LLVM::LLVMStructType innerStructTy) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return packLayoutToStruct(builder, loc, inner, layoutAttr, innerStructTy);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return packComposedLayoutToStruct(builder, loc, inner, composedAttr, innerStructTy);
  llvm_unreachable("unexpected inner attribute type");
}

Value packNarrowLayoutToStruct(OpBuilder &builder, Location loc, Value layout, Attribute attr,
                               LLVM::LLVMStructType structTy) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return packLayoutToStruct(builder, loc, layout, layoutAttr, structTy);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return packComposedLayoutToStruct(builder, loc, layout, composedAttr, structTy);
  llvm_unreachable("unexpected layout attribute type");
}

Value packCoordTensorToStruct(OpBuilder &builder, Location loc, Value operand, CoordTensorType ty,
                              Type structTy) {
  auto outerStructTy = cast<LLVM::LLVMStructType>(structTy);
  Value result = LLVM::UndefOp::create(builder, loc, outerStructTy);
  int64_t idx = 0;
  if (!ty.getBase().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(outerStructTy.getBody()[idx]);
    Value iter = GetIterOp::create(builder, loc, operand);
    Value baseStruct = packIntTupleToStruct(builder, loc, iter, ty.getBase(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, outerStructTy, result, baseStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!isStaticNarrowLayout(ty.getLayout())) {
    auto fieldTy = cast<LLVM::LLVMStructType>(outerStructTy.getBody()[idx]);
    Value layout = GetLayoutOp::create(builder, loc, operand);
    Value layoutStruct = packNarrowLayoutToStruct(builder, loc, layout, ty.getLayout(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, outerStructTy, result, layoutStruct,
                                         ArrayRef<int64_t>{idx});
  }
  return result;
}

std::pair<Value, Value> packMemRefToPtrAndLayout(OpBuilder &builder, Location loc, Value operand,
                                                 fly::MemRefType memrefTy) {
  Value ptrValue = GetIterOp::create(builder, loc, operand);
  Value layoutValue = GetLayoutOp::create(builder, loc, operand);

  if (!memrefHasDynamicLayout(memrefTy))
    return {ptrValue, Value()};

  auto layoutStructTy = getNarrowLayoutStructType(memrefTy.getContext(), memrefTy.getLayout());
  Value layoutStruct =
      packNarrowLayoutToStruct(builder, loc, layoutValue, memrefTy.getLayout(), layoutStructTy);
  return {ptrValue, layoutStruct};
}

Value packDSLValueToStruct(OpBuilder &builder, Location loc, Value operand, Type ty,
                           LLVM::LLVMStructType structTy) {
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty))
    return packIntTupleToStruct(builder, loc, operand, intTupleTy.getAttr(), structTy);
  if (auto layoutTy = dyn_cast<LayoutType>(ty))
    return packLayoutToStruct(builder, loc, operand, layoutTy.getAttr(), structTy);
  if (auto composedTy = dyn_cast<ComposedLayoutType>(ty))
    return packComposedLayoutToStruct(builder, loc, operand, composedTy.getAttr(), structTy);
  if (auto coordTy = dyn_cast<CoordTensorType>(ty))
    return packCoordTensorToStruct(builder, loc, operand, coordTy, structTy);
  llvm_unreachable("unexpected DSL type");
}

//===----------------------------------------------------------------------===//
// Unpack: LLVM struct arg -> reconstruct normal-form DSL value (for func entry)
//===----------------------------------------------------------------------===//

Value unpackIntTupleFromStruct(OpBuilder &builder, Location loc, Value structVal, IntTupleAttr attr,
                               LLVM::LLVMStructType structTy) {
  SmallVector<Value> dyncElems;
  for (size_t i = 0; i < structTy.getBody().size(); ++i) {
    Type fieldTy = structTy.getBody()[i];
    Value val = LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal,
                                             ArrayRef<int64_t>{static_cast<int64_t>(i)});
    dyncElems.push_back(val);
  }
  return MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr), dyncElems);
}

Value unpackLayoutFromStruct(OpBuilder &builder, Location loc, Value structVal, LayoutAttr attr,
                             LLVM::LLVMStructType structTy) {
  int64_t idx = 0;
  Value shape;
  if (!attr.getShape().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    shape = unpackIntTupleFromStruct(builder, loc, fieldVal, attr.getShape(), fieldTy);
    idx++;
  } else {
    shape = MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr.getShape()), ValueRange{});
  }

  Value stride;
  if (!attr.getStride().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    stride = unpackIntTupleFromStruct(builder, loc, fieldVal, attr.getStride(), fieldTy);
  } else {
    stride =
        MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr.getStride()), ValueRange{});
  }

  return MakeLayoutOp::create(builder, loc, LayoutType::get(attr), shape, stride);
}

Value unpackComposedInnerFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                    Attribute attr, LLVM::LLVMStructType innerStructTy);

Type getNarrowLayoutType(Attribute attr) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return LayoutType::get(layout);
  if (auto composed = dyn_cast<ComposedLayoutAttr>(attr))
    return ComposedLayoutType::get(composed);
  llvm_unreachable("unexpected layout attribute type");
}

Value unpackComposedLayoutFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                     ComposedLayoutAttr attr, LLVM::LLVMStructType structTy) {
  int64_t idx = 0;
  Value outer;
  if (!attr.getOuter().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    outer = unpackLayoutFromStruct(builder, loc, fieldVal, attr.getOuter(), fieldTy);
    idx++;
  } else {
    outer = StaticOp::create(builder, loc, LayoutType::get(attr.getOuter()));
  }

  Value offset;
  if (!attr.getOffset().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    offset = unpackIntTupleFromStruct(builder, loc, fieldVal, attr.getOffset(), fieldTy);
    idx++;
  } else {
    offset =
        MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr.getOffset()), ValueRange{});
  }

  Value inner;
  if (!attr.isStaticInner()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    inner = unpackComposedInnerFromStruct(builder, loc, fieldVal, attr.getInner(), fieldTy);
  } else {
    auto innerAttr = attr.getInner();
    Type innerTy;
    if (auto layout = dyn_cast<LayoutAttr>(innerAttr))
      innerTy = LayoutType::get(layout);
    if (auto composed = dyn_cast<ComposedLayoutAttr>(innerAttr))
      innerTy = ComposedLayoutType::get(composed);
    if (auto swizzle = dyn_cast<SwizzleAttr>(innerAttr))
      innerTy = SwizzleType::get(swizzle);
    inner = StaticOp::create(builder, loc, innerTy);
  }

  return MakeComposedLayoutOp::create(builder, loc, inner, offset, outer);
}

Value unpackComposedInnerFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                    Attribute attr, LLVM::LLVMStructType innerStructTy) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return unpackLayoutFromStruct(builder, loc, structVal, layoutAttr, innerStructTy);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return unpackComposedLayoutFromStruct(builder, loc, structVal, composedAttr, innerStructTy);
  llvm_unreachable("unexpected inner attribute type");
}

Value unpackNarrowLayoutFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                   Attribute attr, Type structTy) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return unpackLayoutFromStruct(builder, loc, structVal, layout,
                                  cast<LLVM::LLVMStructType>(structTy));
  if (auto composedLayout = dyn_cast<ComposedLayoutAttr>(attr))
    return unpackComposedLayoutFromStruct(builder, loc, structVal, composedLayout,
                                          cast<LLVM::LLVMStructType>(structTy));
  llvm_unreachable("unexpected layout attribute type");
}

Value unpackCoordTensorFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                  CoordTensorType coordTy, LLVM::LLVMStructType structTy) {
  auto outerStructTy = cast<LLVM::LLVMStructType>(structVal.getType());
  int64_t idx = 0;

  Value iter;
  if (!coordTy.getBase().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(outerStructTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    iter = unpackIntTupleFromStruct(builder, loc, fieldVal, coordTy.getBase(), fieldTy);
    idx++;
  } else {
    iter = MakeIntTupleOp::create(builder, loc, IntTupleType::get(coordTy.getBase()), ValueRange{});
  }

  Value layout;
  if (!isStaticNarrowLayout(coordTy.getLayout())) {
    Type fieldTy = outerStructTy.getBody()[idx];
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    layout = unpackNarrowLayoutFromStruct(builder, loc, fieldVal, coordTy.getLayout(), fieldTy);
  } else {
    layout = StaticOp::create(builder, loc, getNarrowLayoutType(coordTy.getLayout()));
  }

  return MakeViewOp::create(builder, loc, iter, layout);
}

Value unpackDSLValueFromStruct(OpBuilder &builder, Location loc, Value structVal, Type oldType) {
  auto structTy = cast<LLVM::LLVMStructType>(structVal.getType());
  if (auto intTupleTy = dyn_cast<IntTupleType>(oldType))
    return unpackIntTupleFromStruct(builder, loc, structVal, intTupleTy.getAttr(), structTy);
  if (auto layoutTy = dyn_cast<LayoutType>(oldType))
    return unpackLayoutFromStruct(builder, loc, structVal, layoutTy.getAttr(), structTy);
  if (auto composedTy = dyn_cast<ComposedLayoutType>(oldType))
    return unpackComposedLayoutFromStruct(builder, loc, structVal, composedTy.getAttr(), structTy);
  if (auto coordTy = dyn_cast<CoordTensorType>(oldType))
    return unpackCoordTensorFromStruct(builder, loc, structVal, coordTy, structTy);
  llvm_unreachable("unexpected DSL type");
}

// Remove static DSL arguments from function signatures
void sinkStaticArgsFromFunc(FunctionOpInterface funcOp) {
  Location loc = funcOp.getLoc();
  OpBuilder builder(funcOp.getContext());
  ArrayRef<Type> argTypes = funcOp.getArgumentTypes();

  if (argTypes.empty())
    return;

  bool hasBody = !funcOp.getFunctionBody().empty();
  Block *entry = hasBody ? &funcOp.getFunctionBody().front() : nullptr;
  if (hasBody)
    builder.setInsertionPointToStart(entry);

  SmallVector<size_t> staticArgIndices;
  SmallVector<Type> newArgTypes;
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (auto mayStatic = dyn_cast<MayStaticTypeInterface>(argTypes[i]);
        mayStatic && mayStatic.isStatic()) {
      staticArgIndices.push_back(i);
      if (hasBody) {
        BlockArgument arg = entry->getArgument(i);
        Value staticVal = StaticOp::create(builder, loc, arg.getType());
        arg.replaceAllUsesWith(staticVal);
      }
    } else {
      newArgTypes.push_back(argTypes[i]);
    }
  }
  if (staticArgIndices.empty())
    return;

  funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes, funcOp.getResultTypes()));

  if (hasBody) {
    for (int i = static_cast<int>(staticArgIndices.size()) - 1; i >= 0; --i)
      entry->eraseArgument(staticArgIndices[i]);
  }
}

// Remove static operands from gpu.launch_func
void removeStaticOperandsFromLaunchFunc(gpu::LaunchFuncOp launchOp) {
  SmallVector<Value> oldOperands(launchOp.getKernelOperands().begin(),
                                 launchOp.getKernelOperands().end());
  SmallVector<Value> newOperands;
  bool changed = false;

  for (auto operand : oldOperands) {
    if (auto mayStatic = dyn_cast<MayStaticTypeInterface>(operand.getType());
        mayStatic && mayStatic.isStatic()) {
      changed = true;
      continue;
    }
    newOperands.push_back(operand);
  }

  if (changed)
    launchOp.getKernelOperandsMutable().assign(newOperands);
}

bool rewriteDSLArgsToStructFromFunc(FunctionOpInterface op) {
  ArrayRef<Type> argTypes = op.getArgumentTypes();
  SmallVector<Type> oldInputs(argTypes.begin(), argTypes.end());

  SmallVector<ExpandedArg> expandedArgs;
  SmallVector<Type> newInputs;
  bool changed = false;

  for (Type oldType : oldInputs) {
    auto expanded = expandArgType(oldType);
    expandedArgs.push_back(expanded);
    for (Type t : expanded.types)
      newInputs.push_back(t);
    if (expanded.kind != ArgKind::PassThrough)
      changed = true;
  }

  if (!changed)
    return false;

  op.setType(FunctionType::get(op.getContext(), newInputs, op.getResultTypes()));

  if (op.getFunctionBody().empty())
    return true;

  Block &entry = op.getFunctionBody().front();
  Location loc = op.getLoc();

  for (int i = static_cast<int>(oldInputs.size()) - 1; i >= 0; --i) {
    if (expandedArgs[i].kind == ArgKind::MemRefDynamic) {
      BlockArgument oldArg = entry.getArgument(i);
      oldArg.setType(expandedArgs[i].types[0]);
      entry.insertArgument(i + 1, expandedArgs[i].types[1], loc);
    } else if (expandedArgs[i].kind == ArgKind::MemRefStatic) {
      BlockArgument oldArg = entry.getArgument(i);
      oldArg.setType(expandedArgs[i].types[0]);
    } else if (expandedArgs[i].kind != ArgKind::PassThrough) {
      BlockArgument oldArg = entry.getArgument(i);
      oldArg.setType(expandedArgs[i].types[0]);
    }
  }

  OpBuilder builder(&entry, entry.begin());

  size_t newArgIdx = 0;
  for (size_t i = 0; i < oldInputs.size(); ++i) {
    if (expandedArgs[i].kind == ArgKind::PassThrough) {
      newArgIdx++;
      continue;
    }

    if (expandedArgs[i].kind == ArgKind::MemRefStatic) {
      auto memrefTy = cast<fly::MemRefType>(oldInputs[i]);
      BlockArgument ptrArg = entry.getArgument(newArgIdx);

      Value layout = StaticOp::create(builder, loc, getNarrowLayoutType(memrefTy.getLayout()));
      Value view = MakeViewOp::create(builder, loc, ptrArg, layout);

      llvm::SmallPtrSet<Operation *, 8> except;
      except.insert(view.getDefiningOp());
      ptrArg.replaceAllUsesExcept(view, except);

      newArgIdx++;
      continue;
    }

    if (expandedArgs[i].kind == ArgKind::MemRefDynamic) {
      auto memrefTy = cast<fly::MemRefType>(oldInputs[i]);
      BlockArgument ptrArg = entry.getArgument(newArgIdx);
      BlockArgument layoutStructArg = entry.getArgument(newArgIdx + 1);

      Value layout = unpackNarrowLayoutFromStruct(builder, loc, layoutStructArg,
                                                  memrefTy.getLayout(), layoutStructArg.getType());
      Value view = MakeViewOp::create(builder, loc, ptrArg, layout);

      llvm::SmallPtrSet<Operation *, 8> except;
      except.insert(view.getDefiningOp());
      for (Operation *user : layoutStructArg.getUsers())
        except.insert(user);
      ptrArg.replaceAllUsesExcept(view, except);

      newArgIdx += 2;
      continue;
    }

    BlockArgument structArg = entry.getArgument(newArgIdx);

    SmallVector<OpOperand *, 16> usesToReplace;
    for (OpOperand &use : structArg.getUses())
      usesToReplace.push_back(&use);

    Value reconstructed = unpackDSLValueFromStruct(builder, loc, structArg, oldInputs[i]);

    for (OpOperand *use : usesToReplace)
      use->set(reconstructed);

    newArgIdx++;
  }

  return true;
}

void packDSLOperandsFromLaunchFunc(gpu::LaunchFuncOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();

  SmallVector<Value> oldKernelOperands(op.getKernelOperands().begin(),
                                       op.getKernelOperands().end());
  SmallVector<Value> newKernelOperands;
  bool changed = false;

  for (Value operand : oldKernelOperands) {
    Type ty = operand.getType();
    auto expanded = expandArgType(ty);

    if (expanded.kind == ArgKind::PassThrough) {
      newKernelOperands.push_back(operand);
      continue;
    }

    changed = true;

    if (expanded.kind == ArgKind::MemRefStatic || expanded.kind == ArgKind::MemRefDynamic) {
      auto memrefTy = cast<fly::MemRefType>(ty);
      auto [ptr, layoutStruct] = packMemRefToPtrAndLayout(builder, loc, operand, memrefTy);
      newKernelOperands.push_back(ptr);
      if (layoutStruct)
        newKernelOperands.push_back(layoutStruct);
    } else {
      newKernelOperands.push_back(packDSLValueToStruct(
          builder, loc, operand, ty, cast<LLVM::LLVMStructType>(expanded.types[0])));
    }
  }

  if (changed)
    op.getKernelOperandsMutable().assign(newKernelOperands);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class RewriteFuncSignaturePass
    : public mlir::fly::impl::FlyRewriteFuncSignaturePassBase<RewriteFuncSignaturePass> {
public:
  using mlir::fly::impl::FlyRewriteFuncSignaturePassBase<
      RewriteFuncSignaturePass>::FlyRewriteFuncSignaturePassBase;

  void runOnOperation() override {
    getOperation()->walk([&](FunctionOpInterface funcOp) {
      sinkStaticArgsFromFunc(funcOp);
      rewriteDSLArgsToStructFromFunc(funcOp);
    });
    getOperation()->walk([&](gpu::LaunchFuncOp launchOp) {
      removeStaticOperandsFromLaunchFunc(launchOp);
      packDSLOperandsFromLaunchFunc(launchOp);
    });
  }
};

} // namespace

namespace impl {

std::unique_ptr<::mlir::Pass> createRewriteFuncSignaturePass() {
  return std::make_unique<RewriteFuncSignaturePass>();
}

} // namespace impl
