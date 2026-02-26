
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/NormalForm.h"
#include "flydsl/Dialect/Fly/Utils/TiledOpUtils.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <string>

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYLAYOUTLOWERINGPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

// Helper to check if an operation is a make_int_tuple-like op
static bool isMakeIntTupleLikeOp(Operation *op) {
  return isa_and_nonnull<MakeIntTupleOp, MakeShapeOp, MakeStrideOp, MakeCoordOp>(op);
}

static void collectDynamicLeaves(IntTupleAttr attr, SmallVectorImpl<IntAttr> &dynamicLeaves) {
  if (attr.isLeaf()) {
    if (auto intAttr = dyn_cast<IntAttr>(attr.getValue())) {
      if (!intAttr.isStatic() && !intAttr.isNone()) {
        dynamicLeaves.push_back(intAttr);
      }
    }
    return;
  }
  for (int i = 0; i < attr.rank(); ++i) {
    collectDynamicLeaves(attr.at(i), dynamicLeaves);
  }
}

static std::optional<LLVM::LLVMStructType> getIntTupleStructType(IntTupleAttr profile,
                                                                 MLIRContext *ctx) {
  SmallVector<IntAttr> dynamicLeaves;
  collectDynamicLeaves(profile, dynamicLeaves);
  if (dynamicLeaves.empty())
    return std::nullopt;

  SmallVector<Type> fields;
  fields.reserve(dynamicLeaves.size());
  for (size_t i = 0; i < dynamicLeaves.size(); ++i) {
    if (dynamicLeaves[i].getWidth() == 32) {
      fields.push_back(IntegerType::get(ctx, 32));
    } else {
      fields.push_back(IntegerType::get(ctx, 64));
    }
  }
  // Use packed struct to avoid padding between fields
  return LLVM::LLVMStructType::getLiteral(ctx, fields, /*isPacked=*/true);
}

static LLVM::LLVMStructType getIntTupleStructTypeOrEmpty(IntTupleAttr profile, MLIRContext *ctx) {
  SmallVector<IntAttr> dynamicLeaves;
  collectDynamicLeaves(profile, dynamicLeaves);
  if (dynamicLeaves.empty())
    return LLVM::LLVMStructType::getLiteral(ctx, {}, /*isPacked=*/true);

  SmallVector<Type> fields;
  fields.reserve(dynamicLeaves.size());
  for (auto leaf : dynamicLeaves) {
    if (leaf.getWidth() == 32) {
      fields.push_back(IntegerType::get(ctx, 32));
    } else {
      fields.push_back(IntegerType::get(ctx, 64));
    }
  }
  // Use packed struct to avoid padding between fields
  return LLVM::LLVMStructType::getLiteral(ctx, fields, /*isPacked=*/true);
}

static LLVM::LLVMStructType getLayoutStructTypeOrEmpty(LayoutAttr layoutAttr, MLIRContext *ctx) {
  SmallVector<Type> fields;
  fields.reserve(2);
  fields.push_back(getIntTupleStructTypeOrEmpty(layoutAttr.getShape(), ctx));
  fields.push_back(getIntTupleStructTypeOrEmpty(layoutAttr.getStride(), ctx));
  // Use packed struct to avoid padding between fields
  return LLVM::LLVMStructType::getLiteral(ctx, fields, /*isPacked=*/true);
}

static std::optional<LLVM::LLVMStructType> getLayoutStructType(LayoutAttr layoutAttr,
                                                               MLIRContext *ctx) {
  SmallVector<IntAttr> shapeLeaves;
  SmallVector<IntAttr> strideLeaves;
  collectDynamicLeaves(layoutAttr.getShape(), shapeLeaves);
  collectDynamicLeaves(layoutAttr.getStride(), strideLeaves);
  if (shapeLeaves.empty() && strideLeaves.empty())
    return std::nullopt;

  SmallVector<Type> fields;
  fields.reserve(2);
  fields.push_back(getIntTupleStructTypeOrEmpty(layoutAttr.getShape(), ctx));
  fields.push_back(getIntTupleStructTypeOrEmpty(layoutAttr.getStride(), ctx));
  // Use packed struct to avoid padding between fields
  return LLVM::LLVMStructType::getLiteral(ctx, fields, /*isPacked=*/true);
}

static unsigned mapAddressSpace(AddressSpace space) {
  switch (space) {
  case AddressSpace::Global:
    return 0;
  case AddressSpace::Shared:
    return 1;
  case AddressSpace::Register:
    return 2;
  }
  return 0;
}

// Get the fly.ptr type for a MemRef
static PointerType getMemRefPtrType(fly::MemRefType memrefTy) {
  auto *ctx = memrefTy.getContext();
  return PointerType::get(ctx, memrefTy.getElemTy(), memrefTy.getAddressSpace(),
                          memrefTy.getAlignment(), memrefTy.getSwizzle());
}

// Get the layout struct type for a MemRef (reuses existing layout struct logic)
// Returns nullopt if the layout is fully static (no dynamic elements)
static std::optional<LLVM::LLVMStructType> getMemRefLayoutStructType(fly::MemRefType memrefTy) {
  auto layoutStructTy = getLayoutStructType(memrefTy.getLayout(), memrefTy.getContext());
  return layoutStructTy; // Returns nullopt if fully static
}

// Check if a MemRef has any dynamic layout elements
static bool memrefHasDynamicLayout(fly::MemRefType memrefTy) {
  SmallVector<IntAttr> shapeLeaves, strideLeaves;
  collectDynamicLeaves(memrefTy.getLayout().getShape(), shapeLeaves);
  collectDynamicLeaves(memrefTy.getLayout().getStride(), strideLeaves);
  return !shapeLeaves.empty() || !strideLeaves.empty();
}

static Value castToFieldType(OpBuilder &builder, Location loc, Value value, Type fieldTy) {
  if (value.getType() == fieldTy)
    return value;
  if (fieldTy.isIndex())
    return arith::IndexCastOp::create(builder, loc, fieldTy, value);
  if (auto intTy = dyn_cast<IntegerType>(fieldTy)) {
    if (value.getType().isIndex())
      return arith::IndexCastOp::create(builder, loc, fieldTy, value);
    if (auto srcInt = dyn_cast<IntegerType>(value.getType())) {
      if (srcInt.getWidth() < intTy.getWidth())
        return arith::ExtSIOp::create(builder, loc, fieldTy, value);
      if (srcInt.getWidth() > intTy.getWidth())
        return arith::TruncIOp::create(builder, loc, fieldTy, value);
    }
  }
  return nullptr;
}

static std::optional<Value> packIntTupleToStruct(OpBuilder &builder, Location loc, Value tuple,
                                                 LLVM::LLVMStructType structTy) {
  auto tupleTy = dyn_cast<IntTupleType>(tuple.getType());
  if (!tupleTy)
    return std::nullopt;

  IntTupleAttr profile = tupleTy.getAttr();
  SmallVector<IntAttr> dynamicLeaves;
  collectDynamicLeaves(profile, dynamicLeaves);
  if (dynamicLeaves.empty()) {
    return LLVM::UndefOp::create(builder, loc, structTy);
  }

  Operation *defOp = tuple.getDefiningOp();
  if (!defOp || !isMakeIntTupleLikeOp(defOp))
    return std::nullopt;

  if (defOp->getNumOperands() != dynamicLeaves.size())
    return std::nullopt;

  Value result = LLVM::UndefOp::create(builder, loc, structTy);
  for (size_t i = 0; i < dynamicLeaves.size(); ++i) {
    Type valueFieldTy = structTy.getBody()[i];
    Value value = castToFieldType(builder, loc, defOp->getOperand(i), valueFieldTy);
    if (!value)
      return std::nullopt;
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, value,
                                         llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)});
  }
  return result;
}

static std::optional<SmallVector<Value>> collectDynamicOperands(Value tuple, IntTupleAttr profile) {
  SmallVector<IntAttr> dynamicLeaves;
  collectDynamicLeaves(profile, dynamicLeaves);
  if (dynamicLeaves.empty())
    return SmallVector<Value>{};

  Operation *defOp = tuple.getDefiningOp();
  if (!defOp || !isMakeIntTupleLikeOp(defOp))
    return std::nullopt;

  if (defOp->getNumOperands() != dynamicLeaves.size())
    return std::nullopt;

  SmallVector<Value> operands(defOp->getOperands().begin(), defOp->getOperands().end());
  return operands;
}

static std::optional<Value> packLayoutToStruct(OpBuilder &builder, Location loc, Value layout,
                                               LLVM::LLVMStructType structTy,
                                               LayoutAttr layoutAttr) {
  auto layoutOp = layout.getDefiningOp<MakeLayoutOp>();
  if (!layoutOp) {
    if (!layoutAttr.isStatic())
      return std::nullopt;
    auto shapeStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[0]);
    auto strideStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[1]);
    Value shapeStruct = LLVM::UndefOp::create(builder, loc, shapeStructTy);
    Value strideStruct = LLVM::UndefOp::create(builder, loc, strideStructTy);
    Value result = LLVM::UndefOp::create(builder, loc, structTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, shapeStruct,
                                         llvm::ArrayRef<int64_t>{0});
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, strideStruct,
                                         llvm::ArrayRef<int64_t>{1});
    return result;
  }

  auto shapeOps = collectDynamicOperands(layoutOp.getShape(), layoutAttr.getShape());
  auto strideOps = collectDynamicOperands(layoutOp.getStride(), layoutAttr.getStride());
  if (!shapeOps || !strideOps)
    return std::nullopt;

  auto shapeStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[0]);
  auto strideStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[1]);

  Value shapeStruct = LLVM::UndefOp::create(builder, loc, shapeStructTy);
  for (size_t i = 0; i < shapeOps->size(); ++i) {
    Type fieldTy = shapeStructTy.getBody()[i];
    Value casted = castToFieldType(builder, loc, (*shapeOps)[i], fieldTy);
    if (!casted)
      return std::nullopt;
    shapeStruct = LLVM::InsertValueOp::create(builder, loc, shapeStructTy, shapeStruct, casted,
                                              llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)});
  }

  Value strideStruct = LLVM::UndefOp::create(builder, loc, strideStructTy);
  for (size_t i = 0; i < strideOps->size(); ++i) {
    Type fieldTy = strideStructTy.getBody()[i];
    Value casted = castToFieldType(builder, loc, (*strideOps)[i], fieldTy);
    if (!casted)
      return std::nullopt;
    strideStruct = LLVM::InsertValueOp::create(builder, loc, strideStructTy, strideStruct, casted,
                                               llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)});
  }

  Value result = LLVM::UndefOp::create(builder, loc, structTy);
  result = LLVM::InsertValueOp::create(builder, loc, structTy, result, shapeStruct,
                                       llvm::ArrayRef<int64_t>{0});
  result = LLVM::InsertValueOp::create(builder, loc, structTy, result, strideStruct,
                                       llvm::ArrayRef<int64_t>{1});
  return result;
}

// Pack an IntTuple value to LLVM struct by extracting dynamic leaf values using GetLeafOp
// Uses recursive GetLeafOp calls to navigate nested IntTuple structure
static Value packIntTupleToStructGeneric(OpBuilder &builder, Location loc, Value intTuple,
                                         IntTupleAttr profile, LLVM::LLVMStructType structTy) {
  SmallVector<IntAttr> dynamicLeaves;
  collectDynamicLeaves(profile, dynamicLeaves);

  Value result = LLVM::UndefOp::create(builder, loc, structTy);

  // If no dynamic leaves, return undef struct
  if (dynamicLeaves.empty())
    return result;

  // Recursively extract dynamic leaf values
  int32_t structIdx = 0;
  std::function<void(Value, IntTupleAttr)> extractLeaves = [&](Value currentTuple,
                                                               IntTupleAttr currentAttr) {
    if (currentAttr.isLeaf()) {
      if (!currentAttr.isStatic()) {
        // Dynamic leaf - extract the scalar value using GetScalarOp
        Value scalarVal = GetScalarOp::create(builder, loc, currentTuple);
        Type fieldTy = structTy.getBody()[structIdx];
        Value casted = castToFieldType(builder, loc, scalarVal, fieldTy);
        result =
            LLVM::InsertValueOp::create(builder, loc, structTy, result, casted,
                                        llvm::ArrayRef<int64_t>{static_cast<int64_t>(structIdx)});
        structIdx++;
      }
      return;
    }
    // Non-leaf: recurse into children using GetLeafOp
    for (int32_t i = 0; i < currentAttr.rank(); ++i) {
      Value childTuple = GetLeafOp::create(builder, loc, currentTuple, static_cast<uint32_t>(i));
      extractLeaves(childTuple, currentAttr.at(i));
    }
  };

  extractLeaves(intTuple, profile);
  return result;
}

// Pack layout to struct - generic version that works for any layout Value (not just MakeLayoutOp)
static Value packLayoutToStructGeneric(OpBuilder &builder, Location loc, Value layout,
                                       LayoutAttr layoutAttr, LLVM::LLVMStructType structTy) {
  auto shapeStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[0]);
  auto strideStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[1]);

  Value shapeValue = nullptr;
  Value strideValue = nullptr;

  // Try to get shape and stride from MakeLayoutOp directly
  if (auto layoutOp = layout.getDefiningOp<MakeLayoutOp>()) {
    shapeValue = layoutOp.getShape();
    strideValue = layoutOp.getStride();
  } else {
    // Otherwise, create GetShapeOp and GetStrideOp
    shapeValue = GetShapeOp::create(builder, loc, layout);
    strideValue = GetStrideOp::create(builder, loc, layout);
  }

  Value shapeStruct =
      packIntTupleToStructGeneric(builder, loc, shapeValue, layoutAttr.getShape(), shapeStructTy);
  Value strideStruct = packIntTupleToStructGeneric(builder, loc, strideValue,
                                                   layoutAttr.getStride(), strideStructTy);

  Value result = LLVM::UndefOp::create(builder, loc, structTy);
  result = LLVM::InsertValueOp::create(builder, loc, structTy, result, shapeStruct,
                                       llvm::ArrayRef<int64_t>{0});
  result = LLVM::InsertValueOp::create(builder, loc, structTy, result, strideStruct,
                                       llvm::ArrayRef<int64_t>{1});
  return result;
}

// Extract ptr and layout values from a MemRef, returns {ptr, layoutStruct}
static std::pair<Value, Value> unpackMemRefToPtrAndLayout(OpBuilder &builder, Location loc,
                                                          Value memref, fly::MemRefType memrefTy) {
  Value ptrValue = nullptr;
  Value layoutValue = nullptr;

  if (auto makeView = memref.getDefiningOp<MakeViewOp>()) {
    ptrValue = makeView.getIter();
    layoutValue = makeView.getLayout();
  } else {
    ptrValue = GetIterOp::create(builder, loc, memref);
    layoutValue = GetLayoutOp::create(builder, loc, memref);
  }

  auto layoutAttr = memrefTy.getLayout();
  auto layoutStructTy = getLayoutStructTypeOrEmpty(layoutAttr, memrefTy.getContext());

  Value layoutStruct =
      packLayoutToStructGeneric(builder, loc, layoutValue, layoutAttr, layoutStructTy);

  return std::make_pair(ptrValue, layoutStruct);
}

static void lowerGpuLaunchFuncIntTupleOperands(gpu::LaunchFuncOp op) {
  auto kernelRef = op.getKernel();
  auto gpuFunc = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(op, kernelRef);
  if (!gpuFunc)
    return;

  SmallVector<Value> oldKernelOperands(op.getKernelOperands().begin(),
                                       op.getKernelOperands().end());
  SmallVector<Value> newKernelOperands;

  OpBuilder builder(op);
  bool changed = false;

  for (size_t i = 0; i < oldKernelOperands.size(); ++i) {
    Value operand = oldKernelOperands[i];

    if (auto tupleTy = dyn_cast<IntTupleType>(operand.getType())) {
      auto structTy = getIntTupleStructTypeOrEmpty(tupleTy.getAttr(), op.getContext());
      if (auto packed = packIntTupleToStruct(builder, op.getLoc(), operand, structTy)) {
        newKernelOperands.push_back(*packed);
        changed = true;
      } else {
        newKernelOperands.push_back(operand);
      }
      continue;
    }
    if (auto layoutTy = dyn_cast<LayoutType>(operand.getType())) {
      auto structTy = getLayoutStructTypeOrEmpty(layoutTy.getAttr(), op.getContext());
      if (auto packed =
              packLayoutToStruct(builder, op.getLoc(), operand, structTy, layoutTy.getAttr())) {
        newKernelOperands.push_back(*packed);
        changed = true;
      } else {
        newKernelOperands.push_back(operand);
      }
      continue;
    }
    if (auto memrefTy = dyn_cast<fly::MemRefType>(operand.getType())) {
      // MemRef is split into arguments: fly.ptr and optionally layout struct (if dynamic)
      auto unpacked = unpackMemRefToPtrAndLayout(builder, op.getLoc(), operand, memrefTy);
      newKernelOperands.push_back(unpacked.first); // fly.ptr
      // Only add layout struct if layout has dynamic elements
      if (memrefHasDynamicLayout(memrefTy)) {
        newKernelOperands.push_back(unpacked.second); // layout struct
      }
      changed = true;
      continue;
    }
    // Other types pass through unchanged
    newKernelOperands.push_back(operand);
  }

  if (!changed)
    return;

  op.getKernelOperandsMutable().assign(newKernelOperands);
}

/// Lower function arguments: IntTupleType, LayoutType, and MemRefType arguments are lowered
/// to LLVM structs. Works with any operation implementing FunctionOpInterface.
static bool lowerFuncIntTupleArgs(FunctionOpInterface op) {
  ArrayRef<Type> argTypes = op.getArgumentTypes();
  SmallVector<Type> oldInputs(argTypes.begin(), argTypes.end());

  // First pass: compute new argument types
  SmallVector<Type> newInputs;
  // MemRefStatic: only ptr arg (layout is fully static)
  // MemRefDynamic: ptr arg + layout struct arg
  enum class ArgKind { None, IntTuple, Layout, MemRefStatic, MemRefDynamic };
  SmallVector<ArgKind> argKinds; // One per old argument

  bool changed = false;
  for (Type oldType : oldInputs) {
    if (auto tupleTy = dyn_cast<IntTupleType>(oldType)) {
      auto structTy = getIntTupleStructTypeOrEmpty(tupleTy.getAttr(), op.getContext());
      newInputs.push_back(structTy);
      argKinds.push_back(ArgKind::IntTuple);
      changed = true;
      continue;
    }
    if (auto layoutTy = dyn_cast<LayoutType>(oldType)) {
      auto structTy = getLayoutStructTypeOrEmpty(layoutTy.getAttr(), op.getContext());
      newInputs.push_back(structTy);
      argKinds.push_back(ArgKind::Layout);
      changed = true;
      continue;
    }
    if (auto memrefTy = dyn_cast<fly::MemRefType>(oldType)) {
      // MemRef splits into args: fly.ptr and optionally layout struct (if dynamic)
      auto ptrTy = getMemRefPtrType(memrefTy);
      newInputs.push_back(ptrTy);
      if (memrefHasDynamicLayout(memrefTy)) {
        auto layoutStructTy = *getMemRefLayoutStructType(memrefTy);
        newInputs.push_back(layoutStructTy);
        argKinds.push_back(ArgKind::MemRefDynamic);
      } else {
        argKinds.push_back(ArgKind::MemRefStatic);
      }
      changed = true;
      continue;
    }
    newInputs.push_back(oldType);
    argKinds.push_back(ArgKind::None);
  }

  if (!changed)
    return false;

  // Update function type
  auto newFuncType = FunctionType::get(op.getContext(), newInputs, op.getResultTypes());
  op.setType(newFuncType);

  // Handle empty function (declaration only)
  if (op.getFunctionBody().empty())
    return true;

  Block &entry = op.getFunctionBody().front();
  Location loc = op.getLoc();

  // Transform block arguments: work backwards to handle index shifts from MemRef expansion
  for (int i = oldInputs.size() - 1; i >= 0; --i) {
    BlockArgument oldArg = entry.getArgument(i);

    if (argKinds[i] == ArgKind::None) {
      continue;
    }

    if (argKinds[i] == ArgKind::IntTuple || argKinds[i] == ArgKind::Layout) {
      // Compute which newInputs index this corresponds to
      size_t newIdx = 0;
      for (int j = 0; j < i; ++j) {
        newIdx++;
        if (argKinds[j] == ArgKind::MemRefDynamic)
          newIdx++; // MemRefDynamic adds an extra arg
      }
      oldArg.setType(newInputs[newIdx]);
      continue;
    }

    if (argKinds[i] == ArgKind::MemRefStatic) {
      // Static MemRef: only ptr arg, no layout struct
      size_t newIdx = 0;
      for (int j = 0; j < i; ++j) {
        newIdx++;
        if (argKinds[j] == ArgKind::MemRefDynamic)
          newIdx++;
      }
      oldArg.setType(newInputs[newIdx]);
      continue;
    }

    if (argKinds[i] == ArgKind::MemRefDynamic) {
      // Dynamic MemRef: ptr arg + layout struct arg
      size_t newIdx = 0;
      for (int j = 0; j < i; ++j) {
        newIdx++;
        if (argKinds[j] == ArgKind::MemRefDynamic)
          newIdx++;
      }
      // Change existing arg type to ptr
      oldArg.setType(newInputs[newIdx]);
      // Insert layout struct arg right after
      entry.insertArgument(i + 1, newInputs[newIdx + 1], loc);
    }
  }

  // Now reconstruct the fly values from the new arguments
  OpBuilder builder(&entry, entry.begin());

  // Compute new argument indices for each old argument
  size_t newArgIdx = 0;
  for (size_t i = 0; i < oldInputs.size(); ++i) {
    if (argKinds[i] == ArgKind::None) {
      newArgIdx++;
      continue;
    }

    if (argKinds[i] == ArgKind::IntTuple) {
      auto tupleTy = cast<IntTupleType>(oldInputs[i]);
      auto structTy = cast<LLVM::LLVMStructType>(newInputs[newArgIdx]);
      BlockArgument arg = entry.getArgument(newArgIdx);

      SmallVector<IntAttr> dynamicLeaves;
      collectDynamicLeaves(tupleTy.getAttr(), dynamicLeaves);
      if (dynamicLeaves.empty()) {
        Value tuple = MakeIntTupleOp::create(builder, loc, tupleTy, {});
        arg.replaceAllUsesWith(tuple);
        newArgIdx++;
        continue;
      }

      SmallVector<Value> dyncElems;
      SmallVector<Operation *> extractOps;
      dyncElems.reserve(dynamicLeaves.size());

      for (size_t j = 0; j < dynamicLeaves.size(); ++j) {
        Type fieldTy = structTy.getBody()[j];
        Value val = LLVM::ExtractValueOp::create(builder, loc, fieldTy, arg,
                                                 llvm::ArrayRef<int64_t>{static_cast<int64_t>(j)});
        dyncElems.push_back(val);
        extractOps.push_back(val.getDefiningOp());
      }

      Value tuple = MakeIntTupleOp::create(builder, loc, tupleTy, dyncElems);
      llvm::SmallPtrSet<Operation *, 8> except(extractOps.begin(), extractOps.end());
      arg.replaceAllUsesExcept(tuple, except);
      newArgIdx++;
      continue;
    }

    if (argKinds[i] == ArgKind::Layout) {
      auto layoutTy = cast<LayoutType>(oldInputs[i]);
      auto structTy = cast<LLVM::LLVMStructType>(newInputs[newArgIdx]);
      BlockArgument arg = entry.getArgument(newArgIdx);
      LayoutAttr layoutAttr = layoutTy.getAttr();

      SmallVector<IntAttr> shapeLeaves;
      SmallVector<IntAttr> strideLeaves;
      collectDynamicLeaves(layoutAttr.getShape(), shapeLeaves);
      collectDynamicLeaves(layoutAttr.getStride(), strideLeaves);
      if (shapeLeaves.empty() && strideLeaves.empty()) {
        Value Shape =
            MakeIntTupleOp::create(builder, loc, IntTupleType::get(layoutAttr.getShape()), {});
        Value Stride =
            MakeIntTupleOp::create(builder, loc, IntTupleType::get(layoutAttr.getStride()), {});
        Value layout = MakeLayoutOp::create(builder, loc, layoutTy, Shape, Stride);
        arg.replaceAllUsesWith(layout);
        newArgIdx++;
        continue;
      }

      SmallVector<Value> shapeElems;
      SmallVector<Value> strideElems;
      SmallVector<Operation *> extractOps;

      auto shapeStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[0]);
      auto strideStructTy = cast<LLVM::LLVMStructType>(structTy.getBody()[1]);
      Value shapeStruct = LLVM::ExtractValueOp::create(builder, loc, shapeStructTy, arg,
                                                       llvm::ArrayRef<int64_t>{0});
      Value strideStruct = LLVM::ExtractValueOp::create(builder, loc, strideStructTy, arg,
                                                        llvm::ArrayRef<int64_t>{1});
      extractOps.push_back(shapeStruct.getDefiningOp());
      extractOps.push_back(strideStruct.getDefiningOp());

      for (size_t j = 0; j < shapeLeaves.size(); ++j) {
        Type fieldTy = shapeStructTy.getBody()[j];
        Value val = LLVM::ExtractValueOp::create(builder, loc, fieldTy, shapeStruct,
                                                 llvm::ArrayRef<int64_t>{static_cast<int64_t>(j)});
        shapeElems.push_back(val);
        extractOps.push_back(val.getDefiningOp());
      }
      for (size_t j = 0; j < strideLeaves.size(); ++j) {
        Type fieldTy = strideStructTy.getBody()[j];
        Value val = LLVM::ExtractValueOp::create(builder, loc, fieldTy, strideStruct,
                                                 llvm::ArrayRef<int64_t>{static_cast<int64_t>(j)});
        strideElems.push_back(val);
        extractOps.push_back(val.getDefiningOp());
      }

      IntTupleType shapeTy = IntTupleType::get(op.getContext(), layoutAttr.getShape());
      IntTupleType strideTy = IntTupleType::get(op.getContext(), layoutAttr.getStride());
      Value shape = MakeIntTupleOp::create(builder, loc, shapeTy, shapeElems);
      Value stride = MakeIntTupleOp::create(builder, loc, strideTy, strideElems);
      Value layout = MakeLayoutOp::create(builder, loc, layoutTy, shape, stride);
      llvm::SmallPtrSet<Operation *, 8> except(extractOps.begin(), extractOps.end());
      arg.replaceAllUsesExcept(layout, except);
      newArgIdx++;
      continue;
    }

    if (argKinds[i] == ArgKind::MemRefStatic) {
      // Static MemRef: only ptr arg, layout is fully static
      auto memrefTy = cast<fly::MemRefType>(oldInputs[i]);
      LayoutAttr layoutAttr = memrefTy.getLayout();

      BlockArgument ptrArg = entry.getArgument(newArgIdx);

      // Create static layout using MakeIntTupleOp and MakeLayoutOp
      IntTupleType shapeTy = IntTupleType::get(op.getContext(), layoutAttr.getShape());
      IntTupleType strideTy = IntTupleType::get(op.getContext(), layoutAttr.getStride());
      Value shape = MakeIntTupleOp::create(builder, loc, shapeTy, ValueRange{});
      Value stride = MakeIntTupleOp::create(builder, loc, strideTy, ValueRange{});
      auto layoutTy = LayoutType::get(op.getContext(), layoutAttr);
      Value layout = MakeLayoutOp::create(builder, loc, layoutTy, shape, stride);

      // Create the MakeViewOp with fly.ptr directly
      Value view = MakeViewOp::create(builder, loc, ptrArg, layout);

      // Replace uses of the ptr arg
      llvm::SmallPtrSet<Operation *, 8> except;
      except.insert(view.getDefiningOp());
      ptrArg.replaceAllUsesExcept(view, except);

      newArgIdx++; // Static MemRef uses 1 arg
      continue;
    }

    if (argKinds[i] == ArgKind::MemRefDynamic) {
      // Dynamic MemRef: ptr arg + layout struct arg
      auto memrefTy = cast<fly::MemRefType>(oldInputs[i]);
      LayoutAttr layoutAttr = memrefTy.getLayout();

      BlockArgument ptrArg = entry.getArgument(newArgIdx);
      BlockArgument layoutStructArg = entry.getArgument(newArgIdx + 1);
      auto layoutStructTy = cast<LLVM::LLVMStructType>(layoutStructArg.getType());

      SmallVector<IntAttr> shapeLeaves;
      SmallVector<IntAttr> strideLeaves;
      collectDynamicLeaves(layoutAttr.getShape(), shapeLeaves);
      collectDynamicLeaves(layoutAttr.getStride(), strideLeaves);

      SmallVector<Value> shapeElems;
      SmallVector<Value> strideElems;
      SmallVector<Operation *> extractOps;

      auto shapeStructTy = cast<LLVM::LLVMStructType>(layoutStructTy.getBody()[0]);
      auto strideStructTy = cast<LLVM::LLVMStructType>(layoutStructTy.getBody()[1]);
      Value shapeStruct = LLVM::ExtractValueOp::create(builder, loc, shapeStructTy, layoutStructArg,
                                                       llvm::ArrayRef<int64_t>{0});
      Value strideStruct = LLVM::ExtractValueOp::create(
          builder, loc, strideStructTy, layoutStructArg, llvm::ArrayRef<int64_t>{1});
      extractOps.push_back(shapeStruct.getDefiningOp());
      extractOps.push_back(strideStruct.getDefiningOp());

      for (size_t j = 0; j < shapeLeaves.size(); ++j) {
        Type fieldTy = shapeStructTy.getBody()[j];
        Value val = LLVM::ExtractValueOp::create(builder, loc, fieldTy, shapeStruct,
                                                 llvm::ArrayRef<int64_t>{static_cast<int64_t>(j)});
        shapeElems.push_back(val);
        extractOps.push_back(val.getDefiningOp());
      }
      for (size_t j = 0; j < strideLeaves.size(); ++j) {
        Type fieldTy = strideStructTy.getBody()[j];
        Value val = LLVM::ExtractValueOp::create(builder, loc, fieldTy, strideStruct,
                                                 llvm::ArrayRef<int64_t>{static_cast<int64_t>(j)});
        strideElems.push_back(val);
        extractOps.push_back(val.getDefiningOp());
      }

      IntTupleType shapeTy = IntTupleType::get(op.getContext(), layoutAttr.getShape());
      IntTupleType strideTy = IntTupleType::get(op.getContext(), layoutAttr.getStride());
      Value shape = MakeIntTupleOp::create(builder, loc, shapeTy, shapeElems);
      Value stride = MakeIntTupleOp::create(builder, loc, strideTy, strideElems);
      auto layoutTy = LayoutType::get(op.getContext(), layoutAttr);
      Value layout = MakeLayoutOp::create(builder, loc, layoutTy, shape, stride);

      // Create the MakeViewOp with fly.ptr directly
      Value view = MakeViewOp::create(builder, loc, ptrArg, layout);

      // Replace uses of the ptr arg (which was the original memref arg)
      // Must exclude: extractOps (which use layoutStructArg), and view's defining op (which uses
      // ptrArg)
      llvm::SmallPtrSet<Operation *, 8> except(extractOps.begin(), extractOps.end());
      except.insert(view.getDefiningOp());
      ptrArg.replaceAllUsesExcept(view, except);

      newArgIdx += 2; // Dynamic MemRef uses 2 args
      continue;
    }
  }

  return true;
}

static void collectLeafValues(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                              const IntTupleValueAdaptor &tuple, SmallVectorImpl<Value> &out) {
  if (tuple.isLeaf()) {
    out.push_back(builder.getArithValue(tuple).value);
    return;
  }
  for (int i = 0; i < tuple.rank(); ++i) {
    collectLeafValues(builder, builder.at(tuple, i), out);
  }
}

static void collectLeafAttrs(IntTupleAttr attr, SmallVectorImpl<IntAttr> &out) {
  if (attr.isLeaf()) {
    out.push_back(attr.getLeafAsInt());
    return;
  }
  for (int i = 0; i < attr.rank(); ++i) {
    collectLeafAttrs(attr.at(i), out);
  }
}

static Value castPrintfArg(PatternRewriter &rewriter, Location loc, Value value,
                           std::string &format) {
  Type type = value.getType();
  if (isa<IndexType>(type)) {
    format += "%ld";
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), value);
  }
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() <= 32) {
      format += "%d";
      if (intTy.getWidth() < 32) {
        return arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), value);
      }
      return value;
    }
    format += "%ld";
    if (intTy.getWidth() != 64) {
      return arith::ExtSIOp::create(rewriter, loc, rewriter.getI64Type(), value);
    }
    return value;
  }
  if (auto floatTy = dyn_cast<FloatType>(type)) {
    if (floatTy.getWidth() <= 32) {
      format += "%f";
      if (floatTy.getWidth() < 32) {
        return arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), value);
      }
      return value;
    }
    format += "%lf";
    if (floatTy.getWidth() != 64) {
      return arith::ExtFOp::create(rewriter, loc, rewriter.getF64Type(), value);
    }
    return value;
  }
  return nullptr;
}

static bool appendScalarPrintfArg(PatternRewriter &rewriter, Location loc, Value value,
                                  std::string &format, SmallVectorImpl<Value> &args) {
  Value casted = castPrintfArg(rewriter, loc, value, format);
  if (!casted) {
    return false;
  }
  args.push_back(casted);
  return true;
}

static bool appendIntTuplePrintf(PatternRewriter &rewriter, Location loc,
                                 const IntTupleValueAdaptor &tuple, std::string &format,
                                 SmallVectorImpl<Value> &args) {
  // For single leaf values, don't add parentheses
  if (tuple.isLeaf()) {
    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    Value leafValue = builder.getArithValue(tuple).value;
    return appendScalarPrintfArg(rewriter, loc, leafValue, format, args);
  }

  SmallVector<Value> leaves;
  IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
  collectLeafValues(builder, tuple, leaves);
  format += "(";
  for (size_t i = 0; i < leaves.size(); ++i) {
    if (i > 0) {
      format += ", ";
    }
    if (!appendScalarPrintfArg(rewriter, loc, leaves[i], format, args)) {
      return false;
    }
  }
  format += ")";
  return true;
}

static bool appendIntTuplePrintfStatic(IntTupleAttr attr, std::string &format) {
  // For single leaf values, don't add parentheses
  if (attr.isLeaf()) {
    if (attr.getLeafAsInt().isStatic()) {
      format += std::to_string(attr.getLeafAsInt().getValue());
    } else {
      format += "?";
    }
    return true;
  }

  SmallVector<IntAttr> leaves;
  collectLeafAttrs(attr, leaves);
  format += "(";
  for (size_t i = 0; i < leaves.size(); ++i) {
    if (i > 0) {
      format += ", ";
    }
    if (leaves[i].isStatic()) {
      format += std::to_string(leaves[i].getValue());
      continue;
    }
    format += "?";
  }
  format += ")";
  return true;
}

template <typename OpTy, typename BinaryOpFn>
class IntTupleBinaryOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsTy = dyn_cast<IntTupleType>(lhs.getType());
    auto rhsTy = dyn_cast<IntTupleType>(rhs.getType());
    auto resultTy = dyn_cast<IntTupleType>(op.getResult().getType());
    if (!lhsTy || !rhsTy || !resultTy)
      return failure();

    // Check if inputs are in normal form (StaticOp or MakeIntTupleOp)
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(lhs)) ||
        !isNormalForm(cast<TypedValue<IntTupleType>>(rhs)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor lhsAdaptor = IntTupleValueAdaptor::create(builder, lhs, lhsTy.getAttr());
    IntTupleValueAdaptor rhsAdaptor = IntTupleValueAdaptor::create(builder, rhs, rhsTy.getAttr());

    auto result = BinaryOpFn{}(builder, lhsAdaptor, rhsAdaptor);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

struct IntTupleAddFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleAdd(builder, lhs, rhs);
  }
};
struct IntTupleSubFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleSub(builder, lhs, rhs);
  }
};
struct IntTupleMulFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleMul(builder, lhs, rhs);
  }
};
struct IntTupleDivFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleDiv(builder, lhs, rhs);
  }
};
struct IntTupleModFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleMod(builder, lhs, rhs);
  }
};
struct IntTupleShapeDivFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleShapeDiv(builder, lhs, rhs);
  }
};
struct IntTupleCeilDivFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleCeilDiv(builder, lhs, rhs);
  }
};
struct IntTupleElemLessFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleElemLess(builder, lhs, rhs);
  }
};

using IntTupleAddOpLowering = IntTupleBinaryOpLowering<IntTupleAddOp, IntTupleAddFn>;
using IntTupleSubOpLowering = IntTupleBinaryOpLowering<IntTupleSubOp, IntTupleSubFn>;
using IntTupleMulOpLowering = IntTupleBinaryOpLowering<IntTupleMulOp, IntTupleMulFn>;
using IntTupleDivOpLowering = IntTupleBinaryOpLowering<IntTupleDivOp, IntTupleDivFn>;
using IntTupleModOpLowering = IntTupleBinaryOpLowering<IntTupleModOp, IntTupleModFn>;
using ShapeDivOpLowering = IntTupleBinaryOpLowering<ShapeDivOp, IntTupleShapeDivFn>;
using CeilDivOpLowering = IntTupleBinaryOpLowering<CeilDivOp, IntTupleCeilDivFn>;
using ElemLessOpLowering = IntTupleBinaryOpLowering<ElemLessOp, IntTupleElemLessFn>;

template <typename OpTy, typename UnaryOpFn>
class IntTupleUnaryOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();

    auto inputTy = dyn_cast<IntTupleType>(input.getType());
    auto resultTy = dyn_cast<IntTupleType>(op.getResult().getType());
    if (!inputTy || !resultTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(input)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor inputAdaptor =
        IntTupleValueAdaptor::create(builder, input, inputTy.getAttr());

    auto result = UnaryOpFn{}(builder, inputAdaptor);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

struct IntTupleProductEachFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor input) const {
    return intTupleProductEach(builder, input);
  }
};
struct IntTupleProductFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor input) const {
    return intTupleProduct(builder, input);
  }
};

using IntTupleProductEachOpLowering =
    IntTupleUnaryOpLowering<IntTupleProductEachOp, IntTupleProductEachFn>;
using IntTupleProductOpLowering = IntTupleUnaryOpLowering<IntTupleProductOp, IntTupleProductFn>;

class SelectOpLowering : public OpRewritePattern<SelectOp> {
public:
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tuple = op.getTuple();

    auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType());
    if (!intTupleTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor adaptor =
        IntTupleValueAdaptor::create(builder, tuple, intTupleTy.getAttr());

    ArrayRef<int32_t> indices = op.getIndices();
    IntTupleValueAdaptor result = intTupleSelect(builder, adaptor, indices);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class GroupOpLowering : public OpRewritePattern<GroupOp> {
public:
  using OpRewritePattern<GroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GroupOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tuple = op.getTuple();
    int32_t begin = op.getBegin();
    int32_t end = op.getEnd();

    auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType());
    if (!intTupleTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor adaptor =
        IntTupleValueAdaptor::create(builder, tuple, intTupleTy.getAttr());

    IntTupleValueAdaptor result = intTupleGroup(builder, adaptor, begin, end);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class DiceOpLowering : public OpRewritePattern<DiceOp> {
public:
  using OpRewritePattern<DiceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DiceOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value coord = op.getCoord();

    auto intTupleTy = dyn_cast<IntTupleType>(src.getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    if (!intTupleTy || !coordTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(src)))
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor srcAdaptor =
        IntTupleValueAdaptor::create(builder, src, intTupleTy.getAttr());

    IntTupleValueAdaptor result = intTupleDice(builder, srcAdaptor, coordTy.getAttr());
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

template <class OpTy> class IntTupleReprofileOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto inputTuple = op.getTuple();
    if (auto tupleVal = dyn_cast<TypedValue<IntTupleType>>(inputTuple)) {
      if (isNormalForm(tupleVal)) {
        rewriter.replaceOp(op, MakeIntTupleOp::create(rewriter, op.getLoc(), tupleVal.getType(),
                                                      tupleVal.getDefiningOp()->getOperands()));
        return success();
      }
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// GetShapeOp Lowering
//===----------------------------------------------------------------------===//

class GetShapeLowering : public OpRewritePattern<GetShapeOp> {
public:
  using OpRewritePattern<GetShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetShapeOp op, PatternRewriter &rewriter) const override {
    auto layout = op.getLayout();

    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();
    if (auto defOp = layout.getDefiningOp<MakeLayoutOp>()) {
      rewriter.replaceOp(op, defOp.getShape());
      return success();
    }
    return failure();
  }
};

class GetStrideLowering : public OpRewritePattern<GetStrideOp> {
public:
  using OpRewritePattern<GetStrideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetStrideOp op, PatternRewriter &rewriter) const override {
    auto layout = op.getLayout();

    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();
    if (auto defOp = layout.getDefiningOp<MakeLayoutOp>()) {
      rewriter.replaceOp(op, defOp.getStride());
      return success();
    }
    return failure();
  }
};

class GetLayoutLowering : public OpRewritePattern<GetLayoutOp> {
public:
  using OpRewritePattern<GetLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetLayoutOp op, PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();

    if (auto makeViewOp = memref.getDefiningOp<MakeViewOp>()) {
      rewriter.replaceOp(op, makeViewOp.getLayout());
      return success();
    }
    return failure();
  }
};

class GetIterLowering : public OpRewritePattern<GetIterOp> {
public:
  using OpRewritePattern<GetIterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetIterOp op, PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();

    if (auto makeViewOp = memref.getDefiningOp<MakeViewOp>()) {
      rewriter.replaceOp(op, makeViewOp.getIter());
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// GetLeafOp Lowering
//===----------------------------------------------------------------------===//

class GetLeafOpLowering : public OpRewritePattern<GetLeafOp> {
public:
  using OpRewritePattern<GetLeafOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetLeafOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tuple = op.getTuple();
    int32_t leafIdx = op.getLeafIdx();

    // Handle IntTuple case
    if (auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType())) {
      if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)))
        return failure();

      auto defOp = tuple.getDefiningOp<MakeIntTupleOp>();
      if (!defOp)
        return failure();

      IntTupleAttr profile = intTupleTy.getAttr();
      IntTupleAttr leafProfile = profile.at(leafIdx);
      IntTupleType leafTy = IntTupleType::get(leafProfile);

      // Calculate the dynamic element offset for this leaf
      int32_t dyncOffset = 0;
      for (int32_t i = 0; i < leafIdx; ++i) {
        dyncOffset += profile.at(i).dyncLeafCount();
      }
      int32_t leafDyncCount = leafProfile.dyncLeafCount();

      // Extract the dynamic elements for this leaf
      SmallVector<Value> leafDyncElems;
      for (int32_t i = 0; i < leafDyncCount; ++i) {
        leafDyncElems.push_back(defOp.getDyncElems()[dyncOffset + i]);
      }

      Value newTuple = MakeIntTupleOp::create(rewriter, loc, leafTy, leafDyncElems);
      rewriter.replaceOp(op, newTuple);
      return success();
    }

    // Handle Layout case
    if (auto layoutTy = dyn_cast<LayoutType>(tuple.getType())) {
      if (!isNormalForm(cast<TypedValue<LayoutType>>(tuple)))
        return failure();

      auto defOp = tuple.getDefiningOp<MakeLayoutOp>();
      if (!defOp)
        return failure();

      LayoutAttr profile = layoutTy.getAttr();
      LayoutAttr leafProfile = profile.at(leafIdx);
      LayoutType leafTy = LayoutType::get(op.getContext(), leafProfile);

      // Get shape and stride from the defining MakeLayoutOp
      Value shape = defOp.getShape();
      Value stride = defOp.getStride();

      Value shapeLeaf = GetLeafOp::create(rewriter, loc, shape, leafIdx);
      Value strideLeaf = GetLeafOp::create(rewriter, loc, stride, leafIdx);

      Value newLayout = MakeLayoutOp::create(rewriter, loc, leafTy, shapeLeaf, strideLeaf);
      rewriter.replaceOp(op, newLayout);
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// GetScalarOp Lowering
//===----------------------------------------------------------------------===//

class GetScalarLowering : public OpRewritePattern<GetScalarOp> {
public:
  using OpRewritePattern<GetScalarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetScalarOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value intTuple = op.getIntTuple();

    auto intTupleTy = dyn_cast<IntTupleType>(intTuple.getType());
    if (!intTupleTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(intTuple)))
      return failure();

    IntTupleAttr profile = intTupleTy.getAttr();
    assert(profile.isLeaf() && "IntTuple must be a leaf");

    Type resultTy = op.getResult().getType();
    if (auto intAttr = dyn_cast<IntAttr>(profile.getValue())) {
      if (intAttr.isStatic()) {
        rewriter.replaceOp(
            op, arith::ConstantIntOp::create(rewriter, loc, resultTy, intAttr.getValue()));
        return success();
      } else {
        auto defOp = intTuple.getDefiningOp<MakeIntTupleOp>();
        if (!defOp)
          return failure();
        rewriter.replaceOp(op, defOp->getOperand(0));
        return success();
      }
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// SizeOp Lowering
//===----------------------------------------------------------------------===//

class SizeOpLowering : public OpRewritePattern<SizeOp> {
public:
  using OpRewritePattern<SizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SizeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getIntTuple();

    if (auto intTupleTy = dyn_cast<IntTupleType>(input.getType())) {
      if (!isNormalForm(dyn_cast<TypedValue<IntTupleType>>(input))) {
        return failure();
      }

      auto resultTy = dyn_cast<IntTupleType>(op.getResult().getType());
      if (!resultTy)
        return failure();

      // Use intTupleProduct to compute the size
      IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
      IntTupleValueAdaptor inputAdaptor =
          IntTupleValueAdaptor::create(builder, input, intTupleTy.getAttr());
      IntTupleValueAdaptor productAdaptor = intTupleProduct(builder, inputAdaptor);

      rewriter.replaceOp(op, builder.finalize(productAdaptor));
      return success();
    }

    if (auto layoutTy = dyn_cast<LayoutType>(input.getType())) {
      Value shape = nullptr;
      if (auto layoutVal = dyn_cast<TypedValue<LayoutType>>(input)) {
        if (isNormalForm(layoutVal)) {
          if (auto layoutOp = input.getDefiningOp<MakeLayoutOp>()) {
            shape = layoutOp.getShape();
          }
        }
      }
      if (!shape) {
        shape = GetShapeOp::create(rewriter, loc, input);
      }
      Value size = SizeOp::create(rewriter, loc, shape);
      rewriter.replaceOp(op, size);
      return success();
    }

    if (auto memrefTy = dyn_cast<fly::MemRefType>(input.getType())) {
      Value layout = GetLayoutOp::create(rewriter, loc, input);
      Value shape = GetShapeOp::create(rewriter, loc, layout);
      Value size = SizeOp::create(rewriter, loc, shape);
      rewriter.replaceOp(op, size);
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// SliceOp Lowering
//===----------------------------------------------------------------------===//

class SliceLowering : public OpRewritePattern<SliceOp> {
public:
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value coord = op.getCoord();

    auto srcTy = dyn_cast<IntTupleType>(src.getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());

    if (!srcTy || !coordTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(src)))
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor srcAdaptor = IntTupleValueAdaptor::create(builder, src, srcTy.getAttr());

    IntTupleValueAdaptor result = intTupleSlice(builder, srcAdaptor, coordTy.getAttr());

    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Crd2IdxOp Lowering
//===----------------------------------------------------------------------===//

class Crd2IdxLowering : public OpRewritePattern<Crd2IdxOp> {
public:
  using OpRewritePattern<Crd2IdxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Crd2IdxOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto coord = op.getCoord();
    auto layout = op.getLayout();

    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    auto layoutTy = dyn_cast<LayoutType>(layout.getType());
    if (!coordTy || !layoutTy)
      return failure();

    // Inputs must be in normal form
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);

    IntTupleValueAdaptor coordAdaptor =
        IntTupleValueAdaptor::create(builder, coord, coordTy.getAttr());
    IntTupleValueAdaptor shapeAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(0), layoutTy.getAttr().getShape());
    IntTupleValueAdaptor strideAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(1), layoutTy.getAttr().getStride());

    IntTupleValueAdaptor result = layoutCrd2Idx(builder, coordAdaptor, shapeAdaptor, strideAdaptor);

    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Layout Divide Operations Lowering
//===----------------------------------------------------------------------===//

/// Template for all four layout divide operations:
/// - LogicalDivideOp -> layoutLogicalDivide
/// - ZippedDivideOp -> layoutZippedDivide
/// - TiledDivideOp -> layoutTiledDivide
/// - FlatDivideOp -> layoutFlatDivide
template <typename OpTy,
          LayoutValueAdaptor (*DivideFunc)(LayoutBuilder<LayoutValueAdaptor> &, LayoutValueAdaptor,
                                           LayoutValueAdaptor),
          LayoutValueAdaptor (*DivideTileFunc)(LayoutBuilder<LayoutValueAdaptor> &,
                                               LayoutValueAdaptor, TileAttr)>
class LayoutDivideOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    Value divisorValue = op.getDivisor();

    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());

    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());

    if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorValue.getType())) {
      if (!isNormalForm(cast<TypedValue<LayoutType>>(divisorValue)))
        return failure();

      LayoutValueAdaptor divisorAdaptor(divisorValue, divisorLayoutTy.getAttr());
      LayoutValueAdaptor result = DivideFunc(layoutBuilder, layoutAdaptor, divisorAdaptor);

      rewriter.replaceOp(op, layoutBuilder.getValue(result));
      return success();
    }

    if (auto divisorTileTy = dyn_cast<TileType>(divisorValue.getType())) {
      TileAttr tileAttr = divisorTileTy.getAttr();
      LayoutValueAdaptor result = DivideTileFunc(layoutBuilder, layoutAdaptor, tileAttr);

      rewriter.replaceOp(op, layoutBuilder.getValue(result));
      return success();
    }

    return failure();
  }
};

using LogicalDivideOpLowering =
    LayoutDivideOpLowering<LogicalDivideOp, layoutLogicalDivide<LayoutValueAdaptor>,
                           layoutLogicalDivide<LayoutValueAdaptor>>;
using ZippedDivideOpLowering =
    LayoutDivideOpLowering<ZippedDivideOp, layoutZippedDivide<LayoutValueAdaptor>,
                           layoutZippedDivide<LayoutValueAdaptor>>;
using TiledDivideOpLowering =
    LayoutDivideOpLowering<TiledDivideOp, layoutTiledDivide<LayoutValueAdaptor>,
                           layoutTiledDivide<LayoutValueAdaptor>>;
using FlatDivideOpLowering =
    LayoutDivideOpLowering<FlatDivideOp, layoutFlatDivide<LayoutValueAdaptor>,
                           layoutFlatDivide<LayoutValueAdaptor>>;

template <typename OpTy, LayoutValueAdaptor (*ProductFunc)(LayoutBuilder<LayoutValueAdaptor> &,
                                                           LayoutValueAdaptor, LayoutValueAdaptor)>
class LayoutProductOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    Value tileValue = op.getTile();

    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    auto tileTy = dyn_cast<LayoutType>(tileValue.getType());
    if (!tileTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(tileValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor tileAdaptor(tileValue, tileTy.getAttr());
    LayoutValueAdaptor result = ProductFunc(layoutBuilder, layoutAdaptor, tileAdaptor);

    rewriter.replaceOp(op, layoutBuilder.getValue(result));
    return success();
  }
};

using LogicalProductOpLowering =
    LayoutProductOpLowering<LogicalProductOp, layoutLogicalProduct<LayoutValueAdaptor>>;
using ZippedProductOpLowering =
    LayoutProductOpLowering<ZippedProductOp, layoutZippedProduct<LayoutValueAdaptor>>;
using TiledProductOpLowering =
    LayoutProductOpLowering<TiledProductOp, layoutTiledProduct<LayoutValueAdaptor>>;
using FlatProductOpLowering =
    LayoutProductOpLowering<FlatProductOp, layoutFlatProduct<LayoutValueAdaptor>>;
using BlockedProductOpLowering =
    LayoutProductOpLowering<BlockedProductOp, layoutBlockedProduct<LayoutValueAdaptor>>;
using RakedProductOpLowering =
    LayoutProductOpLowering<RakedProductOp, layoutRakedProduct<LayoutValueAdaptor>>;

class AppendOpLowering : public OpRewritePattern<AppendOp> {
public:
  using OpRewritePattern<AppendOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AppendOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value tupleValue = op.getTuple();
    Value elemValue = op.getElem();

    auto tupleTy = dyn_cast<LayoutType>(tupleValue.getType());
    auto elemTy = dyn_cast<LayoutType>(elemValue.getType());
    if (!tupleTy || !elemTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(tupleValue)))
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(elemValue)))
      return failure();

    int32_t n = op.getN().value_or(-1);

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor tupleAdaptor(tupleValue, tupleTy.getAttr());
    LayoutValueAdaptor elemAdaptor(elemValue, elemTy.getAttr());

    auto tupleShape = layoutBuilder.getShape(tupleAdaptor);
    auto tupleStride = layoutBuilder.getStride(tupleAdaptor);
    auto elemShape = layoutBuilder.getShape(elemAdaptor);
    auto elemStride = layoutBuilder.getStride(elemAdaptor);

    auto resultShape = intTupleAppend(layoutBuilder, tupleShape, elemShape, n);
    auto resultStride = intTupleAppend(layoutBuilder, tupleStride, elemStride, n);
    LayoutValueAdaptor result = layoutBuilder.makeLayout(resultShape, resultStride);
    rewriter.replaceOp(op, layoutBuilder.getValue(result));
    return success();
  }
};

class PrependOpLowering : public OpRewritePattern<PrependOp> {
public:
  using OpRewritePattern<PrependOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrependOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value tupleValue = op.getTuple();
    Value elemValue = op.getElem();

    auto tupleTy = dyn_cast<LayoutType>(tupleValue.getType());
    auto elemTy = dyn_cast<LayoutType>(elemValue.getType());
    if (!tupleTy || !elemTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(tupleValue)))
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(elemValue)))
      return failure();

    int32_t n = op.getN().value_or(-1);

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor tupleAdaptor(tupleValue, tupleTy.getAttr());
    LayoutValueAdaptor elemAdaptor(elemValue, elemTy.getAttr());

    auto tupleShape = layoutBuilder.getShape(tupleAdaptor);
    auto tupleStride = layoutBuilder.getStride(tupleAdaptor);
    auto elemShape = layoutBuilder.getShape(elemAdaptor);
    auto elemStride = layoutBuilder.getStride(elemAdaptor);

    auto resultShape = intTuplePrepend(layoutBuilder, tupleShape, elemShape, n);
    auto resultStride = intTuplePrepend(layoutBuilder, tupleStride, elemStride, n);
    LayoutValueAdaptor result = layoutBuilder.makeLayout(resultShape, resultStride);
    rewriter.replaceOp(op, layoutBuilder.getValue(result));
    return success();
  }
};

class CoalesceOpLowering : public OpRewritePattern<CoalesceOp> {
public:
  using OpRewritePattern<CoalesceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoalesceOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    std::optional<IntTupleAttr> profileAttr;
    if (op.getAttr()) {
      auto attrTy = dyn_cast<IntTupleType>(op.getAttr().getType());
      if (attrTy)
        profileAttr = attrTy.getAttr();
    }

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result = layoutCoalesce(layoutBuilder, layoutAdaptor, profileAttr);
    rewriter.replaceOp(op, layoutBuilder.getValue(result));
    return success();
  }
};

class CompositionOpLowering : public OpRewritePattern<CompositionOp> {
public:
  using OpRewritePattern<CompositionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompositionOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value outerValue = op.getOuter();
    Value innerValue = op.getInner();

    auto outerTy = dyn_cast<LayoutType>(outerValue.getType());
    if (!outerTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(outerValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor outerAdaptor(outerValue, outerTy.getAttr());

    if (auto innerLayoutTy = dyn_cast<LayoutType>(innerValue.getType())) {
      if (!isNormalForm(cast<TypedValue<LayoutType>>(innerValue)))
        return failure();
      LayoutValueAdaptor innerAdaptor(innerValue, innerLayoutTy.getAttr());
      LayoutValueAdaptor result = layoutComposition(layoutBuilder, outerAdaptor, innerAdaptor);
      rewriter.replaceOp(op, layoutBuilder.getValue(result));
      return success();
    }

    if (auto innerTileTy = dyn_cast<TileType>(innerValue.getType())) {
      TileAttr tileAttr = innerTileTy.getAttr();
      LayoutValueAdaptor result = layoutComposition(layoutBuilder, outerAdaptor, tileAttr);
      rewriter.replaceOp(op, layoutBuilder.getValue(result));
      return success();
    }

    return failure();
  }
};

class ComplementOpLowering : public OpRewritePattern<ComplementOp> {
public:
  using OpRewritePattern<ComplementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComplementOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());

    std::optional<IntTupleValueAdaptor> codomainSize;
    if (op.getCodomainSize()) {
      auto codomainTy = dyn_cast<IntTupleType>(op.getCodomainSize().getType());
      if (!codomainTy)
        return failure();
      if (!isNormalForm(cast<TypedValue<IntTupleType>>(op.getCodomainSize())))
        return failure();
      codomainSize =
          IntTupleValueAdaptor::create(layoutBuilder, op.getCodomainSize(), codomainTy.getAttr());
    }

    LayoutValueAdaptor result = layoutComplement(layoutBuilder, layoutAdaptor, codomainSize);
    rewriter.replaceOp(op, layoutBuilder.getValue(result));
    return success();
  }
};

class CosizeOpLowering : public OpRewritePattern<CosizeOp> {
public:
  using OpRewritePattern<CosizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CosizeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    auto result = layoutCosize(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, layoutBuilder.finalize(result));
    return success();
  }
};

class RightInverseOpLowering : public OpRewritePattern<RightInverseOp> {
public:
  using OpRewritePattern<RightInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RightInverseOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result = layoutRightInverse(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, layoutBuilder.getValue(result));
    return success();
  }
};

class RecastLayoutOpLowering : public OpRewritePattern<RecastLayoutOp> {
public:
  using OpRewritePattern<RecastLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RecastLayoutOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getSrc();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    int32_t newTypeBits = op.getNewTypeBits();
    int32_t oldTypeBits = op.getOldTypeBits();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result =
        layoutRecast(layoutBuilder, layoutAdaptor, oldTypeBits, newTypeBits);
    rewriter.replaceOp(op, layoutBuilder.getValue(result));
    return success();
  }
};

class PrintOpLowering : public OpRewritePattern<PrintOp> {
public:
  using OpRewritePattern<PrintOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintOp op, PatternRewriter &rewriter) const override {
    bool isGpuContext = op->getParentOfType<gpu::GPUFuncOp>() != nullptr;

    for (Value val : op.getValues()) {
      if (auto intTupleVal = dyn_cast<TypedValue<IntTupleType>>(val)) {
        if (!isNormalForm(intTupleVal)) {
          return failure();
        }
      } else if (auto layoutVal = dyn_cast<TypedValue<LayoutType>>(val)) {
        if (!isNormalForm(layoutVal)) {
          return failure();
        }
      } else {
        continue;
      }
    }

    auto loc = op.getLoc();
    std::string userFormat = op.getFormat().str();
    std::string format;
    SmallVector<Value> args;

    auto formatValueToString = [&](Value val) -> std::string {
      std::string valFormat;
      if (auto tupleTy = dyn_cast<IntTupleType>(val.getType())) {
        if (tupleTy.getAttr().isStatic()) {
          appendIntTuplePrintfStatic(tupleTy.getAttr(), valFormat);
        } else {
          IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
          IntTupleValueAdaptor tuple =
              IntTupleValueAdaptor::create(builder, val, tupleTy.getAttr());
          appendIntTuplePrintf(rewriter, loc, tuple, valFormat, args);
        }
      } else if (auto layoutTy = dyn_cast<LayoutType>(val.getType())) {
        if (layoutTy.getAttr().isStatic()) {
          appendIntTuplePrintfStatic(layoutTy.getAttr().getShape(), valFormat);
          valFormat += ":";
          appendIntTuplePrintfStatic(layoutTy.getAttr().getStride(), valFormat);
        } else {
          LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
          LayoutValueAdaptor layout(val, layoutTy.getAttr());
          appendIntTuplePrintf(rewriter, loc, layoutBuilder.getShape(layout), valFormat, args);
          valFormat += ":";
          appendIntTuplePrintf(rewriter, loc, layoutBuilder.getStride(layout), valFormat, args);
        }
      } else {
        appendScalarPrintfArg(rewriter, loc, val, valFormat, args);
      }
      return valFormat;
    };

    // For CPU context, we need to interleave text and values
    // Collect text segments and argument indices
    struct PrintSegment {
      std::string text;
      int argIndex = -1; // -1 means text only
    };
    SmallVector<PrintSegment> segments;

    if (!userFormat.empty()) {
      size_t valueIdx = 0;
      size_t argIdx = 0;
      size_t pos = 0;
      while (pos < userFormat.size()) {
        size_t placeholderPos = userFormat.find("{}", pos);
        if (placeholderPos == std::string::npos) {
          segments.push_back({userFormat.substr(pos), -1});
          break;
        }
        if (placeholderPos > pos) {
          segments.push_back({userFormat.substr(pos, placeholderPos - pos), -1});
        }
        if (valueIdx < op.getValues().size()) {
          size_t argStartIdx = args.size();
          std::string staticFormat = formatValueToString(op.getValues()[valueIdx]);
          size_t numArgsAdded = args.size() - argStartIdx;
          if (numArgsAdded == 0 && !staticFormat.empty()) {
            // Static value: add as text segment
            segments.push_back({staticFormat, -1});
          } else {
            for (size_t i = 0; i < numArgsAdded; ++i) {
              if (i > 0) {
                segments.push_back({", ", -1});
              }
              segments.push_back({"", static_cast<int>(argStartIdx + i)});
            }
          }
          valueIdx++;
        }
        pos = placeholderPos + 2;
      }
    } else {
      bool first = true;
      for (Value val : op.getValues()) {
        if (!first) {
          segments.push_back({" ", -1});
        }
        first = false;
        size_t argStartIdx = args.size();
        std::string staticFormat = formatValueToString(val);
        size_t numArgsAdded = args.size() - argStartIdx;
        if (numArgsAdded == 0 && !staticFormat.empty()) {
          // Static value: add as text segment
          segments.push_back({staticFormat, -1});
        } else {
          for (size_t i = 0; i < numArgsAdded; ++i) {
            if (i > 0) {
              segments.push_back({", ", -1});
            }
            segments.push_back({"", static_cast<int>(argStartIdx + i)});
          }
        }
      }
    }

    if (isGpuContext) {
      // For GPU, build printf format string
      for (const auto &seg : segments) {
        if (seg.argIndex >= 0) {
          castPrintfArg(rewriter, loc, args[seg.argIndex], format);
        } else {
          format += seg.text;
        }
      }
      format += "\n";
      gpu::PrintfOp::create(rewriter, loc, rewriter.getStringAttr(format), args);
    } else {
      // For CPU, print segments in order
      for (size_t i = 0; i < segments.size(); ++i) {
        const auto &seg = segments[i];
        if (seg.argIndex >= 0) {
          bool isLast = (i == segments.size() - 1);
          auto punctuation =
              isLast ? vector::PrintPunctuation::NewLine : vector::PrintPunctuation::NoPunctuation;
          vector::PrintOp::create(rewriter, loc, args[seg.argIndex], punctuation);
        } else if (!seg.text.empty()) {
          vector::PrintOp::create(rewriter, loc, seg.text);
        }
      }
      if (segments.empty() || segments.back().argIndex < 0) {
        vector::PrintOp::create(rewriter, loc, vector::PrintPunctuation::NewLine);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TiledCopy/TiledMma Partition Lowering
//===----------------------------------------------------------------------===//

static std::pair<Value, Value> getMemRefPtrAndLayout(OpBuilder &builder, Location loc,
                                                     Value memref) {
  if (auto makeViewOp = memref.getDefiningOp<MakeViewOp>()) {
    return {makeViewOp.getIter(), makeViewOp.getLayout()};
  }
  return {GetIterOp::create(builder, loc, memref), GetLayoutOp::create(builder, loc, memref)};
}

template <typename OpTy, LayoutValueAdaptor (*ThrValViewFunc)(LayoutBuilder<LayoutValueAdaptor> &,
                                                              CopyAtomTypeInterface, LayoutAttr,
                                                              TileAttr, LayoutValueAdaptor)>
class TiledCopyPartitionOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value memref = op->getOperand(1);
    Value coord = op.getCoord();

    auto tiledCopyTy = dyn_cast<TiledCopyType>(op.getTiledCopy().getType());
    auto memrefTy = dyn_cast<fly::MemRefType>(memref.getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    if (!tiledCopyTy || !memrefTy || !coordTy)
      return failure();

    auto [ptr, layoutValue] = getMemRefPtrAndLayout(rewriter, loc, memref);
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy || !isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();

    auto copyAtom = dyn_cast<CopyAtomTypeInterface>(tiledCopyTy.getCopyAtom());
    if (!copyAtom)
      return failure();

    LayoutAttr tiledLayoutThrVal = tiledCopyTy.getLayoutThrVal().getAttr();
    TileAttr tileMN = tiledCopyTy.getTileMN().getAttr();
    LayoutAttr layout = memrefTy.getLayout();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layout);
    LayoutValueAdaptor thrValView =
        ThrValViewFunc(layoutBuilder, copyAtom, tiledLayoutThrVal, tileMN, layoutAdaptor);

    auto thrValShape = layoutBuilder.getShape(thrValView);
    auto thrValStride = layoutBuilder.getStride(thrValView);
    auto expandedShape = intTupleExpand(layoutBuilder, thrValShape, {2});
    auto expandedStride = intTupleExpand(layoutBuilder, thrValStride, {2});
    LayoutValueAdaptor expandedLayout = layoutBuilder.makeLayout(expandedShape, expandedStride);

    Value expandedMemref =
        MakeViewOp::create(rewriter, loc, ptr, layoutBuilder.getValue(expandedLayout));

    SmallVector<Value> dynElems(coord.getDefiningOp()->getOperands());
    SmallVector<Attribute> sliceCoordElems;
    sliceCoordElems.push_back(coordTy.getAttr());
    sliceCoordElems.push_back(IntTupleAttr::getLeafNone(ctx));
    for (int i = 0; i < layout.rank(); ++i)
      sliceCoordElems.push_back(IntTupleAttr::getLeafNone(ctx));
    IntTupleAttr sliceCoordAttr = IntTupleAttr::get(ArrayAttr::get(ctx, sliceCoordElems));

    Value sliceCoord =
        MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(sliceCoordAttr), dynElems);

    Value result = SliceOp::create(rewriter, loc, expandedMemref, sliceCoord);

    rewriter.replaceOp(op, result);
    return success();
  }
};

using TiledCopyPartitionSrcOpLowering =
    TiledCopyPartitionOpLowering<TiledCopyPartitionSrcOp,
                                 layoutTiledCopyThrValViewSrc<LayoutValueAdaptor>>;
using TiledCopyPartitionDstOpLowering =
    TiledCopyPartitionOpLowering<TiledCopyPartitionDstOp,
                                 layoutTiledCopyThrValViewDst<LayoutValueAdaptor>>;

class TiledMmaPartitionOpLowering : public OpRewritePattern<TiledMmaPartitionOp> {
public:
  using OpRewritePattern<TiledMmaPartitionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledMmaPartitionOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto operandId = op.getOperandId();
    Value input = op.getInput();
    Value coord = op.getCoord();

    auto tiledMmaTy = dyn_cast<TiledMmaType>(op.getTiledMma().getType());
    auto memrefTy = dyn_cast<fly::MemRefType>(input.getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    if (!tiledMmaTy || !memrefTy || !coordTy)
      return failure();

    auto [inputPtr, inputLayoutValue] = getMemRefPtrAndLayout(rewriter, loc, input);
    auto inputLayoutTy = dyn_cast<LayoutType>(inputLayoutValue.getType());
    if (!inputLayoutTy || !isNormalForm(cast<TypedValue<LayoutType>>(inputLayoutValue)))
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();

    auto mmaAtom = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
    if (!mmaAtom)
      return failure();

    LayoutAttr atomLayoutMNK = tiledMmaTy.getAtomLayout().getAttr();
    TileAttr permutationMNK = tiledMmaTy.getPermutation().getAttr();
    LayoutAttr inputLayout = memrefTy.getLayout();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor inputLayoutAdaptor(inputLayoutValue, inputLayout);
    LayoutValueAdaptor thrValView = layoutTiledMmaThrValOperandView(
        layoutBuilder, mmaAtom, atomLayoutMNK, permutationMNK, operandId, inputLayoutAdaptor);

    Value thrValMemref =
        MakeViewOp::create(rewriter, loc, inputPtr, layoutBuilder.getValue(thrValView));

    LayoutBuilder<LayoutAttr> attrBuilder(ctx);
    LayoutAttr atomThrIDLayout = cast<LayoutAttr>(mmaAtom.getThrLayout());
    LayoutAttr thrLayoutVMNK = layoutTiledProduct(
        attrBuilder, atomThrIDLayout, attrBuilder.materializeConstantLayout(atomLayoutMNK));

    IntTupleAttr vmnkShape = thrLayoutVMNK.getShape();
    IntTupleAttr vmnkStride = thrLayoutVMNK.getStride();

    IntTupleBuilder<IntTupleValueAdaptor> tupleBuilder(rewriter, loc);
    IntTupleValueAdaptor coordAdaptor =
        IntTupleValueAdaptor::create(tupleBuilder, coord, coordTy.getAttr());

    IntTupleValueAdaptor hierCoord =
        layoutIdx2Crd(tupleBuilder, coordAdaptor, tupleBuilder.materializeConstantTuple(vmnkShape),
                      tupleBuilder.materializeConstantTuple(vmnkStride));

    int32_t vmnkRank = vmnkShape.rank();
    IntTupleAttr flatShape = IntTupleAttr::get(ArrayAttr::get(
        ctx, SmallVector<Attribute>(vmnkRank, IntTupleAttr::get(IntAttr::getStatic(ctx, 1)))));
    IntTupleValueAdaptor flatCoord =
        layoutCrd2Crd(tupleBuilder, hierCoord, tupleBuilder.materializeConstantTuple(vmnkShape),
                      tupleBuilder.materializeConstantTuple(flatShape));

    int thrIdx0, thrIdx1;
    switch (operandId) {
    case MmaOperand::C:
      [[fallthrough]];
    case MmaOperand::D:
      thrIdx0 = 1;
      thrIdx1 = 2;
      break;
    case MmaOperand::A:
      thrIdx0 = 1;
      thrIdx1 = 3;
      break;
    case MmaOperand::B:
      thrIdx0 = 2;
      thrIdx1 = 3;
      break;
    }

    IntTupleValueAdaptor thrV = tupleBuilder.at(flatCoord, 0);
    IntTupleValueAdaptor thrDim0 = tupleBuilder.at(flatCoord, thrIdx0);
    IntTupleValueAdaptor thrDim1 = tupleBuilder.at(flatCoord, thrIdx1);

    IntTupleBuilder<IntTupleValueAdaptor>::ElemCollector innerCollector;
    innerCollector.push_back(thrDim0);
    innerCollector.push_back(thrDim1);
    IntTupleValueAdaptor thrDims = tupleBuilder.makeTuple(innerCollector);

    IntTupleBuilder<IntTupleValueAdaptor>::ElemCollector thrCollector;
    thrCollector.push_back(thrV);
    thrCollector.push_back(thrDims);
    IntTupleValueAdaptor thrCoord = tupleBuilder.makeTuple(thrCollector);

    LayoutAttr thrValViewAttr = layoutBuilder.getLayoutAttr(thrValView);
    IntTupleAttr valModeShapeAttr = thrValViewAttr.getShape().at(1);
    auto buildNoneCoord = [&](auto self, IntTupleAttr shapeAttr) -> IntTupleAttr {
      if (shapeAttr.isLeaf()) {
        return IntTupleAttr::getLeafNone(ctx);
      }
      SmallVector<Attribute> elems;
      for (int i = 0; i < shapeAttr.rank(); ++i) {
        elems.push_back(self(self, shapeAttr.at(i)));
      }
      return IntTupleAttr::get(ArrayAttr::get(ctx, elems));
    };
    IntTupleAttr valNoneAttr = buildNoneCoord(buildNoneCoord, valModeShapeAttr);

    SmallVector<Attribute> sliceCoordElems;
    sliceCoordElems.push_back(tupleBuilder.getAttr(thrCoord));
    sliceCoordElems.push_back(valNoneAttr);
    IntTupleAttr sliceCoordAttr = IntTupleAttr::get(ArrayAttr::get(ctx, sliceCoordElems));

    Value thrCoordValue = tupleBuilder.finalize(thrCoord);
    SmallVector<Value> sliceDynElems(thrCoordValue.getDefiningOp()->getOperands());
    Value sliceCoord =
        MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(sliceCoordAttr), sliceDynElems);

    Value result = SliceOp::create(rewriter, loc, thrValMemref, sliceCoord);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ExpandCopyOpLowering : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  static void emitCopyOrAtomCall(PatternRewriter &rewriter, Location loc, Value copyAtomVal,
                                 CopyAtomTypeInterface copyAtomTy, Value srcPtr,
                                 LayoutValueAdaptor valSrcLayout, Value dstPtr,
                                 LayoutValueAdaptor valDstLayout,
                                 LayoutBuilder<LayoutValueAdaptor> &layoutBuilder) {
    auto *ctx = rewriter.getContext();
    LayoutBuilder<LayoutAttr> attrBuilder(ctx);

    auto thrValLayoutSrc = cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    IntAttr numValSrcAttr = intTupleProductImpl(attrBuilder, thrValLayoutSrc.getShape().at(1));
    int64_t numValSrc = numValSrcAttr.getValue();

    IntTupleAttr valSrcSizeAttr =
        layoutSize(attrBuilder, layoutBuilder.getLayoutAttr(valSrcLayout));
    int64_t valSize = valSrcSizeAttr.getLeafAsInt().getValue();

    Value srcView = MakeViewOp::create(rewriter, loc, srcPtr, layoutBuilder.getValue(valSrcLayout));
    Value dstView = MakeViewOp::create(rewriter, loc, dstPtr, layoutBuilder.getValue(valDstLayout));

    if (valSize == numValSrc) {
      CopyAtomCall::create(rewriter, loc, copyAtomVal, srcView, dstView);
    } else {
      CopyOp::create(rewriter, loc, copyAtomVal, srcView, dstView, /*pred=*/nullptr);
    }
  }

  LogicalResult matchAndRewrite(CopyOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value copyAtomVal = op.getCopyAtom();
    Value src = op.getSrc();
    Value dst = op.getDst();

    CopyAtomTypeInterface copyAtomTy;
    if (auto tiledCopyTy = dyn_cast<TiledCopyType>(copyAtomVal.getType()))
      copyAtomTy = dyn_cast<CopyAtomTypeInterface>(tiledCopyTy.getCopyAtom());
    else
      copyAtomTy = dyn_cast<CopyAtomTypeInterface>(copyAtomVal.getType());
    if (!copyAtomTy)
      return failure();

    auto srcMemRefTy = dyn_cast<fly::MemRefType>(src.getType());
    auto dstMemRefTy = dyn_cast<fly::MemRefType>(dst.getType());
    if (!srcMemRefTy || !dstMemRefTy)
      return failure();

    LayoutAttr srcLayout = srcMemRefTy.getLayout();
    LayoutAttr dstLayout = dstMemRefTy.getLayout();

    int32_t srcRank = srcLayout.rank();
    int32_t dstRank = dstLayout.rank();
    if (srcRank != dstRank)
      return failure();

    auto [srcPtr, srcLayoutValue] = getMemRefPtrAndLayout(rewriter, loc, src);
    auto [dstPtr, dstLayoutValue] = getMemRefPtrAndLayout(rewriter, loc, dst);

    auto srcLayoutTy = dyn_cast<LayoutType>(srcLayoutValue.getType());
    auto dstLayoutTy = dyn_cast<LayoutType>(dstLayoutValue.getType());
    if (!srcLayoutTy || !dstLayoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(srcLayoutValue)))
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(dstLayoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor srcLayoutAdaptor(srcLayoutValue, srcLayout);
    LayoutValueAdaptor dstLayoutAdaptor(dstLayoutValue, dstLayout);

    if (srcRank == 1) {
      emitCopyOrAtomCall(rewriter, loc, copyAtomVal, copyAtomTy, srcPtr, srcLayoutAdaptor, dstPtr,
                         dstLayoutAdaptor, layoutBuilder);
      rewriter.eraseOp(op);
      return success();
    }

    auto srcShapeAdaptor = layoutBuilder.getShape(srcLayoutAdaptor);
    auto srcStrideAdaptor = layoutBuilder.getStride(srcLayoutAdaptor);
    auto dstShapeAdaptor = layoutBuilder.getShape(dstLayoutAdaptor);
    auto dstStrideAdaptor = layoutBuilder.getStride(dstLayoutAdaptor);

    auto groupedSrcShape = intTupleGroup(layoutBuilder, srcShapeAdaptor, 1, srcRank);
    auto groupedSrcStride = intTupleGroup(layoutBuilder, srcStrideAdaptor, 1, srcRank);
    auto groupedDstShape = intTupleGroup(layoutBuilder, dstShapeAdaptor, 1, dstRank);
    auto groupedDstStride = intTupleGroup(layoutBuilder, dstStrideAdaptor, 1, dstRank);

    auto restSrcShape = layoutBuilder.at(groupedSrcShape, 1);
    auto restSrcStride = layoutBuilder.at(groupedSrcStride, 1);
    auto restDstShape = layoutBuilder.at(groupedDstShape, 1);
    auto restDstStride = layoutBuilder.at(groupedDstStride, 1);

    LayoutBuilder<LayoutAttr> attrBuilder(ctx);
    IntTupleAttr restDstShapeAttr = layoutBuilder.getAttr(restDstShape);
    IntAttr restSize = intTupleProductImpl(attrBuilder, restDstShapeAttr);
    if (!restSize.isStatic())
      return failure();
    int32_t numIter = restSize.getValue();

    auto valSrcShape = layoutBuilder.at(groupedSrcShape, 0);
    auto valSrcStride = layoutBuilder.at(groupedSrcStride, 0);
    auto valDstShape = layoutBuilder.at(groupedDstShape, 0);
    auto valDstStride = layoutBuilder.at(groupedDstStride, 0);

    LayoutValueAdaptor valSrcLayoutAdaptor = layoutBuilder.makeLayout(valSrcShape, valSrcStride);
    LayoutValueAdaptor valDstLayoutAdaptor = layoutBuilder.makeLayout(valDstShape, valDstStride);

    for (int32_t i = 0; i < numIter; ++i) {
      auto coordAdaptor = layoutBuilder.makeInt(layoutBuilder.materializeConstantArith(i));

      auto srcOffsetAdaptor =
          layoutCrd2Idx(layoutBuilder, coordAdaptor, restSrcShape, restSrcStride);
      auto dstOffsetAdaptor =
          layoutCrd2Idx(layoutBuilder, coordAdaptor, restDstShape, restDstStride);

      Value srcOffsetValue = layoutBuilder.finalize(srcOffsetAdaptor);
      Value dstOffsetValue = layoutBuilder.finalize(dstOffsetAdaptor);

      Value srcIterPtr = AddOffsetOp::create(rewriter, loc, srcPtr, srcOffsetValue);
      Value dstIterPtr = AddOffsetOp::create(rewriter, loc, dstPtr, dstOffsetValue);

      emitCopyOrAtomCall(rewriter, loc, copyAtomVal, copyAtomTy, srcIterPtr, valSrcLayoutAdaptor,
                         dstIterPtr, valDstLayoutAdaptor, layoutBuilder);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class ExpandGemmOpLowering : public OpRewritePattern<GemmOp> {
public:
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value mmaAtomVal = op.getMmaAtom();
    Value d = op.getD();
    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();

    MmaAtomTypeInterface mmaAtomTy;
    if (auto tiledMmaTy = dyn_cast<TiledMmaType>(mmaAtomVal.getType()))
      mmaAtomTy = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
    else
      mmaAtomTy = dyn_cast<MmaAtomTypeInterface>(mmaAtomVal.getType());
    if (!mmaAtomTy)
      return failure();

    auto dMemRefTy = dyn_cast<fly::MemRefType>(d.getType());
    auto aMemRefTy = dyn_cast<fly::MemRefType>(a.getType());
    auto bMemRefTy = dyn_cast<fly::MemRefType>(b.getType());
    auto cMemRefTy = dyn_cast<fly::MemRefType>(c.getType());
    if (!dMemRefTy || !aMemRefTy || !bMemRefTy || !cMemRefTy)
      return failure();

    LayoutAttr dLayout = dMemRefTy.getLayout();
    LayoutAttr aLayout = aMemRefTy.getLayout();
    LayoutAttr bLayout = bMemRefTy.getLayout();
    LayoutAttr cLayout = cMemRefTy.getLayout();

    int32_t dRank = dLayout.rank();
    int32_t aRank = aLayout.rank();
    int32_t bRank = bLayout.rank();
    int32_t cRank = cLayout.rank();

    auto [dPtr, dLayoutValue] = getMemRefPtrAndLayout(rewriter, loc, d);
    auto [aPtr, aLayoutValue] = getMemRefPtrAndLayout(rewriter, loc, a);
    auto [bPtr, bLayoutValue] = getMemRefPtrAndLayout(rewriter, loc, b);
    auto [cPtr, cLayoutValue] = getMemRefPtrAndLayout(rewriter, loc, c);

    if (!isNormalForm(cast<TypedValue<LayoutType>>(dLayoutValue)) ||
        !isNormalForm(cast<TypedValue<LayoutType>>(aLayoutValue)) ||
        !isNormalForm(cast<TypedValue<LayoutType>>(bLayoutValue)) ||
        !isNormalForm(cast<TypedValue<LayoutType>>(cLayoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutBuilder<LayoutAttr> attrBuilder(ctx);

    if (dRank == 1 && aRank == 1 && bRank == 1 && cRank == 1) {
      MmaAtomCall::create(rewriter, loc, mmaAtomVal, d, a, b, c);
      rewriter.eraseOp(op);
      return success();
    }

    if (dRank != 3 || cRank != 3 || aRank < 2 || bRank < 2)
      return failure();

    LayoutValueAdaptor dLayoutAdaptor(dLayoutValue, dLayout);
    LayoutValueAdaptor aLayoutAdaptor(aLayoutValue, aLayout);
    LayoutValueAdaptor bLayoutAdaptor(bLayoutValue, bLayout);
    LayoutValueAdaptor cLayoutAdaptor(cLayoutValue, cLayout);

    auto dShape = layoutBuilder.getShape(dLayoutAdaptor);
    auto dStride = layoutBuilder.getStride(dLayoutAdaptor);
    auto aShape = layoutBuilder.getShape(aLayoutAdaptor);
    auto aStride = layoutBuilder.getStride(aLayoutAdaptor);
    auto bShape = layoutBuilder.getShape(bLayoutAdaptor);
    auto bStride = layoutBuilder.getStride(bLayoutAdaptor);
    auto cShape = layoutBuilder.getShape(cLayoutAdaptor);
    auto cStride = layoutBuilder.getStride(cLayoutAdaptor);

    auto valDShape = layoutBuilder.at(dShape, 0);
    auto valDStride = layoutBuilder.at(dStride, 0);
    auto valAShape = layoutBuilder.at(aShape, 0);
    auto valAStride = layoutBuilder.at(aStride, 0);
    auto valBShape = layoutBuilder.at(bShape, 0);
    auto valBStride = layoutBuilder.at(bStride, 0);
    auto valCShape = layoutBuilder.at(cShape, 0);
    auto valCStride = layoutBuilder.at(cStride, 0);

    auto valDLayout = layoutBuilder.makeLayout(valDShape, valDStride);
    auto valALayout = layoutBuilder.makeLayout(valAShape, valAStride);
    auto valBLayout = layoutBuilder.makeLayout(valBShape, valBStride);
    auto valCLayout = layoutBuilder.makeLayout(valCShape, valCStride);

    IntTupleAttr mShapeAttr = layoutBuilder.getAttr(layoutBuilder.at(dShape, 1));
    IntTupleAttr nShapeAttr = layoutBuilder.getAttr(layoutBuilder.at(dShape, 2));
    IntAttr mSizeAttr = intTupleProductImpl(attrBuilder, mShapeAttr);
    IntAttr nSizeAttr = intTupleProductImpl(attrBuilder, nShapeAttr);
    if (!mSizeAttr.isStatic() || !nSizeAttr.isStatic())
      return failure();
    int32_t M = mSizeAttr.getValue();
    int32_t N = nSizeAttr.getValue();

    int32_t K = 1;
    if (aRank == 3 && bRank == 3) {
      IntTupleAttr kShapeAttr = layoutBuilder.getAttr(layoutBuilder.at(aShape, 2));
      IntAttr kSizeAttr = intTupleProductImpl(attrBuilder, kShapeAttr);
      if (!kSizeAttr.isStatic())
        return failure();
      K = kSizeAttr.getValue();
    }

    auto mDShape = layoutBuilder.at(dShape, 1);
    auto mDStride = layoutBuilder.at(dStride, 1);
    auto nDShape = layoutBuilder.at(dShape, 2);
    auto nDStride = layoutBuilder.at(dStride, 2);
    auto mCShape = layoutBuilder.at(cShape, 1);
    auto mCStride = layoutBuilder.at(cStride, 1);
    auto nCShape = layoutBuilder.at(cShape, 2);
    auto nCStride = layoutBuilder.at(cStride, 2);

    auto mAShape = layoutBuilder.at(aShape, 1);
    auto mAStride = layoutBuilder.at(aStride, 1);
    auto nBShape = layoutBuilder.at(bShape, 1);
    auto nBStride = layoutBuilder.at(bStride, 1);

    bool hasK = (aRank == 3 && bRank == 3);

    for (int32_t k = 0; k < K; ++k) {
      Value aKPtr = aPtr;
      Value bKPtr = bPtr;

      if (hasK) {
        auto kAShape = layoutBuilder.at(aShape, 2);
        auto kAStride = layoutBuilder.at(aStride, 2);
        auto kBShape = layoutBuilder.at(bShape, 2);
        auto kBStride = layoutBuilder.at(bStride, 2);

        auto kCoord = layoutBuilder.makeInt(layoutBuilder.materializeConstantArith(k));
        Value aKOffsetValue =
            layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, kCoord, kAShape, kAStride));
        Value bKOffsetValue =
            layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, kCoord, kBShape, kBStride));
        aKPtr = AddOffsetOp::create(rewriter, loc, aPtr, aKOffsetValue);
        bKPtr = AddOffsetOp::create(rewriter, loc, bPtr, bKOffsetValue);
      }

      for (int32_t m = 0; m < M; ++m) {
        auto mCoord = layoutBuilder.makeInt(layoutBuilder.materializeConstantArith(m));
        Value dMOffsetValue =
            layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, mCoord, mDShape, mDStride));
        Value aMOffsetValue =
            layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, mCoord, mAShape, mAStride));

        for (int32_t n = 0; n < N; ++n) {
          auto nCoord = layoutBuilder.makeInt(layoutBuilder.materializeConstantArith(n));
          Value dNOffsetValue =
              layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, nCoord, nDShape, nDStride));
          Value bNOffsetValue =
              layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, nCoord, nBShape, nBStride));

          Value dIterPtr = AddOffsetOp::create(
              rewriter, loc, AddOffsetOp::create(rewriter, loc, dPtr, dMOffsetValue),
              dNOffsetValue);
          Value aIterPtr = AddOffsetOp::create(rewriter, loc, aKPtr, aMOffsetValue);
          Value bIterPtr = AddOffsetOp::create(rewriter, loc, bKPtr, bNOffsetValue);

          Value dView =
              MakeViewOp::create(rewriter, loc, dIterPtr, layoutBuilder.getValue(valDLayout));
          Value aView =
              MakeViewOp::create(rewriter, loc, aIterPtr, layoutBuilder.getValue(valALayout));
          Value bView =
              MakeViewOp::create(rewriter, loc, bIterPtr, layoutBuilder.getValue(valBLayout));

          Value accView;
          if (k == 0) {
            Value cMOffsetValue =
                layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, mCoord, mCShape, mCStride));
            Value cNOffsetValue =
                layoutBuilder.finalize(layoutCrd2Idx(layoutBuilder, nCoord, nCShape, nCStride));
            Value cIterPtr = AddOffsetOp::create(
                rewriter, loc, AddOffsetOp::create(rewriter, loc, cPtr, cMOffsetValue),
                cNOffsetValue);
            accView =
                MakeViewOp::create(rewriter, loc, cIterPtr, layoutBuilder.getValue(valCLayout));
          } else {
            accView = dView;
          }

          MmaAtomCall::create(rewriter, loc, mmaAtomVal, dView, aView, bView, accView);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Generated patterns
//===----------------------------------------------------------------------===//

#include "flydsl/Dialect/Fly/Transforms/LayoutLowering.cpp.inc"

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

class FlyLayoutLoweringPass
    : public mlir::fly::impl::FlyLayoutLoweringPassBase<FlyLayoutLoweringPass> {
public:
  using mlir::fly::impl::FlyLayoutLoweringPassBase<
      FlyLayoutLoweringPass>::FlyLayoutLoweringPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    getOperation()->walk([&](FunctionOpInterface funcOp) { lowerFuncIntTupleArgs(funcOp); });
    getOperation()->walk(
        [&](gpu::LaunchFuncOp launchOp) { lowerGpuLaunchFuncIntTupleOperands(launchOp); });

    RewritePatternSet patterns(context);

    patterns.add<GetLeafOpLowering, GetScalarLowering, GetShapeLowering, GetStrideLowering,
                 GetLayoutLowering, GetIterLowering>(context);

    patterns.add<SizeOpLowering, CosizeOpLowering>(context);
    patterns.add<SliceLowering, DiceOpLowering, Crd2IdxLowering>(context);

    patterns
        .add<IntTupleAddOpLowering, IntTupleSubOpLowering, IntTupleMulOpLowering,
             IntTupleDivOpLowering, IntTupleModOpLowering, IntTupleProductEachOpLowering,
             IntTupleProductOpLowering, ShapeDivOpLowering, CeilDivOpLowering, ElemLessOpLowering>(
            context);

    patterns.add<SelectOpLowering, GroupOpLowering>(context);
    patterns.add<AppendOpLowering, PrependOpLowering>(context);

    // Layout algebra lowerings
    patterns.add<CoalesceOpLowering, CompositionOpLowering, ComplementOpLowering>(context);
    patterns.add<LogicalDivideOpLowering, ZippedDivideOpLowering, TiledDivideOpLowering,
                 FlatDivideOpLowering, RightInverseOpLowering, RecastLayoutOpLowering>(context);

    patterns.add<LogicalProductOpLowering, ZippedProductOpLowering, TiledProductOpLowering,
                 FlatProductOpLowering, BlockedProductOpLowering, RakedProductOpLowering>(context);

    patterns.add<TiledCopyPartitionSrcOpLowering, TiledCopyPartitionDstOpLowering,
                 TiledMmaPartitionOpLowering>(context);
    patterns.add<ExpandCopyOpLowering, ExpandGemmOpLowering>(context);

    patterns.add<PrintOpLowering>(context);

    populateWithGenerated(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace impl {

std::unique_ptr<::mlir::Pass> createFlyLayoutLoweringPass() {
  return std::make_unique<FlyLayoutLoweringPass>();
}

} // namespace impl
