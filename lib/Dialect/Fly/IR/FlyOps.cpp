// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Core/Visitor.hpp"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/CoreBridge.h"
#include "flydsl/Dialect/Fly/Utils/LegacyLayout.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#define GET_OP_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyOps.cpp.inc"

#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::fly;

namespace {

/// The first error carried by a Core value, in Core traversal order.
template <class T> std::optional<core::ErrorInfo> getFirstError(const T &value) {
  std::optional<core::ErrorInfo> info;
  core::visit_value(
      value,
      [&](core::Leaf<core::Trait> leaf) {
        if (!info && leaf.isError())
          info = leaf.errorInfo();
      },
      [](const core::Pointer<core::Trait> &) {});
  return info;
}

/// Names the first error of an invalid Core `value` and where it was found.
template <class T> std::string invalidResultMessage(const T &value) {
  std::string message = "invalid layout algebra result";
  if (auto info = getFirstError(value))
    message += ": " + core::toString(*info);
  return message;
}

/// Assigns the `FlyType` of the Core `value`; an invalid value is diagnosed
/// with its Core error.
template <class FlyType, class T>
LogicalResult inferResultType(std::optional<Location> location, MLIRContext *context,
                              SmallVectorImpl<Type> &inferredReturnTypes, const T &value) {
  if (auto type = FlyType::get(context, value)) {
    inferredReturnTypes.assign({type});
    return success();
  }
  return emitOptionalError(location, invalidResultMessage(value));
}

} // namespace

#define FLY_INFER_RETURN_TYPES(OP)                                                                 \
  llvm::LogicalResult OP::inferReturnTypes(                                                        \
      mlir::MLIRContext *context, std::optional<::mlir::Location> location,                        \
      mlir::ValueRange operands, mlir::DictionaryAttr attributes, mlir::PropertyRef properties,    \
      mlir::RegionRange regions, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes)

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(MakeLayoutOp) {
  auto shapeType = dyn_cast<IntTupleType>(operands[0].getType());
  auto strideType = dyn_cast<IntTupleType>(operands[1].getType());
  if (!shapeType)
    return emitOptionalError(location, "MakeLayoutOp: expected IntTupleType for shape, got ",
                             operands[0].getType());
  if (!strideType)
    return emitOptionalError(location, "MakeLayoutOp: expected IntTupleType for stride, got ",
                             operands[1].getType());
  auto layoutAttr = LayoutAttr::get(context, shapeType.getAttr(), strideType.getAttr());
  if (!layoutAttr)
    return emitOptionalError(location, "invalid layout profile");
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeLayoutLikeOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutTy)
    return emitOptionalError(location, "MakeLayoutLikeOp: expected LayoutType, got ",
                             operands[0].getType());
  return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                     core::make_layout_like(layoutTy.getRef()));
}

FLY_INFER_RETURN_TYPES(MakeOrderedLayoutOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto orderTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!shapeTy)
    return emitOptionalError(location, "MakeOrderedLayoutOp: expected IntTupleType for shape, got ",
                             operands[0].getType());
  if (!orderTy)
    return emitOptionalError(location, "MakeOrderedLayoutOp: expected IntTupleType for order, got ",
                             operands[1].getType());
  return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                     core::make_ordered_layout(shapeTy.getRef(), orderTy.getRef()));
}

FLY_INFER_RETURN_TYPES(MakeComposedLayoutOp) {
  auto offsetTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!offsetTy)
    return emitOptionalError(location,
                             "MakeComposedLayoutOp: expected IntTupleType for offset, got ",
                             operands[1].getType());
  Attribute outerAttr;
  if (auto outerLayoutTy = dyn_cast<LayoutType>(operands[2].getType())) {
    outerAttr = outerLayoutTy.getAttr();
  } else if (auto outerComposedTy = dyn_cast<ComposedLayoutType>(operands[2].getType())) {
    outerAttr = outerComposedTy.getAttr();
  } else {
    return emitOptionalError(location,
                             "MakeComposedLayoutOp: expected LayoutType or ComposedLayoutType for "
                             "outer, got ",
                             operands[2].getType());
  }
  Attribute innerAttr = nullptr;
  if (auto innerLayoutTy = dyn_cast<LayoutType>(operands[0].getType())) {
    innerAttr = innerLayoutTy.getAttr();
  } else if (auto innerComposedTy = dyn_cast<ComposedLayoutType>(operands[0].getType())) {
    innerAttr = innerComposedTy.getAttr();
  } else if (auto innerSwizzleTy = dyn_cast<SwizzleType>(operands[0].getType())) {
    innerAttr = innerSwizzleTy.getAttr();
  } else if (auto innerCoordSwizzleTy = dyn_cast<CoordSwizzleType>(operands[0].getType())) {
    innerAttr = innerCoordSwizzleTy.getAttr();
  } else {
    return emitOptionalError(
        location,
        "MakeComposedLayoutOp: expected Layout/ComposedLayout/Swizzle/CoordSwizzle for inner, got ",
        operands[0].getType());
  }
  auto composedAttr = ComposedLayoutAttr::get(context, innerAttr, offsetTy.getAttr(), outerAttr);
  if (!composedAttr)
    return emitOptionalError(location, "invalid composed layout profile");
  inferredReturnTypes.assign({ComposedLayoutType::get(context, composedAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeIdentityLayoutOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!shapeTy)
    return emitOptionalError(location,
                             "MakeIdentityLayoutOp: expected IntTupleType for shape, got ",
                             operands[0].getType());
  return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                     core::make_identity_layout(shapeTy.getRef()));
}

FLY_INFER_RETURN_TYPES(MakeViewOp) {
  Type iterTy = operands[0].getType();
  Type layoutArgTy = operands[1].getType();

  Attribute layoutAttr =
      llvm::TypeSwitch<Type, Attribute>(layoutArgTy)
          .Case<LayoutType, ComposedLayoutType>([](auto type) { return type.getAttr(); })
          .Default([](Type) { return Attribute(); });
  if (!layoutAttr)
    return emitOptionalError(
        location, "MakeViewOp: expected LayoutType or ComposedLayoutType for operand #1, got ",
        layoutArgTy);

  if (auto intTupleTy = dyn_cast<IntTupleType>(iterTy)) {
    inferredReturnTypes.assign({CoordTensorType::get(intTupleTy.getAttr(), layoutAttr)});
    return success();
  } else if (auto ptrTy = dyn_cast<PointerType>(iterTy)) {
    inferredReturnTypes.assign(
        {MemRefType::get(ptrTy.getElemTy(), ptrTy.getAddressSpace(), layoutAttr,
                         ptrTy.getAlignment(), ptrTy.getSwizzle())});
    return success();
  } else {
    return emitOptionalError(
        location, "MakeViewOp: expected IntTupleType or PointerType for operand #0, got ", iterTy);
  }
}

FLY_INFER_RETURN_TYPES(MakeFragmentLayoutLikeOp) {
  auto fragment = [&](const auto &layout) -> LogicalResult {
    if (!core::shape(layout).isStatic())
      return emitOptionalError(location, "MakeFragmentLayoutLikeOp: expected a static shape");
    return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                       core::make_fragment_like(layout));
  };
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType>([&](auto type) { return fragment(type.getRef()); })
      .Case<fly::MemRefType, CoordTensorType>(
          [&](auto type) { return std::visit(fragment, type.getRef().layout()); })
      .Default([&](Type type) {
        return emitOptionalError(location, "MakeFragmentLayoutLikeOp: unsupported input ", type);
      });
}

FLY_INFER_RETURN_TYPES(MakeFragmentLikeOp) {
  TypeAttr dtypeAttr = properties ? properties.as<Properties *>()->dtype : TypeAttr();
  auto fragment = [&](Type elemTy, const auto &layout) -> LogicalResult {
    if (!elemTy)
      return emitOptionalError(location, "MakeFragmentLikeOp: dtype is required for a layout");
    if (!core::shape(layout).isStatic())
      return emitOptionalError(location, "MakeFragmentLikeOp: expected a static shape");
    auto result = core::make_fragment_like(layout);
    if (result.isError())
      return emitOptionalError(location, "MakeFragmentLikeOp: ", invalidResultMessage(result));
    inferredReturnTypes.assign(
        {fly::MemRefType::get(elemTy, AddressSpaceAttr::get(context, AddressSpace::Register),
                              LayoutAttr::get(context, result))});
    return success();
  };
  Type dtype = dtypeAttr ? dtypeAttr.getValue() : Type();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType>(
          [&](auto type) { return fragment(dtype, type.getRef()); })
      .Case<fly::MemRefType, CoordTensorType>([&](auto type) {
        Type elemTy = dtype;
        if constexpr (std::is_same_v<decltype(type), fly::MemRefType>)
          elemTy = dtype ? dtype : type.getElemTy();
        return std::visit([&](const auto &layout) { return fragment(elemTy, layout); },
                          type.getRef().layout());
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "MakeFragmentLikeOp: unsupported input ", type);
      });
}

//===----------------------------------------------------------------------===//
// Extractors
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(GetScalarOp) {
  auto intTupleType = dyn_cast<IntTupleType>(operands[0].getType());
  if (!intTupleType)
    return emitOptionalError(location, "GetScalarOp: expected IntTupleType, got ",
                             operands[0].getType());
  IntTupleAttr scalarAttr = intTupleType.getAttr();
  while (!scalarAttr.isLeaf() && scalarAttr.rank() == 1)
    scalarAttr = scalarAttr.at(0);
  if (!scalarAttr.isLeaf())
    return emitOptionalError(location, "GetScalarOp: expected a scalar IntTuple, got ",
                             intTupleType);
  auto intAttr = scalarAttr.getLeaf().getScale();
  if (intAttr.isRatio())
    return emitOptionalError(location, "GetScalarOp: a ratio is not a scalar");
  auto intType = IntegerType::get(context, intAttr.getWidth());
  if (intAttr.getRef().hasFastDivMod())
    inferredReturnTypes.assign({FastDivModType::get(context, intType)});
  else
    inferredReturnTypes.assign({intType});
  return success();
}

OpFoldResult GetScalarOp::fold(FoldAdaptor) {
  auto profile = cast<IntTupleType>(getIntTuple().getType()).getRef().asRef();
  while (!profile.isLeaf() && profile.rank() == 1)
    profile = profile.at(0);
  if (!profile.isLeaf())
    return {};
  auto leaf = profile.getLeaf().scale();
  if (!leaf.isSInt()) {
    auto *producer = getIntTuple().getDefiningOp();
    if (producer && isa<MakeIntTupleOp, MakeShapeOp, MakeStrideOp, MakeCoordOp>(producer) &&
        producer->getNumOperands() == 1 && producer->getOperand(0).getType() == getType())
      return producer->getOperand(0);
    return {};
  }
  return IntegerAttr::get(getResult().getType(), leaf.staticValue());
}

FLY_INFER_RETURN_TYPES(GetLeavesOp) {
  // The scalars Core binds the profile to, with the Fly fast-divmod handle.
  OpBuilder builder(context);
  auto ctx = makeDefaultEmitContext(builder, location.value_or(builder.getUnknownLoc()));
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType, ComposedLayoutType, TileType, CoordTensorType,
            fly::MemRefType>([&](auto type) {
        llvm::append_range(inferredReturnTypes, core::get_binding_types(type.getRef(), ctx));
        return success();
      })
      .Default([&](Type type) {
        return emitOptionalError(location,
                                 "GetLeavesOp: expected a type with a scalar profile, got ", type);
      });
}

//===----------------------------------------------------------------------===//
// Fast division
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(MakeFastDivModOp) {
  inferredReturnTypes.assign(
      {FastDivModType::get(context, cast<IntegerType>(operands[0].getType()))});
  return success();
}

FLY_INFER_RETURN_TYPES(FastDivModOp) {
  auto divisorType = cast<FastDivModType>(operands[1].getType()).getDivisorType();
  if (operands[0].getType() != divisorType)
    return emitOptionalError(location, "FastDivModOp: expected a dividend of type ", divisorType,
                             ", got ", operands[0].getType());
  inferredReturnTypes.assign({divisorType, divisorType});
  return success();
}

FLY_INFER_RETURN_TYPES(FastDivModGetDivisorOp) {
  inferredReturnTypes.assign({cast<FastDivModType>(operands[0].getType()).getDivisorType()});
  return success();
}

FLY_INFER_RETURN_TYPES(GetShapeOp) {
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>([&](auto type) {
        return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                             core::shape(type.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "GetShapeOp: unsupported input ", type);
      });
}

FLY_INFER_RETURN_TYPES(GetStrideOp) {
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, fly::MemRefType, CoordTensorType>([&](auto type) {
        return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                             core::stride(type.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "GetStrideOp: unsupported input ", type);
      });
}

FLY_INFER_RETURN_TYPES(GetLayoutOp) {
  Attribute layoutAttr;
  if (auto memrefTy = dyn_cast<MemRefType>(operands[0].getType())) {
    layoutAttr = memrefTy.getLayout();
  } else if (auto coordTensorTy = dyn_cast<CoordTensorType>(operands[0].getType())) {
    layoutAttr = coordTensorTy.getLayout();
  } else {
    return emitOptionalError(location, "GetLayoutOp: expected TensorLikeType, got ",
                             operands[0].getType());
  }
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr)) {
    inferredReturnTypes.assign({LayoutType::get(context, layout)});
    return success();
  } else if (auto composed = dyn_cast<ComposedLayoutAttr>(layoutAttr)) {
    inferredReturnTypes.assign({ComposedLayoutType::get(context, composed)});
    return success();
  }
  return emitOptionalError(location, "GetLayoutOp: unsupported layout attribute type ", layoutAttr);
}

FLY_INFER_RETURN_TYPES(GetIterOp) {
  if (auto memrefTy = dyn_cast<MemRefType>(operands[0].getType())) {
    inferredReturnTypes.assign({PointerType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(),
                                                 memrefTy.getAlignment(), memrefTy.getSwizzle())});
    return success();
  }
  if (auto coordTensorTy = dyn_cast<CoordTensorType>(operands[0].getType())) {
    inferredReturnTypes.assign({IntTupleType::get(coordTensorTy.getBase())});
    return success();
  }
  return emitOptionalError(location, "GetIterOp: expected TensorLikeType, got ",
                           operands[0].getType());
}

FLY_INFER_RETURN_TYPES(ComposedGetInnerOp) {
  auto inputTy = dyn_cast<ComposedLayoutType>(operands[0].getType());
  if (!inputTy)
    return emitOptionalError(location, "ComposedGetInnerOp: expected ComposedLayoutType, got ",
                             operands[0].getType());
  auto innerAttr = inputTy.getAttr().getInner();
  if (auto swizzleAttr = dyn_cast<SwizzleAttr>(innerAttr)) {
    inferredReturnTypes.assign({SwizzleType::get(context, swizzleAttr)});
    return success();
  } else if (auto coordSwizzleAttr = dyn_cast<CoordSwizzleAttr>(innerAttr)) {
    inferredReturnTypes.assign({CoordSwizzleType::get(context, coordSwizzleAttr)});
    return success();
  } else if (auto layoutAttr = dyn_cast<LayoutAttr>(innerAttr)) {
    if (!layoutAttr)
      return emitOptionalError(location, "invalid layout profile");
    inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
    return success();
  } else if (auto composedLayoutAttr = dyn_cast<ComposedLayoutAttr>(innerAttr)) {
    inferredReturnTypes.assign({ComposedLayoutType::get(context, composedLayoutAttr)});
    return success();
  }
  return emitOptionalError(location, "ComposedGetInnerOp: unrecognized inner attribute type");
}

FLY_INFER_RETURN_TYPES(ComposedGetOffsetOp) {
  auto inputTy = dyn_cast<ComposedLayoutType>(operands[0].getType());
  if (!inputTy)
    return emitOptionalError(location, "ComposedGetOffsetOp: expected ComposedLayoutType, got ",
                             operands[0].getType());
  inferredReturnTypes.assign({IntTupleType::get(inputTy.getAttr().getOffset())});
  return success();
}

FLY_INFER_RETURN_TYPES(ComposedGetOuterOp) {
  auto inputTy = dyn_cast<ComposedLayoutType>(operands[0].getType());
  if (!inputTy)
    return emitOptionalError(location, "ComposedGetOuterOp: expected ComposedLayoutType, got ",
                             operands[0].getType());
  return std::visit(
      [&](const auto &outer) {
        using Outer = std::decay_t<decltype(outer)>;
        using ResultType = std::conditional_t<std::is_same_v<Outer, core::Layout<core::Trait>>,
                                              LayoutType, ComposedLayoutType>;
        return inferResultType<ResultType>(location, context, inferredReturnTypes, outer);
      },
      inputTy.getRef().outer());
}

//===----------------------------------------------------------------------===//
// IntTuple operations
//===----------------------------------------------------------------------===//

#define FLY_INFER_INT_TUPLE_BINARY(OP, EXPR)                                                       \
  FLY_INFER_RETURN_TYPES(OP) {                                                                     \
    auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());                                    \
    auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());                                    \
    if (!lhsTy || !rhsTy)                                                                          \
      return emitOptionalError(location, #OP ": expected IntTupleType operands");                  \
    const auto &lhs = lhsTy.getRef();                                                              \
    const auto &rhs = rhsTy.getRef();                                                              \
    return inferResultType<IntTupleType>(location, context, inferredReturnTypes, EXPR);            \
  }

FLY_INFER_INT_TUPLE_BINARY(IntTupleAddOp, lhs + rhs)
FLY_INFER_INT_TUPLE_BINARY(IntTupleSubOp, lhs - rhs)
FLY_INFER_INT_TUPLE_BINARY(IntTupleMulOp, lhs *rhs)
FLY_INFER_INT_TUPLE_BINARY(IntTupleDivOp, lhs / rhs)
FLY_INFER_INT_TUPLE_BINARY(IntTupleModOp, lhs % rhs)
FLY_INFER_INT_TUPLE_BINARY(IntTupleProductLikeOp, core::product_like(lhs, rhs))
FLY_INFER_INT_TUPLE_BINARY(ShapeDivOp, core::shape_div(lhs, rhs))
FLY_INFER_INT_TUPLE_BINARY(CeilDivOp, core::ceil_div(lhs, rhs))

#undef FLY_INFER_INT_TUPLE_BINARY

FLY_INFER_RETURN_TYPES(IntTupleProductOp) {
  auto inputTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!inputTy)
    return emitOptionalError(location, "IntTupleProductOp: expected IntTupleType");
  return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                       core::product(inputTy.getRef()));
}

FLY_INFER_RETURN_TYPES(IntTupleProductEachOp) {
  auto inputTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!inputTy)
    return emitOptionalError(location, "IntTupleProductEachOp: expected IntTupleType");
  return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                       core::product_each(inputTy.getRef()));
}

//===----------------------------------------------------------------------===//
// IntTupleLike operations
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(GetOp) {
  ArrayRef<int32_t> mode = properties.as<Properties *>()->mode;
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(
          [&](auto type) -> LogicalResult {
            using Ref = std::decay_t<decltype(type.getRef())>;
            Ref value = type.getRef();
            for (int32_t index : mode) {
              if (index < 0 || index >= value.rank())
                return emitOptionalError(location, "GetOp: mode index is outside the input rank");
              Ref child = core::get(value, index);
              value = std::move(child);
            }
            return inferResultType<decltype(type)>(location, context, inferredReturnTypes, value);
          })
      .Default([&](Type type) {
        return emitOptionalError(location, "GetOp: unsupported input type ", type);
      });
}

FLY_INFER_RETURN_TYPES(TakeOp) {
  int32_t begin = properties.as<Properties *>()->begin.getInt();
  int32_t end = properties.as<Properties *>()->end.getInt();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(
          [&](auto type) -> LogicalResult {
            auto value = type.getRef();
            int32_t rank = value.rank();
            if (begin < 0 || begin > rank || end < -1 || (end != -1 && (end < begin || end > rank)))
              return emitOptionalError(location, "TakeOp: invalid range for the input rank");
            return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                                   core::legacy_take(value, begin, end));
          })
      .Default([&](Type type) {
        return emitOptionalError(location, "TakeOp: unsupported input type ", type);
      });
}

FLY_INFER_RETURN_TYPES(SelectOp) {
  ArrayRef<int32_t> indices = properties.as<Properties *>()->indices.asArrayRef();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(
          [&](auto type) -> LogicalResult {
            auto value = type.getRef();
            if (llvm::any_of(indices, [&](int32_t i) { return i < 0 || i >= value.rank(); }))
              return emitOptionalError(location, "SelectOp: index is outside the input rank");
            return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                                   core::legacy_select(value, indices));
          })
      .Default([&](Type type) {
        return emitOptionalError(location, "SelectOp: unsupported input type ", type);
      });
}

FLY_INFER_RETURN_TYPES(GroupOp) {
  int32_t begin = properties.as<Properties *>()->begin.getInt();
  int32_t end = properties.as<Properties *>()->end.getInt();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(
          [&](auto type) -> LogicalResult {
            auto value = type.getRef();
            int32_t rank = value.rank();
            if (begin < 0 || begin > rank || end < -1 || (end != -1 && (end < begin || end > rank)))
              return emitOptionalError(location, "GroupOp: invalid range for the input rank");
            return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                                   core::group(value, begin, end));
          })
      .Default([&](Type type) {
        return emitOptionalError(location, "GroupOp: unsupported input type ", type);
      });
}

/// append and prepend share their operand dispatch; `n` is -1 when absent.
template <bool Append>
static LogicalResult inferAppendOrPrepend(std::optional<Location> location, MLIRContext *context,
                                          ValueRange operands, int32_t n,
                                          SmallVectorImpl<Type> &inferredReturnTypes) {
  auto apply = [&](const auto &tuple, const auto &elem) {
    if constexpr (Append)
      return n == -1 ? core::append(tuple, elem) : core::append(tuple, elem, n);
    else
      return n == -1 ? core::prepend(tuple, elem) : core::prepend(tuple, elem, n);
  };
  Type elemTy = operands[1].getType();
  if (auto nested = dyn_cast<NestedTypeInterface>(operands[0].getType());
      nested && n != -1 && n < nested.rank())
    return emitOptionalError(location, "expected n (", n, ") to be at least the input rank (",
                             nested.rank(), ")");
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case([&](IntTupleType tuple) -> LogicalResult {
        auto elem = dyn_cast<IntTupleType>(elemTy);
        if (!elem)
          return emitOptionalError(location, "expected an IntTupleType element, got ", elemTy);
        return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                             apply(tuple.getRef(), elem.getRef()));
      })
      .template Case<LayoutType, ComposedLayoutType>([&](auto layout) -> LogicalResult {
        auto elem = dyn_cast<LayoutType>(elemTy);
        if (!elem)
          return emitOptionalError(location, "expected a LayoutType element, got ", elemTy);
        return inferResultType<decltype(layout)>(location, context, inferredReturnTypes,
                                                 apply(layout.getRef(), elem.getRef()));
      })
      .Default(
          [&](Type type) { return emitOptionalError(location, "unsupported input type ", type); });
}

FLY_INFER_RETURN_TYPES(AppendOp) {
  IntegerAttr nAttr = properties ? properties.as<Properties *>()->n : IntegerAttr();
  int32_t n = nAttr ? static_cast<int32_t>(nAttr.getInt()) : -1;
  return inferAppendOrPrepend<true>(location, context, operands, n, inferredReturnTypes);
}

FLY_INFER_RETURN_TYPES(PrependOp) {
  IntegerAttr nAttr = properties ? properties.as<Properties *>()->n : IntegerAttr();
  int32_t n = nAttr ? static_cast<int32_t>(nAttr.getInt()) : -1;
  return inferAppendOrPrepend<false>(location, context, operands, n, inferredReturnTypes);
}

FLY_INFER_RETURN_TYPES(SliceOp) {
  auto coordTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!coordTy)
    return emitOptionalError(location, "SliceOp: expected IntTupleType for coord, got ",
                             operands[1].getType());
  const auto &coord = coordTy.getRef();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(
          [&](auto type) -> LogicalResult {
            auto value = type.getRef();
            if (!core::weakly_congruent(coord, core::shape(value)))
              return emitOptionalError(
                  location, "SliceOp: coordinate profile does not match the input shape");
            return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                                   core::legacy_slice(coord, value));
          })
      .Default([&](Type type) {
        return emitOptionalError(location, "SliceOp: unsupported input type ", type);
      });
}

FLY_INFER_RETURN_TYPES(DiceOp) {
  auto coordTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!coordTy)
    return emitOptionalError(location, "DiceOp: expected IntTupleType for coord, got ",
                             operands[1].getType());
  const auto &coord = coordTy.getRef();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType>([&](auto type) -> LogicalResult {
        auto value = type.getRef();
        if (!core::weakly_congruent(coord, core::shape(value)))
          return emitOptionalError(location,
                                   "DiceOp: coordinate profile does not match the input shape");
        return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                               core::legacy_dice(coord, value));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "DiceOp: unsupported input type ", type);
      });
}

//===----------------------------------------------------------------------===//
// LayoutLike operations
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(SizeOp) {
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(
          [&](auto type) {
            return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                                 core::size(type.getRef()));
          })
      .Default([&](Type type) {
        return emitOptionalError(location, "SizeOp: unsupported input type ", type);
      });
}

/// cosize/coshape/coprofile of a layout, or of the layout of a tensor.
#define FLY_INFER_LAYOUT_QUERY(OP, FN)                                                             \
  FLY_INFER_RETURN_TYPES(OP) {                                                                     \
    auto query = [&](const auto &layout) {                                                         \
      return inferResultType<IntTupleType>(location, context, inferredReturnTypes,                 \
                                           core::FN(layout));                                      \
    };                                                                                             \
    return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())                            \
        .Case<LayoutType, ComposedLayoutType>([&](auto type) { return query(type.getRef()); })     \
        .Case<fly::MemRefType, CoordTensorType>(                                                   \
            [&](auto type) { return std::visit(query, type.getRef().layout()); })                  \
        .Default([&](Type type) {                                                                  \
          return emitOptionalError(location, #OP ": unsupported input type ", type);               \
        });                                                                                        \
  }

FLY_INFER_LAYOUT_QUERY(CoprofileOp, coprofile)
FLY_INFER_LAYOUT_QUERY(CoshapeOp, coshape)
FLY_INFER_LAYOUT_QUERY(CosizeOp, cosize)

#undef FLY_INFER_LAYOUT_QUERY

FLY_INFER_RETURN_TYPES(Crd2IdxOp) {
  auto coordTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!coordTy)
    return emitOptionalError(location, "Crd2IdxOp: expected IntTupleType for coord");
  const auto &coord = coordTy.getRef();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[1].getType())
      .Case<LayoutType, ComposedLayoutType>([&](auto type) {
        return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                             core::crd2idx(coord, type.getRef()));
      })
      .Case<SwizzleType, CoordSwizzleType>([&](auto type) {
        return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                             core::swizzle_apply(type.getRef(), coord));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "Crd2IdxOp: unsupported layout type ", type);
      });
}

/// idx2crd/get_flat_coord/get_1d_coord of an index through a plain layout.
#define FLY_INFER_LAYOUT_COORD(OP, MEMBER)                                                         \
  FLY_INFER_RETURN_TYPES(OP) {                                                                     \
    auto indexTy = dyn_cast<IntTupleType>(operands[0].getType());                                  \
    auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());                                   \
    if (!indexTy || !layoutTy)                                                                     \
      return emitOptionalError(location, #OP ": expected IntTupleType and LayoutType operands");   \
    return inferResultType<IntTupleType>(location, context, inferredReturnTypes,                   \
                                         layoutTy.getRef().MEMBER(indexTy.getRef()));              \
  }

FLY_INFER_LAYOUT_COORD(Idx2CrdOp, get_hier_coord)
FLY_INFER_LAYOUT_COORD(GetFlatCoordOp, get_flat_coord)
FLY_INFER_LAYOUT_COORD(Get1DCoordOp, get_1d_coord)

#undef FLY_INFER_LAYOUT_COORD

FLY_INFER_RETURN_TYPES(CoalesceOp) {
  IntTupleType profileTy;
  if (operands.size() > 1 && operands[1]) {
    profileTy = dyn_cast<IntTupleType>(operands[1].getType());
    if (!profileTy)
      return emitOptionalError(location, "CoalesceOp: expected IntTupleType for pattern");
  }
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>([&](auto type) {
        auto value = type.getRef();
        if (profileTy)
          return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                                 core::coalesce(value, profileTy.getRef()));
        return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                               core::coalesce(value));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "CoalesceOp: unsupported input type ", type);
      });
}

/// Layout-like `lhs` combined with a layout or tile `rhs` through `fn`.
template <class Fn>
static LogicalResult inferWithTiler(std::optional<Location> location, MLIRContext *context,
                                    ValueRange operands, SmallVectorImpl<Type> &inferredReturnTypes,
                                    Fn fn) {
  Type rhsTy = operands[1].getType();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(
          [&](auto lhs) -> LogicalResult {
            return llvm::TypeSwitch<Type, LogicalResult>(rhsTy)
                .Case<LayoutType, TileType>([&](auto rhs) -> LogicalResult {
                  // A tuple tiler cannot have more modes than the input.
                  if constexpr (std::is_same_v<decltype(rhs), TileType>) {
                    auto rank = lhs.getRef().rank();
                    if (!rhs.getRef().isLeaf() && rhs.rank() > rank)
                      return emitOptionalError(location, "expected the tiler rank (", rhs.rank(),
                                               ") to not exceed the input rank (", rank, ")");
                  }
                  return inferResultType<decltype(lhs)>(location, context, inferredReturnTypes,
                                                        fn(lhs.getRef(), rhs.getRef()));
                })
                .Default([&](Type type) {
                  return emitOptionalError(location, "expected LayoutType or TileType, got ", type);
                });
          })
      .Default(
          [&](Type type) { return emitOptionalError(location, "unsupported layout type ", type); });
}

#define FLY_INFER_WITH_TILER(OP, FN)                                                               \
  FLY_INFER_RETURN_TYPES(OP) {                                                                     \
    return inferWithTiler(location, context, operands, inferredReturnTypes,                        \
                          [](const auto &lhs, const auto &rhs) { return core::FN(lhs, rhs); });    \
  }

FLY_INFER_WITH_TILER(CompositionOp, composition)
FLY_INFER_WITH_TILER(LogicalDivideOp, logical_divide)
FLY_INFER_WITH_TILER(ZippedDivideOp, zipped_divide)
FLY_INFER_WITH_TILER(TiledDivideOp, tiled_divide)
FLY_INFER_WITH_TILER(FlatDivideOp, flat_divide)

#undef FLY_INFER_WITH_TILER

FLY_INFER_RETURN_TYPES(ComplementOp) {
  IntTupleType codomainTy;
  if (operands.size() > 1 && operands[1]) {
    codomainTy = dyn_cast<IntTupleType>(operands[1].getType());
    if (!codomainTy)
      return emitOptionalError(location, "ComplementOp: expected IntTupleType for codomain_size");
  }
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType>([&](auto type) {
        if (codomainTy)
          return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                             core::complement(type.getRef(), codomainTy.getRef()));
        return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                           core::complement(type.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "ComplementOp: unsupported input type ", type);
      });
}

FLY_INFER_RETURN_TYPES(RightInverseOp) {
  auto type = dyn_cast<LayoutType>(operands[0].getType());
  if (!type)
    return emitOptionalError(location, "RightInverseOp: unsupported input type ",
                             operands[0].getType());
  return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                     core::right_inverse(type.getRef()));
}

FLY_INFER_RETURN_TYPES(LeftInverseOp) {
  auto type = dyn_cast<LayoutType>(operands[0].getType());
  if (!type)
    return emitOptionalError(location, "LeftInverseOp: unsupported input type ",
                             operands[0].getType());
  return inferResultType<LayoutType>(location, context, inferredReturnTypes,
                                     core::left_inverse(type.getRef()));
}

#define FLY_INFER_LAYOUT_PRODUCT(OP, FN)                                                           \
  FLY_INFER_RETURN_TYPES(OP) {                                                                     \
    auto tileTy = dyn_cast<LayoutType>(operands[1].getType());                                     \
    if (!tileTy)                                                                                   \
      return emitOptionalError(location, #OP ": expected LayoutType for tile");                    \
    return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())                            \
        .Case<LayoutType, ComposedLayoutType>([&](auto type) {                                     \
          return inferResultType<decltype(type)>(location, context, inferredReturnTypes,           \
                                                 core::FN(type.getRef(), tileTy.getRef()));        \
        })                                                                                         \
        .Default([&](Type type) {                                                                  \
          return emitOptionalError(location, #OP ": unsupported input type ", type);               \
        });                                                                                        \
  }

FLY_INFER_LAYOUT_PRODUCT(LogicalProductOp, logical_product)
FLY_INFER_LAYOUT_PRODUCT(ZippedProductOp, zipped_product)
FLY_INFER_LAYOUT_PRODUCT(TiledProductOp, tiled_product)
FLY_INFER_LAYOUT_PRODUCT(FlatProductOp, flat_product)
FLY_INFER_LAYOUT_PRODUCT(BlockedProductOp, blocked_product)
FLY_INFER_LAYOUT_PRODUCT(RakedProductOp, raked_product)

#undef FLY_INFER_LAYOUT_PRODUCT

FLY_INFER_RETURN_TYPES(RecastLayoutOp) {
  int32_t newTypeBits = properties.as<Properties *>()->new_type_bits.getInt();
  int32_t oldTypeBits = properties.as<Properties *>()->old_type_bits.getInt();
  if (newTypeBits <= 0 || oldTypeBits <= 0)
    return emitOptionalError(location, "RecastLayoutOp: type bits must be positive");
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType>([&](auto type) {
        return inferResultType<decltype(type)>(
            location, context, inferredReturnTypes,
            core::recast_layout(newTypeBits, oldTypeBits, type.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "RecastLayoutOp: unsupported input type ", type);
      });
}

FLY_INFER_RETURN_TYPES(TileToShapeOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[1].getType());
  auto orderTy = dyn_cast<IntTupleType>(operands[2].getType());
  if (!shapeTy || !orderTy)
    return emitOptionalError(location, "TileToShapeOp: expected IntTupleType shapes");
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<LayoutType, ComposedLayoutType>([&](auto type) -> LogicalResult {
        // The block is padded up to the target rank, never truncated.
        auto rank = type.getRef().rank();
        if (rank > shapeTy.getRef().rank())
          return emitOptionalError(location, "TileToShapeOp: expected the block rank (", rank,
                                   ") to not exceed the target shape rank (",
                                   shapeTy.getRef().rank(), ")");
        return inferResultType<decltype(type)>(
            location, context, inferredReturnTypes,
            core::tile_to_shape(type.getRef(), shapeTy.getRef(), orderTy.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "TileToShapeOp: unsupported block type ", type);
      });
}

//===----------------------------------------------------------------------===//
// Atom and Tiled Mma/Copy ops
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(AtomSetValueOp) {
  auto atomTy = operands[0].getType();
  inferredReturnTypes.assign({atomTy});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeTiledCopyOp) {
  auto copyAtomTy = operands[0].getType();
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  auto tileTy = dyn_cast<TileType>(operands[2].getType());
  if (!layoutTy)
    return emitOptionalError(location, "MakeTiledCopyOp: expected LayoutType for operand #1, got ",
                             operands[1].getType());
  if (!layoutTy.isStatic()) {
    return emitOptionalError(location, "MakeTiledCopyOp: LayoutType is not static, got ",
                             operands[1].getType());
  }
  if (!tileTy)
    return emitOptionalError(location, "MakeTiledCopyOp: expected TileType for operand #2, got ",
                             operands[2].getType());
  if (!tileTy.isStatic())
    return emitOptionalError(location, "MakeTiledCopyOp: TileType is not static, got ", tileTy);

  auto tiledCopyTy = TiledCopyType::get(context, copyAtomTy, layoutTy, tileTy);
  if (!tiledCopyTy.isValidRef())
    return emitOptionalError(location, "MakeTiledCopyOp: thread-value layout ", layoutTy,
                             " does not tile whole atoms of ", copyAtomTy);
  inferredReturnTypes.assign({tiledCopyTy});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeTiledMmaOp) {
  auto mmaAtomTy = operands[0].getType();
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!layoutTy)
    return emitOptionalError(location, "MakeTiledMmaOp: expected LayoutType for operand #1, got ",
                             operands[1].getType());
  if (!layoutTy.isStatic()) {
    return emitOptionalError(location, "MakeTiledMmaOp: LayoutType is not static, got ",
                             operands[1].getType());
  }

  TileType tileTy = TiledMmaType::getDefaultPermutationMNK(context);
  if (operands.size() > 2 && operands[2]) {
    tileTy = dyn_cast<TileType>(operands[2].getType());
    if (!tileTy)
      return emitOptionalError(location, "MakeTiledMmaOp: expected TileType for operand #2, got ",
                               operands[2].getType());
    if (!tileTy.isStatic())
      return emitOptionalError(location, "MakeTiledMmaOp: TileType is not static, got ", tileTy);
  }

  auto tiledMmaTy = TiledMmaType::get(context, mmaAtomTy, layoutTy, tileTy);
  inferredReturnTypes.assign({tiledMmaTy});
  return success();
}

FLY_INFER_RETURN_TYPES(GetCopyAtomOp) {
  inferredReturnTypes.assign({cast<TiledCopyType>(operands[0].getType()).getCopyAtom()});
  return success();
}

FLY_INFER_RETURN_TYPES(GetMmaAtomOp) {
  inferredReturnTypes.assign({cast<TiledMmaType>(operands[0].getType()).getMmaAtom()});
  return success();
}

/// partition_S/partition_D of a tensor for one thread of a tiled copy.
template <bool Source>
static LogicalResult inferTiledCopyPartition(std::optional<Location> location, MLIRContext *context,
                                             ValueRange operands,
                                             SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tiledCopyTy = dyn_cast<TiledCopyType>(operands[0].getType());
  auto thrIdxTy = dyn_cast<IntTupleType>(operands[2].getType());
  if (!tiledCopyTy || !isa<CopyAtomType>(tiledCopyTy.getCopyAtom()) || !thrIdxTy)
    return emitOptionalError(location, "expected a TiledCopyType and an IntTupleType thread index");
  if (!tiledCopyTy.isValidRef())
    return emitOptionalError(location, "invalid tiled copy ", tiledCopyTy);
  auto tiled = tiledCopyTy.getRef();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[1].getType())
      .Case<fly::MemRefType, CoordTensorType>([&](auto type) {
        auto tensor = type.getRef();
        // CuTe C++ requires the tensor to cover every mode of the copy tiler.
        if (core::rank(tensor) < tiled.tilerMN().rank())
          return emitOptionalError(location, "tensor rank ", core::rank(tensor),
                                   " is less than the tiled copy tiler rank ",
                                   tiled.tilerMN().rank());
        if constexpr (Source)
          return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                                 tiled.partition_S(tensor, thrIdxTy.getRef()));
        else
          return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                                 tiled.partition_D(tensor, thrIdxTy.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "expected a tensor to partition, got ", type);
      });
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionSrcOp) {
  return inferTiledCopyPartition<true>(location, context, operands, inferredReturnTypes);
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionDstOp) {
  return inferTiledCopyPartition<false>(location, context, operands, inferredReturnTypes);
}

FLY_INFER_RETURN_TYPES(TiledCopyRetileOp) {
  auto tiledCopyTy = dyn_cast<TiledCopyType>(operands[0].getType());
  auto memrefTy = dyn_cast<fly::MemRefType>(operands[1].getType());
  if (!tiledCopyTy || !isa<CopyAtomType>(tiledCopyTy.getCopyAtom()) || !memrefTy)
    return emitOptionalError(location, "TiledCopyRetileOp: expected a TiledCopyType and a memref");
  if (!tiledCopyTy.isValidRef())
    return emitOptionalError(location, "TiledCopyRetileOp: invalid tiled copy ", tiledCopyTy);
  auto tiled = tiledCopyTy.getRef();
  auto memref = memrefTy.getRef();
  if (core::rank(memref) <= tiled.tilerMN().rank())
    return emitOptionalError(location, "retile expects a (V, rest...) fragment of rank at least ",
                             tiled.tilerMN().rank() + 1);
  auto result = tiled.retile(memref);
  return inferResultType<fly::MemRefType>(location, context, inferredReturnTypes, result);
}

/// partition_A/B/C of `tensor` for thread `thr` of a tiled MMA; D uses C.
template <class Cat, class Engine>
static auto partitionMma(const core::ThrMMA<Cat> &thr, MmaOperand operand,
                         const core::Tensor<Cat, Engine> &tensor) {
  switch (operand) {
  case MmaOperand::A:
    return thr.partition_A(tensor);
  case MmaOperand::B:
    return thr.partition_B(tensor);
  case MmaOperand::C:
  case MmaOperand::D:
    return thr.partition_C(tensor);
  }
  llvm_unreachable("unknown MMA operand");
}

FLY_INFER_RETURN_TYPES(TiledMmaPartitionOp) {
  auto operandId = properties.as<Properties *>()->operand_id.getValue();
  auto tiledMmaTy = dyn_cast<TiledMmaType>(operands[0].getType());
  auto thrIdxTy = dyn_cast<IntTupleType>(operands[2].getType());
  if (!tiledMmaTy || !isa<MmaAtomType>(tiledMmaTy.getMmaAtom()) || !thrIdxTy)
    return emitOptionalError(location, "TiledMmaPartitionOp: expected a TiledMmaType and an "
                                       "IntTupleType thread index");
  auto thr = tiledMmaTy.getRef().get_slice(thrIdxTy.getRef());
  return llvm::TypeSwitch<Type, LogicalResult>(operands[1].getType())
      .Case<fly::MemRefType, CoordTensorType>([&](auto type) {
        return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                               partitionMma(thr, operandId, type.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "TiledMmaPartitionOp: expected a tensor, got ", type);
      });
}

FLY_INFER_RETURN_TYPES(TiledMmaPartitionShapeOp) {
  auto operandId = properties.as<Properties *>()->operand_id.getValue();
  auto tiledMmaTy = dyn_cast<TiledMmaType>(operands[0].getType());
  auto shapeTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!tiledMmaTy || !isa<MmaAtomType>(tiledMmaTy.getMmaAtom()) || !shapeTy)
    return emitOptionalError(location, "TiledMmaPartitionShapeOp: expected a TiledMmaType and an "
                                       "IntTupleType shape");
  auto tiled = tiledMmaTy.getRef();
  const auto &shape = shapeTy.getRef();
  switch (operandId) {
  case MmaOperand::A:
    return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                         core::partition_shape_A(tiled, shape));
  case MmaOperand::B:
    return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                         core::partition_shape_B(tiled, shape));
  case MmaOperand::C:
  case MmaOperand::D:
    return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                         core::partition_shape_C(tiled, shape));
  }
  llvm_unreachable("unknown MMA operand");
}

FLY_INFER_RETURN_TYPES(MmaMakeFragmentOp) {
  auto operandId = properties.as<Properties *>()->operand_id.getValue();
  auto stagesAttr = properties.as<Properties *>()->stages;
  auto tiledMmaTy = dyn_cast<TiledMmaType>(operands[0].getType());
  auto memrefTy = dyn_cast<fly::MemRefType>(operands[1].getType());
  if (!tiledMmaTy || !memrefTy)
    return emitOptionalError(location, "MmaMakeFragmentOp: expected a TiledMmaType and a memref");
  auto mmaAtom = dyn_cast<MmaAtomType>(tiledMmaTy.getMmaAtom());
  if (!mmaAtom)
    return emitOptionalError(location,
                             "MmaMakeFragmentOp: TiledMmaType's mma atom is not a MmaAtomType");

  Type elemTy;
  switch (operandId) {
  case MmaOperand::A:
    elemTy = mmaAtom.getValTypeA();
    break;
  case MmaOperand::B:
    elemTy = mmaAtom.getValTypeB();
    break;
  case MmaOperand::C:
    elemTy = mmaAtom.getValTypeC();
    break;
  case MmaOperand::D:
    elemTy = mmaAtom.getValTypeD();
    break;
  }

  auto tensor = memrefTy.getRef();
  if (stagesAttr) {
    // A staged fragment adds a trailing stage mode of `stages` buffers.
    int32_t stages = static_cast<int32_t>(stagesAttr.getInt());
    auto *layout = std::get_if<core::Layout<core::Trait>>(&tensor.layout());
    if (stages <= 0 || !layout || layout->rank() < 2)
      return emitOptionalError(location, "MmaMakeFragmentOp: stages requires a positive count and "
                                         "a layout with at least two modes");
    auto stageShape =
        core::IntTuple<core::Trait>::fromLeaf(core::Leaf<core::Trait>::getSInt(stages));
    auto staged = core::make_layout(core::append(core::take(layout->shape(), 0, 2), stageShape),
                                    core::append(core::take(layout->stride(), 0, 2),
                                                 IntTupleAttr::getLeafDynamic(context).getRef()));
    tensor = core::MemRef<core::Trait>(tensor.data(), std::move(staged));
  }

  auto thr = tiledMmaTy.getRef().get_slice(core::IntTuple<core::Trait>::getZero());
  auto fragment = std::visit([](const auto &layout) { return core::make_fragment_like(layout); },
                             partitionMma(thr, operandId, tensor).layout());
  if (fragment.isError())
    return emitOptionalError(location, "MmaMakeFragmentOp: ", invalidResultMessage(fragment));
  inferredReturnTypes.assign(
      {fly::MemRefType::get(elemTy, AddressSpaceAttr::get(context, AddressSpace::Register),
                            LayoutAttr::get(context, fragment))});
  return success();
}

//===----------------------------------------------------------------------===//
// MemRef and Ptr operations
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(GetDynSharedOp) {
  auto i8Ty = IntegerType::get(context, 8);
  auto addrSpaceAttr = AddressSpaceAttr::get(context, AddressSpace::Shared);
  auto alignAttr = AlignAttr::get(context, 1024);
  inferredReturnTypes.assign({PointerType::get(i8Ty, addrSpaceAttr, alignAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(PtrToIntOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return emitOptionalError(location, "PtrToIntOp: expected PointerType, got ",
                             operands[0].getType());
  unsigned width;
  if (isGenericAddressSpace<AddressSpace::Shared>(ptrTy.getAddressSpace())) {
    width = 32;
  } else if (isGenericAddressSpace<AddressSpace::Global>(ptrTy.getAddressSpace())) {
    width = 64;
  } else {
    return emitOptionalError(location, "PtrToIntOp: expected Shared or Global address space, got ",
                             ptrTy.getAddressSpace());
  }

  inferredReturnTypes.assign({IntegerType::get(context, width)});
  return success();
}

FLY_INFER_RETURN_TYPES(ToLLVMPtrOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return emitOptionalError(location, "ToLLVMPtrOp: expected PointerType, got ",
                             operands[0].getType());
  if (!properties)
    return emitOptionalError(location, "ToLLVMPtrOp: missing llvm_address_space");
  int64_t addressSpace = properties.as<Properties *>()->llvm_address_space.getInt();
  if (addressSpace < 0)
    return emitOptionalError(location, "ToLLVMPtrOp: llvm_address_space must be non-negative, got ",
                             addressSpace);
  inferredReturnTypes.assign(
      {LLVM::LLVMPointerType::get(context, static_cast<unsigned>(addressSpace))});
  return success();
}

FLY_INFER_RETURN_TYPES(AddOffsetOp) {
  auto offsetTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!offsetTy)
    return emitOptionalError(location, "AddOffsetOp: expected IntTupleType for offset, got ",
                             operands[1].getType());
  const auto &offset = offsetTy.getRef();
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case([&](PointerType ptrTy) -> LogicalResult {
        if (!offset.isLeaf() || offset.isNone() || !offset.isInt())
          return emitOptionalError(
              location, "AddOffsetOp: offset must be a scalar (leaf int) IntTuple, got ", offsetTy);
        auto pointer = core::add_offset(ptrTy.getRef(), offset);
        auto result = PointerType::get(pointer);
        if (!result)
          return emitOptionalError(location, "AddOffsetOp: invalid pointer offset");
        inferredReturnTypes.assign({result});
        return success();
      })
      .Case([&](IntTupleType tupleTy) {
        return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                             tupleTy.getRef() + offset);
      })
      .Default([&](Type type) {
        return emitOptionalError(location,
                                 "AddOffsetOp: expected PointerType or IntTupleType, got ", type);
      });
}

FLY_INFER_RETURN_TYPES(ApplySwizzleOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return emitOptionalError(location, "ApplySwizzleOp: expected PointerType, got ",
                             operands[0].getType());
  auto swizzleTy = dyn_cast<SwizzleType>(operands[1].getType());
  if (!swizzleTy)
    return emitOptionalError(location, "ApplySwizzleOp: expected SwizzleType, got ",
                             operands[1].getType());
  inferredReturnTypes.assign({PointerType::get(ptrTy.getElemTy(), ptrTy.getAddressSpace(),
                                               ptrTy.getAlignment(), swizzleTy.getAttr())});
  return success();
}

FLY_INFER_RETURN_TYPES(DecompositionOp) {
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<fly::MemRefType, CoordTensorType>([&](auto type) {
        return inferResultType<decltype(type)>(location, context, inferredReturnTypes,
                                               core::decompose(type.getRef()));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "DecompositionOp: expected a tensor, got ", type);
      });
}

static LogicalResult verifyMmaOperandGroups(Operation *op, ValueRange a, ValueRange b) {
  if (a.empty() || b.empty())
    return op->emitOpError("A and B operand groups must each contain at least one tensor");
  return success();
}

LogicalResult MmaAtomCall::verify() { return verifyMmaOperandGroups(*this, getA(), getB()); }

LogicalResult MmaAtomCallSSA::verify() { return verifyMmaOperandGroups(*this, getA(), getB()); }

FLY_INFER_RETURN_TYPES(MemRefLoadOp) {
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case([&](fly::MemRefType memrefTy) {
        inferredReturnTypes.assign({memrefTy.getElemTy()});
        return success();
      })
      .Case([&](CoordTensorType tensorTy) -> LogicalResult {
        // Loading a coordinate tensor yields the coordinate at `indices`.
        auto indicesTy = dyn_cast<IntTupleType>(operands[1].getType());
        if (!indicesTy)
          return emitOptionalError(location, "MemRefLoadOp: expected IntTupleType for indices");
        auto tensor = tensorTy.getRef();
        auto offset = std::visit(
            [&](const auto &layout) { return core::crd2idx(indicesTy.getRef(), layout); },
            tensor.layout());
        return inferResultType<IntTupleType>(location, context, inferredReturnTypes,
                                             core::add_offset(tensor.data(), offset));
      })
      .Default([&](Type type) {
        return emitOptionalError(location, "MemRefLoadOp: expected a tensor, got ", type);
      });
}

FLY_INFER_RETURN_TYPES(MemRefLoadVecOp) {
  if (auto coordTensorTy = dyn_cast<CoordTensorType>(operands[0].getType())) {
    IntTupleAttr baseAttr = coordTensorTy.getBase();
    if (!baseAttr.isLeafInt())
      return emitOptionalError(
          location, "MemRefLoadVecOp: CoordTensor base must be a leaf int, got ", baseAttr);

    auto layoutAttr = dyn_cast<LayoutAttr>(coordTensorTy.getLayout());
    if (!layoutAttr)
      return emitOptionalError(
          location, "MemRefLoadVecOp: CoordTensorTy with ComposedLayout is not supported");
    auto size = IntTupleAttr::get(layoutAttr.getShape().getContext(),
                                  core::product(layoutAttr.getShape().getRef()));

    if (!size.isLeafInt() || !size.isStatic())
      return emitOptionalError(
          location, "MemRefLoadVecOp: layout size must be static and leaf int, got ", size);

    auto offsetProbe = IntTupleAttr::get(
        (IntTupleAttr::getLeafDynamic(context)).getContext(),
        core::crd2idx((IntTupleAttr::getLeafDynamic(context)).getRef(),
                      layoutAttr.getShape().getRef(), layoutAttr.getStride().getRef()));
    auto reprIdx =
        IntTupleAttr::get(baseAttr.getContext(), baseAttr.getRef() + offsetProbe.getRef());
    int32_t width = reprIdx.getLeaf().getScale().getWidth();
    Type elemTy = IntegerType::get(context, width == 64 ? 64 : 32);
    inferredReturnTypes.push_back(VectorType::get({size.getLeaf().getValue()}, elemTy));
    return success();
  }

  auto memrefTy = dyn_cast<MemRefType>(operands[0].getType());
  if (!memrefTy)
    return emitOptionalError(location,
                             "MemRefLoadVecOp: expected MemRefType or CoordTensorType, got ",
                             operands[0].getType());
  auto layoutAttr = dyn_cast<LayoutAttr>(memrefTy.getLayout());
  if (!layoutAttr)
    return emitOptionalError(location,
                             "MemRefLoadVecOp: MemRefType with ComposedLayout is not supported");
  auto size = IntTupleAttr::get(layoutAttr.getShape().getContext(),
                                core::product(layoutAttr.getShape().getRef()));

  if (!size.isLeafInt() || !size.isStatic())
    return emitOptionalError(
        location, "MemRefLoadVecOp: layout size must be static and leaf int, got ", size);

  inferredReturnTypes.push_back(VectorType::get({size.getLeaf().getValue()}, memrefTy.getElemTy()));
  return success();
}

#undef FLY_INFER_RETURN_TYPES
