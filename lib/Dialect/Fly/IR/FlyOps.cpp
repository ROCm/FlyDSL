// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors


#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/TiledOpUtils.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#define GET_OP_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyOps.cpp.inc"

#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::fly;

namespace {

LayoutAttr GetLayoutAttrFromLayoutLikeType(Type type) {
  Attribute layoutAttr;
  if (auto memrefTy = dyn_cast<fly::MemRefType>(type)) {
    layoutAttr = memrefTy.getLayout();
  } else if (auto coordTensorTy = dyn_cast<CoordTensorType>(type)) {
    layoutAttr = coordTensorTy.getLayout();
  } else if (auto layoutTy = dyn_cast<LayoutType>(type)) {
    layoutAttr = layoutTy.getAttr();
  } else if (auto composedTy = dyn_cast<ComposedLayoutType>(type)) {
    layoutAttr = composedTy.getAttr();
  } else {
    return nullptr;
  }

  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr)) {
    return layout;
  } else if (auto composed = dyn_cast<ComposedLayoutAttr>(layoutAttr)) {
    return composed.getOuter();
  } else {
    return nullptr;
  }
}

Type RebuildLayoutLikeType(Type type, LayoutAttr newLayoutAttr) {
  auto replaceOuter = [&](Attribute oldLayout) -> Attribute {
    if (isa<LayoutAttr>(oldLayout))
      return newLayoutAttr;
    auto composed = cast<ComposedLayoutAttr>(oldLayout);
    return ComposedLayoutAttr::get(composed.getInner(), composed.getOffset(), newLayoutAttr);
  };

  if (auto memrefTy = dyn_cast<fly::MemRefType>(type)) {
    return fly::MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(),
                                replaceOuter(memrefTy.getLayout()), memrefTy.getAlignment(),
                                memrefTy.getSwizzle());
  } else if (auto coordTensorTy = dyn_cast<fly::CoordTensorType>(type)) {
    return CoordTensorType::get(coordTensorTy.getBase(), replaceOuter(coordTensorTy.getLayout()));
  } else if (auto layoutTy = dyn_cast<fly::LayoutType>(type)) {
    return LayoutType::get(layoutTy.getContext(), newLayoutAttr);
  } else if (auto composedTy = dyn_cast<fly::ComposedLayoutType>(type)) {
    return ComposedLayoutType::get(composedTy.getAttr().getInner(),
                                   composedTy.getAttr().getOffset(), newLayoutAttr);
  } else {
    llvm_unreachable("Unsupported LayoutLike type");
  }
}

Type applyIntTupleTransform(Type inputTy, function_ref<IntTupleAttr(IntTupleAttr)> fn) {
  if (auto tupleTy = dyn_cast<IntTupleType>(inputTy))
    return IntTupleType::get(fn(tupleTy.getAttr()));

  LayoutAttr outerLayout = GetLayoutAttrFromLayoutLikeType(inputTy);
  if (!outerLayout)
    return {};
  LayoutAttr transformed = LayoutAttr::get(fn(outerLayout.getShape()), fn(outerLayout.getStride()));
  return RebuildLayoutLikeType(inputTy, transformed);
}

Type applyOffsetOnMemRef(LayoutBuilder<LayoutAttr> &builder, fly::MemRefType memrefTy,
                         IntTupleAttr offset, LayoutAttr layoutAttr) {
  if (auto composed = dyn_cast<ComposedLayoutAttr>(memrefTy.getLayout())) {
    IntTupleAttr newOffset = intTupleAdd(builder, composed.getOffset(), offset);
    Attribute newLayout = ComposedLayoutAttr::get(composed.getInner(), newOffset, layoutAttr);
    return fly::MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), newLayout,
                                memrefTy.getAlignment(), memrefTy.getSwizzle());
  } else {
    int32_t valDiv = memrefTy.getValueDivisibility();
    IntAttr offsetInt = offset.extractIntFromLeaf();
    int32_t offsetDiv =
        offsetInt.isStatic() ? std::abs(offsetInt.getValue()) : offsetInt.getDivisibility();
    int32_t newValDiv = (offsetDiv == 0) ? valDiv : utils::divisibilityAdd(valDiv, offsetDiv);
    return fly::MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), layoutAttr,
                                AlignAttr::get(memrefTy.getElemTy(), newValDiv),
                                memrefTy.getSwizzle());
  }
}

} // namespace

#define FLY_INFER_RETURN_TYPES(OP)                                                                 \
  llvm::LogicalResult OP::inferReturnTypes(                                                        \
      mlir::MLIRContext *context, std::optional<::mlir::Location> location,                        \
      mlir::ValueRange operands, mlir::DictionaryAttr attributes,                                  \
      mlir::OpaqueProperties properties, mlir::RegionRange regions,                                \
      llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes)

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
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeTileOp) {
  SmallVector<Attribute> layouts;
  for (auto op : operands) {
    if (auto layoutType = dyn_cast<LayoutType>(op.getType())) {
      layouts.push_back(layoutType.getAttr());
    } else if (auto intTupleType = dyn_cast<IntTupleType>(op.getType())) {
      layouts.push_back(intTupleType.getAttr());
    } else {
      return emitOptionalError(location, "MakeTileOp: operand has unsupported type ", op.getType(),
                               ", expected LayoutType or IntTupleType");
    }
  }
  auto tileAttr = TileAttr::get(ArrayAttr::get(context, layouts));
  inferredReturnTypes.assign({TileType::get(context, tileAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeLayoutLikeOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutTy)
    return emitOptionalError(location, "MakeLayoutLikeOp: expected LayoutType, got ",
                             operands[0].getType());
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutMakeLayoutLike(layoutBuilder, layoutTy.getAttr());
  inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  return success();
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
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr layoutAttr =
      layoutMakeOrderedLayout(layoutBuilder, shapeTy.getAttr(), orderTy.getAttr());
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeComposedLayoutOp) {
  auto offsetTy = dyn_cast<IntTupleType>(operands[1].getType());
  auto outerTy = dyn_cast<LayoutType>(operands[2].getType());
  if (!offsetTy)
    return emitOptionalError(location,
                             "MakeComposedLayoutOp: expected IntTupleType for offset, got ",
                             operands[1].getType());
  if (!outerTy)
    return emitOptionalError(location, "MakeComposedLayoutOp: expected LayoutType for outer, got ",
                             operands[2].getType());
  Attribute innerAttr = nullptr;
  if (auto innerLayoutTy = dyn_cast<LayoutType>(operands[0].getType())) {
    innerAttr = innerLayoutTy.getAttr();
  } else if (auto innerComposedTy = dyn_cast<ComposedLayoutType>(operands[0].getType())) {
    innerAttr = innerComposedTy.getAttr();
  } else if (auto innerSwizzleTy = dyn_cast<SwizzleType>(operands[0].getType())) {
    innerAttr = innerSwizzleTy.getAttr();
  } else {
    return emitOptionalError(
        location, "MakeComposedLayoutOp: expected Layout/ComposedLayout/Swizzle for inner, got ",
        operands[0].getType());
  }
  auto composedAttr =
      ComposedLayoutAttr::get(context, innerAttr, offsetTy.getAttr(), outerTy.getAttr());
  inferredReturnTypes.assign({ComposedLayoutType::get(context, composedAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeIdentityLayoutOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!shapeTy)
    return emitOptionalError(location,
                             "MakeIdentityLayoutOp: expected IntTupleType for shape, got ",
                             operands[0].getType());
  IntTupleAttr shapeAttr = shapeTy.getAttr();
  IntTupleAttr strideAttr = intTupleMakeBasisTupleLike(shapeAttr);
  LayoutAttr layoutAttr = LayoutAttr::get(context, shapeAttr, strideAttr);
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeViewOp) {
  Type iterTy = operands[0].getType();
  Type layoutArgTy = operands[1].getType();

  Attribute layoutAttr;
  if (auto layoutTy = dyn_cast<LayoutType>(layoutArgTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if (auto composedTy = dyn_cast<ComposedLayoutType>(layoutArgTy)) {
    layoutAttr = composedTy.getAttr();
  } else {
    return emitOptionalError(
        location, "MakeViewOp: expected LayoutType or ComposedLayoutType for operand #1, got ",
        layoutArgTy);
  }

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

FLY_INFER_RETURN_TYPES(MakeFragmentLikeOp) {
  TypeAttr dtypeAttr;
  if (properties)
    dtypeAttr = properties.as<Properties *>()->dtype;

  auto srcLayout = GetLayoutAttrFromLayoutLikeType(operands[0].getType());
  if (!srcLayout)
    return emitOptionalError(location, "MakeFragmentLikeOp: expected LayoutLikeType, got ",
                             operands[0].getType());

  Type elemTy;
  if (auto memrefTy = dyn_cast<MemRefType>(operands[0].getType())) {
    elemTy = dtypeAttr ? dtypeAttr.getValue() : memrefTy.getElemTy();
  } else {
    if (!dtypeAttr)
      return emitOptionalError(
          location, "MakeFragmentLikeOp: dtype is required when input is not MemRefType");
    elemTy = dtypeAttr.getValue();
  }

  if (!srcLayout.getShape().isStatic())
    return emitOptionalError(location, "MakeFragmentLikeOp: expected static shape layout, got ",
                             srcLayout);

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr fragmentLayout = layoutMakeFragmentLayout(layoutBuilder, srcLayout);
  inferredReturnTypes.assign({MemRefType::get(
      elemTy, AddressSpaceAttr::get(context, AddressSpace::Register), fragmentLayout)});
  return success();
}

//===----------------------------------------------------------------------===//
// Extractors
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(GetScalarOp) {
  auto intTupleType = dyn_cast<IntTupleType>(operands[0].getType());
  if (!intTupleType)
    return emitOptionalError(location, "GetScalarOp: expected IntTupleType, got ",
                             operands[0].getType());
  if (!intTupleType.getAttr().isLeaf())
    return emitOptionalError(location, "GetScalarOp: expected a leaf IntTuple, got ", intTupleType);
  auto intAttr = intTupleType.getAttr().extractIntFromLeaf();
  inferredReturnTypes.assign({IntegerType::get(context, intAttr.getWidth())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetLeavesOp) {
  auto intTupleType = dyn_cast<IntTupleType>(operands[0].getType());
  if (!intTupleType)
    return emitOptionalError(location, "GetLeavesOp: expected IntTupleType, got ",
                             operands[0].getType());
  // transform_leaf on intTupleAttr,
  return failure();
}

FLY_INFER_RETURN_TYPES(GetShapeOp) {
  auto layout = GetLayoutAttrFromLayoutLikeType(operands[0].getType());
  if (!layout)
    return emitOptionalError(location, "GetShapeOp: expected LayoutLikeType, got ",
                             operands[0].getType());
  inferredReturnTypes.assign({IntTupleType::get(layout.getShape())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetStrideOp) {
  Type inputTy = operands[0].getType();

  if (isa<ComposedLayoutType>(inputTy))
    return emitOptionalError(location, "GetStrideOp: unsupported ComposedLayoutType");
  if (auto memrefTy = dyn_cast<fly::MemRefType>(inputTy);
      memrefTy && isa<ComposedLayoutAttr>(memrefTy.getLayout()))
    return emitOptionalError(location, "GetStrideOp: unsupported MemRefType with ComposedLayout");
  if (auto coordTensorTy = dyn_cast<CoordTensorType>(inputTy);
      coordTensorTy && isa<ComposedLayoutAttr>(coordTensorTy.getLayout()))
    return emitOptionalError(location,
                             "GetStrideOp: unsupported CoordTensorType with ComposedLayout");

  auto layout = GetLayoutAttrFromLayoutLikeType(inputTy);
  if (!layout)
    return emitOptionalError(location, "GetStrideOp: expected LayoutLikeType, got ", inputTy);

  inferredReturnTypes.assign({IntTupleType::get(layout.getStride())});
  return success();
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
  } else if (auto layoutAttr = dyn_cast<LayoutAttr>(innerAttr)) {
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
  inferredReturnTypes.assign({LayoutType::get(context, inputTy.getAttr().getOuter())});
  return success();
}

//===----------------------------------------------------------------------===//
// IntTuple operations
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(IntTupleAddOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "IntTupleAddOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "IntTupleAddOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleAdd(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleSubOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "IntTupleSubOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "IntTupleSubOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleSub(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleMulOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "IntTupleMulOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "IntTupleMulOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleMul(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleDivOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "IntTupleDivOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "IntTupleDivOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleDiv(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleModOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "IntTupleModOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "IntTupleModOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleMod(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleProductEachOp) {
  auto inputTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!inputTy)
    return emitOptionalError(location, "IntTupleProductEachOp: expected IntTupleType, got ",
                             operands[0].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign({IntTupleType::get(intTupleProductEach(builder, inputTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleProductOp) {
  auto inputTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!inputTy)
    return emitOptionalError(location, "IntTupleProductOp: expected IntTupleType, got ",
                             operands[0].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign({IntTupleType::get(intTupleProduct(builder, inputTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleProductLikeOp) {
  auto tupleTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto guideTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!tupleTy)
    return emitOptionalError(location, "IntTupleProductLikeOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!guideTy)
    return emitOptionalError(location, "IntTupleProductLikeOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleProductLike(builder, tupleTy.getAttr(), guideTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(ShapeDivOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "ShapeDivOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "ShapeDivOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleShapeDiv(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(CeilDivOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "CeilDivOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "CeilDivOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleCeilDiv(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(ElemLessOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "ElemLessOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "ElemLessOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleElemLess(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(EqualOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy)
    return emitOptionalError(location, "EqualOp: expected IntTupleType for lhs, got ",
                             operands[0].getType());
  if (!rhsTy)
    return emitOptionalError(location, "EqualOp: expected IntTupleType for rhs, got ",
                             operands[1].getType());
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleEqual(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

//===----------------------------------------------------------------------===//
// IntTupleLike operations
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(GetOp) {
  Type inputTy = operands[0].getType();
  ArrayRef<int32_t> mode = properties.as<Properties *>()->mode;

  int32_t depth = -1;
  if (auto nested = dyn_cast<NestedTypeInterface>(inputTy))
    depth = nested.depth();
  else if (auto memrefTy = dyn_cast<MemRefType>(inputTy))
    depth = cast<NestedAttrInterface>(memrefTy.getLayout()).depth();
  else if (auto coordTensorTy = dyn_cast<CoordTensorType>(inputTy))
    depth = cast<NestedAttrInterface>(coordTensorTy.getLayout()).depth();
  else
    return emitOptionalError(location, "GetOp: unsupported input type ", inputTy);

  if (depth < static_cast<int32_t>(mode.size()))
    return emitOptionalError(location, "GetOp: mode length ", mode.size(), " exceeds input depth ",
                             depth);

  Type resultTy;
  if (auto intTupleTy = dyn_cast<IntTupleType>(inputTy))
    resultTy = intTupleTy.at(mode);
  else if (auto layoutTy = dyn_cast<LayoutType>(inputTy))
    resultTy = layoutTy.at(mode);
  else if (auto composedTy = dyn_cast<ComposedLayoutType>(inputTy))
    resultTy = composedTy.at(mode);
  else if (auto memrefTy = dyn_cast<MemRefType>(inputTy))
    resultTy = memrefTy.at(mode);
  else if (auto coordTensorTy = dyn_cast<CoordTensorType>(inputTy))
    resultTy = coordTensorTy.at(mode);
  else
    return emitOptionalError(location, "GetOp: unsupported input type ", inputTy);

  inferredReturnTypes.assign({resultTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TakeOp) {
  int32_t begin = properties.as<Properties *>()->begin.getInt();
  int32_t end = properties.as<Properties *>()->end.getInt();

  IntTupleBuilder<IntTupleAttr> builder(context);
  Type resultTy = applyIntTupleTransform(operands[0].getType(), [&](IntTupleAttr attr) {
    return intTupleTake(builder, attr, begin, end);
  });
  if (!resultTy)
    return emitOptionalError(location, "TakeOp: unsupported input type ", operands[0].getType());

  inferredReturnTypes.assign({resultTy});
  return success();
}

FLY_INFER_RETURN_TYPES(SelectOp) {
  auto idxArr = properties.as<Properties *>()->indices.asArrayRef();
  SmallVector<int32_t> indices(idxArr.begin(), idxArr.end());

  IntTupleBuilder<IntTupleAttr> builder(context);
  Type resultTy = applyIntTupleTransform(operands[0].getType(), [&](IntTupleAttr attr) {
    return intTupleSelect(builder, attr, indices);
  });
  if (!resultTy)
    return emitOptionalError(location, "SelectOp: unsupported input type ", operands[0].getType());
  inferredReturnTypes.assign({resultTy});
  return success();
}

FLY_INFER_RETURN_TYPES(GroupOp) {
  int32_t begin = properties.as<Properties *>()->begin.getInt();
  int32_t end = properties.as<Properties *>()->end.getInt();

  IntTupleBuilder<IntTupleAttr> builder(context);
  Type resultTy = applyIntTupleTransform(operands[0].getType(), [&](IntTupleAttr attr) {
    return intTupleGroup(builder, attr, begin, end);
  });
  if (!resultTy)
    return emitOptionalError(location, "GroupOp: unsupported input type ", operands[0].getType());
  inferredReturnTypes.assign({resultTy});
  return success();
}

FLY_INFER_RETURN_TYPES(AppendOp) {
  Type tupleTy = operands[0].getType();
  Type elemTy = operands[1].getType();

  int32_t n = -1;
  if (properties) {
    auto nAttr = properties.as<Properties *>()->n;
    if (nAttr)
      n = static_cast<int32_t>(nAttr.getInt());
  }

  IntTupleBuilder<IntTupleAttr> builder(context);

  if (auto tupleIT = dyn_cast<IntTupleType>(tupleTy)) {
    auto elemIT = dyn_cast<IntTupleType>(elemTy);
    if (!elemIT)
      return emitOptionalError(location, "AppendOp: tuple and elem must be the same category");
    IntTupleAttr result = intTupleAppend(builder, tupleIT.getAttr(), elemIT.getAttr(), n);
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  }

  if (auto tupleLayout = dyn_cast<LayoutType>(tupleTy)) {
    auto elemLayout = dyn_cast<LayoutType>(elemTy);
    if (!elemLayout)
      return emitOptionalError(location, "AppendOp: tuple and elem must be the same category");
    LayoutAttr baseAttr = tupleLayout.getAttr();
    LayoutAttr eAttr = elemLayout.getAttr();
    IntTupleAttr newShape = intTupleAppend(builder, baseAttr.getShape(), eAttr.getShape(), n);
    IntTupleAttr newStride = intTupleAppend(builder, baseAttr.getStride(), eAttr.getStride(), n);
    inferredReturnTypes.assign(
        {LayoutType::get(context, LayoutAttr::get(context, newShape, newStride))});
    return success();
  }

  if (auto composedTy = dyn_cast<ComposedLayoutType>(tupleTy)) {
    auto elemLayout = dyn_cast<LayoutType>(elemTy);
    if (!elemLayout)
      return emitOptionalError(location,
                               "AppendOp: elem must be LayoutType when tuple is ComposedLayout");
    ComposedLayoutAttr ca = composedTy.getAttr();
    LayoutAttr outer = ca.getOuter();
    LayoutAttr eAttr = elemLayout.getAttr();
    IntTupleAttr newShape = intTupleAppend(builder, outer.getShape(), eAttr.getShape(), n);
    IntTupleAttr newStride = intTupleAppend(builder, outer.getStride(), eAttr.getStride(), n);
    LayoutAttr newOuter = LayoutAttr::get(context, newShape, newStride);
    inferredReturnTypes.assign({ComposedLayoutType::get(
        context, ComposedLayoutAttr::get(context, ca.getInner(), ca.getOffset(), newOuter))});
    return success();
  }

  return emitOptionalError(location, "AppendOp: unsupported input type ", tupleTy);
}

FLY_INFER_RETURN_TYPES(PrependOp) {
  Type tupleTy = operands[0].getType();
  Type elemTy = operands[1].getType();

  int32_t n = -1;
  if (properties) {
    auto nAttr = properties.as<Properties *>()->n;
    if (nAttr)
      n = static_cast<int32_t>(nAttr.getInt());
  }

  IntTupleBuilder<IntTupleAttr> builder(context);

  if (auto tupleIT = dyn_cast<IntTupleType>(tupleTy)) {
    auto elemIT = dyn_cast<IntTupleType>(elemTy);
    if (!elemIT)
      return emitOptionalError(location, "PrependOp: tuple and elem must be the same category");
    IntTupleAttr result = intTuplePrepend(builder, tupleIT.getAttr(), elemIT.getAttr(), n);
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  }

  if (auto tupleLayout = dyn_cast<LayoutType>(tupleTy)) {
    auto elemLayout = dyn_cast<LayoutType>(elemTy);
    if (!elemLayout)
      return emitOptionalError(location, "PrependOp: tuple and elem must be the same category");
    LayoutAttr baseAttr = tupleLayout.getAttr();
    LayoutAttr eAttr = elemLayout.getAttr();
    IntTupleAttr newShape = intTuplePrepend(builder, baseAttr.getShape(), eAttr.getShape(), n);
    IntTupleAttr newStride = intTuplePrepend(builder, baseAttr.getStride(), eAttr.getStride(), n);
    inferredReturnTypes.assign({LayoutType::get(LayoutAttr::get(newShape, newStride))});
    return success();
  }

  if (auto composedTy = dyn_cast<ComposedLayoutType>(tupleTy)) {
    auto elemLayout = dyn_cast<LayoutType>(elemTy);
    if (!elemLayout)
      return emitOptionalError(location,
                               "PrependOp: elem must be LayoutType when tuple is ComposedLayout");
    ComposedLayoutAttr ca = composedTy.getAttr();
    LayoutAttr outer = ca.getOuter();
    LayoutAttr eAttr = elemLayout.getAttr();
    IntTupleAttr newShape = intTuplePrepend(builder, outer.getShape(), eAttr.getShape(), n);
    IntTupleAttr newStride = intTuplePrepend(builder, outer.getStride(), eAttr.getStride(), n);
    LayoutAttr newOuter = LayoutAttr::get(newShape, newStride);
    inferredReturnTypes.assign({ComposedLayoutType::get(ca.getInner(), ca.getOffset(), newOuter)});
    return success();
  }

  return emitOptionalError(location, "PrependOp: unsupported input type ", tupleTy);
}

FLY_INFER_RETURN_TYPES(SliceOp) {
  Type srcTy = operands[0].getType();
  auto coordTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!coordTy)
    return emitOptionalError(location, "SliceOp: expected IntTupleType for coord, got ",
                             operands[1].getType());

  IntTupleAttr coordAttr = coordTy.getAttr();
  IntTupleBuilder<IntTupleAttr> builder(context);

  auto sliceLayout = [&](LayoutAttr layout) -> LayoutAttr {
    IntTupleAttr newShape = intTupleSlice(builder, layout.getShape(), coordAttr);
    IntTupleAttr newStride = intTupleSlice(builder, layout.getStride(), coordAttr);
    return LayoutAttr::get(context, newShape, newStride);
  };
  auto sliceComposed = [&](ComposedLayoutAttr composed) -> ComposedLayoutAttr {
    LayoutAttr outer = composed.getOuter();
    IntTupleAttr offset = layoutCrd2Idx(builder, coordAttr, outer.getShape(), outer.getStride());
    IntTupleAttr newOffset = intTupleAdd(builder, composed.getOffset(), offset);
    return ComposedLayoutAttr::get(composed.getInner(), newOffset, sliceLayout(outer));
  };

  if (auto srcTupleTy = dyn_cast<IntTupleType>(srcTy)) {
    IntTupleAttr result = intTupleSlice(builder, srcTupleTy.getAttr(), coordAttr);
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  }
  if (auto srcLayoutTy = dyn_cast<LayoutType>(srcTy)) {
    inferredReturnTypes.assign({LayoutType::get(sliceLayout(srcLayoutTy.getAttr()))});
    return success();
  }
  if (auto srcComposedTy = dyn_cast<ComposedLayoutType>(srcTy)) {
    inferredReturnTypes.assign({ComposedLayoutType::get(sliceComposed(srcComposedTy.getAttr()))});
    return success();
  }
  if (auto srcMemRefTy = dyn_cast<fly::MemRefType>(srcTy)) {
    Attribute layout = srcMemRefTy.getLayout();
    Attribute newLayout;
    int32_t valDiv = srcMemRefTy.getValueDivisibility();
    int32_t newValDiv;
    if (auto la = dyn_cast<LayoutAttr>(layout)) {
      newLayout = sliceLayout(la);
      IntTupleAttr offsetAttr = layoutCrd2Idx(builder, coordAttr, la.getShape(), la.getStride());
      IntAttr offsetInt = offsetAttr.extractIntFromLeaf();
      int32_t offsetDiv =
          offsetInt.isStatic() ? std::abs(offsetInt.getValue()) : offsetInt.getDivisibility();
      newValDiv = (offsetDiv == 0) ? valDiv : utils::divisibilityAdd(valDiv, offsetDiv);
    } else {
      newLayout = sliceComposed(cast<ComposedLayoutAttr>(layout));
      newValDiv = valDiv;
    }

    inferredReturnTypes.assign({fly::MemRefType::get(
        srcMemRefTy.getElemTy(), srcMemRefTy.getAddressSpace(), newLayout,
        AlignAttr::get(srcMemRefTy.getElemTy(), newValDiv), srcMemRefTy.getSwizzle())});
    return success();
  }
  if (auto srcCoordTensorTy = dyn_cast<CoordTensorType>(srcTy)) {
    IntTupleAttr newBase = intTupleSlice(builder, srcCoordTensorTy.getBase(), coordAttr);
    Attribute layout = srcCoordTensorTy.getLayout();
    Attribute newLayout;
    if (auto la = dyn_cast<LayoutAttr>(layout))
      newLayout = sliceLayout(la);
    else
      newLayout = sliceComposed(cast<ComposedLayoutAttr>(layout));
    inferredReturnTypes.assign({CoordTensorType::get(newBase, newLayout)});
    return success();
  }

  return emitOptionalError(location, "SliceOp: unsupported input type ", srcTy);
}

FLY_INFER_RETURN_TYPES(DiceOp) {
  Type srcTy = operands[0].getType();
  auto coordTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!coordTy)
    return emitOptionalError(location, "DiceOp: expected IntTupleType for coord, got ",
                             operands[1].getType());

  IntTupleAttr coordAttr = coordTy.getAttr();
  IntTupleBuilder<IntTupleAttr> builder(context);

  if (auto srcTupleTy = dyn_cast<IntTupleType>(srcTy)) {
    IntTupleAttr result = intTupleDice(builder, srcTupleTy.getAttr(), coordAttr);
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  }
  if (auto srcLayoutTy = dyn_cast<LayoutType>(srcTy)) {
    LayoutAttr profile = srcLayoutTy.getAttr();
    IntTupleAttr newShape = intTupleDice(builder, profile.getShape(), coordAttr);
    IntTupleAttr newStride = intTupleDice(builder, profile.getStride(), coordAttr);
    inferredReturnTypes.assign(
        {LayoutType::get(context, LayoutAttr::get(context, newShape, newStride))});
    return success();
  }

  return emitOptionalError(location, "DiceOp: expected IntTupleType or LayoutType, got ", srcTy);
}

//===----------------------------------------------------------------------===//
// LayoutLike operations
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(SizeOp) {
  if (auto intTupleTy = dyn_cast<IntTupleType>(operands[0].getType())) {
    IntTupleBuilder<IntTupleAttr> builder(context);
    IntTupleAttr size = intTupleProduct(builder, intTupleTy.getAttr());
    inferredReturnTypes.assign({IntTupleType::get(size)});
    return success();
  }
  auto layout = GetLayoutAttrFromLayoutLikeType(operands[0].getType());
  if (!layout)
    return emitOptionalError(location, "SizeOp: expected LayoutLikeType, got ",
                             operands[0].getType());
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  inferredReturnTypes.assign({IntTupleType::get(layoutSize(layoutBuilder, layout))});
  return success();
}

FLY_INFER_RETURN_TYPES(CoprofileOp) {
  auto layout = GetLayoutAttrFromLayoutLikeType(operands[0].getType());
  if (!layout)
    return emitOptionalError(location, "CoprofileOp: expected LayoutLikeType, got ",
                             operands[0].getType());
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  inferredReturnTypes.assign({IntTupleType::get(layoutCoprofile(layoutBuilder, layout))});
  return success();
}

FLY_INFER_RETURN_TYPES(CoshapeOp) {
  auto layout = GetLayoutAttrFromLayoutLikeType(operands[0].getType());
  if (!layout)
    return emitOptionalError(location, "CoshapeOp: expected LayoutLikeType, got ",
                             operands[0].getType());
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  inferredReturnTypes.assign({IntTupleType::get(layoutCoshape(layoutBuilder, layout))});
  return success();
}

FLY_INFER_RETURN_TYPES(CosizeOp) {
  auto layout = GetLayoutAttrFromLayoutLikeType(operands[0].getType());
  if (!layout)
    return emitOptionalError(location, "CosizeOp: expected LayoutLikeType, got ",
                             operands[0].getType());
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  inferredReturnTypes.assign({IntTupleType::get(layoutCosize(layoutBuilder, layout))});
  return success();
}

FLY_INFER_RETURN_TYPES(Crd2IdxOp) {
  auto coordTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!coordTy)
    return emitOptionalError(location, "Crd2IdxOp: expected IntTupleType for coord, got ",
                             operands[0].getType());

  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr coordAttr = coordTy.getAttr();

  if (auto layoutTy = dyn_cast<LayoutType>(operands[1].getType())) {
    LayoutAttr layoutAttr = layoutTy.getAttr();
    IntTupleAttr result =
        layoutCrd2Idx(builder, coordAttr, layoutAttr.getShape(), layoutAttr.getStride());
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  } else if (auto composedTy = dyn_cast<ComposedLayoutType>(operands[1].getType())) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    IntTupleAttr result =
        layoutCrd2Idx(layoutBuilder, coordAttr, static_cast<Attribute>(composedTy.getAttr()));
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  }

  return emitOptionalError(location,
                           "Crd2IdxOp: expected LayoutType or ComposedLayoutType for layout, got ",
                           operands[1].getType());
}

FLY_INFER_RETURN_TYPES(Idx2CrdOp) {
  if (auto layoutTy = dyn_cast<LayoutType>(operands[1].getType())) {
    LayoutAttr layoutAttr = layoutTy.getAttr();
    IntTupleBuilder<IntTupleAttr> builder(context);
    auto coordTy = dyn_cast<IntTupleType>(operands[0].getType());
    if (!coordTy)
      return emitOptionalError(location, "Idx2CrdOp: expected IntTupleType for index, got ",
                               operands[0].getType());
    IntTupleAttr result =
        layoutIdx2Crd(builder, coordTy.getAttr(), layoutAttr.getShape(), layoutAttr.getStride());
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  }
  return emitOptionalError(location, "Idx2CrdOp: expected LayoutType for layout, got ",
                           operands[1].getType());
}

FLY_INFER_RETURN_TYPES(GetFlatCoordOp) {
  auto indexTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!indexTy)
    return emitOptionalError(location, "GetFlatCoordOp: expected IntTupleType for index, got ",
                             operands[0].getType());
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!layoutTy)
    return emitOptionalError(location, "GetFlatCoordOp: expected LayoutType for layout, got ",
                             operands[1].getType());

  LayoutAttr layoutAttr = layoutTy.getAttr();
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr hierCoord =
      layoutIdx2Crd(builder, indexTy.getAttr(), layoutAttr.getShape(), layoutAttr.getStride());
  IntTupleAttr flatShape = intTupleTransform(
      builder, [&](IntTupleAttr mode) { return builder.materializeConstantLeaf(1); },
      layoutAttr.getShape());
  IntTupleAttr result = layoutCrd2Crd(builder, hierCoord, layoutAttr.getShape(), flatShape);
  inferredReturnTypes.assign({IntTupleType::get(result)});
  return success();
}

FLY_INFER_RETURN_TYPES(Get1DCoordOp) {
  auto indexTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!indexTy)
    return emitOptionalError(location, "Get1DCoordOp: expected IntTupleType for index, got ",
                             operands[0].getType());
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!layoutTy)
    return emitOptionalError(location, "Get1DCoordOp: expected LayoutType for layout, got ",
                             operands[1].getType());

  LayoutAttr layoutAttr = layoutTy.getAttr();
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr result =
      layoutIdx2Crd(builder, indexTy.getAttr(), layoutAttr.getShape(), layoutAttr.getStride());
  result = layoutCrd2IdxColMajor(builder, result, layoutAttr.getShape());
  inferredReturnTypes.assign({IntTupleType::get(result)});
  return success();
}

FLY_INFER_RETURN_TYPES(CoalesceOp) {
  Type inputTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(inputTy);
  if (!layoutAttr)
    return emitOptionalError(location, "CoalesceOp: expected LayoutLikeType, got ", inputTy);

  std::optional<IntTupleAttr> profileAttr;
  if (operands.size() > 1 && operands[1]) {
    auto profileTy = dyn_cast<IntTupleType>(operands[1].getType());
    if (!profileTy)
      return emitOptionalError(location, "CoalesceOp: expected IntTupleType for profile, got ",
                               operands[1].getType());
    profileAttr = profileTy.getAttr();
  }

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutCoalesce(layoutBuilder, layoutAttr, profileAttr);
  inferredReturnTypes.assign({RebuildLayoutLikeType(inputTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(CompositionOp) {
  Type outerTy = operands[0].getType();
  auto outerLayoutAttr = GetLayoutAttrFromLayoutLikeType(outerTy);
  if (!outerLayoutAttr)
    return emitOptionalError(location, "CompositionOp: expected LayoutLikeType for outer, got ",
                             outerTy);

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  Type innerTy = operands[1].getType();
  LayoutAttr inferred;
  if (auto tileTy = dyn_cast<TileType>(innerTy))
    inferred = layoutComposition(layoutBuilder, outerLayoutAttr, tileTy.getAttr());
  else if (auto innerLayoutTy = dyn_cast<LayoutType>(innerTy))
    inferred = layoutComposition(layoutBuilder, outerLayoutAttr, innerLayoutTy.getAttr());
  else
    return emitOptionalError(
        location, "CompositionOp: expected TileType or LayoutType for inner, got ", innerTy);

  inferredReturnTypes.assign({RebuildLayoutLikeType(outerTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(ComplementOp) {
  Type inputTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(inputTy);
  if (!layoutAttr)
    return emitOptionalError(location, "ComplementOp: expected NarrowLayoutType, got ", inputTy);

  std::optional<IntTupleAttr> codomainSizeAttr;
  if (operands.size() > 1 && operands[1]) {
    auto codomainSizeTy = dyn_cast<IntTupleType>(operands[1].getType());
    if (!codomainSizeTy)
      return emitOptionalError(location,
                               "ComplementOp: expected IntTupleType for codomain_size, got ",
                               operands[1].getType());
    codomainSizeAttr = codomainSizeTy.getAttr();
  }

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutComplement(layoutBuilder, layoutAttr, codomainSizeAttr);
  inferredReturnTypes.assign({RebuildLayoutLikeType(inputTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(RightInverseOp) {
  Type inputTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(inputTy);
  if (!layoutAttr)
    return emitOptionalError(location, "RightInverseOp: expected NarrowLayoutType, got ", inputTy);
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutRightInverse(layoutBuilder, layoutAttr);
  inferredReturnTypes.assign({RebuildLayoutLikeType(inputTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(LeftInverseOp) {
  Type inputTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(inputTy);
  if (!layoutAttr)
    return emitOptionalError(location, "LeftInverseOp: expected NarrowLayoutType, got ", inputTy);
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutLeftInverse(layoutBuilder, layoutAttr);
  inferredReturnTypes.assign({RebuildLayoutLikeType(inputTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(LogicalDivideOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "LogicalDivideOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    inferred = layoutLogicalDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    inferred = layoutLogicalDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return emitOptionalError(
        location, "LogicalDivideOp: expected LayoutType or TileType for divisor, got ", divisorTy);
  }

  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(ZippedDivideOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "ZippedDivideOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    inferred = layoutZippedDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    inferred = layoutZippedDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return emitOptionalError(
        location, "ZippedDivideOp: expected LayoutType or TileType for divisor, got ", divisorTy);
  }

  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledDivideOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "TiledDivideOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    inferred = layoutTiledDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    inferred = layoutTiledDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return emitOptionalError(
        location, "TiledDivideOp: expected LayoutType or TileType for divisor, got ", divisorTy);
  }

  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(FlatDivideOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "FlatDivideOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;
  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    inferred = layoutFlatDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    inferred = layoutFlatDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return emitOptionalError(
        location, "FlatDivideOp: expected LayoutType or TileType for divisor, got ", divisorTy);
  }

  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(LogicalProductOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "LogicalProductOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return emitOptionalError(location, "LogicalProductOp: expected LayoutType for tiler, got ",
                             operands[1].getType());

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutLogicalProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());
  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(ZippedProductOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "ZippedProductOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return emitOptionalError(location, "ZippedProductOp: expected LayoutType for tiler, got ",
                             operands[1].getType());

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutZippedProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());
  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledProductOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "TiledProductOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return emitOptionalError(location, "TiledProductOp: expected LayoutType for tiler, got ",
                             operands[1].getType());

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutTiledProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());
  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(FlatProductOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "FlatProductOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return emitOptionalError(location, "FlatProductOp: expected LayoutType for tiler, got ",
                             operands[1].getType());

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutFlatProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());
  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(BlockedProductOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "BlockedProductOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return emitOptionalError(location, "BlockedProductOp: expected LayoutType for tiler, got ",
                             operands[1].getType());

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutBlockedProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());
  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(RakedProductOp) {
  Type lhsTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(lhsTy);
  if (!layoutAttr)
    return emitOptionalError(location, "RakedProductOp: expected LayoutLikeType for lhs, got ",
                             lhsTy);

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return emitOptionalError(location, "RakedProductOp: expected LayoutType for tiler, got ",
                             operands[1].getType());

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutRakedProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());
  inferredReturnTypes.assign({RebuildLayoutLikeType(lhsTy, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(RecastLayoutOp) {
  int32_t newTypeBits = properties.as<Properties *>()->new_type_bits.getInt();
  int32_t oldTypeBits = properties.as<Properties *>()->old_type_bits.getInt();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  if (auto layoutTy = dyn_cast<LayoutType>(operands[0].getType())) {
    LayoutAttr layoutAttr = layoutTy.getAttr();
    LayoutAttr result = layoutRecast(layoutBuilder, layoutAttr, oldTypeBits, newTypeBits);
    inferredReturnTypes.assign({LayoutType::get(result)});
    return success();
  } else if (auto composedTy = dyn_cast<ComposedLayoutType>(operands[0].getType())) {
    ComposedLayoutAttr composedAttr = composedTy.getAttr();
    Attribute result =
        layoutRecast(layoutBuilder, static_cast<Attribute>(composedAttr), oldTypeBits, newTypeBits);
    assert(isa<ComposedLayoutAttr>(result));
    inferredReturnTypes.assign({ComposedLayoutType::get(cast<ComposedLayoutAttr>(result))});
    return success();
  } else {
    return emitOptionalError(
        location, "RecastLayoutOp: expected LayoutType or ComposedLayoutType for operand #0, got ",
        operands[0].getType());
  }
}

FLY_INFER_RETURN_TYPES(TileToShapeOp) {
  Type blockTy = operands[0].getType();
  auto layoutAttr = GetLayoutAttrFromLayoutLikeType(blockTy);
  if (!layoutAttr)
    return emitOptionalError(location, "TileToShapeOp: expected NarrowLayoutType for block, got ",
                             blockTy);

  auto trgShapeTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!trgShapeTy)
    return emitOptionalError(location, "TileToShapeOp: expected IntTupleType for trg_shape, got ",
                             operands[1].getType());

  auto ordShapeTy = dyn_cast<IntTupleType>(operands[2].getType());
  if (!ordShapeTy)
    return emitOptionalError(location, "TileToShapeOp: expected IntTupleType for ord_shape, got ",
                             operands[2].getType());

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred =
      layoutTileToShape(layoutBuilder, layoutAttr, trgShapeTy.getAttr(), ordShapeTy.getAttr());
  inferredReturnTypes.assign({RebuildLayoutLikeType(blockTy, inferred)});
  return success();
}

//===----------------------------------------------------------------------===//
// Atom and Tiled Mma/Copy ops
//===----------------------------------------------------------------------===//

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

  auto tiledCopyTy = TiledCopyType::get(context, copyAtomTy, layoutTy, tileTy);
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
  }

  auto tiledMmaTy = TiledMmaType::get(context, mmaAtomTy, layoutTy, tileTy);
  inferredReturnTypes.assign({tiledMmaTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionSrcOp) {
  auto tiledCopyTy = dyn_cast<TiledCopyType>(operands[0].getType());
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  auto thrIdxTy = dyn_cast<IntTupleType>(operands[2].getType());
  if (!tiledCopyTy)
    return emitOptionalError(location,
                             "TiledCopyPartitionSrcOp: expected TiledCopyType for operand #0, got ",
                             operands[0].getType());
  if (!memrefTy)
    return emitOptionalError(location,
                             "TiledCopyPartitionSrcOp: expected MemRefType for operand #1, got ",
                             operands[1].getType());
  if (!thrIdxTy)
    return emitOptionalError(location,
                             "TiledCopyPartitionSrcOp: expected IntTupleType for operand #2, got ",
                             operands[2].getType());

  auto copyAtom = dyn_cast<CopyAtomType>(tiledCopyTy.getCopyAtom());
  if (!copyAtom)
    return emitOptionalError(
        location, "TiledCopyPartitionSrcOp: TiledCopyType's copy atom is not a CopyAtomType");

  LayoutAttr tiledLayoutThrVal = tiledCopyTy.getLayoutThrVal().getAttr();
  TileAttr tileMN = tiledCopyTy.getTileMN().getAttr();
  IntTupleAttr thrIdx = thrIdxTy.getAttr();

  LayoutAttr srcLayout = GetLayoutAttrFromLayoutLikeType(operands[1].getType());
  if (!srcLayout)
    return emitOptionalError(location,
                             "TiledCopyPartitionSrcOp: unsupported layout type for operand #1");

  LayoutBuilder<LayoutAttr> builder(context);
  LayoutAttr thrValView =
      layoutTiledCopyThrValViewSrc(builder, copyAtom, tiledLayoutThrVal, tileMN, srcLayout);

  SmallVector<Attribute> coordElems;
  coordElems.push_back(thrIdx);
  coordElems.push_back(IntTupleAttr::getLeafNone(context));
  for (int i = 0; i < srcLayout.rank(); ++i)
    coordElems.push_back(IntTupleAttr::getLeafNone(context));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(context, coordElems));

  IntTupleAttr resultShape =
      intTupleSlice(builder, intTupleExpand(builder, thrValView.getShape(), {2}), sliceCoord);
  IntTupleAttr resultStride =
      intTupleSlice(builder, intTupleExpand(builder, thrValView.getStride(), {2}), sliceCoord);
  LayoutAttr partitioned = LayoutAttr::get(resultShape, resultStride);

  IntTupleAttr thrShape = builder.at(thrValView.getShape(), 0);
  IntTupleAttr thrStride = builder.at(thrValView.getStride(), 0);
  IntTupleAttr offset = layoutCrd2Idx(builder, thrIdx, thrShape, thrStride);
  inferredReturnTypes.assign({applyOffsetOnMemRef(builder, memrefTy, offset, partitioned)});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionDstOp) {
  auto tiledCopyTy = dyn_cast<TiledCopyType>(operands[0].getType());
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  auto thrIdxTy = dyn_cast<IntTupleType>(operands[2].getType());
  if (!tiledCopyTy)
    return emitOptionalError(location,
                             "TiledCopyPartitionDstOp: expected TiledCopyType for operand #0, got ",
                             operands[0].getType());
  if (!memrefTy)
    return emitOptionalError(location,
                             "TiledCopyPartitionDstOp: expected MemRefType for operand #1, got ",
                             operands[1].getType());
  if (!thrIdxTy)
    return emitOptionalError(location,
                             "TiledCopyPartitionDstOp: expected IntTupleType for operand #2, got ",
                             operands[2].getType());

  auto copyAtom = dyn_cast<CopyAtomType>(tiledCopyTy.getCopyAtom());
  if (!copyAtom)
    return emitOptionalError(
        location, "TiledCopyPartitionDstOp: TiledCopyType's copy atom is not a CopyAtomType");

  LayoutAttr tiledLayoutThrVal = tiledCopyTy.getLayoutThrVal().getAttr();
  TileAttr tileMN = tiledCopyTy.getTileMN().getAttr();
  IntTupleAttr thrIdx = thrIdxTy.getAttr();

  LayoutAttr dstLayout = GetLayoutAttrFromLayoutLikeType(operands[1].getType());
  if (!dstLayout)
    return emitOptionalError(location,
                             "TiledCopyPartitionDstOp: unsupported layout type for operand #1");

  LayoutBuilder<LayoutAttr> builder(context);
  LayoutAttr thrValView =
      layoutTiledCopyThrValViewDst(builder, copyAtom, tiledLayoutThrVal, tileMN, dstLayout);

  SmallVector<Attribute> coordElems;
  coordElems.push_back(thrIdx);
  coordElems.push_back(IntTupleAttr::getLeafNone(context));
  for (int i = 0; i < dstLayout.rank(); ++i)
    coordElems.push_back(IntTupleAttr::getLeafNone(context));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(context, coordElems));

  IntTupleAttr resultShape =
      intTupleSlice(builder, intTupleExpand(builder, thrValView.getShape(), {2}), sliceCoord);
  IntTupleAttr resultStride =
      intTupleSlice(builder, intTupleExpand(builder, thrValView.getStride(), {2}), sliceCoord);
  LayoutAttr partitioned = LayoutAttr::get(resultShape, resultStride);

  IntTupleAttr thrShape = builder.at(thrValView.getShape(), 0);
  IntTupleAttr thrStride = builder.at(thrValView.getStride(), 0);
  IntTupleAttr offset = layoutCrd2Idx(builder, thrIdx, thrShape, thrStride);
  inferredReturnTypes.assign({applyOffsetOnMemRef(builder, memrefTy, offset, partitioned)});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyRetileOp) {
  auto tiledCopyTy = dyn_cast<TiledCopyType>(operands[0].getType());
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  if (!tiledCopyTy)
    return emitOptionalError(location,
                             "TiledCopyRetileOp: expected TiledCopyType for operand #0, got ",
                             operands[0].getType());
  if (!memrefTy)
    return emitOptionalError(location,
                             "TiledCopyRetileOp: expected MemRefType for operand #1, got ",
                             operands[1].getType());
  if (memrefTy.getAddressSpace().getValue() != AddressSpace::Register)
    return emitOptionalError(
        location, "TiledCopyRetileOp: expected Register address space for operand #1, got ",
        memrefTy.getAddressSpace());

  auto copyAtom = dyn_cast<CopyAtomType>(tiledCopyTy.getCopyAtom());
  if (!copyAtom)
    return emitOptionalError(location,
                             "TiledCopyRetileOp: TiledCopyType's copy atom is not a CopyAtomType");

  LayoutAttr tiledLayoutThrVal = tiledCopyTy.getLayoutThrVal().getAttr();
  TileAttr tileMN = tiledCopyTy.getTileMN().getAttr();
  auto inputLayout = dyn_cast<LayoutAttr>(memrefTy.getLayout());
  if (!inputLayout)
    return emitOptionalError(location,
                             "TiledCopyRetileOp: MemRefType with ComposedLayout is not supported");
  LayoutBuilder<LayoutAttr> builder(context);
  LayoutAttr retiled =
      layoutTiledCopyRetile(builder, copyAtom, tiledLayoutThrVal, tileMN, inputLayout);

  inferredReturnTypes.assign({RebuildLayoutLikeType(memrefTy, retiled)});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledMmaPartitionOp) {
  auto operandId = properties.as<Properties *>()->operand_id.getValue();
  auto tiledMmaTy = dyn_cast<TiledMmaType>(operands[0].getType());
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  auto thrIdxTy = dyn_cast<IntTupleType>(operands[2].getType());
  if (!tiledMmaTy)
    return emitOptionalError(location,
                             "TiledMmaPartitionOp: expected TiledMmaType for operand #0, got ",
                             operands[0].getType());
  if (!memrefTy)
    return emitOptionalError(location,
                             "TiledMmaPartitionOp: expected MemRefType for operand #1, got ",
                             operands[1].getType());
  if (!thrIdxTy)
    return emitOptionalError(location,
                             "TiledMmaPartitionOp: expected IntTupleType for operand #2, got ",
                             operands[2].getType());

  auto mmaAtom = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
  if (!mmaAtom)
    return emitOptionalError(
        location,
        "TiledMmaPartitionOp: TiledMmaType's mma atom does not implement MmaAtomTypeInterface");

  LayoutAttr atomLayout = tiledMmaTy.getAtomLayout().getAttr();
  TileAttr permutationMNK = tiledMmaTy.getPermutation().getAttr();
  LayoutAttr inputLayout = GetLayoutAttrFromLayoutLikeType(memrefTy);
  if (!inputLayout)
    return emitOptionalError(location,
                             "TiledMmaPartitionOp: unsupported layout type for operand #1");

  LayoutBuilder<LayoutAttr> builder(context);
  LayoutAttr thrValView = layoutTiledMmaThrValOperandView(builder, mmaAtom, atomLayout,
                                                          permutationMNK, operandId, inputLayout);

  IntTupleAttr thrIdx = thrIdxTy.getAttr();
  SmallVector<Attribute> coordElems;
  coordElems.push_back(thrIdx);
  coordElems.push_back(IntTupleAttr::getLeafNone(context));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(context, coordElems));

  IntTupleAttr resultShape = intTupleSlice(builder, thrValView.getShape(), sliceCoord);
  IntTupleAttr resultStride = intTupleSlice(builder, thrValView.getStride(), sliceCoord);
  LayoutAttr partitioned = LayoutAttr::get(intTupleExpand(builder, resultShape, {1}),
                                           intTupleExpand(builder, resultStride, {1}));

  IntTupleAttr thrShape = builder.at(thrValView.getShape(), 0);
  IntTupleAttr thrStride = builder.at(thrValView.getStride(), 0);
  IntTupleAttr offset = layoutCrd2Idx(builder, thrIdx, thrShape, thrStride);
  inferredReturnTypes.assign({applyOffsetOnMemRef(builder, memrefTy, offset, partitioned)});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledMmaPartitionShapeOp) {
  auto operandId = properties.as<Properties *>()->operand_id.getValue();
  auto tiledMmaTy = dyn_cast<TiledMmaType>(operands[0].getType());
  auto shapeTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!tiledMmaTy)
    return emitOptionalError(location,
                             "TiledMmaPartitionShapeOp: expected TiledMmaType for operand #0, got ",
                             operands[0].getType());
  if (!shapeTy)
    return emitOptionalError(location,
                             "TiledMmaPartitionShapeOp: expected IntTupleType for operand #1, got ",
                             operands[1].getType());

  auto mmaAtom = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
  if (!mmaAtom)
    return emitOptionalError(location, "TiledMmaPartitionShapeOp: TiledMmaType's mma atom does not "
                                       "implement MmaAtomTypeInterface");

  LayoutAttr atomLayout = tiledMmaTy.getAtomLayout().getAttr();
  TileAttr permutationMNK = tiledMmaTy.getPermutation().getAttr();

  LayoutBuilder<LayoutAttr> builder(context);
  IntTupleAttr inputShape = shapeTy.getAttr();
  IntTupleAttr compactStride = intTupleCompactColMajor(builder, inputShape);
  LayoutAttr dummyLayout = LayoutAttr::get(inputShape, compactStride);

  LayoutAttr thrValView = layoutTiledMmaThrValOperandView(builder, mmaAtom, atomLayout,
                                                          permutationMNK, operandId, dummyLayout);

  SmallVector<Attribute> coordElems;
  coordElems.push_back(IntTupleAttr::getLeafStatic(context, 0));
  coordElems.push_back(IntTupleAttr::getLeafNone(context));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(context, coordElems));

  IntTupleAttr resultShape = intTupleSlice(builder, thrValView.getShape(), sliceCoord);
  inferredReturnTypes.assign({IntTupleType::get(intTupleExpand(builder, resultShape, {1}))});
  return success();
}

FLY_INFER_RETURN_TYPES(MmaMakeFragmentOp) {
  auto operandId = properties.as<Properties *>()->operand_id.getValue();
  auto tiledMmaTy = dyn_cast<TiledMmaType>(operands[0].getType());
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  if (!tiledMmaTy)
    return emitOptionalError(location,
                             "MmaMakeFragmentOp: expected TiledMmaType for operand #0, got ",
                             operands[0].getType());
  if (!memrefTy)
    return emitOptionalError(location,
                             "MmaMakeFragmentOp: expected MemRefType for operand #1, got ",
                             operands[1].getType());

  auto mmaAtom = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
  if (!mmaAtom)
    return emitOptionalError(
        location,
        "MmaMakeFragmentOp: TiledMmaType's mma atom does not implement MmaAtomTypeInterface");

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

  LayoutAttr inputLayout = GetLayoutAttrFromLayoutLikeType(memrefTy);
  if (!inputLayout)
    return emitOptionalError(location, "MmaMakeFragmentOp: unsupported layout type for operand #1");

  LayoutAttr atomLayout = tiledMmaTy.getAtomLayout().getAttr();
  TileAttr permutationMNK = tiledMmaTy.getPermutation().getAttr();

  LayoutBuilder<LayoutAttr> builder(context);
  LayoutAttr thrValView = layoutTiledMmaThrValOperandView(builder, mmaAtom, atomLayout,
                                                          permutationMNK, operandId, inputLayout);

  SmallVector<Attribute> coordElems;
  coordElems.push_back(IntTupleAttr::getLeafStatic(context, 0));
  coordElems.push_back(IntTupleAttr::getLeafNone(context));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(context, coordElems));

  IntTupleAttr resultShape = intTupleSlice(builder, thrValView.getShape(), sliceCoord);
  IntTupleAttr resultStride = intTupleSlice(builder, thrValView.getStride(), sliceCoord);
  LayoutAttr partitioned = LayoutAttr::get(intTupleExpand(builder, resultShape, {1}),
                                           intTupleExpand(builder, resultStride, {1}));

  LayoutAttr fragmentLayout = layoutMakeFragmentLayout(builder, partitioned);

  inferredReturnTypes.assign({MemRefType::get(
      elemTy, AddressSpaceAttr::get(context, AddressSpace::Register), fragmentLayout)});
  return success();
}

//===----------------------------------------------------------------------===//
// MemRef and Ptr operations
//===----------------------------------------------------------------------===//

FLY_INFER_RETURN_TYPES(PtrToIntOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return emitOptionalError(location, "PtrToIntOp: expected PointerType, got ",
                             operands[0].getType());
  AddressSpace addrSpace = ptrTy.getAddressSpace().getValue();
  unsigned width;

  switch (addrSpace) {
  case AddressSpace::Shared:
    width = 32;
    break;
  case AddressSpace::Global:
    width = 64;
    break;
  default:
    return emitOptionalError(location, "PtrToIntOp: expected Shared or Global address space, got ",
                             addrSpace);
  }

  inferredReturnTypes.assign({IntegerType::get(context, width)});
  return success();
}

FLY_INFER_RETURN_TYPES(AddOffsetOp) {
  auto offsetTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!offsetTy)
    return emitOptionalError(location, "AddOffsetOp: expected IntTupleType for offset, got ",
                             operands[1].getType());

  if (auto ptrTy = dyn_cast<PointerType>(operands[0].getType())) {
    if (!offsetTy.getAttr().isLeafInt())
      return emitOptionalError(
          location, "AddOffsetOp: offset must be a scalar (leaf int) IntTuple, got ", offsetTy);

    int32_t valDiv = ptrTy.getValueDivisibility();
    IntAttr offsetInt = offsetTy.getAttr().extractIntFromLeaf();
    int32_t offsetDiv =
        offsetInt.isStatic() ? std::abs(offsetInt.getValue()) : offsetInt.getDivisibility();
    int32_t newValDiv = (offsetDiv == 0) ? valDiv : utils::divisibilityAdd(valDiv, offsetDiv);

    inferredReturnTypes.assign(
        {PointerType::get(ptrTy.getElemTy(), ptrTy.getAddressSpace(),
                          AlignAttr::get(ptrTy.getElemTy(), newValDiv), ptrTy.getSwizzle())});
    return success();
  }

  if (auto intTupleTy = dyn_cast<IntTupleType>(operands[0].getType())) {
    IntTupleBuilder<IntTupleAttr> builder(context);
    inferredReturnTypes.assign(
        {IntTupleType::get(intTupleAdd(builder, intTupleTy.getAttr(), offsetTy.getAttr()))});
    return success();
  }

  return emitOptionalError(location,
                           "AddOffsetOp: expected PointerType or IntTupleType for ptr, got ",
                           operands[0].getType());
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
  int32_t valDiv = ptrTy.getValueDivisibility();
  int32_t newValDiv = utils::divisibilityApplySwizzle(valDiv, swizzleTy.getAttr());
  inferredReturnTypes.assign(
      {PointerType::get(ptrTy.getElemTy(), ptrTy.getAddressSpace(),
                        AlignAttr::get(ptrTy.getElemTy(), newValDiv), swizzleTy.getAttr())});
  return success();
}

FLY_INFER_RETURN_TYPES(PtrLoadOp) {
  if (auto ptrTy = dyn_cast<PointerType>(operands[0].getType())) {
    inferredReturnTypes.assign({ptrTy.getElemTy()});
    return success();
  }
  if (auto intTupleTy = dyn_cast<IntTupleType>(operands[0].getType())) {
    inferredReturnTypes.assign({intTupleTy});
    return success();
  }
  return emitOptionalError(location, "PtrLoadOp: expected PointerType or IntTupleType, got ",
                           operands[0].getType());
}

FLY_INFER_RETURN_TYPES(MemRefLoadOp) {
  if (auto memrefTy = dyn_cast<MemRefType>(operands[0].getType())) {
    inferredReturnTypes.push_back(memrefTy.getElemTy());
    return success();
  }
  if (auto coordTensorTy = dyn_cast<CoordTensorType>(operands[0].getType())) {
    inferredReturnTypes.push_back(IntTupleType::get(coordTensorTy.getBase()));
    return success();
  }
  return emitOptionalError(location, "MemRefLoadOp: expected MemRefType or CoordTensorType, got ",
                           operands[0].getType());
}

FLY_INFER_RETURN_TYPES(MemRefLoadVecOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[0].getType());
  if (!memrefTy)
    return emitOptionalError(location, "MemRefLoadVecOp: expected MemRefType, got ",
                             operands[0].getType());
  if (memrefTy.getAddressSpace().getValue() != AddressSpace::Register)
    return emitOptionalError(location, "MemRefLoadVecOp: expected Register address space, got ",
                             memrefTy.getAddressSpace());
  auto layoutAttr = dyn_cast<LayoutAttr>(memrefTy.getLayout());
  if (!layoutAttr)
    return emitOptionalError(location,
                             "MemRefLoadVecOp: MemRefType with ComposedLayout is not supported");
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr size = intTupleProduct(builder, layoutAttr.getShape());

  if (!size.isLeafInt() || !size.isStatic())
    return emitOptionalError(
        location, "MemRefLoadVecOp: layout size must be static and leaf int, got ", size);

  inferredReturnTypes.push_back(
      VectorType::get({size.getLeafAsInt().getValue()}, memrefTy.getElemTy()));
  return success();
}

#undef FLY_INFER_RETURN_TYPES
