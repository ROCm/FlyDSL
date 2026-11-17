// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Dialect/Fly/Utils/CoreBridge.h"

namespace mlir::fly {

using core::EmitOp;

//===----------------------------------------------------------------------===//
// Emit context
//===----------------------------------------------------------------------===//

namespace {

/// The Fly-level hooks: a Fly pointer advances by fly.add_offset, and a
/// fast-divmod handle is a !fly.fast_divmod value of the fast division ops,
/// which convert-fly-to-rocdl lowers with Core's defaults.
class FlyEmitHooks : public core::EmitHooks {
public:
  FailureOr<Value> addOffset(core::EmitContext &ctx, const core::Pointer<EmitOp> &base,
                             core::IntTupleRef<EmitOp> offset,
                             const core::Pointer<core::Trait> &resultRef) const override {
    PointerType baseType = PointerType::get(ctx.builder.getContext(), core::to_trait(base));
    PointerType resultType = PointerType::get(ctx.builder.getContext(), resultRef);
    if (!baseType || !resultType || !base.value() || base.value().getType() != baseType)
      return failure();
    auto value = materializeValue(offset, ctx);
    if (failed(value))
      return failure();
    Value result = AddOffsetOp::create(ctx.builder, ctx.location, base.value(), *value);
    if (result.getType() != resultType)
      return failure();
    return result;
  }

  Type getFastDivModType(core::EmitContext &ctx, unsigned bitWidth) const override {
    return FastDivModType::get(ctx.builder.getContext(), ctx.builder.getIntegerType(bitWidth));
  }

  Value makeFastDivMod(core::EmitContext &ctx, Value divisor) const override {
    return MakeFastDivModOp::create(ctx.builder, ctx.location, divisor);
  }

  Value getFastDivModDivisor(core::EmitContext &ctx, Value handle) const override {
    return FastDivModGetDivisorOp::create(ctx.builder, ctx.location, handle);
  }

  std::pair<Value, Value> fastDivMod(core::EmitContext &ctx, Value dividend,
                                     Value handle) const override {
    auto op = FastDivModOp::create(ctx.builder, ctx.location, dividend, handle);
    return {op.getQuotient(), op.getRemainder()};
  }
};

const core::EmitHooks *getFlyHooks() {
  static const FlyEmitHooks hooks;
  return &hooks;
}

} // namespace

core::EmitContext makeDefaultEmitContext(OpBuilder &builder, Location loc) {
  return core::EmitContext(builder, loc, getFlyHooks());
}

//===----------------------------------------------------------------------===//
// Import
//===----------------------------------------------------------------------===//

namespace {

/// The dynamic scalars of `value`, whose profile is `ref`, extracted with
/// fly.get_leaves; none for a static value. A memref's pointer is extracted
/// separately.
template <class Ref>
core::EmitBindings extractScalars(const Ref &ref, Value value, core::EmitContext &ctx) {
  core::EmitBindings bindings;
  auto types = core::get_binding_types(ref, ctx);
  if (types.empty())
    return bindings;
  auto extract = GetLeavesOp::create(ctx.builder, ctx.location, types, value);
  llvm::append_range(bindings.scalars, extract.getResults());
  return bindings;
}

/// Binds `ref` to the scalars extracted from `value`.
template <class Ref> auto bindExtracted(const Ref &ref, Value value, core::EmitContext &ctx) {
  return core::bind(ref, extractScalars(ref, value, ctx), ctx);
}

/// Imports a value of one of `FlyTypes` as the Core variant `Variant`.
template <class Variant, class... FlyTypes>
FailureOr<Variant> importOneOf(ValueImporter &importer, Value value) {
  FailureOr<Variant> result = failure();
  auto tryImport = [&](auto typed) {
    if (!typed)
      return false;
    if (auto part = importer(typed); succeeded(part))
      result = Variant(std::move(*part));
    return true;
  };
  (void)(tryImport(dyn_cast<TypedValue<FlyTypes>>(value)) || ...);
  return result;
}

} // namespace

ValueImporter::ValueImporter(core::EmitContext &ctx) : ctx(ctx) {
  if (!ctx.hooks)
    ctx.hooks = getFlyHooks();
}

FailureOr<core::IntTuple<EmitOp>> ValueImporter::operator()(TypedValue<IntTupleType> value) {
  const auto &ref = value.getType().getRef();
  Operation *producer = value.getDefiningOp();
  if (ref.isStatic() ||
      !(producer && isa<MakeIntTupleOp, MakeShapeOp, MakeStrideOp, MakeCoordOp>(producer)))
    return bindExtracted(ref, value, ctx);
  core::EmitBindings bindings;
  llvm::append_range(bindings.scalars, producer->getOperands());
  return core::bind(ref, bindings, ctx);
}

FailureOr<core::Layout<EmitOp>> ValueImporter::operator()(TypedValue<LayoutType> value) {
  auto make = value.getDefiningOp<MakeLayoutOp>();
  if (value.getType().isStatic() || !make)
    return bindExtracted(value.getType().getRef(), value, ctx);
  auto shape = (*this)(make.getShape());
  auto stride = (*this)(make.getStride());
  if (failed(shape) || failed(stride))
    return failure();
  return core::Layout<EmitOp>(std::move(*shape), std::move(*stride));
}

FailureOr<core::ComposedLayout<EmitOp>>
ValueImporter::operator()(TypedValue<ComposedLayoutType> value) {
  auto make = value.getDefiningOp<MakeComposedLayoutOp>();
  if (value.getType().isStatic() || !make)
    return bindExtracted(value.getType().getRef(), value, ctx);
  auto inner = importOneOf<core::ComposedInner<EmitOp>, LayoutType, ComposedLayoutType, SwizzleType,
                           CoordSwizzleType>(*this, make.getInner());
  auto offset = importValue<IntTupleType>(make.getOffset(), ctx);
  auto outer = importOneOf<core::ComposedOuter<EmitOp>, LayoutType, ComposedLayoutType>(
      *this, make.getOuter());
  if (failed(inner) || failed(offset) || failed(outer))
    return failure();
  return core::ComposedLayout<EmitOp>(std::move(*inner), std::move(*offset), std::move(*outer));
}

FailureOr<core::Swizzle<EmitOp>> ValueImporter::operator()(TypedValue<SwizzleType> value) {
  return core::bind(value.getType().getRef(), core::EmitBindings{}, ctx);
}

FailureOr<core::CoordSwizzle<EmitOp>>
ValueImporter::operator()(TypedValue<CoordSwizzleType> value) {
  return core::bind(value.getType().getRef(), core::EmitBindings{}, ctx);
}

FailureOr<core::Tile<EmitOp>> ValueImporter::operator()(TypedValue<TileType> value) {
  auto make = value.getDefiningOp<MakeTileOp>();
  if (value.getType().isStatic() || !make)
    return bindExtracted(value.getType().getRef(), value, ctx);
  core::EmitBindings bindings;
  llvm::append_range(bindings.scalars, make.getDyncElems());
  return core::bind(value.getType().getRef(), bindings, ctx);
}

FailureOr<core::Pointer<EmitOp>> ValueImporter::operator()(TypedValue<PointerType> value) {
  return core::Pointer<EmitOp>(value.getType().getRef(), value, ctx);
}

FailureOr<core::MemRef<EmitOp>> ValueImporter::operator()(TypedValue<MemRefType> value) {
  if (auto view = value.getDefiningOp<MakeViewOp>()) {
    auto pointer = importValue<PointerType>(view.getIter(), ctx);
    auto layout = importOneOf<core::AnyLayout<EmitOp>, LayoutType, ComposedLayoutType>(
        *this, view.getLayout());
    if (failed(pointer) || failed(layout))
      return failure();
    return core::MemRef<EmitOp>(std::move(*pointer), std::move(*layout));
  }
  // Any other producer: the layout scalars and the pointer are extracted.
  const auto &ref = value.getType().getRef();
  auto bindings = extractScalars(ref, value, ctx);
  bindings.pointers.push_back(GetIterOp::create(ctx.builder, ctx.location, value));
  return core::bind(ref, bindings, ctx);
}

FailureOr<core::CoordTensor<EmitOp>> ValueImporter::operator()(TypedValue<CoordTensorType> value) {
  auto view = value.getDefiningOp<MakeViewOp>();
  if (!view)
    return bindExtracted(value.getType().getRef(), value, ctx);
  auto base = importValue<IntTupleType>(view.getIter(), ctx);
  auto layout =
      importOneOf<core::AnyLayout<EmitOp>, LayoutType, ComposedLayoutType>(*this, view.getLayout());
  if (failed(base) || failed(layout))
    return failure();
  return core::CoordTensor<EmitOp>(std::move(*base), std::move(*layout));
}

FailureOr<core::TiledCopy<EmitOp>> ValueImporter::operator()(TypedValue<TiledCopyType> value) {
  return core::bind(value.getType().getRef(), core::EmitBindings{}, ctx);
}

FailureOr<core::TiledMMA<EmitOp>> ValueImporter::operator()(TypedValue<TiledMmaType> value) {
  return core::bind(value.getType().getRef(), core::EmitBindings{}, ctx);
}

//===----------------------------------------------------------------------===//
// Materialize
//===----------------------------------------------------------------------===//

FailureOr<Value> ValueMaterializer::operator()(core::IntTupleRef<EmitOp> value) {
  auto bindings = core::get_bindings(value, ctx);
  if (failed(bindings))
    return failure();
  auto type = IntTupleType::get(ctx.builder.getContext(), core::to_trait(value));
  if (!type)
    return failure();
  if (bindings->scalars.empty())
    return StaticOp::create(ctx.builder, ctx.location, type).getResult();
  return MakeIntTupleOp::create(ctx.builder, ctx.location, type, bindings->scalars).getResult();
}

FailureOr<Value> ValueMaterializer::operator()(const core::Layout<EmitOp> &value) {
  if (value.isError())
    return failure();
  auto type = LayoutType::get(ctx.builder.getContext(), core::to_trait(value));
  if (!type)
    return failure();
  if (value.isStatic())
    return StaticOp::create(ctx.builder, ctx.location, type).getResult();
  auto shape = (*this)(value.shape());
  auto stride = (*this)(value.stride());
  if (failed(shape) || failed(stride))
    return failure();
  return MakeLayoutOp::create(ctx.builder, ctx.location, type, *shape, *stride).getResult();
}

FailureOr<Value> ValueMaterializer::operator()(const core::ComposedLayout<EmitOp> &value) {
  if (value.isError())
    return failure();
  auto type = ComposedLayoutType::get(ctx.builder.getContext(), core::to_trait(value));
  if (!type)
    return failure();
  if (value.isStatic())
    return StaticOp::create(ctx.builder, ctx.location, type).getResult();
  auto inner = (*this)(value.inner());
  auto offset = (*this)(value.offset());
  auto outer = (*this)(value.outer());
  if (failed(inner) || failed(offset) || failed(outer))
    return failure();
  return MakeComposedLayoutOp::create(ctx.builder, ctx.location, type, *inner, *offset, *outer)
      .getResult();
}

FailureOr<Value> ValueMaterializer::operator()(core::Swizzle<EmitOp> value) {
  auto type = SwizzleType::get(ctx.builder.getContext(), core::to_trait(value));
  if (!type)
    return failure();
  return StaticOp::create(ctx.builder, ctx.location, type).getResult();
}

FailureOr<Value> ValueMaterializer::operator()(const core::CoordSwizzle<EmitOp> &value) {
  auto type = CoordSwizzleType::get(ctx.builder.getContext(), core::to_trait(value));
  if (!type)
    return failure();
  return StaticOp::create(ctx.builder, ctx.location, type).getResult();
}

FailureOr<Value> ValueMaterializer::operator()(const core::AnyLayout<EmitOp> &value) {
  return std::visit(*this, value);
}

FailureOr<Value> ValueMaterializer::operator()(const core::ComposedInner<EmitOp> &value) {
  return std::visit(*this, value);
}

FailureOr<Value> ValueMaterializer::operator()(const core::Pointer<EmitOp> &value) {
  PointerType type = PointerType::get(ctx.builder.getContext(), core::to_trait(value));
  if (!type || !value.value() || value.value().getType() != type)
    return failure();
  return value.value();
}

namespace {

template <class TensorT>
FailureOr<Value> materializeTensor(ValueMaterializer &materializer, const TensorT &tensor,
                                   Type type) {
  if (!type)
    return failure();
  auto engine = materializer(tensor.data());
  auto layout = materializer(tensor.layout());
  if (failed(engine) || failed(layout))
    return failure();
  Value result =
      MakeViewOp::create(materializer.ctx.builder, materializer.ctx.location, *engine, *layout);
  if (result.getType() != type)
    return failure();
  return result;
}

} // namespace

FailureOr<Value> ValueMaterializer::operator()(const core::MemRef<EmitOp> &value) {
  return materializeTensor(*this, value,
                           MemRefType::get(ctx.builder.getContext(), core::to_trait(value)));
}

FailureOr<Value> ValueMaterializer::operator()(const core::CoordTensor<EmitOp> &value) {
  return materializeTensor(*this, value,
                           CoordTensorType::get(ctx.builder.getContext(), core::to_trait(value)));
}

} // namespace mlir::fly
