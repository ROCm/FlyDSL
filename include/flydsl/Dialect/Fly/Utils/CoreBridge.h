// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_COREBRIDGE_H
#define FLYDSL_DIALECT_FLY_UTILS_COREBRIDGE_H

#include "flydsl/Core/EmitOp.hpp"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

core::EmitContext makeDefaultEmitContext(OpBuilder &builder, Location loc);

/// Binds a Fly value to the EmitOp description of its type.
///
/// A constructor binds its operands component by component; any other producer (a block argument, a
/// call result) binds the scalars fly.get_leaves extracts. A static value binds no scalars and
/// inspects no producer.
/// Compile-time descriptors (swizzles, tiled copies and MMAs) bind their static description.
struct ValueImporter {
  core::EmitContext &ctx;

  /// Fly values bind at the Fly level: a context without hooks gets
  /// the Fly ones.
  explicit ValueImporter(core::EmitContext &ctx);

  FailureOr<core::IntTuple<core::EmitOp>> operator()(TypedValue<IntTupleType> value);
  FailureOr<core::Layout<core::EmitOp>> operator()(TypedValue<LayoutType> value);
  FailureOr<core::ComposedLayout<core::EmitOp>> operator()(TypedValue<ComposedLayoutType> value);
  FailureOr<core::Swizzle<core::EmitOp>> operator()(TypedValue<SwizzleType> value);
  FailureOr<core::CoordSwizzle<core::EmitOp>> operator()(TypedValue<CoordSwizzleType> value);
  FailureOr<core::Tile<core::EmitOp>> operator()(TypedValue<TileType> value);
  FailureOr<core::Pointer<core::EmitOp>> operator()(TypedValue<PointerType> value);
  FailureOr<core::MemRef<core::EmitOp>> operator()(TypedValue<MemRefType> value);
  FailureOr<core::CoordTensor<core::EmitOp>> operator()(TypedValue<CoordTensorType> value);
  FailureOr<core::TiledCopy<core::EmitOp>> operator()(TypedValue<TiledCopyType> value);
  FailureOr<core::TiledMMA<core::EmitOp>> operator()(TypedValue<TiledMmaType> value);
};

/// Rebuilds a Fly value from an EmitOp value: fly.static when it is static, its
/// constructor applied to its rebuilt components otherwise.
struct ValueMaterializer {
  core::EmitContext &ctx;

  template <class T, std::enable_if_t<core::is_int_tuple_v<T>, int> = 0>
  FailureOr<Value> operator()(const T &value) {
    return (*this)(value.asRef());
  }
  FailureOr<Value> operator()(core::IntTupleRef<core::EmitOp> value);

  FailureOr<Value> operator()(const core::Layout<core::EmitOp> &value);
  FailureOr<Value> operator()(const core::ComposedLayout<core::EmitOp> &value);
  FailureOr<Value> operator()(core::Swizzle<core::EmitOp> value);
  FailureOr<Value> operator()(const core::CoordSwizzle<core::EmitOp> &value);
  FailureOr<Value> operator()(const core::AnyLayout<core::EmitOp> &value);
  FailureOr<Value> operator()(const core::ComposedInner<core::EmitOp> &value);
  FailureOr<Value> operator()(const core::Pointer<core::EmitOp> &value);
  FailureOr<Value> operator()(const core::MemRef<core::EmitOp> &value);
  FailureOr<Value> operator()(const core::CoordTensor<core::EmitOp> &value);
};

/// Binds `value`, of Fly type `FlyType`; fails for a value of another type.
template <class FlyType> auto importValue(Value value, core::EmitContext &ctx) {
  ValueImporter importer{ctx};
  using Result = decltype(importer(std::declval<TypedValue<FlyType>>()));
  auto typed = dyn_cast<TypedValue<FlyType>>(value);
  return typed ? importer(typed) : Result(failure());
}

template <class T> FailureOr<Value> materializeValue(const T &value, core::EmitContext &ctx) {
  return ValueMaterializer{ctx}(value);
}

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_COREBRIDGE_H
