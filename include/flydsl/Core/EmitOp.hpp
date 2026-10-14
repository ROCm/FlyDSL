// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_EMIT_OP_HPP
#define FLYDSL_CORE_EMIT_OP_HPP

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Core/Trait.hpp"
#include "flydsl/Core/Visitor.hpp"

#include <optional>

namespace mlir::fly::core {

struct EmitOp {};

struct EmitContext;

class EmitHooks {
public:
  virtual ~EmitHooks();

  /// `pointer` advanced by `offset`, a single leaf, to a pointer of profile
  /// `result`. The default handles byte-addressable LLVM pointers. An override
  /// must check its SSA type against `result`, whose alignment can change the
  /// dialect type.
  virtual mlir::FailureOr<mlir::Value> addOffset(EmitContext &ctx, const Pointer<EmitOp> &pointer,
                                                 IntTupleRef<EmitOp> offset,
                                                 const Pointer<Trait> &result) const;

  /// The handle of `divisor`, an i32 or i64 in [1, 2^(width-1)). The magic
  /// numbers are computed here and nowhere else.
  virtual mlir::Value makeFastDivMod(EmitContext &ctx, mlir::Value divisor) const;
  /// The handle type of a `bitWidth`-bit fast-divmod divisor. The final LLVM
  /// lowering fixes it to emit_op::getFastDivModType; another type is for a
  /// dialect that converts its handle to that one later.
  virtual mlir::Type getFastDivModType(EmitContext &ctx, unsigned bitWidth) const;
  /// The divisor `handle` was made of.
  virtual mlir::Value getFastDivModDivisor(EmitContext &ctx, mlir::Value handle) const;
  /// The quotient and remainder of `dividend`, an integer of the handle width
  /// in [0, 2^(width-1)), by the divisor of `handle`.
  virtual std::pair<mlir::Value, mlir::Value> fastDivMod(EmitContext &ctx, mlir::Value dividend,
                                                         mlir::Value handle) const;
};

/// Borrowed by dynamic leaves and pointers. The context and its builder must outlive their algebra
/// evaluation.
/// materialization always respects the caller's current insertion point.
struct EmitContext {
  mlir::OpBuilder &builder;
  mlir::Location location;
  /// The emission hooks; null selects the defaults for lowered LLVM values.
  const EmitHooks *hooks;

  EmitContext(mlir::OpBuilder &builder, mlir::Location location, const EmitHooks *hooks = nullptr);

  const EmitHooks &getHooks() const;
};

template <> struct Leaf<EmitOp> : public LeafProperty {
  using Cat = EmitOp;

  Leaf() : LeafProperty{} {}
  explicit Leaf(LeafProperty property, mlir::Value value = {}, EmitContext *context = nullptr);

  static Leaf getSInt(int64_t v) { return Leaf(Leaf<Trait>::getSInt(v)); }
  static Leaf getOne() { return getSInt(1); }
  static Leaf getZero() { return getSInt(0); }
  static Leaf getNone() { return Leaf(Leaf<Trait>::getNone()); }
  static Leaf getRatio(int32_t n, int32_t d) { return Leaf(Leaf<Trait>::getRatio(n, d)); }
  static Leaf getError(ErrorInfo info) { return Leaf(Leaf<Trait>::getError(info)); }
  /// A dynamic EmitOp leaf requires a binding; use getDyn or the constructor.
  static Leaf getDInt(LogWidth = LogWidth::I32, int32_t = 1) {
    return getError(ErrorCode::InvalidSSAValue);
  }
  static Leaf getDyn(mlir::Value value, EmitContext &context, int32_t div = 1);
  static Leaf getBasis(Leaf scale, llvm::ArrayRef<int32_t> modes);
  static std::optional<Leaf> fromString(llvm::StringRef text);

  mlir::Value value() const { return value_; }
  EmitContext *context() const { return context_; }
  bool isSound() const;
  Leaf scale() const { return Leaf(Leaf<Trait>(*this).scale(), value_, context_); }

  Leaf withModes(Leaf proto) const;
  Leaf withFastDivMod(EmitContext *context = nullptr) const;
  Leaf withRange(int32_t lo, int32_t hi) const;
  mlir::FailureOr<mlir::Value> materialize(EmitContext &ctx) const;

  friend bool operator==(Leaf a, Leaf b) {
    return Leaf<Trait>(a) == Leaf<Trait>(b) && a.value_ == b.value_;
  }
  friend bool operator!=(Leaf a, Leaf b) { return !(a == b); }

private:
  mlir::Value value_;
  EmitContext *context_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Pointer<EmitOp>
//===----------------------------------------------------------------------===//

/// A bound pointer retains the original dialect SSA type and all opaque
/// metadata. The default hooks handle byte-addressable LLVM pointers;
/// dialect pointers/fat pointers/subbyte addressing override EmitHooks::addOffset.
template <> struct Pointer<EmitOp> : public PointerProperty {
  Pointer(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment, mlir::Value value,
          Swizzle<EmitOp> swizzle = {}, int64_t bitAlignment = 0,
          OpaqueValue addressSpace = OpaqueValue::getType<void>(),
          int32_t storageElementBitWidth = 0);

  Pointer(PointerProperty property, mlir::Value value, EmitContext &context);

  mlir::Value value() const { return value_; }
  EmitContext *context() const { return context_; }

  bool isValid() const { return bool(value_); }
  Swizzle<EmitOp> swizzle() const { return PointerProperty::swizzle<EmitOp>(); }

private:
  mlir::Value value_;
  EmitContext *context_ = nullptr;
};

#define FLYDSL_CORE_CATEGORY EmitOp
#include "flydsl/Core/Category.def"

//===----------------------------------------------------------------------===//
// Binding
//===----------------------------------------------------------------------===//

struct EmitBindings {
  llvm::SmallVector<mlir::Value, 8> scalars;
  llvm::SmallVector<mlir::Value, 2> pointers;
};

namespace emit_op {

struct BindLeaf {
  EmitContext &context;
  const EmitBindings &bindings;
  size_t index = 0;

  Leaf<EmitOp> operator()(Leaf<Trait> leaf) {
    if (leaf.isStatic())
      return Leaf<EmitOp>(leaf);
    return Leaf<EmitOp>(leaf, bindings.scalars[index++], &context);
  }
};

struct BindPointer {
  EmitContext &context;
  const EmitBindings &bindings;
  size_t index = 0;

  Pointer<EmitOp> operator()(const Pointer<Trait> &pointer) {
    return Pointer<EmitOp>(pointer, bindings.pointers[index++], context);
  }
};

} // namespace emit_op

/// bind :: (Value<Trait>, [mlir::Value]) -> Value<EmitOp>
template <class TraitValue>
auto bind(const TraitValue &profile, const EmitBindings &bindings, EmitContext &ctx)
    -> mlir::FailureOr<decltype(transform_value<EmitOp>(profile, emit_op::BindLeaf{ctx, bindings},
                                                        emit_op::BindPointer{ctx, bindings}))> {
  size_t scalarIndex = 0, pointerIndex = 0;
  bool valid = true;
  visit_value(
      profile,
      [&](Leaf<Trait> leaf) {
        if (leaf.isError())
          valid = false;
        if (!leaf.isStatic()) {
          if (scalarIndex >= bindings.scalars.size() ||
              Leaf<EmitOp>(leaf, bindings.scalars[scalarIndex], &ctx).isError())
            valid = false;
          ++scalarIndex;
        }
      },
      [&](const Pointer<Trait> &) {
        if (pointerIndex >= bindings.pointers.size() || !bindings.pointers[pointerIndex])
          valid = false;
        ++pointerIndex;
      });
  if (!valid || scalarIndex != bindings.scalars.size() || pointerIndex != bindings.pointers.size())
    return mlir::failure();
  return transform_value<EmitOp>(profile, emit_op::BindLeaf{ctx, bindings},
                                 emit_op::BindPointer{ctx, bindings});
}

/// get_bindings :: Value<EmitOp> -> [mlir::Value]
template <class EmitValue>
mlir::FailureOr<EmitBindings> get_bindings(const EmitValue &value, EmitContext &ctx) {
  auto bindings = EmitBindings{};
  bool valid = true;
  visit_value(
      value,
      [&](Leaf<EmitOp> leaf) {
        if (leaf.isError() || !leaf.isSound())
          valid = false;
        if (!leaf.isStatic()) {
          if (!leaf.value())
            valid = false;
          bindings.scalars.push_back(leaf.value());
        }
      },
      [&](const Pointer<EmitOp> &pointer) {
        if (!pointer.isValid())
          valid = false;
        bindings.pointers.push_back(pointer.value());
      });
  if (!valid)
    return mlir::failure();
  return bindings;
}

//===----------------------------------------------------------------------===//
// LLVM representation
//===----------------------------------------------------------------------===//

namespace emit_op {

/// getLLVMStructType :: [mlir::Type] -> mlir::LLVM::LLVMStructType
mlir::LLVM::LLVMStructType getLLVMStructType(mlir::MLIRContext &context,
                                             llvm::ArrayRef<mlir::Type> fields);

/// buildLLVMStruct :: [mlir::Value] -> mlir::LLVMStructValue
mlir::Value buildLLVMStruct(EmitContext &ctx, llvm::ArrayRef<mlir::Value> parts);

/// extractLLVMField :: (mlir::LLVMStructValue, [Int]) -> mlir::Value
///
/// The field of `aggregate` at `path` through its nested structs.
mlir::Value extractLLVMField(EmitContext &ctx, mlir::Value aggregate, llvm::ArrayRef<int64_t> path);

/// The path of each dynamic scalar through the nested structs of a lowered
/// value, in binding order.
using FieldPaths = llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 8>;

/// getFastDivModType :: Int -> mlir::LLVM::LLVMStructType
///
/// The lowered handle of a `bitWidth`-bit fast-divmod divisor: the struct
/// {divisor, multiplier, shift}, each `bitWidth` bits wide. This is the handle
/// of the final LLVM ABI, whichever hooks compute it.
mlir::LLVM::LLVMStructType getFastDivModType(mlir::MLIRContext &context, unsigned bitWidth);

/// makeFastDivModF64 :: i32 -> mlir::LLVMStructValue
///
/// The lowered handle of the i32 `divisor`, as EmitHooks::makeFastDivMod
/// makes it but with the multiplier from one f64 division and an exact
/// correction. On a GPU with full-rate FP64 this is about a third of the
/// instructions a 64-bit integer division expands to.
mlir::Value makeFastDivModF64(EmitContext &ctx, mlir::Value divisor);

} // namespace emit_op

/// get_llvm_type :: Value<Trait> -> mlir::LLVMStructType
///
/// A fully static value has no LLVM type (null). Pointer engines and atoms are converted by the
/// dialect.
template <class TraitValue>
mlir::LLVM::LLVMStructType get_llvm_type(const TraitValue &profile, mlir::MLIRContext &context) {
  auto type = reduce_value<mlir::Type>(
      profile,
      [&](Leaf<Trait> leaf) -> mlir::Type {
        if (leaf.isStatic())
          return {};
        if (leaf.hasFastDivMod())
          return emit_op::getFastDivModType(context, leaf.bitWidth());
        return mlir::IntegerType::get(&context, leaf.bitWidth());
      },
      [&](llvm::ArrayRef<mlir::Type> fields) -> mlir::Type {
        return emit_op::getLLVMStructType(context, fields);
      });
  return mlir::cast_or_null<mlir::LLVM::LLVMStructType>(type);
}

/// collect_llvm_scalars :: (Value<Trait>, mlir::LLVMStructValue) -> [mlir::Value]
///
/// The dynamic scalars of `aggregate`, a value of the LLVM type of `profile`
/// (null for a fully static profile), in the order used by bind/get_bindings.
/// The profile locates each scalar, so a fast-divmod handle is one scalar
/// whatever its struct. The type is checked before any extraction.
template <class TraitValue>
mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>>
collect_llvm_scalars(const TraitValue &profile, mlir::Value aggregate, EmitContext &ctx) {
  mlir::Type type = get_llvm_type(profile, *ctx.builder.getContext());
  if ((aggregate ? aggregate.getType() : mlir::Type()) != type)
    return mlir::failure();
  // As in get_llvm_type, a struct has a field for each part with a dynamic
  // scalar.
  auto paths = reduce_value<emit_op::FieldPaths>(
      profile,
      [](Leaf<Trait> leaf) {
        auto paths = emit_op::FieldPaths{};
        if (!leaf.isStatic())
          paths.emplace_back();
        return paths;
      },
      [](llvm::ArrayRef<emit_op::FieldPaths> parts) {
        auto paths = emit_op::FieldPaths{};
        int64_t field = 0;
        for (const auto &part : parts) {
          if (part.empty())
            continue;
          for (const auto &path : part) {
            auto &scalar = paths.emplace_back();
            scalar.push_back(field);
            scalar.append(path.begin(), path.end());
          }
          ++field;
        }
        return paths;
      });
  auto scalars = llvm::SmallVector<mlir::Value, 8>{};
  for (const auto &path : paths)
    scalars.push_back(emit_op::extractLLVMField(ctx, aggregate, path));
  return scalars;
}

/// build_llvm_value :: Value<EmitOp> -> mlir::LLVMStructValue
///
/// `value` as a value of its LLVM type; null when it is fully static.
template <class EmitValue>
mlir::FailureOr<mlir::Value> build_llvm_value(const EmitValue &value, EmitContext &ctx) {
  if (mlir::failed(get_bindings(value, ctx)))
    return mlir::failure();
  return reduce_value<mlir::Value>(
      value, [](Leaf<EmitOp> leaf) { return leaf.isStatic() ? mlir::Value() : leaf.value(); },
      [&](llvm::ArrayRef<mlir::Value> parts) { return emit_op::buildLLVMStruct(ctx, parts); });
}

/// bind_llvm_value :: (Value<Trait>, mlir::LLVMStructValue) -> Value<EmitOp>
///
/// Rebinds `profile` to `aggregate`, a value of its LLVM type (null for a
/// fully static profile), e.g. after the final TypeConverter mapped a block
/// argument, call result or region yield. The type is checked before any
/// extraction, so malformed input leaves no partial IR behind.
template <class TraitValue>
auto bind_llvm_value(const TraitValue &profile, mlir::Value aggregate,
                     EmitContext &ctx) -> decltype(bind(profile, EmitBindings{}, ctx)) {
  // Hooks that bind lowered LLVM values must keep the handle of that ABI.
  visit_value(
      profile,
      [&](Leaf<Trait> leaf) {
        FLYDSL_CORE_ASSERT(
            !leaf.hasFastDivMod() ||
            ctx.getHooks().getFastDivModType(ctx, leaf.bitWidth()) ==
                emit_op::getFastDivModType(*ctx.builder.getContext(), leaf.bitWidth()));
      },
      [](const Pointer<Trait> &) {});
  auto scalars = collect_llvm_scalars(profile, aggregate, ctx);
  if (mlir::failed(scalars))
    return mlir::failure();
  auto bindings = EmitBindings{};
  bindings.scalars = std::move(*scalars);
  return bind(profile, bindings, ctx);
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_EMIT_OP_HPP
