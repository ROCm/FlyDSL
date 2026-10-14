// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_CONFIG_HPP
#define FLYDSL_CORE_ALGEBRA_CONFIG_HPP

#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#ifndef FLYDSL_CORE_ENABLE_ASSERTIONS
#define FLYDSL_CORE_ENABLE_ASSERTIONS 1
#endif

// Range inference is temporarily disabled for the Fly integration. Keep one
// default shared by every evaluation category and by installed headers.
#ifndef FLYDSL_CORE_ENABLE_RANGE_INFERENCE
#define FLYDSL_CORE_ENABLE_RANGE_INFERENCE 0
#endif

#ifndef FLYDSL_CORE_ASSERT_STRINGIFY
#define FLYDSL_CORE_ASSERT_STRINGIFY_IMPL(x) #x
#define FLYDSL_CORE_ASSERT_STRINGIFY(x) FLYDSL_CORE_ASSERT_STRINGIFY_IMPL(x)
#endif

#ifndef FLYDSL_CORE_ASSERT_MSG
#if FLYDSL_CORE_ENABLE_ASSERTIONS
#include <cassert>
#define FLYDSL_CORE_ASSERT_MSG(condition, message) assert((condition) && (message))
#else
#define FLYDSL_CORE_ASSERT_MSG(condition, message) ((void)0)
#endif
#endif // FLYDSL_CORE_ASSERT_MSG

#ifndef FLYDSL_CORE_ASSERT
#define FLYDSL_CORE_ASSERT(condition)                                                              \
  FLYDSL_CORE_ASSERT_MSG((condition), "assertion failed at " __FILE__                              \
                                      ":" FLYDSL_CORE_ASSERT_STRINGIFY(__LINE__))
#endif // FLYDSL_CORE_ASSERT

#if defined(__has_cpp_attribute) && __has_cpp_attribute(clang::lifetimebound)
#define FLYDSL_CORE_LIFETIMEBOUND [[clang::lifetimebound]]
#else
#define FLYDSL_CORE_LIFETIMEBOUND
#endif

namespace mlir::fly::core {

// clang-format off

//===----------------------------------------------------------------------===//
// Layout components
//===----------------------------------------------------------------------===//

template <class Cat> struct Leaf;
template <class Cat> struct IntTupleRef;
template <class Cat> struct IntTuple;
template <class Cat> struct Layout;
template <class Cat> struct Tile;
template <class Cat> struct Swizzle;
template <class Cat> struct CoordSwizzle;
template <class Cat> struct ComposedLayout;

template <class Cat> struct Pointer;
template <class Cat, class Engine> struct Tensor;

template <class Cat> struct CopyAtom;
template <class Cat> struct MMAAtom;
template <class Cat> struct TiledCopy;
template <class Cat> struct TiledMMA;
template <class Cat> struct ThrCopy;
template <class Cat> struct ThrMMA;

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

///
/// Category
///

template <class T> struct CategoryOf;

template <class Cat> struct CategoryOf<Leaf<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<IntTupleRef<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<IntTuple<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<Layout<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<Tile<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<Swizzle<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<CoordSwizzle<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<ComposedLayout<Cat>> { using type = Cat; };

template <class Cat> struct CategoryOf<Pointer<Cat>> { using type = Cat; };
template <class Cat, class Engine> struct CategoryOf<Tensor<Cat, Engine>> { using type = Cat; };

template <class Cat> struct CategoryOf<CopyAtom<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<TiledCopy<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<ThrCopy<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<MMAAtom<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<TiledMMA<Cat>> { using type = Cat; };
template <class Cat> struct CategoryOf<ThrMMA<Cat>> { using type = Cat; };

template <class T> using category_of = typename CategoryOf<std::decay_t<T>>::type;

///
/// Types
///

template <class T> struct is_leaf : std::false_type {};
template <class T> struct is_int_tuple : std::false_type {};
template <class T> struct is_layout : std::false_type {};
template <class T> struct is_tile : std::false_type {};
template <class T> struct is_composed_layout : std::false_type {};
template <class T> struct is_tensor : std::false_type {};
template <class T> struct is_memref : std::false_type {};
template <class T> struct is_coord_tensor : std::false_type {};

template <class Cat> struct is_leaf<Leaf<Cat>> : std::true_type {};
template <class Cat> struct is_int_tuple<IntTuple<Cat>> : std::true_type {};
template <class Cat> struct is_int_tuple<IntTupleRef<Cat>> : std::true_type {};

template <class Cat> struct is_layout<Layout<Cat>> : std::true_type {};
template <class Cat> struct is_layout<ComposedLayout<Cat>> : std::true_type {};

template <class Cat> struct is_tile<Tile<Cat>> : std::true_type {};
template <class Cat> struct is_composed_layout<ComposedLayout<Cat>> : std::true_type {};

template <class Cat, class Engine> struct is_tensor<Tensor<Cat, Engine>> : std::true_type {};
template <class Cat>               struct is_memref<Tensor<Cat, Pointer<Cat>>> : std::true_type {};
template <class Cat>               struct is_coord_tensor<Tensor<Cat, IntTuple<Cat>>> : std::true_type {};


template <class T> constexpr bool is_leaf_v = is_leaf<std::decay_t<T>>::value;
template <class T> constexpr bool is_int_tuple_v = is_int_tuple<std::decay_t<T>>::value;
template <class T> constexpr bool is_layout_v = is_layout<std::decay_t<T>>::value;
template <class T> constexpr bool is_tile_v = is_tile<std::decay_t<T>>::value;
template <class T> constexpr bool is_composed_layout_v = is_composed_layout<std::decay_t<T>>::value;

template <class T> constexpr bool is_tensor_v = is_tensor<std::decay_t<T>>::value;
template <class T> constexpr bool is_memref_v = is_memref<std::decay_t<T>>::value;
template <class T> constexpr bool is_coord_tensor_v = is_coord_tensor<std::decay_t<T>>::value;

// clang-format on

enum class ErrorCode : uint8_t {
  RatioOverflow,
  DivisionByZero,
  DynamicRatioUnsupported,
  UnsupportedRatioOperation,
  InvalidComparison,
  InexactDivision,
  IncompatibleShapeDivision,
  InvalidDynamicRange,
  RangeOutsideBitWidth,
  FastDivModRequiresI32,
  RangeRequiresDynamicInt,
  FastDivModRequiresDynamicInt,
  InvalidBasisScale,
  TooManyBasisModes,
  BasisModeOutOfRange,
  IncompatibleBasis,
  UnsupportedBasisOperation,
  UnsupportedBasisComparison,
  UnsupportedBitwiseOperand,
  ExpectedLeafOperand,
  InvalidLayout,
  UnsupportedDynamicComplement,
  NonInjectiveLayout,
  InvalidSSAValue,
  MissingEmitContext,
  IntegerOverflow,
  TupleCapacityExceeded,
  // Illegal layout-algebra operations.
  IncompatibleProfile,
  TilerRankExceeded,
  IndexOutOfRange,
  ExpectedStaticOperand,
  InvalidRecastFactor,
  UnsupportedSwizzleComposition,
  IncompatibleElementType,
  Last = IncompatibleElementType,
};

inline llvm::StringRef get_error_message(ErrorCode c) {
  switch (c) {
  case ErrorCode::RatioOverflow:
    return "!ratio-overflow";
  case ErrorCode::DivisionByZero:
    return "!division-by-zero";
  case ErrorCode::DynamicRatioUnsupported:
    return "!dynamic-ratio-unsupported";
  case ErrorCode::UnsupportedRatioOperation:
    return "!unsupported-ratio-operation";
  case ErrorCode::InvalidComparison:
    return "!invalid-comparison";
  case ErrorCode::InexactDivision:
    return "!inexact-division";
  case ErrorCode::IncompatibleShapeDivision:
    return "!incompatible-shape-division";
  case ErrorCode::InvalidDynamicRange:
    return "!invalid-dynamic-range";
  case ErrorCode::RangeOutsideBitWidth:
    return "!range-outside-bit-width";
  case ErrorCode::FastDivModRequiresI32:
    return "!fast-divmod-requires-i32";
  case ErrorCode::RangeRequiresDynamicInt:
    return "!range-requires-dynamic-int";
  case ErrorCode::FastDivModRequiresDynamicInt:
    return "!fast-divmod-requires-dynamic-int";
  case ErrorCode::InvalidBasisScale:
    return "!invalid-basis-scale";
  case ErrorCode::TooManyBasisModes:
    return "!too-many-basis-modes";
  case ErrorCode::BasisModeOutOfRange:
    return "!basis-mode-out-of-range";
  case ErrorCode::IncompatibleBasis:
    return "!incompatible-basis";
  case ErrorCode::UnsupportedBasisOperation:
    return "!unsupported-basis-operation";
  case ErrorCode::UnsupportedBasisComparison:
    return "!unsupported-basis-comparison";
  case ErrorCode::UnsupportedBitwiseOperand:
    return "!unsupported-bitwise-operand";
  case ErrorCode::ExpectedLeafOperand:
    return "!expected-leaf-operand";
  case ErrorCode::InvalidLayout:
    return "!invalid-layout";
  case ErrorCode::UnsupportedDynamicComplement:
    return "!unsupported-dynamic-complement";
  case ErrorCode::InvalidSSAValue:
    return "!invalid-ssa-value";
  case ErrorCode::MissingEmitContext:
    return "!missing-emit-context";
  case ErrorCode::IntegerOverflow:
    return "!integer-overflow";
  case ErrorCode::TupleCapacityExceeded:
    return "!tuple-capacity-exceeded";
  case ErrorCode::NonInjectiveLayout:
    return "!non-injective-layout";
  case ErrorCode::IncompatibleProfile:
    return "!incompatible-profile";
  case ErrorCode::TilerRankExceeded:
    return "!tiler-rank-exceeded";
  case ErrorCode::IndexOutOfRange:
    return "!index-out-of-range";
  case ErrorCode::ExpectedStaticOperand:
    return "!expected-static-operand";
  case ErrorCode::InvalidRecastFactor:
    return "!invalid-recast-factor";
  case ErrorCode::UnsupportedSwizzleComposition:
    return "!unsupported-swizzle-composition";
  case ErrorCode::IncompatibleElementType:
    return "!incompatible-element-type";
  }
  return "!unknown";
}

/// The algebra operation an error was detected in or propagated through.
/// Values are grouped by layer; the gaps between groups are reserved.
enum class AlgebraOp : uint8_t {
  None = 0x00,
  // Leaf arithmetic.
  Add = 0x01,
  Sub,
  Mul,
  Mod,
  Floor,
  Ceil,
  Min,
  Max,
  Gcd,
  Exact,
  Safe,
  Shape,
  BitAnd,
  BitOr,
  BitXor,
  Shl,
  Shr,
  Lt,
  Eq,
  Ne,
  // IntTuple algorithms.
  Transform = 0x20,
  FilterTuple,
  Front,
  Back,
  Take,
  Select,
  Group,
  Append,
  Prepend,
  Replace,
  Insert,
  Remove,
  Unflatten,
  Zip,
  Zip2By,
  InnerProduct,
  ElemScale,
  FilterZeros,
  Slice,
  Dice,
  Crd2Idx,
  Idx2Crd,
  Crd2Crd,
  CompactOrder,
  TupleDiv,
  ProductLike,
  // Layout algebra.
  Coalesce = 0x50,
  CoalesceX,
  Filter,
  Composition,
  Complement,
  RightInverse,
  LeftInverse,
  LogicalDivide,
  ZippedDivide,
  TiledDivide,
  FlatDivide,
  LogicalProduct,
  ZippedProduct,
  TiledProduct,
  FlatProduct,
  BlockedProduct,
  RakedProduct,
  TileToShape,
  DomainDistribute,
  Upcast,
  Downcast,
  RecastLayout,
  SwizzleApply,
  // Tensors and atoms.
  InnerPartition = 0x90,
  OuterPartition,
  MaxCommonLayout,
  MaxCommonVector,
  TiledCopyPartition,
  TiledCopyRetile,
  TiledMmaPartition,
  TiledMmaPermutation,
  Last = TiledMmaPermutation,
};

inline llvm::StringRef get_algebra_op_name(AlgebraOp op) {
  switch (op) {
  case AlgebraOp::None:
    return "none";
  case AlgebraOp::Add:
    return "add";
  case AlgebraOp::Sub:
    return "sub";
  case AlgebraOp::Mul:
    return "mul";
  case AlgebraOp::Mod:
    return "mod";
  case AlgebraOp::Floor:
    return "floor_div";
  case AlgebraOp::Ceil:
    return "ceil_div";
  case AlgebraOp::Min:
    return "min";
  case AlgebraOp::Max:
    return "max";
  case AlgebraOp::Gcd:
    return "gcd";
  case AlgebraOp::Exact:
    return "exact_div";
  case AlgebraOp::Safe:
    return "safe_div";
  case AlgebraOp::Shape:
    return "shape_div";
  case AlgebraOp::BitAnd:
    return "bit_and";
  case AlgebraOp::BitOr:
    return "bit_or";
  case AlgebraOp::BitXor:
    return "bit_xor";
  case AlgebraOp::Shl:
    return "shl";
  case AlgebraOp::Shr:
    return "shr";
  case AlgebraOp::Lt:
    return "lt";
  case AlgebraOp::Eq:
    return "eq";
  case AlgebraOp::Ne:
    return "ne";
  case AlgebraOp::Transform:
    return "transform";
  case AlgebraOp::FilterTuple:
    return "filter_tuple";
  case AlgebraOp::Front:
    return "front";
  case AlgebraOp::Back:
    return "back";
  case AlgebraOp::Take:
    return "take";
  case AlgebraOp::Select:
    return "select";
  case AlgebraOp::Group:
    return "group";
  case AlgebraOp::Append:
    return "append";
  case AlgebraOp::Prepend:
    return "prepend";
  case AlgebraOp::Replace:
    return "replace";
  case AlgebraOp::Insert:
    return "insert";
  case AlgebraOp::Remove:
    return "remove";
  case AlgebraOp::Unflatten:
    return "unflatten";
  case AlgebraOp::Zip:
    return "zip";
  case AlgebraOp::Zip2By:
    return "zip2_by";
  case AlgebraOp::InnerProduct:
    return "inner_product";
  case AlgebraOp::ElemScale:
    return "elem_scale";
  case AlgebraOp::FilterZeros:
    return "filter_zeros";
  case AlgebraOp::Slice:
    return "slice";
  case AlgebraOp::Dice:
    return "dice";
  case AlgebraOp::Crd2Idx:
    return "crd2idx";
  case AlgebraOp::Idx2Crd:
    return "idx2crd";
  case AlgebraOp::Crd2Crd:
    return "crd2crd";
  case AlgebraOp::CompactOrder:
    return "compact_order";
  case AlgebraOp::TupleDiv:
    return "tuple_div";
  case AlgebraOp::ProductLike:
    return "product_like";
  case AlgebraOp::Coalesce:
    return "coalesce";
  case AlgebraOp::CoalesceX:
    return "coalesce_x";
  case AlgebraOp::Filter:
    return "filter";
  case AlgebraOp::Composition:
    return "composition";
  case AlgebraOp::Complement:
    return "complement";
  case AlgebraOp::RightInverse:
    return "right_inverse";
  case AlgebraOp::LeftInverse:
    return "left_inverse";
  case AlgebraOp::LogicalDivide:
    return "logical_divide";
  case AlgebraOp::ZippedDivide:
    return "zipped_divide";
  case AlgebraOp::TiledDivide:
    return "tiled_divide";
  case AlgebraOp::FlatDivide:
    return "flat_divide";
  case AlgebraOp::LogicalProduct:
    return "logical_product";
  case AlgebraOp::ZippedProduct:
    return "zipped_product";
  case AlgebraOp::TiledProduct:
    return "tiled_product";
  case AlgebraOp::FlatProduct:
    return "flat_product";
  case AlgebraOp::BlockedProduct:
    return "blocked_product";
  case AlgebraOp::RakedProduct:
    return "raked_product";
  case AlgebraOp::TileToShape:
    return "tile_to_shape";
  case AlgebraOp::DomainDistribute:
    return "domain_distribute";
  case AlgebraOp::Upcast:
    return "upcast";
  case AlgebraOp::Downcast:
    return "downcast";
  case AlgebraOp::RecastLayout:
    return "recast_layout";
  case AlgebraOp::SwizzleApply:
    return "swizzle_apply";
  case AlgebraOp::InnerPartition:
    return "inner_partition";
  case AlgebraOp::OuterPartition:
    return "outer_partition";
  case AlgebraOp::MaxCommonLayout:
    return "max_common_layout";
  case AlgebraOp::MaxCommonVector:
    return "max_common_vector";
  case AlgebraOp::TiledCopyPartition:
    return "tiled_copy_partition";
  case AlgebraOp::TiledCopyRetile:
    return "tiled_copy_retile";
  case AlgebraOp::TiledMmaPartition:
    return "tiled_mma_partition";
  case AlgebraOp::TiledMmaPermutation:
    return "tiled_mma_permutation";
  }
  return "unknown";
}

/// An ErrorCode together with the algebra operations that locate it, packed
/// into the 64-bit payload of an error leaf:
///
///    63      56 55     40 39     24 23      8 7      0
///   +----------+---------+---------+---------+--------+
///   | pending  | frame 2 | frame 1 | frame 0 | reason |
///   +----------+---------+---------+---------+--------+
///
/// A frame is {op:8, mode:8} with the op in the low byte; an empty frame has
/// op None. Frame 0 is the operation that detected the error, frame 1 the
/// operation enclosing it, and frame 2 the outermost operation the error has
/// propagated through so far. `pending` is a mode recorded by `withMode` that
/// no frame has taken yet, stored as mode + 1 so that zero means none.
class ErrorInfo {
public:
  static constexpr unsigned kNumFrames = 3;
  /// The mode of a frame that is not tied to one mode.
  static constexpr uint8_t kNoMode = 0xff;

  ErrorInfo(ErrorCode reason) { llvm::Bitfield::set<Reason>(raw_, reason); }

  static ErrorInfo fromRaw(uint64_t raw) {
    auto info = ErrorInfo(ErrorCode{});
    info.raw_ = raw;
    return info;
  }
  uint64_t raw() const { return raw_; }

  ErrorCode reason() const { return llvm::Bitfield::get<Reason>(raw_); }

  unsigned depth() const {
    unsigned frames = 0;
    while (frames < kNumFrames && op(frames) != AlgebraOp::None)
      ++frames;
    return frames;
  }

  AlgebraOp op(unsigned frame) const {
    switch (frame) {
    case 0:
      return llvm::Bitfield::get<FrameOp<0>>(raw_);
    case 1:
      return llvm::Bitfield::get<FrameOp<1>>(raw_);
    default:
      FLYDSL_CORE_ASSERT(frame == 2);
      return llvm::Bitfield::get<FrameOp<2>>(raw_);
    }
  }

  uint8_t mode(unsigned frame) const {
    switch (frame) {
    case 0:
      return llvm::Bitfield::get<FrameMode<0>>(raw_);
    case 1:
      return llvm::Bitfield::get<FrameMode<1>>(raw_);
    default:
      FLYDSL_CORE_ASSERT(frame == 2);
      return llvm::Bitfield::get<FrameMode<2>>(raw_);
    }
  }

  /// Records that the error passed through `op`, at `mode` when the operation
  /// works mode by mode; without a mode, `op` takes the one left by
  /// `withMode`. Once every frame is used, `op` replaces the outermost frame,
  /// so the root cause and its direct caller are always kept. An operation
  /// recursing into itself keeps a single frame with the innermost mode, which
  /// leaves the other frames for the operations around it.
  ErrorInfo withFrame(AlgebraOp op, int64_t mode = -1) const {
    FLYDSL_CORE_ASSERT(op != AlgebraOp::None);
    auto pending = llvm::Bitfield::get<PendingMode>(raw_);
    auto encoded = mode >= 0      ? encodeMode(mode)
                   : pending != 0 ? static_cast<uint8_t>(pending - 1)
                                  : kNoMode;
    auto result = *this;
    llvm::Bitfield::set<PendingMode>(result.raw_, 0);
    auto frames = depth();
    if (frames != 0 && this->op(frames - 1) == op) {
      if (this->mode(frames - 1) == kNoMode)
        result.setMode(frames - 1, encoded);
      return result;
    }
    switch (std::min(frames, kNumFrames - 1)) {
    case 0:
      setFrame<0>(result.raw_, op, encoded);
      break;
    case 1:
      setFrame<1>(result.raw_, op, encoded);
      break;
    default:
      setFrame<2>(result.raw_, op, encoded);
      break;
    }
    return result;
  }

  /// Records that the error came from mode `mode` of a mode-wise step that
  /// does not know its operation, such as `transform_layout`; the operation
  /// that reports the error next takes the mode for its frame.
  ErrorInfo withMode(int64_t mode) const {
    FLYDSL_CORE_ASSERT(mode >= 0);
    auto result = *this;
    llvm::Bitfield::set<PendingMode>(result.raw_, static_cast<uint8_t>(encodeMode(mode) + 1));
    return result;
  }

  friend bool operator==(ErrorInfo lhs, ErrorInfo rhs) { return lhs.raw_ == rhs.raw_; }
  friend bool operator!=(ErrorInfo lhs, ErrorInfo rhs) { return !(lhs == rhs); }

private:
  using Reason = llvm::Bitfield::Element<ErrorCode, 0, 8, ErrorCode::Last>;
  template <unsigned I>
  using FrameOp = llvm::Bitfield::Element<AlgebraOp, 8 + 16 * I, 8, AlgebraOp::Last>;
  template <unsigned I> using FrameMode = llvm::Bitfield::Element<uint8_t, 16 + 16 * I, 8>;
  using PendingMode = llvm::Bitfield::Element<uint8_t, 56, 8>;
  static_assert(
      llvm::Bitfield::areContiguous<Reason, FrameOp<0>, FrameMode<0>, FrameOp<1>, FrameMode<1>,
                                    FrameOp<2>, FrameMode<2>, PendingMode>(),
      "error frames and the pending mode must be packed after the reason");
  static_assert(PendingMode::NextBit == 64, "the error must fit the 64-bit leaf payload");

  /// A mode as a frame stores it; modes past the last one saturate.
  static uint8_t encodeMode(int64_t mode) {
    return static_cast<uint8_t>(std::min<int64_t>(mode, kNoMode - 1));
  }

  template <unsigned I> static void setFrame(uint64_t &raw, AlgebraOp op, uint8_t mode) {
    llvm::Bitfield::set<FrameOp<I>>(raw, op);
    llvm::Bitfield::set<FrameMode<I>>(raw, mode);
  }

  void setMode(unsigned frame, uint8_t mode) {
    switch (frame) {
    case 0:
      llvm::Bitfield::set<FrameMode<0>>(raw_, mode);
      break;
    case 1:
      llvm::Bitfield::set<FrameMode<1>>(raw_, mode);
      break;
    default:
      FLYDSL_CORE_ASSERT(frame == 2);
      llvm::Bitfield::set<FrameMode<2>>(raw_, mode);
      break;
    }
  }

  uint64_t raw_ = 0;
};

/// Prints the reason followed by the frames, innermost first, e.g.
/// `!tiler-rank-exceeded (in composition[mode 1] <- logical_divide)`.
inline void print(ErrorInfo info, llvm::raw_ostream &os) {
  os << get_error_message(info.reason());
  auto depth = info.depth();
  for (unsigned i = 0; i < depth; ++i) {
    os << (i == 0 ? " (in " : " <- ") << get_algebra_op_name(info.op(i));
    if (info.mode(i) != ErrorInfo::kNoMode)
      os << "[mode " << static_cast<unsigned>(info.mode(i)) << "]";
  }
  if (depth != 0)
    os << ')';
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_CONFIG_HPP
