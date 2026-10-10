// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_LEAF_HPP
#define FLYDSL_CORE_ALGEBRA_LEAF_HPP

#include "llvm/ADT/ArrayRef.h"

#include "flydsl/Core/Algebra/Config.hpp"

#include "llvm/ADT/Hashing.h"
#include <cstdint>

namespace mlir::fly::core {

/// static leaf has bit0 = 0.
enum LeafKind : uint8_t {
  StaticInt = 0x00,
  DynamicInt = 0x01,
  None = 0x02,
  Ratio = 0x04,
  Error = 0x06,
  Basis = 1 << 7,
};

constexpr uint8_t kMaxBasisModes = 7;

enum DynIntFlag : uint8_t {
  FastDivMod = 1 << 0,
  Range = 1 << 1,
};

enum LogWidth : uint8_t {
  I1 = 0,
  I8 = 3,
  I16 = 4,
  I32 = 5,
  I64 = 6,
};

struct LeafProperty {
public:
  uint32_t kind_ : 8;
  uint32_t logWidth_ : 4;
  uint32_t flags_ : 8;
  uint32_t : 12; // unused
  int32_t div_;
  union {
    int64_t payload64_;
    struct {
      int32_t lo;
      int32_t hi;
    } payload32_;
  };
  uint8_t modes_[kMaxBasisModes];
  uint8_t nmodes_;

  bool isError() const { return kind() == LeafKind::Error; }
  bool isNone() const { return kind() == LeafKind::None; }

  /// None is a wildcard slice marker but participates in algebra as static 0.
  bool isSInt() const { return kind_ == LeafKind::StaticInt || kind_ == LeafKind::None; }
  bool isDInt() const { return kind() == LeafKind::DynamicInt; }
  bool isRatio() const { return kind() == LeafKind::Ratio; }
  bool isBasis() const { return (kind_ & LeafKind::Basis) != 0; }
  bool isBasis(LeafKind scaleLeafKind) const {
    return kind_ == static_cast<uint32_t>(LeafKind::Basis | scaleLeafKind);
  }

  bool isInt() const { return isSInt() || isDInt(); }

  /// Static forms, including static Basis, keep bit 0 clear.
  bool isStatic() const { return (kind_ & 1u) == 0; }
  /// None matches only zero: it is never a unit or any other static value.
  bool isSInt(int64_t value) const {
    return (kind_ == LeafKind::StaticInt && payload64_ == value) ||
           (kind_ == LeafKind::None && value == 0);
  }

  bool isTop() const { return kind_ == LeafKind::DynamicInt && div_ == 1 && flags_ == 0; }
  bool isZero() const { return isSInt(0); }
  bool isOne() const { return isSInt(1); }

  bool hasFastDivMod() const {
    FLYDSL_CORE_ASSERT(isDInt());
    return (flags_ & DynIntFlag::FastDivMod) != 0;
  }
  bool hasRange() const {
    FLYDSL_CORE_ASSERT(isDInt());
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
    return (flags_ & DynIntFlag::Range) != 0;
#else
    return false;
#endif
  }

  LeafKind kind() const { return static_cast<LeafKind>(kind_); }

  ErrorCode errorCode() const { return errorInfo().reason(); }
  ErrorInfo errorInfo() const {
    FLYDSL_CORE_ASSERT(isError());
    return ErrorInfo::fromRaw(static_cast<uint64_t>(payload64_));
  }

  int64_t staticValue() const {
    FLYDSL_CORE_ASSERT(isSInt());
    return payload64_;
  }

  LogWidth logWidth() const {
    FLYDSL_CORE_ASSERT(isSInt() || isDInt() || isRatio());
    return static_cast<LogWidth>(logWidth_);
  }
  int32_t bitWidth() const { return 1u << logWidth_; }

  int32_t divisibility() const {
    FLYDSL_CORE_ASSERT(kind_ & LeafKind::DynamicInt);
    return div_;
  }

  int32_t rangeMin() const {
    FLYDSL_CORE_ASSERT(hasRange());
    return payload32_.lo;
  }
  int32_t rangeMax() const {
    FLYDSL_CORE_ASSERT(hasRange());
    return payload32_.hi;
  }

  int32_t num() const {
    FLYDSL_CORE_ASSERT(isRatio());
    return payload32_.lo;
  }
  int32_t den() const {
    FLYDSL_CORE_ASSERT(isRatio());
    return payload32_.hi;
  }

  uint8_t modeCount() const { return nmodes_; }
  uint8_t mode(unsigned i) const {
    FLYDSL_CORE_ASSERT(isBasis() && i < nmodes_);
    return modes_[i];
  }
  llvm::ArrayRef<uint8_t> modes() const { return llvm::ArrayRef<uint8_t>(modes_, nmodes_); }

  friend llvm::hash_code hash_value(const LeafProperty &leaf) {
    auto hash = llvm::hash_combine(leaf.kind_);
    if (leaf.isBasis())
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(leaf.modes().begin(), leaf.modes().end()));
    auto kind = leaf.kind_ & ~LeafKind::Basis;
    if (kind == LeafKind::None)
      return hash;
    if (kind == LeafKind::DynamicInt) {
      hash = llvm::hash_combine(hash, leaf.logWidth_, leaf.div_, leaf.flags_);
      if (leaf.flags_ & DynIntFlag::Range)
        hash = llvm::hash_combine(hash, leaf.payload32_.lo, leaf.payload32_.hi);
      return hash;
    }
    if (kind == LeafKind::Ratio)
      return llvm::hash_combine(hash, leaf.logWidth_, leaf.payload32_.lo, leaf.payload32_.hi);
    // The frames of an error only locate it; they do not change its identity.
    if (kind == LeafKind::Error)
      return llvm::hash_combine(
          hash, ErrorInfo::fromRaw(static_cast<uint64_t>(leaf.payload64_)).reason());
    hash = llvm::hash_combine(hash, leaf.payload64_);
    return llvm::hash_combine(hash, leaf.logWidth_);
  }

  friend bool operator==(const LeafProperty &lhs, const LeafProperty &rhs) {
    if (lhs.kind_ != rhs.kind_)
      return false;
    if (lhs.isBasis() && lhs.modes() != rhs.modes())
      return false;
    auto kind = lhs.kind_ & ~LeafKind::Basis;
    if (kind == LeafKind::None)
      return true;
    if (kind == LeafKind::DynamicInt)
      return lhs.logWidth_ == rhs.logWidth_ && lhs.div_ == rhs.div_ && lhs.flags_ == rhs.flags_ &&
             (!(lhs.flags_ & DynIntFlag::Range) ||
              (lhs.payload32_.lo == rhs.payload32_.lo && lhs.payload32_.hi == rhs.payload32_.hi));
    if (kind == LeafKind::Ratio)
      return lhs.logWidth_ == rhs.logWidth_ && lhs.payload32_.lo == rhs.payload32_.lo &&
             lhs.payload32_.hi == rhs.payload32_.hi;
    if (kind == LeafKind::Error)
      return ErrorInfo::fromRaw(static_cast<uint64_t>(lhs.payload64_)).reason() ==
             ErrorInfo::fromRaw(static_cast<uint64_t>(rhs.payload64_)).reason();
    return lhs.payload64_ == rhs.payload64_ && lhs.logWidth_ == rhs.logWidth_;
  }
  friend bool operator!=(const LeafProperty &lhs, const LeafProperty &rhs) { return !(lhs == rhs); }
};

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_LEAF_HPP
