// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_OPAQUE_VALUE_HPP
#define FLYDSL_CORE_ALGEBRA_OPAQUE_VALUE_HPP

#include "llvm/ADT/Hashing.h"

#include "flydsl/Core/Algebra/Config.hpp"

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

namespace mlir::fly::core {

/// A C++ type tag that can be stored in an OpaqueValue.
template <class T> struct Identity {
  friend constexpr bool operator==(Identity, Identity) { return true; }
  friend constexpr bool operator!=(Identity, Identity) { return false; }
};

/// Type-erased, value-semantic metadata independent of the algebra category.
///
/// Boundary adapters wrap and recover external values such as element types
/// or operation descriptors. Algebra treats the payload as opaque and carries
/// it unchanged across categories; semantic properties needed by algebra,
/// such as bit width, are exposed separately. Category-specific dynamic values
/// belong in Leaf<Cat> or Pointer<Cat> rather than in this metadata.
///
/// The stored C++ type must support equality and hashing. Copies share const
/// storage; equality compares both the stored type and its value. Recovery
/// requires the same decayed C++ type used when wrapping the value.
///
/// This supports in-memory round trips, not serialization. Storing a handle
/// such as mlir::Type preserves that handle without extending the lifetime of
/// its external owner or MLIRContext.
struct OpaqueValue {
public:
  OpaqueValue() = delete;

  template <class T> static OpaqueValue get(T value) {
    using StoredT = std::decay_t<T>;
    return OpaqueValue(std::make_shared<StorageModel<StoredT>>(std::move(value)));
  }

  template <class T> bool is() const { return storage_->typeID() == getTypeID<std::decay_t<T>>(); }

  /// The returned reference borrows from this value's shared storage.
  template <class T> const std::decay_t<T> &get() const;

  template <class T> static OpaqueValue getType() { return get(Identity<T>{}); }

  template <class T> bool isType() const { return is<Identity<T>>(); }

  friend bool operator==(const OpaqueValue &lhs, const OpaqueValue &rhs) {
    return equals(lhs, rhs);
  }

  friend bool operator!=(const OpaqueValue &lhs, const OpaqueValue &rhs) { return !(lhs == rhs); }

  friend llvm::hash_code hash_value(const OpaqueValue &value) { return value.storage_->hash(); }

private:
  static bool equals(const OpaqueValue &lhs, const OpaqueValue &rhs);

  struct StorageConcept {
    virtual ~StorageConcept() = default;
    virtual const void *typeID() const = 0;
    virtual llvm::hash_code hash() const = 0;
    virtual bool equals(const StorageConcept &other) const = 0;
  };

  template <class T> struct StorageModel final : StorageConcept {
    explicit StorageModel(T value) : value(std::move(value)) {}

    const void *typeID() const override { return getTypeID<T>(); }

    llvm::hash_code hash() const override {
      using llvm::hash_value;
      return llvm::hash_combine(typeID(), hash_value(value));
    }

    bool equals(const StorageConcept &other) const override;

    T value;
  };

  explicit OpaqueValue(std::shared_ptr<const StorageConcept> storage)
      : storage_(std::move(storage)) {}

  template <class T> static const void *getTypeID() {
    static const char id = 0;
    return &id;
  }

  std::shared_ptr<const StorageConcept> storage_;
};

template <class T> llvm::hash_code hash_value(Identity<T>);

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

template <class T> const std::decay_t<T> &OpaqueValue::get() const {
  using StoredT = std::decay_t<T>;
  FLYDSL_CORE_ASSERT(is<StoredT>());
  return static_cast<const StorageModel<StoredT> &>(*storage_).value;
}

inline bool OpaqueValue::equals(const OpaqueValue &lhs, const OpaqueValue &rhs) {
  if (lhs.storage_ == rhs.storage_)
    return true;
  return lhs.storage_->equals(*rhs.storage_);
}

template <class T> bool OpaqueValue::StorageModel<T>::equals(const StorageConcept &other) const {
  if (typeID() != other.typeID())
    return false;
  return value == static_cast<const StorageModel &>(other).value;
}

template <class T> llvm::hash_code hash_value(Identity<T>) {
  static const char id = 0;
  return llvm::hash_value(&id);
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_OPAQUE_VALUE_HPP
