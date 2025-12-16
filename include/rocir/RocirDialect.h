#ifndef ROCIR_DIALECT_H
#define ROCIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "rocir/RocirTypes.h"

namespace mlir::rocir {

// Type declarations
class IntType : public Type::TypeBase<IntType, Type, detail::IntTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.int";
  static IntType get(MLIRContext *context);
};

class ShapeType : public Type::TypeBase<ShapeType, Type, detail::StructuredTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.shape";
  static ShapeType get(MLIRContext *context, int rank);
  /// Create a ShapeType from a canonical textual spec, e.g. "(9,(4,8))" or "(?,(?,?))".
  static ShapeType get(MLIRContext *context, ::llvm::StringRef spec);
  /// Create a ShapeType from a tuple key (structure encoding + flattened dims).
  static ShapeType get(MLIRContext *context,
                       ::llvm::ArrayRef<int32_t> structure,
                       ::llvm::ArrayRef<int64_t> dims);
  int getRank() const;
  ArrayRef<int32_t> getStructure() const;
  ::llvm::StringRef getSpec() const;
  ::llvm::ArrayRef<int64_t> getDims() const;
};

class StrideType : public Type::TypeBase<StrideType, Type, detail::StructuredTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.stride";
  static StrideType get(MLIRContext *context, int rank);
  /// Create a StrideType from a canonical textual spec, e.g. "(59,(13,1))" or "(?,(?,?))".
  static StrideType get(MLIRContext *context, ::llvm::StringRef spec);
  /// Create a StrideType from a tuple key (structure encoding + flattened dims).
  static StrideType get(MLIRContext *context,
                        ::llvm::ArrayRef<int32_t> structure,
                        ::llvm::ArrayRef<int64_t> dims);
  int getRank() const;
  ArrayRef<int32_t> getStructure() const;
  ::llvm::StringRef getSpec() const;
  ::llvm::ArrayRef<int64_t> getDims() const;
};

class LayoutType : public Type::TypeBase<LayoutType, Type, detail::LayoutTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.layout";
  /// Rank-only layout (backward compatible): layout<rank> / layout<(?,?,?)>
  static LayoutType get(MLIRContext *context, int rank);
  /// Layout with structured shape/stride type information.
  static LayoutType get(MLIRContext *context, ShapeType shape, StrideType stride);
  /// Low-level constructor from tuple encodings (structure + flattened dims).
  static LayoutType get(MLIRContext *context,
                        ::llvm::ArrayRef<int32_t> shapeStructure,
                        ::llvm::ArrayRef<int64_t> shapeDims,
                        ::llvm::ArrayRef<int32_t> strideStructure,
                        ::llvm::ArrayRef<int64_t> strideDims);
  int getRank() const;
  ::llvm::StringRef getShapeSpec() const;
  ::llvm::ArrayRef<int32_t> getShapeStructure() const;
  ::llvm::ArrayRef<int64_t> getShapeDims() const;
  ::llvm::StringRef getStrideSpec() const;
  ::llvm::ArrayRef<int32_t> getStrideStructure() const;
  ::llvm::ArrayRef<int64_t> getStrideDims() const;
  ShapeType getShapeType() const;
  StrideType getStrideType() const;
};

class CoordType : public Type::TypeBase<CoordType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.coord";
  static CoordType get(MLIRContext *context, int rank);
  int getRank() const;
};

} // namespace mlir::rocir

// Include generated dialect declarations
#include "rocir/RocirDialect.h.inc"

// Include generated operation declarations
#endif // ROCIR_DIALECT_H
