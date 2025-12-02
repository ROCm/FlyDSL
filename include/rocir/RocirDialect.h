#ifndef ROCIR_DIALECT_H
#define ROCIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::rocir {

// Forward declarations for type storage
namespace detail {
struct IntTypeStorage;
struct RankedTypeStorage;
struct StructureTypeStorage;
}

// Type declarations
class IntType : public Type::TypeBase<IntType, Type, detail::IntTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.int";
  static IntType get(MLIRContext *context);
};

class ShapeType : public Type::TypeBase<ShapeType, Type, detail::StructureTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.shape";
  static ShapeType get(MLIRContext *context, int rank);
  static ShapeType get(MLIRContext *context, ArrayRef<int32_t> structure);
  int getRank() const;
  ArrayRef<int32_t> getStructure() const;
};

class StrideType : public Type::TypeBase<StrideType, Type, detail::StructureTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.stride";
  static StrideType get(MLIRContext *context, int rank);
  static StrideType get(MLIRContext *context, ArrayRef<int32_t> structure);
  int getRank() const;
  ArrayRef<int32_t> getStructure() const;
};

class LayoutType : public Type::TypeBase<LayoutType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "rocir.layout";
  static LayoutType get(MLIRContext *context, int rank);
  int getRank() const;
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
#define GET_OP_CLASSES
#include "rocir/RocirOps.h.inc"

#endif // ROCIR_DIALECT_H
