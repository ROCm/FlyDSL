#ifndef CUTE_TYPES_H
#define CUTE_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir::cute {

namespace detail {
struct IntTypeStorage;
struct RankedTypeStorage;
}

class IntType : public Type::TypeBase<IntType, Type, detail::IntTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.int";
  static IntType get(MLIRContext *context);
};

class ShapeType : public Type::TypeBase<ShapeType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.shape";
  static ShapeType get(MLIRContext *context, int rank);
  int getRank() const;
};

class StrideType : public Type::TypeBase<StrideType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.stride";
  static StrideType get(MLIRContext *context, int rank);
  int getRank() const;
};

class LayoutType : public Type::TypeBase<LayoutType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.layout";
  static LayoutType get(MLIRContext *context, int rank);
  int getRank() const;
};

class CoordType : public Type::TypeBase<CoordType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.coord";
  static CoordType get(MLIRContext *context, int rank);
  int getRank() const;
};

} // namespace mlir::cute

#endif // CUTE_TYPES_H
