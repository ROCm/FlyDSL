//===- RocirTypes.cpp - Rocir Type Implementation -------------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirTypes.h"

using namespace mlir;
using namespace mlir::rocir;

// Implement getStructure() methods for compatibility
ArrayRef<int32_t> ShapeType::getStructure() const {
  return ArrayRef<int32_t>();
}

ArrayRef<int32_t> StrideType::getStructure() const {
  return ArrayRef<int32_t>();
}

