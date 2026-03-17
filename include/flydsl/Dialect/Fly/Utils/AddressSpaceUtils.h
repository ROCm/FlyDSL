#ifndef FLYDSL_DIALECT_FLY_UTILS_ADDRESSSPACEUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_ADDRESSSPACEUTILS_H

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

inline unsigned mapToLLVMAddressSpace(AddressSpace space) {
  switch (space) {
  case AddressSpace::Global:
    return 1;
  case AddressSpace::Shared:
    return 3;
  case AddressSpace::Register:
    return 5;
  case AddressSpace::BufferDesc:
    return 8;
  }
  return 0;
}

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_ADDRESSSPACEUTILS_H
