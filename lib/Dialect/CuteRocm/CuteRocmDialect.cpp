#include "cute/CuteRocmDialect.h"
#include "cute/CuteRocmDialect.cpp.inc"

void mlir::cute::rocm::CuteRocmDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "cute/CuteRocmOps.cpp.inc"
  >();
}
