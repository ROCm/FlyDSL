#include "flir/FlirRocmDialect.h"
#include "flir/FlirRocmDialect.cpp.inc"

void mlir::flir::rocm::FlirRocmDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "flir/FlirRocmOps.cpp.inc"
  >();
}
