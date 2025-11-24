#include "cute/CuteNvgpuDialect.h"
#include "cute/CuteNvgpuDialect.cpp.inc"

void mlir::cute::nvgpu::CuteNvgpuDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "cute/CuteNvgpuOps.cpp.inc"
  >();
}
