#ifndef CUTE_PASSES_H
#define CUTE_PASSES_H

#include "mlir/Pass/Pass.h"
#include "cute/CutePasses.h.inc"

namespace mlir {
namespace cute {
std::unique_ptr<Pass> createCuteToRocmPass();
}
}

#endif // CUTE_PASSES_H
