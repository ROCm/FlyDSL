//===- FlirDCE.cpp - Trivial DCE (standalone) ----------------------------===//
//
// Provides a general dead-code elimination pass for our embedded MLIR
// environment. Upstream MLIR's DCE passes are not registered in Python here, so
// we provide a standalone pass under a distinct name ('trivial-dce').
//
//===----------------------------------------------------------------------===//

#include "flir/FlirPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::flir;

namespace {

#define GEN_PASS_DEF_FLIRTRIVIALDCEPASS
#include "flir/FlirPasses.h.inc"

struct FlirTrivialDCEPass
    : public impl::FlirTrivialDCEPassBase<FlirTrivialDCEPass> {
  using impl::FlirTrivialDCEPassBase<
      FlirTrivialDCEPass>::FlirTrivialDCEPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Iteratively erase dead ops to a fixed point, since erasing one op may
    // render its operands dead.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;
      module.walk<WalkOrder::PostOrder>([&](Operation *op) {
        if (mlir::isOpTriviallyDead(op))
          toErase.push_back(op);
      });
      for (Operation *op : toErase) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace flir {

std::unique_ptr<Pass> createFlirTrivialDCEPass() {
  return std::make_unique<FlirTrivialDCEPass>();
}

} // namespace flir
} // namespace mlir


