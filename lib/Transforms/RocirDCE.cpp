//===- RocirDCE.cpp - Trivial Dead Code Elimination Pass ------------------===//
//
// Provide a standalone, general DCE pass ("trivial-dce") for the Rocir toolchain.
// This is intentionally lightweight: iteratively erase operations that
// - have no uses,
// - are memory-effect free (no side effects),
// - and are not terminators.
//
// This exists because upstream MLIR's generic dce/adce passes may not be
// registered/available in our embedded Python environment.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace rocir {

namespace {

struct RocirTrivialDCEPass
    : public PassWrapper<RocirTrivialDCEPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocirTrivialDCEPass)

  StringRef getArgument() const final { return "trivial-dce"; }
  StringRef getDescription() const final {
    return "Erase trivially-dead operations (general DCE)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool changed = false;
    do {
      changed = false;
      llvm::SmallVector<Operation *, 128> dead;

      // Post-order ensures we consider inner ops before their parents.
      mod.walk<WalkOrder::PostOrder>([&](Operation *op) {
        // Never erase the top-level op provided by the pass manager.
        if (op == mod.getOperation())
          return;

        // Conservatively skip ops with regions (scf/func/gpu/etc). Those usually
        // encode control flow; even if results are unused, erasing may be
        // surprising. The goal here is to eliminate obviously dead leaf ops.
        if (op->getNumRegions() != 0)
          return;

        // Don't touch terminators.
        if (op->hasTrait<OpTrait::IsTerminator>())
          return;

        // Only erase ops with no uses.
        if (!op->use_empty())
          return;

        // Only erase ops with no side effects.
        if (!isMemoryEffectFree(op))
          return;

        dead.push_back(op);
      });

      // Erase in reverse to keep parent operations valid.
      for (Operation *op : llvm::reverse(dead)) {
        op->erase();
        changed = true;
      }
    } while (changed);
  }
};

} // namespace

std::unique_ptr<Pass> createRocirTrivialDCEPass() {
  return std::make_unique<RocirTrivialDCEPass>();
}

} // namespace rocir
} // namespace mlir


