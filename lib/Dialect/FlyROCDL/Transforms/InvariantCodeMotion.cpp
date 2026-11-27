// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Loop-invariant code motion that also moves AMDGPU lane intrinsics.
//
// The lane intrinsics read other lanes of the wave, so they have no MLIR memory effects and
// generic code motion leaves them in loops. Code generation keeps them there too: they are
// convergent. What the hardware does with them decides where they may move instead, which this
// pass applies on top of the generic rule. The same holds for the LDS address of an LDS DMA, which
// the hardware reads from the first active lane into M0.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace fly_rocdl {
#define GEN_PASS_DEF_ROCDLINVARIANTCODEMOTIONPASS
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"
} // namespace fly_rocdl
} // namespace mlir

namespace {

/// Whether values are the same in every lane, by the rules of the pass description.
class UniformValues {
public:
  bool isUniform(Value value) {
    auto [it, inserted] = known.try_emplace(value, false);
    if (!inserted)
      return it->second;
    bool uniform = compute(value);
    known[value] = uniform;
    return uniform;
  }

  /// Whether `loop` runs the same iterations in every lane.
  bool isUniformLoop(LoopLikeOpInterface loop) {
    auto forOp = dyn_cast<scf::ForOp>(loop.getOperation());
    return forOp && isUniform(forOp.getLowerBound()) && isUniform(forOp.getUpperBound()) &&
           isUniform(forOp.getStep());
  }

private:
  bool compute(Value value) {
    if (matchPattern(value, m_Constant()))
      return true;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Operation *owner = arg.getOwner()->getParentOp();
      if (auto func = dyn_cast<gpu::GPUFuncOp>(owner))
        return func.isKernel() && arg.getOwner()->isEntryBlock();
      if (auto loop = dyn_cast<scf::ForOp>(owner))
        return value == loop.getInductionVar() && isUniformLoop(loop);
      return false;
    }
    Operation *op = value.getDefiningOp();
    if (isa<gpu::BlockIdOp, gpu::GridDimOp, gpu::BlockDimOp, gpu::ClusterIdOp, gpu::ClusterDimOp,
            gpu::ClusterBlockIdOp, gpu::ClusterDimBlocksOp, ROCDL::ReadfirstlaneOp,
            ROCDL::ReadlaneOp, ROCDL::BallotOp>(op))
      return true;
    // Inline assembly may depend on the lane whatever its operands.
    if (!isa<arith::ArithDialect, LLVM::LLVMDialect>(op->getDialect()) ||
        isa<LLVM::InlineAsmOp>(op) || op->getNumOperands() == 0 || op->getNumRegions() != 0 ||
        !isMemoryEffectFree(op))
      return false;
    return llvm::all_of(op->getOperands(), [&](Value operand) { return isUniform(operand); });
  }

  DenseMap<Value, bool> known;
};

/// Where the hardware lets a lane intrinsic move, when its operands do not change.
enum class LaneOpMotion {
  Anywhere,        // its result does not depend on which lanes are active
  WithActiveLanes, // its result depends on which lanes are active
};

/// The AMDGPU intrinsic `op` stands for, as a ROCDL op or an llvm.call_intrinsic.
llvm::Intrinsic::ID getIntrinsicID(Operation *op) {
  if (auto call = dyn_cast<LLVM::CallIntrinsicOp>(op))
    return llvm::Intrinsic::lookupIntrinsicID(call.getIntrin());
  using namespace llvm::Intrinsic;
  return llvm::TypeSwitch<Operation *, ID>(op)
      .Case([](ROCDL::ReadlaneOp) { return amdgcn_readlane; })
      .Case([](ROCDL::ReadfirstlaneOp) { return amdgcn_readfirstlane; })
      .Case([](ROCDL::BallotOp) { return amdgcn_ballot; })
      .Case([](ROCDL::DPPUpdateOp) { return amdgcn_update_dpp; })
      .Case([](ROCDL::PermlaneX16Op) { return amdgcn_permlanex16; })
      .Case([](ROCDL::Permlane16SwapOp) { return amdgcn_permlane16_swap; })
      .Case([](ROCDL::Permlane32SwapOp) { return amdgcn_permlane32_swap; })
      .Case([](ROCDL::Permlane16VarOp) { return amdgcn_permlane16_var; })
      .Case([](ROCDL::PermlaneX16VarOp) { return amdgcn_permlanex16_var; })
      .Case([](ROCDL::DsSwizzleOp) { return amdgcn_ds_swizzle; })
      .Case([](ROCDL::DsBpermuteOp) { return amdgcn_ds_bpermute; })
      .Case([](ROCDL::LoadToLDSOp) { return amdgcn_load_to_lds; })
      .Case([](ROCDL::LoadAsyncToLDSOp) { return amdgcn_load_async_to_lds; })
      .Case([](ROCDL::GlobalLoadLDSOp) { return amdgcn_global_load_lds; })
      .Case([](ROCDL::GlobalLoadAsyncLDSOp) { return amdgcn_global_load_async_lds; })
      .Case([](ROCDL::RawPtrBufferLoadLdsOp) { return amdgcn_raw_ptr_buffer_load_lds; })
      .Case([](ROCDL::RawPtrBufferLoadAsyncLdsOp) { return amdgcn_raw_ptr_buffer_load_async_lds; })
      .Default([](Operation *) { return not_intrinsic; });
}

/// Where `op`, a convergent AMDGPU intrinsic without memory effects, may move by what the hardware
/// does with it; std::nullopt for any other op.
std::optional<LaneOpMotion> getLaneOpMotion(Operation *op, UniformValues &uniform) {
  // These read or write the lane they name, or work on scalars, whichever lanes are active. Their
  // scalar operands are read from the first active lane, though, so they are only independent of
  // the active lanes when those operands are the same in every lane.
  auto scalarOperands = [&](std::initializer_list<unsigned> indices) {
    return llvm::all_of(indices, [&](unsigned i) { return uniform.isUniform(op->getOperand(i)); })
               ? LaneOpMotion::Anywhere
               : LaneOpMotion::WithActiveLanes;
  };
  using namespace llvm::Intrinsic;
  switch (getIntrinsicID(op)) {
  case amdgcn_readlane:
    return scalarOperands({1});
  case amdgcn_writelane:
    return scalarOperands({0, 1});
  case amdgcn_inverse_ballot:
  case amdgcn_s_bitreplicate:
  case amdgcn_s_quadmask:
  case amdgcn_s_wqm:
    return scalarOperands({0});
  // These read the active lanes.
  case amdgcn_readfirstlane:
  case amdgcn_ballot:
  case amdgcn_icmp:
  case amdgcn_fcmp:
  case amdgcn_wqm_vote:
  case amdgcn_mov_dpp:
  case amdgcn_mov_dpp8:
  case amdgcn_update_dpp:
  case amdgcn_permlane16:
  case amdgcn_permlanex16:
  case amdgcn_permlane16_var:
  case amdgcn_permlanex16_var:
  case amdgcn_permlane16_swap:
  case amdgcn_permlane32_swap:
  case amdgcn_permlane64:
  case amdgcn_permlane_bcast:
  case amdgcn_permlane_down:
  case amdgcn_permlane_up:
  case amdgcn_permlane_xor:
  case amdgcn_permlane_idx_gen:
  case amdgcn_wave_shuffle:
  case amdgcn_ds_swizzle:
  case amdgcn_ds_permute:
  case amdgcn_ds_bpermute:
  case amdgcn_ds_bpermute_fi_b32:
  case amdgcn_set_inactive:
  case amdgcn_wwm:
  case amdgcn_strict_wwm:
  case amdgcn_strict_wqm:
  case amdgcn_wave_reduce_add:
  case amdgcn_wave_reduce_sub:
  case amdgcn_wave_reduce_min:
  case amdgcn_wave_reduce_max:
  case amdgcn_wave_reduce_umin:
  case amdgcn_wave_reduce_umax:
  case amdgcn_wave_reduce_and:
  case amdgcn_wave_reduce_or:
  case amdgcn_wave_reduce_xor:
  case amdgcn_wave_reduce_fadd:
  case amdgcn_wave_reduce_fsub:
  case amdgcn_wave_reduce_fmin:
  case amdgcn_wave_reduce_fmax:
    return LaneOpMotion::WithActiveLanes;
  default:
    return std::nullopt;
  }
}

/// The operand of `op` that the hardware reads from the first active lane into M0: the LDS
/// address of an LDS DMA. std::nullopt for any other op.
std::optional<unsigned> getM0Operand(Operation *op) {
  using namespace llvm::Intrinsic;
  switch (getIntrinsicID(op)) {
  case amdgcn_load_to_lds:
  case amdgcn_load_async_to_lds:
  case amdgcn_global_load_lds:
  case amdgcn_global_load_async_lds:
  case amdgcn_raw_buffer_load_lds:
  case amdgcn_raw_ptr_buffer_load_lds:
  case amdgcn_struct_buffer_load_lds:
  case amdgcn_struct_ptr_buffer_load_lds:
  case amdgcn_raw_buffer_load_async_lds:
  case amdgcn_raw_ptr_buffer_load_async_lds:
  case amdgcn_struct_buffer_load_async_lds:
  case amdgcn_struct_ptr_buffer_load_async_lds:
    return 1;
  default:
    return std::nullopt;
  }
}

/// Reads before `loop`, a uniform loop, the loop-invariant LDS addresses that the LDS DMAs at the
/// top level of its body read into M0, so that each is read once rather than on every iteration.
void readM0BeforeLoop(LoopLikeOpInterface loop) {
  OpBuilder builder(loop);
  DenseMap<Value, Value> reads;
  for (Region *region : loop.getLoopRegions()) {
    for (Operation &op : region->getOps()) {
      std::optional<unsigned> index = getM0Operand(&op);
      if (!index)
        continue;
      Value address = op.getOperand(*index);
      if (!loop.isDefinedOutsideOfLoop(address) || address.getDefiningOp<ROCDL::ReadfirstlaneOp>())
        continue;
      Value &read = reads[address];
      if (!read)
        read = ROCDL::ReadfirstlaneOp::create(builder, op.getLoc(), address.getType(), address);
      op.setOperand(*index, read);
    }
  }
}

class ROCDLInvariantCodeMotionPass
    : public mlir::fly_rocdl::impl::ROCDLInvariantCodeMotionPassBase<ROCDLInvariantCodeMotionPass> {
public:
  using mlir::fly_rocdl::impl::ROCDLInvariantCodeMotionPassBase<
      ROCDLInvariantCodeMotionPass>::ROCDLInvariantCodeMotionPassBase;

  void runOnOperation() override {
    UniformValues uniform;
    // Inner loops first, so what moves out of one can move on out of the next.
    getOperation()->walk([&](LoopLikeOpInterface loop) {
      bool uniformLoop = uniform.isUniformLoop(loop);
      // moveLoopInvariantCode only considers the top level of the body, where the lanes of a
      // uniform loop are those active before it.
      moveLoopInvariantCode(
          loop.getLoopRegions(),
          [&](Value value, Region *) { return loop.isDefinedOutsideOfLoop(value); },
          [&](Operation *op, Region *) {
            if (isPure(op))
              return true;
            std::optional<LaneOpMotion> motion = getLaneOpMotion(op, uniform);
            return motion && (*motion == LaneOpMotion::Anywhere || uniformLoop);
          },
          [&](Operation *op, Region *) { loop.moveOutOfLoop(op); });
      if (uniformLoop)
        readM0BeforeLoop(loop);
    });
  }
};

} // namespace
