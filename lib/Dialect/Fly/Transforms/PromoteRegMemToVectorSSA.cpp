// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"

#include <numeric>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::fly;

namespace llvm {

template <> struct DenseMapInfo<mlir::fly::MakePtrOp> {
  static unsigned getHashValue(mlir::fly::MakePtrOp op) {
    return DenseMapInfo<void *>::getHashValue(op.getAsOpaquePointer());
  }

  static bool isEqual(mlir::fly::MakePtrOp lhs, mlir::fly::MakePtrOp rhs) { return lhs == rhs; }
};

} // namespace llvm

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYPROMOTEREGMEMTOVECTORSSAPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

using VectorValue = TypedValue<VectorType>;
using RegMem2VectorSSAMap = DenseMap<MakePtrOp, VectorValue>;

/// The lanes an access covers in its allocation's vector SSA value.
struct RegAccessInfo {
  MakePtrOp makePtrOp;
  int32_t offset;
  int32_t width;
};

struct RegAllocaInfo {
  MakePtrOp makePtrOp;
  int32_t allocSize;
  Type elemTy;
  /// The width of one vector SSA lane: the element width, narrowed to cover
  /// every access in whole lanes.
  int64_t laneBits;
  VectorType vectorSSATy;
};

bool isRegValue(Value value) {
  if (auto ptrTy = dyn_cast<PointerType>(value.getType()))
    return isGenericAddressSpace<AddressSpace::Register>(ptrTy.getAddressSpace());
  if (auto memRefTy = dyn_cast<fly::MemRefType>(value.getType()))
    return isGenericAddressSpace<AddressSpace::Register>(memRefTy.getAddressSpace());
  return false;
}

bool isRegOperandOrResult(Operation *op) {
  for (Value result : op->getOpResults()) {
    if (isRegValue(result))
      return true;
  }
  for (Value operand : op->getOperands()) {
    if (isRegValue(operand))
      return true;
  }
  return false;
}

/// The bit width of the elements of a pointer, scalar or vector type.
int64_t getElemBits(Type type) {
  if (auto ptrTy = dyn_cast<PointerType>(type))
    type = ptrTy.getElemTy();
  return getElementTypeOrSelf(type).getIntOrFloatBitWidth();
}

/// Whether promotion can hold `valueType` in vector lanes: an integer or float
/// scalar, or a 1-D vector of them.
bool isLaneAccessType(Type valueType) {
  if (auto vecTy = dyn_cast<VectorType>(valueType)) {
    if (vecTy.getRank() != 1 || vecTy.isScalable())
      return false;
    valueType = vecTy.getElementType();
  }
  return isa<IntegerType, FloatType>(valueType);
}

int64_t getTotalBits(Type valueType) {
  auto vecTy = dyn_cast<VectorType>(valueType);
  return getElemBits(valueType) * (vecTy ? vecTy.getNumElements() : 1);
}

/// A register make_ptr with a static allocation size, which promotion may turn
/// into a vector SSA value.
bool isAllocaCandidate(MakePtrOp makePtrOp) {
  if (!isRegValue(makePtrOp))
    return false;
  auto allocSizeAttr = makePtrOp.getDictAttrs()->getAs<IntegerAttr>("allocSize");
  return allocSizeAttr && allocSizeAttr.getInt() > 0;
}

/// What a register pointer or view addresses: the candidate allocations it may
/// point into, and its offset there in bits, so that typed views of the same
/// storage (recast_iter) share one address space. It is resolved when it names
/// one allocation at one static offset on every path that reaches it.
class RegAddress {
public:
  /// The state of a value the analysis has not reached.
  RegAddress() = default;

  static RegAddress of(MakePtrOp root) {
    RegAddress address;
    address.initialized = true;
    address.roots.push_back(root);
    address.offset = 0;
    return address;
  }

  /// An address that is not derived from a candidate allocation.
  static RegAddress untracked() {
    RegAddress address;
    address.initialized = true;
    address.hasUntracked = true;
    return address;
  }

  bool isUninitialized() const { return !initialized; }
  ArrayRef<Operation *> getRoots() const { return roots; }

  std::optional<std::pair<MakePtrOp, int64_t>> getResolved() const {
    if (!initialized || hasUntracked || roots.size() != 1 || !offset)
      return std::nullopt;
    return std::pair<MakePtrOp, int64_t>{cast<MakePtrOp>(roots.front()), *offset};
  }

  /// This address advanced by `delta` bits; a dynamic delta loses the offset.
  RegAddress advanced(std::optional<int64_t> delta) const {
    RegAddress address = *this;
    if (address.offset && delta)
      *address.offset += *delta;
    else
      address.offset.reset();
    return address;
  }

  bool operator==(const RegAddress &rhs) const {
    return initialized == rhs.initialized && hasUntracked == rhs.hasUntracked &&
           roots == rhs.roots && offset == rhs.offset;
  }

  static RegAddress join(const RegAddress &lhs, const RegAddress &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    RegAddress address;
    address.initialized = true;
    address.hasUntracked = lhs.hasUntracked || rhs.hasUntracked;
    std::set_union(lhs.roots.begin(), lhs.roots.end(), rhs.roots.begin(), rhs.roots.end(),
                   std::back_inserter(address.roots));
    if (lhs.offset == rhs.offset)
      address.offset = lhs.offset;
    return address;
  }

  void print(raw_ostream &os) const {
    if (!initialized) {
      os << "<uninitialized>";
      return;
    }
    os << "roots=" << roots.size() << (hasUntracked ? " untracked" : "") << " offset=";
    if (offset)
      os << *offset;
    else
      os << "?";
  }

private:
  bool initialized = false;
  bool hasUntracked = false;
  /// Sorted, so that equal sets compare equal.
  SmallVector<Operation *, 1> roots;
  std::optional<int64_t> offset;
};

/// Leaves every value an unknown constant, so that dead-code analysis takes
/// every branch and region as reachable and every op is analyzed.
class UnknownConstants
    : public dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<dataflow::ConstantValue>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnknownConstants)
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *, ArrayRef<const dataflow::Lattice<dataflow::ConstantValue> *>,
                 ArrayRef<dataflow::Lattice<dataflow::ConstantValue> *> results) override {
    setAllToEntryStates(results);
    return success();
  }

  void setToEntryState(dataflow::Lattice<dataflow::ConstantValue> *lattice) override {
    propagateIfChanged(lattice, lattice->join(dataflow::ConstantValue::getUnknownConstant()));
  }
};

class RegAddressLattice : public dataflow::Lattice<RegAddress> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegAddressLattice)
  using Lattice::Lattice;
};

/// Tracks register addresses through make_view, get_iter, recast_iter,
/// add_offset, select and the results and arguments of SCF regions, so that an
/// access resolves to its allocation however its address was formed.
class RegAddressAnalysis : public dataflow::SparseForwardDataFlowAnalysis<RegAddressLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegAddressAnalysis)
  using dataflow::SparseForwardDataFlowAnalysis<RegAddressLattice>::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, ArrayRef<const RegAddressLattice *> operands,
                               ArrayRef<RegAddressLattice *> results) override {
    for (auto [result, lattice] : llvm::zip(op->getResults(), results)) {
      if (isRegValue(result))
        propagateIfChanged(lattice, lattice->join(transfer(op, operands)));
    }
    return success();
  }

  void setToEntryState(RegAddressLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(RegAddress::untracked()));
  }

private:
  /// The address a view, offset or recast derives from its first operand; an
  /// operand outside register memory derives no tracked address.
  static RegAddress sourceAddress(Operation *op, ArrayRef<const RegAddressLattice *> operands) {
    return isRegValue(op->getOperand(0)) ? operands[0]->getValue() : RegAddress::untracked();
  }

  static RegAddress transfer(Operation *op, ArrayRef<const RegAddressLattice *> operands) {
    return llvm::TypeSwitch<Operation *, RegAddress>(op)
        .Case([](MakePtrOp makePtrOp) {
          return isAllocaCandidate(makePtrOp) ? RegAddress::of(makePtrOp) : RegAddress::untracked();
        })
        .Case([&](AddOffsetOp addOffsetOp) {
          LeafAttr leaf = addOffsetOp.getOffset().getType().getAttr().getLeaf();
          std::optional<int64_t> delta;
          if (leaf.isStatic())
            delta = int64_t(leaf.getValue()) * getElemBits(addOffsetOp.getPtr().getType());
          return sourceAddress(op, operands).advanced(delta);
        })
        .Case<MakeViewOp, GetIterOp, RecastIterOp>(
            [&](Operation *) { return sourceAddress(op, operands); })
        .Case([&](arith::SelectOp) {
          return RegAddress::join(operands[1]->getValue(), operands[2]->getValue());
        })
        .Default([](Operation *) { return RegAddress::untracked(); });
  }
};

/// Whether `use` passes a register address on to another address (a view, a
/// recast, an offset, a select, or an SCF region value) without reading
/// through it.
bool isForwardingUse(OpOperand &use) {
  return llvm::TypeSwitch<Operation *, bool>(use.getOwner())
      .Case<MakeViewOp, GetIterOp, AddOffsetOp, arith::SelectOp, scf::ForOp, scf::WhileOp,
            scf::ConditionOp>([](Operation *) { return true; })
      .Case([](RecastIterOp recastOp) { return isRegValue(recastOp.getResult()); })
      .Case([](scf::YieldOp yieldOp) {
        return isa<scf::IfOp, scf::ForOp, scf::WhileOp>(yieldOp->getParentOp());
      })
      .Default([](Operation *) { return false; });
}

/// The value type `op` loads or stores, if it is a scalar or vector access that
/// promotion rewrites into vector SSA.
Type getAccessType(Operation *op) {
  Type valueType;
  if (auto loadOp = dyn_cast<PtrLoadOp>(op))
    valueType = loadOp.getResult().getType();
  else if (auto storeOp = dyn_cast<PtrStoreOp>(op))
    valueType = storeOp.getValue().getType();
  return valueType && isLaneAccessType(valueType) ? valueType : Type();
}

/// The pointer an access from `getAccessType` goes through.
Value getAccessPtr(Operation *op) {
  if (auto loadOp = dyn_cast<PtrLoadOp>(op))
    return loadOp.getPtr();
  return cast<PtrStoreOp>(op).getPtr();
}

int64_t getAllocBits(Operation *root) {
  auto makePtrOp = cast<MakePtrOp>(root);
  return makePtrOp.getDictAttrs()->getAs<IntegerAttr>("allocSize").getInt() *
         getElemBits(makePtrOp.getType());
}

class FlyPromoteRegMemToVectorSSAPass
    : public mlir::fly::impl::FlyPromoteRegMemToVectorSSAPassBase<FlyPromoteRegMemToVectorSSAPass> {
public:
  using mlir::fly::impl::FlyPromoteRegMemToVectorSSAPassBase<
      FlyPromoteRegMemToVectorSSAPass>::FlyPromoteRegMemToVectorSSAPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp->walk([&](gpu::GPUFuncOp funcOp) {
      if (failed(processFunction(funcOp)))
        signalPassFailure();
    });

    auto context = moduleOp->getContext();
    IRRewriter rewriter(context);
    DominanceInfo domInfo(getOperation());
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());
  }

private:
  DenseMap<MakePtrOp, RegAllocaInfo> regAllocaInfos;
  SmallVector<MakePtrOp> allocaOrder;
  const DataFlowSolver *solver = nullptr;

  LogicalResult processFunction(gpu::GPUFuncOp funcOp) {
    OpBuilder opBuilder(funcOp.getContext());

    regAllocaInfos.clear();
    allocaOrder.clear();

    funcOp.walk([&](MakePtrOp makePtrOp) {
      if (!isAllocaCandidate(makePtrOp))
        return;
      auto allocSizeAttr = makePtrOp.getDictAttrs()->getAs<IntegerAttr>("allocSize");
      PointerType ptrTy = cast<PointerType>(makePtrOp.getType());
      Type originElemTy = ptrTy.getElemTy();
      Type ssaElemTy = projectToLLVMCompatibleElemTy(ptrTy.getElemTy());
      regAllocaInfos.try_emplace(
          makePtrOp, RegAllocaInfo{makePtrOp, static_cast<int32_t>(allocSizeAttr.getInt()),
                                   originElemTy, getElemBits(originElemTy),
                                   VectorType::get({allocSizeAttr.getInt()}, ssaElemTy)});
      allocaOrder.push_back(makePtrOp);
    });

    if (allocaOrder.empty())
      return success();

    DataFlowSolver functionSolver;
    functionSolver.load<dataflow::DeadCodeAnalysis>();
    functionSolver.load<UnknownConstants>();
    functionSolver.load<RegAddressAnalysis>();
    if (failed(functionSolver.initializeAndRun(funcOp)))
      return failure();
    solver = &functionSolver;

    // Register memory is lexically scoped: it is accessed only inside the
    // region that allocates it.
    bool outOfScope = false;
    funcOp.walk([&](Operation *op) {
      if (isOpTriviallyDead(op))
        return;
      for (OpOperand &use : op->getOpOperands()) {
        const RegAddress *address = getAddress(use.get());
        if (!address || isForwardingUse(use))
          continue;
        for (Operation *root : address->getRoots()) {
          if (!root->getParentRegion()->isAncestor(op->getParentRegion())) {
            op->emitOpError("register memory accessed outside the region that allocates it");
            outOfScope = true;
            break;
          }
        }
      }
    });
    if (outOfScope)
      return failure();

    // An allocation stays in memory, which LLVM still compiles, when one of its
    // addresses is not resolved to it at a static offset (a dynamic offset, or
    // different allocations or offsets on different paths), or reaches an op
    // that neither accesses nor forwards it.
    auto exclude = [&](Value value) {
      for (Operation *root : getAddress(value)->getRoots())
        regAllocaInfos.erase(cast<MakePtrOp>(root));
    };
    auto checkValue = [&](Value value) {
      if (getAddress(value) && !getAddress(value)->getResolved())
        exclude(value);
    };
    funcOp.walk([&](Operation *op) {
      for (Value result : op->getResults())
        checkValue(result);
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments())
            checkValue(arg);
        }
      }
      if (isOpTriviallyDead(op))
        return;
      for (OpOperand &use : op->getOpOperands()) {
        if (getAddress(use.get()) && !isForwardingUse(use) && !getAccessType(op))
          exclude(use.get());
      }
    });

    llvm::erase_if(allocaOrder, [&](MakePtrOp r) { return !regAllocaInfos.count(r); });

    if (allocaOrder.empty())
      return success();

    // Promoted storage is packed: bit i of an allocation is bit i of its vector
    // SSA value, whatever the views it is accessed through. Each access covers
    // whole lanes, so the lanes narrow to the gcd of the element width and the
    // bit offset and width of every access.
    funcOp.walk([&](Operation *op) {
      Type accessType = getAccessType(op);
      auto slot = accessType ? getPromotedSlot(getAccessPtr(op)) : std::nullopt;
      if (!slot)
        return;
      RegAllocaInfo *info = getAllocaInfo(slot->first);
      info->laneBits = std::gcd(info->laneBits, std::gcd(slot->second, getTotalBits(accessType)));
    });

    // A static access outside a promoted allocation cannot index its vector.
    bool outOfBounds = false;
    funcOp.walk([&](Operation *op) {
      Type accessType = getAccessType(op);
      auto slot = accessType ? getPromotedSlot(getAccessPtr(op)) : std::nullopt;
      if (!slot)
        return;
      int64_t end = slot->second + getTotalBits(accessType);
      int64_t size = getAllocBits(slot->first);
      if (slot->second < 0 || end > size) {
        op->emitOpError("out-of-bounds register access: bits [")
            << slot->second << ", " << end << ") of a " << size << "-bit allocation";
        outOfBounds = true;
      }
    });
    if (outOfBounds)
      return failure();

    for (MakePtrOp root : allocaOrder) {
      RegAllocaInfo &info = regAllocaInfos.find(root)->second;
      int64_t elemBits = getElemBits(info.elemTy);
      if (info.laneBits != elemBits)
        info.vectorSSATy = VectorType::get({info.allocSize * elemBits / info.laneBits},
                                           IntegerType::get(root.getContext(), info.laneBits));
    }

    Block *oldEntry = &funcOp.getBody().front();
    Block *newEntry = opBuilder.createBlock(&funcOp.getBody());
    for (BlockArgument arg : oldEntry->getArguments())
      newEntry->addArgument(arg.getType(), arg.getLoc());

    IRMapping mapping;
    for (auto it : llvm::zip(oldEntry->getArguments(), newEntry->getArguments()))
      mapping.map(std::get<0>(it), std::get<1>(it));

    RegMem2VectorSSAMap state;
    if (failed(rewriteBlock(oldEntry, newEntry, mapping, state))) {
      newEntry->erase();
      return failure();
    }

    oldEntry->erase();
    cleanupDeadOps(funcOp);

    // Addresses of promoted allocations are rewritten from a poison
    // placeholder; one still in use was not rewritten away.
    WalkResult remaining = funcOp.walk([&](ub::PoisonOp poisonOp) {
      if (!isRegValue(poisonOp))
        return WalkResult::advance();
      Operation *user = *poisonOp->getUsers().begin();
      user->emitOpError("register operand/result remain after rmem SSA promotion");
      return WalkResult::interrupt();
    });
    return failure(remaining.wasInterrupted());
  }

  RegAllocaInfo *getAllocaInfo(MakePtrOp makePtrOp) {
    auto it = regAllocaInfos.find(makePtrOp);
    if (it == regAllocaInfos.end())
      return nullptr;
    return &it->second;
  }

  /// The address the analysis computed for `value`; null if `value` is not a
  /// register address.
  const RegAddress *getAddress(Value value) const {
    if (!isRegValue(value))
      return nullptr;
    const auto *lattice = solver->lookupState<RegAddressLattice>(value);
    if (!lattice || lattice->getValue().isUninitialized())
      return nullptr;
    return &lattice->getValue();
  }

  /// The promoted allocation and static bit offset `value` addresses.
  std::optional<std::pair<MakePtrOp, int64_t>> getPromotedSlot(Value value) const {
    if (!isRegValue(value) || !getAddress(value))
      return std::nullopt;
    auto resolved = getAddress(value)->getResolved();
    if (!resolved || !regAllocaInfos.count(resolved->first))
      return std::nullopt;
    return resolved;
  }

  bool isPromoted(Value value) const { return getPromotedSlot(value).has_value(); }

  /// The lanes of its promoted allocation that an access of `valueType`
  /// through `ptr` covers.
  std::optional<RegAccessInfo> getRegAccessInfo(Value ptr, Type valueType) {
    auto slot = getPromotedSlot(ptr);
    if (!slot || !isLaneAccessType(valueType))
      return std::nullopt;
    const RegAllocaInfo *info = getAllocaInfo(slot->first);
    return RegAccessInfo{slot->first, static_cast<int32_t>(slot->second / info->laneBits),
                         static_cast<int32_t>(getTotalBits(valueType) / info->laneBits)};
  }

  LogicalResult collectTouchedRegAllocaInRegion(Region &region, Operation *boundaryOp,
                                                DenseSet<MakePtrOp> &touchedRoots) {
    auto tryRecord = [&](Value ptr) {
      auto slot = getPromotedSlot(ptr);
      if (slot && !boundaryOp->isProperAncestor(slot->first))
        touchedRoots.insert(slot->first);
    };
    for (Block &block : region) {
      for (Operation &op : block) {
        if (auto loadOp = dyn_cast<PtrLoadOp>(&op))
          tryRecord(loadOp.getPtr());
        else if (auto storeOp = dyn_cast<PtrStoreOp>(&op))
          tryRecord(storeOp.getPtr());

        for (Region &nested : op.getRegions()) {
          if (failed(collectTouchedRegAllocaInRegion(nested, boundaryOp, touchedRoots)))
            return failure();
        }
      }
    }
    return success();
  }

  LogicalResult collectTouchedRegAlloca(Operation *boundaryOp, SmallVectorImpl<MakePtrOp> &roots) {
    DenseSet<MakePtrOp> touchedRoots;
    for (Region &region : boundaryOp->getRegions()) {
      if (failed(collectTouchedRegAllocaInRegion(region, boundaryOp, touchedRoots)))
        return failure();
    }
    for (MakePtrOp makePtrOp : allocaOrder) {
      if (touchedRoots.contains(makePtrOp))
        roots.push_back(makePtrOp);
    }
    return success();
  }

  void appendVectorSSATypes(ArrayRef<MakePtrOp> roots, SmallVectorImpl<Type> &types) {
    for (MakePtrOp makePtrOp : roots) {
      const RegAllocaInfo *info = getAllocaInfo(makePtrOp);
      assert(info && "missing register makePtrOp info");
      types.push_back(info->vectorSSATy);
    }
  }

  void appendVectorSSAValues(RegMem2VectorSSAMap &state, ArrayRef<MakePtrOp> roots,
                             SmallVectorImpl<Value> &values) {
    for (MakePtrOp makePtrOp : roots) {
      auto it = state.find(makePtrOp);
      assert(it != state.end() && "missing state for register makePtrOp");
      values.push_back(it->second);
    }
  }

  Value bitcastScalarViaVector(OpBuilder &builder, Location loc, Value value, Type targetTy) {
    auto srcVecTy = VectorType::get({1}, value.getType());
    auto dstVecTy = VectorType::get({1}, targetTy);
    Value srcVec = vector::FromElementsOp::create(builder, loc, srcVecTy, value);
    Value dstVec = vector::BitCastOp::create(builder, loc, dstVecTy, srcVec);
    return vector::ExtractOp::create(builder, loc, dstVec, ArrayRef<int64_t>{0});
  }

  /// `value` as `width` lanes of `laneTy`: a scalar when a scalar fills one
  /// lane, a vector otherwise.
  Value bitcastToLanes(OpBuilder &builder, Location loc, Value value, Type laneTy, int64_t width) {
    Type valueTy = value.getType();
    if (!isa<VectorType>(valueTy)) {
      if (width == 1)
        return valueTy == laneTy ? value : bitcastScalarViaVector(builder, loc, value, laneTy);
      value = vector::FromElementsOp::create(builder, loc, VectorType::get({1}, valueTy), value);
    }
    auto lanesTy = VectorType::get({width}, laneTy);
    if (value.getType() == lanesTy)
      return value;
    return vector::BitCastOp::create(builder, loc, lanesTy, value);
  }

  /// The lanes `bitcastToLanes` produced, as a value of `resultTy`.
  Value bitcastFromLanes(OpBuilder &builder, Location loc, Value lanes, Type resultTy) {
    if (!isa<VectorType>(lanes.getType()))
      return lanes.getType() == resultTy ? lanes
                                         : bitcastScalarViaVector(builder, loc, lanes, resultTy);
    if (auto resultVecTy = dyn_cast<VectorType>(resultTy)) {
      if (lanes.getType() == resultVecTy)
        return lanes;
      return vector::BitCastOp::create(builder, loc, resultVecTy, lanes);
    }
    Value vec = vector::BitCastOp::create(builder, loc, VectorType::get({1}, resultTy), lanes);
    return vector::ExtractOp::create(builder, loc, vec, ArrayRef<int64_t>{0});
  }

  LogicalResult rewritePtrStore(PtrStoreOp storeOp, OpBuilder &builder, IRMapping &mapping,
                                RegMem2VectorSSAMap &state) {
    auto access = getRegAccessInfo(storeOp.getPtr(), storeOp.getValue().getType());
    if (!access)
      return failure();

    auto stateIt = state.find(access->makePtrOp);
    assert(stateIt != state.end() && "missing state for register ptr.store");
    VectorValue currentVec = stateIt->second;
    VectorValue updatedVec;

    auto loc = storeOp.getLoc();
    Value storedValue = mapping.lookupOrDefault(storeOp.getValue());

    const RegAllocaInfo *info = getAllocaInfo(access->makePtrOp);
    assert(info && "missing alloca info for register ptr.store");
    storedValue = bitcastToLanes(builder, loc, storedValue, info->vectorSSATy.getElementType(),
                                 access->width);

    if (isa<IntegerType, FloatType>(storedValue.getType())) {
      assert(access->width == 1 && "expected scalar type with width 1");
      updatedVec = vector::InsertOp::create(builder, loc, storedValue, currentVec, access->offset);
    } else {
      assert(isa<VectorType>(storedValue.getType()) && "expected vector type");
      assert(cast<VectorType>(storedValue.getType()).getNumElements() == access->width &&
             "expected vector type with same width as access");

      updatedVec = vector::InsertStridedSliceOp::create(builder, loc, storedValue, currentVec,
                                                        ArrayRef<int64_t>{access->offset},
                                                        ArrayRef<int64_t>{1});
    }
    state[access->makePtrOp] = updatedVec;
    return success();
  }

  LogicalResult rewritePtrLoad(PtrLoadOp loadOp, OpBuilder &builder, IRMapping &mapping,
                               RegMem2VectorSSAMap &state) {
    auto access = getRegAccessInfo(loadOp.getPtr(), loadOp.getResult().getType());
    if (!access)
      return failure();

    auto stateIt = state.find(access->makePtrOp);
    assert(stateIt != state.end() && "missing state for register ptr.load");
    VectorValue currentVec = stateIt->second;

    auto loc = loadOp.getLoc();
    Type resultType = loadOp.getResult().getType();
    Value extracted;

    if (!isa<VectorType>(resultType) && access->width == 1) {
      extracted = vector::ExtractOp::create(builder, loc, currentVec, access->offset);
    } else {
      extracted = vector::ExtractStridedSliceOp::create(
          builder, loc, currentVec, ArrayRef<int64_t>{access->offset},
          ArrayRef<int64_t>{access->width}, ArrayRef<int64_t>{1});
    }
    extracted = bitcastFromLanes(builder, loc, extracted, resultType);
    mapping.map(loadOp, extracted);
    return success();
  }

  /// The positions of values a region op carries (results, region arguments
  /// and the terminator operands that feed them) that address a promoted
  /// allocation. They are dropped: accesses through them resolve via the
  /// analysis, so nothing reads the carried address.
  SmallVector<bool> promotedPositions(std::initializer_list<ValueRange> groups) const {
    SmallVector<bool> dropped(groups.begin()->size(), false);
    for (ValueRange group : groups) {
      for (auto [index, value] : llvm::enumerate(group))
        dropped[index] = dropped[index] || isPromoted(value);
    }
    return dropped;
  }

  static SmallVector<Value> keptValues(ValueRange values, ArrayRef<bool> dropped,
                                       const IRMapping &mapping) {
    SmallVector<Value> kept;
    for (auto [value, drop] : llvm::zip(values, dropped)) {
      if (!drop)
        kept.push_back(mapping.lookupOrDefault(value));
    }
    return kept;
  }

  /// Maps `oldValues` to `newValues` at the kept positions, and the dropped
  /// ones to poison placeholders created before `newOp`, which promotion
  /// leaves unused.
  static void mapCarried(ValueRange oldValues, ValueRange newValues, ArrayRef<bool> dropped,
                         Operation *newOp, IRMapping &mapping) {
    OpBuilder placeholderBuilder(newOp);
    unsigned next = 0;
    for (auto [oldValue, drop] : llvm::zip(oldValues, dropped)) {
      if (drop)
        mapping.map(oldValue,
                    ub::PoisonOp::create(placeholderBuilder, oldValue.getLoc(), oldValue.getType())
                        .getResult());
      else
        mapping.map(oldValue, newValues[next++]);
    }
  }

  LogicalResult rewriteIfOp(scf::IfOp oldIf, OpBuilder &builder, IRMapping &mapping,
                            RegMem2VectorSSAMap &state) {
    SmallVector<MakePtrOp> touchedRoots;
    if (failed(collectTouchedRegAlloca(oldIf, touchedRoots)))
      return failure();

    bool hasElse = !oldIf.getElseRegion().empty();
    SmallVector<bool> dropped =
        promotedPositions({oldIf.getResults(), oldIf.thenYield().getOperands(),
                           hasElse ? oldIf.elseYield().getOperands() : ValueRange()});
    SmallVector<Type> newResultTypes;
    for (auto [type, drop] : llvm::zip(oldIf.getResultTypes(), dropped)) {
      if (!drop)
        newResultTypes.push_back(type);
    }
    unsigned numKept = newResultTypes.size();
    appendVectorSSATypes(touchedRoots, newResultTypes);

    bool withElse = hasElse || !touchedRoots.empty();
    auto newIf = scf::IfOp::create(builder, oldIf.getLoc(), TypeRange(newResultTypes),
                                   mapping.lookupOrDefault(oldIf.getCondition()), withElse);
    newIf->setDiscardableAttrs(oldIf->getDiscardableAttrDictionary());

    {
      // process then block
      IRMapping thenMapping = mapping;
      RegMem2VectorSSAMap thenState = state;
      Block *oldThen = &oldIf.getThenRegion().front();
      Block *newThen = &newIf.getThenRegion().front();
      if (!newThen->empty())
        newThen->back().erase();
      if (failed(rewriteBlock(oldThen, newThen, thenMapping, thenState)))
        return failure();

      auto oldYield = cast<scf::YieldOp>(oldThen->getTerminator());
      SmallVector<Value> newYieldOperands =
          keptValues(oldYield.getOperands(), dropped, thenMapping);
      appendVectorSSAValues(thenState, touchedRoots, newYieldOperands);

      OpBuilder thenYieldBuilder = OpBuilder::atBlockEnd(newThen);
      scf::YieldOp::create(thenYieldBuilder, oldYield.getLoc(), newYieldOperands);
    }

    if (hasElse) {
      IRMapping elseMapping = mapping;
      RegMem2VectorSSAMap elseState = state;
      Block *oldElse = &oldIf.getElseRegion().front();
      Block *newElse = &newIf.getElseRegion().front();
      if (!newElse->empty())
        newElse->back().erase();
      if (failed(rewriteBlock(oldElse, newElse, elseMapping, elseState)))
        return failure();

      auto oldYield = cast<scf::YieldOp>(oldElse->getTerminator());
      SmallVector<Value> newYieldOperands =
          keptValues(oldYield.getOperands(), dropped, elseMapping);
      appendVectorSSAValues(elseState, touchedRoots, newYieldOperands);

      OpBuilder elseYieldBuilder = OpBuilder::atBlockEnd(newElse);
      scf::YieldOp::create(elseYieldBuilder, oldYield.getLoc(), newYieldOperands);
    } else if (withElse) {
      Block *newElse = &newIf.getElseRegion().front();
      if (!newElse->empty())
        newElse->back().erase();
      SmallVector<Value> elseYieldOperands;
      appendVectorSSAValues(state, touchedRoots, elseYieldOperands);
      OpBuilder elseYieldBuilder = OpBuilder::atBlockEnd(newElse);
      scf::YieldOp::create(elseYieldBuilder, oldIf.getLoc(), elseYieldOperands);
    }

    mapCarried(oldIf.getResults(), newIf.getResults(), dropped, newIf, mapping);
    for (auto it : llvm::enumerate(touchedRoots))
      state[it.value()] = cast<VectorValue>(newIf.getResult(numKept + it.index()));

    return success();
  }

  LogicalResult rewriteForOp(scf::ForOp oldFor, OpBuilder &builder, IRMapping &mapping,
                             RegMem2VectorSSAMap &state) {
    SmallVector<MakePtrOp> touchedRoots;
    if (failed(collectTouchedRegAlloca(oldFor, touchedRoots)))
      return failure();

    auto oldYield = cast<scf::YieldOp>(oldFor.getBody()->getTerminator());
    SmallVector<bool> dropped = promotedPositions({oldFor.getInitArgs(), oldFor.getRegionIterArgs(),
                                                   oldYield.getOperands(), oldFor.getResults()});

    SmallVector<Value> newInitArgs = keptValues(oldFor.getInitArgs(), dropped, mapping);
    unsigned numKept = newInitArgs.size();
    appendVectorSSAValues(state, touchedRoots, newInitArgs);

    auto newFor = scf::ForOp::create(
        builder, oldFor.getLoc(), mapping.lookupOrDefault(oldFor.getLowerBound()),
        mapping.lookupOrDefault(oldFor.getUpperBound()), mapping.lookupOrDefault(oldFor.getStep()),
        newInitArgs, [](OpBuilder &, Location, Value, ValueRange) {}, oldFor.getUnsignedCmp());
    newFor->setDiscardableAttrs(oldFor->getDiscardableAttrDictionary());

    IRMapping bodyMapping = mapping;
    Block *oldBody = oldFor.getBody();
    Block *newBody = newFor.getBody();
    newBody->getArgument(0).setLoc(oldFor.getInductionVar().getLoc());
    bodyMapping.map(oldFor.getInductionVar(), newFor.getInductionVar());
    mapCarried(oldFor.getRegionIterArgs(), newFor.getRegionIterArgs(), dropped, newFor,
               bodyMapping);
    for (auto [oldArg, drop] : llvm::zip(oldFor.getRegionIterArgs(), dropped)) {
      if (!drop)
        cast<BlockArgument>(bodyMapping.lookup(oldArg)).setLoc(oldArg.getLoc());
    }
    for (unsigned i = 1 + numKept; i < newBody->getNumArguments(); ++i)
      newBody->getArgument(i).setLoc(oldFor.getLoc());

    RegMem2VectorSSAMap bodyState = state;
    unsigned carriedArgBase = 1 + numKept;
    for (auto it : llvm::enumerate(touchedRoots))
      bodyState[it.value()] = cast<VectorValue>(newBody->getArgument(carriedArgBase + it.index()));

    if (failed(rewriteBlock(oldBody, newBody, bodyMapping, bodyState)))
      return failure();

    SmallVector<Value> newYieldOperands = keptValues(oldYield.getOperands(), dropped, bodyMapping);
    for (MakePtrOp makePtrOp : touchedRoots)
      newYieldOperands.push_back(bodyState[makePtrOp]);

    OpBuilder yieldBuilder = OpBuilder::atBlockEnd(newBody);
    scf::YieldOp::create(yieldBuilder, oldYield.getLoc(), newYieldOperands);

    mapCarried(oldFor.getResults(), newFor.getResults(), dropped, newFor, mapping);
    for (auto it : llvm::enumerate(touchedRoots))
      state[it.value()] = cast<VectorValue>(newFor.getResult(numKept + it.index()));

    return success();
  }

  LogicalResult rewriteWhileOp(scf::WhileOp oldWhile, OpBuilder &builder, IRMapping &mapping,
                               RegMem2VectorSSAMap &state) {
    SmallVector<MakePtrOp> touchedRoots;
    if (failed(collectTouchedRegAlloca(oldWhile, touchedRoots)))
      return failure();

    Block *oldBefore = oldWhile.getBeforeBody();
    Block *oldAfter = oldWhile.getAfterBody();
    auto oldCondition = cast<scf::ConditionOp>(oldBefore->getTerminator());
    auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
    // The inits feed the before block and the after block's yield loops back
    // to it; the condition forwards to the after block and the results.
    SmallVector<bool> droppedBefore = promotedPositions(
        {oldWhile.getInits(), oldWhile.getBeforeArguments(), oldYield.getOperands()});
    SmallVector<bool> droppedAfter = promotedPositions(
        {oldCondition.getArgs(), oldWhile.getAfterArguments(), oldWhile.getResults()});

    SmallVector<Value> newInitArgs = keptValues(oldWhile.getInits(), droppedBefore, mapping);
    unsigned numKeptBefore = newInitArgs.size();
    appendVectorSSAValues(state, touchedRoots, newInitArgs);

    SmallVector<Type> newResultTypes;
    SmallVector<Location> afterArgLocs;
    for (auto [arg, drop] : llvm::zip(oldWhile.getAfterArguments(), droppedAfter)) {
      if (!drop) {
        newResultTypes.push_back(arg.getType());
        afterArgLocs.push_back(arg.getLoc());
      }
    }
    unsigned numKeptAfter = newResultTypes.size();
    appendVectorSSATypes(touchedRoots, newResultTypes);

    auto newWhile =
        scf::WhileOp::create(builder, oldWhile.getLoc(), TypeRange(newResultTypes), newInitArgs);
    newWhile->setDiscardableAttrs(oldWhile->getDiscardableAttrDictionary());

    SmallVector<Type> beforeArgTypes;
    beforeArgTypes.reserve(newInitArgs.size());
    for (Value initArg : newInitArgs)
      beforeArgTypes.push_back(initArg.getType());
    SmallVector<Location> beforeArgLocs;
    for (auto [arg, drop] : llvm::zip(oldWhile.getBeforeArguments(), droppedBefore)) {
      if (!drop)
        beforeArgLocs.push_back(arg.getLoc());
    }
    beforeArgLocs.resize(beforeArgTypes.size(), oldWhile.getLoc());
    Block *newBefore =
        builder.createBlock(&newWhile.getBefore(), {}, beforeArgTypes, beforeArgLocs);

    afterArgLocs.resize(newResultTypes.size(), oldWhile.getLoc());
    Block *newAfter = builder.createBlock(&newWhile.getAfter(), {}, newResultTypes, afterArgLocs);

    {
      // process before block
      IRMapping beforeMapping = mapping;
      mapCarried(oldWhile.getBeforeArguments(), newBefore->getArguments(), droppedBefore, newWhile,
                 beforeMapping);

      RegMem2VectorSSAMap beforeState = state;
      for (auto it : llvm::enumerate(touchedRoots))
        beforeState[it.value()] =
            cast<VectorValue>(newBefore->getArgument(numKeptBefore + it.index()));

      if (failed(rewriteBlock(oldBefore, newBefore, beforeMapping, beforeState)))
        return failure();

      SmallVector<Value> newConditionArgs =
          keptValues(oldCondition.getArgs(), droppedAfter, beforeMapping);
      for (MakePtrOp makePtrOp : touchedRoots)
        newConditionArgs.push_back(beforeState[makePtrOp]);

      OpBuilder condBuilder = OpBuilder::atBlockEnd(newBefore);
      scf::ConditionOp::create(condBuilder, oldCondition.getLoc(),
                               beforeMapping.lookupOrDefault(oldCondition.getCondition()),
                               newConditionArgs);
    }

    {
      // process after block
      IRMapping afterMapping = mapping;
      mapCarried(oldWhile.getAfterArguments(), newAfter->getArguments(), droppedAfter, newWhile,
                 afterMapping);

      RegMem2VectorSSAMap afterState = state;
      for (auto it : llvm::enumerate(touchedRoots))
        afterState[it.value()] =
            cast<VectorValue>(newAfter->getArgument(numKeptAfter + it.index()));

      if (failed(rewriteBlock(oldAfter, newAfter, afterMapping, afterState)))
        return failure();

      SmallVector<Value> newYieldOperands =
          keptValues(oldYield.getOperands(), droppedBefore, afterMapping);
      for (MakePtrOp makePtrOp : touchedRoots)
        newYieldOperands.push_back(afterState[makePtrOp]);

      OpBuilder yieldBuilder = OpBuilder::atBlockEnd(newAfter);
      scf::YieldOp::create(yieldBuilder, oldYield.getLoc(), newYieldOperands);
    }

    mapCarried(oldWhile.getResults(), newWhile.getResults(), droppedAfter, newWhile, mapping);
    for (auto it : llvm::enumerate(touchedRoots))
      state[it.value()] = cast<VectorValue>(newWhile.getResult(numKeptAfter + it.index()));

    return success();
  }

  LogicalResult rewriteBlock(Block *oldBlock, Block *newBlock, IRMapping &mapping,
                             RegMem2VectorSSAMap &state) {
    OpBuilder builder(oldBlock->getParentOp()->getContext());

    for (Operation &op : *oldBlock) {
      builder.setInsertionPointToEnd(newBlock);

      if (isa<scf::YieldOp, scf::ConditionOp>(op))
        continue;

      if (auto makePtrOp = dyn_cast<MakePtrOp>(&op); makePtrOp && isRegValue(makePtrOp)) {
        if (auto *info = getAllocaInfo(makePtrOp)) {
          mapping.map(
              makePtrOp.getResult(),
              ub::PoisonOp::create(builder, makePtrOp.getLoc(), makePtrOp.getType()).getResult());
          state[makePtrOp] = cast<VectorValue>(
              ub::PoisonOp::create(builder, makePtrOp.getLoc(), info->vectorSSATy).getResult());
        } else {
          builder.clone(op, mapping);
        }
        continue;
      }

      if (auto storeOp = dyn_cast<PtrStoreOp>(&op); storeOp && isRegValue(storeOp.getPtr())) {
        if (getRegAccessInfo(storeOp.getPtr(), storeOp.getValue().getType())) {
          if (failed(rewritePtrStore(storeOp, builder, mapping, state)))
            return failure();
        } else {
          builder.clone(op, mapping);
        }
        continue;
      }

      if (auto loadOp = dyn_cast<PtrLoadOp>(&op); loadOp && isRegValue(loadOp.getPtr())) {
        if (getRegAccessInfo(loadOp.getPtr(), loadOp.getResult().getType())) {
          if (failed(rewritePtrLoad(loadOp, builder, mapping, state)))
            return failure();
        } else {
          builder.clone(op, mapping);
        }
        continue;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        if (failed(rewriteForOp(forOp, builder, mapping, state)))
          return failure();
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (failed(rewriteIfOp(ifOp, builder, mapping, state)))
          return failure();
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
        if (failed(rewriteWhileOp(whileOp, builder, mapping, state)))
          return failure();
        continue;
      }

      if (op.getNumRegions() != 0) {
        bool hasNestedRegOperandOrResult = false;
        op.walk([&](Operation *nestedOp) {
          if (nestedOp == &op)
            return;
          if (isRegOperandOrResult(nestedOp))
            hasNestedRegOperandOrResult = true;
        });
        if (hasNestedRegOperandOrResult)
          return op.emitOpError("unsupported region op with register values during rmem SSA");
      }

      builder.clone(op, mapping);
    }

    return success();
  }

  void cleanupDeadOps(gpu::GPUFuncOp funcOp) {
    bool changed = true;
    while (changed) {
      changed = false;
      funcOp.walk([&](Block *block) {
        for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
          if (isOpTriviallyDead(&op)) {
            op.erase();
            changed = true;
          }
        }
      });
    }
  }
};

} // namespace
