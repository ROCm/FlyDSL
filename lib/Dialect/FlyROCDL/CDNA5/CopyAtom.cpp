// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <limits>

#include "llvm/Support/MathExtras.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/TdmGeometry.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

//===----------------------------------------------------------------------===//
// CopyOpCDNA5TensorLoadType / CopyOpCDNA5TensorStoreType — N-D TDM whole-tile DMA
// (rank 1-5), one type per hardware instruction (TENSOR_LOAD_TO_LDS /
// TENSOR_STORE_FROM_LDS), addressed by a tile coordinate.
//
// Everything about the *global tensor* is fixed for the atom's life and arrives
// as construction arguments of `fly.make_copy_atom`: the base pointer, the
// per-dim stride, and every dim's extent, all measured from the tensor origin.
// Everything about the *tile* is a coordinate, and it arrives as the
// `!fly.coord_tensor` operand of the copy: the kernel tiles and slices that
// tensor with ordinary layout algebra, which folds the tile's position into its
// type, and the operand's runtime value is whatever of that position was
// genuinely dynamic.
//
// Why a coordinate and not an address. TDM's `global_addr` is the address of the
// tile, not of the tensor, so the hardware measures `tensor_dim` from the tile start
// as well: moving the tile moves the base *and* shrinks the in-bounds window. Those
// two must agree — a base that has advanced past a clamp that has not silently reads
// outside the tensor — so both are derived here from the one coordinate, and there is
// no way to set either directly.
//
// Which dims clamp is atom state (`boundary_check`, an int_tuple with a leaf per *global tensor
// mode*, all-clamp by default), not a type parameter, because the cost is per dim *and*
// per call: a clamping dim spends a subtract and a max on its extent, while a dim left
// alone passes that extent straight through as `tensor_dim` and spends nothing. A
// tiled loop is usually ragged in only some dims, and often only on some iterations —
// the K-loop that needs its K bound checked on the last trip and nowhere else is the
// shape this exists for. The tuple is set as a unit, which is what makes its rank
// checkable; a static leaf folds the select and kills the losing side's arithmetic, so
// the state slots only cost anything when a flag is genuinely dynamic.
//
// The flags are indexed by the tensor's modes and not the descriptor's, because the
// descriptor's are not the caller's to know: it takes its order from the LDS tile's
// majorness, and packs the tail of a tensor with more than five modes. `tensor2tdm` is the
// translation from one to the other, and it is a type parameter because a call site far from the
// builder must still be able to spell `set_value("boundary_check", ...)` in the tensor's terms. It
// is shaped like the tensor rather than flattened, so the `boundary_check` tuple can be
// checked against it for profile and not merely for leaf count.
//===----------------------------------------------------------------------===//

namespace {

constexpr unsigned kMaxTdmRank = 5;
constexpr uint64_t kMaxTensorStride = (uint64_t{1} << 48) - 1;
constexpr int32_t kMaxAtomicBarrierAddress = 0x7FFF8;
constexpr int32_t kMaxIterateCount = 256;

// Filler for the extent slots past the atom's rank, which the lowering never reads --
// a value that would be obviously wrong in a descriptor, so a stray read shows up.
constexpr int32_t kUnusedExtent = -1;

// LDS. `atomic_barrier_addr` is a pointer into it rather than an integer, so the barrier
// the transfer arrives on is named the way the kernel already holds it -- what an
// allocator hands back -- instead of being flattened to an address at the call site.
// It says only *which* barrier: whether there is one at all is the atom's type, so no
// pointer value is reserved to mean "none" and offset 0 is a barrier like any other.
constexpr unsigned kSharedAddrSpace = ROCDL::ROCDLDialect::kSharedMemoryAddressSpace;

// TDM padding descriptor bitfield:
//   encoded_interval = log2(interval_dw) - 1  -> bits [24:22] (3 bits)
//   encoded_amount   = amount_dw - 1          -> bits [31:25] (7 bits)
//   where *_dw = pad_*_elems * elem_bits / 32
struct PadEncoding {
  int32_t interval = 0;
  int32_t amount = 0;
  bool enable = false;
};

// Encoding for active padding, a disabled encoding when none is requested, or
// failure() when it cannot be represented (not dword-aligned, dword interval not a
// power of two, or a field out of range) — failing here avoids silently emitting a
// wrong descriptor.
FailureOr<PadEncoding> computePadEncoding(int32_t padIntervalElems, int32_t padAmountElems,
                                          int32_t elemBits) {
  PadEncoding e;
  if (padIntervalElems <= 0 || padAmountElems <= 0)
    return e; // disabled
  if ((padIntervalElems * elemBits) % 32 != 0 || (padAmountElems * elemBits) % 32 != 0)
    return failure();
  int32_t intervalDw = padIntervalElems * elemBits / 32;
  int32_t amountDw = padAmountElems * elemBits / 32;
  if (intervalDw <= 0 || amountDw <= 0 || (intervalDw & (intervalDw - 1)) != 0)
    return failure();
  int32_t encInterval = llvm::Log2_32(static_cast<uint32_t>(intervalDw)) - 1;
  int32_t encAmount = amountDw - 1;
  if (encInterval < 0 || encInterval > 0x7 || encAmount < 0 || encAmount > 0x7F)
    return failure();
  e.interval = encInterval;
  e.amount = encAmount;
  e.enable = true;
  return e;
}

Value i32Const(OpBuilder &b, Location loc, int32_t v) {
  return arith::ConstantIntOp::create(b, loc, v, 32);
}

//===----------------------------------------------------------------------===//
// Atom-state struct layout
//===----------------------------------------------------------------------===//

constexpr unsigned kNoSlot = ~0u;

// The state-struct layout is a per-type property, so the load's MCAST fields simply
// do not exist on the store type instead of sitting there as slots the hardware
// ignores. The geometry slots (extent, boundary_check) are present on both.
struct TdmSlots {
  unsigned workgroupMask;     // i32, load only (kNoSlot on store)
  unsigned earlyTimeout;      // i32, load only (kNoSlot on store)
  unsigned atomicBarrierAddr; // !llvm.ptr<3>, which barrier; *whether* is the type
  unsigned basePtr;           // !llvm.ptr<1>, the tensor origin
  unsigned stride0;           // stride_i at stride0 + i, i < kMaxTdmRank - 1 (i64)
  unsigned extent0;           // extent_i at extent0 + i, i < kMaxTdmRank (i32)
  unsigned boundaryCheck0;    // boundary_check_i at boundaryCheck0 + i, i < kMaxTdmRank (i1)
  unsigned iterStride;        // i64, global step between descriptor replays
  unsigned numSlots;
  // Whether `atomicBarrierAddr` is a field a call may write. The *index* stays valid
  // either way, so the struct layout is the same for both and does not depend on a type
  // parameter; only `tdmFieldIndex` hides it, which is what makes `set_value` fail.
  bool hasAtomicBarrier = true;
};

// {mask, early_timeout, atomic_barrier_addr, base, stride_0..3, extent_0..4, boundary_check_0..4,
// iter}
constexpr TdmSlots kLoadSlots = {0, 1, 2, 3, 4, 8, 13, 18, 19};
// {atomic_barrier_addr, base, stride_0..3, extent_0..4, boundary_check_0..4, iter}
constexpr TdmSlots kStoreSlots = {kNoSlot, kNoSlot, 0, 1, 2, 6, 11, 16, 17};

// The `atomic_barrier_addr` field belongs to an atom whose *type* enables the HW
// auto-barrier and to no other: on one that does not, `atom.set_value` finds no slot
// and fails to legalize instead of writing an address nothing arrives on. The slot
// index itself stays reserved so a type parameter does not move the fields around it;
// a slot no call writes dies with the scalarized struct, like an unread extent.
TdmSlots tdmSlots(const TdmSlots &base, bool hasAtomicBarrier) {
  TdmSlots slots = base;
  slots.hasAtomicBarrier = hasAtomicBarrier;
  return slots;
}

// A field the store type does not carry maps to no slot, so `atom.set_value` on it
// fails to legalize instead of writing a descriptor bit the engine ignores.
std::optional<unsigned> optSlot(unsigned slot) {
  if (slot == kNoSlot)
    return std::nullopt;
  return slot;
}

std::optional<unsigned> tdmFieldIndex(const TdmSlots &slots, AtomStateField field) {
  switch (field) {
  case AtomStateField::WorkgroupMask:
    return optSlot(slots.workgroupMask);
  case AtomStateField::EarlyTimeout:
    return optSlot(slots.earlyTimeout);
  case AtomStateField::AtomicBarrierAddr:
    return slots.hasAtomicBarrier ? optSlot(slots.atomicBarrierAddr) : std::nullopt;
  default:
    return std::nullopt;
  }
}

Type tdmConvertedType(MLIRContext *ctx, const TdmSlots &slots) {
  auto i32 = IntegerType::get(ctx, 32);
  auto i64 = IntegerType::get(ctx, 64);
  SmallVector<Type> fields(slots.numSlots, i32);
  fields[slots.basePtr] = LLVM::LLVMPointerType::get(ctx, /*global*/ 1);
  // Unconditional, so the two atoms have one layout: an atom whose type carries no
  // barrier still has the slot, it is simply never written and dies with the scalarized
  // struct. Making it conditional would make `getConvertedType` disagree with the state
  // the builder hands back.
  fields[slots.atomicBarrierAddr] = LLVM::LLVMPointerType::get(ctx, kSharedAddrSpace);
  for (unsigned i = 0; i + 1 < kMaxTdmRank; ++i)
    fields[slots.stride0 + i] = i64;
  for (unsigned i = 0; i < kMaxTdmRank; ++i)
    fields[slots.boundaryCheck0 + i] = IntegerType::get(ctx, 1);
  fields[slots.iterStride] = i64;
  return LLVM::LLVMStructType::getLiteral(ctx, fields);
}

// Construction arguments of `fly.make_copy_atom`, in order:
//   base pointer, stride_0..stride_{rank-2} (i64), extent_0..extent_{rank-1} (i32) —
// all in tensor dim order, all measured from the tensor origin. The innermost stride
// is 1 by construction and is not passed. Every dim passes its extent whether or not
// it currently clamps: `boundary_check` is per-call state, so a later call site may switch its
// dim's clamping on and needs the bound already in the atom. An extent no call reads
// dies with the rest of the scalarized state struct, so this costs nothing.
Value tdmInitialState(OpBuilder &builder, Location loc, const TdmSlots &slots, unsigned rank,
                      IntTupleAttr tensor2tdm, bool iterates, ValueRange args) {
  unsigned expected = 1 + (rank - 1) + rank + (iterates ? 1 : 0);
  if (args.size() != expected) {
    mlir::emitError(loc) << "cdna5 TDM: expected " << expected << " construction arguments (base + "
                         << (rank - 1) << " strides + " << rank << " extents"
                         << (iterates ? " + 1 iteration stride" : "") << "), got " << args.size();
    return nullptr;
  }
  if (iterates && !args.back().getType().isInteger(64)) {
    mlir::emitError(loc) << "cdna5 TDM: the iteration stride must be i64, got "
                         << args.back().getType();
    return nullptr;
  }
  if (!isa<LLVM::LLVMPointerType>(args[0].getType())) {
    mlir::emitError(loc) << "cdna5 TDM: the base must be a pointer, got " << args[0].getType();
    return nullptr;
  }
  for (unsigned i = 0; i + 1 < rank; ++i)
    if (!args[1 + i].getType().isInteger(64)) {
      mlir::emitError(loc) << "cdna5 TDM: stride_" << i << " must be i64, got "
                           << args[1 + i].getType();
      return nullptr;
    }
  for (unsigned i = 0; i < rank; ++i)
    if (!args[rank + i].getType().isInteger(32)) {
      mlir::emitError(loc) << "cdna5 TDM: extent_" << i << " must be i32, got "
                           << args[rank + i].getType();
      return nullptr;
    }

  auto structTy = cast<LLVM::LLVMStructType>(tdmConvertedType(builder.getContext(), slots));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  auto set = [&](unsigned slot, Value v) {
    state = LLVM::InsertValueOp::create(builder, loc, state, v, ArrayRef<int64_t>{slot});
  };
  if (slots.workgroupMask != kNoSlot)
    set(slots.workgroupMask, i32Const(builder, loc, 0));
  if (slots.earlyTimeout != kNoSlot)
    set(slots.earlyTimeout, i32Const(builder, loc, 0));
  set(slots.atomicBarrierAddr,
      LLVM::ZeroOp::create(builder, loc,
                           LLVM::LLVMPointerType::get(builder.getContext(), kSharedAddrSpace)));
  set(slots.basePtr, args[0]);
  // Slots past the atom's rank are never read by the lowering; they only exist so
  // the struct layout is rank-independent.
  for (unsigned i = 0; i + 1 < kMaxTdmRank; ++i)
    set(slots.stride0 + i,
        i + 1 < rank ? args[1 + i] : arith::ConstantIntOp::create(builder, loc, 0, 64));
  for (unsigned i = 0; i < kMaxTdmRank; ++i)
    set(slots.extent0 + i, i < rank ? args[rank + i] : i32Const(builder, loc, kUnusedExtent));
  // A dim starts clamping when some tensor mode can put a bound on it. The builder
  // normally overwrites this immediately with the caller's `boundary_check`; this is what an atom
  // built without one does, and "clamp what can be clamped" is the safe default.
  SmallVector<int32_t> axes;
  tdm::boundaryCheckAxes(tensor2tdm, axes);
  for (unsigned i = 0; i < kMaxTdmRank; ++i) {
    bool bounded = llvm::is_contained(axes, static_cast<int32_t>(i));
    set(slots.boundaryCheck0 + i, arith::ConstantIntOp::create(builder, loc, bounded, 1));
  }
  set(slots.iterStride, iterates ? args.back() : arith::ConstantIntOp::create(builder, loc, 0, 64));
  return state;
}

// Per-dim clamping is TDM-private, so it does not go through the shared AtomStateField
// enum: it is set as a whole, as one int_tuple — a nonzero leaf clamps that descriptor
// dim to the extent baked in at construction, zero lets the tile run its full size there.
// Setting it as one aggregate rather than a leaf at a time is what makes its rank
// checkable.
//
// A caller writes `"boundary_check"` in the *tensor's* modes, because the descriptor's are not
// theirs to know. `fly-rocdl-expand-ops` translates that through `tensor2tdm` into
// the `"boundary_check_axes"` this reads: one leaf per descriptor axis, already OR-merged where
// several modes shared one. So nothing about modes survives to here, and a `"boundary_check"` that
// reaches this point is a pipeline that skipped the pass rather than something to
// translate a second time.
//
// A static leaf lives in the tuple's *type*, so it reaches the descriptor math as a
// constant and the select it feeds folds away; a dynamic leaf is an i32/i64 operand of
// `make_int_tuple` and is compared against zero to reach the i1 the slot holds.
LogicalResult tdmUnpackBoundaryCheckAxes(OpBuilder &builder, Location loc, Value flags,
                                         unsigned rank, SmallVectorImpl<Value> &out) {
  auto tupleTy = dyn_cast<IntTupleType>(flags.getType());
  if (!tupleTy)
    return mlir::emitError(loc) << "cdna5 TDM: \"boundary_check_axes\" must be an int_tuple, got "
                                << flags.getType();
  IntTupleAttr attr = tupleTy.getAttr();
  if (attr.isLeaf() || attr.rank() != static_cast<int32_t>(rank))
    return mlir::emitError(loc) << "cdna5 TDM: \"boundary_check_axes\" is " << attr
                                << ", expected the flat " << rank
                                << "-leaf tuple `fly-rocdl-expand-ops` produces, one leaf per "
                                   "descriptor axis";

  auto tupleOp = flags.getDefiningOp<MakeIntTupleOp>();
  OperandRange dyn = tupleOp ? tupleOp.getDyncElems() : OperandRange(nullptr, 0);
  auto dynIt = dyn.begin();
  out.clear();
  for (int32_t axis = 0; axis < attr.rank(); ++axis) {
    IntTupleAttr leaf = attr.at(axis);
    if (!leaf.isLeaf())
      return mlir::emitError(loc) << "cdna5 TDM: \"boundary_check_axes\" leaf " << axis
                                  << " is nested";
    IntAttr value = leaf.extractIntFromLeaf();
    if (value.isStatic()) {
      out.push_back(arith::ConstantIntOp::create(builder, loc, value.getValue() != 0, 1));
      continue;
    }
    if (!tupleOp || dynIt == dyn.end())
      return mlir::emitError(loc) << "cdna5 TDM: \"boundary_check_axes\" leaf " << axis
                                  << " is dynamic but the tuple is not normal form";
    Value v = *dynIt++;
    Value zero = arith::ConstantIntOp::create(builder, loc, v.getType(), 0);
    out.push_back(arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, v, zero));
  }
  return success();
}

Value tdmSetAtomState(OpBuilder &builder, Location loc, const TdmSlots &slots, unsigned rank,
                      Value atomStruct, Attribute fieldAttr, Value fieldValue) {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto insert = [&](unsigned slot, Value v) {
    atomStruct = LLVM::InsertValueOp::create(builder, loc, atomStruct, v, ArrayRef<int64_t>{slot});
  };

  if (fieldStr.getValue() == "boundary_check") {
    mlir::emitError(loc)
        << "cdna5 TDM: \"boundary_check\" is written in the global tensor's modes and has "
           "to be translated by `fly-rocdl-expand-ops` into \"boundary_check_axes\" before "
           "it can be lowered";
    return nullptr;
  }
  if (fieldStr.getValue() == "boundary_check_axes") {
    SmallVector<Value> boundaryCheck;
    if (failed(tdmUnpackBoundaryCheckAxes(builder, loc, fieldValue, rank, boundaryCheck)))
      return nullptr;
    for (auto [i, v] : llvm::enumerate(boundaryCheck))
      insert(slots.boundaryCheck0 + i, v);
    return atomStruct;
  }

  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  std::optional<unsigned> idx = tdmFieldIndex(slots, *field);
  if (!idx)
    return nullptr;
  if (*field == AtomStateField::AtomicBarrierAddr) {
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(fieldValue.getType());
    if (!ptrTy || ptrTy.getAddressSpace() != kSharedAddrSpace) {
      mlir::emitError(loc) << "cdna5 TDM: \"atomic_barrier_addr\" is the barrier itself, so it "
                              "takes a shared-memory pointer, got "
                           << fieldValue.getType();
      return nullptr;
    }
  }
  insert(*idx, fieldValue);
  return atomStruct;
}

// Only the width of the descriptor's data type reaches the hardware (`data_size`);
// the type itself is there so the atom says which element it moves. The verifier has
// already rejected anything without a width.
int32_t tdmElemBits(Type dataType) {
  return static_cast<int32_t>(dataType.getIntOrFloatBitWidth());
}

// One TDM call is issued by a single thread and moves the whole tile, so the atom is
// (1 thread) x (product(tileShape) * elemBits bits).
int32_t tdmNumBits(ArrayRef<int32_t> tileShape, int32_t elemBits, int32_t iterCount) {
  int64_t numElems = iterCount;
  for (int32_t d : tileShape)
    numElems *= d;
  return static_cast<int32_t>(numElems * elemBits);
}

// Spelled out instead of using the FxLayout/FxShape macros: those expand to a bare
// `getContext()` call and so only work inside a type member function.
Attribute tdmThrBitLayout(MLIRContext *ctx, ArrayRef<int32_t> tileShape, int32_t elemBits,
                          int32_t iterCount) {
  Attribute one = IntTupleAttr::getLeafStatic(ctx, 1);
  Attribute bits = IntTupleAttr::getLeafStatic(ctx, tdmNumBits(tileShape, elemBits, iterCount));
  return LayoutAttr::get(IntTupleAttr::get(ArrayAttr::get(ctx, {one, bits})),
                         IntTupleAttr::get(ArrayAttr::get(ctx, {one, one})));
}

// Shared verifier. `padInterval` / `padAmount` are always zero on the store type,
// which has no padding parameters at all.
LogicalResult tdmVerify(function_ref<InFlightDiagnostic()> emitError, ArrayRef<int32_t> tileShape,
                        Type dataType, IntTupleAttr tensor2tdm, int32_t iterCount,
                        int32_t padInterval, int32_t padAmount) {
  int32_t rank = static_cast<int32_t>(tileShape.size());
  if (rank < 1 || rank > static_cast<int32_t>(kMaxTdmRank))
    return emitError() << "TDM rank must be in [1, " << kMaxTdmRank << "], got " << rank;
  IntTupleBuilder<IntTupleAttr> tupleBuilder(tensor2tdm.getContext());
  SmallVector<IntTupleAttr> mapLeaves;
  intTupleFlattenToVector(tupleBuilder, tensor2tdm, mapLeaves);
  // A tensor always has at least as many modes as the descriptor has dims: dims are
  // built from modes, and several modes can share one, never the other way round.
  if (static_cast<int32_t>(mapLeaves.size()) < rank)
    return emitError() << "TDM tensor2tdm has " << mapLeaves.size()
                       << " modes but the descriptor has " << rank << " dims";
  // Only a static map is representable: a mode either lands on a known axis at a known
  // scale, or it has no bound to give. A mode with a dynamic stride never shares a dim
  // with another, so it either owns its axis at scale 1 or reaches here as a `0` leaf.
  for (auto [mode, leaf] : llvm::enumerate(mapLeaves)) {
    if (leaf.isLeafBasis()) {
      BasisAttr basis = leaf.getLeafAsBasis();
      if (basis.getModes().size() != 1)
        return emitError() << "TDM tensor2tdm mode " << mode
                           << " must name one descriptor dim, got " << leaf;
      int32_t dim = basis.getModes().front();
      if (dim < 0 || dim >= rank)
        return emitError() << "TDM tensor2tdm mode " << mode << " lands on dim " << dim
                           << ", which is not a descriptor dim in [0, " << rank << ")";
      if (!basis.getValue().isStatic())
        return emitError() << "TDM tensor2tdm mode " << mode << " must have a static scale, got "
                           << leaf;
    } else if (!leaf.isLeafStaticValue(0)) {
      return emitError() << "TDM tensor2tdm mode " << mode << " must be a basis stride or 0, got "
                         << leaf;
    }
  }
  for (int32_t d : tileShape)
    if (d < 1)
      return emitError() << "TDM tile shape dims must be >= 1, got " << d;
  // A width is all the descriptor takes from the data type, so a type without one
  // (a memref, a tuple) has nothing to give it -- and `getIntOrFloatBitWidth` asserts
  // rather than answering, so this guard comes first.
  if (!dataType.isIntOrFloat())
    return emitError() << "TDM dataType must be an integer or float type, got " << dataType;
  int32_t elemBits = tdmElemBits(dataType);
  // data_size is exactly two bits: 0/1/2/3 encode 1/2/4/8 bytes.
  if (elemBits != 8 && elemBits != 16 && elemBits != 32 && elemBits != 64)
    return emitError() << "TDM element width must be one of 8, 16, 32 or 64 bits, got " << dataType;
  if ((padInterval == 0) != (padAmount == 0))
    return emitError() << "padInterval and padAmount must both be zero or both non-zero";
  if (padInterval != 0) {
    if (padInterval < 0 || padAmount < 0)
      return emitError() << "padInterval and padAmount must be non-negative, got " << padInterval
                         << ", " << padAmount;
    // interval_in_dwords is a power of two iff padInterval (in elements) is, since
    // element bits are a power of two; the exact dword/bitfield check needs the
    // element type and runs at lowering.
    if ((padInterval & (padInterval - 1)) != 0)
      return emitError() << "padInterval must be a power of two (in elements), got " << padInterval;
    // The hardware pad counter is free-running over the whole transfer -- it is seeded
    // once, before the dim loops, and only reset where the pad fires -- so the interval
    // is not tied to tile_dim0 and may span several rows. What it cannot straddle is a
    // *call* boundary: each instruction re-seeds the counter, so an interval that does
    // not divide one call's element count would put the second call's holes in the
    // wrong place. (Descriptor iteration re-seeds it too, which is why padding and
    // iterCount > 1 are refused together below.)
    int64_t tileElems = 1;
    for (int32_t dim : tileShape)
      tileElems *= dim;
    if (tileElems % padInterval != 0)
      return emitError() << "padInterval must divide the descriptor's tile (" << tileElems
                         << " elements), so every call starts on a pad boundary, got "
                         << padInterval;
    if ((static_cast<int64_t>(tileShape.back()) * elemBits) % 32 != 0)
      return emitError() << "padded TDM tile_dim0 must span a whole number of dwords, got "
                         << tileShape.back() << " elements at " << elemBits << " bits";
  }
  // `iterate_count` is a 16-bit field encoded as value-minus-one, and iteration is paid
  // for out of GROUP2's own slots -- see `tdm::kMaxIterateRank` for which ones.
  if (iterCount < 1 || iterCount > kMaxIterateCount)
    return emitError() << "TDM iterCount must be in [1, " << kMaxIterateCount << "], got "
                       << iterCount;
  if (iterCount > 1 && rank > tdm::kMaxIterateRank)
    return emitError() << "TDM descriptor iteration takes dim 2's stride for its own, so it "
                          "needs a descriptor of at most "
                       << tdm::kMaxIterateRank << " dims, got " << rank;
  if (iterCount > 1 && padInterval != 0)
    return emitError() << "TDM descriptor iteration and LDS padding are not modelled together";
  int64_t totalBits = elemBits;
  for (int32_t dim : tileShape) {
    if (totalBits > std::numeric_limits<int32_t>::max() / dim)
      return emitError() << "TDM tile bit count exceeds the layout integer range";
    totalBits *= dim;
  }
  return success();
}

// Rebuild a tile coordinate from a lowered coordinate tensor.
//
// A coordinate tensor lowers to its coordinate the way a memref lowers to its
// pointer (see `MakeViewOpLowering`), and an `IntTuple` keeps its static leaves in
// its type and only its dynamic ones as SSA operands. So the coordinate is read from
// both halves: constants for the leaves the layout algebra already folded, and the
// `make_int_tuple` operands for the ones a kernel computed.
LogicalResult unpackCoord(OpBuilder &builder, Location loc, IntTupleAttr baseAttr, Value coordValue,
                          SmallVectorImpl<Value> &out) {
  IntTupleBuilder<IntTupleAttr> tupleBuilder(builder.getContext());
  SmallVector<IntTupleAttr> leaves;
  intTupleFlattenToVector(tupleBuilder, baseAttr, leaves);

  // Only the dynamic leaves are carried by the value, and only a normal-form tuple
  // exposes them in order.
  auto tupleOp = coordValue.getDefiningOp<MakeIntTupleOp>();
  OperandRange dyn = tupleOp ? tupleOp.getDyncElems() : OperandRange(nullptr, 0);
  auto dynIt = dyn.begin();
  auto dynEnd = dyn.end();
  Type i32Ty = builder.getI32Type();
  for (auto [dim, leaf] : llvm::enumerate(leaves)) {
    auto intAttr = leaf.extractIntFromLeaf();
    if (intAttr.isStatic()) {
      out.push_back(arith::ConstantIntOp::create(builder, loc, i32Ty, intAttr.getValue()));
      continue;
    }
    if (!tupleOp || dynIt == dynEnd)
      return mlir::emitError(loc) << "cdna5 TDM: coordinate leaf " << dim
                                  << " is dynamic but the coordinate tensor is not in normal form";
    Value v = *dynIt++;
    if (!v.getType().isInteger(32))
      return mlir::emitError(loc) << "cdna5 TDM: coordinate leaf " << dim
                                  << " must be i32 (a tile index in elements), got " << v.getType();
    out.push_back(v);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Descriptor construction + instruction emission (shared by both directions)
//===----------------------------------------------------------------------===//

// The compile-time half of the descriptor, read off the atom type.
struct TdmStatic {
  ArrayRef<int32_t> tileShape;
  int32_t elemBits;
  int32_t padInterval; // always 0 on the store path
  int32_t padAmount;   // always 0 on the store path
  int32_t cacheModifier;
  int32_t iterCount; // 1 = no descriptor iteration
  bool isLoad;
};

// Read off the atom type, once per direction: the two `emitAtomCall` overloads of a type
// want the same config, and the direction is the only thing that varies between them.
TdmStatic tdmConfig(CopyOpCDNA5TensorLoadType ty) {
  return {ty.getTileShape(), tdmElemBits(ty.getDataType()), ty.getPadInterval(),
          ty.getPadAmount(), ty.getCacheModifier(),         ty.getIterCount(),
          /*isLoad=*/true};
}

TdmStatic tdmConfig(CopyOpCDNA5TensorStoreType ty) {
  return {ty.getTileShape(), tdmElemBits(ty.getDataType()), /*padInterval=*/0,
          /*padAmount=*/0,   ty.getCacheModifier(),         ty.getIterCount(),
          /*isLoad=*/false};
}

// Emits one TENSOR_LOAD_TO_LDS / TENSOR_STORE_FROM_LDS. When `pred` is non-null the
// call is wrapped in an `scf.if`.
LogicalResult emitTdmAtomCall(OpBuilder &builder, Location loc, const TdmStatic &cfg,
                              const TdmSlots &slots, Type srcTyArg, Type dstTyArg, Value atomVal,
                              Value src, Value dst, Type predMemTyArg = nullptr,
                              Value pred = nullptr) {
  // The direction is a property of the atom type (one type per opcode). The global
  // side is the coordinate tensor — it has no pointer, the atom holds the base — and
  // the LDS side is a shared memref.
  Type glbTy = cfg.isLoad ? srcTyArg : dstTyArg;
  Type ldsTyArg = cfg.isLoad ? dstTyArg : srcTyArg;
  Value ldsPtr = cfg.isLoad ? dst : src;
  auto ldsMemTy = dyn_cast<fly::MemRefType>(ldsTyArg);
  if (!isa<fly::CoordTensorType>(glbTy) || !ldsMemTy ||
      !isGenericAddressSpace<fly::AddressSpace::Shared>(ldsMemTy.getAddressSpace()))
    return mlir::emitError(loc) << "cdna5 TDM: " << (cfg.isLoad ? "tensor_load" : "tensor_store")
                                << " needs a "
                                << (cfg.isLoad ? "coord-tensor source and a shared destination"
                                               : "shared source and a coord-tensor destination")
                                << ", got " << srcTyArg << " -> " << dstTyArg;

  OpBuilder::InsertionGuard guard(builder);
  if (pred) {
    auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
    Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }

  SmallVector<int32_t> tileShape(cfg.tileShape.begin(), cfg.tileShape.end());
  int32_t rank = static_cast<int32_t>(tileShape.size());
  auto coordTy = cast<fly::CoordTensorType>(glbTy);

  // `elemBits` is the *descriptor's* unit, which a recast may
  // have widened past the tensor's own element. The LDS operand keeps the tensor's
  // element type, so the two are checked against each other in bits rather than by
  // width: what has to agree is how much data one call moves, not how it is counted.
  int32_t elemBits = cfg.elemBits;
  int32_t ldsElemBits = static_cast<int32_t>(ldsMemTy.getElemTy().getIntOrFloatBitWidth());
  int32_t elemBytes = elemBits / 8; // verified byte-granular, power-of-two
  int32_t dataSizeCode = llvm::Log2_32(static_cast<uint32_t>(elemBytes));

  Type i32Ty = builder.getI32Type();
  Type i64Ty = builder.getI64Type();
  Value zeroC = i32Const(builder, loc, 0);
  Value zero64 = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value c16 = i32Const(builder, loc, 16);
  Value mask16 = i32Const(builder, loc, 0xFFFF);
  auto slotField = [&](unsigned slot) {
    return LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{slot});
  };

  // Tensor geometry, from the tensor origin, baked in at construction.
  SmallVector<Value> strideElems(rank);
  for (int32_t i = 0; i < rank - 1; ++i)
    strideElems[i] = slotField(slots.stride0 + i);
  strideElems[rank - 1] = arith::ConstantIntOp::create(builder, loc, 1, 64); // innermost contiguous
  for (int32_t i = 0; i < rank - 1; ++i) {
    Value inRange =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, strideElems[i],
                              arith::ConstantIntOp::create(builder, loc, kMaxTensorStride, 64));
    LLVM::AssumeOp::create(builder, loc, inRange);
  }

  // The tile coordinate comes from the coordinate-tensor operand, whose runtime value
  // is that coordinate: static leaves live in its type and dynamic ones are the
  // operands of the `make_int_tuple` that built it. A static leaf therefore costs a
  // constant the descriptor math folds away, and a dynamic one costs exactly the
  // integer the kernel already had.
  Value glbCoord = cfg.isLoad ? src : dst;
  SmallVector<Value> coord;
  if (failed(unpackCoord(builder, loc, coordTy.getBase(), glbCoord, coord)))
    return failure();
  if (static_cast<int32_t>(coord.size()) != rank)
    return mlir::emitError(loc) << "cdna5 TDM: the coordinate has " << coord.size()
                                << " leaves but the atom's tile is rank " << rank;

  // global_addr = base + elem_bytes * sum_i coord_i * stride_i. A coord left at zero
  // folds its whole term away, so a tile that never moves along a dim costs nothing.
  Value glbAddr = LLVM::PtrToIntOp::create(builder, loc, i64Ty, slotField(slots.basePtr));
  Value elemOff = zero64;
  for (int32_t i = 0; i < rank; ++i) {
    Value coordNonNegative =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, coord[i], zeroC);
    LLVM::AssumeOp::create(builder, loc, coordNonNegative);
    Value c64 = arith::ExtUIOp::create(builder, loc, i64Ty, coord[i]);
    elemOff = arith::AddIOp::create(builder, loc, elemOff,
                                    arith::MulIOp::create(builder, loc, c64, strideElems[i]));
  }
  glbAddr = arith::AddIOp::create(
      builder, loc, glbAddr,
      arith::MulIOp::create(builder, loc, elemOff,
                            arith::ConstantIntOp::create(builder, loc, elemBytes, 64)));

  // The bounds move with the tile, from the same coordinate. `boundary_check_i` selects between
  // the moving bound and the tensor's own extent; it is atom state rather than a
  // compile-time flag, so a call site can turn a dim's clamping off (or back on)
  // without a second atom. The usual case is a constant, and then the select folds and
  // the losing side's arithmetic dies -- the knob costs nothing until it varies.
  SmallVector<Value> tensorDim(rank);
  for (int32_t i = 0; i < rank; ++i) {
    Value extent = slotField(slots.extent0 + i);
    // Clamping off is the caller asserting the tile lies inside the tensor, i.e.
    // `coord_i + tile_i <= extent_i`. The un-shifted extent is then already a bound the
    // tile cannot reach, so the off side needs no subtract and no separate "never out of
    // bounds" sentinel: it reuses an SGPR that is live anyway, and stays a length the
    // hardware reads as positive -- where a saturated 0xFFFFFFFF would be an empty window
    // to a signed read.
    //
    // On side: `max(extent - coord, 0)` rather than the equivalent `coord < extent ? ... : 0`.
    // Extents and coordinates are non-negative, so the two agree, but the compare form is
    // the `usub.sat` idiom and AMDGPU only has that on the VALU -- which would drag a
    // uniform descriptor field onto the vector path and back through readfirstlane.
    Value rem = arith::SubIOp::create(builder, loc, extent, coord[i]);
    Value clamped = arith::MaxSIOp::create(builder, loc, rem, zeroC);
    tensorDim[i] =
        arith::SelectOp::create(builder, loc, slotField(slots.boundaryCheck0 + i), clamped, extent);
  }

  Value ldsAddr = LLVM::PtrToIntOp::create(builder, loc, i32Ty, ldsPtr);

  auto ldsLayout = dyn_cast<LayoutAttr>(ldsMemTy.getLayout());
  if (!ldsLayout || !ldsLayout.isStatic())
    return mlir::emitError(loc) << "cdna5 TDM: the LDS operand needs a fully static layout";
  LayoutBuilder<LayoutAttr> layoutBuilder(builder.getContext());
  int64_t ldsCapacity = layoutCosize(layoutBuilder, ldsLayout).getLeafAsInt().getValue();
  int64_t tileElems = 1;
  for (int32_t dim : tileShape)
    tileElems *= dim;
  // Every replay stacks another box into LDS, so the footprint counts them all.
  int64_t requiredLds = tileElems * cfg.iterCount;
  if (cfg.padAmount) {
    int64_t rows = tileElems / cfg.padInterval;
    requiredLds += (rows - 1) * cfg.padAmount;
  }
  // Both sides in bits, since a recast leaves the two counting in different units.
  if (ldsCapacity * ldsElemBits < requiredLds * elemBits)
    return mlir::emitError(loc) << "cdna5 TDM: LDS view holds " << ldsCapacity * ldsElemBits
                                << " bits, less than the descriptor footprint of "
                                << requiredLds * elemBits;
  if (tileElems * cfg.iterCount * elemBits % ldsElemBits != 0)
    return mlir::emitError(loc) << "cdna5 TDM: the descriptor moves "
                                << tileElems * cfg.iterCount * elemBits
                                << " bits, which is not a whole number of the LDS operand's "
                                << ldsElemBits << "-bit elements";

  // GROUP0 (vector<4xi32>): count, lds_addr, glb_lo, glb_hi | type. `count = 1` is
  // "valid tensor" (0 would be a NULL Tensor that moves nothing). The global address
  // is split from the full i64, so a tile coordinate walking past 4 GiB carries into
  // glb_hi automatically.
  Value g0s2 = LLVM::TruncOp::create(builder, loc, i32Ty, glbAddr);
  Value glbHiRaw = LLVM::LShrOp::create(builder, loc, glbAddr,
                                        arith::ConstantIntOp::create(builder, loc, 32, 64));
  // No mask on the high word: bits [30:25] are reserved and a canonical AMDGPU VA
  // (48-bit) never reaches them, so an `and` here only costs a SALU op per issue site
  // inside the K loop -- `s_or`/`s_bitset1` instead of `s_and`+`s_or`.
  Value g0s3 =
      arith::OrIOp::create(builder, loc, LLVM::TruncOp::create(builder, loc, i32Ty, glbHiRaw),
                           i32Const(builder, loc, /*type field [31:30]=2*/ 1 << 31));
  Value dgroup0 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, i32Ty),
      ValueRange{i32Const(builder, loc, /*count=*/1), ldsAddr, g0s2, g0s3});

  // Padding describes the padded LDS tile the DMA engine fills, which only the load
  // direction has: the store type carries no padding parameters, so this always
  // encodes "disabled" there.
  FailureOr<PadEncoding> padOr = computePadEncoding(cfg.padInterval, cfg.padAmount, elemBits);
  if (failed(padOr))
    return mlir::emitError(loc)
           << "cdna5 TDM: padding (interval=" << cfg.padInterval << ", amount=" << cfg.padAmount
           << " elements at " << elemBits
           << "-bit) is not encodable — the dword interval must be a power of two and the encoded "
              "fields must fit the descriptor bitfield";
  PadEncoding pad = *padOr;

  // Descriptor dims are innermost-first: descriptor dim j maps to tensor dim
  // (rank-1-j). LDS padding is carried by the pad bitfield, never by widening
  // tile_dim, so the global transfer extent stays the true tile size either way.
  auto descTensorDim = [&](int32_t j) -> Value { return tensorDim[rank - 1 - j]; };
  auto descTileDim = [&](int32_t j) -> int32_t { return tileShape[rank - 1 - j]; };
  // 48-bit stride slots: descriptor stride k = stride of tensor dim (rank-2-k).
  auto descStrideLo32 = [&](int32_t k) -> Value {
    return LLVM::TruncOp::create(builder, loc, i32Ty, strideElems[rank - 2 - k]);
  };
  auto descStrideHi16 = [&](int32_t k) -> Value {
    Value hi = LLVM::LShrOp::create(builder, loc, strideElems[rank - 2 - k],
                                    arith::ConstantIntOp::create(builder, loc, 32, 64));
    return arith::AndIOp::create(builder, loc, LLVM::TruncOp::create(builder, loc, i32Ty, hi),
                                 mask16);
  };
  auto lo16 = [&](Value v) { return arith::AndIOp::create(builder, loc, v, mask16); };
  auto hi16 = [&](Value v) {
    return arith::AndIOp::create(builder, loc, arith::ShRUIOp::create(builder, loc, v, c16),
                                 mask16);
  };
  auto shl16 = [&](Value v) { return arith::ShLIOp::create(builder, loc, v, c16); };
  auto orr = [&](Value a, Value b) { return arith::OrIOp::create(builder, loc, a, b); };

  // Whether the transfer arrives on a barrier is the atom's *type* and nothing else, so
  // config bit [18] is a constant here and the state says only which barrier -- the
  // pointer the kernel already holds, flattened to the atomic_barrier_address[18:3] the
  // descriptor wants (GROUP1 bits 47:32, i.e. sgpr1 [15:0]). An LDS pointer is 32-bit, so
  // that flattening is a bitcast and not a truncation.
  //
  // The pointer is never read as an enable, which is what lets LDS offset 0 be a barrier
  // like any other. It also means an atom whose type asks for a barrier and whose state
  // was never given one arrives on offset 0 rather than nowhere: naming the barrier is
  // the caller's to do, exactly as the base pointer is.
  Value barrierEnableBit = zeroC, barrierAddrField = zeroC;
  if (slots.hasAtomicBarrier) {
    Value barrier =
        LLVM::PtrToIntOp::create(builder, loc, i32Ty, slotField(slots.atomicBarrierAddr));
    Value barrierInRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, barrier,
                                                 i32Const(builder, loc, kMaxAtomicBarrierAddress));
    Value barrierAligned = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq,
        arith::AndIOp::create(builder, loc, barrier, i32Const(builder, loc, 7)), zeroC);
    LLVM::AssumeOp::create(builder, loc,
                           arith::AndIOp::create(builder, loc, barrierInRange, barrierAligned));
    barrierEnableBit = i32Const(builder, loc, 1 << 18);
    barrierAddrField =
        lo16(arith::ShRUIOp::create(builder, loc, barrier, i32Const(builder, loc, 3)));
  }

  // MCAST mask and its early-timeout companion exist on the load only.
  Value maskValue = slots.workgroupMask == kNoSlot ? zeroC : slotField(slots.workgroupMask);
  if (slots.workgroupMask != kNoSlot) {
    Value maskInRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, maskValue,
                                              i32Const(builder, loc, 0xFFFF));
    LLVM::AssumeOp::create(builder, loc, maskInRange);
  }
  Value maskLow = lo16(maskValue);
  Value earlyTimeoutBit = zeroC;
  if (slots.earlyTimeout != kNoSlot) {
    Value early = slotField(slots.earlyTimeout);
    Value earlyIsBool = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, early,
                                              i32Const(builder, loc, 1));
    Value earlyOff = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, early, zeroC);
    Value hasMask = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, maskValue, zeroC);
    LLVM::AssumeOp::create(
        builder, loc,
        arith::AndIOp::create(builder, loc, earlyIsBool,
                              arith::OrIOp::create(builder, loc, earlyOff, hasMask)));
    earlyTimeoutBit = arith::ShLIOp::create(
        builder, loc, arith::AndIOp::create(builder, loc, early, i32Const(builder, loc, 1)),
        i32Const(builder, loc, 21));
  }

  // GROUP1: config | mask, tensor_dim0/1 | barrier addr, tile_dim0/1/2, stride0/1.
  int32_t g1s0Upper = (dataSizeCode << 16) | ((cfg.iterCount > 1 ? 1 : 0) << 19) |
                      ((pad.enable ? 1 : 0) << 20) | (pad.interval << 22) | (pad.amount << 25);
  Value g1s0 =
      orr(orr(orr(i32Const(builder, loc, g1s0Upper), maskLow), barrierEnableBit), earlyTimeoutBit);

  Value td0 = descTensorDim(0);
  Value td1 = rank >= 2 ? descTensorDim(1) : zeroC;
  int32_t tile0 = descTileDim(0);
  int32_t tile1 = rank >= 2 ? descTileDim(1) : 0;
  int32_t tile2 = rank >= 3 ? descTileDim(2) : 0;
  Value g1s1 = orr(shl16(lo16(td0)), barrierAddrField); // tensor_dim0 lo16 | barrier addr
  Value g1s2 = orr(hi16(td0), shl16(lo16(td1)));        // dim0 hi16 | dim1 lo16
  auto upper16 = [](int32_t value) {
    return static_cast<int32_t>(static_cast<uint32_t>(value) << 16);
  };
  Value g1s3 = orr(hi16(td1), i32Const(builder, loc, upper16(tile0))); // dim1 hi16 | tile0
  int32_t g1s4c = (tile1 & 0xFFFF) | upper16(tile2);
  Value g1s5 = zeroC, g1s6 = zeroC, g1s7 = zeroC;
  if (rank >= 2) {
    g1s5 = descStrideLo32(0);
    g1s6 = descStrideHi16(0);
    if (rank >= 3) {
      g1s6 = orr(g1s6, shl16(lo16(descStrideLo32(1))));
      g1s7 = orr(hi16(descStrideLo32(1)), shl16(descStrideHi16(1)));
    }
  }
  Value dgroup1 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, i32Ty),
      ValueRange{g1s0, g1s1, g1s2, g1s3, i32Const(builder, loc, g1s4c), g1s5, g1s6, g1s7});

  // GROUP2 (rank>=3): tensor_dim2, tensor_dim3, stride2, tile_dim3 -- or, when the
  // descriptor iterates, the same three slots redefined as lds_addr_increment,
  // global_addr_increment and iterate_count. That is why iteration costs the fourth dim.
  Value g2s0 = zeroC, g2s1 = zeroC, g2s2 = zeroC, g2s3 = zeroC;
  if (rank >= 3)
    g2s0 = descTensorDim(2);
  if (cfg.iterCount > 1) {
    // Each replay lands the next box directly after the previous one in LDS, so the LDS
    // step is one whole box; the global step is the residual axis's stride, baked in as
    // a construction argument because a tensor stride can be dynamic.
    int64_t boxElems = 1;
    for (int32_t d : tileShape)
      boxElems *= d;
    Value glbInc = slotField(slots.iterStride);
    g2s1 = i32Const(builder, loc, static_cast<int32_t>(boxElems));
    g2s2 = LLVM::TruncOp::create(builder, loc, i32Ty, glbInc);
    Value glbIncHi = arith::AndIOp::create(
        builder, loc,
        LLVM::TruncOp::create(
            builder, loc, i32Ty,
            LLVM::LShrOp::create(builder, loc, glbInc,
                                 arith::ConstantIntOp::create(builder, loc, 32, 64))),
        mask16);
    g2s3 = orr(glbIncHi, i32Const(builder, loc, upper16(cfg.iterCount - 1)));
  } else if (rank >= 4) {
    g2s1 = descTensorDim(3);
    g2s2 = descStrideLo32(2);
    g2s3 = orr(descStrideHi16(2), i32Const(builder, loc, upper16(descTileDim(3))));
  }
  Value dg2 = vector::FromElementsOp::create(builder, loc, VectorType::get({4}, i32Ty),
                                             ValueRange{g2s0, g2s1, g2s2, g2s3});

  // GROUP3 (rank==5): stride3, tensor_dim4, tile_dim4.
  Value g3s0 = zeroC, g3s1 = zeroC, g3s2 = zeroC, g3s3 = zeroC;
  if (rank == 5) {
    Value td4 = descTensorDim(4);
    g3s0 = descStrideLo32(3);
    g3s1 = orr(descStrideHi16(3), shl16(lo16(td4)));
    g3s2 = orr(hi16(td4), i32Const(builder, loc, upper16(descTileDim(4))));
  }
  Value dg3 = vector::FromElementsOp::create(builder, loc, VectorType::get({4}, i32Ty),
                                             ValueRange{g3s0, g3s1, g3s2, g3s3});

  Value dg4 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, i32Ty),
      ValueRange{zeroC, zeroC, zeroC, zeroC, zeroC, zeroC, zeroC, zeroC});

  // The ROCDL intrinsic takes the cache policy as an attribute, which is why
  // `cacheModifier` has to stay a compile-time atom-type parameter.
  auto cachePolicy = builder.getI32IntegerAttr(static_cast<int32_t>(cfg.cacheModifier));
  ArrayAttr noAliasScopes;
  if (cfg.isLoad)
    ROCDL::TensorLoadToLDSOp::create(builder, loc, dgroup0, dgroup1, dg2, dg3, dg4, cachePolicy,
                                     noAliasScopes, noAliasScopes, noAliasScopes);
  else
    ROCDL::TensorStoreFromLDSOp::create(builder, loc, dgroup0, dgroup1, dg2, dg3, dg4, cachePolicy,
                                        noAliasScopes, noAliasScopes, noAliasScopes);

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// CopyOpCDNA5TensorLoadType — TENSOR_LOAD_TO_LDS (global -> LDS)
//===----------------------------------------------------------------------===//

// Static, so it answers for the direction rather than for one atom: whether a given
// atom carries `atomic_barrier_addr` is a type parameter, and `setAtomState` reads it.
std::optional<unsigned> CopyOpCDNA5TensorLoadType::getFieldIndex(AtomStateField field) {
  return tdmFieldIndex(kLoadSlots, field);
}

Type CopyOpCDNA5TensorLoadType::getConvertedType(MLIRContext *ctx) const {
  return tdmConvertedType(ctx, kLoadSlots);
}

// The tensor geometry is not optional for this atom, so there is no resting state
// to hand back: an atom without its base pointer could only lower to a descriptor
// pointing at nothing. `getAtomState` is the way in.
Value CopyOpCDNA5TensorLoadType::getDefaultState(OpBuilder &builder, Location loc) const {
  mlir::emitError(loc) << "cdna5 TDM: this atom is built from a tensor and needs its "
                          "construction arguments (base, strides, extents)";
  return nullptr;
}

Value CopyOpCDNA5TensorLoadType::getAtomState(OpBuilder &builder, Location loc,
                                              ValueRange args) const {
  return tdmInitialState(builder, loc, tdmSlots(kLoadSlots, getAtomicBarrier()),
                         getTileShape().size(), getTensor2tdm(), getIterCount() > 1, args);
}

Value CopyOpCDNA5TensorLoadType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                              Attribute fieldAttr, Value fieldValue) const {
  return tdmSetAtomState(builder, loc, tdmSlots(kLoadSlots, getAtomicBarrier()),
                         getTileShape().size(), atomStruct, fieldAttr, fieldValue);
}

Attribute CopyOpCDNA5TensorLoadType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA5TensorLoadType::getThrBitLayoutSrc() const {
  return tdmThrBitLayout(getContext(), getTileShape(), tdmElemBits(getDataType()), getIterCount());
}
Attribute CopyOpCDNA5TensorLoadType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }
Attribute CopyOpCDNA5TensorLoadType::getThrBitLayoutRef() const { return getThrBitLayoutSrc(); }

LogicalResult CopyOpCDNA5TensorLoadType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                ArrayRef<int32_t> tileShape, Type dataType,
                                                IntTupleAttr tensor2tdm, bool atomicBarrier,
                                                int32_t cacheModifier, int32_t iterCount,
                                                int32_t padInterval, int32_t padAmount) {
  return tdmVerify(emitError, tileShape, dataType, tensor2tdm, iterCount, padInterval, padAmount);
}

FailureOr<Value> CopyOpCDNA5TensorLoadType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                            Type resultTy, Type copyAtomTyArg,
                                                            Type srcTyArg, Type dstTyArg,
                                                            Value atomVal, Value src,
                                                            Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpCDNA5TensorLoadType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

LogicalResult CopyOpCDNA5TensorLoadType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcTyArg,
                                                      Type dstTyArg, Value atomVal, Value src,
                                                      Value dst) const {
  TdmStatic cfg = tdmConfig(*this);
  return emitTdmAtomCall(builder, loc, cfg, tdmSlots(kLoadSlots, getAtomicBarrier()), srcTyArg,
                         dstTyArg, atomVal, src, dst);
}

LogicalResult CopyOpCDNA5TensorLoadType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcTyArg,
                                                      Type dstTyArg, Type predMemTyArg,
                                                      Value atomVal, Value src, Value dst,
                                                      Value pred) const {
  TdmStatic cfg = tdmConfig(*this);
  return emitTdmAtomCall(builder, loc, cfg, tdmSlots(kLoadSlots, getAtomicBarrier()), srcTyArg,
                         dstTyArg, atomVal, src, dst, predMemTyArg, pred);
}

//===----------------------------------------------------------------------===//
// CopyOpCDNA5TensorStoreType — TENSOR_STORE_FROM_LDS (LDS -> global)
//===----------------------------------------------------------------------===//

std::optional<unsigned> CopyOpCDNA5TensorStoreType::getFieldIndex(AtomStateField field) {
  return tdmFieldIndex(kStoreSlots, field);
}

Type CopyOpCDNA5TensorStoreType::getConvertedType(MLIRContext *ctx) const {
  return tdmConvertedType(ctx, kStoreSlots);
}

// The tensor geometry is not optional for this atom, so there is no resting state
// to hand back: an atom without its base pointer could only lower to a descriptor
// pointing at nothing. `getAtomState` is the way in.
Value CopyOpCDNA5TensorStoreType::getDefaultState(OpBuilder &builder, Location loc) const {
  mlir::emitError(loc) << "cdna5 TDM: this atom is built from a tensor and needs its "
                          "construction arguments (base, strides, extents)";
  return nullptr;
}

Value CopyOpCDNA5TensorStoreType::getAtomState(OpBuilder &builder, Location loc,
                                               ValueRange args) const {
  return tdmInitialState(builder, loc, tdmSlots(kStoreSlots, getAtomicBarrier()),
                         getTileShape().size(), getTensor2tdm(), getIterCount() > 1, args);
}

Value CopyOpCDNA5TensorStoreType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                               Attribute fieldAttr, Value fieldValue) const {
  return tdmSetAtomState(builder, loc, tdmSlots(kStoreSlots, getAtomicBarrier()),
                         getTileShape().size(), atomStruct, fieldAttr, fieldValue);
}

Attribute CopyOpCDNA5TensorStoreType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA5TensorStoreType::getThrBitLayoutSrc() const {
  return tdmThrBitLayout(getContext(), getTileShape(), tdmElemBits(getDataType()), getIterCount());
}
Attribute CopyOpCDNA5TensorStoreType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }
Attribute CopyOpCDNA5TensorStoreType::getThrBitLayoutRef() const { return getThrBitLayoutSrc(); }

LogicalResult CopyOpCDNA5TensorStoreType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                 ArrayRef<int32_t> tileShape, Type dataType,
                                                 IntTupleAttr tensor2tdm, bool atomicBarrier,
                                                 int32_t cacheModifier, int32_t iterCount) {
  return tdmVerify(emitError, tileShape, dataType, tensor2tdm, iterCount, /*padInterval=*/0,
                   /*padAmount=*/0);
}

FailureOr<Value> CopyOpCDNA5TensorStoreType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                             Type resultTy, Type copyAtomTyArg,
                                                             Type srcTyArg, Type dstTyArg,
                                                             Value atomVal, Value src,
                                                             Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpCDNA5TensorStoreType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

LogicalResult CopyOpCDNA5TensorStoreType::emitAtomCall(OpBuilder &builder, Location loc,
                                                       Type copyAtomTyArg, Type srcTyArg,
                                                       Type dstTyArg, Value atomVal, Value src,
                                                       Value dst) const {
  TdmStatic cfg = tdmConfig(*this);
  return emitTdmAtomCall(builder, loc, cfg, tdmSlots(kStoreSlots, getAtomicBarrier()), srcTyArg,
                         dstTyArg, atomVal, src, dst);
}

LogicalResult CopyOpCDNA5TensorStoreType::emitAtomCall(OpBuilder &builder, Location loc,
                                                       Type copyAtomTyArg, Type srcTyArg,
                                                       Type dstTyArg, Type predMemTyArg,
                                                       Value atomVal, Value src, Value dst,
                                                       Value pred) const {
  TdmStatic cfg = tdmConfig(*this);
  return emitTdmAtomCall(builder, loc, cfg, tdmSlots(kStoreSlots, getAtomicBarrier()), srcTyArg,
                         dstTyArg, atomVal, src, dst, predMemTyArg, pred);
}

} // namespace mlir::fly_rocdl
